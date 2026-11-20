//! Audio for the decoded-audio hooks: the views a hook takes of a chunk
//! ([`mono`], [`rms`], [`dbfs`]), and the tap the engine feeds decoded audio
//! into, which hands each hook its own chunks.
//!
//! The tap converts each hook's audio as one continuous stream — one
//! resampler per hook for the whole track — so a hook asking for 16 kHz mono
//! gets exactly what resampling the whole track at once would give, cut into
//! chunks afterwards. Nothing rings or clicks at a chunk's edge.

use std::sync::Arc;

use anyhow::{Context, Result};
use codec::audio::AudioFrame;
use codec::audio::resample::AlignedResampler;

use super::{AudioDelivery, AudioEvent, Hooks};

/// `samples` (`channels` interleaved) mixed down to one channel: the
/// equal-weight average of each frame.
pub fn mono(samples: &[f32], channels: u8) -> Vec<f32> {
    let ch = usize::from(channels.max(1));
    if ch == 1 {
        return samples.to_vec();
    }
    let scale = 1.0 / ch as f32;
    samples
        .chunks_exact(ch)
        .map(|f| f.iter().sum::<f32>() * scale)
        .collect()
}

/// The root-mean-square level of `samples`, `0.0` for none.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

/// A level (`rms`, a peak) in dB relative to full scale; `-inf` for silence.
pub fn dbfs(level: f32) -> f32 {
    20.0 * level.log10()
}

/// `frame`'s samples with `want` channels: mixed down to mono for one, the
/// first `want` kept for fewer, the last repeated for more. Only a track
/// whose channel count changes part-way through (a broadcast capture
/// switching from stereo to 5.1) ever needs anything but the first case.
fn reshape(frame: &AudioFrame, want: u8) -> Vec<f32> {
    let have = frame.channels.max(1);
    if have == want {
        return frame.samples.clone();
    }
    if want == 1 {
        return mono(&frame.samples, have);
    }
    let (have, want) = (usize::from(have), usize::from(want));
    let mut out = Vec::with_capacity(frame.samples.len() / have * want);
    for f in frame.samples.chunks_exact(have) {
        out.extend((0..want).map(|c| f[c.min(have - 1)]));
    }
    out
}

/// One hook's stream: its conversion and the chunk it is filling.
struct Out {
    entry: usize,
    delivery: AudioDelivery,
    /// Built on the first frame, when the decoded rate and layout are known.
    resampler: Option<AlignedResampler>,
    channels: u8,
    rate: u32,
    chunk_frames: usize,
    buf: Vec<f32>,
    /// Frames (per channel) emitted so far, at `rate`.
    emitted: u64,
    index: u64,
}

impl Out {
    fn start(&mut self, frame: &AudioFrame) -> Result<()> {
        if self.resampler.is_some() {
            return Ok(());
        }
        self.channels = if self.delivery.mono {
            1
        } else {
            frame.channels.clamp(1, 8)
        };
        self.rate = self.delivery.sample_rate.unwrap_or(frame.sample_rate);
        let seconds = if self.delivery.chunk_seconds > 0.0 {
            self.delivery.chunk_seconds
        } else {
            AudioDelivery::default().chunk_seconds
        };
        self.chunk_frames = ((seconds * f64::from(self.rate)).round() as usize).max(1);
        self.resampler = Some(
            AlignedResampler::new(frame.sample_rate, self.rate, self.channels)
                .context("the audio hook's resampler")?,
        );
        Ok(())
    }
}

/// Where the engine sends a source's decoded audio when hooks want it.
pub(crate) struct AudioTap {
    hooks: Hooks,
    clip: usize,
    codec: String,
    outs: Vec<Out>,
}

impl AudioTap {
    /// A tap for source `clip`, or `None` when no hook takes audio.
    pub(crate) fn new(hooks: &Hooks, clip: usize, codec: &str) -> Option<Self> {
        let outs: Vec<Out> = hooks
            .audio_deliveries()
            .into_iter()
            .map(|(entry, delivery)| Out {
                entry,
                delivery,
                resampler: None,
                channels: 0,
                rate: 0,
                chunk_frames: 0,
                buf: Vec::new(),
                emitted: 0,
                index: 0,
            })
            .collect();
        (!outs.is_empty()).then(|| Self {
            hooks: hooks.clone(),
            clip,
            codec: codec.to_string(),
            outs,
        })
    }

    /// The next decoded `frame`, in order. `Err` when a hook rejected the job.
    pub(crate) fn push(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.samples.is_empty() || frame.sample_rate == 0 {
            return Ok(());
        }
        for i in 0..self.outs.len() {
            let out = &mut self.outs[i];
            out.start(frame)?;
            let shaped = AudioFrame {
                samples: reshape(frame, out.channels),
                sample_rate: frame.sample_rate,
                channels: out.channels,
                pts: frame.pts,
            };
            // Built on the first frame; a rate change part-way (rare, and
            // never in the codecs that state one rate for the stream) starts
            // a new resampler from that frame's rate.
            let resampler = out.resampler.as_mut().expect("started");
            if resampler.in_rate() != frame.sample_rate {
                let mut tail = Vec::new();
                resampler.flush(&mut tail)?;
                out.buf.extend_from_slice(&tail);
                *resampler = AlignedResampler::new(frame.sample_rate, out.rate, out.channels)?;
            }
            resampler.process(&shaped, &mut out.buf)?;
            self.drain(i, false)?;
        }
        Ok(())
    }

    /// The end of the track: what the resamplers hold, and the last chunks.
    pub(crate) fn finish(mut self) -> Result<()> {
        for i in 0..self.outs.len() {
            let out = &mut self.outs[i];
            if let Some(r) = out.resampler.as_mut() {
                r.flush(&mut out.buf)?;
            }
            self.drain(i, true)?;
        }
        Ok(())
    }

    /// Hand out the whole chunks `outs[i]` holds, keeping one back until
    /// more audio arrives (so the final chunk is known to be the last, even
    /// when the track divides into whole chunks exactly); with `last`,
    /// everything left, the final one marked.
    fn drain(&mut self, i: usize, last: bool) -> Result<()> {
        let out = &mut self.outs[i];
        let ch = usize::from(out.channels.max(1));
        let whole = out.chunk_frames * ch;
        while out.buf.len() > whole || (last && !out.buf.is_empty()) {
            let take = whole.min(out.buf.len());
            let is_last = last && out.buf.len() <= whole;
            let samples: Arc<[f32]> = out.buf.drain(..take).collect();
            let event = AudioEvent {
                clip: self.clip,
                index: out.index,
                seconds: out.emitted as f64 / f64::from(out.rate),
                codec: self.codec.clone(),
                sample_rate: out.rate,
                channels: out.channels,
                samples,
                last: is_last,
            };
            out.emitted += (take / ch) as u64;
            out.index += 1;
            self.hooks.emit_decoded_audio(out.entry, event)?;
        }
        Ok(())
    }
}
