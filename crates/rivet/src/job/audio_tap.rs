//! The source's audio, decoded for the decoded-audio hooks.
//!
//! A pass of its own over the track, apart from [`prepare_audio`]: the output
//! is whatever it would have been without hooks (a passthrough stays a
//! byte-for-byte copy, a transcode is not touched), and the hooks see the
//! same PCM whatever the output does with the track. Decoding audio costs
//! little beside a video decode, and a job with no audio hook skips this
//! pass entirely.
//!
//! [`prepare_audio`]: super::audio::prepare_audio

use anyhow::{Context, Result};
use codec::audio::create_decoder;
use container::demux::AudioTrack;
use container::edit::{AudioEdit, AudioGap};

use super::audio::{PcmWindow, opus_pre_skip_edit};
use crate::hooks::{Hooks, Stage, audio::AudioTap};
use crate::spec::{AudioDecodeDeny, HeAacPolicy};

/// Whether the hooks were handed the track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tapped {
    /// No hook takes audio, or there is no track: nothing was decoded.
    NotWanted,
    /// Every chunk was handed over.
    Delivered,
    /// The track cannot be decoded here (no decoder, a tool the decoder
    /// refuses, a codec `audio-decode-deny` names); the hooks got nothing.
    Undecodable,
}

/// Decode `track` (with its `edit` and `gaps`, as the demuxer gave them) and
/// hand it to the decoded-audio hooks as source `clip`. `Err` only when a
/// hook rejects the job.
///
/// `he_aac` is honoured for `core` (the hooks hear what a core-only decode
/// gives); under `passthrough` an HE-AAC track is still decoded in full here,
/// since the output is untouched. `deny` is honoured: a denied codec is
/// never decoded, for hooks or anything else.
pub(crate) fn tap_audio(
    hooks: &Hooks,
    clip: usize,
    track: Option<&AudioTrack>,
    edit: Option<AudioEdit>,
    gaps: &[AudioGap],
    he_aac: HeAacPolicy,
    deny: AudioDecodeDeny,
) -> Result<Tapped> {
    let Some(track) = track else {
        return Ok(Tapped::NotWanted);
    };
    if !hooks.wants(Stage::DecodedAudio) || track.samples.is_empty() {
        return Ok(Tapped::NotWanted);
    }
    let codec = track.codec.to_ascii_lowercase();
    let Some(mut tap) = AudioTap::new(hooks, clip, &codec) else {
        return Ok(Tapped::NotWanted);
    };
    if deny.denies(&codec) {
        tracing::warn!(
            codec,
            "decoded-audio hooks: decoding {codec} is denied by audio-decode-deny; the hooks get no audio"
        );
        return Ok(Tapped::Undecodable);
    }
    let core_only = he_aac == HeAacPolicy::Core;
    let private = if codec == "aac" {
        &track.asc
    } else {
        &track.codec_private
    };
    let extra = (!private.is_empty()).then_some(private.as_slice());
    let mut dec: Box<dyn codec::audio::AudioDecoder> = if codec == "aac" {
        let first = track.samples.first().map_or(&[][..], Vec::as_slice);
        if let Err(e) = codec::audio::decode::aac::probe(&track.asc, first, core_only) {
            tracing::warn!(codec, error = %e, "decoded-audio hooks: this AAC stream cannot be decoded; the hooks get no audio");
            return Ok(Tapped::Undecodable);
        }
        if core_only {
            Box::new(
                codec::audio::decode::aac::AacDecoder::new_core_only(extra)
                    .context("audio decoder")?,
            )
        } else {
            match create_decoder(&codec, extra, track.sample_rate, track.channels as u8) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(codec, error = %e, "decoded-audio hooks: no decoder; the hooks get no audio");
                    return Ok(Tapped::Undecodable);
                }
            }
        }
    } else {
        match create_decoder(&codec, extra, track.sample_rate, track.channels as u8) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(codec, error = %e, "decoded-audio hooks: no decoder; the hooks get no audio");
                return Ok(Tapped::Undecodable);
            }
        }
    };

    let edit = edit.or_else(|| opus_pre_skip_edit(&codec, track));
    let mut window: Option<PcmWindow> = None;
    let mut holes = gaps.iter().peekable();
    let mut pts: i64 = 0;
    // The rate the decoder really outputs (an HE-AAC track's SBR rate, Opus's
    // 48 kHz), from its first frame: the edit's window is built on it.
    let mut pcm_rate: Option<u32> = None;
    let mut last_channels: u8 = track.channels.clamp(1, 255) as u8;

    let feed = |frame: codec::audio::AudioFrame,
                window: &mut Option<PcmWindow>,
                pcm_rate: &mut Option<u32>,
                tap: &mut AudioTap|
     -> Result<()> {
        if pcm_rate.is_none() {
            *pcm_rate = Some(frame.sample_rate);
            *window = edit.map(|e| PcmWindow::new(&e, track.timescale, frame.sample_rate));
        }
        let frame = match window.as_mut() {
            None => frame,
            Some(w) => match w.take(&frame) {
                Some(f) => f,
                None => return Ok(()),
            },
        };
        tap.push(&frame)
    };

    for (index, packet) in track.samples.iter().enumerate() {
        let frames = match dec.decode(packet, pts) {
            Ok(frames) => frames,
            Err(codec::audio::AudioError::Unsupported(reason)) => {
                tracing::warn!(codec, %reason, "decoded-audio hooks: the decoder refuses this stream; the hooks got audio up to here");
                tap.finish()?;
                return Ok(Tapped::Undecodable);
            }
            Err(e) => return Err(e).context("audio decode for hooks"),
        };
        for frame in frames {
            pts =
                pts.saturating_add((frame.samples.len() as i64) / i64::from(frame.channels.max(1)));
            last_channels = frame.channels.max(1);
            feed(frame, &mut window, &mut pcm_rate, &mut tap)?;
        }
        while let Some(hole) = holes.next_if(|h| h.after_packet == index) {
            let rate = pcm_rate.unwrap_or(track.sample_rate);
            let length = container::edit::rescale_round(hole.ticks, rate, track.timescale);
            let silence = codec::audio::AudioFrame {
                samples: vec![0.0; length as usize * usize::from(last_channels)],
                sample_rate: rate,
                channels: last_channels,
                pts,
            };
            pts = pts.saturating_add(length as i64);
            feed(silence, &mut window, &mut pcm_rate, &mut tap)?;
        }
    }
    for frame in dec.flush().context("audio flush for hooks")? {
        feed(frame, &mut window, &mut pcm_rate, &mut tap)?;
    }
    tap.finish()?;
    Ok(Tapped::Delivered)
}
