//! [`Transcribe`]: the decoded-audio hook that turns speech into timed text.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::Result;
use rivet::hooks::{
    ActivityGate, AudioDelivery, AudioEvent, DecodedAudioHook, HookContext, HookOutcome, HookReport,
};
use serde_json::{Value, json};

use crate::decode::Segment;
use crate::model::{Engine, SpeechModel, Task, device_or_cpu};

/// How to transcribe.
#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// The language spoken (`en`, `pt`, ...); `None` detects it from the first
    /// active window and keeps it for the rest of the job.
    pub language: Option<String>,
    pub task: Task,
    /// What decides which audio the model hears: windows with nothing active
    /// are skipped, and the quiet inside the rest is silenced.
    pub gate: ActivityGate,
    /// A window whose no-speech probability is above this, and whose tokens
    /// were chosen with a mean log-probability below `logprob_threshold`, is
    /// taken to hold no speech (Whisper's own test: 0.6 and -1.0).
    pub no_speech_threshold: f64,
    pub logprob_threshold: f64,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            task: Task::Transcribe,
            gate: ActivityGate::default(),
            no_speech_threshold: 0.6,
            logprob_threshold: -1.0,
        }
    }
}

/// Speech to timed text: a [`DecodedAudioHook`] running a Whisper model.
///
/// It takes 30-second windows of 16 kHz mono ([`AudioDelivery::speech`]).
/// Each window's record carries
/// - `segments`: `[[start, end, text], ...]`, seconds from the start of the
///   source's audio;
/// - `language` and `model` (the [`SpeechModel`]'s label);
/// - `carried_seconds`: audio held for the next window;
/// - `skipped: true` when no window was decoded for this chunk (it was all
///   carried), or the audio was not 16 kHz.
///
/// [`Transcribe::segments`] collects them from a job's report. Inference
/// takes seconds per window: register it with
/// `HookPolicy::background()` in a transcode, so the ladder does not wait on
/// it. In [`run_audio_hooks`](rivet::run_audio_hooks), its only job, either
/// policy does.
///
/// A sentence a window cuts off is not kept from that window: the audio from
/// the end of the last sentence the model finished is carried into the next
/// window, which hears it whole (as Whisper's own long-form decoding seeks).
/// So a chunk's record holds the segments finished by then, and the last
/// chunk's record (`AudioEvent::last`) everything left.
///
/// One `Transcribe` can serve several jobs at once: what it carries, and the
/// language it detected, are kept per job and per source.
pub struct Transcribe {
    engine: Mutex<Engine>,
    label: String,
    options: TranscribeOptions,
    /// `(job id, clip)` → what that source carries between chunks.
    streams: Mutex<HashMap<(String, usize), Stream>>,
}

/// One source's audio not yet decoded, and its language once known.
#[derive(Default)]
struct Stream {
    /// Where `samples` starts, in seconds from the start of the audio.
    start: f64,
    samples: Vec<f32>,
    language: Option<String>,
}

/// Whisper's window, in samples at 16 kHz.
const WINDOW: usize = 30 * 16_000;
/// A window is advanced at least this far, in seconds, so a model that keeps
/// closing its first sentence early cannot stall on one stretch of audio.
const MIN_ADVANCE: f64 = 1.0;
/// Audio active this close to a window's end, in seconds, runs past it.
const RUNS_TO_END: f64 = 0.5;

impl Transcribe {
    /// Load `model` (on a GPU when built with `cuda` or `metal` and one is
    /// there, the CPU otherwise). Refused when the directory is incomplete
    /// or the weights do not load.
    pub fn load(model: &SpeechModel, options: TranscribeOptions) -> Result<Self> {
        let engine = Engine::load(model, device_or_cpu())?;
        if let Some(lang) = &options.language
            && engine.tokens().language(lang).is_none()
        {
            anyhow::bail!(
                "the model at {} has no language `{lang}`",
                model.dir.display()
            );
        }
        Ok(Self {
            engine: Mutex::new(engine),
            label: model.label.clone(),
            options,
            streams: Mutex::new(HashMap::new()),
        })
    }

    /// Every segment the hook registered as `hook` recorded in `report`, in
    /// order of time.
    pub fn segments(report: &HookReport, hook: &str) -> Vec<Segment> {
        let mut out: Vec<Segment> = report
            .by_hook(hook)
            .filter_map(|r| r.annotation("segments"))
            .filter_map(Value::as_array)
            .flatten()
            .filter_map(|s| {
                let s = s.as_array()?;
                Some(Segment {
                    start: s.first()?.as_f64()?,
                    end: s.get(1)?.as_f64()?,
                    text: s.get(2)?.as_str()?.to_string(),
                })
            })
            .collect();
        out.sort_by(|a, b| a.start.total_cmp(&b.start));
        out
    }

    /// The language the hook registered as `hook` recorded, if it heard any
    /// speech.
    pub fn language(report: &HookReport, hook: &str) -> Option<String> {
        report
            .by_hook(hook)
            .filter_map(|r| r.annotation("language"))
            .find_map(|v| v.as_str().map(str::to_string))
    }

    /// One window (at most 30 s of 16 kHz mono starting at second 0 of
    /// `piece`): the segments to keep, in seconds from the window's start,
    /// and how many seconds of it were used. With `final_piece` nothing more
    /// follows, so a sentence the window cut is kept as it is.
    fn window(
        &self,
        piece: &[f32],
        final_piece: bool,
        stream: &mut Stream,
    ) -> Result<(Vec<Segment>, f64)> {
        let length = piece.len() as f64 / 16_000.0;
        let active = self.options.gate.detect(piece, 16_000, 0.0);
        if active.is_empty() {
            return Ok((Vec::new(), length));
        }
        // The quiet between active ranges, silenced: what the model would
        // otherwise fill with invented words.
        let mut heard = vec![0f32; piece.len()];
        for r in &active {
            let a = (r.start * 16_000.0) as usize;
            let b = ((r.end * 16_000.0) as usize).min(piece.len());
            heard[a..b].copy_from_slice(&piece[a..b]);
        }
        let mut engine = self.engine.lock().unwrap();
        let language = match (&self.options.language, &stream.language) {
            (Some(l), _) | (None, Some(l)) => l.clone(),
            (None, None) => {
                let l = engine.detect_language(&heard)?.to_string();
                stream.language = Some(l.clone());
                l
            }
        };
        let result = engine.transcribe_window(&heard, &language, self.options.task)?;
        drop(engine);
        if result.no_speech_prob > self.options.no_speech_threshold
            && result.avg_logprob < self.options.logprob_threshold
        {
            return Ok((Vec::new(), length));
        }
        let overlaps = |s: &Segment| active.iter().any(|r| s.start < r.end && r.start < s.end);
        // Where the next window starts: after the last sentence the model
        // finished. A sentence left open was cut by the window; so, likely,
        // was the last one when the audio is still active at the window's
        // end, even if the model closed it (it often closes a cut sentence a
        // moment before the boundary).
        let runs_to_end = active.last().is_some_and(|r| r.end >= length - RUNS_TO_END);
        let restart = match result.closed_end {
            Some(end) => Some(end),
            None if runs_to_end => result.segments.last().map(|s| s.start),
            None => None,
        };
        let (keep_until, used) = match restart {
            Some(at) if !final_piece && at >= MIN_ADVANCE => (at, at),
            _ => (f64::INFINITY, length),
        };
        let segments = result
            .segments
            .into_iter()
            .filter(|s| s.end <= keep_until + 1e-6 && overlaps(s))
            .collect();
        Ok((segments, used))
    }
}

impl DecodedAudioHook for Transcribe {
    fn delivery(&self) -> AudioDelivery {
        AudioDelivery::speech()
    }

    fn on_decoded_audio(&self, ctx: &HookContext, audio: &AudioEvent) -> Result<HookOutcome> {
        let outcome = HookOutcome::proceed().annotate("model", self.label.clone());
        if audio.sample_rate != 16_000 {
            tracing::warn!(
                rate = audio.sample_rate,
                "Transcribe needs 16 kHz audio (AudioDelivery::speech)"
            );
            return Ok(outcome
                .annotate("segments", Vec::<Value>::new())
                .annotate("skipped", true));
        }
        let key = (ctx.job_id.clone(), audio.clip);
        // Taken out while it is worked on, so other jobs' chunks are not held
        // up behind this one's inference by the map's lock.
        let mut stream = {
            let mut streams = self.streams.lock().unwrap();
            match streams.remove(&key) {
                Some(s) if audio.index > 0 => s,
                _ => Stream {
                    start: audio.seconds,
                    ..Stream::default()
                },
            }
        };
        if audio.channels == 1 {
            stream.samples.extend_from_slice(&audio.samples);
        } else {
            stream
                .samples
                .extend(rivet::hooks::audio::mono(&audio.samples, audio.channels));
        }
        let mut segments: Vec<Value> = Vec::new();
        let mut decoded = false;
        while stream.samples.len() >= WINDOW || (audio.last && !stream.samples.is_empty()) {
            let n = stream.samples.len().min(WINDOW);
            let final_piece = audio.last && stream.samples.len() <= WINDOW;
            let piece = stream.samples[..n].to_vec();
            let (found, used) = self.window(&piece, final_piece, &mut stream)?;
            decoded = true;
            segments.extend(
                found
                    .iter()
                    .map(|s| json!([stream.start + s.start, stream.start + s.end, s.text])),
            );
            let used = ((used * 16_000.0).round() as usize).clamp(1, n);
            stream.samples.drain(..used);
            stream.start += used as f64 / 16_000.0;
        }
        let carried = stream.samples.len() as f64 / 16_000.0;
        let language = stream
            .language
            .clone()
            .or_else(|| self.options.language.clone());
        if !audio.last {
            self.streams.lock().unwrap().insert(key, stream);
        }
        let mut outcome = outcome
            .annotate("segments", segments)
            .annotate("carried_seconds", carried);
        if !decoded {
            outcome = outcome.annotate("skipped", true);
        }
        if let Some(l) = language {
            outcome = outcome.annotate("language", l);
        }
        Ok(outcome)
    }

    fn describe(&self) -> String {
        format!("transcribe ({})", self.label)
    }
}
