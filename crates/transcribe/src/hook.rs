//! [`Transcribe`]: the decoded-audio hook that turns speech into timed text.

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
/// - `skipped: true` when the gate found nothing to hear.
///
/// [`Transcribe::segments`] collects them from a job's report. Inference
/// takes seconds per window: register it with
/// `HookPolicy::background()` in a transcode, so the ladder does not wait on
/// it. In [`run_audio_hooks`](rivet::run_audio_hooks), its only job, either
/// policy does.
///
/// A window is decoded on its own, so a word spoken across a 30-second
/// boundary may be cut or lost.
pub struct Transcribe {
    engine: Mutex<Engine>,
    label: String,
    options: TranscribeOptions,
    detected: Mutex<Option<&'static str>>,
}

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
            detected: Mutex::new(None),
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

    fn language_for(&self, engine: &mut Engine, samples: &[f32]) -> Result<String> {
        if let Some(l) = &self.options.language {
            return Ok(l.clone());
        }
        let mut detected = self.detected.lock().unwrap();
        if let Some(l) = *detected {
            return Ok(l.to_string());
        }
        let l = engine.detect_language(samples)?;
        *detected = Some(l);
        Ok(l.to_string())
    }
}

impl DecodedAudioHook for Transcribe {
    fn delivery(&self) -> AudioDelivery {
        AudioDelivery::speech()
    }

    fn on_decoded_audio(&self, _ctx: &HookContext, audio: &AudioEvent) -> Result<HookOutcome> {
        let rate = audio.sample_rate;
        let mono;
        let samples: &[f32] = if audio.channels == 1 {
            &audio.samples
        } else {
            mono = rivet::hooks::audio::mono(&audio.samples, audio.channels);
            &mono
        };
        let outcome = HookOutcome::proceed().annotate("model", self.label.clone());
        let active = self.options.gate.detect(samples, rate, 0.0);
        if active.is_empty() || rate != 16_000 {
            if rate != 16_000 {
                tracing::warn!(
                    rate,
                    "Transcribe needs 16 kHz audio (AudioDelivery::speech)"
                );
            }
            return Ok(outcome
                .annotate("segments", Vec::<Value>::new())
                .annotate("skipped", true));
        }
        // The quiet between active ranges, silenced: what the model would
        // otherwise fill with invented words.
        let mut heard = vec![0f32; samples.len()];
        for r in &active {
            let a = (r.start * f64::from(rate)) as usize;
            let b = ((r.end * f64::from(rate)) as usize).min(samples.len());
            heard[a..b].copy_from_slice(&samples[a..b]);
        }
        let mut engine = self.engine.lock().unwrap();
        let language = self.language_for(&mut engine, &heard)?;
        let result = engine.transcribe_window(&heard, &language, self.options.task)?;
        drop(engine);
        let silent = result.no_speech_prob > self.options.no_speech_threshold
            && result.avg_logprob < self.options.logprob_threshold;
        let overlaps = |s: &Segment| active.iter().any(|r| s.start < r.end && r.start < s.end);
        let segments: Vec<Value> = if silent {
            Vec::new()
        } else {
            result
                .segments
                .iter()
                .filter(|s| overlaps(s))
                .map(|s| json!([audio.seconds + s.start, audio.seconds + s.end, s.text]))
                .collect()
        };
        Ok(outcome
            .annotate("segments", segments)
            .annotate("language", language)
            .annotate("avg_logprob", result.avg_logprob)
            .annotate("no_speech_prob", result.no_speech_prob))
    }

    fn describe(&self) -> String {
        format!("transcribe ({})", self.label)
    }
}
