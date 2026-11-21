//! A Whisper model from a directory, and speech to timed text with it.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{
    self as w, audio,
    model::{AudioEncoder, Whisper},
};
use serde::Deserialize;

use crate::decode::{Rules, Segment, closed_end, segments_of};
use crate::decoder::CachedDecoder;
use crate::mel;
use crate::tokens::Tokens;

/// The files a model directory must hold.
pub const REQUIRED_FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

/// A Whisper checkpoint on disk: the directory and the name to record.
///
/// Any Whisper checkpoint in Hugging Face's layout (tiny to large-v3,
/// large-v3-turbo, a fine-tune of one) loads from the same code: the sizes
/// differ only in the numbers `config.json` holds.
#[derive(Debug, Clone)]
pub struct SpeechModel {
    pub dir: PathBuf,
    /// What generated a track, recorded with it: an image tag, a revision.
    pub label: String,
}

impl SpeechModel {
    pub fn new(dir: impl Into<PathBuf>, label: impl Into<String>) -> Self {
        Self {
            dir: dir.into(),
            label: label.into(),
        }
    }

    /// The files it needs, or which are missing.
    pub fn check(&self) -> Result<()> {
        if !self.dir.is_dir() {
            bail!(
                "the speech model directory {} does not exist",
                self.dir.display()
            );
        }
        let missing: Vec<&str> = REQUIRED_FILES
            .iter()
            .copied()
            .filter(|f| !self.dir.join(f).is_file())
            .collect();
        if !missing.is_empty() {
            bail!(
                "the speech model directory {} is missing {}",
                self.dir.display(),
                missing.join(", ")
            );
        }
        Ok(())
    }
}

/// `generation_config.json`, when the directory has one: where newer
/// checkpoints keep the tokens decoding must never emit.
#[derive(Deserialize, Default)]
struct GenerationConfig {
    #[serde(default)]
    suppress_tokens: Vec<u32>,
    #[serde(default)]
    begin_suppress_tokens: Vec<u32>,
}

/// What to do with the speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Text in the language spoken.
    Transcribe,
    /// English text, whatever was spoken.
    Translate,
}

/// What one window produced.
#[derive(Debug, Clone)]
pub struct WindowResult {
    /// Every segment, text left open at the end running to the window's end.
    pub segments: Vec<Segment>,
    /// Where the last segment the model closed ends, when text was left open
    /// after it: a sentence the window cut. The next window can start there
    /// and hear that sentence whole. `None` when nothing was left open.
    pub closed_end: Option<f64>,
    /// Mean log-probability of the tokens chosen.
    pub avg_logprob: f64,
    /// The model's probability that the window holds no speech.
    pub no_speech_prob: f64,
    /// The language it was transcribed as: the one asked for, or detected.
    pub language: String,
}

/// A loaded model.
pub struct Engine {
    encoder: AudioEncoder,
    decoder: CachedDecoder,
    config: w::Config,
    tokens: Tokens,
    filters: Vec<f32>,
    rules: Rules,
    device: Device,
}

/// Tokens a window's text may run to: half the decoder's context, as
/// Whisper's own decoding allows.
fn max_tokens(cfg: &w::Config) -> usize {
    cfg.max_target_positions / 2
}

impl Engine {
    /// Load `model` on `device`.
    pub fn load(model: &SpeechModel, device: Device) -> Result<Self> {
        model.check()?;
        let read = |name: &str| -> Result<String> {
            let p = model.dir.join(name);
            std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))
        };
        let mut config: w::Config =
            serde_json::from_str(&read("config.json")?).context("config.json")?;
        let tokens = Tokens::load(&model.dir.join("tokenizer.json"))?;
        let generation: GenerationConfig = match model.dir.join("generation_config.json") {
            p if p.is_file() => serde_json::from_str(&read("generation_config.json")?)
                .context("generation_config.json")?,
            _ => GenerationConfig::default(),
        };
        if config.suppress_tokens.is_empty() {
            config.suppress_tokens = generation.suppress_tokens.clone();
        }
        let weights = model.dir.join("model.safetensors");
        // SAFETY: the file is mapped read-only and not modified while the
        // model lives; that is the contract candle's mmap loader asks for.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], w::DTYPE, &device)? };
        let filters = mel::filters(config.num_mel_bins);
        let rules = Rules::new(
            &tokens,
            config.vocab_size,
            &config.suppress_tokens,
            &generation.begin_suppress_tokens,
        );
        let loading = || format!("loading the weights in {}", weights.display());
        // candle's encoder, and the decoder with its self-attention cached
        // (decoder.rs). candle loads only a whole Whisper; its decoder is
        // dropped here.
        let Whisper { encoder, .. } = Whisper::load(&vb, config.clone()).with_context(loading)?;
        let decoder = CachedDecoder::load(vb.pp("model.decoder"), &config).with_context(loading)?;
        Ok(Self {
            encoder,
            decoder,
            config,
            tokens,
            filters,
            rules,
            device,
        })
    }

    pub fn tokens(&self) -> &Tokens {
        &self.tokens
    }

    /// The audio features of up to 30 s of 16 kHz mono.
    fn encode(&mut self, samples: &[f32]) -> Result<Tensor> {
        let n_mels = self.config.num_mel_bins;
        let mel = audio::pcm_to_mel(&self.config, samples, &self.filters);
        let frames = mel.len() / n_mels;
        let mel = Tensor::from_vec(mel, (1, n_mels, frames), &self.device)?;
        // The encoder takes exactly 30 s (3000 frames); pcm_to_mel pads past
        // the audio with at least that much.
        let mel = mel.narrow(2, 0, w::N_FRAMES.min(frames))?;
        Ok(self.encoder.forward(&mel, true)?)
    }

    /// The language most likely spoken, from the encoder's output.
    fn language_of(&mut self, features: &Tensor) -> Result<&'static str> {
        self.decoder.set_audio(features)?;
        let logits = self.decoder.step(&[self.tokens.sot])?;
        let (code, _) = self
            .tokens
            .languages
            .iter()
            .max_by(|a, b| logits[a.1 as usize].total_cmp(&logits[b.1 as usize]))
            .copied()
            .context("no language tokens")?;
        Ok(code)
    }

    /// Up to 30 s of 16 kHz mono to timed segments (seconds from the
    /// window's start), greedily, under Whisper's timestamp rules.
    ///
    /// `language` `None` detects it from the same encoder pass the
    /// transcription uses (the encoder is most of the cost: on large-v3-turbo
    /// a second pass doubled a short clip's time).
    /// `heard` is told each timestamp the model writes, in seconds from the
    /// window's start, as it writes it.
    pub fn transcribe_window(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        task: Task,
        heard: &mut dyn FnMut(f64),
    ) -> Result<WindowResult> {
        let features = self.encode(samples)?;
        let language = match language {
            Some(l) => l.to_string(),
            None => self.language_of(&features)?.to_string(),
        };
        let lang = self
            .tokens
            .language(&language)
            .with_context(|| format!("this model has no language `{language}`"))?;
        let task_token = match task {
            Task::Transcribe => self.tokens.transcribe,
            Task::Translate => self.tokens.translate,
        };
        let prompt = [self.tokens.sot, lang, task_token];
        let mut tokens: Vec<u32> = prompt.to_vec();
        let window_seconds = samples.len() as f64 / mel::SAMPLE_RATE;
        let max = max_tokens(&self.config);
        let mut sum_logprob = 0.0;
        self.decoder.set_audio(&features)?;
        let mut logits = self.decoder.step(&prompt)?;
        let no_speech_prob = self
            .tokens
            .no_speech
            .map_or(0.0, |ns| softmax_at(&logits, ns as usize));
        for _ in 0..max {
            let sampled = &tokens[prompt.len()..];
            self.rules
                .apply(&mut logits, sampled, &self.tokens, window_seconds);
            let next = argmax(&logits) as u32;
            sum_logprob += log_softmax_at(&logits, next as usize);
            tokens.push(next);
            if self.tokens.is_timestamp(next) {
                heard(self.tokens.seconds(next).min(window_seconds));
            }
            if next == self.tokens.eot || self.decoder.room() == 0 {
                break;
            }
            logits = self.decoder.step(&[next])?;
        }
        let sampled = &tokens[prompt.len()..];
        let n = sampled.len().max(1) as f64;
        Ok(WindowResult {
            segments: segments_of(sampled, &self.tokens, window_seconds),
            closed_end: closed_end(sampled, &self.tokens, window_seconds),
            language,
            avg_logprob: sum_logprob / n,
            no_speech_prob,
        })
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn log_sum_exp(v: &[f32]) -> f64 {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let s: f64 = v.iter().map(|&x| f64::from(x - m).exp()).sum();
    f64::from(m) + s.ln()
}

fn log_softmax_at(v: &[f32], i: usize) -> f64 {
    f64::from(v[i]) - log_sum_exp(v)
}

fn softmax_at(v: &[f32], i: usize) -> f64 {
    log_softmax_at(v, i).exp()
}

pub(crate) fn device_or_cpu() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    Device::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_directory_or_file_is_named() {
        let dir = std::env::temp_dir().join("rivet-transcribe-no-such-model");
        let err = SpeechModel::new(&dir, "x").check().unwrap_err();
        assert!(format!("{err:#}").contains("does not exist"));

        let tmp = std::env::temp_dir().join(format!("rivet-transcribe-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("config.json"), "{}").unwrap();
        let err = SpeechModel::new(&tmp, "x").check().unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("tokenizer.json") && msg.contains("model.safetensors"),
            "{msg}"
        );
        assert!(!msg.contains("config.json,"), "{msg}");
        let err = Engine::load(&SpeechModel::new(&tmp, "x"), Device::Cpu)
            .err()
            .unwrap();
        assert!(format!("{err:#}").contains("missing"));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn log_softmax_is_normalised() {
        let v = [1.0f32, 2.0, 3.0];
        let total: f64 = (0..3).map(|i| softmax_at(&v, i)).sum();
        assert!((total - 1.0).abs() < 1e-9);
        assert_eq!(argmax(&v), 2);
    }
}
