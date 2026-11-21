//! Speech to timed text on rivet's decoded-audio hook.
//!
//! [`Transcribe`] is a [`DecodedAudioHook`](rivet::hooks::DecodedAudioHook)
//! that runs a Whisper model, through candle, on 30-second windows of the
//! source's audio, and records timed segments in the job's hook report. It
//! runs in any rivet job, and [`transcribe`] runs it over a file's audio
//! alone.
//!
//! **The crate holds no model and fetches none.** A [`SpeechModel`] is a
//! directory the caller provides, in Hugging Face's layout:
//!
//! | File | Required | What |
//! |------|:-:|------|
//! | `config.json` | ✓ | the checkpoint's sizes |
//! | `tokenizer.json` | ✓ | the vocabulary |
//! | `model.safetensors` | ✓ | the weights, memory-mapped |
//! | `generation_config.json` | | the tokens never to emit, where newer checkpoints keep them |
//!
//! Any Whisper checkpoint loads: tiny to large-v3, large-v3-turbo, a
//! fine-tune. Nothing is downloaded at build or run time, and a directory
//! that is missing a file is refused by name.
//!
//! Which audio the model hears is decided by rivet's
//! [`ActivityGate`](rivet::hooks::ActivityGate): a window with nothing
//! active is skipped, and the quiet inside the rest is silenced, because
//! Whisper fills long silence with invented words.
//!
//! ```no_run
//! use rivet_transcribe::{SpeechModel, Transcribe, TranscribeOptions, transcribe};
//!
//! let model = SpeechModel::new("/opt/models/whisper", "whisper-large-v3-turbo@41f01f3");
//! let transcriber = std::sync::Arc::new(Transcribe::load(&model, TranscribeOptions::default())?);
//! let bytes = std::fs::read("talk.mp4")?;
//! let transcript = transcribe(&bytes, &transcriber)?;
//! for s in transcript.segments {
//!     println!("{:.2} --> {:.2}  {}", s.start, s.end, s.text);
//! }
//! # anyhow::Ok(())
//! ```

mod decode;
mod decoder;
mod hook;
pub mod mel;
mod model;
pub mod tokens;

use std::sync::Arc;

use anyhow::Result;
use rivet::hooks::Hooks;

pub use decode::Segment;
pub use hook::{OnHeard, Transcribe, TranscribeOptions};
pub use model::{REQUIRED_FILES, SpeechModel, Task};

/// What [`transcribe`] heard.
#[derive(Debug, Clone)]
pub struct Transcript {
    /// The language spoken, or `None` when nothing was heard.
    pub language: Option<String>,
    pub segments: Vec<Segment>,
    /// The audio codec of the source.
    pub audio_codec: String,
}

/// The name [`transcribe`] registers its hook under.
pub const HOOK_NAME: &str = "transcribe";

/// Transcribe `input`'s audio: any file rivet reads audio from, including
/// an HLS audio rendition's `init.mp4` followed by its segments.
pub fn transcribe(input: &[u8], transcriber: &Arc<Transcribe>) -> Result<Transcript> {
    let hooks = Hooks::new().decoded_audio(HOOK_NAME, Arc::clone(transcriber));
    let out = rivet::run_audio_hooks_blocking(input, &hooks)?;
    Ok(Transcript {
        language: Transcribe::language(&out.hooks, HOOK_NAME),
        segments: Transcribe::segments(&out.hooks, HOOK_NAME),
        audio_codec: out.audio_codec,
    })
}
