//! Transcribe a file's audio and print it as WebVTT.
//!
//! ```sh
//! cargo run --release -p rivet-transcribe --example transcribe -- \
//!     --model /path/to/whisper-dir [--language en] [--translate] input.mp4
//! ```

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rivet_transcribe::{SpeechModel, Task, Transcribe, TranscribeOptions, transcribe};

fn stamp(seconds: f64) -> String {
    let ms = (seconds * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut model, mut input) = (None, None);
    let mut options = TranscribeOptions::default();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next(),
            "--language" => options.language = args.next(),
            "--translate" => options.task = Task::Translate,
            _ if input.is_none() => input = Some(a),
            other => bail!("unexpected argument `{other}`"),
        }
    }
    let model = model.context("--model DIR is required")?;
    let input = input.context("an input file is required")?;
    let loading = Instant::now();
    let transcriber = Arc::new(Transcribe::load(
        &SpeechModel::new(&model, &model),
        options,
    )?);
    eprintln!("loaded {model} in {:.1}s", loading.elapsed().as_secs_f64());
    let bytes = std::fs::read(&input).with_context(|| format!("reading {input}"))?;
    let running = Instant::now();
    let t = transcribe(&bytes, &transcriber)?;
    eprintln!(
        "{} audio, language {}, {} segments in {:.1}s",
        t.audio_codec,
        t.language.as_deref().unwrap_or("none"),
        t.segments.len(),
        running.elapsed().as_secs_f64()
    );
    println!("WEBVTT\n");
    for s in &t.segments {
        println!("{} --> {}\n{}\n", stamp(s.start), stamp(s.end), s.text);
    }
    Ok(())
}
