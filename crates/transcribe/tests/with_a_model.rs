//! End to end with real weights, when there are some.
//!
//! Set `RIVET_WHISPER_MODEL` to a model directory and `RIVET_WHISPER_SPEECH`
//! to a file of someone speaking; optionally `RIVET_WHISPER_EXPECT` to words
//! the transcript must contain. Without them the test says so and passes:
//! the crate has no weights of its own, and CI fetches none.

use std::sync::Arc;

use rivet_transcribe::{OnHeard, SpeechModel, Transcribe, TranscribeOptions, transcribe};

#[test]
fn transcribes_speech_with_real_weights() {
    let (Ok(model), Ok(speech)) = (
        std::env::var("RIVET_WHISPER_MODEL"),
        std::env::var("RIVET_WHISPER_SPEECH"),
    ) else {
        eprintln!("RIVET_WHISPER_MODEL / RIVET_WHISPER_SPEECH not set: skipped");
        return;
    };
    let heard = Arc::new(std::sync::Mutex::new(Vec::<f64>::new()));
    let t = Arc::new(
        Transcribe::load(
            &SpeechModel::new(&model, "test"),
            TranscribeOptions {
                on_heard: Some(OnHeard::new({
                    let heard = Arc::clone(&heard);
                    move |s| heard.lock().unwrap().push(s)
                })),
                ..TranscribeOptions::default()
            },
        )
        .expect("the model loads"),
    );
    let bytes = std::fs::read(&speech).expect("the speech file");
    let out = transcribe(&bytes, &t).expect("transcribed");
    assert!(!out.segments.is_empty(), "nothing heard in {speech}");
    assert!(out.language.is_some());
    for pair in out.segments.windows(2) {
        assert!(pair[0].start <= pair[1].start, "{:?}", out.segments);
    }
    for s in &out.segments {
        assert!(s.end > s.start, "{s:?}");
    }
    let text: String = out
        .segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    eprintln!("{text}");

    // Progress was told as the model wrote, more often than once a window,
    // and never past the end of what it heard.
    let heard = heard.lock().unwrap().clone();
    let end = out.segments.last().map_or(0.0, |s| s.end);
    eprintln!(
        "on_heard: {} calls, up to {:.1} s",
        heard.len(),
        heard.iter().cloned().fold(0.0, f64::max)
    );
    assert!(heard.len() > out.segments.len() / 2, "{heard:?}");
    assert!(
        heard.iter().all(|s| *s >= 0.0 && *s <= end + 31.0),
        "{heard:?}"
    );
    if let Ok(expect) = std::env::var("RIVET_WHISPER_EXPECT") {
        assert!(
            text.contains(&expect.to_lowercase()),
            "`{expect}` not in `{text}`"
        );
    }

    // Silence goes nowhere near the model and comes back empty.
    let silence = silent_wav(3.0);
    let quiet = transcribe(&silence, &t).expect("silence");
    assert!(quiet.segments.is_empty(), "{:?}", quiet.segments);
}

/// A 16 kHz mono 16-bit WAV of silence.
fn silent_wav(seconds: f64) -> Vec<u8> {
    let n = (seconds * 16_000.0) as u32;
    let data = n * 2;
    let mut w = Vec::new();
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&16_000u32.to_le_bytes());
    w.extend_from_slice(&32_000u32.to_le_bytes());
    w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data.to_le_bytes());
    w.resize(w.len() + data as usize, 0);
    w
}
