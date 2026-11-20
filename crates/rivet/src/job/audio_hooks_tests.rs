//! The decoded-audio hooks: what they are handed, through a transcode and
//! through `run_audio_hooks`, and the activity gate over real decoded audio.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use bytes::Bytes;

use crate::hooks::{
    ActivityGate, AudioDelivery, AudioEvent, DecodedAudioHook, HookContext, HookOutcome, Hooks,
    Stage, StageSet,
};
use crate::progress::NullSink;
use crate::spec::{Container, OutputSpec};

const RATE: u32 = 48_000;

/// A bare `.mp3` from the workspace's own encoder, 48 kHz mono: `quiet`
/// seconds of silence, `tone` seconds of a 440 Hz tone at -8 dBFS, `quiet`
/// seconds of silence.
fn quiet_tone_quiet(quiet: f32, tone: f32) -> Vec<u8> {
    let n = |s: f32| (s * RATE as f32) as usize;
    let mut samples = vec![0.0f32; n(quiet)];
    samples.extend(
        (0..n(tone))
            .map(|i| 0.4 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / RATE as f32).sin()),
    );
    samples.extend(vec![0.0f32; n(quiet)]);
    let mut enc = codec::audio::create_encoder(codec::audio::AudioEncoderConfig::new(
        codec::audio::AudioCodec::Mp3,
        RATE,
        1,
        64_000,
    ))
    .unwrap();
    let mut packets = enc
        .encode(&codec::audio::AudioFrame {
            samples,
            sample_rate: RATE,
            channels: 1,
            pts: 0,
        })
        .unwrap();
    packets.extend(enc.flush().unwrap());
    packets.into_iter().flat_map(|p| p.data).collect()
}

/// Keeps every chunk it is handed.
#[derive(Clone)]
struct Recorder {
    delivery: AudioDelivery,
    chunks: Arc<Mutex<Vec<AudioEvent>>>,
}

impl Recorder {
    fn new(delivery: AudioDelivery) -> Self {
        Self {
            delivery,
            chunks: Arc::default(),
        }
    }
    fn chunks(&self) -> Vec<AudioEvent> {
        self.chunks.lock().unwrap().clone()
    }
}

impl DecodedAudioHook for Recorder {
    fn delivery(&self) -> AudioDelivery {
        self.delivery
    }
    fn on_decoded_audio(&self, _: &HookContext, audio: &AudioEvent) -> Result<HookOutcome> {
        self.chunks.lock().unwrap().push(audio.clone());
        Ok(HookOutcome::proceed())
    }
}

/// The chunks join end to end, from zero, and add up to `seconds`. A bare
/// MP3 with no gapless header plays its encoder's padding too (1728 samples
/// at 48 kHz, 36 ms, from this encoder), hence the tolerance.
fn assert_contiguous(chunks: &[AudioEvent], seconds: f64) {
    assert!(!chunks.is_empty(), "no chunks");
    let mut at = 0.0;
    for (i, c) in chunks.iter().enumerate() {
        assert_eq!(c.index, i as u64);
        assert!(
            (c.seconds - at).abs() < 1e-9,
            "chunk {i} starts at {} not {at}",
            c.seconds
        );
        at += c.duration();
    }
    assert!(
        (at - seconds).abs() < 0.05,
        "{at} seconds delivered, {seconds} expected"
    );
}

#[test]
fn the_stage_parses_and_fits_in_a_set() {
    assert_eq!(
        "decoded-audio".parse::<Stage>().unwrap(),
        Stage::DecodedAudio
    );
    assert_eq!("audio".parse::<Stage>().unwrap(), Stage::DecodedAudio);
    let set = StageSet::parse("source,decoded-audio,failed").unwrap();
    assert!(set.contains(Stage::DecodedAudio) && set.contains(Stage::Failed));
    assert!(!set.contains(Stage::Still));
    assert_eq!(StageSet::ALL.iter().count(), Stage::ALL.len());
}

#[test]
fn each_hook_gets_its_own_chunks_at_its_own_rate() {
    let mp3 = quiet_tone_quiet(1.0, 1.0);
    let native = Recorder::new(AudioDelivery::chunks_of(0.5));
    let speech = Recorder::new(AudioDelivery::speech());
    let hooks = Hooks::new()
        .decoded_audio("native", native.clone())
        .decoded_audio("speech", speech.clone());
    let out = crate::job::run_audio_hooks_blocking(&mp3, &hooks).unwrap();
    assert_eq!(out.audio_codec, "mp3");

    let n = native.chunks();
    assert!(n.iter().all(|c| c.sample_rate == RATE && c.channels == 1));
    // Six whole half-seconds, and the padding as a short seventh.
    assert_eq!(n.len(), 7);
    assert!(n[..6].iter().all(|c| c.frames() == 24_000));
    assert!(n[6].frames() < 24_000);
    assert_contiguous(&n, 3.0);

    let s = speech.chunks();
    assert!(s.iter().all(|c| c.sample_rate == 16_000 && c.channels == 1));
    assert_eq!(s.len(), 1, "three seconds is one 30-second chunk");
    assert_contiguous(&s, 3.0);

    // One report record per chunk handed over.
    assert_eq!(out.hooks.by_hook("native").count(), 7);
    assert_eq!(out.hooks.by_hook("speech").count(), 1);
}

#[test]
fn the_activity_gate_finds_the_tone_and_not_the_silence() {
    let mp3 = quiet_tone_quiet(1.0, 1.0);
    let gate = ActivityGate::new();
    let hooks = Hooks::new().decoded_audio("gate", gate);
    let out = crate::job::run_audio_hooks_blocking(&mp3, &hooks).unwrap();
    let ranges = gate.ranges(&out.hooks, "gate");
    assert_eq!(ranges.len(), 1, "{ranges:?}");
    let r = ranges[0];
    assert!((r.start - (1.0 - gate.padding)).abs() < 0.06, "{r:?}");
    assert!((r.end - (2.0 + gate.padding)).abs() < 0.06, "{r:?}");
    let total: f64 = out
        .hooks
        .by_hook("gate")
        .filter_map(|r| r.annotation("active_seconds"))
        .filter_map(serde_json::Value::as_f64)
        .sum();
    assert!((total - r.duration()).abs() < 1e-6);
}

#[test]
fn the_gate_joins_a_range_that_two_chunks_split() {
    // Chunks of 1.5 s cut the tone (1.0 to 2.0 s) in two.
    let gate = ActivityGate::new();
    let samples = |from: f64, to: f64| -> Vec<f32> {
        (0..((to - from) * 16_000.0) as usize)
            .map(|i| {
                let t = from + i as f64 / 16_000.0;
                if (1.0..2.0).contains(&t) {
                    (0.4 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()) as f32
                } else {
                    0.0
                }
            })
            .collect()
    };
    let a = gate.detect(&samples(0.0, 1.5), 16_000, 0.0);
    let b = gate.detect(&samples(1.5, 3.0), 16_000, 1.5);
    assert_eq!((a.len(), b.len()), (1, 1));
    let mut report = crate::hooks::HookReport::default();
    for (i, ranges) in [a, b].into_iter().enumerate() {
        let list: Vec<serde_json::Value> = ranges
            .iter()
            .map(|r| serde_json::json!([r.start, r.end]))
            .collect();
        report.records.push(crate::hooks::HookRecord {
            hook: "gate".into(),
            kind: crate::hooks::HookKind::DecodedAudio,
            stage: Stage::DecodedAudio,
            subject: crate::hooks::Subject::Audio {
                clip: 0,
                seconds: i as f64 * 1.5,
                duration: 1.5,
            },
            verdict: Some(crate::hooks::Verdict::Continue),
            annotations: vec![crate::hooks::Annotation {
                key: "active".into(),
                value: list.into(),
            }],
            error: None,
            elapsed: std::time::Duration::ZERO,
            background: false,
        });
    }
    let joined = gate.ranges(&report, "gate");
    assert_eq!(joined.len(), 1, "{joined:?}");
    assert!(joined[0].start < 1.0 && joined[0].end > 2.0);
}

#[test]
fn silence_is_never_active() {
    let gate = ActivityGate::new();
    assert!(gate.detect(&vec![0.0; 48_000], 16_000, 0.0).is_empty());
    // Room tone at -60 dBFS: under the floor however quiet the background.
    let hiss: Vec<f32> = (0..48_000u32)
        .map(|i| 0.001 * (((i.wrapping_mul(2_654_435_761)) >> 16) as f32 / 32_768.0 - 1.0))
        .collect();
    assert!(gate.detect(&hiss, 16_000, 0.0).is_empty());
}

#[test]
fn a_passthrough_is_decoded_for_the_hooks_and_left_byte_identical() {
    let mp3 = Bytes::from(quiet_tone_quiet(0.5, 0.5));
    let spec = OutputSpec::audio_only_in(Container::Mp3);
    let plain =
        crate::job::run_job_blocking(&mp3, &spec, None, Arc::new(NullSink)).expect("plain job");
    let rec = Recorder::new(AudioDelivery::chunks_of(0.25));
    let hooked_spec = spec
        .clone()
        .with_hooks(Hooks::new().decoded_audio("rec", rec.clone()));
    let hooked = crate::job::run_job_blocking(&mp3, &hooked_spec, None, Arc::new(NullSink))
        .expect("hooked job");
    assert!(
        plain.audio_handling.contains("passthrough"),
        "expected a passthrough: {}",
        plain.audio_handling
    );
    assert_eq!(plain.rungs.len(), hooked.rungs.len());
    for (a, b) in plain.rungs.iter().zip(&hooked.rungs) {
        match (&a.artifact, &b.artifact) {
            (crate::job::RungArtifact::File(x), crate::job::RungArtifact::File(y)) => {
                assert!(x == y, "the hook changed the output")
            }
            _ => panic!("expected files"),
        }
    }
    assert_contiguous(&rec.chunks(), 1.5);
    assert_eq!(hooked.hooks.by_hook("rec").count(), rec.chunks().len());
}

#[test]
fn a_set_with_no_audio_hook_is_refused() {
    let err = crate::job::run_audio_hooks_blocking(&quiet_tone_quiet(0.1, 0.1), &Hooks::new())
        .unwrap_err();
    assert!(format!("{err:#}").contains("decoded-audio hook"));
}

#[test]
fn an_hls_audio_rendition_is_read_as_its_init_and_segments_joined() {
    use super::audio::{AudioOutput, AudioRequest, build_audio_rendition, prepare_audio};
    use crate::spec::AudioCodecPolicy;
    // AAC 5.1, passed through into CMAF: the rendition a video job writes.
    let path = format!(
        "{}/tests/data/audio/tones_51_aac.m4a",
        env!("CARGO_MANIFEST_DIR")
    );
    let source = Bytes::from(std::fs::read(&path).unwrap());
    let src = container::streaming::demux_audio(source.clone())
        .unwrap()
        .unwrap();
    let req = AudioRequest {
        output: AudioOutput::Cmaf,
        ..AudioRequest::plain(AudioCodecPolicy::Auto)
    };
    let prepared = prepare_audio(Some(&src.track), src.edit, &src.gaps, req)
        .unwrap()
        .unwrap();
    assert!(
        prepared.handling.contains("passthrough"),
        "{}",
        prepared.handling
    );
    let dir = tempfile::tempdir().unwrap();
    build_audio_rendition(dir.path(), &prepared, 0.5, "audio", "Audio")
        .unwrap()
        .unwrap();
    let audio_dir = dir.path().join("audio");
    let mut segments: Vec<_> = std::fs::read_dir(&audio_dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "m4s"))
        .collect();
    segments.sort();
    assert!(segments.len() > 1, "want several segments: {segments:?}");
    let mut joined = std::fs::read(audio_dir.join("init.mp4")).unwrap();
    for s in &segments {
        joined.extend(std::fs::read(s).unwrap());
    }

    let heard = |bytes: &[u8]| {
        let rec = Recorder::new(AudioDelivery::speech());
        let hooks = Hooks::new().decoded_audio("rec", rec.clone());
        let out = crate::job::run_audio_hooks_blocking(bytes, &hooks).unwrap();
        assert_eq!(out.audio_codec, "aac");
        let chunks = rec.chunks();
        let total: f64 = chunks.iter().map(AudioEvent::duration).sum();
        (total, chunks)
    };
    let (from_file, _) = heard(&source);
    let (from_rendition, chunks) = heard(&joined);
    assert!(
        chunks
            .iter()
            .all(|c| c.sample_rate == 16_000 && c.channels == 1)
    );
    assert!(
        (from_file - from_rendition).abs() < 0.03,
        "the file is {from_file}s, the rendition {from_rendition}s"
    );
}
