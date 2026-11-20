//! [`ActivityGate`]: where in a source's audio something is happening, as
//! opposed to silence or background.

use anyhow::Result;
use serde_json::{Value, json};

use super::audio::{dbfs, rms};
use super::{AudioDelivery, AudioEvent, DecodedAudioHook, HookContext, HookOutcome, HookReport};

/// A stretch of audio, in seconds from the start of the source's audio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivityRange {
    pub start: f64,
    pub end: f64,
}

impl ActivityRange {
    pub fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }
}

/// An activity gate: the stretches of a source's audio loud enough, against
/// that stretch's own background, to hold something — speech above all.
///
/// It is a gate, not a classifier. Anything voiced and louder than the
/// background passes — singing, shouting, music with a vocal — and what it
/// leaves out is the quiet: silence, room tone, a fan. That is what it is
/// for: a speech model handed only these ranges does not spend time on
/// silence, and does not invent words to fill it, which is what speech
/// models do with long quiet stretches. Whether a range really holds words
/// is the model's call.
///
/// Registered as a [`DecodedAudioHook`]
/// ([`Hooks::decoded_audio`](super::Hooks::decoded_audio)), it takes 16 kHz
/// mono in 30-second chunks and annotates each chunk with
/// - `active`: its ranges, `[[start, end], ...]` in seconds from the start of
///   the source's audio, and
/// - `active_seconds`: their total.
///
/// [`ranges`](Self::ranges) joins them across chunks from a job's
/// [`HookReport`].
///
/// # How
///
/// The audio is cut into 30 ms frames and each frame's RMS level taken in
/// dBFS (digital silence counts as -120). A chunk's background is the 10th
/// percentile of its frames' levels, but never above `max_background_dbfs`:
/// a chunk that is loud all the way through (music, a crowd) has no quiet to
/// measure against, and is passed as active rather than hidden. A frame is
/// active when it is `margin_db` above the background, and above
/// `floor_dbfs` in any case. Active frames closer than `hangover` seconds
/// are joined, runs shorter than `min_active` dropped, and each range
/// widened by `padding` on both sides so a word's soft start and end are
/// kept.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivityGate {
    /// How far above a chunk's background a frame must be, in dB.
    pub margin_db: f32,
    /// The level below which nothing is active, in dBFS.
    pub floor_dbfs: f32,
    /// The loudest a chunk's background is taken to be, in dBFS.
    pub max_background_dbfs: f32,
    /// Gaps shorter than this, in seconds, are joined.
    pub hangover: f64,
    /// Runs shorter than this, in seconds, are dropped.
    pub min_active: f64,
    /// Seconds added before and after each range.
    pub padding: f64,
}

/// Frame length, seconds.
const FRAME: f64 = 0.030;

/// The level digital silence is counted at.
const SILENCE_DBFS: f32 = -120.0;

impl Default for ActivityGate {
    fn default() -> Self {
        Self {
            margin_db: 12.0,
            floor_dbfs: -50.0,
            max_background_dbfs: -40.0,
            hangover: 0.35,
            min_active: 0.25,
            padding: 0.15,
        }
    }
}

impl ActivityGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// The ranges in `samples` (mono at `rate`), which start at `offset`
    /// seconds.
    pub fn detect(&self, samples: &[f32], rate: u32, offset: f64) -> Vec<ActivityRange> {
        if rate == 0 || samples.is_empty() {
            return Vec::new();
        }
        let frame = ((FRAME * f64::from(rate)).round() as usize).max(1);
        let levels: Vec<f32> = samples
            .chunks(frame)
            .map(|f| dbfs(rms(f)).max(SILENCE_DBFS))
            .collect();
        let mut sorted = levels.clone();
        sorted.sort_by(f32::total_cmp);
        let background = sorted[sorted.len() / 10].min(self.max_background_dbfs);
        let threshold = (background + self.margin_db).max(self.floor_dbfs);
        let seconds = |i: usize| offset + (i * frame) as f64 / f64::from(rate);
        let end_of = |i: usize| offset + ((i * frame).min(samples.len())) as f64 / f64::from(rate);

        let mut runs: Vec<ActivityRange> = Vec::new();
        for (i, &level) in levels.iter().enumerate() {
            if level <= threshold {
                continue;
            }
            let (start, end) = (seconds(i), end_of(i + 1));
            match runs.last_mut() {
                Some(last) if start - last.end < self.hangover => last.end = end,
                _ => runs.push(ActivityRange { start, end }),
            }
        }
        let limit = end_of(levels.len());
        runs.into_iter()
            .filter(|r| r.duration() >= self.min_active)
            .map(|r| ActivityRange {
                start: (r.start - self.padding).max(offset),
                end: (r.end + self.padding).min(limit),
            })
            .collect()
    }

    /// Every range the hook registered as `hook` recorded in `report`, in
    /// order, joined where chunks split one or `hangover` separates them.
    pub fn ranges(&self, report: &HookReport, hook: &str) -> Vec<ActivityRange> {
        let mut all: Vec<ActivityRange> = report
            .by_hook(hook)
            .filter_map(|r| r.annotation("active"))
            .filter_map(Value::as_array)
            .flatten()
            .filter_map(|pair| {
                let pair = pair.as_array()?;
                Some(ActivityRange {
                    start: pair.first()?.as_f64()?,
                    end: pair.get(1)?.as_f64()?,
                })
            })
            .collect();
        all.sort_by(|a, b| a.start.total_cmp(&b.start));
        let mut joined: Vec<ActivityRange> = Vec::with_capacity(all.len());
        for r in all {
            match joined.last_mut() {
                Some(last) if r.start - last.end < self.hangover => last.end = last.end.max(r.end),
                _ => joined.push(r),
            }
        }
        joined
    }
}

impl DecodedAudioHook for ActivityGate {
    fn delivery(&self) -> AudioDelivery {
        AudioDelivery::speech()
    }

    fn on_decoded_audio(&self, _ctx: &HookContext, audio: &AudioEvent) -> Result<HookOutcome> {
        // Delivered mono; a caller registering it through the general trait
        // with another delivery still gets a mono view.
        let mono;
        let samples: &[f32] = if audio.channels == 1 {
            &audio.samples
        } else {
            mono = super::audio::mono(&audio.samples, audio.channels);
            &mono
        };
        let ranges = self.detect(samples, audio.sample_rate, audio.seconds);
        let total: f64 = ranges.iter().map(ActivityRange::duration).sum();
        let list: Vec<Value> = ranges.iter().map(|r| json!([r.start, r.end])).collect();
        Ok(HookOutcome::proceed()
            .annotate("active", list)
            .annotate("active_seconds", total))
    }

    fn describe(&self) -> String {
        format!(
            "activity gate ( {} dB over background, floor {} dBFS)",
            self.margin_db, self.floor_dbfs
        )
    }
}
