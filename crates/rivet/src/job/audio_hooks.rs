//! A job whose only product is what its hooks say about the audio:
//! [`run_audio_hooks`].

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use bytes::Bytes;
use container::streaming;

use super::audio_tap::{Tapped, tap_audio};
use crate::hooks::{HookReport, Hooks, JobKind, MediaSummary, Stage};
use crate::spec::{AudioDecodeDeny, HeAacPolicy};

/// What [`run_audio_hooks`] returns.
#[derive(Debug, Clone)]
pub struct AudioHooksOutput {
    /// The source's audio codec (`aac`, `opus`, ...).
    pub audio_codec: String,
    /// Everything the hooks said.
    pub hooks: HookReport,
    pub elapsed: Duration,
}

/// Run `hooks` over `input`'s audio and write nothing.
///
/// The input is anything rivet reads audio from: a media file, an audio
/// file, or an HLS audio rendition's `init.mp4` followed by its segments (a
/// fragmented MP4). Source and probe hooks run as in any job, then the audio
/// is decoded and handed to the decoded-audio hooks ([`Hooks::decoded_audio`])
/// chunk by chunk, then the completed hooks — or the failed hooks, when a
/// hook rejects the job or the audio cannot be decoded.
///
/// For the analyses that need the sound and nothing else: where it is active
/// ([`ActivityGate`](crate::hooks::ActivityGate)), a speech model, a loudness
/// measurement. A set with no decoded-audio hook is refused: there
/// would be nothing for the job to do.
pub async fn run_audio_hooks(input: Bytes, hooks: &Hooks) -> Result<AudioHooksOutput> {
    if !hooks.wants(Stage::DecodedAudio) {
        bail!("run_audio_hooks needs at least one decoded-audio hook");
    }
    let started = Instant::now();
    let _slot = crate::thread_budget::enter_job();
    let hooks = hooks.ensure_session(JobKind::AudioAnalysis);
    let session = hooks.clone();
    let run = async move {
        let source = input.clone();
        session.offload(move |h| h.emit_source(0, &source)).await?;
        let h = session.clone();
        tokio::task::spawn_blocking(move || analyse(&h, input))
            .await
            .map_err(|e| anyhow::anyhow!("audio hooks task: {e}"))?
    };
    let audio_codec = hooks.run(|_: &String| Vec::new(), run).await?;
    Ok(AudioHooksOutput {
        audio_codec,
        hooks: hooks.report(),
        elapsed: started.elapsed(),
    })
}

/// [`run_audio_hooks`] on a runtime of its own, for callers without one.
pub fn run_audio_hooks_blocking(input: &[u8], hooks: &Hooks) -> Result<AudioHooksOutput> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building Tokio runtime")?;
    rt.block_on(run_audio_hooks(Bytes::copy_from_slice(input), hooks))
}

fn analyse(hooks: &Hooks, input: Bytes) -> Result<String> {
    let head = input.slice(..input.len().min(64));
    let src = streaming::demux_audio(input)
        .context("demux")?
        .context("the input has no audio track this build reads")?;
    let codec = src.track.codec.to_ascii_lowercase();
    hooks.emit_probe(
        0,
        MediaSummary {
            container: container::sniff_container(&head).label().to_string(),
            audio_codec: Some(codec.clone()),
            ..Default::default()
        },
    )?;
    match tap_audio(
        hooks,
        0,
        Some(&src.track),
        src.edit,
        &src.gaps,
        HeAacPolicy::Auto,
        AudioDecodeDeny::NONE,
    )? {
        Tapped::Undecodable => bail!(
            "the {codec} track cannot be decoded by this build, so the audio hooks could not hear all of it"
        ),
        Tapped::NotWanted if src.track.samples.is_empty() => {
            bail!("the {codec} track has no packets rivet can read")
        }
        _ => Ok(codec),
    }
}
