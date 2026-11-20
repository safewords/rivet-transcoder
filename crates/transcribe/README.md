# rivet-transcribe

Speech to timed text on rivet's decoded-audio hook: a Whisper model run
through [candle](https://github.com/huggingface/candle) on 30-second windows
of a source's audio. It runs inside any rivet job as a hook, or over a file's
audio alone with `transcribe`.

It is a crate of its own, not part of rivet, as
[decisions §33](../../docs/decisions.md) asks of model inference.

## The model is not in this crate

The crate holds no weights and downloads none, at build time or at run time.
A model is a directory you provide, in Hugging Face's layout:

| File | Required | What |
|------|:-:|------|
| `config.json` | yes | the checkpoint's sizes |
| `tokenizer.json` | yes | the vocabulary |
| `model.safetensors` | yes | the weights, memory-mapped |
| `generation_config.json` | no | the tokens never to emit, where newer checkpoints keep them; include it when the checkpoint has one |

Any Whisper checkpoint loads from the same code (tiny to large-v3,
large-v3-turbo, a fine-tune), because the sizes differ only in the numbers in
`config.json`. A directory missing a required file is refused, naming the
file. Where the directory comes from (a mounted image, a volume) is the
deployment's business; pin the checkpoint to a revision and record it as the
`SpeechModel`'s label, which is written into every record.

The mel filter bank Whisper needs is computed (`mel.rs`), not shipped, and
the tokenizer is read directly from `tokenizer.json` (`tokens.rs`): no BPE
library and no C dependency.

## Use

```rust
use std::sync::Arc;
use rivet_transcribe::{SpeechModel, Transcribe, TranscribeOptions, transcribe};

let model = SpeechModel::new("/opt/models/whisper", "whisper-large-v3-turbo@<revision>");
let t = Arc::new(Transcribe::load(&model, TranscribeOptions::default())?);
let transcript = transcribe(&std::fs::read("talk.mp4")?, &t)?;
```

In a transcode, register it like any decoded-audio hook, in the background so
the ladder does not wait on inference:

```rust
let hooks = rivet::hooks::Hooks::new()
    .decoded_audio_with("transcribe", t.clone(), rivet::hooks::HookPolicy::background());
```

`Transcribe::segments(&report, "transcribe")` reads the segments back.

From the command line:

```sh
cargo run --release -p rivet-transcribe --example transcribe -- --model DIR input.mp4
```

## How it decides what to hear

rivet's `ActivityGate` runs on each window first. A window with nothing
active never reaches the model, and the quiet inside the rest is silenced,
because Whisper fills long silence with invented words. A window the model
itself judges speechless (no-speech probability over 0.6 with a mean token
log-probability under -1.0, Whisper's own test) records no segments.

Decoding is greedy under Whisper's timestamp rules (`decode.rs`), with the
decoder's self-attention cached (`decoder.rs`) so each step runs one token:
candle's own decoder re-runs the whole text every step, which on
whisper-tiny took 28.5 s for 62 s of speech, against 2.6 s cached.

A sentence a window cuts is decoded again from its start in the next window:
the audio after the last finished sentence is carried over, as Whisper's own
long-form decoding seeks. A sentence counts as cut when the model left it
open, or when the audio is still active at the window's end even though the
model closed it (whisper-tiny closes a cut sentence a moment before the
boundary). Carried audio and the detected language are kept per job and per
source, so one `Transcribe` can serve several jobs.

Measured on this laptop's CPU (Apple silicon, release build) with
whisper-tiny: 52 s of synthetic speech in 4.0 s, every sentence whole.
large-v3-turbo has not been measured.

Not done yet: temperature fallback for windows that decode badly.

## Tests

`cargo test -p rivet-transcribe` needs no weights. The end-to-end test runs
when `RIVET_WHISPER_MODEL` (a model directory) and `RIVET_WHISPER_SPEECH` (a
file of speech) are set, and checks `RIVET_WHISPER_EXPECT` is in the
transcript when that is set too; otherwise it says it was skipped.

## Features

`cuda` and `metal` run the model on a GPU when one is there; the CPU
otherwise.
