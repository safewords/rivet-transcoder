//! Whisper's tokens, from the model's `tokenizer.json`: what each id is, the
//! special tokens decoding steers with, and text out of ids.
//!
//! Only decoding is needed — the prompt is special tokens, looked up by name
//! — so this reads the vocabulary and reverses GPT-2's byte-level encoding
//! rather than bringing in a BPE implementation.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// Whisper's language codes, in its own order (large-v3 added `yue`).
pub const LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// Seconds per timestamp token.
pub const TIMESTAMP_STEP: f64 = 0.02;

#[derive(Deserialize)]
struct TokenizerJson {
    model: ModelJson,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
}

#[derive(Deserialize)]
struct ModelJson {
    vocab: HashMap<String, u32>,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
}

/// The vocabulary and the ids decoding needs.
#[derive(Debug, Clone)]
pub struct Tokens {
    by_id: HashMap<u32, String>,
    by_text: HashMap<String, u32>,
    pub sot: u32,
    pub eot: u32,
    pub transcribe: u32,
    pub translate: u32,
    pub no_timestamps: u32,
    /// `<|nospeech|>` or, in older vocabularies, `<|nocaptions|>`.
    pub no_speech: Option<u32>,
    /// `<|0.00|>`: every id from here up is a timestamp, 20 ms apart.
    pub timestamp_begin: u32,
    /// `(code, id)` for each language the vocabulary has.
    pub languages: Vec<(&'static str, u32)>,
}

impl Tokens {
    /// From a `tokenizer.json` file.
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("{}", path.display()))
    }

    /// From a `tokenizer.json` document.
    pub fn parse(json: &str) -> Result<Self> {
        let doc: TokenizerJson = serde_json::from_str(json).context("not a tokenizer.json")?;
        let mut by_text = doc.model.vocab;
        for t in doc.added_tokens {
            by_text.insert(t.content, t.id);
        }
        let by_id: HashMap<u32, String> = by_text.iter().map(|(t, &i)| (i, t.clone())).collect();
        let need = |name: &str| -> Result<u32> {
            by_text
                .get(name)
                .copied()
                .with_context(|| format!("the vocabulary has no {name}: not a Whisper tokenizer"))
        };
        let no_timestamps = need("<|notimestamps|>")?;
        // Whisper's timestamps follow `<|notimestamps|>`; a tokenizer.json
        // that lists them names the first.
        let timestamp_begin = by_text
            .get("<|0.00|>")
            .copied()
            .unwrap_or(no_timestamps + 1);
        let languages: Vec<(&'static str, u32)> = LANGUAGES
            .iter()
            .filter_map(|&code| by_text.get(&format!("<|{code}|>")).map(|&id| (code, id)))
            .collect();
        if languages.is_empty() {
            bail!("the vocabulary has no language tokens: not a Whisper tokenizer");
        }
        Ok(Self {
            sot: need("<|startoftranscript|>")?,
            eot: need("<|endoftext|>")?,
            transcribe: need("<|transcribe|>")?,
            translate: need("<|translate|>")?,
            no_timestamps,
            no_speech: by_text
                .get("<|nospeech|>")
                .or_else(|| by_text.get("<|nocaptions|>"))
                .copied(),
            timestamp_begin,
            languages,
            by_id,
            by_text,
        })
    }

    /// The id of `<|code|>`, for a language the vocabulary has.
    pub fn language(&self, code: &str) -> Option<u32> {
        self.languages
            .iter()
            .find(|(c, _)| c.eq_ignore_ascii_case(code))
            .map(|&(_, id)| id)
    }

    pub fn id(&self, text: &str) -> Option<u32> {
        self.by_text.get(text).copied()
    }

    pub fn is_timestamp(&self, id: u32) -> bool {
        id >= self.timestamp_begin
    }

    /// A timestamp token's time in seconds.
    pub fn seconds(&self, id: u32) -> f64 {
        f64::from(id.saturating_sub(self.timestamp_begin)) * TIMESTAMP_STEP
    }

    /// The text of `ids`, special tokens left out.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        let table = unicode_to_byte();
        for &id in ids {
            if id >= self.eot {
                // Every special, language and timestamp token comes after
                // `<|endoftext|>` in Whisper's vocabularies.
                continue;
            }
            let Some(piece) = self.by_id.get(&id) else {
                continue;
            };
            for ch in piece.chars() {
                match table.get(&ch) {
                    Some(&b) => bytes.push(b),
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// GPT-2's byte-level encoding reversed: the printable character each byte
/// was stored as, back to the byte.
fn unicode_to_byte() -> &'static HashMap<char, u8> {
    static TABLE: std::sync::OnceLock<HashMap<char, u8>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
        let mut table = HashMap::new();
        let mut shifted = 0u32;
        for b in 0u8..=255 {
            let ch = if printable(b) {
                char::from(b)
            } else {
                let c = char::from_u32(256 + shifted).expect("valid");
                shifted += 1;
                c
            };
            table.insert(ch, b);
        }
        table
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tokenizer.json with Whisper's shape and a handful of words.
    pub(crate) fn tiny_tokenizer_json() -> String {
        let mut vocab = serde_json::Map::new();
        for (i, w) in ["Hello", "Ġworld", "!", "Ġcaf", "Ã©", "Ġ"]
            .iter()
            .enumerate()
        {
            vocab.insert((*w).into(), (i as u32).into());
        }
        let mut added = vec![
            (100, "<|endoftext|>"),
            (101, "<|startoftranscript|>"),
            (102, "<|en|>"),
            (103, "<|de|>"),
            (104, "<|translate|>"),
            (105, "<|transcribe|>"),
            (106, "<|nospeech|>"),
            (107, "<|notimestamps|>"),
        ]
        .into_iter()
        .map(|(id, c)| serde_json::json!({ "id": id, "content": c, "special": true }))
        .collect::<Vec<_>>();
        added.push(serde_json::json!({ "id": 108, "content": "<|0.00|>", "special": true }));
        serde_json::json!({ "model": { "vocab": vocab }, "added_tokens": added }).to_string()
    }

    #[test]
    fn finds_the_special_tokens() {
        let t = Tokens::parse(&tiny_tokenizer_json()).unwrap();
        assert_eq!(
            (t.eot, t.sot, t.transcribe, t.translate),
            (100, 101, 105, 104)
        );
        assert_eq!(
            (t.no_timestamps, t.timestamp_begin, t.no_speech),
            (107, 108, Some(106))
        );
        assert_eq!(t.language("en"), Some(102));
        assert_eq!(t.language("DE"), Some(103));
        assert_eq!(t.language("fr"), None);
        assert!(t.is_timestamp(108) && t.is_timestamp(158));
        assert!((t.seconds(158) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn decodes_byte_level_text_and_skips_specials() {
        let t = Tokens::parse(&tiny_tokenizer_json()).unwrap();
        assert_eq!(t.decode(&[108, 0, 1, 2, 120]), "Hello world!");
        assert_eq!(t.decode(&[3, 4]), " café");
        assert_eq!(t.decode(&[101, 102, 105]), "");
    }

    #[test]
    fn refuses_a_tokenizer_that_is_not_whispers() {
        let err = Tokens::parse(r#"{"model":{"vocab":{"a":0}}}"#).unwrap_err();
        assert!(format!("{err:#}").contains("not a Whisper tokenizer"));
        assert!(Tokens::parse("{}").is_err());
    }
}
