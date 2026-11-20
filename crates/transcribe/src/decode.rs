//! Greedy decoding's rules, and timed segments out of the tokens chosen.
//!
//! The rules are Whisper's own (its `SuppressTokens`, `SuppressBlank` and
//! `ApplyTimestampRules` logit filters), applied to one step's logits:
//!
//! - tokens the model must never emit (the checkpoint's suppress list, the
//!   task and language tokens, `<|notimestamps|>`) are masked;
//! - the first token may not be a blank or the end;
//! - text starts with a timestamp, no later than 1 s in;
//! - timestamps come in pairs: after `<|a|> text <|b|>` comes another
//!   timestamp or the end, never more text without one; after a pair comes
//!   text;
//! - time never runs backwards, nor past the window;
//! - when the timestamps together are likelier than any one text token, a
//!   timestamp is taken.

use crate::tokens::{TIMESTAMP_STEP, Tokens};

/// One stretch of speech: `[start, end)` seconds and what was said.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// The masks for one model's vocabulary.
#[derive(Debug, Clone)]
pub struct Rules {
    suppress: Vec<u32>,
    begin_suppress: Vec<u32>,
}

/// The latest the first timestamp may be, in seconds.
const MAX_INITIAL_TIMESTAMP: f64 = 1.0;

impl Rules {
    pub fn new(
        tokens: &Tokens,
        vocab_size: usize,
        suppress_tokens: &[u32],
        begin_suppress_tokens: &[u32],
    ) -> Self {
        let mut suppress: Vec<u32> = suppress_tokens
            .iter()
            .copied()
            .filter(|&t| (t as usize) < vocab_size)
            .collect();
        suppress.extend([
            tokens.sot,
            tokens.transcribe,
            tokens.translate,
            tokens.no_timestamps,
        ]);
        suppress.extend(tokens.languages.iter().map(|&(_, id)| id));
        suppress.extend(tokens.no_speech);
        suppress.sort_unstable();
        suppress.dedup();
        let begin_suppress = if begin_suppress_tokens.is_empty() {
            // Whisper's default: a lone space, and the end.
            tokens.id("Ġ").into_iter().chain([tokens.eot]).collect()
        } else {
            begin_suppress_tokens.to_vec()
        };
        Self {
            suppress,
            begin_suppress,
        }
    }

    /// Mask `logits` for the next token after `sampled` (the tokens chosen
    /// so far, after the prompt) in a window `window_seconds` long.
    pub fn apply(&self, logits: &mut [f32], sampled: &[u32], tokens: &Tokens, window_seconds: f64) {
        let n = logits.len();
        let mask = |logits: &mut [f32], from: usize, to: usize| {
            for l in &mut logits[from.min(n)..to.min(n)] {
                *l = f32::NEG_INFINITY;
            }
        };
        for &t in &self.suppress {
            if (t as usize) < n {
                logits[t as usize] = f32::NEG_INFINITY;
            }
        }
        if sampled.is_empty() {
            for &t in &self.begin_suppress {
                if (t as usize) < n {
                    logits[t as usize] = f32::NEG_INFINITY;
                }
            }
        }

        let ts = tokens.timestamp_begin as usize;
        let eot = tokens.eot as usize;
        let is_ts = |t: &u32| tokens.is_timestamp(*t);
        let last_ts = sampled.last().is_some_and(is_ts);
        let penultimate_ts = sampled.len() < 2 || is_ts(&sampled[sampled.len() - 2]);
        if last_ts {
            if penultimate_ts {
                // A pair just closed: text next.
                mask(logits, ts, n);
            } else {
                // A segment just ended: another timestamp or the end.
                mask(logits, 0, eot);
            }
        }
        // Time never runs backwards: a segment's closing timestamp may equal
        // its opening one, anything after must be later.
        if let Some(&latest) = sampled.iter().filter(|t| is_ts(t)).max() {
            let floor = if last_ts && !penultimate_ts {
                latest as usize
            } else {
                latest as usize + 1
            };
            mask(logits, ts, floor);
        }
        if sampled.is_empty() {
            // Text opens with a timestamp, in the first second.
            mask(logits, 0, ts);
            let last_allowed = ts + (MAX_INITIAL_TIMESTAMP / TIMESTAMP_STEP).round() as usize;
            mask(logits, last_allowed + 1, n);
        }
        // Nothing past the window's end.
        let window_end = ts + (window_seconds / TIMESTAMP_STEP).ceil() as usize;
        mask(logits, window_end + 1, n);

        // The timestamps together likelier than the best text token: take one.
        if ts < n {
            let ts_mass = log_sum_exp(&logits[ts..]);
            let best_text = logits[..ts]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            if ts_mass > f64::from(best_text) {
                mask(logits, 0, ts);
            }
        }
    }
}

fn log_sum_exp(v: &[f32]) -> f64 {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    f64::from(m) + v.iter().map(|&x| f64::from(x - m).exp()).sum::<f64>().ln()
}

/// The segments in `sampled` (`<|a|> text <|b|> <|b|> text <|c|> ...`),
/// times capped at `window_seconds`. Text left open at the end runs to the
/// window's end.
pub fn segments_of(sampled: &[u32], tokens: &Tokens, window_seconds: f64) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut start: Option<f64> = None;
    let mut text: Vec<u32> = Vec::new();
    let mut last_end = 0.0;
    let close = |start: f64, end: f64, text: &mut Vec<u32>, out: &mut Vec<Segment>| {
        let words = tokens.decode(text).trim().to_string();
        text.clear();
        if !words.is_empty() && end > start {
            out.push(Segment {
                start,
                end,
                text: words,
            });
        }
    };
    for &id in sampled {
        if id == tokens.eot {
            break;
        }
        if tokens.is_timestamp(id) {
            let t = tokens.seconds(id).min(window_seconds);
            match start {
                Some(s) if !text.is_empty() => {
                    close(s, t, &mut text, &mut out);
                    last_end = t;
                    start = None;
                }
                _ => start = Some(t),
            }
        } else if id < tokens.eot {
            if start.is_none() {
                start = Some(last_end);
            }
            text.push(id);
        }
    }
    if let Some(s) = start
        && !text.is_empty()
    {
        close(s, window_seconds, &mut text, &mut out);
    }
    out
}

/// Where the last closed segment in `sampled` ends, when text follows it
/// that no timestamp closed: the sentence a window cut off. `None` when
/// every segment was closed, or none was.
pub fn closed_end(sampled: &[u32], tokens: &Tokens, window_seconds: f64) -> Option<f64> {
    let body: Vec<u32> = sampled
        .iter()
        .copied()
        .take_while(|&t| t != tokens.eot)
        .collect();
    let last_ts = body.iter().rposition(|&t| tokens.is_timestamp(t))?;
    let open_tail = body[last_ts + 1..].iter().any(|&t| t < tokens.eot);
    // The timestamp a segment closed on: one with text before it.
    let closes = |i: usize| i > 0 && !tokens.is_timestamp(body[i - 1]);
    if !open_tail {
        return None;
    }
    // The opening timestamp of the open text, or the closing one before it.
    let closing = (0..=last_ts)
        .rev()
        .find(|&i| tokens.is_timestamp(body[i]) && closes(i))?;
    Some(tokens.seconds(body[closing]).min(window_seconds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::tests::tiny_tokenizer_json;

    fn tokens() -> Tokens {
        Tokens::parse(&tiny_tokenizer_json()).unwrap()
    }

    /// `<|t|>` for `t` seconds, in the tiny vocabulary (`<|0.00|>` is 108).
    fn ts(t: f64) -> u32 {
        108 + (t / TIMESTAMP_STEP).round() as u32
    }

    #[test]
    fn pairs_of_timestamps_make_segments() {
        let t = tokens();
        // <|0.00|> Hello world! <|1.50|><|1.50|> café <|3.00|> <eot>
        let s = segments_of(
            &[ts(0.0), 0, 1, 2, ts(1.5), ts(1.5), 3, 4, ts(3.0), t.eot],
            &t,
            30.0,
        );
        assert_eq!(
            s,
            vec![
                Segment {
                    start: 0.0,
                    end: 1.5,
                    text: "Hello world!".into()
                },
                Segment {
                    start: 1.5,
                    end: 3.0,
                    text: "café".into()
                },
            ]
        );
    }

    #[test]
    fn open_text_runs_to_the_window_end_and_times_are_capped() {
        let t = tokens();
        let s = segments_of(&[ts(2.0), 0, 1], &t, 4.0);
        assert_eq!(
            s,
            vec![Segment {
                start: 2.0,
                end: 4.0,
                text: "Hello world".into()
            }]
        );
        let s = segments_of(&[ts(1.0), 0, ts(9.0)], &t, 4.0);
        assert_eq!(s[0].end, 4.0);
    }

    #[test]
    fn a_sentence_left_open_reports_where_the_closed_ones_end() {
        let t = tokens();
        // <|0.00|> Hello <|1.50|><|1.50|> world (no closing timestamp)
        assert_eq!(
            closed_end(&[ts(0.0), 0, ts(1.5), ts(1.5), 1], &t, 30.0),
            Some(1.5)
        );
        // Everything closed, then the end: nothing open.
        assert_eq!(closed_end(&[ts(0.0), 0, ts(1.5), t.eot], &t, 30.0), None);
        // Open text and nothing closed before it: nowhere earlier to start.
        assert_eq!(closed_end(&[ts(0.0), 0, 1], &t, 30.0), None);
    }

    fn rules() -> (Rules, Tokens) {
        let t = tokens();
        (Rules::new(&t, 2000, &[2], &[]), t)
    }

    fn allowed(logits: &[f32]) -> Vec<usize> {
        logits
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_finite())
            .map(|(i, _)| i)
            .collect()
    }

    #[test]
    fn the_first_token_is_an_early_timestamp() {
        let (r, t) = rules();
        let mut logits = vec![0.0f32; 2000];
        r.apply(&mut logits, &[], &t, 30.0);
        let ok = allowed(&logits);
        assert_eq!(ok.first(), Some(&(ts(0.0) as usize)));
        assert_eq!(ok.last(), Some(&(ts(1.0) as usize)));
    }

    #[test]
    fn after_a_closing_timestamp_comes_a_timestamp_or_the_end() {
        let (r, t) = rules();
        let mut logits = vec![0.0f32; 2000];
        // The end well ahead, so the timestamp-mass rule leaves text open.
        logits[t.eot as usize] = 50.0;
        r.apply(&mut logits, &[ts(0.0), 0, ts(1.0)], &t, 30.0);
        let ok = allowed(&logits);
        assert!(ok.contains(&(t.eot as usize)));
        assert!(!ok.contains(&0) && !ok.contains(&1));
        // Not before the timestamp that closed the segment.
        assert!(!ok.contains(&(ts(0.98) as usize)));
        assert!(ok.contains(&(ts(1.0) as usize)));
    }

    #[test]
    fn after_a_pair_comes_text_and_suppressed_tokens_never() {
        let (r, t) = rules();
        let mut logits = vec![0.0f32; 2000];
        // Text well ahead of any timestamp, so the mass rule does not fire.
        logits[1] = 50.0;
        r.apply(&mut logits, &[ts(0.0), 0, ts(1.0), ts(1.0)], &t, 30.0);
        let ok = allowed(&logits);
        assert!(ok.iter().all(|&i| i < t.timestamp_begin as usize));
        assert!(!ok.contains(&2), "the checkpoint's suppress list");
        for special in [
            t.sot,
            t.transcribe,
            t.no_timestamps,
            t.language("en").unwrap(),
        ] {
            assert!(!ok.contains(&(special as usize)));
        }
    }

    #[test]
    fn timestamps_never_pass_the_window() {
        let (r, t) = rules();
        let mut logits = vec![0.0f32; 2000];
        r.apply(&mut logits, &[ts(0.0), 0], &t, 2.0);
        assert!(allowed(&logits).iter().all(|&i| i <= ts(2.0) as usize));
    }
}
