//! Whisper's text decoder with its self-attention cached, so each step runs
//! only the new token.
//!
//! candle's `TextDecoder` caches the cross-attention keys and values (the
//! audio) but not the self-attention's, so it is handed the whole token
//! sequence every step and re-attends every earlier token to all 1500 audio
//! frames: quadratic in the text. Measured on whisper-tiny, CPU: 28.5 s for
//! 62 s of speech. This is the same network on the same weights (Hugging
//! Face's names, as candle loads them), run incrementally.

use candle_core::{D, IndexOp, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::whisper::Config;

fn layer_norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        vb.get(size, "weight")?,
        vb.get(size, "bias")?,
        1e-5,
    ))
}

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
}

impl Attention {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            query: candle_nn::linear(n_state, n_state, vb.pp("q_proj"))?,
            key: candle_nn::linear_no_bias(n_state, n_state, vb.pp("k_proj"))?,
            value: candle_nn::linear(n_state, n_state, vb.pp("v_proj"))?,
            out: candle_nn::linear(n_state, n_state, vb.pp("out_proj"))?,
            n_head,
        })
    }

    /// `(batch, ctx, state)` → `(batch, head, ctx, state / head)`.
    fn heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, s) = x.dims3()?;
        x.reshape((b, n, self.n_head, s / self.n_head))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// Scaled attention of `q` over `k` / `v` (already split into heads).
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let d = q.dim(D::Minus1)?;
        let scale = (d as f64).powf(-0.25);
        let q = (q * scale)?;
        let k = (k.transpose(2, 3)? * scale)?;
        let mut w = q.matmul(&k.contiguous()?)?;
        if let Some(mask) = mask {
            w = w.broadcast_add(mask)?;
        }
        let w = candle_nn::ops::softmax_last_dim(&w)?;
        let out = w.matmul(v)?.transpose(1, 2)?.flatten_from(2)?;
        self.out.forward(&out)
    }
}

struct Block {
    attn: Attention,
    attn_ln: LayerNorm,
    cross: Attention,
    cross_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    mlp_ln: LayerNorm,
    /// Self-attention keys and values of every token so far, in heads.
    self_kv: Option<(Tensor, Tensor)>,
    /// The audio's keys and values, in heads, set once per window.
    cross_kv: Option<(Tensor, Tensor)>,
}

impl Block {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(n_state, n_head, vb.pp("self_attn"))?,
            attn_ln: layer_norm(n_state, vb.pp("self_attn_layer_norm"))?,
            cross: Attention::load(n_state, n_head, vb.pp("encoder_attn"))?,
            cross_ln: layer_norm(n_state, vb.pp("encoder_attn_layer_norm"))?,
            fc1: candle_nn::linear(n_state, n_state * 4, vb.pp("fc1"))?,
            fc2: candle_nn::linear(n_state * 4, n_state, vb.pp("fc2"))?,
            mlp_ln: layer_norm(n_state, vb.pp("final_layer_norm"))?,
            self_kv: None,
            cross_kv: None,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let h = self.attn_ln.forward(x)?;
        let q = self.attn.heads(&self.attn.query.forward(&h)?)?;
        let k = self.attn.heads(&self.attn.key.forward(&h)?)?;
        let v = self.attn.heads(&self.attn.value.forward(&h)?)?;
        let (k, v) = match self.self_kv.take() {
            Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?),
            None => (k, v),
        };
        self.self_kv = Some((k.clone(), v.clone()));
        let x = (x + self.attn.attend(&q, &k, &v, mask)?)?;

        let h = self.cross_ln.forward(&x)?;
        let q = self.cross.heads(&self.cross.query.forward(&h)?)?;
        let (ck, cv) = self.cross_kv.as_ref().expect("set_audio first");
        let x = (&x + self.cross.attend(&q, ck, cv, None)?)?;

        let h = self
            .fc2
            .forward(&self.fc1.forward(&self.mlp_ln.forward(&x)?)?.gelu()?)?;
        x + h
    }
}

/// The decoder, run a step at a time.
pub struct CachedDecoder {
    embed: Embedding,
    positions: Tensor,
    blocks: Vec<Block>,
    ln: LayerNorm,
    /// Tokens consumed since `set_audio`.
    len: usize,
}

impl CachedDecoder {
    /// From the `model.decoder` weights.
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let n_state = cfg.d_model;
        Ok(Self {
            embed: candle_nn::embedding(cfg.vocab_size, n_state, vb.pp("embed_tokens"))?,
            positions: vb.get(
                (cfg.max_target_positions, n_state),
                "embed_positions.weight",
            )?,
            blocks: (0..cfg.decoder_layers)
                .map(|i| {
                    Block::load(
                        n_state,
                        cfg.decoder_attention_heads,
                        vb.pp(format!("layers.{i}")),
                    )
                })
                .collect::<Result<_>>()?,
            ln: layer_norm(n_state, vb.pp("layer_norm"))?,
            len: 0,
        })
    }

    /// Start a new sequence over `audio` (the encoder's output).
    pub fn set_audio(&mut self, audio: &Tensor) -> Result<()> {
        for b in &mut self.blocks {
            b.self_kv = None;
            let k = b.cross.heads(&b.cross.key.forward(audio)?)?;
            let v = b.cross.heads(&b.cross.value.forward(audio)?)?;
            b.cross_kv = Some((k, v));
        }
        self.len = 0;
        Ok(())
    }

    /// Feed `tokens` (the prompt, then one at a time) and return the logits
    /// for what follows the last of them.
    pub fn step(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let n = tokens.len();
        let device = self.positions.device().clone();
        let ids = Tensor::new(tokens, &device)?.unsqueeze(0)?;
        let pos = self.positions.narrow(0, self.len, n)?;
        let mut x = self.embed.forward(&ids)?.broadcast_add(&pos)?;
        // Several tokens at once (the prompt): each sees the cache and the
        // new tokens up to itself.
        let mask = if n > 1 {
            let base = self.len;
            let total = base + n;
            let m: Vec<f32> = (0..n)
                .flat_map(|i| {
                    (0..total).map(move |j| if j > base + i { f32::NEG_INFINITY } else { 0.0 })
                })
                .collect();
            Some(Tensor::from_vec(m, (n, total), &device)?)
        } else {
            None
        };
        for b in &mut self.blocks {
            x = b.forward(&x, mask.as_ref())?;
        }
        self.len += n;
        let last = self.ln.forward(&x.i((.., n - 1..n, ..))?)?;
        let logits = last.matmul(&self.embed.embeddings().t()?.unsqueeze(0)?)?;
        logits.flatten_all()?.to_vec1::<f32>()
    }

    /// Tokens the context still has room for.
    pub fn room(&self) -> usize {
        self.positions.dim(0).unwrap_or(0).saturating_sub(self.len)
    }
}
