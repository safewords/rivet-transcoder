//! The mel filter bank Whisper's spectrogram is taken through.
//!
//! Whisper's reference implementation loads it from `mel_filters.npz`, which
//! is `librosa.filters.mel(sr=16000, n_fft=400, n_mels=N)` with librosa's
//! defaults: the Slaney mel scale (linear below 1 kHz, logarithmic above) and
//! Slaney normalisation (each triangle scaled to unit area in Hz). That is a
//! formula, so it is computed here rather than shipped as a file beside the
//! weights.

/// Whisper's FFT size (25 ms at 16 kHz).
pub const N_FFT: usize = 400;
/// Whisper's sample rate.
pub const SAMPLE_RATE: f64 = 16_000.0;

const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;

fn log_step() -> f64 {
    6.4f64.ln() / 27.0
}

/// Hz to mels on the Slaney scale.
pub fn hz_to_mel(hz: f64) -> f64 {
    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / log_step()
    } else {
        hz / F_SP
    }
}

/// Mels to Hz on the Slaney scale.
pub fn mel_to_hz(mel: f64) -> f64 {
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (log_step() * (mel - MIN_LOG_MEL)).exp()
    } else {
        F_SP * mel
    }
}

/// `n_mels` triangular filters over the `N_FFT / 2 + 1` bins of a 16 kHz
/// spectrum, row-major (`filters[m * bins + k]`): the layout
/// `candle_transformers::models::whisper::audio::pcm_to_mel` reads.
pub fn filters(n_mels: usize) -> Vec<f32> {
    let bins = N_FFT / 2 + 1;
    let fft_hz: Vec<f64> = (0..bins)
        .map(|k| k as f64 * SAMPLE_RATE / N_FFT as f64)
        .collect();
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE / 2.0));
    let points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut out = vec![0f32; n_mels * bins];
    for m in 0..n_mels {
        let (left, centre, right) = (points[m], points[m + 1], points[m + 2]);
        // Slaney: each triangle scaled to unit area.
        let norm = 2.0 / (right - left);
        for (k, &f) in fft_hz.iter().enumerate() {
            let rising = (f - left) / (centre - left);
            let falling = (right - f) / (right - centre);
            let w = rising.min(falling).max(0.0);
            out[m * bins + k] = (w * norm) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scale_is_linear_below_1khz_and_round_trips() {
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-9);
        assert!((hz_to_mel(500.0) - 7.5).abs() < 1e-9);
        for hz in [0.0, 100.0, 999.0, 1000.0, 3000.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 1e-6, "{hz}");
        }
    }

    #[test]
    fn each_filter_is_a_triangle_of_unit_area() {
        for n in [80, 128] {
            let f = filters(n);
            let bins = N_FFT / 2 + 1;
            assert_eq!(f.len(), n * bins);
            let df = SAMPLE_RATE / N_FFT as f64;
            let mut last_peak = 0;
            for m in 0..n {
                let row = &f[m * bins..(m + 1) * bins];
                assert!(row.iter().all(|&w| w >= 0.0));
                let peak = (0..bins)
                    .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                    .unwrap();
                assert!(peak >= last_peak, "peaks rise with the filter index");
                last_peak = peak;
                // Wide enough to cover several bins (the high filters), the
                // sampled area is the triangle's: 1.
                let nonzero = row.iter().filter(|&&w| w > 0.0).count();
                if nonzero >= 6 {
                    let area: f64 = row.iter().map(|&w| f64::from(w) * df).sum();
                    assert!((area - 1.0).abs() < 0.05, "filter {m} of {n}: area {area}");
                }
            }
        }
    }
}
