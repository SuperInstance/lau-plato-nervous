//! Spectral analysis of alert patterns: Fourier decomposition of alert time series.
//!
//! Alerts arrive as time series. This module decomposes them into frequency
//! components to detect periodic patterns, bursts, and anomalies.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A single alert event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    pub timestamp: f64,
    pub severity: f64,
    pub source: String,
    pub message: String,
}

/// Result of spectral decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralDecomposition {
    /// Frequencies (Hz).
    pub frequencies: Vec<f64>,
    /// Amplitudes at each frequency.
    pub amplitudes: Vec<f64>,
    /// Phases at each frequency.
    pub phases: Vec<f64>,
    /// Dominant frequency index.
    pub dominant_index: usize,
}

/// Compute the Discrete Fourier Transform (DFT) of a real-valued time series.
/// Returns (frequencies, complex amplitudes as (re, im) pairs).
pub fn dft(samples: &[f64], sample_rate: f64) -> (Vec<f64>, Vec<(f64, f64)>) {
    let n = samples.len();
    let mut frequencies = Vec::with_capacity(n);
    let mut amplitudes = Vec::with_capacity(n);

    for k in 0..n {
        let freq = k as f64 * sample_rate / n as f64;
        let mut re = 0.0;
        let mut im = 0.0;
        for (t, &x) in samples.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        frequencies.push(freq);
        amplitudes.push((re, im));
    }
    (frequencies, amplitudes)
}

/// Perform full spectral decomposition on an alert time series.
pub fn spectral_decompose(samples: &[f64], sample_rate: f64) -> SpectralDecomposition {
    let (frequencies, amplitudes) = dft(samples, sample_rate);
    let n = samples.len();

    let magnitudes: Vec<f64> = amplitudes
        .iter()
        .map(|(re, im)| (re * re + im * im).sqrt() / n as f64)
        .collect();

    let phases: Vec<f64> = amplitudes
        .iter()
        .map(|(re, im)| im.atan2(*re))
        .collect();

    let dominant_index = magnitudes
        .iter()
        .enumerate()
        .skip(1) // skip DC component
        .take(n / 2)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    SpectralDecomposition {
        frequencies,
        amplitudes: magnitudes,
        phases,
        dominant_index,
    }
}

/// Detect if alerts have a periodic pattern.
/// Returns the dominant frequency and its strength (SNR-like ratio).
pub fn detect_periodicity(samples: &[f64], sample_rate: f64) -> (f64, f64) {
    let decomp = spectral_decompose(samples, sample_rate);
    if decomp.dominant_index == 0 {
        return (0.0, 0.0);
    }
    let dominant_amp = decomp.amplitudes[decomp.dominant_index];
    let total_energy: f64 = decomp.amplitudes.iter().skip(1).map(|a| a * a).sum();
    let dominant_energy = dominant_amp * dominant_amp;
    let snr = if total_energy > 0.0 {
        dominant_energy / total_energy
    } else {
        0.0
    };
    (decomp.frequencies[decomp.dominant_index], snr)
}

/// Compute the power spectral density.
pub fn power_spectral_density(samples: &[f64], sample_rate: f64) -> DVector<f64> {
    let (_, amplitudes) = dft(samples, sample_rate);
    let n = samples.len() as f64;
    let psd: Vec<f64> = amplitudes
        .iter()
        .map(|(re, im)| (re * re + im * im) / n)
        .collect();
    DVector::from_vec(psd)
}

/// Build a spectrogram matrix from overlapping windows.
pub fn spectrogram(
    samples: &[f64],
    window_size: usize,
    hop: usize,
    sample_rate: f64,
) -> DMatrix<f64> {
    let num_windows = (samples.len().saturating_sub(window_size)) / hop + 1;
    if num_windows == 0 {
        return DMatrix::zeros(0, 0);
    }
    let mut cols = Vec::with_capacity(num_windows);
    for i in 0..num_windows {
        let start = i * hop;
        let end = (start + window_size).min(samples.len());
        let window = &samples[start..end];
        let psd = power_spectral_density(window, sample_rate);
        cols.push(psd);
    }
    let nrows = cols[0].nrows();
    let mut data = Vec::with_capacity(nrows * num_windows);
    for col in &cols {
        data.extend(col.iter().cloned());
    }
    DMatrix::from_vec(nrows, num_windows, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dft_dc_component() {
        let samples = vec![1.0, 1.0, 1.0, 1.0];
        let (freqs, amps) = dft(&samples, 1.0);
        assert_eq!(freqs.len(), 4);
        let dc_mag = (amps[0].0.powi(2) + amps[0].1.powi(2)).sqrt();
        assert!((dc_mag - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_dft_sine_wave() {
        // 4 Hz sine wave sampled at 16 Hz, 16 samples
        let n = 16;
        let sr = 16.0;
        let freq = 4.0;
        let samples: Vec<f64> = (0..n)
            .map(|t| (2.0 * std::f64::consts::PI * freq * t as f64 / sr).sin())
            .collect();
        let decomp = spectral_decompose(&samples, sr);
        assert!(decomp.dominant_index > 0);
    }

    #[test]
    fn test_spectral_decompose_dominant() {
        let samples = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        let decomp = spectral_decompose(&samples, 1.0);
        assert!(decomp.amplitudes.len() == 8);
        assert!(decomp.dominant_index > 0);
    }

    #[test]
    fn test_detect_periodicity_pure_sine() {
        let n = 32;
        let sr = 32.0;
        let samples: Vec<f64> = (0..n)
            .map(|t| (2.0 * std::f64::consts::PI * 4.0 * t as f64 / sr).sin())
            .collect();
        let (freq, snr) = detect_periodicity(&samples, sr);
        assert!(snr > 0.5, "SNR should be high for pure sine: {}", snr);
    }

    #[test]
    fn test_detect_periodicity_noise() {
        let samples = vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.15, -0.25, 0.08];
        let (_, snr) = detect_periodicity(&samples, 1.0);
        // Noise should have low SNR
        assert!(snr < 0.9);
    }

    #[test]
    fn test_psd_length() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let psd = power_spectral_density(&samples, 1.0);
        assert_eq!(psd.nrows(), 4);
    }

    #[test]
    fn test_psd_nonnegative() {
        let samples = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let psd = power_spectral_density(&samples, 1.0);
        for &v in psd.iter() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_spectrogram_dimensions() {
        let samples: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();
        let sg = spectrogram(&samples, 16, 4, 16.0);
        assert!(sg.nrows() > 0);
        assert!(sg.ncols() > 0);
    }

    #[test]
    fn test_alert_event_serialization() {
        let evt = AlertEvent {
            timestamp: 1000.0,
            severity: 0.8,
            source: "room-1".into(),
            message: "high CPU".into(),
        };
        let json = serde_json::to_string(&evt).unwrap();
        let back: AlertEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.source, "room-1");
    }

    #[test]
    fn test_spectral_decompose_constant() {
        let samples = vec![5.0; 8];
        let decomp = spectral_decompose(&samples, 1.0);
        // DC should dominate
        let dc = decomp.amplitudes[0];
        let others: f64 = decomp.amplitudes.iter().skip(1).sum();
        assert!(dc > others);
    }
}
