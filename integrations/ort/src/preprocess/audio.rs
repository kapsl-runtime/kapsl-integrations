//! Float32 mono PCM input to log-mel ONNX feature tensors.

use crate::tensor::{BorrowedTensor, OwnedTensor};
use crate::{invalid_argument, FfiResult};
use kapsl_backend_abi::{KAPSL_DTYPE_F32, KAPSL_DTYPE_I32, KAPSL_DTYPE_I64};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use serde::Deserialize;
use std::sync::Arc;

const MAX_FFT_SIZE: usize = 1 << 20;
const MAX_MEL_BINS: usize = 65_536;
const MAX_STATIC_ELEMENTS: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum MelScale {
    #[default]
    Htk,
    Slaney,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum MelNorm {
    #[default]
    None,
    Slaney,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum LogKind {
    None,
    Ln,
    #[default]
    Log10,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AudioLayout {
    #[default]
    MelTime,
    TimeMel,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FeatureNormalization {
    #[default]
    None,
    PerFeature,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum LengthDtype {
    Int32,
    #[default]
    Int64,
}

fn default_sample_rate() -> u32 {
    16_000
}

fn default_n_fft() -> usize {
    400
}

fn default_hop_length() -> usize {
    160
}

fn default_n_mels() -> usize {
    80
}

fn default_power() -> f32 {
    2.0
}

fn default_center() -> bool {
    true
}

fn default_log_eps() -> f32 {
    1e-10
}

fn default_normalize_eps() -> f32 {
    1e-5
}

#[derive(Debug, Deserialize)]
pub(super) struct AudioConfig {
    #[serde(default = "default_sample_rate")]
    sample_rate: u32,
    #[serde(default = "default_n_fft")]
    n_fft: usize,
    #[serde(default = "default_hop_length")]
    hop_length: usize,
    #[serde(default = "default_n_mels")]
    n_mels: usize,
    #[serde(default)]
    f_min: f32,
    #[serde(default)]
    f_max: Option<f32>,
    #[serde(default)]
    mel_scale: MelScale,
    #[serde(default)]
    norm: MelNorm,
    #[serde(default)]
    log: LogKind,
    #[serde(default = "default_power")]
    power: f32,
    #[serde(default = "default_center")]
    center: bool,
    #[serde(default)]
    layout: AudioLayout,
    #[serde(default = "default_log_eps")]
    log_eps: f32,
    #[serde(default, alias = "normalize_type")]
    normalize: FeatureNormalization,
    #[serde(default = "default_normalize_eps")]
    normalize_eps: f32,
    #[serde(default)]
    length_input: Option<String>,
    #[serde(default)]
    length_dtype: LengthDtype,
}

pub(crate) struct AudioPreprocessor {
    config: AudioConfig,
    window: Vec<f32>,
    mel_filterbank: Vec<f32>,
    frequencies: usize,
    fft: Arc<dyn Fft<f32>>,
}

impl AudioPreprocessor {
    pub(super) fn new(config: AudioConfig) -> FfiResult<Self> {
        config.validate()?;
        let frequencies = config.n_fft / 2 + 1;
        let window = hann_window(config.n_fft);
        let mel_filterbank = mel_filterbank(&config, frequencies)?;
        let fft = FftPlanner::new().plan_fft_forward(config.n_fft);
        Ok(Self {
            config,
            window,
            mel_filterbank,
            frequencies,
            fft,
        })
    }

    pub(super) fn apply(&self, input: &BorrowedTensor<'_>) -> FfiResult<OwnedTensor> {
        let samples = self.samples(input)?;
        let signal = self.maybe_center(&samples)?;
        let frames = self.frame_count(signal.len())?;
        let elements = self
            .config
            .n_mels
            .checked_mul(frames)
            .ok_or_else(|| invalid_argument("audio feature tensor size overflows"))?;
        let mut mels = vec![0.0; elements];
        let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); self.config.n_fft];
        let mut scratch = vec![Complex::<f32>::new(0.0, 0.0); self.fft.get_inplace_scratch_len()];
        for frame in 0..frames {
            let start = frame
                .checked_mul(self.config.hop_length)
                .ok_or_else(|| invalid_argument("audio frame offset overflows"))?;
            for index in 0..self.config.n_fft {
                buffer[index] = Complex::new(signal[start + index] * self.window[index], 0.0);
            }
            self.fft.process_with_scratch(&mut buffer, &mut scratch);
            for mel in 0..self.config.n_mels {
                let filter =
                    &self.mel_filterbank[mel * self.frequencies..(mel + 1) * self.frequencies];
                let mut energy = 0.0;
                for (bin, weight) in filter.iter().copied().enumerate() {
                    if weight == 0.0 {
                        continue;
                    }
                    let value = buffer[bin];
                    let square = value.re * value.re + value.im * value.im;
                    let magnitude = if self.config.power == 1.0 {
                        square.sqrt()
                    } else {
                        square
                    };
                    energy += weight * magnitude;
                }
                mels[mel * frames + frame] = self.compress(energy);
            }
        }
        self.normalize(&mut mels, frames);

        let (shape, values) = match self.config.layout {
            AudioLayout::MelTime => (vec![1, self.config.n_mels as i64, frames as i64], mels),
            AudioLayout::TimeMel => {
                let mut transposed = vec![0.0; elements];
                for mel in 0..self.config.n_mels {
                    for frame in 0..frames {
                        transposed[frame * self.config.n_mels + mel] = mels[mel * frames + frame];
                    }
                }
                (
                    vec![1, frames as i64, self.config.n_mels as i64],
                    transposed,
                )
            }
        };
        Ok(OwnedTensor {
            name: "input".to_string(),
            dtype: KAPSL_DTYPE_F32,
            shape,
            data: encode_f32(values),
        })
    }

    pub(super) fn derived_inputs(&self, output: &OwnedTensor) -> FfiResult<Vec<OwnedTensor>> {
        let Some(name) = self.config.length_input.as_ref() else {
            return Ok(Vec::new());
        };
        let axis = match self.config.layout {
            AudioLayout::MelTime => 2,
            AudioLayout::TimeMel => 1,
        };
        let frames = output.shape.get(axis).copied().ok_or_else(|| {
            invalid_argument(format!(
                "audio preprocessing cannot derive length from shape {:?}",
                output.shape
            ))
        })?;
        let (dtype, data) = match self.config.length_dtype {
            LengthDtype::Int64 => (KAPSL_DTYPE_I64, frames.to_ne_bytes().to_vec()),
            LengthDtype::Int32 => {
                let frames = i32::try_from(frames).map_err(|_| {
                    invalid_argument(format!("audio frame count {frames} does not fit int32"))
                })?;
                (KAPSL_DTYPE_I32, frames.to_ne_bytes().to_vec())
            }
        };
        Ok(vec![OwnedTensor {
            name: name.clone(),
            dtype,
            shape: vec![1],
            data,
        }])
    }

    pub(super) fn planned_bytes(&self, input: &BorrowedTensor<'_>) -> FfiResult<usize> {
        self.validate_input(input)?;
        let samples = input.data.len() / 4;
        let signal = self.centered_length(samples)?;
        let frames = self.frame_count(signal)?;
        let mel_elements = self
            .config
            .n_mels
            .checked_mul(frames)
            .ok_or_else(|| invalid_argument("audio feature tensor size overflows"))?;
        let sample_bytes = samples
            .checked_mul(4)
            .ok_or_else(|| invalid_argument("audio sample memory estimate overflows"))?;
        let centered_bytes = signal
            .checked_mul(4)
            .ok_or_else(|| invalid_argument("audio centered signal estimate overflows"))?;
        let fft_bytes = self
            .config
            .n_fft
            .checked_mul(std::mem::size_of::<Complex<f32>>())
            .ok_or_else(|| invalid_argument("audio FFT memory estimate overflows"))?;
        let scratch_bytes = self
            .fft
            .get_inplace_scratch_len()
            .checked_mul(std::mem::size_of::<Complex<f32>>())
            .ok_or_else(|| invalid_argument("audio FFT scratch estimate overflows"))?;
        let feature_copies = if self.config.layout == AudioLayout::TimeMel {
            3
        } else {
            2
        };
        let feature_bytes = mel_elements
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_mul(feature_copies))
            .ok_or_else(|| invalid_argument("audio feature memory estimate overflows"))?;
        sample_bytes
            .checked_add(centered_bytes)
            .and_then(|bytes| bytes.checked_add(fft_bytes))
            .and_then(|bytes| bytes.checked_add(scratch_bytes))
            .and_then(|bytes| bytes.checked_add(feature_bytes))
            .and_then(|bytes| {
                bytes.checked_add(if self.config.length_input.is_some() {
                    8
                } else {
                    0
                })
            })
            .ok_or_else(|| invalid_argument("audio preprocessing memory estimate overflows"))
    }

    pub(super) fn resident_bytes(&self) -> usize {
        let window = self
            .window
            .capacity()
            .saturating_mul(std::mem::size_of::<f32>());
        let filterbank = self
            .mel_filterbank
            .capacity()
            .saturating_mul(std::mem::size_of::<f32>());
        // RustFFT does not expose the heap retained by a concrete plan. Use a
        // conservative linear allowance in addition to its dynamic object.
        let plan = std::mem::size_of_val(self.fft.as_ref()).saturating_add(
            self.config
                .n_fft
                .saturating_mul(4 * std::mem::size_of::<Complex<f32>>()),
        );
        window.saturating_add(filterbank).saturating_add(plan)
    }

    pub(super) fn expected_shape(&self) -> Vec<i64> {
        match self.config.layout {
            AudioLayout::MelTime => vec![1, self.config.n_mels as i64, -1],
            AudioLayout::TimeMel => vec![1, -1, self.config.n_mels as i64],
        }
    }

    pub(super) fn length_contract(&self) -> Option<(&str, &str)> {
        self.config.length_input.as_deref().map(|name| {
            let dtype = match self.config.length_dtype {
                LengthDtype::Int32 => "int32",
                LengthDtype::Int64 => "int64",
            };
            (name, dtype)
        })
    }

    fn samples(&self, input: &BorrowedTensor<'_>) -> FfiResult<Vec<f32>> {
        self.validate_input(input)?;
        let samples = input
            .data
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four bytes")))
            .collect::<Vec<_>>();
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(invalid_argument(
                "audio preprocessing requires finite PCM samples",
            ));
        }
        Ok(samples)
    }

    fn validate_input(&self, input: &BorrowedTensor<'_>) -> FfiResult<()> {
        if input.dtype != KAPSL_DTYPE_F32 {
            return Err(invalid_argument(format!(
                "audio preprocessing expects float32 PCM samples, received dtype {}",
                input.dtype
            )));
        }
        if input.data.is_empty() || !input.data.len().is_multiple_of(4) {
            return Err(invalid_argument(
                "audio preprocessing requires a non-empty float32 waveform",
            ));
        }
        Ok(())
    }

    fn centered_length(&self, samples: usize) -> FfiResult<usize> {
        if self.config.center {
            samples
                .checked_add(2 * (self.config.n_fft / 2))
                .ok_or_else(|| invalid_argument("centered audio length overflows"))
        } else {
            Ok(samples)
        }
    }

    fn frame_count(&self, signal_length: usize) -> FfiResult<usize> {
        if signal_length < self.config.n_fft {
            return Err(invalid_argument(format!(
                "audio waveform is too short for n_fft={} after centering",
                self.config.n_fft
            )));
        }
        Ok(1 + (signal_length - self.config.n_fft) / self.config.hop_length)
    }

    fn maybe_center(&self, samples: &[f32]) -> FfiResult<Vec<f32>> {
        if !self.config.center {
            return Ok(samples.to_vec());
        }
        let pad = self.config.n_fft / 2;
        let mut output = Vec::with_capacity(self.centered_length(samples.len())?);
        for index in 0..pad {
            output.push(samples[reflect_index((pad - index) as isize, samples.len())]);
        }
        output.extend_from_slice(samples);
        for index in 0..pad {
            output.push(
                samples[reflect_index(samples.len() as isize - 2 - index as isize, samples.len())],
            );
        }
        Ok(output)
    }

    fn compress(&self, energy: f32) -> f32 {
        match self.config.log {
            LogKind::None => energy,
            LogKind::Ln => (energy + self.config.log_eps).ln(),
            LogKind::Log10 => (energy + self.config.log_eps).log10(),
        }
    }

    fn normalize(&self, mels: &mut [f32], frames: usize) {
        if self.config.normalize != FeatureNormalization::PerFeature {
            return;
        }
        for row in mels.chunks_exact_mut(frames) {
            let mean = row.iter().copied().sum::<f32>() / frames as f32;
            let variance = if frames > 1 {
                row.iter()
                    .map(|value| {
                        let centered = *value - mean;
                        centered * centered
                    })
                    .sum::<f32>()
                    / (frames - 1) as f32
            } else {
                0.0
            };
            let denominator = variance.sqrt() + self.config.normalize_eps;
            for value in row {
                *value = (*value - mean) / denominator;
            }
        }
    }
}

impl AudioConfig {
    fn validate(&self) -> FfiResult<()> {
        if self.sample_rate == 0 || self.n_fft == 0 || self.hop_length == 0 || self.n_mels == 0 {
            return Err(invalid_argument(
                "audio sample_rate, n_fft, hop_length, and n_mels must be non-zero",
            ));
        }
        if self.n_fft > MAX_FFT_SIZE || self.n_mels > MAX_MEL_BINS {
            return Err(invalid_argument(format!(
                "audio preprocessing supports n_fft <= {MAX_FFT_SIZE} and n_mels <= {MAX_MEL_BINS}"
            )));
        }
        let frequencies = self.n_fft / 2 + 1;
        let static_elements = self
            .n_mels
            .checked_mul(frequencies)
            .ok_or_else(|| invalid_argument("audio mel filterbank size overflows"))?;
        if static_elements > MAX_STATIC_ELEMENTS {
            return Err(invalid_argument(format!(
                "audio mel filterbank exceeds {MAX_STATIC_ELEMENTS} elements"
            )));
        }
        let maximum_frequency = self.f_max();
        let nyquist = self.sample_rate as f32 / 2.0;
        if !self.f_min.is_finite()
            || !maximum_frequency.is_finite()
            || self.f_min < 0.0
            || maximum_frequency <= self.f_min
            || maximum_frequency > nyquist
        {
            return Err(invalid_argument(format!(
                "audio frequencies must satisfy 0 <= f_min < f_max <= Nyquist ({nyquist})"
            )));
        }
        if !matches!(self.power, 1.0 | 2.0) {
            return Err(invalid_argument("audio power must be 1.0 or 2.0"));
        }
        if !self.log_eps.is_finite() || self.log_eps <= 0.0 {
            return Err(invalid_argument(
                "audio log_eps must be finite and greater than zero",
            ));
        }
        if !self.normalize_eps.is_finite() || self.normalize_eps <= 0.0 {
            return Err(invalid_argument(
                "audio normalize_eps must be finite and greater than zero",
            ));
        }
        if self
            .length_input
            .as_ref()
            .is_some_and(|name| name.trim().is_empty() || name == "input")
        {
            return Err(invalid_argument(
                "audio length_input must be non-empty and may not use the reserved name `input`",
            ));
        }
        Ok(())
    }

    fn f_max(&self) -> f32 {
        self.f_max.unwrap_or(self.sample_rate as f32 / 2.0)
    }
}

fn hz_to_mel(frequency: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + frequency / 700.0).log10(),
        MelScale::Slaney => {
            let spacing = 200.0 / 3.0;
            let logarithmic_start = 1000.0;
            let logarithmic_mel = logarithmic_start / spacing;
            let logarithmic_step = 6.4_f32.ln() / 27.0;
            if frequency < logarithmic_start {
                frequency / spacing
            } else {
                logarithmic_mel + (frequency / logarithmic_start).ln() / logarithmic_step
            }
        }
    }
}

fn mel_to_hz(mel: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 700.0 * (10_f32.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => {
            let spacing = 200.0 / 3.0;
            let logarithmic_start = 1000.0;
            let logarithmic_mel = logarithmic_start / spacing;
            let logarithmic_step = 6.4_f32.ln() / 27.0;
            if mel < logarithmic_mel {
                mel * spacing
            } else {
                logarithmic_start * (logarithmic_step * (mel - logarithmic_mel)).exp()
            }
        }
    }
}

fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|index| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / length as f32).cos())
        .collect()
}

fn mel_filterbank(config: &AudioConfig, frequencies: usize) -> FfiResult<Vec<f32>> {
    let minimum = hz_to_mel(config.f_min, config.mel_scale);
    let maximum = hz_to_mel(config.f_max(), config.mel_scale);
    let edges = (0..config.n_mels + 2)
        .map(|index| {
            let mel = minimum + (maximum - minimum) * index as f32 / (config.n_mels + 1) as f32;
            mel_to_hz(mel, config.mel_scale)
        })
        .collect::<Vec<_>>();
    let elements = config
        .n_mels
        .checked_mul(frequencies)
        .ok_or_else(|| invalid_argument("audio mel filterbank size overflows"))?;
    let mut filterbank = vec![0.0; elements];
    for mel in 0..config.n_mels {
        let (left, center, right) = (edges[mel], edges[mel + 1], edges[mel + 2]);
        let left_width = (center - left).max(f32::EPSILON);
        let right_width = (right - center).max(f32::EPSILON);
        for bin in 0..frequencies {
            let frequency = bin as f32 * config.sample_rate as f32 / config.n_fft as f32;
            let weight = if frequency >= left && frequency <= center {
                (frequency - left) / left_width
            } else if frequency > center && frequency <= right {
                (right - frequency) / right_width
            } else {
                0.0
            };
            filterbank[mel * frequencies + bin] = weight;
        }
        if config.norm == MelNorm::Slaney {
            let normalization = 2.0 / (right - left).max(f32::EPSILON);
            for bin in 0..frequencies {
                filterbank[mel * frequencies + bin] *= normalization;
            }
        }
    }
    Ok(filterbank)
}

fn reflect_index(index: isize, length: usize) -> usize {
    if length == 1 {
        return 0;
    }
    let length = length as i128;
    let period = 2 * (length - 1);
    let folded = (index as i128).rem_euclid(period);
    if folded >= length {
        (period - folded) as usize
    } else {
        folded as usize
    }
}

fn encode_f32(values: Vec<f32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AudioConfig {
        AudioConfig {
            sample_rate: 16_000,
            n_fft: 4,
            hop_length: 2,
            n_mels: 2,
            f_min: 0.0,
            f_max: Some(8_000.0),
            mel_scale: MelScale::Htk,
            norm: MelNorm::None,
            log: LogKind::None,
            power: 2.0,
            center: false,
            layout: AudioLayout::MelTime,
            log_eps: 1e-10,
            normalize: FeatureNormalization::None,
            normalize_eps: 1e-5,
            length_input: Some("length".to_string()),
            length_dtype: LengthDtype::Int64,
        }
    }

    #[test]
    fn emits_mel_features_and_derived_frame_count() {
        let bytes = [1.0_f32, 0.0, -1.0, 0.0]
            .iter()
            .flat_map(|sample| sample.to_ne_bytes())
            .collect::<Vec<_>>();
        let shape = [4_i64];
        let input = BorrowedTensor {
            name: "input",
            dtype: KAPSL_DTYPE_F32,
            shape: &shape,
            data: &bytes,
        };
        let preprocessor = AudioPreprocessor::new(config()).unwrap();
        let output = preprocessor.apply(&input).unwrap();
        assert_eq!(output.shape, [1, 2, 1]);
        assert!(output
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .all(f32::is_finite));
        let derived = preprocessor.derived_inputs(&output).unwrap();
        assert_eq!(derived.len(), 1);
        assert_eq!(derived[0].name, "length");
        assert_eq!(derived[0].dtype, KAPSL_DTYPE_I64);
        assert_eq!(
            i64::from_ne_bytes(derived[0].data.clone().try_into().unwrap()),
            1
        );
        assert!(preprocessor.planned_bytes(&input).unwrap() >= output.data.len());
    }

    #[test]
    fn reflection_is_constant_time_for_short_waveforms() {
        assert_eq!(reflect_index(1_000_000, 2), 0);
        assert_eq!(reflect_index(-1_000_001, 2), 1);
        assert_eq!(reflect_index(8, 4), 2);
    }

    #[test]
    fn log10_silence_matches_embedded_profile() {
        let mut config = config();
        config.log = LogKind::Log10;
        config.length_input = None;
        let bytes = [0.0_f32; 4]
            .iter()
            .flat_map(|sample| sample.to_ne_bytes())
            .collect::<Vec<_>>();
        let shape = [4_i64];
        let input = BorrowedTensor {
            name: "input",
            dtype: KAPSL_DTYPE_F32,
            shape: &shape,
            data: &bytes,
        };
        let output = AudioPreprocessor::new(config)
            .unwrap()
            .apply(&input)
            .unwrap();
        assert!(output
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .all(|value| (value + 10.0).abs() < 1e-3));
    }
}
