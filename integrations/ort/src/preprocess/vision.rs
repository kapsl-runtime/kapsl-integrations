//! Encoded JPEG/PNG input to normalized ONNX image tensors.

use crate::tensor::{BorrowedTensor, OwnedTensor};
use crate::{invalid_argument, FfiResult};
use image::{imageops::FilterType, ImageReader, Limits, RgbImage};
use kapsl_backend_abi::{KAPSL_DTYPE_F32, KAPSL_DTYPE_U8};
use serde::Deserialize;
use std::io::Cursor;

const DEFAULT_MAX_DECODE_DIMENSION: u32 = 32_768;
const DEFAULT_MAX_DECODE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ResizeMode {
    #[default]
    Stretch,
    Letterbox,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Layout {
    #[default]
    Nchw,
    Nhwc,
}

fn default_width() -> u32 {
    224
}

fn default_height() -> u32 {
    224
}

fn default_scale() -> f32 {
    1.0 / 255.0
}

fn default_mean() -> [f32; 3] {
    [0.0; 3]
}

fn default_std() -> [f32; 3] {
    [1.0; 3]
}

fn default_max_decode_dimension() -> u32 {
    DEFAULT_MAX_DECODE_DIMENSION
}

fn default_max_decode_bytes() -> u64 {
    DEFAULT_MAX_DECODE_BYTES
}

#[derive(Debug, Deserialize)]
pub(super) struct VisionConfig {
    #[serde(default = "default_width")]
    width: u32,
    #[serde(default = "default_height")]
    height: u32,
    #[serde(default)]
    resize: ResizeMode,
    #[serde(default)]
    layout: Layout,
    #[serde(default = "default_scale")]
    scale: f32,
    #[serde(default = "default_mean")]
    mean: [f32; 3],
    #[serde(default = "default_std")]
    std: [f32; 3],
    #[serde(default)]
    pad: u8,
    #[serde(default = "default_max_decode_dimension")]
    max_decode_width: u32,
    #[serde(default = "default_max_decode_dimension")]
    max_decode_height: u32,
    #[serde(default = "default_max_decode_bytes")]
    max_decode_bytes: u64,
}

pub(crate) struct VisionPreprocessor {
    config: VisionConfig,
}

impl VisionPreprocessor {
    pub(super) fn new(config: VisionConfig) -> FfiResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub(super) fn apply(&self, input: &BorrowedTensor<'_>) -> FfiResult<OwnedTensor> {
        self.validate_input(input)?;
        let decoded = self.reader(input.data)?.decode().map_err(|error| {
            invalid_argument(format!(
                "vision preprocessing could not decode image: {error}"
            ))
        })?;
        let rgb = decoded.to_rgb8();
        let fitted = self.fit(&rgb);
        let values = self.normalize(&fitted)?;
        let shape = match self.config.layout {
            Layout::Nchw => vec![
                1,
                3,
                i64::from(self.config.height),
                i64::from(self.config.width),
            ],
            Layout::Nhwc => vec![
                1,
                i64::from(self.config.height),
                i64::from(self.config.width),
                3,
            ],
        };
        Ok(OwnedTensor {
            name: "input".to_string(),
            dtype: KAPSL_DTYPE_F32,
            shape,
            data: encode_f32(values),
        })
    }

    pub(super) fn planned_bytes(&self, input: &BorrowedTensor<'_>) -> FfiResult<usize> {
        self.validate_input(input)?;
        let (source_width, source_height) =
            self.reader(input.data)?
                .into_dimensions()
                .map_err(|error| {
                    invalid_argument(format!(
                        "vision preprocessing could not inspect image dimensions: {error}"
                    ))
                })?;
        let source_pixels = checked_pixels(source_width, source_height, "decoded image")?;
        let target_pixels = checked_pixels(self.config.width, self.config.height, "image tensor")?;
        // Conservative peak for decoder/original conversion plus resize,
        // optional letterbox canvas, normalized f32 values, and encoded output.
        source_pixels
            .checked_mul(8)
            .and_then(|bytes| {
                target_pixels
                    .checked_mul(19)
                    .and_then(|target| bytes.checked_add(target))
            })
            .ok_or_else(|| invalid_argument("vision preprocessing memory estimate overflows"))
    }

    pub(super) fn expected_shape(&self) -> Vec<i64> {
        match self.config.layout {
            Layout::Nchw => vec![
                1,
                3,
                i64::from(self.config.height),
                i64::from(self.config.width),
            ],
            Layout::Nhwc => vec![
                1,
                i64::from(self.config.height),
                i64::from(self.config.width),
                3,
            ],
        }
    }

    fn validate_input(&self, input: &BorrowedTensor<'_>) -> FfiResult<()> {
        if input.dtype != KAPSL_DTYPE_U8 {
            return Err(invalid_argument(format!(
                "vision preprocessing expects encoded image bytes as uint8, received dtype {}",
                input.dtype
            )));
        }
        if input.data.is_empty() {
            return Err(invalid_argument(
                "vision preprocessing received an empty image payload",
            ));
        }
        Ok(())
    }

    fn reader<'a>(&self, data: &'a [u8]) -> FfiResult<ImageReader<Cursor<&'a [u8]>>> {
        let mut reader = ImageReader::new(Cursor::new(data))
            .with_guessed_format()
            .map_err(|error| {
                invalid_argument(format!(
                    "vision preprocessing could not inspect image: {error}"
                ))
            })?;
        let mut limits = Limits::default();
        limits.max_image_width = Some(self.config.max_decode_width);
        limits.max_image_height = Some(self.config.max_decode_height);
        limits.max_alloc = Some(self.config.max_decode_bytes);
        reader.limits(limits);
        Ok(reader)
    }

    fn fit(&self, image: &RgbImage) -> RgbImage {
        let (width, height) = (self.config.width, self.config.height);
        match self.config.resize {
            ResizeMode::Stretch => {
                image::imageops::resize(image, width, height, FilterType::Triangle)
            }
            ResizeMode::Letterbox => {
                let image_width = image.width() as f64;
                let image_height = image.height() as f64;
                let ratio = (f64::from(width) / image_width).min(f64::from(height) / image_height);
                let resized_width = ((image_width * ratio).round().max(1.0) as u32).min(width);
                let resized_height = ((image_height * ratio).round().max(1.0) as u32).min(height);
                let resized = image::imageops::resize(
                    image,
                    resized_width,
                    resized_height,
                    FilterType::Triangle,
                );
                let pad = image::Rgb([self.config.pad; 3]);
                let mut canvas = RgbImage::from_pixel(width, height, pad);
                let x = i64::from((width - resized_width) / 2);
                let y = i64::from((height - resized_height) / 2);
                image::imageops::overlay(&mut canvas, &resized, x, y);
                canvas
            }
        }
    }

    fn normalize(&self, image: &RgbImage) -> FfiResult<Vec<f32>> {
        let width = usize::try_from(self.config.width)
            .map_err(|_| invalid_argument("vision target width exceeds this platform"))?;
        let height = usize::try_from(self.config.height)
            .map_err(|_| invalid_argument("vision target height exceeds this platform"))?;
        let pixels = width
            .checked_mul(height)
            .ok_or_else(|| invalid_argument("vision target dimensions overflow"))?;
        let elements = pixels
            .checked_mul(3)
            .ok_or_else(|| invalid_argument("vision tensor element count overflows"))?;
        let mut output = vec![0.0; elements];
        match self.config.layout {
            Layout::Nchw => {
                for (index, pixel) in image.pixels().enumerate() {
                    for channel in 0..3 {
                        output[channel * pixels + index] = self.normalized(pixel[channel], channel);
                    }
                }
            }
            Layout::Nhwc => {
                for (index, pixel) in image.pixels().enumerate() {
                    for channel in 0..3 {
                        output[index * 3 + channel] = self.normalized(pixel[channel], channel);
                    }
                }
            }
        }
        Ok(output)
    }

    fn normalized(&self, value: u8, channel: usize) -> f32 {
        (f32::from(value) * self.config.scale - self.config.mean[channel])
            / self.config.std[channel]
    }
}

impl VisionConfig {
    fn validate(&self) -> FfiResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(invalid_argument(
                "vision preprocessing width and height must be non-zero",
            ));
        }
        if self.max_decode_width == 0 || self.max_decode_height == 0 || self.max_decode_bytes == 0 {
            return Err(invalid_argument(
                "vision preprocessing decode limits must be non-zero",
            ));
        }
        if !self.scale.is_finite()
            || self.mean.iter().any(|value| !value.is_finite())
            || self
                .std
                .iter()
                .any(|value| !value.is_finite() || *value == 0.0)
        {
            return Err(invalid_argument(
                "vision preprocessing scale/mean/std must be finite and std must be non-zero",
            ));
        }
        checked_pixels(self.width, self.height, "vision target")?
            .checked_mul(3)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| invalid_argument("vision target tensor size overflows"))?;
        Ok(())
    }
}

fn checked_pixels(width: u32, height: u32, label: &str) -> FfiResult<usize> {
    let width = usize::try_from(width)
        .map_err(|_| invalid_argument(format!("{label} width exceeds this platform")))?;
    let height = usize::try_from(height)
        .map_err(|_| invalid_argument(format!("{label} height exceeds this platform")))?;
    width
        .checked_mul(height)
        .ok_or_else(|| invalid_argument(format!("{label} dimensions overflow")))
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
    use image::{DynamicImage, ImageFormat};

    fn config(layout: Layout) -> VisionConfig {
        VisionConfig {
            width: 2,
            height: 1,
            resize: ResizeMode::Stretch,
            layout,
            scale: 1.0,
            mean: [0.0; 3],
            std: [1.0; 3],
            pad: 0,
            max_decode_width: 16,
            max_decode_height: 16,
            max_decode_bytes: 1024 * 1024,
        }
    }

    fn encoded_image() -> Vec<u8> {
        let image = RgbImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 255, 0])
            }
        });
        let mut output = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image)
            .write_to(&mut output, ImageFormat::Png)
            .unwrap();
        output.into_inner()
    }

    #[test]
    fn emits_nchw_normalized_tensor_and_memory_plan() {
        let bytes = encoded_image();
        let shape = [bytes.len() as i64];
        let input = BorrowedTensor {
            name: "input",
            dtype: KAPSL_DTYPE_U8,
            shape: &shape,
            data: &bytes,
        };
        let preprocessor = VisionPreprocessor::new(config(Layout::Nchw)).unwrap();
        let output = preprocessor.apply(&input).unwrap();
        assert_eq!(output.shape, [1, 3, 1, 2]);
        let values = output
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, [255.0, 0.0, 0.0, 255.0, 0.0, 0.0]);
        assert!(preprocessor.planned_bytes(&input).unwrap() >= output.data.len());
    }

    #[test]
    fn decoder_dimension_limit_rejects_oversized_source() {
        let bytes = encoded_image();
        let shape = [bytes.len() as i64];
        let input = BorrowedTensor {
            name: "input",
            dtype: KAPSL_DTYPE_U8,
            shape: &shape,
            data: &bytes,
        };
        let mut config = config(Layout::Nchw);
        config.max_decode_width = 1;
        let preprocessor = VisionPreprocessor::new(config).unwrap();
        assert!(preprocessor.apply(&input).is_err());
        assert!(preprocessor.planned_bytes(&input).is_err());
    }
}
