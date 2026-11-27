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

/// Order of the three color channels in the emitted tensor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ChannelOrder {
    /// Red, green, blue: the order the image is decoded in.
    #[default]
    Rgb,
    /// Blue, green, red: for models trained on OpenCV-loaded images.
    Bgr,
}

impl ChannelOrder {
    /// Index into a decoded RGB pixel for output channel `channel`.
    fn source_channel(self, channel: usize) -> usize {
        match self {
            Self::Rgb => channel,
            Self::Bgr => 2 - channel,
        }
    }
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
    #[serde(default)]
    channel_order: ChannelOrder,
    #[serde(default = "default_max_decode_dimension")]
    max_decode_width: u32,
    #[serde(default = "default_max_decode_dimension")]
    max_decode_height: u32,
    #[serde(default = "default_max_decode_bytes")]
    max_decode_bytes: u64,
}

/// Where a source image lands inside the model input: the single source of
/// truth for resize, scale and pad. The preprocessor applies it and detection
/// postprocessing inverts it to report boxes in original-image coordinates.
/// The arithmetic is deliberately single precision so the rounded resize is
/// identical to the embedded ORT backend for every source size.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FitGeometry {
    /// Source (original image) size in pixels.
    source_width: u32,
    source_height: u32,
    /// Size the source is resized to before padding. Equals the model input
    /// for `stretch`; smaller on one axis for `letterbox`.
    fit_width: u32,
    fit_height: u32,
    /// Top-left of the resized image inside the model input.
    pad_x: u32,
    pad_y: u32,
}

impl FitGeometry {
    /// Horizontal and vertical factor from source pixels to model-input
    /// pixels: the exact ratio of the rounded resize, not the nominal
    /// letterbox ratio.
    fn scale(&self) -> (f32, f32) {
        (
            self.fit_width as f32 / self.source_width as f32,
            self.fit_height as f32 / self.source_height as f32,
        )
    }

    /// Map a point from model-input space back to source-image space
    /// without clipping.
    fn to_source(self, x: f32, y: f32) -> (f32, f32) {
        let (scale_x, scale_y) = self.scale();
        (
            (x - self.pad_x as f32) / scale_x,
            (y - self.pad_y as f32) / scale_y,
        )
    }
}

/// The geometry one request's image was preprocessed with, kept so task
/// postprocessing can report results against the image the client sent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ImagePlacement {
    geometry: FitGeometry,
    model_width: u32,
    model_height: u32,
}

impl ImagePlacement {
    /// Original image size in pixels.
    pub(crate) fn source_size(&self) -> (f32, f32) {
        (
            self.geometry.source_width as f32,
            self.geometry.source_height as f32,
        )
    }

    /// Model input size in pixels, after resize and padding.
    pub(crate) fn model_size(&self) -> (f32, f32) {
        (self.model_width as f32, self.model_height as f32)
    }

    /// Map a model-input point back to the original image without clipping.
    pub(crate) fn to_source(self, x: f32, y: f32) -> (f32, f32) {
        self.geometry.to_source(x, y)
    }
}

pub(crate) struct VisionPreprocessor {
    config: VisionConfig,
}

impl VisionPreprocessor {
    pub(super) fn new(config: VisionConfig) -> FfiResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub(super) fn apply(
        &self,
        input: &BorrowedTensor<'_>,
    ) -> FfiResult<(OwnedTensor, ImagePlacement)> {
        self.validate_input(input)?;
        let decoded = self.reader(input.data)?.decode().map_err(|error| {
            invalid_argument(format!(
                "vision preprocessing could not decode image: {error}"
            ))
        })?;
        let rgb = decoded.to_rgb8();
        let geometry = self.config.geometry(rgb.width(), rgb.height());
        let fitted = self.fit(&rgb, &geometry);
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
        let tensor = OwnedTensor {
            name: "input".to_string(),
            dtype: KAPSL_DTYPE_F32,
            shape,
            data: encode_f32(values),
        };
        let placement = ImagePlacement {
            geometry,
            model_width: self.config.width,
            model_height: self.config.height,
        };
        Ok((tensor, placement))
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

    fn fit(&self, image: &RgbImage, geometry: &FitGeometry) -> RgbImage {
        let resized = image::imageops::resize(
            image,
            geometry.fit_width,
            geometry.fit_height,
            FilterType::Triangle,
        );
        match self.config.resize {
            ResizeMode::Stretch => resized,
            ResizeMode::Letterbox => {
                let pad = image::Rgb([self.config.pad; 3]);
                let mut canvas = RgbImage::from_pixel(self.config.width, self.config.height, pad);
                image::imageops::overlay(
                    &mut canvas,
                    &resized,
                    i64::from(geometry.pad_x),
                    i64::from(geometry.pad_y),
                );
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
        let order = self.config.channel_order;
        let mut output = vec![0.0; elements];
        match self.config.layout {
            Layout::Nchw => {
                for (index, pixel) in image.pixels().enumerate() {
                    for channel in 0..3 {
                        output[channel * pixels + index] =
                            self.normalized(pixel[order.source_channel(channel)], channel);
                    }
                }
            }
            Layout::Nhwc => {
                for (index, pixel) in image.pixels().enumerate() {
                    for channel in 0..3 {
                        output[index * 3 + channel] =
                            self.normalized(pixel[order.source_channel(channel)], channel);
                    }
                }
            }
        }
        Ok(output)
    }

    /// `(pixel * scale - mean[c]) / std[c]`, where `c` indexes the *output*
    /// channel, that is, after any `channel_order` swap.
    fn normalized(&self, value: u8, channel: usize) -> f32 {
        (f32::from(value) * self.config.scale - self.config.mean[channel])
            / self.config.std[channel]
    }
}

impl VisionConfig {
    /// Resize/pad geometry for a `source_width x source_height` image.
    fn geometry(&self, source_width: u32, source_height: u32) -> FitGeometry {
        let (width, height) = (self.width, self.height);
        match self.resize {
            ResizeMode::Stretch => FitGeometry {
                source_width,
                source_height,
                fit_width: width,
                fit_height: height,
                pad_x: 0,
                pad_y: 0,
            },
            ResizeMode::Letterbox => {
                // Scale to fit within the box, preserving aspect ratio.
                let (image_width, image_height) = (source_width as f32, source_height as f32);
                let ratio = (width as f32 / image_width).min(height as f32 / image_height);
                let fit_width = ((image_width * ratio).round().max(1.0) as u32).min(width);
                let fit_height = ((image_height * ratio).round().max(1.0) as u32).min(height);
                FitGeometry {
                    source_width,
                    source_height,
                    fit_width,
                    fit_height,
                    // Centered within the canvas.
                    pad_x: (width - fit_width) / 2,
                    pad_y: (height - fit_height) / 2,
                }
            }
        }
    }

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
            channel_order: ChannelOrder::Rgb,
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
        let (output, placement) = preprocessor.apply(&input).unwrap();
        assert_eq!(output.shape, [1, 3, 1, 2]);
        assert_eq!(placement.source_size(), (2.0, 1.0));
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

    // ---- parity with the embedded ORT vision preprocessor ----
    //
    // Ported from `kapsl-backends/src/preprocess/vision_tests.rs` with the same
    // numeric cases, so both routes are held to one specification.

    /// A vision block parsed exactly as `metadata.preprocess` is.
    fn parsed(yaml: &str) -> VisionConfig {
        serde_yaml::from_str(yaml).unwrap()
    }

    /// Encode a solid-color `width x height` image to PNG bytes.
    fn solid_png(width: u32, height: u32, color: [u8; 3]) -> Vec<u8> {
        let image = RgbImage::from_pixel(width, height, image::Rgb(color));
        let mut output = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image)
            .write_to(&mut output, ImageFormat::Png)
            .unwrap();
        output.into_inner()
    }

    fn run(
        preprocessor: &VisionPreprocessor,
        bytes: &[u8],
    ) -> (Vec<i64>, Vec<f32>, ImagePlacement) {
        let shape = [bytes.len() as i64];
        let input = BorrowedTensor {
            name: "input",
            dtype: KAPSL_DTYPE_U8,
            shape: &shape,
            data: bytes,
        };
        let (output, placement) = preprocessor.apply(&input).unwrap();
        assert_eq!(output.dtype, KAPSL_DTYPE_F32);
        let values = output
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_ne_bytes(*chunk))
            .collect();
        (output.shape, values, placement)
    }

    #[test]
    fn nchw_solid_color_normalizes_per_channel() {
        let preprocessor = VisionPreprocessor::new(parsed(
            "{width: 4, height: 4, resize: stretch, layout: nchw}",
        ))
        .unwrap();
        // Solid red 8x8 stretched to 4x4.
        let (shape, values, _) = run(&preprocessor, &solid_png(8, 8, [255, 0, 0]));
        assert_eq!(shape, [1, 3, 4, 4]);
        let plane = 16;
        // Default scale 1/255, mean 0, std 1: R plane is 1.0, G and B are 0.0.
        for red in &values[0..plane] {
            assert!(
                (red - 1.0).abs() < 1e-6,
                "R channel expected 1.0, got {red}"
            );
        }
        for other in &values[plane..3 * plane] {
            assert!(other.abs() < 1e-6, "G/B channel expected 0.0, got {other}");
        }
    }

    #[test]
    fn mean_std_normalization_applied() {
        let preprocessor = VisionPreprocessor::new(parsed(
            "{width: 2, height: 2, mean: [0.5, 0.0, 0.0], std: [0.5, 1.0, 1.0]}",
        ))
        .unwrap();
        let (_, values, _) = run(&preprocessor, &solid_png(2, 2, [255, 0, 0]));
        // R = (255/255 - 0.5) / 0.5 = 1.0
        assert!((values[0] - 1.0).abs() < 1e-6, "got {}", values[0]);
    }

    #[test]
    fn nhwc_layout_is_interleaved() {
        let preprocessor =
            VisionPreprocessor::new(parsed("{width: 1, height: 1, layout: nhwc}")).unwrap();
        let (shape, values, _) = run(&preprocessor, &solid_png(1, 1, [255, 128, 0]));
        assert_eq!(shape, [1, 1, 1, 3]);
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[1] - 128.0 / 255.0).abs() < 1e-6);
        assert!(values[2].abs() < 1e-6);
    }

    #[test]
    fn letterbox_pads_non_square_input() {
        let preprocessor =
            VisionPreprocessor::new(parsed("{width: 10, height: 10, resize: letterbox, pad: 0}"))
                .unwrap();
        // Wide 20x10 white image fits to 10x5 centered; top/bottom rows pad 0.
        let (shape, values, _) = run(&preprocessor, &solid_png(20, 10, [255, 255, 255]));
        assert_eq!(shape, [1, 3, 10, 10]);
        assert!(values[0].abs() < 1e-6, "padded corner, got {}", values[0]);
        let center = 5 * 10 + 5;
        assert!(
            (values[center] - 1.0).abs() < 1e-6,
            "image center, got {}",
            values[center]
        );
    }

    #[test]
    fn rejects_non_uint8_and_undecodable_input() {
        let preprocessor = VisionPreprocessor::new(parsed("{}")).unwrap();
        let shape = [1_i64];
        let floats = BorrowedTensor {
            name: "input",
            dtype: KAPSL_DTYPE_F32,
            shape: &shape,
            data: &[0, 0, 0, 0],
        };
        assert!(preprocessor.apply(&floats).is_err());
        let junk_shape = [5_i64];
        let junk = BorrowedTensor {
            name: "input",
            dtype: KAPSL_DTYPE_U8,
            shape: &junk_shape,
            data: &[1, 2, 3, 4, 5],
        };
        assert!(preprocessor.apply(&junk).is_err());
    }

    #[test]
    fn zero_std_config_rejected() {
        assert!(VisionPreprocessor::new(parsed("{std: [0.0, 1.0, 1.0]}")).is_err());
    }

    #[test]
    fn bgr_channel_order_swaps_planes_and_keeps_mean_std_per_output_channel() {
        for (layout, shape) in [("nchw", [1, 3, 2, 2]), ("nhwc", [1, 2, 2, 3])] {
            // mean is indexed by output channel: out[0] is B, so it takes mean[0].
            let preprocessor = VisionPreprocessor::new(parsed(&format!(
                "{{width: 2, height: 2, resize: stretch, layout: {layout}, scale: 1.0, \
                 mean: [10.0, 0.0, 0.0], channel_order: bgr}}"
            )))
            .unwrap();
            let (output_shape, values, _) = run(&preprocessor, &solid_png(2, 2, [200, 100, 50]));
            assert_eq!(output_shape, shape);
            let expected = [50.0 - 10.0, 100.0, 200.0]; // B-mean, G, R
            for (index, value) in values.iter().enumerate() {
                let channel = if layout == "nchw" {
                    index / 4
                } else {
                    index % 3
                };
                assert!(
                    (value - expected[channel]).abs() < 1e-4,
                    "{layout} index {index}: got {value}, want {}",
                    expected[channel]
                );
            }
        }
    }

    #[test]
    fn channel_order_defaults_to_rgb_and_parses_from_manifest_yaml() {
        assert_eq!(parsed("{}").channel_order, ChannelOrder::Rgb);
        assert_eq!(
            parsed("channel_order: bgr").channel_order,
            ChannelOrder::Bgr
        );
        assert_eq!(parsed("width: 8").channel_order, ChannelOrder::Rgb);
        assert!(serde_yaml::from_str::<VisionConfig>("channel_order: gbr").is_err());
    }

    #[test]
    fn letterbox_geometry_centers_and_inverts() {
        let geometry = parsed("{width: 100, height: 100, resize: letterbox}").geometry(200, 100);
        assert_eq!(
            (
                geometry.fit_width,
                geometry.fit_height,
                geometry.pad_x,
                geometry.pad_y
            ),
            (100, 50, 0, 25)
        );
        assert_eq!(geometry.scale(), (0.5, 0.5));
        // Content corners map back to source corners.
        assert_eq!(geometry.to_source(0.0, 25.0), (0.0, 0.0));
        assert_eq!(geometry.to_source(100.0, 75.0), (200.0, 100.0));
    }

    #[test]
    fn letterbox_geometry_uses_the_rounded_resize_for_scale() {
        // 300x101 -> ratio 1/3; fit 100x34 (33.67 rounds up), so the y scale
        // is 34/101, not the nominal 1/3.
        let geometry = parsed("{width: 100, height: 100, resize: letterbox}").geometry(300, 101);
        assert_eq!((geometry.fit_width, geometry.fit_height), (100, 34));
        assert!((geometry.scale().1 - 34.0 / 101.0).abs() < 1e-6);
        assert_eq!(geometry.pad_y, 33);
    }

    #[test]
    fn letterbox_rounding_is_single_precision_like_embedded_ort() {
        // These sources land within one ulp of a half pixel. Embedded ORT
        // computes the ratio in f32 and rounds up; double precision rounds
        // down and would resize, pad and normalize a different image.
        let config = parsed("{width: 224, height: 224, resize: letterbox}");
        for (source, fit, double_precision_fit) in [
            ((320, 175), (224, 123), (224, 122)),
            ((90, 640), (32, 224), (31, 224)),
            ((1472, 391), (224, 59), (224, 60)),
        ] {
            let geometry = config.geometry(source.0, source.1);
            assert_eq!((geometry.fit_width, geometry.fit_height), fit, "{source:?}");
            let ratio = (224.0 / f64::from(source.0)).min(224.0 / f64::from(source.1));
            let double = (
                (f64::from(source.0) * ratio).round() as u32,
                (f64::from(source.1) * ratio).round() as u32,
            );
            assert_eq!(double, double_precision_fit, "{source:?}");
        }
    }

    #[test]
    fn stretch_geometry_has_no_pad() {
        let geometry = parsed("{width: 64, height: 32, resize: stretch}").geometry(128, 128);
        assert_eq!(
            (
                geometry.fit_width,
                geometry.fit_height,
                geometry.pad_x,
                geometry.pad_y
            ),
            (64, 32, 0, 0)
        );
        assert_eq!(geometry.scale(), (0.5, 0.25));
    }

    #[test]
    fn geometry_matches_the_pixels_the_preprocessor_emits() {
        // A white 8x4 image letterboxed into 8x8 with pad 0: content is rows 2..6.
        let preprocessor = VisionPreprocessor::new(parsed(
            "{width: 8, height: 8, resize: letterbox, layout: nchw}",
        ))
        .unwrap();
        let (_, values, placement) = run(&preprocessor, &solid_png(8, 4, [255, 255, 255]));
        let geometry = placement.geometry;
        assert_eq!(geometry, preprocessor.config.geometry(8, 4));
        for row in 0..8_u32 {
            let inside = row >= geometry.pad_y && row < geometry.pad_y + geometry.fit_height;
            assert_eq!(values[(row * 8) as usize] > 0.5, inside, "row {row}");
        }
    }

    #[test]
    fn placement_reports_the_decoded_source_and_model_sizes() {
        let preprocessor =
            VisionPreprocessor::new(parsed("{width: 64, height: 32, resize: letterbox}")).unwrap();
        let (_, _, placement) = run(&preprocessor, &solid_png(13, 7, [1, 2, 3]));
        assert_eq!(placement.source_size(), (13.0, 7.0));
        assert_eq!(placement.model_size(), (64.0, 32.0));
    }
}
