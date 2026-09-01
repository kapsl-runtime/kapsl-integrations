use crate::{backend_error, invalid_argument, FfiResult};
use half::f16;
use kapsl_backend_abi::*;
use ort::session::SessionInputValue;
use ort::value::{DynValue, TensorRef, Value};
use std::borrow::Cow;

const MAX_INPUTS: usize = 256;
const MAX_NAME_BYTES: usize = 1024;
const MAX_TENSOR_RANK: usize = 32;

#[derive(Clone, Copy)]
pub(crate) struct BorrowedTensor<'a> {
    pub(crate) name: &'a str,
    pub(crate) dtype: u32,
    pub(crate) shape: &'a [i64],
    pub(crate) data: &'a [u8],
}

pub(crate) struct OwnedTensor {
    pub(crate) name: String,
    pub(crate) dtype: u32,
    pub(crate) shape: Vec<i64>,
    pub(crate) data: Vec<u8>,
}

impl OwnedTensor {
    pub(crate) fn as_borrowed(&self) -> BorrowedTensor<'_> {
        BorrowedTensor {
            name: &self.name,
            dtype: self.dtype,
            shape: &self.shape,
            data: &self.data,
        }
    }
}

pub(crate) unsafe fn request_tensors<'a>(
    request: *const KapslInferenceRequestV1,
) -> FfiResult<(u64, Vec<BorrowedTensor<'a>>)> {
    if request.is_null() {
        return Err(invalid_argument("native ORT request is null"));
    }
    // Read only struct_size until the caller proves the complete header exists.
    let struct_size = unsafe { request.cast::<u32>().read() };
    if struct_size < std::mem::size_of::<KapslInferenceRequestV1>() as u32 {
        return Err(invalid_argument("native ORT request struct is truncated"));
    }
    // SAFETY: struct_size covers every v1 request field read below.
    let request = unsafe { &*request };
    if request.wire_format != KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1 || request.reserved != 0 {
        return Err(invalid_argument(
            "native ORT request has an unsupported wire format or reserved value",
        ));
    }
    let count = usize::try_from(request.input_count)
        .map_err(|_| invalid_argument("native ORT input count exceeds this platform"))?;
    if count == 0 || count > MAX_INPUTS || request.inputs.is_null() {
        return Err(invalid_argument(format!(
            "native ORT request must contain 1..={MAX_INPUTS} inputs"
        )));
    }
    let mut tensors = Vec::with_capacity(count);
    for index in 0..count {
        // SAFETY: input_count is bounded and the host promises an array with
        // that many entries for this synchronous call.
        let input = unsafe { request.inputs.add(index) };
        let struct_size = unsafe { input.cast::<u32>().read() };
        if struct_size < std::mem::size_of::<KapslNamedTensorViewV1>() as u32 {
            return Err(invalid_argument(format!(
                "native ORT named tensor {index} is truncated"
            )));
        }
        // SAFETY: struct_size covers the complete named tensor entry.
        tensors.push(unsafe { tensor_from_wire(&*input) }?);
    }
    for left in 0..tensors.len() {
        for right in (left + 1)..tensors.len() {
            if tensors[left].name == tensors[right].name {
                return Err(invalid_argument(format!(
                    "native ORT request repeats input name `{}`",
                    tensors[left].name
                )));
            }
        }
    }
    Ok((request.request_id, tensors))
}

unsafe fn tensor_from_wire<'a>(input: &'a KapslNamedTensorViewV1) -> FfiResult<BorrowedTensor<'a>> {
    if input.struct_size < std::mem::size_of::<KapslNamedTensorViewV1>() as u32
        || input.tensor.struct_size < std::mem::size_of::<KapslTensorViewV1>() as u32
    {
        return Err(invalid_argument("native ORT tensor view is truncated"));
    }
    if input.reserved != 0 {
        return Err(invalid_argument(
            "native ORT named tensor has a non-zero reserved field",
        ));
    }
    // SAFETY: the host retains name storage for this synchronous call.
    let name = unsafe { input.name.as_bytes() }
        .ok_or_else(|| invalid_argument("native ORT tensor name has a null pointer"))?;
    if name.is_empty() || name.len() > MAX_NAME_BYTES {
        return Err(invalid_argument(format!(
            "native ORT tensor names must contain 1..={MAX_NAME_BYTES} bytes"
        )));
    }
    let name = std::str::from_utf8(name).map_err(|error| {
        invalid_argument(format!("native ORT tensor name is not UTF-8: {error}"))
    })?;

    let tensor = &input.tensor;
    let supported_flags = KAPSL_TENSOR_FLAG_CONTIGUOUS | KAPSL_TENSOR_FLAG_READ_ONLY;
    if tensor.flags & KAPSL_TENSOR_FLAG_CONTIGUOUS == 0 || tensor.flags & !supported_flags != 0 {
        return Err(invalid_argument(
            "native ORT accepts only contiguous tensors with known flags",
        ));
    }
    if !matches!(
        tensor.memory_kind,
        KAPSL_MEMORY_HOST | KAPSL_MEMORY_HOST_PINNED
    ) {
        return Err(invalid_argument(format!(
            "CPU ORT adapter cannot consume memory kind {}",
            tensor.memory_kind
        )));
    }
    if tensor.device_id != -1 {
        return Err(invalid_argument(
            "CPU ORT host tensors must use device_id -1",
        ));
    }
    let rank = usize::try_from(tensor.rank)
        .map_err(|_| invalid_argument("native ORT tensor rank exceeds this platform"))?;
    if rank > MAX_TENSOR_RANK || (rank > 0 && tensor.shape.is_null()) {
        return Err(invalid_argument(format!(
            "native ORT tensor rank must be at most {MAX_TENSOR_RANK}"
        )));
    }
    let shape = if rank == 0 {
        &[]
    } else {
        // SAFETY: rank is bounded and the host retains shape storage.
        unsafe { std::slice::from_raw_parts(tensor.shape, rank) }
    };
    let elements = shape_elements(shape)?;
    let element_bytes = dtype_bytes(tensor.dtype)?;
    let expected = elements
        .checked_mul(element_bytes)
        .ok_or_else(|| invalid_argument("native ORT tensor byte length overflows"))?;
    let actual = usize::try_from(tensor.byte_len)
        .map_err(|_| invalid_argument("native ORT tensor byte length exceeds this platform"))?;
    if actual != expected || (actual > 0 && tensor.data.is_null()) {
        return Err(invalid_argument(format!(
            "native ORT tensor `{name}` requires {expected} bytes, received {actual}"
        )));
    }
    let data = if actual == 0 {
        &[]
    } else {
        // SAFETY: byte length was validated and host storage remains borrowed.
        unsafe { std::slice::from_raw_parts(tensor.data.cast::<u8>(), actual) }
    };
    Ok(BorrowedTensor {
        name,
        dtype: tensor.dtype,
        shape,
        data,
    })
}

fn shape_elements(shape: &[i64]) -> FfiResult<usize> {
    if shape.is_empty() {
        return Ok(1);
    }
    shape.iter().try_fold(1_usize, |elements, dimension| {
        let dimension = usize::try_from(*dimension).map_err(|_| {
            invalid_argument(format!(
                "native ORT tensor has invalid dimension {dimension}"
            ))
        })?;
        if dimension == 0 {
            return Err(invalid_argument(
                "native ORT request tensors may not contain zero-sized dimensions",
            ));
        }
        elements
            .checked_mul(dimension)
            .ok_or_else(|| invalid_argument("native ORT tensor element count overflows"))
    })
}

pub(crate) fn dtype_bytes(dtype: u32) -> FfiResult<usize> {
    match dtype {
        KAPSL_DTYPE_U8 => Ok(1),
        KAPSL_DTYPE_F16 => Ok(2),
        KAPSL_DTYPE_I32 | KAPSL_DTYPE_F32 => Ok(4),
        KAPSL_DTYPE_I64 | KAPSL_DTYPE_F64 => Ok(8),
        other => Err(invalid_argument(format!(
            "CPU ORT adapter does not support input dtype {other}"
        ))),
    }
}

enum PreparedInput<'a> {
    F32(Cow<'a, [f32]>),
    F64(Cow<'a, [f64]>),
    F16(Vec<f16>),
    I32(Cow<'a, [i32]>),
    I64(Cow<'a, [i64]>),
    U8(&'a [u8]),
}

pub(crate) fn to_session_input<'a>(input: &BorrowedTensor<'a>) -> FfiResult<SessionInputValue<'a>> {
    let shape = input
        .shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension)
                .map_err(|_| invalid_argument("native ORT tensor dimension is negative"))
        })
        .collect::<FfiResult<Vec<_>>>()?;
    let elements = shape_elements(input.shape)?;
    let prepared = match input.dtype {
        KAPSL_DTYPE_F32 => PreparedInput::F32(parse_f32(input.data, elements)),
        KAPSL_DTYPE_F64 => PreparedInput::F64(parse_f64(input.data, elements)),
        KAPSL_DTYPE_F16 => PreparedInput::F16(parse_f16(input.data, elements)),
        KAPSL_DTYPE_I32 => PreparedInput::I32(parse_i32(input.data, elements)),
        KAPSL_DTYPE_I64 => PreparedInput::I64(parse_i64(input.data, elements)),
        KAPSL_DTYPE_U8 => PreparedInput::U8(input.data),
        other => {
            return Err(invalid_argument(format!(
                "CPU ORT adapter does not support input dtype {other}"
            )))
        }
    };
    let value = match prepared {
        PreparedInput::F32(values) => value_from_cow(shape, values),
        PreparedInput::F64(values) => value_from_cow(shape, values),
        PreparedInput::F16(values) => Value::from_array((shape, values)).map(Into::into),
        PreparedInput::I32(values) => value_from_cow(shape, values),
        PreparedInput::I64(values) => value_from_cow(shape, values),
        PreparedInput::U8(values) => TensorRef::from_array_view((shape, values)).map(Into::into),
    }
    .map_err(|error| backend_error(format!("construct ORT input tensor: {error}")))?;
    Ok(value)
}

fn value_from_cow<'a, T>(
    shape: Vec<usize>,
    values: Cow<'a, [T]>,
) -> ort::Result<SessionInputValue<'a>>
where
    T: ort::tensor::PrimitiveTensorElementType + Clone + std::fmt::Debug + 'static,
{
    match values {
        Cow::Borrowed(values) => TensorRef::from_array_view((shape, values)).map(Into::into),
        Cow::Owned(values) => Value::from_array((shape, values)).map(Into::into),
    }
}

fn aligned_slice<T: Copy>(bytes: &[u8]) -> Option<&[T]> {
    // SAFETY: call sites use only primitive numeric POD types and validate the
    // exact byte length before reaching this helper.
    let (prefix, values, suffix) = unsafe { bytes.align_to::<T>() };
    (prefix.is_empty() && suffix.is_empty()).then_some(values)
}

fn parse_f32(bytes: &[u8], elements: usize) -> Cow<'_, [f32]> {
    aligned_slice(bytes).map(Cow::Borrowed).unwrap_or_else(|| {
        Cow::Owned(
            bytes
                .chunks_exact(4)
                .take(elements)
                .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte chunk")))
                .collect(),
        )
    })
}

fn parse_f64(bytes: &[u8], elements: usize) -> Cow<'_, [f64]> {
    aligned_slice(bytes).map(Cow::Borrowed).unwrap_or_else(|| {
        Cow::Owned(
            bytes
                .chunks_exact(8)
                .take(elements)
                .map(|chunk| f64::from_ne_bytes(chunk.try_into().expect("eight-byte chunk")))
                .collect(),
        )
    })
}

fn parse_f16(bytes: &[u8], elements: usize) -> Vec<f16> {
    bytes
        .chunks_exact(2)
        .take(elements)
        .map(|chunk| {
            f16::from_bits(u16::from_ne_bytes(
                chunk.try_into().expect("two-byte chunk"),
            ))
        })
        .collect()
}

fn parse_i32(bytes: &[u8], elements: usize) -> Cow<'_, [i32]> {
    aligned_slice(bytes).map(Cow::Borrowed).unwrap_or_else(|| {
        Cow::Owned(
            bytes
                .chunks_exact(4)
                .take(elements)
                .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("four-byte chunk")))
                .collect(),
        )
    })
}

fn parse_i64(bytes: &[u8], elements: usize) -> Cow<'_, [i64]> {
    aligned_slice(bytes).map(Cow::Borrowed).unwrap_or_else(|| {
        Cow::Owned(
            bytes
                .chunks_exact(8)
                .take(elements)
                .map(|chunk| i64::from_ne_bytes(chunk.try_into().expect("eight-byte chunk")))
                .collect(),
        )
    })
}

pub(crate) fn from_ort_value(value: &DynValue, name: &str) -> FfiResult<OwnedTensor> {
    if let Ok((shape, values)) = value.try_extract_tensor::<f32>() {
        return Ok(OwnedTensor {
            name: name.to_string(),
            dtype: KAPSL_DTYPE_F32,
            shape: shape.to_vec(),
            data: primitive_bytes(values),
        });
    }
    if let Ok((shape, values)) = value.try_extract_tensor::<f64>() {
        return Ok(OwnedTensor {
            name: name.to_string(),
            dtype: KAPSL_DTYPE_F64,
            shape: shape.to_vec(),
            data: primitive_bytes(values),
        });
    }
    if let Ok((shape, values)) = value.try_extract_tensor::<f16>() {
        return Ok(OwnedTensor {
            name: name.to_string(),
            dtype: KAPSL_DTYPE_F16,
            shape: shape.to_vec(),
            data: primitive_bytes(values),
        });
    }
    if let Ok((shape, values)) = value.try_extract_tensor::<i32>() {
        return Ok(OwnedTensor {
            name: name.to_string(),
            dtype: KAPSL_DTYPE_I32,
            shape: shape.to_vec(),
            data: primitive_bytes(values),
        });
    }
    if let Ok((shape, values)) = value.try_extract_tensor::<i64>() {
        return Ok(OwnedTensor {
            name: name.to_string(),
            dtype: KAPSL_DTYPE_I64,
            shape: shape.to_vec(),
            data: primitive_bytes(values),
        });
    }
    if let Ok((shape, values)) = value.try_extract_tensor::<u8>() {
        return Ok(OwnedTensor {
            name: name.to_string(),
            dtype: KAPSL_DTYPE_U8,
            shape: shape.to_vec(),
            data: values.to_vec(),
        });
    }
    Err(backend_error(
        "ORT returned an unsupported output tensor dtype",
    ))
}

fn primitive_bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    let byte_len = values
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .expect("ORT output byte length overflow");
    // SAFETY: ORT exposes a contiguous initialized primitive slice, and the
    // Kapsl tensor contract uses the same native-endian representation.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len) }.to_vec()
}
