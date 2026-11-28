//! Session options every profile applies, checked against the real ONNX
//! Runtime without a GPU.

use super::*;
use crate::profile::COMPILED_PROFILE;
use ort::AsPointer;
use std::ffi::{c_int, CString};

const DISABLE_CPU_EP_FALLBACK_KEY: &str = "session.disable_cpu_ep_fallback";
const USE_ENV_ALLOCATORS_KEY: &str = "session.use_env_allocators";
const ONNX_FLOAT: u64 = 1;
const ONNX_INT64: u64 = 7;
const ONNX_ATTRIBUTE_INT: u64 = 2;

fn session_config_entry(
    builder: &ort::session::builder::SessionBuilder,
    key: &str,
) -> Option<String> {
    let api = ort::api();
    let key = CString::new(key).unwrap();
    let mut present: c_int = 0;
    // SAFETY: the builder and key outlive the call and `present` is a valid
    // output slot.
    let status = unsafe { (api.HasSessionConfigEntry)(builder.ptr(), key.as_ptr(), &mut present) };
    assert!(
        status.0.is_null(),
        "ORT must report session option presence"
    );
    if present == 0 {
        return None;
    }
    let mut value = [0u8; 256];
    let mut size = value.len();
    // SAFETY: as above; `value` has `size` writable bytes and ORT writes a
    // NUL-terminated string on success.
    let status = unsafe {
        (api.GetSessionConfigEntry)(
            builder.ptr(),
            key.as_ptr(),
            value.as_mut_ptr().cast(),
            &mut size,
        )
    };
    assert!(
        status.0.is_null(),
        "ORT must expose the applied session option"
    );
    Some(String::from_utf8(value[..size - 1].to_vec()).unwrap())
}

fn varint(output: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        output.push((value as u8) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

fn int_field(output: &mut Vec<u8>, field: u64, value: u64) {
    varint(output, field << 3);
    varint(output, value);
}

fn bytes_field(output: &mut Vec<u8>, field: u64, value: &[u8]) {
    varint(output, (field << 3) | 2);
    varint(output, value.len() as u64);
    output.extend_from_slice(value);
}

fn node(op_type: &str, inputs: &[&str], output: &str, attribute: Option<&[u8]>) -> Vec<u8> {
    let mut node = Vec::new();
    for input in inputs {
        bytes_field(&mut node, 1, input.as_bytes());
    }
    bytes_field(&mut node, 2, output.as_bytes());
    bytes_field(&mut node, 3, output.as_bytes());
    bytes_field(&mut node, 4, op_type.as_bytes());
    if let Some(attribute) = attribute {
        bytes_field(&mut node, 5, attribute);
    }
    node
}

/// A float tensor shaped `[batch, seq, 4]`.
fn float_value_info(name: &str) -> Vec<u8> {
    let mut shape = Vec::new();
    for parameter in ["batch", "seq"] {
        let mut dimension = Vec::new();
        bytes_field(&mut dimension, 2, parameter.as_bytes());
        bytes_field(&mut shape, 1, &dimension);
    }
    let mut dimension = Vec::new();
    int_field(&mut dimension, 1, 4);
    bytes_field(&mut shape, 1, &dimension);
    let mut tensor_type = Vec::new();
    int_field(&mut tensor_type, 1, ONNX_FLOAT);
    bytes_field(&mut tensor_type, 2, &shape);
    let mut value_type = Vec::new();
    bytes_field(&mut value_type, 1, &tensor_type);
    let mut value_info = Vec::new();
    bytes_field(&mut value_info, 1, name.as_bytes());
    bytes_field(&mut value_info, 2, &value_type);
    value_info
}

/// `y = x + seq_len(x)`, with the sequence length computed by the
/// `Shape -> Gather -> Cast` pattern found in transformer exports. ORT 1.23.2's
/// CUDA provider writes `Shape` output to host memory and leaves the `Gather`
/// and `Cast` that consume it to the CPU provider (`GetCpuPreferredNodes`), as
/// it does for the stable gate's GPT-2 generation model.
fn cpu_preferred_shape_subgraph_onnx() -> Vec<u8> {
    let mut seq_axis = Vec::new();
    int_field(&mut seq_axis, 2, ONNX_INT64);
    bytes_field(&mut seq_axis, 8, b"seq_axis");
    bytes_field(&mut seq_axis, 9, &1_i64.to_le_bytes());
    let mut cast_to_float = Vec::new();
    bytes_field(&mut cast_to_float, 1, b"to");
    int_field(&mut cast_to_float, 3, ONNX_FLOAT);
    int_field(&mut cast_to_float, 20, ONNX_ATTRIBUTE_INT);

    let mut graph = Vec::new();
    bytes_field(&mut graph, 1, &node("Shape", &["x"], "x_shape", None));
    bytes_field(
        &mut graph,
        1,
        &node("Gather", &["x_shape", "seq_axis"], "seq_len", None),
    );
    bytes_field(
        &mut graph,
        1,
        &node("Cast", &["seq_len"], "seq_len_f", Some(&cast_to_float)),
    );
    bytes_field(&mut graph, 1, &node("Add", &["x", "seq_len_f"], "y", None));
    bytes_field(&mut graph, 2, b"cpu_preferred_shape_subgraph");
    bytes_field(&mut graph, 5, &seq_axis);
    bytes_field(&mut graph, 11, &float_value_info("x"));
    bytes_field(&mut graph, 12, &float_value_info("y"));

    let mut opset = Vec::new();
    bytes_field(&mut opset, 1, b"");
    int_field(&mut opset, 2, 13);
    let mut model = Vec::new();
    int_field(&mut model, 1, 8);
    bytes_field(&mut model, 7, &graph);
    bytes_field(&mut model, 8, &opset);
    model
}

#[test]
fn sessions_accept_nodes_that_ort_places_on_the_cpu_provider() {
    retain_ort_environment().unwrap();
    let builder = session_builder(&OrtTuning::default()).unwrap();
    assert_eq!(
        session_config_entry(&builder, DISABLE_CPU_EP_FALLBACK_KEY),
        None,
        "ORT 1.23.2 CUDA places shape subgraphs on the CPU provider and this option rejects them"
    );
    #[cfg(any(feature = "profile-cuda12", feature = "profile-tensorrt10"))]
    assert_eq!(
        crate::allocator::USE_ENV_ALLOCATORS_KEY,
        USE_ENV_ALLOCATORS_KEY
    );
    let governed = COMPILED_PROFILE
        .requires_governed_device_memory()
        .then(|| "1".to_string());
    assert_eq!(
        session_config_entry(&builder, USE_ENV_ALLOCATORS_KEY),
        governed,
        "accelerator device memory must come from the governed environment allocator"
    );

    // With `session.disable_cpu_ep_fallback=1` this commit fails exactly as the
    // v0.3.0 GPU gate did: "This session contains graph nodes that are assigned
    // to the default CPU EP, but fallback to CPU EP has been explicitly
    // disabled by the user."
    let directory = tempfile::tempdir().unwrap();
    let model_path = directory.path().join("cpu-preferred-shape-subgraph.onnx");
    std::fs::write(&model_path, cpu_preferred_shape_subgraph_onnx()).unwrap();
    let mut session = builder
        .commit_from_file(&model_path)
        .expect("profile session options must accept CPU-provider shape nodes");
    let input = ort::value::Tensor::from_array((
        [1_usize, 2, 4],
        (0..8).map(|value| value as f32).collect::<Vec<_>>(),
    ))
    .unwrap();
    let outputs = session.run(ort::inputs![input]).unwrap();
    let (shape, values) = outputs[0].try_extract_tensor::<f32>().unwrap();
    assert_eq!(&shape[..], &[1, 2, 4]);
    let expected = (0..8).map(|value| value as f32 + 2.0).collect::<Vec<_>>();
    assert_eq!(values, expected.as_slice());
}

#[cfg(any(feature = "profile-cuda12", feature = "profile-tensorrt10"))]
#[test]
fn an_unavailable_accelerator_provider_fails_instead_of_running_on_cpu() {
    use ort::ep::ExecutionProvider;

    retain_ort_environment().unwrap();
    #[cfg(feature = "profile-cuda12")]
    let available = ort::ep::CUDA::default().is_available().unwrap();
    #[cfg(feature = "profile-tensorrt10")]
    let available = ort::ep::TensorRT::default().is_available().unwrap()
        && ort::ep::CUDA::default().is_available().unwrap();
    if available {
        eprintln!(
            "skipped: this ONNX Runtime provides {}; CI links the CPU runtime",
            COMPILED_PROFILE.label()
        );
        return;
    }
    let builder = session_builder(&OrtTuning::default()).unwrap();
    let error = configure_accelerator_provider(builder, 0)
        .err()
        .expect("an unavailable accelerator provider must not degrade to the CPU provider");
    assert!(
        error.1.starts_with("register required"),
        "unexpected registration error: {}",
        error.1
    );
}
