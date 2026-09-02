use crate::tensor::{dtype_bytes, from_ort_value, to_session_input, BorrowedTensor, OwnedTensor};
use crate::{backend_error, invalid_argument, FfiResult};
use kapsl_engine_api::{EngineModelInfo, MemoryAllocationClass, MemoryDomain, MemoryReport};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::run_options::{OutputSelector, RunOptions};
use ort::session::{Session, SessionInputValue};
use ort::tensor::TensorElementType;
use ort::value::ValueType;
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::time::Instant;

const MAX_SESSION_BUCKETS: usize = 64;
static ORT_ENVIRONMENT: OnceLock<Result<Arc<ort::environment::Environment>, String>> =
    OnceLock::new();

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct OrtTuning {
    pub(crate) memory_pattern: Option<bool>,
    pub(crate) disable_cpu_mem_arena: Option<bool>,
    pub(crate) session_buckets: Option<usize>,
    pub(crate) bucket_dim_granularity: Option<usize>,
    pub(crate) bucket_max_dims: Option<usize>,
    pub(crate) peak_concurrency_hint: Option<u32>,
}

#[derive(Debug, Clone)]
struct ModelMetadata {
    input_names: Vec<String>,
    output_names: Vec<String>,
    input_shapes: Vec<Vec<i64>>,
    output_shapes: Vec<Vec<i64>>,
    input_dtypes: Vec<String>,
    output_dtypes: Vec<String>,
}

struct LoadedModel {
    model_path: PathBuf,
    bytes: usize,
    metadata: ModelMetadata,
    primary_pool: Arc<SessionPool>,
    bucket_sessions: Mutex<BucketSessionState>,
}

#[derive(Default)]
struct BucketSessionState {
    primary_bucket_key: Option<String>,
    sessions: HashMap<String, Arc<SessionPool>>,
    lru: VecDeque<String>,
}

struct SessionPool {
    inner: Mutex<SessionPoolInner>,
    condvar: Condvar,
    max_sessions: usize,
    waits_total: AtomicU64,
    wait_micros_total: AtomicU64,
}

struct SessionPoolInner {
    idle: Vec<Session>,
    total: usize,
}

struct PooledSession<'a> {
    session: Option<Session>,
    pool: &'a SessionPool,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct SessionPoolStats {
    pub(crate) total_sessions: usize,
    pub(crate) idle_sessions: usize,
    pub(crate) waits_total: u64,
    pub(crate) wait_seconds_total: f64,
}

impl SessionPool {
    fn new(initial: Session, max_sessions: usize) -> Self {
        Self {
            inner: Mutex::new(SessionPoolInner {
                idle: vec![initial],
                total: 1,
            }),
            condvar: Condvar::new(),
            max_sessions: max_sessions.max(1),
            waits_total: AtomicU64::new(0),
            wait_micros_total: AtomicU64::new(0),
        }
    }

    fn acquire<F>(&self, mut create_session: F) -> FfiResult<PooledSession<'_>>
    where
        F: FnMut() -> FfiResult<Session>,
    {
        loop {
            let mut inner = self.lock_inner()?;
            if let Some(session) = inner.idle.pop() {
                return Ok(PooledSession {
                    session: Some(session),
                    pool: self,
                });
            }
            if inner.total < self.max_sessions {
                inner.total += 1;
                drop(inner);
                return match create_session() {
                    Ok(session) => Ok(PooledSession {
                        session: Some(session),
                        pool: self,
                    }),
                    Err(error) => {
                        self.release_reserved_slot();
                        Err(error)
                    }
                };
            }

            self.waits_total.fetch_add(1, Ordering::Relaxed);
            let started = Instant::now();
            let guard = self.condvar.wait(inner).map_err(|_| {
                backend_error("native ORT session pool lock is poisoned while waiting")
            })?;
            self.record_wait(started);
            drop(guard);
        }
    }

    fn reserve_slot(&self) -> FfiResult<bool> {
        let mut inner = self.lock_inner()?;
        if inner.total >= self.max_sessions {
            return Ok(false);
        }
        inner.total += 1;
        Ok(true)
    }

    fn add_reserved_session(&self, session: Session) -> FfiResult<()> {
        let mut inner = self.lock_inner()?;
        inner.idle.push(session);
        self.condvar.notify_one();
        Ok(())
    }

    fn release_reserved_slot(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.total = inner.total.saturating_sub(1);
            self.condvar.notify_one();
        }
    }

    fn stats(&self) -> SessionPoolStats {
        let (total_sessions, idle_sessions) = self
            .inner
            .lock()
            .map(|inner| (inner.total, inner.idle.len()))
            .unwrap_or((0, 0));
        SessionPoolStats {
            total_sessions,
            idle_sessions,
            waits_total: self.waits_total.load(Ordering::Relaxed),
            wait_seconds_total: self.wait_micros_total.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        }
    }

    fn lock_inner(&self) -> FfiResult<std::sync::MutexGuard<'_, SessionPoolInner>> {
        self.inner
            .lock()
            .map_err(|_| backend_error("native ORT session pool lock is poisoned"))
    }

    fn record_wait(&self, started: Instant) {
        let micros = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        self.wait_micros_total.fetch_add(micros, Ordering::Relaxed);
    }
}

impl Deref for PooledSession<'_> {
    type Target = Session;

    fn deref(&self) -> &Self::Target {
        self.session
            .as_ref()
            .expect("pooled ORT session missing before drop")
    }
}

impl DerefMut for PooledSession<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.session
            .as_mut()
            .expect("pooled ORT session missing before drop")
    }
}

impl Drop for PooledSession<'_> {
    fn drop(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };
        match self.pool.inner.lock() {
            Ok(mut inner) => {
                inner.idle.push(session);
                self.pool.condvar.notify_one();
            }
            Err(_) => self.pool.release_reserved_slot(),
        }
    }
}

pub(crate) struct OrtBackend {
    tuning: OrtTuning,
    loaded: RwLock<Option<Arc<LoadedModel>>>,
}

impl OrtBackend {
    pub(crate) fn new(tuning: OrtTuning) -> FfiResult<Self> {
        retain_ort_environment()?;
        Ok(Self {
            tuning,
            loaded: RwLock::new(None),
        })
    }

    pub(crate) fn planned_memory(
        &self,
        model_path: &Path,
        allocation_id: &str,
    ) -> FfiResult<MemoryReport> {
        let bytes = file_bytes(model_path)?.saturating_mul(self.session_pool_size());
        Ok(MemoryReport::single(
            allocation_id,
            MemoryDomain::Host,
            MemoryAllocationClass::ModelSession,
            bytes,
        ))
    }

    pub(crate) fn load(&self, model_path: &Path) -> FfiResult<()> {
        if self.is_loaded()? {
            return Err(backend_error(
                "ORT model is already loaded; unload it before loading another model",
            ));
        }
        let canonical = model_path.canonicalize().map_err(|error| {
            backend_error(format!(
                "resolve ONNX model {}: {error}",
                model_path.display()
            ))
        })?;
        let bytes = file_bytes(&canonical)?;
        let session = self.create_session(&canonical)?;
        let metadata = metadata(&session)?;
        let primary_pool = Arc::new(SessionPool::new(session, self.session_pool_size()));
        self.prewarm_session_pool(&primary_pool, &canonical)?;
        let loaded = Arc::new(LoadedModel {
            model_path: canonical,
            bytes,
            metadata,
            primary_pool,
            bucket_sessions: Mutex::new(BucketSessionState::default()),
        });
        let mut slot = self
            .loaded
            .write()
            .map_err(|_| backend_error("native ORT model state lock is poisoned"))?;
        if slot.is_some() {
            return Err(backend_error(
                "ORT model was loaded concurrently; refusing to replace it",
            ));
        }
        *slot = Some(loaded);
        Ok(())
    }

    pub(crate) fn infer(&self, inputs: &[BorrowedTensor<'_>]) -> FfiResult<OwnedTensor> {
        let loaded = self.loaded_model()?;
        if inputs.is_empty() {
            return Err(invalid_argument("native ORT request has no primary input"));
        }
        let pool = self.pool_for_request(&loaded, &inputs[0])?;
        let mut session = pool.acquire(|| self.create_session(&loaded.model_path))?;
        infer_with_session(&mut session, inputs, &loaded.metadata)
    }

    pub(crate) fn infer_batch(
        &self,
        requests: &[Vec<BorrowedTensor<'_>>],
    ) -> FfiResult<Vec<OwnedTensor>> {
        let mut results = (0..requests.len()).map(|_| None).collect::<Vec<_>>();
        let mut groups: HashMap<BatchKey, Vec<usize>> = HashMap::new();

        for (index, inputs) in requests.iter().enumerate() {
            let Some(primary) = inputs.first() else {
                return Err(invalid_argument(format!(
                    "native ORT batch request {index} has no primary input"
                )));
            };
            if inputs.len() != 1 || primary.shape.len() < 2 {
                results[index] = Some(self.infer(inputs)?);
                continue;
            }
            groups
                .entry(BatchKey {
                    dtype: primary.dtype,
                    trailing_shape: primary.shape[1..].to_vec(),
                })
                .or_default()
                .push(index);
        }

        for indices in groups.into_values() {
            if indices.len() == 1 {
                let index = indices[0];
                results[index] = Some(self.infer(&requests[index])?);
                continue;
            }
            let stacked = stack_group(requests, &indices)
                .and_then(|stacked| self.infer_stacked(stacked, requests, &indices));
            match stacked {
                Ok(outputs) => {
                    for (index, output) in indices.iter().copied().zip(outputs) {
                        results[index] = Some(output);
                    }
                }
                Err(_) => {
                    // Fixed-batch graphs and unusual output layouts may reject
                    // an otherwise compatible stack. Preserve correctness by
                    // using the same per-request path as embedded ORT.
                    for index in indices {
                        results[index] = Some(self.infer(&requests[index])?);
                    }
                }
            }
        }

        results
            .into_iter()
            .enumerate()
            .map(|(index, result)| {
                result.ok_or_else(|| backend_error(format!("missing ORT batch result {index}")))
            })
            .collect()
    }

    fn infer_stacked(
        &self,
        stacked: StackedTensor,
        requests: &[Vec<BorrowedTensor<'_>>],
        indices: &[usize],
    ) -> FfiResult<Vec<OwnedTensor>> {
        let borrowed = BorrowedTensor {
            name: "input",
            dtype: stacked.dtype,
            shape: &stacked.shape,
            data: &stacked.data,
        };
        let output = self.infer(&[borrowed])?;
        let rows = indices
            .iter()
            .map(|index| requests[*index][0].shape[0])
            .collect::<Vec<_>>();
        split_stacked_output(output, &rows)
    }

    pub(crate) fn unload(&self) -> FfiResult<()> {
        let mut loaded = self
            .loaded
            .write()
            .map_err(|_| backend_error("native ORT model state lock is poisoned"))?;
        *loaded = None;
        Ok(())
    }

    pub(crate) fn is_loaded(&self) -> FfiResult<bool> {
        self.loaded
            .read()
            .map(|loaded| loaded.is_some())
            .map_err(|_| backend_error("native ORT model state lock is poisoned"))
    }

    pub(crate) fn loaded_bytes(&self) -> usize {
        self.loaded_model().map_or(0, |loaded| {
            loaded
                .bytes
                .saturating_mul(collect_pool_stats(&loaded).total_sessions)
        })
    }

    pub(crate) fn session_pool_stats(&self) -> SessionPoolStats {
        self.loaded_model()
            .map(|loaded| collect_pool_stats(&loaded))
            .unwrap_or_default()
    }

    pub(crate) fn actual_memory(&self, allocation_id: &str) -> MemoryReport {
        self.loaded_model().map_or_else(
            |_| MemoryReport::default(),
            |loaded| {
                MemoryReport::single(
                    allocation_id,
                    MemoryDomain::Host,
                    MemoryAllocationClass::ModelSession,
                    loaded
                        .bytes
                        .saturating_mul(collect_pool_stats(&loaded).total_sessions),
                )
            },
        )
    }

    pub(crate) fn model_info(&self) -> FfiResult<EngineModelInfo> {
        let model = self.loaded_model()?;
        Ok(EngineModelInfo {
            input_names: model.metadata.input_names.clone(),
            output_names: model.metadata.output_names.clone(),
            input_shapes: model.metadata.input_shapes.clone(),
            output_shapes: model.metadata.output_shapes.clone(),
            input_dtypes: model.metadata.input_dtypes.clone(),
            output_dtypes: model.metadata.output_dtypes.clone(),
            framework: Some("onnx".to_string()),
            model_version: None,
            peak_concurrency: Some(self.session_pool_size() as u32),
        })
    }

    fn loaded_model(&self) -> FfiResult<Arc<LoadedModel>> {
        self.loaded
            .read()
            .map_err(|_| backend_error("native ORT model state lock is poisoned"))?
            .clone()
            .ok_or_else(|| backend_error("ORT model is not loaded"))
    }

    fn session_pool_size(&self) -> usize {
        self.tuning.peak_concurrency_hint.unwrap_or(1).max(1) as usize
    }

    fn max_bucket_sessions(&self) -> usize {
        self.tuning
            .session_buckets
            .unwrap_or(4)
            .clamp(1, MAX_SESSION_BUCKETS)
    }

    fn pool_for_request(
        &self,
        loaded: &Arc<LoadedModel>,
        primary: &BorrowedTensor<'_>,
    ) -> FfiResult<Arc<SessionPool>> {
        if self.max_bucket_sessions() <= 1 {
            return Ok(Arc::clone(&loaded.primary_pool));
        }
        let bucket_key = self.bucket_key(primary);
        let mut state = loaded
            .bucket_sessions
            .lock()
            .map_err(|_| backend_error("native ORT bucket-session lock is poisoned"))?;
        let primary_key = state
            .primary_bucket_key
            .get_or_insert_with(|| bucket_key.clone());
        if *primary_key == bucket_key {
            return Ok(Arc::clone(&loaded.primary_pool));
        }
        if let Some(pool) = state.sessions.get(&bucket_key).cloned() {
            touch_bucket_lru(&mut state, &bucket_key);
            return Ok(pool);
        }

        let secondary_capacity = self.max_bucket_sessions().saturating_sub(1).max(1);
        while state.sessions.len() >= secondary_capacity {
            let Some(evicted) = state.lru.pop_front() else {
                break;
            };
            state.sessions.remove(&evicted);
        }
        let session = self.create_session(&loaded.model_path)?;
        let pool = Arc::new(SessionPool::new(session, self.session_pool_size()));
        state.sessions.insert(bucket_key.clone(), Arc::clone(&pool));
        touch_bucket_lru(&mut state, &bucket_key);
        Ok(pool)
    }

    fn bucket_key(&self, primary: &BorrowedTensor<'_>) -> String {
        let mut key = format!("{}:r{}", primary.dtype, primary.shape.len());
        let max_dims = self.tuning.bucket_max_dims.unwrap_or(4).max(1);
        let granularity = self.tuning.bucket_dim_granularity.unwrap_or(64).max(1) as i64;
        for (index, dimension) in primary.shape.iter().take(max_dims).enumerate() {
            let rounded = if index == 0 {
                *dimension
            } else {
                ((*dimension + granularity - 1) / granularity) * granularity
            };
            key.push(':');
            key.push_str(&rounded.to_string());
        }
        if primary.shape.len() > max_dims {
            key.push_str(":*");
        }
        key
    }

    fn prewarm_session_pool(&self, pool: &SessionPool, model_path: &Path) -> FfiResult<()> {
        while pool.reserve_slot()? {
            match self.create_session(model_path) {
                Ok(session) => pool.add_reserved_session(session)?,
                Err(error) => {
                    pool.release_reserved_slot();
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    fn create_session(&self, model_path: &Path) -> FfiResult<Session> {
        let mut builder = Session::builder()
            .map_err(|error| backend_error(format!("create ORT session builder: {error}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| backend_error(format!("configure ORT optimization: {error}")))?
            .with_memory_pattern(self.tuning.memory_pattern.unwrap_or(true))
            .map_err(|error| backend_error(format!("configure ORT memory pattern: {error}")))?;
        if self.tuning.disable_cpu_mem_arena.unwrap_or(false) {
            builder = builder
                .with_config_entry("session.disable_cpu_mem_arena", "1")
                .map_err(|error| backend_error(format!("disable ORT CPU arena: {error}")))?;
        }
        builder
            .commit_from_file(model_path)
            .map_err(|error| backend_error(format!("load ONNX model: {error}")))
    }
}

fn retain_ort_environment() -> FfiResult<()> {
    match ORT_ENVIRONMENT.get_or_init(|| {
        ort::environment::get_environment()
            .map_err(|error| format!("initialize shared ORT environment: {error}"))
    }) {
        Ok(_) => Ok(()),
        Err(message) => Err(backend_error(message.clone())),
    }
}

#[derive(Hash, Eq, PartialEq)]
struct BatchKey {
    dtype: u32,
    trailing_shape: Vec<i64>,
}

struct StackedTensor {
    dtype: u32,
    shape: Vec<i64>,
    data: Vec<u8>,
}

fn stack_group(
    requests: &[Vec<BorrowedTensor<'_>>],
    indices: &[usize],
) -> FfiResult<StackedTensor> {
    let first = &requests[indices[0]][0];
    let mut total_rows = 0_i64;
    let mut data = Vec::new();
    for index in indices {
        let input = &requests[*index][0];
        if input.dtype != first.dtype || input.shape[1..] != first.shape[1..] {
            return Err(invalid_argument(
                "native ORT batch group contains incompatible tensors",
            ));
        }
        total_rows = total_rows
            .checked_add(input.shape[0])
            .ok_or_else(|| invalid_argument("native ORT batch row count overflows"))?;
        data.extend_from_slice(input.data);
    }
    let mut shape = first.shape.to_vec();
    shape[0] = total_rows;
    Ok(StackedTensor {
        dtype: first.dtype,
        shape,
        data,
    })
}

fn split_stacked_output(output: OwnedTensor, rows: &[i64]) -> FfiResult<Vec<OwnedTensor>> {
    if output.shape.is_empty() {
        return Err(backend_error("stacked ORT output has no batch dimension"));
    }
    let total_rows = rows.iter().try_fold(0_usize, |total, rows| {
        let rows = usize::try_from(*rows)
            .map_err(|_| backend_error("stacked ORT output has an invalid row count"))?;
        total
            .checked_add(rows)
            .ok_or_else(|| backend_error("stacked ORT output row count overflows"))
    })?;
    if usize::try_from(output.shape[0]).ok() != Some(total_rows)
        || total_rows == 0
        || !output.data.len().is_multiple_of(total_rows)
    {
        return Err(backend_error(
            "stacked ORT output does not preserve its batch dimension",
        ));
    }
    let row_bytes = output.data.len() / total_rows;
    if !row_bytes.is_multiple_of(dtype_bytes(output.dtype)?) {
        return Err(backend_error(
            "stacked ORT output row is not element aligned",
        ));
    }
    let mut offset = 0_usize;
    let mut results = Vec::with_capacity(rows.len());
    for rows in rows {
        let row_count = usize::try_from(*rows)
            .map_err(|_| backend_error("stacked ORT output has an invalid row count"))?;
        let byte_len = row_count
            .checked_mul(row_bytes)
            .ok_or_else(|| backend_error("stacked ORT split byte count overflows"))?;
        let end = offset
            .checked_add(byte_len)
            .ok_or_else(|| backend_error("stacked ORT split offset overflows"))?;
        let mut shape = output.shape.clone();
        shape[0] = *rows;
        results.push(OwnedTensor {
            name: output.name.clone(),
            dtype: output.dtype,
            shape,
            data: output.data[offset..end].to_vec(),
        });
        offset = end;
    }
    Ok(results)
}

fn infer_with_session(
    session: &mut Session,
    inputs: &[BorrowedTensor<'_>],
    metadata: &ModelMetadata,
) -> FfiResult<OwnedTensor> {
    if inputs.len() != metadata.input_names.len() {
        return Err(invalid_argument(format!(
            "ONNX model requires {} inputs, request supplied {}",
            metadata.input_names.len(),
            inputs.len()
        )));
    }
    let primary_output_name = metadata
        .output_names
        .first()
        .ok_or_else(|| backend_error("ONNX model declares no outputs"))?;
    let run_options = if metadata.output_names.len() > 1 {
        Some(
            RunOptions::new()
                .map_err(|error| backend_error(format!("create ORT run options: {error}")))?
                .with_outputs(OutputSelector::no_default().with(primary_output_name.as_str())),
        )
    } else {
        None
    };

    let outputs = if inputs.len() == 1 {
        let value = to_session_input(&inputs[0])?;
        if let Some(options) = run_options.as_ref() {
            session.run_with_options([value], options)
        } else {
            session.run([value])
        }
    } else {
        let mut values: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(inputs.len());
        for (index, model_name) in metadata.input_names.iter().enumerate() {
            // The engine ABI bridge names the primary tensor `input`; as with
            // embedded ORT it maps positionally to model input zero. Every
            // additional tensor is matched by the model's declared name.
            let input = if index == 0 {
                &inputs[0]
            } else {
                inputs
                    .iter()
                    .skip(1)
                    .find(|input| input.name == model_name)
                    .ok_or_else(|| {
                        invalid_argument(format!("ONNX request is missing input `{model_name}`"))
                    })?
            };
            values.push((Cow::Borrowed(model_name.as_str()), to_session_input(input)?));
        }
        if let Some(options) = run_options.as_ref() {
            session.run_with_options(values, options)
        } else {
            session.run(values)
        }
    }
    .map_err(|error| backend_error(format!("ORT inference failed: {error}")))?;
    if outputs.len() == 0 {
        return Err(backend_error("ORT returned no primary output"));
    }
    from_ort_value(&outputs[0], primary_output_name)
}

fn collect_pool_stats(loaded: &LoadedModel) -> SessionPoolStats {
    let mut stats = loaded.primary_pool.stats();
    if let Ok(state) = loaded.bucket_sessions.lock() {
        for pool in state.sessions.values() {
            let pool = pool.stats();
            stats.total_sessions = stats.total_sessions.saturating_add(pool.total_sessions);
            stats.idle_sessions = stats.idle_sessions.saturating_add(pool.idle_sessions);
            stats.waits_total = stats.waits_total.saturating_add(pool.waits_total);
            stats.wait_seconds_total += pool.wait_seconds_total;
        }
    }
    stats
}

fn touch_bucket_lru(state: &mut BucketSessionState, bucket_key: &str) {
    if let Some(position) = state.lru.iter().position(|key| key == bucket_key) {
        state.lru.remove(position);
    }
    state.lru.push_back(bucket_key.to_string());
}

fn file_bytes(path: &Path) -> FfiResult<usize> {
    let metadata = std::fs::metadata(path)
        .map_err(|error| backend_error(format!("stat ONNX model {}: {error}", path.display())))?;
    if !metadata.is_file() {
        return Err(invalid_argument(format!(
            "ONNX model {} is not a regular file",
            path.display()
        )));
    }
    usize::try_from(metadata.len())
        .map_err(|_| backend_error("ONNX model size exceeds this platform"))
}

fn metadata(session: &Session) -> FfiResult<ModelMetadata> {
    let input_names = session
        .inputs()
        .iter()
        .map(|input| input.name().to_string())
        .collect::<Vec<_>>();
    let output_names = session
        .outputs()
        .iter()
        .map(|output| output.name().to_string())
        .collect::<Vec<_>>();
    if input_names.is_empty() || output_names.is_empty() {
        return Err(backend_error(
            "ONNX model must declare at least one input and output",
        ));
    }

    let (input_shapes, input_dtypes) = session
        .inputs()
        .iter()
        .map(|input| value_metadata(input.dtype()))
        .unzip();
    let (output_shapes, output_dtypes) = session
        .outputs()
        .iter()
        .map(|output| value_metadata(output.dtype()))
        .unzip();
    Ok(ModelMetadata {
        input_names,
        output_names,
        input_shapes,
        output_shapes,
        input_dtypes,
        output_dtypes,
    })
}

fn value_metadata(value_type: &ValueType) -> (Vec<i64>, String) {
    match value_type {
        ValueType::Tensor { ty, shape, .. } => (
            shape.iter().copied().collect(),
            tensor_element_name(*ty).to_string(),
        ),
        _ => (Vec::new(), "unsupported".to_string()),
    }
}

fn tensor_element_name(element_type: TensorElementType) -> &'static str {
    match element_type {
        TensorElementType::Float32 => "float32",
        TensorElementType::Float64 => "float64",
        TensorElementType::Float16 => "float16",
        TensorElementType::Int32 => "int32",
        TensorElementType::Int64 => "int64",
        TensorElementType::Uint8 => "uint8",
        _ => "unsupported",
    }
}
