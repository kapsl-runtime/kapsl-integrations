//! ORT CUDA allocator forwarding to the backend ABI host.
//!
//! ORT registers one environment allocator per CUDA device, while Kapsl opens
//! one native adapter instance per model replica. The registry below therefore
//! owns one allocator per device and registers each model/replica host as a
//! distinct client. A scoped ORT call selects the exact client; allocations
//! made on an unscoped provider thread fail rather than being charged to an
//! unrelated model.

use kapsl_backend_abi::{
    KapslDeviceAllocationScopeV1, KapslDeviceAllocationV1, KapslDeviceFreeFn,
    KapslDeviceSynchronizeFn, KapslLogFn, KapslScopedDeviceAllocateFn,
    KapslScopedDeviceAllocationRequestV1, KapslSlice, KAPSL_ALLOCATION_SCOPE_MODEL,
    KAPSL_ALLOCATION_SCOPE_REPLICA, KAPSL_ALLOCATION_SCOPE_REQUEST,
    KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH, KAPSL_LOG_ERROR, KAPSL_MEMORY_CUDA, KAPSL_STATUS_OK,
};
use kapsl_llm::allocation_scope::{
    DeviceAllocationClass, DeviceAllocationScope, DeviceAllocationScopeGuard,
    DeviceAllocationScopeKind, DeviceAllocationScopeProvider,
};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::sys as ort_sys;
use ort::AsPointer;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) const USE_ENV_ALLOCATORS_KEY: &str = "session.use_env_allocators";

const CUDA_ALLOCATION_ALIGNMENT: u64 = 256;

// ORT API 23 added callback fields after Reserve, but ort-sys rc.11 models
// only the API-22 prefix. Advertising 22 prevents ORT from reading beyond it.
const ORT_ALLOCATOR_ABI_VERSION: u32 = 22;

#[derive(Clone, Copy)]
pub(crate) struct HostDeviceCallbacks {
    user_data: usize,
    log: Option<KapslLogFn>,
    allocate_scoped: KapslScopedDeviceAllocateFn,
    free: KapslDeviceFreeFn,
    synchronize: KapslDeviceSynchronizeFn,
}

// SAFETY: ABI v1 host callbacks remain valid until adapter shutdown returns
// and are explicitly designed to be invoked by concurrent backend work.
unsafe impl Send for HostDeviceCallbacks {}
unsafe impl Sync for HostDeviceCallbacks {}

impl HostDeviceCallbacks {
    pub(crate) fn new(
        user_data: *mut c_void,
        log: Option<KapslLogFn>,
        allocate_scoped: KapslScopedDeviceAllocateFn,
        free: KapslDeviceFreeFn,
        synchronize: KapslDeviceSynchronizeFn,
    ) -> Self {
        Self {
            user_data: user_data as usize,
            log,
            allocate_scoped,
            free,
            synchronize,
        }
    }

    fn emit_error(self, message: &str) {
        if let Some(log) = self.log {
            // SAFETY: callback storage is retained by the matching adapter
            // instance and the message is borrowed only for this call.
            unsafe {
                log(
                    self.user_data as *mut c_void,
                    KAPSL_LOG_ERROR,
                    KapslSlice::from_bytes(message.as_bytes()),
                );
            }
        }
    }

    fn allocate(
        self,
        request: &KapslScopedDeviceAllocationRequestV1,
    ) -> Result<KapslDeviceAllocationV1, i32> {
        let mut allocation = KapslDeviceAllocationV1::empty();
        // SAFETY: the host table remains live and both values are valid for
        // this synchronous callback.
        let status = unsafe {
            (self.allocate_scoped)(self.user_data as *mut c_void, request, &mut allocation)
        };
        if status == KAPSL_STATUS_OK {
            Ok(allocation)
        } else {
            Err(status)
        }
    }

    fn free(self, allocation: &KapslDeviceAllocationV1) -> i32 {
        // SAFETY: allocation is the exact identity returned by this host.
        unsafe { (self.free)(self.user_data as *mut c_void, allocation) }
    }

    fn synchronize(self, device_id: u32) -> i32 {
        // SAFETY: the host table remains live through adapter shutdown.
        unsafe { (self.synchronize)(self.user_data as *mut c_void, device_id) }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ClientKey {
    model_id: u32,
    replica_id: u32,
}

impl ClientKey {
    pub(crate) const fn new(model_id: u32, replica_id: u32) -> Self {
        Self {
            model_id,
            replica_id,
        }
    }
}

#[derive(Clone)]
struct AllocationContext {
    device_id: i32,
    client: ClientKey,
    allocation_class: u32,
    scope_kind: u32,
    scope_id: u64,
    request_ids: Vec<u64>,
}

thread_local! {
    static ALLOCATION_CONTEXT: RefCell<Option<AllocationContext>> = const { RefCell::new(None) };
}

pub(crate) struct AllocationScope {
    previous: Option<AllocationContext>,
}

impl AllocationScope {
    fn enter(
        device_id: i32,
        client: ClientKey,
        allocation_class: u32,
        scope_kind: u32,
        scope_id: u64,
        request_ids: &[u64],
    ) -> Result<Self, String> {
        let scope = KapslDeviceAllocationScopeV1::new(
            scope_kind,
            scope_id,
            client.model_id,
            client.replica_id,
            request_ids,
        );
        if device_id < 0 || request_ids.contains(&0) || !scope.is_well_formed() {
            return Err("ORT governed allocator received an invalid allocation scope".to_string());
        }
        let previous = ALLOCATION_CONTEXT.with(|active| {
            active.borrow_mut().replace(AllocationContext {
                device_id,
                client,
                allocation_class,
                scope_kind,
                scope_id,
                request_ids: request_ids.to_vec(),
            })
        });
        Ok(Self { previous })
    }
}

impl Drop for AllocationScope {
    fn drop(&mut self) {
        ALLOCATION_CONTEXT.with(|active| *active.borrow_mut() = self.previous.take());
    }
}

#[derive(Clone)]
pub(crate) struct AllocationScopeBridge {
    device_id: i32,
    client: ClientKey,
    next_scope_id: Arc<AtomicU64>,
}

impl AllocationScopeBridge {
    pub(crate) fn new(device_id: i32, client: ClientKey) -> Self {
        Self {
            device_id,
            client,
            next_scope_id: Arc::new(AtomicU64::new(1)),
        }
    }

    pub(crate) fn enter_adapter_scope(
        &self,
        allocation_class: u32,
        request_ids: &[u64],
    ) -> Result<AllocationScope, String> {
        let scope_id = self
            .next_scope_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| "ORT governed allocation scope IDs exhausted".to_string())?;
        let scope_kind = match request_ids.len() {
            0 if allocation_class == kapsl_backend_abi::KAPSL_ALLOCATION_CLASS_WEIGHTS => {
                KAPSL_ALLOCATION_SCOPE_MODEL
            }
            0 => KAPSL_ALLOCATION_SCOPE_REPLICA,
            1 => KAPSL_ALLOCATION_SCOPE_REQUEST,
            _ => KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH,
        };
        AllocationScope::enter(
            self.device_id,
            self.client,
            allocation_class,
            scope_kind,
            scope_id,
            request_ids,
        )
    }
}

impl DeviceAllocationScopeProvider for AllocationScopeBridge {
    fn enter(
        &self,
        scope: &DeviceAllocationScope,
    ) -> Result<Box<dyn DeviceAllocationScopeGuard>, String> {
        if !scope.is_well_formed()
            || i32::try_from(scope.device_id).ok() != Some(self.device_id)
            || scope.model_id != self.client.model_id
            || scope.replica_id != self.client.replica_id
            || scope.request_ids.contains(&0)
        {
            return Err(
                "ORT generation allocation scope does not match its adapter instance".to_string(),
            );
        }
        let scope_kind = match scope.kind {
            DeviceAllocationScopeKind::Model => KAPSL_ALLOCATION_SCOPE_MODEL,
            DeviceAllocationScopeKind::Replica => KAPSL_ALLOCATION_SCOPE_REPLICA,
            DeviceAllocationScopeKind::Request => KAPSL_ALLOCATION_SCOPE_REQUEST,
            DeviceAllocationScopeKind::RequestBatch => KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH,
        };
        let allocation_class = match scope.allocation_class {
            DeviceAllocationClass::PersistentWeights => {
                kapsl_backend_abi::KAPSL_ALLOCATION_CLASS_WEIGHTS
            }
            DeviceAllocationClass::KvCache => kapsl_backend_abi::KAPSL_ALLOCATION_CLASS_KV,
            DeviceAllocationClass::TransientWorkspace => {
                kapsl_backend_abi::KAPSL_ALLOCATION_CLASS_WORKSPACE
            }
            DeviceAllocationClass::BlockTable | DeviceAllocationClass::RequestTransient => {
                kapsl_backend_abi::KAPSL_ALLOCATION_CLASS_REQUEST
            }
            DeviceAllocationClass::Other => kapsl_backend_abi::KAPSL_ALLOCATION_CLASS_OTHER,
        };
        AllocationScope::enter(
            self.device_id,
            self.client,
            allocation_class,
            scope_kind,
            scope.scope_id,
            &scope.request_ids,
        )
        .map(|guard| Box::new(guard) as Box<dyn DeviceAllocationScopeGuard>)
    }
}

struct LiveAllocation {
    allocation: KapslDeviceAllocationV1,
    callbacks: HostDeviceCallbacks,
    client: ClientKey,
}

// SAFETY: the opaque device pointer is never dereferenced here. The matching
// host callbacks are the sole authority that consumes it.
unsafe impl Send for LiveAllocation {}

#[derive(Default)]
struct AllocatorInner {
    clients: HashMap<ClientKey, HostDeviceCallbacks>,
    live: HashMap<usize, LiveAllocation>,
}

struct AllocatorState {
    memory_info: MemoryInfo,
    device_id: i32,
    inner: Mutex<AllocatorInner>,
}

#[repr(C)]
struct HostOrtAllocator {
    // MUST remain first: ORT retains a pointer to this vtable and callbacks
    // cast it back to the containing allocation.
    ort: ort_sys::OrtAllocator,
    state: AllocatorState,
}

// SAFETY: callbacks serialize mutable state through `inner`; memory info is
// immutable and the registry pins this allocation at a stable address.
unsafe impl Send for HostOrtAllocator {}

fn emit_error(state: &Mutex<AllocatorInner>, message: &str) {
    let callback = state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clients
        .values()
        .next()
        .copied();
    if let Some(callback) = callback {
        callback.emit_error(message);
    }
}

fn allocate_scoped(device_id: i32, state: &Mutex<AllocatorInner>, size: usize) -> *mut c_void {
    let context = ALLOCATION_CONTEXT.with(|active| active.borrow().clone());
    let Some(context) = context.filter(|context| context.device_id == device_id) else {
        emit_error(
            state,
            &format!(
                "ORT governed allocator on device {device_id} rejected an unscoped allocation of {size} bytes"
            ),
        );
        return std::ptr::null_mut();
    };
    let mut inner = state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(callbacks) = inner.clients.get(&context.client).copied() else {
        drop(inner);
        emit_error(
            state,
            "ORT governed allocator rejected an allocation for an inactive client",
        );
        return std::ptr::null_mut();
    };
    let Ok(bytes) = u64::try_from(size) else {
        callbacks.emit_error("ORT governed allocation exceeds the backend ABI byte range");
        return std::ptr::null_mut();
    };
    let scope = KapslDeviceAllocationScopeV1::new(
        context.scope_kind,
        context.scope_id,
        context.client.model_id,
        context.client.replica_id,
        &context.request_ids,
    );
    let request = KapslScopedDeviceAllocationRequestV1::new(
        device_id as u32,
        KAPSL_MEMORY_CUDA,
        context.allocation_class,
        scope,
        bytes,
        CUDA_ALLOCATION_ALIGNMENT,
    );
    let allocation = match callbacks.allocate(&request) {
        Ok(allocation) => allocation,
        Err(status) => {
            callbacks.emit_error(&format!(
                "ORT governed allocation of {size} bytes was rejected with status {status}"
            ));
            return std::ptr::null_mut();
        }
    };
    let pointer = allocation.device_ptr as usize;
    let valid = allocation.struct_size >= std::mem::size_of::<KapslDeviceAllocationV1>() as u32
        && allocation.reserved == 0
        && allocation.allocation_id != 0
        && pointer != 0
        && allocation.granted_bytes >= bytes
        && pointer.is_multiple_of(CUDA_ALLOCATION_ALIGNMENT as usize);
    if !valid || inner.live.contains_key(&pointer) {
        let _ = callbacks.free(&allocation);
        callbacks.emit_error(
            "ORT governed allocator received an invalid or duplicate host allocation identity",
        );
        return std::ptr::null_mut();
    }
    inner.live.insert(
        pointer,
        LiveAllocation {
            allocation,
            callbacks,
            client: context.client,
        },
    );
    pointer as *mut c_void
}

fn free_scoped(state: &Mutex<AllocatorInner>, pointer: *mut c_void) {
    let mut inner = state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(live) = inner.live.remove(&(pointer as usize)) else {
        drop(inner);
        emit_error(
            state,
            "ORT governed allocator rejected a free for an unknown pointer",
        );
        return;
    };
    let status = live.callbacks.free(&live.allocation);
    if status != KAPSL_STATUS_OK {
        live.callbacks.emit_error(&format!(
            "ORT governed free was rejected with status {status}"
        ));
        inner.live.insert(pointer as usize, live);
    }
}

unsafe extern "system" fn host_alloc(
    this_: *mut ort_sys::OrtAllocator,
    size: usize,
) -> *mut c_void {
    catch_unwind(AssertUnwindSafe(|| {
        if this_.is_null() || size == 0 {
            return std::ptr::null_mut();
        }
        // SAFETY: ORT passes the registered vtable pointer, which is the first
        // field of a registry-pinned HostOrtAllocator.
        let state = unsafe { &(*(this_.cast::<HostOrtAllocator>())).state };
        allocate_scoped(state.device_id, &state.inner, size)
    }))
    .unwrap_or(std::ptr::null_mut())
}

unsafe extern "system" fn host_free(this_: *mut ort_sys::OrtAllocator, pointer: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if this_.is_null() || pointer.is_null() {
            return;
        }
        // SAFETY: same registered-vtable invariant as host_alloc.
        let state = unsafe { &(*(this_.cast::<HostOrtAllocator>())).state };
        free_scoped(&state.inner, pointer);
    }));
}

unsafe extern "system" fn host_info(
    this_: *const ort_sys::OrtAllocator,
) -> *const ort_sys::OrtMemoryInfo {
    if this_.is_null() {
        return std::ptr::null();
    }
    // SAFETY: ORT passes the registered, registry-pinned vtable pointer.
    let state = unsafe { &(*(this_.cast::<HostOrtAllocator>())).state };
    state.memory_info.ptr()
}

unsafe extern "system" fn host_reserve(
    this_: *const ort_sys::OrtAllocator,
    size: usize,
) -> *mut c_void {
    // Reserve uses the same governed allocation path as Alloc.
    unsafe { host_alloc(this_.cast_mut(), size) }
}

struct Registration {
    environment: Arc<ort::environment::Environment>,
    allocator: Box<HostOrtAllocator>,
}

static REGISTRY: OnceLock<Mutex<HashMap<i32, Registration>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<i32, Registration>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn status_to_result(status: ort_sys::OrtStatusPtr) -> Result<(), String> {
    if status.0.is_null() {
        return Ok(());
    }
    // SAFETY: ORT owns a valid error status and message until ReleaseStatus.
    unsafe {
        let message = CStr::from_ptr((ort::api().GetErrorMessage)(status.0))
            .to_string_lossy()
            .into_owned();
        (ort::api().ReleaseStatus)(status.0);
        Err(message)
    }
}

pub(crate) struct AllocatorLease {
    device_id: i32,
    client: ClientKey,
    callbacks: HostDeviceCallbacks,
}

impl Drop for AllocatorLease {
    fn drop(&mut self) {
        if let Err(error) = unregister_client(self.device_id, self.client, self.callbacks) {
            self.callbacks.emit_error(&error);
        }
    }
}

pub(crate) fn register_client(
    device_id: i32,
    client: ClientKey,
    callbacks: HostDeviceCallbacks,
) -> Result<AllocatorLease, String> {
    let mut registry = registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(registration) = registry.get_mut(&device_id) {
        let mut inner = registration
            .allocator
            .state
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if inner.clients.contains_key(&client) {
            return Err(format!(
                "ORT governed allocator client {}/{} is already registered on device {device_id}",
                client.model_id, client.replica_id
            ));
        }
        inner.clients.insert(client, callbacks);
        return Ok(AllocatorLease {
            device_id,
            client,
            callbacks,
        });
    }

    let environment = ort::environment::get_environment()
        .map_err(|error| format!("obtain shared ORT environment: {error}"))?;
    let memory_info = MemoryInfo::new(
        AllocationDevice::CUDA,
        device_id,
        AllocatorType::Device,
        MemoryType::Default,
    )
    .map_err(|error| format!("create ORT CUDA memory info: {error}"))?;
    let mut clients = HashMap::new();
    clients.insert(client, callbacks);
    let mut allocator = Box::new(HostOrtAllocator {
        ort: ort_sys::OrtAllocator {
            version: ORT_ALLOCATOR_ABI_VERSION,
            Alloc: Some(host_alloc),
            Free: Some(host_free),
            Info: Some(host_info),
            Reserve: Some(host_reserve),
        },
        state: AllocatorState {
            memory_info,
            device_id,
            inner: Mutex::new(AllocatorInner {
                clients,
                live: HashMap::new(),
            }),
        },
    });
    // SAFETY: the Box pins the allocator until successful unregistration.
    let status =
        unsafe { (ort::api().RegisterAllocator)(environment.ptr().cast_mut(), &mut allocator.ort) };
    status_to_result(status).map_err(|error| format!("ORT RegisterAllocator failed: {error}"))?;
    registry.insert(
        device_id,
        Registration {
            environment,
            allocator,
        },
    );
    Ok(AllocatorLease {
        device_id,
        client,
        callbacks,
    })
}

fn unregister_client(
    device_id: i32,
    client: ClientKey,
    callbacks: HostDeviceCallbacks,
) -> Result<(), String> {
    let synchronize_status = callbacks.synchronize(device_id as u32);
    let mut registry = registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(registration) = registry.get_mut(&device_id) else {
        return Err(format!(
            "ORT governed allocator for device {device_id} was already unregistered"
        ));
    };
    let mut reclaimed = 0_usize;
    let clients_remaining = {
        let mut inner = registration
            .allocator
            .state
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let live_pointers = inner
            .live
            .iter()
            .filter_map(|(pointer, live)| (live.client == client).then_some(*pointer))
            .collect::<Vec<_>>();
        for pointer in live_pointers {
            if let Some(live) = inner.live.remove(&pointer) {
                let _ = live.callbacks.free(&live.allocation);
                reclaimed += 1;
            }
        }
        if inner.clients.remove(&client).is_none() {
            return Err(format!(
                "ORT governed allocator client {}/{} was already unregistered on device {device_id}",
                client.model_id, client.replica_id
            ));
        }
        !inner.clients.is_empty()
    };

    let mut errors = Vec::new();
    if synchronize_status != KAPSL_STATUS_OK {
        errors.push(format!(
            "device synchronization failed with status {synchronize_status}"
        ));
    }
    if reclaimed != 0 {
        errors.push(format!(
            "reclaimed {reclaimed} governed ORT allocations during client shutdown"
        ));
    }
    if !clients_remaining {
        // SAFETY: every client and live allocation has been removed, so ORT
        // can no longer legally call this allocator after unregistration.
        let status = unsafe {
            (ort::api().UnregisterAllocator)(
                registration.environment.ptr().cast_mut(),
                registration.allocator.state.memory_info.ptr(),
            )
        };
        match status_to_result(status) {
            Ok(()) => {
                registry.remove(&device_id);
            }
            Err(error) => errors.push(format!("ORT UnregisterAllocator failed: {error}")),
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "ORT governed allocator shutdown on device {device_id}: {}",
            errors.join("; ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_backend_abi::{
        KAPSL_ALLOCATION_CLASS_WEIGHTS, KAPSL_ALLOCATION_CLASS_WORKSPACE,
        KAPSL_STATUS_BACKEND_ERROR, KAPSL_STATUS_INVALID_ARGUMENT,
    };
    use std::alloc::{alloc, dealloc, Layout};
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    struct HostAllocation {
        pointer: usize,
        layout: Layout,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct ObservedRequest {
        device_id: u32,
        allocation_class: u32,
        scope_kind: u32,
        scope_id: u64,
        model_id: u32,
        replica_id: u32,
        request_ids: Vec<u64>,
        bytes: u64,
        alignment: u64,
    }

    #[derive(Default)]
    struct HostProbe {
        next_id: AtomicU64,
        requests: Mutex<Vec<ObservedRequest>>,
        live: Mutex<HashMap<u64, HostAllocation>>,
        logs: Mutex<Vec<String>>,
        synchronizations: AtomicUsize,
    }

    impl Drop for HostProbe {
        fn drop(&mut self) {
            let live = self
                .live
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .drain()
                .map(|(_, allocation)| allocation)
                .collect::<Vec<_>>();
            for allocation in live {
                // SAFETY: every pointer was created with this exact layout.
                unsafe { dealloc(allocation.pointer as *mut u8, allocation.layout) };
            }
        }
    }

    unsafe extern "C" fn test_allocate(
        user_data: *mut c_void,
        request: *const KapslScopedDeviceAllocationRequestV1,
        allocation_out: *mut KapslDeviceAllocationV1,
    ) -> i32 {
        if user_data.is_null() || request.is_null() || allocation_out.is_null() {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: test callbacks receive pointers retained by this test.
        let probe = unsafe { &*user_data.cast::<HostProbe>() };
        let request = unsafe { *request };
        if !request.is_well_formed() {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: the adapter retains the request-ID slice for this callback.
        let Some(request_ids) = (unsafe { request.scope.request_ids() }) else {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        };
        let Ok(size) = usize::try_from(request.bytes) else {
            return KAPSL_STATUS_BACKEND_ERROR;
        };
        let Ok(alignment) = usize::try_from(request.alignment) else {
            return KAPSL_STATUS_BACKEND_ERROR;
        };
        let Ok(layout) = Layout::from_size_align(size, alignment) else {
            return KAPSL_STATUS_BACKEND_ERROR;
        };
        // SAFETY: layout is non-zero and valid; test_free owns deallocation.
        let pointer = unsafe { alloc(layout) };
        if pointer.is_null() {
            return KAPSL_STATUS_BACKEND_ERROR;
        }
        let allocation_id = probe.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        probe
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(ObservedRequest {
                device_id: request.device_id,
                allocation_class: request.allocation_class,
                scope_kind: request.scope.scope_kind,
                scope_id: request.scope.scope_id,
                model_id: request.scope.model_id,
                replica_id: request.scope.replica_id,
                request_ids: request_ids.to_vec(),
                bytes: request.bytes,
                alignment: request.alignment,
            });
        probe
            .live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                allocation_id,
                HostAllocation {
                    pointer: pointer as usize,
                    layout,
                },
            );
        // SAFETY: allocation_out was checked and is writable for this call.
        unsafe {
            *allocation_out = KapslDeviceAllocationV1 {
                struct_size: std::mem::size_of::<KapslDeviceAllocationV1>() as u32,
                reserved: 0,
                allocation_id,
                device_ptr: pointer.cast(),
                granted_bytes: request.bytes,
            }
        };
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn test_free(
        user_data: *mut c_void,
        allocation: *const KapslDeviceAllocationV1,
    ) -> i32 {
        if user_data.is_null() || allocation.is_null() {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: test callbacks receive pointers retained by this test.
        let probe = unsafe { &*user_data.cast::<HostProbe>() };
        let allocation = unsafe { *allocation };
        let Some(host) = probe
            .live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&allocation.allocation_id)
        else {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        };
        if host.pointer != allocation.device_ptr as usize
            || host.layout.size() as u64 != allocation.granted_bytes
        {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: pointer and layout are the exact pair created by test_allocate.
        unsafe { dealloc(host.pointer as *mut u8, host.layout) };
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn test_synchronize(user_data: *mut c_void, _device_id: u32) -> i32 {
        if user_data.is_null() {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: the test retains its probe through this callback.
        let probe = unsafe { &*user_data.cast::<HostProbe>() };
        probe.synchronizations.fetch_add(1, Ordering::Relaxed);
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn test_log(user_data: *mut c_void, _level: u32, message: KapslSlice) {
        if user_data.is_null() {
            return;
        }
        // SAFETY: the test retains its probe and the message for this call.
        let probe = unsafe { &*user_data.cast::<HostProbe>() };
        let message = unsafe { message.as_bytes() }
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
            .unwrap_or("invalid log message")
            .to_string();
        probe
            .logs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(message);
    }

    fn callbacks(probe: &HostProbe) -> HostDeviceCallbacks {
        HostDeviceCallbacks::new(
            (probe as *const HostProbe).cast_mut().cast(),
            Some(test_log),
            test_allocate,
            test_free,
            test_synchronize,
        )
    }

    #[test]
    fn scoped_allocations_route_to_the_exact_model_replica_and_free_identity() {
        let first = Box::new(HostProbe::default());
        let second = Box::new(HostProbe::default());
        let first_key = ClientKey::new(11, 2);
        let second_key = ClientKey::new(19, 4);
        let state = Mutex::new(AllocatorInner {
            clients: HashMap::from([
                (first_key, callbacks(&first)),
                (second_key, callbacks(&second)),
            ]),
            live: HashMap::new(),
        });

        assert!(allocate_scoped(0, &state, 512).is_null());
        let logged_unscoped = [&first, &second].into_iter().any(|probe| {
            probe
                .logs
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .any(|message| message.contains("unscoped allocation"))
        });
        assert!(logged_unscoped);

        let first_pointer = {
            let _scope = AllocationScope::enter(
                0,
                first_key,
                KAPSL_ALLOCATION_CLASS_WEIGHTS,
                KAPSL_ALLOCATION_SCOPE_MODEL,
                1,
                &[],
            )
            .unwrap();
            allocate_scoped(0, &state, 513)
        };
        assert!(!first_pointer.is_null());
        assert_eq!((first_pointer as usize) % 256, 0);

        let second_pointer = {
            let _outer = AllocationScope::enter(
                0,
                first_key,
                KAPSL_ALLOCATION_CLASS_WEIGHTS,
                KAPSL_ALLOCATION_SCOPE_MODEL,
                2,
                &[],
            )
            .unwrap();
            let _inner = AllocationScope::enter(
                0,
                second_key,
                KAPSL_ALLOCATION_CLASS_WORKSPACE,
                KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH,
                3,
                &[71, 72],
            )
            .unwrap();
            allocate_scoped(0, &state, 1024)
        };
        assert!(!second_pointer.is_null());

        let first_request = first
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())[0]
            .clone();
        assert_eq!(first_request.device_id, 0);
        assert_eq!(first_request.model_id, 11);
        assert_eq!(first_request.replica_id, 2);
        assert_eq!(first_request.scope_kind, KAPSL_ALLOCATION_SCOPE_MODEL);
        assert_eq!(first_request.scope_id, 1);
        assert!(first_request.request_ids.is_empty());
        assert_eq!(
            first_request.allocation_class,
            KAPSL_ALLOCATION_CLASS_WEIGHTS
        );
        assert_eq!(first_request.bytes, 513);
        assert_eq!(first_request.alignment, 256);

        let second_request = second
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())[0]
            .clone();
        assert_eq!(second_request.model_id, 19);
        assert_eq!(second_request.replica_id, 4);
        assert_eq!(
            second_request.scope_kind,
            KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH
        );
        assert_eq!(second_request.scope_id, 3);
        assert_eq!(second_request.request_ids, [71, 72]);
        assert_eq!(
            second_request.allocation_class,
            KAPSL_ALLOCATION_CLASS_WORKSPACE
        );

        free_scoped(&state, first_pointer);
        free_scoped(&state, second_pointer);
        assert!(state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .live
            .is_empty());
        assert!(first
            .live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_empty());
        assert!(second
            .live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_empty());
    }

    #[test]
    fn generation_bridge_preserves_scope_identity_and_rejects_foreign_owners() {
        let bridge = AllocationScopeBridge::new(0, ClientKey::new(11, 2));
        let scope = DeviceAllocationScope {
            kind: DeviceAllocationScopeKind::Request,
            scope_id: 44,
            device_id: 0,
            model_id: 11,
            replica_id: 2,
            allocation_class: DeviceAllocationClass::KvCache,
            request_ids: vec![91],
        };
        let guard = DeviceAllocationScopeProvider::enter(&bridge, &scope).unwrap();
        let active = ALLOCATION_CONTEXT.with(|context| context.borrow().clone().unwrap());
        assert_eq!(active.scope_kind, KAPSL_ALLOCATION_SCOPE_REQUEST);
        assert_eq!(active.scope_id, 44);
        assert_eq!(
            active.allocation_class,
            kapsl_backend_abi::KAPSL_ALLOCATION_CLASS_KV
        );
        assert_eq!(active.request_ids, [91]);
        drop(guard);
        assert!(ALLOCATION_CONTEXT.with(|context| context.borrow().is_none()));

        let mut foreign = scope;
        foreign.replica_id = 3;
        assert!(DeviceAllocationScopeProvider::enter(&bridge, &foreign).is_err());
    }

    #[test]
    fn adapter_scope_ids_remain_unique_across_sequential_loads() {
        let bridge = AllocationScopeBridge::new(0, ClientKey::new(7, 1));
        let first = bridge
            .enter_adapter_scope(KAPSL_ALLOCATION_CLASS_WEIGHTS, &[])
            .unwrap();
        let first_id =
            ALLOCATION_CONTEXT.with(|context| context.borrow().as_ref().unwrap().scope_id);
        drop(first);
        let second = bridge
            .enter_adapter_scope(KAPSL_ALLOCATION_CLASS_WEIGHTS, &[])
            .unwrap();
        let second_id =
            ALLOCATION_CONTEXT.with(|context| context.borrow().as_ref().unwrap().scope_id);
        drop(second);
        assert_ne!(first_id, second_id);
    }
}
