//! ORT CUDA allocator forwarding to the backend ABI host.
//!
//! ORT registers one environment allocator per CUDA device, while Kapsl opens
//! one native adapter instance per model replica. The registry below therefore
//! owns one allocator per device and registers each model/replica host as a
//! distinct client. A scoped ORT call selects the exact client; allocations
//! made on an unscoped provider thread fail rather than being charged to an
//! unrelated model.

use kapsl_backend_abi::{
    KapslDeviceAllocateFn, KapslDeviceAllocationRequestV1, KapslDeviceAllocationV1,
    KapslDeviceFreeFn, KapslDeviceSynchronizeFn, KapslLogFn, KapslSlice, KAPSL_LOG_ERROR,
    KAPSL_MEMORY_CUDA, KAPSL_STATUS_OK,
};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::sys as ort_sys;
use ort::AsPointer;
use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
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
    allocate: KapslDeviceAllocateFn,
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
        allocate: KapslDeviceAllocateFn,
        free: KapslDeviceFreeFn,
        synchronize: KapslDeviceSynchronizeFn,
    ) -> Self {
        Self {
            user_data: user_data as usize,
            log,
            allocate,
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
        request: &KapslDeviceAllocationRequestV1,
    ) -> Result<KapslDeviceAllocationV1, i32> {
        let mut allocation = KapslDeviceAllocationV1::empty();
        // SAFETY: the host table remains live and both values are valid for
        // this synchronous callback.
        let status =
            unsafe { (self.allocate)(self.user_data as *mut c_void, request, &mut allocation) };
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

#[derive(Clone, Copy)]
struct AllocationContext {
    device_id: i32,
    client: ClientKey,
    allocation_class: u32,
}

thread_local! {
    static ALLOCATION_CONTEXT: Cell<Option<AllocationContext>> = const { Cell::new(None) };
}

pub(crate) struct AllocationScope {
    previous: Option<AllocationContext>,
}

impl AllocationScope {
    pub(crate) fn enter(device_id: i32, client: ClientKey, allocation_class: u32) -> Self {
        let previous = ALLOCATION_CONTEXT.with(|active| {
            active.replace(Some(AllocationContext {
                device_id,
                client,
                allocation_class,
            }))
        });
        Self { previous }
    }
}

impl Drop for AllocationScope {
    fn drop(&mut self) {
        ALLOCATION_CONTEXT.with(|active| active.set(self.previous));
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
    let context = ALLOCATION_CONTEXT.with(Cell::get);
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
    let request = KapslDeviceAllocationRequestV1::new(
        device_id as u32,
        KAPSL_MEMORY_CUDA,
        context.allocation_class,
        context.client.model_id,
        context.client.replica_id,
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

    #[derive(Default)]
    struct HostProbe {
        next_id: AtomicU64,
        requests: Mutex<Vec<KapslDeviceAllocationRequestV1>>,
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
        request: *const KapslDeviceAllocationRequestV1,
        allocation_out: *mut KapslDeviceAllocationV1,
    ) -> i32 {
        if user_data.is_null() || request.is_null() || allocation_out.is_null() {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: test callbacks receive pointers retained by this test.
        let probe = unsafe { &*user_data.cast::<HostProbe>() };
        let request = unsafe { *request };
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
            .push(request);
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
            let _scope = AllocationScope::enter(0, first_key, KAPSL_ALLOCATION_CLASS_WEIGHTS);
            allocate_scoped(0, &state, 513)
        };
        assert!(!first_pointer.is_null());
        assert_eq!((first_pointer as usize) % 256, 0);

        let second_pointer = {
            let _outer = AllocationScope::enter(0, first_key, KAPSL_ALLOCATION_CLASS_WEIGHTS);
            let _inner = AllocationScope::enter(0, second_key, KAPSL_ALLOCATION_CLASS_WORKSPACE);
            allocate_scoped(0, &state, 1024)
        };
        assert!(!second_pointer.is_null());

        let first_request = first
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())[0];
        assert_eq!(first_request.device_id, 0);
        assert_eq!(first_request.model_id, 11);
        assert_eq!(first_request.replica_id, 2);
        assert_eq!(
            first_request.allocation_class,
            KAPSL_ALLOCATION_CLASS_WEIGHTS
        );
        assert_eq!(first_request.bytes, 513);
        assert_eq!(first_request.alignment, 256);

        let second_request = second
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())[0];
        assert_eq!(second_request.model_id, 19);
        assert_eq!(second_request.replica_id, 4);
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
}
