//! Private adapter-to-provider bridge. This is not an engine ABI extension.
//!
//! Each ORT execution provider opens a separate handle. A handle captures the
//! calling thread's engine scope at construction and at each Run, so provider
//! worker threads never infer ownership from a process-wide current request.

use super::*;

pub(crate) const SESSION_ALLOCATOR_KEY: &str = "kapsl.ort.scoped_allocator.v1";
const ERROR: i32 = kapsl_backend_abi::KAPSL_STATUS_INVALID_ARGUMENT;
static NEXT_PROVIDER_ID: AtomicU64 = AtomicU64::new(1);

/// Keep in sync with runtime/include/kapsl_ort_allocator.h. The table and its
/// broker live until all session builders and providers have been destroyed.
#[repr(C)]
pub(crate) struct ProviderAllocatorV1 {
    struct_size: u32,
    version: u32,
    user_data: *mut c_void,
    open: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    close: unsafe extern "C" fn(*mut c_void),
    begin: unsafe extern "C" fn(*mut c_void) -> i32,
    end: unsafe extern "C" fn(*mut c_void) -> i32,
    allocate: unsafe extern "C" fn(*mut c_void, u64, u64) -> *mut c_void,
    free: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32,
}

#[repr(C)]
pub(crate) struct ProviderAllocator {
    table: ProviderAllocatorV1,
    device_id: i32,
    client: ClientKey,
    inner: Arc<Mutex<AllocatorInner>>,
}

// SAFETY: the broker is pinned in a Box. Mutable allocation state is locked;
// the user_data pointer refers only to this immutable, retained broker.
unsafe impl Send for ProviderAllocator {}
unsafe impl Sync for ProviderAllocator {}

impl ProviderAllocator {
    pub(super) fn new(
        device_id: i32,
        client: ClientKey,
        inner: Arc<Mutex<AllocatorInner>>,
    ) -> Box<Self> {
        let mut broker = Box::new(Self {
            table: ProviderAllocatorV1 {
                struct_size: std::mem::size_of::<ProviderAllocatorV1>() as u32,
                version: 1,
                user_data: std::ptr::null_mut(),
                open,
                close,
                begin,
                end,
                allocate,
                free,
            },
            device_id,
            client,
            inner,
        });
        broker.table.user_data = (&mut *broker as *mut Self).cast();
        broker
    }

    pub(crate) fn address(&self) -> String {
        (&self.table as *const ProviderAllocatorV1 as usize).to_string()
    }
}

struct ProviderSession {
    id: u64,
    device_id: i32,
    client: ClientKey,
    inner: Arc<Mutex<AllocatorInner>>,
    active: Mutex<Option<AllocationContext>>,
}

fn capture(device_id: i32, client: ClientKey) -> Option<AllocationContext> {
    ALLOCATION_CONTEXT.with(|active| {
        active
            .borrow()
            .as_ref()
            .filter(|scope| scope.device_id == device_id && scope.client == client)
            .cloned()
    })
}

unsafe extern "C" fn open(user_data: *mut c_void) -> *mut c_void {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: a provider uses the broker retained by its adapter lease.
        let Some(broker) = (unsafe { user_data.cast::<ProviderAllocator>().as_ref() }) else {
            return std::ptr::null_mut();
        };
        let Some(scope) = capture(broker.device_id, broker.client) else {
            return std::ptr::null_mut();
        };
        if !broker
            .inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clients
            .contains_key(&broker.client)
        {
            return std::ptr::null_mut();
        }
        let Ok(id) = NEXT_PROVIDER_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        else {
            return std::ptr::null_mut();
        };
        Box::into_raw(Box::new(ProviderSession {
            id,
            device_id: broker.device_id,
            client: broker.client,
            inner: Arc::clone(&broker.inner),
            active: Mutex::new(Some(scope)),
        }))
        .cast()
    }))
    .unwrap_or(std::ptr::null_mut())
}

unsafe extern "C" fn begin(handle: *mut c_void) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: open returned this handle; close follows all provider work.
        let Some(session) = (unsafe { handle.cast::<ProviderSession>().as_ref() }) else {
            return ERROR;
        };
        let Some(scope) = capture(session.device_id, session.client) else {
            return ERROR;
        };
        let mut active = session.active.lock().unwrap_or_else(|p| p.into_inner());
        if active.is_some()
            || !session
                .inner
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clients
                .contains_key(&session.client)
        {
            return ERROR;
        }
        *active = Some(scope);
        KAPSL_STATUS_OK
    }))
    .unwrap_or(ERROR)
}

unsafe extern "C" fn end(handle: *mut c_void) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: same open/close lifetime contract as begin.
        let Some(session) = (unsafe { handle.cast::<ProviderSession>().as_ref() }) else {
            return ERROR;
        };
        // Allocation holds this lock through the host callback. Ending a scope
        // cannot race an allocation into a subsequent request's ownership.
        if session
            .active
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
            .is_none()
        {
            return ERROR;
        }
        KAPSL_STATUS_OK
    }))
    .unwrap_or(ERROR)
}

unsafe extern "C" fn allocate(handle: *mut c_void, bytes: u64, alignment: u64) -> *mut c_void {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: same open/close lifetime contract as begin.
        let Some(session) = (unsafe { handle.cast::<ProviderSession>().as_ref() }) else {
            return std::ptr::null_mut();
        };
        let Ok(size) = usize::try_from(bytes) else {
            return std::ptr::null_mut();
        };
        // TensorRT permits zero (default) or a power-of-two CUDA alignment.
        if alignment != 0 && !alignment.is_power_of_two() {
            return std::ptr::null_mut();
        }
        let active = session.active.lock().unwrap_or_else(|p| p.into_inner());
        let Some(scope) = active.as_ref() else {
            return std::ptr::null_mut();
        };
        allocate_with_context(
            &session.inner,
            scope,
            size,
            alignment.max(CUDA_ALLOCATION_ALIGNMENT),
            Some(session.id),
        )
    }))
    .unwrap_or(std::ptr::null_mut())
}

fn free_owned(session: &ProviderSession, pointer: *mut c_void) -> i32 {
    if pointer.is_null() {
        return KAPSL_STATUS_OK;
    }
    let live = {
        let mut inner = session.inner.lock().unwrap_or_else(|p| p.into_inner());
        if !inner.clients.contains_key(&session.client) {
            return ERROR;
        }
        match inner.live.get(&(pointer as usize)) {
            Some(live) if live.client == session.client && live.provider_id == Some(session.id) => {
            }
            _ => return ERROR,
        }
        // Remove while synchronizing to reject concurrent/double frees. Do not
        // hold any allocator lock while waiting for device work to complete.
        inner
            .live
            .remove(&(pointer as usize))
            .expect("validated allocation")
    };
    let mut status = live.callbacks.synchronize(session.device_id as u32);
    if status == KAPSL_STATUS_OK {
        status = live.callbacks.free(&live.allocation);
    }
    if status != KAPSL_STATUS_OK {
        live.callbacks.emit_error(&format!("ORT provider retains allocation {} after synchronization/free failed with status {status}", live.allocation.allocation_id));
        session
            .inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .live
            .insert(pointer as usize, live);
    }
    status
}

unsafe extern "C" fn free(handle: *mut c_void, pointer: *mut c_void) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: same open/close lifetime contract as begin.
        let Some(session) = (unsafe { handle.cast::<ProviderSession>().as_ref() }) else {
            return ERROR;
        };
        free_owned(session, pointer)
    }))
    .unwrap_or(ERROR)
}

unsafe extern "C" fn close(handle: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            return;
        }
        // SAFETY: close consumes the unique handle from open exactly once,
        // after the provider has drained its calls and destroyed its objects.
        let session = unsafe { Box::from_raw(handle.cast::<ProviderSession>()) };
        session
            .active
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        let pointers = session
            .inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .live
            .iter()
            .filter_map(|(pointer, live)| {
                (live.client == session.client && live.provider_id == Some(session.id))
                    .then_some(*pointer)
            })
            .collect::<Vec<_>>();
        for pointer in pointers {
            // Failed cleanup stays in the client ledger for unload retry. No
            // ownership or allocation identity is discarded with this handle.
            let _ = free_owned(&session, pointer as *mut c_void);
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::tests::{callbacks, HostProbe};
    use kapsl_backend_abi::{KAPSL_ALLOCATION_CLASS_WEIGHTS, KAPSL_ALLOCATION_CLASS_WORKSPACE};

    fn broker(probe: &HostProbe, device: i32, client: ClientKey) -> Box<ProviderAllocator> {
        ProviderAllocator::new(
            device,
            client,
            Arc::new(Mutex::new(AllocatorInner {
                clients: HashMap::from([(client, callbacks(probe))]),
                live: HashMap::new(),
            })),
        )
    }

    fn scope(client: ClientKey, device: i32, id: u64, requests: &[u64]) -> AllocationScope {
        let (kind, class) = match requests.len() {
            0 => (KAPSL_ALLOCATION_SCOPE_MODEL, KAPSL_ALLOCATION_CLASS_WEIGHTS),
            1 => (
                KAPSL_ALLOCATION_SCOPE_REQUEST,
                KAPSL_ALLOCATION_CLASS_WORKSPACE,
            ),
            _ => (
                KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH,
                KAPSL_ALLOCATION_CLASS_WORKSPACE,
            ),
        };
        AllocationScope::enter(device, client, class, kind, id, requests).unwrap()
    }

    #[test]
    fn provider_worker_threads_preserve_load_request_and_batch_ownership() {
        let probe = HostProbe::default();
        let client = ClientKey::new(41, 2);
        let broker = broker(&probe, 3, client);
        assert_eq!(
            broker.table.struct_size as usize,
            std::mem::size_of::<ProviderAllocatorV1>()
        );
        assert_eq!(broker.table.version, 1);
        assert_eq!(
            std::mem::size_of::<ProviderAllocatorV1>(),
            8 + 7 * std::mem::size_of::<usize>()
        );
        assert!(unsafe { (broker.table.open)(broker.table.user_data) }.is_null());
        let model = scope(client, 3, 7, &[]);
        let handle = unsafe { (broker.table.open)(broker.table.user_data) };
        assert!(!handle.is_null());
        // No ambient scope exists in this worker. It must use the provider's
        // captured owner, including the original engine scope ID.
        let address = handle as usize;
        let weight = std::thread::spawn(move || unsafe {
            allocate(address as *mut c_void, 512, 0) as usize
        })
        .join()
        .unwrap();
        assert_ne!(weight, 0);
        assert_eq!(unsafe { (broker.table.end)(handle) }, KAPSL_STATUS_OK);
        drop(model);
        assert!(unsafe { allocate(handle, 64, 256) }.is_null());
        assert_eq!(unsafe { begin(handle) }, ERROR);
        for (id, requests) in [(8, vec![71]), (9, vec![72, 73])] {
            let _scope = scope(client, 3, id, &requests);
            assert_eq!(unsafe { begin(handle) }, KAPSL_STATUS_OK);
            assert_eq!(unsafe { begin(handle) }, ERROR); // Reject overlap.
            let pointer = std::thread::spawn(move || unsafe {
                allocate(address as *mut c_void, 128, 512) as usize
            })
            .join()
            .unwrap();
            assert_ne!(pointer, 0);
            assert_eq!(unsafe { end(handle) }, KAPSL_STATUS_OK);
            assert!(unsafe { allocate(handle, 64, 256) }.is_null());
            assert_eq!(
                unsafe { free(handle, pointer as *mut c_void) },
                KAPSL_STATUS_OK
            );
        }
        let requests = probe.requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        for request in requests.iter() {
            assert_eq!(
                (request.device_id, request.model_id, request.replica_id),
                (3, 41, 2)
            );
        }
        assert_eq!(requests[0].scope_id, 7);
        assert_eq!(requests[0].scope_kind, KAPSL_ALLOCATION_SCOPE_MODEL);
        assert_eq!(requests[0].allocation_class, KAPSL_ALLOCATION_CLASS_WEIGHTS);
        assert_eq!(requests[1].request_ids, [71]);
        assert_eq!(requests[1].scope_id, 8);
        assert_eq!(requests[2].request_ids, [72, 73]);
        assert_eq!(requests[2].scope_kind, KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH);
        drop(requests);
        assert_eq!(
            unsafe { free(handle, weight as *mut c_void) },
            KAPSL_STATUS_OK
        );
        assert_eq!(unsafe { free(handle, weight as *mut c_void) }, ERROR);
        unsafe { close(handle) };
        assert!(broker.inner.lock().unwrap().live.is_empty());
    }

    #[test]
    fn providers_reject_foreign_devices_models_replicas_and_frees() {
        let probe = HostProbe::default();
        let client = ClientKey::new(51, 3);
        let broker = broker(&probe, 2, client);
        for (owner, device) in [
            (ClientKey::new(52, 3), 2),
            (ClientKey::new(51, 4), 2),
            (client, 1),
        ] {
            let _scope = scope(owner, device, 1, &[]);
            assert!(unsafe { open(broker.table.user_data) }.is_null());
        }
        let _model = scope(client, 2, 2, &[]);
        let first = unsafe { open(broker.table.user_data) };
        let second = unsafe { open(broker.table.user_data) };
        let pointer = unsafe { allocate(first, 64, 256) };
        assert!(!pointer.is_null());
        assert_eq!(unsafe { free(second, pointer) }, ERROR);
        assert_eq!(probe.free_attempts.load(Ordering::Relaxed), 0);
        assert_eq!(unsafe { free(first, pointer) }, KAPSL_STATUS_OK);
        unsafe {
            close(first);
            close(second);
        }
    }

    #[test]
    fn concurrent_providers_never_share_a_current_request() {
        let probe = HostProbe::default();
        let client = ClientKey::new(61, 1);
        let broker = broker(&probe, 4, client);
        let table = &broker.table;
        let model = scope(client, 4, 1, &[]);
        let first = unsafe { (table.open)(table.user_data) } as usize;
        let second = unsafe { (table.open)(table.user_data) } as usize;
        for handle in [first, second] {
            assert_eq!(unsafe { end(handle as *mut c_void) }, KAPSL_STATUS_OK);
        }
        drop(model);
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let workers = [(first, 101), (second, 102)]
            .into_iter()
            .map(|(address, request)| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let handle = address as *mut c_void;
                    let _scope = scope(client, 4, request, &[request]);
                    assert_eq!(unsafe { begin(handle) }, KAPSL_STATUS_OK);
                    barrier.wait();
                    let pointer = unsafe { allocate(handle, request, 0) };
                    assert!(!pointer.is_null());
                    barrier.wait();
                    assert_eq!(unsafe { end(handle) }, KAPSL_STATUS_OK);
                    assert_eq!(unsafe { free(handle, pointer) }, KAPSL_STATUS_OK);
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
        let requests = probe.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        for request in requests.iter() {
            assert_eq!(request.request_ids, [request.bytes]);
        }
        drop(requests);
        unsafe {
            close(first as *mut c_void);
            close(second as *mut c_void);
        }
    }

    #[test]
    fn admission_cancellation_failed_frees_and_provider_failure_keep_host_authority() {
        let probe = HostProbe::default();
        let client = ClientKey::new(71, 0);
        let broker = broker(&probe, 5, client);
        let _scope = scope(client, 5, 1, &[701]);
        let handle = unsafe { open(broker.table.user_data) };
        for (bytes, alignment) in [(0, 256), (32, 3)] {
            assert!(unsafe { allocate(handle, bytes, alignment) }.is_null());
        }
        assert!(probe.requests.lock().unwrap().is_empty());
        probe.reject_allocations.store(true, Ordering::Relaxed);
        assert!(unsafe { allocate(handle, 64, 256) }.is_null());
        probe.reject_allocations.store(false, Ordering::Relaxed);
        let pointer = unsafe { allocate(handle, 64, 256) };
        assert!(!pointer.is_null());
        // Cancellation/revoked admission is checked on every allocation, even
        // when the provider captured its scope before the request was cancelled.
        probe.reject_allocations.store(true, Ordering::Relaxed);
        assert!(unsafe { allocate(handle, 64, 256) }.is_null());
        probe.fail_synchronize.store(true, Ordering::Relaxed);
        assert_ne!(unsafe { free(handle, pointer) }, KAPSL_STATUS_OK);
        assert_eq!(probe.free_attempts.load(Ordering::Relaxed), 0);
        assert_eq!(broker.inner.lock().unwrap().live.len(), 1);
        probe.fail_synchronize.store(false, Ordering::Relaxed);
        probe.fail_free.store(true, Ordering::Relaxed);
        assert_ne!(unsafe { free(handle, pointer) }, KAPSL_STATUS_OK);
        assert_eq!(broker.inner.lock().unwrap().live.len(), 1);
        // Constructor/inference failure may skip normal End and destroy the
        // provider immediately. Cleanup retries with the original identity.
        unsafe { close(handle) };
        assert_eq!(broker.inner.lock().unwrap().live.len(), 1);
        probe.fail_free.store(false, Ordering::Relaxed);
        free_scoped(&broker.inner, pointer); // Adapter unload retry.
        assert!(broker.inner.lock().unwrap().live.is_empty());
        // A provider retained past client retirement cannot call stale hosts.
        let stale = unsafe { open(broker.table.user_data) };
        broker.inner.lock().unwrap().clients.clear();
        assert!(unsafe { allocate(stale, 64, 256) }.is_null());
        assert!(unsafe { open(broker.table.user_data) }.is_null());
        unsafe { close(stale) };
    }
}
