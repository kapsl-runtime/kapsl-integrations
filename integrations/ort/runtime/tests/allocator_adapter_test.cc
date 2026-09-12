// Host-only fixture for the actual patched ORT allocator-adapter methods.
// test_host.py extracts those definitions from authenticated prepared source.
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

#define ORT_ENFORCE(condition, ...)                 \
  do {                                             \
    if (!(condition)) throw std::runtime_error("allocation rejected"); \
  } while (false)

struct OrtSyncStream {};
struct Stream : OrtSyncStream {};
struct OrtAllocator {
  uint32_t version;
  void* (*Alloc)(OrtAllocator*, size_t);
  void* (*Reserve)(OrtAllocator*, size_t);
  void* (*AllocOnStream)(OrtAllocator*, size_t, OrtSyncStream*);
};

class IAllocatorImplWrappingOrtAllocator {
 public:
  explicit IAllocatorImplWrappingOrtAllocator(OrtAllocator* allocator)
      : ort_allocator_(allocator, [](OrtAllocator*) {}) {}
  void* Alloc(size_t size);
  bool IsStreamAware() const;
  void* AllocOnStream(size_t size, Stream* stream);
  void* Reserve(size_t size);

 private:
  std::unique_ptr<OrtAllocator, std::function<void(OrtAllocator*)>> ort_allocator_;
};

// Includes the pinned version guards and actual production method definitions.
#include "allocator_adapter_methods.inc"

static int allocation_calls = 0;
static int reserve_calls = 0;
static int stream_calls = 0;
static bool reject = false;
static char storage;
static void* result(size_t size) { return reject || size == 0 ? nullptr : &storage; }
static void* allocate(OrtAllocator*, size_t size) {
  ++allocation_calls;
  return result(size);
}
static void* reserve(OrtAllocator*, size_t size) {
  ++reserve_calls;
  return result(size);
}
static void* on_stream(OrtAllocator*, size_t size, OrtSyncStream*) {
  ++stream_calls;
  return result(size);
}

template <typename Callback>
static void must_reject(Callback callback) {
  bool rejected = false;
  try {
    (void)callback();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
}

int main() {
  Stream stream;
  for (uint32_t version : {kOrtAllocatorReserveMinVersion - 1,
                           kOrtAllocatorReserveMinVersion,
                           kOrtAllocatorAllocOnStreamMinVersion}) {
    for (bool optional_callbacks : {false, true}) {
      OrtAllocator api{version, allocate, optional_callbacks ? reserve : nullptr,
                       optional_callbacks ? on_stream : nullptr};
      IAllocatorImplWrappingOrtAllocator wrapper(&api);
      const bool has_reserve = optional_callbacks && version >= kOrtAllocatorReserveMinVersion;
      const bool has_stream = optional_callbacks && version >= kOrtAllocatorAllocOnStreamMinVersion;
      assert(wrapper.IsStreamAware() == has_stream);
      allocation_calls = reserve_calls = stream_calls = 0;
      reject = false;
      assert(wrapper.Alloc(16) == &storage);
      assert(wrapper.Reserve(16) == &storage);
      assert(wrapper.AllocOnStream(16, &stream) == &storage);
      assert(allocation_calls == 1 + !has_reserve + !has_stream);
      assert(reserve_calls == has_reserve && stream_calls == has_stream);
      // ORT permits a null result for an empty tensor/allocation.
      assert(wrapper.Alloc(0) == nullptr);
      assert(wrapper.Reserve(0) == nullptr);
      assert(wrapper.AllocOnStream(0, &stream) == nullptr);
      // Cancellation/admission rejection must stop execution before any kernel
      // can consume a null device pointer, including legacy callback fallbacks.
      reject = true;
      must_reject([&] { return wrapper.Alloc(16); });
      must_reject([&] { return wrapper.Reserve(16); });
      must_reject([&] { return wrapper.AllocOnStream(16, &stream); });
    }
  }
}
