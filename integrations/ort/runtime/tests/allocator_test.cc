// SPDX-License-Identifier: MIT
#include "kapsl_ort_gpu_allocator.h"
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <unordered_set>

struct Probe {
  bool active = true;
  bool fail_free = false;
  int opens = 0;
  int closes = 0;
  int allocations = 0;
  std::unordered_set<void*> live;
};

static KapslOrtAllocatorV1 Table(Probe& probe) {
  return {
      sizeof(KapslOrtAllocatorV1), 1, &probe,
      [](void* data) -> void* {
        auto& p = *static_cast<Probe*>(data);
        if (!p.active) return nullptr;
        ++p.opens;
        return data;
      },
      [](void* data) {
        auto& p = *static_cast<Probe*>(data);
        ++p.closes;
        for (void* pointer : p.live) std::free(pointer);
        p.live.clear();
      },
      [](void* data) -> int32_t {
        auto& p = *static_cast<Probe*>(data);
        if (p.active) return -1;
        p.active = true;
        return 0;
      },
      [](void* data) -> int32_t {
        auto& p = *static_cast<Probe*>(data);
        if (!p.active) return -1;
        p.active = false;
        return 0;
      },
      [](void* data, uint64_t bytes, uint64_t alignment) -> void* {
        auto& p = *static_cast<Probe*>(data);
        if (!p.active || !bytes || (alignment && (alignment & (alignment - 1)))) return nullptr;
        ++p.allocations;
        auto* pointer = std::malloc(bytes);
        p.live.insert(pointer);
        return pointer;
      },
      [](void* data, void* pointer) -> int32_t {
        auto& p = *static_cast<Probe*>(data);
        if (!pointer) return 0;
        if (p.fail_free || !p.live.erase(pointer)) return -1;
        std::free(pointer);
        return 0;
      }};
}

template <typename Mutate>
static void Reject(Mutate mutate) {
  Probe probe;
  auto table = Table(probe);
  mutate(table);
  bool rejected = false;
  try { kapsl_ort::GpuAllocator allocator(&table); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected && probe.opens == 0 && probe.closes == 0);
}

int main() {
  static_assert(sizeof(KapslOrtAllocatorV1) == 8 + 7 * sizeof(void*));
  static_assert(offsetof(KapslOrtAllocatorV1, user_data) == 8);
  Reject([](auto& t) { --t.struct_size; });
  Reject([](auto& t) { ++t.version; });
  Reject([](auto& t) { t.user_data = nullptr; });
  Reject([](auto& t) { t.open = nullptr; });
  Reject([](auto& t) { t.close = nullptr; });
  Reject([](auto& t) { t.begin = nullptr; });
  Reject([](auto& t) { t.end = nullptr; });
  Reject([](auto& t) { t.allocate = nullptr; });
  Reject([](auto& t) { t.free = nullptr; });
  Probe probe;
  auto table = Table(probe);
  {
    kapsl_ort::GpuAllocator allocator(&table);
    assert(probe.opens == 1);
    // Exercise TensorRT 10.9's actual virtual interface/default async methods.
    nvinfer1::IGpuAllocator& trt = allocator;
    assert(!trt.allocate(0, 256, 0));
    assert(!trt.allocate(8, 3, 0));
    assert(!trt.allocate(8, 256, 2));
    assert(probe.allocations == 0);
    void* weight = trt.allocate(64, 256, 1);
    assert(weight && probe.live.size() == 1);
    assert(!trt.reallocate(weight, 256, 128));
    assert(probe.live.count(weight));
    assert(allocator.End());
    assert(!trt.allocateAsync(64, 256, 0, nullptr));
    assert(allocator.Begin());
    assert(!allocator.Begin());
    void* request = trt.allocateAsync(128, 256, 0, nullptr);
    assert(request);
    probe.fail_free = true;
    assert(!trt.deallocateAsync(request, nullptr));
    assert(probe.live.count(request));
    probe.fail_free = false;
    assert(trt.deallocateAsync(request, nullptr));
    assert(!trt.deallocateAsync(request, nullptr));
    assert(allocator.End());
    assert(trt.deallocate(weight));
    assert(trt.deallocate(nullptr));
  }
  assert(probe.closes == 1 && probe.live.empty());
  // Exception during provider construction destroys the allocator after other
  // members unwind and reclaims an allocation still owned by the failed build.
  probe.active = true;
  try {
    kapsl_ort::GpuAllocator allocator(&table);
    assert(allocator.allocate(64, 0, 0));
    throw std::runtime_error("simulated provider construction failure");
  } catch (const std::runtime_error&) {}
  assert(probe.opens == 2 && probe.closes == 2 && probe.live.empty());
}
