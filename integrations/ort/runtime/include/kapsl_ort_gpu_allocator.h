// SPDX-License-Identifier: MIT
#pragma once
#include "kapsl_ort_allocator.h"
#include <NvInferRuntime.h>
#include <stdexcept>

namespace kapsl_ort {

// Declared before every TensorRT object owned by an execution provider, so
// reverse member destruction destroys the allocator last, including failures.
class GpuAllocator final : public nvinfer1::IGpuAllocator {
 public:
  explicit GpuAllocator(const KapslOrtAllocatorV1* table) {
    if (!table || table->struct_size < sizeof(KapslOrtAllocatorV1) || table->version != 1 ||
        !table->user_data || !table->open || !table->close || !table->begin || !table->end ||
        !table->allocate || !table->free) {
      throw std::invalid_argument("TensorRT requires the Kapsl scoped allocator v1 table");
    }
    table_ = *table;
    handle_ = table_.open(table_.user_data);
    if (!handle_) {
      throw std::invalid_argument("TensorRT provider construction requires an active engine allocation scope");
    }
  }

  GpuAllocator(const GpuAllocator&) = delete;
  GpuAllocator& operator=(const GpuAllocator&) = delete;
  ~GpuAllocator() override { table_.close(handle_); }

  bool Begin() noexcept { return table_.begin(handle_) == 0; }
  bool End() noexcept { return table_.end(handle_) == 0; }

  void* allocate(uint64_t bytes, uint64_t alignment, nvinfer1::AllocatorFlags flags) noexcept override {
    // kRESIZABLE is a permitted hint; a later reallocate may still fail while
    // retaining the original pointer, as allowed by IGpuAllocator.
    constexpr uint32_t supported_flags = 1U << static_cast<uint32_t>(nvinfer1::AllocatorFlag::kRESIZABLE);
    if ((flags & ~supported_flags) != 0 || bytes == 0) return nullptr;
    return table_.allocate(handle_, bytes, alignment);
  }

  bool deallocate(void* pointer) noexcept override { return table_.free(handle_, pointer) == 0; }

  // TensorRT's default async implementations delegate to these synchronous
  // methods. The bridge synchronizes through the engine before every free.
  // Reallocation retains the base class's documented failure behavior: NULL,
  // leaving the original allocation intact. No private cudaMalloc fallback.

 private:
  KapslOrtAllocatorV1 table_{};
  void* handle_ = nullptr;
};

}  // namespace kapsl_ort
