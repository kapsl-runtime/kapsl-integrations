// SPDX-License-Identifier: MIT
// Private interface between the ORT adapter and its pinned provider build.
// The engine continues to implement kapsl-backend-abi 0.2.0 unchanged.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KapslOrtAllocatorV1 {
  uint32_t struct_size;
  uint32_t version;
  void* user_data;
  // open captures the caller's active model/request scope into a new provider
  // handle. No scope or an inactive client returns NULL. The broker must live
  // until open returns; the handle owns its state thereafter.
  void* (*open)(void* user_data);
  // close consumes a handle once, after all provider objects and calls drain.
  // Failed reclamation remains in the adapter's governed allocation ledger.
  void (*close)(void* handle);
  // A successful begin captures the current calling thread's request/batch.
  // Only one scope may be active per handle. Separate handles can run together.
  int32_t (*begin)(void* handle);
  int32_t (*end)(void* handle);
  // Allocations on worker threads use the handle's captured scope. The host
  // remains authoritative for admission, cancellation and active ownership.
  void* (*allocate)(void* handle, uint64_t bytes, uint64_t alignment);
  // Null free succeeds. Other frees validate exact provider/client ownership
  // and synchronize before releasing memory. Failure preserves the identity.
  int32_t (*free)(void* handle, void* pointer);
} KapslOrtAllocatorV1;

#ifdef __cplusplus
}
#endif
