// SPDX-License-Identifier: MIT
// Host test only: opaque CUDA handles used in TensorRT's public declarations.
// No CUDA functions, allocator implementations or emulated kernels are supplied.
#pragma once
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
