// NVIDIA-compatible function attribute enum (CUfunction_attribute).
//
// Values and meanings mirror NVIDIA's cuda.h (CUDA 12, cu126 pip wheel):
// kept in sync by hand, verified 2026-10-15. Only MAX_THREADS_PER_BLOCK
// through MAX_DYNAMIC_SHARED_SIZE_BYTES are implemented (see
// docs/driver-api.md); the rest are clean INVALID_VALUE. Reused by the
// driver, cudart, and tests so nobody passes magic numbers.
#ifndef ACUDA_DRIVER_ATTRS_H
#define ACUDA_DRIVER_ATTRS_H

typedef enum {
    // Max threads per block before a launch of the function would fail.
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    // Statically-allocated shared memory bytes (no dynamic part).
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    // User-allocated constant memory bytes.
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    // Local memory bytes used by each thread.
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    // Registers used by each thread.
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
    // PTX arch as major*10+minor (13 means PTX 1.3).
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    // Binary arch as major*10+minor (10 for legacy cubins).
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    // Whether compiled with -Xptxas --dlcm=ca.
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    // Max dynamic shared memory bytes usable by this function.
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    // Shared-memory carveout preference, in percent (hint only).
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    // Kernel must launch with a valid cluster size specified.
    CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10,
    // Required cluster width/height/depth in blocks.
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13,
    // Whether a non-portable cluster size may be launched (1/0).
    CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14,
    // Block scheduling policy (CUclusterSchedulingPolicy).
    CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15,

    CU_FUNC_ATTRIBUTE_MAX
} CUfunction_attribute;

#endif
