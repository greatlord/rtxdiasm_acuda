#include "driver_internal.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

CUresult acuda_stub_cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                                      const char *name) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuModuleGetGlobal(%s)\n",
                name ? name : "(null)");
    // `bytes` may be NULL when caller does not request the size.
    if (!dptr || !hmod || !name) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? cuModuleGetGlobal(%s) -> invalid dptr=%p bytes=%p hmod=%p name=%p\n",
                    name ? name : "(null)", (void *)dptr, (void *)bytes, (void *)hmod, (void *)name);
        return CUDA_ERROR_INVALID_VALUE;
    }
    *dptr = 0;
    if (bytes) *bytes = 0;
    for (size_t i = 0; i < hmod->nglobals; i++) {
        if (strcmp(hmod->globals[i].name, name) == 0) {
            *dptr = hmod->globals[i].dptr;
            if (bytes) *bytes = hmod->globals[i].bytes;
            ACUDA_DEBUG_PRINT("acuda-trace ?? cuModuleGetGlobal(%s) -> cached 0x%llx %zu\n",
                        name, (unsigned long long)*dptr, hmod->globals[i].bytes);
            return CUDA_SUCCESS;
        }
    }
    // Legacy PhysX kernels keep their parameter block in a 10 KiB
    // __constant__ symbol.  Other names keep a generous default.
    size_t gsize = strcmp(name, "parameters_cmem") == 0 ? 10240 : 65536;
    CUdeviceptr p = 0;
    CUresult r = mem_alloc(&p, gsize);
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuModuleGetGlobal(%s) alloc %zu -> r=%d p=0x%llx\n",
                name, gsize, (int)r, (unsigned long long)p);
    if (r != CUDA_SUCCESS)
        return r;
    if (hmod->nglobals == hmod->globals_cap) {
        size_t cap = hmod->globals_cap ? hmod->globals_cap * 2 : 4;
        void *ng = realloc(hmod->globals, cap * sizeof(*hmod->globals));
        if (!ng) {
            cuMemFree_v2(p);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        hmod->globals = ng;
        hmod->globals_cap = cap;
    }
    snprintf(hmod->globals[hmod->nglobals].name,
             sizeof(hmod->globals[hmod->nglobals].name), "%s", name);
    hmod->globals[hmod->nglobals].dptr = p;
    hmod->globals[hmod->nglobals].bytes = gsize;
    hmod->nglobals++;
    *dptr = p;
    if (bytes) *bytes = gsize;
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuModuleGetGlobal(%s) -> 0x%llx %zu\n",
                name, (unsigned long long)*dptr, gsize);
    return CUDA_SUCCESS;
}

CUresult acuda_stub_cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                                         const char *name) {
    return acuda_stub_cuModuleGetGlobal(dptr, bytes, hmod, name);
}

CUresult acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                                CUfunction func,
                                                                int blockSize,
                                                                size_t dynamicSMemSize) {
    (void)func; (void)blockSize; (void)dynamicSMemSize;
    if (!numBlocks) return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS) return CUDA_ERROR_NOT_INITIALIZED;
    *numBlocks = 1;
    return CUDA_SUCCESS;
}

CUresult acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
    (void)flags;
    return acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func,
                                                                  blockSize,
                                                                  dynamicSMemSize);
}

CUresult acuda_stub_cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *maxBlockSize,
                                                     CUfunction func, size_t blockSizeLimit,
                                                     size_t dynamicSMemSize) {
    (void)func; (void)blockSizeLimit; (void)dynamicSMemSize;
    if (need_init() != CUDA_SUCCESS) return CUDA_ERROR_NOT_INITIALIZED;
    if (minGridSize) *minGridSize = 1;
    if (maxBlockSize) *maxBlockSize = 256;
    return CUDA_SUCCESS;
}

CUresult acuda_stub_cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *maxBlockSize, CUfunction func,
    size_t blockSizeLimit, size_t dynamicSMemSize, unsigned int flags) {
    (void)flags;
    return acuda_stub_cuOccupancyMaxPotentialBlockSize(minGridSize, maxBlockSize, func,
                                                       blockSizeLimit, dynamicSMemSize);
}

CUresult acuda_stub_cuTensorMapEncodeTiled(void *tensorMap, int tensorDataType,
                                           unsigned int tensorRank, void *globalAddress,
                                           const unsigned long long *globalDim, const unsigned long long *globalStrides,
                                           const unsigned int *boxDim, const unsigned int *elementStrides,
                                           int interleave, int swizzle,
                                           int l2Promotion, int oobFill) {
    (void)tensorMap;(void)tensorDataType;(void)tensorRank;(void)globalAddress;(void)globalDim;(void)globalStrides;(void)boxDim;(void)elementStrides;(void)interleave;(void)swizzle;(void)l2Promotion;(void)oobFill;
    if (need_init() != CUDA_SUCCESS) return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_ERROR_NOT_SUPPORTED;
}


/* Exported aliases for clients that bind these names directly. */
CUresult ACUDA_CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                           const char *name) {
    return acuda_stub_cuModuleGetGlobal(dptr, bytes, hmod, name);
}
CUresult ACUDA_CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                              const char *name) {
    return acuda_stub_cuModuleGetGlobal(dptr, bytes, hmod, name);
}
CUresult ACUDA_CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
    return acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks, func, blockSize, dynamicSMemSize);
}
CUresult ACUDA_CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
    return acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks, func, blockSize, dynamicSMemSize, flags);
}
CUresult ACUDA_CUDAAPI cuOccupancyMaxPotentialBlockSize(
    int *minGridSize, int *maxBlockSize, CUfunction func,
    size_t blockSizeLimit, size_t dynamicSMemSize) {
    return acuda_stub_cuOccupancyMaxPotentialBlockSize(
        minGridSize, maxBlockSize, func, blockSizeLimit, dynamicSMemSize);
}
CUresult ACUDA_CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *maxBlockSize, CUfunction func,
    size_t blockSizeLimit, size_t dynamicSMemSize, unsigned int flags) {
    return acuda_stub_cuOccupancyMaxPotentialBlockSizeWithFlags(
        minGridSize, maxBlockSize, func, blockSizeLimit, dynamicSMemSize,
        flags);
}
CUresult ACUDA_CUDAAPI cuTensorMapEncodeTiled(
    void *tensorMap, int tensorDataType, unsigned int tensorRank,
    void *globalAddress, const unsigned long long *globalDim,
    const unsigned long long *globalStrides, const unsigned int *boxDim,
    const unsigned int *elementStrides, int interleave, int swizzle,
    int l2Promotion, int oobFill) {
    return acuda_stub_cuTensorMapEncodeTiled(
        tensorMap, tensorDataType, tensorRank, globalAddress, globalDim,
        globalStrides, boxDim, elementStrides, interleave, swizzle,
        l2Promotion, oobFill);
}
