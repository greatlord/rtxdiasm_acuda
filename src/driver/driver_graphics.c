#include "acuda/driver/driver_internal.h"
#include <string.h>
#include <stdlib.h>

typedef enum {
    ACUDA_GRES_NONE = 0,
    ACUDA_GRES_DEVICEPTR = 1,
    ACUDA_GRES_ARRAY = 2,
    ACUDA_GRES_MIPMAPPED_ARRAY = 3,
} acuda_gres_kind_t;

typedef struct CUgraphicsResource_st {
    acuda_gres_kind_t kind;
    unsigned int flags;
    int mapped;
    CUdeviceptr devptr;
    size_t bytes;
    CUarray array;
    CUmipmappedArray mipmapped;
};

static struct CUgraphicsResource_st *g_gfx[256];
#if defined(_WIN32)
static SRWLOCK g_gfx_lock = SRWLOCK_INIT;
#endif

static struct CUgraphicsResource_st *gfx_alloc(void) {
    struct CUgraphicsResource_st *r = (struct CUgraphicsResource_st *)calloc(1, sizeof(*r));
    return r;
}

CUresult ACUDA_CUDAAPI cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
    (void)hStream;
    if (count > 0 && !resources) return CUDA_ERROR_INVALID_VALUE;
    for (unsigned int i = 0; i < count; ++i) {
        struct CUgraphicsResource_st *r = (struct CUgraphicsResource_st *)(uintptr_t)resources[i];
        if (!r) return CUDA_ERROR_INVALID_HANDLE;
        if (r->kind == ACUDA_GRES_NONE) return CUDA_ERROR_INVALID_HANDLE;
        r->mapped = 1;
    }
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
    (void)hStream;
    if (count > 0 && !resources) return CUDA_ERROR_INVALID_VALUE;
    for (unsigned int i = 0; i < count; ++i) {
        struct CUgraphicsResource_st *r = (struct CUgraphicsResource_st *)(uintptr_t)resources[i];
        if (!r) return CUDA_ERROR_INVALID_HANDLE;
        if (r->kind == ACUDA_GRES_NONE) return CUDA_ERROR_INVALID_HANDLE;
        r->mapped = 0;
    }
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource) {
    struct CUgraphicsResource_st *r = (struct CUgraphicsResource_st *)(uintptr_t)resource;
    if (!pDevPtr || !pSize || !r) return CUDA_ERROR_INVALID_VALUE;
    if (r->kind != ACUDA_GRES_DEVICEPTR) return CUDA_ERROR_NOT_MAPPED;
    if (!r->mapped) return CUDA_ERROR_NOT_MAPPED;
    *pDevPtr = r->devptr;
    *pSize = r->bytes;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags) {
    struct CUgraphicsResource_st *r = (struct CUgraphicsResource_st *)(uintptr_t)resource;
    if (!r) return CUDA_ERROR_INVALID_VALUE;
    r->flags = flags;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) {
    struct CUgraphicsResource_st *r = (struct CUgraphicsResource_st *)(uintptr_t)resource;
    (void)arrayIndex; (void)mipLevel;
    if (!pArray || !r) return CUDA_ERROR_INVALID_VALUE;
    if (!r->mapped) return CUDA_ERROR_NOT_MAPPED;
    if (r->kind == ACUDA_GRES_ARRAY) {
        *pArray = r->array;
        return CUDA_SUCCESS;
    }
    if (r->kind == ACUDA_GRES_MIPMAPPED_ARRAY && r->mipmapped) {
        return cuMipmappedArrayGetLevel(pArray, r->mipmapped, mipLevel);
    }
    return CUDA_ERROR_NOT_MAPPED;
}

CUresult ACUDA_CUDAAPI cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource) {
    struct CUgraphicsResource_st *r = (struct CUgraphicsResource_st *)(uintptr_t)resource;
    if (!pMipmappedArray || !r) return CUDA_ERROR_INVALID_VALUE;
    if (!r->mapped) return CUDA_ERROR_NOT_MAPPED;
    if (r->kind != ACUDA_GRES_MIPMAPPED_ARRAY) return CUDA_ERROR_NOT_MAPPED;
    *pMipmappedArray = r->mipmapped;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource) {
    struct CUgraphicsResource_st *r = (struct CUgraphicsResource_st *)(uintptr_t)resource;
    if (!r) return CUDA_ERROR_INVALID_VALUE;
    r->kind = ACUDA_GRES_NONE;
    free(r);
    return CUDA_SUCCESS;
}

/* Internal helpers for ACuda/cudart registration paths */
CUresult acuda_graphics_register_deviceptr(CUgraphicsResource *resource, CUdeviceptr devptr, size_t bytes, unsigned int flags) {
    struct CUgraphicsResource_st *r;
    if (!resource) return CUDA_ERROR_INVALID_VALUE;
    r = gfx_alloc();
    if (!r) return CUDA_ERROR_OUT_OF_MEMORY;
    r->kind = ACUDA_GRES_DEVICEPTR;
    r->flags = flags;
    r->devptr = devptr;
    r->bytes = bytes;
    *resource = (CUgraphicsResource)(uintptr_t)r;
    return CUDA_SUCCESS;
}

CUresult acuda_graphics_register_array(CUgraphicsResource *resource, CUarray array, unsigned int flags) {
    struct CUgraphicsResource_st *r;
    if (!resource || !array) return CUDA_ERROR_INVALID_VALUE;
    r = gfx_alloc();
    if (!r) return CUDA_ERROR_OUT_OF_MEMORY;
    r->kind = ACUDA_GRES_ARRAY;
    r->flags = flags;
    r->array = array;
    *resource = (CUgraphicsResource)(uintptr_t)r;
    return CUDA_SUCCESS;
}

CUresult acuda_graphics_register_mipmapped_array(CUgraphicsResource *resource, CUmipmappedArray mipmapped, unsigned int flags) {
    struct CUgraphicsResource_st *r;
    if (!resource || !mipmapped) return CUDA_ERROR_INVALID_VALUE;
    r = gfx_alloc();
    if (!r) return CUDA_ERROR_OUT_OF_MEMORY;
    r->kind = ACUDA_GRES_MIPMAPPED_ARRAY;
    r->flags = flags;
    r->mipmapped = mipmapped;
    *resource = (CUgraphicsResource)(uintptr_t)r;
    return CUDA_SUCCESS;
}
