#include "driver_internal.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

CUresult driver_memcpy_common(size_t src_xb, size_t src_y, size_t src_z,
                                     CUmemorytype src_type, const void *src_host,
                                     CUdeviceptr src_dev, CUarray src_array,
                                     size_t src_pitch, size_t src_height,
                                     size_t dst_xb, size_t dst_y, size_t dst_z,
                                     CUmemorytype dst_type, void *dst_host,
                                     CUdeviceptr dst_dev, CUarray dst_array,
                                     size_t dst_pitch, size_t dst_height,
                                     size_t wbytes, size_t h, size_t depth) {
    // Host-to-array.
    if (src_type == CU_MEMORYTYPE_HOST && dst_type == CU_MEMORYTYPE_ARRAY) {
        unsigned char *hp = (unsigned char *)src_host +
                            src_y * src_pitch + src_xb;
        if (depth > 1)
            hp += src_z * src_pitch * (src_height ? src_height : h);
        return driver_array_rw(dst_array, 1, dst_xb, dst_y, dst_z, hp,
                               wbytes, h, depth, src_pitch);
    }
    // Array-to-host.
    if (src_type == CU_MEMORYTYPE_ARRAY && dst_type == CU_MEMORYTYPE_HOST) {
        unsigned char *hp = (unsigned char *)dst_host +
                            dst_y * dst_pitch + dst_xb;
        if (depth > 1)
            hp += dst_z * dst_pitch * (dst_height ? dst_height : h);
        return driver_array_rw(src_array, 0, src_xb, src_y, src_z, hp,
                               wbytes, h, depth, dst_pitch);
    }
    // Device-to-array / array-to-device via the host-visible arena.
    if (src_type == CU_MEMORYTYPE_DEVICE && dst_type == CU_MEMORYTYPE_ARRAY) {
        void *base = NULL;
        if (host_for(src_dev, src_pitch * h * depth, &base) != CUDA_SUCCESS)
            return CUDA_ERROR_INVALID_VALUE;
        unsigned char *hp = (unsigned char *)base +
                            src_y * src_pitch + src_xb;
        if (depth > 1)
            hp += src_z * src_pitch * (src_height ? src_height : h);
        return driver_array_rw(dst_array, 1, dst_xb, dst_y, dst_z, hp,
                               wbytes, h, depth, src_pitch);
    }
    if (src_type == CU_MEMORYTYPE_ARRAY && dst_type == CU_MEMORYTYPE_DEVICE) {
        void *base = NULL;
        if (host_for(dst_dev, dst_pitch * h * depth, &base) != CUDA_SUCCESS)
            return CUDA_ERROR_INVALID_VALUE;
        unsigned char *hp = (unsigned char *)base +
                            dst_y * dst_pitch + dst_xb;
        if (depth > 1)
            hp += dst_z * dst_pitch * (dst_height ? dst_height : h);
        return driver_array_rw(src_array, 0, src_xb, src_y, src_z, hp,
                               wbytes, h, depth, dst_pitch);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult ACUDA_CUDAAPI cuMemcpy2D(const void *pCopy) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    const CUDA_MEMCPY2D *m = pCopy;
    if (!m)
        return CUDA_ERROR_INVALID_VALUE;
    return driver_memcpy_common(
        m->srcXInBytes, m->srcY, 0, m->srcMemoryType, m->srcHost,
        m->srcDevice, m->srcArray, m->srcPitch, 0, m->dstXInBytes, m->dstY,
        0, m->dstMemoryType, m->dstHost, m->dstDevice, m->dstArray,
        m->dstPitch, 0, m->WidthInBytes, m->Height, 1);
}

CUresult ACUDA_CUDAAPI cuMemcpy2DAsync(const void *pCopy, CUstream hStream) {
    (void)hStream;
    return cuMemcpy2D(pCopy);
}

CUresult ACUDA_CUDAAPI cuMemcpy3D(const void *pCopy) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    const CUDA_MEMCPY3D *m = pCopy;
    if (!m)
        return CUDA_ERROR_INVALID_VALUE;
    return driver_memcpy_common(
        m->srcXInBytes, m->srcY, m->srcZ, m->srcMemoryType, m->srcHost,
        m->srcDevice, m->srcArray, m->srcPitch, m->srcHeight,
        m->dstXInBytes, m->dstY, m->dstZ, m->dstMemoryType, m->dstHost,
        m->dstDevice, m->dstArray, m->dstPitch, m->dstHeight,
        m->WidthInBytes, m->Height, m->Depth);
}

CUresult ACUDA_CUDAAPI cuMemcpy3DAsync(const void *pCopy, CUstream hStream) {
    (void)hStream;
    return cuMemcpy3D(pCopy);
}

/* CUDA 6.x-era compatibility: entries the 32-bit CUDA 6.5 runtime binds by
 * name at init through the module handle in export-table slot 5.  Every
 * probed name must resolve or the call dispatches through NULL (EIP=0
 * execute fault).  Real behavior where meaningful, success no-ops for
 * inapplicable concepts (single device, no interop), NOT_SUPPORTED for
 * unmodelled APIs.  All prototypes are arity-exact: the driver is
 * __stdcall on 32-bit Windows, so the callee must clean exactly the
 * pushed bytes. */

/* 2D/3D copy variants the runtime probes for. */
CUresult ACUDA_CUDAAPI cuMemcpy2D_v2(const void *pCopy) {
    return cuMemcpy2D(pCopy);
}

CUresult ACUDA_CUDAAPI cuMemcpy2DAsync_v2(const void *pCopy,
                                          CUstream hStream) {
    return cuMemcpy2DAsync(pCopy, hStream);
}

CUresult ACUDA_CUDAAPI cuMemcpy2DUnaligned(const void *pCopy) {
    return cuMemcpy2D(pCopy);
}

CUresult ACUDA_CUDAAPI cuMemcpy2DUnaligned_v2(const void *pCopy) {
    return cuMemcpy2D(pCopy);
}

CUresult ACUDA_CUDAAPI cuMemcpy3D_v2(const void *pCopy) {
    return cuMemcpy3D(pCopy);
}

CUresult ACUDA_CUDAAPI cuMemcpy3DAsync_v2(const void *pCopy,
                                          CUstream hStream) {
    return cuMemcpy3DAsync(pCopy, hStream);
}

/* Single-device peer copies are plain device-to-device copies. */
CUresult ACUDA_CUDAAPI cuMemcpyPeer(CUdeviceptr dst, CUcontext dstCtx,
                                    CUdeviceptr src, CUcontext srcCtx,
                                    size_t bytes) {
    (void)dstCtx;
    (void)srcCtx;
    return memcpy_d2d(dst, src, bytes);
}

CUresult ACUDA_CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dst, CUcontext dstCtx,
                                         CUdeviceptr src, CUcontext srcCtx,
                                         size_t bytes, CUstream hStream) {
    (void)hStream;
    return cuMemcpyPeer(dst, dstCtx, src, srcCtx, bytes);
}



 CUresult memcpy_h2d(CUdeviceptr dst, const void *src, size_t bytes) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemcpyHtoD(dst=%llx bytes=%zu)\n",
                (unsigned long long)dst, bytes);
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (!src)
        return CUDA_ERROR_INVALID_VALUE;
    if (bytes == 0)
        return CUDA_SUCCESS;
    void *host = NULL;
    r = host_for(dst, bytes, &host);
    if (r != CUDA_SUCCESS)
        return r;
    // Host-visible coherent: plain copy, ordered after default-stream work.
    r = map_status(acuda_stream_synchronize(g_actx, NULL));
    if (r != CUDA_SUCCESS)
        return r;
    memcpy(host, src, bytes);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t bytes) {
    return memcpy_h2d(dst, src, bytes);
}

CUresult ACUDA_CUDAAPI cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t bytes) {
    return memcpy_h2d(dst, src, bytes);
}

 CUresult memcpy_d2h(void *dst, CUdeviceptr src, size_t bytes) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemcpyDtoH(dst=%p src=%llx bytes=%zu)\n",
                dst, (unsigned long long)src, bytes);
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (!dst)
        return CUDA_ERROR_INVALID_VALUE;
    if (bytes == 0)
        return CUDA_SUCCESS;
    void *host = NULL;
    r = host_for(src, bytes, &host);
    if (r != CUDA_SUCCESS)
        return r;
    r = map_status(acuda_stream_synchronize(g_actx, NULL));
    if (r != CUDA_SUCCESS)
        return r;
    memcpy(dst, host, bytes);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t bytes) {
    return memcpy_d2h(dst, src, bytes);
}

CUresult ACUDA_CUDAAPI cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t bytes) {
    return memcpy_d2h(dst, src, bytes);
}

CUresult memcpy_d2d(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemcpyDtoD(dst=%llx src=%llx bytes=%zu)\n",
                (unsigned long long)dst, (unsigned long long)src, bytes);
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (bytes == 0)
        return CUDA_SUCCESS;
    void *hd = NULL, *hs = NULL;
    r = host_for(dst, bytes, &hd);
    if (r != CUDA_SUCCESS)
        return r;
    r = host_for(src, bytes, &hs);
    if (r != CUDA_SUCCESS)
        return r;
    r = map_status(acuda_stream_synchronize(g_actx, NULL));
    if (r != CUDA_SUCCESS)
        return r;
    memmove(hd, hs, bytes);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    return memcpy_d2d(dst, src, bytes);
}

CUresult ACUDA_CUDAAPI cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    return memcpy_d2d(dst, src, bytes);
}

// Generic copy used by CUDA 13 cudart for the scalar `.item()` path.
// Dispatch on which side is arena-backed; host/host stays a plain copy.
static CUresult memcpy_generic(CUdeviceptr dst, CUdeviceptr src,
                               size_t bytes) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (bytes == 0)
        return CUDA_SUCCESS;
    if (!dst || !src)
        return CUDA_ERROR_INVALID_VALUE;
    size_t cap = 0;
    int ddev = acuda_arena_capacity(dst, &cap) == 0;
    int sdev = acuda_arena_capacity(src, &cap) == 0;
    if (ddev && sdev)
        return memcpy_d2d(dst, src, bytes);
    if (ddev)
        return memcpy_h2d(dst, (const void *)(uintptr_t)src, bytes);
    if (sdev)
        return memcpy_d2h((void *)(uintptr_t)dst, src, bytes);
    memcpy((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, bytes);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    return memcpy_generic(dst, src, bytes);
}

CUresult ACUDA_CUDAAPI cuMemcpy_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    return memcpy_generic(dst, src, bytes);
}

// CUDA 13.4 cudart probes the attributed-copy entry point for scalar
// readback.  The attributes are placement hints; the copy itself is
// the same.
CUresult ACUDA_CUDAAPI cuMemcpyWithAttributesAsync(CUdeviceptr dst, CUdeviceptr src,
                                     size_t bytes, void *attr,
                                     void *stream) {
    (void)attr;
    (void)stream;
    return memcpy_generic(dst, src, bytes);
}

CUresult ACUDA_CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src,
                              size_t bytes, void *stream) {
    (void)stream;
    return cuMemcpyHtoD_v2(dst, src, bytes);
}
CUresult ACUDA_CUDAAPI cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src,
                              size_t bytes, void *stream) {
    (void)stream;
    return cuMemcpyDtoH_v2(dst, src, bytes);
}
CUresult ACUDA_CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src,
                              size_t bytes, void *stream) {
    (void)stream;
    return cuMemcpyDtoD_v2(dst, src, bytes);
}

CUresult ACUDA_CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src,
                           size_t bytes, void *stream) {
    return cuMemcpyHtoDAsync_v2(dst, src, bytes, stream);
}
CUresult ACUDA_CUDAAPI cuMemcpyDtoHAsync(void *dst, CUdeviceptr src, size_t bytes,
                           void *stream) {
    return cuMemcpyDtoHAsync_v2(dst, src, bytes, stream);
}
CUresult ACUDA_CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dst, CUdeviceptr src,
                           size_t bytes, void *stream) {
    return cuMemcpyDtoDAsync_v2(dst, src, bytes, stream);
}
CUresult ACUDA_CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                       size_t bytes, int kind, void *stream) {
    (void)stream;
    switch (kind) {
    case 1: /* HtoD */
        return cuMemcpyHtoD_v2(dst, (const void *)(uintptr_t)src, bytes);
    case 2: /* DtoH */
        return cuMemcpyDtoH_v2((void *)(uintptr_t)dst, src, bytes);
    case 3: /* DtoD */
        return cuMemcpyDtoD_v2(dst, src, bytes);
    default:
        if (kind == 0) { /* HtoH */
            memcpy((void *)(uintptr_t)dst, (const void *)(uintptr_t)src,
                   bytes);
            return CUDA_SUCCESS;
        }
        return CUDA_ERROR_INVALID_VALUE;
    }
}
