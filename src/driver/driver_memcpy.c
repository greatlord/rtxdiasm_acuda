#include "acuda/driver/driver_internal.h"
#include "acuda/runtime_api.h"
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
    /* Array-to-array: stage through host buffer. */
    if (src_type == CU_MEMORYTYPE_ARRAY && dst_type == CU_MEMORYTYPE_ARRAY) {
        if (wbytes == 0 || h == 0 || depth == 0) return CUDA_SUCCESS;
        size_t buf = wbytes * h * depth;
        void *tmp = malloc(buf ? buf : 1);
        if (!tmp) return CUDA_ERROR_OUT_OF_MEMORY;
        CUresult r = driver_memcpy_common(src_xb, src_y, src_z, src_type, src_host, src_dev, src_array,
                                          src_pitch, src_height, 0, 0, 0, CU_MEMORYTYPE_HOST, tmp, 0, NULL,
                                          wbytes, 0, wbytes, h, depth);
        if (r == CUDA_SUCCESS)
            r = driver_memcpy_common(0, 0, 0, CU_MEMORYTYPE_HOST, tmp, 0, NULL, wbytes, 0,
                                     dst_xb, dst_y, dst_z, dst_type, dst_host, dst_dev, dst_array,
                                     dst_pitch, dst_height, wbytes, h, depth);
        free(tmp);
        return r;
    }
    /* 2D/3D host<->device / device<->device / host<->host (non-array). */
    if (src_type != CU_MEMORYTYPE_ARRAY && dst_type != CU_MEMORYTYPE_ARRAY) {
        if (wbytes == 0 || h == 0 || depth == 0)
            return CUDA_SUCCESS;
        if (src_pitch == 0) src_pitch = wbytes;
        if (dst_pitch == 0) dst_pitch = wbytes;
        size_t src_plane = src_pitch * (src_height ? src_height : h);
        size_t dst_plane = dst_pitch * (dst_height ? dst_height : h);
        unsigned char *sp = NULL, *dp = NULL;
        CUresult r = CUDA_SUCCESS;
        if (src_type == CU_MEMORYTYPE_DEVICE) {
            void *base = NULL;
            r = host_for(src_dev, src_z * src_plane + src_y * src_pitch + src_xb + (depth - 1) * src_plane + (h - 1) * src_pitch + wbytes, &base);
            if (r != CUDA_SUCCESS) return r;
            sp = (unsigned char *)base;
        } else {
            if (!src_host && (wbytes * h * depth)) return CUDA_ERROR_INVALID_VALUE;
            sp = (unsigned char *)src_host;
        }
        if (dst_type == CU_MEMORYTYPE_DEVICE) {
            void *base = NULL;
            r = host_for(dst_dev, dst_z * dst_plane + dst_y * dst_pitch + dst_xb + (depth - 1) * dst_plane + (h - 1) * dst_pitch + wbytes, &base);
            if (r != CUDA_SUCCESS) return r;
            dp = (unsigned char *)base;
        } else {
            if (!dst_host && (wbytes * h * depth)) return CUDA_ERROR_INVALID_VALUE;
            dp = (unsigned char *)dst_host;
        }
        r = map_status(acuda_stream_synchronize(g_actx, NULL));
        if (r != CUDA_SUCCESS) return r;
        for (size_t z = 0; z < depth; z++) {
            const unsigned char *splane = sp + (src_z + z) * src_plane;
            unsigned char *dplane = dp + (dst_z + z) * dst_plane;
            for (size_t y = 0; y < h; y++) {
                const unsigned char *srow = splane + (src_y + y) * src_pitch + src_xb;
                unsigned char *drow = dplane + (dst_y + y) * dst_pitch + dst_xb;
                memmove(drow, srow, wbytes);
            }
        }
        return CUDA_SUCCESS;
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
    { static FILE *_fp; static int _i; static long long _n; if(!_i){_i=1; const char*_p=getenv("ACUDA_DRVTRACE"); if(_p&&*_p)_fp=fopen(_p,"a");} if(_fp){fprintf(_fp,"%lld memcpy_h2d dst=%llx sz=%zu\n",_n++,(unsigned long long)dst,(size_t)bytes);fflush(_fp);} }
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemcpyHtoD(dst=%llx bytes=%zu)\n",
                (unsigned long long)dst, bytes);
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        { { static FILE *_fp; static int _i; static long long _n; if(!_i){_i=1; const char*_p=getenv("ACUDA_DRVTRACE"); if(_p&&*_p)_fp=fopen(_p,"a");} if(_fp){fprintf(_fp,"%lld memcpy_h2d_ret r=%d\n",_n++,(int)r);fflush(_fp);} } return r; }
    if (!src)
        { { static FILE *_fp; static int _i; static long long _n; if(!_i){_i=1; const char*_p=getenv("ACUDA_DRVTRACE"); if(_p&&*_p)_fp=fopen(_p,"a");} if(_fp){fprintf(_fp,"%lld memcpy_h2d_ret r=%d\n",_n++,(int)CUDA_ERROR_INVALID_VALUE);fflush(_fp);} } return CUDA_ERROR_INVALID_VALUE; }
    if (bytes == 0)
        { { static FILE *_fp; static int _i; static long long _n; if(!_i){_i=1; const char*_p=getenv("ACUDA_DRVTRACE"); if(_p&&*_p)_fp=fopen(_p,"a");} if(_fp){fprintf(_fp,"%lld memcpy_h2d_ret r=0\n",_n++);fflush(_fp);} } return CUDA_SUCCESS; }
    void *host = NULL;
    r = host_for(dst, bytes, &host);
    if (r != CUDA_SUCCESS)
        { { static FILE *_fp; static int _i; static long long _n; if(!_i){_i=1; const char*_p=getenv("ACUDA_DRVTRACE"); if(_p&&*_p)_fp=fopen(_p,"a");} if(_fp){fprintf(_fp,"%lld memcpy_h2d_ret r=%d\n",_n++,(int)r);fflush(_fp);} } return r; }
    // Host-visible coherent: plain copy, ordered after default-stream work.
    r = map_status(acuda_stream_synchronize(g_actx, NULL));
    if (r != CUDA_SUCCESS)
        { { static FILE *_fp; static int _i; static long long _n; if(!_i){_i=1; const char*_p=getenv("ACUDA_DRVTRACE"); if(_p&&*_p)_fp=fopen(_p,"a");} if(_fp){fprintf(_fp,"%lld memcpy_h2d_ret r=%d\n",_n++,(int)r);fflush(_fp);} } return r; }
    memcpy(host, src, bytes);
    { static FILE *_fp; static int _i; static long long _n; if(!_i){_i=1; const char*_p=getenv("ACUDA_DRVTRACE"); if(_p&&*_p)_fp=fopen(_p,"a");} if(_fp){fprintf(_fp,"%lld memcpy_h2d_ret r=0\n",_n++);fflush(_fp);} }
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t bytes) {
    return memcpy_h2d(dst, src, bytes);
}

CUresult ACUDA_CUDAAPI cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t bytes) {
    return memcpy_h2d(dst, src, bytes);
}

 CUresult memcpy_d2h(void *dst, CUdeviceptr src, size_t bytes) {
    { static FILE *_fp; static int _i; static long long _n; if(!_i){_i=1; const char*_p=getenv("ACUDA_DRVTRACE"); if(_p&&*_p)_fp=fopen(_p,"a");} if(_fp){fprintf(_fp,"%lld memcpy_d2h src=%llx sz=%zu\n",_n++,(unsigned long long)src,(size_t)bytes);fflush(_fp);} }
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemcpyDtoH(dst=%p src=%llx bytes=%zu)\n",
                dst, (unsigned long long)src, bytes);
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        { { static FILE *_fp; static int _i; static long long _n; if(!_i){_i=1; const char*_p=getenv("ACUDA_DRVTRACE"); if(_p&&*_p)_fp=fopen(_p,"a");} if(_fp){fprintf(_fp,"%lld memcpy_d2h_ret r=%d\n",_n++,(int)r);fflush(_fp);} } return r; }
    if (!dst)
        { { static FILE *_fp; static int _i; static long long _n; if(!_i){_i=1; const char*_p=getenv("ACUDA_DRVTRACE"); if(_p&&*_p)_fp=fopen(_p,"a");} if(_fp){fprintf(_fp,"%lld memcpy_d2h_ret r=%d\n",_n++,(int)CUDA_ERROR_INVALID_VALUE);fflush(_fp);} } return CUDA_ERROR_INVALID_VALUE; }
    if (bytes == 0)
        { { static FILE *_fp; static int _i; static long long _n; if(!_i){_i=1; const char*_p=getenv("ACUDA_DRVTRACE"); if(_p&&*_p)_fp=fopen(_p,"a");} if(_fp){fprintf(_fp,"%lld memcpy_d2h_ret r=0\n",_n++);fflush(_fp);} } return CUDA_SUCCESS; }
    void *host = NULL;
    r = host_for(src, bytes, &host);
    if (r != CUDA_SUCCESS)
        { { static FILE *_fp; static int _i; static long long _n; if(!_i){_i=1; const char*_p=getenv("ACUDA_DRVTRACE"); if(_p&&*_p)_fp=fopen(_p,"a");} if(_fp){fprintf(_fp,"%lld memcpy_d2h_ret r=%d\n",_n++,(int)r);fflush(_fp);} } return r; }
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
    /* Asynchronous DtoD: device memory is host mapped, so apply the copy
     * immediately and let the caller order it on its stream.  Taking the
     * synchronising path here blocks PhysX's worker thread inside
     * vkWaitSemaphores when a queued signal does not arrive. */
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemcpyDtoDAsync(dst=%llx src=%llx bytes=%zu)\n",
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
    memmove(hd, hs, bytes);
    return CUDA_SUCCESS;
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

// Copy a rectangle between host memory and an array level/layer.
CUresult driver_array_rw(CUarray a, int write, size_t xb, size_t y,
                                size_t z, void *host, size_t wbytes,
                                size_t h, size_t depth, size_t pitch) {
    if (!a || !a->image || !host)
        return CUDA_ERROR_INVALID_VALUE;
    acuda_buffer_t img = a->image;
    uint32_t level = a->level;
    uint32_t lw = img->width >> level;
    if (!lw) lw = 1;
    uint32_t lh = img->dim1d ? 1u : (img->height >> level);
    if (!lh) lh = 1;
    uint32_t ld = img->dim3d ? (img->depth >> level) : 1u;
    if (!ld) ld = 1;
    size_t elem = (size_t)driver_elem_bytes(img->format);
    if (elem == 0 || xb + wbytes > (size_t)lw * elem || y + h > lh ||
        z + depth > (img->dim3d ? ld : img->layers))
        return CUDA_ERROR_INVALID_VALUE;
    size_t layer_bytes = (size_t)lw * lh * elem;
    size_t total = img->dim3d ? layer_bytes * ld : layer_bytes;
    void *tmp = malloc(total);
    if (!tmp)
        return CUDA_ERROR_OUT_OF_MEMORY;
    acuda_status_t st;
    if (img->dim3d) {
        st = acuda_image_download(g_actx, img, level, 0, tmp);
        if (st != ACUDA_SUCCESS) {
            free(tmp);
            return map_status(st);
        }
    }
    for (size_t dz = 0; dz < depth; dz++) {
        void *lb;
        uint32_t layer = 0;
        if (img->dim3d) {
            lb = tmp;
        } else {
            layer = (uint32_t)(z + dz);
            lb = tmp;
            st = acuda_image_download(g_actx, img, level, layer, lb);
            if (st != ACUDA_SUCCESS) {
                fprintf(stderr, "ACuda-dbg: array_rw download fail st=%d\n", (int)st);
                free(tmp);
                return map_status(st);
            }
        }
        for (size_t row = 0; row < h; row++) {
            size_t ry = y + row;
            size_t rz = img->dim3d ? (z + dz) : 0;
            unsigned char *lp = (unsigned char *)lb +
                                ((size_t)rz * lh + ry) * lw * elem + xb;
            unsigned char *hp = (unsigned char *)host +
                                (dz * h + row) * pitch;
            if (write)
                memcpy(lp, hp, wbytes);
            else
                memcpy(hp, lp, wbytes);
        }
        if (!img->dim3d) {
            if (write && dz == 0 && y == 0) {
                const float *fv = (const float *)host;
                fprintf(stderr, "ACuda-dbg: array_rw write wbytes=%zu h=%zu depth=%zu pitch=%zu elem=%zu lw=%u lh=%u first=%f %f %f %f\n",
                        wbytes, h, depth, pitch, elem, (unsigned)lw, (unsigned)lh,
                        fv[0], fv[1], fv[2], fv[3]);
            }
            st = acuda_image_upload(g_actx, img, level, layer, lb);
            if (st != ACUDA_SUCCESS) {
                fprintf(stderr, "ACuda-dbg: array_rw upload fail st=%d\n", (int)st);
                free(tmp);
                return map_status(st);
            }
        }
    }
    if (img->dim3d) {
        st = acuda_image_upload(g_actx, img, level, 0, tmp);
        if (st != ACUDA_SUCCESS) {
            free(tmp);
            return map_status(st);
        }
    }
    free(tmp);
    return CUDA_SUCCESS;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    if ((!dst || !src) && count)
        return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
    if (!count)
        return (cudaError_t)CUDA_SUCCESS;
    switch ((int)kind) {
    case 0: /* cudaMemcpyHostToHost */
        memcpy(dst, src, count);
        return (cudaError_t)CUDA_SUCCESS;
    case 1: /* cudaMemcpyHostToDevice */
        return (cudaError_t)cuMemcpyHtoD_v2((CUdeviceptr)(uintptr_t)dst, src, count);
    case 2: /* cudaMemcpyDeviceToHost */
        return (cudaError_t)cuMemcpyDtoH_v2(dst, (CUdeviceptr)(uintptr_t)src, count);
    case 3: /* cudaMemcpyDeviceToDevice */
        return (cudaError_t)cuMemcpyDtoD_v2((CUdeviceptr)(uintptr_t)dst,
                                            (CUdeviceptr)(uintptr_t)src, count);
    case 4: /* cudaMemcpyDefault */
        memcpy(dst, src, count);
        return (cudaError_t)CUDA_SUCCESS;
    default:
        return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult ACUDA_CUDAAPI cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount) {
    CUcontext c = NULL; CUresult r = need_ctx(&c); if (r != CUDA_SUCCESS) return r;
    if (!dstArray || (!srcHost && ByteCount)) return CUDA_ERROR_INVALID_VALUE;
    return driver_memcpy_common(dstOffset, 0, 0, CU_MEMORYTYPE_HOST, srcHost, 0, NULL, ByteCount, 0,
                                0, 0, 0, CU_MEMORYTYPE_ARRAY, NULL, 0, dstArray, 0, 0, ByteCount, 1, 1);
}
CUresult ACUDA_CUDAAPI cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount) {
    return cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount);
}
CUresult ACUDA_CUDAAPI cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
    CUcontext c = NULL; CUresult r = need_ctx(&c); if (r != CUDA_SUCCESS) return r;
    if (!srcArray || (!dstHost && ByteCount)) return CUDA_ERROR_INVALID_VALUE;
    return driver_memcpy_common(srcOffset, 0, 0, CU_MEMORYTYPE_ARRAY, NULL, 0, srcArray, 0, 0,
                                0, 0, 0, CU_MEMORYTYPE_HOST, dstHost, 0, NULL, ByteCount, 0, ByteCount, 1, 1);
}
CUresult ACUDA_CUDAAPI cuMemcpyAtoH(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
    return cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount);
}
CUresult ACUDA_CUDAAPI cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
    CUcontext c = NULL; CUresult r = need_ctx(&c); if (r != CUDA_SUCCESS) return r;
    if (!dstArray || !srcArray) return CUDA_ERROR_INVALID_VALUE;
    return driver_memcpy_common(srcOffset, 0, 0, CU_MEMORYTYPE_ARRAY, NULL, 0, srcArray, 0, 0,
                                dstOffset, 0, 0, CU_MEMORYTYPE_ARRAY, NULL, 0, dstArray, 0, 0, ByteCount, 1, 1);
}
CUresult ACUDA_CUDAAPI cuMemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
    return cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
}
CUresult ACUDA_CUDAAPI cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream) {
    (void)hStream; return cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount);
}
CUresult ACUDA_CUDAAPI cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) {
    (void)hStream; return cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount);
}
CUresult ACUDA_CUDAAPI cuMemcpyAtoAAsync_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) {
    (void)hStream; return cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
}
