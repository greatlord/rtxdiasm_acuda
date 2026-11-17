#include "driver_internal.h"
#include <string.h>
#include <stdint.h>

static CUresult memset_d(CUdeviceptr dst, unsigned long long value, size_t count,
                         int is32) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    size_t bytes = count * (is32 ? 4u : 1u);
    if (bytes == 0)
        return CUDA_SUCCESS;
    void *host = NULL;
    r = host_for(dst, bytes, &host);
    if (r != CUDA_SUCCESS)
        return r;
    r = map_status(acuda_stream_synchronize(g_actx, NULL));
    if (r != CUDA_SUCCESS)
        return r;
    if (is32) {
        // Word fill (cuMemsetD32 semantics); offset assumed 4-aligned.
        uint32_t *w = host;
        for (size_t i = 0; i < count; i++)
            w[i] = (uint32_t)value;
    } else {
        memset(host, (int)(unsigned char)value, bytes);
    }
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemsetD32_v2(CUdeviceptr dst, unsigned int value, size_t count) {
    return memset_d(dst, value, count, 1);
}

CUresult ACUDA_CUDAAPI cuMemsetD32(CUdeviceptr dst, unsigned int value, size_t count) {
    return memset_d(dst, value, count, 1);
}

CUresult ACUDA_CUDAAPI cuMemsetD8_v2(CUdeviceptr dst, unsigned char value, size_t count) {
    return memset_d(dst, value, count, 0);
}

CUresult ACUDA_CUDAAPI cuMemsetD8(CUdeviceptr dst, unsigned char value, size_t count) {
    return memset_d(dst, value, count, 0);
}

static CUresult memset_d16(CUdeviceptr dst, unsigned short value, size_t count) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (count == 0)
        return CUDA_SUCCESS;
    void *host = NULL;
    r = host_for(dst, count * 2u, &host);
    if (r != CUDA_SUCCESS)
        return r;
    r = map_status(acuda_stream_synchronize(g_actx, NULL));
    if (r != CUDA_SUCCESS)
        return r;
    {
        uint16_t *w = host;
        for (size_t i = 0; i < count; i++)
            w[i] = value;
    }
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemsetD16_v2(CUdeviceptr dst, unsigned short value,
                                      size_t count) {
    return memset_d16(dst, value, count);
}

CUresult ACUDA_CUDAAPI cuMemsetD16(CUdeviceptr dst, unsigned short value,
                                   size_t count) {
    return memset_d16(dst, value, count);
}

CUresult ACUDA_CUDAAPI cuMemsetD16Async(CUdeviceptr dst, unsigned short value,
                                       size_t count, CUstream hStream) {
    (void)hStream;
    return memset_d16(dst, value, count);
}

// 2D device memset used by cudaMemset2D.  Width is in elements (bytes for
// D8, 16-bit units for D16, 32-bit units for D32); the driver is
// synchronous so the Async variants run inline.
static CUresult memset_d2d(CUdeviceptr dst, size_t pitch,
                           unsigned long long value, size_t width,
                           size_t height, int elem_size) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (width == 0 || height == 0)
        return CUDA_SUCCESS;
    if (pitch < width * (size_t)elem_size)
        return CUDA_ERROR_INVALID_VALUE;
    r = map_status(acuda_stream_synchronize(g_actx, NULL));
    if (r != CUDA_SUCCESS)
        return r;
    for (size_t y = 0; y < height; y++) {
        void *host = NULL;
        r = host_for(dst + y * pitch, width * (size_t)elem_size, &host);
        if (r != CUDA_SUCCESS)
            return r;
        if (elem_size == 4) {
            uint32_t *w = host;
            for (size_t i = 0; i < width; i++)
                w[i] = (uint32_t)value;
        } else if (elem_size == 2) {
            uint16_t *w = host;
            for (size_t i = 0; i < width; i++)
                w[i] = (uint16_t)value;
        } else {
            memset(host, (int)(unsigned char)value, width);
        }
    }
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemsetD2D8_v2(CUdeviceptr dst, size_t pitch,
                                       unsigned char value, size_t width,
                                       size_t height) {
    return memset_d2d(dst, pitch, value, width, height, 1);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D8(CUdeviceptr dst, size_t pitch,
                                    unsigned char value, size_t width,
                                    size_t height) {
    return memset_d2d(dst, pitch, value, width, height, 1);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D8Async(CUdeviceptr dst, size_t pitch,
                                        unsigned char value, size_t width,
                                        size_t height, CUstream hStream) {
    (void)hStream;
    return memset_d2d(dst, pitch, value, width, height, 1);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D16_v2(CUdeviceptr dst, size_t pitch,
                                       unsigned short value, size_t width,
                                       size_t height) {
    return memset_d2d(dst, pitch, value, width, height, 2);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D16(CUdeviceptr dst, size_t pitch,
                                    unsigned short value, size_t width,
                                    size_t height) {
    return memset_d2d(dst, pitch, value, width, height, 2);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D16Async(CUdeviceptr dst, size_t pitch,
                                         unsigned short value, size_t width,
                                         size_t height, CUstream hStream) {
    (void)hStream;
    return memset_d2d(dst, pitch, value, width, height, 2);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D32_v2(CUdeviceptr dst, size_t pitch,
                                       unsigned int value, size_t width,
                                       size_t height) {
    return memset_d2d(dst, pitch, value, width, height, 4);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D32(CUdeviceptr dst, size_t pitch,
                                    unsigned int value, size_t width,
                                    size_t height) {
    return memset_d2d(dst, pitch, value, width, height, 4);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D32Async(CUdeviceptr dst, size_t pitch,
                                         unsigned int value, size_t width,
                                         size_t height, CUstream hStream) {
    (void)hStream;
    return memset_d2d(dst, pitch, value, width, height, 4);
}
