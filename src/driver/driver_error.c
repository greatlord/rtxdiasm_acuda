#include "acuda/driver.h"
#include "acuda/runtime_api.h"
#include <stddef.h>

static const struct { CUresult code; const char *name, *str; } err_table[] = {    { CUDA_SUCCESS, "CUDA_SUCCESS", "no error" },
    { CUDA_ERROR_INVALID_VALUE, "CUDA_ERROR_INVALID_VALUE", "invalid argument" },
    { CUDA_ERROR_OUT_OF_MEMORY, "CUDA_ERROR_OUT_OF_MEMORY", "out of memory" },
    { CUDA_ERROR_NOT_INITIALIZED, "CUDA_ERROR_NOT_INITIALIZED",
      "driver not initialized" },
    { CUDA_ERROR_DEINITIALIZED, "CUDA_ERROR_DEINITIALIZED", "driver shutting down" },
    { CUDA_ERROR_NO_DEVICE, "CUDA_ERROR_NO_DEVICE", "no CUDA-capable device" },
    { CUDA_ERROR_INVALID_DEVICE, "CUDA_ERROR_INVALID_DEVICE", "invalid device ordinal" },
    { CUDA_ERROR_INVALID_IMAGE, "CUDA_ERROR_INVALID_IMAGE",
      "device program image invalid (cubin/SASS unsupported; PTX only)" },
    { CUDA_ERROR_INVALID_CONTEXT, "CUDA_ERROR_INVALID_CONTEXT", "invalid context" },
    { CUDA_ERROR_INVALID_HANDLE, "CUDA_ERROR_INVALID_HANDLE", "invalid handle" },
    { CUDA_ERROR_NOT_FOUND, "CUDA_ERROR_NOT_FOUND", "named symbol not found" },
    { CUDA_ERROR_NOT_READY, "CUDA_ERROR_NOT_READY", "operation not ready" },
    { CUDA_ERROR_NOT_SUPPORTED, "CUDA_ERROR_NOT_SUPPORTED", "operation not supported" },
    { CUDA_ERROR_UNKNOWN, "CUDA_ERROR_UNKNOWN", "unknown error" },
};

CUresult ACUDA_CUDAAPI cuGetErrorString(CUresult error, const char **pStr) {
    if (!pStr)
        return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < sizeof(err_table) / sizeof(err_table[0]); i++) {
        if (err_table[i].code == error) {
            *pStr = err_table[i].str;
            return CUDA_SUCCESS;
        }
    }
    *pStr = "unknown error";
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuGetErrorName(CUresult error, const char **pStr) {
    if (!pStr)
        return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < sizeof(err_table) / sizeof(err_table[0]); i++) {
        if (err_table[i].code == error) {
            *pStr = err_table[i].name;
            return CUDA_SUCCESS;
        }
    }
    *pStr = "CUDA_ERROR_UNKNOWN";
    return CUDA_SUCCESS;
}

/* Single runtime error state for the driver.  The CUDA 13 hybrid runtime
 * reaches cudaGetLastError/cudaPeekAtLastError through the dd9e98a4 interface,
 * and the rest of the exported runtime API lives in this DLL too, so there is
 * one per-thread state here instead of a second one in the interface shim. */
#if defined(_MSC_VER)
#define ACUDA_RT_TLS __declspec(thread)
#else
#define ACUDA_RT_TLS __thread
#endif

static ACUDA_RT_TLS cudaError_t g_cuda_last_error = cudaSuccess;

cudaError_t cudaGetLastError(void) {
    cudaError_t e = g_cuda_last_error;
    g_cuda_last_error = cudaSuccess;
    return e;
}

cudaError_t cudaPeekAtLastError(void) { return g_cuda_last_error; }

void cudaSetLastError(cudaError_t e) { g_cuda_last_error = e; }
