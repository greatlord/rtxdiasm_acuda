#include "acuda/runtime.h"

const char *acuda_strerror(acuda_status_t status) {
    switch (status) {
    case ACUDA_SUCCESS: return "success";
    case ACUDA_ERROR_UNINITIALIZED: return "uninitialized or never recorded";
    case ACUDA_ERROR_INVALID_VALUE: return "invalid value";
    case ACUDA_ERROR_OUT_OF_MEMORY: return "out of memory";
    case ACUDA_ERROR_NO_DEVICE: return "no suitable device";
    case ACUDA_ERROR_NOT_SUPPORTED: return "not supported";
    case ACUDA_ERROR_INTERNAL: return "internal error";
    case ACUDA_ERROR_INVALID_HANDLE: return "invalid handle";
    case ACUDA_ERROR_NOT_READY: return "not ready";
    case ACUDA_ERROR_TIMEOUT: return "timeout";
    default: return "unknown status";
    }
}
