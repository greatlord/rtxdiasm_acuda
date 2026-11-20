#ifndef ACUDA_DRIVER_EXPORT_DD9E98A4_TYPES_H
#define ACUDA_DRIVER_EXPORT_DD9E98A4_TYPES_H

#include "acuda/driver.h"

#include <stddef.h>

typedef struct {
    unsigned int x, y, z;
} acuda_dd9e98a4_dim3;

#define ACUDA_DD9E98A4_MAX_FATBINS 8
#define ACUDA_DD9E98A4_MAX_FUNCS 256
#define ACUDA_DD9E98A4_MAX_VARS 256

typedef struct {
    const void *blob;
    CUmodule module;
    int state; /* 0 = not loaded, 1 = loaded, -1 = failed */
} acuda_dd9e98a4_fatbin;

typedef struct {
    const void *host_fn;
    const char *device_name; /* owned by the executable image */
    acuda_dd9e98a4_fatbin *fatbin;
} acuda_dd9e98a4_kernel;

typedef struct {
    acuda_dd9e98a4_dim3 grid;
    acuda_dd9e98a4_dim3 block;
    size_t shared;
    void *stream;
} acuda_dd9e98a4_launch_config;

typedef struct {
    void *fatbin_handle; /* 1-based index as returned by register_fat_binary */
    void *host_var;
    const char *device_name;
    void *device_address; /* host-side symbol address, if provided */
    size_t size;
    int constant;
    int global;
    int ext;
} acuda_dd9e98a4_var;

#if defined(_MSC_VER)
#define ACUDA_DD9E98A4_TLS __declspec(thread)
#else
#define ACUDA_DD9E98A4_TLS __thread
#endif

#endif /* ACUDA_DRIVER_EXPORT_DD9E98A4_TYPES_H */
