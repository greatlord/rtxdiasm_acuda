#ifndef ACUDA_DRIVER_INTERNAL_H
#define ACUDA_DRIVER_INTERNAL_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rtxdiasm_fatbin.h"
#if defined(_M_IX86)
#include <intrin.h>
#endif
#include "internal.h"
#include "arena.h"
#include "rtxdisam_bridge.h"
#include "acuda/driver.h"
#include "acuda/compiler.h"
#include "acuda/context.h"
#include "acuda/memory.h"
#include "acuda/stream.h"

#if defined(_WIN32)
#include <windows.h>
#include <direct.h>
#ifndef setenv
#define setenv(name, value, overwrite) _putenv_s((name), (value))
#endif
#ifndef unsetenv
#define unsetenv(name) _putenv_s((name), "")
#endif
#else
#include <dlfcn.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define ACUDA_CC_MAJOR 8
#define ACUDA_CC_MINOR 6
#define ACUDA_DRIVER_VERSION 13040
#if defined(__i386__) || (defined(_WIN32) && !defined(_WIN64))
#define ACUDA_DEVICE_MEM (512ull << 20)
#else
#define ACUDA_DEVICE_MEM (8ull << 30)
#endif
#define ACUDA_CTX_STACK 16
#define ACUDA_PTX_EAGER_VALIDATE_MAX (1u << 20)

#ifndef ACUDA_DEBUG_PRINT
#define ACUDA_DEBUG_PRINT(...) do { \
    if (getenv("ACUDA_TRACE")) fprintf(stderr, __VA_ARGS__); \
} while (0)
#endif
#ifndef ACUDA_DEBUG_PRINT_PROC
#define ACUDA_DEBUG_PRINT_PROC(...) do { \
    if (getenv("ACUDA_TRACE_PROC")) fprintf(stderr, __VA_ARGS__); \
} while (0)
#endif

struct CUctx_st {
    int device;
};
struct CUmod_st;
struct CUtexref_st;
struct CUfunc_st;
struct CUkernel_st;
struct CUstream_st {
    CUcontext ctx;
    acuda_stream_t s;
};
struct CUevent_st {
    CUcontext ctx;
    acuda_event_t e;
};
struct CUlink_st {
    char **blobs;
    size_t *sizes;
    size_t n, cap;
    char *linked;
    size_t linked_size;
};

extern int g_init;
extern acuda_context_t g_actx;
extern int g_actx_exit_registered;
extern CUcontext g_stack[ACUDA_CTX_STACK];
extern size_t g_depth;

unsigned long long acuda_device_mem_total(void);
CUresult map_status(acuda_status_t st);
CUresult need_init(void);
CUresult need_actx(void);
CUresult need_ctx(CUcontext *out);
CUresult ctx_push(CUcontext ctx);
CUresult mem_alloc(CUdeviceptr *dptr, size_t bytesize);
CUresult host_for(CUdeviceptr dptr, size_t bytes, void **host);
CUresult bind_for(CUdeviceptr dptr, size_t bytes, VkBuffer *buf,
                  VkDeviceSize *offset);
CUresult memcpy_h2d(CUdeviceptr dst, const void *src, size_t bytes);
CUresult memcpy_d2h(void *dst, CUdeviceptr src, size_t bytes);
CUresult memcpy_d2d(CUdeviceptr dst, CUdeviceptr src, size_t bytes);

#endif
