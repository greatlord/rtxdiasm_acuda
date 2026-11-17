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
struct CUmod_st {
    CUcontext ctx;
    char *ptx;
    unsigned char *image;
    size_t image_size;
    int ptx_oom;
    struct { char name[ACUDA_ENTRY_NAME_MAX]; acuda_kernel_t *k; } *funcs;
    size_t nfuncs, fcap;
    struct CUtexref_st **texrefs;
    size_t ntexrefs, texrefs_cap;
    struct {
        char name[ACUDA_ENTRY_NAME_MAX];
        CUdeviceptr dptr;
        size_t bytes;
    } *globals;
    size_t nglobals, globals_cap;
};

struct CUarray_st {
    acuda_buffer_t image;
    CUDA_ARRAY_DESCRIPTOR desc2;
    CUDA_ARRAY3D_DESCRIPTOR desc3;
    int is3d;
    unsigned int level;
    int own;
    struct CUarray_st *parent;
};

struct CUmipmappedArray_st {
    acuda_buffer_t image;
    CUDA_ARRAY3D_DESCRIPTOR desc;
    unsigned int levels;
    int is3d;
};

typedef struct {
    CUtexObject id;
    acuda_buffer_t tex;
    acuda_buffer_t sampler;
    acuda_buffer_t own_image;
    CUDA_RESOURCE_DESC res;
    CUDA_TEXTURE_DESC td;
} driver_texobj_t;

typedef struct {
    CUsurfObject id;
    uint32_t slot;
    acuda_buffer_t image;
    CUDA_RESOURCE_DESC res;
} driver_surfobj_t;

struct CUtexref_st {
    CUmodule mod;
    char name[ACUDA_ENTRY_NAME_MAX];
    CUdeviceptr addr;
    size_t bytes;
    CUarray array;
    int is_array;
    int fmt_set;
    CUarray_format fmt;
    int channels;
    unsigned int flags;
    CUfilter_mode filter;
    CUaddress_mode address_mode[3];
    unsigned int max_aniso;
    CUtexObject obj;
    CUdeviceptr built_addr;
    size_t built_bytes;
    CUarray_format built_fmt;
    int built_channels;
};

extern driver_texobj_t *g_texobjs;
extern size_t g_ntexobjs, g_ctexobjs;
extern CUtexObject g_next_texid;
extern driver_surfobj_t *g_surfobjs;
extern size_t g_nsurfobjs, g_csurfobjs;
extern acuda_buffer_t *g_surf_slots;
extern uint32_t g_surf_slot_cap;

driver_surfobj_t *driver_surf_find(CUsurfObject id);
void driver_img_type(acuda_buffer_t img, int *dim, int *arrayed, int *isint);

struct CUfunc_st {
    uint32_t magic;
    CUmodule mod;
    acuda_kernel_t *k;
    char name[ACUDA_ENTRY_NAME_MAX];
    int maxdyn_set;
    unsigned maxdyn;
};
struct CUkernel_st {
    uint32_t magic;
    CUmodule mod;
    char name[ACUDA_ENTRY_NAME_MAX];
    CUfunction func;
};
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




driver_texobj_t *driver_tex_find(CUtexObject id);
int driver_elem_bytes(uint32_t fmt);

typedef void (ACUDA_CUDAAPI *acuda_stream_cb_t)(CUstream hStream, CUresult status, void *userData);

CUresult driver_array_rw(CUarray a, int write, size_t xb, size_t y, size_t z, void *host, size_t wbytes, size_t h, size_t depth, size_t pitch);
CUresult driver_memcpy_common(size_t src_xb, size_t src_y, size_t src_z, CUmemorytype src_type, const void *src_host, CUdeviceptr src_dev, CUarray src_array, size_t src_pitch, size_t src_height, size_t dst_xb, size_t dst_y, size_t dst_z, CUmemorytype dst_type, void *dst_host, CUdeviceptr dst_dev, CUarray dst_array, size_t dst_pitch, size_t dst_height, size_t wbytes, size_t h, size_t depth);

#if defined(_WIN32)
int acuda_host_heap_init(size_t need);
#endif

#if defined(_WIN32) && !defined(_WIN64)
#include <stdio.h>
static inline void acuda_trace_va(const char *tag) {
    if (!getenv("ACUDA_TRACE"))
        return;
    MEMORY_BASIC_INFORMATION mbi;
    unsigned char *p = (unsigned char *)0x10000;
    size_t total_free = 0, largest = 0, committed = 0;
    while ((uintptr_t)p < (uintptr_t)0x80000000u) {
        if (VirtualQuery(p, &mbi, sizeof(mbi)) != sizeof(mbi))
            break;
        if (mbi.State == MEM_FREE) {
            total_free += mbi.RegionSize;
            if (mbi.RegionSize > largest)
                largest = mbi.RegionSize;
        } else if (mbi.State == MEM_COMMIT) {
            committed += mbi.RegionSize;
        }
        unsigned char *next =
            (unsigned char *)mbi.BaseAddress + mbi.RegionSize;
        if (next <= p)
            break;
        p = next;
    }
    fprintf(stderr,
            "acuda-va %s: free=%zu largest=%zu committed=%zu\n",
            tag, total_free, largest, committed);
}
#else
static inline void acuda_trace_va(const char *tag) { (void)tag; }
#endif

CUcontext current_ctx(void);

#ifndef ACUDA_MAX_DEVICES
#define ACUDA_MAX_DEVICES 8
#endif
extern CUcontext primary_ctx[ACUDA_MAX_DEVICES];
extern int primary_refs[ACUDA_MAX_DEVICES];
extern CUmemPool default_pool[ACUDA_MAX_DEVICES];
CUresult check_dev(int dev);

struct CUmemPool_st {
    int device;
};

uint32_t rd32le(const unsigned char *p);
uint64_t rd64le(const unsigned char *p);
int cubin_backend(void);
char *fatbin_first_ptx(const unsigned char *blob, size_t cap);
CUresult attach_sass_image(CUmodule module, const unsigned char *data, size_t size);
CUresult module_from_sass_image(CUmodule *module, const unsigned char *data, size_t size);

#define ACUDA_HANDLE_FUNC 0x46554e43u
#define ACUDA_HANDLE_KERNEL 0x4b45524eu
CUresult handle_as_func(void *handle, CUfunction *out);

CUresult bind_for(CUdeviceptr dptr, size_t bytes, VkBuffer *buf, size_t *offset);

void acuda_install_veh(void);

int cc_major(void);
int cc_minor(void);
CUresult need_actx(void);
