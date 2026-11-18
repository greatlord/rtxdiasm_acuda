#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_stub.h"
#include "acuda/runtime_api.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#if defined(_WIN32)
#include <windows.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#else
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#endif

static void acuda_log_caller(const char *tag) {
#if defined(_WIN32) && defined(_MSC_VER)
    void *retaddr = _ReturnAddress();
    MEMORY_BASIC_INFORMATION mbi;
    char modpath[MAX_PATH] = "?";
    HMODULE mod = NULL;
    if (VirtualQuery(retaddr, &mbi, sizeof(mbi))) {
        mod = (HMODULE)mbi.AllocationBase;
        GetModuleFileNameA(mod, modpath, MAX_PATH);
    }
    ACUDA_DEBUG_PRINT( "ACuda-debug: %s caller=%p modbase=%p mod=%s\n",
            tag, retaddr, (void *)mod, modpath);
#else
    (void)tag;
#endif
}

// CUDA 13's runtime fetches this small interface table with
// cuGetExportTable instead of resolving the corresponding entry points
// one by one.  It is queried during device enumeration; slot 12 is the
// "load compilers" callback, which is a no-op for our software path.
static CUresult ACUDA_CUDAAPI cudart_iface_load_compilers(void) {
    return CUDA_SUCCESS;
}

static CUresult ACUDA_CUDAAPI cudart_iface_primary_ctx(CUcontext *pctx, CUdevice dev) {
    return cuDevicePrimaryCtxRetain(pctx, dev);
}

static CUresult ACUDA_CUDAAPI cudart_iface_primary_ctx_flags(CUdevice dev,
                                                              unsigned int flags) {
    CUcontext ctx;
    CUresult r = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (r != CUDA_SUCCESS)
        return r;
    return cuDevicePrimaryCtxSetFlags(dev, flags);
}

static CUresult ACUDA_CUDAAPI cudart_iface_module_from_cubin(
    CUmodule *pmod, const void *fatbin) {
    if (!pmod || !fatbin)
        return CUDA_ERROR_INVALID_VALUE;
    return cuModuleLoadData(pmod, fatbin);
}

static CUresult ACUDA_CUDAAPI cudart_iface_module_ext1(
    CUmodule *pmod, const void *fatbin_wrapper, void *arg3, void *arg4,
    unsigned int arg5) {
    (void)arg3; (void)arg4; (void)arg5;
    if (!pmod || !fatbin_wrapper)
        return CUDA_ERROR_INVALID_VALUE;
    return cuModuleLoadFatBinary(pmod, fatbin_wrapper);
}

static void ACUDA_CUDAAPI cudart_iface_fn7(size_t arg) {
    (void)arg;
}

static CUresult ACUDA_CUDAAPI cudart_iface_module_ext2(
    const void *fatbin_header, CUmodule *pmod, void *arg3, void *arg4,
    unsigned int arg5) {
    (void)arg3; (void)arg4; (void)arg5;
    if (!pmod || !fatbin_header)
        return CUDA_ERROR_INVALID_VALUE;
    return cuModuleLoadData(pmod, fatbin_header);
}

// Opaque scratch buffers requested by the tools runtime hooks.  The
// caller treats them as private storage; zeroed memory is enough.
static unsigned int g_tools_buffer1[1024];
static unsigned int g_tools_buffer2[14];

static void ACUDA_CUDAAPI tools_runtime_buffer1(void **ptr, size_t *size) {
    if (ptr)
        *ptr = g_tools_buffer1;
    if (size)
        *size = sizeof(g_tools_buffer1) / sizeof(g_tools_buffer1[0]);
}

static void ACUDA_CUDAAPI tools_runtime_buffer2(void **ptr, size_t *size) {
    if (ptr)
        *ptr = g_tools_buffer2;
    if (size)
        *size = sizeof(g_tools_buffer2) / sizeof(g_tools_buffer2[0]);
}

// CUDA 4.x cudart calls this slot from cudaMallocHost with
// (event_id, event_data).  Nothing in the data is consumed by our
// software path, so a success return is enough.
static int ACUDA_CUDAAPI tools_runtime_event(unsigned int code, void *data) {
    (void)code;
    (void)data;
    return 0;
}

static const void *g_tools_tls[4] = {
    (const void *)(uintptr_t)(4 * sizeof(void *)),   /* [0] table size */
    NULL, NULL, NULL
};

static const void *g_tools_runtime_hooks[7] = {
    (const void *)(uintptr_t)(7 * sizeof(void *)),   /* [0] table size */
    (const void *)(uintptr_t)tools_runtime_event,    /* [1] */
    (const void *)(uintptr_t)tools_runtime_buffer1,  /* [2] */
    NULL, NULL, NULL,
    (const void *)(uintptr_t)tools_runtime_buffer2   /* [6] */
};

/* Legacy 32-bit CUDA 4.x tools table.  CUDA 4.x runtimes query this
 * interface during context setup and expect six callable entries.  The
 * calls only carry bookkeeping/query data for tools; safe zeroed replies
 * are enough for our non-instrumented runtime. */
static int ACUDA_CUDAAPI tools_legacy_query_index(unsigned index, void *out) {
    (void)index;
    if (out)
        *(unsigned int *)out = 0;
    return 0;
}
static int ACUDA_CUDAAPI tools_legacy_pair_a(void *a, void *b) {
    if (a)
        *(unsigned int *)a = 0;
    if (b)
        *(unsigned int *)b = 0x400;
    return 0;
}
static int ACUDA_CUDAAPI tools_legacy_ctx(void *ctx, void *out, void *extra) {
    (void)ctx;
    (void)extra;
    if (out)
        *(unsigned int *)out = 0;
    return 0;
}
static int ACUDA_CUDAAPI tools_legacy_ctx2(void *ctx, void *out) {
    (void)ctx;
    if (out) {
        ((unsigned int *)out)[0] = 0;
        ((unsigned int *)out)[1] = 0;
    }
    return 0;
}
static int ACUDA_CUDAAPI tools_legacy_pair_b(void *a, void *b) {
    if (a)
        *(unsigned int *)a = 0;
    if (b)
        *(unsigned int *)b = 0xE;
    return 0;
}
static int ACUDA_CUDAAPI tools_legacy_zero(void *a, void *b) {
    if (a)
        *(unsigned int *)a = 0;
    if (b)
        *(unsigned int *)b = 0;
    return 0;
}
#if defined(__GNUC__)
__attribute__((unused))
#endif
static void *g_tools_runtime_hooks_legacy[7] = {
    (void *)(uintptr_t)(7 * sizeof(void *)),
    (void *)(uintptr_t)tools_legacy_query_index,
    (void *)(uintptr_t)tools_legacy_pair_a,
    (void *)(uintptr_t)tools_legacy_ctx,
    (void *)(uintptr_t)tools_legacy_ctx2,
    (void *)(uintptr_t)tools_legacy_pair_b,
    (void *)(uintptr_t)tools_legacy_zero
};

// Context-local storage (C693336E), used by the tools layer to attach
// small records to a context.  A fixed table is enough for our use.
#define ACUDA_CTX_STORAGE_MAX 64
typedef struct {
    CUcontext ctx;
    void *key;
    void *value;
    void (*dtor)(CUcontext, void *, void *);
} acuda_ctx_slot_t;
static acuda_ctx_slot_t g_ctx_slots[ACUDA_CTX_STORAGE_MAX];

static CUresult ACUDA_CUDAAPI ctx_storage_put(CUcontext ctx, void *key, void *value,
                                void (*dtor)(CUcontext, void *, void *)) {
    for (int i = 0; i < ACUDA_CTX_STORAGE_MAX; i++) {
        if (g_ctx_slots[i].ctx == ctx && g_ctx_slots[i].key == key) {
            g_ctx_slots[i].value = value;
            g_ctx_slots[i].dtor = dtor;
            return CUDA_SUCCESS;
        }
    }
    for (int i = 0; i < ACUDA_CTX_STORAGE_MAX; i++) {
        if (g_ctx_slots[i].key == NULL) {
            g_ctx_slots[i].ctx = ctx;
            g_ctx_slots[i].key = key;
            g_ctx_slots[i].value = value;
            g_ctx_slots[i].dtor = dtor;
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
}

static CUresult ACUDA_CUDAAPI ctx_storage_delete(CUcontext ctx, void *key) {
    for (int i = 0; i < ACUDA_CTX_STORAGE_MAX; i++) {
        if (g_ctx_slots[i].ctx == ctx && g_ctx_slots[i].key == key) {
            g_ctx_slots[i].key = NULL;
            g_ctx_slots[i].ctx = NULL;
            g_ctx_slots[i].value = NULL;
            g_ctx_slots[i].dtor = NULL;
            return CUDA_SUCCESS;
        }
    }
    return CUDA_SUCCESS;
}

static CUresult ACUDA_CUDAAPI ctx_storage_get(void **value, CUcontext ctx,
                                              void *key) {
    if (!value)
        return CUDA_ERROR_INVALID_VALUE;
    for (int i = 0; i < ACUDA_CTX_STORAGE_MAX; i++) {
        if (g_ctx_slots[i].ctx == ctx && g_ctx_slots[i].key == key) {
            *value = g_ctx_slots[i].value;
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_INVALID_HANDLE;
}

// The context-storage export table has no size slot: [0] put (4 args),
// [1] delete (2 args), [2] get (3 args).  The real driver's table is
// exactly three function pointers.
static const void *g_ctx_storage[3] = {
    (const void *)(uintptr_t)ctx_storage_put,
    (const void *)(uintptr_t)ctx_storage_delete,
    (const void *)(uintptr_t)ctx_storage_get
};

// CTX_CREATE_BYPASS: the CUDA 4.x runtime uses this interface to create
// the context through the driver without its own thread-local wrapper.
static CUresult ACUDA_CUDAAPI ctx_create_v2_bypass(CUcontext *pctx,
                                                   unsigned int flags,
                                                   CUdevice dev) {
    return cuCtxCreate_v2(pctx, flags, dev);
}

static const void *g_ctx_create_bypass[2] = {
    (const void *)(uintptr_t)(2 * sizeof(void *)),
    (const void *)(uintptr_t)ctx_create_v2_bypass
};

// CONTEXT_CHECKS (263E8860): both probes are advisory; cudart only
// records the answers.
static CUresult ACUDA_CUDAAPI context_check(CUcontext ctx, unsigned int *result1,
                              const void **result2) {
    (void)ctx;
    (void)result2;
    if (result1)
        *result1 = 0;
    return CUDA_SUCCESS;
}

static unsigned int ACUDA_CUDAAPI check_fn3(void) { return 0; }

static const void *g_context_checks[4] = {
    (const void *)(uintptr_t)(4 * sizeof(void *)),  /* [0] table size */
    NULL,
    (const void *)(uintptr_t)context_check,         /* [2] */
    (const void *)(uintptr_t)check_fn3              /* [3] */
};

// HEAP_ACCESS (195BCBF4-D67D-024A-ACC5-1D29CEA631AE): cudart 6.x queries
// this table during context setup and fails every allocation with
// "unknown error" (30) when it is missing.  The driver allocates an
// opaque per-context heap record; cudart stores the pointer and hands
// it back to heap_free at teardown.  Contents are driver-private, so a
// zeroed block is enough.
static CUresult ACUDA_CUDAAPI heap_alloc(void **rec, void *a, void *b) {
    (void)a;
    (void)b;
    if (!rec)
        return CUDA_ERROR_INVALID_VALUE;
    void *p = calloc(1, 64);
    if (!p)
        return CUDA_ERROR_OUT_OF_MEMORY;
    *rec = p;
    return CUDA_SUCCESS;
}

static CUresult ACUDA_CUDAAPI heap_free(void *rec, void *extra) {
    (void)extra;
    free(rec);
    return CUDA_SUCCESS;
}

static const void *g_heap_access[3] = {
    (const void *)(uintptr_t)(3 * sizeof(void *)),  /* [0] table size */
    (const void *)(uintptr_t)heap_alloc,            /* [1] */
    (const void *)(uintptr_t)heap_free              /* [2] */
};

static const void *g_cudart_interface[13];
static const void *g_integrity_check[3];

// INTEGRITY_CHECK (D4082055): CUDA 13 recomputes this checksum to
// confirm that the driver and runtime interface tables agree.  The
// mixing table and round function are the interoperability contract; the
// C implementation below is independent, cross-checked against the
// ZLUDA reference (dark_api/src/lib.rs) and its test vectors.
static const unsigned char acuda_integrity_mix[256] = {
    0x29, 0x2e, 0x43, 0xc9, 0xa2, 0xd8, 0x7c, 0x01, 0x3d, 0x36, 0x54, 0xa1, 0xec, 0xf0, 0x06, 0x13,
    0x62, 0xa7, 0x05, 0xf3, 0xc0, 0xc7, 0x73, 0x8c, 0x98, 0x93, 0x2b, 0xd9, 0xbc, 0x4c, 0x82, 0xca,
    0x1e, 0x9b, 0x57, 0x3c, 0xfd, 0xd4, 0xe0, 0x16, 0x67, 0x42, 0x6f, 0x18, 0x8a, 0x17, 0xe5, 0x12,
    0xbe, 0x4e, 0xc4, 0xd6, 0xda, 0x9e, 0xde, 0x49, 0xa0, 0xfb, 0xf5, 0x8e, 0xbb, 0x2f, 0xee, 0x7a,
    0xa9, 0x68, 0x79, 0x91, 0x15, 0xb2, 0x07, 0x3f, 0x94, 0xc2, 0x10, 0x89, 0x0b, 0x22, 0x5f, 0x21,
    0x80, 0x7f, 0x5d, 0x9a, 0x5a, 0x90, 0x32, 0x27, 0x35, 0x3e, 0xcc, 0xe7, 0xbf, 0xf7, 0x97, 0x03,
    0xff, 0x19, 0x30, 0xb3, 0x48, 0xa5, 0xb5, 0xd1, 0xd7, 0x5e, 0x92, 0x2a, 0xac, 0x56, 0xaa, 0xc6,
    0x4f, 0xb8, 0x38, 0xd2, 0x96, 0xa4, 0x7d, 0xb6, 0x76, 0xfc, 0x6b, 0xe2, 0x9c, 0x74, 0x04, 0xf1,
    0x45, 0x9d, 0x70, 0x59, 0x64, 0x71, 0x87, 0x20, 0x86, 0x5b, 0xcf, 0x65, 0xe6, 0x2d, 0xa8, 0x02,
    0x1b, 0x60, 0x25, 0xad, 0xae, 0xb0, 0xb9, 0xf6, 0x1c, 0x46, 0x61, 0x69, 0x34, 0x40, 0x7e, 0x0f,
    0x55, 0x47, 0xa3, 0x23, 0xdd, 0x51, 0xaf, 0x3a, 0xc3, 0x5c, 0xf9, 0xce, 0xba, 0xc5, 0xea, 0x26,
    0x2c, 0x53, 0x0d, 0x6e, 0x85, 0x28, 0x84, 0x09, 0xd3, 0xdf, 0xcd, 0xf4, 0x41, 0x81, 0x4d, 0x52,
    0x6a, 0xdc, 0x37, 0xc8, 0x6c, 0xc1, 0xab, 0xfa, 0x24, 0xe1, 0x7b, 0x08, 0x0c, 0xbd, 0xb1, 0x4a,
    0x78, 0x88, 0x95, 0x8b, 0xe3, 0x63, 0xe8, 0x6d, 0xe9, 0xcb, 0xd5, 0xfe, 0x3b, 0x00, 0x1d, 0x39,
    0xf2, 0xef, 0xb7, 0x0e, 0x66, 0x58, 0xd0, 0xe4, 0xa6, 0x77, 0x72, 0xf8, 0xeb, 0x75, 0x4b, 0x0a,
    0x31, 0x44, 0x50, 0xb4, 0x8f, 0xed, 0x1f, 0x1a, 0xdb, 0x99, 0x8d, 0x33, 0x9f, 0x11, 0x83, 0x14,
};

static void acuda_integrity_round(unsigned char state[66], unsigned char v) {
    unsigned t1 = state[0x40];
    state[t1 + 0x10] = v;
    unsigned t2 = t1;
    unsigned t3 = (t1 + 1) & 0xf;
    state[t2 + 0x20] = state[t2] ^ v;
    unsigned t4 = acuda_integrity_mix[(v ^ state[0x41]) & 0xff];
    unsigned t5 = state[t2 + 0x30];
    state[t2 + 0x30] = (unsigned char)(t4 ^ t5);
    state[0x41] = (unsigned char)(t4 ^ t5);
    state[0x40] = (unsigned char)t3;
    if (t3 != 0)
        return;
    unsigned char a = 0x29;
    unsigned char c = 0;
    for (;;) {
        a ^= state[0];
        state[0] = a;
        for (unsigned i = 1; i < 0x30; i++) {
            a = (unsigned char)(state[i] ^ acuda_integrity_mix[a]);
            state[i] = a;
        }
        a = (unsigned char)(a + c);
        c = (unsigned char)(c + 1);
        if (c == 0x12)
            break;
        a = acuda_integrity_mix[a];
    }
}

static void acuda_integrity_hash(unsigned char state[66], const void *mix,
                                 size_t len, unsigned char x) {
    const unsigned char *p = (const unsigned char *)mix;
    for (size_t i = 0; i < len; i++)
        acuda_integrity_round(state, (unsigned char)(p[i] ^ x));
}

static void acuda_integrity_finish(unsigned char state[66],
                                   unsigned long long out[2]) {
    unsigned char t = (unsigned char)(16 - state[64]);
    for (unsigned i = 0; i < t; i++)
        acuda_integrity_round(state, t);
    for (unsigned i = 0x30; i < 0x40; i++)
        acuda_integrity_round(state, state[i]);
    memcpy(&out[0], state, sizeof(out[0]));
    memcpy(&out[1], state + 8, sizeof(out[1]));
}

typedef struct {
    uint32_t driver_version;
    uint32_t version;
    uint32_t current_process;
    uint32_t current_thread;
    const void *cudart_table;
    const void *integrity_table;
    const void *fn_address;
    uint64_t unix_seconds;
} acuda_integrity_pass3_t;

typedef struct {
    CUuuid guid;
    int32_t pci_domain;
    int32_t pci_bus;
    int32_t pci_device;
} acuda_integrity_dev_t;

static CUresult ACUDA_CUDAAPI integrity_check(unsigned int version,
                                unsigned long long unix_seconds,
                                unsigned long long result[2]) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? integrity_check(version=%u)\n",
                version);
    if (!result)
        return CUDA_ERROR_INVALID_VALUE;
    switch (version % 10) {
    case 0:
        result[0] = 0x3341181C03CB675CULL;
        result[1] = 0x8ED383AA1F4CD1E8ULL;
        return CUDA_SUCCESS;
    case 1:
        result[0] = 0x1841181C03CB675CULL;
        result[1] = 0x8ED383AA1F4CD1E8ULL;
        return CUDA_SUCCESS;
    default:
        break;
    }

    static const unsigned char pass1[16] = {
        0x14, 0x6a, 0xdd, 0xae, 0x53, 0xa9, 0xa7, 0x52,
        0xaa, 0x08, 0x41, 0x36, 0x0b, 0xf5, 0x5a, 0x9f
    };
    unsigned char state[66];
    memset(state, 0, sizeof(state));
    acuda_integrity_hash(state, pass1, sizeof(pass1), 0x36);

#if defined(_WIN32)
    uint32_t pid = (uint32_t)GetCurrentProcessId();
    uint32_t tid = (uint32_t)GetCurrentThreadId();
#else
    uint32_t pid = (uint32_t)getpid();
    uint32_t tid = (uint32_t)(uintptr_t)pthread_self();
#endif
    acuda_integrity_pass3_t p3;
    memset(&p3, 0, sizeof(p3));
    p3.driver_version = ACUDA_DRIVER_VERSION;
    p3.version = version;
    p3.current_process = pid;
    p3.current_thread = tid;
    p3.cudart_table = (const void *)g_cudart_interface;
    p3.integrity_table = (const void *)g_integrity_check;
    p3.fn_address = (const void *)(uintptr_t)integrity_check;
    p3.unix_seconds = unix_seconds;
    acuda_integrity_hash(state, &p3, sizeof(p3), 0);

    int ndev = 0;
    if (cuDeviceGetCount(&ndev) == CUDA_SUCCESS) {
        for (int d = 0; d < ndev; d++) {
            acuda_integrity_dev_t dev;
            memset(&dev, 0, sizeof(dev));
            cuDeviceGetUuid(&dev.guid, d);
            int v = 0;
            if (cuDeviceGetAttribute(&v, 50, d) == CUDA_SUCCESS)
                dev.pci_domain = v;
            v = 0;
            if (cuDeviceGetAttribute(&v, 33, d) == CUDA_SUCCESS)
                dev.pci_bus = v;
            v = 0;
            if (cuDeviceGetAttribute(&v, 34, d) == CUDA_SUCCESS)
                dev.pci_device = v;
            acuda_integrity_hash(state, &dev, sizeof(dev), 0);
        }
    }

    acuda_integrity_finish(state, result);   /* pass5 #1 */
    unsigned long long pass5_1[2] = { result[0], result[1] };
    memset(state, 0, 16);
    memset(state + 48, 0, 66 - 48);
    acuda_integrity_hash(state, pass1, sizeof(pass1), 0x5c);
    acuda_integrity_hash(state, pass5_1, sizeof(pass5_1), 0);
    acuda_integrity_finish(state, result);
    return CUDA_SUCCESS;
}

static const void *g_integrity_check[3] = {
    (const void *)(uintptr_t)(3 * sizeof(void *)),  /* [0] table size */
    (const void *)(uintptr_t)integrity_check        /* [1] */
};
// CUDA 13.4 asks for this large table during stream setup.  Only the
// size slot is needed for initialisation; unknown entries stay NULL and
// the optional tool paths that use them are not reached by our runtime.
static const void *g_cuda134_tools[64] = {
    (const void *)(uintptr_t)(64 * sizeof(void *)),
};

/* Slot 5 is the driver module handle: the 32-bit CUDA 6.5 runtime passes
 * table[5] as the HMODULE to resolve its ~140 driver entries by name.
 * NULL here fails every resolution, leaving NULL call targets (EIP=0).
 * Filled lazily with our own base; name-independent. */
static const void *g_cudart_interface[13] = {
    (const void *)(uintptr_t)(13 * sizeof(void *)),  /* [0] table size */
    (const void *)(uintptr_t)cudart_iface_module_from_cubin, /* [1] */
    (const void *)(uintptr_t)cudart_iface_primary_ctx,/* [2] primary ctx */
    NULL,                                             /* [3] */
    (const void *)(uintptr_t)cudart_iface_primary_ctx_flags, /* [4] */
    NULL,                                             /* [5] */
    (const void *)(uintptr_t)cudart_iface_module_ext1, /* [6] */
    (const void *)(uintptr_t)cudart_iface_fn7,         /* [7] */
    (const void *)(uintptr_t)cudart_iface_module_ext2, /* [8] */
    NULL,                                             /* [9] */
    NULL,                                             /* [10] */
    NULL,                                             /* [11] */
    (const void *)(uintptr_t)cudart_iface_load_compilers /* [12] */
};

/* See the slot-5 note above. */
static void ensure_cudart_iface(void) {
    if (g_cudart_interface[5])
        return;
#if defined(_WIN32)
    {
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery((LPCVOID)(uintptr_t)&cuGetExportTable, &mbi,
                         sizeof(mbi)) &&
            mbi.AllocationBase)
            g_cudart_interface[5] = mbi.AllocationBase;
    }
#else
    {
        Dl_info info;
        if (dladdr((const void *)(uintptr_t)&cuGetExportTable, &info) &&
            info.dli_fbase)
            g_cudart_interface[5] = info.dli_fbase;
    }
#endif
    ACUDA_DEBUG_PRINT("acuda-trace .. cudart_iface slot5=%p\n",
                g_cudart_interface[5]);
}

/* acuda_infoocial_cudart_interface_dd9e98a4 traced stubs (signatures unknown; log and return CUDA_SUCCESS) */
static void acuda_dump_mem(const char *tag, const void *p, int n) {
    if (!p) { ACUDA_DEBUG_PRINT( "ACuda-debug: %s NULL\n", tag); return; }
    const unsigned char *u = (const unsigned char *)p;
    ACUDA_DEBUG_PRINT( "ACuda-debug: %s %p:", tag, p);
    for (int i = 0; i < n; i++) ACUDA_DEBUG_PRINT( " %02x", u[i]);
    ACUDA_DEBUG_PRINT( "\n");
}
typedef struct { unsigned int id; void *fn; } acuda_dd9e98a4_id_entry_t;
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_000(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x0 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_001(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_002(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x2 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_003(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x3 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_004(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x4 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_005(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x5 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_006(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x6 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_007(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x7 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_008(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x8 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_009(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x9 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_00a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xa call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_00b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xb call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_00c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xc call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_00d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xd call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_00e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xe call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_00f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xf call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_010(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x10 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_011(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x11 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_012(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x12 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_013(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x13 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_014(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x14 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_015(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x15 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_016(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x16 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_017(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x17 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_018(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x18 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_019(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x19 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_01a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_01b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_01c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_01d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_01e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_01f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_020(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x20 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_021(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x21 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_022(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x22 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_023(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x23 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_024(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x24 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_025(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x25 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_026(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x26 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_027(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x27 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_028(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x28 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_029(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x29 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_02a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x2a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_02b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x2b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_02c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x2c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_02d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x2d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_02e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x2e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_02f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x2f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_030(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x30 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_031(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x31 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_032(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x32 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_033(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x33 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_034(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x34 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_035(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x35 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_036(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x36 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_037(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x37 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_038(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x38 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_039(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x39 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_03a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x3a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_03b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x3b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_03c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x3c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_03d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x3d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_03e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x3e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_03f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x3f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_040(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x40 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_041(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x41 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_042(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x42 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_043(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x43 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_044(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x44 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_045(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x45 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_046(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x46 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_047(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x47 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_048(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x48 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_049(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x49 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_04a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x4a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_04b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x4b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_04c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x4c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_04d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x4d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_04e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x4e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_04f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x4f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_050(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x50 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_051(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x51 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_052(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x52 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_053(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x53 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_054(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x54 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_055(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x55 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_056(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x56 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_057(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x57 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_058(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x58 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_059(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x59 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_05a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x5a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_05b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x5b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_05c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x5c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_05d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x5d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_05e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x5e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_05f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x5f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_060(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x60 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_061(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x61 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_062(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x62 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_063(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x63 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_064(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x64 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_065(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x65 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_066(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x66 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_067(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x67 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_068(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x68 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_069(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x69 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_06a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x6a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_06b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x6b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_06c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x6c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_06d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x6d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_06e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x6e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_06f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x6f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_070(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x70 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_071(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x71 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_072(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x72 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_073(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x73 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_074(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x74 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_075(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x75 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_076(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x76 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_077(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x77 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_078(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x78 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_079(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x79 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_07a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x7a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_07b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x7b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_07c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x7c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_07d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x7d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_07e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x7e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_07f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x7f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_080(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x80 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_081(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x81 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_082(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x82 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_083(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x83 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_084(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x84 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_085(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x85 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_086(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x86 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_087(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x87 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_088(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x88 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_089(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x89 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_08a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x8a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_08b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x8b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_08c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x8c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_08d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x8d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_08e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x8e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_08f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x8f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_090(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x90 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_091(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x91 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_092(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x92 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_093(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x93 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_094(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x94 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_095(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x95 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_096(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x96 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_097(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x97 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_098(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x98 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_099(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x99 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_09a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x9a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_09b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x9b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_09c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x9c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_09d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x9d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_09e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x9e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_09f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x9f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0a0(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xa0 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0a1(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xa1 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0a2(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xa2 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0a3(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xa3 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0a4(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xa4 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0a5(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xa5 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0a6(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xa6 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0a7(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xa7 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0a8(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xa8 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0a9(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xa9 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0aa(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xaa call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0ab(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xab call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0ac(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xac call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0ad(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xad call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0ae(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xae call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0af(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xaf call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0b0(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xb0 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0b1(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xb1 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0b2(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xb2 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0b3(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xb3 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0b4(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xb4 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0b5(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xb5 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0b6(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xb6 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0b7(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xb7 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0b8(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xb8 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0b9(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xb9 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0ba(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xba call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0bb(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xbb call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0bc(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xbc call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0bd(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xbd call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0be(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xbe call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0bf(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xbf call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0c0(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xc0 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0c1(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xc1 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0c2(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xc2 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0c3(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xc3 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0c4(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xc4 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0c5(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xc5 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0c6(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xc6 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0c7(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xc7 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0c8(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xc8 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0c9(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xc9 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0ca(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xca call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0cb(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xcb call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0cc(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xcc call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0cd(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xcd call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0ce(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xce call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0cf(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xcf call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0d0(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xd0 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0d1(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xd1 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0d2(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xd2 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0d3(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xd3 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0d4(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xd4 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0d5(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xd5 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0d6(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xd6 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0d7(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xd7 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0d8(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xd8 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0d9(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xd9 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0da(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xda call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0db(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xdb call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0dc(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xdc call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0dd(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xdd call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0de(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xde call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0df(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xdf call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0e0(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xe0 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0e1(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xe1 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0e2(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xe2 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0e3(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xe3 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0e4(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xe4 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0e5(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xe5 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0e6(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xe6 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0e7(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xe7 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0e8(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xe8 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0e9(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xe9 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0ea(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xea call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0eb(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xeb call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0ec(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xec call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0ed(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xed call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0ee(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xee call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0ef(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xef call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0f0(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xf0 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0f1(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xf1 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0f2(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xf2 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0f3(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xf3 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0f4(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xf4 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0f5(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xf5 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0f6(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xf6 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0f7(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xf7 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0f8(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xf8 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0f9(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xf9 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0fa(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xfa call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0fb(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xfb call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0fc(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xfc call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0fd(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xfd call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0fe(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xfe call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_0ff(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0xff call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_100(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x100 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_101(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x101 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_102(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x102 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_103(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x103 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_104(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x104 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_105(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x105 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_106(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x106 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_107(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x107 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_108(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x108 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_109(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x109 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_10a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x10a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_10b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x10b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_10c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x10c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_10d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x10d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_10e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x10e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_10f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x10f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_110(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x110 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_111(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x111 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_112(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x112 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_113(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x113 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_114(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x114 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_115(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x115 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_116(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x116 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_117(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x117 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_118(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x118 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_119(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x119 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_11a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x11a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_11b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x11b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_11c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x11c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_11d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x11d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_11e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x11e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_11f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x11f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_120(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x120 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_121(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x121 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_122(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x122 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_123(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x123 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_124(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x124 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_125(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x125 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_126(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x126 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_127(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x127 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_128(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x128 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_129(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x129 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_12a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x12a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_12b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x12b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_12c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x12c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_12d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x12d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_12e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x12e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_12f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x12f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_130(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x130 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_131(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x131 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_132(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x132 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_133(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x133 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_134(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x134 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_135(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x135 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_136(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x136 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_137(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x137 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_138(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x138 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_139(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x139 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_13a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x13a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_13b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x13b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_13c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x13c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_13d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x13d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_13e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x13e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_13f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x13f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_140(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x140 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_141(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x141 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_142(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x142 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_143(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x143 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_144(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x144 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_145(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x145 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_146(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x146 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_147(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x147 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_148(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x148 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_149(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x149 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_14a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x14a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_14b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x14b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_14c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x14c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_14d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x14d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_14e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x14e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_14f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x14f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_150(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x150 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_151(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x151 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_152(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x152 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_153(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x153 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_154(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x154 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_155(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x155 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_156(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x156 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_157(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x157 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_158(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x158 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_159(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x159 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_15a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x15a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_15b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x15b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_15c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x15c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_15d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x15d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_15e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x15e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_15f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x15f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_160(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x160 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_161(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x161 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_162(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x162 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_163(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x163 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_164(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x164 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_165(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x165 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_166(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x166 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_167(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x167 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_168(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x168 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_169(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x169 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_16a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x16a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_16b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x16b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_16c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x16c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_16d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x16d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_16e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x16e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_16f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x16f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_170(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x170 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_171(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x171 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_172(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x172 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_173(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x173 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_174(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x174 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_175(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x175 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_176(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x176 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_177(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x177 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_178(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x178 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_179(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x179 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_17a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x17a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_17b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x17b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_17c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x17c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_17d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x17d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_17e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x17e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_17f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x17f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_180(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x180 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_181(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x181 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_182(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x182 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_183(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x183 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_184(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x184 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_185(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x185 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_186(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x186 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_187(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x187 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_188(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x188 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_189(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x189 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_18a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x18a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_18b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x18b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_18c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x18c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_18d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x18d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_18e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x18e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_18f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x18f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_190(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x190 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_191(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x191 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_192(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x192 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_193(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x193 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_194(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x194 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_195(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x195 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_196(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x196 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_197(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x197 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_198(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x198 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_199(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x199 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_19a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x19a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_19b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x19b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_19c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x19c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_19d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x19d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_19e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x19e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_19f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x19f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1a0(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1a0 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1a1(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1a1 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1a2(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1a2 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1a3(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1a3 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1a4(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1a4 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1a5(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1a5 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1a6(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1a6 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1a7(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1a7 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1a8(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1a8 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1a9(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1a9 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1aa(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1aa call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1ab(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1ab call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1ac(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1ac call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1ad(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1ad call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1ae(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1ae call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1af(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1af call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1b0(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1b0 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1b1(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1b1 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1b2(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1b2 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1b3(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1b3 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1b4(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1b4 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1b5(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1b5 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1b6(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1b6 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1b7(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1b7 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1b8(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1b8 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1b9(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1b9 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1ba(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1ba call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1bb(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1bb call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1bc(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1bc call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1bd(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1bd call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1be(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1be call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1bf(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1bf call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1c0(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1c0 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1c1(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1c1 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1c2(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1c2 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1c3(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1c3 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1c4(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1c4 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1c5(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1c5 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1c6(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1c6 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1c7(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1c7 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1c8(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1c8 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1c9(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1c9 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1ca(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1ca call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1cb(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1cb call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1cc(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1cc call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1cd(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1cd call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1ce(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1ce call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1cf(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1cf call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1d0(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1d0 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1d1(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1d1 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1d2(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1d2 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1d3(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1d3 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1d4(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1d4 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1d5(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1d5 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1d6(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1d6 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1d7(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1d7 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1d8(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1d8 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1d9(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1d9 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1da(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1da call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1db(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1db call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1dc(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1dc call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1dd(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1dd call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1de(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1de call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1df(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1df call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1e0(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1e0 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1e1(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1e1 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1e2(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1e2 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1e3(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1e3 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1e4(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1e4 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1e5(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1e5 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1e6(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1e6 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1e7(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1e7 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1e8(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1e8 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1e9(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1e9 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1ea(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1ea call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1eb(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1eb call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1ec(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1ec call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1ed(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1ed call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1ee(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1ee call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1ef(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1ef call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1f0(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1f0 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1f1(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1f1 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1f2(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1f2 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1f3(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1f3 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1f4(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1f4 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1f5(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1f5 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1f6(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1f6 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1f7(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1f7 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1f8(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1f8 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1f9(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1f9 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1fa(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1fa call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1fb(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1fb call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1fc(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1fc call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1fd(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1fd call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1fe(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1fe call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_1ff(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1ff call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_200(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x200 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_201(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x201 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_202(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x202 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_203(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x203 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_204(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x204 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_205(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x205 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_206(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x206 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_207(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x207 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_208(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x208 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_209(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x209 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_20a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x20a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_20b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x20b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_20c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x20c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_20d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x20d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_20e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x20e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_20f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x20f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_210(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x210 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_211(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x211 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_212(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x212 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_213(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x213 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_214(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x214 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_215(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x215 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_216(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x216 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_217(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x217 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_218(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x218 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_219(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x219 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_21a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x21a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_21b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x21b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_21c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x21c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_21d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x21d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_21e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x21e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_21f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x21f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_220(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x220 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_221(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x221 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_222(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x222 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_223(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x223 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_224(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x224 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_225(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x225 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_226(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x226 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_227(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x227 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_228(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x228 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_229(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x229 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_22a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x22a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_22b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x22b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_22c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x22c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_22d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x22d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_22e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x22e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_22f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x22f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_230(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x230 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_231(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x231 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_232(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x232 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_233(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x233 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_234(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x234 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_235(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x235 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_236(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x236 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_237(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x237 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_238(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x238 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_239(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x239 call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_23a(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x23a call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_23b(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x23b call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_23c(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x23c call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_23d(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x23d call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_23e(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x23e call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_idstub_23f(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0; int i2 = n++;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x23f call[%d] a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", i2,a,b,c,d,e,f,g,h);
    return CUDA_SUCCESS;
}
/* --- dd9e98a4 interface functions mapped to CUDA runtime API entry points ---
 * The CUDA 13.x hybrid runtime resolves these IDs through the dispatcher and
 * tail-calls them from its cuda* wrapper functions (see docs/unknown notes).
 * id 0x3 is cudaGetDeviceCount(). */
extern CUresult ACUDA_CUDAAPI cuDeviceGetCount(int *count);
extern CUresult ACUDA_CUDAAPI cuInit(unsigned int flags);

/* The CUDA 13.x hybrid runtime never calls cuInit() or cuCtxCreate(): it
 * expects the driver interface to be usable right away, so bring up the
 * primary context before serving interface requests. */
static int acuda_dd9e98a4_driver_ready(void) {
    static int state = 0; /* 0 unknown, 1 ok, -1 failed */
    if (state == 0) {
        CUresult r = cuInit(0);
        if (r == CUDA_SUCCESS) {
            CUcontext ctx = NULL;
            r = cuDevicePrimaryCtxRetain(&ctx, 0);
            if (r == CUDA_SUCCESS)
                r = cuCtxSetCurrent(ctx);
        }
        ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 implicit init -> %d\n", (int)r);
        state = (r == CUDA_SUCCESS) ? 1 : -1;
    }
    return state == 1;
}

static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_count(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    CUresult r;
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    r = cuDeviceGetCount((int *)a);
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x3 cudaGetDeviceCount -> %d (count=%d)\n",
            (int)r, a ? *(int *)a : -1);
    return r;
}

/* id 0xc8: cudaDeviceGetAttribute(int *value, int attr, int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_attribute(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuDeviceGetAttribute((int *)a, (int)(intptr_t)b, (CUdevice)(intptr_t)c);
}

/* id 0x14: cudaMalloc(void **devPtr, size_t size) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_malloc(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuMemAlloc((CUdeviceptr *)a, (size_t)b);
}

/* id 0x19: cudaMallocHost(void **ptr, size_t size) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_malloc_host(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    size_t size = (size_t)b;
    (void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!a)
        return CUDA_ERROR_INVALID_VALUE;
    *(void **)a = NULL;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    *(void **)a = malloc(size ? size : 1);
    return *(void **)a ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

/* id 0x10: cudaSetDevice(int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_set_device(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    CUcontext ctx = NULL;
    CUresult r;
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    r = cuDevicePrimaryCtxRetain(&ctx, (CUdevice)(intptr_t)a);
    if (r != CUDA_SUCCESS)
        return r;
    return cuCtxSetCurrent(ctx);
}

/* id 0x1b8: cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_device_properties(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    struct cudaDeviceProp *prop = (struct cudaDeviceProp *)a;
    int device = (int)(intptr_t)b;
    char name[256] = { 0 };
    int major = 0, minor = 0;
    size_t total = 0;
    (void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!prop)
        return CUDA_ERROR_INVALID_VALUE;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (cuDeviceGetName(name, (int)sizeof(name), (CUdevice)device) != CUDA_SUCCESS)
        return CUDA_ERROR_INVALID_DEVICE;
    cuDeviceComputeCapability(&major, &minor, (CUdevice)device);
    cuDeviceTotalMem(&total, (CUdevice)device);
    prop->name[0] = 0;
    {
        size_t i;
        for (i = 0; i + 1 < sizeof(prop->name) && name[i]; i++)
            prop->name[i] = name[i];
        prop->name[i] = 0;
    }
    prop->major = major;
    prop->minor = minor;
    prop->totalGlobalMem = total;
    prop->sharedMemPerBlock = 65536;
    prop->regsPerBlock = 65536;
    prop->warpSize = 32;
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->clockRate = 2000000;
    prop->totalConstMem = 65536;
    prop->textureAlignment = 512;
    prop->multiProcessorCount = 16;
    prop->computeMode = 0;
    prop->concurrentKernels = 1;
    prop->unifiedAddressing = 1;
    prop->memoryClockRate = 1600000;
    prop->memoryBusWidth = 256;
    prop->l2CacheSize = 4 << 20;
    prop->maxThreadsPerMultiProcessor = 2048;
    prop->sharedMemPerMultiprocessor = 65536;
    prop->regsPerMultiprocessor = 65536;
    prop->sharedMemPerBlockOptin = 65536;
    prop->maxBlocksPerMultiProcessor = 32;
    prop->pageableMemoryAccess = 1;
    prop->hostRegisterSupported = 1;
    prop->memoryPoolsSupported = 1;
    prop->clusterLaunch = 0;
    prop->unifiedFunctionPointers = 0;
    prop->deviceOverlap = 1;
    prop->canMapHostMemory = 1;
    prop->asyncEngineCount = 1;
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 id=0x1b8 cudaGetDeviceProperties(%d) name=%s cc=%d.%d\n",
            device, prop->name, major, minor);
    return CUDA_SUCCESS;
}

/* id 0xd1: cudaGetErrorString(cudaError_t error) -> const char * */
static const char *acuda_dd9e98a4_error_string(int e) {
    switch (e) {
    case 0: return "no error";
    case 1: return "invalid argument";
    case 2: return "out of memory";
    case 3: return "initialization error";
    case 4: return "cudaErrorCudartUnloading";
    case 100: return "no CUDA-capable device is detected";
    case 101: return "invalid device ordinal";
    case 200: return "invalid device function";
    case 209: return "invalid configuration argument";
    case 400: return "invalid resource handle";
    case 500: return "named symbol not found";
    case 700: return "an illegal memory access was encountered";
    case 719: return "unspecified launch failure";
    case 801: return "operation not supported";
    default: return "unknown error";
    }
}

static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_error_string(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return (CUresult)(uintptr_t)acuda_dd9e98a4_error_string((int)(intptr_t)a);
}

/* --- memory, events, synchronisation, profiler ------------------------- */

/* id 0x31: cudaMemset(void *devPtr, int value, size_t count) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_memset(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 memset(%p,%d,%zu)\n", a, (int)(intptr_t)b, (size_t)c);
    (void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!a)
        return CUDA_ERROR_INVALID_VALUE;
    if (!c)
        return CUDA_SUCCESS;
    return cuMemsetD8((CUdeviceptr)(uintptr_t)a, (unsigned char)(uintptr_t)b,
                      (size_t)c);
}

/* id 0x29: cudaMemcpyAsync(void *dst, const void *src, size_t count,
 *                          cudaMemcpyKind kind, cudaStream_t stream)
 * ACuda's device memory is directly addressable host memory, so the async
 * copies are executed synchronously, exactly like ACuda's cudart does. */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_memcpy_async(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    int kind = (int)(intptr_t)d;
    size_t count = (size_t)c;
    (void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if ((!a || !b) && count)
        return CUDA_ERROR_INVALID_VALUE;
    if (!count)
        return CUDA_SUCCESS;
    switch (kind) {
    case 0: /* cudaMemcpyHostToHost */
        memmove(a, b, count);
        return CUDA_SUCCESS;
    case 1: /* cudaMemcpyHostToDevice */
        return cuMemcpyHtoD((CUdeviceptr)(uintptr_t)a, b, count);
    case 2: /* cudaMemcpyDeviceToHost */
        return cuMemcpyDtoH(a, (CUdeviceptr)(uintptr_t)b, count);
    case 3: /* cudaMemcpyDeviceToDevice */
        return cuMemcpyDtoD((CUdeviceptr)(uintptr_t)a, (CUdeviceptr)(uintptr_t)b,
                            count);
    default:
        return CUDA_ERROR_INVALID_VALUE;
    }
}

/* id 0x16: cudaFree(void *devPtr) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_free(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!a)
        return CUDA_SUCCESS;
    return cuMemFree((CUdeviceptr)(uintptr_t)a);
}

/* id 0x1a: cudaFreeHost(void *ptr) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_free_host(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    free(a);
    return CUDA_SUCCESS;
}

/* id 0xa5: cudaDeviceSynchronize(void) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_synchronize(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuCtxSynchronize();
}

/* id 0x85: cudaEventCreate(cudaEvent_t *event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_create(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventCreate((CUevent *)a, 0);
}

/* id 0x87: cudaEventRecord(cudaEvent_t event, cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_record(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventRecord((CUevent)a, (CUstream)b);
}

/* id 0x8a: cudaEventQuery(cudaEvent_t event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_query(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventQuery((CUevent)a);
}

/* id 0x1e6: cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_elapsed_time(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventElapsedTime((float *)a, (CUevent)b, (CUevent)c);
}

/* id 0x88: cudaEventDestroy(cudaEvent_t event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_destroy(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventDestroy((CUevent)a);
}

/* id 0xab / 0xac: cudaProfilerStart() / cudaProfilerStop() */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_profiler_noop(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    return CUDA_SUCCESS;
}

/* --- kernel launch -----------------------------------------------------
 * The hybrid runtime reports the executable's fatbin once at startup and
 * then every __global__ host stub that gets launched.  Launches arrive with
 * the host stub address as the function key, which is exactly what ACuda's
 * cudart does for statically linked executables. */

typedef struct {
    unsigned int x, y, z;
} acuda_dd9e98a4_dim3;

#define ACUDA_DD9E98A4_MAX_FATBINS 8
#define ACUDA_DD9E98A4_MAX_FUNCS 256

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

static acuda_dd9e98a4_fatbin g_dd9e98a4_fatbins[ACUDA_DD9E98A4_MAX_FATBINS];
static unsigned g_dd9e98a4_nfatbins;
static acuda_dd9e98a4_kernel g_dd9e98a4_kernels[ACUDA_DD9E98A4_MAX_FUNCS];
static unsigned g_dd9e98a4_nkernels;

/* id 0x1ee: __cudaRegisterFatBinary(void *fatCubin) -> handle */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_register_fat_binary(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    unsigned i;
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!a)
        return CUDA_ERROR_INVALID_VALUE;
    /* The hybrid runtime keeps the returned handle in a 32-bit field, so a
     * pointer cannot be handed back; use a 1-based index instead. */
    for (i = 0; i < g_dd9e98a4_nfatbins; i++) {
        if (g_dd9e98a4_fatbins[i].blob == a)
            return (CUresult)(i + 1);
    }
    if (g_dd9e98a4_nfatbins >= ACUDA_DD9E98A4_MAX_FATBINS)
        return CUDA_ERROR_OUT_OF_MEMORY;
    i = g_dd9e98a4_nfatbins++;
    g_dd9e98a4_fatbins[i].blob = a;
    g_dd9e98a4_fatbins[i].module = NULL;
    g_dd9e98a4_fatbins[i].state = 0;
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 register fatbin %p -> %u\n", a,
                      i + 1);
    return (CUresult)(i + 1);
}

/* id 0x1ef: __cudaRegisterFatBinaryEnd(void **handle) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_register_fat_binary_end(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    return CUDA_SUCCESS;
}

/* id 0x1f3: __cudaRegisterFunction(handle, hostFun, deviceFun, deviceName,
 *                                  threadLimit, tid, bid, bDim, gDim, wSize) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_register_function(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    acuda_dd9e98a4_kernel *k;
    (void)c;(void)e;(void)f;(void)g;(void)h;
    if (!a || !b || !d)
        return CUDA_ERROR_INVALID_VALUE;
    if (g_dd9e98a4_nkernels >= ACUDA_DD9E98A4_MAX_FUNCS)
        return CUDA_ERROR_OUT_OF_MEMORY;
    k = &g_dd9e98a4_kernels[g_dd9e98a4_nkernels++];
    k->host_fn = b;
    k->device_name = (const char *)d;
    k->fatbin = NULL;
    {
        unsigned h = (unsigned)(uintptr_t)a;
        if (h >= 1 && h <= g_dd9e98a4_nfatbins)
            k->fatbin = &g_dd9e98a4_fatbins[h - 1];
        else if (g_dd9e98a4_nfatbins)
            k->fatbin = &g_dd9e98a4_fatbins[g_dd9e98a4_nfatbins - 1];
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 register kernel %p -> %s\n", b,
                      k->device_name);
    return CUDA_SUCCESS;
}

/* id 0x1f8: __cudaGetKernel(void **kernel, const void *func) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_kernel(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 get kernel slot=%p func=%p\n", a, b);
    (void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!a || !b)
        return CUDA_ERROR_INVALID_VALUE;
    *(const void **)a = b;
    return CUDA_SUCCESS;
}

/* Pushed <<<grid, block, shared, stream>>> configuration. */
typedef struct {
    acuda_dd9e98a4_dim3 grid;
    acuda_dd9e98a4_dim3 block;
    size_t shared;
    void *stream;
} acuda_dd9e98a4_launch_config;

#if defined(_MSC_VER)
#define ACUDA_DD9E98A4_TLS __declspec(thread)
#else
#define ACUDA_DD9E98A4_TLS __thread
#endif

static ACUDA_DD9E98A4_TLS acuda_dd9e98a4_launch_config
    g_dd9e98a4_configs[16];
static ACUDA_DD9E98A4_TLS unsigned g_dd9e98a4_config_depth;

/* id 0x1f6: __cudaPushCallConfiguration(dim3 *grid, dim3 *block,
 *                                       size_t sharedMem, void *stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_push_config(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    acuda_dd9e98a4_launch_config *cfg;
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 push config\n");
    (void)e;(void)f;(void)g;(void)h;
    if (!a || !b || g_dd9e98a4_config_depth >= 16)
        return CUDA_ERROR_INVALID_VALUE;
    cfg = &g_dd9e98a4_configs[g_dd9e98a4_config_depth++];
    cfg->grid = *(const acuda_dd9e98a4_dim3 *)a;
    cfg->block = *(const acuda_dd9e98a4_dim3 *)b;
    cfg->shared = (size_t)c;
    cfg->stream = d;
    return CUDA_SUCCESS;
}

/* id 0x1f7: __cudaPopCallConfiguration(dim3 *grid, dim3 *block,
 *                                      size_t *sharedMem, void **stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_pop_config(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    acuda_dd9e98a4_launch_config *cfg;
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 pop config\n");
    (void)e;(void)f;(void)g;(void)h;
    if (g_dd9e98a4_config_depth == 0)
        return CUDA_ERROR_INVALID_VALUE;
    cfg = &g_dd9e98a4_configs[--g_dd9e98a4_config_depth];
    if (a)
        *(acuda_dd9e98a4_dim3 *)a = cfg->grid;
    if (b)
        *(acuda_dd9e98a4_dim3 *)b = cfg->block;
    if (c)
        *(size_t *)c = cfg->shared;
    if (d)
        *(void **)d = cfg->stream;
    return CUDA_SUCCESS;
}

/* id 0x1f9: cudaLaunchKernel(const void *func, dim3 *grid, dim3 *block,
 *                            void **args, size_t sharedMem, void *stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_launch_kernel(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    const acuda_dd9e98a4_dim3 *grid = (const acuda_dd9e98a4_dim3 *)b;
    const acuda_dd9e98a4_dim3 *block = (const acuda_dd9e98a4_dim3 *)c;
    acuda_dd9e98a4_kernel *k = NULL;
    acuda_dd9e98a4_fatbin *fb;
    CUfunction fn = NULL;
    CUresult r;
    unsigned i;
    (void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!a || !grid || !block)
        return CUDA_ERROR_INVALID_VALUE;
    for (i = 0; i < g_dd9e98a4_nkernels; i++) {
        if (g_dd9e98a4_kernels[i].host_fn == a) {
            k = &g_dd9e98a4_kernels[i];
            break;
        }
    }
    if (!k || !k->fatbin)
        return CUDA_ERROR_NOT_FOUND;
    fb = k->fatbin;
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 launch %s grid=(%u,%u,%u) block=(%u,%u,%u)\n",
                      k->device_name, grid->x, grid->y, grid->z, block->x,
                      block->y, block->z);
    if (fb->state == 0) {
        r = cuModuleLoadFatBinary(&fb->module, fb->blob);
        fb->state = (r == CUDA_SUCCESS) ? 1 : -1;
        ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 module load -> %d\n", (int)r);
    }
    if (fb->state != 1)
        return CUDA_ERROR_INVALID_IMAGE;
    r = cuModuleGetFunction(&fn, fb->module, k->device_name);
    if (r != CUDA_SUCCESS)
        return r;
    return cuLaunchKernel(fn, grid->x, grid->y, grid->z, block->x, block->y,
                          block->z, (unsigned int)(size_t)e, (CUstream)f, d,
                          NULL);
}

static void *acuda_dd9e98a4_get_id_stub(unsigned int id) {
    switch(id){
    case 0x0: return (void*)&acuda_dd9e98a4_idstub_000;
    case 0x10: return (void*)&acuda_dd9e98a4_api_set_device;
    case 0x14: return (void*)&acuda_dd9e98a4_api_malloc;
    case 0x19: return (void*)&acuda_dd9e98a4_api_malloc_host;
    case 0x1b8: return (void*)&acuda_dd9e98a4_api_get_device_properties;
    case 0xc8: return (void*)&acuda_dd9e98a4_api_device_get_attribute;
    case 0xd1: return (void*)&acuda_dd9e98a4_api_get_error_string;
    case 0x3: return (void*)&acuda_dd9e98a4_api_device_get_count;
    case 0x16: return (void*)&acuda_dd9e98a4_api_free;
    case 0x1a: return (void*)&acuda_dd9e98a4_api_free_host;
    case 0x29: return (void*)&acuda_dd9e98a4_api_memcpy_async;
    case 0x31: return (void*)&acuda_dd9e98a4_api_memset;
    case 0x85: return (void*)&acuda_dd9e98a4_api_event_create;
    case 0x87: return (void*)&acuda_dd9e98a4_api_event_record;
    case 0x88: return (void*)&acuda_dd9e98a4_api_event_destroy;
    case 0x8a: return (void*)&acuda_dd9e98a4_api_event_query;
    case 0xa5: return (void*)&acuda_dd9e98a4_api_device_synchronize;
    case 0xab: return (void*)&acuda_dd9e98a4_api_profiler_noop;
    case 0xac: return (void*)&acuda_dd9e98a4_api_profiler_noop;
    case 0x1e6: return (void*)&acuda_dd9e98a4_api_event_elapsed_time;
    case 0x1ee: return (void*)&acuda_dd9e98a4_api_register_fat_binary;
    case 0x1ef: return (void*)&acuda_dd9e98a4_api_register_fat_binary_end;
    case 0x1f3: return (void*)&acuda_dd9e98a4_api_register_function;
    case 0x1f6: return (void*)&acuda_dd9e98a4_api_push_config;
    case 0x1f7: return (void*)&acuda_dd9e98a4_api_pop_config;
    case 0x1f8: return (void*)&acuda_dd9e98a4_api_get_kernel;
    case 0x1f9: return (void*)&acuda_dd9e98a4_api_launch_kernel;
    case 0x1: return (void*)&acuda_dd9e98a4_idstub_001;
    case 0x2: return (void*)&acuda_dd9e98a4_idstub_002;
    case 0x4: return (void*)&acuda_dd9e98a4_idstub_004;
    case 0x5: return (void*)&acuda_dd9e98a4_idstub_005;
    case 0x6: return (void*)&acuda_dd9e98a4_idstub_006;
    case 0x7: return (void*)&acuda_dd9e98a4_idstub_007;
    case 0x8: return (void*)&acuda_dd9e98a4_idstub_008;
    case 0x9: return (void*)&acuda_dd9e98a4_idstub_009;
    case 0xa: return (void*)&acuda_dd9e98a4_idstub_00a;
    case 0xb: return (void*)&acuda_dd9e98a4_idstub_00b;
    case 0xc: return (void*)&acuda_dd9e98a4_idstub_00c;
    case 0xd: return (void*)&acuda_dd9e98a4_idstub_00d;
    case 0xe: return (void*)&acuda_dd9e98a4_idstub_00e;
    case 0xf: return (void*)&acuda_dd9e98a4_idstub_00f;
    case 0x11: return (void*)&acuda_dd9e98a4_idstub_011;
    case 0x12: return (void*)&acuda_dd9e98a4_idstub_012;
    case 0x13: return (void*)&acuda_dd9e98a4_idstub_013;
    case 0x15: return (void*)&acuda_dd9e98a4_idstub_015;
    case 0x17: return (void*)&acuda_dd9e98a4_idstub_017;
    case 0x18: return (void*)&acuda_dd9e98a4_idstub_018;
    case 0x1b: return (void*)&acuda_dd9e98a4_idstub_01b;
    case 0x1c: return (void*)&acuda_dd9e98a4_idstub_01c;
    case 0x1d: return (void*)&acuda_dd9e98a4_idstub_01d;
    case 0x1e: return (void*)&acuda_dd9e98a4_idstub_01e;
    case 0x1f: return (void*)&acuda_dd9e98a4_idstub_01f;
    case 0x20: return (void*)&acuda_dd9e98a4_idstub_020;
    case 0x21: return (void*)&acuda_dd9e98a4_idstub_021;
    case 0x22: return (void*)&acuda_dd9e98a4_idstub_022;
    case 0x23: return (void*)&acuda_dd9e98a4_idstub_023;
    case 0x24: return (void*)&acuda_dd9e98a4_idstub_024;
    case 0x25: return (void*)&acuda_dd9e98a4_idstub_025;
    case 0x26: return (void*)&acuda_dd9e98a4_idstub_026;
    case 0x27: return (void*)&acuda_dd9e98a4_idstub_027;
    case 0x28: return (void*)&acuda_dd9e98a4_idstub_028;
    case 0x2a: return (void*)&acuda_dd9e98a4_idstub_02a;
    case 0x2b: return (void*)&acuda_dd9e98a4_idstub_02b;
    case 0x2c: return (void*)&acuda_dd9e98a4_idstub_02c;
    case 0x2d: return (void*)&acuda_dd9e98a4_idstub_02d;
    case 0x2e: return (void*)&acuda_dd9e98a4_idstub_02e;
    case 0x2f: return (void*)&acuda_dd9e98a4_idstub_02f;
    case 0x30: return (void*)&acuda_dd9e98a4_idstub_030;
    case 0x32: return (void*)&acuda_dd9e98a4_idstub_032;
    case 0x33: return (void*)&acuda_dd9e98a4_idstub_033;
    case 0x34: return (void*)&acuda_dd9e98a4_idstub_034;
    case 0x35: return (void*)&acuda_dd9e98a4_idstub_035;
    case 0x36: return (void*)&acuda_dd9e98a4_idstub_036;
    case 0x37: return (void*)&acuda_dd9e98a4_idstub_037;
    case 0x38: return (void*)&acuda_dd9e98a4_idstub_038;
    case 0x39: return (void*)&acuda_dd9e98a4_idstub_039;
    case 0x3a: return (void*)&acuda_dd9e98a4_idstub_03a;
    case 0x3b: return (void*)&acuda_dd9e98a4_idstub_03b;
    case 0x3c: return (void*)&acuda_dd9e98a4_idstub_03c;
    case 0x3d: return (void*)&acuda_dd9e98a4_idstub_03d;
    case 0x3e: return (void*)&acuda_dd9e98a4_idstub_03e;
    case 0x3f: return (void*)&acuda_dd9e98a4_idstub_03f;
    case 0x40: return (void*)&acuda_dd9e98a4_idstub_040;
    case 0x41: return (void*)&acuda_dd9e98a4_idstub_041;
    case 0x42: return (void*)&acuda_dd9e98a4_idstub_042;
    case 0x43: return (void*)&acuda_dd9e98a4_idstub_043;
    case 0x44: return (void*)&acuda_dd9e98a4_idstub_044;
    case 0x45: return (void*)&acuda_dd9e98a4_idstub_045;
    case 0x46: return (void*)&acuda_dd9e98a4_idstub_046;
    case 0x47: return (void*)&acuda_dd9e98a4_idstub_047;
    case 0x48: return (void*)&acuda_dd9e98a4_idstub_048;
    case 0x49: return (void*)&acuda_dd9e98a4_idstub_049;
    case 0x4a: return (void*)&acuda_dd9e98a4_idstub_04a;
    case 0x4b: return (void*)&acuda_dd9e98a4_idstub_04b;
    case 0x4c: return (void*)&acuda_dd9e98a4_idstub_04c;
    case 0x4d: return (void*)&acuda_dd9e98a4_idstub_04d;
    case 0x4e: return (void*)&acuda_dd9e98a4_idstub_04e;
    case 0x4f: return (void*)&acuda_dd9e98a4_idstub_04f;
    case 0x50: return (void*)&acuda_dd9e98a4_idstub_050;
    case 0x51: return (void*)&acuda_dd9e98a4_idstub_051;
    case 0x52: return (void*)&acuda_dd9e98a4_idstub_052;
    case 0x53: return (void*)&acuda_dd9e98a4_idstub_053;
    case 0x54: return (void*)&acuda_dd9e98a4_idstub_054;
    case 0x55: return (void*)&acuda_dd9e98a4_idstub_055;
    case 0x56: return (void*)&acuda_dd9e98a4_idstub_056;
    case 0x57: return (void*)&acuda_dd9e98a4_idstub_057;
    case 0x58: return (void*)&acuda_dd9e98a4_idstub_058;
    case 0x59: return (void*)&acuda_dd9e98a4_idstub_059;
    case 0x5a: return (void*)&acuda_dd9e98a4_idstub_05a;
    case 0x5b: return (void*)&acuda_dd9e98a4_idstub_05b;
    case 0x5c: return (void*)&acuda_dd9e98a4_idstub_05c;
    case 0x5d: return (void*)&acuda_dd9e98a4_idstub_05d;
    case 0x5e: return (void*)&acuda_dd9e98a4_idstub_05e;
    case 0x5f: return (void*)&acuda_dd9e98a4_idstub_05f;
    case 0x60: return (void*)&acuda_dd9e98a4_idstub_060;
    case 0x61: return (void*)&acuda_dd9e98a4_idstub_061;
    case 0x62: return (void*)&acuda_dd9e98a4_idstub_062;
    case 0x63: return (void*)&acuda_dd9e98a4_idstub_063;
    case 0x64: return (void*)&acuda_dd9e98a4_idstub_064;
    case 0x65: return (void*)&acuda_dd9e98a4_idstub_065;
    case 0x66: return (void*)&acuda_dd9e98a4_idstub_066;
    case 0x67: return (void*)&acuda_dd9e98a4_idstub_067;
    case 0x68: return (void*)&acuda_dd9e98a4_idstub_068;
    case 0x69: return (void*)&acuda_dd9e98a4_idstub_069;
    case 0x6a: return (void*)&acuda_dd9e98a4_idstub_06a;
    case 0x6b: return (void*)&acuda_dd9e98a4_idstub_06b;
    case 0x6c: return (void*)&acuda_dd9e98a4_idstub_06c;
    case 0x6d: return (void*)&acuda_dd9e98a4_idstub_06d;
    case 0x6e: return (void*)&acuda_dd9e98a4_idstub_06e;
    case 0x6f: return (void*)&acuda_dd9e98a4_idstub_06f;
    case 0x70: return (void*)&acuda_dd9e98a4_idstub_070;
    case 0x71: return (void*)&acuda_dd9e98a4_idstub_071;
    case 0x72: return (void*)&acuda_dd9e98a4_idstub_072;
    case 0x73: return (void*)&acuda_dd9e98a4_idstub_073;
    case 0x74: return (void*)&acuda_dd9e98a4_idstub_074;
    case 0x75: return (void*)&acuda_dd9e98a4_idstub_075;
    case 0x76: return (void*)&acuda_dd9e98a4_idstub_076;
    case 0x77: return (void*)&acuda_dd9e98a4_idstub_077;
    case 0x78: return (void*)&acuda_dd9e98a4_idstub_078;
    case 0x79: return (void*)&acuda_dd9e98a4_idstub_079;
    case 0x7a: return (void*)&acuda_dd9e98a4_idstub_07a;
    case 0x7b: return (void*)&acuda_dd9e98a4_idstub_07b;
    case 0x7c: return (void*)&acuda_dd9e98a4_idstub_07c;
    case 0x7d: return (void*)&acuda_dd9e98a4_idstub_07d;
    case 0x7e: return (void*)&acuda_dd9e98a4_idstub_07e;
    case 0x7f: return (void*)&acuda_dd9e98a4_idstub_07f;
    case 0x80: return (void*)&acuda_dd9e98a4_idstub_080;
    case 0x81: return (void*)&acuda_dd9e98a4_idstub_081;
    case 0x82: return (void*)&acuda_dd9e98a4_idstub_082;
    case 0x83: return (void*)&acuda_dd9e98a4_idstub_083;
    case 0x84: return (void*)&acuda_dd9e98a4_idstub_084;
    case 0x86: return (void*)&acuda_dd9e98a4_idstub_086;
    case 0x89: return (void*)&acuda_dd9e98a4_idstub_089;
    case 0x8b: return (void*)&acuda_dd9e98a4_idstub_08b;
    case 0x8c: return (void*)&acuda_dd9e98a4_idstub_08c;
    case 0x8d: return (void*)&acuda_dd9e98a4_idstub_08d;
    case 0x8e: return (void*)&acuda_dd9e98a4_idstub_08e;
    case 0x8f: return (void*)&acuda_dd9e98a4_idstub_08f;
    case 0x90: return (void*)&acuda_dd9e98a4_idstub_090;
    case 0x91: return (void*)&acuda_dd9e98a4_idstub_091;
    case 0x92: return (void*)&acuda_dd9e98a4_idstub_092;
    case 0x93: return (void*)&acuda_dd9e98a4_idstub_093;
    case 0x94: return (void*)&acuda_dd9e98a4_idstub_094;
    case 0x95: return (void*)&acuda_dd9e98a4_idstub_095;
    case 0x96: return (void*)&acuda_dd9e98a4_idstub_096;
    case 0x97: return (void*)&acuda_dd9e98a4_idstub_097;
    case 0x98: return (void*)&acuda_dd9e98a4_idstub_098;
    case 0x99: return (void*)&acuda_dd9e98a4_idstub_099;
    case 0x9a: return (void*)&acuda_dd9e98a4_idstub_09a;
    case 0x9b: return (void*)&acuda_dd9e98a4_idstub_09b;
    case 0x9c: return (void*)&acuda_dd9e98a4_idstub_09c;
    case 0x9d: return (void*)&acuda_dd9e98a4_idstub_09d;
    case 0x9e: return (void*)&acuda_dd9e98a4_idstub_09e;
    case 0x9f: return (void*)&acuda_dd9e98a4_idstub_09f;
    case 0xa0: return (void*)&acuda_dd9e98a4_idstub_0a0;
    case 0xa1: return (void*)&acuda_dd9e98a4_idstub_0a1;
    case 0xa2: return (void*)&acuda_dd9e98a4_idstub_0a2;
    case 0xa3: return (void*)&acuda_dd9e98a4_idstub_0a3;
    case 0xa4: return (void*)&acuda_dd9e98a4_idstub_0a4;
    case 0xa6: return (void*)&acuda_dd9e98a4_idstub_0a6;
    case 0xa7: return (void*)&acuda_dd9e98a4_idstub_0a7;
    case 0xa8: return (void*)&acuda_dd9e98a4_idstub_0a8;
    case 0xa9: return (void*)&acuda_dd9e98a4_idstub_0a9;
    case 0xaa: return (void*)&acuda_dd9e98a4_idstub_0aa;
    case 0xad: return (void*)&acuda_dd9e98a4_idstub_0ad;
    case 0xae: return (void*)&acuda_dd9e98a4_idstub_0ae;
    case 0xaf: return (void*)&acuda_dd9e98a4_idstub_0af;
    case 0xb0: return (void*)&acuda_dd9e98a4_idstub_0b0;
    case 0xb1: return (void*)&acuda_dd9e98a4_idstub_0b1;
    case 0xb2: return (void*)&acuda_dd9e98a4_idstub_0b2;
    case 0xb3: return (void*)&acuda_dd9e98a4_idstub_0b3;
    case 0xb4: return (void*)&acuda_dd9e98a4_idstub_0b4;
    case 0xb5: return (void*)&acuda_dd9e98a4_idstub_0b5;
    case 0xb6: return (void*)&acuda_dd9e98a4_idstub_0b6;
    case 0xb7: return (void*)&acuda_dd9e98a4_idstub_0b7;
    case 0xb8: return (void*)&acuda_dd9e98a4_idstub_0b8;
    case 0xb9: return (void*)&acuda_dd9e98a4_idstub_0b9;
    case 0xba: return (void*)&acuda_dd9e98a4_idstub_0ba;
    case 0xbb: return (void*)&acuda_dd9e98a4_idstub_0bb;
    case 0xbc: return (void*)&acuda_dd9e98a4_idstub_0bc;
    case 0xbd: return (void*)&acuda_dd9e98a4_idstub_0bd;
    case 0xbe: return (void*)&acuda_dd9e98a4_idstub_0be;
    case 0xbf: return (void*)&acuda_dd9e98a4_idstub_0bf;
    case 0xc0: return (void*)&acuda_dd9e98a4_idstub_0c0;
    case 0xc1: return (void*)&acuda_dd9e98a4_idstub_0c1;
    case 0xc2: return (void*)&acuda_dd9e98a4_idstub_0c2;
    case 0xc3: return (void*)&acuda_dd9e98a4_idstub_0c3;
    case 0xc4: return (void*)&acuda_dd9e98a4_idstub_0c4;
    case 0xc5: return (void*)&acuda_dd9e98a4_idstub_0c5;
    case 0xc6: return (void*)&acuda_dd9e98a4_idstub_0c6;
    case 0xc7: return (void*)&acuda_dd9e98a4_idstub_0c7;
    case 0xc9: return (void*)&acuda_dd9e98a4_idstub_0c9;
    case 0xca: return (void*)&acuda_dd9e98a4_idstub_0ca;
    case 0xcb: return (void*)&acuda_dd9e98a4_idstub_0cb;
    case 0xcc: return (void*)&acuda_dd9e98a4_idstub_0cc;
    case 0xcd: return (void*)&acuda_dd9e98a4_idstub_0cd;
    case 0xce: return (void*)&acuda_dd9e98a4_idstub_0ce;
    case 0xcf: return (void*)&acuda_dd9e98a4_idstub_0cf;
    case 0xd0: return (void*)&acuda_dd9e98a4_idstub_0d0;
    case 0xd2: return (void*)&acuda_dd9e98a4_idstub_0d2;
    case 0xd3: return (void*)&acuda_dd9e98a4_idstub_0d3;
    case 0xd4: return (void*)&acuda_dd9e98a4_idstub_0d4;
    case 0xd5: return (void*)&acuda_dd9e98a4_idstub_0d5;
    case 0xd6: return (void*)&acuda_dd9e98a4_idstub_0d6;
    case 0xd7: return (void*)&acuda_dd9e98a4_idstub_0d7;
    case 0xd8: return (void*)&acuda_dd9e98a4_idstub_0d8;
    case 0xd9: return (void*)&acuda_dd9e98a4_idstub_0d9;
    case 0xda: return (void*)&acuda_dd9e98a4_idstub_0da;
    case 0xdb: return (void*)&acuda_dd9e98a4_idstub_0db;
    case 0xdc: return (void*)&acuda_dd9e98a4_idstub_0dc;
    case 0xdd: return (void*)&acuda_dd9e98a4_idstub_0dd;
    case 0xde: return (void*)&acuda_dd9e98a4_idstub_0de;
    case 0xdf: return (void*)&acuda_dd9e98a4_idstub_0df;
    case 0xe0: return (void*)&acuda_dd9e98a4_idstub_0e0;
    case 0xe1: return (void*)&acuda_dd9e98a4_idstub_0e1;
    case 0xe2: return (void*)&acuda_dd9e98a4_idstub_0e2;
    case 0xe3: return (void*)&acuda_dd9e98a4_idstub_0e3;
    case 0xe4: return (void*)&acuda_dd9e98a4_idstub_0e4;
    case 0xe5: return (void*)&acuda_dd9e98a4_idstub_0e5;
    case 0xe6: return (void*)&acuda_dd9e98a4_idstub_0e6;
    case 0xe7: return (void*)&acuda_dd9e98a4_idstub_0e7;
    case 0xe8: return (void*)&acuda_dd9e98a4_idstub_0e8;
    case 0xe9: return (void*)&acuda_dd9e98a4_idstub_0e9;
    case 0xea: return (void*)&acuda_dd9e98a4_idstub_0ea;
    case 0xeb: return (void*)&acuda_dd9e98a4_idstub_0eb;
    case 0xec: return (void*)&acuda_dd9e98a4_idstub_0ec;
    case 0xed: return (void*)&acuda_dd9e98a4_idstub_0ed;
    case 0xee: return (void*)&acuda_dd9e98a4_idstub_0ee;
    case 0xef: return (void*)&acuda_dd9e98a4_idstub_0ef;
    case 0xf0: return (void*)&acuda_dd9e98a4_idstub_0f0;
    case 0xf1: return (void*)&acuda_dd9e98a4_idstub_0f1;
    case 0xf2: return (void*)&acuda_dd9e98a4_idstub_0f2;
    case 0xf3: return (void*)&acuda_dd9e98a4_idstub_0f3;
    case 0xf4: return (void*)&acuda_dd9e98a4_idstub_0f4;
    case 0xf5: return (void*)&acuda_dd9e98a4_idstub_0f5;
    case 0xf6: return (void*)&acuda_dd9e98a4_idstub_0f6;
    case 0xf7: return (void*)&acuda_dd9e98a4_idstub_0f7;
    case 0xf8: return (void*)&acuda_dd9e98a4_idstub_0f8;
    case 0xf9: return (void*)&acuda_dd9e98a4_idstub_0f9;
    case 0xfa: return (void*)&acuda_dd9e98a4_idstub_0fa;
    case 0xfb: return (void*)&acuda_dd9e98a4_idstub_0fb;
    case 0xfc: return (void*)&acuda_dd9e98a4_idstub_0fc;
    case 0xfd: return (void*)&acuda_dd9e98a4_idstub_0fd;
    case 0xfe: return (void*)&acuda_dd9e98a4_idstub_0fe;
    case 0xff: return (void*)&acuda_dd9e98a4_idstub_0ff;
    case 0x100: return (void*)&acuda_dd9e98a4_idstub_100;
    case 0x101: return (void*)&acuda_dd9e98a4_idstub_101;
    case 0x102: return (void*)&acuda_dd9e98a4_idstub_102;
    case 0x103: return (void*)&acuda_dd9e98a4_idstub_103;
    case 0x104: return (void*)&acuda_dd9e98a4_idstub_104;
    case 0x105: return (void*)&acuda_dd9e98a4_idstub_105;
    case 0x106: return (void*)&acuda_dd9e98a4_idstub_106;
    case 0x107: return (void*)&acuda_dd9e98a4_idstub_107;
    case 0x108: return (void*)&acuda_dd9e98a4_idstub_108;
    case 0x109: return (void*)&acuda_dd9e98a4_idstub_109;
    case 0x10a: return (void*)&acuda_dd9e98a4_idstub_10a;
    case 0x10b: return (void*)&acuda_dd9e98a4_idstub_10b;
    case 0x10c: return (void*)&acuda_dd9e98a4_idstub_10c;
    case 0x10d: return (void*)&acuda_dd9e98a4_idstub_10d;
    case 0x10e: return (void*)&acuda_dd9e98a4_idstub_10e;
    case 0x10f: return (void*)&acuda_dd9e98a4_idstub_10f;
    case 0x110: return (void*)&acuda_dd9e98a4_idstub_110;
    case 0x111: return (void*)&acuda_dd9e98a4_idstub_111;
    case 0x112: return (void*)&acuda_dd9e98a4_idstub_112;
    case 0x113: return (void*)&acuda_dd9e98a4_idstub_113;
    case 0x114: return (void*)&acuda_dd9e98a4_idstub_114;
    case 0x115: return (void*)&acuda_dd9e98a4_idstub_115;
    case 0x116: return (void*)&acuda_dd9e98a4_idstub_116;
    case 0x117: return (void*)&acuda_dd9e98a4_idstub_117;
    case 0x118: return (void*)&acuda_dd9e98a4_idstub_118;
    case 0x119: return (void*)&acuda_dd9e98a4_idstub_119;
    case 0x11a: return (void*)&acuda_dd9e98a4_idstub_11a;
    case 0x11b: return (void*)&acuda_dd9e98a4_idstub_11b;
    case 0x11c: return (void*)&acuda_dd9e98a4_idstub_11c;
    case 0x11d: return (void*)&acuda_dd9e98a4_idstub_11d;
    case 0x11e: return (void*)&acuda_dd9e98a4_idstub_11e;
    case 0x11f: return (void*)&acuda_dd9e98a4_idstub_11f;
    case 0x120: return (void*)&acuda_dd9e98a4_idstub_120;
    case 0x121: return (void*)&acuda_dd9e98a4_idstub_121;
    case 0x122: return (void*)&acuda_dd9e98a4_idstub_122;
    case 0x123: return (void*)&acuda_dd9e98a4_idstub_123;
    case 0x124: return (void*)&acuda_dd9e98a4_idstub_124;
    case 0x125: return (void*)&acuda_dd9e98a4_idstub_125;
    case 0x126: return (void*)&acuda_dd9e98a4_idstub_126;
    case 0x127: return (void*)&acuda_dd9e98a4_idstub_127;
    case 0x128: return (void*)&acuda_dd9e98a4_idstub_128;
    case 0x129: return (void*)&acuda_dd9e98a4_idstub_129;
    case 0x12a: return (void*)&acuda_dd9e98a4_idstub_12a;
    case 0x12b: return (void*)&acuda_dd9e98a4_idstub_12b;
    case 0x12c: return (void*)&acuda_dd9e98a4_idstub_12c;
    case 0x12d: return (void*)&acuda_dd9e98a4_idstub_12d;
    case 0x12e: return (void*)&acuda_dd9e98a4_idstub_12e;
    case 0x12f: return (void*)&acuda_dd9e98a4_idstub_12f;
    case 0x130: return (void*)&acuda_dd9e98a4_idstub_130;
    case 0x131: return (void*)&acuda_dd9e98a4_idstub_131;
    case 0x132: return (void*)&acuda_dd9e98a4_idstub_132;
    case 0x133: return (void*)&acuda_dd9e98a4_idstub_133;
    case 0x134: return (void*)&acuda_dd9e98a4_idstub_134;
    case 0x135: return (void*)&acuda_dd9e98a4_idstub_135;
    case 0x136: return (void*)&acuda_dd9e98a4_idstub_136;
    case 0x137: return (void*)&acuda_dd9e98a4_idstub_137;
    case 0x138: return (void*)&acuda_dd9e98a4_idstub_138;
    case 0x139: return (void*)&acuda_dd9e98a4_idstub_139;
    case 0x13a: return (void*)&acuda_dd9e98a4_idstub_13a;
    case 0x13b: return (void*)&acuda_dd9e98a4_idstub_13b;
    case 0x13c: return (void*)&acuda_dd9e98a4_idstub_13c;
    case 0x13d: return (void*)&acuda_dd9e98a4_idstub_13d;
    case 0x13e: return (void*)&acuda_dd9e98a4_idstub_13e;
    case 0x13f: return (void*)&acuda_dd9e98a4_idstub_13f;
    case 0x140: return (void*)&acuda_dd9e98a4_idstub_140;
    case 0x141: return (void*)&acuda_dd9e98a4_idstub_141;
    case 0x142: return (void*)&acuda_dd9e98a4_idstub_142;
    case 0x143: return (void*)&acuda_dd9e98a4_idstub_143;
    case 0x144: return (void*)&acuda_dd9e98a4_idstub_144;
    case 0x145: return (void*)&acuda_dd9e98a4_idstub_145;
    case 0x146: return (void*)&acuda_dd9e98a4_idstub_146;
    case 0x147: return (void*)&acuda_dd9e98a4_idstub_147;
    case 0x148: return (void*)&acuda_dd9e98a4_idstub_148;
    case 0x149: return (void*)&acuda_dd9e98a4_idstub_149;
    case 0x14a: return (void*)&acuda_dd9e98a4_idstub_14a;
    case 0x14b: return (void*)&acuda_dd9e98a4_idstub_14b;
    case 0x14c: return (void*)&acuda_dd9e98a4_idstub_14c;
    case 0x14d: return (void*)&acuda_dd9e98a4_idstub_14d;
    case 0x14e: return (void*)&acuda_dd9e98a4_idstub_14e;
    case 0x14f: return (void*)&acuda_dd9e98a4_idstub_14f;
    case 0x150: return (void*)&acuda_dd9e98a4_idstub_150;
    case 0x151: return (void*)&acuda_dd9e98a4_idstub_151;
    case 0x152: return (void*)&acuda_dd9e98a4_idstub_152;
    case 0x153: return (void*)&acuda_dd9e98a4_idstub_153;
    case 0x154: return (void*)&acuda_dd9e98a4_idstub_154;
    case 0x155: return (void*)&acuda_dd9e98a4_idstub_155;
    case 0x156: return (void*)&acuda_dd9e98a4_idstub_156;
    case 0x157: return (void*)&acuda_dd9e98a4_idstub_157;
    case 0x158: return (void*)&acuda_dd9e98a4_idstub_158;
    case 0x159: return (void*)&acuda_dd9e98a4_idstub_159;
    case 0x15a: return (void*)&acuda_dd9e98a4_idstub_15a;
    case 0x15b: return (void*)&acuda_dd9e98a4_idstub_15b;
    case 0x15c: return (void*)&acuda_dd9e98a4_idstub_15c;
    case 0x15d: return (void*)&acuda_dd9e98a4_idstub_15d;
    case 0x15e: return (void*)&acuda_dd9e98a4_idstub_15e;
    case 0x15f: return (void*)&acuda_dd9e98a4_idstub_15f;
    case 0x160: return (void*)&acuda_dd9e98a4_idstub_160;
    case 0x161: return (void*)&acuda_dd9e98a4_idstub_161;
    case 0x162: return (void*)&acuda_dd9e98a4_idstub_162;
    case 0x163: return (void*)&acuda_dd9e98a4_idstub_163;
    case 0x164: return (void*)&acuda_dd9e98a4_idstub_164;
    case 0x165: return (void*)&acuda_dd9e98a4_idstub_165;
    case 0x166: return (void*)&acuda_dd9e98a4_idstub_166;
    case 0x167: return (void*)&acuda_dd9e98a4_idstub_167;
    case 0x168: return (void*)&acuda_dd9e98a4_idstub_168;
    case 0x169: return (void*)&acuda_dd9e98a4_idstub_169;
    case 0x16a: return (void*)&acuda_dd9e98a4_idstub_16a;
    case 0x16b: return (void*)&acuda_dd9e98a4_idstub_16b;
    case 0x16c: return (void*)&acuda_dd9e98a4_idstub_16c;
    case 0x16d: return (void*)&acuda_dd9e98a4_idstub_16d;
    case 0x16e: return (void*)&acuda_dd9e98a4_idstub_16e;
    case 0x16f: return (void*)&acuda_dd9e98a4_idstub_16f;
    case 0x170: return (void*)&acuda_dd9e98a4_idstub_170;
    case 0x171: return (void*)&acuda_dd9e98a4_idstub_171;
    case 0x172: return (void*)&acuda_dd9e98a4_idstub_172;
    case 0x173: return (void*)&acuda_dd9e98a4_idstub_173;
    case 0x174: return (void*)&acuda_dd9e98a4_idstub_174;
    case 0x175: return (void*)&acuda_dd9e98a4_idstub_175;
    case 0x176: return (void*)&acuda_dd9e98a4_idstub_176;
    case 0x177: return (void*)&acuda_dd9e98a4_idstub_177;
    case 0x178: return (void*)&acuda_dd9e98a4_idstub_178;
    case 0x179: return (void*)&acuda_dd9e98a4_idstub_179;
    case 0x17a: return (void*)&acuda_dd9e98a4_idstub_17a;
    case 0x17b: return (void*)&acuda_dd9e98a4_idstub_17b;
    case 0x17c: return (void*)&acuda_dd9e98a4_idstub_17c;
    case 0x17d: return (void*)&acuda_dd9e98a4_idstub_17d;
    case 0x17e: return (void*)&acuda_dd9e98a4_idstub_17e;
    case 0x17f: return (void*)&acuda_dd9e98a4_idstub_17f;
    case 0x180: return (void*)&acuda_dd9e98a4_idstub_180;
    case 0x181: return (void*)&acuda_dd9e98a4_idstub_181;
    case 0x182: return (void*)&acuda_dd9e98a4_idstub_182;
    case 0x183: return (void*)&acuda_dd9e98a4_idstub_183;
    case 0x184: return (void*)&acuda_dd9e98a4_idstub_184;
    case 0x185: return (void*)&acuda_dd9e98a4_idstub_185;
    case 0x186: return (void*)&acuda_dd9e98a4_idstub_186;
    case 0x187: return (void*)&acuda_dd9e98a4_idstub_187;
    case 0x188: return (void*)&acuda_dd9e98a4_idstub_188;
    case 0x189: return (void*)&acuda_dd9e98a4_idstub_189;
    case 0x18a: return (void*)&acuda_dd9e98a4_idstub_18a;
    case 0x18b: return (void*)&acuda_dd9e98a4_idstub_18b;
    case 0x18c: return (void*)&acuda_dd9e98a4_idstub_18c;
    case 0x18d: return (void*)&acuda_dd9e98a4_idstub_18d;
    case 0x18e: return (void*)&acuda_dd9e98a4_idstub_18e;
    case 0x18f: return (void*)&acuda_dd9e98a4_idstub_18f;
    case 0x190: return (void*)&acuda_dd9e98a4_idstub_190;
    case 0x191: return (void*)&acuda_dd9e98a4_idstub_191;
    case 0x192: return (void*)&acuda_dd9e98a4_idstub_192;
    case 0x193: return (void*)&acuda_dd9e98a4_idstub_193;
    case 0x194: return (void*)&acuda_dd9e98a4_idstub_194;
    case 0x195: return (void*)&acuda_dd9e98a4_idstub_195;
    case 0x196: return (void*)&acuda_dd9e98a4_idstub_196;
    case 0x197: return (void*)&acuda_dd9e98a4_idstub_197;
    case 0x198: return (void*)&acuda_dd9e98a4_idstub_198;
    case 0x199: return (void*)&acuda_dd9e98a4_idstub_199;
    case 0x19a: return (void*)&acuda_dd9e98a4_idstub_19a;
    case 0x19b: return (void*)&acuda_dd9e98a4_idstub_19b;
    case 0x19c: return (void*)&acuda_dd9e98a4_idstub_19c;
    case 0x19d: return (void*)&acuda_dd9e98a4_idstub_19d;
    case 0x19e: return (void*)&acuda_dd9e98a4_idstub_19e;
    case 0x19f: return (void*)&acuda_dd9e98a4_idstub_19f;
    case 0x1a0: return (void*)&acuda_dd9e98a4_idstub_1a0;
    case 0x1a1: return (void*)&acuda_dd9e98a4_idstub_1a1;
    case 0x1a2: return (void*)&acuda_dd9e98a4_idstub_1a2;
    case 0x1a3: return (void*)&acuda_dd9e98a4_idstub_1a3;
    case 0x1a4: return (void*)&acuda_dd9e98a4_idstub_1a4;
    case 0x1a5: return (void*)&acuda_dd9e98a4_idstub_1a5;
    case 0x1a6: return (void*)&acuda_dd9e98a4_idstub_1a6;
    case 0x1a7: return (void*)&acuda_dd9e98a4_idstub_1a7;
    case 0x1a8: return (void*)&acuda_dd9e98a4_idstub_1a8;
    case 0x1a9: return (void*)&acuda_dd9e98a4_idstub_1a9;
    case 0x1aa: return (void*)&acuda_dd9e98a4_idstub_1aa;
    case 0x1ab: return (void*)&acuda_dd9e98a4_idstub_1ab;
    case 0x1ac: return (void*)&acuda_dd9e98a4_idstub_1ac;
    case 0x1ad: return (void*)&acuda_dd9e98a4_idstub_1ad;
    case 0x1ae: return (void*)&acuda_dd9e98a4_idstub_1ae;
    case 0x1af: return (void*)&acuda_dd9e98a4_idstub_1af;
    case 0x1b0: return (void*)&acuda_dd9e98a4_idstub_1b0;
    case 0x1b1: return (void*)&acuda_dd9e98a4_idstub_1b1;
    case 0x1b2: return (void*)&acuda_dd9e98a4_idstub_1b2;
    case 0x1b3: return (void*)&acuda_dd9e98a4_idstub_1b3;
    case 0x1b4: return (void*)&acuda_dd9e98a4_idstub_1b4;
    case 0x1b5: return (void*)&acuda_dd9e98a4_idstub_1b5;
    case 0x1b6: return (void*)&acuda_dd9e98a4_idstub_1b6;
    case 0x1b7: return (void*)&acuda_dd9e98a4_idstub_1b7;
    case 0x1b9: return (void*)&acuda_dd9e98a4_idstub_1b9;
    case 0x1ba: return (void*)&acuda_dd9e98a4_idstub_1ba;
    case 0x1bb: return (void*)&acuda_dd9e98a4_idstub_1bb;
    case 0x1bc: return (void*)&acuda_dd9e98a4_idstub_1bc;
    case 0x1bd: return (void*)&acuda_dd9e98a4_idstub_1bd;
    case 0x1be: return (void*)&acuda_dd9e98a4_idstub_1be;
    case 0x1bf: return (void*)&acuda_dd9e98a4_idstub_1bf;
    case 0x1c0: return (void*)&acuda_dd9e98a4_idstub_1c0;
    case 0x1c1: return (void*)&acuda_dd9e98a4_idstub_1c1;
    case 0x1c2: return (void*)&acuda_dd9e98a4_idstub_1c2;
    case 0x1c3: return (void*)&acuda_dd9e98a4_idstub_1c3;
    case 0x1c4: return (void*)&acuda_dd9e98a4_idstub_1c4;
    case 0x1c5: return (void*)&acuda_dd9e98a4_idstub_1c5;
    case 0x1c6: return (void*)&acuda_dd9e98a4_idstub_1c6;
    case 0x1c7: return (void*)&acuda_dd9e98a4_idstub_1c7;
    case 0x1c8: return (void*)&acuda_dd9e98a4_idstub_1c8;
    case 0x1c9: return (void*)&acuda_dd9e98a4_idstub_1c9;
    case 0x1ca: return (void*)&acuda_dd9e98a4_idstub_1ca;
    case 0x1cb: return (void*)&acuda_dd9e98a4_idstub_1cb;
    case 0x1cc: return (void*)&acuda_dd9e98a4_idstub_1cc;
    case 0x1cd: return (void*)&acuda_dd9e98a4_idstub_1cd;
    case 0x1ce: return (void*)&acuda_dd9e98a4_idstub_1ce;
    case 0x1cf: return (void*)&acuda_dd9e98a4_idstub_1cf;
    case 0x1d0: return (void*)&acuda_dd9e98a4_idstub_1d0;
    case 0x1d1: return (void*)&acuda_dd9e98a4_idstub_1d1;
    case 0x1d2: return (void*)&acuda_dd9e98a4_idstub_1d2;
    case 0x1d3: return (void*)&acuda_dd9e98a4_idstub_1d3;
    case 0x1d4: return (void*)&acuda_dd9e98a4_idstub_1d4;
    case 0x1d5: return (void*)&acuda_dd9e98a4_idstub_1d5;
    case 0x1d6: return (void*)&acuda_dd9e98a4_idstub_1d6;
    case 0x1d7: return (void*)&acuda_dd9e98a4_idstub_1d7;
    case 0x1d8: return (void*)&acuda_dd9e98a4_idstub_1d8;
    case 0x1d9: return (void*)&acuda_dd9e98a4_idstub_1d9;
    case 0x1da: return (void*)&acuda_dd9e98a4_idstub_1da;
    case 0x1db: return (void*)&acuda_dd9e98a4_idstub_1db;
    case 0x1dc: return (void*)&acuda_dd9e98a4_idstub_1dc;
    case 0x1dd: return (void*)&acuda_dd9e98a4_idstub_1dd;
    case 0x1de: return (void*)&acuda_dd9e98a4_idstub_1de;
    case 0x1df: return (void*)&acuda_dd9e98a4_idstub_1df;
    case 0x1e0: return (void*)&acuda_dd9e98a4_idstub_1e0;
    case 0x1e1: return (void*)&acuda_dd9e98a4_idstub_1e1;
    case 0x1e2: return (void*)&acuda_dd9e98a4_idstub_1e2;
    case 0x1e3: return (void*)&acuda_dd9e98a4_idstub_1e3;
    case 0x1e4: return (void*)&acuda_dd9e98a4_idstub_1e4;
    case 0x1e5: return (void*)&acuda_dd9e98a4_idstub_1e5;
    case 0x1e7: return (void*)&acuda_dd9e98a4_idstub_1e7;
    case 0x1e8: return (void*)&acuda_dd9e98a4_idstub_1e8;
    case 0x1e9: return (void*)&acuda_dd9e98a4_idstub_1e9;
    case 0x1ea: return (void*)&acuda_dd9e98a4_idstub_1ea;
    case 0x1eb: return (void*)&acuda_dd9e98a4_idstub_1eb;
    case 0x1ec: return (void*)&acuda_dd9e98a4_idstub_1ec;
    case 0x1ed: return (void*)&acuda_dd9e98a4_idstub_1ed;
    case 0x1f0: return (void*)&acuda_dd9e98a4_idstub_1f0;
    case 0x1f1: return (void*)&acuda_dd9e98a4_idstub_1f1;
    case 0x1f2: return (void*)&acuda_dd9e98a4_idstub_1f2;
    case 0x1f4: return (void*)&acuda_dd9e98a4_idstub_1f4;
    case 0x1f5: return (void*)&acuda_dd9e98a4_idstub_1f5;
    case 0x1fa: return (void*)&acuda_dd9e98a4_idstub_1fa;
    case 0x1fb: return (void*)&acuda_dd9e98a4_idstub_1fb;
    case 0x1fc: return (void*)&acuda_dd9e98a4_idstub_1fc;
    case 0x1fd: return (void*)&acuda_dd9e98a4_idstub_1fd;
    case 0x1fe: return (void*)&acuda_dd9e98a4_idstub_1fe;
    case 0x1ff: return (void*)&acuda_dd9e98a4_idstub_1ff;
    case 0x200: return (void*)&acuda_dd9e98a4_idstub_200;
    case 0x201: return (void*)&acuda_dd9e98a4_idstub_201;
    case 0x202: return (void*)&acuda_dd9e98a4_idstub_202;
    case 0x203: return (void*)&acuda_dd9e98a4_idstub_203;
    case 0x204: return (void*)&acuda_dd9e98a4_idstub_204;
    case 0x205: return (void*)&acuda_dd9e98a4_idstub_205;
    case 0x206: return (void*)&acuda_dd9e98a4_idstub_206;
    case 0x207: return (void*)&acuda_dd9e98a4_idstub_207;
    case 0x208: return (void*)&acuda_dd9e98a4_idstub_208;
    case 0x209: return (void*)&acuda_dd9e98a4_idstub_209;
    case 0x20a: return (void*)&acuda_dd9e98a4_idstub_20a;
    case 0x20b: return (void*)&acuda_dd9e98a4_idstub_20b;
    case 0x20c: return (void*)&acuda_dd9e98a4_idstub_20c;
    case 0x20d: return (void*)&acuda_dd9e98a4_idstub_20d;
    case 0x20e: return (void*)&acuda_dd9e98a4_idstub_20e;
    case 0x20f: return (void*)&acuda_dd9e98a4_idstub_20f;
    case 0x210: return (void*)&acuda_dd9e98a4_idstub_210;
    case 0x211: return (void*)&acuda_dd9e98a4_idstub_211;
    case 0x212: return (void*)&acuda_dd9e98a4_idstub_212;
    case 0x213: return (void*)&acuda_dd9e98a4_idstub_213;
    case 0x214: return (void*)&acuda_dd9e98a4_idstub_214;
    case 0x215: return (void*)&acuda_dd9e98a4_idstub_215;
    case 0x216: return (void*)&acuda_dd9e98a4_idstub_216;
    case 0x217: return (void*)&acuda_dd9e98a4_idstub_217;
    case 0x218: return (void*)&acuda_dd9e98a4_idstub_218;
    case 0x219: return (void*)&acuda_dd9e98a4_idstub_219;
    case 0x21a: return (void*)&acuda_dd9e98a4_idstub_21a;
    case 0x21b: return (void*)&acuda_dd9e98a4_idstub_21b;
    case 0x21c: return (void*)&acuda_dd9e98a4_idstub_21c;
    case 0x21d: return (void*)&acuda_dd9e98a4_idstub_21d;
    case 0x21e: return (void*)&acuda_dd9e98a4_idstub_21e;
    case 0x21f: return (void*)&acuda_dd9e98a4_idstub_21f;
    case 0x220: return (void*)&acuda_dd9e98a4_idstub_220;
    case 0x221: return (void*)&acuda_dd9e98a4_idstub_221;
    case 0x222: return (void*)&acuda_dd9e98a4_idstub_222;
    case 0x223: return (void*)&acuda_dd9e98a4_idstub_223;
    case 0x224: return (void*)&acuda_dd9e98a4_idstub_224;
    case 0x225: return (void*)&acuda_dd9e98a4_idstub_225;
    case 0x226: return (void*)&acuda_dd9e98a4_idstub_226;
    case 0x227: return (void*)&acuda_dd9e98a4_idstub_227;
    case 0x228: return (void*)&acuda_dd9e98a4_idstub_228;
    case 0x229: return (void*)&acuda_dd9e98a4_idstub_229;
    case 0x22a: return (void*)&acuda_dd9e98a4_idstub_22a;
    case 0x22b: return (void*)&acuda_dd9e98a4_idstub_22b;
    case 0x22c: return (void*)&acuda_dd9e98a4_idstub_22c;
    case 0x22d: return (void*)&acuda_dd9e98a4_idstub_22d;
    case 0x22e: return (void*)&acuda_dd9e98a4_idstub_22e;
    case 0x22f: return (void*)&acuda_dd9e98a4_idstub_22f;
    case 0x230: return (void*)&acuda_dd9e98a4_idstub_230;
    case 0x231: return (void*)&acuda_dd9e98a4_idstub_231;
    case 0x232: return (void*)&acuda_dd9e98a4_idstub_232;
    case 0x233: return (void*)&acuda_dd9e98a4_idstub_233;
    case 0x234: return (void*)&acuda_dd9e98a4_idstub_234;
    case 0x235: return (void*)&acuda_dd9e98a4_idstub_235;
    case 0x236: return (void*)&acuda_dd9e98a4_idstub_236;
    case 0x237: return (void*)&acuda_dd9e98a4_idstub_237;
    case 0x238: return (void*)&acuda_dd9e98a4_idstub_238;
    case 0x239: return (void*)&acuda_dd9e98a4_idstub_239;
    case 0x23a: return (void*)&acuda_dd9e98a4_idstub_23a;
    case 0x23b: return (void*)&acuda_dd9e98a4_idstub_23b;
    case 0x23c: return (void*)&acuda_dd9e98a4_idstub_23c;
    case 0x23d: return (void*)&acuda_dd9e98a4_idstub_23d;
    case 0x23e: return (void*)&acuda_dd9e98a4_idstub_23e;
    case 0x23f: return (void*)&acuda_dd9e98a4_idstub_23f;
    default: return (void*)&acuda_dd9e98a4_idstub_000;
    }
}

static void *ACUDA_CUDAAPI acuda_dd9e98a4_dispatcher(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0;
    unsigned int id = 0xFFFFFFFFu, sub = 0;
    if (a) {
        const unsigned char *p = (const unsigned char *)a;
        id = (unsigned int)p[8] | ((unsigned int)p[9] << 8) | ((unsigned int)p[10] << 16) | ((unsigned int)p[11] << 24);
        sub = (unsigned int)p[12] | ((unsigned int)p[13] << 8) | ((unsigned int)p[14] << 16) | ((unsigned int)p[15] << 24);
    }
    if (n < 80) {
        ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4_dispatch[%d] id=0x%x sub=%u a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n",
                n, id, sub, a,b,c,d,e,f,g,h);
    } else if (n == 80) {
        ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4_dispatch ... (%d calls so far)\n", n);
    }
    n++;
    (void)f;
    return acuda_dd9e98a4_get_id_stub(id);
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_slot01(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 slot01 a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", a,b,c,d,e,f,g,h);
    acuda_dump_mem("slot01[a]", a, 32);
    acuda_dump_mem("slot01[b]", b, 32);
    acuda_dump_mem("slot01[c]", c, 32);
    /* NVIDIA probe: slot01 sets *b to dispatcher pointer (dispatcher returns fn ptr in RAX) */
    if (b) *(void**)b = (void *)&acuda_dd9e98a4_dispatcher;
    return CUDA_SUCCESS;
}
#define ACUDA_DD9E98A4_STUB(n) \
    static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_slot##n(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) { \
        ACUDA_DEBUG_PRINT( "ACuda-debug: acuda_infoocial_cudart_interface_dd9e98a4 slot " #n " called a=%p b=%p c=%p d=%p\n", a,b,c,d); \
        return CUDA_SUCCESS; \
    }
ACUDA_DD9E98A4_STUB(02)
ACUDA_DD9E98A4_STUB(10)
ACUDA_DD9E98A4_STUB(11)
ACUDA_DD9E98A4_STUB(19)
ACUDA_DD9E98A4_STUB(20)
ACUDA_DD9E98A4_STUB(21)
ACUDA_DD9E98A4_STUB(22)
ACUDA_DD9E98A4_STUB(31)
#undef ACUDA_DD9E98A4_STUB
CUresult ACUDA_CUDAAPI cuGetExportTable(const void **ppExportTable,
                          const struct CUuuid_st *pExportTableId) {
    acuda_log_caller("cuGetExportTable");
    if (pExportTableId) {
        const unsigned char *b = (const unsigned char *)pExportTableId->bytes;
        if (b[0]==0xdd && b[1]==0x9e && b[2]==0x98 && b[3]==0xa4 &&
            b[4]==0xf4 && b[5]==0x10 && b[6]==0x48 && b[7]==0x44 &&
            b[8]==0x99 && b[9]==0xb0 && b[10]==0xe2 && b[11]==0xae &&
            b[12]==0xb7 && b[13]==0xb4 && b[14]==0x32 && b[15]==0xb5) {
            static const void *tbl[64];
            static int inited = 0;
            if (getenv("ACUDA_DD9E98A4_FALLBACK")) {
                ACUDA_DEBUG_PRINT( "ACuda-debug: cuGetExportTable -> forcing NOT_SUPPORTED (fallback experiment)\n");
                return CUDA_ERROR_NOT_SUPPORTED;
            }
            if (!inited) {
                /* Layout mirrored from NVIDIA nvcuda.dll probe (first 32 entries) */
                tbl[0]  = (const void *)(uintptr_t)0x18;
                tbl[1]  = (const void *)&acuda_dd9e98a4_slot01;
                tbl[2]  = (const void *)&acuda_dd9e98a4_slot02;
                tbl[3]  = (const void *)0;
                tbl[4]  = (const void *)(uintptr_t)0x1;
                tbl[5]  = (const void *)(uintptr_t)0xe943fb0113fafa73ULL;
                tbl[6]  = (const void *)(uintptr_t)0x781ef6113590168cULL;
                tbl[7]  = (const void *)(uintptr_t)0xffffffffffffffffULL;
                tbl[8]  = (const void *)(uintptr_t)0xffffffffULL;
                tbl[9]  = (const void *)(uintptr_t)0x18;
                tbl[10] = (const void *)&acuda_dd9e98a4_slot10;
                tbl[11] = (const void *)&acuda_dd9e98a4_slot11;
                tbl[12] = (const void *)0;
                tbl[13] = (const void *)(uintptr_t)0x1;
                tbl[14] = (const void *)(uintptr_t)0xe943fb0113fafa73ULL;
                tbl[15] = (const void *)(uintptr_t)0x781ef6113590168cULL;
                tbl[16] = (const void *)(uintptr_t)0xffffffffffffffffULL;
                tbl[17] = (const void *)(uintptr_t)0xffffffffULL;
                tbl[18] = (const void *)(uintptr_t)0x28;
                tbl[19] = (const void *)&acuda_dd9e98a4_slot19;
                tbl[20] = (const void *)&acuda_dd9e98a4_slot20;
                tbl[21] = (const void *)&acuda_dd9e98a4_slot21;
                tbl[22] = (const void *)&acuda_dd9e98a4_slot22;
                tbl[23] = (const void *)0;
                tbl[24] = (const void *)(uintptr_t)0x1;
                tbl[25] = (const void *)(uintptr_t)0xe943fb0113fafa73ULL;
                tbl[26] = (const void *)(uintptr_t)0x781ef6113590168cULL;
                tbl[27] = (const void *)(uintptr_t)0xffffffffffffffffULL;
                tbl[28] = (const void *)(uintptr_t)0xffffffffffffffffULL;
                tbl[29] = (const void *)(uintptr_t)0xffffffffULL;
                tbl[30] = (const void *)(uintptr_t)0x68;
                tbl[31] = (const void *)&acuda_dd9e98a4_slot31;
                for (int i = 32; i < 64; i++) tbl[i] = (const void *)0;
                inited = 1;
            }
            *ppExportTable = (const void *)tbl;
            ACUDA_DEBUG_PRINT( "ACuda-debug: cuGetExportTable -> SUCCESS (acuda_infoocial_cudart_interface_dd9e98a4 traced table) table=%p\n", *ppExportTable);
            return CUDA_SUCCESS;
        }
    }

    // {6BD5FB6C-5BF4-E74A-8987-D93912FD9DF9}
    static const char cudart_guid[16] = {
        (char)0x6b, (char)0xd5, (char)0xfb, (char)0x6c,
        (char)0x5b, (char)0xf4, (char)0xe7, (char)0x4a,
        (char)0x89, (char)0x87, (char)0xd9, (char)0x39,
        (char)0x12, (char)0xfd, (char)0x9d, (char)0xf9
    };
    if (!ppExportTable || !pExportTableId) {
        ACUDA_DEBUG_PRINT( "ACuda-debug: cuGetExportTable -> INVALID_VALUE (pp=%p id=%p)\n",
                (const void *)ppExportTable, (const void *)pExportTableId);
        return CUDA_ERROR_INVALID_VALUE;
    }
    ACUDA_DEBUG_PRINT( "ACuda-debug: cuGetExportTable(%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x)\n",
            (unsigned char)pExportTableId->bytes[0],
            (unsigned char)pExportTableId->bytes[1],
            (unsigned char)pExportTableId->bytes[2],
            (unsigned char)pExportTableId->bytes[3],
            (unsigned char)pExportTableId->bytes[4],
            (unsigned char)pExportTableId->bytes[5],
            (unsigned char)pExportTableId->bytes[6],
            (unsigned char)pExportTableId->bytes[7],
            (unsigned char)pExportTableId->bytes[8],
            (unsigned char)pExportTableId->bytes[9],
            (unsigned char)pExportTableId->bytes[10],
            (unsigned char)pExportTableId->bytes[11],
            (unsigned char)pExportTableId->bytes[12],
            (unsigned char)pExportTableId->bytes[13],
            (unsigned char)pExportTableId->bytes[14],
            (unsigned char)pExportTableId->bytes[15]);
    if (getenv("ACUDA_TRACE")) {
#if defined(_M_IX86)
        void *retaddr = _ReturnAddress();
        MEMORY_BASIC_INFORMATION mbi;
        char modpath[MAX_PATH] = "?";
        if (VirtualQuery(retaddr, &mbi, sizeof(mbi)))
            GetModuleFileNameA((HMODULE)mbi.AllocationBase, modpath, MAX_PATH);
        ACUDA_DEBUG_PRINT( "acuda-trace .. cuGetExportTable ret=%p rva=%Ix mod=%s pp=%p\n",
                retaddr, (uintptr_t)retaddr - (uintptr_t)mbi.AllocationBase,
                modpath, (const void *)ppExportTable);
#endif
        ACUDA_DEBUG_PRINT( "acuda-trace ?? cuGetExportTable(%02x%02x%02x%02x-"
                "%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x)\n",
                (unsigned char)pExportTableId->bytes[0],
                (unsigned char)pExportTableId->bytes[1],
                (unsigned char)pExportTableId->bytes[2],
                (unsigned char)pExportTableId->bytes[3],
                (unsigned char)pExportTableId->bytes[4],
                (unsigned char)pExportTableId->bytes[5],
                (unsigned char)pExportTableId->bytes[6],
                (unsigned char)pExportTableId->bytes[7],
                (unsigned char)pExportTableId->bytes[8],
                (unsigned char)pExportTableId->bytes[9],
                (unsigned char)pExportTableId->bytes[10],
                (unsigned char)pExportTableId->bytes[11],
                (unsigned char)pExportTableId->bytes[12],
                (unsigned char)pExportTableId->bytes[13],
                (unsigned char)pExportTableId->bytes[14],
                (unsigned char)pExportTableId->bytes[15]);
    }
    // {A094798C-2E74-2E74-93F2-0800200C0A66}
    static const char tools_guid[16] = {
        (char)0xa0, (char)0x94, (char)0x79, (char)0x8c,
        (char)0x2e, (char)0x74, (char)0x2e, (char)0x74,
        (char)0x93, (char)0xf2, (char)0x08, (char)0x00,
        (char)0x20, (char)0x0c, (char)0x0a, (char)0x66
    };
    if (memcmp(pExportTableId->bytes, cudart_guid, sizeof(cudart_guid)) == 0) {
        ensure_cudart_iface();
        *ppExportTable = (const void *)g_cudart_interface;
        ACUDA_DEBUG_PRINT( "ACuda-debug: cuGetExportTable -> SUCCESS (cudart interface) table=%p\n", *ppExportTable);
        return CUDA_SUCCESS;
    }
    if (memcmp(pExportTableId->bytes, tools_guid, sizeof(tools_guid)) == 0) {
        ACUDA_DEBUG_PRINT("acuda-trace .. tools branch pp=%p\n",
                    (const void *)ppExportTable);
        *ppExportTable = (const void *)g_tools_runtime_hooks;
        ACUDA_DEBUG_PRINT( "ACuda-debug: cuGetExportTable -> SUCCESS (tools hooks) table=%p\n", *ppExportTable);
        return CUDA_SUCCESS;
    }
    // {42D85A81-23F6-CB47-8298-F6E78A3AECDC}
    static const char tls_guid[16] = {
        (char)0x42, (char)0xd8, (char)0x5a, (char)0x81,
        (char)0x23, (char)0xf6, (char)0xcb, (char)0x47,
        (char)0x82, (char)0x98, (char)0xf6, (char)0xe7,
        (char)0x8a, (char)0x3a, (char)0xec, (char)0xdc
    };
    if (memcmp(pExportTableId->bytes, tls_guid, sizeof(tls_guid)) == 0) {
        *ppExportTable = (const void *)g_tools_tls;
        ACUDA_DEBUG_PRINT( "ACuda-debug: cuGetExportTable -> SUCCESS (tools tls) table=%p\n", *ppExportTable);
        return CUDA_SUCCESS;
    }
    // {C693336E-1121-DF11-A8C3-68F355D89593}
    static const char ctx_storage_guid[16] = {
        (char)0xc6, (char)0x93, (char)0x33, (char)0x6e,
        (char)0x11, (char)0x21, (char)0xdf, (char)0x11,
        (char)0xa8, (char)0xc3, (char)0x68, (char)0xf3,
        (char)0x55, (char)0xd8, (char)0x95, (char)0x93
    };
    if (memcmp(pExportTableId->bytes, ctx_storage_guid,
               sizeof(ctx_storage_guid)) == 0) {
        *ppExportTable = (const void *)g_ctx_storage;
        return CUDA_SUCCESS;
    }
    // {0CA50B8C-1004-929A-89A7-D0DF10E77286}
    static const char ctx_create_guid[16] = {
        (char)0x0c, (char)0xa5, (char)0x0b, (char)0x8c,
        (char)0x10, (char)0x04, (char)0x92, (char)0x9a,
        (char)0x89, (char)0xa7, (char)0xd0, (char)0xdf,
        (char)0x10, (char)0xe7, (char)0x72, (char)0x86
    };
    if (memcmp(pExportTableId->bytes, ctx_create_guid,
               sizeof(ctx_create_guid)) == 0) {
        *ppExportTable = (const void *)g_ctx_create_bypass;
        return CUDA_SUCCESS;
    }
    // {263E8860-7CD2-6143-92F6-BBD5006DFA7E}
    static const char ctx_checks_guid[16] = {
        (char)0x26, (char)0x3e, (char)0x88, (char)0x60,
        (char)0x7c, (char)0xd2, (char)0x61, (char)0x43,
        (char)0x92, (char)0xf6, (char)0xbb, (char)0xd5,
        (char)0x00, (char)0x6d, (char)0xfa, (char)0x7e
    };
    if (memcmp(pExportTableId->bytes, ctx_checks_guid,
               sizeof(ctx_checks_guid)) == 0) {
        *ppExportTable = (const void *)g_context_checks;
        return CUDA_SUCCESS;
    }
    // {D4082055-BDE6-704B-8D34-BA123C66E1F2}
    static const char integrity_guid[16] = {
        (char)0xd4, (char)0x08, (char)0x20, (char)0x55,
        (char)0xbd, (char)0xe6, (char)0x70, (char)0x4b,
        (char)0x8d, (char)0x34, (char)0xba, (char)0x12,
        (char)0x3c, (char)0x66, (char)0xe1, (char)0xf2
    };
    if (memcmp(pExportTableId->bytes, integrity_guid,
               sizeof(integrity_guid)) == 0) {
        *ppExportTable = (const void *)g_integrity_check;
        return CUDA_SUCCESS;
    }
    // {21318C60-9714-3248-8CA6-41FF7324C8F2} (CUDA 13.4)
    static const char cuda134_guid[16] = {
        (char)0x21, (char)0x31, (char)0x8c, (char)0x60,
        (char)0x97, (char)0x14, (char)0x32, (char)0x48,
        (char)0x8c, (char)0xa6, (char)0x41, (char)0xff,
        (char)0x73, (char)0x24, (char)0xc8, (char)0xf2
    };
    if (memcmp(pExportTableId->bytes, cuda134_guid,
               sizeof(cuda134_guid)) == 0) {
        *ppExportTable = (const void *)g_cuda134_tools;
        return CUDA_SUCCESS;
    }
    // {195BCBF4-D67D-024A-ACC5-1D29CEA631AE} (heap access, cudart 6.x allocs)
    static const char heap_guid[16] = {
        (char)0x19, (char)0x5b, (char)0xcb, (char)0xf4,
        (char)0xd6, (char)0x7d, (char)0x02, (char)0x4a,
        (char)0xac, (char)0xc5, (char)0x1d, (char)0x29,
        (char)0xce, (char)0xa6, (char)0x31, (char)0xae
    };
    if (memcmp(pExportTableId->bytes, heap_guid,
               sizeof(heap_guid)) == 0) {
        *ppExportTable = (const void *)g_heap_access;
        return CUDA_SUCCESS;
    }
    ACUDA_DEBUG_PRINT( "ACuda-debug: cuGetExportTable -> NOT_FOUND\n");
    return CUDA_ERROR_NOT_FOUND;
}
