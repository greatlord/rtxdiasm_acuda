#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_stub.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#if defined(_WIN32)
#include <windows.h>
#if defined(_M_IX86)
#include <intrin.h>
#endif
#else
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#endif

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

CUresult ACUDA_CUDAAPI cuGetExportTable(const void **ppExportTable,
                          const struct CUuuid_st *pExportTableId) {
    // {6BD5FB6C-5BF4-E74A-8987-D93912FD9DF9}
    static const char cudart_guid[16] = {
        (char)0x6b, (char)0xd5, (char)0xfb, (char)0x6c,
        (char)0x5b, (char)0xf4, (char)0xe7, (char)0x4a,
        (char)0x89, (char)0x87, (char)0xd9, (char)0x39,
        (char)0x12, (char)0xfd, (char)0x9d, (char)0xf9
    };
    if (!ppExportTable || !pExportTableId)
        return CUDA_ERROR_INVALID_VALUE;
    if (getenv("ACUDA_TRACE")) {
#if defined(_M_IX86)
        void *retaddr = _ReturnAddress();
        MEMORY_BASIC_INFORMATION mbi;
        char modpath[MAX_PATH] = "?";
        if (VirtualQuery(retaddr, &mbi, sizeof(mbi)))
            GetModuleFileNameA((HMODULE)mbi.AllocationBase, modpath, MAX_PATH);
        fprintf(stderr, "acuda-trace .. cuGetExportTable ret=%p rva=%Ix mod=%s pp=%p\n",
                retaddr, (uintptr_t)retaddr - (uintptr_t)mbi.AllocationBase,
                modpath, (const void *)ppExportTable);
#endif
        fprintf(stderr, "acuda-trace ?? cuGetExportTable(%02x%02x%02x%02x-"
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
        return CUDA_SUCCESS;
    }
    if (memcmp(pExportTableId->bytes, tools_guid, sizeof(tools_guid)) == 0) {
        ACUDA_DEBUG_PRINT("acuda-trace .. tools branch pp=%p\n",
                    (const void *)ppExportTable);
        *ppExportTable = (const void *)g_tools_runtime_hooks;
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
    *ppExportTable = NULL;
    return CUDA_ERROR_NOT_FOUND;
}
