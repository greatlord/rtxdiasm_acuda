#include "acuda/driver/driver_internal.h"
#include "acuda/driver/arena.h"
#include <stdlib.h>
#include <string.h>

CUresult ACUDA_CUDAAPI cuPointerGetAttribute(void *data,
                                              CUpointer_attribute attribute,
                                              CUdeviceptr ptr) {
    if (!data)
        return CUDA_ERROR_INVALID_VALUE;
    size_t cap = 0;
    int device = acuda_arena_capacity((uint64_t)ptr, &cap) == 0;
    switch (attribute) {
    case CU_POINTER_ATTRIBUTE_CONTEXT: {
        CUcontext *out = (CUcontext *)data;
        return need_ctx(out) == CUDA_SUCCESS ? CUDA_SUCCESS
                                             : CUDA_ERROR_NOT_INITIALIZED;
    }
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
        *(CUmemorytype *)data =
            device ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
        *(CUdeviceptr *)data = ptr;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_HOST_POINTER:
        *(void **)data = (void *)(uintptr_t)ptr;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_IS_MANAGED:
    case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
    case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
    case CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE:
        *(int *)data = 0;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_BUFFER_ID:
    case CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID:
        *(unsigned long long *)data = (unsigned long long)ptr;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
        *(int *)data = 0;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS:
        *(unsigned int *)data = 3; /* read/write */
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
        *(CUdeviceptr *)data = ptr;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
        *(size_t *)data = cap ? cap : 0;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_MAPPED:
        *(int *)data = device ? 1 : 0;
        return CUDA_SUCCESS;
    default:
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult ACUDA_CUDAAPI cuPointerGetAttributes(unsigned int numAttributes,
                                              CUpointer_attribute *attributes,
                                              void **data,
                                              CUdeviceptr ptr) {
    if (numAttributes && (!attributes || !data))
        return CUDA_ERROR_INVALID_VALUE;
    for (unsigned int i = 0; i < numAttributes; i++) {
        CUresult r = cuPointerGetAttribute(data[i], attributes[i], ptr);
        if (r != CUDA_SUCCESS)
            return r;
    }
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemHostGetDevicePointer(void **pdst, void *p, unsigned int flags) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemHostGetDevicePointer(%p flags=%x)\n",
                p, flags);
    (void)flags;
    if (!pdst || !p)
        return CUDA_ERROR_INVALID_VALUE;
    *pdst = p; // unified addressing
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemHostGetFlags(unsigned int *flags, void *p) {
    (void)p;
    if (!flags)
        return CUDA_ERROR_INVALID_VALUE;
    *flags = 0;
    return CUDA_SUCCESS;
}

#if defined(_WIN32)
/* The 32-bit process runs out of contiguous address space once the device
 * arena has grown to ~1.4 GB, so a later 128 MB VirtualAlloc fails and
 * PhysX dereferences the failure.  Reserve a host heap early (on the first
 * host allocation) and suballocate/decommit from it.  Address space is
 * reserved once; commit charge is released on cuMemFreeHost. */
#define ACUDA_HOST_HEAP_RESERVE (160u << 20)
/* Fallback fixed base if no above-2 GiB window exists.  A heap reserved at
 * the first free low address fragments the region the NVIDIA OpenGL driver
 * needs for its Optimus DxPresent surface, which then fails with OOM. */
#define ACUDA_HOST_HEAP_BASE ((void *)(uintptr_t)0x70000000u)
#define ACUDA_HOST_DESC_MAX 256
typedef struct acuda_host_desc {
    size_t off;
    size_t size;
    int free;
} acuda_host_desc_t;
static unsigned char *g_host_base;
static size_t g_host_size;
static acuda_host_desc_t g_host_desc[ACUDA_HOST_DESC_MAX];
static int g_host_desc_n;
static int g_host_init_done;

/* Find the highest free block that can hold the heap.  When `above_2gb`
 * is set, only blocks fully above 0x80000000 are considered (the LAA
 * region a large-address-aware 32-bit process has on 64-bit Windows);
 * otherwise only blocks fully below 0x80000000 qualify.  The host heap
 * is plain CPU staging memory with no GPU-visibility requirement, so
 * placing it above 2 GiB keeps the low region - where the NVIDIA OpenGL
 * driver and PhysX must allocate (depth FBOs, DxPresent surfaces, CRT
 * heap) - unfragmented.  On a non-LAA process no above-2 GiB block
 * exists and the caller falls back below. */
static void *acuda_host_heap_find_high(size_t reserve, int above_2gb) {
    MEMORY_BASIC_INFORMATION mbi;
    unsigned char *p = (unsigned char *)0x10000;
    uintptr_t best = 0;
    while ((uintptr_t)p < (uintptr_t)0xFFFFFFFFu) {
        uintptr_t base, end;
        unsigned char *next;
        if (VirtualQuery(p, &mbi, sizeof(mbi)) != sizeof(mbi))
            break;
        if (mbi.State == MEM_FREE && mbi.RegionSize >= reserve) {
            base = ((uintptr_t)mbi.BaseAddress + 0xFFFFu) & ~(uintptr_t)0xFFFFu;
            end = (uintptr_t)mbi.BaseAddress + mbi.RegionSize;
            if (above_2gb) {
                if (base >= (uintptr_t)0x80000000u &&
                    base + reserve >= base && base > best)
                    best = base;
            } else if (base + reserve <= (uintptr_t)0x80000000u &&
                       base > best) {
                best = base;
            }
        }
        next = (unsigned char *)mbi.BaseAddress + mbi.RegionSize;
        if (next <= p)
            break;
        p = next;
    }
    return best ? (void *)best : NULL;
}

 int acuda_host_heap_init(size_t need) {
    size_t reserve;
    const char *noheap;
    const char *heap_mb;
    if (g_host_init_done)
        return g_host_base != NULL;
    g_host_init_done = 1;
    /* Diagnostics: a lower/no reservation leaves more of the 32-bit
     * address space to the NVIDIA OpenGL driver (depth FBOs). */
    noheap = getenv("ACUDA_NO_HOST_HEAP");
    if (noheap && *noheap)
        return 0;
    reserve = ACUDA_HOST_HEAP_RESERVE;
    heap_mb = getenv("ACUDA_HOST_HEAP_MB");
    if (heap_mb && *heap_mb) {
        unsigned long mb = strtoul(heap_mb, NULL, 0);
        reserve = mb ? (size_t)mb << 20 : 0;
        if (!reserve)
            return 0;
    }
    if (need > reserve)
        reserve = ((need + 0xFFFFu) & ~(size_t)0xFFFFu) + (16u << 20);
    /* Prefer the LAA region above 2 GiB so the low 2 GiB stays contiguous
     * for the NVIDIA driver and PhysX.  Fall back to the previous order
     * (fixed base, highest free below 2 GiB, anywhere) on failure or on
     * non-LAA processes where no above-2 GiB window exists. */
    {
        void *high = acuda_host_heap_find_high(reserve, 1);
        if (high) {
            g_host_base = (unsigned char *)VirtualAlloc(high, reserve,
                                                        MEM_RESERVE,
                                                        PAGE_NOACCESS);
            if (g_host_base)
                ACUDA_DEBUG_PRINT(
                        "acuda-trace .. host_heap reserve %zu above 2GiB at %p\n",
                        reserve, (void *)g_host_base);
        }
    }
    if (!g_host_base)
        g_host_base = (unsigned char *)VirtualAlloc(ACUDA_HOST_HEAP_BASE, reserve,
                                                    MEM_RESERVE, PAGE_NOACCESS);
    if (!g_host_base) {
        void *high = acuda_host_heap_find_high(reserve, 0);
        if (high)
            g_host_base = (unsigned char *)VirtualAlloc(high, reserve,
                                                        MEM_RESERVE,
                                                        PAGE_NOACCESS);
    }
    if (!g_host_base)
        g_host_base = (unsigned char *)VirtualAlloc(NULL, reserve, MEM_RESERVE,
                                                    PAGE_NOACCESS);
    if (!g_host_base)
        return 0;
    g_host_size = reserve;
    g_host_desc[0].off = 0;
    g_host_desc[0].size = reserve;
    g_host_desc[0].free = 1;
    g_host_desc_n = 1;
    return 1;
}

static void *acuda_host_heap_alloc(size_t bytes) {
    size_t need = (bytes + 0xFFFu) & ~(size_t)0xFFFu;
    int best = -1;
    void *p;
    if (need == 0)
        return NULL;
    if (!acuda_host_heap_init(need))
        return NULL;
    for (int i = 0; i < g_host_desc_n; i++) {
        if (g_host_desc[i].free && g_host_desc[i].size >= need) {
            if (best < 0 || g_host_desc[i].size < g_host_desc[best].size)
                best = i;
            if (g_host_desc[i].size == need)
                break;
        }
    }
    if (best < 0)
        return NULL;
    p = g_host_base + g_host_desc[best].off;
    if (!VirtualAlloc(p, need, MEM_COMMIT, PAGE_READWRITE))
        return NULL;
    if (g_host_desc[best].size > need && g_host_desc_n < ACUDA_HOST_DESC_MAX) {
        for (int j = g_host_desc_n; j > best + 1; j--)
            g_host_desc[j] = g_host_desc[j - 1];
        g_host_desc_n++;
        g_host_desc[best + 1].off = g_host_desc[best].off + need;
        g_host_desc[best + 1].size = g_host_desc[best].size - need;
        g_host_desc[best + 1].free = 1;
        g_host_desc[best].size = need;
    }
    g_host_desc[best].free = 0;
    ACUDA_DEBUG_PRINT("acuda-trace .. host_heap alloc %zu at %p\n", bytes, p);
    return p;
}

static int acuda_host_heap_free(void *p) {
    size_t off;
    int i;
    if (!g_host_base || !p)
        return 0;
    if ((unsigned char *)p < g_host_base ||
        (unsigned char *)p >= g_host_base + g_host_size)
        return 0;
    off = (size_t)((unsigned char *)p - g_host_base);
    for (i = 0; i < g_host_desc_n; i++) {
        if (g_host_desc[i].off == off && !g_host_desc[i].free) {
            g_host_desc[i].free = 1;
            if (g_host_desc[i].size)
                VirtualFree(p, g_host_desc[i].size, MEM_DECOMMIT);
            if (i + 1 < g_host_desc_n && g_host_desc[i + 1].free) {
                g_host_desc[i].size += g_host_desc[i + 1].size;
                memmove(&g_host_desc[i + 1], &g_host_desc[i + 2],
                        (size_t)(g_host_desc_n - i - 2) * sizeof(g_host_desc[0]));
                g_host_desc_n--;
            }
            if (i > 0 && g_host_desc[i - 1].free) {
                g_host_desc[i - 1].size += g_host_desc[i].size;
                memmove(&g_host_desc[i], &g_host_desc[i + 1],
                        (size_t)(g_host_desc_n - i - 1) * sizeof(g_host_desc[0]));
                g_host_desc_n--;
            }
            ACUDA_DEBUG_PRINT("acuda-trace .. host_heap free %p\n", p);
            return 1;
        }
    }
    return 0;
}
#endif

// Pinned host memory.  Prefer the ACuda arena so the pointer is
// registered for launch binding, acuda_arena_capacity/host lookups, and
// zero-copy kernel access. Fall back to host heap/OS allocation for
// contexts where the arena is unavailable.
CUresult ACUDA_CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize, unsigned int flags) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemHostAlloc(%zu flags=%x)\n",
                bytesize, flags);
    if (!pp)
        return CUDA_ERROR_INVALID_VALUE;
    *pp = NULL;
    if (bytesize == 0)
        return CUDA_ERROR_INVALID_VALUE;
    CUcontext c = NULL;
    if (need_ctx(&c) == CUDA_SUCCESS) {
        uint64_t virt = 0;
        VkBuffer buf = VK_NULL_HANDLE;
        if (acuda_arena_alloc(bytesize, &virt, &buf) == 0) {
            *pp = (void *)(uintptr_t)virt;
            return CUDA_SUCCESS;
        }
    }
#if defined(_WIN32)
    void *p = acuda_host_heap_alloc(bytesize);
    if (!p)
        p = VirtualAlloc(NULL, bytesize, MEM_COMMIT | MEM_RESERVE,
                         PAGE_READWRITE);
#else
    void *p = malloc(bytesize);
#endif
    if (!p) {
        ACUDA_DEBUG_PRINT("acuda-trace .. cuMemHostAlloc(%zu) -> OOM\n",
                    bytesize);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *pp = p;
#if defined(_WIN32) && !defined(_WIN64)
    acuda_trace_va("cuMemHostAlloc");
#endif
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemHostAlloc_v2(void **pp, size_t bytesize, unsigned int flags) {
    return cuMemHostAlloc(pp, bytesize, flags);
}

CUresult ACUDA_CUDAAPI cuMemAllocHost(void **pp, size_t bytesize) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemAllocHost(%zu)\n", bytesize);
    return cuMemHostAlloc(pp, bytesize, 0);
}

CUresult ACUDA_CUDAAPI cuMemAllocHost_v2(void **pp, size_t bytesize) {
    return cuMemAllocHost(pp, bytesize);
}

CUresult ACUDA_CUDAAPI cuMemFreeHost(void *p) {
    if (!p)
        return CUDA_SUCCESS;
    if (acuda_arena_free((uint64_t)(uintptr_t)p) == 0)
        return CUDA_SUCCESS;
#if defined(_WIN32)
    if (!acuda_host_heap_free(p))
        VirtualFree(p, 0, MEM_RELEASE);
#else
    free(p);
#endif
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemHostRegister(void *p, size_t bytesize, unsigned int flags) {
    (void)p;
    (void)bytesize;
    (void)flags;
    // All host memory is already accessible to the arena copies.
    return p ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult ACUDA_CUDAAPI cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int flags) {
    return cuMemHostRegister(p, bytesize, flags);
}

CUresult ACUDA_CUDAAPI cuMemHostUnregister(void *p) {
    return p ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult ACUDA_CUDAAPI cuMemHostGetDevicePointer_v2(void **pdst, void *p,
                                                    unsigned int flags) {
    return cuMemHostGetDevicePointer(pdst, p, flags);
}
