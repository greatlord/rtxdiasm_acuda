#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include "arena.h"

typedef struct {
    size_t offset, size; // free span within arena
} span_t;

typedef struct {
    int committed;
    VkDeviceMemory mem;
    void *mapped;
} arena_t;

typedef struct {
    uint64_t virt;
    size_t size;
    VkBuffer buf;
    int arena;
} alloc_t;

static acuda_context_t g_ctx;
static unsigned char *g_virt; // PROT_NONE reservation, ARENA_MAX * SIZE
#if ACUDA_ARENA_SLOTS > ACUDA_ARENA_MAX
static unsigned char *g_big_virt; // PROT_NONE reservation for the big slot
#endif
static arena_t g_arenas[ACUDA_ARENA_SLOTS];
static span_t *g_spans[ACUDA_ARENA_SLOTS];
static size_t g_nspans[ACUDA_ARENA_SLOTS], g_span_cap[ACUDA_ARENA_SLOTS];
static alloc_t *g_allocs;
static size_t g_nallocs, g_acap;
static size_t g_live;

static int commit_arena(int idx);

// Size of one arena slot's virtual window.
static size_t arena_vsize(int idx) {
#if ACUDA_ARENA_SLOTS > ACUDA_ARENA_MAX
    if (idx == ACUDA_ARENA_BIG_INDEX)
        return (size_t)ACUDA_ARENA_BIG_SIZE;
#endif
    (void)idx;
    return (size_t)ACUDA_ARENA_SIZE;
}

// Base virtual address of one arena slot's window.
static uint64_t arena_vbase(int idx) {
#if ACUDA_ARENA_SLOTS > ACUDA_ARENA_MAX
    if (idx == ACUDA_ARENA_BIG_INDEX)
        return (uint64_t)(uintptr_t)g_big_virt;
#endif
    return (uint64_t)(uintptr_t)g_virt + (uint64_t)(unsigned)idx * (uint64_t)ACUDA_ARENA_SIZE;
}

int acuda_arena_boot(acuda_context_t ctx) {
    g_ctx = ctx;
    if (!g_virt) {
        size_t bytes = ACUDA_ARENA_SIZE * ACUDA_ARENA_MAX;
#if defined(_WIN32)
        void *p = VirtualAlloc(NULL, bytes, MEM_RESERVE, PAGE_NOACCESS);
#else
        void *p = mmap(NULL, bytes, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED)
            p = NULL;
#endif
        if (!p)
            return -1;
        g_virt = p;
    }
    // Commit the small-object arena and the large-object arena now, while
    // the 32-bit address space is still unfragmented.  Deferring a 32 MiB
    // host-visible commit until the fluid solver asks for its first large
    // buffer makes it fail with VK_ERROR_OUT_OF_DEVICE_MEMORY even though
    // the heap is empty: the loader cannot find a free 32 MiB window in
    // the low 2 GiB anymore.  Committing all four slots up front leaves
    // too little address space for PhysX's own 16 MiB host buffer.
    for (int i = 0; i < 2 && i < ACUDA_ARENA_MAX; i++) {
        if (!g_arenas[i].committed)
            (void)commit_arena(i); // best effort; retried on demand
    }
#if ACUDA_ARENA_SLOTS > ACUDA_ARENA_MAX
    // Reserve the big slot above 2 GiB when the process is large-address
    // aware so the low region stays contiguous for the NVIDIA driver.
    // Committed eagerly: a late 256 MiB host-visible commit would face
    // the same fragmentation the small arenas avoid by booting early.
    // ACUDA_BIG_ARENA_LOW=1 forces the legacy anywhere-placement for
    // A/B testing (some 2011-era code mis-handles pointers >= 2 GiB).
    {
        size_t bytes = (size_t)ACUDA_ARENA_BIG_SIZE;
        void *p = NULL;
        int force_low = 0;
        {
            const char *s = getenv("ACUDA_BIG_ARENA_LOW");
            force_low = (s && *s) ? 1 : 0;
        }
#if defined(_WIN32)
        if (!force_low) {
            MEMORY_BASIC_INFORMATION mbi;
            unsigned char *q = (unsigned char *)0x80000000u;
            uintptr_t best = 0;
            while ((uintptr_t)q < (uintptr_t)0xFFFFFFFFu) {
                if (VirtualQuery(q, &mbi, sizeof(mbi)) != sizeof(mbi))
                    break;
                if (mbi.State == MEM_FREE && mbi.RegionSize >= bytes)
                    best = (uintptr_t)mbi.BaseAddress;
                {
                    unsigned char *next =
                        (unsigned char *)mbi.BaseAddress + mbi.RegionSize;
                    if (next <= q)
                        break;
                    q = next;
                }
            }
            if (best)
                p = VirtualAlloc((void *)best, bytes, MEM_RESERVE,
                                 PAGE_NOACCESS);
        }
        if (!p)
            p = VirtualAlloc(NULL, bytes, MEM_RESERVE, PAGE_NOACCESS);
#else
        (void)force_low;
        p = mmap(NULL, bytes, PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED)
            p = NULL;
#endif
        if (p) {
            g_big_virt = p;
            (void)commit_arena(ACUDA_ARENA_BIG_INDEX); // best effort
        }
    }
#endif
    return 0;
}

void acuda_arena_shutdown(void) {
    // Buffers freed per-allocation; arenas released here.
    if (!g_ctx)
        return;
    for (int i = 0; i < ACUDA_ARENA_SLOTS; i++) {
        if (g_arenas[i].committed) {
            vkUnmapMemory(g_ctx->device, g_arenas[i].mem);
            vkFreeMemory(g_ctx->device, g_arenas[i].mem, NULL);
        }
        free(g_spans[i]);
        g_spans[i] = NULL;
        g_nspans[i] = 0;
        g_span_cap[i] = 0;
        g_arenas[i].committed = 0;
        g_arenas[i].mem = VK_NULL_HANDLE;
    }
    free(g_allocs);
    g_allocs = NULL;
    g_nallocs = g_acap = 0;
    g_live = 0;
#if defined(_WIN32)
    if (g_virt)
        VirtualFree(g_virt, 0, MEM_RELEASE);
#else
    if (g_virt)
        munmap(g_virt, ACUDA_ARENA_SIZE * ACUDA_ARENA_MAX);
#endif
    g_virt = NULL;
#if ACUDA_ARENA_SLOTS > ACUDA_ARENA_MAX
#if defined(_WIN32)
    if (g_big_virt)
        VirtualFree(g_big_virt, 0, MEM_RELEASE);
#else
    if (g_big_virt)
        munmap(g_big_virt, (size_t)ACUDA_ARENA_BIG_SIZE);
#endif
    g_big_virt = NULL;
#endif
}

static int commit_arena(int idx) {
    // A 32-bit client can be too fragmented late in a run for a large
    // host-visible Vulkan allocation even though the heap has room: the
    // loader has to map the whole allocation into the low 2 GiB.  Retry
    // progressively smaller chunks instead of failing the first CUDA
    // allocation outright.  The virtual window stays at the arena's slot
    // size; only the committed span shrinks.
    size_t try_size = arena_vsize(idx);
    while (try_size >= (4ull << 20)) {
        VkBufferCreateInfo bci = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = try_size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     (g_ctx->bda_ok ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0),
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VkBuffer tmp = VK_NULL_HANDLE;
        if (vkCreateBuffer(g_ctx->device, &bci, NULL, &tmp) != VK_SUCCESS) {
            try_size >>= 1;
            continue;
        }
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(g_ctx->device, tmp, &req);
        vkDestroyBuffer(g_ctx->device, tmp, NULL);
        if (req.size > arena_vsize(idx)) {
            // Alignment pushed the allocation past this arena's slot in
            // the shared virtual window; try the next smaller chunk.
            try_size >>= 1;
            continue;
        }
        VkMemoryAllocateFlagsInfo addr_flags = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
        };
        VkMemoryAllocateInfo mai = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = req.size,
            .memoryTypeIndex = g_ctx->mem_type,
            .pNext = g_ctx->bda_ok ? &addr_flags : NULL,
        };
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkResult ar = vkAllocateMemory(g_ctx->device, &mai, NULL, &mem);
        if (ar != VK_SUCCESS) {
            if (getenv("ACUDA_TRACE"))
                fprintf(stderr,
                        "acuda-arena commit[%d] vkAllocateMemory failed "
                        "size=%llu type=%u r=%d\n",
                        idx, (unsigned long long)req.size,
                        (unsigned)g_ctx->mem_type, (int)ar);
            try_size >>= 1;
            continue;
        }
        void *mapped = NULL;
        VkResult mr = vkMapMemory(g_ctx->device, mem, 0, req.size, 0, &mapped);
        if (mr != VK_SUCCESS) {
            if (getenv("ACUDA_TRACE"))
                fprintf(stderr,
                        "acuda-arena commit[%d] vkMapMemory failed "
                        "size=%llu r=%d\n",
                        idx, (unsigned long long)req.size, (int)mr);
            vkFreeMemory(g_ctx->device, mem, NULL);
            try_size >>= 1;
            continue;
        }
        g_arenas[idx].mem = mem;
        g_arenas[idx].mapped = mapped;
        g_arenas[idx].committed = 1;
        g_spans[idx] = malloc(sizeof(span_t));
        if (!g_spans[idx]) {
            vkUnmapMemory(g_ctx->device, mem);
            vkFreeMemory(g_ctx->device, mem, NULL);
            g_arenas[idx].committed = 0;
            return -1;
        }
        g_spans[idx][0].offset = 0;
        g_spans[idx][0].size = req.size;
        g_nspans[idx] = 1;
        g_span_cap[idx] = 1;
        if (getenv("ACUDA_TRACE") && req.size != arena_vsize(idx))
            fprintf(stderr,
                    "acuda-arena commit[%d] shrank to %llu bytes\n",
                    idx, (unsigned long long)req.size);
        return 0;
    }
    return -1;
}

static size_t align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

int acuda_arena_alloc(size_t size, uint64_t *virt_out, VkBuffer *buf_out) {
    if (!g_ctx || size == 0)
        return -1;
    size = align_up(size, ACUDA_ARENA_ALIGN);
    // Arena 1 is reserved for large objects: keep it empty so a 32 MiB
    // cuMemAlloc has a contiguous span that is not fragmented by the
    // many small kernel global/constant allocations.  Small objects use
    // arena 0 and then the on-demand arenas; large objects try arena 1
    // first and may still fall back to the others.
    int order[ACUDA_ARENA_SLOTS];
    int norder = 0;
#if ACUDA_ARENA_SLOTS > ACUDA_ARENA_MAX
    // Requests that cannot fit a standard slot go to the big slot only;
    // keep it pristine for the PhysX-style 256 MiB heap.
    if (size > (size_t)ACUDA_ARENA_SIZE) {
        order[norder++] = ACUDA_ARENA_BIG_INDEX;
    } else
#endif
    if (size >= (16u << 20)) {
        if (ACUDA_ARENA_MAX > 1)
            order[norder++] = 1;
        order[norder++] = 0;
        for (int i = 2; i < ACUDA_ARENA_MAX; i++)
            order[norder++] = i;
    } else {
        order[norder++] = 0;
        for (int i = 2; i < ACUDA_ARENA_MAX; i++)
            order[norder++] = i;
        if (ACUDA_ARENA_MAX > 1)
            order[norder++] = 1; // last resort
    }
    for (int oi = 0; oi < norder; oi++) {
        int i = order[oi];
        if (!g_arenas[i].committed) {
            if (commit_arena(i) != 0)
                continue;
        }
        for (size_t s = 0; s < g_nspans[i]; s++) {
            size_t off = align_up(g_spans[i][s].offset, ACUDA_ARENA_ALIGN);
            size_t pad = off - g_spans[i][s].offset;
            if (g_spans[i][s].size < pad + size)
                continue;
            VkBufferCreateInfo bci = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         (g_ctx->bda_ok
                              ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0),
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };
            VkBuffer buf = VK_NULL_HANDLE;
            if (vkCreateBuffer(g_ctx->device, &bci, NULL, &buf) != VK_SUCCESS) {
                if (getenv("ACUDA_TRACE")) fprintf(stderr, "acuda-arena vkCreateBuffer failed size=%zu\n", size);
                return -1;
            }
            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(g_ctx->device, buf, &req);
            size_t aligned = align_up(off, req.alignment > 0 ? req.alignment : 1);
            size_t pad2 = aligned - g_spans[i][s].offset;
            if (g_spans[i][s].size < pad2 + size) {
                vkDestroyBuffer(g_ctx->device, buf, NULL);
                continue;
            }
            if (vkBindBufferMemory(g_ctx->device, buf, g_arenas[i].mem,
                                   aligned) != VK_SUCCESS) {
                if (getenv("ACUDA_TRACE")) fprintf(stderr, "acuda-arena vkBindBufferMemory failed\n");
                vkDestroyBuffer(g_ctx->device, buf, NULL);
                return -1;
            }
            // Carve [aligned, aligned+size) out of the span.
            size_t span_end = g_spans[i][s].offset + g_spans[i][s].size;
            size_t alloc_end = aligned + size;
            if (aligned == g_spans[i][s].offset) {
                g_spans[i][s].offset = alloc_end;
                g_spans[i][s].size = span_end - alloc_end;
                if (g_spans[i][s].size == 0) {
                    memmove(&g_spans[i][s], &g_spans[i][s + 1],
                            sizeof(span_t) * (g_nspans[i] - s - 1));
                    g_nspans[i]--;
                }
            } else {
                // Split: shrink current span to the head fragment.
                size_t head = aligned - g_spans[i][s].offset;
                g_spans[i][s].size = head;
                if (alloc_end < span_end) {
                    // Append tail fragment (grow list if needed).
                    if (g_nspans[i] == g_span_cap[i]) {
                        size_t ncap = g_span_cap[i] * 2;
                        span_t *ns = malloc(sizeof(*ns) * ncap);
                        if (!ns) {
                            vkDestroyBuffer(g_ctx->device, buf, NULL);
                            // NOTE: span left shrunk (leaks head pad); acceptable
                            // on OOM path only.
                            return -1;
                        }
                        memcpy(ns, g_spans[i], sizeof(*ns) * g_nspans[i]);
                        free(g_spans[i]);
                        g_spans[i] = ns;
                        g_span_cap[i] = ncap;
                    }
                    g_spans[i][g_nspans[i]].offset = alloc_end;
                    g_spans[i][g_nspans[i]].size = span_end - alloc_end;
                    g_nspans[i]++;
                }
            }
            if (g_nallocs == g_acap) {
                size_t ncap = g_acap ? g_acap * 2 : 16;
                alloc_t *na = malloc(sizeof(*na) * ncap);
                if (!na) {
                    vkDestroyBuffer(g_ctx->device, buf, NULL);
                    return -1;
                }
                memcpy(na, g_allocs, sizeof(*na) * g_nallocs);
                free(g_allocs);
                g_allocs = na;
                g_acap = ncap;
            }
            uint64_t virt = arena_vbase(i) + aligned;
            g_allocs[g_nallocs].virt = virt;
            g_allocs[g_nallocs].size = size;
            g_allocs[g_nallocs].buf = buf;
            g_allocs[g_nallocs].arena = i;
            g_nallocs++;
            g_live += size;
            *virt_out = virt;
            *buf_out = buf;
            return 0;
        }
    }
    return -1;
}

static alloc_t *find_alloc(uint64_t virt, size_t size) {
    for (size_t i = 0; i < g_nallocs; i++) {
        if (virt >= g_allocs[i].virt &&
            virt + size <= g_allocs[i].virt + g_allocs[i].size)
            return &g_allocs[i];
    }
    return NULL;
}

int acuda_arena_free(uint64_t virt) {
    for (size_t i = 0; i < g_nallocs; i++) {
        if (g_allocs[i].virt == virt) {
            int ai = g_allocs[i].arena;
            size_t off = (size_t)(virt - arena_vbase(ai));
            size_t size = g_allocs[i].size;
            vkDestroyBuffer(g_ctx->device, g_allocs[i].buf, NULL);
            // Return span, coalescing neighbors.
            size_t lo = off, hi = off + size;
            size_t w = 0;
            for (size_t s = 0; s < g_nspans[ai]; s++) {
                size_t so = g_spans[ai][s].offset;
                size_t se = so + g_spans[ai][s].size;
                if (se == lo) {
                    lo = so;
                    continue;
                }
                if (so == hi) {
                    hi = se;
                    continue;
                }
                g_spans[ai][w++] = g_spans[ai][s];
            }
            if (w == g_nspans[ai]) {
                if (g_nspans[ai] == g_span_cap[ai]) {
                    size_t ncap = g_span_cap[ai] * 2;
                    span_t *ns = malloc(sizeof(*ns) * ncap);
                    if (!ns)
                        return -1;
                    memcpy(ns, g_spans[ai], sizeof(*ns) * g_nspans[ai]);
                    free(g_spans[ai]);
                    g_spans[ai] = ns;
                    g_span_cap[ai] = ncap;
                }
                g_spans[ai][w].offset = lo;
                g_spans[ai][w].size = hi - lo;
                w++;
            } else {
                // Coalesced away existing span(s); append merged.
                g_spans[ai][w].offset = lo;
                g_spans[ai][w].size = hi - lo;
                w++;
            }
            g_nspans[ai] = w;
            g_live -= size;
            memmove(&g_allocs[i], &g_allocs[i + 1],
                    sizeof(alloc_t) * (g_nallocs - i - 1));
            g_nallocs--;
            return 0;
        }
    }
    return -1;
}

int acuda_arena_host(uint64_t virt, size_t size, void **host_out) {
    alloc_t *a = find_alloc(virt, size);
    if (!a)
        return -1;
    size_t base = (size_t)arena_vbase(a->arena);
    *host_out = (unsigned char *)g_arenas[a->arena].mapped + (virt - base);
    return 0;
}

int acuda_arena_bind(uint64_t virt, size_t size, VkBuffer *buf_out,
                     size_t *offset_out) {
    alloc_t *a = find_alloc(virt, size);
    if (!a)
        return -1;
    *buf_out = a->buf;
    // The buffer starts at the allocation base; the descriptor carries
    // the intra-allocation offset.
    *offset_out = (size_t)(virt - a->virt);
    return 0;
}

int acuda_arena_capacity(uint64_t virt, size_t *remain_out) {
    for (size_t i = 0; i < g_nallocs; i++) {
        if (virt >= g_allocs[i].virt &&
            virt < g_allocs[i].virt + g_allocs[i].size) {
            *remain_out = (size_t)(g_allocs[i].virt + g_allocs[i].size - virt);
            return 0;
        }
    }
    return -1;
}

size_t acuda_arena_live(void) {
    return g_live;
}
