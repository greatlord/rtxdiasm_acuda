#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_dd9e98a4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* Surface dummy helpers (moved from driver.c for launch). */
typedef struct {
    int used, dim, arrayed, isint;
    acuda_buffer_t img;
} driver_dummy_t;
static driver_dummy_t g_surf_dummies[ACUDA_SURF_TABLE_MAX_TYPES];

static acuda_buffer_t driver_surf_dummy(int dim, int arrayed, int isint) {
    for (int i = 0; i < ACUDA_SURF_TABLE_MAX_TYPES; i++)
        if (g_surf_dummies[i].used &&
            g_surf_dummies[i].dim == dim &&
            g_surf_dummies[i].arrayed == arrayed &&
            g_surf_dummies[i].isint == isint)
            return g_surf_dummies[i].img;
    int slot = -1;
    for (int i = 0; i < ACUDA_SURF_TABLE_MAX_TYPES; i++)
        if (!g_surf_dummies[i].used) {
            slot = i;
            break;
        }
    if (slot < 0)
        return NULL;
    acuda_image_desc d = {
        .width = 1,
        .height = 1,
        .depth = 1,
        .layers = arrayed ? 2u : 1u,
        .levels = 1,
        .samples = 1,
        .format = isint ? 3u : 0u,
        .dim = dim + 1,
        .cube = 0,
    };
    acuda_buffer_t img = NULL;
    if (acuda_surface_create(g_actx, &img, &d, NULL) != ACUDA_SUCCESS)
        return NULL;
    g_surf_dummies[slot].used = 1;
    g_surf_dummies[slot].dim = dim;
    g_surf_dummies[slot].arrayed = arrayed;
    g_surf_dummies[slot].isint = isint;
    g_surf_dummies[slot].img = img;
    return img;
}

// Six words consumed by surface suq (geometry, levels/samples,
// channel type/order); mirrors the surface branch of tex_meta_words.
static void driver_surf_meta(acuda_buffer_t h, uint32_t out[6]) {
    memset(out, 0, 6 * sizeof(uint32_t));
    if (!h)
        return;
    out[0] = h->width;
    out[1] = h->dim1d ? 1u : h->height;
    out[2] = h->dim3d ? h->depth : 1u;
    uint32_t asz = 0;
    if (!h->dim3d) {
        if (h->cube) {
            if (h->layers > 6)
                asz = h->layers / 6;
        } else if (h->layers > 1) {
            asz = h->layers;
        }
    }
    out[3] = asz;
    uint32_t cdt = 2, order = 1;
    switch (h->format) {
    case 0: cdt = 2; order = 3; break; // RGBA32F
    case 1: cdt = 1; order = 3; break; // RGBA8 UNORM
    case 2: cdt = 2; order = 1; break; // R32F
    case 3: cdt = 1; order = 1; break; // R32_UINT
    }
    out[4] = (cdt & 0xffu) | (order << 8);
}

/* Default host-call stub for driver API launches. */
static int driver_host_identity(void *user, const uint8_t *args,
                                size_t args_bytes, uint8_t *ret,
                                size_t ret_bytes) {
    (void)user;
    memset(ret, 0, ret_bytes);
    if (args_bytes > ret_bytes)
        args_bytes = ret_bytes;
    memcpy(ret, args, args_bytes);
    return 0;
}

static void release_param_buffers(acuda_buffer_t *bufs, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        if (bufs[i]) {
            acuda_free(g_actx, bufs[i]);
            bufs[i] = NULL;
        }
    }
}
CUresult ACUDA_CUDAAPI cuLaunchKernelEx(const void *config, CUfunction f,
                          void **kernelParams, void **extra) {
    if (!config || !f)
        return CUDA_ERROR_INVALID_VALUE;
    uint32_t grid[3], block[3], shm = 0, nattr = 0;
    void *stream = NULL;
    memcpy(grid, (const char *)config + 0, sizeof(grid));
    memcpy(block, (const char *)config + 12, sizeof(block));
    memcpy(&shm, (const char *)config + 24, sizeof(shm));
    memcpy(&stream, (const char *)config + 32, sizeof(stream));
    memcpy(&nattr, (const char *)config + 48, sizeof(nattr));
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (nattr != 0)
        return CUDA_ERROR_INVALID_VALUE; // cluster/etc. unsupported
    return cuLaunchKernel(f, grid[0], grid[1], grid[2], block[0], block[1],
                          block[2], shm, (CUstream)stream, kernelParams,
                          extra);
}

static CUresult ensure_func(CUfunction f) {
    if (!f)
        return CUDA_ERROR_INVALID_HANDLE;
    if (f->k)
        return CUDA_SUCCESS;
    CUfunction tmp = NULL;
    CUresult r = cuModuleGetFunction(&tmp, f->mod, f->name);
    ACUDA_DEBUG_PRINT("acuda-lazy compile %s rc=%d tmp=%p k=%p\n",
                f->name, r, (void *)tmp, tmp ? (void *)tmp->k : NULL);
    if (r != CUDA_SUCCESS)
        return r;
    f->k = tmp->k;
    f->maxdyn_set = tmp->maxdyn_set;
    f->maxdyn = tmp->maxdyn;
    free(tmp);
    return CUDA_SUCCESS;
}

// A zeroed dummy buffer bound in place of a null buffer argument (for
// example layer-norm's optional gamma/beta).  The shader's null check
// still sees a zero device address; the descriptor only has to be
// valid for the pipeline layout.
#define ACUDA_NULL_PARAM_BYTES 65536u
static acuda_buffer_t g_null_param_buffer;

static acuda_status_t ensure_null_param_buffer(void) {
    if (g_null_param_buffer)
        return ACUDA_SUCCESS;
    acuda_status_t st = acuda_malloc_host(
        g_actx, &g_null_param_buffer, ACUDA_NULL_PARAM_BYTES);
    if (st != ACUDA_SUCCESS) {
        g_null_param_buffer = NULL;
        return st;
    }
    void *mapped = NULL;
    if (vkMapMemory(g_actx->device, g_null_param_buffer->memory, 0,
                    VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
        memset(mapped, 0, ACUDA_NULL_PARAM_BYTES);
        VkMappedMemoryRange range = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = g_null_param_buffer->memory,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        vkFlushMappedMemoryRanges(g_actx->device, 1, &range);
        vkUnmapMemory(g_actx->device, g_null_param_buffer->memory);
    }
    g_null_param_buffer->gpu_dirty = 1;
    return ACUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra) {
    {
        static FILE *_dtfp = NULL; static int _dtinit = 0; static long long _dtseq = 0;
        if (!_dtinit) { _dtinit = 1; const char *_dtp = getenv("ACUDA_DRIVER_TRACE_FILE"); if (_dtp && _dtp[0]) _dtfp = fopen(_dtp, "w"); }
        if (_dtfp) { fprintf(_dtfp, "%lld cuLaunchKernel 0 f=%p g=%ux%ux%u b=%ux%ux%u s=%p k=%p e=%p\n", _dtseq++, (void*)f, gridDimX,gridDimY,gridDimZ, blockDimX,blockDimY,blockDimZ, (void*)hStream, (void*)kernelParams, (void*)extra); if((_dtseq&0x3f)==0) fflush(_dtfp); }
    }
    if (!f)
        { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
    {
        CUfunction real = NULL;
        CUresult hr = handle_as_func((void *)f, &real);
        if (hr != CUDA_SUCCESS)
            return hr;
        f = real;
    }
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    acuda_dd9e98a4_check_ready_canary("cuLaunchKernel_entry");
    ACUDA_DEBUG_PRINT("acuda-trace >> cuLaunchKernel %s\n", f->name);
    {
        // Diagnostic: skip named kernels entirely (pretend success) to
        // isolate a hanging launch. Comma-separated substring list.
        const char *skip = getenv("ACUDA_SKIP_KERNEL");
        if (skip && skip[0]) {
            char buf[1024];
            snprintf(buf, sizeof(buf), "%s", skip);
            for (char *tok = strtok(buf, ","); tok;
                 tok = strtok(NULL, ",")) {
                if (strstr(f->name, tok) != NULL) {
                    ACUDA_DEBUG_PRINT("acuda-trace .. cuLaunchKernel %s SKIPPED\n",
                                f->name);
                    return CUDA_SUCCESS;
                }
            }
        }
    }
    r = ensure_func(f);
    if (r != CUDA_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel ensure_func %s "
                    "rc=%d\n", f->name, (int)r);
        return r;
    }
    /* One parameter-model fetch serves the launch-info log, the packed
     * argument walk, and the launch below.  acuda_kernel_info_t is
     * ~283 KiB (per-parameter name tables); keep exactly one live copy
     * — worker threads that drive PhysX do not have stack to spare. */
    const acuda_kernel_info_t *info = acuda_kernel_info_ptr(f->k);
    if (!info)
        { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_UNKNOWN, __LINE__); return CUDA_ERROR_UNKNOWN; }
    if (getenv("ACUDA_TRACE")) {
        unsigned stackleft = 0;
#if defined(_WIN32)
        ULONG_PTR low = 0, high = 0;
        GetCurrentThreadStackLimits(&low, &high);
        stackleft = (unsigned)((high - (ULONG_PTR)&low) / 1024u);
#endif
        ACUDA_DEBUG_PRINT("acuda-launch-info %s params=%u ptr=%u bda=%u "
                "grid=%u,%u,%u block=%u,%u,%u shm=%u kparams=%p extra=%p "
                "stackleft=%uk\n",
                info->entry, info->num_params, info->num_ptr_params, info->bda,
                gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                blockDimZ, sharedMemBytes, (void *)kernelParams,
                (void *)extra, stackleft);
    }
    void **kparams = kernelParams;
    void *argv_buf[ACUDA_MAX_PARAMS]; // synthetic argv for packed buffers
    if (extra != NULL) {
        // cudart 6.5 passes kernel arguments as a packed buffer via
        // CU_LAUNCH_PARAM_* pairs (kernelParams == NULL).  Unpack it
        // into a synthetic argument vector using our per-parameter
        // model: scalars/pointers are 4 bytes in a 32-bit client,
        // 64-bit scalars 8 bytes, by-value structs wide_words*8 bytes
        // (8-byte stride, matching the launcher), mixed blobs 8 bytes.
        // Hidden texture params occupy no client bytes.  Pointer-word
        // groups (kind 5) ride inside the head's struct bytes.
        void **e = extra;
        void *pbuf = NULL;
        for (int guard = 0; guard < 66; guard++) {
            uintptr_t key = (uintptr_t)e[0];
            if (key == 0)
                break;
            if (key == 1)
                pbuf = e[1];
            else if (key != 2) {
                ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel "
                            "extra[%d] UNKNOWN key=%lu\n", guard,
                            (unsigned long)key);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
            }
            e += 2;
        }
        if (!pbuf) {
            ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel packed "
                        "buffer missing\n");
            { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
        }
        {
            size_t off = 0;
            for (uint32_t i = 0; i < info->num_params &&
                                i < ACUDA_MAX_PARAMS;
                 i++) {
                if (info->hidden_texref[i]) {
                    argv_buf[i] = NULL;
                    continue;
                }
                // Walk with declared strides (what cudart packed),
                // not kind-implied sizes: a .u64 pointer still
                // occupies 8 bytes even though only 4 are read.
                // A 64-bit client aligns 8-byte params to 8; a 32-bit
                // client packs everything tightly at 4 bytes (strides
                // are already 4-multiples there).
                size_t sz = info->param_stride[i];
#if !defined(_M_IX86) && !defined(__i386__)
                if (sz >= 8)
                    off = (off + 7u) & ~(size_t)7u;
#endif
                argv_buf[i] = (char *)pbuf + off;
                off += sz;
            }
            ACUDA_DEBUG_PRINT("acuda-trace .. cuLaunchKernel packed walk "
                        "n=%u bytes=%zu\n", info->num_params, off);
            {
                const unsigned char *pb = (const unsigned char *)pbuf;
                for (size_t o = 0; o < off; o += 16) {
                    char line[80];
                    size_t n = 0;
                    for (size_t j = o; j < o + 16 && j < off && n < sizeof(line) - 4; j++) {
                        n += (size_t)snprintf(line + n, sizeof(line) - n,
                                              "%02x ", pb[j]);
                    }
                    ACUDA_DEBUG_PRINT("acuda-trace .. packed[%04zu] %s\n", o, line);
                }
            }
            kparams = argv_buf;
        }
    }
    if (!kparams) {
        ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel kernelParams=NULL\n");
        { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
    }
    if (gridDimX == 0 || gridDimY == 0 || gridDimZ == 0 ||
        blockDimX == 0 || blockDimY == 0 || blockDimZ == 0) {
        ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel zero dim "
                    "grid=%u,%u,%u block=%u,%u,%u\n", gridDimX, gridDimY,
                    gridDimZ, blockDimX, blockDimY, blockDimZ);
        { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
    }
    if (getenv("ACUDA_TRACE_PARAMS")) {
        // Diagnostic: dump by-value struct arguments word by word so a
        // lift/runtime mismatch can be correlated with the PTX offsets.
        fprintf(stderr, "acuda-params %s", info->entry);
        for (uint32_t i = 0; i < info->num_params && i < ACUDA_MAX_PARAMS; i++) {
            if (!info->wide_words[i])
                continue;
            const unsigned char *p = (const unsigned char *)kparams[i];
            if (!p)
                continue;
            uint32_t words = info->wide_words[i] * 8;
            fprintf(stderr, " [%u]", i);
            for (uint32_t w = 0; w + 4 <= words && w < 2048; w += 4) {
                uint32_t v;
                memcpy(&v, p + w, 4);
                fprintf(stderr, " %08x", v);
            }
        }
        fprintf(stderr, "\n");
    }
    ACUDA_DEBUG_PRINT("acuda-launch-info %s params=%u ptr=%u bda=%u "
                "grid=%u,%u,%u block=%u,%u,%u shm=%u\n",
                info->entry, info->num_params, info->num_ptr_params, info->bda,
                gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                blockDimZ, sharedMemBytes);
    // Give unresolved .extern .func calls a usable default when the
    // CUDA API is used without ACuda's host-call registration.  Keep a
    // callback the application already registered.
    for (uint32_t hi = 0; hi < info->host_call_count; hi++) {
        const char *hname = info->host_call_names[hi];
        if (!acuda_host_call_find(g_actx, hname))
            acuda_host_call_register(g_actx, hname, driver_host_identity,
                                     NULL);
    }
    // Static .shared lives in Workgroup storage; dynamic .shared is
    // sized by this launch (spec-constant extern array). Cap the
    // total against the advertised per-block limit, like hardware,
    // and honor a raised MAX_DYNAMIC cap when set.
    if (info->shared_bytes > 65536 ||
        (uint64_t)info->shared_bytes + (uint64_t)sharedMemBytes > 65536)
        { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, __LINE__); return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES; }
    if (f->maxdyn_set && sharedMemBytes > f->maxdyn)
        { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, __LINE__); return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES; }
    uint32_t nptr = 0;
    for (uint32_t i = 0; i < info->num_params; i++)
        nptr += (info->is_ptr[i] == 1 || info->is_ptr[i] == 5 ||
                 info->is_ptr[i] == 6 ||
                 (info->is_ptr[i] == 4 && info->param_ssbo[i]))
                    ? 1u : 0u;
    VkBuffer *vkb = NULL;
    size_t *offs = NULL, *ranges = NULL;
    void **hostbufs = NULL;
    acuda_buffer_t *texh = NULL;
    if (nptr > 0) {
        vkb = malloc(sizeof(*vkb) * nptr);
        offs = malloc(sizeof(*offs) * nptr);
        ranges = malloc(sizeof(*ranges) * nptr);
        hostbufs = calloc(nptr, sizeof(*hostbufs));
        texh = calloc(nptr, sizeof(*texh));
        if (!vkb || !offs || !ranges || !hostbufs || !texh) {
            free(vkb);
            free(offs);
            free(ranges);
            free(hostbufs);
            free(texh);
            { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_OUT_OF_MEMORY, __LINE__); return CUDA_ERROR_OUT_OF_MEMORY; }
        }
    }
    uint32_t *words = NULL;
    if (info->push_bytes > 0) {
        words = malloc(info->push_bytes);
        if (!words) {
            free(vkb);
            free(offs);
            free(ranges);
            { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_OUT_OF_MEMORY, __LINE__); return CUDA_ERROR_OUT_OF_MEMORY; }
        }
    }
    acuda_stream_t s = NULL;
    if (hStream) {
        if (hStream->ctx != c) {
            free(vkb);
            free(offs);
            free(ranges);
            free(hostbufs);
            free(texh);
            free(words);
            { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
        }
        s = hStream->s;
    }
    acuda_stream_t resolved_stream = NULL;
    if (acuda_resolve_stream(g_actx, s, &resolved_stream) != ACUDA_SUCCESS) {
        free(vkb);
        free(offs);
        free(ranges);
        free(hostbufs);
        free(texh);
        free(words);
        { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
    }
    acuda_buffer_t owned[ACUDA_MAX_PARAMS];
    memset(owned, 0, sizeof(owned));
    // Sizes per binding are unknown here; resolve each pointer to its full
    // allocation range via a zero-size probe first, then bind the range
    // actually needed. Simpler and correct: bind the whole allocation.
    uint32_t bi = 0, wi = 0, harg = 0, cont_off = 8;
    for (uint32_t i = 0; i < info->num_params; i++) {
        uint32_t pkind = info->is_ptr[i];
        void *karg;
        if (info->hidden_texref[i]) {
            CUtexref tr = NULL;
            for (size_t ti = 0; ti < f->mod->ntexrefs; ti++) {
                if (f->mod->texrefs[ti] &&
                    strcmp(f->mod->texrefs[ti]->name,
                           info->hidden_texref_name[i]) == 0) {
                    tr = f->mod->texrefs[ti];
                    break;
                }
            }
            driver_texobj_t *to = tr ? driver_tex_find(tr->obj) : NULL;
            ACUDA_DEBUG_PRINT("acuda-trace ?? hidden texref [%u] mod=%p want='%s' found=%d obj=%llx to=%p\n",
                    i, (void *)f->mod, info->hidden_texref_name[i], tr != NULL,
                    (unsigned long long)(tr ? (uint64_t)tr->obj : 0u),
                    (void *)to);
            if (!to) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
            }
            texh[bi] = to->tex;
            vkb[bi] = VK_NULL_HANDLE;
            ranges[bi] = 0;
            bi++;
            continue;
        }
        if (info->hidden_symbol[i]) {
            CUdeviceptr dp = 0;
            size_t bytes = 0;
            for (size_t gi = 0; gi < f->mod->nglobals; gi++) {
                if (strcmp(f->mod->globals[gi].name, info->hidden_symbol_name[i]) == 0) {
                    dp = f->mod->globals[gi].dptr;
                    bytes = f->mod->globals[gi].bytes;
                    break;
                }
            }
            if (dp == 0 || bytes == 0) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb); free(offs); free(ranges); free(hostbufs); free(texh); free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_NOT_FOUND, __LINE__); return CUDA_ERROR_NOT_FOUND; }
            }
            if (bind_for(dp, bytes, &vkb[bi], &offs[bi]) != CUDA_SUCCESS ||
                acuda_arena_host((uint64_t)dp, bytes, &hostbufs[bi]) != 0) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb); free(offs); free(ranges); free(hostbufs); free(texh); free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
            }
            ranges[bi] = bytes;
            if (getenv("ACUDA_DEBUG_SYMBOLS") || getenv("ACUDA_DEBUG_BDA_PUSH")) {
                VkBufferDeviceAddressInfo _bdai = {0};
                _bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
                _bdai.buffer = vkb[bi];
                VkDeviceAddress _bda = vkGetBufferDeviceAddress(g_actx->device, &_bdai);
                fprintf(stderr, "ACuda: hidden symbol bind param=%u bi=%u name=%s dp=0x%llx bytes=%zu vkb=%p off=%zu bda=0x%llx\n",
                        i, bi, info->hidden_symbol_name[i], (unsigned long long)dp, bytes,
                        (void *)(uintptr_t)vkb[bi], offs[bi], (unsigned long long)(_bda + (VkDeviceAddress)offs[bi]));
            }
            bi++;
            continue;
        }
        if (pkind == 5) {
            // Next pointer word of the previous by-value argument,
            // not a separate host arg.
            if (harg == 0) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
            }
            karg = (char *)kparams[harg - 1] + cont_off;
            cont_off += 8;
        } else {
            karg = kparams[harg++];
            cont_off = 8;
        }
        if (!karg) {
            release_param_buffers(owned, ACUDA_MAX_PARAMS);
            free(vkb);
            free(offs);
            free(ranges);
            free(hostbufs);
            free(texh);
            free(words);
            { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
        }
        if (info->is_ptr[i] == 3) {
            continue; // unused param: no arg read, no words
        } else if (info->is_ptr[i] == 1 && info->is_tex[i] == 3) {
            // Surface parameter: the argument is an opaque CUsurfObject
            // id; bind its borrowed storage image.
            CUdeviceptr dp = 0;
            memcpy(&dp, karg, sizeof(dp));
            driver_surfobj_t *so = driver_surf_find((CUsurfObject)dp);
            if (!so || !so->image) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
            }
            texh[bi] = so->image;
            vkb[bi] = VK_NULL_HANDLE;
            ranges[bi] = 0;
            bi++;
        } else if (info->is_ptr[i] == 1 && info->is_tex[i]) {
            // Texture/sampler parameter: the argument is an opaque
            // CUtexObject id, not a device address.
            CUdeviceptr dp = 0;
            memcpy(&dp, karg, sizeof(dp));
            driver_texobj_t *to = driver_tex_find((CUtexObject)dp);
            if (!to) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
            }
            texh[bi] = info->is_tex[i] == 2 ? to->sampler : to->tex;
            vkb[bi] = VK_NULL_HANDLE;
            ranges[bi] = 0;
            bi++;
        } else if (info->is_ptr[i] == 1 || info->is_ptr[i] == 5) {
            CUdeviceptr dp = 0;
            memcpy(&dp, karg, sizeof(dp));
            if (dp == 0) {
                // Optional pointer argument (e.g. layer-norm
                // gamma/beta).  Bind a zeroed dummy so the descriptor
                // is valid, but push a zero device address: the
                // shader's null check must still see null.
                acuda_status_t nst = ensure_null_param_buffer();
                if (nst != ACUDA_SUCCESS) {
                    release_param_buffers(owned, ACUDA_MAX_PARAMS);
                    free(vkb);
                    free(offs);
                    free(ranges);
                    free(hostbufs);
                    free(texh);
                    free(words);
                    return map_status(nst);
                }
                vkb[bi] = g_null_param_buffer->buffer;
                offs[bi] = 0;
                ranges[bi] = ACUDA_NULL_PARAM_BYTES;
                hostbufs[bi] = NULL;
                texh[bi] = ACUDA_NULL_PARAM_HANDLE;
                bi++;
                continue;
            }
            // Range unknown: bind from dp to end of its allocation.
            // arena_bind validates containment of at least 1 byte.
            size_t cap = 0;
            ACUDA_DEBUG_PRINT("acuda-trace ?? bind ptr param [%u] dp=%llx stride=%u\n",
                        i, (unsigned long long)dp,
                        info->param_stride
                            ? (unsigned)info->param_stride[i] : 9999u);
            if (acuda_arena_capacity((uint64_t)dp, &cap) != 0) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
            }
            if (bind_for(dp, cap, &vkb[bi], &offs[bi]) != CUDA_SUCCESS ||
                acuda_arena_host((uint64_t)dp, cap, &hostbufs[bi]) != 0) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
            }
            ranges[bi] = cap;
            bi++;
        } else if (info->is_ptr[i] == 6) {
            // Mixed blob: bind word0 target like a plain pointer,
            // then push word1 as the scalar word.
            CUdeviceptr dp6 = 0;
            memcpy(&dp6, karg, sizeof(dp6));
            ACUDA_DEBUG_PRINT("acuda-trace ?? bind kind-6 param [%u] dp=%llx stride=%u\n",
                        i, (unsigned long long)dp6,
                        info->param_stride
                            ? (unsigned)info->param_stride[i] : 9999u);
            if (dp6 == 0) {
                acuda_status_t nst6 = ensure_null_param_buffer();
                if (nst6 != ACUDA_SUCCESS) {
                    release_param_buffers(owned, ACUDA_MAX_PARAMS);
                    free(vkb);
                    free(offs);
                    free(ranges);
                    free(hostbufs);
                    free(texh);
                    free(words);
                    return map_status(nst6);
                }
                vkb[bi] = g_null_param_buffer->buffer;
                offs[bi] = 0;
                ranges[bi] = ACUDA_NULL_PARAM_BYTES;
                hostbufs[bi] = NULL;
                texh[bi] = ACUDA_NULL_PARAM_HANDLE;
                bi++;
            } else {
                size_t cap6 = 0;
                if (acuda_arena_capacity((uint64_t)dp6, &cap6) != 0) {
                    release_param_buffers(owned, ACUDA_MAX_PARAMS);
                    free(vkb);
                    free(offs);
                    free(ranges);
                    free(hostbufs);
                    free(texh);
                    free(words);
                    { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
                }
                if (bind_for(dp6, cap6, &vkb[bi], &offs[bi]) != CUDA_SUCCESS ||
                    acuda_arena_host((uint64_t)dp6, cap6, &hostbufs[bi]) != 0) {
                    release_param_buffers(owned, ACUDA_MAX_PARAMS);
                    free(vkb);
                    free(offs);
                    free(ranges);
                    free(hostbufs);
                    free(texh);
                    free(words);
                    { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
                }
                ranges[bi] = cap6;
                bi++;
            }
            memcpy(&words[wi++], (const char *)karg + 4, 4);
        } else if (info->is_ptr[i] == 2) {
            // 64-bit scalars take two push words and are even-aligned
            // by the compiler; honor the same padding here.
            if (wi & 1u)
                wi++;
            uint64_t v = 0;
            memcpy(&v, karg, sizeof(v));
            // A u64 scalar can still be a device pointer that the
            // parameter analysis only tracked as a value (e.g. an
            // output pointer used solely as a store base).  Translate
            // an arena address to the real Vulkan device address, the
            // same way by-value struct words are translated below, so
            // the shader's PhysicalStorageBuffer access hits the buffer.
            if (v) {
                size_t vcap = 0;
                VkBuffer vbuf = VK_NULL_HANDLE;
                size_t voff = 0;
                if (acuda_arena_capacity(v, &vcap) == 0 &&
                    acuda_arena_bind(v, vcap, &vbuf, &voff) == 0 &&
                    vbuf != VK_NULL_HANDLE) {
                    VkBufferDeviceAddressInfo vbdai = {
                        .sType =
                            VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                        .buffer = vbuf,
                    };
                    VkDeviceAddress va = vkGetBufferDeviceAddress(
                        g_actx->device, &vbdai);
                    if (va)
                        v = (uint64_t)va + (uint64_t)voff;
                }
            }
            words[wi++] = (uint32_t)(v & 0xffffffffu);
            words[wi++] = (uint32_t)(v >> 32);
        } else if (info->is_ptr[i] == 4 && info->param_ssbo[i]) {
            // Large by-value struct: upload the raw argument bytes into
            // a storage buffer; the shader reads it by byte offset.
            uint32_t nq = info->wide_words[i];
            if (nq < 2)
                nq = 2;
            size_t bytes = (size_t)nq * 8u;
            acuda_buffer_t pb = NULL;
            acuda_status_t pst = acuda_malloc_host(g_actx, &pb, bytes);
            if (pst == ACUDA_SUCCESS) {
                void *mapped = NULL;
                VkResult vr = vkMapMemory(g_actx->device, pb->memory, 0,
                                          VK_WHOLE_SIZE, 0, &mapped);
                if (vr != VK_SUCCESS) {
                    pst = ACUDA_ERROR_INTERNAL;
                } else {
                    memcpy(mapped, karg, bytes);

                    // The argument is metadata whose pointer fields use
                    // ACuda's arena virtual addresses.  Translate any
                    // such 8-byte word to the real Vulkan device address
                    // so the shader's BDA loads hit the right buffer.
                    uint64_t *qw = (uint64_t *)mapped;
                    for (uint32_t q = 0; q < nq; q++) {
                        size_t qcap = 0;
                        VkBuffer qbuf = VK_NULL_HANDLE;
                        size_t qoff = 0;
                        if (acuda_arena_capacity(qw[q], &qcap) != 0 ||
                            acuda_arena_bind(qw[q], qcap, &qbuf, &qoff) != 0 ||
                            qbuf == VK_NULL_HANDLE)
                            continue;
                        VkBufferDeviceAddressInfo qbdai = {
                            .sType =
                                VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                            .buffer = qbuf,
                        };
                        VkDeviceAddress qa = vkGetBufferDeviceAddress(
                            g_actx->device, &qbdai);
                        if (qa)
                            qw[q] = (uint64_t)qa + (uint64_t)qoff;
                    }
                    VkMappedMemoryRange range = {
                        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                        .memory = pb->memory,
                        .offset = 0,
                        .size = VK_WHOLE_SIZE,
                    };
                    VkResult fr = vkFlushMappedMemoryRanges(
                        g_actx->device, 1, &range);
                    vkUnmapMemory(g_actx->device, pb->memory);
                    if (fr != VK_SUCCESS) {
                        pst = ACUDA_ERROR_INTERNAL;
                    } else {
                        pb->gpu_dirty = 1;
                        pb->gpu_read = 0;
                    }
                }
            }
            if (pst != ACUDA_SUCCESS) {
                if (pb)
                    acuda_free(g_actx, pb);
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                return map_status(pst);
            }
            owned[bi] = pb;
            texh[bi] = pb;
            vkb[bi] = pb->buffer;
            offs[bi] = 0;
            ranges[bi] = bytes;
            bi++;
        } else if (info->is_ptr[i] == 4) {
            // By-value struct of scalar words: two push words per
            // 8-byte word, packed from one host argument.
            uint32_t nq = info->wide_words[i];
            if (nq < 2)
                nq = 2;
            if (wi & 1u)
                wi++;
            // A by-value struct can mix scalars and pointers (e.g. a
            // functor plus the output pointer).  Translate any qword
            // that lands inside the arena to its Vulkan device address.
            for (uint32_t q = 0; q < nq; q++) {
                uint64_t v = 0;
                memcpy(&v, (const char *)karg + (size_t)q * 8u, 8u);
                size_t qcap = 0;
                VkBuffer qbuf = VK_NULL_HANDLE;
                size_t qoff = 0;
                if (acuda_arena_capacity(v, &qcap) == 0 &&
                    acuda_arena_bind(v, qcap, &qbuf, &qoff) == 0 &&
                    qbuf != VK_NULL_HANDLE) {
                    VkBufferDeviceAddressInfo qbdai = {
                        .sType =
                            VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                        .buffer = qbuf,
                    };
                    VkDeviceAddress qa = vkGetBufferDeviceAddress(
                        g_actx->device, &qbdai);
                    if (qa)
                        v = (uint64_t)qa + (uint64_t)qoff;
                }
                words[wi++] = (uint32_t)(v & 0xffffffffu);
                words[wi++] = (uint32_t)(v >> 32);
            }
        } else {
            memcpy(&words[wi++], karg, 4);
        }
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. cuLaunchKernel %s bind done "
                "nptr=%u bi(words=%u)\n",
                f->name, nptr, wi);
    if (getenv("ACUDA_TRACE_PARAMS") && getenv("ACUDA_TRACE")) {
        // Scalar push words, in the same order the shader's
        // PushConstant block reads them.  Pair with acuda-params for
        // by-value structs when a launch reads a wrong value.
        fprintf(stderr, "acuda-push %s", f->name);
        for (uint32_t k = 0; k < wi; k++)
            fprintf(stderr, " %08x", words[k]);
        fprintf(stderr, "\n");
    }
    {
        // Headless-hang diagnosis: snapshot the exact launch inputs
        // (push words, grid/block, per-binding bytes) so a device-loss
        // dispatch can be replayed without the application.
        const char *fdir = getenv("ACUDA_DUMP_FORCE_INPUTS");
        if (fdir && *fdir &&
            (strstr(f->name, "KernelFluidForce") != NULL ||
             strstr(f->name, "KernelFluidDensity") != NULL)) {
            static int fserial = 0;
            char path[1024];
            snprintf(path, sizeof(path), "%s/forcedump_%d.bin", fdir,
                     fserial++);
            FILE *fp = fopen(path, "wb");
            if (fp) {
                uint32_t hdr[16];
                memset(hdr, 0, sizeof(hdr));
                hdr[0] = 0x46445243u; // 'FDRC'
                hdr[1] = nptr;
                hdr[2] = wi;
                hdr[3] = gridDimX;
                hdr[4] = gridDimY;
                hdr[5] = gridDimZ;
                hdr[6] = blockDimX;
                hdr[7] = blockDimY;
                hdr[8] = blockDimZ;
                fwrite(hdr, sizeof(hdr), 1, fp);
                fwrite(words, sizeof(uint32_t), wi, fp);
                for (uint32_t b = 0; b < nptr; b++) {
                    uint64_t rb = (uint64_t)ranges[b];
                    fwrite(&rb, 8, 1, fp);
                    size_t take = ranges[b] < 262144 ? ranges[b] : 262144;
                    if (take && hostbufs && hostbufs[b])
                        fwrite(hostbufs[b], 1, take, fp);
                }
                fclose(fp);
                ACUDA_DEBUG_PRINT("acuda-trace .. forcedump %s nptr=%u words=%u\n",
                            path, nptr, wi);
            }
        }
    }
    acuda_buffer_t *tables = NULL;
    acuda_buffer_t smeta = NULL;
    uint32_t surf_slots = 0;
    if (info->surf_table) {
        acuda_buffer_t fallback = NULL;
        for (uint32_t i = 1; i < g_surf_slot_cap; i++)
            if (g_surf_slots[i]) {
                fallback = g_surf_slots[i];
                break;
            }
        if (!fallback) {
            release_param_buffers(owned, ACUDA_MAX_PARAMS);
            free(vkb);
            free(offs);
            free(ranges);
            free(hostbufs);
            free(texh);
            free(words);
            { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
        }
        surf_slots = 0;
        for (uint32_t i = g_surf_slot_cap; i > 0; i--)
            if (g_surf_slots[i - 1]) {
                surf_slots = i;
                break;
            }
        if (surf_slots == 0) {
            release_param_buffers(owned, ACUDA_MAX_PARAMS);
            free(vkb);
            free(offs);
            free(ranges);
            free(hostbufs);
            free(texh);
            free(words);
            { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
        }
        uint32_t nt = info->surf_table_count;
        if (nt) {
            tables = calloc((size_t)nt * surf_slots, sizeof(*tables));
            if (!tables) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_OUT_OF_MEMORY, __LINE__); return CUDA_ERROR_OUT_OF_MEMORY; }
            }
        }
        for (uint32_t t = 0; t < nt; t++) {
            int dim = (int)info->surf_table_dim[t];
            int arrayed = (int)info->surf_table_arrayed[t];
            int isint = (int)info->surf_table_int[t];
            for (uint32_t i = 0; i < surf_slots; i++) {
                acuda_buffer_t img = g_surf_slots[i];
                int id = -1, ia = -1, ii = -1;
                if (img)
                    driver_img_type(img, &id, &ia, &ii);
                if (!img || id != dim || ia != arrayed || ii != isint)
                    img = driver_surf_dummy(dim, arrayed, isint);
                if (!img) {
                    release_param_buffers(owned, ACUDA_MAX_PARAMS);
                    free(tables);
                    free(vkb);
                    free(offs);
                    free(ranges);
                    free(hostbufs);
                    free(texh);
                    free(words);
                    { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_OUT_OF_MEMORY, __LINE__); return CUDA_ERROR_OUT_OF_MEMORY; }
                }
                tables[(size_t)t * surf_slots + i] = img;
            }
        }
        size_t mbytes = surf_slots * 6u * sizeof(uint32_t);
        uint32_t *mw = calloc(1, mbytes);
        acuda_status_t mst = mw ? acuda_malloc(g_actx, &smeta, mbytes)
                                : ACUDA_ERROR_OUT_OF_MEMORY;
        if (mst == ACUDA_SUCCESS) {
            for (uint32_t i = 0; i < surf_slots; i++) {
                acuda_buffer_t img = g_surf_slots[i];
                driver_surf_meta(img ? img : fallback, mw + i * 6);
            }
            mst = acuda_memcpy_h2d(g_actx, smeta, mw, mbytes, NULL);
        }
        free(mw);
        if (mst != ACUDA_SUCCESS) {
            if (smeta)
                acuda_free(g_actx, smeta);
            release_param_buffers(owned, ACUDA_MAX_PARAMS);
            free(tables);
            free(vkb);
            free(offs);
            free(ranges);
            free(hostbufs);
            free(texh);
            free(words);
            return map_status(mst);
        }
    }
    acuda_status_t st =
        acuda_launch_raw_ex(g_actx, f->k, vkb, offs, ranges, nptr,
                            hostbufs, texh, tables,
                            info->surf_table_count, surf_slots, smeta, words,
                            info->push_bytes,
                            gridDimX, gridDimY, gridDimZ,
                            blockDimX, blockDimY, blockDimZ,
                            sharedMemBytes, s);
    ACUDA_DEBUG_PRINT("acuda-trace .. cuLaunchKernel %s raw_ex st=%d\n",
                f->name, (int)st);
    free(tables);
    if (smeta && st != ACUDA_SUCCESS)
        acuda_free(g_actx, smeta);
    free(vkb);
    free(offs);
    free(ranges);
    free(hostbufs);
    free(texh);
    free(words);
    if (st != ACUDA_SUCCESS) {
        release_param_buffers(owned, ACUDA_MAX_PARAMS);
        return map_status(st);
    }
    // Storage-buffer parameter data must outlive the recorded launch.
    // Attach each temporary buffer to the stream; on the rare tracking
    // failure, synchronize before releasing it.
    for (uint32_t i = 0; i < nptr && i < ACUDA_MAX_PARAMS; i++) {
        if (!owned[i])
            continue;
        if (acuda_stream_track_buffer_pending(
                g_actx, resolved_stream, owned[i]) != ACUDA_SUCCESS) {
            acuda_stream_synchronize(g_actx, s);
            acuda_free(g_actx, owned[i]);
        }
        owned[i] = NULL;
    }
    // NOTE: the CUfunction shell stays owned by the caller (like real
    // CUDA, functions die with the module, not the launch). Freeing it
    // here used to turn every handle reuse into use-after-free.
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuFuncGetAttribute(int *pi, int attrib, CUfunction hfunc) {
    if (!pi)
        return CUDA_ERROR_INVALID_VALUE;
    *pi = 0;
    if (!hfunc)
        return CUDA_ERROR_INVALID_HANDLE;
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuFuncGetAttribute(%d, %s)\n",
                attrib, hfunc->name);
    {
        CUfunction real = NULL;
        CUresult hr = handle_as_func((void *)hfunc, &real);
        if (hr != CUDA_SUCCESS)
            return hr;
        hfunc = real;
    }
    if (!hfunc->k) {
        // Lazy library kernel: answer the probe without compiling.
        if (attrib == CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            *pi = ACUDA_MAX_BLOCK_X;
        else if (attrib == CU_FUNC_ATTRIBUTE_CACHE_MODE_CA)
            *pi = 1;
        else if (attrib == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)
            *pi = hfunc->maxdyn_set ? (int)hfunc->maxdyn : 0;
        else if (attrib == CU_FUNC_ATTRIBUTE_PTX_VERSION ||
                 attrib == CU_FUNC_ATTRIBUTE_BINARY_VERSION)
            // cudart validates a library kernel's PTX/binary version
            // before launching; answering 0 makes it report
            // cudaErrorInvalidDeviceFunction (e.g. for CUB's
            // runtime-API kernels).
            *pi = cc_major() * 10 + cc_minor();
        ACUDA_DEBUG_PRINT("acuda-trace .. cuFuncGetAttribute -> %d\n", *pi);
        return CUDA_SUCCESS;
    }
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (attrib < 0)
        return CUDA_ERROR_INVALID_VALUE;
    if (attrib > CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) {
        // Newer attributes (cluster geometry and the like) are not
        // modelled.  Answer with a benign zero instead of an error: the
        // CUDA 12.8+ runtime caches a failed post-launch attribute query
        // per function, which then makes later
        // cudaFuncGetAttributes/cudaOccupancyMaxPotentialBlockSize calls
        // fail even though the kernel itself is fine.
        *pi = 0;
        return CUDA_SUCCESS;
    }
    const acuda_kernel_info_t *info = acuda_kernel_info_ptr(hfunc->k);
    if (!info)
        return CUDA_ERROR_UNKNOWN;
    switch (attrib) {
    case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
        *pi = ACUDA_MAX_BLOCK_X; // any 1..1024 launches (specialized)
        break;
    case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: // static .shared total
        *pi = (int)info->shared_bytes;
        break;
    case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: // no .const: always 0
    case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: // no .local: always 0
        *pi = 0;
        break;
    case CU_FUNC_ATTRIBUTE_NUM_REGS: // declared registers (estimate)
        *pi = (int)info->num_regs;
        break;
    case CU_FUNC_ATTRIBUTE_PTX_VERSION: // .version as major*10+minor
        *pi = (int)info->ptx_version;
        break;
    case CU_FUNC_ATTRIBUTE_BINARY_VERSION: // .target sm
        *pi = (int)info->sm_version;
        break;
    case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA: // always cached, report 1
        *pi = 1;
        break;
    case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
        *pi = hfunc->maxdyn_set ? (int)hfunc->maxdyn : 0;
        break;
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. cuFuncGetAttribute -> %d\n", *pi);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuFuncSetAttribute(CUfunction hfunc, int attrib, int value) {
    if (!hfunc)
        return CUDA_ERROR_INVALID_HANDLE;
    {
        CUfunction real = NULL;
        CUresult hr = handle_as_func((void *)hfunc, &real);
        if (hr != CUDA_SUCCESS)
            return hr;
        hfunc = real;
    }
    if (!hfunc->k)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (attrib < 0)
        return CUDA_ERROR_INVALID_VALUE;
    // Carveout/cluster hints and other newer attributes are advisory on
    // this backend.  Accept and ignore them: the CUDA 12.8+ runtime
    // caches a failure from a launch-time attribute probe and then
    // rejects later cudaFuncGetAttributes/cudaOccupancy queries.
    if (attrib > CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)
        return CUDA_SUCCESS;
    // Single-path backend: most attributes are advisory only. Static
    // .shared is real (Workgroup storage, reported via
    // SHARED_SIZE_BYTES); dynamic .shared is sized per launch up to
    // the 64KB total cap (and MAX_DYNAMIC when raised here).
    // Negative values are rejected like the real driver.
    if (value < 0)
        return CUDA_ERROR_INVALID_VALUE;
    if (attrib == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) {
        hfunc->maxdyn_set = 1;
        hfunc->maxdyn = (unsigned)value;
    }
    return CUDA_SUCCESS;
}
