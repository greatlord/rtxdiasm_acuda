#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "acuda/compiler.h"
#include "internal.h"
#include "time_compat.h"
#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#endif

static int lprof_on = -1;
static uint64_t lprof_n, lprof_pre;
static double lprof_now(void) {
    return (double)acuda_now_ns();
}
static void lprof_dump(void) {
    if (lprof_n)
        fprintf(stderr, "ACuda launch pre-work: %.2f us/launch\n",
                (double)lprof_pre / (double)lprof_n / 1e3);
}

#define ACUDA_BIND_CACHE_N 8
#define ACUDA_BIND_CACHE_PARAMS 8

// One cached descriptor set for an identical binding tuple, valid only
// inside the stream's current open batch (stream id + batch generation).
typedef struct {
    uint64_t stream_id;
    uint64_t gen;
    VkDescriptorSet dset;
    uint32_t nbufs;
    int has_handles;
    VkBuffer bufs[ACUDA_BIND_CACHE_PARAMS];
    size_t offsets[ACUDA_BIND_CACHE_PARAMS];
    size_t ranges[ACUDA_BIND_CACHE_PARAMS];
    acuda_buffer_t handles[ACUDA_BIND_CACHE_PARAMS];
    // Device addresses of the bound buffers.  Descriptor sets reference
    // a buffer's memory, while the VkBuffer handle itself is recyclable:
    // a destroyed buffer's handle can be handed to a later allocation
    // whose memory lives elsewhere.  Matching on the handle alone would
    // then reuse a descriptor set pointing at freed memory.
    VkDeviceAddress addrs[ACUDA_BIND_CACHE_PARAMS];
} acuda_bind_cache_t;

// Resolve the device address of every bound buffer so descriptor-cache
// identity follows the memory, not the recyclable VkBuffer handle.
static void bind_cache_addresses(acuda_context_t ctx, acuda_bind_cache_t *c,
                                 const VkBuffer *bufs, uint32_t nbufs) {
    for (uint32_t i = 0; i < nbufs && i < ACUDA_BIND_CACHE_PARAMS; i++) {
        c->addrs[i] = 0;
        if (!ctx->bda_ok || !bufs || bufs[i] == VK_NULL_HANDLE)
            continue;
        VkBufferDeviceAddressInfo bi = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = bufs[i],
        };
        c->addrs[i] = vkGetBufferDeviceAddress(ctx->device, &bi);
    }
}

typedef struct {
    const acuda_kernel_t *kernel;
    uint64_t serial; // kernel pointers are recycled; match serial too
    acuda_context_t ctx;
    uint32_t block_x; // workgroup size: specialized per pipeline
    uint32_t block_y;
    uint32_t block_z;
    uint32_t dyn_words; // dynamic-.shared words: specialized per pipeline
    VkShaderModule mod;
    VkDescriptorSetLayout dsl;
    VkDescriptorSetLayout *surf_dsl; // one per dynamic surface type
    uint32_t surf_dsl_count;
    VkPipelineLayout pl;
    VkPipeline pipe;
    // Descriptor-set reuse inside one open batch. A few entries cover
    // double/triple buffering as well as repeated identical bindings.
    acuda_bind_cache_t bind_cache[ACUDA_BIND_CACHE_N];
    uint32_t bind_next;
    // Cross-batch steady-state cache for repeated launches with the same
    // buffers and block shape. The descriptor set and its pool are retained
    // so a render loop does not rebuild descriptor arrays and pools per frame.
    acuda_bind_cache_t steady;
    VkDescriptorPool steady_pool;
    int steady_valid;
    // A fully recorded single-dispatch command buffer. It is reused only
    // when the stream is idle, the binding tuple is the steady one, and
    // every push-constant byte matches. Animated values therefore need to
    // live in a buffer, not in push constants, for this path to hit.
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    uint64_t cmd_stream_id;
    VkDescriptorSet cmd_dset;
    uint32_t cmd_grid;
    uint32_t cmd_grid_y;
    uint32_t cmd_grid_z;
    uint32_t cmd_push_bytes;
    uint8_t cmd_push[512];
    int cmd_pending; // signature seen once; record on the next match
    int cmd_valid;   // command buffer recorded for that signature
} launch_cache_t;

// The cache is shared by the user thread and the ACUDA_HIGH_PERF
// keep-busy thread. Entries live on the heap and the global array only
// holds pointers, so growing it never invalidates an entry a caller is
// using. The lock guards the array itself (lookup/insert/clear); an
// entry is only mutated by the thread that owns its kernel.
static launch_cache_t **cache;
static size_t cache_count, cache_cap;
#if defined(_WIN32)
static SRWLOCK cache_lock = SRWLOCK_INIT;
static void cache_lock_acquire(void) { AcquireSRWLockExclusive(&cache_lock); }
static void cache_lock_release(void) { ReleaseSRWLockExclusive(&cache_lock); }
#else
static pthread_mutex_t cache_lock = PTHREAD_MUTEX_INITIALIZER;
static void cache_lock_acquire(void) { pthread_mutex_lock(&cache_lock); }
static void cache_lock_release(void) { pthread_mutex_unlock(&cache_lock); }
#endif

static uint32_t surf_max_slots(acuda_context_t ctx, uint32_t count) {
    uint32_t m = ctx->props.limits.maxPerStageDescriptorStorageImages;
    if (ctx->props.limits.maxDescriptorSetStorageImages &&
        m > ctx->props.limits.maxDescriptorSetStorageImages)
        m = ctx->props.limits.maxDescriptorSetStorageImages;
    if (ctx->props.limits.maxPerStageResources &&
        m > ctx->props.limits.maxPerStageResources)
        m = ctx->props.limits.maxPerStageResources;
    if (count > 1)
        m /= count;
    if (m > 8)
        m -= 8; // leave room for params/metadata
    return m ? m : 1;
}

// Optional ISA introspection: ACUDA_DUMP_ISA prints the internal
// representations (disassembly) that the driver reports for a compute
// pipeline. Requires VK_KHR_pipeline_executable_properties; silently
// does nothing when unavailable.
static void dump_pipeline_isa(acuda_context_t ctx, VkPipeline pipe,
                              const char *entry) {
    static int enabled = -1;
    if (enabled < 0)
        enabled = getenv("ACUDA_DUMP_ISA") != NULL;
    if (!enabled || !ctx->pipe_exec_ok)
        return;
    VkPipelineInfoKHR pi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR,
        .pipeline = pipe,
    };
    uint32_t n = 0;
    if (ctx->get_pipe_exec_props(ctx->device, &pi, &n, NULL) != VK_SUCCESS ||
        n == 0)
        return;
    VkPipelineExecutablePropertiesKHR *props =
        calloc(n, sizeof(*props));
    if (!props)
        return;
    for (uint32_t i = 0; i < n; i++)
        props[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    if (ctx->get_pipe_exec_props(ctx->device, &pi, &n, props) != VK_SUCCESS) {
        free(props);
        return;
    }
    for (uint32_t i = 0; i < n; i++) {
        fprintf(stderr, "ACuda: pipeline '%s' executable %u: %s (%s), "
                        "subgroup=%u\n",
                entry, i, props[i].name, props[i].description,
                props[i].subgroupSize);
        VkPipelineExecutableInfoKHR ei = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
            .pipeline = pipe,
            .executableIndex = i,
        };
        uint32_t rn = 0;
        VkResult rvr =
            ctx->get_pipe_exec_reprs(ctx->device, &ei, &rn, NULL);
        if (rvr != VK_SUCCESS) {
            fprintf(stderr, "ACuda: executable representations unavailable "
                            "(VkResult %d)\n", (int)rvr);
            continue;
        }
        if (rn == 0) {
            fprintf(stderr, "ACuda: no internal representations "
                            "(driver does not expose them)\n");
            continue;
        }

        VkPipelineExecutableInternalRepresentationKHR *reps =
            calloc(rn, sizeof(*reps));
        if (!reps)
            continue;
        for (uint32_t r = 0; r < rn; r++)
            reps[r].sType =
                VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
        VkResult rvr2 =
            ctx->get_pipe_exec_reprs(ctx->device, &ei, &rn, reps);
        if (rvr2 != VK_SUCCESS) {
            fprintf(stderr, "ACuda: representation query 2 failed: %d\n",
                    (int)rvr2);
            free(reps);
            continue;
        }
        for (uint32_t r = 0; r < rn; r++)
            if (reps[r].dataSize)
                reps[r].pData = malloc(reps[r].dataSize);
        if (ctx->get_pipe_exec_reprs(ctx->device, &ei, &rn, reps) ==
            VK_SUCCESS) {
            for (uint32_t r = 0; r < rn; r++) {
                const char *data = reps[r].pData;
                if (!data || reps[r].dataSize == 0)
                    continue;
                fprintf(stderr, "ACuda: --- %s (%s), %zu bytes ---\n",
                        reps[r].name, reps[r].description,
                        reps[r].dataSize);
                if (reps[r].isText) {
                    fwrite(data, 1, reps[r].dataSize, stderr);
                    if (data[reps[r].dataSize - 1] != '\n')
                        fputc('\n', stderr);
                } else {
                    fprintf(stderr, "(%zu bytes of binary)\n",
                            reps[r].dataSize);
                }
            }
        }
        for (uint32_t r = 0; r < rn; r++)
            free(reps[r].pData);
        free(reps);
    }
    free(props);
}

static void cache_destroy_entry(acuda_context_t ctx, launch_cache_t *e) {
    if (e->pipe)
        vkDestroyPipeline(ctx->device, e->pipe, NULL);
    if (e->pl)
        vkDestroyPipelineLayout(ctx->device, e->pl, NULL);
    if (e->surf_dsl) {
        for (uint32_t i = 0; i < e->surf_dsl_count; i++)
            if (e->surf_dsl[i])
                vkDestroyDescriptorSetLayout(ctx->device, e->surf_dsl[i],
                                             NULL);
        free(e->surf_dsl);
    }
    if (e->dsl)
        vkDestroyDescriptorSetLayout(ctx->device, e->dsl, NULL);
    if (e->mod)
        vkDestroyShaderModule(ctx->device, e->mod, NULL);
    if (e->steady_pool)
        vkDestroyDescriptorPool(ctx->device, e->steady_pool, NULL);
    if (e->cmd_pool)
        vkDestroyCommandPool(ctx->device, e->cmd_pool, NULL);
    free(e);
}

// Called from acuda_shutdown (same library) to avoid validation errors
// from destroying a device with live pipelines.
void acuda_launch_cache_clear_ctx(acuda_context_t ctx) {
    cache_lock_acquire();
    size_t keep = 0;
    for (size_t i = 0; i < cache_count; i++) {
        if (cache[i]->ctx == ctx) {
            cache_destroy_entry(ctx, cache[i]);
        } else {
            cache[keep++] = cache[i];
        }
    }
    cache_count = keep;
    cache_lock_release();
}

// Descriptor type for a binding kind (0 buffer, 1 image sampler,
// 3 sampler).
static VkDescriptorType binding_vktype(uint32_t kind) {
    switch (kind) {
    case 1: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    case 2: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case 3: return VK_DESCRIPTOR_TYPE_SAMPLER;
    default: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
}

static acuda_status_t cache_lookup(acuda_context_t ctx, const acuda_kernel_t *k,
                                   const acuda_kernel_info_t *info,
                                   uint32_t block_dim_x, uint32_t block_dim_y,
                                   uint32_t block_dim_z, uint32_t dyn_words,
                                   launch_cache_t **out) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: entry ctx=%p k=%p info=%p cache=%p count=%zu\n",
                      (void*)ctx, (void*)k, (void*)info, (void*)cache, cache_count);
    cache_lock_acquire();
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: locked cache=%p count=%zu\n", (void*)cache, cache_count);
    for (size_t i = 0; i < cache_count; i++) {
        launch_cache_t *ce = cache ? cache[i] : NULL;
        ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: i=%zu ce=%p\n", i, (void*)ce);
        if (!ce) {
            ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: NULL entry at %zu\n", i);
            continue;
        }
        if (ce->kernel == k && ce->serial == info->serial &&
            ce->ctx == ctx && ce->block_x == block_dim_x &&
            ce->block_y == block_dim_y &&
            ce->block_z == block_dim_z &&
            ce->dyn_words == dyn_words) {
            ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: hit i=%zu\n", i);
            *out = ce;
            cache_lock_release();
            return ACUDA_SUCCESS;
        }
    }
    cache_lock_release();
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: miss\n");
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: miss post-release ctx=%p info=%p\n", (void*)ctx, (void*)info);
    uint64_t invocations = (uint64_t)block_dim_x * block_dim_y * block_dim_z;
    if (invocations == 0 ||
        invocations > ctx->props.limits.maxComputeWorkGroupInvocations ||
        block_dim_x > ctx->props.limits.maxComputeWorkGroupSize[0] ||
        block_dim_y > ctx->props.limits.maxComputeWorkGroupSize[1] ||
        block_dim_z > ctx->props.limits.maxComputeWorkGroupSize[2]) {
        return ACUDA_ERROR_NOT_SUPPORTED;
    }
    uint32_t eff_sg = 0;
    if (info->uses_subgroup && ctx->subgroup32_ok)
        eff_sg = 32;
    else if (ctx->pref_subgroup_size && !info->uses_barrier)
        eff_sg = ctx->pref_subgroup_size;
    if (info->uses_subgroup && !ctx->subgroup32_ok && eff_sg == 0 &&
        (ctx->subgroup_size == 8 || ctx->subgroup_size == 16))
        eff_sg = (uint32_t)ctx->subgroup_size;
    if (eff_sg && ctx->max_workgroup_subgroups > 0 &&
        (block_dim_x + eff_sg - 1u) / eff_sg >
            ctx->max_workgroup_subgroups) {
        fprintf(stderr,
                "ACuda: kernel '%s' needs %u %u-lane subgroups, device "
                "max is %u\n", info->entry,
                (block_dim_x + eff_sg - 1u) / eff_sg, eff_sg,
                ctx->max_workgroup_subgroups);
        return ACUDA_ERROR_NOT_SUPPORTED;
    }
    // PTX defines warps as 32 lanes. Request a 32-wide subgroup when
    // the device supports the control. Without the control, 32- and
    // 64-lane subgroups are exact (64-lane waves are split into 32-lane
    // segments). For 8/16-lane physical subgroups, the compiler emits
    // adaptive SPIR-V that emulates 32-lane logical warps via shared
    // memory when SubgroupSize < 32. If the launch fits in one native
    // subgroup, hardware subgroup ops are exact regardless of size.
    if (info->uses_subgroup && !ctx->subgroup32_ok &&
        ctx->subgroup_size != 32 && ctx->subgroup_size != 64 &&
        block_dim_x > (uint32_t)ctx->subgroup_size) {
        if (ctx->subgroup_size != 8 && ctx->subgroup_size != 16) {
            fprintf(stderr,
                    "ACuda: kernel '%s' needs 32-lane warp semantics but this "
                    "device reports %d-lane subgroups and cannot require 32\n",
                    info->entry, ctx->subgroup_size);
            return ACUDA_ERROR_NOT_SUPPORTED;
        }
        /* Emulation uses non-uniform subgroup arithmetic/elect ops. */
        {
            const uint32_t need = VK_SUBGROUP_FEATURE_BASIC_BIT |
                                  VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
            if ((ctx->subgroup_ops & need) != need ||
                !(ctx->subgroup_stages & VK_SHADER_STAGE_COMPUTE_BIT)) {
                fprintf(stderr,
                        "ACuda: kernel '%s' needs 8/16-lane warp emulation but "
                        "device subgroup features are insufficient (ops=0x%x stages=0x%x)\n",
                        info->entry, ctx->subgroup_ops, ctx->subgroup_stages);
                return ACUDA_ERROR_NOT_SUPPORTED;
            }
        }
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: pre calloc\n");
    launch_cache_t *e = calloc(1, sizeof(*e));
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: post calloc e=%p\n", (void*)e);
    if (!e)
        return ACUDA_ERROR_OUT_OF_MEMORY;
    e->kernel = k;
    e->serial = info->serial;
    e->ctx = ctx;
    e->block_x = block_dim_x;
    e->block_y = block_dim_y;
    e->block_z = block_dim_z;
    e->dyn_words = dyn_words;
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: entry initialized\n");

    const uint32_t *words = NULL;
    size_t nwords = 0;
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: pre spirv k=%p\n", (void*)k);
    acuda_status_t st = acuda_kernel_spirv(k, &words, &nwords);
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: post spirv st=%d words=%p nwords=%zu\n", (int)st, (const void*)words, nwords);
    if (st != ACUDA_SUCCESS)
        return st;
    // Drivers without maintenance4 cannot consume OpExecutionModeId
    // LocalSizeId. Patch a private copy to OpExecutionMode LocalSize using
    // the launch's actual block size; the pipeline cache is already keyed by
    // block_x, so no fixed workgroup size is assumed.
    uint32_t *patched_words = NULL;
    const uint32_t *shader_words = words;
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: pre patch maint4=%d api=%u\n", ctx->maintenance4_ok, (unsigned)ctx->props.apiVersion);
    if (!ctx->maintenance4_ok ||
        ctx->props.apiVersion < VK_API_VERSION_1_3) {
        patched_words = malloc(nwords * sizeof(uint32_t));
        if (!patched_words) {
            cache_destroy_entry(ctx, e);
            return ACUDA_ERROR_OUT_OF_MEMORY;
        }
        memcpy(patched_words, words, nwords * sizeof(uint32_t));
        // SPIR-V 1.6 is only legal for Vulkan 1.3+ semantics. Vulkan 1.2
        // devices (including Dozen) accept SPIR-V 1.4.
        if (ctx->props.apiVersion < VK_API_VERSION_1_3 && nwords > 1)
            patched_words[1] = 0x00010400u;
        for (size_t i = 5; i + 5 < nwords;) {
            uint32_t wc = patched_words[i] >> 16;
            uint32_t op = patched_words[i] & 0xffffu;
            if (wc < 1 || i + wc > nwords)
                break;
            if (op == 331 && wc == 6 && patched_words[i + 2] == 38) {
                patched_words[i] = (6u << 16) | 16; // OpExecutionMode
                patched_words[i + 2] = 17;           // LocalSize
                patched_words[i + 3] = block_dim_x;
                patched_words[i + 4] = block_dim_y;
                patched_words[i + 5] = block_dim_z;
                break;
            }
            i += wc;
        }
        shader_words = patched_words;
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: pre shader module\n");
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = nwords * 4,
        .pCode = shader_words,
    };
    VkResult vr = vkCreateShaderModule(ctx->device, &smci, NULL, &e->mod);
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: post shader module vr=%d mod=%p\n", (int)vr, (void*)(uintptr_t)e->mod);
    free(patched_words);
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: pre release_spirv\n");
    acuda_kernel_release_spirv((acuda_kernel_t *)k);
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: post release_spirv\n");
    if (vr != VK_SUCCESS)
        return acuda_from_vk(vr);

    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: pre launch_ptr_params k=%p\n", (void*)k);
    uint32_t full_ptrs = acuda_kernel_launch_ptr_params(k);
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: post launch_ptr_params full_ptrs=%u\n", full_ptrs);
    uint32_t nbind = full_ptrs + (info->surf_table ? 1u : 0u) +
                     (info->surf_lock ? 1u : 0u);
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: nbind=%u surf_table=%u surf_lock=%u surf_count=%u info=%p\n",
                      nbind, info ? info->surf_table : 0u, info ? info->surf_lock : 0u, info ? info->surf_table_count : 0u, (void*)info);
    VkDescriptorSetLayoutBinding *bindings = NULL;
    if (nbind > 0) {
        bindings = calloc(nbind, sizeof(*bindings));
        ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: bindings=%p full_ptrs=%u\n", (void*)bindings, full_ptrs);
        if (!bindings) {
            cache_destroy_entry(ctx, e);
            return ACUDA_ERROR_OUT_OF_MEMORY;
        }
        for (uint32_t i = 0; i < full_ptrs; i++) {
            ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: bind i=%u kind_ptr=%p\n", i, (void*)(info ? info->binding_kind : NULL));
            uint32_t kind = info->binding_kind ? info->binding_kind[i] : 0u;
            bindings[i].binding = i;
            bindings[i].descriptorType = binding_vktype(kind);
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].pImmutableSamplers = NULL;
        }
        ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: bind loop done\n");
        uint32_t extra = full_ptrs;
        if (info->surf_table) {
            VkDescriptorSetLayoutBinding *mb = &bindings[extra++];
            mb->binding = ACUDA_SURF_META_BINDING;
            mb->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            mb->descriptorCount = 1;
            mb->stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            mb->pImmutableSamplers = NULL;
        }
        if (info->surf_lock) {
            VkDescriptorSetLayoutBinding *lb = &bindings[extra++];
            lb->binding = ACUDA_SURF_LOCK_BINDING;
            lb->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            lb->descriptorCount = 1;
            lb->stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            lb->pImmutableSamplers = NULL;
        }
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: pre dsl create dev=%p nbind=%u bindings=%p\n", (void*)ctx->device, nbind, (void*)bindings);
    VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = nbind,
        .pBindings = bindings,
    };
    vr = vkCreateDescriptorSetLayout(ctx->device, &dlci, NULL, &e->dsl);
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: post dsl create vr=%d\n", (int)vr);
    free(bindings);
    if (vr != VK_SUCCESS) {
        cache_destroy_entry(ctx, e);
        return acuda_from_vk(vr);
    }

    uint32_t max_slots = surf_max_slots(ctx, info->surf_table_count);
    uint32_t nsets = 1;
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: pre surf layouts max_slots=%u st=%u stc=%u\n", max_slots, info->surf_table, info->surf_table_count);
    if (info->surf_table && info->surf_table_count > 0) {
        e->surf_dsl = calloc(info->surf_table_count, sizeof(*e->surf_dsl));
        if (!e->surf_dsl) {
            cache_destroy_entry(ctx, e);
            return ACUDA_ERROR_OUT_OF_MEMORY;
        }
        e->surf_dsl_count = info->surf_table_count;
        for (uint32_t t = 0; t < info->surf_table_count; t++) {
            VkDescriptorSetLayoutBinding tb = {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = max_slots,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = NULL,
            };
            VkDescriptorBindingFlags flag =
                VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
            VkDescriptorSetLayoutBindingFlagsCreateInfo fb = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                .bindingCount = 1,
                .pBindingFlags = &flag,
            };
            VkDescriptorSetLayoutCreateInfo lci = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = &fb,
                .bindingCount = 1,
                .pBindings = &tb,
            };
            vr = vkCreateDescriptorSetLayout(ctx->device, &lci, NULL,
                                             &e->surf_dsl[t]);
            if (vr != VK_SUCCESS) {
                cache_destroy_entry(ctx, e);
                return acuda_from_vk(vr);
            }
        }
        nsets += info->surf_table_count;
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: post surf layouts nsets=%u\n", nsets);
    uint32_t addr_base = (info->push_bytes + info->tex_meta_bytes + 7u) & ~7u;
    uint32_t push_total = addr_base + info->addr_push_bytes;
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: push_total=%u max=%u\n", push_total, ctx->props.limits.maxPushConstantsSize);
    if (push_total > ctx->props.limits.maxPushConstantsSize) {
        cache_destroy_entry(ctx, e);
        return ACUDA_ERROR_NOT_SUPPORTED;
    }
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_total,
    };
    VkDescriptorSetLayout *sls = malloc(sizeof(*sls) * nsets);
    if (!sls) {
        cache_destroy_entry(ctx, e);
        return ACUDA_ERROR_OUT_OF_MEMORY;
    }
    sls[0] = e->dsl;
    for (uint32_t t = 0; t + 1 < nsets; t++)
        sls[t + 1] = e->surf_dsl[t];
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = nsets,
        .pSetLayouts = sls,
        .pushConstantRangeCount = push_total > 0 ? 1u : 0u,
        .pPushConstantRanges = push_total > 0 ? &pcr : NULL,
    };
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: pre pipeline layout\n");
    vr = vkCreatePipelineLayout(ctx->device, &plci, NULL, &e->pl);
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: post pipeline layout vr=%d\n", (int)vr);
    free(sls);
    if (vr != VK_SUCCESS) {
        cache_destroy_entry(ctx, e);
        return acuda_from_vk(vr);
    }

    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: pre pipeline create entry=%s\n", info->entry);
    VkPipelineShaderStageCreateInfo stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = e->mod,
        .pName = info->entry,
    };
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_req = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .requiredSubgroupSize = 32,
    };
    uint32_t req_sg = 0;
    if (info->uses_subgroup && ctx->subgroup32_ok)
        req_sg = 32;
    else if (ctx->pref_subgroup_size && !info->uses_barrier)
        req_sg = ctx->pref_subgroup_size;
    /* 8/16-lane devices: prefer a controlled physical subgroup size for
     * predictable warp emulation grouping when supported. */
    if (info->uses_subgroup && !ctx->subgroup32_ok && req_sg == 0 &&
        (ctx->subgroup_size == 8 || ctx->subgroup_size == 16)) {
        if (ctx->subgroup_min <= (uint32_t)ctx->subgroup_size &&
            ctx->subgroup_max >= (uint32_t)ctx->subgroup_size &&
            ctx->subgroup_size >= 8)
            req_sg = (uint32_t)ctx->subgroup_size;
    }
    if (req_sg) {
        subgroup_req.requiredSubgroupSize = req_sg;
        stage.pNext = &subgroup_req;
        if (getenv("ACUDA_DEBUG_SUBGROUP"))
            fprintf(stderr,
                    "ACuda: kernel '%s' requires subgroup size %u\n",
                    info->entry, req_sg);
    }
    // Only promise full subgroups when the specialized workgroup size is
    // already a multiple of the effective subgroup size.
    uint32_t esg = req_sg ? req_sg
                          : (ctx->subgroup_size > 0
                                 ? (uint32_t)ctx->subgroup_size
                                 : 1u);
    if (ctx->full_subgroups_ok && block_dim_x % esg == 0 &&
        ctx->props.apiVersion >= VK_API_VERSION_1_3)
        stage.flags |=
            VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    // Specialize the workgroup-size spec constants (0, 1, 2) so any
    // blockDimX in 1..1024 launches without recompiling the module,
    // plus the dynamic-.shared word count (3) for extern kernels.
    uint32_t spec_data[4] = { block_dim_x, block_dim_y, block_dim_z,
                              dyn_words };
    VkSpecializationMapEntry spec_map[4] = {
        { .constantID = 0, .offset = 0, .size = sizeof(uint32_t) },
        { .constantID = 1, .offset = 4, .size = sizeof(uint32_t) },
        { .constantID = 2, .offset = 8, .size = sizeof(uint32_t) },
        { .constantID = 3, .offset = 12, .size = sizeof(uint32_t) },
    };
    VkSpecializationInfo spec_info = {
        .mapEntryCount = info->has_extern ? 4u : 3u,
        .pMapEntries = spec_map,
        .dataSize = info->has_extern ? sizeof(spec_data) : 3 * sizeof(uint32_t),
        .pData = spec_data,
    };
    stage.pSpecializationInfo = &spec_info;
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage,
        .layout = e->pl,
    };
    if (ctx->pipe_exec_ok && getenv("ACUDA_DUMP_ISA"))
        cpci.flags |=
            VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
#if defined(_WIN32)
    if (getenv("ACUDA_TRACE") && !HeapValidate(GetProcessHeap(), 0, NULL))
        ACUDA_DEBUG_PRINT("acuda-trace !! HEAP CORRUPT before create pipeline\n");
#endif
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: pre create pipeline dev=%p mod=%p pl=%p entry=%p(%s) stageNext=%p specN=%u specSize=%zu specData=%p[%u,%u,%u,%u] specMap=%p\n",
                      (void*)ctx->device, (void*)(uintptr_t)stage.module, (void*)(uintptr_t)cpci.layout,
                      (void*)stage.pName, stage.pName ? stage.pName : "(null)",
                      stage.pNext, spec_info.mapEntryCount, spec_info.dataSize, spec_info.pData,
                      spec_data[0], spec_data[1], spec_data[2], spec_data[3],
                      (void*)spec_info.pMapEntries);
    if (getenv("ACUDA_TRACE")) {
        for (uint32_t si = 0; si < spec_info.mapEntryCount; si++)
            ACUDA_DEBUG_PRINT("acuda-trace ..   spec[%u] id=%u off=%u size=%zu\n",
                              si, spec_map[si].constantID, spec_map[si].offset, spec_map[si].size);
        if (stage.pNext == &subgroup_req)
            ACUDA_DEBUG_PRINT("acuda-trace ..   subgroup_req sType=%d req=%u pNext=%p\n",
                              (int)subgroup_req.sType, subgroup_req.requiredSubgroupSize, subgroup_req.pNext);
    }
    vr = vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                  &e->pipe);
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: post create pipeline vr=%d pipe=%p\n", (int)vr, (void*)(uintptr_t)e->pipe);
    if (vr != VK_SUCCESS) {
        cache_destroy_entry(ctx, e);
        return acuda_from_vk(vr);
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: pre dump isa\n");
    dump_pipeline_isa(ctx, e->pipe, info->entry);
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: post dump isa\n");
    // Publish the fully built entry. Another thread may have created the
    // same one while we were building; keep theirs and drop ours.
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: pre publish lock\n");
    cache_lock_acquire();
    for (size_t i = 0; i < cache_count; i++) {
        if (cache[i]->kernel == k && cache[i]->serial == info->serial &&
            cache[i]->ctx == ctx && cache[i]->block_x == block_dim_x &&
            cache[i]->block_y == block_dim_y &&
            cache[i]->block_z == block_dim_z &&
            cache[i]->dyn_words == dyn_words) {
            *out = cache[i];
            cache_lock_release();
            cache_destroy_entry(ctx, e);
            return ACUDA_SUCCESS;
        }
    }
    if (cache_count == cache_cap) {
        size_t ncap = cache_cap ? cache_cap * 2 : 4;
        launch_cache_t **nc = malloc(sizeof(*nc) * ncap);
        if (!nc) {
            cache_lock_release();
            cache_destroy_entry(ctx, e);
            return ACUDA_ERROR_OUT_OF_MEMORY;
        }
        memcpy(nc, cache, sizeof(*nc) * cache_count);
        free(cache);
        cache = nc;
        cache_cap = ncap;
    }
    cache[cache_count++] = e;
    cache_lock_release();
    ACUDA_DEBUG_PRINT("acuda-trace .. cache_lookup: published e=%p count=%zu\n", (void*)e, cache_count);
    *out = e;
    return ACUDA_SUCCESS;
}

typedef struct {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSet sets[1 + ACUDA_SURF_TABLE_MAX_TYPES];
    uint32_t set_count;
    const void *push_data;
    size_t push_bytes;
    uint32_t grid;
    uint32_t grid_y;
    uint32_t grid_z;
    VkBuffer *bufs;
    const size_t *offsets;
    const size_t *ranges;
    uint32_t num_bufs;
    int has_storage;
    int mem_barrier; // insert global shader-write -> read/write barrier
    acuda_stream_t stream;
    acuda_buffer_t *handles; // NULL for the raw VkBuffer path
    const uint32_t *written; // per-param write flags (binding order)
} launch_rec_t;

static acuda_status_t record_launch(VkCommandBuffer cmd, void *user) {
    launch_rec_t *r = user;
    ACUDA_DEBUG_PRINT("acuda-trace .. record_launch grid=%u bufs=%u push=%u\n",
                      r->grid, r->num_bufs, (unsigned)r->push_bytes);
    for (uint32_t i = 0; i < r->num_bufs; i++) {
        if (!r->bufs[i])
            continue;
        acuda_buffer_t h =
            (r->handles && i < r->num_bufs) ? r->handles[i] : NULL;
        if (h == ACUDA_NULL_PARAM_HANDLE)
            h = NULL; // null argument: dummy buffer needs no tracking
        if (h && h->kind != 0)
            h = NULL; // textures/images use their own synchronization
        int writes = r->written && r->written[i] != 0;
        // Acquire an earlier write, or order this write after earlier
        // reads (WAR). Read-after-read needs no barrier.
        if (!h || h->gpu_dirty || (writes && h->gpu_read))
            acuda_buf_barrier(cmd, r->bufs[i], r->offsets[i],
                              r->ranges[i]);
        if (h) {
            h->gpu_dirty = 0;
            h->gpu_read = 0;
        }
    }
    if (r->mem_barrier) {
        VkMemoryBarrier2 mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
                             VK_ACCESS_2_SHADER_WRITE_BIT,
        };
        VkDependencyInfo dep = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &mb,
        };
        vkCmdPipelineBarrier2(cmd, &dep);
    }
    acuda_stream_t s = r->stream;
    int layout_changed = s ? (r->layout != s->bind_layout) : 1;
    if (!s || r->pipeline != s->bind_pipeline) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r->pipeline);
        if (s)
            s->bind_pipeline = r->pipeline;
    }
    if (layout_changed && s) {
        // A layout change invalidates the batch's descriptor/push state.
        s->bind_set_count = 0;
        s->bind_push_bytes = 0;
    }
    int same_sets = s && r->set_count == s->bind_set_count;
    if (same_sets) {
        for (uint32_t i = 0; i < r->set_count; i++)
            if (r->sets[i] != s->bind_sets[i]) {
                same_sets = 0;
                break;
            }
    }
    if (!same_sets && r->set_count > 0) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r->layout,
                                0, r->set_count, r->sets, 0, NULL);
        if (s) {
            s->bind_set_count = r->set_count;
            memcpy(s->bind_sets, r->sets,
                   sizeof(VkDescriptorSet) * r->set_count);
        }
    }
    int same_push = s && r->push_bytes == s->bind_push_bytes &&
                    r->push_bytes <= sizeof(s->bind_push) &&
                    (r->push_bytes == 0 ||
                     memcmp(r->push_data, s->bind_push, r->push_bytes) == 0);
    if (!same_push && r->push_bytes > 0) {
        vkCmdPushConstants(cmd, r->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           (uint32_t)r->push_bytes, r->push_data);
        if (s && r->push_bytes <= sizeof(s->bind_push)) {
            memcpy(s->bind_push, r->push_data, r->push_bytes);
            s->bind_push_bytes = (uint32_t)r->push_bytes;
        } else if (s) {
            s->bind_push_bytes = 0; // too large to cache
        }
    }
    if (s)
        s->bind_layout = r->layout;
    ACUDA_DEBUG_PRINT("acuda-trace .. record_launch dispatch grid=%u\n",
                      r->grid);
    vkCmdDispatch(cmd, r->grid, r->grid_y ? r->grid_y : 1,
                  r->grid_z ? r->grid_z : 1);
    if (r->handles && r->written) {
        for (uint32_t i = 0; i < r->num_bufs; i++) {
            acuda_buffer_t h = r->handles[i];
            if (!h || h == ACUDA_NULL_PARAM_HANDLE || h->kind != 0)
                continue;
            if (r->written[i]) {
                h->gpu_dirty = 1;
                h->gpu_read = 0;
            } else {
                h->gpu_read = 1;
            }
        }
    }
    if (r->mem_barrier) {
        VkMemoryBarrier2 mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
                             VK_ACCESS_2_SHADER_WRITE_BIT,
        };
        VkDependencyInfo dep = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &mb,
        };
        vkCmdPipelineBarrier2(cmd, &dep);
    }
    for (uint32_t i = 0; i < r->num_bufs; i++)
        if (r->bufs[i])
            acuda_buf_barrier(cmd, r->bufs[i], r->offsets[i],
                              r->ranges[i]);
    return ACUDA_SUCCESS;
}

// Record the single-dispatch command buffer that the steady-state path
// resubmits. `rec->stream` is cleared so recording always emits the full
// bind sequence instead of relying on the open batch's cached bind state.
static acuda_status_t cache_record_cmd(acuda_context_t ctx, launch_cache_t *e,
                                       const launch_rec_t *rec) {
    if (!e->cmd_pool) {
        VkCommandPoolCreateInfo pci = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = ctx->queue_family,
        };
        VkResult vr = vkCreateCommandPool(ctx->device, &pci, NULL,
                                          &e->cmd_pool);
        if (vr != VK_SUCCESS)
            return acuda_from_vk(vr);
    }
    if (!e->cmd) {
        VkCommandBufferAllocateInfo ai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = e->cmd_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkResult vr = vkAllocateCommandBuffers(ctx->device, &ai, &e->cmd);
        if (vr != VK_SUCCESS)
            return acuda_from_vk(vr);
    }
    VkResult vr = vkResetCommandBuffer(e->cmd, 0);
    if (vr != VK_SUCCESS)
        return acuda_from_vk(vr);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    vr = vkBeginCommandBuffer(e->cmd, &bi);
    if (vr != VK_SUCCESS)
        return acuda_from_vk(vr);
    launch_rec_t tmp = *rec;
    tmp.stream = NULL;
    // Force unconditional acquire/release barriers for every buffer: the
    // command may be resubmitted after unrelated work or a host write.
    tmp.handles = NULL;
    acuda_status_t st = record_launch(e->cmd, &tmp);
    vr = vkEndCommandBuffer(e->cmd);
    if (st != ACUDA_SUCCESS)
        return st;
    if (vr != VK_SUCCESS)
        return acuda_from_vk(vr);
    return ACUDA_SUCCESS;
}

// One raw dispatch, without host-call servicing.
// PTX addr_mode enum: wrap 0, mirror 1, clamp_ogl 2, clamp_to_edge 3,
// clamp_to_border 4.
static uint32_t tex_addr_code(uint32_t a) {
    switch (a) {
    case 1: return 1;
    case 2: return 3;
    case 3: return 4;
    default: return 0;
    }
}

// Seven texture/sampler metadata words consumed by txq and tiled 1D
// linear textures. Word 6 is the 1D tile stride (2D image width); 0
// means a native/non-tiled texture.
static void tex_meta_words(acuda_buffer_t h, uint32_t kind, uint32_t out[7]) {
    memset(out, 0, 7 * sizeof(uint32_t));
    if (kind == 1 || kind == 2) {
        if (h->tiled1d) {
            out[0] = h->logical_width ? h->logical_width : h->width;
            out[1] = 1;
            out[2] = 1;
            out[6] = h->tile_w ? h->tile_w : h->width;
        } else {
            out[0] = h->width;
            out[1] = h->dim1d ? 1u : h->height;
            out[2] = h->dim3d ? h->depth : 1u;
        }
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
        uint32_t levels = h->levels > 1 ? h->levels : 0;
        uint32_t samples = h->samples > 1 ? h->samples : 0;
        out[4] = (levels & 0xffffu) | (samples << 16);
        if (kind == 2) {
            uint32_t cdt = 2; // FLOAT
            uint32_t order = 1;
            switch (h->format) {
            case 0: cdt = 2; order = 3; break; // RGBA32F
            case 1: cdt = 1; order = 3; break; // RGBA8 UNORM
            case 2: cdt = 2; order = 1; break; // R32F
            case 3: cdt = 1; order = 1; break; // R32_UINT
            }
            out[4] = (cdt & 0xffu) | (order << 8);
        }
    }
    uint32_t a0 = tex_addr_code(h->saddr[0]);
    uint32_t a1 = tex_addr_code(h->saddr[1]);
    uint32_t a2 = tex_addr_code(h->saddr[2]);
    out[5] = (h->sfilter & 1u) | ((h->snorm ? 1u : 0u) << 2) |
             (a0 << 4) | (a1 << 8) | (a2 << 12);
}

static acuda_status_t launch_raw_once(acuda_context_t ctx,
                                      acuda_kernel_t *kernel,
                               VkBuffer *bufs, const size_t *offsets,
                               const size_t *ranges, uint32_t num_bufs,
                               acuda_buffer_t *handles,
                               acuda_buffer_t *surf_table,
                               uint32_t nsurf_table, uint32_t surf_slots,
                               acuda_buffer_t surf_meta,
                               const void *push_data, size_t push_bytes,
                               uint32_t grid_dim_x, uint32_t grid_dim_y,
                               uint32_t grid_dim_z, uint32_t block_dim_x,
                               uint32_t block_dim_y, uint32_t block_dim_z,
                               uint32_t dyn_shared_bytes,
                               acuda_stream_t stream) {
    ACUDA_DEBUG_PRINT("acuda-trace .. launch_raw_once: entry ctx=%p k=%p nb=%u\n", (void*)ctx, (void*)kernel, num_bufs);
    acuda_stream_t s = NULL;
    acuda_status_t st = acuda_resolve_stream(ctx, stream, &s);
    ACUDA_DEBUG_PRINT("acuda-trace .. launch_raw_once: resolve st=%d s=%p\n", (int)st, (void*)s);
    if (st != ACUDA_SUCCESS)
        return st;
    if (lprof_on < 0) {
        lprof_on = getenv("ACUDA_PROFILE_LAUNCH") != NULL;
        if (lprof_on)
            atexit(lprof_dump);
    }
    double lpt0 = lprof_on ? lprof_now() : 0;
    if (!kernel)
        return ACUDA_ERROR_INVALID_VALUE;
    const acuda_kernel_info_t *info = acuda_kernel_info_ptr(kernel);
    ACUDA_DEBUG_PRINT("acuda-trace .. launch_raw_once: info=%p surf_table=%u surf_lock=%u\n", (void*)info, info ? info->surf_table : 0u, info ? info->surf_lock : 0u);
    if (!info)
        return ACUDA_ERROR_INTERNAL;
    if (info->surf_table && (!surf_table || !surf_meta))
        return ACUDA_ERROR_NOT_SUPPORTED;
    ACUDA_DEBUG_PRINT("acuda-trace .. launch_raw_once: check lock_buf ctx=%p\n", (void*)ctx);
    if (info->surf_lock && !ctx->lock_buf) {
        st = acuda_malloc(ctx, &ctx->lock_buf, ACUDA_SURF_LOCK_BYTES);
        if (st != ACUDA_SUCCESS)
            return st;
        void *p = NULL;
        VkResult vr = vkMapMemory(ctx->device, ctx->lock_buf->memory, 0,
                                  ACUDA_SURF_LOCK_BYTES, 0, &p);
        if (vr != VK_SUCCESS || !p)
            return acuda_from_vk(vr);
        memset(p, 0, ACUDA_SURF_LOCK_BYTES);
        vkUnmapMemory(ctx->device, ctx->lock_buf->memory);
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. launch_raw_once: surf checks done\n");
    if (info->surf_table) {
        if (!ctx->desc_idx_ok)
            return ACUDA_ERROR_NOT_SUPPORTED;
        if (surf_slots == 0)
            return ACUDA_ERROR_INVALID_VALUE;
        if (surf_slots > surf_max_slots(ctx, info->surf_table_count))
            return ACUDA_ERROR_NOT_SUPPORTED;
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. launch_raw_once: pre ptr_params\n");
    if (num_bufs != acuda_kernel_launch_ptr_params(kernel))
        return ACUDA_ERROR_INVALID_VALUE;
    ACUDA_DEBUG_PRINT("acuda-trace .. launch_raw_once: post ptr_params\n");
    if (num_bufs > 0 && (!bufs || !offsets || !ranges))
        return ACUDA_ERROR_INVALID_VALUE;
    if (push_bytes != info->push_bytes ||
        (push_bytes > 0 && !push_data) || (push_bytes == 0 && push_data))
        return ACUDA_ERROR_INVALID_VALUE;
    if (block_dim_x == 0 || block_dim_x > ACUDA_MAX_BLOCK_X ||
        block_dim_y == 0 || block_dim_z == 0 ||
        grid_dim_x == 0 || grid_dim_y == 0 || grid_dim_z == 0)
        return ACUDA_ERROR_INVALID_VALUE;
    if (grid_dim_x > ctx->props.limits.maxComputeWorkGroupCount[0] ||
        grid_dim_y > ctx->props.limits.maxComputeWorkGroupCount[1] ||
        grid_dim_z > ctx->props.limits.maxComputeWorkGroupCount[2])
        return ACUDA_ERROR_NOT_SUPPORTED;
    ACUDA_DEBUG_PRINT("acuda-trace .. launch_raw_once: post dim checks\n");
    uint64_t shared_needed =
        (uint64_t)info->shared_bytes +
        (info->has_extern ? (uint64_t)dyn_shared_bytes : 0);
    if (shared_needed > ctx->props.limits.maxComputeSharedMemorySize)
        return ACUDA_ERROR_NOT_SUPPORTED;
    for (uint32_t i = 0; i < num_bufs; i++) {
        if (handles && i < ACUDA_MAX_PARAMS &&
            info->binding_kind[i] != 0) {
            int want = info->binding_kind[i] == 1 ? 3
                       : (info->binding_kind[i] == 2 ? 2 : 4);
            if (!handles[i] || handles[i]->kind != want)
                return ACUDA_ERROR_INVALID_VALUE;
            if (info->binding_kind[i] == 1 && handles[i]->samples > 1 &&
                handles[i]->sfilter)
                return ACUDA_ERROR_INVALID_VALUE; // MS + linear
            continue;
        }
        if (!bufs[i] || ranges[i] == 0)
            return ACUDA_ERROR_INVALID_VALUE;
        if (!info->bda &&
            (uint64_t)ranges[i] > ctx->props.limits.maxStorageBufferRange)
            return ACUDA_ERROR_NOT_SUPPORTED;
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. launch_raw_once: post buf validation\n");
    ACUDA_DEBUG_PRINT("acuda-trace .. launch_raw_once: pre f64 req=%d ctx=%p info=%p k=%p\n", info ? (int)info->requires_float64 : -1, (void *)ctx, (void *)info, (void *)kernel);

    // f64 kernels need shaderFloat64: fail cleanly here, never garbage.
    if (info->requires_float64 == 2) {
        if (!ctx->f64_ok)
            return ACUDA_ERROR_NOT_SUPPORTED;
    } else if (info->requires_float64 == 0) {
        /* old blob: scan SPIR-V */
        ACUDA_DEBUG_PRINT("acuda-trace .. launch_raw_once: f64 scan spirv k=%p\n", (void *)kernel);
        const uint32_t *words = NULL;
        size_t nwords = 0;
        st = acuda_kernel_spirv(kernel, &words, &nwords);
        ACUDA_DEBUG_PRINT("acuda-trace .. launch_raw_once: spirv st=%d words=%p nwords=%zu\n", (int)st, (const void *)words, nwords);
        if (st != ACUDA_SUCCESS)
            return st;
        int need_f64 = 0;
        for (size_t i = 5; i + 1 < nwords;) {
            uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
            if (wc < 1 || i + wc > nwords)
                break;
            if (wc == 2 && op == 17 && words[i + 1] == 10) { need_f64 = 1; break; }
            i += wc;
        }
        acuda_kernel_release_spirv((acuda_kernel_t *)kernel);
        if (need_f64 && !ctx->f64_ok)
            return ACUDA_ERROR_NOT_SUPPORTED;
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. launch_raw_once: pre cache_lookup dyn_shared=%zu\n", dyn_shared_bytes);

    launch_cache_t *e = NULL;
    // Dynamic words specialize the extern array (min 1: zero-length
    // arrays are invalid SPIR-V even before use).
    uint32_t dyn_words = dyn_shared_bytes / 4 + (dyn_shared_bytes % 4 ? 1u : 0u);
    if (dyn_words == 0)
        dyn_words = 1;
    st = cache_lookup(ctx, kernel, info, block_dim_x, block_dim_y,
                      block_dim_z, dyn_words, &e);
    ACUDA_DEBUG_PRINT("acuda-trace .. launch_raw_once: post cache_lookup st=%d e=%p\n", (int)st, (void *)e);
    if (st != ACUDA_SUCCESS)
        return st;
    ACUDA_DEBUG_PRINT("acuda-trace .. launch cache_lookup ok pipe=%p layout=%p\n",
                      (void *)(uintptr_t)e->pipe, (void *)(uintptr_t)e->pl);

    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorPool dpool_t = VK_NULL_HANDLE;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    VkDescriptorSet tsets[ACUDA_SURF_TABLE_MAX_TYPES];
    memset(tsets, 0, sizeof(tsets));
    uint32_t ntsets = info->surf_table ? info->surf_table_count : 0;
    VkWriteDescriptorSet *writes = NULL;
    VkDescriptorBufferInfo *infos = NULL;
    VkDescriptorImageInfo *iinfos = NULL;
    VkDescriptorImageInfo *tinfos = NULL;
    int desc_bad = 0;
    int reused_set = 0;
    int can_cache = s && bufs && num_bufs > 0 &&
                    num_bufs <= ACUDA_BIND_CACHE_PARAMS &&
                    !info->surf_table;
    if (can_cache && e->steady_valid &&
        e->steady.nbufs == num_bufs &&
        e->steady.has_handles == (handles != NULL) &&
        memcmp(e->steady.bufs, bufs, sizeof(VkBuffer) * num_bufs) == 0 &&
        memcmp(e->steady.offsets, offsets,
               sizeof(size_t) * num_bufs) == 0 &&
        memcmp(e->steady.ranges, ranges,
               sizeof(size_t) * num_bufs) == 0 &&
        (!handles || memcmp(e->steady.handles, handles,
                            sizeof(acuda_buffer_t) * num_bufs) == 0)) {
        // A recycled VkBuffer handle does not imply the same memory; the
        // cached descriptor set still points at whatever memory it was
        // written with.  Reuse only when every buffer resolves to the
        // same device address as when the set was written.
        acuda_bind_cache_t cur;
        bind_cache_addresses(ctx, &cur, bufs, num_bufs);
        if (memcmp(e->steady.addrs, cur.addrs,
                   sizeof(VkDeviceAddress) * num_bufs) == 0) {
            dset = e->steady.dset;
            reused_set = 1;
        }
    }
    if (can_cache && !reused_set) {
        for (uint32_t k = 0; k < ACUDA_BIND_CACHE_N; k++) {
            acuda_bind_cache_t *c = &e->bind_cache[k];
            if (c->stream_id == s->id && c->gen == s->batch_gen &&
                c->nbufs == num_bufs &&
                memcmp(c->bufs, bufs, sizeof(VkBuffer) * num_bufs) == 0 &&
                memcmp(c->offsets, offsets,
                       sizeof(size_t) * num_bufs) == 0 &&
                memcmp(c->ranges, ranges,
                       sizeof(size_t) * num_bufs) == 0 &&
                c->has_handles == (handles != NULL) &&
                (!handles || memcmp(c->handles, handles,
                                    sizeof(acuda_buffer_t) * num_bufs) == 0)) {
                dset = c->dset;
                reused_set = 1;
                break;
            }
        }
    }
    if (!reused_set && (num_bufs > 0 || info->surf_table || info->surf_lock)) {
        uint32_t nst = 0, nimg = 0, nsmp = 0, nstor = 0;
        for (uint32_t i = 0; i < num_bufs; i++) {
            if (info->binding_kind[i] == 1)
                nimg++;
            else if (info->binding_kind[i] == 2)
                nstor++;
            else if (info->binding_kind[i] == 3)
                nsmp++;
            else
                nst++;
        }
        if (info->surf_table)
            nst++;
        if (info->surf_lock)
            nst++;
        VkDescriptorPoolSize psz[4];
        uint32_t nps = 0;
        if (nst > 0) {
            psz[nps].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            psz[nps].descriptorCount = nst;
            nps++;
        }
        if (nimg > 0) {
            psz[nps].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            psz[nps].descriptorCount = nimg;
            nps++;
        }
        if (nstor > 0) {
            psz[nps].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            psz[nps].descriptorCount = nstor;
            nps++;
        }
        if (nsmp > 0) {
            psz[nps].type = VK_DESCRIPTOR_TYPE_SAMPLER;
            psz[nps].descriptorCount = nsmp;
            nps++;
        }
        VkDescriptorPoolCreateInfo dpci = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = nps,
            .pPoolSizes = psz,
        };
        // Reuse a completed pool when possible. A pool created for a
        // different layout may be too small; fall back to a fresh one.
        VkResult vr = VK_ERROR_OUT_OF_POOL_MEMORY;
        if (s->free_dpool_count > 0) {
            dpool = s->free_dpools[--s->free_dpool_count];
            VkDescriptorSetAllocateInfo rai = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = dpool,
                .descriptorSetCount = 1,
                .pSetLayouts = &e->dsl,
            };
            vr = vkAllocateDescriptorSets(ctx->device, &rai, &dset);
            if (vr != VK_SUCCESS) {
                vkDestroyDescriptorPool(ctx->device, dpool, NULL);
                dpool = VK_NULL_HANDLE;
            }
        }
        if (!dpool) {
            vr = vkCreateDescriptorPool(ctx->device, &dpci, NULL, &dpool);
            if (vr != VK_SUCCESS)
                return acuda_from_vk(vr);
            VkDescriptorSetAllocateInfo ai = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = dpool,
                .descriptorSetCount = 1,
                .pSetLayouts = &e->dsl,
            };
            vr = vkAllocateDescriptorSets(ctx->device, &ai, &dset);
            if (vr != VK_SUCCESS) {
                vkDestroyDescriptorPool(ctx->device, dpool, NULL);
                return acuda_from_vk(vr);
            }
        }
        uint32_t n0 = num_bufs + (info->surf_table ? 1u : 0u) +
                      (info->surf_lock ? 1u : 0u);
        writes = calloc(n0 ? n0 : 1, sizeof(*writes));
        infos = calloc(n0 ? n0 : 1, sizeof(*infos));
        iinfos = calloc(num_bufs ? num_bufs : 1, sizeof(*iinfos));
        if (!writes || !infos || !iinfos) {
            desc_bad = 1;
        }
        for (uint32_t i = 0; !desc_bad && i < num_bufs; i++) {
            uint32_t kind = info->binding_kind[i];
            int isimg = handles && kind == 1;
            int isstor = handles && kind == 2;
            int issmp = handles && kind == 3;
            if (isimg) {
                if (!handles[i] || handles[i]->kind != 3 ||
                    !handles[i]->view || !handles[i]->sampler) {
                    desc_bad = 1;
                    break;
                }
                iinfos[i].sampler = handles[i]->sampler;
                iinfos[i].imageView = handles[i]->view;
                iinfos[i].imageLayout =
                    handles[i]->is_surface
                        ? VK_IMAGE_LAYOUT_GENERAL
                        : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            } else if (isstor) {
                if (!handles[i] || handles[i]->kind != 2 ||
                    !handles[i]->view) {
                    desc_bad = 1;
                    break;
                }
                VkImageView iv = handles[i]->view;
                if (info->surf_f32_cas[i]) {
                    // The compiler's portable f32 min/max fallback uses
                    // an R32ui image and integer CAS.
                    iv = handles[i]->uint_view;
                    if (!iv) {
                        desc_bad = 1;
                        break;
                    }
                }
                iinfos[i].sampler = VK_NULL_HANDLE;
                iinfos[i].imageView = iv;
                iinfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            } else if (issmp) {
                if (!handles[i] || handles[i]->kind != 4 ||
                    !handles[i]->sampler) {
                    desc_bad = 1;
                    break;
                }
                iinfos[i].sampler = handles[i]->sampler;
                iinfos[i].imageView = VK_NULL_HANDLE;
                iinfos[i].imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            } else {
                infos[i].buffer = bufs[i];
                if (info->bda) {
                    // Plain pointer parameters are accessed through the
                    // device address pushed in the push-constant block, so
                    // this storage-buffer binding is not read by the
                    // shader.  Vulkan still requires the descriptor offset
                    // to be a multiple of minStorageBufferOffsetAlignment,
                    // so bind from zero instead of the (possibly
                    // unaligned) pointer offset.
                    infos[i].offset = 0;
                    infos[i].range = offsets[i] + ranges[i];
                } else {
                    infos[i].offset = offsets[i];
                    infos[i].range = ranges[i];
                }
            }
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = dset;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = binding_vktype(kind);
            writes[i].pImageInfo =
                (isimg || isstor || issmp) ? &iinfos[i] : NULL;
            writes[i].pBufferInfo =
                (isimg || isstor || issmp) ? NULL : &infos[i];
        }
        if (!desc_bad && info->surf_table) {
            infos[num_bufs].buffer =
                surf_meta ? surf_meta->buffer : VK_NULL_HANDLE;
            infos[num_bufs].offset = 0;
            infos[num_bufs].range = surf_slots * 6u * sizeof(uint32_t);
            VkWriteDescriptorSet *mw = &writes[num_bufs];
            mw->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            mw->dstSet = dset;
            mw->dstBinding = ACUDA_SURF_META_BINDING;
            mw->descriptorCount = 1;
            mw->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            mw->pBufferInfo = &infos[num_bufs];
            if (!surf_meta || !surf_meta->buffer ||
                nsurf_table != info->surf_table_count)
                desc_bad = 1;
        }
        if (!desc_bad && info->surf_lock) {
            uint32_t li = num_bufs + (info->surf_table ? 1u : 0u);
            infos[li].buffer = ctx->lock_buf ? ctx->lock_buf->buffer
                                             : VK_NULL_HANDLE;
            infos[li].offset = 0;
            infos[li].range = ACUDA_SURF_LOCK_BYTES;
            VkWriteDescriptorSet *lw = &writes[li];
            lw->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            lw->dstSet = dset;
            lw->dstBinding = ACUDA_SURF_LOCK_BINDING;
            lw->descriptorCount = 1;
            lw->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            lw->pBufferInfo = &infos[li];
            if (!ctx->lock_buf || !ctx->lock_buf->buffer)
                desc_bad = 1;
        }
        if (!desc_bad)
            vkUpdateDescriptorSets(ctx->device, n0, writes, 0, NULL);

        if (!desc_bad && ntsets > 0) {
            VkDescriptorPoolSize tps = {
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = ntsets * surf_slots,
            };
            VkDescriptorPoolCreateInfo tpci = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = ntsets,
                .poolSizeCount = 1,
                .pPoolSizes = &tps,
            };
            VkResult tvr = vkCreateDescriptorPool(ctx->device, &tpci, NULL,
                                                  &dpool_t);
            if (tvr != VK_SUCCESS) {
                desc_bad = 1;
            }
            tinfos = calloc((size_t)ntsets * surf_slots, sizeof(*tinfos));
            VkWriteDescriptorSet *twrites =
                calloc(ntsets ? ntsets : 1, sizeof(*twrites));
            if (!tinfos || !twrites)
                desc_bad = 1;
            for (uint32_t t = 0; !desc_bad && t < ntsets; t++) {
                uint32_t vcount = surf_slots;
                VkDescriptorSetVariableDescriptorCountAllocateInfo vdca = {
                    .sType =
                        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
                    .descriptorSetCount = 1,
                    .pDescriptorCounts = &vcount,
                };
                VkDescriptorSetAllocateInfo tai = {
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .pNext = &vdca,
                    .descriptorPool = dpool_t,
                    .descriptorSetCount = 1,
                    .pSetLayouts = &e->surf_dsl[t],
                };
                tvr = vkAllocateDescriptorSets(ctx->device, &tai, &tsets[t]);
                if (tvr != VK_SUCCESS) {
                    desc_bad = 1;
                    break;
                }
                for (uint32_t i = 0; i < surf_slots; i++) {
                    acuda_buffer_t img =
                        surf_table ? surf_table[t * surf_slots + i] : NULL;
                    if (!img || img->kind != 2 || !img->view) {
                        desc_bad = 1;
                        break;
                    }
                    int d = img->dim3d ? 2 : (img->dim1d ? 0 : 1);
                    int a = !img->dim3d && img->layers > 1;
                    int fi = img->format == 3 ? 1
                             : (img->format == 0 ? 0 : -1);
                    if (info->surf_table_cas[t] ||
                        info->surf_table_lock[t]) {
                        // The fallback image is R32ui/R32f and requires
                        // the R32_SFLOAT host format.
                        fi = img->format == 2 ? 0 : -1;
                    }
                    if (d != (int)info->surf_table_dim[t] ||
                        a != (int)info->surf_table_arrayed[t] ||
                        fi != (int)info->surf_table_int[t]) {
                        desc_bad = 1;
                        break;
                    }
                    VkImageView iv = img->view;
                    if (info->surf_table_cas[t]) {
                        iv = img->uint_view;
                        if (!iv) {
                            desc_bad = 1;
                            break;
                        }
                    }
                    tinfos[(size_t)t * surf_slots + i].sampler =
                        VK_NULL_HANDLE;
                    tinfos[(size_t)t * surf_slots + i].imageView = iv;
                    tinfos[(size_t)t * surf_slots + i].imageLayout =
                        VK_IMAGE_LAYOUT_GENERAL;
                }
                twrites[t].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                twrites[t].dstSet = tsets[t];
                twrites[t].dstBinding = 0;
                twrites[t].descriptorCount = surf_slots;
                twrites[t].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                twrites[t].pImageInfo = &tinfos[(size_t)t * surf_slots];
            }
            if (!desc_bad)
                vkUpdateDescriptorSets(ctx->device, ntsets, twrites, 0, NULL);
            free(twrites);
        }
        free(writes);
        free(infos);
        free(iinfos);
        free(tinfos);
        writes = NULL;
        infos = NULL;
        iinfos = NULL;
        tinfos = NULL;
        if (!desc_bad && can_cache) {
            acuda_bind_cache_t *c =
                &e->bind_cache[e->bind_next++ % ACUDA_BIND_CACHE_N];
            c->stream_id = s->id;
            c->gen = s->batch_gen;
            c->dset = dset;
            c->nbufs = num_bufs;
            c->has_handles = handles != NULL;
            memcpy(c->bufs, bufs, sizeof(VkBuffer) * num_bufs);
            memcpy(c->offsets, offsets, sizeof(size_t) * num_bufs);
            memcpy(c->ranges, ranges, sizeof(size_t) * num_bufs);
            if (handles)
                memcpy(c->handles, handles,
                       sizeof(acuda_buffer_t) * num_bufs);
            bind_cache_addresses(ctx, c, bufs, num_bufs);
            // Promote this binding tuple to the cross-batch steady cache.
            // The pool is deliberately retained; it is released by
            // cache_destroy_entry when the pipeline cache is cleared.
            e->steady = *c;
            e->steady_pool = dpool;
            e->steady_valid = 1;
            e->cmd_pending = 0;
            e->cmd_valid = 0;
            dpool = VK_NULL_HANDLE;
        }
        if (desc_bad) {
            if (dpool_t)
                vkDestroyDescriptorPool(ctx->device, dpool_t, NULL);
            if (dpool)
                vkDestroyDescriptorPool(ctx->device, dpool, NULL);
            return ACUDA_ERROR_INVALID_VALUE;
        }
    }

    size_t apush_base = ((size_t)info->push_bytes +
                         info->tex_meta_bytes + 7u) & ~(size_t)7u;
    size_t total_push = apush_base + info->addr_push_bytes;
    void *pushbuf = NULL;
    if (total_push > info->push_bytes) {
        if (total_push > ctx->props.limits.maxPushConstantsSize)
            return ACUDA_ERROR_NOT_SUPPORTED;
        pushbuf = calloc(1, total_push);
        if (!pushbuf)
            return ACUDA_ERROR_OUT_OF_MEMORY;
        if (info->push_bytes > 0 && push_data)
            memcpy(pushbuf, push_data, info->push_bytes);
        uint32_t *mw = (uint32_t *)((char *)pushbuf + info->push_bytes);
        uint32_t k = 0;
        for (uint32_t i = 0; i < num_bufs; i++) {
            uint32_t kind = info->binding_kind[i];
            if (kind != 1 && kind != 2 && kind != 3)
                continue;
            if (!handles || !handles[i]) {
                free(pushbuf);
                return ACUDA_ERROR_INVALID_VALUE;
            }
            tex_meta_words(handles[i], kind, mw + k * 7);
            k++;
        }
        for (uint32_t k = 0; k < info->addr_push_bytes / 8u; k++) {
            uint32_t idx = info->bda_param[k];
            if (idx >= num_bufs) {
                free(pushbuf);
                return ACUDA_ERROR_INVALID_VALUE;
            }
            VkDeviceAddress a = 0;
            int null_arg = handles &&
                           handles[idx] == ACUDA_NULL_PARAM_HANDLE;
            if (null_arg) {
                // Null pointer argument: keep the pushed value zero so
                // the shader's null check selects the right path.  The
                // descriptor still points at the dummy buffer.
                a = 0;
            } else if (handles && handles[idx] && handles[idx]->addr) {
                a = handles[idx]->addr;
            } else if (bufs && bufs[idx] != VK_NULL_HANDLE) {
                // Arena-backed driver path: the launch has VkBuffers but
                // no acuda_buffer_t handles.
                VkBufferDeviceAddressInfo bdai = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                    .buffer = bufs[idx],
                };
                a = vkGetBufferDeviceAddress(ctx->device, &bdai);
            }
            if (a == 0 && !null_arg) {
                free(pushbuf);
                return ACUDA_ERROR_NOT_SUPPORTED;
            }
            if (offsets && !null_arg)
                a += offsets[idx];
            if (getenv("ACUDA_DEBUG_BDA_PUSH"))
                fprintf(stderr,
                        "ACuda: bda push %u: idx=%u addr=0x%llx base=%zu\n",
                        k, idx, (unsigned long long)a, apush_base);
            memcpy((char *)pushbuf + apush_base + (size_t)k * 8u, &a, 8);
        }
    }
    int has_storage = 0;
    for (uint32_t i = 0; i < num_bufs; i++)
        if (info->binding_kind[i] == 2)
            has_storage = 1;
    if (info->surf_table)
        has_storage = 1;
    // Raw device addresses (PhysicalStorageBuffer) bypass the per-buffer
    // dependency tracking, so a kernel that can resolve them needs a
    // global memory barrier around its dispatch.  Without it, a kernel
    // reading a buffer written by the previous launch can observe stale
    // (e.g. freshly zeroed) contents.
    int mem_barrier = has_storage || info->uses_bda_access != 0;
    // Image/texture work keeps the full-completion wait; buffer-only
    // submissions rely on the per-buffer dependency tracking.
    int serialize = has_storage;
    for (uint32_t d = 0; d < info->num_params; d++)
        if (info->is_ptr[d] == 1 && info->is_tex[d] != 0)
            serialize = 1;
    launch_rec_t rec = {
        .pipeline = e->pipe,
        .layout = e->pl,
        .push_data = pushbuf ? pushbuf : push_data,
        .push_bytes = total_push,
        .has_storage = has_storage,
        .mem_barrier = mem_barrier,
        .stream = s,
        .grid = grid_dim_x,
        .grid_y = grid_dim_y,
        .grid_z = grid_dim_z,
        .bufs = bufs,
        .offsets = offsets,
        .ranges = ranges,
        .num_bufs = num_bufs,
        .handles = handles,
        .written = info->param_written,
    };
    if (dset)
        rec.sets[rec.set_count++] = dset;
    for (uint32_t t = 0; t < ntsets; t++)
        rec.sets[rec.set_count++] = tsets[t];

    // If every launch parameter is invariant (animated values live in a
    // buffer, not in push constants), record one command buffer and
    // resubmit it verbatim. This skips vkCmdBind*/vkCmdPushConstants and
    // the per-launch record path entirely. It is only used when the
    // stream is idle, so the cached command buffer and its resources are
    // not in flight.
    int cmd_cache_ok = reused_set && e->steady_valid &&
                       dset == e->steady.dset && !info->surf_table &&
                       s->queue_family == ctx->queue_family &&
                       !has_storage && !serialize &&
                       !getenv("ACUDA_NO_CMD_CACHE") &&
                       (!e->cmd_valid || e->cmd_stream_id == s->id) &&
                       total_push <= sizeof(e->cmd_push);
    if (cmd_cache_ok) {
        const void *pdata = pushbuf ? pushbuf : push_data;
        int signature_match =
            e->cmd_stream_id == s->id && e->cmd_dset == dset &&
            e->cmd_grid == grid_dim_x &&
            e->cmd_grid_y == grid_dim_y &&
            e->cmd_grid_z == grid_dim_z &&
            e->cmd_push_bytes == total_push &&
            (total_push == 0 ||
             memcmp(e->cmd_push, pdata, total_push) == 0);
        if (e->cmd_valid && signature_match) {
            if (acuda_stream_cached_ready(ctx, s) == ACUDA_SUCCESS) {
                acuda_status_t sub =
                    acuda_stream_submit_cached(ctx, s, e->cmd, NULL);
                if (sub == ACUDA_SUCCESS) {
                    if (handles) {
                        for (uint32_t i = 0; i < num_bufs; i++) {
                            acuda_buffer_t h = handles[i];
                            if (!h || h == ACUDA_NULL_PARAM_HANDLE ||
                                h->kind != 0)
                                continue;
                            if (info->param_written[i]) {
                                h->gpu_dirty = 1;
                                h->gpu_read = 0;
                            } else {
                                h->gpu_read = 1;
                            }
                        }
                    }
                    free(pushbuf);
                    return ACUDA_SUCCESS;
                }
                if (sub != ACUDA_ERROR_NOT_READY) {
                    free(pushbuf);
                    return sub;
                }
            }
        } else if (!e->cmd_valid && e->cmd_pending && signature_match) {
            // The same invariant signature appeared twice. Record it once
            // and submit it from now on.
            if (acuda_stream_cached_ready(ctx, s) == ACUDA_SUCCESS) {
                acuda_status_t r = cache_record_cmd(ctx, e, &rec);
                if (r == ACUDA_SUCCESS) {
                    e->cmd_valid = 1;
                    acuda_status_t sub =
                        acuda_stream_submit_cached(ctx, s, e->cmd, NULL);
                    if (sub == ACUDA_SUCCESS) {
                        if (handles) {
                            for (uint32_t i = 0; i < num_bufs; i++) {
                                acuda_buffer_t h = handles[i];
                                if (!h || h == ACUDA_NULL_PARAM_HANDLE ||
                                    h->kind != 0)
                                    continue;
                                if (info->param_written[i]) {
                                    h->gpu_dirty = 1;
                                    h->gpu_read = 0;
                                } else {
                                    h->gpu_read = 1;
                                }
                            }
                        }
                        free(pushbuf);
                        return ACUDA_SUCCESS;
                    }
                    if (sub != ACUDA_ERROR_NOT_READY) {
                        free(pushbuf);
                        return sub;
                    }
                } else {
                    e->cmd_pending = 0;
                    e->cmd_valid = 0;
                }
            }
        } else {
            // New signature: remember it, but stay on the normal batch
            // path for this launch. If the next launch matches, record.
            e->cmd_stream_id = s->id;
            e->cmd_dset = dset;
            e->cmd_grid = grid_dim_x;
            e->cmd_grid_y = grid_dim_y;
            e->cmd_grid_z = grid_dim_z;
            e->cmd_push_bytes = (uint32_t)total_push;
            if (total_push)
                memcpy(e->cmd_push, pdata, total_push);
            e->cmd_pending = 1;
            e->cmd_valid = 0;
        }
    }
    if (lprof_on) {
        lprof_pre += (uint64_t)(lprof_now() - lpt0);
        lprof_n++;
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. launch appending grid=%u\n",
                      grid_dim_x);
    st = acuda_stream_append(ctx, s, record_launch, &rec, serialize);
    free(pushbuf);
    if (st != ACUDA_SUCCESS) {
        if (dpool_t)
            vkDestroyDescriptorPool(ctx->device, dpool_t, NULL);
        if (dpool)
            vkDestroyDescriptorPool(ctx->device, dpool, NULL);
        if (surf_meta)
            acuda_free(ctx, surf_meta);
        return st;
    }
    // The command is recorded; its resources must live until the batch
    // is submitted and completes.
    if (dpool)
        st = acuda_stream_track_dpool_pending(ctx, s, dpool);
    if (st == ACUDA_SUCCESS && dpool_t)
        st = acuda_stream_track_dpool_pending(ctx, s, dpool_t);
    if (st == ACUDA_SUCCESS && surf_meta)
        st = acuda_stream_track_buffer_pending(ctx, s, surf_meta);
    if (st != ACUDA_SUCCESS)
        return st;
    ACUDA_DEBUG_PRINT("acuda-trace .. launch flushing\n");
    st = acuda_stream_maybe_flush(ctx, s);
    ACUDA_DEBUG_PRINT("acuda-trace .. launch flushed st=%d submitted=%llu\n",
                (int)st, (unsigned long long)s->last_submitted);
    return st;
}

static acuda_status_t launch_raw_once(acuda_context_t ctx,
                                      acuda_kernel_t *kernel,
                                      VkBuffer *bufs,
                                      const size_t *offsets,
                                      const size_t *ranges,
                                      uint32_t num_bufs,
                                      acuda_buffer_t *handles,
                                      acuda_buffer_t *surf_table,
                                      uint32_t nsurf_table,
                                      uint32_t surf_slots,
                                      acuda_buffer_t surf_meta,
                                      const void *push_data,
                                      size_t push_bytes,
                                      uint32_t grid_dim_x,
                                      uint32_t grid_dim_y,
                                      uint32_t grid_dim_z,
                                      uint32_t block_dim_x,
                                      uint32_t block_dim_y,
                                      uint32_t block_dim_z,
                                      uint32_t dyn_shared_bytes,
                                      acuda_stream_t stream);
static acuda_status_t launch_host_chain_raw(acuda_context_t ctx,
                                            acuda_kernel_t *kernel,
                                            VkBuffer *bufs,
                                            const size_t *offsets,
                                            const size_t *ranges,
                                            uint32_t num_bufs,
                                            void **host_ptrs,
                                            const void *push_data,
                                            size_t push_bytes,
                                            uint32_t grid_dim_x,
                                            uint32_t grid_dim_y,
                                            uint32_t grid_dim_z,
                                            uint32_t block_dim_x,
                                            uint32_t block_dim_y,
                                            uint32_t block_dim_z,
                                            uint32_t dyn_shared_bytes,
                                            acuda_stream_t stream,
                                            const acuda_kernel_info_t *info);

// Common host-call service loop. `handles` is non-NULL when the caller
// has host-side buffer handles (needed to map global pointer arguments);
// the raw driver path has only VkBuffers and passes NULL.
static acuda_status_t launch_host_common(acuda_context_t ctx,
                                         acuda_kernel_t *kernel,
                                         VkBuffer *ubufs,
                                         const size_t *uoffs,
                                         const size_t *uranges,
                                         acuda_buffer_t *handles,
                                         void **host_ptrs,
                                         uint32_t num_bufs,
                                         const void *push_data,
                                         size_t push_bytes,
                                         uint32_t grid_dim_x,
                                         uint32_t grid_dim_y,
                                         uint32_t grid_dim_z,
                                         uint32_t block_dim_x,
                                         uint32_t block_dim_y,
                                         uint32_t block_dim_z,
                                         uint32_t dyn_shared_bytes,
                                         acuda_stream_t stream,
                                         const acuda_kernel_info_t *info) {
    if (num_bufs != info->num_ptr_params)
        return ACUDA_ERROR_INVALID_VALUE;
    if (grid_dim_x == 0 || grid_dim_y == 0 || grid_dim_z == 0 ||
        block_dim_x == 0 || block_dim_y == 0 || block_dim_z == 0 ||
        block_dim_x > ACUDA_MAX_BLOCK_X)
        return ACUDA_ERROR_INVALID_VALUE;
    if (grid_dim_x > ctx->props.limits.maxComputeWorkGroupCount[0] ||
        grid_dim_y > ctx->props.limits.maxComputeWorkGroupCount[1] ||
        grid_dim_z > ctx->props.limits.maxComputeWorkGroupCount[2] ||
        block_dim_x > ctx->props.limits.maxComputeWorkGroupSize[0] ||
        block_dim_y > ctx->props.limits.maxComputeWorkGroupSize[1] ||
        block_dim_z > ctx->props.limits.maxComputeWorkGroupSize[2])
        return ACUDA_ERROR_NOT_SUPPORTED;
    uint64_t inv64 = (uint64_t)grid_dim_x * grid_dim_y * grid_dim_z *
                     block_dim_x * block_dim_y * block_dim_z;
    if (inv64 == 0 || inv64 > (uint64_t)SIZE_MAX)
        return ACUDA_ERROR_INVALID_VALUE;
    size_t invocations = (size_t)inv64;
    uint64_t groups64 = (uint64_t)grid_dim_x * grid_dim_y * grid_dim_z;
    if (groups64 == 0 || groups64 > (uint64_t)SIZE_MAX)
        return ACUDA_ERROR_INVALID_VALUE;
    size_t groups = (size_t)groups64;
    if (info->host_state_stride == 0 || info->host_request_stride == 0)
        return ACUDA_ERROR_INTERNAL;
    if (invocations > SIZE_MAX / info->host_state_stride ||
        invocations > SIZE_MAX / info->host_request_stride)
        return ACUDA_ERROR_INVALID_VALUE;
    size_t shared_stride = info->host_shared_stride +
                           (((size_t)dyn_shared_bytes + 3u) & ~(size_t)3u);
    if (groups > SIZE_MAX / shared_stride)
        return ACUDA_ERROR_INVALID_VALUE;
    size_t shared_total = groups * shared_stride;
    if (shared_total > SIZE_MAX - 16)
        return ACUDA_ERROR_INVALID_VALUE;
    size_t state_offset = 16 + shared_total;
    if (invocations > (SIZE_MAX - state_offset) / info->host_state_stride)
        return ACUDA_ERROR_INVALID_VALUE;
    size_t state_size = state_offset + invocations * info->host_state_stride;
    size_t req_size = invocations * info->host_request_stride;

    acuda_buffer_t state = NULL, req = NULL;
    uint8_t *zero = NULL, *host_req = NULL;
    void **maps = NULL;
    VkDeviceMemory *map_mems = NULL;
    void *state_map = NULL;
    VkBuffer *raw = NULL;
    size_t *offs = NULL;
    size_t *ranges = NULL;
    acuda_status_t st = acuda_malloc(ctx, &state, state_size);
    if (st == ACUDA_SUCCESS)
        st = acuda_malloc(ctx, &req, req_size);
    if (st != ACUDA_SUCCESS)
        goto out;
    size_t zero_size = req_size > state_size ? req_size : state_size;
    zero = calloc(zero_size, 1);
    host_req = calloc(req_size, 1);
    if (!zero || !host_req) {
        st = ACUDA_ERROR_OUT_OF_MEMORY;
        goto out;
    }
    {
        uint32_t hdr[4] = {(uint32_t)state_offset,
                           (uint32_t)shared_stride, 0, 0};
        memcpy(zero, hdr, sizeof(hdr));
    }
    st = acuda_memcpy_h2d(ctx, state, zero, state_size, stream);
    if (st != ACUDA_SUCCESS)
        goto out;
    memset(zero, 0, req_size > 0 ? req_size : 1);
    st = acuda_memcpy_h2d(ctx, req, zero, req_size, stream);
    if (st != ACUDA_SUCCESS)
        goto out;

    // Resolve host mappings for user buffers. Handle-based launches use
    // their VkDeviceMemory; the raw path may still map buffers created
    // by acuda_malloc on this context. Explicit host pointers override
    // both (the arena driver passes them).
    if (num_bufs > 0) {
        maps = calloc(num_bufs, sizeof(*maps));
        map_mems = calloc(num_bufs, sizeof(*map_mems));
        if (!maps || !map_mems) {
            st = ACUDA_ERROR_OUT_OF_MEMORY;
            goto out;
        }
        for (uint32_t i = 0; i < num_bufs; i++) {
            if (host_ptrs && host_ptrs[i])
                continue;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            if (handles && handles[i] != ACUDA_NULL_PARAM_HANDLE)
                mem = handles[i]->memory;
            else {
                acuda_buffer_t b = acuda_buffer_find(ctx, ubufs[i]);
                if (b)
                    mem = b->memory;
            }
            if (mem == VK_NULL_HANDLE)
                continue; // no host mapping available for this binding
            VkResult vr = vkMapMemory(ctx->device, mem, 0, VK_WHOLE_SIZE, 0,
                                      &maps[i]);
            if (vr != VK_SUCCESS) {
                for (uint32_t j = 0; j < i; j++)
                    if (map_mems[j])
                        vkUnmapMemory(ctx->device, map_mems[j]);
                free(maps);
                free(map_mems);
                maps = NULL;
                map_mems = NULL;
                st = acuda_from_vk(vr);
                goto out;
            }
            map_mems[i] = mem;
        }
    }

    uint32_t total = num_bufs + 2;
    raw = malloc(sizeof(*raw) * total);
    offs = malloc(sizeof(*offs) * total);
    ranges = malloc(sizeof(*ranges) * total);
    if (!raw || !offs || !ranges) {
        st = ACUDA_ERROR_OUT_OF_MEMORY;
        goto out;
    }
    for (uint32_t i = 0; i < num_bufs; i++) {
        raw[i] = ubufs[i];
        offs[i] = uoffs[i];
        ranges[i] = uranges[i];
    }
    raw[num_bufs] = state->buffer;
    offs[num_bufs] = 0;
    ranges[num_bufs] = state_size;
    raw[num_bufs + 1] = req->buffer;
    offs[num_bufs + 1] = 0;
    ranges[num_bufs + 1] = req_size;

    int done = 0;
    uint64_t limit = UINT64_MAX;
    const char *lim_env = getenv("ACUDA_HOST_CALL_LIMIT");
    if (lim_env && lim_env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(lim_env, &end, 0);
        if (end != lim_env && v > 0)
            limit = (uint64_t)v;
    }
    for (uint64_t iter = 0; iter < limit && !done; iter++) {
        st = launch_raw_once(ctx, kernel, raw, offs, ranges, total, NULL,
                             NULL, 0, 0, NULL, push_data, push_bytes,
                             grid_dim_x, grid_dim_y, grid_dim_z,
                             block_dim_x, block_dim_y, block_dim_z,
                             dyn_shared_bytes, stream);
        if (st != ACUDA_SUCCESS)
            break;
        st = acuda_stream_synchronize(ctx, stream);
        if (st != ACUDA_SUCCESS)
            break;
        st = acuda_memcpy_d2h(ctx, host_req, req, req_size, stream);
        if (st != ACUDA_SUCCESS)
            break;

        size_t pending = 0;
        for (size_t i = 0; i < invocations; i++) {
            uint32_t p = 0;
            memcpy(&p, host_req + i * info->host_request_stride + 4, 4);
            if (p)
                pending++;
        }
        if (pending == 0) {
            done = 1;
            break;
        }

        {
            VkResult vr = vkMapMemory(ctx->device, state->memory, 0,
                                      VK_WHOLE_SIZE, 0, &state_map);
            if (vr != VK_SUCCESS) {
                st = acuda_from_vk(vr);
                break;
            }
        }

        for (size_t i = 0; i < invocations; i++) {
            uint8_t *slot = host_req + i * info->host_request_stride;
            uint32_t call_id = 0, p = 0;
            memcpy(&call_id, slot + 0, 4);
            memcpy(&p, slot + 4, 4);
            if (!p)
                continue;
            if (call_id >= info->host_call_count) {
                st = ACUDA_ERROR_INTERNAL;
                break;
            }
            const char *name = info->host_call_names[call_id];
            acuda_host_entry_t *e = acuda_host_call_find(ctx, name);
            if (!e || !e->fn) {
                fprintf(stderr,
                        "ACuda: no host callback registered for '%s'\n",
                        name);
                st = ACUDA_ERROR_NOT_SUPPORTED;
                break;
            }
            size_t abytes = info->host_arg_bytes[call_id];
            size_t rbytes = info->host_ret_bytes[call_id];
            if (abytes > ACUDA_HOST_REQ_RET_OFF -
                             ACUDA_HOST_REQ_ARGS_OFF ||
                rbytes > ACUDA_HOST_REQ_STRIDE -
                             ACUDA_HOST_REQ_RET_OFF) {
                st = ACUDA_ERROR_NOT_SUPPORTED;
                break;
            }
            for (uint32_t a = 0; a < info->host_nargs[call_id]; a++) {
                uint32_t bind = 0xffffffffu;
                memcpy(&bind,
                       slot + ACUDA_HOST_REQ_PTR_BIND_OFF + a * 4, 4);
                if (bind == 0xffffffffu)
                    continue;
                size_t aoff = ACUDA_HOST_REQ_ARGS_OFF +
                              info->host_arg_offsets[call_id][a];
                uint64_t off = 0;
                memcpy(&off, slot + aoff, 8);
                uint8_t *base = NULL;
                if (bind == 0xfffffffeu || bind == 0xfffffffdu) {
                    base = state_map ? (uint8_t *)state_map : NULL;
                } else if (host_ptrs && bind < num_bufs && host_ptrs[bind]) {
                    base = host_ptrs[bind];
                } else if (maps && bind < num_bufs && maps[bind]) {
                    base = (uint8_t *)maps[bind] + uoffs[bind];
                }
                if (!base) {
                    // No host mapping: externally created VkBuffers have
                    // no discoverable VkDeviceMemory. Use a handle-based
                    // launch or acuda_launch_raw_ex with host pointers.
                    st = ACUDA_ERROR_NOT_SUPPORTED;
                    break;
                }
                uint64_t hp = (uint64_t)(uintptr_t)(base + off);
                memcpy(slot + aoff, &hp, 8);
            }
            if (st != ACUDA_SUCCESS)
                break;
            int rc = e->fn(e->user, slot + ACUDA_HOST_REQ_ARGS_OFF,
                           abytes, slot + ACUDA_HOST_REQ_RET_OFF, rbytes);
            p = 0;
            memcpy(slot + 4, &p, 4);
            memcpy(slot + 8, &rc, 4);
            if (rc != 0) {
                st = ACUDA_ERROR_INTERNAL;
                break;
            }
            if (rbytes > 0 && state_map)
                memcpy((uint8_t *)state_map + state_offset +
                           i * info->host_state_stride +
                           info->host_ret_offsets[call_id],
                       slot + ACUDA_HOST_REQ_RET_OFF, rbytes);
        }
        if (state_map) {
            vkUnmapMemory(ctx->device, state->memory);
            state_map = NULL;
        }
        if (st != ACUDA_SUCCESS)
            break;
        st = acuda_memcpy_h2d(ctx, req, host_req, req_size, stream);
        if (st != ACUDA_SUCCESS)
            break;
    }
    if (st == ACUDA_SUCCESS && !done)
        st = ACUDA_ERROR_INTERNAL;

out:
    if (state_map)
        vkUnmapMemory(ctx->device, state->memory);
    if (maps) {
        for (uint32_t i = 0; i < num_bufs; i++)
            if (map_mems && map_mems[i])
                vkUnmapMemory(ctx->device, map_mems[i]);
        free(maps);
    }
    free(map_mems);
    free(raw);
    free(offs);
    free(ranges);
    free(zero);
    free(host_req);
    if (state)
        acuda_free(ctx, state);
    if (req)
        acuda_free(ctx, req);
    return st;
}

static acuda_status_t launch_host_chain(acuda_context_t ctx,
                                        acuda_kernel_t *kernel,
                                        acuda_buffer_t *bufs,
                                        uint32_t num_bufs,
                                        const void *push_data,
                                        size_t push_bytes,
                                        uint32_t grid_dim_x,
                                        uint32_t grid_dim_y,
                                        uint32_t grid_dim_z,
                                        uint32_t block_dim_x,
                                        uint32_t block_dim_y,
                                        uint32_t block_dim_z,
                                        uint32_t dyn_shared_bytes,
                                        acuda_stream_t stream,
                                        const acuda_kernel_info_t *info) {
    VkBuffer *raw = NULL;
    size_t *offs = NULL, *ranges = NULL;
    if (num_bufs > 0) {
        raw = malloc(sizeof(*raw) * num_bufs);
        offs = malloc(sizeof(*offs) * num_bufs);
        ranges = malloc(sizeof(*ranges) * num_bufs);
        if (!raw || !offs || !ranges) {
            free(raw);
            free(offs);
            free(ranges);
            return ACUDA_ERROR_OUT_OF_MEMORY;
        }
        for (uint32_t i = 0; i < num_bufs; i++) {
            raw[i] = bufs[i]->buffer;
            offs[i] = 0;
            ranges[i] = bufs[i]->size;
        }
    }
    acuda_status_t st = launch_host_common(
        ctx, kernel, raw, offs, ranges, bufs, NULL, num_bufs, push_data,
        push_bytes, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
        block_dim_y, block_dim_z, dyn_shared_bytes, stream, info);
    free(raw);
    free(offs);
    free(ranges);
    return st;
}

static acuda_status_t launch_host_chain_raw(acuda_context_t ctx,
                                            acuda_kernel_t *kernel,
                                            VkBuffer *bufs,
                                            const size_t *offsets,
                                            const size_t *ranges,
                                            uint32_t num_bufs,
                                            void **host_ptrs,
                                            const void *push_data,
                                            size_t push_bytes,
                                            uint32_t grid_dim_x,
                                            uint32_t grid_dim_y,
                                            uint32_t grid_dim_z,
                                            uint32_t block_dim_x,
                                            uint32_t block_dim_y,
                                            uint32_t block_dim_z,
                                            uint32_t dyn_shared_bytes,
                                            acuda_stream_t stream,
                                            const acuda_kernel_info_t *info) {
    return launch_host_common(ctx, kernel, bufs, offsets, ranges, NULL,
                              host_ptrs, num_bufs, push_data, push_bytes,
                              grid_dim_x, grid_dim_y, grid_dim_z,
                              block_dim_x, block_dim_y, block_dim_z,
                              dyn_shared_bytes, stream, info);
}

// Raw entry points used by the driver/cudart arena path. Host-call
// kernels are detected here and routed through the state-machine
// service loop automatically.
acuda_status_t acuda_launch_raw_ex(acuda_context_t ctx,
                                   acuda_kernel_t *kernel,
                                   VkBuffer *bufs,
                                   const size_t *offsets,
                                   const size_t *ranges,
                                   uint32_t num_bufs,
                                   void **host_ptrs,
                                   acuda_buffer_t *tex_handles,
                                   acuda_buffer_t *surf_table,
                                   uint32_t nsurf_table,
                                   uint32_t surf_slots,
                                   acuda_buffer_t surf_meta,
                                   const void *push_data,
                                   size_t push_bytes,
                                   uint32_t grid_dim_x,
                                   uint32_t grid_dim_y,
                                   uint32_t grid_dim_z,
                                   uint32_t block_dim_x,
                                   uint32_t block_dim_y,
                                   uint32_t block_dim_z,
                                   uint32_t dyn_shared_bytes,
                                   acuda_stream_t stream) {
    ACUDA_DEBUG_PRINT("acuda-trace .. acuda_launch_raw_ex: entry k=%p\n", (void *)kernel);
    const acuda_kernel_info_t *info = acuda_kernel_info_ptr(kernel);
    int have_info = info != NULL;
    ACUDA_DEBUG_PRINT("acuda-trace .. acuda_launch_raw_ex: info=%p have=%d\n", (void *)info, have_info);
    if (have_info && info->host_segment_count > 0) {
        uint32_t user = num_bufs;
        if (num_bufs == acuda_kernel_launch_ptr_params(kernel))
            user = info->num_ptr_params;
        else if (num_bufs != info->num_ptr_params)
            return ACUDA_ERROR_INVALID_VALUE;
        return launch_host_chain_raw(ctx, kernel, bufs, offsets, ranges,
                                     user, host_ptrs, push_data, push_bytes,
                                     grid_dim_x, grid_dim_y, grid_dim_z,
                                     block_dim_x, block_dim_y, block_dim_z,
                                     dyn_shared_bytes, stream, info);
    }
    if (tex_handles) {
        // Texture/sampler bindings have no VkBuffer; clear the buffer
        // slots so barrier emission skips them. Storage-buffer handles
        // (including large by-value parameter buffers) keep their
        // VkBuffer while still participating in barrier tracking.
        ACUDA_DEBUG_PRINT("acuda-trace .. acuda_launch_raw_ex: tex_handles loop n=%u\n", num_bufs);
        for (uint32_t i = 0; i < num_bufs; i++)
            if (tex_handles[i] &&
                (!have_info || info->binding_kind[i] != 0))
                bufs[i] = VK_NULL_HANDLE;
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. acuda_launch_raw_ex: calling launch_raw_once\n");
    return launch_raw_once(ctx, kernel, bufs, offsets, ranges, num_bufs,
                           tex_handles, surf_table, nsurf_table, surf_slots,
                           surf_meta,
                           push_data, push_bytes, grid_dim_x, grid_dim_y,
                           grid_dim_z, block_dim_x, block_dim_y, block_dim_z,
                           dyn_shared_bytes, stream);
}

acuda_status_t acuda_launch_raw(acuda_context_t ctx, acuda_kernel_t *kernel,
                               VkBuffer *bufs, const size_t *offsets,
                               const size_t *ranges, uint32_t num_bufs,
                               const void *push_data, size_t push_bytes,
                               uint32_t grid_dim_x, uint32_t block_dim_x,
                               uint32_t dyn_shared_bytes,
                               acuda_stream_t stream) {
    return acuda_launch_raw_ex(ctx, kernel, bufs, offsets, ranges, num_bufs,
                               NULL, NULL, NULL, 0, 0, NULL, push_data,
                               push_bytes, grid_dim_x, 1, 1, block_dim_x,
                               1, 1, dyn_shared_bytes, stream);
}

acuda_status_t acuda_launch_kernel_ex(acuda_context_t ctx,
                                      acuda_kernel_t *kernel,
                                      acuda_buffer_t *bufs,
                                      uint32_t num_bufs,
                                      const void *push_data,
                                      size_t push_bytes,
                                      uint32_t grid_dim_x,
                                      uint32_t block_dim_x,
                                      uint32_t dyn_shared_bytes,
                                      acuda_stream_t stream) {
    if (num_bufs > 0 && !bufs)
        return ACUDA_ERROR_INVALID_VALUE;
    {
        const acuda_kernel_info_t *hinfo = acuda_kernel_info_ptr(kernel);
        if (hinfo && hinfo->host_segment_count > 0)
            return launch_host_chain(ctx, kernel, bufs, num_bufs,
                                     push_data, push_bytes, grid_dim_x,
                                     1, 1, block_dim_x, 1, 1,
                                     dyn_shared_bytes, stream, hinfo);
    }
    VkBuffer *raw = NULL;
    size_t *offs = NULL, *ranges = NULL;
    if (num_bufs > 0) {
        raw = malloc(sizeof(*raw) * num_bufs);
        offs = malloc(sizeof(*offs) * num_bufs);
        ranges = malloc(sizeof(*ranges) * num_bufs);
        if (!raw || !offs || !ranges) {
            free(raw);
            free(offs);
            free(ranges);
            return ACUDA_ERROR_OUT_OF_MEMORY;
        }
        for (uint32_t i = 0; i < num_bufs; i++) {
            if (!bufs[i]) {
                free(raw);
                free(offs);
                free(ranges);
                return ACUDA_ERROR_INVALID_VALUE;
            }
            raw[i] = bufs[i]->buffer;
            offs[i] = 0;
            ranges[i] = bufs[i]->size;
        }
    }
    acuda_status_t st = launch_raw_once(ctx, kernel, raw, offs, ranges,
                                        num_bufs, bufs, NULL, 0, 0, NULL,
                                        push_data, push_bytes, grid_dim_x,
                                        1, 1, block_dim_x, 1, 1,
                                        dyn_shared_bytes, stream);
    free(raw);
    free(offs);
    free(ranges);
    return st;
}

acuda_status_t acuda_launch_kernel(acuda_context_t ctx, acuda_kernel_t *kernel,
                                   acuda_buffer_t *bufs, uint32_t num_bufs,
                                   const void *push_data, size_t push_bytes,
                                   uint32_t grid_dim_x, uint32_t block_dim_x,
                                   acuda_stream_t stream) {
    return acuda_launch_kernel_ex(ctx, kernel, bufs, num_bufs, push_data,
                                  push_bytes, grid_dim_x, block_dim_x, 0,
                                  stream);
}
