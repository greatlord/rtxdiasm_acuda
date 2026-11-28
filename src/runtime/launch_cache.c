#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "acuda/compiler.h"
#include "acuda/driver/arena.h"
#include "acuda/runtime/internal.h"
#include "acuda/runtime/launch_internal.h"
#include "acuda/runtime/time_compat.h"
#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#endif
#include "acuda/runtime/launch_internal.h"

int lprof_on = -1;
uint64_t lprof_n, lprof_pre;
double lprof_now(void) {
    return (double)acuda_now_ns();
}
void lprof_dump(void) {
    if (lprof_n)
        fprintf(stderr, "ACuda launch pre-work: %.2f us/launch\n",
                (double)lprof_pre / (double)lprof_n / 1e3);
}



// Resolve the device address of every bound buffer so descriptor-cache
// identity follows the memory, not the recyclable VkBuffer handle.
void bind_cache_addresses(acuda_context_t ctx, acuda_bind_cache_t *c,
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

uint32_t surf_max_slots(acuda_context_t ctx, uint32_t count) {
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
VkDescriptorType binding_vktype(uint32_t kind) {
    switch (kind) {
    case 1: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    case 2: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case 3: return VK_DESCRIPTOR_TYPE_SAMPLER;
    default: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
}

acuda_status_t cache_lookup(acuda_context_t ctx, const acuda_kernel_t *k,
                                   const acuda_kernel_info_t *info,
                                   uint32_t block_dim_x, uint32_t block_dim_y,
                                   uint32_t block_dim_z, uint32_t dyn_words,
                                   launch_cache_t **out) {
    cache_lock_acquire();
    for (size_t i = 0; i < cache_count; i++) {
        if (cache[i]->kernel == k && cache[i]->serial == info->serial &&
            cache[i]->ctx == ctx && cache[i]->block_x == block_dim_x &&
            cache[i]->block_y == block_dim_y &&
            cache[i]->block_z == block_dim_z &&
            cache[i]->dyn_words == dyn_words &&
            cache[i]->arena_epoch == acuda_arena_token_epoch()) {
            *out = cache[i];
            cache_lock_release();
            return ACUDA_SUCCESS;
        }
    }
    cache_lock_release();
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
    launch_cache_t *e = calloc(1, sizeof(*e));
    if (!e)
        return ACUDA_ERROR_OUT_OF_MEMORY;
    e->kernel = k;
    e->serial = info->serial;
    e->ctx = ctx;
    e->block_x = block_dim_x;
    e->block_y = block_dim_y;
    e->block_z = block_dim_z;
    e->dyn_words = dyn_words;
    e->arena_epoch = acuda_arena_token_epoch();

    const uint32_t *words = NULL;
    size_t nwords = 0;
    acuda_status_t st = acuda_kernel_spirv(k, &words, &nwords);
    if (st != ACUDA_SUCCESS)
        return st;
    // Drivers without maintenance4 cannot consume OpExecutionModeId
    // LocalSizeId. Patch a private copy to OpExecutionMode LocalSize using
    // the launch's actual block size; the pipeline cache is already keyed by
    // block_x, so no fixed workgroup size is assumed.
    uint32_t *patched_words = NULL;
    const uint32_t *shader_words = words;
    if (!ctx->maintenance4_ok ||
        ctx->props.apiVersion < VK_API_VERSION_1_3) {
        patched_words = malloc(nwords * sizeof(uint32_t));
        if (!patched_words) {
            cache_destroy_entry(ctx, e);
            return ACUDA_ERROR_OUT_OF_MEMORY;
        }
        memcpy(patched_words, words, nwords * sizeof(uint32_t));
        // Vulkan 1.3+ supports SPIR-V 1.6; downgrade for older devices.
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
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = nwords * 4,
        .pCode = shader_words,
    };
    VkResult vr = vkCreateShaderModule(ctx->device, &smci, NULL, &e->mod);
    free(patched_words);
    acuda_kernel_release_spirv((acuda_kernel_t *)k);
    if (vr != VK_SUCCESS)
        return acuda_from_vk(vr);

    uint32_t full_ptrs = acuda_kernel_launch_ptr_params(k);
    if (full_ptrs > ACUDA_MAX_PARAMS) {
        cache_destroy_entry(ctx, e);
        return ACUDA_ERROR_NOT_SUPPORTED;
    }
    uint32_t nbind = full_ptrs + (info->surf_table ? 1u : 0u) +
                     (info->surf_lock ? 1u : 0u);
    VkDescriptorSetLayoutBinding *bindings = NULL;
    if (nbind > 0) {
        bindings = calloc(nbind, sizeof(*bindings));
        if (!bindings) {
            cache_destroy_entry(ctx, e);
            return ACUDA_ERROR_OUT_OF_MEMORY;
        }
        for (uint32_t i = 0; i < full_ptrs; i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType =
                binding_vktype(info->binding_kind[i]);
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].pImmutableSamplers = NULL;
        }
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
    VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = nbind,
        .pBindings = bindings,
    };
    vr = vkCreateDescriptorSetLayout(ctx->device, &dlci, NULL, &e->dsl);
    free(bindings);
    if (vr != VK_SUCCESS) {
        cache_destroy_entry(ctx, e);
        return acuda_from_vk(vr);
    }

    uint32_t max_slots = surf_max_slots(ctx, info->surf_table_count);
    uint32_t nsets = 1;
    if (info->surf_table && info->surf_table_count > 0) {
        if (info->surf_table_count > ACUDA_SURF_TABLE_MAX_TYPES) {
            cache_destroy_entry(ctx, e);
            return ACUDA_ERROR_NOT_SUPPORTED;
        }
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
    uint32_t addr_base = (info->push_bytes + info->tex_meta_bytes + 7u) & ~7u;
    uint32_t push_total = addr_base + info->addr_push_bytes;
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
    vr = vkCreatePipelineLayout(ctx->device, &plci, NULL, &e->pl);
    free(sls);
    if (vr != VK_SUCCESS) {
        cache_destroy_entry(ctx, e);
        return acuda_from_vk(vr);
    }

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
    // plus the dynamic-.shared word count (3) that sizes an
    // `.extern .shared` array. Slot 3 is always supplied: a kernel
    // without an extern array has no spec constant id 3 and Vulkan
    // ignores the extra map entry, while gating it on info->has_extern
    // left the array at its default length 1 and corrupted shared-
    // memory indexing in kernels (e.g. the fluid reorder) whose extern
    // flag was not propagated.
    uint32_t spec_data[4 + 32] = { block_dim_x, block_dim_y, block_dim_z,
                                   dyn_words };
    VkSpecializationMapEntry spec_map[4 + 32] = {
        { .constantID = 0, .offset = 0, .size = sizeof(uint32_t) },
        { .constantID = 1, .offset = 4, .size = sizeof(uint32_t) },
        { .constantID = 2, .offset = 8, .size = sizeof(uint32_t) },
        { .constantID = 3, .offset = 12, .size = sizeof(uint32_t) },
    };
    uint32_t map_count = 4u;
    // Arena token table: SpecId 4+4i+{0,1,2,3} = {base, size, dlo, dhi}.
    // Unused slots keep size 0, which makes the shader's range check false.
    {
        uint32_t tb[8], ts[8], tlo[8], thi[8];
        int nt = acuda_arena_token_table(tb, ts, tlo, thi, 8);
        for (int i = 0; i < nt; i++) {
            uint32_t w = 4u + (uint32_t)i * 4u;
            spec_data[w + 0] = tb[i];
            spec_data[w + 1] = ts[i];
            spec_data[w + 2] = tlo[i];
            spec_data[w + 3] = thi[i];
            for (uint32_t k = 0; k < 4u; k++)
                spec_map[w + k] = (VkSpecializationMapEntry){
                    .constantID = w + k,
                    .offset = (w + k) * (uint32_t)sizeof(uint32_t),
                    .size = sizeof(uint32_t),
                };
        }
        map_count = 4u + (uint32_t)nt * 4u;
    }
    VkSpecializationInfo spec_info = {
        .mapEntryCount = map_count,
        .pMapEntries = spec_map,
        .dataSize = map_count * sizeof(uint32_t),
        .pData = spec_data,
    };
    if (getenv("ACUDA_TRACE"))
        ACUDA_DEBUG_PRINT("acuda-trace .. specialize %s has_extern=%u dyn_words=%u block=%u\n",
                          info->entry, info->has_extern, dyn_words,
                          block_dim_x);
    stage.pSpecializationInfo = &spec_info;
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage,
        .layout = e->pl,
    };
    if (ctx->pipe_exec_ok && getenv("ACUDA_DUMP_ISA"))
        cpci.flags |=
            VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    vr = vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                  &e->pipe);
    if (vr != VK_SUCCESS) {
        cache_destroy_entry(ctx, e);
        return acuda_from_vk(vr);
    }
    dump_pipeline_isa(ctx, e->pipe, info->entry);
    // Publish the fully built entry. Another thread may have created the
    // same one while we were building; keep theirs and drop ours.
    cache_lock_acquire();
    for (size_t i = 0; i < cache_count; i++) {
        if (cache[i]->kernel == k && cache[i]->serial == info->serial &&
            cache[i]->ctx == ctx && cache[i]->block_x == block_dim_x &&
            cache[i]->block_y == block_dim_y &&
            cache[i]->block_z == block_dim_z &&
            cache[i]->dyn_words == dyn_words &&
            cache[i]->arena_epoch == acuda_arena_token_epoch()) {
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
        if (cache_count > 0)
            memcpy(nc, cache, sizeof(*nc) * cache_count);
        free(cache);
        cache = nc;
        cache_cap = ncap;
    }
    cache[cache_count++] = e;
    cache_lock_release();
    *out = e;
    return ACUDA_SUCCESS;
}


acuda_status_t record_launch(VkCommandBuffer cmd, void *user) {
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
    uint32_t rset_count = r->set_count;
    if (rset_count > 1u + ACUDA_SURF_TABLE_MAX_TYPES)
        rset_count = 1u + ACUDA_SURF_TABLE_MAX_TYPES;
    int same_sets = s && rset_count == s->bind_set_count;
    if (same_sets) {
        for (uint32_t i = 0; i < rset_count; i++)
            if (r->sets[i] != s->bind_sets[i]) {
                same_sets = 0;
                break;
            }
    }
    if (!same_sets && rset_count > 0) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r->layout,
                                0, rset_count, r->sets, 0, NULL);
        if (s) {
            s->bind_set_count = rset_count;
            memcpy(s->bind_sets, r->sets,
                   sizeof(VkDescriptorSet) * rset_count);
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
acuda_status_t cache_record_cmd(acuda_context_t ctx, launch_cache_t *e,
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
uint32_t tex_addr_code(uint32_t a) {
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
void tex_meta_words(acuda_buffer_t h, uint32_t kind, uint32_t out[7]) {
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
