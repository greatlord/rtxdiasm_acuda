#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "acuda/compiler.h"
#include "acuda/runtime/internal.h"
#include "acuda/runtime/launch_internal.h"
#include "acuda/driver/driver_export_dd9e98a4.h"
#include "acuda/runtime/time_compat.h"
#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#endif
#include "acuda/runtime/launch_internal.h"
#include "acuda/driver/driver_export_dd9e98a4.h"

acuda_status_t launch_raw_once(acuda_context_t ctx,
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
    acuda_stream_t s = NULL;
    acuda_status_t st = acuda_resolve_stream(ctx, stream, &s);
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
    if (!info)
        return ACUDA_ERROR_INTERNAL;
    if (info->surf_table && (!surf_table || !surf_meta))
        return ACUDA_ERROR_NOT_SUPPORTED;
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
    if (info->surf_table) {
        if (!ctx->desc_idx_ok)
            return ACUDA_ERROR_NOT_SUPPORTED;
        if (info->surf_table_count > ACUDA_SURF_TABLE_MAX_TYPES)
            return ACUDA_ERROR_NOT_SUPPORTED;
        if (surf_slots == 0)
            return ACUDA_ERROR_INVALID_VALUE;
        if (surf_slots > surf_max_slots(ctx, info->surf_table_count))
            return ACUDA_ERROR_NOT_SUPPORTED;
    }
    if (num_bufs != acuda_kernel_launch_ptr_params(kernel))
        return ACUDA_ERROR_INVALID_VALUE;
    if (num_bufs > ACUDA_MAX_PARAMS)
        return ACUDA_ERROR_NOT_SUPPORTED;
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

    // f64 kernels need shaderFloat64: fail cleanly here, never garbage.
    if (info->requires_float64 == 2) {
        if (!ctx->f64_ok)
            return ACUDA_ERROR_NOT_SUPPORTED;
    } else if (info->requires_float64 == 0) {
        /* old blob: scan SPIR-V */
        const uint32_t *words = NULL;
        size_t nwords = 0;
        st = acuda_kernel_spirv(kernel, &words, &nwords);
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

    launch_cache_t *e = NULL;
    // Dynamic words specialize the extern array (min 1: zero-length
    // arrays are invalid SPIR-V even before use).
    uint32_t dyn_words = dyn_shared_bytes / 4 + (dyn_shared_bytes % 4 ? 1u : 0u);
    if (dyn_words == 0)
        dyn_words = 1;
    st = cache_lookup(ctx, kernel, info, block_dim_x, block_dim_y,
                      block_dim_z, dyn_words, &e);
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
    {
        static int _rtinit = 0; static long long _rtseq = 0;
        if (!_rtinit) { _rtinit = 1; /* truncate once */ const char *_rtp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_rtp && _rtp[0]) { FILE *_f = fopen(_rtp,"w"); if(_f) fclose(_f); else fprintf(stderr, "ACuda-debug: runtime trace fopen failed: %s\n", _rtp); } }
        { const char *_rtp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_rtp && *_rtp) { FILE *_f = fopen(_rtp,"a"); if(_f){ fprintf(_f,"%lld append g=%u\n", _rtseq++, grid_dim_x); fflush(_f); fclose(_f);} } }
        ACUDA_DEBUG_PRINT("acuda-trace .. launch appending grid=%u\n",
                          grid_dim_x);
        st = acuda_stream_append(ctx, s, record_launch, &rec, serialize);
        { const char *_rtp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_rtp && *_rtp) { FILE *_f = fopen(_rtp,"a"); if(_f){ fprintf(_f,"%lld appended st=%d\n", _rtseq++, (int)st); fflush(_f); fclose(_f);} } }
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
        if (st != ACUDA_SUCCESS) {
            { const char *_rtp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_rtp && *_rtp) { FILE *_f = fopen(_rtp,"a"); if(_f){ fprintf(_f,"%lld track_err st=%d\n", _rtseq++, (int)st); fflush(_f); fclose(_f);} } }
            return st;
        }
        ACUDA_DEBUG_PRINT("acuda-trace .. launch flushing\n");
        { const char *_rtp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_rtp && *_rtp) { FILE *_f = fopen(_rtp,"a"); if(_f){ fprintf(_f,"%lld flush\n", _rtseq++); fflush(_f); fclose(_f);} } }
        st = acuda_stream_maybe_flush(ctx, s);
        ACUDA_DEBUG_PRINT("acuda-trace .. launch flushed st=%d submitted=%llu\n",
                    (int)st, (unsigned long long)s->last_submitted);
        { const char *_rtp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_rtp && *_rtp) { FILE *_f = fopen(_rtp,"a"); if(_f){ fprintf(_f,"%lld flushed st=%d sub=%llu\n", _rtseq++, (int)st, (unsigned long long)s->last_submitted); fflush(_f); fclose(_f);} } }
        return st;
    }
}
