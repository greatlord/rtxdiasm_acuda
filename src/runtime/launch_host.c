#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "internal.h"
#include "launch_internal.h"

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
    size_t shared_total = 0;
    if (shared_stride == 0) {
        shared_total = 0;
    } else {
        if (groups > SIZE_MAX / shared_stride)
            return ACUDA_ERROR_INVALID_VALUE;
        shared_total = groups * shared_stride;
    }
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

acuda_status_t launch_host_chain(acuda_context_t ctx,
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

acuda_status_t launch_host_chain_raw(acuda_context_t ctx,
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
