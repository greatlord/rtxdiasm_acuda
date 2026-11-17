#include <stdlib.h>
#include <string.h>
#include "acuda/runtime/internal.h"
#include "acuda/runtime/launch_internal.h"

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
    const acuda_kernel_info_t *info = acuda_kernel_info_ptr(kernel);
    int have_info = info != NULL;
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
        for (uint32_t i = 0; i < num_bufs; i++)
            if (tex_handles[i] &&
                (!have_info || info->binding_kind[i] != 0))
                bufs[i] = VK_NULL_HANDLE;
    }
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
