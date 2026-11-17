#ifndef ACUDA_LAUNCH_INTERNAL_H
#define ACUDA_LAUNCH_INTERNAL_H

#include "internal.h"

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
                               acuda_stream_t stream);

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
                                     const acuda_kernel_info_t *info);

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
                                 const acuda_kernel_info_t *info);

#endif
