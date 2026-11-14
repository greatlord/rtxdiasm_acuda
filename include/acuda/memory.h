// ACuda device-memory management. C ABI.
#ifndef ACUDA_MEMORY_H
#define ACUDA_MEMORY_H

#include <stddef.h>
#include <stdint.h>
#include "acuda/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Vulkan-image-backed texture objects.
// format: 0 = RGBA32F, 1 = RGBA8_UNORM, 2 = R32F, 3 = R32_UINT,
//         4 = D32_SFLOAT (depth texture).
// dim: 0 = infer (3D when depth > 1, 2D otherwise), 1 = 1D, 2 = 2D,
//      3 = 3D.  cube = 1 marks a cube or cube-array texture (layers is
//      a multiple of six).  samples > 1 creates a real multisampled
//      2D/2D-array image (single mip level, float-sampled formats);
//      the source texels are replicated to every sample.
typedef struct acuda_image_desc {
    uint32_t width;
    uint32_t height;
    uint32_t depth;   // 1 for 1D/2D
    uint32_t layers;  // 1 for non-arrayed
    uint32_t levels;  // mip levels
    uint32_t samples; // 1 for non-multisample
    // 0 = RGBA32F, 1 = RGBA8_UNORM, 2 = R32_SFLOAT, 3 = R32_UINT,
    // 4 = D32_SFLOAT (textures only), 5 = R64_UINT (surfaces,
    // SPV_EXT_shader_image_int64).
    uint32_t format;
    uint32_t dim;
    uint32_t cube;
} acuda_image_desc;

// filter: 0 nearest, 1 linear.  addr: 0 wrap, 1 mirror, 2 clamp, 3 border.
typedef struct acuda_sampler_desc {
    uint32_t filter;
    uint32_t mip_filter;
    uint32_t addr[3];
    uint32_t normalized;
    float border[4];
    uint32_t compare;    // 1 = depth-compare sampler
    uint32_t compare_op; // VkCompareOp when compare
} acuda_sampler_desc;

acuda_status_t acuda_image_create(acuda_context_t ctx, acuda_buffer_t *out,
                                  const acuda_image_desc *desc,
                                  const void *texels);
// Upload one mip level / array layer of an image.  texels holds
// width*height*depth texels of the level's extent (clamped to >= 1).
acuda_status_t acuda_image_upload(acuda_context_t ctx, acuda_buffer_t image,
                                  uint32_t level, uint32_t layer,
                                  const void *texels);
// Read one mip level / array layer back into host memory.
acuda_status_t acuda_image_download(acuda_context_t ctx, acuda_buffer_t image,
                                    uint32_t level, uint32_t layer,
                                    void *texels);
acuda_status_t acuda_sampler_create(acuda_context_t ctx, acuda_buffer_t *out,
                                    const acuda_sampler_desc *desc);
// Build a combined-image-sampler handle from image + sampler handles.
acuda_status_t acuda_texture_create_image(acuda_context_t ctx,
                                          acuda_buffer_t *out,
                                          acuda_buffer_t image,
                                          acuda_buffer_t sampler);

// Convenience texture descriptor: builds an image + sampler and returns
// a combined handle.  Supported layouts: channels = 4 with format = 0
// (RGBA32F), channels = 1 with format = 0 (R32F), and channels = 1 with
// format = 1/2 (R32_UINT).  texels are tightly packed 32-bit components.
// filter = 0 nearest, 1 linear; addr modes = 0 wrap, 1 mirror,
// 2 clamp, 3 border.  normalized = 1 selects normalized coordinates.
// The returned handle owns the image and sampler.
typedef struct acuda_texture_desc {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t channels;
    uint32_t format;
    uint32_t filter;
    uint32_t normalized;
    uint32_t addr[3];
} acuda_texture_desc;

acuda_status_t acuda_texture_create(acuda_context_t ctx,
                                    acuda_buffer_t *out_buf,
                                    const acuda_texture_desc *desc,
                                    const void *texels);

// Read-write surface (storage image).  format: 0 RGBA32F, 1 RGBA8,
// 2 R32F, 3 R32_UINT.  .surfref parameters bind these.
acuda_status_t acuda_surface_create(acuda_context_t ctx, acuda_buffer_t *out,
                                    const acuda_image_desc *desc,
                                    const void *texels);
// Whole level-0 upload/download (all layers, tightly packed).
acuda_status_t acuda_surface_upload(acuda_context_t ctx, acuda_buffer_t surf,
                                    const void *texels);
acuda_status_t acuda_surface_download(acuda_context_t ctx,
                                      acuda_buffer_t surf, void *texels);

// Host-visible device memory. size in bytes.
acuda_status_t acuda_malloc(acuda_context_t ctx, acuda_buffer_t *out_buf, size_t bytes);
// Buffer device address (Vulkan 1.2 bufferDeviceAddress). Returns
// ACUDA_ERROR_NOT_SUPPORTED when the device/feature is unavailable.
acuda_status_t acuda_buffer_device_address(acuda_buffer_t buf,
                                           uint64_t *out_addr);
// Device-local buffer (VRAM) when the device has such a memory type;
// falls back to host-visible memory on integrated devices. H2D/D2H use a
// staging copy; D2D and memset stay on the queue.
acuda_status_t acuda_malloc_device(acuda_context_t ctx, acuda_buffer_t *buf,
                                   size_t bytes);
acuda_status_t acuda_free(acuda_context_t ctx, acuda_buffer_t buf);
acuda_status_t acuda_buffer_size(acuda_buffer_t buf, size_t *out_bytes);

// stream may be NULL (= default stream). H2D/D2H synchronize the stream
// first so host access is ordered w.r.t. prior device work.
acuda_status_t acuda_memcpy_h2d(acuda_context_t ctx, acuda_buffer_t dst,
                                const void *src, size_t bytes, acuda_stream_t stream);
acuda_status_t acuda_memcpy_d2h(acuda_context_t ctx, void *dst,
                                acuda_buffer_t src, size_t bytes, acuda_stream_t stream);
// Queued on stream (or default), returns before completion.
acuda_status_t acuda_memcpy_d2d(acuda_context_t ctx, acuda_buffer_t dst,
                                acuda_buffer_t src, size_t bytes, acuda_stream_t stream);
// offset/bytes must be multiples of 4.
acuda_status_t acuda_memset_d32(acuda_context_t ctx, acuda_buffer_t buf,
                                uint32_t value, size_t offset, size_t bytes,
                                acuda_stream_t stream);
acuda_status_t acuda_memset_d8(acuda_context_t ctx, acuda_buffer_t buf,
                               uint8_t value, size_t offset, size_t bytes,
                               acuda_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif
