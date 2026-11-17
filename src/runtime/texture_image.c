// Vulkan-image-backed texture objects: images, samplers and combined
// image-sampler handles.  The runtime's ordinary allocations remain
// VkBuffers; these handles extend the same opaque acuda_buffer_t type.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/memory.h"
#include "acuda/runtime/internal.h"
#include "acuda/runtime/texture_image_internal.h"

VkFormat image_vkformat(uint32_t f) {
    switch (f) {
    case 0: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case 1: return VK_FORMAT_R8G8B8A8_UNORM;
    case 2: return VK_FORMAT_R32_SFLOAT;
    case 3: return VK_FORMAT_R32_UINT;
    case 4: return VK_FORMAT_D32_SFLOAT;
    case 5: return VK_FORMAT_R64_UINT;
    default: return VK_FORMAT_UNDEFINED;
    }
}

uint32_t image_elem_bytes(uint32_t f) {
    switch (f) {
    case 0: return 16;
    case 1: return 4;
    case 2: return 4;
    case 3: return 4;
    case 4: return 4;
    case 5: return 8;
    default: return 0;
    }
}

int image_is_depth(uint32_t f) {
    return f == 4;
}

int image_format_storage_ok(acuda_context_t ctx, VkFormat fmt) {
    VkFormatProperties3 fp3 = {
        .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3,
    };
    VkFormatProperties2 fp2 = {
        .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
        .pNext = &fp3,
    };
    vkGetPhysicalDeviceFormatProperties2(ctx->phys, fmt, &fp2);
    return (fp3.optimalTilingFeatures &
            VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
}

uint32_t find_image_mem(acuda_context_t ctx, uint32_t bits) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(ctx->phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(bits & (1u << i)))
            continue;
        if (mp.memoryTypes[i].propertyFlags &
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            return i;
    }
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if (bits & (1u << i))
            return i;
    return 0;
}

void image_range(acuda_buffer_t b, VkImageSubresourceRange *r) {
    r->aspectMask = image_is_depth(b->format) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                              : VK_IMAGE_ASPECT_COLOR_BIT;
    r->baseMipLevel = 0;
    r->levelCount = b->levels ? b->levels : 1;
    r->baseArrayLayer = 0;
    r->layerCount = b->layers ? b->layers : 1;
}

void image_barrier_img(VkCommandBuffer cmd, acuda_buffer_t b,
                              VkImage img, VkImageLayout from,
                              VkImageLayout to, VkAccessFlags src,
                              VkAccessFlags dst, VkPipelineStageFlags sp,
                              VkPipelineStageFlags dp) {
    VkImageMemoryBarrier2 bar = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = sp,
        .dstStageMask = dp,
        .srcAccessMask = src,
        .dstAccessMask = dst,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img,
    };
    image_range(b, &bar.subresourceRange);
    VkDependencyInfo dep = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &bar,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

void image_barrier(VkCommandBuffer cmd, acuda_buffer_t b,
                          VkImageLayout from, VkImageLayout to,
                          VkAccessFlags src, VkAccessFlags dst,
                          VkPipelineStageFlags sp, VkPipelineStageFlags dp) {
    image_barrier_img(cmd, b, b->image, from, to, src, dst, sp, dp);
}

// Per-level extent (clamped to at least one texel).
void image_level_extent(acuda_buffer_t b, uint32_t level,
                               uint32_t *w, uint32_t *h, uint32_t *d) {
    if (b->tiled1d) {
        uint32_t lw = b->logical_width ? b->logical_width : b->width;
        lw = lw >> level;
        if (lw == 0) lw = 1;
        *w = b->tile_w ? b->tile_w : lw;
        *h = (lw + *w - 1u) / *w;
        if (*h == 0) *h = 1;
        *d = 1;
        return;
    }
    *w = b->width >> level;
    if (*w == 0) *w = 1;
    *h = b->height >> level;
    if (*h == 0) *h = 1;
    *d = b->depth >> level;
    if (*d == 0) *d = 1;
}

// One-shot command buffer helper used by create/upload.
acuda_status_t image_begin(acuda_context_t ctx, VkCommandPool *pool,
                                  VkCommandBuffer *cmd) {
    VkCommandPoolCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = ctx->queue_family,
    };
    VkResult vr = vkCreateCommandPool(ctx->device, &pci, NULL, pool);
    if (vr != VK_SUCCESS) {
        return acuda_from_vk(vr);
    }
    VkCommandBufferAllocateInfo cai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = *pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    vr = vkAllocateCommandBuffers(ctx->device, &cai, cmd);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(ctx->device, *pool, NULL);
        *pool = VK_NULL_HANDLE;
        return acuda_from_vk(vr);
    }
    VkCommandBufferBeginInfo cbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vr = vkBeginCommandBuffer(*cmd, &cbi);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(ctx->device, *pool, NULL);
        *pool = VK_NULL_HANDLE;
        *cmd = VK_NULL_HANDLE;
        return acuda_from_vk(vr);
    }
    return ACUDA_SUCCESS;
}

void image_end(acuda_context_t ctx, VkCommandPool pool,
                      VkCommandBuffer cmd) {
    // Direct-queue image work must not overtake an open launch batch.
    acuda_stream_flush_all(ctx);
    vkEndCommandBuffer(cmd);
    VkCommandBufferSubmitInfo cbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = cmd,
    };
    VkSubmitInfo2 si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cbi,
    };
    // Wait on this submission only; vkQueueWaitIdle would also drain
    // unrelated work already queued on the compute queue.
    VkFenceCreateInfo fci = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(ctx->device, &fci, NULL, &fence) != VK_SUCCESS)
        fence = VK_NULL_HANDLE;
    VkResult vr = vkQueueSubmit2(ctx->queue, 1, &si, fence);
    if (vr == VK_SUCCESS && fence)
        vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    else
        vkQueueWaitIdle(ctx->queue);
    if (fence)
        vkDestroyFence(ctx->device, fence, NULL);
    vkDestroyCommandPool(ctx->device, pool, NULL);
}

// Copy one mip level (all layers or the 3D depth) from a staging buffer.
void image_copy_level(acuda_context_t ctx, VkCommandBuffer cmd,
                             acuda_buffer_t b, VkImage img, VkBuffer stage,
                             uint32_t level, uint32_t layer,
                             size_t buffer_offset) {
    uint32_t w, h, d;
    image_level_extent(b, level, &w, &h, &d);
    int depth3d = b->dim2d ? 0 : (b->depth > 1 ? 1 : 0);
    VkBufferImageCopy2 region = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        .bufferOffset = buffer_offset,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            image_is_depth(b->format) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                      : VK_IMAGE_ASPECT_COLOR_BIT,
            level,
            depth3d ? 0u : layer,
            1,
        },
        .imageExtent = { w, (b->dim1d && !b->tiled1d) ? 1u : h, depth3d ? d : 1u },
    };
    VkCopyBufferToImageInfo2 cinfo = {
        .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
        .srcBuffer = stage,
        .dstImage = img,
        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .regionCount = 1,
        .pRegions = &region,
    };
    vkCmdCopyBufferToImage2(cmd, &cinfo);
    (void)ctx;
}

// ---- multisample fill -------------------------------------------------
//
// Vulkan cannot copy into a multisampled image, so a graphics pass
// renders the single-sample source into the MS target.  The fragment
// shader samples the source at the pixel centre, which makes every
