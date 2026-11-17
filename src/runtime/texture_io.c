#include <stdlib.h>
#include <string.h>
#include "acuda/memory.h"
#include "acuda/runtime/internal.h"
#include "acuda/runtime/texture_image_internal.h"


int acuda_surface_alias_supported(acuda_context_t ctx) {
    if (!ctx)
        return 0;
    acuda_image_desc d = { 1, 1, 1, 1, 1, 1, 2, 2, 0 };
    acuda_buffer_t b = NULL;
    if (acuda_surface_create(ctx, &b, &d, NULL) != ACUDA_SUCCESS)
        return 0;
    int ok = b && b->uint_view != VK_NULL_HANDLE;
    if (b)
        acuda_free(ctx, b);
    return ok;
}

acuda_status_t acuda_surface_upload(acuda_context_t ctx, acuda_buffer_t surf,
                                    const void *texels) {
    if (!ctx || !surf || surf->kind != 2 || !texels || !surf->is_surface)
        return ACUDA_ERROR_INVALID_VALUE;
    uint32_t layers = surf->dim3d ? 1u : surf->layers;
    size_t layer_bytes = (size_t)surf->width * surf->height *
                         (surf->dim3d ? surf->depth : 1) *
                         image_elem_bytes(surf->format);
    size_t bytes = layer_bytes * layers;
    acuda_buffer_t stage = NULL;
    acuda_status_t st = acuda_malloc(ctx, &stage, bytes ? bytes : 4);
    if (st != ACUDA_SUCCESS)
        return st;
    st = acuda_memcpy_h2d(ctx, stage, texels, bytes, NULL);
    if (st != ACUDA_SUCCESS) {
        acuda_free(ctx, stage);
        return st;
    }
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    st = image_begin(ctx, &pool, &cmd);
    if (st != ACUDA_SUCCESS) {
        acuda_free(ctx, stage);
        return st;
    }
    image_barrier_img(cmd, surf, surf->image, VK_IMAGE_LAYOUT_GENERAL,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_SHADER_WRITE_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT);
    for (uint32_t l = 0; l < layers; l++)
        image_copy_level(ctx, cmd, surf, surf->image, stage->buffer, 0, l,
                         (size_t)l * layer_bytes);
    image_barrier_img(cmd, surf, surf->image,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_GENERAL,
                      VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    image_end(ctx, pool, cmd);
    acuda_free(ctx, stage);
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_surface_download(acuda_context_t ctx,
                                      acuda_buffer_t surf, void *texels) {
    if (!ctx || !surf || surf->kind != 2 || !texels || !surf->is_surface)
        return ACUDA_ERROR_INVALID_VALUE;
    uint32_t layers = surf->dim3d ? 1u : surf->layers;
    size_t layer_bytes = (size_t)surf->width * surf->height *
                         (surf->dim3d ? surf->depth : 1) *
                         image_elem_bytes(surf->format);
    size_t bytes = layer_bytes * layers;
    acuda_buffer_t stage = NULL;
    acuda_status_t st = acuda_malloc(ctx, &stage, bytes ? bytes : 4);
    if (st != ACUDA_SUCCESS)
        return st;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    st = image_begin(ctx, &pool, &cmd);
    if (st != ACUDA_SUCCESS) {
        acuda_free(ctx, stage);
        return st;
    }
    image_barrier_img(cmd, surf, surf->image, VK_IMAGE_LAYOUT_GENERAL,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_ACCESS_SHADER_WRITE_BIT,
                      VK_ACCESS_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy2 region = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        .imageSubresource = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers,
        },
        .imageExtent = { surf->width, surf->dim1d ? 1u : surf->height,
                         surf->dim3d ? surf->depth : 1u },
    };
    VkCopyImageToBufferInfo2 cinfo = {
        .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
        .srcImage = surf->image,
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstBuffer = stage->buffer,
        .regionCount = 1,
        .pRegions = &region,
    };
    vkCmdCopyImageToBuffer2(cmd, &cinfo);
    image_barrier_img(cmd, surf, surf->image,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_IMAGE_LAYOUT_GENERAL,
                      VK_ACCESS_TRANSFER_READ_BIT,
                      VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    image_end(ctx, pool, cmd);
    st = acuda_memcpy_d2h(ctx, texels, stage, bytes, NULL);
    acuda_free(ctx, stage);
    return st;
}

acuda_status_t acuda_image_upload(acuda_context_t ctx, acuda_buffer_t image,
                                  uint32_t level, uint32_t layer,
                                  const void *texels) {
    if (!ctx || !image || image->kind != 2 || !texels)
        return ACUDA_ERROR_INVALID_VALUE;
    if (level >= image->levels)
        return ACUDA_ERROR_INVALID_VALUE;
    uint32_t layers = image->dim3d ? 1u : image->layers;
    if (layer >= layers)
        return ACUDA_ERROR_INVALID_VALUE;
    uint32_t w, h, d;
    image_level_extent(image, level, &w, &h, &d);
    size_t bytes = (size_t)w * h * d * image_elem_bytes(image->format);
    acuda_buffer_t stage = NULL;
    acuda_status_t st = acuda_malloc(ctx, &stage, bytes ? bytes : 4);
    if (st != ACUDA_SUCCESS)
        return st;
    st = acuda_memcpy_h2d(ctx, stage, texels, bytes, NULL);
    if (st != ACUDA_SUCCESS) {
        acuda_free(ctx, stage);
        return st;
    }
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    st = image_begin(ctx, &pool, &cmd);
    if (st != ACUDA_SUCCESS) {
        acuda_free(ctx, stage);
        return st;
    }
    VkImage target = image->is_ms ? image->ms_src_image : image->image;
    VkImageLayout idle_layout = image->is_surface
                                    ? VK_IMAGE_LAYOUT_GENERAL
                                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAccessFlags idle_access =
        image->is_surface ? (VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT)
                          : VK_ACCESS_SHADER_READ_BIT;
    image_barrier_img(cmd, image, target, idle_layout,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, idle_access,
                      VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT);
    image_copy_level(ctx, cmd, image, target, stage->buffer, level, layer,
                     0);
    image_barrier_img(cmd, image, target,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, idle_layout,
                      VK_ACCESS_TRANSFER_WRITE_BIT, idle_access,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    image_end(ctx, pool, cmd);
    acuda_free(ctx, stage);
    if (image->is_ms)
        return ms_fill(ctx, image);
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_image_download(acuda_context_t ctx, acuda_buffer_t image,
                                    uint32_t level, uint32_t layer,
                                    void *texels) {
    if (!ctx || !image || image->kind != 2 || !texels)
        return ACUDA_ERROR_INVALID_VALUE;
    if (level >= image->levels)
        return ACUDA_ERROR_INVALID_VALUE;
    uint32_t layers = image->dim3d ? 1u : image->layers;
    if (layer >= layers)
        return ACUDA_ERROR_INVALID_VALUE;
    uint32_t w, h, d;
    image_level_extent(image, level, &w, &h, &d);
    size_t bytes = (size_t)w * h * d * image_elem_bytes(image->format);
    acuda_buffer_t stage = NULL;
    acuda_status_t st = acuda_malloc(ctx, &stage, bytes ? bytes : 4);
    if (st != ACUDA_SUCCESS)
        return st;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    st = image_begin(ctx, &pool, &cmd);
    if (st != ACUDA_SUCCESS) {
        acuda_free(ctx, stage);
        return st;
    }
    VkImage src = image->is_ms ? image->ms_src_image : image->image;
    VkImageLayout idle_layout = image->is_surface
                                    ? VK_IMAGE_LAYOUT_GENERAL
                                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAccessFlags idle_access =
        image->is_surface ? (VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT)
                          : VK_ACCESS_SHADER_READ_BIT;
    image_barrier_img(cmd, image, src, idle_layout,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, idle_access,
                      VK_ACCESS_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT);
    uint32_t w2, h2, d2;
    image_level_extent(image, level, &w2, &h2, &d2);
    VkBufferImageCopy2 region = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        .imageSubresource = {
            image_is_depth(image->format) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                          : VK_IMAGE_ASPECT_COLOR_BIT,
            level,
            image->dim3d ? 0u : layer,
            1,
        },
        .imageExtent = { w2, image->dim1d ? 1u : h2,
                         image->dim3d ? d2 : 1u },
    };
    VkCopyImageToBufferInfo2 cinfo = {
        .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
        .srcImage = src,
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstBuffer = stage->buffer,
        .regionCount = 1,
        .pRegions = &region,
    };
    vkCmdCopyImageToBuffer2(cmd, &cinfo);
    image_barrier_img(cmd, image, src,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, idle_layout,
                      VK_ACCESS_TRANSFER_READ_BIT, idle_access,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    image_end(ctx, pool, cmd);
    st = acuda_memcpy_d2h(ctx, texels, stage, bytes, NULL);
    acuda_free(ctx, stage);
    return st;
}
