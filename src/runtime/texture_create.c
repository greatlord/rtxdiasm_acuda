#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/memory.h"
#include "acuda/runtime/internal.h"
#include "acuda/runtime/texture_image_internal.h"

static acuda_status_t image_obj_create(acuda_context_t ctx, VkFormat fmt,
                                       VkImageType type, uint32_t w,
                                       uint32_t h, uint32_t d,
                                       uint32_t layers, uint32_t levels,
                                       VkSampleCountFlagBits samples,
                                       VkImageUsageFlags usage,
                                       VkImageCreateFlags flags,
                                       VkImage *img, VkDeviceMemory *mem,
                                       uint32_t *mem_type_bits) {
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = flags,
        .imageType = type,
        .format = fmt,
        .extent = { w, h, d },
        .mipLevels = levels,
        .arrayLayers = layers,
        .samples = samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkResult vr = vkCreateImage(ctx->device, &ici, NULL, img);
    if (vr != VK_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? image_obj_create: vkCreateImage failed vr=%d w=%u fmt=%d\n",
                    (int)vr, w, (int)fmt);
        return acuda_from_vk(vr);
    }
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(ctx->device, *img, &req);
    uint32_t mt = find_image_mem(ctx, req.memoryTypeBits);
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = mt,
    };
    vr = vkAllocateMemory(ctx->device, &mai, NULL, mem);
    if (vr != VK_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? image_obj_create: vkAllocateMemory failed vr=%d size=%zu\n",
                    (int)vr, (size_t)req.size);
        vkDestroyImage(ctx->device, *img, NULL);
        return acuda_from_vk(vr);
    }
    vr = vkBindImageMemory(ctx->device, *img, *mem, 0);
    if (vr != VK_SUCCESS) {
        vkFreeMemory(ctx->device, *mem, NULL);
        vkDestroyImage(ctx->device, *img, NULL);
        return acuda_from_vk(vr);
    }
    if (mem_type_bits)
        *mem_type_bits = mt;
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_image_create(acuda_context_t ctx, acuda_buffer_t *out,
                                  const acuda_image_desc *desc,
                                  const void *texels) {
    if (!ctx || !out || !desc || desc->width == 0 || desc->height == 0 ||
        desc->depth == 0 || desc->layers == 0 || desc->levels == 0 ||
        desc->samples == 0) {
        return ACUDA_ERROR_INVALID_VALUE;
    }
    VkFormat fmt = image_vkformat(desc->format);
    if (fmt == VK_FORMAT_UNDEFINED) {
        return ACUDA_ERROR_NOT_SUPPORTED;
    }
    int dim = desc->dim;
    if (dim == 0)
        dim = desc->depth > 1 ? 3 : 2;
    int ms = desc->samples > 1;
    if (ms) {
        // Multisampled storage is 2D/2D-array only, single-level, and
        // limited to float-sampled formats the fill shader can read.
        if (dim != 2 || desc->cube || desc->levels != 1) {
            return ACUDA_ERROR_NOT_SUPPORTED;
        }
        if (desc->format != 0 && desc->format != 1 && desc->format != 2) {
            return ACUDA_ERROR_NOT_SUPPORTED;
        }
    }
    acuda_buffer_t b = calloc(1, sizeof(*b));
    if (!b) {
        return ACUDA_ERROR_OUT_OF_MEMORY;
    }
    // Color arrays are created storage-capable so surface objects can
    // alias them; depth, cube and multisampled images stay sampled-only.
    int storable = !ms && !image_is_depth(desc->format) && !desc->cube &&
                   image_format_storage_ok(ctx, fmt);
    b->kind = 2;
    b->width = desc->width;
    b->height = desc->height;
    b->depth = desc->depth;
    b->layers = desc->layers;
    b->levels = desc->levels;
    b->samples = ms ? desc->samples : 1;
    b->format = desc->format;
    b->dim1d = dim == 1;
    b->dim2d = dim == 2;
    b->dim3d = dim == 3;
    b->cube = desc->cube ? 1 : 0;
    b->is_ms = ms;
    b->is_surface = storable;

    // 1D textures are represented as 2D tiled images so large linear
    // texrefs can exceed maxImageDimension1D (common for PhysX buffers).
    // The compiler treats geom 1d as 2D and tiles coordinates using the
    // stride in tex metadata (word 6).
    uint32_t tile_w = 0, tile_h = 1;
    VkImageType it;
    uint32_t iw, ih, id;
    uint32_t layers = dim == 3 ? 1u : desc->layers;
    if (dim == 1) {
        
        tile_w = desc->width;
        uint32_t max2 = ctx->max_image_dimension2d ? ctx->max_image_dimension2d : 16384u;
        if (tile_w == 0) tile_w = 1;
        if (tile_w > max2) tile_w = max2;
        tile_h = (desc->width + tile_w - 1u) / tile_w;
        if (tile_h == 0) tile_h = 1;
        if (tile_h > max2) {
            // Try to find a tile width that keeps height in range.
            tile_w = (desc->width + max2 - 1u) / max2;
            if (tile_w == 0) tile_w = 1;
            if (tile_w > max2) tile_w = max2;
            tile_h = (desc->width + tile_w - 1u) / tile_w;
        }
        if (tile_h > max2) {
            free(b);
            return ACUDA_ERROR_NOT_SUPPORTED;
        }
        b->tiled1d = 1;
        b->tile_w = tile_w;
        b->logical_width = desc->width;
        b->height = 1;
        b->depth = 1;
        it = VK_IMAGE_TYPE_2D;
        iw = tile_w;
        ih = tile_h;
        id = 1;
        layers = desc->layers; // 1d array stays arrayed
    } else {
        it = dim == 3 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        iw = desc->width;
        ih = dim == 1 ? 1u : desc->height;
        id = dim == 3 ? desc->depth : 1u;
    }
    VkImageCreateFlags flags =
        b->cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    if (storable && desc->format == 2)
        flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (storable)
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (ms)
        usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VkSampleCountFlagBits sc =
        ms ? (VkSampleCountFlagBits)desc->samples : VK_SAMPLE_COUNT_1_BIT;
    acuda_status_t st = image_obj_create(
        ctx, fmt, it, iw, ih, id, layers, desc->levels, sc, usage, flags,
        &b->image, &b->memory, NULL);
    if (st != ACUDA_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? acuda_image_create: image_obj_create failed st=%d w=%u fmt=%d\n",
                    (int)st, desc->width, (int)desc->format);
        free(b);
        return st;
    }
    if (ms) {
        st = image_obj_create(ctx, fmt, VK_IMAGE_TYPE_2D, desc->width,
                              desc->height, 1, layers, 1,
                              VK_SAMPLE_COUNT_1_BIT,
                              VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                              0, &b->ms_src_image, &b->ms_src_memory, NULL);
        if (st != ACUDA_SUCCESS) {
            vkFreeMemory(ctx->device, b->memory, NULL);
            vkDestroyImage(ctx->device, b->image, NULL);
            free(b);
            return st;
        }
    }

    VkImageViewType vt;
    if (b->cube) {
        if (desc->layers == 6)
            vt = VK_IMAGE_VIEW_TYPE_CUBE;
        else if (desc->layers > 6 && desc->layers % 6 == 0)
            vt = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
        else {
            { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT before_free texture_create.c:200\n"); fflush(_f); fclose(_f);} } }
            acuda_free(ctx, b);
            return ACUDA_ERROR_INVALID_VALUE;
        }
    } else if (dim == 1) {
        // Tiled 1D is stored as a 2D image (shader treats geom 1d as 2D).
        vt = desc->layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                              : VK_IMAGE_VIEW_TYPE_2D;
    } else if (dim == 3) {
        vt = VK_IMAGE_VIEW_TYPE_3D;
    } else {
        vt = desc->layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                              : VK_IMAGE_VIEW_TYPE_2D;
    }
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = b->image,
        .viewType = vt,
        .format = fmt,
    };
    image_range(b, &vci.subresourceRange);
    VkResult vr = vkCreateImageView(ctx->device, &vci, NULL, &b->view);
    if (vr != VK_SUCCESS) {
        { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT before_free texture_create.c:222\n"); fflush(_f); fclose(_f);} } }
        acuda_free(ctx, b);
        return acuda_from_vk(vr);
    }
    if (storable && desc->format == 2) {
        VkImageViewCreateInfo uci = vci;
        uci.format = VK_FORMAT_R32_UINT;
        if (vkCreateImageView(ctx->device, &uci, NULL, &b->uint_view) !=
            VK_SUCCESS)
            b->uint_view = VK_NULL_HANDLE;
    }
    if (ms) {
        VkImageViewCreateInfo svci = vci;
        svci.image = b->ms_src_image;
        vr = vkCreateImageView(ctx->device, &svci, NULL, &b->ms_src_view);
        if (vr != VK_SUCCESS) {
            { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT before_free texture_create.c:237\n"); fflush(_f); fclose(_f);} } }
            acuda_free(ctx, b);
            return acuda_from_vk(vr);
        }
    }

    VkImageLayout idle_layout = b->is_surface
                                    ? VK_IMAGE_LAYOUT_GENERAL
                                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAccessFlags idle_access =
        b->is_surface ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)
                      : VK_ACCESS_SHADER_READ_BIT;
    if (!texels) {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        st = image_begin(ctx, &pool, &cmd);
        if (st != ACUDA_SUCCESS) {
            { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT before_free texture_create.c:253\n"); fflush(_f); fclose(_f);} } }
            acuda_free(ctx, b);
            return st;
        }
        image_barrier(cmd, b, VK_IMAGE_LAYOUT_UNDEFINED, idle_layout, 0,
                      idle_access,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        if (ms)
            image_barrier_img(cmd, b, b->ms_src_image,
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0,
                              VK_ACCESS_SHADER_READ_BIT,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        image_end(ctx, pool, cmd);
        *out = b;
        return ACUDA_SUCCESS;
    }

    // Upload level 0 (every layer for array textures).
    size_t elem_bytes = image_elem_bytes(desc->format);
    size_t elems = (size_t)desc->width * desc->height *
                   (dim == 3 ? desc->depth : 1) * layers;
    size_t bytes = elems * elem_bytes;
    size_t upload_bytes = bytes;
    if (b->tiled1d) {
        // Pad to full tile rows so the 2D copy covers the image extent.
        size_t tile_elems = (size_t)b->tile_w * ((desc->width + b->tile_w - 1u) / b->tile_w);
        upload_bytes = tile_elems * elem_bytes * layers;
    }
    acuda_buffer_t stage = NULL;
    st = acuda_malloc_host(ctx, &stage, upload_bytes ? upload_bytes : 4);
    if (st != ACUDA_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? acuda_image_create: stage malloc failed st=%d bytes=%zu\n",
                    (int)st, upload_bytes);
        { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT before_free texture_create.c:288\n"); fflush(_f); fclose(_f);} } }
        acuda_free(ctx, b);
        return st;
    }
    st = acuda_memcpy_h2d(ctx, stage, texels, bytes, NULL);
    if (st != ACUDA_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? acuda_image_create: upload memcpy failed st=%d bytes=%zu\n",
                    (int)st, bytes);
        acuda_free(ctx, stage);
        { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT before_free texture_create.c:296\n"); fflush(_f); fclose(_f);} } }
        acuda_free(ctx, b);
        return st;
    }
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    st = image_begin(ctx, &pool, &cmd);
    if (st != ACUDA_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? acuda_image_create: image_begin failed st=%d\n",
                    (int)st);
        acuda_free(ctx, stage);
        { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT before_free texture_create.c:306\n"); fflush(_f); fclose(_f);} } }
        acuda_free(ctx, b);
        return st;
    }
    VkImage target = ms ? b->ms_src_image : b->image;
    image_barrier_img(cmd, b, target, VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                      VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT);
    size_t layer_bytes = b->tiled1d
        ? (size_t)b->tile_w * ((desc->width + b->tile_w - 1u) / b->tile_w) * elem_bytes
        : (size_t)desc->width * desc->height *
              (dim == 3 ? desc->depth : 1) * elem_bytes;
    for (uint32_t l = 0; l < layers; l++)
        image_copy_level(ctx, cmd, b, target, stage->buffer, 0, l,
                         (size_t)l * layer_bytes);
    image_barrier_img(cmd, b, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      idle_layout, VK_ACCESS_TRANSFER_WRITE_BIT,
                      idle_access, VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    image_end(ctx, pool, cmd);
    acuda_free(ctx, stage);
    if (ms) {
        st = ms_fill(ctx, b);
        if (st != ACUDA_SUCCESS) {
            { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT before_free texture_create.c:331\n"); fflush(_f); fclose(_f);} } }
            acuda_free(ctx, b);
            return st;
        }
    }
    *out = b;
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_surface_create(acuda_context_t ctx, acuda_buffer_t *out,
                                    const acuda_image_desc *desc,
                                    const void *texels) {
    if (!ctx || !out || !desc || desc->width == 0 || desc->height == 0 ||
        desc->depth == 0 || desc->layers == 0 || desc->levels == 0)
        return ACUDA_ERROR_INVALID_VALUE;
    if (desc->cube || desc->samples > 1)
        return ACUDA_ERROR_NOT_SUPPORTED;
    if (desc->format > 5 || desc->format == 4)
        return ACUDA_ERROR_NOT_SUPPORTED;
    VkFormat fmt = image_vkformat(desc->format);
    if (fmt == VK_FORMAT_UNDEFINED)
        return ACUDA_ERROR_NOT_SUPPORTED;
    int dim = desc->dim;
    if (dim == 0)
        dim = desc->depth > 1 ? 3 : 2;
    acuda_buffer_t b = calloc(1, sizeof(*b));
    if (!b)
        return ACUDA_ERROR_OUT_OF_MEMORY;
    b->kind = 2;
    b->width = desc->width;
    b->height = desc->height;
    b->depth = desc->depth;
    b->layers = desc->layers;
    b->levels = desc->levels;
    b->samples = 1;
    b->format = desc->format;
    b->dim1d = dim == 1;
    b->dim2d = dim == 2;
    b->dim3d = dim == 3;
    b->is_surface = 1;
    VkImageType it = dim == 1 ? VK_IMAGE_TYPE_1D
                              : (dim == 3 ? VK_IMAGE_TYPE_3D
                                          : VK_IMAGE_TYPE_2D);
    uint32_t layers = dim == 3 ? 1u : desc->layers;
    VkImageCreateFlags icflags =
        desc->format == 2 ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
    acuda_status_t st = image_obj_create(
        ctx, fmt, it, desc->width, dim == 1 ? 1u : desc->height,
        dim == 3 ? desc->depth : 1u, layers, desc->levels,
        VK_SAMPLE_COUNT_1_BIT,
        ((desc->format == 5 ? 0u : VK_IMAGE_USAGE_SAMPLED_BIT) |
         VK_IMAGE_USAGE_STORAGE_BIT |
         VK_IMAGE_USAGE_TRANSFER_DST_BIT |
         VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
        icflags, &b->image, &b->memory, NULL);
    if (st != ACUDA_SUCCESS) {
        free(b);
        return st;
    }
    VkImageViewType vt;
    if (dim == 1)
        vt = desc->layers > 1 ? VK_IMAGE_VIEW_TYPE_1D_ARRAY
                              : VK_IMAGE_VIEW_TYPE_1D;
    else if (dim == 3)
        vt = VK_IMAGE_VIEW_TYPE_3D;
    else
        vt = desc->layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                              : VK_IMAGE_VIEW_TYPE_2D;
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = b->image,
        .viewType = vt,
        .format = fmt,
    };
    image_range(b, &vci.subresourceRange);
    VkResult vr = vkCreateImageView(ctx->device, &vci, NULL, &b->view);
    if (vr != VK_SUCCESS) {
        { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT before_free texture_create.c:407\n"); fflush(_f); fclose(_f);} } }
        acuda_free(ctx, b);
        return acuda_from_vk(vr);
    }
    if (desc->format == 2) {
        VkImageViewCreateInfo uci = vci;
        uci.format = VK_FORMAT_R32_UINT;
        if (vkCreateImageView(ctx->device, &uci, NULL, &b->uint_view) !=
            VK_SUCCESS)
            b->uint_view = VK_NULL_HANDLE;
    }

    acuda_buffer_t stage = NULL;
    size_t bytes = 0;
    if (texels) {
        size_t elems = (size_t)desc->width * desc->height *
                       (dim == 3 ? desc->depth : 1) * layers;
        bytes = elems * image_elem_bytes(desc->format);
        st = acuda_malloc(ctx, &stage, bytes ? bytes : 4);
        if (st != ACUDA_SUCCESS) {
            { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT before_free texture_create.c:426\n"); fflush(_f); fclose(_f);} } }
            acuda_free(ctx, b);
            return st;
        }
        st = acuda_memcpy_h2d(ctx, stage, texels, bytes, NULL);
        if (st != ACUDA_SUCCESS) {
            acuda_free(ctx, stage);
            { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT before_free texture_create.c:432\n"); fflush(_f); fclose(_f);} } }
            acuda_free(ctx, b);
            return st;
        }
    }
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    st = image_begin(ctx, &pool, &cmd);
    if (st != ACUDA_SUCCESS) {
        if (stage)
            acuda_free(ctx, stage);
        { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT before_free texture_create.c:442\n"); fflush(_f); fclose(_f);} } }
        acuda_free(ctx, b);
        return st;
    }
    if (texels) {
        image_barrier_img(cmd, b, b->image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT);
        size_t layer_bytes = (size_t)desc->width * desc->height *
                             (dim == 3 ? desc->depth : 1) *
                             image_elem_bytes(desc->format);
        for (uint32_t l = 0; l < layers; l++)
            image_copy_level(ctx, cmd, b, b->image, stage->buffer, 0, l,
                             (size_t)l * layer_bytes);
        image_barrier_img(cmd, b, b->image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    } else {
        image_barrier_img(cmd, b, b->image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_GENERAL, 0,
                          VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    image_end(ctx, pool, cmd);
    if (stage)
        acuda_free(ctx, stage);
    *out = b;
    return ACUDA_SUCCESS;
}
