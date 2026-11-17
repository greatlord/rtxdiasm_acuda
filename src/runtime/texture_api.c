#include <stdlib.h>
#include <string.h>
#include "acuda/memory.h"
#include "acuda/runtime/internal.h"

acuda_status_t acuda_sampler_create(acuda_context_t ctx, acuda_buffer_t *out,
                                    const acuda_sampler_desc *desc) {
    if (!ctx || !out || !desc)
        return ACUDA_ERROR_INVALID_VALUE;
    acuda_buffer_t b = calloc(1, sizeof(*b));
    if (!b)
        return ACUDA_ERROR_OUT_OF_MEMORY;
    b->kind = 4;
    VkFilter f = desc->filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    VkSamplerAddressMode am[3];
    for (int i = 0; i < 3; i++) {
        switch (desc->addr[i]) {
        case 1: am[i] = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; break;
        case 2: am[i] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; break;
        case 3: am[i] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER; break;
        default: am[i] = VK_SAMPLER_ADDRESS_MODE_REPEAT; break;
        }
    }
    VkBorderColor bc = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    int unnorm = !desc->normalized;
    if (unnorm) {
        // Unnormalized samplers must clamp and cannot mip/anisotropy.
        am[0] = am[1] = am[2] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
    VkSamplerCreateInfo sci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = f,
        .minFilter = f,
        .mipmapMode = (desc->mip_filter && !unnorm)
                          ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                          : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = am[0],
        .addressModeV = am[1],
        .addressModeW = am[2],
        .borderColor = bc,
        .unnormalizedCoordinates = unnorm ? VK_TRUE : VK_FALSE,
        .compareEnable = desc->compare ? VK_TRUE : VK_FALSE,
        .compareOp = (VkCompareOp)desc->compare_op,
        .maxLod = unnorm ? 0.0f : VK_LOD_CLAMP_NONE,
        .maxAnisotropy = 1.0f,
    };
    VkResult vr = vkCreateSampler(ctx->device, &sci, NULL, &b->sampler);
    if (vr != VK_SUCCESS) {
        free(b);
        return acuda_from_vk(vr);
    }
    b->normalized = 1;
    b->sfilter = desc->filter ? 1u : 0u;
    b->smip = desc->mip_filter ? 1u : 0u;
    for (int i = 0; i < 3; i++)
        b->saddr[i] = desc->addr[i];
    b->snorm = desc->normalized ? 1u : 0u;
    b->scompare = desc->compare ? 1u : 0u;
    *out = b;
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_texture_create_image(acuda_context_t ctx,
                                          acuda_buffer_t *out,
                                          acuda_buffer_t image,
                                          acuda_buffer_t sampler) {
    if (!ctx || !out || !image || !sampler ||
        image->kind != 2 || sampler->kind != 4)
        return ACUDA_ERROR_INVALID_VALUE;
    acuda_buffer_t b = calloc(1, sizeof(*b));
    if (!b)
        return ACUDA_ERROR_OUT_OF_MEMORY;
    b->kind = 3;
    b->view = image->view;
    b->sampler = sampler->sampler;
    b->width = image->width;
    b->height = image->height;
    b->depth = image->depth;
    b->layers = image->layers;
    b->levels = image->levels;
    b->samples = image->samples;
    b->format = image->format;
    b->dim1d = image->dim1d;
    b->dim2d = image->dim2d;
    b->dim3d = image->dim3d;
    b->cube = image->cube;
    b->tiled1d = image->tiled1d;
    b->tile_w = image->tile_w;
    b->logical_width = image->logical_width;
    b->is_surface = image->is_surface;
    b->normalized = 1;
    b->sfilter = sampler->sfilter;
    b->smip = sampler->smip;
    for (int i = 0; i < 3; i++)
        b->saddr[i] = sampler->saddr[i];
    b->snorm = sampler->snorm;
    b->scompare = sampler->scompare;
    *out = b;
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_texture_create(acuda_context_t ctx,
                                    acuda_buffer_t *out_buf,
                                    const acuda_texture_desc *desc,
                                    const void *texels) {
    if (!ctx || !out_buf || !desc)
        return ACUDA_ERROR_INVALID_VALUE;
    if (desc->width == 0 || desc->height == 0 || desc->depth == 0 ||
        desc->format > 2)
        return ACUDA_ERROR_INVALID_VALUE;
    uint32_t fmt;
    if (desc->format == 0 && desc->channels == 4)
        fmt = 0; // RGBA32F
    else if (desc->channels == 1)
        fmt = desc->format == 0 ? 2u : 3u; // R32F / R32_UINT
    else
        return ACUDA_ERROR_NOT_SUPPORTED;
    acuda_image_desc id = {
        .width = desc->width,
        .height = desc->height,
        .depth = desc->depth,
        .layers = 1,
        .levels = 1,
        .samples = 1,
        .format = fmt,
        .dim = desc->depth > 1 ? 3u : 2u,
        .cube = 0,
    };
    acuda_buffer_t img = NULL, smp = NULL, tex = NULL;
    acuda_status_t st = acuda_image_create(ctx, &img, &id, texels);
    if (st != ACUDA_SUCCESS)
        return st;
    acuda_sampler_desc sd;
    memset(&sd, 0, sizeof(sd));
    sd.filter = desc->filter ? 1u : 0u;
    sd.mip_filter = 0;
    for (int i = 0; i < 3; i++)
        sd.addr[i] = desc->addr[i];
    sd.normalized = desc->normalized ? 1u : 0u;
    st = acuda_sampler_create(ctx, &smp, &sd);
    if (st != ACUDA_SUCCESS) {
        acuda_free(ctx, img);
        return st;
    }
    st = acuda_texture_create_image(ctx, &tex, img, smp);
    if (st != ACUDA_SUCCESS) {
        acuda_free(ctx, img);
        acuda_free(ctx, smp);
        return st;
    }
    // Move ownership of the Vulkan objects into the combined handle.
    acuda_buffer_unregister(ctx, img);
    acuda_buffer_unregister(ctx, smp);
    tex->image = img->image;
    tex->memory = img->memory;
    tex->sampler = smp->sampler;
    tex->owns_storage = 1;
    free(img);
    free(smp);
    *out_buf = tex;
    return ACUDA_SUCCESS;
}
