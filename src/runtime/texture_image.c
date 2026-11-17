// Vulkan-image-backed texture objects: images, samplers and combined
// image-sampler handles.  The runtime's ordinary allocations remain
// VkBuffers; these handles extend the same opaque acuda_buffer_t type.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/memory.h"
#include "acuda/runtime/internal.h"

static VkFormat image_vkformat(uint32_t f) {
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

static uint32_t image_elem_bytes(uint32_t f) {
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

static int image_is_depth(uint32_t f) {
    return f == 4;
}

static int image_format_storage_ok(acuda_context_t ctx, VkFormat fmt) {
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

static uint32_t find_image_mem(acuda_context_t ctx, uint32_t bits) {
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

static void image_range(acuda_buffer_t b, VkImageSubresourceRange *r) {
    r->aspectMask = image_is_depth(b->format) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                              : VK_IMAGE_ASPECT_COLOR_BIT;
    r->baseMipLevel = 0;
    r->levelCount = b->levels ? b->levels : 1;
    r->baseArrayLayer = 0;
    r->layerCount = b->layers ? b->layers : 1;
}

static void image_barrier_img(VkCommandBuffer cmd, acuda_buffer_t b,
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

static void image_barrier(VkCommandBuffer cmd, acuda_buffer_t b,
                          VkImageLayout from, VkImageLayout to,
                          VkAccessFlags src, VkAccessFlags dst,
                          VkPipelineStageFlags sp, VkPipelineStageFlags dp) {
    image_barrier_img(cmd, b, b->image, from, to, src, dst, sp, dp);
}

// Per-level extent (clamped to at least one texel).
static void image_level_extent(acuda_buffer_t b, uint32_t level,
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
static acuda_status_t image_begin(acuda_context_t ctx, VkCommandPool *pool,
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

static void image_end(acuda_context_t ctx, VkCommandPool pool,
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
static void image_copy_level(acuda_context_t ctx, VkCommandBuffer cmd,
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
// sample of each pixel hold the source texel.

#define OP_Capability 17
#define OP_MemoryModel 14
#define OP_EntryPoint 15
#define OP_ExecutionMode 16
#define OP_Decorate 71
#define OP_TypeVoid 19
#define OP_TypeInt 21
#define OP_TypeFloat 22
#define OP_TypeVector 23
#define OP_TypeImage 25
#define OP_TypeSampledImage 27
#define OP_TypePointer 32
#define OP_TypeFunction 33
#define OP_Constant 43
#define OP_Variable 59
#define OP_Function 54
#define OP_Label 248
#define OP_Return 253
#define OP_FunctionEnd 56
#define OP_Load 61
#define OP_Store 62
#define OP_CompositeConstruct 80
#define OP_ImageSampleImplicitLod 87
#define OP_ConvertUToF 112
#define OP_FMul 133
#define OP_FSub 131
#define OP_ShiftLeftLogical 196
#define OP_BitwiseAnd 199

typedef struct {
    uint32_t w[256];
    int n;
} sb_t;

static void sb_emit(sb_t *s, uint32_t op, int nops, ...) {
    s->w[s->n++] = ((uint32_t)(nops + 1) << 16) | op;
    va_list ap;
    va_start(ap, nops);
    for (int i = 0; i < nops; i++)
        s->w[s->n++] = va_arg(ap, uint32_t);
    va_end(ap);
}

static void sb_header(sb_t *s) {
    s->n = 0;
    s->w[s->n++] = 0x07230203;
    s->w[s->n++] = 0x00010000;
    s->w[s->n++] = 0;
    s->w[s->n++] = 0; // bound patched
    s->w[s->n++] = 0;
}

static void build_fill_vertex(sb_t *s) {
    sb_header(s);
    uint32_t nid = 1;
    uint32_t void_t = nid++, f32 = nid++, u32 = nid++, v2 = nid++, v4 = nid++;
    uint32_t poutv4 = nid++, poutv2 = nid++, pin_u32 = nid++, fn = nid++;
    uint32_t glvi = nid++, glpos = nid++, uv = nid++, mainfn = nid++;
    uint32_t label = nid++, f0 = nid++, f1 = nid++, f2 = nid++;
    uint32_t u1 = nid++, u2 = nid++;
    sb_emit(s, OP_Capability, 1, 1);
    sb_emit(s, OP_MemoryModel, 2, 0, 1);
    sb_emit(s, OP_EntryPoint, 7, 0, mainfn, 0x6e69616d, 0, glvi, glpos,
            uv);
    sb_emit(s, OP_Decorate, 3, glvi, 11, 42);  // BuiltIn VertexIndex
    sb_emit(s, OP_Decorate, 3, glpos, 11, 0);  // BuiltIn Position
    sb_emit(s, OP_Decorate, 3, uv, 30, 0);     // Location 0
    sb_emit(s, OP_TypeVoid, 1, void_t);
    sb_emit(s, OP_TypeFloat, 2, f32, 32);
    sb_emit(s, OP_TypeInt, 3, u32, 32, 0);
    sb_emit(s, OP_TypeVector, 3, v2, f32, 2);
    sb_emit(s, OP_TypeVector, 3, v4, f32, 4);
    sb_emit(s, OP_TypePointer, 3, poutv4, 3, v4);
    sb_emit(s, OP_TypePointer, 3, poutv2, 3, v2);
    sb_emit(s, OP_TypePointer, 3, pin_u32, 1, u32);
    sb_emit(s, OP_TypeFunction, 2, fn, void_t);
    sb_emit(s, OP_Variable, 3, pin_u32, glvi, 1);
    sb_emit(s, OP_Variable, 3, poutv4, glpos, 3);
    sb_emit(s, OP_Variable, 3, poutv2, uv, 3);
    sb_emit(s, OP_Constant, 3, f32, f0, 0);
    sb_emit(s, OP_Constant, 3, f32, f1, 0x3f800000u);
    sb_emit(s, OP_Constant, 3, f32, f2, 0x40000000u);
    sb_emit(s, OP_Constant, 3, u32, u1, 1);
    sb_emit(s, OP_Constant, 3, u32, u2, 2);
    sb_emit(s, OP_Function, 4, void_t, mainfn, 0, fn);
    sb_emit(s, OP_Label, 1, label);
    uint32_t idx = nid++, sh = nid++, xu = nid++, x = nid++, yu = nid++;
    uint32_t y = nid++, pv = nid++, x2 = nid++, x1 = nid++, y2 = nid++;
    uint32_t y1 = nid++, pos = nid++;
    sb_emit(s, OP_Load, 3, u32, idx, glvi);
    sb_emit(s, OP_ShiftLeftLogical, 4, u32, sh, idx, u1);
    sb_emit(s, OP_BitwiseAnd, 4, u32, xu, sh, u2);
    sb_emit(s, OP_ConvertUToF, 3, f32, x, xu);
    sb_emit(s, OP_BitwiseAnd, 4, u32, yu, idx, u2);
    sb_emit(s, OP_ConvertUToF, 3, f32, y, yu);
    sb_emit(s, OP_CompositeConstruct, 4, v2, pv, x, y);
    sb_emit(s, OP_FMul, 4, f32, x2, x, f2);
    sb_emit(s, OP_FSub, 4, f32, x1, x2, f1);
    sb_emit(s, OP_FMul, 4, f32, y2, y, f2);
    sb_emit(s, OP_FSub, 4, f32, y1, y2, f1);
    sb_emit(s, OP_CompositeConstruct, 6, v4, pos, x1, y1, f0, f1);
    sb_emit(s, OP_Store, 2, glpos, pos);
    sb_emit(s, OP_Store, 2, uv, pv);
    sb_emit(s, OP_Return, 0);
    sb_emit(s, OP_FunctionEnd, 0);
    s->w[3] = nid;
}

static void build_fill_fragment(sb_t *s) {
    sb_header(s);
    uint32_t nid = 1;
    uint32_t void_t = nid++, f32 = nid++, v2 = nid++, v4 = nid++;
    uint32_t poutv4 = nid++, pinv2 = nid++, img = nid++, simg = nid++;
    uint32_t psimg = nid++, tex = nid++, uv = nid++, outc = nid++;
    uint32_t mainfn = nid++, label = nid++, fn = nid++;
    uint32_t si = nid++, c = nid++, col = nid++;
    sb_emit(s, OP_Capability, 1, 1);
    sb_emit(s, OP_MemoryModel, 2, 0, 1);
    sb_emit(s, OP_EntryPoint, 6, 4, mainfn, 0x6e69616d, 0, uv, outc);
    sb_emit(s, OP_ExecutionMode, 2, mainfn, 7); // OriginUpperLeft
    sb_emit(s, OP_Decorate, 3, uv, 30, 0);
    sb_emit(s, OP_Decorate, 3, outc, 30, 0);
    sb_emit(s, OP_Decorate, 3, tex, 34, 0); // DescriptorSet 0
    sb_emit(s, OP_Decorate, 3, tex, 33, 0); // Binding 0
    sb_emit(s, OP_TypeVoid, 1, void_t);
    sb_emit(s, OP_TypeFloat, 2, f32, 32);
    sb_emit(s, OP_TypeVector, 3, v2, f32, 2);
    sb_emit(s, OP_TypeVector, 3, v4, f32, 4);
    sb_emit(s, OP_TypePointer, 3, poutv4, 3, v4);
    sb_emit(s, OP_TypePointer, 3, pinv2, 1, v2);
    sb_emit(s, OP_TypeImage, 8, img, f32, 1, 0, 0, 0, 1, 0);
    sb_emit(s, OP_TypeSampledImage, 2, simg, img);
    sb_emit(s, OP_TypePointer, 3, psimg, 0, simg);
    sb_emit(s, OP_TypeFunction, 2, fn, void_t);
    sb_emit(s, OP_Variable, 3, psimg, tex, 0);
    sb_emit(s, OP_Variable, 3, pinv2, uv, 1);
    sb_emit(s, OP_Variable, 3, poutv4, outc, 3);
    sb_emit(s, OP_Function, 4, void_t, mainfn, 0, fn);
    sb_emit(s, OP_Label, 1, label);
    sb_emit(s, OP_Load, 3, simg, si, tex);
    sb_emit(s, OP_Load, 3, v2, c, uv);
    sb_emit(s, OP_ImageSampleImplicitLod, 4, v4, col, si, c);
    sb_emit(s, OP_Store, 2, outc, col);
    sb_emit(s, OP_Return, 0);
    sb_emit(s, OP_FunctionEnd, 0);
    s->w[3] = nid;
}

// Transition a single array layer. The MS fill renders and clears one
// layer at a time, so an UNDEFINED transition must not discard the
// layers already filled.
static void image_barrier_layer(VkCommandBuffer cmd, acuda_buffer_t b,
                                VkImage img, uint32_t layer,
                                VkImageLayout from, VkImageLayout to,
                                VkAccessFlags src, VkAccessFlags dst,
                                VkPipelineStageFlags sp,
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
        .subresourceRange = {
            .aspectMask = image_is_depth(b->format)
                              ? VK_IMAGE_ASPECT_DEPTH_BIT
                              : VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = layer,
            .layerCount = 1,
        },
    };
    VkDependencyInfo dep = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &bar,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

static acuda_status_t ms_fill(acuda_context_t ctx, acuda_buffer_t b) {
    if (!b->is_ms)
        return ACUDA_SUCCESS;
    VkFormat fmt = image_vkformat(b->format);
    uint32_t layers = b->layers ? b->layers : 1;
    VkSampleCountFlagBits sc = (VkSampleCountFlagBits)b->samples;

    sb_t vs, fs;
    build_fill_vertex(&vs);
    build_fill_fragment(&fs);
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = (size_t)vs.n * 4,
        .pCode = vs.w,
    };
    VkShaderModule vmod = VK_NULL_HANDLE, fmod = VK_NULL_HANDLE;
    VkResult vr = vkCreateShaderModule(ctx->device, &smci, NULL, &vmod);
    if (vr != VK_SUCCESS)
        return acuda_from_vk(vr);
    smci.codeSize = (size_t)fs.n * 4;
    smci.pCode = fs.w;
    vr = vkCreateShaderModule(ctx->device, &smci, NULL, &fmod);
    if (vr != VK_SUCCESS) {
        vkDestroyShaderModule(ctx->device, vmod, NULL);
        return acuda_from_vk(vr);
    }

    VkAttachmentDescription att = {
        .format = fmt,
        .samples = sc,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkAttachmentReference ref = {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    VkSubpassDescription sub = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &ref,
    };
    VkSubpassDependency dep = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };
    VkRenderPassCreateInfo rpci = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &att,
        .subpassCount = 1,
        .pSubpasses = &sub,
        .dependencyCount = 1,
        .pDependencies = &dep,
    };
    // Prefer Vulkan 1.4 dynamic rendering; the render pass remains as a
    // fallback for devices that do not expose the feature.
    int dyn = ctx->dynamic_rendering_ok;
    VkRenderPass rp = VK_NULL_HANDLE;
    if (!dyn) {
        vr = vkCreateRenderPass(ctx->device, &rpci, NULL, &rp);
        if (vr != VK_SUCCESS) {
            vkDestroyShaderModule(ctx->device, vmod, NULL);
            vkDestroyShaderModule(ctx->device, fmod, NULL);
            return acuda_from_vk(vr);
        }
    }

    VkDescriptorSetLayoutBinding bnd = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &bnd,
    };
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    vr = vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL, &dsl);
    if (vr != VK_SUCCESS) {
        if (rp) vkDestroyRenderPass(ctx->device, rp, NULL);
        vkDestroyShaderModule(ctx->device, vmod, NULL);
        vkDestroyShaderModule(ctx->device, fmod, NULL);
        return acuda_from_vk(vr);
    }
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &dsl,
    };
    VkPipelineLayout pl = VK_NULL_HANDLE;
    vr = vkCreatePipelineLayout(ctx->device, &plci, NULL, &pl);
    if (vr != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(ctx->device, dsl, NULL);
        if (rp) vkDestroyRenderPass(ctx->device, rp, NULL);
        vkDestroyShaderModule(ctx->device, vmod, NULL);
        vkDestroyShaderModule(ctx->device, fmod, NULL);
        return acuda_from_vk(vr);
    }

    VkPipelineShaderStageCreateInfo stages[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vmod,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fmod,
            .pName = "main",
        },
    };
    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    VkViewport vp = { 0.0f, 0.0f, (float)b->width, (float)b->height,
                      0.0f, 1.0f };
    VkRect2D scissor = { { 0, 0 }, { b->width, b->height } };
    VkPipelineViewportStateCreateInfo vps = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &vp,
        .scissorCount = 1,
        .pScissors = &scissor,
    };
    VkPipelineRasterizationStateCreateInfo rs = {
        .sType =
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    VkPipelineMultisampleStateCreateInfo mss = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = sc,
    };
    VkPipelineColorBlendAttachmentState cba = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                          VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT |
                          VK_COLOR_COMPONENT_A_BIT,
    };
    VkPipelineColorBlendStateCreateInfo cbs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &cba,
    };
    VkPipelineRenderingCreateInfo prci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &fmt,
    };
    VkGraphicsPipelineCreateInfo gpci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = dyn ? &prci : NULL,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vi,
        .pInputAssemblyState = &ia,
        .pViewportState = &vps,
        .pRasterizationState = &rs,
        .pMultisampleState = &mss,
        .pColorBlendState = &cbs,
        .layout = pl,
        .renderPass = rp,
    };
    VkPipeline pipe = VK_NULL_HANDLE;
    vr = vkCreateGraphicsPipelines(ctx->device, VK_NULL_HANDLE, 1, &gpci,
                                   NULL, &pipe);
    if (vr != VK_SUCCESS) {
        vkDestroyPipelineLayout(ctx->device, pl, NULL);
        vkDestroyDescriptorSetLayout(ctx->device, dsl, NULL);
        if (rp) vkDestroyRenderPass(ctx->device, rp, NULL);
        vkDestroyShaderModule(ctx->device, vmod, NULL);
        vkDestroyShaderModule(ctx->device, fmod, NULL);
        return acuda_from_vk(vr);
    }

    VkSamplerCreateInfo sci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxLod = 0.0f,
    };
    VkSampler smp = VK_NULL_HANDLE;
    vr = vkCreateSampler(ctx->device, &sci, NULL, &smp);
    if (vr != VK_SUCCESS) {
        vkDestroyPipeline(ctx->device, pipe, NULL);
        vkDestroyPipelineLayout(ctx->device, pl, NULL);
        vkDestroyDescriptorSetLayout(ctx->device, dsl, NULL);
        if (rp) vkDestroyRenderPass(ctx->device, rp, NULL);
        vkDestroyShaderModule(ctx->device, vmod, NULL);
        vkDestroyShaderModule(ctx->device, fmod, NULL);
        return acuda_from_vk(vr);
    }
    VkDescriptorPoolSize psz = {
        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &psz,
    };
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    vr = vkCreateDescriptorPool(ctx->device, &dpci, NULL, &dpool);
    if (vr != VK_SUCCESS) {
        vkDestroySampler(ctx->device, smp, NULL);
        vkDestroyPipeline(ctx->device, pipe, NULL);
        vkDestroyPipelineLayout(ctx->device, pl, NULL);
        vkDestroyDescriptorSetLayout(ctx->device, dsl, NULL);
        if (rp) vkDestroyRenderPass(ctx->device, rp, NULL);
        vkDestroyShaderModule(ctx->device, vmod, NULL);
        vkDestroyShaderModule(ctx->device, fmod, NULL);
        return acuda_from_vk(vr);
    }
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = dpool,
        .descriptorSetCount = 1,
        .pSetLayouts = &dsl,
    };
    VkDescriptorSet dset = VK_NULL_HANDLE;
    vr = vkAllocateDescriptorSets(ctx->device, &dsai, &dset);

    acuda_status_t st = ACUDA_SUCCESS;
    if (vr != VK_SUCCESS) {
        st = acuda_from_vk(vr);
        goto done_fill;
    }

    for (uint32_t layer = 0; layer < layers; layer++) {
        VkImageViewCreateInfo svci = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = b->ms_src_image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = fmt,
        };
        svci.subresourceRange.aspectMask = image_is_depth(b->format)
                                               ? VK_IMAGE_ASPECT_DEPTH_BIT
                                               : VK_IMAGE_ASPECT_COLOR_BIT;
        svci.subresourceRange.levelCount = 1;
        svci.subresourceRange.baseArrayLayer = layer;
        svci.subresourceRange.layerCount = 1;
        VkImageView sview = VK_NULL_HANDLE;
        vr = vkCreateImageView(ctx->device, &svci, NULL, &sview);
        if (vr != VK_SUCCESS) {
            st = acuda_from_vk(vr);
            break;
        }
        VkImageViewCreateInfo mvci = svci;
        mvci.image = b->image;
        VkImageView mview = VK_NULL_HANDLE;
        vr = vkCreateImageView(ctx->device, &mvci, NULL, &mview);
        if (vr != VK_SUCCESS) {
            vkDestroyImageView(ctx->device, sview, NULL);
            st = acuda_from_vk(vr);
            break;
        }
        VkFramebuffer fb = VK_NULL_HANDLE;
        if (!dyn) {
            VkFramebufferCreateInfo fci = {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = rp,
                .attachmentCount = 1,
                .pAttachments = &mview,
                .width = b->width,
                .height = b->height,
                .layers = 1,
            };
            vr = vkCreateFramebuffer(ctx->device, &fci, NULL, &fb);
            if (vr != VK_SUCCESS) {
                vkDestroyImageView(ctx->device, sview, NULL);
                vkDestroyImageView(ctx->device, mview, NULL);
                st = acuda_from_vk(vr);
                break;
            }
        }
        VkDescriptorImageInfo dii = {
            .sampler = smp,
            .imageView = sview,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        VkWriteDescriptorSet wr = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = dset,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &dii,
        };
        vkUpdateDescriptorSets(ctx->device, 1, &wr, 0, NULL);

        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        st = image_begin(ctx, &pool, &cmd);
        if (st != ACUDA_SUCCESS) {
            vkDestroyFramebuffer(ctx->device, fb, NULL);
            vkDestroyImageView(ctx->device, sview, NULL);
            vkDestroyImageView(ctx->device, mview, NULL);
            break;
        }
        if (dyn) {
            // UNDEFINED discards the previous contents, matching the
            // render pass load op, then hand the image back to sampling.
            image_barrier_layer(
                cmd, b, b->image, layer, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
            VkRenderingAttachmentInfo color = {
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = mview,
                .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = { .color = { { 0, 0, 0, 0 } } },
            };
            VkRenderingInfo ri = {
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .renderArea = { { 0, 0 }, { b->width, b->height } },
                .layerCount = 1,
                .colorAttachmentCount = 1,
                .pColorAttachments = &color,
            };
            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl,
                                    0, 1, &dset, 0, NULL);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRendering(cmd);
            image_barrier_layer(
                cmd, b, b->image, layer,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        } else {
            VkClearValue clear = { .color = { { 0, 0, 0, 0 } } };
            VkRenderPassBeginInfo rbi = {
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = rp,
                .framebuffer = fb,
                .renderArea = { { 0, 0 }, { b->width, b->height } },
                .clearValueCount = 1,
                .pClearValues = &clear,
            };
            vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl,
                                    0, 1, &dset, 0, NULL);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRenderPass(cmd);
        }
        image_end(ctx, pool, cmd);
        if (fb)
            vkDestroyFramebuffer(ctx->device, fb, NULL);
        vkDestroyImageView(ctx->device, sview, NULL);
        vkDestroyImageView(ctx->device, mview, NULL);
    }

done_fill:
    vkDestroyDescriptorPool(ctx->device, dpool, NULL);
    vkDestroySampler(ctx->device, smp, NULL);
    vkDestroyPipeline(ctx->device, pipe, NULL);
    vkDestroyPipelineLayout(ctx->device, pl, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, dsl, NULL);
    if (rp) vkDestroyRenderPass(ctx->device, rp, NULL);
    vkDestroyShaderModule(ctx->device, vmod, NULL);
    vkDestroyShaderModule(ctx->device, fmod, NULL);
    return st;
}

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
        acuda_free(ctx, b);
        return st;
    }
    st = acuda_memcpy_h2d(ctx, stage, texels, bytes, NULL);
    if (st != ACUDA_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? acuda_image_create: upload memcpy failed st=%d bytes=%zu\n",
                    (int)st, bytes);
        acuda_free(ctx, stage);
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
            acuda_free(ctx, b);
            return st;
        }
        st = acuda_memcpy_h2d(ctx, stage, texels, bytes, NULL);
        if (st != ACUDA_SUCCESS) {
            acuda_free(ctx, stage);
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
