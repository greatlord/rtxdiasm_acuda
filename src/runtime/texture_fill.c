#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "acuda/runtime/internal.h"
#include "acuda/runtime/texture_image_internal.h"

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

acuda_status_t ms_fill(acuda_context_t ctx, acuda_buffer_t b) {
    { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT ms_fill in b=%p ms=%d\n",(void*)b,b?b->is_ms:0); fflush(_f); fclose(_f);} } }
    if (!b->is_ms) {
        { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT ms_fill out st=0 (noop)\n"); fflush(_f); fclose(_f);} } }
        return ACUDA_SUCCESS;
    }
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
    { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT ms_fill out st=%d\n",(int)st); fflush(_f); fclose(_f);} } }
    return st;
}
