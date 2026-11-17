#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/memory.h"
#include "acuda/runtime/internal.h"

static int ctx_buffer_add(acuda_context_t ctx, acuda_buffer_t b) {
    if (ctx->nbufs + 1 > ctx->capbufs) {
        size_t ncap = ctx->capbufs ? ctx->capbufs * 2 : 16;
        acuda_buffer_t *nb = realloc(ctx->buffers, sizeof(*nb) * ncap);
        if (!nb)
            return -1;
        ctx->buffers = nb;
        ctx->capbufs = ncap;
    }
    ctx->buffers[ctx->nbufs++] = b;
    return 0;
}

static void ctx_buffer_remove(acuda_context_t ctx, acuda_buffer_t b) {
    for (size_t i = 0; i < ctx->nbufs; i++) {
        if (ctx->buffers[i] == b) {
            ctx->buffers[i] = ctx->buffers[--ctx->nbufs];
            return;
        }
    }
}

void acuda_buffer_unregister(acuda_context_t ctx, acuda_buffer_t b) {
    ctx_buffer_remove(ctx, b);
}

acuda_buffer_t acuda_buffer_find(acuda_context_t ctx, VkBuffer buffer) {
    if (!ctx || buffer == VK_NULL_HANDLE)
        return NULL;
    for (size_t i = 0; i < ctx->nbufs; i++)
        if (ctx->buffers[i] && ctx->buffers[i]->buffer == buffer)
            return ctx->buffers[i];
    return NULL;
}

static acuda_status_t alloc_impl(acuda_context_t ctx, acuda_buffer_t *out_buf,
                                 size_t bytes, uint32_t mtype,
                                 int host_visible) {
    if (!ctx || !out_buf)
        return ACUDA_ERROR_INVALID_HANDLE;
    *out_buf = NULL;
    if (bytes == 0 || mtype == UINT32_MAX)
        return ACUDA_ERROR_INVALID_VALUE;
    // maintenance4 exposes the hard device buffer-size limit; the
    // per-binding storage range limit is checked at launch time so that
    // large copy-only staging buffers stay legal.
    if (ctx->max_buffer_size &&
        (uint64_t)bytes > (uint64_t)ctx->max_buffer_size)
        return ACUDA_ERROR_NOT_SUPPORTED;

    // Subword stores are lowered via 32-bit atomic RMWs, so every storage
    // buffer must cover at least one aligned word even for smaller logical
    // allocations (e.g. 2-byte outputs).
    size_t alloc_bytes = bytes;
    if (alloc_bytes < 4)
        alloc_bytes = 4;
    alloc_bytes = (alloc_bytes + 3u) & ~(size_t)3u;
    if (ctx->max_buffer_size &&
        (uint64_t)alloc_bytes > (uint64_t)ctx->max_buffer_size)
        alloc_bytes = bytes;

    acuda_buffer_t b = calloc(1, sizeof(*b));
    if (!b)
        return ACUDA_ERROR_OUT_OF_MEMORY;
    b->size = alloc_bytes;
    b->host_visible = host_visible;

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (ctx->bda_ok)
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = alloc_bytes,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkResult vr = vkCreateBuffer(ctx->device, &bci, NULL, &b->buffer);
    if (vr != VK_SUCCESS) {
        free(b);
        return acuda_from_vk(vr);
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx->device, b->buffer, &req);
    if (!(req.memoryTypeBits & (1u << mtype))) {
        vkDestroyBuffer(ctx->device, b->buffer, NULL);
        free(b);
        return ACUDA_ERROR_NOT_SUPPORTED;
    }
    VkMemoryAllocateFlagsInfo addr_flags = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = mtype,
        .pNext = ctx->bda_ok ? &addr_flags : NULL,
    };
    vr = vkAllocateMemory(ctx->device, &mai, NULL, &b->memory);
    if (vr != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, b->buffer, NULL);
        free(b);
        return acuda_from_vk(vr);
    }
    vr = vkBindBufferMemory(ctx->device, b->buffer, b->memory, 0);
    if (vr != VK_SUCCESS) {
        vkFreeMemory(ctx->device, b->memory, NULL);
        vkDestroyBuffer(ctx->device, b->buffer, NULL);
        free(b);
        return acuda_from_vk(vr);
    }
    if (ctx->bda_ok) {
        VkBufferDeviceAddressInfo bdai = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = b->buffer,
        };
        b->addr = vkGetBufferDeviceAddress(ctx->device, &bdai);
    }
    if (ctx_buffer_add(ctx, b) != 0) {
        vkFreeMemory(ctx->device, b->memory, NULL);
        vkDestroyBuffer(ctx->device, b->buffer, NULL);
        free(b);
        return ACUDA_ERROR_OUT_OF_MEMORY;
    }
    *out_buf = b;
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_malloc(acuda_context_t ctx, acuda_buffer_t *out_buf,
                            size_t bytes) {
    return alloc_impl(ctx, out_buf, bytes, ctx ? ctx->mem_type : UINT32_MAX, 1);
}

acuda_status_t acuda_malloc_host(acuda_context_t ctx, acuda_buffer_t *out,
                                 size_t bytes) {
    return alloc_impl(ctx, out, bytes, ctx ? ctx->mem_type : UINT32_MAX, 1);
}

acuda_status_t acuda_malloc_device(acuda_context_t ctx, acuda_buffer_t *out,
                                   size_t bytes) {
    if (!ctx)
        return ACUDA_ERROR_INVALID_HANDLE;
    return alloc_impl(ctx, out, bytes, ctx->mem_type_dev,
                      ctx->mem_type_dev == ctx->mem_type);
}

acuda_status_t acuda_buffer_device_address(acuda_buffer_t buf,
                                           uint64_t *out_addr) {
    if (!buf || !out_addr)
        return ACUDA_ERROR_INVALID_VALUE;
    if (buf->addr == 0)
        return ACUDA_ERROR_NOT_SUPPORTED;
    *out_addr = (uint64_t)buf->addr;
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_free(acuda_context_t ctx, acuda_buffer_t buf) {
    if (!ctx || !buf)
        return ACUDA_ERROR_INVALID_HANDLE;
    // Submit any recorded batch that may still reference this buffer
    // before its handles are destroyed (the caller owns completion).
    acuda_stream_flush_all(ctx);
    ctx_buffer_remove(ctx, buf);
    if (buf->kind == 2) { // image
        if (buf->ms_src_view)
            vkDestroyImageView(ctx->device, buf->ms_src_view, NULL);
        if (buf->ms_src_image)
            vkDestroyImage(ctx->device, buf->ms_src_image, NULL);
        if (buf->ms_src_memory)
            vkFreeMemory(ctx->device, buf->ms_src_memory, NULL);
        if (buf->uint_view)
            vkDestroyImageView(ctx->device, buf->uint_view, NULL);
        if (buf->view)
            vkDestroyImageView(ctx->device, buf->view, NULL);
        if (buf->image)
            vkDestroyImage(ctx->device, buf->image, NULL);
        if (buf->memory)
            vkFreeMemory(ctx->device, buf->memory, NULL);
    } else if (buf->kind == 3 && buf->owns_storage) {
        if (buf->uint_view)
            vkDestroyImageView(ctx->device, buf->uint_view, NULL);
        if (buf->view)
            vkDestroyImageView(ctx->device, buf->view, NULL);
        if (buf->image)
            vkDestroyImage(ctx->device, buf->image, NULL);
        if (buf->memory)
            vkFreeMemory(ctx->device, buf->memory, NULL);
        if (buf->sampler)
            vkDestroySampler(ctx->device, buf->sampler, NULL);
    } else if (buf->kind == 4) { // sampler
        if (buf->sampler)
            vkDestroySampler(ctx->device, buf->sampler, NULL);
    } else if (buf->kind == 0) { // ordinary buffer
        vkDestroyBuffer(ctx->device, buf->buffer, NULL);
        vkFreeMemory(ctx->device, buf->memory, NULL);
    }
    // kind 1 (buffer texture) and 3 (combined texture) are borrowed
    // descriptors: the underlying buffer/image/sampler owns storage.
    free(buf);
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_buffer_size(acuda_buffer_t buf, size_t *out_bytes) {
    if (!buf || !out_bytes)
        return ACUDA_ERROR_INVALID_VALUE;
    *out_bytes = buf->size;
    return ACUDA_SUCCESS;
}

static acuda_status_t stage_copy(acuda_context_t ctx, acuda_stream_t s,
                                 acuda_buffer_t dst, acuda_buffer_t src,
                                 size_t bytes);

acuda_status_t acuda_memcpy_h2d(acuda_context_t ctx, acuda_buffer_t dst,
                                const void *src, size_t bytes, acuda_stream_t stream) {
    acuda_stream_t s = NULL;
    acuda_status_t st = acuda_resolve_stream(ctx, stream, &s);
    if (st != ACUDA_SUCCESS)
        return st;
    if (!dst || !src)
        return ACUDA_ERROR_INVALID_VALUE;
    if (bytes > dst->size)
        return ACUDA_ERROR_INVALID_VALUE;
    // Order host write after prior device work on this stream.
    st = acuda_stream_synchronize(ctx, s);
    if (st != ACUDA_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? memcpy_h2d: synchronize failed st=%d bytes=%zu\n",
                    (int)st, bytes);
        return st;
    }
    if (bytes == 0)
        return ACUDA_SUCCESS;
    if (!dst->host_visible) {
        // Device-local destination: upload through a host-visible staging
        // buffer and a queued transfer.
        acuda_buffer_t stage = NULL;
        acuda_status_t cst = acuda_malloc_host(ctx, &stage, bytes);
        if (cst != ACUDA_SUCCESS)
            return cst;
        void *p = NULL;
        VkResult vr = vkMapMemory(ctx->device, stage->memory, 0, bytes, 0, &p);
        if (vr == VK_SUCCESS) {
            memcpy(p, src, bytes);
            vkUnmapMemory(ctx->device, stage->memory);
        } else {
            cst = acuda_from_vk(vr);
        }
        if (cst == ACUDA_SUCCESS)
            cst = stage_copy(ctx, s, dst, stage, bytes);
        acuda_free(ctx, stage);
        return cst;
    }
    void *p = NULL;
    VkResult vr = vkMapMemory(ctx->device, dst->memory, 0, bytes, 0, &p);
    if (vr != VK_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? memcpy_h2d: vkMapMemory failed vr=%d bytes=%zu\n",
                    (int)vr, bytes);
        return acuda_from_vk(vr);
    }
    memcpy(p, src, bytes);
    vkUnmapMemory(ctx->device, dst->memory);
    dst->gpu_dirty = 1; // host write: acquire before the next device use
    dst->gpu_read = 0;
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_memcpy_d2h(acuda_context_t ctx, void *dst,
                                acuda_buffer_t src, size_t bytes, acuda_stream_t stream) {
    acuda_stream_t s = NULL;
    acuda_status_t st = acuda_resolve_stream(ctx, stream, &s);
    if (st != ACUDA_SUCCESS)
        return st;
    if (!dst || !src)
        return ACUDA_ERROR_INVALID_VALUE;
    if (bytes > src->size)
        return ACUDA_ERROR_INVALID_VALUE;
    // Order host read after prior device work on this stream.
    st = acuda_stream_synchronize(ctx, s);
    if (st != ACUDA_SUCCESS)
        return st;
    if (bytes == 0)
        return ACUDA_SUCCESS;
    if (!src->host_visible) {
        acuda_buffer_t stage = NULL;
        acuda_status_t cst = acuda_malloc_host(ctx, &stage, bytes);
        if (cst != ACUDA_SUCCESS)
            return cst;
        cst = stage_copy(ctx, s, stage, src, bytes);
        if (cst == ACUDA_SUCCESS) {
            void *p = NULL;
            VkResult vr = vkMapMemory(ctx->device, stage->memory, 0, bytes,
                                      0, &p);
            if (vr == VK_SUCCESS) {
                memcpy(dst, p, bytes);
                vkUnmapMemory(ctx->device, stage->memory);
            } else {
                cst = acuda_from_vk(vr);
            }
        }
        acuda_free(ctx, stage);
        return cst;
    }
    void *p = NULL;
    VkResult vr = vkMapMemory(ctx->device, src->memory, 0, bytes, 0, &p);
    if (vr != VK_SUCCESS)
        return acuda_from_vk(vr);
    memcpy(dst, p, bytes);
    vkUnmapMemory(ctx->device, src->memory);
    src->gpu_read = 0; // host completion wait drains the read
    return ACUDA_SUCCESS;
}

// Conservative full-buffer barrier covering transfer + shader access.
// Correctness first; narrowed barriers are future work.
static void full_barrier(VkCommandBuffer cmd, acuda_buffer_t buf) {
    VkBufferMemoryBarrier2 b = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT |
                         VK_ACCESS_2_SHADER_WRITE_BIT |
                         VK_ACCESS_2_TRANSFER_READ_BIT |
                         VK_ACCESS_2_SHADER_READ_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT |
                         VK_ACCESS_2_TRANSFER_WRITE_BIT |
                         VK_ACCESS_2_SHADER_READ_BIT |
                         VK_ACCESS_2_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buf->buffer,
        .offset = 0,
        .size = buf->size,
    };
    VkDependencyInfo dep = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &b,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

typedef struct copy_args {
    acuda_buffer_t dst;
    acuda_buffer_t src;
    size_t bytes;
} copy_args_t;

static acuda_status_t record_copy(VkCommandBuffer cmd, void *user) {
    copy_args_t *a = user;
    full_barrier(cmd, a->src);
    full_barrier(cmd, a->dst);
    a->src->gpu_dirty = 0;
    a->src->gpu_read = 0;
    a->dst->gpu_dirty = 0;
    a->dst->gpu_read = 0;
    VkBufferCopy2 region = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
        .srcOffset = 0,
        .dstOffset = 0,
        .size = a->bytes,
    };
    VkCopyBufferInfo2 cinfo = {
        .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
        .srcBuffer = a->src->buffer,
        .dstBuffer = a->dst->buffer,
        .regionCount = 1,
        .pRegions = &region,
    };
    vkCmdCopyBuffer2(cmd, &cinfo);
    full_barrier(cmd, a->dst);
    a->src->gpu_read = 1;
    a->dst->gpu_dirty = 1;
    a->dst->gpu_read = 0;
    return ACUDA_SUCCESS;
}

static acuda_status_t stage_copy(acuda_context_t ctx, acuda_stream_t s,
                                 acuda_buffer_t dst, acuda_buffer_t src,
                                 size_t bytes) {
    copy_args_t a = { dst, src, bytes };
    acuda_status_t st = acuda_stream_append(ctx, s, record_copy, &a, 0);
    if (st == ACUDA_SUCCESS)
        st = acuda_stream_maybe_flush(ctx, s);
    if (st == ACUDA_SUCCESS)
        st = acuda_stream_synchronize(ctx, s);
    return st;
}

acuda_status_t acuda_memcpy_d2d(acuda_context_t ctx, acuda_buffer_t dst,
                                acuda_buffer_t src, size_t bytes, acuda_stream_t stream) {
    acuda_stream_t s = NULL;
    acuda_status_t st = acuda_resolve_stream(ctx, stream, &s);
    if (st != ACUDA_SUCCESS)
        return st;
    if (!dst || !src)
        return ACUDA_ERROR_INVALID_VALUE;
    if (bytes > dst->size || bytes > src->size)
        return ACUDA_ERROR_INVALID_VALUE;
    if (bytes == 0)
        return ACUDA_SUCCESS;
    copy_args_t a = { dst, src, bytes };
    st = acuda_stream_append(ctx, s, record_copy, &a, 0);
    if (st != ACUDA_SUCCESS)
        return st;
    return acuda_stream_maybe_flush(ctx, s);
}

typedef struct fill_args {
    acuda_buffer_t buf;
    uint32_t value;
    size_t offset;
    size_t bytes;
} fill_args_t;

static acuda_status_t record_fill(VkCommandBuffer cmd, void *user) {
    fill_args_t *a = user;
    full_barrier(cmd, a->buf);
    a->buf->gpu_dirty = 0;
    a->buf->gpu_read = 0;
    vkCmdFillBuffer(cmd, a->buf->buffer, a->offset, a->bytes, a->value);
    full_barrier(cmd, a->buf);
    a->buf->gpu_dirty = 1;
    a->buf->gpu_read = 0;
    return ACUDA_SUCCESS;
}

static acuda_status_t memset_common(acuda_context_t ctx, acuda_buffer_t buf,
                                    uint32_t word, size_t offset, size_t bytes,
                                    acuda_stream_t stream) {
    acuda_stream_t s = NULL;
    acuda_status_t st = acuda_resolve_stream(ctx, stream, &s);
    if (st != ACUDA_SUCCESS)
        return st;
    if (!buf)
        return ACUDA_ERROR_INVALID_VALUE;
    if (offset % 4 != 0 || bytes % 4 != 0 || offset + bytes > buf->size)
        return ACUDA_ERROR_INVALID_VALUE;
    if (bytes == 0)
        return ACUDA_SUCCESS;
    fill_args_t a = { buf, word, offset, bytes };
    st = acuda_stream_append(ctx, s, record_fill, &a, 0);
    if (st != ACUDA_SUCCESS)
        return st;
    return acuda_stream_maybe_flush(ctx, s);
}

acuda_status_t acuda_memset_d32(acuda_context_t ctx, acuda_buffer_t buf,
                                uint32_t value, size_t offset, size_t bytes,
                                acuda_stream_t stream) {
    return memset_common(ctx, buf, value, offset, bytes, stream);
}

acuda_status_t acuda_memset_d8(acuda_context_t ctx, acuda_buffer_t buf,
                               uint8_t value, size_t offset, size_t bytes,
                               acuda_stream_t stream) {
    uint32_t word = (uint32_t)value * 0x01010101u;
    return memset_common(ctx, buf, word, offset, bytes, stream);
}

