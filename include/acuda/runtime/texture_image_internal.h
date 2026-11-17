#ifndef ACUDA_TEXTURE_IMAGE_INTERNAL_H
#define ACUDA_TEXTURE_IMAGE_INTERNAL_H

#include "acuda/runtime/internal.h"

uint32_t image_elem_bytes(uint32_t f);
int image_is_depth(uint32_t f);
void image_barrier_img(VkCommandBuffer cmd, acuda_buffer_t b,
                       VkImage img, VkImageLayout from, VkImageLayout to,
                       VkAccessFlags src, VkAccessFlags dst,
                       VkPipelineStageFlags sp, VkPipelineStageFlags dp);
void image_level_extent(acuda_buffer_t b, uint32_t level,
                        uint32_t *w, uint32_t *h, uint32_t *d);
acuda_status_t image_begin(acuda_context_t ctx, VkCommandPool *pool,
                           VkCommandBuffer *cmd);
void image_end(acuda_context_t ctx, VkCommandPool pool, VkCommandBuffer cmd);
void image_copy_level(acuda_context_t ctx, VkCommandBuffer cmd,
                      acuda_buffer_t b, VkImage img, VkBuffer stage,
                      uint32_t level, uint32_t layer,
                      size_t buffer_offset);
acuda_status_t ms_fill(acuda_context_t ctx, acuda_buffer_t b);

#endif
