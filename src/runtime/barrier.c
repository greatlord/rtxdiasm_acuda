// Shared conservative buffer barrier: makes prior writes available and
// visible (acquire) and this batch's writes available for later batches
// (release). Correctness first; narrowed scopes are future work.
//
// Uses synchronization2 (required at device selection): the 1.3 barrier
// API takes 64-bit stage/access masks directly.
#include "acuda/runtime/internal.h"

void acuda_buf_barrier(VkCommandBuffer cmd, VkBuffer buf,
                       VkDeviceSize offset, VkDeviceSize size) {
    VkBufferMemoryBarrier2 b = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        // Compute/transfer only: graphics stages never touch SSBOs here.
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
        .buffer = buf,
        .offset = offset,
        .size = size,
    };
    VkDependencyInfo dep = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &b,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}
