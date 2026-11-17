// Driver-internal arena allocator: real addresses for CUdeviceptr.
// One PROT_NONE virtual reservation (collision-free window); committed
// arenas underneath are host-visible Vulkan buffers, persistently mapped.
// Sub-allocations become individual VkBuffers bound into arena memory,
// so the existing launch path works via a raw (buffer+offset) entry.
// Single-threaded (matches driver).
#ifndef ACUDA_ARENA_H
#define ACUDA_ARENA_H

#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include "acuda/runtime/internal.h"

#if defined(__i386__) || (defined(_WIN32) && !defined(_WIN64))
// A 32-bit host cannot reserve the 8 GiB virtual window used on 64-bit hosts.
// Keep the reservation small: FluidMark's own host heap still needs large
// contiguous blocks for PhysX particle buffers.  Large 128 MiB host-visible
// commits also fail late in a fragmented 32-bit process (PhysX has already
// mapped most of the low 2 GiB), so commit in smaller 32 MiB chunks.
#define ACUDA_ARENA_SIZE (32ull << 20) // 32 MiB committed per arena
#define ACUDA_ARENA_MAX 4
// One extra slot holds a single large PhysX-style GPU heap (the fluid
// solver requests one 256 MiB block).  No 32 MiB slot can serve it, so
// without this every cuMemAlloc(256 MiB) fails and PhysX reports no GPU.
// Committed eagerly at boot while VA is unfragmented; best effort.
#define ACUDA_ARENA_BIG_SIZE ((256ull << 20) + (16ull << 20))
#define ACUDA_ARENA_BIG_INDEX ACUDA_ARENA_MAX
#define ACUDA_ARENA_SLOTS (ACUDA_ARENA_MAX + 1)
#else
#define ACUDA_ARENA_SIZE (1ull << 30) // 1 GiB committed per arena
#define ACUDA_ARENA_MAX 8
#define ACUDA_ARENA_BIG_SIZE 0ull
#define ACUDA_ARENA_BIG_INDEX -1
#define ACUDA_ARENA_SLOTS ACUDA_ARENA_MAX
#endif
#define ACUDA_ARENA_ALIGN 256

int acuda_arena_boot(acuda_context_t ctx);
void acuda_arena_shutdown(void);
// Allocates a VkBuffer bound into arena memory. *virt_out receives the
// stable virtual address (base+offset); *buf_out the buffer (caller
// destroys it on free). 0 ok, -1 on OOM/exhaustion.
int acuda_arena_alloc(size_t size, uint64_t *virt_out, VkBuffer *buf_out);
// Frees by virtual address (looks up size internally). 0 ok, -1 unknown.
int acuda_arena_free(uint64_t virt);
// Resolves [virt, virt+size) to mapped host memory. 0 ok, -1 invalid.
int acuda_arena_host(uint64_t virt, size_t size, void **host_out);
// Remaining bytes from virt to the end of its allocation. 0 ok, -1 invalid.
int acuda_arena_capacity(uint64_t virt, size_t *remain_out);
// Resolves to (buffer, buffer-offset) for descriptor binding.
int acuda_arena_bind(uint64_t virt, size_t size, VkBuffer *buf_out,
                     size_t *offset_out);
size_t acuda_arena_live(void);

#endif
