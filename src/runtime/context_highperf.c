#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/compiler.h"
#include "acuda/runtime/internal.h"
#include "acuda/runtime/context_internal.h"
#include "acuda/runtime/time_compat.h"
#if defined(_WIN32)
#include <windows.h>
#define acuda_setenv(name, value) _putenv_s((name), (value))
#define acuda_unsetenv(name) _putenv_s((name), "")
#define acuda_strdup _strdup
#else
#include <pthread.h>
#define acuda_setenv(name, value) setenv((name), (value), 1)
#define acuda_unsetenv(name) unsetenv((name))
#define acuda_strdup strdup
#endif

// ---- GPU clock-ramp helpers (ACUDA_HIGH_PERF / warm-up) ----
void acuda_warmup_gpu(acuda_context_t ctx) {
    const char *ms_env = getenv("ACUDA_WARMUP_MS");
    int ms = ms_env ? atoi(ms_env) : 2000;
    if (ms < 0)
        ms = 0;
    if (ms == 0)
        return;
    if (!ctx || !ctx->device || ctx->queue == VK_NULL_HANDLE)
        return;

    static const char ramp_ptx[] =
        ".version 8.0\n"
        ".target sm_80\n"
        ".address_size 64\n"
        "\n"
        ".visible .entry gpu_warmup(\n"
        "    .param .u32 iterations\n"
        ")\n"
        "{\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .u32 %r<3>;\n"
        "    .reg .f32 %f<4>;\n"
        "\n"
        "    ld.param.u32 %r0, [iterations];\n"
        "\n"
        "    mov.u32 %r1, 0;\n"
        "    mov.f32 %f0, 1.001;\n"
        "    mov.f32 %f1, 1.002;\n"
        "    mov.f32 %f2, 0.001;\n"
        "\n"
        "LOOP:\n"
        "    fma.rn.f32 %f3, %f0, %f1, %f2;\n"
        "    mov.f32 %f2, %f3;\n"
        "\n"
        "    add.u32 %r1, %r1, 1;\n"
        "    setp.lt.u32 %p0, %r1, %r0;\n"
        "    @%p0 bra LOOP;\n"
        "\n"
        "    ret;\n"
        "}\n";

    acuda_kernel_t *k = NULL;
    char err[256];
    int compiled = (acuda_compile_ptx(ramp_ptx, "gpu_warmup", &k, err,
                                      sizeof(err), &ctx->compiler_caps) == ACUDA_SUCCESS);
    if (compiled) {
        uint32_t reps = 1000000;
        uint64_t deadline = acuda_now_ns() + (uint64_t)ms * 1000000ull;
        while (acuda_now_ns() < deadline) {
            acuda_launch_kernel(ctx, k, NULL, 0, &reps, sizeof(reps), 64,
                                256, NULL);
            acuda_stream_synchronize(ctx, NULL);
        }
        acuda_kernel_free(k);
        return;
    }

    fprintf(stderr, "ACuda: ramp PTX compile failed (%s); using fill warm-up\n",
            err);

    VkBuffer vbuf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = 64ull * 1024 * 1024;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(ctx->device, &bci, NULL, &vbuf) != VK_SUCCESS)
        return;
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(ctx->device, vbuf, &mr);
    uint32_t mt = (ctx->mem_type_dev != UINT32_MAX) ? ctx->mem_type_dev
                                                     : ctx->mem_type;
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mt;
    if (vkAllocateMemory(ctx->device, &mai, NULL, &mem) != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, vbuf, NULL);
        return;
    }
    vkBindBufferMemory(ctx->device, vbuf, mem, 0);
    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.queueFamilyIndex = ctx->queue_family;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(ctx->device, &cpci, NULL, &pool) != VK_SUCCESS) {
        vkFreeMemory(ctx->device, mem, NULL);
        vkDestroyBuffer(ctx->device, vbuf, NULL);
        return;
    }
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(ctx->device, &cbai, &cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(ctx->device, pool, NULL);
        vkFreeMemory(ctx->device, mem, NULL);
        vkDestroyBuffer(ctx->device, vbuf, NULL);
        return;
    }
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(ctx->device, &fci, NULL, &fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(ctx->device, pool, 1, &cmd);
        vkDestroyCommandPool(ctx->device, pool, NULL);
        vkFreeMemory(ctx->device, mem, NULL);
        vkDestroyBuffer(ctx->device, vbuf, NULL);
        return;
    }
    uint64_t deadline = acuda_now_ns() + (uint64_t)ms * 1000000ull;
    while (acuda_now_ns() < deadline) {
        VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdFillBuffer(cmd, vbuf, 0, VK_WHOLE_SIZE, 0xABCD1234u);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(ctx->queue, 1, &si, fence);
        vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(ctx->device, 1, &fence);
        vkResetCommandBuffer(cmd, 0);
    }
    vkDestroyFence(ctx->device, fence, NULL);
    vkFreeCommandBuffers(ctx->device, pool, 1, &cmd);
    vkDestroyCommandPool(ctx->device, pool, NULL);
    vkFreeMemory(ctx->device, mem, NULL);
    vkDestroyBuffer(ctx->device, vbuf, NULL);
}

#if defined(_WIN32)
static DWORD WINAPI acuda_bg_thread_proc(LPVOID arg)
#else
static void *acuda_bg_thread_proc(void *arg)
#endif
{
    acuda_context_t ctx = (acuda_context_t)arg;
    acuda_context_t work = (ctx && ctx->bg_ctx) ? ctx->bg_ctx : ctx;
    if (work && work->bg_kernel && work->bg_stream) {
        acuda_kernel_t *k = (acuda_kernel_t *)work->bg_kernel;
        uint32_t reps = work->bg_reps;
        while (work && !work->bg_stop) {
            acuda_launch_kernel(work, k, NULL, 0, &reps, sizeof(reps), 64, 256,
                                work->bg_stream);
            acuda_stream_synchronize(work, work->bg_stream);
        }
    } else {
        while (work && !work->bg_stop) {
            if (work->bg_cmd == VK_NULL_HANDLE || work->bg_buffer == VK_NULL_HANDLE ||
                work->bg_queue == VK_NULL_HANDLE)
                break;
            VkCommandBufferBeginInfo bi = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            if (vkBeginCommandBuffer(work->bg_cmd, &bi) != VK_SUCCESS)
                break;
            vkCmdFillBuffer(work->bg_cmd, work->bg_buffer, 0, VK_WHOLE_SIZE,
                            0xABCD1234u);
            vkEndCommandBuffer(work->bg_cmd);
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
            si.commandBufferCount = 1;
            si.pCommandBuffers = &work->bg_cmd;
            if (vkQueueSubmit(work->bg_queue, 1, &si, work->bg_fence) != VK_SUCCESS)
                break;
            vkWaitForFences(work->device, 1, &work->bg_fence, VK_TRUE, UINT64_MAX);
            vkResetFences(work->device, 1, &work->bg_fence);
            vkResetCommandBuffer(work->bg_cmd, 0);
        }
    }
#if defined(_WIN32)
    return 0;
#else
    return NULL;
#endif
}



static int start_high_perf_isolated(acuda_context_t main) {
    // Keep-busy runs on a second, unvalidated ACuda context pinned to the
    // same physical GPU.  Sharing the validated VkInstance/device makes the
    // validation layer see concurrent queue use and live keep-busy objects
    // during teardown.
    char *old_nv = NULL, *old_hp = NULL, *old_dev = NULL, *old_warm = NULL;
    const char *v = getenv("ACUDA_NO_VALIDATION");
    if (v) old_nv = acuda_strdup(v);
    v = getenv("ACUDA_HIGH_PERF");
    if (v) old_hp = acuda_strdup(v);
    v = getenv("ACUDA_DEVICE");
    if (v) old_dev = acuda_strdup(v);
    v = getenv("ACUDA_WARMUP_MS");
    if (v) old_warm = acuda_strdup(v);

    acuda_setenv("ACUDA_NO_VALIDATION", "1");
    acuda_setenv("ACUDA_HIGH_PERF", "0");
    acuda_setenv("ACUDA_DEVICE", main->props.deviceName);
    acuda_setenv("ACUDA_WARMUP_MS", "0");
    acuda_context_t bg = NULL;
    acuda_status_t st = acuda_init(&bg);
    if (old_nv) { acuda_setenv("ACUDA_NO_VALIDATION", old_nv); free(old_nv); }
    else acuda_unsetenv("ACUDA_NO_VALIDATION");
    if (old_hp) { acuda_setenv("ACUDA_HIGH_PERF", old_hp); free(old_hp); }
    else acuda_unsetenv("ACUDA_HIGH_PERF");
    if (old_dev) { acuda_setenv("ACUDA_DEVICE", old_dev); free(old_dev); }
    else acuda_unsetenv("ACUDA_DEVICE");
    if (old_warm) { acuda_setenv("ACUDA_WARMUP_MS", old_warm); free(old_warm); }
    else acuda_unsetenv("ACUDA_WARMUP_MS");
    if (st != ACUDA_SUCCESS || !bg)
        return 0;

    static const char ramp_ptx[] =
        ".version 8.0\n"
        ".target sm_80\n"
        ".address_size 64\n"
        "\n"
        ".visible .entry gpu_warmup(\n"
        "    .param .u32 iterations\n"
        ")\n"
        "{\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .u32 %r<3>;\n"
        "    .reg .f32 %f<4>;\n"
        "\n"
        "    ld.param.u32 %r0, [iterations];\n"
        "\n"
        "    mov.u32 %r1, 0;\n"
        "    mov.f32 %f0, 1.001;\n"
        "    mov.f32 %f1, 1.002;\n"
        "    mov.f32 %f2, 0.001;\n"
        "\n"
        "LOOP:\n"
        "    fma.rn.f32 %f3, %f0, %f1, %f2;\n"
        "    mov.f32 %f2, %f3;\n"
        "\n"
        "    add.u32 %r1, %r1, 1;\n"
        "    setp.lt.u32 %p0, %r1, %r0;\n"
        "    @%p0 bra LOOP;\n"
        "\n"
        "    ret;\n"
        "}\n";
    acuda_kernel_t *k = NULL;
    char err[256];
    if (acuda_compile_ptx(ramp_ptx, "gpu_warmup", &k, err, sizeof(err), &bg->compiler_caps) !=
        ACUDA_SUCCESS) {
        acuda_shutdown(bg);
        return 0;
    }
    acuda_stream_t st_stream = NULL;
    if (acuda_stream_create(bg, &st_stream) != ACUDA_SUCCESS) {
        acuda_kernel_free(k);
        acuda_shutdown(bg);
        return 0;
    }
    bg->bg_kernel = k;
    bg->bg_stream = st_stream;
    bg->bg_reps = 1000000;
    bg->bg_stop = 0;
    main->bg_ctx = bg;
    main->bg_stop = 0;
#if defined(_WIN32)
    main->bg_thread = CreateThread(NULL, 0, acuda_bg_thread_proc, main, 0, NULL);
    main->bg_thread_created = (main->bg_thread != NULL);
#else
    pthread_t *pt = (pthread_t *)malloc(sizeof(pthread_t));
    if (pt && pthread_create(pt, NULL, acuda_bg_thread_proc, main) == 0) {
        main->bg_thread = pt;
        main->bg_thread_created = 1;
    } else {
        free(pt);
        main->bg_thread = NULL;
        main->bg_thread_created = 0;
    }
#endif
    if (!main->bg_thread_created) {
        main->bg_ctx = NULL;
        acuda_kernel_free(k);
        acuda_stream_destroy(bg, st_stream);
        acuda_shutdown(bg);
        return 0;
    }
    return 1;
}

void acuda_start_high_perf(acuda_context_t ctx) {
    const char *hp = getenv("ACUDA_HIGH_PERF");
    const char *async = getenv("ACUDA_ASYNC_STREAMS");
    // Keep-busy is OFF by default; only enabled when explicitly requested.
    if (!ctx || !hp || strcmp(hp, "1") != 0 ||
        (async && *async && atoi(async) == 0))
        return;
    if (ctx->validation_on) {
        start_high_perf_isolated(ctx);
        return;
    }
    if (getenv("ACUDA_DEBUG_HIGH_PERF"))
        fprintf(stderr,
                "ACuda high-perf: primary family %u, bg family %u, %s\n",
                ctx->queue_family, ctx->bg_queue_family,
                ctx->bg_queue == VK_NULL_HANDLE
                    ? "no dedicated queue (disabled)"
                    : "keep-busy enabled");
    // The keep-busy loop submits from a background thread.  Without a
    // dedicated secondary queue it shares the primary queue and races
    // user submissions (Vulkan validation flags simultaneous
    // vkQueueSubmit2 and the driver can crash).  Some devices expose only
    // one queue in this family, so skip keep-busy there.
    if (ctx->bg_queue == VK_NULL_HANDLE)
        return;

    // Prefer a PTX compute kernel for real core load.
    static const char ramp_ptx[] =
        ".version 8.0\n"
        ".target sm_80\n"
        ".address_size 64\n"
        "\n"
        ".visible .entry gpu_warmup(\n"
        "    .param .u32 iterations\n"
        ")\n"
        "{\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .u32 %r<3>;\n"
        "    .reg .f32 %f<4>;\n"
        "\n"
        "    ld.param.u32 %r0, [iterations];\n"
        "\n"
        "    mov.u32 %r1, 0;\n"
        "    mov.f32 %f0, 1.001;\n"
        "    mov.f32 %f1, 1.002;\n"
        "    mov.f32 %f2, 0.001;\n"
        "\n"
        "LOOP:\n"
        "    fma.rn.f32 %f3, %f0, %f1, %f2;\n"
        "    mov.f32 %f2, %f3;\n"
        "\n"
        "    add.u32 %r1, %r1, 1;\n"
        "    setp.lt.u32 %p0, %r1, %r0;\n"
        "    @%p0 bra LOOP;\n"
        "\n"
        "    ret;\n"
        "}\n";
    acuda_kernel_t *k = NULL;
    char err[256];
    if (acuda_compile_ptx(ramp_ptx, "gpu_warmup", &k, err, sizeof(err), &ctx->compiler_caps) ==
        ACUDA_SUCCESS) {
        acuda_stream_t st = NULL;
        if (acuda_stream_create_on_queue(ctx, ctx->bg_queue,
                                         ctx->bg_queue_family, &st) ==
            ACUDA_SUCCESS) {
            // The stream is bound to the dedicated keep-busy queue and its
            // family, so its command pool is valid there. Sharing the
            // primary queue races user submissions (Vulkan validation flags
            // simultaneous vkQueueSubmit2 from two threads).
            ctx->bg_kernel = k;
            ctx->bg_buf = NULL;
            ctx->bg_reps = 1000000;
            ctx->bg_stream = st;
                ctx->bg_stop = 0;
                ctx->bg_high_perf = 1;
#if defined(_WIN32)
                ctx->bg_thread = CreateThread(NULL, 0, acuda_bg_thread_proc, ctx, 0, NULL);
                ctx->bg_thread_created = (ctx->bg_thread != NULL);
#else
                pthread_t *pt = (pthread_t *)malloc(sizeof(pthread_t));
                if (pt && pthread_create(pt, NULL, acuda_bg_thread_proc, ctx) == 0) {
                    ctx->bg_thread = pt;
                    ctx->bg_thread_created = 1;
                } else {
                    free(pt);
                    ctx->bg_thread = NULL;
                    ctx->bg_thread_created = 0;
                }
#endif
                if (ctx->bg_thread_created)
                    return;
                // thread failed: fall through to fill path (clean up PTX)
                ctx->bg_kernel = NULL;
                ctx->bg_buf = NULL;
                ctx->bg_stream = NULL;
                acuda_stream_destroy(ctx, st);
                acuda_kernel_free(k);
            } else {
                acuda_kernel_free(k);
            }
        } else {
        }

    // Fallback: Vulkan fill on a second queue (original behavior).
    if (ctx->bg_queue == VK_NULL_HANDLE)
        return;

    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = 64ull * 1024 * 1024;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(ctx->device, &bci, NULL, &ctx->bg_buffer) != VK_SUCCESS)
        return;
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(ctx->device, ctx->bg_buffer, &mr);
    uint32_t mt = (ctx->mem_type_dev != UINT32_MAX) ? ctx->mem_type_dev
                                                     : ctx->mem_type;
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mt;
    if (vkAllocateMemory(ctx->device, &mai, NULL, &ctx->bg_memory) != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, ctx->bg_buffer, NULL);
        ctx->bg_buffer = VK_NULL_HANDLE;
        return;
    }
    vkBindBufferMemory(ctx->device, ctx->bg_buffer, ctx->bg_memory, 0);

    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.queueFamilyIndex = ctx->bg_queue_family;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(ctx->device, &cpci, NULL, &ctx->bg_pool) != VK_SUCCESS) {
        vkFreeMemory(ctx->device, ctx->bg_memory, NULL);
        vkDestroyBuffer(ctx->device, ctx->bg_buffer, NULL);
        ctx->bg_memory = VK_NULL_HANDLE;
        ctx->bg_buffer = VK_NULL_HANDLE;
        return;
    }
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = ctx->bg_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(ctx->device, &cbai, &ctx->bg_cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(ctx->device, ctx->bg_pool, NULL);
        vkFreeMemory(ctx->device, ctx->bg_memory, NULL);
        vkDestroyBuffer(ctx->device, ctx->bg_buffer, NULL);
        ctx->bg_pool = VK_NULL_HANDLE;
        ctx->bg_memory = VK_NULL_HANDLE;
        ctx->bg_buffer = VK_NULL_HANDLE;
        return;
    }
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(ctx->device, &fci, NULL, &ctx->bg_fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(ctx->device, ctx->bg_pool, 1, &ctx->bg_cmd);
        vkDestroyCommandPool(ctx->device, ctx->bg_pool, NULL);
        vkFreeMemory(ctx->device, ctx->bg_memory, NULL);
        vkDestroyBuffer(ctx->device, ctx->bg_buffer, NULL);
        ctx->bg_cmd = VK_NULL_HANDLE;
        ctx->bg_pool = VK_NULL_HANDLE;
        ctx->bg_memory = VK_NULL_HANDLE;
        ctx->bg_buffer = VK_NULL_HANDLE;
        return;
    }

    ctx->bg_stop = 0;
    ctx->bg_high_perf = 1;
#if defined(_WIN32)
    ctx->bg_thread = CreateThread(NULL, 0, acuda_bg_thread_proc, ctx, 0, NULL);
    ctx->bg_thread_created = (ctx->bg_thread != NULL);
#else
    pthread_t *pt = (pthread_t *)malloc(sizeof(pthread_t));
    if (pt && pthread_create(pt, NULL, acuda_bg_thread_proc, ctx) == 0) {
        ctx->bg_thread = pt;
        ctx->bg_thread_created = 1;
    } else {
        free(pt);
        ctx->bg_thread = NULL;
        ctx->bg_thread_created = 0;
    }
#endif
}

void acuda_stop_high_perf(acuda_context_t ctx) {
    if (!ctx)
        return;
    acuda_context_t work = ctx->bg_ctx ? ctx->bg_ctx : ctx;
    if (ctx->bg_thread_created) {
        ctx->bg_stop = 1;
        if (work)
            work->bg_stop = 1;
#if defined(_WIN32)
        WaitForSingleObject((HANDLE)ctx->bg_thread, INFINITE);
        CloseHandle((HANDLE)ctx->bg_thread);
#else
        pthread_join(*(pthread_t *)ctx->bg_thread, NULL);
        free(ctx->bg_thread);
#endif
        ctx->bg_thread = NULL;
        ctx->bg_thread_created = 0;
    }
    if (work->bg_kernel) {
        acuda_kernel_free((acuda_kernel_t *)work->bg_kernel);
        work->bg_kernel = NULL;
    }
    if (work->bg_buf) {
        acuda_free(work, (acuda_buffer_t)work->bg_buf);
        work->bg_buf = NULL;
    }
    if (work->bg_stream) {
        if (work->device)
            vkDeviceWaitIdle(work->device);
        acuda_stream_destroy(work, work->bg_stream);
        work->bg_stream = NULL;
    }
    if (work->bg_fence != VK_NULL_HANDLE) {
        vkDestroyFence(work->device, work->bg_fence, NULL);
        work->bg_fence = VK_NULL_HANDLE;
    }
    if (work->bg_cmd != VK_NULL_HANDLE && work->bg_pool != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(work->device, work->bg_pool, 1, &work->bg_cmd);
        work->bg_cmd = VK_NULL_HANDLE;
    }
    if (work->bg_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(work->device, work->bg_pool, NULL);
        work->bg_pool = VK_NULL_HANDLE;
    }
    if (work->bg_memory != VK_NULL_HANDLE) {
        vkFreeMemory(work->device, work->bg_memory, NULL);
        work->bg_memory = VK_NULL_HANDLE;
    }
    if (work->bg_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(work->device, work->bg_buffer, NULL);
        work->bg_buffer = VK_NULL_HANDLE;
    }
    // The isolated keep-busy context owns a second VkDevice.  Destroying
    // that device at process exit crashes inside the Vulkan driver, so leave
    // it for the OS to reclaim.  The validated main context is untouched.
    if (ctx->bg_ctx)
        ctx->bg_ctx = NULL;
}

void acuda_high_perf_shutdown(acuda_context_t ctx) {
    acuda_stop_high_perf(ctx);
}
