#include "driver_internal.h"
#include "rtxdisam_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult ACUDA_CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, CUjit_option *options,
                            void **optionValues) {
    (void)numOptions;
    (void)options;
    (void)optionValues;
    if (getenv("ACUDA_TRACE") && image)
        fprintf(stderr, "acuda-trace .. cuModuleLoadDataEx magic=%08x\n",
                rd32le((const unsigned char *)image));
    if (!module)
        return CUDA_ERROR_INVALID_VALUE;
    *module = NULL;
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    if (!image)
        return CUDA_ERROR_INVALID_VALUE;
    const char *text = (const char *)image;
    const unsigned char *bytes = (const unsigned char *)image;
    // A __fatBinC_Wrapper_t is what cuModuleLoadData receives from old
    // PhysX/CUDA 4.x runtimes.  It points at the real 0xBA55ED50 blob.
    if (rd32le(bytes) == 0x466243B1u)
        return cuModuleLoadFatBinary(module, image);
    // A raw ELF cubin or fatbin blob has no PTX for ACuda to import. Ask the
    // optional rtxdisam native library to lift its SASS first.
    if ((bytes[0] == 0x7f && strncmp(text + 1, "ELF", 3) == 0) ||
        rd32le(bytes) == 0xBA55ED50u) {
        size_t size = acuda_image_size(bytes);
        if (!size)
            return CUDA_ERROR_INVALID_IMAGE;
        return load_sass_image(module, bytes, size);
    }
    if (!looks_like_ptx(text))
        return CUDA_ERROR_INVALID_IMAGE;
    size_t tlen = strlen(text);
    int ptx_oom = 0;
    if (tlen > ACUDA_PTX_EAGER_VALIDATE_MAX) {
        /* Large module: skip the eager whole-text parse. cuModuleGetFunction
         * extracts and compiles only the requested entry, which validates it
         * there; a full parse here just burns 32-bit address space and can
         * fail/crash before any function is ever needed. */
        ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleLoadDataEx: skip eager validate (%zu bytes)\n", tlen);
#if defined(_WIN32) && !defined(_WIN64)
        // The per-entry 32-bit parser is also likely to OOM on a module this
        // large; use the 64-bit compile host directly.
        ptx_oom = 1;
#endif
    } else {
        char err[1024];
        if (acuda_ptx_validate(text, err, sizeof(err)) != ACUDA_SUCCESS) {
            /* 32-bit large modules: full parse can OOM during validation; store
             * text and let cuModuleGetFunction/compile handle per-function. */
            if (!(err[0] && strstr(err, "out of memory")))
                return CUDA_ERROR_INVALID_IMAGE;
            ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleLoadDataEx: validate OOM, storing ptx text (%zu)\n", tlen);
#if defined(_WIN32) && !defined(_WIN64)
            // The same 32-bit parser will OOM again for every entry in this
            // module; skip straight to the 64-bit compile host.
            ptx_oom = 1;
#endif
        }
    }
    CUmodule m = calloc(1, sizeof(*m));
    if (!m)
        return CUDA_ERROR_OUT_OF_MEMORY;
    m->ctx = c;
    m->ptx_oom = ptx_oom;
    m->ptx = malloc(strlen(text) + 1);
    if (!m->ptx) {
        free(m);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    strcpy(m->ptx, text);
    *module = m;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuModuleLoadData(CUmodule *module, const void *image) {
    return cuModuleLoadDataEx(module, image, 0, NULL, NULL);
}

// ---- NVIDIA fatbin container parsing ----
//
// Core PTX member extraction/decompression (raw/LZ4/ZSTD) is shared via
// rtxdiasm-fatbin-parse (rtxdiasm_fatbin_extract_first_ptx) so acuda and
// cudart use identical fatbin decode logic. Driver-specific wrapper handling,
// validation/OOM acceptance, and SASS fallback remain here.

uint32_t rd32le(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

uint64_t rd64le(const unsigned char *p) {
    uint64_t lo = rd32le(p), hi = rd32le(p + 4);
    return lo | (hi << 32);
}

char *fatbin_first_ptx(const unsigned char *blob, size_t cap) {
    char *ptx = NULL;
    size_t ptx_len = 0;
    if (rtxdiasm_fatbin_extract_first_ptx(blob, cap, &ptx, &ptx_len) != RTXDIASM_FATBIN_OK)
        return NULL;
    /* Preserve driver behavior: accept large PTX without eager full validate,
     * and tolerate 32-bit validation OOM/internal failures (compile host revalidates). */
    if (ptx_len > ACUDA_PTX_EAGER_VALIDATE_MAX) {
        ACUDA_DEBUG_PRINT("acuda-trace .. extracted ptx large, accept without eager validate (%zu bytes)\n", ptx_len);
        return ptx;
    }
    {
        char err[256];
        err[0] = '\0';
        if (acuda_ptx_validate(ptx, err, sizeof(err)) == ACUDA_SUCCESS)
            return ptx;
        if (!err[0] || strstr(err, "out of memory")) {
            ACUDA_DEBUG_PRINT("acuda-trace .. extracted ptx validate OOM/internal, accepting as ptx (%zu bytes)\n", ptx_len);
            return ptx;
        }
        ACUDA_DEBUG_PRINT("acuda-trace .. extracted ptx validate failed: %s\n", err);
    }
    free(ptx);
    return NULL;
}

CUresult load_fatbin_member(CUmodule *module,
                                   const unsigned char *data, size_t size) {
    const char *dump = getenv("ACUDA_DUMP_FATBIN");
    if (dump && *dump) {
        // Numbered dumps: PhysX loads several fatbins per run and the
        // interesting one is rarely the last.
        static unsigned long dump_serial;
        char path[1024];
        size_t dl = strlen(dump);
        if (dl > 7 && strcmp(dump + dl - 7, ".fatbin") == 0)
            snprintf(path, sizeof(path), "%.*s_%lu.fatbin", (int)(dl - 7),
                     dump, ++dump_serial);
        else
            snprintf(path, sizeof(path), "%s_%lu", dump, ++dump_serial);
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(data, 1, size, f);
            fclose(f);
            ACUDA_DEBUG_PRINT("acuda-trace .. dumped fatbin %zu bytes\n", size);
        }
    }
    char *ptx = cubin_backend() == 2 ? NULL : fatbin_first_ptx(data, size);
    if (!ptx)
        return load_sass_image(module, data, size);
    CUresult r = cuModuleLoadData(module, ptx);
    free(ptx);
    if (r == CUDA_SUCCESS && module && *module)
        attach_sass_image(*module, data, size);
    return r;
}

CUresult ACUDA_CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleLoadFatBinary\n");
    if (!module)
        return CUDA_ERROR_INVALID_VALUE;
    *module = NULL;
    if (!fatCubin)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    // The argument is the __fatBinC_Wrapper_t control struct, not the
    // raw blob (verified field-by-field against real torch fatbins).
    const unsigned char *w = (const unsigned char *)fatCubin;
    if (rd32le(w) != 0x466243B1u)
        return CUDA_ERROR_INVALID_IMAGE;
    uint32_t ver = rd32le(w + 4);
    ACUDA_DEBUG_PRINT("acuda-trace .. fatbin wrapper ver=%u\n", ver);
    if (ver != 1 && ver != 2)
        return CUDA_ERROR_INVALID_IMAGE;
    if (ver == 1) {
        const unsigned char *data = NULL;
        memcpy(&data, w + 8, sizeof(data));
        if (!data)
            return CUDA_ERROR_INVALID_IMAGE;
        // Bound the walk by the blob's own declared size (+ header).
        if (rd32le(data) != 0xBA55ED50u)
            return CUDA_ERROR_INVALID_IMAGE;
        uint64_t files_size = rd64le(data + 8);
        if (files_size > (uint64_t)512 << 20)
            return CUDA_ERROR_INVALID_IMAGE;
        return load_fatbin_member(module, data,
                                  16 + (size_t)files_size);
    }
    // Version 2: NULL-terminated array of wrapper pointers (submodules).
    const void *const *arr = NULL;
    memcpy(&arr, w + 16, sizeof(arr));
    if (!arr)
        return CUDA_ERROR_INVALID_IMAGE;
    for (size_t i = 0; i < 4096; i++) {
        const unsigned char *sub = NULL;
        memcpy(&sub, &arr[i], sizeof(sub));
        if (!sub)
            break;
        if (rd32le(sub) != 0x466243B1u || rd32le(sub + 4) != 1)
            continue;
        const unsigned char *data = NULL;
        memcpy(&data, sub + 8, sizeof(data));
        if (!data || rd32le(data) != 0xBA55ED50u)
            continue;
        uint64_t files_size = rd64le(data + 8);
        if (files_size > (uint64_t)512 << 20)
            continue;
        CUresult r = load_fatbin_member(module, data,
                                        16 + (size_t)files_size);
        if (r == CUDA_SUCCESS)
            return r;
    }
    return CUDA_ERROR_INVALID_IMAGE;
}
