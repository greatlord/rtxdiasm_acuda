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
