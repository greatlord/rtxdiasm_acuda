#include "acuda/driver/driver_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// PTX-passthrough linker: accepts PTX inputs, concatenates them, and
// hands the text to cuModuleLoadData as the "cubin". No SASS involved.
CUresult ACUDA_CUDAAPI cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options,
                         void **optionValues, CUlinkState *stateOut) {
    (void)numOptions;
    (void)options;
    (void)optionValues;
    if (!stateOut)
        return CUDA_ERROR_INVALID_VALUE;
    *stateOut = NULL;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    CUlinkState s = calloc(1, sizeof(*s));
    if (!s)
        return CUDA_ERROR_OUT_OF_MEMORY;
    *stateOut = s;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                      void **optionValues, CUlinkState *stateOut) {
    return cuLinkCreate_v2(numOptions, options, optionValues, stateOut);
}

CUresult ACUDA_CUDAAPI cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data,
                          size_t size, const char *name,
                          unsigned int numOptions, CUjit_option *options,
                          void **optionValues) {
    (void)name;
    (void)numOptions;
    (void)options;
    (void)optionValues;
    if (!state)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!data)
        return CUDA_ERROR_INVALID_VALUE;
    if (type != CU_JIT_INPUT_PTX)
        return CUDA_ERROR_INVALID_IMAGE; // cubin/fatbin/object unsupported
    char *copy = malloc(size + 1);
    if (!copy)
        return CUDA_ERROR_OUT_OF_MEMORY;
    memcpy(copy, data, size);
    copy[size] = '\0';
    char err[1024];
    if (acuda_ptx_validate(copy, err, sizeof(err)) != ACUDA_SUCCESS) {
        free(copy);
        return CUDA_ERROR_INVALID_IMAGE;
    }
    if (state->n == state->cap) {
        size_t ncap = state->cap ? state->cap * 2 : 4;
        char **nb = malloc(sizeof(*nb) * ncap);
        size_t *ns = malloc(sizeof(*ns) * ncap);
        if (!nb || !ns) {
            free(nb);
            free(ns);
            free(copy);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        memcpy(nb, state->blobs, sizeof(*nb) * state->n);
        memcpy(ns, state->sizes, sizeof(*ns) * state->n);
        free(state->blobs);
        free(state->sizes);
        state->blobs = nb;
        state->sizes = ns;
        state->cap = ncap;
    }
    state->blobs[state->n] = copy;
    state->sizes[state->n] = size;
    state->n++;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type, void *data,
                       size_t size, const char *name,
                       unsigned int numOptions, CUjit_option *options,
                       void **optionValues) {
    return cuLinkAddData_v2(state, type, data, size, name, numOptions, options,
                            optionValues);
}

CUresult ACUDA_CUDAAPI cuLinkAddFile_v2(CUlinkState state, CUjitInputType type,
                          const char *path, unsigned int numOptions,
                          CUjit_option *options, void **optionValues) {
    if (!state)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!path)
        return CUDA_ERROR_INVALID_VALUE;
    FILE *f = fopen(path, "rb");
    if (!f)
        return CUDA_ERROR_INVALID_IMAGE; // unreadable input, documented
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    CUresult r = CUDA_ERROR_INVALID_IMAGE;
    if (n >= 0) {
        char *buf = malloc((size_t)n ? (size_t)n : 1);
        if (buf && (n == 0 || fread(buf, 1, (size_t)n, f) == (size_t)n))
            r = cuLinkAddData_v2(state, type, buf, (size_t)n, path,
                                 numOptions, options, optionValues);
        free(buf);
    }
    fclose(f);
    return r;
}

CUresult ACUDA_CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                       const char *path, unsigned int numOptions,
                       CUjit_option *options, void **optionValues) {
    return cuLinkAddFile_v2(state, type, path, numOptions, options,
                            optionValues);
}

CUresult ACUDA_CUDAAPI cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) {
    if (!state)
        return CUDA_ERROR_INVALID_HANDLE;
    if (!cubinOut || !sizeOut)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (state->n == 0)
        return CUDA_ERROR_INVALID_IMAGE;
    size_t total = 0;
    for (size_t i = 0; i < state->n; i++)
        total += state->sizes[i] + 1;
    char *out = malloc(total + 1);
    if (!out)
        return CUDA_ERROR_OUT_OF_MEMORY;
    size_t pos = 0;
    for (size_t i = 0; i < state->n; i++) {
        memcpy(out + pos, state->blobs[i], state->sizes[i]);
        pos += state->sizes[i];
        out[pos++] = '\n';
    }
    out[pos] = '\0';
    free(state->linked);
    state->linked = out;
    state->linked_size = pos;
    *cubinOut = out;
    *sizeOut = pos;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuLinkDestroy(CUlinkState state) {
    if (!state)
        return CUDA_ERROR_INVALID_HANDLE;
    for (size_t i = 0; i < state->n; i++)
        free(state->blobs[i]);
    free(state->blobs);
    free(state->sizes);
    free(state->linked);
    free(state);
    return CUDA_SUCCESS;
}
