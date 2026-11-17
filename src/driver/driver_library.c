#include "driver_internal.h"
#include "rtxdisam_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult ACUDA_CUDAAPI cuModuleGetLoadingMode(int *mode) {
    if (!mode)
        return CUDA_ERROR_INVALID_VALUE;
    *mode = 0x1; /* CU_MODULE_EAGER_LOADING */
    return CUDA_SUCCESS;
}
/* CUDA 12/13 library API.  A CUlibrary is a CUmodule and a CUkernel is a
 * CUfunction; the module loader already knows how to lift SASS-only
 * cubins/fatbins through the optional rtxdisam bridge. */
CUresult ACUDA_CUDAAPI cuLibraryLoadData(void **library, const void *code, void *jitOptions,
                           void **jitOptionsValues, unsigned int numJitOptions,
                           void *libraryOptions, void **libraryOptionValues,
                           unsigned int numLibraryOptions) {
    (void)jitOptions; (void)jitOptionsValues; (void)numJitOptions;
    (void)libraryOptions; (void)libraryOptionValues; (void)numLibraryOptions;
    if (!library || !code)
        return CUDA_ERROR_INVALID_VALUE;
    *library = NULL;
    const unsigned char *b = (const unsigned char *)code;
    ACUDA_DEBUG_PRINT("acuda-lib magic=%08x v=%08x\n",
                rd32le(b), rd32le(b + 4));
    const unsigned char *image = b;
    // Library blobs are ELF cubins, raw fatbins, or (rarely) the
    // __fatBinC_Wrapper_t control structure.  Store the image without
    // lifting it: cuLibraryGetKernel() lifts only the requested entry.
    if (rd32le(b) == 0x466243B1u) {
        uint32_t ver = rd32le(b + 4);
        const unsigned char *data = NULL;
        if (ver == 1) {
            memcpy(&data, b + 8, sizeof(data));
        } else if (ver == 2) {
            const void *const *arr = NULL;
            memcpy(&arr, b + 16, sizeof(arr));
            for (size_t i = 0; arr && i < 4096; i++) {
                const unsigned char *sub = NULL;
                memcpy(&sub, &arr[i], sizeof(sub));
                if (!sub)
                    break;
                ACUDA_DEBUG_PRINT("acuda-lib v2 sub[%zu]=%p magic=%08x "
                            "v=%08x\n", i, (const void *)sub,
                            rd32le(sub), rd32le(sub + 4));
                if (rd32le(sub) == 0xBA55ED50u) {
                    data = sub;
                    break;
                }
                if (rd32le(sub) == 0x466243B1u && rd32le(sub + 4) == 1) {
                    memcpy(&data, sub + 8, sizeof(data));
                    break;
                }
            }
        }
        if (!data || rd32le(data) != 0xBA55ED50u)
            return CUDA_ERROR_INVALID_IMAGE;
        image = data;
    }
    if (!(image[0] == 0x7f && image[1] == 'E' && image[2] == 'L' &&
          image[3] == 'F') &&
        rd32le(image) != 0xBA55ED50u)
        return cuModuleLoadData((CUmodule *)library, code);
    size_t size = acuda_image_size(image);
    ACUDA_DEBUG_PRINT("acuda-lib image size=%zu\n", size);
    if (!size)
        return CUDA_ERROR_INVALID_IMAGE;
    // Library loads stay lazy: cuLibraryGetKernel lifts only the entry
    // that is actually used.  backend=1 (C parser only) is the exception:
    // it has to import a PTX member now, and has nothing for SASS-only
    // containers.
    if (rd32le(image) == 0xBA55ED50u && cubin_backend() == 1) {
        char *ptx = fatbin_first_ptx(image, size);
        if (!ptx)
            return CUDA_ERROR_INVALID_IMAGE;
        CUresult r = cuModuleLoadData((CUmodule *)library, ptx);
        free(ptx);
        if (r == CUDA_SUCCESS && *(CUmodule *)library)
            attach_sass_image(*(CUmodule *)library, image, size);
        return r;
    }
    return module_from_sass_image((CUmodule *)library, image, size);
}
CUresult ACUDA_CUDAAPI cuLibraryLoadFromFile(void **library, const char *fileName,
                               void *jitOptions, void **jitOptionsValues,
                               unsigned int numJitOptions, void *libraryOptions,
                               void **libraryOptionValues,
                               unsigned int numLibraryOptions) {
    if (!library || !fileName)
        return CUDA_ERROR_INVALID_VALUE;
    FILE *f = fopen(fileName, "rb");
    if (!f)
        return CUDA_ERROR_NOT_FOUND;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return CUDA_ERROR_INVALID_VALUE;
    }
    long len = ftell(f);
    if (len <= 0 || len > (long)(512u << 20)) {
        fclose(f);
        return CUDA_ERROR_INVALID_VALUE;
    }
    rewind(f);
    void *data = malloc((size_t)len);
    if (!data) {
        fclose(f);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (fread(data, 1, (size_t)len, f) != (size_t)len) {
        free(data);
        fclose(f);
        return CUDA_ERROR_INVALID_VALUE;
    }
    fclose(f);
    CUresult r = cuLibraryLoadData(library, data, jitOptions,
                                   jitOptionsValues, numJitOptions,
                                   libraryOptions, libraryOptionValues,
                                   numLibraryOptions);
    free(data);
    return r;
}
CUresult ACUDA_CUDAAPI cuLibraryUnload(void *library) {
    if (!library)
        return CUDA_ERROR_INVALID_HANDLE;
    return cuModuleUnload((CUmodule)library);
}
CUresult ACUDA_CUDAAPI cuLibraryGetModule(void **pMod, void *library) {
    if (!pMod)
        return CUDA_ERROR_INVALID_VALUE;
    *pMod = library;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuLibraryGetKernel(void **pKernel, void *library, const char *name) {
    if (!pKernel)
        return CUDA_ERROR_INVALID_VALUE;
    *pKernel = NULL;
    if (!library || !name)
        return CUDA_ERROR_INVALID_HANDLE;
    // cudart enumerates every kernel while bringing up a library, but
    // only asks for a launchable CUfunction when that kernel is used.
    // Defer the (expensive) SASS lift until then.
    struct CUkernel_st *k = calloc(1, sizeof(*k));
    if (!k)
        return CUDA_ERROR_OUT_OF_MEMORY;
    k->magic = ACUDA_HANDLE_KERNEL;
    k->mod = (CUmodule)library;
    snprintf(k->name, sizeof(k->name), "%s", name);
    *pKernel = (void *)k;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuKernelGetFunction(void **pFunc, void *kernel) {
    if (!pFunc)
        return CUDA_ERROR_INVALID_VALUE;
    struct CUkernel_st *k = (struct CUkernel_st *)kernel;
    if (!k)
        return CUDA_ERROR_INVALID_HANDLE;
    if (!k->func) {
        CUfunction f = calloc(1, sizeof(*f));
        if (!f)
            return CUDA_ERROR_OUT_OF_MEMORY;
        f->magic = ACUDA_HANDLE_FUNC;
        f->mod = k->mod;
        snprintf(f->name, sizeof(f->name), "%s", k->name);
        k->func = f;
        ACUDA_DEBUG_PRINT("acuda-kfunc f=%p mod=%p k=%p name=%s\n",
                    (void *)f, (void *)f->mod, (void *)f->k, f->name);
    }
    *pFunc = (void *)k->func;
    return CUDA_SUCCESS;
}
// CUDA 13 places the attribute before the kernel handle:
//   cuKernelGetAttribute(int *pi, CUfunction_attribute attrib,
//                        CUkernel kernel, CUdevice dev)
CUresult ACUDA_CUDAAPI cuKernelGetAttribute(int *pi, int attrib, void *kernel, CUdevice dev) {
    (void)dev;
    if (!pi)
        return CUDA_ERROR_INVALID_VALUE;
    if (!kernel)
        return CUDA_ERROR_INVALID_HANDLE;
    // Accept either the CUkernel or the CUfunction returned by
    // cuKernelGetFunction.
    CUfunction f = NULL;
    CUresult r = handle_as_func(kernel, &f);
    if (r != CUDA_SUCCESS)
        return r;
    return cuFuncGetAttribute(pi, attrib, f);
}
CUresult ACUDA_CUDAAPI cuKernelGetLibrary(void **pLib, void *kernel) {
    if (!pLib)
        return CUDA_ERROR_INVALID_VALUE;
    *pLib = NULL;
    struct CUkernel_st *k = (struct CUkernel_st *)kernel;
    if (!k)
        return CUDA_ERROR_INVALID_HANDLE;
    *pLib = (void *)k->mod;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuKernelSetAttribute(int attrib, int val, void *kernel, CUdevice dev) {
    (void)dev;
    struct CUkernel_st *k = (struct CUkernel_st *)kernel;
    if (!k)
        return CUDA_ERROR_INVALID_HANDLE;
    if (!k->func)
        return CUDA_SUCCESS;
    return cuFuncSetAttribute(k->func, attrib, val);
}
CUresult ACUDA_CUDAAPI cuKernelSetCacheConfig(void *kernel, int config, CUdevice dev) {
    (void)kernel; (void)config; (void)dev;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuKernelGetName(const char **name, void *kernel) {
    if (!name)
        return CUDA_ERROR_INVALID_VALUE;
    struct CUkernel_st *k = (struct CUkernel_st *)kernel;
    if (!k)
        return CUDA_ERROR_INVALID_HANDLE;
    *name = k->name;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuKernelGetParamInfo(void *kernel, size_t paramIndex,
                              size_t *paramOffset, size_t *paramSize) {
    (void)kernel; (void)paramIndex;
    if (paramOffset) *paramOffset = 0;
    if (paramSize) *paramSize = 0;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuKernelGetParamCount(void *kernel, size_t *count) {
    (void)kernel;
    if (count) *count = 0;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                            void *library, const char *name) {
    (void)library; (void)name;
    if (dptr) *dptr = 0;
    if (bytes) *bytes = 0;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes,
                             void *library, const char *name) {
    return cuLibraryGetGlobal(dptr, bytes, library, name);
}
CUresult ACUDA_CUDAAPI cuLibraryGetUnifiedFunction(void **fptr, void *library,
                                     const char *symbol) {
    (void)library; (void)symbol;
    if (fptr) *fptr = NULL;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuLibraryGetKernelCount(unsigned int *count, void *library) {
    (void)library;
    if (count) *count = 0;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuLibraryEnumerateKernels(void **kernels, unsigned int numKernels,
                                   void *library) {
    (void)kernels; (void)numKernels; (void)library;
    return CUDA_SUCCESS;
}





CUresult ACUDA_CUDAAPI cuModuleUnload(CUmodule module) {
    if (!module)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    for (size_t i = 0; i < module->nfuncs; i++)
        acuda_kernel_free(module->funcs[i].k);
    for (size_t i = 0; i < module->ntexrefs; i++) {
        struct CUtexref_st *tr = module->texrefs[i];
        if (tr) {
            if (tr->obj)
                cuTexObjectDestroy(tr->obj);
            free(tr);
        }
    }
    for (size_t i = 0; i < module->nglobals; i++)
        if (module->globals[i].dptr)
            cuMemFree_v2(module->globals[i].dptr);
    free(module->globals);
    free(module->texrefs);
    free(module->funcs);
    free(module->ptx);
    free(module->image);
    free(module);
    return CUDA_SUCCESS;
}
