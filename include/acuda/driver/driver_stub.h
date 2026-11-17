#ifndef ACUDA_DRIVER_STUB_H
#define ACUDA_DRIVER_STUB_H

#include <stdio.h>
#include "acuda/driver.h"

#ifndef ACUDA_DEBUG_PRINT
#define ACUDA_DEBUG_PRINT(...) do { } while (0)
#endif

static CUresult compat_unimpl(const char *name) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? %s -> NOT_SUPPORTED (compat stub)\n", name);
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult compat_noop(const char *name) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? %s -> SUCCESS (compat no-op)\n", name);
    return CUDA_SUCCESS;
}

#define ACUDA_COMPAT_UNIMPL0(fn) \
    CUresult ACUDA_CUDAAPI fn(void) { return compat_unimpl(#fn); }
#define ACUDA_COMPAT_UNIMPL1(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a) { \
        (void)a; \
        return compat_unimpl(#fn); \
    }
#define ACUDA_COMPAT_UNIMPL2(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b) { \
        (void)a; (void)b; \
        return compat_unimpl(#fn); \
    }
#define ACUDA_COMPAT_UNIMPL3(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b, void *c) { \
        (void)a; (void)b; (void)c; \
        return compat_unimpl(#fn); \
    }
#define ACUDA_COMPAT_UNIMPL4(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b, void *c, void *d) { \
        (void)a; (void)b; (void)c; (void)d; \
        return compat_unimpl(#fn); \
    }
#define ACUDA_COMPAT_UNIMPL5(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b, void *c, void *d, void *e) { \
        (void)a; (void)b; (void)c; (void)d; (void)e; \
        return compat_unimpl(#fn); \
    }
#define ACUDA_COMPAT_UNIMPL6(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b, void *c, void *d, void *e, void *f) { \
        (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; \
        return compat_unimpl(#fn); \
    }
#define ACUDA_COMPAT_NOOP0(fn) \
    CUresult ACUDA_CUDAAPI fn(void) { return compat_noop(#fn); }
#define ACUDA_COMPAT_NOOP1(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a) { \
        (void)a; \
        return compat_noop(#fn); \
    }
#define ACUDA_COMPAT_NOOP2(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b) { \
        (void)a; (void)b; \
        return compat_noop(#fn); \
    }
#define ACUDA_COMPAT_NOOP3(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b, void *c) { \
        (void)a; (void)b; (void)c; \
        return compat_noop(#fn); \
    }
#define ACUDA_COMPAT_NOOP4(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b, void *c, void *d) { \
        (void)a; (void)b; (void)c; (void)d; \
        return compat_noop(#fn); \
    }

#endif

/* Declarations for compat stubs defined in driver_compat_stubs.c */
CUresult ACUDA_CUDAAPI cuMemcpy3DPeer(void *a1);
CUresult ACUDA_CUDAAPI cuMemcpy3DPeerAsync(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuMemAllocManaged(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuModuleLoad(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuModuleGetSurfRef(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuSurfRefSetArray(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuTexRefSetAddress2D_v3(void *a1, void *a2, void *a3, void *a4);
CUresult ACUDA_CUDAAPI cuTexRefSetMipmapFilterMode(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuTexRefSetMipmapLevelBias(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuTexRefSetMipmapLevelClamp(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuTexRefSetMipmappedArray(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuTexObjectGetResourceViewDesc(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuFuncSetCacheConfig(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuFuncSetSharedMemConfig(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuCtxEnablePeerAccess(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuCtxDisablePeerAccess(void *a1);
CUresult ACUDA_CUDAAPI cuProfilerInitialize(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuProfilerStart(void);
CUresult ACUDA_CUDAAPI cuProfilerStop(void);
CUresult ACUDA_CUDAAPI cuStreamAttachMemAsync(void *a1, void *a2, void *a3, void *a4);
CUresult ACUDA_CUDAAPI cuD3D10CtxCreateOnDevice(void *a1, void *a2, void *a3, void *a4);
CUresult ACUDA_CUDAAPI cuD3D10GetDevice(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuD3D10GetDevices(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuD3D10GetDirect3DDevice(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuD3D10MapResources(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuD3D10RegisterResource(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuD3D10UnmapResources(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuD3D10UnregisterResource(void *a1);
CUresult ACUDA_CUDAAPI cuD3D11CtxCreateOnDevice(void *a1, void *a2, void *a3, void *a4);
CUresult ACUDA_CUDAAPI cuD3D11GetDevice(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuD3D11GetDevices(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuD3D11GetDirect3DDevice(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuGLCtxCreate_v2(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuGLGetDevices(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuGLInit(void);
CUresult ACUDA_CUDAAPI cuGLMapBufferObject_v2(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuGLMapBufferObjectAsync_v2(void *a1, void *a2, void *a3, void *a4);
CUresult ACUDA_CUDAAPI cuGLRegisterBufferObject(void *a1);
CUresult ACUDA_CUDAAPI cuGLSetBufferObjectMapFlags(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuGLUnmapBufferObject(void *a1);
CUresult ACUDA_CUDAAPI cuGLUnmapBufferObjectAsync(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuGLUnregisterBufferObject(void *a1);
CUresult ACUDA_CUDAAPI cuGraphicsD3D10RegisterResource(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuGraphicsD3D11RegisterResource(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuGraphicsD3D9RegisterResource(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuGraphicsGLRegisterBuffer(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuGraphicsGLRegisterImage(void *a1, void *a2, void *a3, void *a4);
CUresult ACUDA_CUDAAPI cuGraphicsMapResources(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuGraphicsUnmapResources(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuGraphicsUnregisterResource(void *a1);
CUresult ACUDA_CUDAAPI cuGraphicsResourceGetMappedPointer_v2(void *a1, void *a2, void *a3);
CUresult ACUDA_CUDAAPI cuGraphicsResourceSetMapFlags_v2(void *a1, void *a2);
CUresult ACUDA_CUDAAPI cuGraphicsSubResourceGetMappedArray(void *a1, void *a2, void *a3, void *a4);
CUresult ACUDA_CUDAAPI cuGraphicsResourceGetMappedMipmappedArray(void *a1, void *a2, void *a3, void *a4);
CUresult ACUDA_CUDAAPI cuWGLGetDevice(void *a1, void *a2);
