#include "acuda/driver/driver_stub.h"

ACUDA_COMPAT_UNIMPL1(cuMemcpy3DPeer)
ACUDA_COMPAT_UNIMPL2(cuMemcpy3DPeerAsync)
ACUDA_COMPAT_UNIMPL3(cuMemAllocManaged)
ACUDA_COMPAT_UNIMPL2(cuModuleLoad)
ACUDA_COMPAT_UNIMPL3(cuModuleGetSurfRef)
ACUDA_COMPAT_UNIMPL3(cuSurfRefSetArray)
ACUDA_COMPAT_UNIMPL4(cuTexRefSetAddress2D_v3)
ACUDA_COMPAT_NOOP2(cuTexRefSetMipmapFilterMode)
ACUDA_COMPAT_NOOP2(cuTexRefSetMipmapLevelBias)
ACUDA_COMPAT_NOOP3(cuTexRefSetMipmapLevelClamp)
ACUDA_COMPAT_NOOP3(cuTexRefSetMipmappedArray)
/* cuTexObjectGetResourceViewDesc implemented in driver_texture.c */
ACUDA_COMPAT_NOOP2(cuFuncSetCacheConfig)
ACUDA_COMPAT_NOOP2(cuFuncSetSharedMemConfig)
ACUDA_COMPAT_NOOP2(cuCtxEnablePeerAccess)
ACUDA_COMPAT_NOOP1(cuCtxDisablePeerAccess)
ACUDA_COMPAT_NOOP3(cuProfilerInitialize)
ACUDA_COMPAT_NOOP0(cuProfilerStart)
ACUDA_COMPAT_NOOP0(cuProfilerStop)
ACUDA_COMPAT_NOOP4(cuStreamAttachMemAsync)
ACUDA_COMPAT_UNIMPL4(cuD3D10CtxCreateOnDevice)
ACUDA_COMPAT_UNIMPL2(cuD3D10GetDevice)
ACUDA_COMPAT_UNIMPL3(cuD3D10GetDevices)
ACUDA_COMPAT_UNIMPL2(cuD3D10GetDirect3DDevice)
ACUDA_COMPAT_UNIMPL2(cuD3D10MapResources)
ACUDA_COMPAT_UNIMPL3(cuD3D10RegisterResource)
ACUDA_COMPAT_UNIMPL2(cuD3D10UnmapResources)
ACUDA_COMPAT_UNIMPL1(cuD3D10UnregisterResource)
ACUDA_COMPAT_UNIMPL4(cuD3D11CtxCreateOnDevice)
ACUDA_COMPAT_UNIMPL2(cuD3D11GetDevice)
ACUDA_COMPAT_UNIMPL3(cuD3D11GetDevices)
ACUDA_COMPAT_UNIMPL2(cuD3D11GetDirect3DDevice)
ACUDA_COMPAT_UNIMPL3(cuGLCtxCreate_v2)
ACUDA_COMPAT_UNIMPL3(cuGLGetDevices)
ACUDA_COMPAT_UNIMPL0(cuGLInit)
ACUDA_COMPAT_UNIMPL3(cuGLMapBufferObject_v2)
ACUDA_COMPAT_UNIMPL4(cuGLMapBufferObjectAsync_v2)
ACUDA_COMPAT_UNIMPL1(cuGLRegisterBufferObject)
ACUDA_COMPAT_UNIMPL2(cuGLSetBufferObjectMapFlags)
ACUDA_COMPAT_UNIMPL1(cuGLUnmapBufferObject)
ACUDA_COMPAT_UNIMPL2(cuGLUnmapBufferObjectAsync)
ACUDA_COMPAT_UNIMPL1(cuGLUnregisterBufferObject)
ACUDA_COMPAT_UNIMPL3(cuGraphicsD3D10RegisterResource)
ACUDA_COMPAT_UNIMPL3(cuGraphicsD3D11RegisterResource)
ACUDA_COMPAT_UNIMPL3(cuGraphicsD3D9RegisterResource)
ACUDA_COMPAT_UNIMPL3(cuGraphicsGLRegisterBuffer)
ACUDA_COMPAT_UNIMPL4(cuGraphicsGLRegisterImage)
/* cuGraphics map/unmap/query/unregister implemented in driver_graphics.c */
ACUDA_COMPAT_UNIMPL2(cuWGLGetDevice)

/* IPC handles: 64-byte structs passed by value, like the real driver. */
CUresult ACUDA_CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr) {
    (void)dptr;
    return compat_unimpl("cuIpcCloseMemHandle");
}

CUresult ACUDA_CUDAAPI cuIpcGetEventHandle(CUipcEventHandle *handle,
                                           CUevent event) {
    (void)handle;
    (void)event;
    return compat_unimpl("cuIpcGetEventHandle");
}

CUresult ACUDA_CUDAAPI cuIpcGetMemHandle(CUipcMemHandle *handle,
                                        CUdeviceptr dptr) {
    (void)handle;
    (void)dptr;
    return compat_unimpl("cuIpcGetMemHandle");
}

CUresult ACUDA_CUDAAPI cuIpcOpenEventHandle(CUevent *event,
                                           CUipcEventHandle handle) {
    (void)event;
    (void)handle;
    return compat_unimpl("cuIpcOpenEventHandle");
}

CUresult ACUDA_CUDAAPI cuIpcOpenMemHandle(CUdeviceptr *dptr,
                                         CUipcMemHandle handle,
                                         unsigned int flags) {
    (void)dptr;
    (void)handle;
    (void)flags;
    return compat_unimpl("cuIpcOpenMemHandle");
}
