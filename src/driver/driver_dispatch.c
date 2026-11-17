#include "driver_internal.h"
#include "driver_stub.h"
#include "acuda/driver.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
/* extra decls added as needed */

    ENTRY(cuMemsetD8)
    ENTRY(cuModuleLoadDataEx)
    ENTRY(cuModuleLoadData)
    ENTRY(cuModuleLoadFatBinary)
    ENTRY(cuModuleUnload)
    ENTRY(cuModuleGetFunction)
    ENTRY(cuFuncGetAttribute)
    ENTRY(cuFuncSetAttribute)
    ENTRY(cuLaunchKernel)
    ENTRY(cuLaunchKernelEx)
    ENTRY(cuStreamCreate)
    ENTRY(cuStreamDestroy_v2)
    ENTRY(cuStreamDestroy)
    ENTRY(cuStreamSynchronize)
    ENTRY(cuEventCreate)
    ENTRY(cuEventDestroy_v2)
    ENTRY(cuEventDestroy)
    ENTRY(cuEventRecord)
    ENTRY(cuEventSynchronize)
    ENTRY(cuEventQuery)
    ENTRY(cuEventElapsedTime)
    ENTRY(cuLinkCreate_v2)
    ENTRY(cuLinkCreate)
    ENTRY(cuLinkAddData_v2)
    ENTRY(cuLinkAddData)
    ENTRY(cuLinkAddFile_v2)
    ENTRY(cuLinkAddFile)
    ENTRY(cuLinkComplete)
    ENTRY(cuLinkDestroy)
    ENTRY(cuGetErrorString)
    ENTRY(cuGetErrorName)
    ENTRY(cuGetProcAddress)
    ENTRY(cuGetProcAddress_v2)
    ENTRY(cuDeviceGetUuid)
    ENTRY(cuDeviceGetPCIBusId)
    ENTRY(cuDeviceGetByPCIBusId)
    ENTRY(cuArrayCreate)
    ENTRY(cuArrayCreate_v2)
    ENTRY(cuArray3DCreate)
    ENTRY(cuArray3DCreate_v2)
    ENTRY(cuArrayDestroy)
    ENTRY(cuArrayGetDescriptor)
    ENTRY(cuArray3DGetDescriptor)
    ENTRY(cuMipmappedArrayCreate)
    ENTRY(cuMipmappedArrayDestroy)
    ENTRY(cuMipmappedArrayGetLevel)
    ENTRY(cuModuleGetTexRef)
    ENTRY(cuTexRefSetAddress_v2)
    ENTRY(cuTexRefSetAddress)
    ENTRY(cuTexRefSetArray)
    ENTRY(cuTexRefSetFormat)
    ENTRY(cuTexRefSetFlags)
    ENTRY(cuTexRefSetFilterMode)
    ENTRY(cuTexRefSetAddressMode)
    ENTRY(cuTexRefSetMaxAnisotropy)
    ENTRY(cuTexRefGetAddress_v2)
    ENTRY(cuTexRefGetAddress)
    ENTRY(cuTexRefGetArray)
    ENTRY(cuTexRefGetFormat)
    ENTRY(cuTexRefGetFlags)
    ENTRY(cuTexRefGetFilterMode)
    ENTRY(cuTexRefGetAddressMode)
    ENTRY(cuTexRefGetMaxAnisotropy)
    ENTRY(cuTexObjectCreate)
    ENTRY(cuTexObjectDestroy)
    ENTRY(cuTexObjectGetResourceDesc)
    ENTRY(cuTexObjectGetTextureDesc)
    ENTRY(cuSurfObjectCreate)
    ENTRY(cuSurfObjectDestroy)
    ENTRY(cuSurfObjectGetResourceDesc)
    ENTRY(cuMemcpy2D)
    ENTRY(cuMemcpy2DAsync)
    ENTRY(cuMemcpy3D)
    ENTRY(cuMemcpy3DAsync)
    ENTRY(cuDeviceGetTexture1DLinearMaxWidth)
    ENTRY(cuDeviceGetP2PAttribute)
    ENTRY(cuDeviceCanAccessPeer)
    ENTRY(cuFlushGPUDirectRDMAWrites)
    ENTRY(cuDevicePrimaryCtxRetain)
    ENTRY(cuDevicePrimaryCtxRelease)
    ENTRY(cuDevicePrimaryCtxReset)
    ENTRY(cuDevicePrimaryCtxSetFlags)
    ENTRY(cuDevicePrimaryCtxGetState)
    ENTRY(cuCtxGetDevice_v2)
    ENTRY(cuCtxSynchronize_v2)
    ENTRY(cuDeviceTotalMem_v2)
    ENTRY(cuDeviceGetUuid_v2)
    ENTRY(cuDevicePrimaryCtxRelease_v2)
    ENTRY(cuDevicePrimaryCtxReset_v2)
    ENTRY(cuDevicePrimaryCtxSetFlags_v2)
    ENTRY(cuCtxGetCurrent)
    ENTRY(cuCtxSetCurrent)
    ENTRY(cuCtxGetDevice)
    ENTRY(cuCtxGetApiVersion)
    ENTRY(cuCtxGetLimit)
    ENTRY(cuCtxSetLimit)
    ENTRY(cuCtxSetCacheConfig)
    ENTRY(cuCtxGetCacheConfig)
    ENTRY(cuCtxDetach)
    ENTRY(cuMemGetInfo)
    ENTRY(cuDeviceGetDefaultMemPool)
    ENTRY(cuDeviceGetMemPool)
    ENTRY(cuDeviceSetMemPool)
    ENTRY(cuMemPoolTrimTo)
    ENTRY(cuStreamQuery)
    ENTRY(cuStreamWaitEvent)
    ENTRY(cuStreamGetFlags)
    ENTRY(cuStreamGetPriority)
    ENTRY(cuStreamGetCtx)
    ENTRY(cuStreamGetCtx_v2)
    ENTRY(cuStreamGetDevice)
    ENTRY(cuStreamGetId)
    ENTRY(cuPointerGetAttribute)
    ENTRY(cuPointerGetAttributes)
    ENTRY(cuMemHostGetDevicePointer)
    ENTRY(cuMemHostGetFlags)
    ENTRY(cuMemHostAlloc)
    ENTRY(cuMemHostAlloc_v2)
    ENTRY(cuMemFreeHost)
    ENTRY(cuMemHostRegister)
    ENTRY(cuMemHostRegister_v2)
    ENTRY(cuMemHostUnregister)
    ENTRY(cuModuleGetLoadingMode)
    ENTRY(cuMemAddressReserve)
    ENTRY(cuMemAddressFree)
    ENTRY(cuMemCreate)
    ENTRY(cuMemRelease)
    ENTRY(cuMemMap)
    ENTRY(cuMemUnmap)
    ENTRY(cuMemSetAccess)
    ENTRY(cuMemGetAllocationGranularity)
    ENTRY(cuMemGetAllocationPropertiesFromHandle)
    ENTRY(cuMemExportToShareableHandle)
    ENTRY(cuMemImportFromShareableHandle)
    ENTRY(cuMemRetainAllocationHandle)
    ENTRY(cuMemsetD32Async)
    ENTRY(cuMemsetD8Async)
    ENTRY(cuMemcpyHtoDAsync)
    ENTRY(cuMemcpyDtoHAsync)
    ENTRY(cuMemcpyDtoDAsync)
    ENTRY(cuMemcpyHtoDAsync_v2)
    ENTRY(cuMemcpyDtoHAsync_v2)
    ENTRY(cuMemcpyDtoDAsync_v2)
    ENTRY(cuMemcpyAsync)
    ENTRY(cuStreamWriteValue32)
    ENTRY(cuStreamWaitValue32)
    ENTRY(cuStreamWriteValue64)
    ENTRY(cuStreamWaitValue64)
    ENTRY(cuStreamBatchMemOp)
    ENTRY(cuCtxGetStreamPriorityRange)
    ENTRY(cuStreamIsCapturing)
    ENTRY(cuStreamBeginCapture)
    ENTRY(cuStreamEndCapture)
    ENTRY(cuThreadExchangeStreamCaptureMode)
    ENTRY(cuStreamGetCaptureInfo)
    ENTRY(cuStreamGetCaptureInfo_v2)
    ENTRY(cuStreamGetCaptureInfo_v3)
    ENTRY(cuMemGetAddressRange)
    ENTRY(cuMemGetAddressRange_v2)
    ENTRY(cuKernelGetFunction)
    ENTRY(cuKernelGetLibrary)
    ENTRY(cuKernelGetAttribute)
    ENTRY(cuKernelSetAttribute)
    ENTRY(cuKernelSetCacheConfig)
    ENTRY(cuKernelGetName)
    ENTRY(cuKernelGetParamInfo)
    ENTRY(cuKernelGetParamCount)
    ENTRY(cuLibraryLoadData)
    ENTRY(cuLibraryLoadFromFile)
    ENTRY(cuLibraryUnload)
    ENTRY(cuLibraryGetModule)
    ENTRY(cuLibraryGetKernel)
    ENTRY(cuLibraryGetGlobal)
    ENTRY(cuLibraryGetManaged)
    ENTRY(cuLibraryGetUnifiedFunction)
    ENTRY(cuLibraryGetKernelCount)
    ENTRY(cuLibraryEnumerateKernels)
    ENTRY(cuModuleGetGlobal)
    ENTRY(cuModuleGetGlobal_v2)
    ENTRY(cuOccupancyMaxActiveBlocksPerMultiprocessor)
    ENTRY(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
    ENTRY(cuOccupancyMaxPotentialBlockSize)
    ENTRY(cuOccupancyMaxPotentialBlockSizeWithFlags)
    ENTRY(acuda_stub_cuModuleGetGlobal)
    ENTRY(acuda_stub_cuModuleGetGlobal_v2)
    ENTRY(acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessor)
    ENTRY(acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
    ENTRY(acuda_stub_cuOccupancyMaxPotentialBlockSize)
    ENTRY(acuda_stub_cuOccupancyMaxPotentialBlockSizeWithFlags)
    ENTRY(cuGetExportTable)
    ENTRY(acuda_stub_cuTensorMapEncodeTiled)
    /* CUDA 6.x-era names the runtime binds by handle (see above). */
    ENTRY(cuMemsetD16_v2)
    ENTRY(cuMemsetD16)
    ENTRY(cuMemsetD16Async)
    ENTRY(cuMemsetD2D8_v2)
    ENTRY(cuMemsetD2D8)
    ENTRY(cuMemsetD2D8Async)
    ENTRY(cuMemsetD2D16_v2)
    ENTRY(cuMemsetD2D16)
    ENTRY(cuMemsetD2D16Async)
    ENTRY(cuMemsetD2D32_v2)
    ENTRY(cuMemsetD2D32)
    ENTRY(cuMemsetD2D32Async)
    ENTRY(cuMemcpy2D_v2)
    ENTRY(cuMemcpy2DAsync_v2)
    ENTRY(cuMemcpy2DUnaligned)
    ENTRY(cuMemcpy2DUnaligned_v2)
    ENTRY(cuMemcpy3D_v2)
    ENTRY(cuMemcpy3DAsync_v2)
    ENTRY(cuMemcpyPeer)
    ENTRY(cuMemcpyPeerAsync)
    ENTRY(cuMemcpy3DPeer)
    ENTRY(cuMemcpy3DPeerAsync)
    ENTRY(cuMemGetInfo_v2)
    ENTRY(cuMemHostGetDevicePointer_v2)
    ENTRY(cuMemAllocPitch_v2)
    ENTRY(cuMemAllocManaged)
    ENTRY(cuArrayGetDescriptor_v2)
    ENTRY(cuArray3DGetDescriptor_v2)
    ENTRY(cuLinkAddData)
    ENTRY(cuLinkAddFile)
    ENTRY(cuModuleLoad)
    ENTRY(cuModuleGetSurfRef)
    ENTRY(cuSurfRefSetArray)
    ENTRY(cuTexRefCreate)
    ENTRY(cuTexRefDestroy)
    ENTRY(cuTexRefSetAddress2D_v3)
    ENTRY(cuTexRefSetMipmapFilterMode)
    ENTRY(cuTexRefSetMipmapLevelBias)
    ENTRY(cuTexRefSetMipmapLevelClamp)
    ENTRY(cuTexRefSetMipmappedArray)
    ENTRY(cuTexObjectGetResourceViewDesc)
    ENTRY(cuFuncSetCacheConfig)
    ENTRY(cuFuncSetSharedMemConfig)
    ENTRY(cuCtxEnablePeerAccess)
    ENTRY(cuCtxDisablePeerAccess)
    ENTRY(cuProfilerInitialize)
    ENTRY(cuProfilerStart)
    ENTRY(cuProfilerStop)
    ENTRY(cuStreamCreateWithPriority)
    ENTRY(cuStreamAddCallback)
    ENTRY(cuStreamAttachMemAsync)
    ENTRY(cuD3D10CtxCreateOnDevice)
    ENTRY(cuD3D10GetDevice)
    ENTRY(cuD3D10GetDevices)
    ENTRY(cuD3D10GetDirect3DDevice)
    ENTRY(cuD3D10MapResources)
    ENTRY(cuD3D10RegisterResource)
    ENTRY(cuD3D10UnmapResources)
    ENTRY(cuD3D10UnregisterResource)
    ENTRY(cuD3D11CtxCreateOnDevice)
    ENTRY(cuD3D11GetDevice)
    ENTRY(cuD3D11GetDevices)
    ENTRY(cuD3D11GetDirect3DDevice)
    ENTRY(cuGLCtxCreate_v2)
    ENTRY(cuGLGetDevices)
    ENTRY(cuGLInit)
    ENTRY(cuGLMapBufferObject_v2)
    ENTRY(cuGLMapBufferObjectAsync_v2)
    ENTRY(cuGLRegisterBufferObject)
    ENTRY(cuGLSetBufferObjectMapFlags)
    ENTRY(cuGLUnmapBufferObject)
    ENTRY(cuGLUnmapBufferObjectAsync)
    ENTRY(cuGLUnregisterBufferObject)
    ENTRY(cuGraphicsD3D10RegisterResource)
    ENTRY(cuGraphicsD3D11RegisterResource)
    ENTRY(cuGraphicsD3D9RegisterResource)
    ENTRY(cuGraphicsGLRegisterBuffer)
    ENTRY(cuGraphicsGLRegisterImage)
    ENTRY(cuGraphicsMapResources)
    ENTRY(cuGraphicsUnmapResources)
    ENTRY(cuGraphicsUnregisterResource)
    ENTRY(cuGraphicsResourceGetMappedPointer_v2)
    ENTRY(cuGraphicsResourceSetMapFlags_v2)
    ENTRY(cuGraphicsSubResourceGetMappedArray)
    ENTRY(cuGraphicsResourceGetMappedMipmappedArray)
    ENTRY(cuWGLGetDevice)
    ENTRY(cuIpcCloseMemHandle)
    ENTRY(cuIpcGetEventHandle)
    ENTRY(cuIpcGetMemHandle)
    ENTRY(cuIpcOpenEventHandle)
    ENTRY(cuIpcOpenMemHandle)
#undef ENTRY
    return CUDA_ERROR_NOT_FOUND;
}

// Resolve a base driver symbol for the ABI version the caller was built
// against. CUDA 12+ clients request the base name and expect the _v2
// entry point when one exists. cuGetProcAddress itself is the clearest
// case: v1 takes four arguments, v2 adds the symbolStatus out-parameter.
static CUresult proc_address(const char *symbol, void **pfn, int cudaVersion) {
    if (!symbol || !pfn)
        return CUDA_ERROR_INVALID_VALUE;
    *pfn = NULL;
    size_t n = strlen(symbol);
    if (cudaVersion >= 12000 && n > 3 &&
        strcmp(symbol + n - 3, "_v2") != 0) {
        char versioned[160];
        if (n + 3 < sizeof(versioned)) {
            memcpy(versioned, symbol, n);
            memcpy(versioned + n, "_v2", 4);
            if (lookup_symbol(versioned, pfn) == CUDA_SUCCESS)
                return CUDA_SUCCESS;
        }
    }
    CUresult r = lookup_symbol(symbol, pfn);
    if (r != CUDA_SUCCESS && getenv("ACUDA_TRACE"))
        fprintf(stderr, "acuda-trace ?? cuGetProcAddress(%s) -> NOT_FOUND\n",
                symbol);
    return r;
}

CUresult ACUDA_CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          unsigned long long flags) {
    (void)flags;
    if (!pfn)
        return CUDA_ERROR_INVALID_VALUE;
    if (!symbol || !*symbol) {
        *pfn = NULL;
        return CUDA_ERROR_NOT_FOUND;
    }
    return proc_address(symbol, pfn, cudaVersion);
}

CUresult ACUDA_CUDAAPI cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             unsigned long long flags,
                             CUdriverProcAddressQueryResult *symbolStatus) {
    (void)flags;
    if (!pfn)
        return CUDA_ERROR_INVALID_VALUE;
    if (cudaVersion > ACUDA_DRIVER_VERSION)
        return CUDA_ERROR_INVALID_VALUE;
    if (!symbol || !*symbol) {
        *pfn = NULL;
        if (symbolStatus)
            *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        return CUDA_SUCCESS;
    }
    CUresult r = proc_address(symbol, pfn, cudaVersion);
    ACUDA_DEBUG_PRINT_PROC("acuda-proc %s v=%d found=%d pfn=%p\n",
                symbol ? symbol : "(null)", cudaVersion, r == CUDA_SUCCESS,
                pfn ? *pfn : NULL);
    if (r != CUDA_SUCCESS) {
        *pfn = NULL;
        if (symbolStatus)
            *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        // The v2 ABI reports lookup failures through symbolStatus and
        // still returns CUDA_SUCCESS; callers that need the legacy
        // error use the four-argument entry point.
        return CUDA_SUCCESS;
    }
    if (symbolStatus)
        *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    return CUDA_SUCCESS;
}

/* ---- Device characterization, primary contexts, misc (torch init path) ---- */

struct CUmemPool_st {
    int device;
};

#define ACUDA_MAX_DEVICES 8
static CUcontext primary_ctx[ACUDA_MAX_DEVICES];
static int primary_refs[ACUDA_MAX_DEVICES];
static CUmemPool default_pool[ACUDA_MAX_DEVICES];

static CUresult check_dev(int dev) {
    int n = 0;
    CUresult r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS)
        return r;
    return (dev >= 0 && dev < n) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE;
}

static int device_pci_info(CUdevice dev, acuda_device_info_t *info) {
    if (!g_actx || acuda_device_get(g_actx, (uint32_t)dev, info) != ACUDA_SUCCESS)
        return 0;
    return 1;
}

CUresult ACUDA_CUDAAPI cuDeviceGetAttribute(int *value, int attrib, CUdevice dev);

CUresult ACUDA_CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    if (!uuid)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = check_dev(dev);
    if (r != CUDA_SUCCESS)
        return r;
    // Deterministic pseudo-UUID: "ACUD" + ordinal + FNV of nothing (stable).
    memset(uuid->bytes, 0, 16);
    uuid->bytes[0] = 'A';
    uuid->bytes[1] = 'C';
    uuid->bytes[2] = 'U';
    uuid->bytes[3] = 'D';
    uuid->bytes[4] = (char)dev;
    uint32_t h = 2166136261u + (uint32_t)dev * 16777619u;
    memcpy(uuid->bytes + 8, &h, 4);
