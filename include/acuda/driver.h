// ACuda CUDA Driver API compatibility (libcuda.so). C ABI.
//
// Implements the driver functions needed by JIT-style clients (Numba):
// init, devices, contexts, allocation, copies, memset, PTX modules,
// functions, launches, streams, events, PTX-passthrough linking, errors.
//
// Divergences from NVIDIA (see docs/driver-api.md): CUdeviceptr values
// are opaque handles (no host-side arithmetic); only compiled entries
// launch (1D, block 256); cubin/SASS rejected; the linker passes PTX
// through instead of JIT-compiling.
#ifndef ACUDA_DRIVER_H
#define ACUDA_DRIVER_H

#include <stddef.h>
#include <stdint.h>
#include "acuda/driver_attrs.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(_M_IX86)
#define ACUDA_CUDAAPI __stdcall
#else
#define ACUDA_CUDAAPI
#endif

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef struct CUlink_st *CUlinkState;
typedef struct CUmemPool_st *CUmemPool;
/* CUdeviceptr matches the process pointer width, like the NVIDIA headers:
 * 64-bit on 64-bit hosts, 32-bit on 32-bit hosts.  The 32-bit CUDA 6.x
 * runtime passes 4-byte device pointers (its own memset fallback stubs
 * clean 12 bytes for a 3-argument call); a wider type would unbalance
 * every stdcall driver call that takes a device pointer by value. */
#if defined(__i386__) || (defined(_WIN32) && !defined(_WIN64))
typedef unsigned int CUdeviceptr;
#else
typedef unsigned long long CUdeviceptr;
#endif

/* Context creation flags.  ACuda accepts the documented bits and ignores
 * the scheduling/mapping hints. */
typedef enum acuda_ctx_flags_enum {
    ACUDA_CTX_SCHED_AUTO = 0x00,
    ACUDA_CTX_SCHED_SPIN = 0x01,
    ACUDA_CTX_SCHED_YIELD = 0x02,
    ACUDA_CTX_SCHED_BLOCKING_SYNC = 0x04,
    ACUDA_CTX_BLOCKING_SYNC = 0x04,
    ACUDA_CTX_SCHED_MASK = 0x07,
    ACUDA_CTX_MAP_HOST = 0x08,
    ACUDA_CTX_LMEM_RESIZE_TO_MAX = 0x10,
    ACUDA_CTX_COREDUMP_ENABLE = 0x20,
    ACUDA_CTX_USER_COREDUMP_ENABLE = 0x40,
    ACUDA_CTX_SYNC_MEMOPS = 0x80,
    ACUDA_CTX_FLAGS_MASK = 0xFF
} acuda_ctx_flags;

/* Legacy CUDA 2.x/4.x device properties (used by PhysX 2.8). */
typedef struct CUdevprop_st {
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int sharedMemPerBlock;
    int totalConstantMemory;
    int SIMDWidth;
    int memPitch;
    int regsPerBlock;
    int clockRate;
    int textureAlign;
} CUdevprop;

typedef struct CUipcMemHandle_st {
    char reserved[64];
} CUipcMemHandle;
typedef struct CUipcEventHandle_st {
    char reserved[64];
} CUipcEventHandle;

typedef struct CUuuid_st {
    char bytes[16];
} CUuuid;
typedef int CUarray_format;

/* Texture / array objects. */
typedef struct CUarray_st *CUarray;
typedef struct CUmipmappedArray_st *CUmipmappedArray;
typedef unsigned long long CUtexObject;
typedef unsigned long long CUsurfObject;
typedef struct CUtexref_st *CUtexref;

#define CU_TRSF_READ_AS_INTEGER 0x01
#define CU_TRSF_NORMALIZED_COORDINATES 0x02
#define CU_TRSF_SRGB 0x10
#define CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION 0x20

typedef enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
    CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
    CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
    CU_AD_FORMAT_SIGNED_INT8 = 0x08,
    CU_AD_FORMAT_SIGNED_INT16 = 0x09,
    CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
    CU_AD_FORMAT_HALF = 0x10,
    CU_AD_FORMAT_FLOAT = 0x20
} CUarray_format_enum;

typedef struct CUDA_ARRAY_DESCRIPTOR_st {
    size_t Width;
    size_t Height;
    CUarray_format Format;
    unsigned int NumChannels;
} CUDA_ARRAY_DESCRIPTOR;

typedef struct CUDA_ARRAY3D_DESCRIPTOR_st {
    size_t Width;
    size_t Height;
    size_t Depth;
    CUarray_format Format;
    unsigned int NumChannels;
    unsigned int Flags;
} CUDA_ARRAY3D_DESCRIPTOR;

#define CUDA_ARRAY3D_LAYERED 0x01
#define CUDA_ARRAY3D_SURFACE_LDST 0x02
#define CUDA_ARRAY3D_2DARRAY 0x01
#define CUDA_ARRAY3D_CUBEMAP 0x04
#define CUDA_ARRAY3D_DEPTH_TEXTURE 0x10

typedef enum CUresourcetype_enum {
    CU_RESOURCE_TYPE_ARRAY = 0x00,
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01,
    CU_RESOURCE_TYPE_LINEAR = 0x02,
    CU_RESOURCE_TYPE_PITCH2D = 0x03
} CUresourcetype;

typedef enum CUfilter_mode_enum {
    CU_TR_FILTER_MODE_POINT = 0,
    CU_TR_FILTER_MODE_LINEAR = 1
} CUfilter_mode;

typedef enum CUaddress_mode_enum {
    CU_TR_ADDRESS_MODE_WRAP = 0,
    CU_TR_ADDRESS_MODE_CLAMP = 1,
    CU_TR_ADDRESS_MODE_MIRROR = 2,
    CU_TR_ADDRESS_MODE_BORDER = 3
} CUaddress_mode;

typedef struct CUDA_RESOURCE_DESC_st {
    CUresourcetype resType;
    union {
        struct { CUarray hArray; } array;
        struct { CUmipmappedArray hMipmappedArray; } mipmap;
        struct {
            CUdeviceptr devPtr;
            CUarray_format format;
            unsigned int numChannels;
            size_t sizeInBytes;
        } linear;
        struct {
            CUdeviceptr devPtr;
            CUarray_format format;
            unsigned int numChannels;
            size_t width;
            size_t height;
            size_t pitchInBytes;
        } pitch2D;
   } res;
    unsigned int flags;
} CUDA_RESOURCE_DESC;

typedef struct CUDA_TEXTURE_DESC_st {
    CUaddress_mode addressMode[3];
    CUfilter_mode filterMode;
    unsigned int flags;
    unsigned int maxAnisotropy;
    CUfilter_mode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    float borderColor[4];
    int reserved[12];
} CUDA_TEXTURE_DESC;

typedef enum CUresourceViewFormat_enum {
    CU_RES_VIEW_FORMAT_NONE = 0x00,
    CU_RES_VIEW_FORMAT_UINT_1X8 = 0x01,
    CU_RES_VIEW_FORMAT_UINT_2X8 = 0x02,
    CU_RES_VIEW_FORMAT_UINT_4X8 = 0x03,
    CU_RES_VIEW_FORMAT_SINT_1X8 = 0x04,
    CU_RES_VIEW_FORMAT_UINT_1X16 = 0x08,
    CU_RES_VIEW_FORMAT_FLOAT_1X32 = 0x15,
    CU_RES_VIEW_FORMAT_FLOAT_2X32 = 0x16,
    CU_RES_VIEW_FORMAT_FLOAT_4X32 = 0x17
} CUresourceViewFormat;

typedef struct CUDA_RESOURCE_VIEW_DESC_st {
    CUresourceViewFormat format;
    size_t width;
    size_t height;
    size_t depth;
    unsigned int firstMipmapLevel;
    unsigned int lastMipmapLevel;
    unsigned int firstLayer;
    unsigned int lastLayer;
    unsigned int reserved[16];
} CUDA_RESOURCE_VIEW_DESC;

typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST = 0x01,
    CU_MEMORYTYPE_DEVICE = 0x02,
    CU_MEMORYTYPE_ARRAY = 0x03,
    CU_MEMORYTYPE_UNIFIED = 0x04
} CUmemorytype;

typedef enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
    CU_POINTER_ATTRIBUTE_MAPPED = 13,
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,
    CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20,
    CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE = 21
} CUpointer_attribute;

typedef struct CUDA_MEMCPY2D_st {
    size_t srcXInBytes;
    size_t srcY;
    CUmemorytype srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    CUmemorytype dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
} CUDA_MEMCPY2D;

typedef struct CUDA_MEMCPY3D_st {
    size_t srcXInBytes;
    size_t srcY;
    size_t srcZ;
    size_t srcLOD;
    CUmemorytype srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    void *reserved0;
    size_t srcPitch;
    size_t srcHeight;
    size_t dstXInBytes;
    size_t dstY;
    size_t dstZ;
    size_t dstLOD;
    CUmemorytype dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    void *reserved1;
    size_t dstPitch;
    size_t dstHeight;
    size_t WidthInBytes;
    size_t Height;
    size_t Depth;
} CUDA_MEMCPY3D;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_NOT_INITIALIZED 3
#define CUDA_ERROR_DEINITIALIZED 4
#define CUDA_ERROR_NO_DEVICE 100
#define CUDA_ERROR_INVALID_DEVICE 101
#define CUDA_ERROR_INVALID_IMAGE 200
#define CUDA_ERROR_INVALID_CONTEXT 201
/* No loaded module/kernel matches the installed device; NVIDIA returns this
 * when a module carries no PTX and none of its cubins match the GPU. */
#define CUDA_ERROR_NO_BINARY_FOR_GPU 209
#define CUDA_ERROR_INVALID_HANDLE 400
#define CUDA_ERROR_NOT_FOUND 500
#define CUDA_ERROR_NOT_READY 600
#define CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES 701
#define CUDA_ERROR_NOT_SUPPORTED 801
#define CUDA_ERROR_UNKNOWN 999

#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK 1
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X 2
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y 3
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z 4
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X 5
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y 6
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z 7
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK 8
/* Common characterization attributes (values match CUDA headers). */
#define CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY 9
#define CU_DEVICE_ATTRIBUTE_WARP_SIZE 10
#define CU_DEVICE_ATTRIBUTE_MAX_PITCH 11
#define CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK 12
#define CU_DEVICE_ATTRIBUTE_CLOCK_RATE 13
#define CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT 14
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT 17
#define CU_DEVICE_ATTRIBUTE_INTEGRATED 18
#define CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY 19
#define CU_DEVICE_ATTRIBUTE_COMPUTE_MODE 20
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH 21
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH 22
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT 23
#define CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT 30
#define CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS 31
#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID 33
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID 34
#define CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE 36
#define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID 50
#define CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH 37
#define CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE 38
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR 39
#define CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT 40
#define CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING 41
#define CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT 51

typedef enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_NUM_OPTIONS = 12
} CUjit_option;

typedef enum CUjitInputType_enum {
    CU_JIT_INPUT_CUBIN = 0,
    CU_JIT_INPUT_PTX = 1,
    CU_JIT_INPUT_FATBINARY = 2,
    CU_JIT_INPUT_OBJECT = 3,
    CU_JIT_INPUT_LIBRARY = 4,
    CU_JIT_NUM_INPUT_TYPES = 5
} CUjitInputType;

CUresult ACUDA_CUDAAPI cuInit(unsigned int flags);
CUresult ACUDA_CUDAAPI cuDriverGetVersion(int *driverVersion);
CUresult ACUDA_CUDAAPI cuDeviceGetCount(int *count);
CUresult ACUDA_CUDAAPI cuDeviceGet(CUdevice *device, int ordinal);
CUresult ACUDA_CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult ACUDA_CUDAAPI cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
CUresult ACUDA_CUDAAPI cuDeviceGetProperties(CUdevprop *prop, CUdevice dev);
CUresult ACUDA_CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev);
struct cudaDeviceProp;
CUresult ACUDA_CUDAAPI acuda_query_device_properties(struct cudaDeviceProp *prop, int device);
CUresult ACUDA_CUDAAPI cuDeviceGetAttribute(int *value, int attrib, CUdevice dev);

CUresult ACUDA_CUDAAPI cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult ACUDA_CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult ACUDA_CUDAAPI cuCtxDestroy_v2(CUcontext ctx);
CUresult ACUDA_CUDAAPI cuCtxDestroy(CUcontext ctx);
CUresult ACUDA_CUDAAPI cuCtxPushCurrent_v2(CUcontext ctx);
CUresult ACUDA_CUDAAPI cuCtxPushCurrent(CUcontext ctx);
CUresult ACUDA_CUDAAPI cuCtxPopCurrent_v2(CUcontext *pctx);
CUresult ACUDA_CUDAAPI cuCtxPopCurrent(CUcontext *pctx);
CUresult ACUDA_CUDAAPI cuCtxSynchronize(void);

CUresult ACUDA_CUDAAPI cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult ACUDA_CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult ACUDA_CUDAAPI cuMemFree_v2(CUdeviceptr dptr);
CUresult ACUDA_CUDAAPI cuMemFree(CUdeviceptr dptr);
CUresult ACUDA_CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t bytes);
CUresult ACUDA_CUDAAPI cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t bytes);
CUresult ACUDA_CUDAAPI cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t bytes);
CUresult ACUDA_CUDAAPI cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t bytes);
CUresult ACUDA_CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes);
CUresult ACUDA_CUDAAPI cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t bytes);
CUresult ACUDA_CUDAAPI cuMemsetD32_v2(CUdeviceptr dst, unsigned int value, size_t count);
CUresult ACUDA_CUDAAPI cuMemsetD32(CUdeviceptr dst, unsigned int value, size_t count);
CUresult ACUDA_CUDAAPI cuMemsetD8_v2(CUdeviceptr dst, unsigned char value, size_t count);
CUresult ACUDA_CUDAAPI cuMemsetD8(CUdeviceptr dst, unsigned char value, size_t count);

CUresult ACUDA_CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, CUjit_option *options,
                            void **optionValues);
CUresult ACUDA_CUDAAPI cuModuleLoadData(CUmodule *module, const void *image);
CUresult ACUDA_CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin);
CUresult ACUDA_CUDAAPI cuModuleUnload(CUmodule module);
CUresult ACUDA_CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule module,
                             const char *name);
CUresult ACUDA_CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra);
CUresult ACUDA_CUDAAPI cuLaunchKernelEx(const void *config, CUfunction f,
                          void **kernelParams, void **extra);

CUresult ACUDA_CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int flags);
CUresult ACUDA_CUDAAPI cuStreamDestroy_v2(CUstream hStream);
CUresult ACUDA_CUDAAPI cuStreamDestroy(CUstream hStream);
CUresult ACUDA_CUDAAPI cuStreamSynchronize(CUstream hStream);

CUresult ACUDA_CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int flags);
CUresult ACUDA_CUDAAPI cuEventDestroy_v2(CUevent hEvent);
CUresult ACUDA_CUDAAPI cuEventDestroy(CUevent hEvent);
CUresult ACUDA_CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult ACUDA_CUDAAPI cuEventSynchronize(CUevent hEvent);
CUresult ACUDA_CUDAAPI cuEventQuery(CUevent hEvent);
CUresult ACUDA_CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);

/* Function attributes (see acuda/driver_attrs.h for the official
 * NVIDIA CUfunction_attribute list). Only MAX_THREADS_PER_BLOCK
 * through MAX_DYNAMIC_SHARED_SIZE_BYTES are implemented. */
CUresult ACUDA_CUDAAPI cuFuncGetAttribute(int *pi, int attrib, CUfunction hfunc);
CUresult ACUDA_CUDAAPI cuFuncSetAttribute(CUfunction hfunc, int attrib, int value);

/* IPC handles: present so clients (numba) can wrap them; no cross-process
 * memory exists in this model, so all uses fail NOT_SUPPORTED. */
CUresult ACUDA_CUDAAPI cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr);
CUresult ACUDA_CUDAAPI cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle,
                            unsigned int flags);
CUresult ACUDA_CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr);
CUresult ACUDA_CUDAAPI cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event);
CUresult ACUDA_CUDAAPI cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle);
CUresult ACUDA_CUDAAPI cuIpcCloseEventHandle(CUevent event);

CUresult ACUDA_CUDAAPI cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options,
                         void **optionValues, CUlinkState *stateOut);
CUresult ACUDA_CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                      void **optionValues, CUlinkState *stateOut);
CUresult ACUDA_CUDAAPI cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data,
                          size_t size, const char *name,
                          unsigned int numOptions, CUjit_option *options,
                          void **optionValues);
CUresult ACUDA_CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type, void *data,
                       size_t size, const char *name,
                       unsigned int numOptions, CUjit_option *options,
                       void **optionValues);
CUresult ACUDA_CUDAAPI cuLinkAddFile_v2(CUlinkState state, CUjitInputType type,
                          const char *path, unsigned int numOptions,
                          CUjit_option *options, void **optionValues);
CUresult ACUDA_CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                       const char *path, unsigned int numOptions,
                       CUjit_option *options, void **optionValues);
CUresult ACUDA_CUDAAPI cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut);
CUresult ACUDA_CUDAAPI cuLinkDestroy(CUlinkState state);

CUresult ACUDA_CUDAAPI cuGetErrorString(CUresult error, const char **pStr);
CUresult ACUDA_CUDAAPI cuGetErrorName(CUresult error, const char **pStr);

/* Device characterization, primary contexts, pools (torch init path). */
CUresult ACUDA_CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev);
CUresult ACUDA_CUDAAPI cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev);
CUresult ACUDA_CUDAAPI cuDeviceGetByPCIBusId(CUdevice *device, const char *pciBusId);
CUresult ACUDA_CUDAAPI cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements,
                                            CUarray_format format,
                                            unsigned numChannels,
                                            CUdevice dev);
CUresult ACUDA_CUDAAPI cuDeviceGetP2PAttribute(int *value, int attrib,
                                 CUdevice srcDevice, CUdevice dstDevice);
CUresult ACUDA_CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                               CUdevice peerDev);
CUresult ACUDA_CUDAAPI cuFlushGPUDirectRDMAWrites(int target, int scope);
CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev);
CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags,
                                    int *active);
CUresult ACUDA_CUDAAPI cuCtxGetCurrent(CUcontext *pctx);
CUresult ACUDA_CUDAAPI cuCtxSetCurrent(CUcontext ctx);
CUresult ACUDA_CUDAAPI cuCtxGetDevice(CUdevice *device);
CUresult ACUDA_CUDAAPI cuCtxGetDevice_v2(CUdevice *device);
CUresult ACUDA_CUDAAPI cuCtxSynchronize_v2(void);
CUresult ACUDA_CUDAAPI cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
CUresult ACUDA_CUDAAPI cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev);
CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxRelease_v2(CUdevice dev);
CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxReset_v2(CUdevice dev);
CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags);
CUresult ACUDA_CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int *version);
CUresult ACUDA_CUDAAPI cuCtxGetLimit(size_t *pvalue, int limit);
CUresult ACUDA_CUDAAPI cuCtxSetLimit(int limit, size_t value);
CUresult ACUDA_CUDAAPI cuCtxSetCacheConfig(int config);
CUresult ACUDA_CUDAAPI cuCtxGetCacheConfig(int *config);
CUresult ACUDA_CUDAAPI cuCtxDetach(CUcontext ctx);
CUresult ACUDA_CUDAAPI cuCtxAttach(CUcontext *pctx, unsigned int flags);
CUresult ACUDA_CUDAAPI cuMemGetInfo(size_t *freeBytes, size_t *totalBytes);
CUresult ACUDA_CUDAAPI cuDeviceGetDefaultMemPool(CUmemPool *pool_out, CUdevice dev);
CUresult ACUDA_CUDAAPI cuDeviceGetMemPool(CUmemPool *pool_out, CUdevice dev);
CUresult ACUDA_CUDAAPI cuDeviceSetMemPool(CUdevice dev, CUmemPool pool);
CUresult ACUDA_CUDAAPI cuMemPoolTrimTo(CUmemPool pool, size_t minBytesToKeep);
CUresult ACUDA_CUDAAPI cuStreamQuery(CUstream hStream);
CUresult ACUDA_CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                           unsigned int flags);
CUresult ACUDA_CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags);
CUresult ACUDA_CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority);

/* Texture arrays and texture objects. */
CUresult ACUDA_CUDAAPI cuArrayCreate(CUarray *pHandle,
                       const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
CUresult ACUDA_CUDAAPI cuArrayCreate_v2(CUarray *pHandle,
                          const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
CUresult ACUDA_CUDAAPI cuArray3DCreate(CUarray *pHandle,
                         const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
CUresult ACUDA_CUDAAPI cuArray3DCreate_v2(CUarray *pHandle,
                            const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
CUresult ACUDA_CUDAAPI cuArrayDestroy(CUarray hArray);
CUresult ACUDA_CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor,
                              CUarray hArray);
CUresult ACUDA_CUDAAPI cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor,
                                CUarray hArray);
CUresult ACUDA_CUDAAPI cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                                const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                                unsigned int numMipmapLevels);
CUresult ACUDA_CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray);
CUresult ACUDA_CUDAAPI cuMipmappedArrayGetLevel(CUarray *pLevelArray,
                                  CUmipmappedArray hMipmappedArray,
                                  unsigned int level);
CUresult ACUDA_CUDAAPI cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod,
                                          const char *name);
CUresult ACUDA_CUDAAPI cuTexRefSetAddress_v2(size_t *ByteOffset,
                                             CUtexref hTexRef,
                                             CUdeviceptr dptr, size_t bytes);
CUresult ACUDA_CUDAAPI cuTexRefSetAddress(size_t *ByteOffset,
                                          CUtexref hTexRef,
                                          CUdeviceptr dptr, size_t bytes);
CUresult ACUDA_CUDAAPI cuTexRefSetArray(CUtexref hTexRef, CUarray hArray,
                                        unsigned int Flags);
CUresult ACUDA_CUDAAPI cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt,
                                         int NumPackedComponents);
CUresult ACUDA_CUDAAPI cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags);
CUresult ACUDA_CUDAAPI cuTexRefSetFilterMode(CUtexref hTexRef,
                                             CUfilter_mode fm);
CUresult ACUDA_CUDAAPI cuTexRefSetAddressMode(CUtexref hTexRef, int dim,
                                              CUaddress_mode am);
CUresult ACUDA_CUDAAPI cuTexRefSetMaxAnisotropy(CUtexref hTexRef,
                                                unsigned int maxAniso);
CUresult ACUDA_CUDAAPI cuTexRefGetAddress_v2(CUdeviceptr *pdptr,
                                             CUtexref hTexRef);
CUresult ACUDA_CUDAAPI cuTexRefGetAddress(CUdeviceptr *pdptr,
                                          CUtexref hTexRef);
CUresult ACUDA_CUDAAPI cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef);
CUresult ACUDA_CUDAAPI cuTexRefGetFormat(CUarray_format *pFormat,
                                         int *pNumChannels, CUtexref hTexRef);
CUresult ACUDA_CUDAAPI cuTexRefGetFlags(unsigned int *pFlags,
                                        CUtexref hTexRef);
CUresult ACUDA_CUDAAPI cuTexRefGetFilterMode(CUfilter_mode *pfm,
                                             CUtexref hTexRef);
CUresult ACUDA_CUDAAPI cuTexRefGetAddressMode(CUaddress_mode *pam,
                                              CUtexref hTexRef, int dim);
CUresult ACUDA_CUDAAPI cuTexRefGetMaxAnisotropy(int *pmaxAniso,
                                                CUtexref hTexRef);

CUresult ACUDA_CUDAAPI cuTexObjectCreate(CUtexObject *pTexObject,
                           const CUDA_RESOURCE_DESC *pResDesc,
                           const CUDA_TEXTURE_DESC *pTexDesc,
                           const CUDA_RESOURCE_VIEW_DESC *pResViewDesc);
CUresult ACUDA_CUDAAPI cuTexObjectDestroy(CUtexObject texObject);
CUresult ACUDA_CUDAAPI cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                    CUtexObject texObject);
CUresult ACUDA_CUDAAPI cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc,
                                   CUtexObject texObject);
CUresult ACUDA_CUDAAPI cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc,
                                                      CUtexObject texObject);
CUresult ACUDA_CUDAAPI cuSurfObjectCreate(CUsurfObject *pSurfObject,
                            const CUDA_RESOURCE_DESC *pResDesc);
CUresult ACUDA_CUDAAPI cuSurfObjectDestroy(CUsurfObject surfObject);
CUresult ACUDA_CUDAAPI cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                     CUsurfObject surfObject);
CUresult ACUDA_CUDAAPI cuMemcpy2D(const void *pCopy);
CUresult ACUDA_CUDAAPI cuMemcpy2DAsync(const void *pCopy, CUstream hStream);
CUresult ACUDA_CUDAAPI cuMemcpy3D(const void *pCopy);
CUresult ACUDA_CUDAAPI cuMemcpy3DAsync(const void *pCopy, CUstream hStream);

// Result reported by cuGetProcAddress_v2 (CUDA 12.0+ ABI).
typedef enum CUdriverProcAddressQueryResult_enum {
    CU_GET_PROC_ADDRESS_SUCCESS = 0,
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1,
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2
} CUdriverProcAddressQueryResult;

// Forward-compatibility dispatch (CUDA 11.3+ clients dlsym this first;
// without it they unload the library and fall back to a legacy path).
// _v2 takes the fifth symbolStatus out-parameter; CUDA 13's cuda.h aliases
// cuGetProcAddress to that five-argument form.
CUresult ACUDA_CUDAAPI cuGetExportTable(const void **ppExportTable,
                          const CUuuid *pExportTableId);
CUresult ACUDA_CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          unsigned long long flags);
CUresult ACUDA_CUDAAPI cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             unsigned long long flags,
                             CUdriverProcAddressQueryResult *symbolStatus);

// CUDA 12+ library objects. A library handle is module-compatible in
// ACuda, so kernels can be fetched lazily from the loaded image.
CUresult ACUDA_CUDAAPI cuLibraryLoadData(void **library, const void *code, void *jitOptions,
                           void **jitOptionsValues, unsigned int numJitOptions,
                           void *libraryOptions, void **libraryOptionValues,
                           unsigned int numLibraryOptions);
CUresult ACUDA_CUDAAPI cuLibraryLoadFromFile(void **library, const char *fileName,
                               void *jitOptions, void **jitOptionsValues,
                               unsigned int numJitOptions, void *libraryOptions,
                               void **libraryOptionValues,
                               unsigned int numLibraryOptions);
CUresult ACUDA_CUDAAPI cuLibraryUnload(void *library);
CUresult ACUDA_CUDAAPI cuLibraryGetKernel(void **pKernel, void *library, const char *name);
CUresult ACUDA_CUDAAPI cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                            void *library, const char *name);
CUresult ACUDA_CUDAAPI cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes,
                             void *library, const char *name);
CUresult ACUDA_CUDAAPI cuLibraryGetUnifiedFunction(void **fptr, void *library,
                                     const char *symbol);
CUresult ACUDA_CUDAAPI cuLibraryGetKernelCount(unsigned int *count, void *library);

#ifdef __cplusplus
}
#endif

#endif
