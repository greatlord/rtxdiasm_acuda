// ACuda CUDA Runtime API compatibility (libcudart.so). C ABI.
//
// Layout-critical types (cudaDeviceProp, cudaFuncAttributes, enums) match
// the CUDA 12 headers field-for-field; verified against the
// nvidia-cuda-runtime pip wheel headers. Values reported are documented
// in docs/cudart-api.md.
#ifndef ACUDA_RUNTIME_API_H
#define ACUDA_RUNTIME_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError_enum {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorInvalidImage = 200,
    cudaErrorInvalidContext = 201,
    cudaErrorInvalidPtx = 218,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorInvalidHandle = 400,
    cudaErrorNotFound = 500,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchFailure = 719,
    cudaErrorNotSupported = 801,
    cudaErrorUnknown = 999
} cudaError_t;

typedef enum cudaMemcpyKind_enum {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

typedef enum cudaMemoryType_enum {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3
} cudaMemoryType;

typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;
typedef struct CUmemPoolHandle_st *cudaMemPool_t;
typedef struct CUgraph_st *cudaGraph_t;
typedef struct CUgraphNode_st *cudaGraphNode_t;
typedef struct CUgraphExec_st *cudaGraphExec_t;
typedef struct CUuserObject_st *cudaUserObject_t;
typedef struct cudaIpcEventHandle_st {
    char reserved[64];
} cudaIpcEventHandle_t;
typedef struct cudaIpcMemHandle_st {
    char reserved[64];
} cudaIpcMemHandle_t;
typedef struct cudaLaunchConfig_st cudaLaunchConfig_t;

typedef enum cudaDriverEntryPointQueryResult_enum {
    cudaDriverEntryPointSuccess = 0,
    cudaDriverEntryPointSymbolNotFound = 1
} cudaDriverEntryPointQueryResult;

typedef enum cudaStreamCaptureStatus_enum {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1
} cudaStreamCaptureStatus;

typedef void (*cudaStreamCallback_t)(cudaStream_t stream, int status,
                                     void *userData);

#define cudaStreamLegacy ((cudaStream_t)0)
#define cudaStreamPerThread ((cudaStream_t)0x2)
#define cudaStreamDefault ((cudaStream_t)0)
#define cudaStreamNonBlocking 0x01
#define cudaEventDefault 0x00
#define cudaEventBlockingSync 0x01
#define cudaEventDisableTiming 0x02

typedef struct dim3 {
    unsigned int x, y, z;
} dim3;

typedef struct cudaUUID_st {
    char bytes[16];
} cudaUUID_t;

/* Texture and array objects. */
typedef struct cudaArray *cudaArray_t;
typedef struct cudaArray *cudaArray_const_t;
typedef struct cudaMipmappedArray *cudaMipmappedArray_t;
typedef struct cudaMipmappedArray *cudaMipmappedArray_const_t;
typedef unsigned long long cudaTextureObject_t;
typedef unsigned long long cudaSurfaceObject_t;

#define cudaArrayDefault 0x00
#define cudaArrayLayered 0x01
#define cudaArraySurfaceLoadStore 0x02
#define cudaArrayCubemap 0x04
#define cudaArrayTextureGather 0x08
#define cudaArrayColorAttachment 0x20
#define cudaArrayDeferredMapping 0x80

typedef enum cudaChannelFormatKind_enum {
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3,
    cudaChannelFormatKindNVJPEG = 4
} cudaChannelFormatKind;

typedef struct cudaChannelFormatDesc_st {
    int x;
    int y;
    int z;
    int w;
    cudaChannelFormatKind f;
} cudaChannelFormatDesc;

typedef struct cudaExtent_st {
    size_t width;
    size_t height;
    size_t depth;
} cudaExtent;

typedef struct cudaPos_st {
    size_t x;
    size_t y;
    size_t z;
} cudaPos;

typedef struct cudaPitchedPtr_st {
    void *ptr;
    size_t pitch;
    size_t xsize;
    size_t ysize;
} cudaPitchedPtr;

typedef enum cudaResourceType_enum {
    cudaResourceTypeArray = 0x00,
    cudaResourceTypeMipmappedArray = 0x01,
    cudaResourceTypeLinear = 0x02,
    cudaResourceTypePitch2D = 0x03
} cudaResourceType;

typedef enum cudaTextureAddressMode_enum {
    cudaAddressModeWrap = 0,
    cudaAddressModeClamp = 1,
    cudaAddressModeMirror = 2,
    cudaAddressModeBorder = 3
} cudaTextureAddressMode;

typedef enum cudaTextureFilterMode_enum {
    cudaFilterModePoint = 0,
    cudaFilterModeLinear = 1
} cudaTextureFilterMode;

typedef enum cudaTextureReadMode_enum {
    cudaReadModeElementType = 0,
    cudaReadModeNormalizedFloat = 1
} cudaTextureReadMode;

typedef struct cudaResourceDesc_st {
    cudaResourceType resType;
    union {
        struct {
            cudaArray_t array;
        } array;
        struct {
            cudaMipmappedArray_t mipmap;
        } mipmap;
        struct {
            void *devPtr;
            cudaChannelFormatDesc desc;
            size_t sizeInBytes;
        } linear;
        struct {
            void *devPtr;
            cudaChannelFormatDesc desc;
            size_t width;
            size_t height;
            size_t pitchInBytes;
        } pitch2D;
    } res;
} cudaResourceDesc;

typedef struct cudaTextureDesc_st {
    cudaTextureAddressMode addressMode[3];
    cudaTextureFilterMode filterMode;
    cudaTextureReadMode readMode;
    int sRGB;
    float borderColor[4];
    int normalizedCoords;
    unsigned int maxAnisotropy;
    cudaTextureFilterMode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    int disableTrilinearOptimization;
    int seamlessCubemap;
} cudaTextureDesc;

typedef enum cudaResourceViewFormat_enum {
    cudaResViewFormatNone = 0x00,
    cudaResViewFormatUnsignedChar1 = 0x01,
    cudaResViewFormatUnsignedChar2 = 0x02,
    cudaResViewFormatUnsignedChar4 = 0x03,
    cudaResViewFormatSignedChar1 = 0x04,
    cudaResViewFormatUnsignedShort1 = 0x08,
    cudaResViewFormatFloat1 = 0x15,
    cudaResViewFormatFloat2 = 0x16,
    cudaResViewFormatFloat4 = 0x17
} cudaResourceViewFormat;

typedef struct cudaResourceViewDesc_st {
    cudaResourceViewFormat format;
    size_t width;
    size_t height;
    size_t depth;
    unsigned int firstMipmapLevel;
    unsigned int lastMipmapLevel;
    unsigned int firstLayer;
    unsigned int lastLayer;
} cudaResourceViewDesc;

typedef struct cudaMemcpy3DParms_st {
    cudaArray_t srcArray;
    struct cudaPos_st srcPos;
    struct cudaPitchedPtr_st srcPtr;
    cudaArray_t dstArray;
    struct cudaPos_st dstPos;
    struct cudaPitchedPtr_st dstPtr;
    struct cudaExtent_st extent;
    cudaMemcpyKind kind;
} cudaMemcpy3DParms;

struct cudaDeviceProp {
    char name[256];
    cudaUUID_t uuid;
    char luid[8];
    unsigned int luidDeviceNodeMask;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int maxBlocksPerMultiProcessor;
    int accessPolicyMaxWindowSize;
    size_t reservedSharedMemPerBlock;
    int hostRegisterSupported;
    int sparseCudaArraySupported;
    int hostRegisterReadOnlySupported;
    int timelineSemaphoreInteropSupported;
    int memoryPoolsSupported;
    int gpuDirectRDMASupported;
    unsigned int gpuDirectRDMAFlushWritesOptions;
    int gpuDirectRDMAWritesOrdering;
    unsigned int memoryPoolSupportedHandleTypes;
    int deferredMappingCudaArraySupported;
    int ipcEventSupported;
    int clusterLaunch;
    int unifiedFunctionPointers;
    int reserved2[2];
    int reserved1[1];
    int reserved[60];
};

struct cudaFuncAttributes {
    size_t sharedSizeBytes;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int ptxVersion;
    int binaryVersion;
    int cacheModeCA;
    int maxDynamicSharedSizeBytes;
    int preferredShmemCarveout;
    int clusterDimMustBeSet;
    int requiredClusterWidth;
    int requiredClusterHeight;
    int requiredClusterDepth;
    int clusterSchedulingPolicyPreference;
    int nonPortableClusterSizeAllowed;
    int reserved[16];
};

struct cudaPointerAttributes {
    cudaMemoryType type;
    int device;
    void *devicePointer;
    void *hostPointer;
};

typedef enum cudaMemPoolAttr_enum {
    cudaMemPoolReuseAllowOpportunistic = 0x2,
    cudaMemPoolAttrReleaseThreshold = 0x4,
    cudaMemPoolAttrUsedMemCurrent = 0x7
} cudaMemPoolAttr;

typedef enum cudaStreamCaptureMode_enum {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2
} cudaStreamCaptureMode;

typedef enum cudaLimit_enum {
    cudaLimitStackSize = 0x00,
    cudaLimitPrintfFifoSize = 0x01,
    cudaLimitMallocHeapSize = 0x02
} cudaLimit;

/* --- initialization / version --- */
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *prop,
                                       int device);
cudaError_t cudaDeviceGetAttribute(int *value, int attrib, int device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaDriverGetVersion(int *driverVersion);
cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceGetLimit(size_t *pValue, int limit);
cudaError_t cudaDeviceSetLimit(int limit, size_t value);
cudaError_t cudaThreadExchangeStreamCaptureMode(int *mode);

/* --- errors --- */
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char *cudaGetErrorString(cudaError_t error);
const char *cudaGetErrorName(cudaError_t error);

/* --- memory --- */
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream);
cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                              size_t spitch, size_t width, size_t height,
                              cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src,
                                int srcDevice, size_t count,
                                cudaStream_t stream);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
                            cudaStream_t stream);
/* --- texture arrays and texture objects --- */
cudaError_t cudaMallocArray(cudaArray_t *array,
                            const cudaChannelFormatDesc *desc, size_t width,
                            size_t height, unsigned int flags);
cudaError_t cudaMalloc3DArray(cudaArray_t *array,
                              const cudaChannelFormatDesc *desc,
                              struct cudaExtent_st extent,
                              unsigned int flags);
cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray,
                                     const cudaChannelFormatDesc *desc,
                                     struct cudaExtent_st extent,
                                     unsigned int numLevels,
                                     unsigned int flags);
cudaError_t cudaFreeArray(cudaArray_t array);
cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t *levelArray,
                                       cudaMipmappedArray_const_t mipmappedArray,
                                       unsigned int level);
cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject,
                                    const cudaResourceDesc *pResDesc,
                                    const cudaTextureDesc *pTexDesc,
                                    const cudaResourceViewDesc *pResViewDesc);
cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);
cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc *pResDesc,
                                             cudaTextureObject_t texObject);
cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc *pTexDesc,
                                            cudaTextureObject_t texObject);
cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject,
                                    const cudaResourceDesc *pResDesc);
cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject);
cudaError_t cudaGetSurfaceObjectResourceDesc(
    cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject);
cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset,
                                size_t hOffset, const void *src, size_t spitch,
                                size_t width, size_t height,
                                cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset,
                                     size_t hOffset, const void *src,
                                     size_t spitch, size_t width,
                                     size_t height, cudaMemcpyKind kind,
                                     cudaStream_t stream);
cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch,
                                  cudaArray_const_t src, size_t wOffset,
                                  size_t hOffset, size_t width, size_t height,
                                  cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
                                       cudaArray_const_t src, size_t wOffset,
                                       size_t hOffset, size_t width,
                                       size_t height,
                                       cudaMemcpyKind kind,
                                       cudaStream_t stream);
cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms_st *p);
cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms_st *p,
                              cudaStream_t stream);
cudaError_t cudaMemGetInfo(size_t *freeBytes, size_t *totalBytes);
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attrs,
                                     const void *ptr);
cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags);
cudaError_t cudaFreeHost(void *ptr);
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool,
                                   int attr, void *value);
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool,
                                   int attr, void *value);
cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *pool_out, int device);
cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t pool);
cudaError_t cudaMemPoolTrimTo(cudaMemPool_t pool, size_t minBytesToKeep);

/* --- streams / events --- */
cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream,
                                      unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                               unsigned int flags);
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                    unsigned int flags);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                 cudaEvent_t end);

/* --- launch --- */
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream);
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                const void *func, void **args);
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
                                  const void *func);
cudaError_t cudaFuncSetAttribute(const void *func, int attr, int value);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags);
cudaError_t cudaLaunchHostFunc(cudaStream_t stream,
                               void (*fn)(void *userData), void *userData);
cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem,
                                        cudaStream_t stream);
unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem, void *stream);
cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                       size_t *sharedMem, void *stream);
void __cudaInitModule(void);
cudaError_t cudaProfilerStart(void);
cudaError_t cudaProfilerStop(void);
cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr,
                                    unsigned long long flags,
                                    int *driverStatus);
cudaError_t cudaGetDriverEntryPointByVersion(const char *symbol,
                                             void **funcPtr,
                                             unsigned int cudaVersion,
                                             unsigned long long flags,
                                             int *driverStatus);

/* --- fatbin registration (compiler ABI, called by CUDA C++ runtime) --- */
void **__cudaRegisterFatBinary(void *fatCubin);
void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, void *tid, void *bid,
                            void *bDim, void *gDim, int *wSize);
void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                       char *deviceAddress, const char *deviceName, int ext,
                       size_t size, int constant, int global);
void __cudaUnregisterFatBinary(void **fatCubinHandle);

/* --- extended surface required by torch (peer, streams, symbols) --- */
cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
                                    int peerDevice);
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority,
                                             int *greatestPriority);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                         unsigned int flags, int priority);
cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority);
cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags);
cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void *userData, unsigned int flags);
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, int *pCaptureStatus);
cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream,
                                        int *captureStatus_out,
                                        unsigned long long *id_out);
cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src,
                                int srcDevice, size_t count,
                                cudaStream_t stream);
cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
                               size_t count, size_t offset, int kind);
cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol);
cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void *ptr);

#ifdef __cplusplus
}
#endif

#endif
