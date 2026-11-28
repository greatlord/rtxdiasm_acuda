#include "acuda/driver/driver_internal.h"
#include "acuda/runtime_api.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

driver_texobj_t *g_texobjs;
size_t g_ntexobjs, g_ctexobjs;
CUtexObject g_next_texid = 1;

driver_surfobj_t *g_surfobjs;
size_t g_nsurfobjs, g_csurfobjs;
acuda_buffer_t *g_surf_slots;
uint32_t g_surf_slot_cap;

// ---- CUDA texture arrays and texture objects -------------------------

int driver_elem_bytes(uint32_t fmt) {
    switch (fmt) {
    case 0: return 16;
    case 1: return 4;
    case 2: return 4;
    case 3: return 4;
    case 4: return 4;
    }
    return 0;
}

static int driver_array_format(CUarray_format f, unsigned nchans,
                               unsigned flags, uint32_t *fmt) {
    if (flags & CUDA_ARRAY3D_DEPTH_TEXTURE) {
        if (f == CU_AD_FORMAT_FLOAT && nchans == 1) {
            *fmt = 4;
            return 0;
        }
        return -1;
    }
    switch (f) {
    case CU_AD_FORMAT_FLOAT:
        if (nchans == 4) { *fmt = 0; return 0; }
        if (nchans == 1) { *fmt = 2; return 0; }
        return -1;
    case CU_AD_FORMAT_UNSIGNED_INT32:
        if (nchans == 1) { *fmt = 3; return 0; }
        return -1;
    case CU_AD_FORMAT_UNSIGNED_INT8:
        if (nchans == 4) { *fmt = 1; return 0; }
        return -1;
    default:
        return -1;
    }
}

static void driver_array_shape(const CUDA_ARRAY3D_DESCRIPTOR *d,
                               uint32_t *dim, uint32_t *layers,
                               uint32_t *cube) {
    *cube = (d->Flags & CUDA_ARRAY3D_CUBEMAP) ? 1u : 0u;
    if (d->Flags & CUDA_ARRAY3D_LAYERED) {
        *dim = d->Height <= 1 ? 1u : 2u;
        *layers = (uint32_t)d->Depth;
    } else if (d->Depth > 1) {
        *dim = 3;
        *layers = 1;
    } else if (d->Height <= 1) {
        *dim = 1;
        *layers = 1;
    } else {
        *dim = 2;
        *layers = 1;
    }
    if (*cube && *layers < 6)
        *layers = 6; // CUDA cube arrays are six layers per cube
}

static CUresult driver_array_alloc(CUarray *out,
                                   const CUDA_ARRAY3D_DESCRIPTOR *d,
                                   unsigned levels, acuda_buffer_t shared) {
    if (!out || !d || d->Width == 0)
        return CUDA_ERROR_INVALID_VALUE;
    uint32_t fmt;
    if (driver_array_format(d->Format, d->NumChannels, d->Flags, &fmt) != 0)
        return CUDA_ERROR_NOT_SUPPORTED;
    uint32_t dim, layers, cube;
    driver_array_shape(d, &dim, &layers, &cube);
    if (cube && (layers % 6) != 0)
        return CUDA_ERROR_INVALID_VALUE;
    acuda_buffer_t img = shared;
    if (!img) {
        acuda_image_desc id = {
            .width = (uint32_t)d->Width,
            .height = dim == 1 ? 1u : (uint32_t)d->Height,
            .depth = dim == 3 ? (uint32_t)d->Depth : 1u,
            .layers = layers,
            .levels = levels ? levels : 1,
            .samples = 1,
            .format = fmt,
            .dim = dim,
            .cube = cube,
        };
        if (acuda_image_create(g_actx, &img, &id, NULL) != ACUDA_SUCCESS)
            return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUarray a = calloc(1, sizeof(*a));
    if (!a) {
        if (!shared)
            acuda_free(g_actx, img);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    a->image = img;
    a->own = shared ? 0 : 1;
    a->level = 0;
    a->desc3 = *d;
    a->is3d = 1;
    a->desc2.Width = d->Width;
    a->desc2.Height = d->Height;
    a->desc2.Format = d->Format;
    a->desc2.NumChannels = d->NumChannels;
    *out = a;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuArrayCreate_v2(CUarray *pHandle,
                          const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    if (!pHandle || !pAllocateArray)
        return CUDA_ERROR_INVALID_VALUE;
    CUDA_ARRAY3D_DESCRIPTOR d = {
        .Width = pAllocateArray->Width,
        .Height = pAllocateArray->Height,
        .Depth = 1,
        .Format = pAllocateArray->Format,
        .NumChannels = pAllocateArray->NumChannels,
        .Flags = 0,
    };
    return driver_array_alloc(pHandle, &d, 1, NULL);
}

CUresult ACUDA_CUDAAPI cuArrayCreate(CUarray *pHandle,
                       const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
    return cuArrayCreate_v2(pHandle, pAllocateArray);
}

CUresult ACUDA_CUDAAPI cuArray3DCreate_v2(CUarray *pHandle,
                            const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    return driver_array_alloc(pHandle, pAllocateArray, 1, NULL);
}

CUresult ACUDA_CUDAAPI cuArray3DCreate(CUarray *pHandle,
                         const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
    return cuArray3DCreate_v2(pHandle, pAllocateArray);
}

CUresult ACUDA_CUDAAPI cuArrayDestroy(CUarray hArray) {
    if (!hArray)
        return CUDA_ERROR_INVALID_VALUE;
    if (hArray->own && hArray->image)
        acuda_free(g_actx, hArray->image);
    free(hArray);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor,
                              CUarray hArray) {
    if (!pArrayDescriptor || !hArray)
        return CUDA_ERROR_INVALID_VALUE;
    *pArrayDescriptor = hArray->desc2;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor,
                                CUarray hArray) {
    if (!pArrayDescriptor || !hArray)
        return CUDA_ERROR_INVALID_VALUE;
    *pArrayDescriptor = hArray->desc3;
    return CUDA_SUCCESS;
}

cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc *desc, cudaArray_t array) {
    if (!desc) return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
    memset(desc, 0, sizeof(*desc));
    if (!array) return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
    {
        CUDA_ARRAY_DESCRIPTOR ad;
        if (cuArrayGetDescriptor(&ad, (CUarray)(uintptr_t)array) == CUDA_SUCCESS) {
            unsigned n = ad.NumChannels; int bits = 32; int fmt = 0;
            switch (ad.Format) {
            case CU_AD_FORMAT_UNSIGNED_INT8: bits = 8; fmt = 1; break;
            case CU_AD_FORMAT_UNSIGNED_INT16: bits = 16; fmt = 1; break;
            case CU_AD_FORMAT_UNSIGNED_INT32: bits = 32; fmt = 1; break;
            case CU_AD_FORMAT_SIGNED_INT8: bits = 8; fmt = 2; break;
            case CU_AD_FORMAT_SIGNED_INT16: bits = 16; fmt = 2; break;
            case CU_AD_FORMAT_SIGNED_INT32: bits = 32; fmt = 2; break;
            case CU_AD_FORMAT_HALF: bits = 16; fmt = 0; break;
            case CU_AD_FORMAT_FLOAT: bits = 32; fmt = 0; break;
            default: break;
            }
            desc->x = (n >= 1) ? bits : 0;
            desc->y = (n >= 2) ? bits : 0;
            desc->z = (n >= 3) ? bits : 0;
            desc->w = (n >= 4) ? bits : 0;
            desc->f = (enum cudaChannelFormatKind)fmt;
            return (cudaError_t)CUDA_SUCCESS;
        }
    }
    {
        CUDA_ARRAY3D_DESCRIPTOR ad3;
        if (cuArray3DGetDescriptor(&ad3, (CUarray)(uintptr_t)array) == CUDA_SUCCESS) {
            unsigned n = ad3.NumChannels; int bits = 32; int fmt = 0;
            switch (ad3.Format) {
            case CU_AD_FORMAT_UNSIGNED_INT8: bits = 8; fmt = 1; break;
            case CU_AD_FORMAT_UNSIGNED_INT16: bits = 16; fmt = 1; break;
            case CU_AD_FORMAT_UNSIGNED_INT32: bits = 32; fmt = 1; break;
            case CU_AD_FORMAT_SIGNED_INT8: bits = 8; fmt = 2; break;
            case CU_AD_FORMAT_SIGNED_INT16: bits = 16; fmt = 2; break;
            case CU_AD_FORMAT_SIGNED_INT32: bits = 32; fmt = 2; break;
            case CU_AD_FORMAT_HALF: bits = 16; fmt = 0; break;
            case CU_AD_FORMAT_FLOAT: bits = 32; fmt = 0; break;
            default: break;
            }
            desc->x = (n >= 1) ? bits : 0;
            desc->y = (n >= 2) ? bits : 0;
            desc->z = (n >= 3) ? bits : 0;
            desc->w = (n >= 4) ? bits : 0;
            desc->f = (enum cudaChannelFormatKind)fmt;
            return (cudaError_t)CUDA_SUCCESS;
        }
    }
    return (cudaError_t)CUDA_ERROR_INVALID_HANDLE;
}

cudaError_t cudaMallocArray(cudaArray_t *array,
                            const cudaChannelFormatDesc *desc, size_t width,
                            size_t height, unsigned int flags) {
    CUDA_ARRAY_DESCRIPTOR ad;
    CUarray a = NULL;
    CUresult r;
    if (!array || !desc) return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
    {
        unsigned n = (desc->x?1u:0u)+(desc->y?1u:0u)+(desc->z?1u:0u)+(desc->w?1u:0u);
        int bits = desc->x;
        CUarray_format fmt = CU_AD_FORMAT_FLOAT;
        switch (desc->f) {
        case 0: fmt = (bits==8)?CU_AD_FORMAT_SIGNED_INT8:(bits==16)?CU_AD_FORMAT_SIGNED_INT16:CU_AD_FORMAT_SIGNED_INT32; break;
        case 1: fmt = (bits==8)?CU_AD_FORMAT_UNSIGNED_INT8:(bits==16)?CU_AD_FORMAT_UNSIGNED_INT16:CU_AD_FORMAT_UNSIGNED_INT32; break;
        case 2: fmt = (bits==16)?CU_AD_FORMAT_HALF:CU_AD_FORMAT_FLOAT; break;
        default: fmt = (bits==16)?CU_AD_FORMAT_HALF:CU_AD_FORMAT_FLOAT; break;
        }
        ad.Width = width;
        ad.Height = height;
        ad.Format = fmt;
        ad.NumChannels = n;
    }
    r = cuArrayCreate_v2(&a, &ad);
    if (r != CUDA_SUCCESS) return (cudaError_t)r;
    *array = (cudaArray_t)(void *)a;
    (void)flags;
    return (cudaError_t)CUDA_SUCCESS;
}

CUresult acuda_cuda_resource_desc_to_cu(const void *pResDesc, CUDA_RESOURCE_DESC *rd) {
    typedef struct { int x, y, z, w, f; } acuda_chfmt;
    typedef struct { void *array; } acuda_res_array;
    typedef struct { void *mipmap; } acuda_res_mip;
    typedef struct { void *devPtr; acuda_chfmt desc; size_t sizeInBytes; } acuda_res_linear;
    typedef struct { void *devPtr; acuda_chfmt desc; size_t width; size_t height; size_t pitchInBytes; } acuda_res_pitch2d;
    typedef struct { int resType; int pad; union { acuda_res_array array; acuda_res_mip mipmap; acuda_res_linear linear; acuda_res_pitch2d pitch2D; char raw[64]; } res; } acuda_cuda_resdesc;
    const acuda_cuda_resdesc *src = (const acuda_cuda_resdesc *)pResDesc;
    memset(rd, 0, sizeof(*rd));
    if (!src) return CUDA_ERROR_INVALID_VALUE;
    rd->resType = (CUresourcetype)src->resType;
    if (src->resType == 0) { /* cudaResourceTypeArray */
        rd->res.array.hArray = (CUarray)(uintptr_t)src->res.array.array;
        return CUDA_SUCCESS;
    }
    if (src->resType == 1) { /* cudaResourceTypeMipmappedArray */
        rd->res.mipmap.hMipmappedArray = (CUmipmappedArray)(uintptr_t)src->res.mipmap.mipmap;
        return CUDA_SUCCESS;
    }
    if (src->resType == 2) { /* cudaResourceTypeLinear */
        unsigned n = (src->res.linear.desc.x?1u:0u)+(src->res.linear.desc.y?1u:0u)+(src->res.linear.desc.z?1u:0u)+(src->res.linear.desc.w?1u:0u);
        int bits = src->res.linear.desc.x;
        CUarray_format fmt = CU_AD_FORMAT_FLOAT;
        switch (src->res.linear.desc.f) {
        case 0: /* signed */
            fmt = (bits == 8) ? CU_AD_FORMAT_SIGNED_INT8 : (bits == 16) ? CU_AD_FORMAT_SIGNED_INT16 : CU_AD_FORMAT_SIGNED_INT32;
            break;
        case 1: /* unsigned */
            fmt = (bits == 8) ? CU_AD_FORMAT_UNSIGNED_INT8 : (bits == 16) ? CU_AD_FORMAT_UNSIGNED_INT16 : CU_AD_FORMAT_UNSIGNED_INT32;
            break;
        case 2: /* float */
            fmt = (bits == 16) ? CU_AD_FORMAT_HALF : CU_AD_FORMAT_FLOAT;
            break;
        default:
            fmt = (bits == 16) ? CU_AD_FORMAT_HALF : CU_AD_FORMAT_FLOAT;
            break;
        }
        rd->res.linear.devPtr = (CUdeviceptr)(uintptr_t)src->res.linear.devPtr;
        rd->res.linear.format = fmt;
        rd->res.linear.numChannels = n;
        rd->res.linear.sizeInBytes = src->res.linear.sizeInBytes;
        return CUDA_SUCCESS;
    }
    if (src->resType == 3) { /* cudaResourceTypePitch2D */
        unsigned n = (src->res.pitch2D.desc.x?1u:0u)+(src->res.pitch2D.desc.y?1u:0u)+(src->res.pitch2D.desc.z?1u:0u)+(src->res.pitch2D.desc.w?1u:0u);
        int bits = src->res.pitch2D.desc.x;
        CUarray_format fmt = CU_AD_FORMAT_FLOAT;
        switch (src->res.pitch2D.desc.f) {
        case 0:
            fmt = (bits == 8) ? CU_AD_FORMAT_SIGNED_INT8 : (bits == 16) ? CU_AD_FORMAT_SIGNED_INT16 : CU_AD_FORMAT_SIGNED_INT32;
            break;
        case 1:
            fmt = (bits == 8) ? CU_AD_FORMAT_UNSIGNED_INT8 : (bits == 16) ? CU_AD_FORMAT_UNSIGNED_INT16 : CU_AD_FORMAT_UNSIGNED_INT32;
            break;
        case 2:
            fmt = (bits == 16) ? CU_AD_FORMAT_HALF : CU_AD_FORMAT_FLOAT;
            break;
        default:
            fmt = (bits == 16) ? CU_AD_FORMAT_HALF : CU_AD_FORMAT_FLOAT;
            break;
        }
        rd->res.pitch2D.devPtr = (CUdeviceptr)(uintptr_t)src->res.pitch2D.devPtr;
        rd->res.pitch2D.format = fmt;
        rd->res.pitch2D.numChannels = n;
        rd->res.pitch2D.width = src->res.pitch2D.width;
        rd->res.pitch2D.height = src->res.pitch2D.height;
        rd->res.pitch2D.pitchInBytes = src->res.pitch2D.pitchInBytes;
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject,
                                    const cudaResourceDesc *pResDesc,
                                    const cudaTextureDesc *pTexDesc,
                                    const cudaResourceViewDesc *pResViewDesc) {
    CUDA_RESOURCE_DESC rd;
    CUDA_TEXTURE_DESC td;
    CUDA_RESOURCE_VIEW_DESC vd;
    CUDA_RESOURCE_VIEW_DESC *pvd = NULL;
    CUtexObject tex = 0;
    CUresult r;
    if (!pTexObject || !pResDesc || !pTexDesc) return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
    r = acuda_cuda_resource_desc_to_cu(pResDesc, &rd);
    if (r != CUDA_SUCCESS) return (cudaError_t)r;
    memset(&td, 0, sizeof(td));
    for (int i = 0; i < 3; i++) td.addressMode[i] = (CUaddress_mode)pTexDesc->addressMode[i];
    td.filterMode = (CUfilter_mode)pTexDesc->filterMode;
    td.mipmapFilterMode = (CUfilter_mode)pTexDesc->mipmapFilterMode;
    td.flags = pTexDesc->normalizedCoords ? 0x2u : 0u;
    td.maxAnisotropy = pTexDesc->maxAnisotropy;
    td.mipmapLevelBias = pTexDesc->mipmapLevelBias;
    td.minMipmapLevelClamp = pTexDesc->minMipmapLevelClamp;
    td.maxMipmapLevelClamp = pTexDesc->maxMipmapLevelClamp;
    for (int i = 0; i < 4; i++) td.borderColor[i] = pTexDesc->borderColor[i];
    if (pResViewDesc) {
        memset(&vd, 0, sizeof(vd));
        vd.format = (CUresourceViewFormat)pResViewDesc->format;
        vd.width = pResViewDesc->width;
        vd.height = pResViewDesc->height;
        vd.depth = pResViewDesc->depth;
        vd.firstMipmapLevel = pResViewDesc->firstMipmapLevel;
        vd.lastMipmapLevel = pResViewDesc->lastMipmapLevel;
        vd.firstLayer = pResViewDesc->firstLayer;
        vd.lastLayer = pResViewDesc->lastLayer;
        pvd = &vd;
    }
    r = cuTexObjectCreate(&tex, &rd, &td, pvd);
    if (r != CUDA_SUCCESS) return (cudaError_t)r;
    *pTexObject = (cudaTextureObject_t)tex;
    return (cudaError_t)CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMipmappedArrayCreate(
    CUmipmappedArray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
    unsigned int numMipmapLevels) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    if (!pHandle || !pMipmappedArrayDesc || numMipmapLevels == 0)
        return CUDA_ERROR_INVALID_VALUE;
    uint32_t fmt;
    if (driver_array_format(pMipmappedArrayDesc->Format,
                            pMipmappedArrayDesc->NumChannels,
                            pMipmappedArrayDesc->Flags, &fmt) != 0)
        return CUDA_ERROR_NOT_SUPPORTED;
    uint32_t dim, layers, cube;
    driver_array_shape(pMipmappedArrayDesc, &dim, &layers, &cube);
    acuda_image_desc id = {
        .width = (uint32_t)pMipmappedArrayDesc->Width,
        .height = dim == 1 ? 1u : (uint32_t)pMipmappedArrayDesc->Height,
        .depth = dim == 3 ? (uint32_t)pMipmappedArrayDesc->Depth : 1u,
        .layers = layers,
        .levels = numMipmapLevels,
        .samples = 1,
        .format = fmt,
        .dim = dim,
        .cube = cube,
    };
    CUmipmappedArray m = calloc(1, sizeof(*m));
    if (!m)
        return CUDA_ERROR_OUT_OF_MEMORY;
    if (acuda_image_create(g_actx, &m->image, &id, NULL) != ACUDA_SUCCESS) {
        free(m);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    m->desc = *pMipmappedArrayDesc;
    m->levels = numMipmapLevels;
    m->is3d = dim == 3;
    *pHandle = m;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
    if (!hMipmappedArray)
        return CUDA_ERROR_INVALID_VALUE;
    acuda_free(g_actx, hMipmappedArray->image);
    free(hMipmappedArray);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMipmappedArrayGetLevel(CUarray *pLevelArray,
                                  CUmipmappedArray hMipmappedArray,
                                  unsigned int level) {
    if (!pLevelArray || !hMipmappedArray || level >= hMipmappedArray->levels)
        return CUDA_ERROR_INVALID_VALUE;
    CUarray a = calloc(1, sizeof(*a));
    if (!a)
        return CUDA_ERROR_OUT_OF_MEMORY;
    a->image = hMipmappedArray->image;
    a->own = 0;
    a->parent = NULL;
    a->level = level;
    a->desc3 = hMipmappedArray->desc;
    a->desc2.Width = hMipmappedArray->desc.Width;
    a->desc2.Height = hMipmappedArray->desc.Height;
    a->desc2.Format = hMipmappedArray->desc.Format;
    a->desc2.NumChannels = hMipmappedArray->desc.NumChannels;
    a->is3d = hMipmappedArray->is3d;
    *pLevelArray = a;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexObjectCreate(CUtexObject *pTexObject,
                           const CUDA_RESOURCE_DESC *pResDesc,
                           const CUDA_TEXTURE_DESC *pTexDesc,
                           const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    if (!pTexObject || !pResDesc)
        return CUDA_ERROR_INVALID_VALUE;
    acuda_buffer_t image = NULL;
    acuda_buffer_t own_img = NULL;
    int depth_tex = 0;
    int is_array = 0;
    if (pResDesc->resType == CU_RESOURCE_TYPE_ARRAY) {
        CUarray a = pResDesc->res.array.hArray;
        if (!a)
            return CUDA_ERROR_INVALID_VALUE;
        image = a->image;
        depth_tex = (a->desc3.Flags & CUDA_ARRAY3D_DEPTH_TEXTURE) != 0;
        is_array = 1;
    } else if (pResDesc->resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY) {
        CUmipmappedArray m = pResDesc->res.mipmap.hMipmappedArray;
        if (!m)
            return CUDA_ERROR_INVALID_VALUE;
        image = m->image;
        depth_tex = (m->desc.Flags & CUDA_ARRAY3D_DEPTH_TEXTURE) != 0;
        is_array = 1;
    } else if (pResDesc->resType == CU_RESOURCE_TYPE_LINEAR) {
        // Linear textures snapshot the backing memory into a 1D image.
        uint32_t fmt = 0;
        if (driver_array_format(pResDesc->res.linear.format,
                                pResDesc->res.linear.numChannels, 0,
                                &fmt) != 0)
            return CUDA_ERROR_NOT_SUPPORTED;
        size_t elem = (size_t)driver_elem_bytes(fmt);
        size_t bytes = pResDesc->res.linear.sizeInBytes;
        if (elem == 0 || bytes < elem)
            return CUDA_ERROR_INVALID_VALUE;
        void *host = NULL;
        if (host_for(pResDesc->res.linear.devPtr, bytes, &host) !=
            CUDA_SUCCESS)
            return CUDA_ERROR_INVALID_VALUE;
        acuda_image_desc id = {
            .width = (uint32_t)(bytes / elem),
            .height = 1,
            .depth = 1,
            .layers = 1,
            .levels = 1,
            .samples = 1,
            .format = fmt,
            .dim = 1,
            .cube = 0,
        };
        acuda_status_t icst = acuda_image_create(g_actx, &own_img, &id,
                                                         host);
        if (icst != ACUDA_SUCCESS) {
            ACUDA_DEBUG_PRINT("acuda-trace ?? cuTexObjectCreate(linear): image create failed st=%d w=%u fmt=%d bytes=%zu\n",
                        (int)icst, (unsigned)id.width, (int)fmt, bytes);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        image = own_img;
    } else if (pResDesc->resType == CU_RESOURCE_TYPE_PITCH2D) {
        uint32_t fmt = 0;
        if (driver_array_format(pResDesc->res.pitch2D.format,
                                pResDesc->res.pitch2D.numChannels, 0,
                                &fmt) != 0)
            return CUDA_ERROR_NOT_SUPPORTED;
        size_t elem = (size_t)driver_elem_bytes(fmt);
        size_t w = pResDesc->res.pitch2D.width;
        size_t h = pResDesc->res.pitch2D.height;
        size_t pitch = pResDesc->res.pitch2D.pitchInBytes;
        size_t row = w * elem;
        if (elem == 0 || w == 0 || h == 0 || row > pitch)
            return CUDA_ERROR_INVALID_VALUE;
        void *host = NULL;
        if (host_for(pResDesc->res.pitch2D.devPtr, pitch * h, &host) !=
            CUDA_SUCCESS)
            return CUDA_ERROR_INVALID_VALUE;
        unsigned char *tight = malloc(row * h);
        if (!tight)
            return CUDA_ERROR_OUT_OF_MEMORY;
        for (size_t y = 0; y < h; y++)
            memcpy(tight + y * row,
                   (const unsigned char *)host + y * pitch, row);
        acuda_image_desc id = {
            .width = (uint32_t)w,
            .height = (uint32_t)h,
            .depth = 1,
            .layers = 1,
            .levels = 1,
            .samples = 1,
            .format = fmt,
            .dim = 2,
            .cube = 0,
        };
        acuda_status_t ist = acuda_image_create(g_actx, &own_img, &id,
                                                tight);
        free(tight);
        if (ist != ACUDA_SUCCESS)
            return CUDA_ERROR_OUT_OF_MEMORY;
        image = own_img;
    } else {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    acuda_sampler_desc sd;
    memset(&sd, 0, sizeof(sd));
    CUDA_TEXTURE_DESC td;
    if (pTexDesc)
        td = *pTexDesc;
    else
        memset(&td, 0, sizeof(td));
    sd.filter = td.filterMode == CU_TR_FILTER_MODE_LINEAR ? 1u : 0u;
    sd.mip_filter = td.mipmapFilterMode == CU_TR_FILTER_MODE_LINEAR ? 1u : 0u;
    for (int i = 0; i < 3; i++) {
        switch (td.addressMode[i]) {
        case CU_TR_ADDRESS_MODE_CLAMP: sd.addr[i] = 2; break;
        case CU_TR_ADDRESS_MODE_MIRROR: sd.addr[i] = 1; break;
        case CU_TR_ADDRESS_MODE_BORDER: sd.addr[i] = 3; break;
        default: sd.addr[i] = 0; break;
        }
    }
    // CUDA arrays support unnormalized coordinates; honour the flag for all.
    sd.normalized = (td.flags & 0x2u) ? 1u : 0u;
    sd.compare = depth_tex ? 1u : 0u;
    sd.compare_op = 3; // LESS_OR_EQUAL for depth texture references
    for (int i = 0; i < 4; i++)
        sd.border[i] = td.borderColor[i];

    acuda_buffer_t smp = NULL, tex = NULL;
    if (acuda_sampler_create(g_actx, &smp, &sd) != ACUDA_SUCCESS) {
        if (own_img)
            acuda_free(g_actx, own_img);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (acuda_texture_create_image(g_actx, &tex, image, smp) !=
        ACUDA_SUCCESS) {
        acuda_free(g_actx, smp);
        if (own_img)
            acuda_free(g_actx, own_img);
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (g_ntexobjs == g_ctexobjs) {
        size_t ncap = g_ctexobjs ? g_ctexobjs * 2 : 8;
        driver_texobj_t *n = realloc(g_texobjs, sizeof(*n) * ncap);
        if (!n) {
            acuda_free(g_actx, tex);
            acuda_free(g_actx, smp);
            if (own_img)
                acuda_free(g_actx, own_img);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        g_texobjs = n;
        g_ctexobjs = ncap;
    }
    driver_texobj_t *e = &g_texobjs[g_ntexobjs++];
    memset(e, 0, sizeof(*e));
    e->id = g_next_texid++;
    if (e->id == 0)
        e->id = g_next_texid++;
    e->tex = tex;
    e->sampler = smp;
    e->own_image = own_img;
    e->res = *pResDesc;
    e->td = td;
    if (pResViewDesc) {
        e->resView = *pResViewDesc;
        e->has_resView = 1;
    } else {
        memset(&e->resView, 0, sizeof(e->resView));
        e->has_resView = 0;
    }
    *pTexObject = e->id;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexObjectDestroy(CUtexObject texObject) {
    for (size_t i = 0; i < g_ntexobjs; i++) {
        if (g_texobjs[i].id != texObject)
            continue;
        acuda_free(g_actx, g_texobjs[i].tex);
        acuda_free(g_actx, g_texobjs[i].sampler);
        if (g_texobjs[i].own_image)
            acuda_free(g_actx, g_texobjs[i].own_image);
        g_texobjs[i] = g_texobjs[--g_ntexobjs];
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_HANDLE;
}

CUresult ACUDA_CUDAAPI cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                    CUtexObject texObject) {
    driver_texobj_t *e = driver_tex_find(texObject);
    if (!pResDesc || !e)
        return CUDA_ERROR_INVALID_VALUE;
    *pResDesc = e->res;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc,
                                   CUtexObject texObject) {
    driver_texobj_t *e = driver_tex_find(texObject);
    if (!pTexDesc || !e)
        return CUDA_ERROR_INVALID_VALUE;
    *pTexDesc = e->td;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc,
                                                      CUtexObject texObject) {
    driver_texobj_t *e = driver_tex_find(texObject);
    if (!pResViewDesc || !e)
        return CUDA_ERROR_INVALID_VALUE;
    if (e->has_resView)
        *pResViewDesc = e->resView;
    else
        memset(pResViewDesc, 0, sizeof(*pResViewDesc));
    return CUDA_SUCCESS;
}

// ---- legacy texture references (CUDA 4.x / PhysX 2.8) ----------------

static CUresult texref_rebuild(CUtexref h, int force) {
    if (!h)
        return CUDA_ERROR_INVALID_VALUE;
    if (h->is_array) {
        if (h->obj) {
            cuTexObjectDestroy(h->obj);
            h->obj = 0;
        }
        h->built_addr = 0;
        h->built_bytes = 0;
    } else if (h->bytes == 0 || !h->fmt_set) {
        if (h->obj) {
            cuTexObjectDestroy(h->obj);
            h->obj = 0;
        }
        h->built_addr = 0;
        h->built_bytes = 0;
        return CUDA_SUCCESS;
    } else if (!force && h->obj && h->built_addr == h->addr &&
               h->built_bytes == h->bytes && h->built_fmt == h->fmt &&
               h->built_channels == h->channels) {
        // Bind state unchanged (filter/flag/mipmap re-set): keep the
        // existing snapshot instead of re-uploading every frame.
        return CUDA_SUCCESS;
    } else {
        if (h->obj) {
            cuTexObjectDestroy(h->obj);
            h->obj = 0;
        }
        h->built_addr = 0;
        h->built_bytes = 0;
    }
    /* The linear snapshot below copies from the host-visible mirror.
       Make sure previously submitted work (e.g. a sort that just filled
       this buffer) has completed first; otherwise the image captures
       stale bytes and kernels hang on the degenerate data. Only needed
       when actually (re)building; unchanged binds keep their snapshot. */
    if (!h->is_array && h->bytes != 0 && h->addr != 0) {
        CUcontext cc = NULL;
        if (need_ctx(&cc) == CUDA_SUCCESS)
            acuda_stream_synchronize(g_actx, NULL);
    }
    CUDA_RESOURCE_DESC rd;
    memset(&rd, 0, sizeof(rd));
    if (h->is_array) {
        if (!h->array)
            return CUDA_SUCCESS;
        rd.resType = CU_RESOURCE_TYPE_ARRAY;
        rd.res.array.hArray = h->array;
    } else {
        rd.resType = CU_RESOURCE_TYPE_LINEAR;
        rd.res.linear.devPtr = h->addr;
        rd.res.linear.format = h->fmt;
        rd.res.linear.numChannels = (unsigned int)h->channels;
        rd.res.linear.sizeInBytes = h->bytes;
    }
    CUDA_TEXTURE_DESC td;
    memset(&td, 0, sizeof(td));
    td.addressMode[0] = h->address_mode[0];
    td.addressMode[1] = h->address_mode[1];
    td.addressMode[2] = h->address_mode[2];
    td.filterMode = h->filter;
    td.flags = h->flags;
    td.maxAnisotropy = h->max_aniso ? h->max_aniso : 1;
    td.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    td.borderColor[0] = td.borderColor[1] = td.borderColor[2] = td.borderColor[3] = 0.0f;
    CUresult tcr = cuTexObjectCreate(&h->obj, &rd, &td, NULL);
    ACUDA_DEBUG_PRINT("acuda-trace ?? texref_rebuild(%s) create rc=%d obj=%llx\n",
                h->name, (int)tcr, (unsigned long long)(uint64_t)h->obj);
    if (tcr == CUDA_SUCCESS && !h->is_array) {
        h->built_addr = h->addr;
        h->built_bytes = h->bytes;
        h->built_fmt = h->fmt;
        h->built_channels = h->channels;
    }
    return tcr;
}

CUresult ACUDA_CUDAAPI cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod,
                                          const char *name) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuModuleGetTexRef(%s) mod=%p ntexrefs=%zu\n",
                name ? name : "(null)", (void *)hmod,
                hmod ? hmod->ntexrefs : 0);
    if (!pTexRef || !hmod || !name)
        return CUDA_ERROR_INVALID_VALUE;
    *pTexRef = NULL;
    for (size_t i = 0; i < hmod->ntexrefs; i++) {
        if (hmod->texrefs[i] &&
            strcmp(hmod->texrefs[i]->name, name) == 0) {
            *pTexRef = hmod->texrefs[i];
            return CUDA_SUCCESS;
        }
    }
    CUtexref tr = calloc(1, sizeof(*tr));
    if (!tr)
        return CUDA_ERROR_OUT_OF_MEMORY;
    tr->mod = hmod;
    strncpy(tr->name, name, sizeof(tr->name) - 1);
    tr->filter = CU_TR_FILTER_MODE_POINT;
    tr->address_mode[0] = tr->address_mode[1] = tr->address_mode[2] =
        CU_TR_ADDRESS_MODE_WRAP;
    tr->max_aniso = 1;
    if (hmod->ntexrefs == hmod->texrefs_cap) {
        size_t ncap = hmod->texrefs_cap ? hmod->texrefs_cap * 2 : 4;
        void *np = realloc(hmod->texrefs, ncap * sizeof(*hmod->texrefs));
        if (!np) {
            free(tr);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        hmod->texrefs = np;
        hmod->texrefs_cap = ncap;
    }
    hmod->texrefs[hmod->ntexrefs++] = tr;
    *pTexRef = tr;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefSetAddress_v2(size_t *ByteOffset,
                                             CUtexref hTexRef,
                                             CUdeviceptr dptr, size_t bytes) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuTexRefSetAddress(%s mod=%p dptr=%llx bytes=%zu)\n",
                hTexRef ? hTexRef->name : "(null)",
                (void *)(hTexRef ? hTexRef->mod : NULL),
                (unsigned long long)dptr, bytes);
    if (!hTexRef)
        return CUDA_ERROR_INVALID_HANDLE;
    if (ByteOffset)
        *ByteOffset = 0;
    if (bytes == 0 && dptr != 0) {
        // cudart's texture bind path passes size 0; real CUDA binds to the
        // end of the allocation. Probe the arena for the same extent, but
        // cap the snapshot: PhysX sub-allocates particle buffers from one
        // 256 MiB heap, so the raw remainder would build a ~250 MiB image
        // per texture and exhaust device memory. 16 MiB covers ~1M float4
        // particles, far above FluidMark's range.
        size_t cap = 0;
        if (acuda_arena_capacity((uint64_t)dptr, &cap) == 0 && cap != 0) {
            size_t want = cap > (16u << 20) ? (16u << 20) : cap;
            ACUDA_DEBUG_PRINT("acuda-trace ?? cuTexRefSetAddress(%s) size0 -> cap=%zu use=%zu\n",
                        hTexRef->name, cap, want);
            bytes = want;
        }
    }
    hTexRef->addr = dptr;
    hTexRef->bytes = bytes;
    hTexRef->is_array = 0;
    hTexRef->array = NULL;
    {
        // Diagnostic: snapshot the buffer a texture is bound to, named after
        // the texture symbol.  The arena is persistently mapped, so this is a
        // host copy with no GPU round trip.  Used to inspect the fluid cell
        // table (inputCellStartTex / inputCellEndTex), which is only ever
        // reachable through a texture binding.
        const char *dir = getenv("ACUDA_DUMP_TEXREF");
        if (dir && *dir && dptr && bytes) {
            void *host = NULL;
            size_t cap = 0;
            if (acuda_arena_capacity((uint64_t)dptr, &cap) != 0)
                cap = bytes;
            size_t take = bytes < cap ? bytes : cap;
            if (take > (4u << 20))
                take = (4u << 20);
            int ok = take && acuda_arena_host((uint64_t)dptr, take, &host) == 0;
            ACUDA_DEBUG_PRINT("acuda-texref dump %s addr=%llx bytes=%zu take=%zu ok=%d\n",
                              hTexRef->name ? hTexRef->name : "tex",
                              (unsigned long long)dptr, bytes, take, ok);
            if (ok && host) {
                char path[1024];
                // One file per texture name, overwritten on every bind: the
                // interesting state is the last one before a kernel hangs.
                snprintf(path, sizeof(path), "%s/%s.bin", dir,
                         hTexRef->name ? hTexRef->name : "tex");
                FILE *f = fopen(path, "wb");
                if (f) {
                    fwrite(host, 1, take, f);
                    fclose(f);
                }
            }
        }
    }
    return texref_rebuild(hTexRef, 1);
}

CUresult ACUDA_CUDAAPI cuTexRefSetAddress(size_t *ByteOffset,
                                          CUtexref hTexRef,
                                          CUdeviceptr dptr, size_t bytes) {
    return cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes);
}

CUresult ACUDA_CUDAAPI cuTexRefSetArray(CUtexref hTexRef, CUarray hArray,
                                        unsigned int Flags) {
    (void)Flags;
    if (!hTexRef || !hArray)
        return CUDA_ERROR_INVALID_VALUE;
    hTexRef->array = hArray;
    hTexRef->is_array = 1;
    hTexRef->addr = 0;
    hTexRef->bytes = 0;
    return texref_rebuild(hTexRef, 0);
}

CUresult ACUDA_CUDAAPI cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt,
                                         int NumPackedComponents) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuTexRefSetFormat(%s fmt=%d nch=%d)\n",
                hTexRef ? hTexRef->name : "(null)", (int)fmt,
                NumPackedComponents);
    if (!hTexRef || NumPackedComponents < 1 || NumPackedComponents > 4)
        return CUDA_ERROR_INVALID_VALUE;
    hTexRef->fmt = fmt;
    hTexRef->channels = NumPackedComponents;
    hTexRef->fmt_set = 1;
    return texref_rebuild(hTexRef, 0);
}

CUresult ACUDA_CUDAAPI cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
    if (!hTexRef)
        return CUDA_ERROR_INVALID_HANDLE;
    hTexRef->flags = Flags;
    return texref_rebuild(hTexRef, 0);
}

CUresult ACUDA_CUDAAPI cuTexRefSetFilterMode(CUtexref hTexRef,
                                             CUfilter_mode fm) {
    if (!hTexRef || (fm != CU_TR_FILTER_MODE_POINT &&
                     fm != CU_TR_FILTER_MODE_LINEAR))
        return CUDA_ERROR_INVALID_VALUE;
    hTexRef->filter = fm;
    return texref_rebuild(hTexRef, 0);
}

CUresult ACUDA_CUDAAPI cuTexRefSetAddressMode(CUtexref hTexRef, int dim,
                                              CUaddress_mode am) {
    if (!hTexRef || dim < 0 || dim > 2 || am > CU_TR_ADDRESS_MODE_BORDER)
        return CUDA_ERROR_INVALID_VALUE;
    hTexRef->address_mode[dim] = am;
    return texref_rebuild(hTexRef, 0);
}

CUresult ACUDA_CUDAAPI cuTexRefSetMaxAnisotropy(CUtexref hTexRef,
                                                unsigned int maxAniso) {
    if (!hTexRef)
        return CUDA_ERROR_INVALID_HANDLE;
    hTexRef->max_aniso = maxAniso;
    return texref_rebuild(hTexRef, 0);
}

CUresult ACUDA_CUDAAPI cuTexRefGetAddress_v2(CUdeviceptr *pdptr,
                                             CUtexref hTexRef) {
    if (!pdptr || !hTexRef)
        return CUDA_ERROR_INVALID_VALUE;
    *pdptr = hTexRef->addr;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefGetAddress(CUdeviceptr *pdptr,
                                          CUtexref hTexRef) {
    return cuTexRefGetAddress_v2(pdptr, hTexRef);
}

CUresult ACUDA_CUDAAPI cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
    if (!phArray || !hTexRef)
        return CUDA_ERROR_INVALID_VALUE;
    *phArray = hTexRef->array;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefGetFormat(CUarray_format *pFormat,
                                         int *pNumChannels, CUtexref hTexRef) {
    if (!pFormat || !pNumChannels || !hTexRef)
        return CUDA_ERROR_INVALID_VALUE;
    *pFormat = hTexRef->fmt;
    *pNumChannels = hTexRef->channels;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefGetFlags(unsigned int *pFlags,
                                        CUtexref hTexRef) {
    if (!pFlags || !hTexRef)
        return CUDA_ERROR_INVALID_VALUE;
    *pFlags = hTexRef->flags;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefGetFilterMode(CUfilter_mode *pfm,
                                             CUtexref hTexRef) {
    if (!pfm || !hTexRef)
        return CUDA_ERROR_INVALID_VALUE;
    *pfm = hTexRef->filter;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefGetAddressMode(CUaddress_mode *pam,
                                              CUtexref hTexRef, int dim) {
    if (!pam || !hTexRef || dim < 0 || dim > 2)
        return CUDA_ERROR_INVALID_VALUE;
    *pam = hTexRef->address_mode[dim];
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefGetMaxAnisotropy(int *pmaxAniso,
                                                CUtexref hTexRef) {
    if (!pmaxAniso || !hTexRef)
        return CUDA_ERROR_INVALID_VALUE;
    *pmaxAniso = (int)hTexRef->max_aniso;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuSurfObjectCreate(CUsurfObject *pSurfObject,
                            const CUDA_RESOURCE_DESC *pResDesc) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    if (!pSurfObject || !pResDesc)
        return CUDA_ERROR_INVALID_VALUE;
    acuda_buffer_t image = NULL;
    if (pResDesc->resType == CU_RESOURCE_TYPE_ARRAY) {
        CUarray a = pResDesc->res.array.hArray;
        if (!a || !a->image)
            return CUDA_ERROR_INVALID_VALUE;
        image = a->image;
    } else if (pResDesc->resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY) {
        CUmipmappedArray m = pResDesc->res.mipmap.hMipmappedArray;
        if (!m || !m->image)
            return CUDA_ERROR_INVALID_VALUE;
        image = m->image;
    } else {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    // Only color arrays get VK_IMAGE_USAGE_STORAGE_BIT; depth, cube and
    // multisampled images cannot back a surface object.
    if (image->kind != 2 || !image->is_surface)
        return CUDA_ERROR_NOT_SUPPORTED;
    if (g_nsurfobjs == g_csurfobjs) {
        size_t ncap = g_csurfobjs ? g_csurfobjs * 2 : 8;
        driver_surfobj_t *n = realloc(g_surfobjs, sizeof(*n) * ncap);
        if (!n)
            return CUDA_ERROR_OUT_OF_MEMORY;
        g_surfobjs = n;
        g_csurfobjs = ncap;
    }
    uint32_t slot = 0;
    for (uint32_t i = 1; i < g_surf_slot_cap; i++)
        if (!g_surf_slots[i]) {
            slot = i;
            break;
        }
    if (slot == 0) {
        uint32_t ncap = g_surf_slot_cap ? g_surf_slot_cap * 2 : 16;
        acuda_buffer_t *ns =
            realloc(g_surf_slots, sizeof(*ns) * (size_t)ncap);
        if (!ns)
            return CUDA_ERROR_OUT_OF_MEMORY;
        memset(ns + g_surf_slot_cap, 0,
               sizeof(*ns) * (size_t)(ncap - g_surf_slot_cap));
        slot = g_surf_slot_cap ? g_surf_slot_cap : 1;
        g_surf_slots = ns;
        g_surf_slot_cap = ncap;
    }
    driver_surfobj_t *e = &g_surfobjs[g_nsurfobjs++];
    memset(e, 0, sizeof(*e));
    e->id = slot;
    e->slot = slot;
    e->image = image;
    e->res = *pResDesc;
    g_surf_slots[slot] = image;
    *pSurfObject = e->id;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuSurfObjectDestroy(CUsurfObject surfObject) {
    for (size_t i = 0; i < g_nsurfobjs; i++) {
        if (g_surfobjs[i].id != surfObject)
            continue;
        // The array owns the image; surface objects are borrowed views.
        if (g_surfobjs[i].slot < g_surf_slot_cap)
            g_surf_slots[g_surfobjs[i].slot] = NULL;
        g_surfobjs[i] = g_surfobjs[--g_nsurfobjs];
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_HANDLE;
}

CUresult ACUDA_CUDAAPI cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                     CUsurfObject surfObject) {
    driver_surfobj_t *e = driver_surf_find(surfObject);
    if (!pResDesc || !e)
        return CUDA_ERROR_INVALID_VALUE;
    *pResDesc = e->res;
    return CUDA_SUCCESS;
}


CUresult ACUDA_CUDAAPI cuTexRefCreate(CUtexref *pTexRef) {
    if (!pTexRef)
        return CUDA_ERROR_INVALID_VALUE;
    CUtexref tr = calloc(1, sizeof(*tr));
    if (!tr)
        return CUDA_ERROR_OUT_OF_MEMORY;
    snprintf(tr->name, sizeof(tr->name), "orphan");
    *pTexRef = tr;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefDestroy(CUtexref hTexRef) {
    if (!hTexRef)
        return CUDA_ERROR_INVALID_HANDLE;
    free(hTexRef);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pDesc,
                                               CUarray hArray) {
    return cuArrayGetDescriptor(pDesc, hArray);
}

CUresult ACUDA_CUDAAPI
cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pDesc, CUarray hArray) {
    return cuArray3DGetDescriptor(pDesc, hArray);
}

void driver_img_type(acuda_buffer_t img, int *dim, int *arrayed,
                            int *isint) {
    *dim = img->dim3d ? 2 : (img->dim1d ? 0 : 1);
    *arrayed = !img->dim3d && img->layers > 1;
    *isint = img->format == 3 ? 1 : (img->format == 0 ? 0 : -1);
}

driver_texobj_t *driver_tex_find(CUtexObject id) {
    for (size_t i = 0; i < g_ntexobjs; i++)
        if (g_texobjs[i].id == id)
            return &g_texobjs[i];
    return NULL;
}

driver_surfobj_t *driver_surf_find(CUsurfObject id) {
    for (size_t i = 0; i < g_nsurfobjs; i++)
        if (g_surfobjs[i].id == id)
            return &g_surfobjs[i];
    return NULL;
}
