#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_6bd5fb6c.h"

#include <string.h>

/* {6BD5FB6C-5BF4-E74A-8987-D93912FD9DF9} */
static const unsigned char acuda_export_guid_6bd5fb6c[16] = {
    0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
    0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9
};

int acuda_export_table_is_6bd5fb6c(const struct CUuuid_st *id) {
    return id && memcmp(id->bytes, acuda_export_guid_6bd5fb6c,
                        sizeof(acuda_export_guid_6bd5fb6c)) == 0;
}

CUresult acuda_export_table_get_6bd5fb6c(const void **ppExportTable) {
    if (!ppExportTable)
        return CUDA_ERROR_INVALID_VALUE;
    *ppExportTable = acuda_export_6bd5fb6c_table();
    ACUDA_DEBUG_PRINT("ACuda-debug: cuGetExportTable -> SUCCESS (cudart interface) table=%p\n",
                      *ppExportTable);
    return CUDA_SUCCESS;
}
