#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_21318c60.h"

#include <string.h>

/* {21318C60-9714-3248-8CA6-41FF7324C8F2} */
static const unsigned char acuda_export_guid_21318c60[16] = {
    0x21, 0x31, 0x8c, 0x60, 0x97, 0x14, 0x32, 0x48,
    0x8c, 0xa6, 0x41, 0xff, 0x73, 0x24, 0xc8, 0xf2
};

int acuda_export_table_is_21318c60(const struct CUuuid_st *id) {
    return id && memcmp(id->bytes, acuda_export_guid_21318c60,
                        sizeof(acuda_export_guid_21318c60)) == 0;
}

CUresult acuda_export_table_get_21318c60(const void **ppExportTable) {
    if (!ppExportTable)
        return CUDA_ERROR_INVALID_VALUE;
    *ppExportTable = acuda_export_21318c60_table();
    ACUDA_DEBUG_PRINT("ACuda-debug: cuGetExportTable -> SUCCESS (cuda 13.4 tools) table=%p\n",
                      *ppExportTable);
    return CUDA_SUCCESS;
}
