#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_c693336e.h"

#include <string.h>

/* {C693336E-1121-DF11-A8C3-68F355D89593} */
static const unsigned char acuda_export_guid_c693336e[16] = {
    0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11,
    0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93
};

int acuda_export_table_is_c693336e(const struct CUuuid_st *id) {
    return id && memcmp(id->bytes, acuda_export_guid_c693336e,
                        sizeof(acuda_export_guid_c693336e)) == 0;
}

CUresult acuda_export_table_get_c693336e(const void **ppExportTable) {
    if (!ppExportTable)
        return CUDA_ERROR_INVALID_VALUE;
    *ppExportTable = acuda_export_c693336e_table();
    ACUDA_DEBUG_PRINT("ACuda-debug: cuGetExportTable -> SUCCESS (ctx storage) table=%p\n",
                      *ppExportTable);
    return CUDA_SUCCESS;
}
