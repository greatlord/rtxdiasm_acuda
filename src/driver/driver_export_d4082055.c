#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_d4082055.h"

#include <string.h>

/* {D4082055-BDE6-704B-8D34-BA123C66E1F2} */
static const unsigned char acuda_export_guid_d4082055[16] = {
    0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b,
    0x8d, 0x34, 0xba, 0x12, 0x3c, 0x66, 0xe1, 0xf2
};

int acuda_export_table_is_d4082055(const struct CUuuid_st *id) {
    return id && memcmp(id->bytes, acuda_export_guid_d4082055,
                        sizeof(acuda_export_guid_d4082055)) == 0;
}

CUresult acuda_export_table_get_d4082055(const void **ppExportTable) {
    if (!ppExportTable)
        return CUDA_ERROR_INVALID_VALUE;
    *ppExportTable = acuda_export_d4082055_table();
    ACUDA_DEBUG_PRINT("ACuda-debug: cuGetExportTable -> SUCCESS (integrity) table=%p\n",
                      *ppExportTable);
    return CUDA_SUCCESS;
}
