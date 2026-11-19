#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_42d85a81.h"

#include <string.h>

/* {42D85A81-23F6-CB47-8298-F6E78A3AECDC} */
static const unsigned char acuda_export_guid_42d85a81[16] = {
    0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47,
    0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc
};

int acuda_export_table_is_42d85a81(const struct CUuuid_st *id) {
    return id && memcmp(id->bytes, acuda_export_guid_42d85a81,
                        sizeof(acuda_export_guid_42d85a81)) == 0;
}

CUresult acuda_export_table_get_42d85a81(const void **ppExportTable) {
    if (!ppExportTable)
        return CUDA_ERROR_INVALID_VALUE;
    *ppExportTable = acuda_export_42d85a81_table();
    ACUDA_DEBUG_PRINT("ACuda-debug: cuGetExportTable -> SUCCESS (tools tls) table=%p\n",
                      *ppExportTable);
    return CUDA_SUCCESS;
}
