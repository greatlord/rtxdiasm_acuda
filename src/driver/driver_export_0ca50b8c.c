#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_0ca50b8c.h"

#include <string.h>

/* {0CA50B8C-1004-929A-89A7-D0DF10E77286} */
static const unsigned char acuda_export_guid_0ca50b8c[16] = {
    0x0c, 0xa5, 0x0b, 0x8c, 0x10, 0x04, 0x92, 0x9a,
    0x89, 0xa7, 0xd0, 0xdf, 0x10, 0xe7, 0x72, 0x86
};

int acuda_export_table_is_0ca50b8c(const struct CUuuid_st *id) {
    return id && memcmp(id->bytes, acuda_export_guid_0ca50b8c,
                        sizeof(acuda_export_guid_0ca50b8c)) == 0;
}

CUresult acuda_export_table_get_0ca50b8c(const void **ppExportTable) {
    if (!ppExportTable)
        return CUDA_ERROR_INVALID_VALUE;
    *ppExportTable = acuda_export_0ca50b8c_table();
    ACUDA_DEBUG_PRINT("ACuda-debug: cuGetExportTable -> SUCCESS (ctx create bypass) table=%p\n",
                      *ppExportTable);
    return CUDA_SUCCESS;
}
