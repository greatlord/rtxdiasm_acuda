#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_195bcbf4.h"

#include <string.h>

/* {195BCBF4-D67D-024A-ACC5-1D29CEA631AE} */
static const unsigned char acuda_export_guid_195bcbf4[16] = {
    0x19, 0x5b, 0xcb, 0xf4, 0xd6, 0x7d, 0x02, 0x4a,
    0xac, 0xc5, 0x1d, 0x29, 0xce, 0xa6, 0x31, 0xae
};

int acuda_export_table_is_195bcbf4(const struct CUuuid_st *id) {
    return id && memcmp(id->bytes, acuda_export_guid_195bcbf4,
                        sizeof(acuda_export_guid_195bcbf4)) == 0;
}

CUresult acuda_export_table_get_195bcbf4(const void **ppExportTable) {
    if (!ppExportTable)
        return CUDA_ERROR_INVALID_VALUE;
    *ppExportTable = acuda_export_195bcbf4_table();
    ACUDA_DEBUG_PRINT("ACuda-debug: cuGetExportTable -> SUCCESS (heap access) table=%p\n",
                      *ppExportTable);
    return CUDA_SUCCESS;
}
