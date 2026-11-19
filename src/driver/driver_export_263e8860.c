#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_263e8860.h"

#include <string.h>

/* {263E8860-7CD2-6143-92F6-BBD5006DFA7E} */
static const unsigned char acuda_export_guid_263e8860[16] = {
    0x26, 0x3e, 0x88, 0x60, 0x7c, 0xd2, 0x61, 0x43,
    0x92, 0xf6, 0xbb, 0xd5, 0x00, 0x6d, 0xfa, 0x7e
};

int acuda_export_table_is_263e8860(const struct CUuuid_st *id) {
    return id && memcmp(id->bytes, acuda_export_guid_263e8860,
                        sizeof(acuda_export_guid_263e8860)) == 0;
}

CUresult acuda_export_table_get_263e8860(const void **ppExportTable) {
    if (!ppExportTable)
        return CUDA_ERROR_INVALID_VALUE;
    *ppExportTable = acuda_export_263e8860_table();
    ACUDA_DEBUG_PRINT("ACuda-debug: cuGetExportTable -> SUCCESS (ctx checks) table=%p\n",
                      *ppExportTable);
    return CUDA_SUCCESS;
}
