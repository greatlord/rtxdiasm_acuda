#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_a094798c.h"

#include <string.h>

/* {A094798C-2E74-2E74-93F2-0800200C0A66} */
static const unsigned char acuda_export_guid_a094798c[16] = {
    0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74,
    0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66
};

int acuda_export_table_is_a094798c(const struct CUuuid_st *id) {
    return id && memcmp(id->bytes, acuda_export_guid_a094798c,
                        sizeof(acuda_export_guid_a094798c)) == 0;
}

CUresult acuda_export_table_get_a094798c(const void **ppExportTable) {
    if (!ppExportTable)
        return CUDA_ERROR_INVALID_VALUE;
    *ppExportTable = acuda_export_a094798c_table();
    ACUDA_DEBUG_PRINT("ACuda-debug: cuGetExportTable -> SUCCESS (tools hooks) table=%p\n",
                      *ppExportTable);
    return CUDA_SUCCESS;
}
