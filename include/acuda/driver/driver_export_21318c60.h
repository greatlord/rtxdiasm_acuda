#ifndef ACUDA_DRIVER_EXPORT_21318C60_H
#define ACUDA_DRIVER_EXPORT_21318C60_H

#include "acuda/driver.h"

#ifdef __cplusplus
extern "C" {
#endif

/* {21318C60-9714-3248-8CA6-41FF7324C8F2} CUDA 13.4 tools table.  Only
 * the size slot is needed for initialisation; the optional tool paths
 * that use the remaining entries are not reached by our runtime. */

int acuda_export_table_is_21318c60(const struct CUuuid_st *id);
CUresult acuda_export_table_get_21318c60(const void **ppExportTable);

const void *acuda_export_21318c60_table(void);

#ifdef __cplusplus
}
#endif

#endif /* ACUDA_DRIVER_EXPORT_21318C60_H */
