#ifndef ACUDA_DRIVER_EXPORT_A094798C_H
#define ACUDA_DRIVER_EXPORT_A094798C_H

#include "acuda/driver.h"

#ifdef __cplusplus
extern "C" {
#endif

/* {A094798C-2E74-2E74-93F2-0800200C0A66} tools runtime hooks.  CUDA 4.x
 * cudart hands these callbacks to the profiler/debugger layer during
 * context setup; the table is seven slots, [0] being its size. */

int acuda_export_table_is_a094798c(const struct CUuuid_st *id);
CUresult acuda_export_table_get_a094798c(const void **ppExportTable);

const void *acuda_export_a094798c_table(void);

#ifdef __cplusplus
}
#endif

#endif /* ACUDA_DRIVER_EXPORT_A094798C_H */
