#ifndef ACUDA_DRIVER_EXPORT_D4082055_H
#define ACUDA_DRIVER_EXPORT_D4082055_H

#include "acuda/driver.h"

#ifdef __cplusplus
extern "C" {
#endif

/* {D4082055-BDE6-704B-8D34-BA123C66E1F2} integrity check.  CUDA 13
 * recomputes this checksum to confirm that the driver and runtime
 * interface tables agree; without a correct answer every runtime call
 * fails with an initialisation error. */

int acuda_export_table_is_d4082055(const struct CUuuid_st *id);
CUresult acuda_export_table_get_d4082055(const void **ppExportTable);

const void *acuda_export_d4082055_table(void);

#ifdef __cplusplus
}
#endif

#endif /* ACUDA_DRIVER_EXPORT_D4082055_H */
