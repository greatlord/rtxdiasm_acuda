#ifndef ACUDA_DRIVER_EXPORT_6BD5FB6C_H
#define ACUDA_DRIVER_EXPORT_6BD5FB6C_H

#include "acuda/driver.h"

#ifdef __cplusplus
extern "C" {
#endif

/* {6BD5FB6C-5BF4-E74A-8987-D93912FD9DF9} driver/runtime interface table.
 * CUDA 4.x/6.x cudart fetches this 13-slot table with cuGetExportTable
 * instead of resolving the handful of driver entry points it needs. */

int acuda_export_table_is_6bd5fb6c(const struct CUuuid_st *id);
CUresult acuda_export_table_get_6bd5fb6c(const void **ppExportTable);

/* Table pointer handed to the runtime (slot 5 is filled lazily with this
 * module's base so cudart can resolve driver entry points by name). */
const void *acuda_export_6bd5fb6c_table(void);

#ifdef __cplusplus
}
#endif

#endif /* ACUDA_DRIVER_EXPORT_6BD5FB6C_H */
