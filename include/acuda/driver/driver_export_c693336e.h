#ifndef ACUDA_DRIVER_EXPORT_C693336E_H
#define ACUDA_DRIVER_EXPORT_C693336E_H

#include "acuda/driver.h"

#ifdef __cplusplus
extern "C" {
#endif

/* {C693336E-1121-DF11-A8C3-68F355D89593} context-local storage.  The
 * tools layer attaches small records to a context through this table;
 * three function pointers, no size slot. */

int acuda_export_table_is_c693336e(const struct CUuuid_st *id);
CUresult acuda_export_table_get_c693336e(const void **ppExportTable);

const void *acuda_export_c693336e_table(void);

#ifdef __cplusplus
}
#endif

#endif /* ACUDA_DRIVER_EXPORT_C693336E_H */
