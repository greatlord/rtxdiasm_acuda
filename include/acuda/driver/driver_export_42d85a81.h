#ifndef ACUDA_DRIVER_EXPORT_42D85A81_H
#define ACUDA_DRIVER_EXPORT_42D85A81_H

#include "acuda/driver.h"

#ifdef __cplusplus
extern "C" {
#endif

/* {42D85A81-23F6-CB47-8298-F6E78A3AECDC} tools thread-local storage.  A
 * four-slot table; [0] is its size and the remaining entries are unused
 * for our non-instrumented runtime. */

int acuda_export_table_is_42d85a81(const struct CUuuid_st *id);
CUresult acuda_export_table_get_42d85a81(const void **ppExportTable);

const void *acuda_export_42d85a81_table(void);

#ifdef __cplusplus
}
#endif

#endif /* ACUDA_DRIVER_EXPORT_42D85A81_H */
