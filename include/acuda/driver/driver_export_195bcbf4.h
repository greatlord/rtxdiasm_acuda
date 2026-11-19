#ifndef ACUDA_DRIVER_EXPORT_195BCBF4_H
#define ACUDA_DRIVER_EXPORT_195BCBF4_H

#include "acuda/driver.h"

#ifdef __cplusplus
extern "C" {
#endif

/* {195BCBF4-D67D-024A-ACC5-1D29CEA631AE} heap access.  CUDA 6.x cudart
 * queries this table during context setup and fails every allocation
 * with "unknown error" (30) when it is missing.  Three slots: [0] size,
 * [1] alloc, [2] free. */

int acuda_export_table_is_195bcbf4(const struct CUuuid_st *id);
CUresult acuda_export_table_get_195bcbf4(const void **ppExportTable);

const void *acuda_export_195bcbf4_table(void);

#ifdef __cplusplus
}
#endif

#endif /* ACUDA_DRIVER_EXPORT_195BCBF4_H */
