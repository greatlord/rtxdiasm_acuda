#ifndef ACUDA_DRIVER_EXPORT_0CA50B8C_H
#define ACUDA_DRIVER_EXPORT_0CA50B8C_H

#include "acuda/driver.h"

#ifdef __cplusplus
extern "C" {
#endif

/* {0CA50B8C-1004-929A-89A7-D0DF10E77286} context-create bypass.  The
 * CUDA 4.x runtime creates the context through this interface so it can
 * skip its own thread-local wrapper.  Two slots: [0] size, [1] create. */

int acuda_export_table_is_0ca50b8c(const struct CUuuid_st *id);
CUresult acuda_export_table_get_0ca50b8c(const void **ppExportTable);

const void *acuda_export_0ca50b8c_table(void);

#ifdef __cplusplus
}
#endif

#endif /* ACUDA_DRIVER_EXPORT_0CA50B8C_H */
