#ifndef ACUDA_DRIVER_EXPORT_263E8860_H
#define ACUDA_DRIVER_EXPORT_263E8860_H

#include "acuda/driver.h"

#ifdef __cplusplus
extern "C" {
#endif

/* {263E8860-7CD2-6143-92F6-BBD5006DFA7E} context checks.  Both probes
 * are advisory; cudart only records the answers. */

int acuda_export_table_is_263e8860(const struct CUuuid_st *id);
CUresult acuda_export_table_get_263e8860(const void **ppExportTable);

const void *acuda_export_263e8860_table(void);

#ifdef __cplusplus
}
#endif

#endif /* ACUDA_DRIVER_EXPORT_263E8860_H */
