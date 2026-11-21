#ifndef ACUDA_DRIVER_EXPORT_DD9E98A4_H
#define ACUDA_DRIVER_EXPORT_DD9E98A4_H

#include "acuda/driver.h"

#ifdef __cplusplus
extern "C" {
#endif

/* {DD9E98A4-F410-4844-99B0-E2AEB7B432B5} hybrid runtime export table.
 * The CUDA 13.x statically linked runtime resolves cudart entry points
 * through this table instead of cuGetProcAddress. */

int acuda_export_table_is_dd9e98a4(const struct CUuuid_st *id);
CUresult acuda_export_table_get_dd9e98a4(const void **ppExportTable);


void acuda_dd9e98a4_dump_mem(const char *tag, const void *p, int n);
void *acuda_dd9e98a4_get_generic_stub(unsigned int id);

/* Returns 1 if dd9e98a4 driver-ready state/canaries look intact, 0 if
 * corruption detected (and logs). For diagnostic use from hot paths. */
int acuda_dd9e98a4_check_ready_canary(const char *where);

/* Ensure driver/interface is ready (implicit init, used for all export tables/GUIDs).
 * Returns 1 if ready, 0 on init failure. */
int acuda_driver_ready(void);

/* Diagnostic getters. */
int acuda_dd9e98a4_ready_state(void);
int acuda_dd9e98a4_was_ready_flag(void);

#ifdef __cplusplus
}
#endif

#endif /* ACUDA_DRIVER_EXPORT_DD9E98A4_H */
