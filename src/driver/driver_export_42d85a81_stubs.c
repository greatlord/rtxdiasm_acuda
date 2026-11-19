#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_42d85a81.h"

/* {42D85A81-23F6-CB47-8298-F6E78A3AECDC} tools thread-local storage.
 * Four slots; [0] is the table size.  No TLS keys are handed out for our
 * non-instrumented runtime. */

static const void *g_tools_tls[4] = {
    (const void *)(uintptr_t)(4 * sizeof(void *)),   /* [0] table size */
    NULL, NULL, NULL
};

const void *acuda_export_42d85a81_table(void) {
    return (const void *)g_tools_tls;
}
