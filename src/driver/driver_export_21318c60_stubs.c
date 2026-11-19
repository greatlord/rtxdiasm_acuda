#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_21318c60.h"

/* {21318C60-9714-3248-8CA6-41FF7324C8F2} CUDA 13.4 tools table.
 *
 * CUDA 13.4 asks for this large table during stream setup.  Only the size
 * slot is needed for initialisation; unknown entries stay NULL and the
 * optional tool paths that use them are not reached by our runtime. */

static const void *g_cuda134_tools[64] = {
    (const void *)(uintptr_t)(64 * sizeof(void *)),
};

const void *acuda_export_21318c60_table(void) {
    return (const void *)g_cuda134_tools;
}
