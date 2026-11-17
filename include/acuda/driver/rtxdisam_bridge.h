#ifndef ACUDA_RTXDISAM_BRIDGE_H
#define ACUDA_RTXDISAM_BRIDGE_H

#include <stddef.h>

typedef enum {
    // The image was lifted to PTX.
    ACUDA_RTXDISAM_OK = 0,
    // The container has multiple text sections, so the caller must retry with
    // a specific entry/function name.
    ACUDA_RTXDISAM_ENTRY_REQUIRED = 1,
    // The library is missing or the image cannot be lifted.
    ACUDA_RTXDISAM_ERROR = -1
} acuda_rtxdisam_status_t;

// Lazily load the optional rtxdisam native library and convert a cubin or
// fatbin image to PTX. Returns a malloc'd NUL-terminated PTX string, or NULL
// when the library is unavailable or the image cannot be lifted. On failure,
// err (when non-NULL) receives a diagnostic.
char *acuda_rtxdisam_cubin_to_ptx(const unsigned char *data, size_t size,
                                  const char *entry,
                                  char *err, size_t errlen);

// Like acuda_rtxdisam_cubin_to_ptx, but distinguishes an image that needs an
// explicit entry name from a hard failure. On success, *ptx receives a
// malloc'd NUL-terminated string.
acuda_rtxdisam_status_t acuda_rtxdisam_cubin_to_ptx_status(
    const unsigned char *data, size_t size, const char *entry,
    char **ptx, char *err, size_t errlen);

// Target-aware variant. target is a preferred architecture name such as
// "sm_80"; the native library selects a matching fatbin member when possible
// and uses it for the emitted PTX target.
char *acuda_rtxdisam_cubin_to_ptx_for_target(
    const unsigned char *data, size_t size, const char *entry,
    const char *target, char *err, size_t errlen);

acuda_rtxdisam_status_t acuda_rtxdisam_cubin_to_ptx_status_for_target(
    const unsigned char *data, size_t size, const char *entry,
    const char *target, char **ptx, char *err, size_t errlen);

// Returns the size of an ELF cubin or raw fatbin blob. Returns 0 for an
// unrecognized image.
size_t acuda_image_size(const unsigned char *data);

#endif
