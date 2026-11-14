// Driver call tracer (env-gated, zero cost when off).
// Enabled with ACUDA_TRACE=1. Needs -finstrument-functions.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int trace_on(void) {
    static int on = -2;
    if (on == -2)
        on = getenv("ACUDA_TRACE") ? 1 : 0;
    return on == 1;
}

static const char *symname(void *fn) {
    Dl_info di;
    if (dladdr(fn, &di) && di.dli_sname)
        return di.dli_sname;
    return "?";
}

void __cyg_profile_func_enter(void *fn, void *site) {
    (void)site;
    if (trace_on())
        fprintf(stderr, "acuda-trace >> %s\n", symname(fn));
}

void __cyg_profile_func_exit(void *fn, void *site) {
    (void)fn;
    (void)site;
    // NOTE: return values are NOT captured here (reading %eax proved
    // unreliable); entry tracing only. Diagnose results explicitly.
}
