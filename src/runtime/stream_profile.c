#include <stdio.h>
#include <stdlib.h>
#include "acuda/runtime/internal.h"
#include "acuda/runtime/stream_internal.h"
#include "acuda/runtime/time_compat.h"

int stream_prof_on = -1;
uint64_t stream_prof_cmds, stream_prof_flushes, stream_prof_record, stream_prof_submit;
double stream_prof_now_ns(void) {
    return (double)acuda_now_ns();
}
void stream_prof_dump(void) {
    if (stream_prof_flushes == 0)
        return;
    double c = (double)(stream_prof_cmds ? stream_prof_cmds : 1);
    double f = (double)stream_prof_flushes;
    fprintf(stderr,
            "ACuda launch profile: %llu commands / %llu flushes, "
            "record %.2f us/command, submit %.2f us/flush\n",
            (unsigned long long)stream_prof_cmds, (unsigned long long)stream_prof_flushes,
            stream_prof_record / c / 1e3, stream_prof_submit / f / 1e3);
}
