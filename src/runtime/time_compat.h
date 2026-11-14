// Monotonic nanosecond clock used by the runtime's optional profiling paths.
#ifndef ACUDA_TIME_COMPAT_H
#define ACUDA_TIME_COMPAT_H

#include <stdint.h>

#if defined(_WIN32)
#include <windows.h>

static inline uint64_t acuda_now_ns(void)
{
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (uint64_t)((double)counter.QuadPart * 1000000000.0 /
                      (double)frequency.QuadPart);
}
#else
#include <time.h>

static inline uint64_t acuda_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
#endif

#endif
