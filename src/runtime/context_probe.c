#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "acuda/compiler.h"
#include "acuda/runtime/internal.h"
#include "acuda/runtime/context_internal.h"
#include "acuda/runtime/time_compat.h"

int probe_shader_fma(acuda_context_t ctx) {
    static const char ptx[] =
        ".version 6.5\n"
        ".target sm_30\n"
        ".address_size 64\n"
        ".visible .entry __acuda_fma_probe(\n"
        "    .param .u64 input,\n"
        "    .param .u64 output\n"
        ")\n"
        "{\n"
        "    .reg .u64 ia, oa;\n"
        "    .reg .f32 a, b, r;\n"
        "    ld.param.u64 ia, [input];\n"
        "    ld.param.u64 oa, [output];\n"
        "    ld.f32 a, [ia];\n"
        "    ld.f32 b, [ia+4];\n"
        "    add.rp.f32 r, a, b;\n"
        "    st.f32 [oa], r;\n"
        "    ret;\n"
        "}\n";
    acuda_kernel_t *k = NULL;
    char err[256];
    int ok = 0;
    acuda_compiler_caps_t caps;
    acuda_compiler_caps_init(&caps);
    acuda_compiler_set_shader_fma(&caps, 1);
    if (acuda_compile_ptx(ptx, "__acuda_fma_probe", &k, err, sizeof(err), &caps) !=
        ACUDA_SUCCESS)
        return 0;
    acuda_buffer_t in = NULL, out = NULL;
    uint32_t inbits[2] = { 0x3f800000u, 0x33800000u }; // 1.0 + 2^-24
    uint32_t got = 0;
    if (acuda_malloc(ctx, &in, sizeof(inbits)) == ACUDA_SUCCESS &&
        acuda_malloc(ctx, &out, 4) == ACUDA_SUCCESS &&
        acuda_memcpy_h2d(ctx, in, inbits, sizeof(inbits), NULL) ==
            ACUDA_SUCCESS) {
        acuda_buffer_t bufs[2] = { in, out };
        if (acuda_launch_kernel(ctx, k, bufs, 2, NULL, 0, 1, 1, NULL) ==
                ACUDA_SUCCESS &&
            acuda_stream_synchronize(ctx, NULL) == ACUDA_SUCCESS &&
            acuda_memcpy_d2h(ctx, &got, out, sizeof(got), NULL) ==
                ACUDA_SUCCESS &&
            got == 0x3f800001u)
            ok = 1;
    }
    if (in)
        acuda_free(ctx, in);
    if (out)
        acuda_free(ctx, out);
    acuda_kernel_free(k);
    return ok;
}

int probe_shader_fma64(acuda_context_t ctx) {
    static const char ptx[] =
        ".version 6.5\n"
        ".target sm_30\n"
        ".address_size 64\n"
        ".visible .entry __acuda_fma64_probe(\n"
        "    .param .u64 input,\n"
        "    .param .u64 output\n"
        ")\n"
        "{\n"
        "    .reg .u64 ia, oa;\n"
        "    .reg .f64 a, b, r, f1, f2, f3;\n"
        "    ld.param.u64 ia, [input];\n"
        "    ld.param.u64 oa, [output];\n"
        "    ld.f64 a, [ia];\n"
        "    ld.f64 b, [ia+8];\n"
        "    add.rp.f64 r, a, b;\n"
        "    st.f64 [oa], r;\n"
        "    ld.f64 f1, [ia+16];\n"
        "    ld.f64 f2, [ia+24];\n"
        "    ld.f64 f3, [ia+32];\n"
        "    fma.rn.f64 r, f1, f2, f3;\n"
        "    st.f64 [oa+8], r;\n"
        "    ret;\n"
        "}\n";
    acuda_kernel_t *k = NULL;
    char err[256];
    int ok = 0;
    acuda_compiler_caps_t caps;
    acuda_compiler_caps_init(&caps);
    acuda_compiler_set_shader_fma_f64(&caps, 1);
    if (acuda_compile_ptx(ptx, "__acuda_fma64_probe", &k, err,
                          sizeof(err), &caps) != ACUDA_SUCCESS)
        return 0;
    acuda_buffer_t in = NULL, out = NULL;
    uint64_t inbits[5] = {
        0x3ff0000000000000ull, // 1.0 + 2^-53 -> 1 + 2^-52 under rp
        0x3ca0000000000000ull,
        0x3ff0000000000001ull, // 1 + 2^-52
        0x3fefffffffffffffull, // 1 - 2^-52
        0xbff0000000000000ull  // -1
    };
    uint64_t got[2] = { 0, 0 };
    if (acuda_malloc(ctx, &in, sizeof(inbits)) == ACUDA_SUCCESS &&
        acuda_malloc(ctx, &out, sizeof(got)) == ACUDA_SUCCESS &&
        acuda_memcpy_h2d(ctx, in, inbits, sizeof(inbits), NULL) ==
            ACUDA_SUCCESS) {
        acuda_buffer_t bufs[2] = { in, out };
        if (acuda_launch_kernel(ctx, k, bufs, 2, NULL, 0, 1, 1, NULL) ==
                ACUDA_SUCCESS &&
            acuda_stream_synchronize(ctx, NULL) == ACUDA_SUCCESS &&
            acuda_memcpy_d2h(ctx, got, out, sizeof(got), NULL) ==
                ACUDA_SUCCESS &&
            got[0] == 0x3ff0000000000001ull &&
            got[1] == 0x3c9ffffffffffffeull)
            ok = 1;
    }
    if (in)
        acuda_free(ctx, in);
    if (out)
        acuda_free(ctx, out);
    acuda_kernel_free(k);
    return ok;
}

// Native 16-bit float arithmetic is much faster than the f32 emulation
// on devices with shaderFloat16, but not universally. Compile/run the
// same small packed-f16 FMA kernel both ways and keep the faster path.
int probe_f16_native(acuda_context_t ctx) {
    if (getenv("ACUDA_TRACE")) {
        fprintf(stderr, "ACuda-debug: probe_f16_native(ctx=%p)\n", (void *)ctx);
        fflush(stderr);
    }
    static const char ptx[] =
        ".version 8.8\n"
        ".target sm_120\n"
        ".address_size 64\n"
        ".visible .entry __acuda_f16_probe(.param .u64 p,"
        " .param .u32 reps)\n"
        "{\n"
        "    .reg .u64 r; .reg .u32 n; .reg .pred q;\n"
        "    .reg .b32 a0,a1,a2,a3,b0,b1,b2,b3;\n"
        "    .reg .b32 c0,c1,c2,c3;\n"
        "    ld.param.u64 r,[p];\n"
        "    ld.param.u32 n,[reps];\n"
        "    ld.global.b32 a0,[r]; ld.global.b32 a1,[r+4];\n"
        "    ld.global.b32 a2,[r+8]; ld.global.b32 a3,[r+12];\n"
        "    ld.global.b32 b0,[r+16]; ld.global.b32 b1,[r+20];\n"
        "    ld.global.b32 b2,[r+24]; ld.global.b32 b3,[r+28];\n"
        "    mov.b32 c0,0; mov.b32 c1,0; mov.b32 c2,0; mov.b32 c3,0;\n"
        "L:\n"
        "    fma.rn.f16x2 c0,a0,b0,c0; fma.rn.f16x2 c1,a1,b1,c1;\n"
        "    fma.rn.f16x2 c2,a2,b2,c2; fma.rn.f16x2 c3,a3,b3,c3;\n"
        "    fma.rn.f16x2 c0,a0,b0,c0; fma.rn.f16x2 c1,a1,b1,c1;\n"
        "    fma.rn.f16x2 c2,a2,b2,c2; fma.rn.f16x2 c3,a3,b3,c3;\n"
        "    fma.rn.f16x2 c0,a0,b0,c0; fma.rn.f16x2 c1,a1,b1,c1;\n"
        "    fma.rn.f16x2 c2,a2,b2,c2; fma.rn.f16x2 c3,a3,b3,c3;\n"
        "    fma.rn.f16x2 c0,a0,b0,c0; fma.rn.f16x2 c1,a1,b1,c1;\n"
        "    fma.rn.f16x2 c2,a2,b2,c2; fma.rn.f16x2 c3,a3,b3,c3;\n"
        "    fma.rn.f16x2 c0,a0,b0,c0; fma.rn.f16x2 c1,a1,b1,c1;\n"
        "    fma.rn.f16x2 c2,a2,b2,c2; fma.rn.f16x2 c3,a3,b3,c3;\n"
        "    fma.rn.f16x2 c0,a0,b0,c0; fma.rn.f16x2 c1,a1,b1,c1;\n"
        "    fma.rn.f16x2 c2,a2,b2,c2; fma.rn.f16x2 c3,a3,b3,c3;\n"
        "    fma.rn.f16x2 c0,a0,b0,c0; fma.rn.f16x2 c1,a1,b1,c1;\n"
        "    fma.rn.f16x2 c2,a2,b2,c2; fma.rn.f16x2 c3,a3,b3,c3;\n"
        "    fma.rn.f16x2 c0,a0,b0,c0; fma.rn.f16x2 c1,a1,b1,c1;\n"
        "    fma.rn.f16x2 c2,a2,b2,c2; fma.rn.f16x2 c3,a3,b3,c3;\n"
        "    sub.u32 n,n,1; setp.ne.u32 q,n,0; @q bra L;\n"
        "    add.f16x2 c0,c0,c1; add.f16x2 c0,c0,c2;\n"
        "    add.f16x2 c0,c0,c3;\n"
        "    st.global.b32 [r+64],c0;\n"
        "    ret;\n"
        "}\n";
    acuda_kernel_t *kn = NULL, *ke = NULL;
    acuda_compiler_caps_t caps_n, caps_e;
    acuda_compiler_caps_init(&caps_n);
    acuda_compiler_caps_init(&caps_e);
    acuda_compiler_set_f16_native(&caps_n, 1);
    acuda_compiler_set_f16_native(&caps_e, 0);
    int ok = acuda_compile_ptx(ptx, "__acuda_f16_probe", &kn, NULL, 0, &caps_n) ==
             ACUDA_SUCCESS;
    ok = ok && acuda_compile_ptx(ptx, "__acuda_f16_probe", &ke, NULL, 0, &caps_e) ==
                   ACUDA_SUCCESS;
    if (!ok) {
        acuda_kernel_free(kn);
        acuda_kernel_free(ke);
        return 0;
    }
    acuda_buffer_t in = NULL;
    uint32_t vals[32];
    for (int i = 0; i < 32; i++)
        vals[i] = 0x3c003c00u;
    if (acuda_malloc(ctx, &in, sizeof(vals)) != ACUDA_SUCCESS ||
        acuda_memcpy_h2d(ctx, in, vals, sizeof(vals), NULL) !=
            ACUDA_SUCCESS) {
        if (in) acuda_free(ctx, in);
        acuda_kernel_free(kn);
        acuda_kernel_free(ke);
        return 0;
    }
    uint32_t reps = 128;
    double tn = 1e9, te = 1e9;
    for (int round = 0; round < 2; round++) {
        for (int pass = 0; pass < 2; pass++) {
            acuda_kernel_t *k = pass == 0 ? kn : ke;
            for (int i = 0; i < 2; i++)
                acuda_launch_kernel(ctx, k, &in, 1, &reps, sizeof(reps),
                                    128, 64, NULL);
            uint64_t t0 = acuda_now_ns();
            for (int i = 0; i < 2; i++)
                acuda_launch_kernel(ctx, k, &in, 1, &reps, sizeof(reps),
                                    128, 64, NULL);
            acuda_stream_synchronize(ctx, NULL);
            uint64_t t1 = acuda_now_ns();
            double ms = (double)(t1 - t0) / 1e6;
            if (pass == 0 && ms < tn)
                tn = ms;
            if (pass == 1 && ms < te)
                te = ms;
        }
    }
    int native = tn <= te;
    if (getenv("ACUDA_DEBUG_F16"))
        fprintf(stderr,
                "ACuda: f16 probe native=%.3f ms emulated=%.3f ms -> %s\n",
                tn, te, native ? "native" : "emulated");
    acuda_free(ctx, in);
    acuda_kernel_free(kn);
    acuda_kernel_free(ke);
    return native;
}

// Fused-op probe: 1+2^-23 times 1-2^-23 is 1-2^-46; adding -1
// leaves -2^-46. A non-fused mul+add rounds the product to 1.0 first
// and returns 0, so this distinguishes true fma from mul+add.
int probe_shader_fma_rn(acuda_context_t ctx) {
    static const char ptx[] =
        ".version 6.5\n"
        ".target sm_30\n"
        ".address_size 64\n"
        ".visible .entry __acuda_fma_rn_probe(\n"
        "    .param .u64 input,\n"
        "    .param .u64 output\n"
        ")\n"
        "{\n"
        "    .reg .u64 ia, oa;\n"
        "    .reg .f32 a, b, c, r;\n"
        "    ld.param.u64 ia, [input];\n"
        "    ld.param.u64 oa, [output];\n"
        "    ld.f32 a, [ia];\n"
        "    ld.f32 b, [ia+4];\n"
        "    ld.f32 c, [ia+8];\n"
        "    fma.rn.f32 r, a, b, c;\n"
        "    st.f32 [oa], r;\n"
        "    ret;\n"
        "}\n";
    acuda_kernel_t *k = NULL;
    char err[256];
    int ok = 0;
    acuda_compiler_caps_t caps;
    acuda_compiler_caps_init(&caps);
    acuda_compiler_set_shader_fma(&caps, 1);
    if (acuda_compile_ptx(ptx, "__acuda_fma_rn_probe", &k, err,
                          sizeof(err), &caps) != ACUDA_SUCCESS)
        return 0;
    acuda_buffer_t in = NULL, out = NULL;
    uint32_t inbits[3] = { 0x3f800001u, 0x3f7fffffu, 0xbf800000u };
    uint32_t got = 0;
    if (acuda_malloc(ctx, &in, sizeof(inbits)) == ACUDA_SUCCESS &&
        acuda_malloc(ctx, &out, 4) == ACUDA_SUCCESS &&
        acuda_memcpy_h2d(ctx, in, inbits, sizeof(inbits), NULL) ==
            ACUDA_SUCCESS) {
        acuda_buffer_t bufs[2] = { in, out };
        if (acuda_launch_kernel(ctx, k, bufs, 2, NULL, 0, 1, 1, NULL) ==
                ACUDA_SUCCESS &&
            acuda_stream_synchronize(ctx, NULL) == ACUDA_SUCCESS &&
            acuda_memcpy_d2h(ctx, &got, out, sizeof(got), NULL) ==
                ACUDA_SUCCESS &&
            got == 0xa8800000u)
            ok = 1;
    }
    if (in)
        acuda_free(ctx, in);
    if (out)
        acuda_free(ctx, out);
    acuda_kernel_free(k);
    return ok;
}
