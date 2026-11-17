#include "driver_internal.h"
#include "rtxdisam_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult ACUDA_CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, CUjit_option *options,
                            void **optionValues) {
    (void)numOptions;
    (void)options;
    (void)optionValues;
    if (getenv("ACUDA_TRACE") && image)
        fprintf(stderr, "acuda-trace .. cuModuleLoadDataEx magic=%08x\n",
                rd32le((const unsigned char *)image));
    if (!module)
        return CUDA_ERROR_INVALID_VALUE;
    *module = NULL;
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    if (!image)
        return CUDA_ERROR_INVALID_VALUE;
    const char *text = (const char *)image;
    const unsigned char *bytes = (const unsigned char *)image;
    // A __fatBinC_Wrapper_t is what cuModuleLoadData receives from old
    // PhysX/CUDA 4.x runtimes.  It points at the real 0xBA55ED50 blob.
    if (rd32le(bytes) == 0x466243B1u)
        return cuModuleLoadFatBinary(module, image);
    // A raw ELF cubin or fatbin blob has no PTX for ACuda to import. Ask the
    // optional rtxdisam native library to lift its SASS first.
    if ((bytes[0] == 0x7f && strncmp(text + 1, "ELF", 3) == 0) ||
        rd32le(bytes) == 0xBA55ED50u) {
        size_t size = acuda_image_size(bytes);
        if (!size)
            return CUDA_ERROR_INVALID_IMAGE;
        return load_sass_image(module, bytes, size);
    }
    if (!looks_like_ptx(text))
        return CUDA_ERROR_INVALID_IMAGE;
    size_t tlen = strlen(text);
    int ptx_oom = 0;
    if (tlen > ACUDA_PTX_EAGER_VALIDATE_MAX) {
        /* Large module: skip the eager whole-text parse. cuModuleGetFunction
         * extracts and compiles only the requested entry, which validates it
         * there; a full parse here just burns 32-bit address space and can
         * fail/crash before any function is ever needed. */
        ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleLoadDataEx: skip eager validate (%zu bytes)\n", tlen);
#if defined(_WIN32) && !defined(_WIN64)
        // The per-entry 32-bit parser is also likely to OOM on a module this
        // large; use the 64-bit compile host directly.
        ptx_oom = 1;
#endif
    } else {
        char err[1024];
        if (acuda_ptx_validate(text, err, sizeof(err)) != ACUDA_SUCCESS) {
            /* 32-bit large modules: full parse can OOM during validation; store
             * text and let cuModuleGetFunction/compile handle per-function. */
            if (!(err[0] && strstr(err, "out of memory")))
                return CUDA_ERROR_INVALID_IMAGE;
            ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleLoadDataEx: validate OOM, storing ptx text (%zu)\n", tlen);
#if defined(_WIN32) && !defined(_WIN64)
            // The same 32-bit parser will OOM again for every entry in this
            // module; skip straight to the 64-bit compile host.
            ptx_oom = 1;
#endif
        }
    }
    CUmodule m = calloc(1, sizeof(*m));
    if (!m)
        return CUDA_ERROR_OUT_OF_MEMORY;
    m->ctx = c;
    m->ptx_oom = ptx_oom;
    m->ptx = malloc(strlen(text) + 1);
    if (!m->ptx) {
        free(m);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    strcpy(m->ptx, text);
    *module = m;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuModuleLoadData(CUmodule *module, const void *image) {
    return cuModuleLoadDataEx(module, image, 0, NULL, NULL);
}

// ---- NVIDIA fatbin container parsing ----
//
// Core PTX member extraction/decompression (raw/LZ4/ZSTD) is shared via
// rtxdiasm-fatbin-parse (rtxdiasm_fatbin_extract_first_ptx) so acuda and
// cudart use identical fatbin decode logic. Driver-specific wrapper handling,
// validation/OOM acceptance, and SASS fallback remain here.

uint32_t rd32le(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

uint64_t rd64le(const unsigned char *p) {
    uint64_t lo = rd32le(p), hi = rd32le(p + 4);
    return lo | (hi << 32);
}

char *fatbin_first_ptx(const unsigned char *blob, size_t cap) {
    char *ptx = NULL;
    size_t ptx_len = 0;
    if (rtxdiasm_fatbin_extract_first_ptx(blob, cap, &ptx, &ptx_len) != RTXDIASM_FATBIN_OK)
        return NULL;
    /* Preserve driver behavior: accept large PTX without eager full validate,
     * and tolerate 32-bit validation OOM/internal failures (compile host revalidates). */
    if (ptx_len > ACUDA_PTX_EAGER_VALIDATE_MAX) {
        ACUDA_DEBUG_PRINT("acuda-trace .. extracted ptx large, accept without eager validate (%zu bytes)\n", ptx_len);
        return ptx;
    }
    {
        char err[256];
        err[0] = '\0';
        if (acuda_ptx_validate(ptx, err, sizeof(err)) == ACUDA_SUCCESS)
            return ptx;
        if (!err[0] || strstr(err, "out of memory")) {
            ACUDA_DEBUG_PRINT("acuda-trace .. extracted ptx validate OOM/internal, accepting as ptx (%zu bytes)\n", ptx_len);
            return ptx;
        }
        ACUDA_DEBUG_PRINT("acuda-trace .. extracted ptx validate failed: %s\n", err);
    }
    free(ptx);
    return NULL;
}

CUresult load_fatbin_member(CUmodule *module,
                                   const unsigned char *data, size_t size) {
    const char *dump = getenv("ACUDA_DUMP_FATBIN");
    if (dump && *dump) {
        // Numbered dumps: PhysX loads several fatbins per run and the
        // interesting one is rarely the last.
        static unsigned long dump_serial;
        char path[1024];
        size_t dl = strlen(dump);
        if (dl > 7 && strcmp(dump + dl - 7, ".fatbin") == 0)
            snprintf(path, sizeof(path), "%.*s_%lu.fatbin", (int)(dl - 7),
                     dump, ++dump_serial);
        else
            snprintf(path, sizeof(path), "%s_%lu", dump, ++dump_serial);
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(data, 1, size, f);
            fclose(f);
            ACUDA_DEBUG_PRINT("acuda-trace .. dumped fatbin %zu bytes\n", size);
        }
    }
    char *ptx = cubin_backend() == 2 ? NULL : fatbin_first_ptx(data, size);
    if (!ptx)
        return load_sass_image(module, data, size);
    CUresult r = cuModuleLoadData(module, ptx);
    free(ptx);
    if (r == CUDA_SUCCESS && module && *module)
        attach_sass_image(*module, data, size);
    return r;
}

CUresult ACUDA_CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleLoadFatBinary\n");
    if (!module)
        return CUDA_ERROR_INVALID_VALUE;
    *module = NULL;
    if (!fatCubin)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    // The argument is the __fatBinC_Wrapper_t control struct, not the
    // raw blob (verified field-by-field against real torch fatbins).
    const unsigned char *w = (const unsigned char *)fatCubin;
    if (rd32le(w) != 0x466243B1u)
        return CUDA_ERROR_INVALID_IMAGE;
    uint32_t ver = rd32le(w + 4);
    ACUDA_DEBUG_PRINT("acuda-trace .. fatbin wrapper ver=%u\n", ver);
    if (ver != 1 && ver != 2)
        return CUDA_ERROR_INVALID_IMAGE;
    if (ver == 1) {
        const unsigned char *data = NULL;
        memcpy(&data, w + 8, sizeof(data));
        if (!data)
            return CUDA_ERROR_INVALID_IMAGE;
        // Bound the walk by the blob's own declared size (+ header).
        if (rd32le(data) != 0xBA55ED50u)
            return CUDA_ERROR_INVALID_IMAGE;
        uint64_t files_size = rd64le(data + 8);
        if (files_size > (uint64_t)512 << 20)
            return CUDA_ERROR_INVALID_IMAGE;
        return load_fatbin_member(module, data,
                                  16 + (size_t)files_size);
    }
    // Version 2: NULL-terminated array of wrapper pointers (submodules).
    const void *const *arr = NULL;
    memcpy(&arr, w + 16, sizeof(arr));
    if (!arr)
        return CUDA_ERROR_INVALID_IMAGE;
    for (size_t i = 0; i < 4096; i++) {
        const unsigned char *sub = NULL;
        memcpy(&sub, &arr[i], sizeof(sub));
        if (!sub)
            break;
        if (rd32le(sub) != 0x466243B1u || rd32le(sub + 4) != 1)
            continue;
        const unsigned char *data = NULL;
        memcpy(&data, sub + 8, sizeof(data));
        if (!data || rd32le(data) != 0xBA55ED50u)
            continue;
        uint64_t files_size = rd64le(data + 8);
        if (files_size > (uint64_t)512 << 20)
            continue;
        CUresult r = load_fatbin_member(module, data,
                                        16 + (size_t)files_size);
        if (r == CUDA_SUCCESS)
            return r;
    }
    return CUDA_ERROR_INVALID_IMAGE;
}

/* PTX cache, compile-host helpers, and module function loading. */

static void reported_sm_target(char *buf, size_t len) {
    /* Diagnostics: the SASS lift target normally follows the reported
     * compute capability (old software may need an era CC to accept the
     * GPU, while the lifter still needs a supported arch).
     * ACUDA_SASS_TARGET=<sm_xx> pins the lift/cubin-selection target. */
    const char *t = getenv("ACUDA_SASS_TARGET");
    if (t && t[0]) {
        snprintf(buf, len, "%s", t);
        return;
    }
    snprintf(buf, len, "sm_%d%d", cc_major(), cc_minor());
}

#if defined(_WIN32)
#define ACUDA_CACHE_SEP '\\'
#else
#define ACUDA_CACHE_SEP '/'
#endif

static int acuda_cache_dir(char *buf, size_t bufn) {
    const char *d = getenv("ACUDA_CACHE_DIR");
    if (d && d[0]) {
        snprintf(buf, bufn, "%s", d);
        return 1;
    }
#if defined(_WIN32)
    {
        const char *la = getenv("LOCALAPPDATA");
        if (la && la[0]) {
            snprintf(buf, bufn, "%s\\acuda\\kernel-cache", la);
            return 1;
        }
        {
            char tmp[MAX_PATH];
            DWORD n = GetTempPathA(MAX_PATH, tmp);
            if (n > 0 && n < MAX_PATH) {
                snprintf(buf, bufn, "%sacuda-kernel-cache", tmp);
                return 1;
            }
        }
    }
#else
    {
        const char *xdg = getenv("XDG_CACHE_HOME");
        const char *home = getenv("HOME");
        if (xdg && xdg[0]) {
            snprintf(buf, bufn, "%s/acuda/kernel-cache", xdg);
            return 1;
        }
        if (home && home[0]) {
            snprintf(buf, bufn, "%s/.cache/acuda/kernel-cache", home);
            return 1;
        }
        snprintf(buf, bufn, "/tmp/acuda-kernel-cache");
        return 1;
    }
#endif
    return 0;
}

static void acuda_cache_mkdir(const char *dir) {
    char tmp[1024];
    size_t n;
    snprintf(tmp, sizeof(tmp), "%s", dir);
    n = strlen(tmp);
    for (size_t i = 2; i < n; i++) {
        if (tmp[i] == '/' || tmp[i] == '\\') {
            char c = tmp[i];
            tmp[i] = 0;
#if defined(_WIN32)
            _mkdir(tmp);
#else
            mkdir(tmp, 0700);
#endif
            tmp[i] = c;
        }
    }
#if defined(_WIN32)
    _mkdir(tmp);
#else
    mkdir(tmp, 0700);
#endif
}

/* Default namespace for the kernel cache when ACUDA_CACHE_TAG is unset.
 * Derived from this module's PE link timestamp: stable for every run of
 * one build, but automatically different after a rebuild, so a new
 * lifter never silently reuses kernels (or .failed markers) produced by
 * an older one.  ACUDA_CACHE_TAG overrides it. */
static const char *acuda_cache_default_tag(void) {
    static char tag[64];
    if (!tag[0]) {
        unsigned long stamp = 0;
#if defined(_WIN32)
        HMODULE self = NULL;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR)&acuda_cache_default_tag, &self) &&
            self) {
            const IMAGE_DOS_HEADER *dos = (const IMAGE_DOS_HEADER *)self;
            if (dos->e_magic == IMAGE_DOS_SIGNATURE) {
                const IMAGE_NT_HEADERS *nt =
                    (const IMAGE_NT_HEADERS *)((const char *)self +
                                               dos->e_lfanew);
                if (nt->Signature == IMAGE_NT_SIGNATURE)
                    stamp = nt->FileHeader.TimeDateStamp;
            }
        }
#endif
        snprintf(tag, sizeof(tag), "acuda-kernel-cache-%08lx", stamp);
    }
    return tag;
}

static uint64_t acuda_cache_hash(const char *ptx, const char *entry) {
    uint64_t h = 1469598103934665603ull;
    /* The tag is the cache namespace: ACUDA_CACHE_TAG wins when set
     * (a build id, a timestamp, ...); otherwise use the build stamp. */
    const char *tag = getenv("ACUDA_CACHE_TAG");
    if (!tag || !*tag)
        tag = acuda_cache_default_tag();
    const unsigned char *s;
    for (s = (const unsigned char *)tag; *s; s++) {
        h ^= *s;
        h *= 1099511628211ull;
    }
    if (ptx)
        for (s = (const unsigned char *)ptx; *s; s++) {
            h ^= *s;
            h *= 1099511628211ull;
        }
    h ^= 0xffu;
    h *= 1099511628211ull;
    if (entry)
        for (s = (const unsigned char *)entry; *s; s++) {
            h ^= *s;
            h *= 1099511628211ull;
        }
    h ^= 0xfeu;
    h *= 1099511628211ull;
    {
        const char *emu = getenv("ACUDA_EMU_F64");
        if (emu)
            for (s = (const unsigned char *)emu; *s; s++) {
                h ^= *s;
                h *= 1099511628211ull;
            }
    }
    h ^= 0xfdu;
    h *= 1099511628211ull;
    {
        char target[16];
        reported_sm_target(target, sizeof(target));
        for (s = (const unsigned char *)target; *s; s++) {
            h ^= *s;
            h *= 1099511628211ull;
        }
    }
    return h;
}

static acuda_status_t acuda_cache_load(const char *ptx, const char *entry,
                                       acuda_kernel_t **out) {
    char dir[1024], path[1200], tpath[1200];
    uint64_t h;
    FILE *src, *dst;
    char buf[65536];
    size_t r;
    static unsigned long long serial;
    acuda_status_t st;
    *out = NULL;
    if (!ptx || !entry || getenv("ACUDA_NO_CACHE"))
        return ACUDA_ERROR_INVALID_VALUE;
    if (!acuda_cache_dir(dir, sizeof(dir)))
        return ACUDA_ERROR_INVALID_VALUE;
    h = acuda_cache_hash(ptx, entry);
    snprintf(path, sizeof(path), "%s%c%016llx.akb", dir, ACUDA_CACHE_SEP,
             (unsigned long long)h);
    src = fopen(path, "rb");
    if (!src)
        return ACUDA_ERROR_INVALID_VALUE;
    fclose(src);
#if defined(_WIN32)
    {
        char tmp[MAX_PATH];
        GetTempPathA(MAX_PATH, tmp);
        snprintf(tpath, sizeof(tpath), "%sacuda_cache_%lu_%llu.akb", tmp,
                 (unsigned long)GetCurrentProcessId(), ++serial);
    }
#else
    snprintf(tpath, sizeof(tpath), "/tmp/acuda_cache_%ld_%llu.akb",
             (long)getpid(), ++serial);
#endif
    src = fopen(path, "rb");
    dst = fopen(tpath, "wb");
    if (!src || !dst) {
        if (src)
            fclose(src);
        if (dst)
            fclose(dst);
        remove(tpath);
        return ACUDA_ERROR_INVALID_VALUE;
    }
    while ((r = fread(buf, 1, sizeof(buf), src)) > 0) {
        if (fwrite(buf, 1, r, dst) != r)
            break;
    }
    fclose(src);
    fclose(dst);
    st = acuda_kernel_deserialize_file(tpath, out);
    if (st != ACUDA_SUCCESS)
        remove(tpath);
    ACUDA_DEBUG_PRINT("acuda-trace .. cache load %016llx (%s) st=%d\n",
                (unsigned long long)h, entry, (int)st);
    return st;
}

static void acuda_cache_fail_path(const char *dir, uint64_t h, char *out,
                                  size_t outlen) {
    snprintf(out, outlen, "%s%c%016llx.failed", dir, ACUDA_CACHE_SEP,
             (unsigned long long)h);
}

/* Tag-scoped negative cache: deterministic compile failures (unsupported
 * SASS, missing BDA types, ...) are re-attempted on every run, each
 * paying a full lift+compile roundtrip.  A .failed marker skips them.
 * The marker key includes the cache tag (via acuda_cache_hash), so a
 * compiler fix automatically re-attempts.  OOM/resource failures are
 * transient and never cached. */
static int acuda_cache_is_transient(const char *err) {
    static const char *words[] = {"oom", "out of memory", "out_of_memory",
                                  "no memory", "alloc fail", "resource"};
    if (!err || !*err)
        return 1;
    for (size_t i = 0; i < sizeof(words) / sizeof(words[0]); i++)
        if (strstr(err, words[i]) != NULL)
            return 1;
    {
        // Case-insensitive OOM check without extra deps.
        char lower[128];
        size_t n = strlen(err);
        if (n >= sizeof(lower))
            n = sizeof(lower) - 1;
        for (size_t i = 0; i < n; i++) {
            char ch = err[i];
            lower[i] = (char)((ch >= 'A' && ch <= 'Z') ? ch + 32 : ch);
        }
        lower[n] = 0;
        if (strstr(lower, "oom") != NULL ||
            strstr(lower, "out of memory") != NULL)
            return 1;
    }
    return 0;
}

static int acuda_cache_failed(const char *dir, uint64_t h, int *code,
                              char *err, size_t errlen) {
    char path[1200];
    FILE *f;
    if (!dir || getenv("ACUDA_NO_CACHE"))
        return 0;
    acuda_cache_fail_path(dir, h, path, sizeof(path));
    f = fopen(path, "rb");
    if (!f)
        return 0;
    {
        char line[32];
        if (!fgets(line, sizeof(line), f)) {
            fclose(f);
            return 0;
        }
        if (code)
            *code = atoi(line);
    }
    if (err && errlen > 0) {
        size_t n = fread(err, 1, errlen - 1, f);
        err[n] = 0;
    }
    fclose(f);
    return 1;
}

static void acuda_cache_mark_failed(const char *dir, uint64_t h, int code,
                                    const char *err) {
    char path[1200];
    FILE *f;
    if (!dir || getenv("ACUDA_NO_CACHE") || !err || !*err)
        return;
    if (acuda_cache_is_transient(err))
        return;
    acuda_cache_fail_path(dir, h, path, sizeof(path));
    f = fopen(path, "wb");
    if (!f)
        return;
    fprintf(f, "%d\n%s", code, err);
    fclose(f);
}

static void acuda_cache_store(const char *ptx, const char *entry,
                              const acuda_kernel_t *k) {
    char dir[1024], path[1200], tpath[1200];
    uint64_t h;
    uint8_t *blob = NULL;
    size_t bsz = 0;
    static unsigned long long serial;
    FILE *f;
    if (!ptx || !entry || !k || getenv("ACUDA_NO_CACHE"))
        return;
    if (!acuda_cache_dir(dir, sizeof(dir)))
        return;
    acuda_cache_mkdir(dir);
    h = acuda_cache_hash(ptx, entry);
    snprintf(path, sizeof(path), "%s%c%016llx.akb", dir, ACUDA_CACHE_SEP,
             (unsigned long long)h);
    if (acuda_kernel_serialize(k, &blob, &bsz) != ACUDA_SUCCESS || !blob)
        return;
#if defined(_WIN32)
    {
        char tmp[MAX_PATH];
        GetTempPathA(MAX_PATH, tmp);
        snprintf(tpath, sizeof(tpath), "%sacuda_cache_%lu_%llu.tmp", tmp,
                 (unsigned long)GetCurrentProcessId(), ++serial);
    }
#else
    snprintf(tpath, sizeof(tpath), "/tmp/acuda_cache_%ld_%llu.tmp",
             (long)getpid(), ++serial);
#endif
    f = fopen(tpath, "wb");
    if (f) {
        size_t wr = fwrite(blob, 1, bsz, f);
        fclose(f);
        if (wr == bsz) {
            remove(path);
            if (rename(tpath, path) != 0)
                remove(tpath);
        } else {
            remove(tpath);
        }
    }
    free(blob);
    ACUDA_DEBUG_PRINT("acuda-trace .. cache store %016llx (%s) %zu bytes\n",
                (unsigned long long)h, entry, bsz);
}

CUresult attach_sass_image(CUmodule module, const unsigned char *data,
                                  size_t size) {
    if (!module || !data || size == 0)
        return CUDA_ERROR_INVALID_VALUE;
    unsigned char *copy = malloc(size);
    if (!copy)
        return CUDA_ERROR_OUT_OF_MEMORY;
    memcpy(copy, data, size);
    free(module->image);
    module->image = copy;
    module->image_size = size;
    return CUDA_SUCCESS;
}

CUresult module_from_sass_image(CUmodule *module,
                                       const unsigned char *data,
                                       size_t size) {
    if (!module)
        return CUDA_ERROR_INVALID_VALUE;
    *module = NULL;
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    CUmodule m = calloc(1, sizeof(*m));
    if (!m)
        return CUDA_ERROR_OUT_OF_MEMORY;
    m->ctx = c;
    r = attach_sass_image(m, data, size);
    if (r != CUDA_SUCCESS) {
        free(m);
        return r;
    }
    *module = m;
    return CUDA_SUCCESS;
}

/* Select the cubin/fatbin backend (shared with rtxdiasm_guard/cudart):
 *   0 (default) - C fatbin parser when the image carries PTX, otherwise
 *                 lift SASS with rtxdisam (C#).
 *   1           - C fatbin parser only; never call the SASS lifter.
 *   2           - rtxdisam (C#) only; skip the C fatbin parser.
 * Unset, empty or invalid values mean 0. */
int cubin_backend(void) {
    const char *s = getenv("ACUDA_CUBIN_BACKEND");
    if (s && *s) {
        long v = strtol(s, NULL, 0);
        if (v == 1 || v == 2)
            return (int)v;
    }
    return 0;
}

CUresult load_sass_image(CUmodule *module, const unsigned char *data,
                                size_t size) {
    if (!module)
        return CUDA_ERROR_INVALID_VALUE;
    *module = NULL;
    if (cubin_backend() == 1) {
        ACUDA_DEBUG_PRINT("acuda-trace .. backend=1 (C parser only): "
                    "refusing the rtxdisam SASS lift\n");
        return CUDA_ERROR_INVALID_IMAGE;
    }
    char err[1024] = {0};
    char target[16];
    reported_sm_target(target, sizeof(target));
    char *ptx = NULL;
    acuda_rtxdisam_status_t status =
        acuda_rtxdisam_cubin_to_ptx_status_for_target(
            data, size, NULL, target, &ptx, err, sizeof(err));
    if (status == ACUDA_RTXDISAM_ENTRY_REQUIRED)
        return module_from_sass_image(module, data, size);
    if (status != ACUDA_RTXDISAM_OK || !ptx) {
        if (err[0])
            fprintf(stderr, "ACuda: %s\n", err);
        return CUDA_ERROR_INVALID_IMAGE;
    }
    CUmodule m = NULL;
    CUresult r = cuModuleLoadData(&m, ptx);
    free(ptx);
    if (r != CUDA_SUCCESS)
        return r;
    // Keep the original image: a fatbin may contain several single-entry
    // cubins, so a later cuModuleGetFunction(name) may need to lift another
    // member even though the entry-less lift selected the first one.
    r = attach_sass_image(m, data, size);
    if (r != CUDA_SUCCESS) {
        cuModuleUnload(m);
        return r;
    }
    *module = m;
    return CUDA_SUCCESS;
}




static char *replace_text(const char *src, const char *from,
                         const char *to) {
    size_t fl = strlen(from), tl = strlen(to), n = 0;
    const char *p = src, *q;
    while ((q = strstr(p, from)) != NULL) {
        n++;
        p = q + fl;
    }
    size_t len = strlen(src) + n * (tl - fl) + 1;
    char *out = malloc(len);
    if (!out)
        return NULL;
    char *d = out;
    p = src;
    while ((q = strstr(p, from)) != NULL) {
        size_t k = (size_t)(q - p);
        memcpy(d, p, k);
        d += k;
        memcpy(d, to, tl);
        d += tl;
        p = q + fl;
    }
    strcpy(d, p);
    return out;
}

#if defined(_WIN32)
static const char *acuda_compile_host_exe(char *buf, size_t bufn) {
    const char *e = getenv("ACUDA_COMPILE_HOST");
    if (e && e[0]) return e;
    /* next to RTXDISAM_HOST if set */
    const char *rh = getenv("RTXDISAM_HOST");
    if (rh && rh[0]) {
        snprintf(buf, bufn, "%s", rh);
        char *sl = strrchr(buf, '\\'); char *sl2 = strrchr(buf, '/');
        char *s = sl > sl2 ? sl : sl2;
        if (s) {
            strcpy(s + 1, "rtxdiasm_ptx-to-spirv_compiler.exe");
            if (GetFileAttributesA(buf) != INVALID_FILE_ATTRIBUTES) return buf;
            strcpy(s + 1, "acuda_compile_host.exe");
            if (GetFileAttributesA(buf) != INVALID_FILE_ATTRIBUTES) return buf;
        }
    }
    if (GetFileAttributesA("rtxdiasm_ptx-to-spirv_compiler.exe") != INVALID_FILE_ATTRIBUTES) return "rtxdiasm_ptx-to-spirv_compiler.exe";
    if (GetFileAttributesA("acuda_compile_host.exe") != INVALID_FILE_ATTRIBUTES) return "acuda_compile_host.exe";
    if (GetFileAttributesA("out\\win-x64\\rtxdiasm_ptx-to-spirv_compiler.exe") != INVALID_FILE_ATTRIBUTES) return "out\\win-x64\\rtxdiasm_ptx-to-spirv_compiler.exe";
    if (GetFileAttributesA("out\\win-x64\\acuda_compile_host.exe") != INVALID_FILE_ATTRIBUTES) return "out\\win-x64\\acuda_compile_host.exe";
    if (GetFileAttributesA("C:\\Users\\Magnus\\source\\repos\\ConsoleApp2\\out\\win-x64\\rtxdiasm_ptx-to-spirv_compiler.exe") != INVALID_FILE_ATTRIBUTES)
        return "C:\\Users\\Magnus\\source\\repos\\ConsoleApp2\\out\\win-x64\\rtxdiasm_ptx-to-spirv_compiler.exe";
    if (GetFileAttributesA("C:\\Users\\Magnus\\source\\repos\\ConsoleApp2\\out\\win-x64\\acuda_compile_host.exe") != INVALID_FILE_ATTRIBUTES)
        return "C:\\Users\\Magnus\\source\\repos\\ConsoleApp2\\out\\win-x64\\acuda_compile_host.exe";
    return NULL;
}

static size_t acuda_append_caps_args(char *dst, size_t dstsz, const acuda_compiler_caps_t *caps) {
    if (!dst || dstsz == 0) return 0;
    dst[0] = '\0';
    acuda_compiler_caps_t def;
    if (!caps) {
        acuda_compiler_caps_init(&def);
        caps = &def;
    }
    return (size_t)snprintf(dst, dstsz,
        " --cap.shader_fma=%d --cap.shader_fma_f64=%d --cap.soft_fma=%d"
        " --cap.f16_native=%d --cap.coopmat_f16=%d --cap.coopmat_bf16=%d"
        " --cap.mixed_fdot=%d --cap.dot_product=%d --cap.dot_signed=%d"
        " --cap.dot_unsigned=%d --cap.dot_mixed=%d --cap.zero_init_wg=%d"
        " --cap.ssbo8=%d --cap.ssbo16=%d --cap.fc32=%d --cap.fc64=%d"
        " --cap.vmm=%d --cap.bda=%d --cap.atomic_float=%u"
        " --cap.int64_image=%d --cap.surf_alias_ok=%d",
        caps->shader_fma, caps->shader_fma_f64, caps->soft_fma,
        caps->f16_native, caps->coopmat_f16, caps->coopmat_bf16,
        caps->mixed_fdot, caps->dot_product, caps->dot_signed,
        caps->dot_unsigned, caps->dot_mixed, caps->zero_init_wg,
        caps->ssbo8, caps->ssbo16, caps->fc32, caps->fc64,
        caps->vmm, caps->bda, caps->atomic_float,
        caps->int64_image, caps->surf_alias_ok);
}

static acuda_status_t acuda_compile_via_host(const char *ptx, const char *entry,
                                             const acuda_compiler_caps_t *caps,
                                             acuda_kernel_t **out, char *err, size_t errlen) {
    *out = NULL;
    char exebuf[MAX_PATH];
    const char *exe = acuda_compile_host_exe(exebuf, sizeof(exebuf));
    if (!exe) { if (err && errlen) snprintf(err, errlen, "rtxdiasm_ptx-to-spirv_compiler executable not found"); return ACUDA_ERROR_NOT_SUPPORTED; }
    char tmp[MAX_PATH], inpath[MAX_PATH], outpath[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp);
    snprintf(inpath, sizeof(inpath), "%sacuda_ch_%lu_%lu.ptx", tmp, GetCurrentProcessId(), GetTickCount());
    snprintf(outpath, sizeof(outpath), "%sacuda_ch_%lu_%lu.akb", tmp, GetCurrentProcessId(), GetTickCount()+1);
    FILE *fi = fopen(inpath, "wb");
    if (!fi) return ACUDA_ERROR_NOT_SUPPORTED;
    fwrite(ptx, 1, strlen(ptx), fi); fclose(fi);
    char capsargs[1024];
    acuda_append_caps_args(capsargs, sizeof(capsargs), caps);
    char cmd[8192];
    if (entry && *entry)
        snprintf(cmd, sizeof(cmd), "\"%s\" \"%s\" \"%s\" \"%s\"%s", exe, inpath, outpath, entry, capsargs);
    else
        snprintf(cmd, sizeof(cmd), "\"%s\" \"%s\" \"%s\" -%s", exe, inpath, outpath, capsargs);
    STARTUPINFOA si; PROCESS_INFORMATION pi; memset(&si,0,sizeof(si)); si.cb=sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    SECURITY_ATTRIBUTES sa; memset(&sa,0,sizeof(sa)); sa.nLength=sizeof(sa); sa.bInheritHandle=TRUE;
    char logpath[MAX_PATH]; snprintf(logpath,sizeof(logpath),"%sacuda_ch_%lu.log",tmp,GetCurrentProcessId());
    HANDLE lh = CreateFileA(logpath, GENERIC_WRITE, FILE_SHARE_READ|FILE_SHARE_WRITE, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, NULL);
    if (lh != INVALID_HANDLE_VALUE) { si.hStdOutput=lh; si.hStdError=lh; si.hStdInput=GetStdHandle(STD_INPUT_HANDLE); }
    BOOL ok = CreateProcessA(NULL, cmd, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    if (lh != INVALID_HANDLE_VALUE) CloseHandle(lh);
    if (!ok) { DeleteFileA(inpath); if(err&&errlen) snprintf(err,errlen,"cannot start compiler host executable"); return ACUDA_ERROR_NOT_SUPPORTED; }
    WaitForSingleObject(pi.hProcess, 120000);
    DWORD ec=1; GetExitCodeProcess(pi.hProcess, &ec);
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
    DeleteFileA(inpath);
    ACUDA_DEBUG_PRINT("acuda-trace .. compile_host ec=%lu in=%s out=%s exists=%d\n",
                (unsigned long)ec, inpath, outpath, (int)(GetFileAttributesA(outpath)!=INVALID_FILE_ATTRIBUTES));
    if (ec != 0) {
        if (err && errlen) {
            FILE *lf=fopen(logpath,"rb");
            if (lf) { char msg[512]={0}; size_t rn=fread(msg,1,sizeof(msg)-1,lf); fclose(lf); msg[rn]=0;
                /* last non-empty line */
                char *last=msg; for(char *q=msg;*q;q++) if(*q=='\n'&&q[1]) last=q+1;
                snprintf(err,errlen,"compiler host exit %lu: %.400s",(unsigned long)ec,last);
            } else snprintf(err,errlen,"compiler host exit %lu",(unsigned long)ec);
        }
        DeleteFileA(logpath); DeleteFileA(outpath);
        return ACUDA_ERROR_NOT_SUPPORTED;
    }
    DeleteFileA(logpath);
    acuda_status_t st = acuda_kernel_deserialize_file(outpath, out);
    ACUDA_DEBUG_PRINT("acuda-trace .. compile_host deserialize_file=%d\n", (int)st);
    /* deserialize_file removes AKB when not used as SPIR-V backing; if used as
     * backing, kernel owns the file and will remove it on free. */
    if (st != ACUDA_SUCCESS) DeleteFileA(outpath);
    if (st != ACUDA_SUCCESS && err && errlen) snprintf(err,errlen,"compile_host deserialize_file status %d", (int)st);
    return st;
}
#endif

CUresult ACUDA_CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule module,
                             const char *name) {
    if (!hfunc)
        return CUDA_ERROR_INVALID_VALUE;
    *hfunc = NULL;
    if (!module || !name)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    ACUDA_DEBUG_PRINT("acuda-trace >> cuModuleGetFunction(%s)\n", name);
    for (size_t i = 0; i < module->nfuncs; i++) {
        if (strcmp(module->funcs[i].name, name) == 0) {
            CUfunction f = malloc(sizeof(*f));
            if (!f)
                return CUDA_ERROR_OUT_OF_MEMORY;
            f->magic = ACUDA_HANDLE_FUNC;
            f->mod = module;
            f->k = module->funcs[i].k;
            snprintf(f->name, sizeof(f->name), "%s", name);
            f->maxdyn_set = 0;
            f->maxdyn = 0;
            *hfunc = f;
            return CUDA_SUCCESS;
        }
    }
    char err[1024] = {0};
    acuda_kernel_t *k = NULL;
    // Auto-fallback: f64 code on a device without shaderFloat64 compiles
    // emulated (explicit ACUDA_EMU_F64, in either direction, always wins).
    int emu_auto = 0;
    if (!getenv("ACUDA_EMU_F64") && need_actx() == CUDA_SUCCESS && g_actx &&
        !g_actx->f64_ok) {
        setenv("ACUDA_EMU_F64", "1", 1);
        emu_auto = 1;
    }
    acuda_status_t cst = ACUDA_ERROR_INVALID_VALUE;
    // Tag-scoped negative cache: skip kernels whose compilation already
    // failed deterministically (unsupported SASS, missing BDA types...).
    // The key covers the entry and target; the tag is folded in by
    // acuda_cache_hash, so a compiler fix automatically re-attempts.
    char fail_key[2048];
    char fail_dir[1024];
    uint64_t fail_hash = 0;
    int have_fail_dir = 0;
    int skip_attempts = 0;
    {
        char tgt[16];
        reported_sm_target(tgt, sizeof(tgt));
        snprintf(fail_key, sizeof(fail_key), "%s|%s", name ? name : "", tgt);
        have_fail_dir = acuda_cache_dir(fail_dir, sizeof(fail_dir));
        if (have_fail_dir)
            fail_hash = acuda_cache_hash(NULL, fail_key);
    }
    if (have_fail_dir) {
        int fcode = 0;
        if (acuda_cache_failed(fail_dir, fail_hash, &fcode, err,
                               sizeof(err))) {
            ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): cached failure, skipping compile\n",
                        name);
            cst = (acuda_status_t)fcode;
            if (cst == ACUDA_SUCCESS)
                cst = ACUDA_ERROR_NOT_SUPPORTED;
            skip_attempts = 1;
        }
    }
    if (!skip_attempts && module->ptx) {
        acuda_kernel_t *ck = NULL;
        if (acuda_cache_load(module->ptx, name, &ck) == ACUDA_SUCCESS && ck) {
            k = ck;
            cst = ACUDA_SUCCESS;
            ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): cache hit\n", name);
        } else {
            if (!module->ptx_oom) {
                ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): trying ptx compile\n", name);
                cst = acuda_compile_ptx(module->ptx, name, &k, err, sizeof(err), acuda_context_get_compiler_caps(g_actx));
                ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): ptx compile cst=%d err=%s\n", name, (int)cst, err[0]?err:"(none)");
            } else if (getenv("ACUDA_TRACE")) {
                fprintf(stderr, "acuda-trace .. cuModuleGetFunction(%s): skip 32-bit ptx compile (module parse OOMed)\n", name);
            }
#if defined(_WIN32)
            if (cst != ACUDA_SUCCESS && sizeof(void *) <= 4) {
                ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): trying 64-bit compile_host fallback\n", name);
                acuda_kernel_t *hk = NULL; char herr[1024] = {0};
                acuda_status_t hcst = acuda_compile_via_host(module->ptx, name, acuda_context_get_compiler_caps(g_actx), &hk, herr, sizeof(herr));
                ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): compile_host cst=%d err=%s\n", name, (int)hcst, herr[0]?herr:"(none)");
                if (hcst == ACUDA_SUCCESS && hk) { k = hk; cst = ACUDA_SUCCESS; err[0] = 0; }
                else if (herr[0] && !err[0]) snprintf(err, sizeof(err), "%s", herr);
            }
#endif
            if (cst == ACUDA_SUCCESS && k)
                acuda_cache_store(module->ptx, name, k);
        }
    }
    if (!skip_attempts && cst != ACUDA_SUCCESS && module->image &&
        module->image_size && cubin_backend() == 1) {
        ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): "
                    "backend=1 (C parser only): no rtxdisam lift\n", name);
    }
    if (!skip_attempts && cst != ACUDA_SUCCESS && module->image &&
        module->image_size && cubin_backend() != 1) {
        ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): trying rtxdisam fallback\n", name);
        // The entry-less lift either could not choose a section (multi-entry
        // cubin) or chose the first member of a multi-cubin fatbin. Ask the
        // bridge for this specific entry and prefer the device's reported SM.
        char target[16];
        reported_sm_target(target, sizeof(target));
        char *lifted = NULL;
        if (acuda_rtxdisam_cubin_to_ptx_status_for_target(
                module->image, module->image_size, name, target, &lifted,
                err, sizeof(err)) == ACUDA_RTXDISAM_OK &&
            lifted) {
            // The PTX parser caps identifiers at 256 bytes; torch's
            // mangled kernel names are longer.  Compile under a short
            // synthetic entry and keep the public name in module->funcs.
            const char *compileName = name;
            char shortName[64];
            // Always compile under a short synthetic entry: mangled torch
            // names and the lifter's L__text__<name> labels can exceed the
            // parser's 256-byte identifier cap.
            static unsigned long long serial;
            snprintf(shortName, sizeof(shortName), "__acuda_sass_%llu",
                     ++serial);
            if (getenv("ACUDA_DUMP_PTX_DIR")) {
                static unsigned long long dump_serial;
                char path[512];
                snprintf(path, sizeof(path), "%s/%llu.ptx",
                         getenv("ACUDA_DUMP_PTX_DIR"), ++dump_serial);
                FILE *dfp = fopen(path, "wb");
                if (dfp) {
                    fwrite(lifted, 1, strlen(lifted), dfp);
                    fclose(dfp);
                }
                fprintf(stderr, "acuda-ptxdump %llu target=%s name=%s\n",
                        dump_serial, target, name ? name : "(null)");
            }
            char *renamed = replace_text(lifted, name, shortName);
            if (!renamed) {
                free(lifted);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            compileName = shortName;
            cst = acuda_compile_ptx(renamed, compileName, &k, err,
                                    sizeof(err), acuda_context_get_compiler_caps(g_actx));
            free(renamed);
            free(lifted);
        }
    }
    if (emu_auto)
        unsetenv("ACUDA_EMU_F64");
    if (cst != ACUDA_SUCCESS) {
        if (err[0])
            fprintf(stderr, "ACuda: %s\n", err);
        if (have_fail_dir && !skip_attempts)
            acuda_cache_mark_failed(fail_dir, fail_hash, (int)cst, err);
        // Diagnostic: keep the exact PTX that failed, whether it came
        // from the module image or from an entry-specific re-lift.
        const char *dump_dir = getenv("ACUDA_DUMP_PTX_DIR");
        const char *failed_ptx = (module && module->ptx) ? module->ptx : NULL;
        if (dump_dir && *dump_dir && failed_ptx) {
            static unsigned long dump_fail_serial;
            char path[1024];
            snprintf(path, sizeof(path), "%s/fail_%lu.ptx", dump_dir,
                     ++dump_fail_serial);
            FILE *dfp = fopen(path, "wb");
            if (dfp) {
                fwrite(failed_ptx, 1, strlen(failed_ptx), dfp);
                fclose(dfp);
                fprintf(stderr, "acuda-ptxdump fail_%lu name=%s\n",
                        dump_fail_serial, name ? name : "(null)");
            }
        }
        ACUDA_DEBUG_PRINT("acuda-trace !! cuModuleGetFunction(%s): "
                    "compile status %d err=%s\n", name, (int)cst, err);
        return CUDA_ERROR_NOT_FOUND;
    }
    if (module->nfuncs == module->fcap) {
        size_t ncap = module->fcap ? module->fcap * 2 : 4;
        void *np = malloc(ncap * sizeof(*module->funcs));
        if (!np) {
            acuda_kernel_free(k);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        memcpy(np, module->funcs, module->nfuncs * sizeof(*module->funcs));
        free(module->funcs);
        module->funcs = np;
        module->fcap = ncap;
    }
    strncpy(module->funcs[module->nfuncs].name, name,
            sizeof(module->funcs[module->nfuncs].name) - 1);
    module->funcs[module->nfuncs].name[sizeof(module->funcs[module->nfuncs].name) - 1] =
        '\0';
    module->funcs[module->nfuncs].k = k;
    module->nfuncs++;
    CUfunction f = malloc(sizeof(*f));
    if (!f)
        return CUDA_ERROR_OUT_OF_MEMORY;
    f->magic = ACUDA_HANDLE_FUNC;
    f->mod = module;
    f->k = k;
    snprintf(f->name, sizeof(f->name), "%s", name);
    f->maxdyn_set = 0;
    f->maxdyn = 0;
    *hfunc = f;
    ACUDA_DEBUG_PRINT("acuda-mfunc f=%p mod=%p k=%p name=%s\n",
                (void *)f, (void *)f->mod, (void *)f->k, f->name);
    return CUDA_SUCCESS;
}

CUresult handle_as_func(void *handle, CUfunction *out) {
    if (!handle || !out)
        return CUDA_ERROR_INVALID_HANDLE;
    uint32_t magic = *(const uint32_t *)handle;
    if (magic == ACUDA_HANDLE_FUNC) {
        *out = (CUfunction)handle;
        return CUDA_SUCCESS;
    }
    if (magic == ACUDA_HANDLE_KERNEL) {
        struct CUkernel_st *k = (struct CUkernel_st *)handle;
        if (!k->func) {
            CUfunction f = calloc(1, sizeof(*f));
            if (!f)
                return CUDA_ERROR_OUT_OF_MEMORY;
            f->magic = ACUDA_HANDLE_FUNC;
            f->mod = k->mod;
            snprintf(f->name, sizeof(f->name), "%s", k->name);
            k->func = f;
        }
        *out = k->func;
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_HANDLE;
}

int looks_like_ptx(const char *text) {
    // Skips whitespace and comments; PTX starts with ".version".
    for (;;) {
        while (*text == ' ' || *text == '\t' || *text == '\n' || *text == '\r')
            text++;
        if (text[0] == '/' && text[1] == '/') {
            while (*text && *text != '\n')
                text++;
            continue;
        }
        if (text[0] == '/' && text[1] == '*') {
            text += 2;
            while (*text && !(text[0] == '*' && text[1] == '/'))
                text++;
            if (*text)
                text += 2;
            continue;
        }
        break;
    }
    return strncmp(text, ".version", 8) == 0;
}
