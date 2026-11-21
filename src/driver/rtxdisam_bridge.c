// Optional bridge to rtxdisam's native C ABI.
#if !defined(_WIN32) && !defined(_GNU_SOURCE)
// dladdr() lives behind _GNU_SOURCE on glibc; define it before the first
// system header so the sibling-directory lookup below compiles cleanly.
#define _GNU_SOURCE
#endif
//
// ACuda can load PTX from cubin/fatbin containers, but it does not decode
// SASS. When an image contains only SM-architecture code, dynamically load
// rtxdisam_native and ask it to lift the image to PTX. The dependency is
// deliberately optional: a missing library leaves the existing clean
// CUDA_ERROR_INVALID_IMAGE path intact.
#include "acuda/driver/rtxdisam_bridge.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#endif

typedef int (*rtxdisam_bytes_fn)(const uint8_t *data, size_t size,
                                 const char *entry, const char *parameters,
                                 int output, char **sass, char **ptx);
typedef int (*rtxdisam_bytes_target_fn)(
    const uint8_t *data, size_t size, const char *entry,
    const char *parameters, const char *target, int output,
    char **sass, char **ptx);
typedef void (*rtxdisam_free_fn)(void *memory);
typedef const char *(*rtxdisam_error_fn)(void);

static void *g_library;
static rtxdisam_bytes_fn g_bytes_to_text;
static rtxdisam_bytes_target_fn g_bytes_to_text_with_target;
static rtxdisam_free_fn g_free;
static rtxdisam_error_fn g_last_error;

static uint32_t rd32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const unsigned char *p) {
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

static void set_error(char *err, size_t errlen, const char *message) {
    if (err && errlen)
        snprintf(err, errlen, "%s", message ? message : "rtxdisam error");
}

static void *open_library(const char *name) {
#ifdef _WIN32
    return (void *)LoadLibraryA(name);
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void *find_symbol(void *library, const char *name) {
#ifdef _WIN32
    return (void *)GetProcAddress((HMODULE)library, name);
#else
    return dlsym(library, name);
#endif
}

static int load_library(char *err, size_t errlen) {
    if (g_library && g_bytes_to_text && g_free)
        return 0;

    const char *env = getenv("RTXDISAM_LIBRARY");
    const char *candidates[] = {
#ifdef _WIN32
        "rtxdisam_native.dll",
        "rtxdisam.dll",
#else
        "librtxdisam_native.so",
        "rtxdisam_native.so",
        "librtxdisam.so",
        "rtxdisam.so",
#endif
        NULL,
    };
    if (env && env[0]) {
        g_library = open_library(env);
        if (!g_library) {
            char message[512];
            snprintf(message, sizeof(message),
                     "cannot load rtxdisam library '%s'", env);
            set_error(err, errlen, message);
            return -1;
        }
    } else {
        for (int i = 0; candidates[i]; i++) {
            g_library = open_library(candidates[i]);
            if (g_library)
                break;
        }
#ifndef _WIN32
        if (!g_library) {
            // The loader searches the process rpath/LD_LIBRARY_PATH, not
            // the directory this libcuda.so lives in.  ACuda stages the
            // lifter next to its own libraries, so try that first.
            Dl_info di;
            if (dladdr((void *)(uintptr_t)&load_library, &di) &&
                di.dli_fname) {
                const char *slash = strrchr(di.dli_fname, '/');
                if (slash) {
                    char path[1024];
                    int dirn = (int)(slash - di.dli_fname);
                    static const char *siblings[] = {
                        "librtxdisam_native.so", "rtxdisam_native.so",
                        NULL,
                    };
                    for (int i = 0; !g_library && siblings[i]; i++) {
                        snprintf(path, sizeof(path), "%.*s/%s", dirn,
                                 di.dli_fname, siblings[i]);
                        g_library = open_library(path);
                    }
                }
            }
        }
#endif
    }
    if (!g_library) {
        set_error(err, errlen,
                  "rtxdisam native library is not available "
                  "(set RTXDISAM_LIBRARY to override)");
        return -1;
    }

    g_bytes_to_text = (rtxdisam_bytes_fn)find_symbol(
        g_library, "rtxdisam_cubin_bytes_to_text");
    g_bytes_to_text_with_target = (rtxdisam_bytes_target_fn)find_symbol(
        g_library, "rtxdisam_cubin_bytes_to_text_with_target");
    g_free = (rtxdisam_free_fn)find_symbol(g_library, "rtxdisam_free");
    g_last_error = (rtxdisam_error_fn)find_symbol(
        g_library, "rtxdisam_last_error");
    if (!g_bytes_to_text || !g_free) {
        set_error(err, errlen,
                  "rtxdisam library is missing cubin_bytes_to_text/free");
        return -1;
    }
    return 0;
}

size_t acuda_image_size(const unsigned char *data) {
    if (!data)
        return 0;
    if (data[0] == 0x7f && data[1] == 'E' && data[2] == 'L' &&
        data[3] == 'F') {
        int is64 = data[4] == 2;
        uint64_t end = 0;
        if (is64) {
            uint64_t phoff = rd64(data + 0x20);
            uint64_t shoff = rd64(data + 0x28);
            uint16_t phentsize = (uint16_t)(data[0x36] | (data[0x37] << 8));
            uint16_t phnum = (uint16_t)(data[0x38] | (data[0x39] << 8));
            uint16_t shentsize = (uint16_t)(data[0x3a] | (data[0x3b] << 8));
            uint16_t shnum = (uint16_t)(data[0x3c] | (data[0x3d] << 8));
            uint64_t phend = phoff + (uint64_t)phentsize * phnum;
            uint64_t shend = shoff + (uint64_t)shentsize * shnum;
            end = phend > shend ? phend : shend;
        } else {
            uint32_t phoff = rd32(data + 0x1c);
            uint32_t shoff = rd32(data + 0x20);
            uint16_t phentsize = (uint16_t)(data[0x2a] | (data[0x2b] << 8));
            uint16_t phnum = (uint16_t)(data[0x2c] | (data[0x2d] << 8));
            uint16_t shentsize = (uint16_t)(data[0x2e] | (data[0x2f] << 8));
            uint16_t shnum = (uint16_t)(data[0x30] | (data[0x31] << 8));
            uint32_t phend = phoff + (uint32_t)phentsize * phnum;
            uint32_t shend = shoff + (uint32_t)shentsize * shnum;
            end = phend > shend ? phend : shend;
        }
        if (end < 64 || end > ((uint64_t)1 << 30))
            return 0;
        return (size_t)(end + 64);
    }
    if (rd32(data) == 0xBA55ED50u) {
        uint64_t files_size = rd64(data + 8);
        if (files_size > ((uint64_t)1 << 30))
            return 0;
        return (size_t)(16 + files_size);
    }
    return 0;
}

static int needs_entry_error(const char *message) {
    // The current native C ABI reports an entry-less multi-section lift as
    // exception text; treat those two messages as "retry with an entry".
    return message &&
           (strstr(message, "requires an entry name") != NULL ||
            strstr(message, "PTX output is ambiguous") != NULL);
}

static void dump_cubin_on_error(const unsigned char *data, size_t size) {
    // Diagnostic hook: when a SASS lift fails, keep the exact image that
    // failed so the instruction/relocation can be reproduced offline.
    const char *dir = getenv("ACUDA_DUMP_CUBIN_ON_ERROR");
    if (!dir || !*dir || !data || size == 0)
        return;
#ifdef _WIN32
    _mkdir(dir);
#else
    mkdir(dir, 0755);
#endif
    static int dump_n;
    char path[1024];
    snprintf(path, sizeof(path), "%s/cubin_%d.bin", dir, dump_n++);
    FILE *f = fopen(path, "wb");
    if (!f)
        return;
    fwrite(data, 1, size, f);
    fclose(f);
}

#ifdef _WIN32
// NativeAOT has no win-x86 target, so a 32-bit ACuda cannot load the
// 64-bit rtxdisam_native.dll in-process.  Fall back to a small 64-bit
// helper process that lifts the image and writes PTX to a temp file.
static acuda_rtxdisam_status_t lift_with_helper(
    const unsigned char *data, size_t size, const char *entry,
    const char *target, char **ptx_out, char *err, size_t errlen) {
    const char *helper = getenv("RTXDISAM_HOST");
    char helper_buf[MAX_PATH];
    if (!helper || !*helper) {
        helper = "rtxdisam_host.exe";
        /* Search relative to the ACuda module (cuda.dll) and common build locations. */
        HMODULE self = NULL;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR)&lift_with_helper, &self) && self) {
            char mod[MAX_PATH];
            DWORD n = GetModuleFileNameA(self, mod, MAX_PATH);
            if (n > 0 && n < MAX_PATH) {
                for (int i = (int)n - 1; i >= 0; i--) {
                    if (mod[i] == '\\' || mod[i] == '/') { mod[i] = '\0'; break; }
                }
                static const char *rel[] = {
                    "\\rtxdisam_host.exe",
                    "\\..\\..\\..\\..\\..\\out\\win-x64\\rtxdisam_host.exe",
                    "\\..\\..\\..\\..\\..\\out\\win64-test\\rtxdisam_host.exe",
                    "\\..\\..\\..\\..\\out\\win-x64\\rtxdisam_host.exe",
                    "\\..\\win-x64\\rtxdisam_host.exe",
                    "\\..\\..\\win-x64\\rtxdisam_host.exe",
                    "\\..\\..\\..\\win-x64\\rtxdisam_host.exe",
                    NULL
                };
                int found = 0;
                for (int r = 0; rel[r] && !found; r++) {
                    snprintf(helper_buf, sizeof(helper_buf), "%s%s", mod, rel[r]);
                    if (GetFileAttributesA(helper_buf) != INVALID_FILE_ATTRIBUTES) {
                        helper = helper_buf;
                        found = 1;
                    }
                }
                if (!found) {
                    char exedir[MAX_PATH];
                    DWORD en = GetModuleFileNameA(NULL, exedir, MAX_PATH);
                    if (en > 0 && en < MAX_PATH) {
                        for (int i = (int)en - 1; i >= 0; i--) {
                            if (exedir[i] == '\\' || exedir[i] == '/') { exedir[i] = '\0'; break; }
                        }
                        for (int r = 0; rel[r] && !found; r++) {
                            snprintf(helper_buf, sizeof(helper_buf), "%s%s", exedir, rel[r]);
                            if (GetFileAttributesA(helper_buf) != INVALID_FILE_ATTRIBUTES) {
                                helper = helper_buf;
                                found = 1;
                            }
                        }
                    }
                }
                if (!found) {
                    HMODULE nvcuda = GetModuleHandleA("nvcuda.dll");
                    if (nvcuda) {
                        char npath[MAX_PATH];
                        DWORD nn = GetModuleFileNameA(nvcuda, npath, MAX_PATH);
                        if (nn > 0 && nn < MAX_PATH) {
                            for (int i = (int)nn - 1; i >= 0; i--) {
                                if (npath[i] == '\\' || npath[i] == '/') { npath[i] = '\0'; break; }
                            }
                            for (int r = 0; rel[r] && !found; r++) {
                                snprintf(helper_buf, sizeof(helper_buf), "%s%s", npath, rel[r]);
                                if (GetFileAttributesA(helper_buf) != INVALID_FILE_ATTRIBUTES) {
                                    helper = helper_buf;
                                    found = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    char temp[MAX_PATH];
    if (GetTempPathA(MAX_PATH, temp) == 0)
        snprintf(temp, sizeof(temp), ".");
    static unsigned long counter;
    char in_path[MAX_PATH];
    char out_path[MAX_PATH];
    char log_path[MAX_PATH];
    snprintf(in_path, sizeof(in_path), "%sacuda_sass_%lu_%lu.bin",
             temp, (unsigned long)GetCurrentProcessId(), ++counter);
    snprintf(out_path, sizeof(out_path), "%sacuda_sass_%lu_%lu.ptx",
             temp, (unsigned long)GetCurrentProcessId(), counter);
    snprintf(log_path, sizeof(log_path), "%sacuda_sass_%lu_%lu.log",
             temp, (unsigned long)GetCurrentProcessId(), counter);

    FILE *f = fopen(in_path, "wb");
    if (!f) {
        set_error(err, errlen, "rtxdisam: cannot create temp cubin");
        return ACUDA_RTXDISAM_ERROR;
    }
    int wrote = fwrite(data, 1, size, f) == size;
    fclose(f);
    if (!wrote) {
        DeleteFileA(in_path);
        set_error(err, errlen, "rtxdisam: cannot write temp cubin");
        return ACUDA_RTXDISAM_ERROR;
    }

    char command[4096];
    snprintf(command, sizeof(command), "\"%s\" \"%s\" \"%s\" \"%s\" \"%s\"",
             helper, in_path, out_path,
             target && target[0] ? target : "sm_86",
             entry && entry[0] ? entry : "-");
    char mutable_command[4096];
    snprintf(mutable_command, sizeof(mutable_command), "%s", command);

    STARTUPINFOA startup;
    PROCESS_INFORMATION process;
    memset(&startup, 0, sizeof(startup));
    startup.cb = sizeof(startup);
    /* The log handle must be inheritable or the child's redirected stderr
     * is lost (CreateProcess only inherits handles marked inheritable).
     * Without the helper's message we cannot classify retryable failures
     * such as "requires an entry name" and fall back to per-entry lifting. */
    SECURITY_ATTRIBUTES sa;
    memset(&sa, 0, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE log = CreateFileA(
        log_path, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (log != INVALID_HANDLE_VALUE) {
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdOutput = log;
        startup.hStdError = log;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    }
    memset(&process, 0, sizeof(process));
    BOOL started = CreateProcessA(
        NULL, mutable_command, NULL, NULL, TRUE, CREATE_NO_WINDOW,
        NULL, NULL, &startup, &process);
    if (!started) {
        if (log != INVALID_HANDLE_VALUE)
            CloseHandle(log);
        DeleteFileA(in_path);
        DeleteFileA(log_path);
        char message[512];
        snprintf(message, sizeof(message),
                 "rtxdisam: cannot start '%s' (error %lu)",
                 helper, (unsigned long)GetLastError());
        set_error(err, errlen, message);
        return ACUDA_RTXDISAM_ERROR;
    }
    WaitForSingleObject(process.hProcess, 120000);
    DWORD exit_code = 1;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    if (log != INVALID_HANDLE_VALUE)
        CloseHandle(log);

    char *ptx = NULL;
    if (exit_code == 0) {
        FILE *out = fopen(out_path, "rb");
        if (out) {
            if (fseek(out, 0, SEEK_END) == 0) {
                long n = ftell(out);
                if (n > 0 && fseek(out, 0, SEEK_SET) == 0) {
                    ptx = malloc((size_t)n + 1);
                    if (ptx && fread(ptx, 1, (size_t)n, out) == (size_t)n) {
                        ptx[n] = '\0';
                    } else {
                        free(ptx);
                        ptx = NULL;
                    }
                }
            }
            fclose(out);
        }
    }
    DeleteFileA(in_path);
    DeleteFileA(out_path);
    if (!ptx) {
        char helper_error[1024] = {0};
        FILE *lf = fopen(log_path, "rb");
        if (lf) {
            size_t n = fread(helper_error, 1, sizeof(helper_error) - 1, lf);
            helper_error[n] = '\0';
            fclose(lf);
        }
        // .NET writes stderr as UTF-16 (FF FE ... 00-interleaved ASCII),
        // which strstr() cannot match.  Normalize to narrow text first.
        char helper_narrow[1024];
        {
            const unsigned char *raw =
                (const unsigned char *)helper_error;
            size_t k = 0;
            int wide_bom = (raw[0] == 0xFF && raw[1] == 0xFE);
            // Also detect BOM-less UTF-16LE (e.g. .NET stderr when redirected
            // to a file): ASCII char followed by a NUL.
            int wide = wide_bom ||
                       (raw[0] >= 0x20 && raw[0] < 0x7F && raw[1] == 0 &&
                        raw[2] != 0);
            if (wide) {
                size_t i = wide_bom ? 2 : 0;
                while (raw[i] && k + 1 < sizeof(helper_narrow)) {
                    // UTF-16LE ASCII: low byte is the character.
                    if (i + 1 >= sizeof(helper_error))
                        break;
                    if (raw[i + 1] == 0 &&
                        raw[i] >= 0x20 && raw[i] < 0x7F)
                        helper_narrow[k++] = (char)raw[i];
                    else if (raw[i + 1] == 0 && raw[i] != 0 &&
                             (raw[i] < 0x20 || raw[i] >= 0x7F))
                        helper_narrow[k++] = '?';
                    // else: skip non-ASCII code units
                    i += 2;
                }
                helper_narrow[k] = '\0';
            } else {
                snprintf(helper_narrow, sizeof(helper_narrow), "%s",
                         helper_error);
            }
        }
        // Mirror the in-process path: an entry-less lift of a multi-entry
        // image is retryable with a specific entry, not fatal.  Without
        // this, 32-bit clients always fail fatbins the 64-bit path accepts.
        if (needs_entry_error(helper_narrow[0] ? helper_narrow : NULL)) {
            DeleteFileA(log_path);
            return ACUDA_RTXDISAM_ENTRY_REQUIRED;
        }
        char message[1600];
        snprintf(message, sizeof(message),
                 "rtxdisam: helper exit %lu (size=%zu magic=%08x target=%s entry=%s)%s%s",
                 (unsigned long)exit_code, size,
                 size >= 4 ? rd32(data) : 0,
                 target && target[0] ? target : "(none)",
                 entry && entry[0] ? entry : "(none)",
                 helper_narrow[0] ? ": " : "",
                 helper_narrow[0] ? helper_narrow : " (no PTX produced)");
        set_error(err, errlen, message);
        DeleteFileA(log_path);
        return ACUDA_RTXDISAM_ERROR;
    }
    DeleteFileA(log_path);
    *ptx_out = ptx;
    return ACUDA_RTXDISAM_OK;
}
#endif

static acuda_rtxdisam_status_t lift_status(
    const unsigned char *data, size_t size, const char *entry,
    const char *target, char **ptx_out, char *err, size_t errlen) {
    if (!ptx_out) {
        set_error(err, errlen, "rtxdisam: missing PTX output pointer");
        return ACUDA_RTXDISAM_ERROR;
    }
    *ptx_out = NULL;
    if (!data || size == 0) {
        set_error(err, errlen, "rtxdisam: empty image");
        return ACUDA_RTXDISAM_ERROR;
    }
    if (load_library(err, errlen) != 0) {
#ifdef _WIN32
        return lift_with_helper(
            data, size, entry, target, ptx_out, err, errlen);
#else
        return ACUDA_RTXDISAM_ERROR;
#endif
    }

    char *ptx = NULL;
    char *sass = NULL;
    int rc = target && target[0] && g_bytes_to_text_with_target
        ? g_bytes_to_text_with_target(
              data, size, entry, NULL, target, 2, &sass, &ptx)
        : g_bytes_to_text(data, size, entry, NULL, 2, &sass, &ptx);
    if (sass)
        g_free(sass);
    if (rc != 0 || !ptx) {
        const char *message = g_last_error ? g_last_error() : NULL;
        set_error(err, errlen,
                  message ? message : "rtxdisam could not lift the image");
        if (ptx)
            g_free(ptx);
        if (!entry && needs_entry_error(message))
            return ACUDA_RTXDISAM_ENTRY_REQUIRED;
        dump_cubin_on_error(data, size);
        return ACUDA_RTXDISAM_ERROR;
    }
    size_t len = strlen(ptx);
    char *copy = malloc(len + 1);
    if (!copy) {
        g_free(ptx);
        set_error(err, errlen, "rtxdisam: out of memory");
        return ACUDA_RTXDISAM_ERROR;
    }
    memcpy(copy, ptx, len + 1);
    g_free(ptx);
    *ptx_out = copy;
    return ACUDA_RTXDISAM_OK;
}

acuda_rtxdisam_status_t acuda_rtxdisam_cubin_to_ptx_status(
    const unsigned char *data, size_t size, const char *entry,
    char **ptx_out, char *err, size_t errlen) {
    return lift_status(data, size, entry, NULL, ptx_out, err, errlen);
}

acuda_rtxdisam_status_t acuda_rtxdisam_cubin_to_ptx_status_for_target(
    const unsigned char *data, size_t size, const char *entry,
    const char *target, char **ptx_out, char *err, size_t errlen) {
    return lift_status(data, size, entry, target, ptx_out, err, errlen);
}

char *acuda_rtxdisam_cubin_to_ptx_for_target(
    const unsigned char *data, size_t size, const char *entry,
    const char *target, char *err, size_t errlen) {
    char *ptx = NULL;
    if (acuda_rtxdisam_cubin_to_ptx_status_for_target(
            data, size, entry, target, &ptx, err, errlen) !=
        ACUDA_RTXDISAM_OK)
        return NULL;
    return ptx;
}

char *acuda_rtxdisam_cubin_to_ptx(const unsigned char *data, size_t size,
                                  const char *entry,
                                  char *err, size_t errlen) {
    return acuda_rtxdisam_cubin_to_ptx_for_target(
        data, size, entry, NULL, err, errlen);
}
