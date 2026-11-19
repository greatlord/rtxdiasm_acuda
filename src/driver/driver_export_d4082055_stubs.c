#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_d4082055.h"
#include "acuda/driver/driver_export_6bd5fb6c.h"

/* {D4082055-BDE6-704B-8D34-BA123C66E1F2} slot implementations.
 *
 * CUDA 13 recomputes this checksum to confirm that the driver and runtime
 * interface tables agree.  The mixing table and round function are the
 * interoperability contract; the C implementation below is independent,
 * cross-checked against the ZLUDA reference (dark_api/src/lib.rs) and its
 * test vectors. */

static const unsigned char acuda_integrity_mix[256] = {
    0x29, 0x2e, 0x43, 0xc9, 0xa2, 0xd8, 0x7c, 0x01, 0x3d, 0x36, 0x54, 0xa1, 0xec, 0xf0, 0x06, 0x13,
    0x62, 0xa7, 0x05, 0xf3, 0xc0, 0xc7, 0x73, 0x8c, 0x98, 0x93, 0x2b, 0xd9, 0xbc, 0x4c, 0x82, 0xca,
    0x1e, 0x9b, 0x57, 0x3c, 0xfd, 0xd4, 0xe0, 0x16, 0x67, 0x42, 0x6f, 0x18, 0x8a, 0x17, 0xe5, 0x12,
    0xbe, 0x4e, 0xc4, 0xd6, 0xda, 0x9e, 0xde, 0x49, 0xa0, 0xfb, 0xf5, 0x8e, 0xbb, 0x2f, 0xee, 0x7a,
    0xa9, 0x68, 0x79, 0x91, 0x15, 0xb2, 0x07, 0x3f, 0x94, 0xc2, 0x10, 0x89, 0x0b, 0x22, 0x5f, 0x21,
    0x80, 0x7f, 0x5d, 0x9a, 0x5a, 0x90, 0x32, 0x27, 0x35, 0x3e, 0xcc, 0xe7, 0xbf, 0xf7, 0x97, 0x03,
    0xff, 0x19, 0x30, 0xb3, 0x48, 0xa5, 0xb5, 0xd1, 0xd7, 0x5e, 0x92, 0x2a, 0xac, 0x56, 0xaa, 0xc6,
    0x4f, 0xb8, 0x38, 0xd2, 0x96, 0xa4, 0x7d, 0xb6, 0x76, 0xfc, 0x6b, 0xe2, 0x9c, 0x74, 0x04, 0xf1,
    0x45, 0x9d, 0x70, 0x59, 0x64, 0x71, 0x87, 0x20, 0x86, 0x5b, 0xcf, 0x65, 0xe6, 0x2d, 0xa8, 0x02,
    0x1b, 0x60, 0x25, 0xad, 0xae, 0xb0, 0xb9, 0xf6, 0x1c, 0x46, 0x61, 0x69, 0x34, 0x40, 0x7e, 0x0f,
    0x55, 0x47, 0xa3, 0x23, 0xdd, 0x51, 0xaf, 0x3a, 0xc3, 0x5c, 0xf9, 0xce, 0xba, 0xc5, 0xea, 0x26,
    0x2c, 0x53, 0x0d, 0x6e, 0x85, 0x28, 0x84, 0x09, 0xd3, 0xdf, 0xcd, 0xf4, 0x41, 0x81, 0x4d, 0x52,
    0x6a, 0xdc, 0x37, 0xc8, 0x6c, 0xc1, 0xab, 0xfa, 0x24, 0xe1, 0x7b, 0x08, 0x0c, 0xbd, 0xb1, 0x4a,
    0x78, 0x88, 0x95, 0x8b, 0xe3, 0x63, 0xe8, 0x6d, 0xe9, 0xcb, 0xd5, 0xfe, 0x3b, 0x00, 0x1d, 0x39,
    0xf2, 0xef, 0xb7, 0x0e, 0x66, 0x58, 0xd0, 0xe4, 0xa6, 0x77, 0x72, 0xf8, 0xeb, 0x75, 0x4b, 0x0a,
    0x31, 0x44, 0x50, 0xb4, 0x8f, 0xed, 0x1f, 0x1a, 0xdb, 0x99, 0x8d, 0x33, 0x9f, 0x11, 0x83, 0x14,
};

static void acuda_integrity_round(unsigned char state[66], unsigned char v) {
    unsigned t1 = state[0x40];
    state[t1 + 0x10] = v;
    unsigned t2 = t1;
    unsigned t3 = (t1 + 1) & 0xf;
    state[t2 + 0x20] = state[t2] ^ v;
    unsigned t4 = acuda_integrity_mix[(v ^ state[0x41]) & 0xff];
    unsigned t5 = state[t2 + 0x30];
    state[t2 + 0x30] = (unsigned char)(t4 ^ t5);
    state[0x41] = (unsigned char)(t4 ^ t5);
    state[0x40] = (unsigned char)t3;
    if (t3 != 0)
        return;
    unsigned char a = 0x29;
    unsigned char c = 0;
    for (;;) {
        a ^= state[0];
        state[0] = a;
        for (unsigned i = 1; i < 0x30; i++) {
            a = (unsigned char)(state[i] ^ acuda_integrity_mix[a]);
            state[i] = a;
        }
        a = (unsigned char)(a + c);
        c = (unsigned char)(c + 1);
        if (c == 0x12)
            break;
        a = acuda_integrity_mix[a];
    }
}

static void acuda_integrity_hash(unsigned char state[66], const void *mix,
                                 size_t len, unsigned char x) {
    const unsigned char *p = (const unsigned char *)mix;
    for (size_t i = 0; i < len; i++)
        acuda_integrity_round(state, (unsigned char)(p[i] ^ x));
}

static void acuda_integrity_finish(unsigned char state[66],
                                   unsigned long long out[2]) {
    unsigned char t = (unsigned char)(16 - state[64]);
    for (unsigned i = 0; i < t; i++)
        acuda_integrity_round(state, t);
    for (unsigned i = 0x30; i < 0x40; i++)
        acuda_integrity_round(state, state[i]);
    memcpy(&out[0], state, sizeof(out[0]));
    memcpy(&out[1], state + 8, sizeof(out[1]));
}

typedef struct {
    uint32_t driver_version;
    uint32_t version;
    uint32_t current_process;
    uint32_t current_thread;
    const void *cudart_table;
    const void *integrity_table;
    const void *fn_address;
    uint64_t unix_seconds;
} acuda_integrity_pass3_t;

typedef struct {
    CUuuid guid;
    int32_t pci_domain;
    int32_t pci_bus;
    int32_t pci_device;
} acuda_integrity_dev_t;

static CUresult ACUDA_CUDAAPI integrity_check(unsigned int version,
                                unsigned long long unix_seconds,
                                unsigned long long result[2]);

static const void *g_integrity_table[3] = {
    (const void *)(uintptr_t)(3 * sizeof(void *)),  /* [0] table size */
    (const void *)(uintptr_t)integrity_check        /* [1] */
};

static CUresult ACUDA_CUDAAPI integrity_check(unsigned int version,
                                unsigned long long unix_seconds,
                                unsigned long long result[2]) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? integrity_check(version=%u)\n",
                version);
    if (!result)
        return CUDA_ERROR_INVALID_VALUE;
    switch (version % 10) {
    case 0:
        result[0] = 0x3341181C03CB675CULL;
        result[1] = 0x8ED383AA1F4CD1E8ULL;
        return CUDA_SUCCESS;
    case 1:
        result[0] = 0x1841181C03CB675CULL;
        result[1] = 0x8ED383AA1F4CD1E8ULL;
        return CUDA_SUCCESS;
    default:
        break;
    }

    static const unsigned char pass1[16] = {
        0x14, 0x6a, 0xdd, 0xae, 0x53, 0xa9, 0xa7, 0x52,
        0xaa, 0x08, 0x41, 0x36, 0x0b, 0xf5, 0x5a, 0x9f
    };
    unsigned char state[66];
    memset(state, 0, sizeof(state));
    acuda_integrity_hash(state, pass1, sizeof(pass1), 0x36);

#if defined(_WIN32)
    uint32_t pid = (uint32_t)GetCurrentProcessId();
    uint32_t tid = (uint32_t)GetCurrentThreadId();
#else
    uint32_t pid = (uint32_t)getpid();
    uint32_t tid = (uint32_t)(uintptr_t)pthread_self();
#endif
    acuda_integrity_pass3_t p3;
    memset(&p3, 0, sizeof(p3));
    p3.driver_version = ACUDA_DRIVER_VERSION;
    p3.version = version;
    p3.current_process = pid;
    p3.current_thread = tid;
    p3.cudart_table = acuda_export_6bd5fb6c_table();
    p3.integrity_table = (const void *)g_integrity_table;
    p3.fn_address = (const void *)(uintptr_t)integrity_check;
    p3.unix_seconds = unix_seconds;
    acuda_integrity_hash(state, &p3, sizeof(p3), 0);

    int ndev = 0;
    if (cuDeviceGetCount(&ndev) == CUDA_SUCCESS) {
        for (int d = 0; d < ndev; d++) {
            acuda_integrity_dev_t dev;
            memset(&dev, 0, sizeof(dev));
            cuDeviceGetUuid(&dev.guid, d);
            int v = 0;
            if (cuDeviceGetAttribute(&v, 50, d) == CUDA_SUCCESS)
                dev.pci_domain = v;
            v = 0;
            if (cuDeviceGetAttribute(&v, 33, d) == CUDA_SUCCESS)
                dev.pci_bus = v;
            v = 0;
            if (cuDeviceGetAttribute(&v, 34, d) == CUDA_SUCCESS)
                dev.pci_device = v;
            acuda_integrity_hash(state, &dev, sizeof(dev), 0);
        }
    }

    acuda_integrity_finish(state, result);   /* pass5 #1 */
    unsigned long long pass5_1[2] = { result[0], result[1] };
    memset(state, 0, 16);
    memset(state + 48, 0, 66 - 48);
    acuda_integrity_hash(state, pass1, sizeof(pass1), 0x5c);
    acuda_integrity_hash(state, pass5_1, sizeof(pass5_1), 0);
    acuda_integrity_finish(state, result);
    return CUDA_SUCCESS;
}

const void *acuda_export_d4082055_table(void) {
    return (const void *)g_integrity_table;
}
