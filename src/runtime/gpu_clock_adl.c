#include "gpu_clock.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>
#include <string.h>

#include "adl_sdk.h"

typedef int (*ADL2_MAIN_CONTROL_CREATE)(ADL_MAIN_MALLOC_CALLBACK, int, ADL_CONTEXT_HANDLE *);
typedef int (*ADL2_MAIN_CONTROL_DESTROY)(ADL_CONTEXT_HANDLE);
typedef int (*ADL2_ADAPTER_NUMBEROFADAPTERS_GET)(ADL_CONTEXT_HANDLE, int *);
typedef int (*ADL2_ADAPTER_ADAPTERINFO_GET)(ADL_CONTEXT_HANDLE, LPAdapterInfo, int);
typedef int (*ADL2_NEW_QUERYPMLDATA_GET)(ADL_CONTEXT_HANDLE, int, ADLPMLogDataOutput *);

static void *__stdcall acuda_adl_alloc(int size)
{
    return malloc((size_t)size);
}

int acuda_query_adl_gpu_clock_khz(uint32_t pci_domain, uint32_t pci_bus,
                                  uint32_t pci_device, uint32_t pci_function,
                                  int *out_graphics_khz, int *out_memory_khz)
{
    if (out_graphics_khz)
        *out_graphics_khz = 0;
    if (out_memory_khz)
        *out_memory_khz = 0;

    /* ADL AdapterInfo reports bus/device/function (no domain). */
    if (pci_domain != 0)
        return 0;

    HMODULE dll = LoadLibraryW(L"atiadlxx.dll");
    if (!dll)
        return 0;

    ADL2_MAIN_CONTROL_CREATE create =
        (ADL2_MAIN_CONTROL_CREATE)GetProcAddress(dll, "ADL2_Main_Control_Create");
    ADL2_MAIN_CONTROL_DESTROY destroy =
        (ADL2_MAIN_CONTROL_DESTROY)GetProcAddress(dll, "ADL2_Main_Control_Destroy");
    ADL2_ADAPTER_NUMBEROFADAPTERS_GET get_count =
        (ADL2_ADAPTER_NUMBEROFADAPTERS_GET)GetProcAddress(dll, "ADL2_Adapter_NumberOfAdapters_Get");
    ADL2_ADAPTER_ADAPTERINFO_GET get_info =
        (ADL2_ADAPTER_ADAPTERINFO_GET)GetProcAddress(dll, "ADL2_Adapter_AdapterInfo_Get");
    ADL2_NEW_QUERYPMLDATA_GET get_pmlog =
        (ADL2_NEW_QUERYPMLDATA_GET)GetProcAddress(dll, "ADL2_New_QueryPMLogData_Get");

    if (!create || !destroy || !get_count || !get_info || !get_pmlog) {
        FreeLibrary(dll);
        return 0;
    }

    ADL_CONTEXT_HANDLE context = NULL;
    if (create(acuda_adl_alloc, 1, &context) != ADL_OK || !context) {
        FreeLibrary(dll);
        return 0;
    }

    int count = 0;
    int ok = 0;
    if (get_count(context, &count) != ADL_OK || count <= 0)
        goto cleanup;

    {
        LPAdapterInfo adapters = (LPAdapterInfo)calloc((size_t)count, sizeof(AdapterInfo));
        if (!adapters)
            goto cleanup;

        if (get_info(context, adapters, count * (int)sizeof(AdapterInfo)) != ADL_OK) {
            free(adapters);
            goto cleanup;
        }

        for (int i = 0; i < count; ++i) {
            if ((uint32_t)adapters[i].iBusNumber != pci_bus ||
                (uint32_t)adapters[i].iDeviceNumber != pci_device ||
                (uint32_t)adapters[i].iFunctionNumber != pci_function)
                continue;

            ADLPMLogDataOutput data;
            memset(&data, 0, sizeof(data));
            data.size = sizeof(data);
            if (get_pmlog(context, adapters[i].iAdapterIndex, &data) != ADL_OK)
                continue;

            if (out_graphics_khz && data.sensors[PMLOG_CLK_GFXCLK].supported) {
                int v = data.sensors[PMLOG_CLK_GFXCLK].value;
                if (v > 0) {
                    *out_graphics_khz = v * 1000; /* MHz -> kHz */
                    ok = 1;
                }
            }
            if (out_memory_khz && data.sensors[PMLOG_CLK_MEMCLK].supported) {
                int v = data.sensors[PMLOG_CLK_MEMCLK].value;
                if (v > 0) {
                    *out_memory_khz = v * 1000; /* MHz -> kHz */
                    ok = 1;
                }
            }
            /* Matched the target BDF; stop after first successful/attempted PMLog. */
            break;
        }

        free(adapters);
    }

cleanup:
    destroy(context);
    FreeLibrary(dll);
    return ok;
}

int acuda_query_adl_gpu_memory_bytes(uint32_t pci_domain, uint32_t pci_bus,
                                     uint32_t pci_device, uint32_t pci_function,
                                     uint64_t *out_bytes)
{
    typedef int (*ADL2_ADAPTER_MEMORYINFO2_GET)(ADL_CONTEXT_HANDLE, int, ADLMemoryInfo2 *);
    typedef int (*ADL2_ADAPTER_MEMORYINFO_GET)(ADL_CONTEXT_HANDLE, int, ADLMemoryInfo *);

    if (out_bytes)
        *out_bytes = 0;

    if (pci_domain != 0)
        return 0;

    HMODULE dll = LoadLibraryW(L"atiadlxx.dll");
    if (!dll)
        return 0;

    ADL2_MAIN_CONTROL_CREATE create =
        (ADL2_MAIN_CONTROL_CREATE)GetProcAddress(dll, "ADL2_Main_Control_Create");
    ADL2_MAIN_CONTROL_DESTROY destroy =
        (ADL2_MAIN_CONTROL_DESTROY)GetProcAddress(dll, "ADL2_Main_Control_Destroy");
    ADL2_ADAPTER_NUMBEROFADAPTERS_GET get_count =
        (ADL2_ADAPTER_NUMBEROFADAPTERS_GET)GetProcAddress(dll, "ADL2_Adapter_NumberOfAdapters_Get");
    ADL2_ADAPTER_ADAPTERINFO_GET get_info =
        (ADL2_ADAPTER_ADAPTERINFO_GET)GetProcAddress(dll, "ADL2_Adapter_AdapterInfo_Get");
    ADL2_ADAPTER_MEMORYINFO2_GET get_mem2 =
        (ADL2_ADAPTER_MEMORYINFO2_GET)GetProcAddress(dll, "ADL2_Adapter_MemoryInfo2_Get");
    ADL2_ADAPTER_MEMORYINFO_GET get_mem =
        (ADL2_ADAPTER_MEMORYINFO_GET)GetProcAddress(dll, "ADL2_Adapter_MemoryInfo_Get");

    if (!create || !destroy || !get_count || !get_info || (!get_mem2 && !get_mem)) {
        FreeLibrary(dll);
        return 0;
    }

    ADL_CONTEXT_HANDLE context = NULL;
    if (create(acuda_adl_alloc, 1, &context) != ADL_OK || !context) {
        FreeLibrary(dll);
        return 0;
    }

    int count = 0;
    int ok = 0;
    if (get_count(context, &count) != ADL_OK || count <= 0)
        goto cleanup;

    {
        LPAdapterInfo adapters = (LPAdapterInfo)calloc((size_t)count, sizeof(AdapterInfo));
        if (!adapters)
            goto cleanup;

        if (get_info(context, adapters, count * (int)sizeof(AdapterInfo)) != ADL_OK) {
            free(adapters);
            goto cleanup;
        }

        for (int i = 0; i < count; ++i) {
            if ((uint32_t)adapters[i].iBusNumber != pci_bus ||
                (uint32_t)adapters[i].iDeviceNumber != pci_device ||
                (uint32_t)adapters[i].iFunctionNumber != pci_function)
                continue;

            if (get_mem2) {
                ADLMemoryInfo2 mem2;
                memset(&mem2, 0, sizeof(mem2));
                if (get_mem2(context, adapters[i].iAdapterIndex, &mem2) == ADL_OK) {
                    uint64_t bytes = 0;
                    if (mem2.iMemorySize > 0)
                        bytes = (uint64_t)mem2.iMemorySize;
                    if (bytes == 0 && mem2.iVisibleMemorySize > 0)
                        bytes = (uint64_t)mem2.iVisibleMemorySize;
                    if (bytes > 0) {
                        if (out_bytes)
                            *out_bytes = bytes;
                        ok = 1;
                        break;
                    }
                }
            }
            if (get_mem) {
                ADLMemoryInfo mem;
                memset(&mem, 0, sizeof(mem));
                if (get_mem(context, adapters[i].iAdapterIndex, &mem) == ADL_OK &&
                    mem.iMemorySize > 0) {
                    if (out_bytes)
                        *out_bytes = (uint64_t)mem.iMemorySize;
                    ok = 1;
                    break;
                }
            }
            break;
        }

        free(adapters);
    }

cleanup:
    destroy(context);
    FreeLibrary(dll);
    return ok;
}

#else /* !_WIN32 */

int acuda_query_adl_gpu_clock_khz(uint32_t pci_domain, uint32_t pci_bus,
                                  uint32_t pci_device, uint32_t pci_function,
                                  int *out_graphics_khz, int *out_memory_khz)
{
    (void)pci_domain;
    (void)pci_bus;
    (void)pci_device;
    (void)pci_function;
    if (out_graphics_khz)
        *out_graphics_khz = 0;
    if (out_memory_khz)
        *out_memory_khz = 0;
    return 0;
}

int acuda_query_adl_gpu_memory_bytes(uint32_t pci_domain, uint32_t pci_bus,
                                     uint32_t pci_device, uint32_t pci_function,
                                     uint64_t *out_bytes)
{
    (void)pci_domain;
    (void)pci_bus;
    (void)pci_device;
    (void)pci_function;
    if (out_bytes)
        *out_bytes = 0;
    return 0;
}

#endif /* _WIN32 */
