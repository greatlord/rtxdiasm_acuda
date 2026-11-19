#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_stub.h"
#include "acuda/driver/driver_export_dd9e98a4.h"
#include "acuda/driver/driver_export_6bd5fb6c.h"
#include "acuda/driver/driver_export_a094798c.h"
#include "acuda/driver/driver_export_42d85a81.h"
#include "acuda/driver/driver_export_c693336e.h"
#include "acuda/driver/driver_export_0ca50b8c.h"
#include "acuda/driver/driver_export_263e8860.h"
#include "acuda/driver/driver_export_d4082055.h"
#include "acuda/driver/driver_export_21318c60.h"
#include "acuda/driver/driver_export_195bcbf4.h"
#include "acuda/runtime_api.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#if defined(_WIN32)
#include <windows.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#else
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#endif

static void acuda_log_caller(const char *tag) {
#if defined(_WIN32) && defined(_MSC_VER)
    void *retaddr = _ReturnAddress();
    MEMORY_BASIC_INFORMATION mbi;
    char modpath[MAX_PATH] = "?";
    HMODULE mod = NULL;
    if (VirtualQuery(retaddr, &mbi, sizeof(mbi))) {
        mod = (HMODULE)mbi.AllocationBase;
        GetModuleFileNameA(mod, modpath, MAX_PATH);
    }
    ACUDA_DEBUG_PRINT( "ACuda-debug: %s caller=%p modbase=%p mod=%s\n",
            tag, retaddr, (void *)mod, modpath);
#else
    (void)tag;
#endif
}

/* Export-table dispatch.  Each interface has its own
 * driver_export_<guid>.c (GUID match + lookup) and
 * driver_export_<guid>_stubs.c (table contents), mirroring the dd9e98a4
 * hybrid-runtime split. */
CUresult ACUDA_CUDAAPI cuGetExportTable(const void **ppExportTable,
                          const struct CUuuid_st *pExportTableId) {
    acuda_log_caller("cuGetExportTable");
    if (acuda_export_table_is_dd9e98a4(pExportTableId))
        return acuda_export_table_get_dd9e98a4(ppExportTable);

    if (!ppExportTable || !pExportTableId) {
        ACUDA_DEBUG_PRINT( "ACuda-debug: cuGetExportTable -> INVALID_VALUE (pp=%p id=%p)\n",
                (const void *)ppExportTable, (const void *)pExportTableId);
        return CUDA_ERROR_INVALID_VALUE;
    }
    ACUDA_DEBUG_PRINT( "ACuda-debug: cuGetExportTable(%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x)\n",
            (unsigned char)pExportTableId->bytes[0],
            (unsigned char)pExportTableId->bytes[1],
            (unsigned char)pExportTableId->bytes[2],
            (unsigned char)pExportTableId->bytes[3],
            (unsigned char)pExportTableId->bytes[4],
            (unsigned char)pExportTableId->bytes[5],
            (unsigned char)pExportTableId->bytes[6],
            (unsigned char)pExportTableId->bytes[7],
            (unsigned char)pExportTableId->bytes[8],
            (unsigned char)pExportTableId->bytes[9],
            (unsigned char)pExportTableId->bytes[10],
            (unsigned char)pExportTableId->bytes[11],
            (unsigned char)pExportTableId->bytes[12],
            (unsigned char)pExportTableId->bytes[13],
            (unsigned char)pExportTableId->bytes[14],
            (unsigned char)pExportTableId->bytes[15]);
    if (getenv("ACUDA_TRACE")) {
#if defined(_M_IX86)
        void *retaddr = _ReturnAddress();
        MEMORY_BASIC_INFORMATION mbi;
        char modpath[MAX_PATH] = "?";
        if (VirtualQuery(retaddr, &mbi, sizeof(mbi)))
            GetModuleFileNameA((HMODULE)mbi.AllocationBase, modpath, MAX_PATH);
        ACUDA_DEBUG_PRINT( "acuda-trace .. cuGetExportTable ret=%p rva=%Ix mod=%s pp=%p\n",
                retaddr, (uintptr_t)retaddr - (uintptr_t)mbi.AllocationBase,
                modpath, (const void *)ppExportTable);
#endif
        ACUDA_DEBUG_PRINT( "acuda-trace ?? cuGetExportTable(%02x%02x%02x%02x-"
                "%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x)\n",
                (unsigned char)pExportTableId->bytes[0],
                (unsigned char)pExportTableId->bytes[1],
                (unsigned char)pExportTableId->bytes[2],
                (unsigned char)pExportTableId->bytes[3],
                (unsigned char)pExportTableId->bytes[4],
                (unsigned char)pExportTableId->bytes[5],
                (unsigned char)pExportTableId->bytes[6],
                (unsigned char)pExportTableId->bytes[7],
                (unsigned char)pExportTableId->bytes[8],
                (unsigned char)pExportTableId->bytes[9],
                (unsigned char)pExportTableId->bytes[10],
                (unsigned char)pExportTableId->bytes[11],
                (unsigned char)pExportTableId->bytes[12],
                (unsigned char)pExportTableId->bytes[13],
                (unsigned char)pExportTableId->bytes[14],
                (unsigned char)pExportTableId->bytes[15]);
    }

    if (acuda_export_table_is_6bd5fb6c(pExportTableId))
        return acuda_export_table_get_6bd5fb6c(ppExportTable);
    if (acuda_export_table_is_a094798c(pExportTableId))
        return acuda_export_table_get_a094798c(ppExportTable);
    if (acuda_export_table_is_42d85a81(pExportTableId))
        return acuda_export_table_get_42d85a81(ppExportTable);
    if (acuda_export_table_is_c693336e(pExportTableId))
        return acuda_export_table_get_c693336e(ppExportTable);
    if (acuda_export_table_is_0ca50b8c(pExportTableId))
        return acuda_export_table_get_0ca50b8c(ppExportTable);
    if (acuda_export_table_is_263e8860(pExportTableId))
        return acuda_export_table_get_263e8860(ppExportTable);
    if (acuda_export_table_is_d4082055(pExportTableId))
        return acuda_export_table_get_d4082055(ppExportTable);
    if (acuda_export_table_is_21318c60(pExportTableId))
        return acuda_export_table_get_21318c60(ppExportTable);
    if (acuda_export_table_is_195bcbf4(pExportTableId))
        return acuda_export_table_get_195bcbf4(ppExportTable);

    ACUDA_DEBUG_PRINT( "ACuda-debug: cuGetExportTable -> NOT_FOUND\n");
    return CUDA_ERROR_NOT_FOUND;
}
