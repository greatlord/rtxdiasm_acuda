cmake_minimum_required(VERSION 3.21)
project(rtxdiasm_acuda LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# rtxdiasm_acuda: ACuda CUDA Driver API (libcuda/nvcuda) + internal runtime foundation.
# Outputs (as requested):
#   Linux:   libcuda.so.1 (OUTPUT_NAME cuda, SOVERSION 1)
#   Windows: nvcuda.dll

option(ACUDA_BUILD_TESTS "Build ACuda tests/examples (normally built by integrating project)" OFF)
option(ACUDA_USE_HWINFO "Use hwinfo for real GPU clock queries" ON)

if(NOT CMAKE_BUILD_TYPE)
  # When built standalone, allow defaulting; integrating builds may enforce.
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Vulkan REQUIRED)

if(MSVC)
  if(NOT DEFINED ACUDA_WARNINGS)
    set(ACUDA_WARNINGS /W3)
  endif()
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
  if(NOT DEFINED ACUDA_WARNINGS)
    set(ACUDA_WARNINGS -Wall -Wextra -Werror)
  endif()
endif()

# trace.c is GCC -finstrument-functions hook and POSIX-only.
if(WIN32)
  set(ACUDA_TRACE_SOURCE_INTERNAL)
else()
  set(ACUDA_TRACE_SOURCE_INTERNAL src/driver/trace.c)
endif()

# --- hwinfo (real GPU clock, NVML/Level Zero/OpenCL/sysfs backends loaded at runtime) ---
if(ACUDA_USE_HWINFO)
  set(HWINFO_STATIC ON CACHE BOOL "Build hwinfo as static library" FORCE)
  set(HWINFO_SHARED OFF CACHE BOOL "Do not build hwinfo shared library" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "Disable hwinfo examples" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "Disable hwinfo tests" FORCE)
  set(HWINFO_OS OFF CACHE BOOL "" FORCE)
  set(HWINFO_COMPUTER OFF CACHE BOOL "" FORCE)
  set(HWINFO_MAINBOARD OFF CACHE BOOL "" FORCE)
  set(HWINFO_CPU OFF CACHE BOOL "" FORCE)
  set(HWINFO_DISK OFF CACHE BOOL "" FORCE)
  set(HWINFO_RAM OFF CACHE BOOL "" FORCE)
  set(HWINFO_GPU ON CACHE BOOL "" FORCE)
  set(HWINFO_GPU_BACKENDS ON CACHE BOOL "" FORCE)
  set(HWINFO_BATTERY OFF CACHE BOOL "" FORCE)
  set(HWINFO_NETWORK OFF CACHE BOOL "" FORCE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/hwinfo/CMakeLists.txt")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/hwinfo" "${CMAKE_CURRENT_BINARY_DIR}/hwinfo" EXCLUDE_FROM_ALL)
  else()
    message(FATAL_ERROR "ACUDA_USE_HWINFO is ON but external/hwinfo is missing")
  endif()
  set(ACUDA_GPU_CLOCK_SOURCE src/runtime/gpu_clock_hwinfo.cpp)
else()
  set(ACUDA_GPU_CLOCK_SOURCE)
endif()
# AMD ADL2 clock query (Windows, runtime-loaded atiadlxx.dll) with hwinfo fallback.
if(WIN32)
  list(APPEND ACUDA_GPU_CLOCK_SOURCE src/runtime/gpu_clock_adl.c)
endif()
# NVML live clocks/utilisation: the Win32 fallback (hwinfo cannot build for
# Win32) and a live source on Win64.  Stubs exist for other platforms.
list(APPEND ACUDA_GPU_CLOCK_SOURCE src/runtime/gpu_clock_nvml.c)

# --- Runtime library (internal foundation used by driver/consumers) ---
add_library(acuda_runtime STATIC
  src/runtime/error.c
  src/runtime/context_init.c
  src/runtime/context_query.c
  src/runtime/context_highperf.c
  src/runtime/context_probe.c
  src/runtime/context_setup.c
  src/runtime/context_shutdown.c
  src/runtime/stream.c
  src/runtime/stream_profile.c
  src/runtime/stream_sync.c
  src/runtime/stream_batch.c
  src/runtime/stream_flush.c
  src/runtime/stream_exec.c
  src/runtime/stream_lifecycle.c
  src/runtime/stream_track.c
  src/runtime/event.c
  src/runtime/memory.c
  src/runtime/texture_image.c
  src/runtime/texture_create.c
  src/runtime/texture_fill.c
  src/runtime/texture_io.c
  src/runtime/texture_api.c
  src/runtime/launch.c
  src/runtime/launch_api.c
  src/runtime/launch_host.c
  src/runtime/launch_cache.c
  src/runtime/hostcall.c
  src/runtime/barrier.c
  ${ACUDA_GPU_CLOCK_SOURCE}
)
target_include_directories(acuda_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
)
target_link_libraries(acuda_runtime PRIVATE Vulkan::Vulkan)
if(ACUDA_USE_HWINFO AND TARGET lfreist-hwinfo::gpu)
  target_compile_definitions(acuda_runtime PRIVATE ACUDA_USE_HWINFO=1)
  target_link_libraries(acuda_runtime PRIVATE lfreist-hwinfo::gpu)
elseif(ACUDA_USE_HWINFO)
  target_compile_definitions(acuda_runtime PRIVATE ACUDA_USE_HWINFO=1)
endif()
if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/display-library/include/adl_sdk.h")
  target_include_directories(acuda_runtime PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/external/display-library/include")
endif()
target_compile_options(acuda_runtime PRIVATE ${ACUDA_WARNINGS})
set_target_properties(acuda_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- PTX->SPIR-V compiler dependency ---
# Integrating builds can provide the target/path explicitly.
if(NOT TARGET acuda_compiler)
  if(NOT DEFINED RTXDIASM_PTX_TO_SPIRV_DIR)
    # Sibling in rtx_modules by default.
    set(RTXDIASM_PTX_TO_SPIRV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rtxdiasm-ptx-to-spirv")
  endif()
  if(EXISTS "${RTXDIASM_PTX_TO_SPIRV_DIR}/CMakeLists.txt")
    add_subdirectory("${RTXDIASM_PTX_TO_SPIRV_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/rtxdiasm-ptx-to-spirv" EXCLUDE_FROM_ALL)
  else()
    message(FATAL_ERROR "Missing compiler module at ${RTXDIASM_PTX_TO_SPIRV_DIR}. "
      "Provide rtxdiasm-ptx-to-spirv or define RTXDIASM_PTX_TO_SPIRV_DIR / acuda_compiler target.")
  endif()
endif()
if(NOT TARGET acuda_compiler)
  if(TARGET rtxdiasm_ptx_to_spirv_compiler_static)
    add_library(acuda_compiler ALIAS rtxdiasm_ptx_to_spirv_compiler_static)
  else()
    message(FATAL_ERROR "acuda_compiler target not available (expected rtxdiasm_ptx_to_spirv_compiler_static).")
  endif()
endif()
target_link_libraries(acuda_runtime PUBLIC acuda_compiler)

# --- Shared fatbin PTX parse/extract (lz4+zstd) ---
if(NOT TARGET rtxdiasm_fatbin_parse)
  if(NOT DEFINED RTXDIASM_FATBIN_PARSE_DIR)
    set(RTXDIASM_FATBIN_PARSE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rtxdiasm-fatbin-parse")
  endif()
  if(EXISTS "${RTXDIASM_FATBIN_PARSE_DIR}/CMakeLists.txt")
    add_subdirectory("${RTXDIASM_FATBIN_PARSE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/rtxdiasm-fatbin-parse" EXCLUDE_FROM_ALL)
  else()
    message(FATAL_ERROR "Missing fatbin parse module at ${RTXDIASM_FATBIN_PARSE_DIR}.")
  endif()
endif()

# --- CUDA Driver API (libcuda.so.1 / nvcuda.dll) ---
add_library(acuda_driver SHARED
  src/driver/driver_compat_stubs.c
  src/driver/driver_stub_funcs.c
  src/driver/driver_export.c
  src/driver/driver_export_dd9e98a4.c
  src/driver/driver_export_dd9e98a4_stubs.c
  src/driver/driver_export_6bd5fb6c.c
  src/driver/driver_export_6bd5fb6c_stubs.c
  src/driver/driver_export_a094798c.c
  src/driver/driver_export_a094798c_stubs.c
  src/driver/driver_export_42d85a81.c
  src/driver/driver_export_42d85a81_stubs.c
  src/driver/driver_export_c693336e.c
  src/driver/driver_export_c693336e_stubs.c
  src/driver/driver_export_0ca50b8c.c
  src/driver/driver_export_0ca50b8c_stubs.c
  src/driver/driver_export_263e8860.c
  src/driver/driver_export_263e8860_stubs.c
  src/driver/driver_export_d4082055.c
  src/driver/driver_export_d4082055_stubs.c
  src/driver/driver_export_21318c60.c
  src/driver/driver_export_21318c60_stubs.c
  src/driver/driver_export_195bcbf4.c
  src/driver/driver_export_195bcbf4_stubs.c
  src/driver/driver_error.c
  src/driver/driver_texture.c
  src/driver/driver_graphics.c
  src/driver/driver_procaddr.c
  src/driver/driver_event.c
  src/driver/driver_link.c
  src/driver/driver_stream.c
  src/driver/driver_memory.c
  src/driver/driver_context.c
  src/driver/driver_memset.c
  src/driver/driver_memcpy.c
  src/driver/driver_vmm.c
  src/driver/driver_host.c
  src/driver/driver_library.c
  src/driver/driver_launch.c
  src/driver/driver_device.c
  src/driver/driver_module.c
  src/driver/arena.c
  ${ACUDA_TRACE_SOURCE_INTERNAL}
  src/driver/rtxdisam_bridge.c
  src/driver/vulkan32_alternatenames.c
)

set_target_properties(acuda_driver PROPERTIES
  OUTPUT_NAME cuda
  VERSION 1.0.0
  SOVERSION 1
)
if(WIN32)
  set_target_properties(acuda_driver PROPERTIES OUTPUT_NAME nvcuda)
endif()

target_include_directories(acuda_driver
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
)
target_link_libraries(acuda_driver PRIVATE acuda_runtime Vulkan::Vulkan ${CMAKE_DL_LIBS} rtxdiasm_fatbin_parse)

if(NOT WIN32)
  target_link_options(acuda_driver PRIVATE
    "LINKER:-Bsymbolic"
    "LINKER:-z,nodelete"
  )
  target_compile_options(acuda_driver PRIVATE
    -finstrument-functions
    -finstrument-functions-exclude-file-list=trace.c
  )
endif()

if(WIN32)
  set_target_properties(acuda_driver PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
  )
endif()

target_compile_options(acuda_driver PRIVATE ${ACUDA_WARNINGS})

# Compatibility target name for consumers expecting 'cuda' target name in some setups.
if(NOT TARGET cuda)
  add_library(cuda ALIAS acuda_driver)
endif()

# --- 64-bit PTX->SPIR-V compile helper for the 32-bit driver ---
# The 32-bit driver cannot parse large PTX modules without exhausting its own
# address space, so it spawns this executable. It must live next to the driver
# because the driver names it, and it must keep accepting the same
# --cap.<name>=<value> arguments the driver passes.
add_executable(acuda_compile_host src/driver/compile_host.c)
target_link_libraries(acuda_compile_host PRIVATE acuda_compiler)
target_compile_options(acuda_compile_host PRIVATE ${ACUDA_WARNINGS})

# Expose resolved trace source path for sibling modules (e.g. cudart) if needed.
if(ACUDA_TRACE_SOURCE_INTERNAL)
  get_filename_component(_acuda_trace_abs "${CMAKE_CURRENT_SOURCE_DIR}/${ACUDA_TRACE_SOURCE_INTERNAL}" ABSOLUTE)
  set(ACUDA_DRIVER_TRACE_SOURCE "${_acuda_trace_abs}" CACHE FILEPATH "ACuda driver trace.c source" FORCE)
else()
  set(ACUDA_DRIVER_TRACE_SOURCE "" CACHE FILEPATH "ACuda driver trace.c source" FORCE)
endif()
