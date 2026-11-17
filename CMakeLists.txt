cmake_minimum_required(VERSION 3.21)
project(rtxdiasm_acuda LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# rtxdiasm_acuda: ACuda CUDA Driver API (libcuda/nvcuda) + internal runtime foundation.
# Outputs (as requested):
#   Linux:   libcuda.so.1 (OUTPUT_NAME cuda, SOVERSION 1)
#   Windows: nvcuda.dll

option(ACUDA_BUILD_TESTS "Build ACuda tests/examples (normally built by integrating project)" OFF)

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

# --- Runtime library (internal foundation used by driver/consumers) ---
add_library(acuda_runtime STATIC
  src/runtime/error.c
  src/runtime/context.c
  src/runtime/stream.c
  src/runtime/memory.c
  src/runtime/texture_image.c
  src/runtime/launch.c
  src/runtime/hostcall.c
  src/runtime/barrier.c
)
target_include_directories(acuda_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
)
target_link_libraries(acuda_runtime PRIVATE Vulkan::Vulkan)
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
  src/driver/driver.c
  src/driver/driver_compat_stubs.c
  src/driver/driver_stub_funcs.c
  src/driver/driver_export.c
  src/driver/driver_error.c
  src/driver/driver_texture.c
  src/driver/driver_procaddr.c
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
