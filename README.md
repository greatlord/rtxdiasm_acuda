# rtxdiasm_acuda

**A Vulkan-based CUDA Driver API compatibility layer.**

`rtxdiasm_acuda` is a standalone module providing a CUDA-compatible Driver API implementation using Vulkan as its GPU execution backend.

It forms the driver-level foundation of the ACuda project, translating CUDA driver operations into Vulkan-backed execution. CUDA PTX instructions are translated into SPIR-V and executed through Vulkan compute pipelines, allowing compatible CUDA workloads to run on Vulkan-capable GPUs from different vendors.

The project is designed to operate without requiring NVIDIA hardware or the NVIDIA CUDA SDK.

## Features

- CUDA Driver API compatibility through platform-native driver libraries.
- Vulkan-based GPU execution backend.
- Cross-vendor GPU support through Vulkan.
- PTX-to-SPIR-V compilation integration.
- CUBIN/FATBIN handling and SASS-to-PTX lifting integration.
- Device selection and PCI identification.
- Configurable GPU memory management and execution behavior.
- Support for asynchronous streams and command submission.
- Optional hardware-accelerated instruction paths.
- Configurable compilation and pipeline caching.
- Runtime tracing, profiling, validation and debugging.
- Modular CMake integration with other ACuda components.

## Build Outputs

The project produces platform-native CUDA Driver API compatibility libraries.

| Platform | Output | SONAME |
|---|---|---|
| Linux | `libcuda.so.1` | `libcuda.so.1` |
| Windows | `nvcuda.dll` | — |

These libraries expose the CUDA Driver API interface expected by applications and higher-level CUDA compatibility layers.

## Architecture

The module consists of two primary components.

### `acuda_driver`

Shared library implementing the CUDA Driver API compatibility layer.

Responsibilities include:

- Exposing the platform-specific `libcuda` / `nvcuda.dll` interface.
- Managing CUDA-compatible device and context operations.
- Handling driver-level resource and execution requests.
- Connecting driver operations to the internal Vulkan runtime.
- Coordinating PTX and binary input processing.

### `acuda_runtime`

Static internal runtime providing the Vulkan execution infrastructure used by the driver and related tests.

Responsibilities include:

- Vulkan instance and physical-device initialization.
- Logical-device and queue management.
- GPU memory and resource management.
- Command buffer and synchronization management.
- Compute pipeline creation and execution.
- Runtime support for translated GPU workloads.

The runtime is intended as an internal foundation shared by the driver and other ACuda components.

## Dependencies

### Required build dependencies

- **Vulkan SDK and loader** — Vulkan headers, loader and development libraries.
- **CMake** — Build-system integration.
- **`rtxdiasm-ptx-to-spirv`** — PTX-to-SPIR-V compiler module.

Vulkan is discovered through:

```cmake
find_package(Vulkan REQUIRED)
```

The project links against:

```cmake
Vulkan::Vulkan
```

The build requires Vulkan 1.3 header definitions, including `VK_API_VERSION_1_3`.

### PTX-to-SPIR-V compiler

The module depends on:

`rtx_modules/rtxdiasm-ptx-to-spirv`

This compiler translates CUDA PTX instructions into SPIR-V, which is subsequently compiled into Vulkan compute pipelines by the target GPU driver.

The compiler is expected to exist as a sibling module by default.

## Vulkan Requirements

ACuda uses Vulkan as its GPU execution backend. The target system must provide a Vulkan 1.3-capable GPU and driver.

### Build-time requirements

- Vulkan SDK or equivalent Vulkan development environment.
- Vulkan loader and headers discoverable by CMake.
- Vulkan 1.3 API definitions.
- `Vulkan::Vulkan` link target.
- `rtxdiasm-ptx-to-spirv` compiler module.

### Runtime requirements

The physical device must report Vulkan API version 1.3 or newer and provide the required device capabilities.

**Minimum device requirements:**

| Requirement | Vulkan capability | Purpose |
|---|---|---|
| Vulkan API | 1.3 or newer | Required API baseline |
| Compute queue | `VK_QUEUE_COMPUTE_BIT` | Kernel execution |
| Maintenance4 | `maintenance4` | Required execution support, including `LocalSizeId` |
| Synchronization2 | `synchronization2` | Command and execution synchronization |
| Timeline semaphores | `timelineSemaphore` | GPU execution synchronization |
| 64-bit integers | `shaderInt64` | 64-bit integer shader operations |
| Host memory | Host-visible and host-coherent memory | CPU/GPU data transfers |

Device-local memory is preferred when available for GPU resource allocation and performance.

### Queue selection

ACuda selects a Vulkan queue family supporting compute operations.

When possible, it prefers a queue family that supports both compute and graphics operations.

The selection can be overridden using:

```bash
ACUDA_QUEUE_FAMILY=<index>
```

The selected queue family must support compute operations.

### Optional GPU capabilities

The following capabilities are detected and used opportunistically. They are not required for basic device initialization, but enable additional instruction paths and hardware acceleration.

| Feature / Extension | Purpose |
|---|---|
| `shaderFloat64` | Native FP64 operations |
| `shaderIntegerDotProduct` | Native integer dot-product operations, including accelerated DP4A |
| `VK_KHR_shader_integer_dot_product` | Integer dot-product support |
| `shaderFloat16` | Native FP16 operations |
| `VK_KHR_shader_float16_int8` | Additional FP16 and INT8 shader capabilities |
| `VK_KHR_shader_fma` | Fused multiply-add support |
| BFloat16 capabilities and extensions | BF16 instruction paths |
| Mixed-float dot-product capabilities | Specialized mixed-precision arithmetic |

If an optional capability is unavailable, ACuda can use alternative instruction implementations where supported.

### Validation and debugging

Optional Vulkan debugging facilities include:

- `VK_EXT_debug_utils` — Vulkan debug messages and diagnostics.
- `VK_LAYER_KHRONOS_validation` — Vulkan API validation.

Enable validation using:

```bash
ACUDA_VALIDATION=1
```

Validation is disabled by default and can be explicitly disabled using:

```bash
ACUDA_NO_VALIDATION=1
```

### Vulkan compatibility note

**Vulkan API version and Vulkan feature support are two separate requirements.**

A GPU may report Vulkan 1.3 compatibility while still lacking one or more features required by ACuda.

During device initialization, ACuda must verify that the physical device supports the required features and that they can be enabled by the Vulkan implementation.

A Vulkan 1.3 driver alone is therefore not sufficient to guarantee compatibility.

Optional capabilities are detected separately and used when available. This allows ACuda to select optimized execution paths or alternative implementations depending on the GPU and driver.

## Execution Architecture

ACuda uses the following execution pipeline:

```text
        CUDA Application
               |
               v
        CUDA Driver API
               |
               v
         acuda_driver
               |
               v
        acuda_runtime
               |
               v
       CUDA Binary / PTX
               |
               v
      PTX-to-SPIR-V Compiler
               |
               v
             SPIR-V
               |
               v
      Vulkan Compute Pipeline
               |
               v
       Vulkan Command Buffers
               |
               v
          GPU Execution
```

For precompiled CUDA binaries containing SASS, the external rtxdisam lifting bridge can be used to recover PTX before translation.

This provides a path for handling applications that do not include their original PTX source.

## CUBIN/FATBIN Backend

ACuda supports multiple approaches for processing CUDA binary inputs.

The backend can select between its internal C parser and the external rtxdisam SASS-to-PTX lifting implementation.

| Backend | Behavior |
|---|---|
| `0` / `auto` | Automatic selection: internal C parser for PTX inputs, otherwise rtxdisam lifting |
| `1` | Internal C parser only |
| `2` | rtxdisam / C# lifting backend only |

The external lifting bridge supports:

- `rtxdisam_native` — Native shared library.
- `rtxdisam_host` — External helper executable.

Both paths can be configured through environment variables.

This configuration is used by ACuda and the `rtxdiasm_guard` CUDA Runtime API shim.

## Integration

`rtxdiasm_acuda` is designed to integrate directly into the ACuda CMake build system while remaining independently buildable as a module.

Add the module to an existing CMake project:

```cmake
add_subdirectory(rtxdiasm_acuda)
```

This exposes the following targets:

- `acuda_driver`
- `acuda_runtime`

Other project components, including `rtxdiasm-cudart`, can link against these targets as required.

## Environment Variables

ACuda exposes environment variables for configuring device selection, runtime behavior, memory allocation, compilation, caching, execution, tracing and debugging.

These variables allow developers to modify ACuda behavior without changing application code or rebuilding the project.

### 1. Device Selection and Runtime Configuration

| Variable | Description |
|---|---|
| `ACUDA_VALIDATION` | Enable Vulkan validation and debug-utils messenger. Opt-in; disabled by default. |
| `ACUDA_NO_VALIDATION` | Force Vulkan validation off. Overrides the validation enable setting. |
| `ACUDA_DEVICE` | Select Vulkan physical device using an index or device-name hint. |
| `ACUDA_GPU` | Alternative GPU selection hint. |
| `ACUDA_QUEUE_FAMILY` | Override the compute queue family index. Must support compute operations. |
| `ACUDA_LIST_DEVICES` | Enumerate available Vulkan devices for diagnostics. |
| `ACUDA_PCI_DOMAIN` | PCI domain selection or identification hint. |
| `ACUDA_PCI_BUS` | PCI bus selection or identification hint. |
| `ACUDA_PCI_DEVICE` | PCI device selection or identification hint. |
| `ACUDA_PCI_BUS_ID` | PCI bus ID selection or identification hint. |
| `ACUDA_CC_MAJOR` | Override reported or simulated compute capability major version. |
| `ACUDA_CC_MINOR` | Override reported or simulated compute capability minor version. |
| `ACUDA_DEVICE_MEM_MB` | Configure device memory sizing in MB. |
| `ACUDA_HOST_HEAP_MB` | Configure host heap sizing in MB. |
| `ACUDA_BIG_ARENA_LOW` | Control low-memory placement or sizing of the large memory arena. |
| `ACUDA_UNIFIED_ADDRESSING` | Control unified addressing behavior. |
| `ACUDA_SUBGROUP_SIZE` | Override or tune subgroup size. |
| `ACUDA_MAINTENANCE4` | Control the Vulkan maintenance4-related execution path. |
| `ACUDA_NO_MAINTENANCE4` | Disable the maintenance4-related execution path. |
| `ACUDA_KEEP_BUSY` | Control keep-GPU-busy behavior. |
| `ACUDA_KEEP_BUSY_PTX` | Control PTX-based background workload behavior. |
| `ACUDA_KEEP_BUSY_THREADS` | Configure background workload thread behavior. |
| `ACUDA_WARMUP_MS` | Configure GPU/runtime warmup timing in milliseconds. |

### 2. Tracing, Debugging and Diagnostic Dumps

| Variable | Description |
|---|---|
| `ACUDA_TRACE` | Enable runtime and driver tracing/logging. |
| `ACUDA_TRACE_FILE` | Configure trace output destination. |
| `ACUDA_TRACE_PROC` | Control process-specific trace output. |
| `ACUDA_TRACE_PARAMS` | Control parameter-level tracing details. |
| `ACUDA_PROFILE_LAUNCH` | Enable kernel launch and stream profiling output. |
| `ACUDA_REPORT_BID` | Report block IDs for debugging. |
| `ACUDA_DUMP_CUBIN_ON_ERROR` | Dump cubin data when lifting or processing fails. |
| `ACUDA_DUMP_FATBIN` | Enable fatbin input dumping. |
| `ACUDA_FATBIN_DUMP_DIR` | Set the fatbin dump directory. |
| `ACUDA_DUMP_PTX_DIR` | Set the generated PTX dump directory. |
| `ACUDA_DUMP_ISA` | Enable ISA dumping for debugging. |
| `ACUDA_DUMP_FORCE_INPUTS` | Force dumping of compiler inputs. |

**Feature-specific debugging switches:**

| Variable | Description |
|---|---|
| `ACUDA_DEBUG_ATOMIC` | Atomic operation debugging. |
| `ACUDA_DEBUG_BDA` | Buffer device address debugging. |
| `ACUDA_DEBUG_BDA_PUSH` | BDA push-constant debugging. |
| `ACUDA_DEBUG_COOPMAT` | Cooperative matrix debugging. |
| `ACUDA_DEBUG_DOT` | Dot-product debugging. |
| `ACUDA_DEBUG_F16` | FP16 debugging. |
| `ACUDA_DEBUG_FDOT2` | FDOT2 debugging. |
| `ACUDA_DEBUG_FLOAT` | Floating-point debugging. |
| `ACUDA_DEBUG_HIGH_PERF` | High-performance path debugging. |
| `ACUDA_DEBUG_STREAMS` | Stream execution debugging. |
| `ACUDA_DEBUG_SUBGROUP` | Subgroup execution debugging. |
| `FRAMEMT_DUMP` | Additional dump/debug hook. |

### 3. Compilation and PTX-to-SPIR-V Configuration

| Variable | Description |
|---|---|
| `ACUDA_SASS_TARGET` | Specify target SM architecture hint for SASS lifting and compilation. |
| `ACUDA_EMU_F64` | Enable or force FP64 emulation. |
| `ACUDA_F16_EMULATED` | Control emulated FP16 execution. |
| `ACUDA_F16_PROBE_OFF` | Disable FP16 capability probing. |
| `ACUDA_FDOT2` | Control the FDOT2/dot-product execution path. |
| `ACUDA_NO_FDOT2` | Disable the FDOT2 execution path. |
| `ACUDA_COOPMAT` | Enable cooperative matrix functionality. |
| `ACUDA_NO_COOPMAT` | Disable cooperative matrix functionality. |
| `ACUDA_HIGH_PERF` | Enable the high-performance execution path. |
| `ACUDA_FMA` | FMA selection: `0` = auto (probe when Vulkan shader-FMA capability exists, use hardware only if probes pass, otherwise portable/software), `1` = prefer hardware when capability exists (skip probe-fail fallback; fall back to software if capability absent), `2` = always software/portable. Default: `0`. |
| `ACUDA_NO_SURF_ALIAS` | Disable surface aliasing. |
| `ACUDA_NO_HOST_HEAP` | Disable the host heap path. |
| `ACUDA_PTX_NO_ATOMIC_F32_ADD` | Disable PTX floating-point 32-bit atomic addition support. |
| `ACUDA_PTX_NO_ATOMIC_FLOAT` | Disable floating-point atomic operations. |
| `ACUDA_PTX_NO_BDA` | Disable the buffer device address path. |
| `ACUDA_PTX_NO_DOT_ACCEL` | Disable dot-product acceleration. |
| `ACUDA_PTX_NO_DOT_PRODUCT` | Disable dot-product instruction support. |
| `ACUDA_PTX_NO_DYNAMIC_RENDERING` | Disable the dynamic-rendering-dependent path. |
| `ACUDA_PTX_NO_FLOAT_CONTROLS` | Disable floating-point control features. |
| `ACUDA_SPIRV_FMA_SHADER` | Control FMA-related shader/SPIR-V behavior. |
| `ACUDA_PTX_NO_FULL_SUBGROUPS` | Disable assumptions about full subgroup execution. |
| `ACUDA_PTX_NO_INT64_IMAGE` | Disable the 64-bit integer image path. |
| `ACUDA_PTX_NO_SSBO8` | Disable 8-bit storage-buffer functionality. |
| `ACUDA_PTX_NO_SSBO16` | Disable 16-bit storage-buffer functionality. |
| `ACUDA_PTX_NO_SUBGROUP32` | Disable the subgroup-size-32 execution path. |
| `ACUDA_PTX_NO_VMM` | Disable virtual memory management functionality. |
| `ACUDA_PTX_NO_ZERO_INIT_WG` | Disable zero-initialized workgroup behavior. |

### 4. Compilation Cache and Pipeline Management

| Variable | Description |
|---|---|
| `ACUDA_CACHE_DIR` | Specify the cache directory for compiled artifacts. |
| `ACUDA_NO_CACHE` | Disable ACuda caching. |
| `ACUDA_NO_CMD_CACHE` | Disable the command/pipeline cache path. |
| `ACUDA_SPIRV_CACHE` | Control SPIR-V caching. |
| `ACUDA_SPIRV_CACHE_DIR` | Specify the SPIR-V cache directory. |
| `ACUDA_COMPILE_HOST` | Control host compilation behavior. |

### 5. Streams, Kernel Launch and Execution

| Variable | Description |
|---|---|
| `ACUDA_FORCE_SERIAL` | Force serialized execution. |
| `ACUDA_BATCH` | Enable batching. |
| `ACUDA_NO_BATCH` | Disable batching. |
| `ACUDA_ASYNC_STREAMS` | Control asynchronous stream behavior. |
| `ACUDA_STREAM_QUEUES` | Configure stream-to-queue mapping or queue count. |
| `ACUDA_HOST_CALL_LIMIT` | Limit host-call callbacks. |
| `ACUDA_SKIP_KERNEL` | Skip kernel execution for debugging and testing. |
| `ACUDA_PPU_ATTR` | Control PPU/launch attributes. |

### 6. CUBIN/FATBIN Backend and SASS-to-PTX Bridge

| Variable | Description |
|---|---|
| `ACUDA_CUBIN_BACKEND` | Select the cubin/fatbin processing backend. |
| `RTXDISAM_LIBRARY` | Specify the path to the `rtxdisam_native` shared library. |
| `RTXDISAM_HOST` | Specify the path to the `rtxdisam_host` helper executable. |

`ACUDA_CUBIN_BACKEND` supports three modes:

| Value | Behavior |
|---|---|
| `0` / `auto` | Automatic selection: internal C parser for PTX inputs; otherwise rtxdisam lifting. |
| `1` | Internal C parser only. |
| `2` | rtxdisam / C# lifting backend only. |

The external lifting bridge provides an alternative for handling precompiled CUDA binaries containing SASS.

### 7. System and Environment Integration

| Variable | Description |
|---|---|
| `HOME` | Home directory lookup on POSIX-compatible systems. Used for cache and path resolution. |
| `XDG_CACHE_HOME` | Base directory for user-specific cache storage. |
| `LOCALAPPDATA` | Windows local application data directory used for cache and path resolution. |

### Configuration Notes

- Environment variables are read by the relevant ACuda components. Not every variable applies to every execution path.
- Variables prefixed with `NO_` or `PTX_NO_` generally disable specific capabilities or implementation paths.
- Disabling an optimized path may cause ACuda to use an alternative implementation, provided a compatible fallback exists.
- Device selection hints do not override Vulkan hardware capabilities or feature availability.
- Debugging and dump options may generate additional files and affect performance.
- Cache configuration can affect compilation startup time and storage usage.
- When multiple variables control the same feature, their precedence depends on the implementation.

This reference documents configuration variables exposed by the current implementation. It does not imply that every variable is supported by every ACuda module or that all variables are enabled by default.

## Design Goals

The primary goals of `rtxdiasm_acuda` are:

1. **CUDA compatibility** — Provide a CUDA Driver API-compatible interface without requiring NVIDIA hardware.
2. **Cross-vendor execution** — Use Vulkan as the underlying GPU execution backend.
3. **Modularity** — Keep the driver and runtime infrastructure independently maintainable.
4. **Extensibility** — Allow additional CUDA features and instruction paths to be implemented over time.
5. **Hardware acceleration** — Take advantage of optional Vulkan capabilities when available.
6. **Fallback support** — Provide alternative implementations for selected instructions when native GPU features are unavailable.
7. **Integration** — Provide a reusable foundation for other ACuda components and CUDA compatibility layers.

## Copyright and Authorship

Copyright © 2026 Magnus Olsen.

Development was assisted by AI models, including DeepFlash 4.1, Muse Spark 1.3, and Iceman.

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for the complete license text.
