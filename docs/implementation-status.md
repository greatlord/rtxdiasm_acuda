# ACuda Implementation Status

This document summarizes the current implementation status of major ACuda components.

**Status legend**
- `ok` — implemented and usable in the current supported scope
- `partially` — implemented, but with limitations, incomplete coverage, or known gaps
- `todo` — not implemented yet / planned

| Area | Component | Status | Notes |
|---|---|---|---|
| Driver API | Core init/version/device/context APIs | ok | Implemented subset including `cuInit`, version/device queries, context create/push/pop/current/synchronize, and primary context APIs as documented in `driver-api.md`. |
| Driver API | Memory APIs (`cuMemAlloc/Free`, sync memcpy/memset, mem info) | ok | Core allocation/free, `HtoD/DtoH/DtoD`, `MemsetD32/D8`, and live memory tracking implemented for supported paths. |
| Driver API | Module/link/launch APIs (PTX-focused subset) | partially | `cuModuleLoadData[Ex]`, `cuModuleGetFunction`, `cuLaunchKernel`, linker PTX-passthrough (`AddData/Complete`), `cuLaunchKernelEx` basic forwarding implemented; launch/linking have documented constraints (e.g. 1D-oriented launch constraints, PTX-centric flow, limited extended attributes). |
| Driver API | Streams/events API surface | partially | Stream/event create/destroy/query/record/synchronize/wait paths implemented for supported subset; async behavior and full CUDA parity are not complete. |
| Driver API | Texture/surface array/object APIs | partially | Arrays/mipmapped arrays, memcpy2D/3D, texture objects, and surface objects implemented for supported formats/geometries/use-cases; advanced/unsupported cases return `NOT_SUPPORTED` or are limited. |
| Driver API | Torch / broader CUDA client compatibility surface | partially | Targeted compatibility work exists (e.g. selected attributes, fatbinary handling paths, `cuLaunchKernelEx` basic cases), but missing symbols/behaviors remain for wider client compatibility. |
| Driver API | Full CUDA Driver API coverage | todo | Many CUDA driver APIs and semantics outside the implemented/documented subset are not yet supported. |
| Runtime | Vulkan instance/device/queue initialization | ok | Vulkan 1.3-based runtime initialization, device/queue selection, and core context bring-up implemented (subject to required feature availability). |
| Runtime | Command submission and synchronization primitives | ok | Internal stream/event/timeline-semaphore based execution and synchronization paths implemented for supported workloads. |
| Runtime | GPU memory/resource management | partially | Core buffer/image/resource handling implemented; behavior differs from CUDA in areas (e.g. opaque handle/addressing model, host-visible transfer paths) and is scoped to supported use-cases. |
| Runtime | Compute pipeline creation and kernel execution | partially | PTX-to-SPIR-V to Vulkan compute pipeline execution path implemented for supported kernels/features; not full CUDA/PTX coverage. |
| Runtime | Optional capability detection and accelerated paths | partially | Optional Vulkan features/extensions are detected and used opportunistically when available, with fallbacks in some cases. |
| Memory | Linear device allocations and transfers | ok | Implemented for supported driver/runtime paths (`alloc/free`, sync host/device copies, memset operations). |
| Memory | Texture/image/surface backing resources | partially | Vulkan image-backed textures/surfaces and related upload/download/copy paths implemented for supported formats/layouts/geometries. |
| Memory | CUDA memory semantics parity (advanced/pools/unified/etc.) | partially/todo | Some pool-related driver entry points exist, but broader CUDA memory model/parity (and advanced features) is incomplete. |
| Execution | Streams | ok | Stream creation, batching/flush/submit/sync, and tracking paths implemented in runtime (refactored into modular TUs). |
| Execution | Events and timing/query | partially | Event record/query/synchronize/elapsed and stream-wait-event paths implemented for supported usage; full CUDA behavior/timing parity not guaranteed. |
| Execution | Multi-context / concurrency isolation | partially/todo | Current design shares Vulkan device/state with limited isolation semantics; full CUDA context isolation/concurrency behavior is not implemented. |
| Compilation | PTX module load and PTX-to-SPIR-V execution integration | ok | PTX-based module loading/launch path integrated with `rtxdiasm-ptx-to-spirv` for supported kernels. |
| Compilation | Fatbin/cubin handling and SASS-to-PTX bridge integration | partially | Fatbin parsing and optional SASS-to-PTX lifting paths exist (including backend selection behavior), with limitations (e.g. unsupported/compressed cases, lazy/multi-entry constraints). |
| Compilation | Full PTX/SASS instruction and feature coverage | todo | Coverage is workload-driven and incomplete relative to full CUDA/PTX/SASS surface. |
| Debug/Validation | Validation/debug env controls and tracing hooks | partially | Runtime/driver support includes validation toggles and tracing/debug-related options in parts of the stack; coverage and tooling depth vary. |
| Debug/Validation | Profiling and diagnostics | partially | Some profiling/tracing hooks and diagnostics exist, but not a complete CUDA-compatible profiling/debug toolchain. |
| Build/Packaging | Native driver/runtime library build (CMake) | ok | Builds `acuda_driver` and `acuda_runtime` with Vulkan and PTX compiler integration (per module README/build setup). |
| Platform/Test | Windows PTX kit validation (Win32/x86, Vulkan 1.4 SDK env used in this workspace) | ok | Canonical `rtxdiasm_tests/tests/acuda/windows-kit` currently verified at **61/61 passing** in this workspace flow. |
| Platform/Test | Cross-platform / conformant-GPU runtime validation | partially | WSL/software-Vulkan (e.g. llvmpipe) is not conformant for runtime correctness; broader platform/GPU validation depends on available conformant Vulkan environments. |

## Notes
- This table is a high-level snapshot and should be updated as APIs/features evolve.
- Detailed behavior, supported subsets, and known divergences are documented in:
  - `README.md`
  - `docs/driver-api.md`
  - `docs/streams.md`
  - `docs/device-module-linking.md`
  - `docs/host-calls.md`
- Some items are marked `partially` where functionality exists but full CUDA compatibility, performance parity, or edge-case coverage is not yet complete.
