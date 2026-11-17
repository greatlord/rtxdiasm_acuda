# nvcuda.dll / libcuda.so Implementation Status

This document tracks implementation status for the CUDA Driver API compatibility library (`nvcuda.dll` on Windows, `libcuda.so.1` on Linux).

**Status legend**
- `ok` — implemented for the currently supported/documented subset
- `partially` — implemented with limitations, partial coverage, or known gaps
- `todo` — not implemented

| Area | API / Component | Status | Notes |
|---|---|---|---|
| Initialization / Version | `cuInit`, `cuDriverGetVersion` | ok | Driver version reported for CUDA compatibility (see `driver-api.md`). |
| Device | `cuDeviceGetCount/Get/GetName`, compute capability, total memory, UUID, PCI bus ID/query, common attributes | ok | Implemented subset for device discovery/characterization. |
| Device | `cuDeviceGetAttribute` (full attribute coverage) | partially | Common characterization attributes covered; not all CUDA attributes are implemented. |
| Device | P2P/mempool/query helpers (`GetP2PAttribute`, `CanAccessPeer`, default/mempool get-set, trim, RDMA flush no-op, etc.) | partially | Present for compatibility in subset form; behavior may be stubbed/limited compared to CUDA. |
| Context | Create/destroy/push/pop/set/get-current/device/api-version/synchronize, limits/cache-config, detach | ok | Implemented as lightweight current markers over a shared Vulkan device (limited isolation). |
| Context | Primary context APIs (`retain/release/reset/setflags/getstate`) | ok | Implemented for supported usage. |
| Memory | `cuMemAlloc`, `cuMemFree`, `cuMemGetInfo` | ok | Linear allocation/free and live memory tracking for supported paths. |
| Memory | Sync transfers/memset (`cuMemcpyHtoD/DtoH/DtoD`, `cuMemsetD32/D8`) | ok | Synchronous variants implemented; complete before return. |
| Memory | Async memory operations and broader memory API coverage | partially/todo | Some async-related APIs exist in adjacent areas (e.g. memcpy2D/3D async variants), but general async memory API coverage/parity is incomplete. |
| Module | `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleUnload`, `cuModuleGetFunction` | partially | PTX-focused path implemented; binary/cubin/fatbin handling depends on parser/lifter paths and documented constraints. |
| Module | `cuModuleLoadFatBinary` | partially | Supports fatbin parsing/PTX extraction and optional SASS-to-PTX lifting paths (backend-dependent); unsupported/missing cases return errors (e.g. `INVALID_IMAGE`). |
| Module | Globals/other module APIs (`cuModuleGetGlobal_v2`, libraries, etc.) | todo | Noted as missing/not yet on taken paths in current driver docs. |
| Linker | `cuLinkCreate`, `cuLinkAddData`, `cuLinkAddFile[_v2]`, `cuLinkComplete`, `cuLinkDestroy` | partially | PTX-passthrough linker implemented for supported inputs; cubin/fatbin/object inputs rejected where unsupported. |
| Launch / Function | `cuLaunchKernel` | partially | Implemented with documented constraints (e.g. 1D-oriented limits/assumptions, `sharedMem=0`, `extra=NULL`, specific arg marshalling expectations). |
| Launch / Function | `cuLaunchKernelEx` | partially | Plain configs forwarded to `cuLaunchKernel`; extended attributes rejected (`INVALID_VALUE`). |
| Launch / Function | `cuFuncGetAttribute`, `cuFuncSetAttribute` | partially | Implemented with driver/runtime-reported values and advisory/limited behavior where CUDA features are not backed by equivalent runtime capabilities. |
| Occupancy / Execution control | `cuOccupancyMaxActiveBlocksPerMultiprocessor` and related APIs | todo | Driver occupancy APIs noted as missing in current docs. |
| Streams | `cuStreamCreate/Destroy/Query/WaitEvent/GetFlags/GetPriority/Synchronize` (subset) | partially | Supported stream subset implemented; full CUDA stream semantics/parity not complete. |
| Events | `cuEventCreate/Destroy/Record/Query/Synchronize` (subset) | partially | Supported event subset implemented; elapsed timing/query behavior depends on runtime support and is subset-scoped. |
| Arrays / Texture objects | `cuArray*`, `cuMipmappedArray*`, `cuTexObject*` (supported formats/geometries) | partially | Implemented for documented formats/geometries and use-cases; unsupported cases return `NOT_SUPPORTED` or are limited. |
| Memcpy 2D/3D (array/device/host) | `cuMemcpy2D[Async]`, `cuMemcpy3D[Async]` | partially | Supported paths implemented (including rectangle handling/staging as documented); not full CUDA coverage. |
| Surface objects | `cuSurfObjectCreate/Destroy/GetResourceDesc` (supported cases) | partially | Supported array-backed surface cases implemented; unsupported resource types/arrays (e.g. depth/cube/multisampled in unsupported paths) rejected. |
| Dispatch / Errors | `cuGetProcAddress[_v2]`, `cuGetErrorString`, `cuGetErrorName` | ok | Dispatch over implemented symbols and error string/name reporting implemented. |
| Handle lifetime / ABI behavior | Driver handle ownership and compatibility behaviors | partially | Key fixes/behaviors implemented (e.g. function handle not consumed on launch), but full CUDA ABI/behavior parity is not guaranteed. |
| Client compatibility (e.g. cudart/torch-facing symbols) | Additional driver symbols required by specific clients | partially | Targeted symbols/paths added for client compatibility, but some required APIs remain missing depending on workload/client path. |

## References
- `docs/driver-api.md`
- `docs/streams.md`
- `README.md`

## Scope note
This table reflects the current documented/implemented subset of the driver library (`nvcuda.dll` / `libcuda.so`), not full CUDA Driver API compatibility.
