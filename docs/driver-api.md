# CUDA Driver API compatibility (`libcuda.so`, Phase 5a)

`bin/release/libcuda.so.1.0.0` (`SOVERSION 1`) implements the driver subset a
JIT client needs. Error codes match the CUDA 13 driver values for the
implemented set (0, 1, 2, 3, 100, 101, 200, 201, 400, 500, 600, 801, 999).

## Covered

init/version (`cuInit`, `cuDriverGetVersion` → 13040 = CUDA 13.4 compat);
devices (`GetCount/Get/GetName/ComputeCapability` → 8.0, `TotalMem` →
8 GiB convention, `GetUuid`, `GetPCIBusId/ByPCIBusId`,
`GetTexture1DLinearMaxWidth`, `GetP2PAttribute`, `CanAccessPeer`,
`GetAttribute` for the common characterization set,
`GetDefaultMemPool/GetMemPool/SetMemPool`, `MemPoolTrimTo`,
`FlushGPUDirectRDMAWrites` no-op);
contexts (create/destroy/push/pop/set/get-current/device/apiversion/
limits/cache-config/detach/synchronize, plus primary-ctx
retain/release/reset/setflags/getstate; one shared Vulkan device,
contexts are lightweight current-markers on a stack of 16);
memory (`Alloc/Free`, `GetInfo` with live tracking, `HtoD/DtoH/DtoD`,
`MemsetD32/D8`; sync variants complete before return); texture arrays
(`ArrayCreate[_v2]`, `Array3DCreate[_v2]`, `ArrayDestroy`,
`ArrayGetDescriptor`, `Array3DGetDescriptor`, `MipmappedArrayCreate/Destroy/
GetLevel`, `Memcpy2D[Async]`, `Memcpy3D[Async]`) and texture objects
(`TexObjectCreate/Destroy/GetResourceDesc/GetTextureDesc`); surface
objects (`SurfObjectCreate/Destroy/GetResourceDesc`); modules
(`LoadData[Ex]`, `Unload`, `GetFunction`, `LaunchKernel`); streams
(create/destroy/query/wait-event/getflags/getpriority/synchronize);
events (create/destroy/record/query/synchronize); linker
(`Create/AddData/Complete/Destroy`); dispatch (`cuGetProcAddress[_v2]`
over all implemented symbols); errors (`GetErrorString/Name`).

Link hardening: `-Bsymbolic` (glibc weakly defines the instrumentation
hooks too) and `-z nodelete` (NVIDIA's libcudart unloads/reloads the
driver during negotiation; state must survive). `ACUDA_TRACE=1` logs
driver entries via `-finstrument-functions` hooks.

## Torch-needed surface (CUDA 13.4; torch 2.15.0+cu134)

Torch loads NVIDIA's `libcudart.so.13` (RPATH) on top of OUR
`libcuda.so.1` (see `docs/torch-nvrtc.md`), so every driver symbol it
touches must exist or its loader degrades silently. Found via
`LD_DEBUG=libs`, `ltrace`, and gdb backtraces into
`c10::cuda::device_count` / `gpu_kernel_impl_nocast`:
- `cuFuncGetAttribute` — real values from `acuda_kernel_info`
  (block=1024 max, shared/const/local=0, numRegs=declared count, ptx=XY,
  binary=sm). `cuFuncSetAttribute` — validated, recorded, largely
  advisory (no shared memory to carve; launch still caps at 0).
  IDs come from `include/acuda/driver_attrs.h` (official NVIDIA
  `CUfunction_attribute` 0–15, kept in sync with `cuda.h`).
- `cuLinkAddFile[_v2]` — read file, delegate to `AddData`.
- `cuModuleLoadFatBinary` - parses the `__fatBinC_Wrapper_t` control
  struct + `0xBA55ED50` container (layout verified field-by-field
  against torch's real fatbins with `cuobjdump`; kind 1=PTX/2=ELF,
  LZ4/ZSTD members stay unloadable). Extracts the first PTX member and
  still fully PTX-validates it. If a container has no PTX member, the
  optional `rtxdisam_native` library is dynamically loaded to lift its
  SASS to PTX. Multi-entry SASS-only containers keep the original image
  and are lifted lazily per `cuModuleGetFunction` entry name. Lifting asks
  rtxdisam to prefer the device's reported compute capability (`sm_XX`), so
  a multi-architecture fatbin selects the matching SASS member. Set
  `RTXDISAM_LIBRARY` to override the library path; a missing library
  leaves the clean `INVALID_IMAGE` path intact.
- `ACUDA_CUBIN_BACKEND` — selects who handles cubin/fatbin containers,
  in both ACuda and the `rtxdiasm_guard` cudart shim:
  - `0` (default/auto): use the C fatbin parser when the image carries
    PTX, otherwise lift SASS with rtxdisam (C#).
  - `1`: C fatbin parser only. A SASS-only container (e.g. torch's
    `libtorch_cuda.so`, which has no PTX) fails with
    `CUDA_ERROR_INVALID_IMAGE` instead of calling the lifter.
  - `2`: rtxdisam (C#) only. The C fatbin parser is skipped; rtxdisam
    parses the container and lifts SASS, and a PTX-only fatbin is
    returned verbatim by the C# container path.
  Unset, empty or invalid values behave like `0`. The lifter is looked
  up via `RTXDISAM_LIBRARY`, then the loader search path, then next to
  `libcuda.so` itself.
- `cuLaunchKernelEx` — plain configs forward to `cuLaunchKernel`
  (offsets verified vs NVIDIA `cuda.h`); any extended attributes are
  rejected (`INVALID_VALUE`).
- Handle lifetime: `cuLaunchKernel` does NOT consume the `CUfunction`
  shell (caller/module-owned, like real CUDA). Freeing it here used to
  turn every handle reuse into use-after-free (found via `test_driver`
  crash with `k=0xd`).
- Still missing (flagged by the loader, not yet on any taken path):
  `cuModuleGetGlobal_v2`, driver `cuOccupancyMaxActiveBlocksPerMultiprocessor`,
  `cuTensorMapEncodeTiled` (sm_90+, N/A at our 8.0), `cuLibraryLoadData`.

## Texture arrays and texture objects

`cuArrayCreate`/`cuArray3DCreate` allocate a Vulkan image behind the
`CUarray` handle. Supported formats: `UNSIGNED_INT8` x4 (RGBA8 UNORM),
`FLOAT` x4 (RGBA32F), `FLOAT` x1 (R32F), `UNSIGNED_INT32` x1 (R32UINT)
and `FLOAT` x1 with `CUDA_ARRAY3D_DEPTH_TEXTURE` (D32 depth compare).
Geometry follows the descriptor: `Height == 0` is 1D, `LAYERED` makes a
2D/1D array with `Depth` layers, `Depth > 1` is 3D, and `CUBEMAP` marks
a cube/cube-array (six layers per cube). `cuMipmappedArrayCreate`
allocates the full level chain and `cuMipmappedArrayGetLevel` returns a
borrowed level view.

`cuMemcpy2D`/`cuMemcpy3D` copy between host memory, device memory and
arrays (the device paths go through the host-visible arena). Rectangles
are honoured (source/dest offsets and pitches); the implementation reads
the affected array level/layer into a staging buffer, patches or copies
the rectangle, and writes it back.

`cuTexObjectCreate` pairs an array (or mipmapped array) with a
`CUDA_TEXTURE_DESC`: filter, mip filter and the three address modes map
to the runtime sampler, and `CUDA_ARRAY3D_DEPTH_TEXTURE` enables depth
comparison (`LESS_OR_EQUAL`). `CU_RESOURCE_TYPE_LINEAR` snapshots the
device buffer into a 1D image and `CU_RESOURCE_TYPE_PITCH2D` repacks the
pitched rows into a 2D image (updates require recreating the object).
`CU_TRSF_NORMALIZED_COORDINATES` selects a normalized sampler; without
it the sampler is created with `unnormalizedCoordinates`, which is what
`tex1Dfetch`-style integer fetches expect. A `CUtexObject` is an opaque 64-bit id: pass it as the value of
a `.param .texref` argument and `cuLaunchKernel` resolves it to the
image sampler binding. Texture objects are process-global and destroyed
with `cuTexObjectDestroy`; arrays must outlive the objects that use
them.

## Surface objects

`cuSurfObjectCreate` accepts `CU_RESOURCE_TYPE_ARRAY` and
`CU_RESOURCE_TYPE_MIPMAPPED_ARRAY` (level 0). Color arrays are created
with `VK_IMAGE_USAGE_STORAGE_BIT` and kept in `GENERAL` layout, so a
surface object is a borrowed view of the array image: kernel writes are
visible to subsequent array copies and vice versa. Depth, cube and
multisampled arrays cannot back a surface object and return
`NOT_SUPPORTED`. The array must outlive the surface object;
`cuSurfObjectDestroy` only drops the opaque handle.
`cuSurfObjectGetResourceDesc` returns the original descriptor.

PTX kernels can consume the handle either as a `.param .surfref` or as
a `.u64` kernel parameter used in `suld`/`sust`/`sured`/`suq` (the
modern `cudaSurfaceObject_t` form). The latter is detected by the
compiler and bound exactly like a `.surfref` parameter. When the handle
is not statically tied to one parameter (for example, loaded from
device memory or selected at runtime), the compiler binds one
runtime-sized storage-image descriptor array per distinct surface type
(dim/arrayed/format), each in its own descriptor set, plus a shared
metadata SSBO. Handles index every table, so one launch can select
among arbitrarily many live surface objects, including mixed
geometry/format operations in the same kernel. The driver builds each
type table from matching live surfaces, supplies dummies for unused
slots, and rejects a launch that exceeds the device's descriptor
limits. This path uses the Vulkan 1.2 core runtime descriptor arrays,
variable descriptor counts, and non-uniform storage-image indexing
(the runtime itself requires Vulkan 1.3); devices without those
features report `NOT_SUPPORTED`. `suq` metadata
is carried in the companion SSBO. The driver and CUDART provide the
tables; the public `acuda_launch_kernel` path reports `NOT_SUPPORTED`
for kernels that need them. The table is runtime-sized: the only bound
is the device's descriptor limits (the test suite exercises >1024 live
handles).

## Divergences (by design, all documented)

- `CUdeviceptr` is an **opaque handle**, not an address. Host-side
  pointer arithmetic is impossible; out-of-range sizes and dead handles
  return `INVALID_VALUE`. (A future arena allocator could make these
  real addresses; see resume.)
- `cuModuleLoadData[Ex]` accepts PTX text (`.version` sniffed past
  comments). ELF cubins and raw fatbin blobs are routed through the
  optional `rtxdisam_native` SASS-to-PTX bridge before validation.
  A SASS-only container with multiple text sections or fatbin members is
  retained and lifted per `cuModuleGetFunction` entry, so every entry is
  reachable. Invalid PTX returns `INVALID_IMAGE` at load; unknown entry
  at `GetFunction` returns `NOT_FOUND` (500).
- The **linker is PTX-passthrough**: `AddData` takes `PTX` only (cubin/
  fatbin/object → `INVALID_IMAGE`), `Complete` concatenates the texts
  and returns the buffer, which lives until `LinkDestroy`. There is no
  SASS JIT — the "cubin" is PTX for `cuModuleLoadData`. This is exactly
  the seam Numba's bind path (`cuLinkCreate/AddData/Complete` → module
  load) needs.
- `cuLaunchKernel` is 1D-only (`gridY/Z = blockY/Z = 1`), `sharedMem =
  0`, `extra = NULL`; any `blockDimX` in 1..1024 launches (workgroup
  size specializes per pipeline, cached). `kernelParams[i]` points at
  the arg value: 8-byte handle for `.u64` params, 4-byte word for
  scalars, in `.param` order.
- One Vulkan device is shared process-wide; contexts don't isolate.
  Single-threaded. `cuCtxCreate` flag mask limited to sched/map/lmem
  bits; stream flags 0/1; event flags 0–3.
- Output params are nulled before fallible calls (matches the test's
  expectation discipline: use a scratch variable for negative tests).

## Unresolved `.extern .func` calls

The CUDA driver API has no callback-registration entry point. When a
loaded module contains unresolved `.extern .func` calls,
`cuLaunchKernel` installs an identity stub for each host-call name that
has no callback registered through `acuda_host_call_register()`. The
stub copies the argument bytes into the return area and zero-fills the
rest. Applications that need real behavior should register a callback on
the same ACuda context before launching; an existing registration is
left untouched. This is what lets the driver-API truth harness run
`extern_func`/`extern_array` PTX without a separate linker.

## Client path (proven pattern, mirrors Numba)

`cuInit → cuDeviceGet → cuCtxCreate → cuMemAlloc → cuMemcpyHtoD →
cuLinkCreate/AddData(PTX)/Complete → cuModuleLoadData →
cuModuleGetFunction → cuLaunchKernel → cuCtxSynchronize →
cuMemcpyDtoH`. Covered end to end by `tests/integration/test_driver.c`
via `dlopen`, including pre-init, bad-image, unknown-symbol, and bad-
launch error paths.

## Python clients

`python-test/driver_vecadd.py` runs the same flow from Python via
ctypes + numpy (2²⁰ floats verified). Run with the cu134 venv, which
has numpy:
`"/home/magnusolsen/git/aicode-ptx/virtual python/ven-cu134/bin/python" python-test/driver_vecadd.py`.
`kernelParams` are built as an array of addresses of 8-byte handle
values (`.u64` params) / 4-byte words (scalars) — the exact marshaling
a Numba/CuPy bridge needs. Numba itself as a *driver client* additionally
needs Phase 6 compiler work first: it emits `.func` (not `.entry`),
generic `ld/st`, `.bXX`/`.sXX` types, signed 64-bit index math,
`bra.uni` diamond CFGs, `st.param` returns, and ~22 unpacked params
(see `supported-isa.md` §9 and `tests/ptx/vecadd_numba.ptx`, which the
script shows being rejected cleanly at link time).
