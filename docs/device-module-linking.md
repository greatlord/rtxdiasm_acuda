# Device module linking (future idea)

Status: idea only, not planned or implemented.

This note records a separate idea from [host-calls.md](host-calls.md):
linking prebuilt **device** modules (for example common BLAS/math kernels)
into an ACuda kernel instead of compiling every callee from PTX source.

It is not the same as calling a host library. `cuBLAS`, `cuDNN`, and similar
libraries are host APIs that enqueue their own kernels; they cannot be linked
into a Vulkan compute shader. What can be linked is device code: a collection
of SPIR-V/PTX functions that the shader is allowed to call.

## Options

### 1. PTX-level linking (easiest, fits today)

Keep a companion PTX module (or a small PTX library) and resolve
`.extern .func` calls at compile time by inlining the callee, exactly like
the existing direct `.func` inliner. This needs no new runtime machinery.

```text
ACuda compile:
    parse entry PTX
    for each undefined .extern .func:
        look up name in device library PTX
        parse and inline
    emit one SPIR-V module
```

Pros: reuses `inl_run`, works with the current one-module pipeline.
Cons: every caller gets a copy of the callee; the library must be available
as PTX at compile time; no binary distribution benefit.

### 2. SPIR-V-level linking at compile time

Merge prebuilt SPIR-V library modules into the kernel module before
`vkCreateShaderModule`:

1. parse both modules,
2. merge types/constants/global variables,
3. renumber IDs to avoid collisions,
4. resolve `LinkageAttributes` imports to exported functions,
5. inline or keep `OpFunctionCall`,
6. validate the merged module.

Vulkan does not link modules at runtime, so this must happen in the compiler
or in a preprocessing step. It is a real linker, not a small feature.

Pros: ship device libraries as SPIR-V; no PTX parser dependency for library
users.
Cons: SPIR-V module surgery, decoration/capability compatibility, and
descriptor/binding layout agreement. High implementation cost.

### 3. Pipeline libraries

`VK_KHR_pipeline_library` is aimed at graphics pipeline stages. It does not
provide general device-function linking for compute shaders. Not a fit.

## Requirements if this is pursued

- The library and the kernel must target the same Vulkan environment and
  declare compatible capabilities/extensions.
- Descriptor bindings must not collide. A linker needs a binding remap or a
  strict library convention.
- The library must not depend on host state (unless it uses the host-call
  mechanism above).
- Inlining is the simplest correctness model; real `OpFunctionCall` support
  requires the whole call graph to be present in the final module.
- The library needs an ABI: parameter slots, return slots, pointer spaces,
  and alignment. This is the same ABI work as host calls, minus the host
  round-trip.

## Possible API sketch

```c
acuda_status_t acuda_device_library_add(acuda_context_t ctx,
                                        const char *name,
                                        const void *ptx_or_spv,
                                        size_t size);

acuda_status_t acuda_device_library_remove(acuda_context_t ctx,
                                           const char *name);
```

Resolution order should be deterministic: explicit kernel-local `.func`
definitions first, then registered libraries in registration order, then a
clean "undefined external function" error.

## Use cases

- Device-side math intrinsics and fast approximations.
- Small BLAS/GEMM building blocks used by multiple kernels.
- Quantization/dequantization helpers.
- A future device runtime library shared by many compiled kernels.

## Status

Not in the current plan. Recorded so the host-call work does not accidentally
grow into a linker, and so the two mechanisms can share the eventual calling
convention and `.param` slot representation.
