# rtxdiasm_acuda

ACuda CUDA Driver API implementation (driver + runtime foundation), split from ACuda as a standalone module.

## Outputs
- **Linux:** `libcuda.so.1` (SONAME `libcuda.so.1`)
- **Windows:** `nvcuda.dll`

## Components
- `acuda_driver` (shared): CUDA Driver API compatibility library (`libcuda`/`nvcuda`)
- `acuda_runtime` (static): internal Vulkan/runtime foundation used by the driver and ACuda tests

## Dependencies
- Vulkan
- `rtx_modules/rtxdiasm-ptx-to-spirv` (PTX-to-SPIR-V compiler module, expected as sibling by default)

## Integration
When used from ACuda, add this module via `add_subdirectory(...)` so it provides `acuda_driver` and `acuda_runtime` targets to tests and other modules (e.g. `rtxdiasm-cudart`).

## Copyright and authorship
Copyright 2026 by Magnus Olsen.

This code had assistance by models DeepFlash 4.1, Muse Spark 1.3, and Iceman.

## License
MIT (see `LICENSE`).
