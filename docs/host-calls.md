# Host calls: split-dispatch state machine

Status: implemented for PTX `.extern .func` calls, including branches,
loops, divergence, predicated calls, pointer/narrow arguments, wide
returns, `.local`/`.shared`/dynamic `.extern .shared` persistence,
barriers, dynamic pointer homes across global and shared storage,
dynamic-home atomics, and re-entrant host launches. See
[Current limits](#current-limits).

This document specifies how ACuda supports a PTX `call` to an unresolved
`.extern .func` as a **host callback**. It separates three things that are
easy to conflate:

| Concept | Meaning | Where it belongs |
|---|---|---|
| Device extern | A `.func` defined in another PTX/SPIR-V module | Compile-time linking, see [device-module-linking.md](device-module-linking.md) |
| Host call | A device invocation that asks the host to run C code and returns a result | This document |
| Debug output | `printf`/`assert`-style one-way reporting | `VK_EXT_debug_printf` or a host-visible ring buffer |

CUDA device code cannot call host code either. A host call is an ACuda
extension, not PTX semantics.

## Why split dispatch

SPIR-V has no host-call instruction. Two emulations are possible:

1. **Spin mailbox** — the shader writes a request and spins until the host
   answers. Simple, but it burns GPU time, can trip the Windows TDR
   watchdog, and deadlocks if the servicing host thread is blocked in the
   launch wait.
2. **Split dispatch** — the kernel stops at the call, the host runs the
   callback, and a continuation dispatch resumes.

ACuda uses split dispatch. The kernel is relaunched until no invocation
has a pending request. There is no shader spin loop.

## State-machine model

A host-call kernel is one SPIR-V module with a per-invocation program
counter (PC) and register file in a state buffer:

```text
entry:
    load PC and registers from state
loop:
    dispatch on PC
  case block_i:
    execute the PTX block
    store registers back to Function variables
    PC = next block / branch target
    continue
  case host_call:
    store PC = continuation, spill registers to state
    write request, return
  case ret/exit:
    store PC = done, return
```

Because each invocation has its own PC and state slot, calls inside
branches, loops, and divergent paths work naturally. The host services
only invocations with a pending request and then relaunches; invocations
already at `done` return immediately.

### State layout

The state buffer begins with a 16-byte header (`state_offset`,
`shared_stride`). It is followed by one shared area per workgroup, then
one state slot per invocation (`host_state_stride` bytes):

```text
per-workgroup shared area: static + dynamic .shared (state-backed)

per-invocation state:
offset 0:   uint32 PC                   (0 = entry, 0xffffffff = done)
offset 16:  register slots, 16 bytes each
            +0 tag, +4 home tag for dynamic pointers, +8 value/offset
after regs: one byte-addressed return area per call id
after rets: per-invocation .local area
```

The compiler classifies each register's pointer home. A register with one
home stores only its offset. A register that selects among several global
buffer bindings stores an offset plus a runtime home tag; loads/stores
through it dispatch on that tag. `.local` storage lives in the per-
invocation state area and `.shared` storage lives in a per-workgroup
area of the state buffer, so both survive relaunches. Plain and counted
`bar.sync` work within a launch; a host-call path that reaches a barrier
must remain uniform, as in CUDA.

### Request layout

Per invocation (`host_request_stride` bytes):

```text
offset 0:   uint32 call id
offset 4:   uint32 pending
offset 8:   int32  callback status
offset 16:  arguments, one 8-byte slot per parameter (up to 32)
offset 272: return bytes
offset 400: uint32 pointer binding per argument (0xffffffff = scalar)
```

Arguments are 8-byte slots even for `.b8`/`.b16` parameters; narrow
values are zero-extended. Pointer arguments store an offset in the slot and a binding/tag in the
pointer-binding table. Global tags are user-buffer bindings; shared and
local tags point into the state buffer. At service time the runtime maps
the right memory and replaces the offset with a real host address, so the
callback receives a dereferenceable `uint64_t` pointer.

Return values are copied into the state buffer at
`host_ret_offsets[call_id]`; the continuation reads any width/offset
from there via `POP_HOST_RESULT`.

## Compiler pipeline

1. Parse `.extern .func` signatures (`.param` arguments and
   `.param .b8 ret[N]` return slots) and indexed `.param` arrays.
2. Treat a call to a declared external with no local body as a host call.
3. Add hidden state/request pointer parameters.
4. Split each block at every host call. The call ends its block; the
   remainder becomes an unlabeled continuation block inserted right
   after it, preserving fallthrough order.
5. Classify pointer homes and emit the PC state machine
   (`src/compiler/spirv_hoststate.c`).
6. Fill `acuda_kernel_info_t` with the call table, state/request strides,
   return offsets, and argument offsets.

Predicated host calls are left predicated: the emitter branches on the
predicate, returns after writing the request when true, and continues to
the call's continuation when false. The result slot is undefined when the
predicate was false, matching CUDA.

## Runtime

`acuda_launch_kernel` detects `host_state_machine` and:

1. allocates and zeroes the state and request buffers,
2. appends them to the user pointer bindings,
3. dispatches the kernel,
4. waits, reads pending requests,
5. maps user buffers, services callbacks, writes return bytes into the
   state buffer, clears pending,
6. relaunches until no request is pending.

The chain is synchronous. The callback runs on the host thread that
called `acuda_launch_kernel`; it must not re-enter the same context or
stream.

### Callback ABI

```c
typedef int (*acuda_host_call_fn)(void *user,
                                  const uint8_t *args, size_t args_bytes,
                                  uint8_t *ret, size_t ret_bytes);
```

- `args` is the request argument area (8-byte slots).
- Pointer arguments contain host virtual addresses for the corresponding
  user buffers.
- `ret` is writable; `ret_bytes` is the declared return size.
- Return 0 on success. A nonzero status aborts the remaining relaunches
  and is reported as `ACUDA_ERROR_INTERNAL`.

## Supported

- unresolved `.extern .func` calls with `.param` signatures,
- calls in branches and loops,
- divergent call sites (per-invocation PC),
- predicated `@p call.uni`,
- scalar, predicate, and pointer registers live across a call,
  including pointers derived by `mov`/`add`/`sub`,
- dynamic pointer homes across global bindings and shared storage, with
  runtime load/store and atomic dispatch,
- pointer arguments (global, shared, or local, mapped to host addresses),
- `.b8`/`.b16` arguments (zero-extended into 8-byte slots),
- return slots larger than 8 bytes, including reads at offsets,
- `.local`, static `.shared`, and dynamic `.extern .shared` storage
  across relaunches (`acuda_launch_kernel_ex` supplies the dynamic size),
- plain, counted, and `bar.arrive` barriers,
- up to 32 host-call ids and 32 arguments per call,
- callbacks that launch work on the same context/stream (re-entrant
  launches are tested),
- multiple distinct calls and multiple dynamic calls per invocation.

## Current limits

- Dynamic homes cover global bindings, static shared, global images,
  and parameter addresses. Parameter-address pointers are supported for
  `ld.param [%rd]`; ordinary memory loads through a dynamic parameter
  address follow the parameter-space path.
- Dynamic pointer arithmetic covers `mov`/`add`/`sub`/`mad.lo`,
  `selp` between pointer operands, and `mad.wide` with a pointer base.
  A shared pointer canonicalized for dynamic dispatch keeps an absolute
  shared-area offset.
- Barriers in a host-call function require the host-call path to be
  uniform across the workgroup, matching CUDA.
- `.extern .func` supports at most 32 parameters and 32 distinct function
  ids.
- Runtime relaunch has no fixed cap. `ACUDA_HOST_CALL_LIMIT` can set an
  optional safety limit; a runaway callback loop otherwise runs until the
  kernel exits.
- Dynamic `.extern .shared` requires `acuda_launch_kernel_ex` (or the
  driver's shared-memory launch path) so the runtime can size the
  per-workgroup state area.
- `acuda_launch_raw` auto-detects host-call kernels and services
  callbacks. It maps user buffers created by `acuda_malloc` on the same
  context, so global pointer callback arguments work without extra
  inputs. Externally created `VkBuffer`s have no discoverable
  `VkDeviceMemory`; use `acuda_launch_raw_ex` (the arena driver does)
  with host addresses, or `acuda_launch_kernel_ex`.
- Indirect calls are resolved through a cross-block `mov`/`selp` chain
  or a `.calltargets` list. A single candidate becomes a direct call;
  several candidates become a runtime comparison chain with one call
  block per callee (the call must be in the final block with a
  straight-line tail). Dispatch targets use the straight-line call
  path; multi-block callees remain limited to the final-block direct
  call form. A `.callprototype` without a `.calltargets` list still
  needs a statically resolvable single target.
- Function pointer tables use `.global .b64/.u64/.s64` initializers
  naming functions. Dynamic indices are supported for `.global` tables;
  `.const` tables follow the existing constant-index-only const-load
  rule.

## Test coverage

`test_hostcall` covers:

- one call and two sequential calls,
- a call inside a branch (taken and skipped),
- three calls inside a loop,
- divergent branch call sites (two threads, one calls),
- a predicated `@p call`,
- a pointer argument dereferenced by the callback,
- `.b8`/`.b16` arguments,
- a 16-byte return read at offsets 0 and 8,
- a pointer derived by `add.u64` and used after the call,
- `.local` storage read after a call,
- static `.shared` storage with a barrier across a relaunch,
- a shared pointer argument written by the callback,
- 10 callback arguments (32-argument path),
- a register holding different global homes on different threads,
- dynamic-home atomics across global and shared storage,
- a register mixing global and shared homes,
- `selp` between two pointer homes,
- `mad.wide` with a pointer base,
- dynamic `.extern .shared` with an 8-byte launch size,
- a dynamic global-image pointer,
- a dynamic parameter-address load,
- a statically resolvable indirect call,
- multi-target indirect dispatch through a dynamic `.global` function
  table and through a `selp` of two function addresses,
- multi-target dispatch inside a host-call kernel (the callback result
  selects the table entry, then the state machine resumes),
- a `brx.idx` jump table whose result feeds a host-call argument,
- the plain `acuda_launch_raw` path auto-servicing a host call with a
  global pointer argument (the runtime maps `acuda_malloc` buffers),
- a callback that launches another kernel on the same context/stream.

`extern_func.ptx` remains in the imported success kit with an identity
callback.

## Future work

Device module linking (prebuilt math/BLAS device code) is a separate idea
in [device-module-linking.md](device-module-linking.md). Host-call
functions with shared/local state would need those spaces included in the
state spill.
