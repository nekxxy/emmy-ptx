# Emmy PTX: a direct Kernel IR backend

A Python 3.12 compiler from Emmy's typed Kernel IR to NVIDIA PTX. It deserializes with `Graph.from_dict` and emits PTX itself. It does not call Emmy's renderer, nvcc, or another compiler.

## Run

```sh
python -m pip install -r requirements.txt
python compile.py /path/to/problem/kernel.ir.json > out.ptx
```

The matching `meta.json` must sit beside the input. Its `arg_order` is authoritative, including for pipeline stages. Output is PTX only; failures go to stderr.

To reproduce the official open-set test, clone the organizer's repository separately, then run:

```sh
python /path/to/emmy-hackathon-2026/submit.py p00_scalar_add --compiler /path/to/this/compile.py
```

Organizer task repository: https://github.com/cloudrift-ai/emmy-hackathon-2026

## Design

- A fresh virtual-register allocator separates 32-bit integers, pointers, floats, and predicates. `ptxas` handles physical allocation.
- Tile extents drive linear thread decoding and the tail guard. Buffer shapes drive row-major address arithmetic.
- Statements are lowered directly: scalar/vector-element loads and stores, arithmetic, branches, strided loops, and reduction updates.
- Reduction state starts at its operator identity. Nested reduction loops reset only the state they own; explicit `Init` state remains carried across chunks.
- Warp combines use XOR shuffles and the IR's own combine expressions, including coupled online-softmax state.
- Cross-warp combines use a simple thread-zero fold in shared memory followed by a block broadcast. This favors clarity and correctness over speed.
- Matrix multiplication uses direct `cp.async` copies, commit/wait groups, and atomic add stores where the IR requests them. Large shared allocations use dynamic shared memory from the launch descriptor.

## Verified status

All 15 published problems pass on the organizer's RTX 5090 runner as of October 27, 2026. See `all15-final.txt` for the complete result: pointwise 4/4, reductions 4/4, matmul 2/2, fused 3/3, attention-like 2/2.

This is **open-set correctness**, not a hidden-set score or prize prediction. Timings are single-run feedback without clock locking, not benchmarks. The code is driven by IR structure rather than problem names, supplied expected outputs, or fixed dimensions.

## Additional tests

`stress.py` supplies fresh random input arrays and checks independent NumPy formulas: 15/15 passed (`stress-final.txt`). `variants.py` changes typed IR shapes, operation mixes, and reduction lengths: 55/55 passed (`variants-final.txt`). Approximate tanh/sigmoid/SiLU variants use an explicitly chosen test budget of rtol=1e-4, atol=1e-5; they have no organizer-provided tolerance. The approximate variants did not pass the add problem's exact-equality budget before this custom test budget was chosen. This is additional engineering evidence, not a hidden evaluation.

Tests need NumPy and requests in addition to the compiler requirements. They read an official task checkout supplied as an argument and do not redistribute its files.

## Limits

The backend targets the contest's static f32 kernels. It is not a general CUDA compiler. Tensor cores, TMA, dynamic shapes, and serial runtime coordinates are not implemented. Exotic schedules, new dtypes, or operations outside the current lowering table may raise an explicit error. Hidden-set performance and correctness are unmeasured.

## Provenance

The lowering code and project documentation were built with AI coding assistance for this entry. The entrant must understand the implementation and record their own explanation. An editable narration draft is available for preparation. No generated voice or substituted understanding is supplied; the entrant must record in their own voice and explain what they understand.

Emmy (`emmy-ml==0.3.11`) is an external Apache-2.0 dependency: https://github.com/cloudrift-ai/emmy

This repository does not redistribute the organizer's task code, problem IRs, reference CUDA, or expected outputs. Obtain those from the official repository.
