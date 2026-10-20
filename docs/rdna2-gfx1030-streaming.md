# RDNA2 / gfx1030 adaptive KV streaming (experimental)

This downstream branch adds a Linux ROCm RDNA2 port of the adaptive KV
streaming implementation maintained at
https://github.com/achyudh/llama.cpp (branch
feature/adaptive-kv-stream-rocm). It is not a replacement for
KVMem, and both KV owners must not be enabled concurrently.

## Design

- Restrict the RDNA4-specific WMMA paged decoder to RDNA4.
- Decode paged K/V on RDNA2 through existing tiled Flash Attention, with
  per-page-run log-sum-exp (LSE) for mathematically stable merging.
- Enforce a single split-K partition for LSE-enabled page runs; ordinary
  attention scheduling is unchanged.
- Require an independent ROCr/HSA SDMA engine for host-to-device KV
  transfers. Fail closed if only the HIP-shared engine is available.
- The HSA direct copy-engine API is not exposed on HIP Windows;
  return unavailable rather than attempting an unsafe streamer there.

## Building the RDNA2 HIP backend on Linux

Install a ROCm toolchain that supports gfx1030 and its HSA runtime.

    export ROCM_PATH=/path/to/rocm
    export HIP_PATH="$ROCM_PATH"
    export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"
    cmake -S . -B build-gfx1030 -G Ninja \
      -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 \
      -DGGML_HIP_NO_VMM=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-gfx1030 --target ggml-hip -j 3

This builds HIP kernels but does not by itself prove correct GPU inference.
The runtime may require matching GPU/ROCr libraries and an independent
HSA SDMA engine. Verify the actual device enumeration before launching.

## Tests

    python3 tests/test-rdna2-paged-merge.py

The host-side oracle compares normalized page-run merging with direct
softmax over 1,024 deterministic masked/extreme-logit cases. It is not a
GPU numerical-equivalence test.

On the contributor's gfx1030 ROCm 10.2 Preview environment, the four
targeted HIP objects (copy-engine.cu, fattn-paged-decode.cu, fattn.cu,
and the D=256 tile instantiation) compiled successfully on 2026-10-20.
This clean upstream-based checkout has **not** yet passed full-model
128K/250K inference or GPU numerical regression tests.

An earlier, more heavily patched llama.cpp + Swift/Qwen hybrid backend
did complete a near-250K KV-streaming recall experiment, but that result
must not be mistaken for validation of this isolated upstream port.

## Contributor branches

- rdna2/sdma-safety: standalone SDMA safety fix.
- rdna2/paged-attention: standalone tiled paged-attention fallback and
  host numerical oracle.
- rdna2/gfx1030: integration of both patches plus this documentation.

Each is based on the upstream adaptive KV branch; the first two may be
proposed as separate upstream PRs after full review/testing.

## Updating

Keep upstream pointing at achyudh/llama.cpp and llama-official
pointing at ggml-org/llama.cpp. Before rebasing a contributor branch,
inspect the new upstream diffs and rerun HIP, numerical and full-model
tests. Do not blindly auto-merge major llama.cpp changes into a running
production model.
