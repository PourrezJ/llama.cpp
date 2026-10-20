# Experimental KV decode auto-tuning (RDNA2 and other backends)

The legacy fixed decode budget remains unchanged without the explicit
--kv-resident-autotune option. This experiment adjusts a **bounded** decode
budget using workload measurements. It does **not** merge KVMem with KV
streaming or alter the prefill budget.

## CLI

~~~sh
llama-server --kv-stream --flash-attn on \
  --kv-resident-mib 256 \
  --kv-resident-mib-decode 1024 \
  --kv-resident-autotune \
  -np 1
~~~

The first limit is the minimum and prefill budget. The second is the maximum
allowed for decode, **not** a fixed target. Four candidates span these bounds:
256/512/768/1024 MiB here. Different limits generate different candidates.
VRAM free-space checks use the queried device's own total/free capacity,
without GPU-specific constants or a fixed 128K token threshold.

Current streaming used-cell tracking handles only the first stream. This
experimental controller rejects -np greater than 1 until multi-stream
workloads are separately validated.

## First-version control loop

1. Begin at the configured minimum for a small decode footprint.
2. Count actual attended and streamed KV pages for each decode ubatch,
   plus requested host-to-device bytes. Observe timing only after the
   scheduler's existing synchronize calls: no new GPU sync per token.
3. When at least 25% of attended pages stream, and after 64 evaluated
   tokens, **probe** the next budget instead of assuming more VRAM helps.
4. Over a further 64 evaluated tokens, keep the new budget only when
   observed decode verification throughput improves by at least 8%.
   Otherwise restore the last accepted budget and suppress retries until
   context occupancy or transfer pressure changes materially.
5. After a major context reset or near-zero streaming pressure, reduce
   the pool. Resize only between batches through resize_kv_stream().
6. Protect free VRAM with device-relative headroom. After failed pool
   allocation, try restoring the old working budget; fail cleanly if
   the restoration cannot allocate either.

Thresholds are conservative tuning choices, **not** claims of optimality.
The observed throughput is per model-evaluated token, including MTP
verification, not necessarily user-visible accepted tokens/second. Counts
reflect demanded host-to-device copies, **not** measured PCIe stall time.

## Verification

~~~sh
g++ -std=c++17 -Wall -Wextra -Werror -O2 \
  tests/test-kv-budget-autotune.cpp -o /tmp/test-kv-budget-autotune
/tmp/test-kv-budget-autotune
~~~

The pure policy test covers promotion, rejection, cooldown, allocation
refusal, context resets, zero traffic, and smaller GPU budget ranges.
A full CPU llama-server build checks the CLI and source integration,
but not GPU correctness or runtime performance.

The next qualification must run on gfx1030 with Swift IQ3_S and external
MTP2, repeating 40K/112K/128K with identical prompts, multiple passes,
actual VRAM statistics and stable MTP acceptance. Include low-VRAM,
prefill/decode switching, long-to-short contexts, and allocation failures.
The severe prior 112K slowdown is a known regression target.

Keep this opt-in, do not alter the stable Ajean preset, and do not send
this as a merge-ready upstream patch before GPU verification.
