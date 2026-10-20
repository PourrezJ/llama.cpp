# Adaptive KV residency for RDNA2 (experimental)

## Purpose

Dynamically grow the **decode** resident KV budget in four equal quartiles
of the configured context. This is distinct from the `--kv-resident-mib`
prefill budget, which remains fixed. No KVMem hooks or KV ownership changes.
The option is disabled unless expressly enabled.

Example for a 131072-token context:

```sh
--kv-stream --kv-resident-mib 256 \
--kv-resident-mib-decode 1024 --kv-resident-adaptive \
-c 131072 -ctk q8_0 -ctv q4_0 --flash-attn on
```

| Occupied context (tokens) | Decode budget |
| --- | ---: |
| 0–32768 | 256 MiB |
| 32769–65536 | 512 MiB |
| 65537–98304 | 768 MiB |
| 98305–131072 | 1024 MiB |

*The 1024 MiB ceiling is from the user's `--kv-resident-mib-decode`
setting. With other context sizes, the boundary tokens scale proportionally;
with other budgets, each step equals one quarter of that ceiling.*

### Safety and implementation

- The backend detects actual occupied KV cells (including inside the hybrid
  attention/GDN memory wrapper) rather than assuming the whole logical
  `-c` context is occupied.
- Changes happen between decode batches, with GPU synchronization. Before a
  prefill, the decode pool is released and the smaller prefill pool restored.
- Downward changes retain a 1024-token hysteresis around context boundaries.
- If VRAM allocation fails, retry at the next lower quarter instead of
  corrupting resident KV pages. The failed upper ceiling is retained for the
  process to avoid repeated VRAM thrashing.
- A resize flushes and rebuilds page views/graph references using the
  existing stream interface. The base non-adaptive path is unchanged.
- `-lv 4` exposes the **chosen budget and actual resident/transfer-ring page
  counts**, so a logical budget change can be distinguished from a physical
  resize (notably when the entire short context already fits in VRAM).

### Experimental results before adaptive testing

- RX 6950 XT gfx1030, Swift + external MTP2, KV Q8/Q4.
- Fixed 256 MiB decode: 40K 26.26 tok/s; 112K 13.74; 128K 6.43.
- Fixed 1024 MiB decode: 40K 18.97 tok/s; 112K 11.92; 128K 10.63.
- This motivates testing an adaptive strategy; the point of crossover and
  actual speed at each step remain empirical questions, not guarantees.

The RDNA2 upstream PR remains unchanged until real long-context testing and
numerical checks of this experimental feature are complete.

## GPU integration validation (2026-10-20)

**Hardware:** RX 6950 XT 16 GiB (`gfx1030`) with ROCm 10.2 Preview,
Swift Qwen3.8-27B IQ3_S with external MTP2 and Q8_0/Q4_0 KV. Test used
one slot, a 131072-token configured context, 256 MiB for prefill and a
1024 MiB adaptive decode ceiling. Logs captured with `-lv 4`.

| Prompt | Actual decode budget | Resident pages | Transfer-ring pages | Decode tok/s | Synthetic marker |
| ---: | ---: | ---: | ---: | ---: | --- |
| 20000 | 256 MiB | 31 | 64 | 20.84 | PASS |
| 39971 | 512 MiB | 65 | 108 | 22.95 | PASS |
| 80000 | 768 MiB | 88 | 241 | 15.90 | PASS |
| 111970 | 1024 MiB | 115 | 339 | 6.12 | PASS |
| 127871 | 1024 MiB | 106 | 406 | 10.08 | PASS |

The physical resident pool increased from 31 to 65 to 88 to 115 pages;
at 127871 tokens the ring requirements rose, leaving 106 resident pages
inside the **same** 1024 MiB combined pool+ring ceiling. These numbers
must not be interpreted as 1024 MiB of resident-only K/V data.

A smaller 8192-token GPU smoke ran sequential 1700, 3700, 5700 and
7699-token prompts. All four budget selections were logged (256/512/768/
1024 MiB respectively) and the marker was retrieved in all four runs.
On that short context, full KV may already fit in VRAM, so it primarily
verifies policy and state transitions, not meaningful pool growth.

**Performance qualification: NOT PASSED.** The adaptive result of 10.08
tok/s at 127871 tokens is below the earlier fixed 1024 MiB result of
10.63 tok/s; at 112K the drop was larger (6.12 vs 11.92 tok/s).
Measurements have different logging verbosity (`-lv 4` in adaptive),
timing conditions and prefix history. These are not controlled A/B
comparisons, so do not attribute the slowdown to one cause without a
same-verbosity rerun. The server emitted many graph reallocation notices
throughout prefill; potential runtime overhead requires profiling.

Original local evidence (not committed to this repository):
`/home/djoe/ai-runtime/experiments/llama-rdna2-validation-20261020/`
(`mtp-adaptive-long-server.log`, `mtp-adaptive-long-*.json`,
`test-adaptive-long.sh`).

**Limits:** Tested with `-np 1` only, and a single synthetic marker.
No GPU failure, HIP error or OOM was observed in the test. This does not
prove compatibility with parallel/multisequence operation, arbitrary
long-context reasoning, or mid-decode context shrink. Keep the feature
opt-in and leave production Ajean unchanged.
