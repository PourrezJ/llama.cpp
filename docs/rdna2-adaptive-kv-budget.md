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
