#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

// Experimental, backend-agnostic controller. The caller measures real KV page traffic and
// completed decode work; no context-position threshold or GPU model is hard-coded here.
// A promoted budget is only retained after a throughput comparison at nearly the same
// context length. Resizes themselves are deliberately performed by llama_context.
class llama_kv_budget_autotune {
public:
    static constexpr size_t MiB = 1024*1024;

    struct sample {
        uint32_t evaluated_tokens = 0;
        uint32_t used_cells       = 0;
        uint64_t elapsed_us       = 0;
        uint64_t attended_pages   = 0;
        uint64_t streamed_pages   = 0;
    };

    void configure(size_t lower, size_t upper, uint32_t context_size) {
        // Four evenly-spaced candidate budgets. For a 256..1024 MiB range they
        // are precisely 256/512/768/1024 MiB, but other GPUs may use other caps.
        lower = std::max(lower, MiB);
        upper = std::max(upper, lower);
        for (int i = 0; i < 4; ++i) {
            budgets[i] = lower + ((upper - lower) / 3 / MiB * i)*MiB;
        }
        budgets[3] = upper;
        n_ctx = std::max(context_size, 1u);
        current = accepted = 0;
        pending = -1;
        rejected_at = 0;
        last_rejected_pressure = 0;
        reset_window();
    }

    size_t budget() const { return budgets[current]; }
    size_t pending_budget() const { return pending >= 0 ? budgets[pending] : budget(); }
    bool wants_resize() const { return pending >= 0 && pending != current; }
    int level() const { return current; }

    void resized(bool success) {
        if (pending < 0) {
            return;
        }
        if (success) {
            current = pending;
        } else {
            // Do not repeatedly attempt an allocation that just failed.
            rejected_at = most_recent_cells;
            last_rejected_pressure = last_pressure;
        }
        pending = -1;
        reset_window();
    }

    void observe(const sample & s) {
        if (s.evaluated_tokens == 0 || s.elapsed_us == 0 || s.attended_pages == 0) {
            return;
        }
        // A substantial context reset invalidates earlier comparisons.
        if (most_recent_cells > s.used_cells &&
                most_recent_cells - s.used_cells > std::max(256u, n_ctx/16)) {
            accepted = 0;
            if (current > 0) {
                pending = 0;
            }
            rejected_at = 0;
            baseline_tps = 0;
            reset_window();
        }
        most_recent_cells = s.used_cells;
        if (pending >= 0) {
            return;
        }

        window_tokens   += s.evaluated_tokens;
        window_us       += s.elapsed_us;
        window_attended += s.attended_pages;
        window_streamed += s.streamed_pages;
        if (window_tokens < 64 || window_us == 0) {
            return;
        }

        const double tps = 1e6*window_tokens / (double) window_us;
        const double pressure = window_attended ? (double) window_streamed / window_attended : 0;
        last_pressure = pressure;
        reset_window();

        if (current != accepted) {
            // Trial budgets must outperform the accepted reference by 8% to
            // justify the disruption and memory cost of the larger pool.
            if (tps > baseline_tps*1.08) {
                accepted = current;
                baseline_tps = tps;
                rejected_at = 0;
            } else {
                rejected_at = s.used_cells;
                last_rejected_pressure = pressure;
                pending = accepted;
            }
            return;
        }

        // The transfer pressure triggers *measurement*, not blind promotion.
        // Failed trials stay vetoed until context or transfer pressure changes.
        const bool new_workload = s.used_cells > rejected_at + std::max(256u, n_ctx/16) ||
                                  pressure > last_rejected_pressure + 0.20;
        if (pressure >= 0.25 && current < 3 &&
                (rejected_at == 0 || new_workload) &&
                budgets[current + 1] > budgets[current]) {
            baseline_tps = tps;
            pending = current + 1;
            return;
        }

        // Release memory when streaming has essentially stopped (e.g. a short
        // replacement conversation), without oscillating around a token boundary.
        if (current > 0 && pressure < 0.02) {
            pending = 0;
            accepted = 0;
        }
    }

private:
    void reset_window() {
        window_tokens = 0;
        window_us = window_attended = window_streamed = 0;
    }

    std::array<size_t, 4> budgets = {};
    uint32_t n_ctx = 1;
    int current = 0;
    int accepted = 0;
    int pending = -1;
    uint32_t most_recent_cells = 0;
    uint32_t rejected_at = 0;
    double last_pressure = 0;
    double last_rejected_pressure = 0;
    double baseline_tps = 0;
    uint64_t window_tokens = 0;
    uint64_t window_us = 0;
    uint64_t window_attended = 0;
    uint64_t window_streamed = 0;
};
