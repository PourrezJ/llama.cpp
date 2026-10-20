#include "../src/llama-kv-budget-autotune.h"

#include <cassert>
#include <cstddef>

int main() {
    using T = llama_kv_budget_autotune;
    T ctl;
    ctl.configure(256*T::MiB, 1024*T::MiB, 131072);
    assert(ctl.budget() == 256*T::MiB);

    // A 128K context's first 64 decode tokens are transfer-limited.
    ctl.observe({64, 112000, 1000000, 10000, 7500});
    assert(ctl.wants_resize());
    assert(ctl.pending_budget() == 512*T::MiB);
    ctl.resized(true);
    assert(ctl.budget() == 512*T::MiB);

    // An improvement is accepted; the next level becomes eligible.
    ctl.observe({64, 112064, 800000, 10000, 6500});
    assert(!ctl.wants_resize());
    assert(ctl.budget() == 512*T::MiB);
    ctl.observe({64, 112128, 800000, 10000, 6500});
    assert(ctl.wants_resize());
    assert(ctl.pending_budget() == 768*T::MiB);
    ctl.resized(true);

    // A higher budget is rejected if latency is worse, without repeated retries.
    ctl.observe({64, 112192, 1500000, 10000, 5200});
    assert(ctl.wants_resize());
    assert(ctl.pending_budget() == 512*T::MiB);
    ctl.resized(true);
    assert(ctl.budget() == 512*T::MiB);
    ctl.observe({64, 112256, 850000, 10000, 6000});
    assert(!ctl.wants_resize());

    // Context reset reduces the pool back to the minimum.
    ctl.observe({64, 2000, 200000, 800, 0});
    assert(ctl.wants_resize());
    assert(ctl.pending_budget() == 256*T::MiB);
    ctl.resized(true);

    // No stream traffic: do not use extra VRAM.
    ctl.observe({64, 2300, 100000, 1000, 0});
    assert(!ctl.wants_resize());

    // Changing the device's available budget changes the actual tier sizes.
    ctl.configure(64*T::MiB, 256*T::MiB, 8192);
    assert(ctl.budget() == 64*T::MiB);
    ctl.observe({64, 6000, 1000000, 10000, 9500});
    assert(ctl.pending_budget() == 128*T::MiB);
    ctl.resized(false);
    assert(ctl.budget() == 64*T::MiB);
    ctl.observe({64, 6020, 1000000, 10000, 9500});
    assert(!ctl.wants_resize());
}
