// The game's core count (BOTW-JOBS-NATIVE.md). sead hard-codes 3 (sead::CoreInfo::configure); with every per-core
// structure the game's jobs use given a slot per core, the core-count patch (tools/patches.py _core_info_configure)
// gives the game this count, and the job pool makes one worker per core. It is the host's physical cores less one
// (the render thread's), at least sead's 3 and at most 32 (the runtime's per-core structures' limit).
// SAVAGEWIND_GAME_CORES=<n> sets it (core-count sweeps).
#include <stdlib.h>

#include "../platform.h"

int rt_game_cores(void) {
    static int n;
    if (!n) {
        const char *e = getenv("SAVAGEWIND_GAME_CORES");
        int v = e ? atoi(e) : plat_physical_cores() - 1;
        n = v < 3 ? 3 : v > 32 ? 32 : v;
    }
    return n;
}
