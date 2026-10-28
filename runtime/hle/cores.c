// The game's core count (BOTW-JOBS-NATIVE.md) and the renderer's translation workers, sized together: as a PC game
// sizes its threads, the threads busy every frame are no more than the cores. sead hard-codes 3
// (sead::CoreInfo::configure); with every per-core structure the game's jobs use given a slot per core, the core-count
// patch (tools/patches.py _core_info_configure) gives the game this count, and the job pool makes one worker per core.
// The renderer's threads besides: its render thread, its recording thread and the translation workers.
//
// Measured 28 Oct (Kakariko 720p, Ryzen 7 7800X3D, 8 cores, one logical processor per core; back to back, the same
// frames): game cores N + translation workers W = 3 + 3: 316-319 fps; 4 + 2: 316; 4 + 3: 312; 5 + 2: 310; 3 + 4: 309;
// 6 + 3: 277; 7 + 3 (the cores less one, the default before): 276. Past the cores the OS time-slices the game's main
// thread and the render thread with helpers (each wait of the main thread 2-3x longer at N = 7, the same jobs ~36%
// more CPU), so N is the physical cores less the renderer's threads, at least sead's 3 and at most 32 (the runtime's
// per-core structures' limit). SAVAGEWIND_GAME_CORES=<n> sets it (core-count sweeps).
#include <stdlib.h>

#include "../platform.h"

// Translation workers when SAVAGEWIND_VK_PAR doesn't say: half the cores less one, at least 2 (1 below 4 cores), at most
// 4. Not a size limit but the render thread's pace: it plans a frame's draws in ~2.6 ms and the workers translate them
// in ~5 ms of work, so ~2 keep up and 1-2 more clear the list's last chunks (3 -> 4 workers: render lists 3.30 ->
// 3.21 ms, its wait for the last chunks 0.78 -> 0.64; Kakariko 720p, 28 Oct). More would idle on cores the game can use:
// at 16 cores the game gets 10, not 7. (runtime/nvn/vulkan/exec.c par_start has the measurements.)
int rt_render_workers_default(void) {
    int cores = plat_physical_cores(), w = cores / 2 - 1;
    return cores < 4 ? 1 : w < 2 ? 2 : w > 4 ? 4 : w;
}

int rt_game_cores(void) {
    static int n;
    if (!n) {
        const char *e = getenv("SAVAGEWIND_GAME_CORES"), *w = getenv("SAVAGEWIND_VK_PAR");
        int workers = w ? atoi(w) : rt_render_workers_default();
        int v = e ? atoi(e) : plat_physical_cores() - (workers > 0 ? workers : 0) - 2;
        n = v < 3 ? 3 : v > 32 ? 32 : v;
    }
    return n;
}
