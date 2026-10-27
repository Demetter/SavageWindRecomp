// Frame rate (Video > Frame rate): Off (no limit), 30, 60 (the default), 90 or 120 frames per second, and the game's
// clock the same at every one of them. Shared by the macOS, Linux and Windows builds.
//
// How BOTW 1.8.2 keeps time (main module, runtime addresses):
//   - Its frame loop (sead GameFrameworkNx, 0x700101287c) asks for a present interval of 2 vsyncs (30 fps), 3 (20 fps)
//     after three slow frames in a row. The present (nvnQueuePresentTexture: objects.c present_pace) keeps that pace;
//     the frame loop's own vsync wait (0x7000d811fc) only runs for the first frames after start-up.
//   - ksys::VFR (the instance at [0x7001d32010]) turns that interval into the game's time step: initialised with a
//     base interval of 2, so one "delta frame" is 1/30 s. Every frame VFR::updateInterval (0x7001409230) sets the
//     interval ratio (interval / 2: 1.0 at 30 fps, 1.5 at 20 fps), and delta frame = ratio x the slowest active time
//     multiplier (slow motion), delta time = delta frame / 30. About 1600 call sites read them: the time of day
//     (TimeMgr adds its step x delta frame), the Blood Moon timer, the Havok step (the physics world is handed delta
//     time every frame), stamina, animation, AI. A few read the interval itself.
//   - The menus' input timing counts frames: holding a direction on a value steps it every 6 frames on the title
//     screen (0.2 s at 30 fps, 0.1 s at 120). The paused game (pause menu, map: the state at 0x7001cb48a4 is 2, 0
//     while playing) updates VFR twice a frame.
// So the game already runs at any frame rate as long as VFR is told how long a frame really took. Savage Wind:
//   - paces the presents at the chosen rate on a steady grid (present_pace, framepace_period_ns; Off: 300 fps at
//     most, FPS_CEILING; with VSync On or Adaptive a little under the display's refresh rate), and skips the frame
//     loop's vsync wait (rt_frame_wait, tools/patches.py);
//   - tells VFR how long the frame really took (rt_vfr_update, inside updateInterval after the game set its ratio):
//     ratio = seconds x 30 (0.5 at 60 fps, 0.25 at 120): one period when the frames keep the pace, the measured time
//     averaged otherwise, changing by a few percent a frame at most, and whatever it gave more or less than the real
//     time is paid back over the next frames, so the game's time follows the real time. The interval follows the frame rate averaged over ~20 frames: 1
//     above ~43 fps (the game's own 60 fps branches, made for its debug viewer), 2 below ~35, 3 below ~23. A step is 1/20 s at most (ratio 1.5, the console's longest):
//     a hitch is paid back over the next frames, but frames slower than that for long (over 1/4 s owed) slow the
//     game down, as on the console, instead of taking steps big enough to break the physics;
//   - runs the menus (the title screen, the pause menu and the map) and the events (cutscenes, talks) at the chosen
//     frame rate too, as Cemu's FPS++ does: the menus' speed follows the frame's length (rt_gui_speed, below), and
//     movies are paced by their pictures' time (movie.c). SAVAGEWIND_MENU_FPS / SAVAGEWIND_EVENT_FPS hold them at a
//     rate (30: the console's; some events' waits count frames);
//   - steps the controllers' hold counters by the frame's length (CONTROLLER_STEP, below).
// Test and debug switches:
//   SAVAGEWIND_FPS=<n>             frame rate instead of the setting (0: no limit)
//   SAVAGEWIND_BACKGROUND_FPS=<n>  frame rate at most while another window is in front (default 60; 0: no cap)
//   SAVAGEWIND_FRAMEPACE=0         the game's own pacing and clock (its 30 / 20 fps, as before)
//   SAVAGEWIND_MOVIE_FPS=<n>       frame rate while a movie plays (default: the chosen one, the pictures paced by
//                                  their time; with SAVAGEWIND_MOVIE_PACE=0 the movie's own; 0: the chosen one)
//   SAVAGEWIND_MENU_FPS=<n>        frame rate in the menus (default 0: the chosen one, or 30 in game code without the
//                                  GUI speed patch; 30: the console's)
//   SAVAGEWIND_EVENT_FPS=<n>       frame rate in the events (default 0: the chosen one; 30: the console's)
//   SAVAGEWIND_LOG_FRAMEPACE=<n>   every n clock updates: target, frame length, VFR ratio and interval
//   SAVAGEWIND_NO_CONTROLLER_STEP=1  the controllers count presented frames (as the game does)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../game_sites.h"
#include "../gamelib.h"
#include "../hle.h"
#include "../settings.h"
#include "../framepace.h"
#include "../window_size.h"

extern volatile int g_movie_playing;  // runtime/hle/movie.c: video decoders running
extern volatile int g_movie_fps;      // runtime/hle/movie.c: the playing movie's frame rate

#define VF(T, base, off) (*(T *)(uintptr_t)((base) + (off)))
// VFR's per-core deltas (runtime/hle/vfr.c: they live there, a slot for each of the game's cores)
float rt_vfr_core0_delta_frame(void);
float rt_vfr_core_delta_frame(int core);
#define PAUSE_STATE g_site.pause_state  // (u32: 0 playing, 2 in the pause menu or the map)

static int env_int(const char *name, int def) {
    const char *e = getenv(name);
    return e && *e ? atoi(e) : def;
}

bool framepace_on(void) {
    static int on = -1;
    if (on < 0) on = env_int("SAVAGEWIND_FRAMEPACE", 1) != 0;
    return on;
}

// ------------------------------------------------------------------ the frame rate
// The menus: the game is paused (pause menu, map), or the title screen's menu is up (screen 1 in the UI screen
// manager [screen_mgr]: +0x28 count, +0x30 screens by id).
static bool in_menu(void) {
    if (*(uint32_t *)(uintptr_t)PAUSE_STATE) return true;
    uint64_t mgr = *(uint64_t *)(uintptr_t)g_site.screen_mgr;
    return mgr && VF(uint32_t, mgr, 0x28) > 1 && VF(uint64_t, mgr, 0x30) &&
           *(uint64_t *)(uintptr_t)(VF(uint64_t, mgr, 0x30) + 8);
}

// An event is running: a cutscene, a talk, an item fanfare (ksys::evt::Manager [event_mgr]: its active event at
// +0x1d2b8, as hasActiveEvent, 1.8.2 0x7000ffa208, reads it; the time of day stands still meanwhile). Some of their
// waits count frames: the new game's opening gave Link control 2 s early at 60 fps, 3 s at 120.
static bool in_event(void) {
    uint64_t evt = *(uint64_t *)(uintptr_t)g_site.event_mgr;
    return evt && *(uint64_t *)(uintptr_t)(evt + 0x1d2b8);
}

// The menus follow the chosen frame rate only in game code generated with the GUI speed patch (tools/patches.py
// _gui_speed, site gui_speed): older game code keeps the game's own GUI speed, too fast above 60 fps, so its menus
// stay at 30.
static bool gui_speed_patched(void) {
    const GameLibTables *t = gamelib_tables();
    for (uint32_t k = 0; t && k < *t->nsites; k++)
        if (!strcmp(t->sites[k].name, "gui_speed")) return true;
    return false;
}

bool framepace_in_event(void) { return in_event(); }

// The game's UI screen `id` exists (the screen manager's array; ids as runtime/ui/game_options.mm uses them)
static bool screen_up(uint32_t id) {
    uint64_t mgr = *(uint64_t *)(uintptr_t)g_site.screen_mgr;
    if (!mgr || VF(uint32_t, mgr, 0x28) <= id) return false;
    uint64_t arr = VF(uint64_t, mgr, 0x30);
    return arr && *(uint64_t *)(uintptr_t)(arr + 8 * id);
}

// (screens: 1 the start screen, 0x38 its load and confirmation menus)
bool framepace_title_ui(void) { return screen_up(1) || screen_up(0x38); }

// (pause state 2: the pause menu's pages and the map; 4 is the quick menus over the game)
bool framepace_menu_ui(void) { return *(uint32_t *)(uintptr_t)PAUSE_STATE == 2; }

int framepace_target(void) {
    static int forced = -2, movie = -2, menu = -2, event = -2;
    if (forced == -2) forced = env_int("SAVAGEWIND_FPS", -1);
    extern bool movie_paced(void);
    if (movie == -2) movie = env_int("SAVAGEWIND_MOVIE_FPS", movie_paced() ? 0 : -1);  // (-1: the movie's own frame rate)
    if (menu == -2) menu = env_int("SAVAGEWIND_MENU_FPS", gui_speed_patched() ? 0 : 30);
    if (event == -2) event = env_int("SAVAGEWIND_EVENT_FPS", 0);
    int fps = forced >= 0 ? forced : g_settings.fps_limit;
    if (fps < 0) fps = FPS_DEFAULT;
    if (movie != 0 && __atomic_load_n(&g_movie_playing, __ATOMIC_RELAXED) > 0) {
        int cap = movie > 0 ? movie : g_movie_fps;
        if (fps == 0 || fps > cap) fps = cap;
    }
    if (menu > 0 && (fps == 0 || fps > menu) && in_menu()) fps = menu;
    if (event > 0 && (fps == 0 || fps > event) && in_event()) fps = event;
    // Another window in front: at most SAVAGEWIND_BACKGROUND_FPS (default 60; 0: no cap), as PC games do. Without a
    // limit the game kept presenting 240-300 frames a second behind the desktop, which lagged (typing, windows).
    // Hidden runs (SAVAGEWIND_HIDDEN, the measurements) never have the focus: no cap.
    static int bg = -2;
    if (bg == -2) bg = getenv("SAVAGEWIND_HIDDEN") ? 0 : env_int("SAVAGEWIND_BACKGROUND_FPS", 60);
    extern bool input_window_focused(void);
    if (bg > 0 && (fps == 0 || fps > bg) && !input_window_focused()) fps = bg;
    return fps;
}

// No limit still stops at 300 fps: shorter steps lose precision in the game's float clocks (the time of day, 0..360,
// takes 1/30 of a minute times the step each frame: at 300 fps that is 1% of the value's rounding unit per frame).
// (measuring) SAVAGEWIND_FPS_CEILING=<fps>: another ceiling, to see how fast a scene could go.
static int fps_ceiling(void) {
    static int c;
    if (!c) c = env_int("SAVAGEWIND_FPS_CEILING", 300) > 0 ? env_int("SAVAGEWIND_FPS_CEILING", 300) : 300;
    return c;
}
#define FPS_CEILING fps_ceiling()

// With VSync On or Adaptive (FIFO), a frame rate at or over the display's refresh rate waits for the display: up to a
// refresh of delay, and with G-SYNC / FreeSync the display leaves its variable rate at the top of its range. So the
// frame rate stays under it there, at refresh - refresh^2 / 3600 (NVIDIA Reflex's cap: 59 fps at 60 Hz, 138 at 144,
// 224 at 240). Fast (MAILBOX) and Off (IMMEDIATE) never wait for the display: no cap.
static double vsync_cap(void) {
    if (g_settings.vsync != VSYNC_ON && g_settings.vsync != VSYNC_ADAPTIVE) return 0;
    double hz = winsize_refresh_rate();
    return hz >= 30 ? hz - hz * hz / 3600 : 0;
}

// A frame rate within 1% of the refresh rate divided by a whole number takes exactly that (60 on a 239.77 Hz display:
// 59.94, a picture every 4 refreshes; 30: 29.97). Otherwise the frames drift against the refreshes, and on a display
// without G-SYNC / FreeSync one in a few hundred waits a refresh less or more (a judder every few seconds). With
// G-SYNC / FreeSync it changes nothing that shows. Not with VSync Off, which doesn't wait for refreshes.
static double on_refresh(double fps) {
    double hz = g_settings.vsync == VSYNC_OFF ? 0 : winsize_refresh_rate();
    if (hz < 30 || fps > hz) return fps;
    double n = round(hz / fps);
    return n >= 1 && fabs(hz / n - fps) < fps * 0.01 ? hz / n : fps;
}

uint64_t framepace_period_ns(void) {
    int target = framepace_target();
    double fps = target > 0 ? target : FPS_CEILING, cap = vsync_cap();
    if (cap > 0 && fps > cap) fps = cap;
    return (uint64_t)(1e9 / on_refresh(fps) + 0.5);
}

// The frame loop's vsync wait: the present keeps the pace instead (1: skipped; 0: the game's own wait).
int rt_frame_wait(void) { return framepace_on(); }

// ------------------------------------------------------------------ presents
// The controllers (sead::ControllerBase::updateDerivativeParams_, 1.8.2 0x7000d9d8a0) count how long each button has
// been held, in frames: a held button's repeat starts after its delay and comes every pulse frames, and the game times
// some holds with the count. The count goes up by a step the game keeps at 1 (a float, controller_step: 1.8.2
// 0x7001c4ba40, read there only): the runtime sets it to the frame's length in console frames, so holds last as long
// as on the console.
#define CONTROLLER_STEP g_site.controller_step
static float g_step = 1;         // the last frame's length between presents, in console frames
static int g_grid_streak;        // presents in a row that kept the pace

void framepace_presented(uint64_t t, bool on_grid) {
    if (!framepace_on()) return;
    static uint64_t prev;
    static double smooth;
    double s = prev && t > prev ? (t - prev) / 1e9 : 1.0 / 30;
    prev = t;
    if (s > 0.25) s = 0.25;
    smooth = smooth > 0 ? smooth + (s - smooth) * 0.25 : s;
    static int measured = -1;  // (probe) SAVAGEWIND_STEP_MEASURED=1
    if (measured < 0) measured = env_int("SAVAGEWIND_STEP_MEASURED", 0);
    float step = (float)((on_grid && !measured ? s : smooth) * 30);
    g_step = step < 0.1f ? 0.1f : step > 1.5f ? 1.5f : step;
    g_grid_streak = on_grid ? g_grid_streak + 1 : 0;
    static int fixed = -1;
    if (fixed < 0) fixed = env_int("SAVAGEWIND_NO_CONTROLLER_STEP", 0);
    if (!fixed) *(float *)(uintptr_t)CONTROLLER_STEP = g_step;
}

// The menus' speed: every frame the game sets a GUI speed to its integer interval x 0.5: 1 at 30 fps, 0.5 at 60, and
// never lower, so the menus' cursor and animations ran faster above 60 fps (and at 60 when the interval follows the
// frame rate). It becomes the frame's length in console frames, as Cemu's FPS++ does on the Wii U.
// SAVAGEWIND_NO_GUI_SPEED=1: the game's own.
float rt_gui_speed(float game) {
    static int off = -1;
    if (off < 0) off = env_int("SAVAGEWIND_NO_GUI_SPEED", 0);
    return off || !framepace_on() ? game : g_step;
}

// ------------------------------------------------------------------ the game's clock (ksys::VFR)
// VFR fields (1.8.2; the layout of zeldaret's ksys::VFR): +0x28 interval changed, +0x29 interval override on,
// +0x30 new interval ratio, +0x34 interval, +0x3c interval ratio (both read through per-core pointers).
#define VFR_CHANGED 0x28
#define VFR_OVERRIDE 0x29
#define VFR_NEW_RATIO 0x30
#define VFR_INTERVAL 0x34
#define VFR_RATIO 0x3c

static void clock_probe(uint64_t vfr);
static void log_screens(void);
static void player_dump(uint64_t vfr);

void rt_vfr_update(uint64_t vfr) {
    extern bool nvn_timeline(void);
    if (nvn_timeline()) LOG("[tl] %.3f %x vfr\n", plat_now_ns() / 1e6, kernel_current_thread_handle());
    extern uint64_t g_fs_vfr;  // (SAVAGEWIND_FRAME_LOG=2, nvn/objects.c)
    g_fs_vfr = plat_now_ns();
    clock_probe(vfr);
    extern void rt_percore_check(void);  // (runtime/hle/percore.c: SAVAGEWIND_PERCORE_POISON)
    rt_percore_check();
    {   // (probe) SAVAGEWIND_CPU_CLOCK=1: every ~2 s, the main thread's and the process's clock while running (cycles /
        // CPU time), the main thread's share of the wall time, and cycles per frame: whether frames got slower
        // because the CPU ran slower or because they took more cycles
        static int on = -1;
        static uint64_t t0, c0, n0, pc0, pn0, frames;
        if (on < 0) on = env_int("SAVAGEWIND_CPU_CLOCK", 0);
        if (on) {
            extern uint64_t perf_thread_cycles(void), perf_thread_cpu_ns(void), perf_process_cycles(void),
                perf_process_cpu_ns(void);
            uint64_t now = plat_now_ns();
            frames++;
            if (!t0) t0 = now, c0 = perf_thread_cycles(), n0 = perf_thread_cpu_ns(), pc0 = perf_process_cycles(),
                     pn0 = perf_process_cpu_ns(), frames = 0;
            else if (now - t0 > 2000000000ull) {
                uint64_t c = perf_thread_cycles(), n = perf_thread_cpu_ns(), pc = perf_process_cycles(), pn = perf_process_cpu_ns();
                LOG("[clock] main %.2f GHz running %.0f%% of the time, %.2f Mcycles a frame; process %.2f GHz on %.2f "
                    "processors, %.2f Mcycles a frame (%llu frames)\n", n > n0 ? (c - c0) / (double)(n - n0) : 0,
                    100.0 * (n - n0) / (now - t0), (c - c0) / 1e6 / frames, pn > pn0 ? (pc - pc0) / (double)(pn - pn0) : 0,
                    (pn - pn0) / (double)(now - t0), (pc - pc0) / 1e6 / frames, (unsigned long long)frames);
                t0 = now, c0 = c, n0 = n, pc0 = pc, pn0 = pn, frames = 0;
            }
        }
    }
    log_screens();
    player_dump(vfr);
    extern void input_cam_frame(float ratio);  // runtime/hle/input.c: the mouse's motion in the frame, for the cameras
    if (!framepace_on()) {
        input_cam_frame(VF(float, vfr, VFR_RATIO));
        return;
    }
    static uint64_t last_t;
    static double real, given;  // seconds of real time and of game time handed out (only their gap matters)
    static double smooth;       // seconds between updates, averaged over about 20
    static double last_step;    // the step handed out last, seconds (0: none yet)
    static int last_target = -1;
    static double slow = 1;     // the step in console frames, averaged over about 20 (the interval follows it)
    static uint32_t last_iv;
    static int log_every = -1;
    static uint64_t nframe;
    if (log_every < 0) log_every = env_int("SAVAGEWIND_LOG_FRAMEPACE", 0);
    uint64_t now = plat_now_ns();
    double s = last_t && now > last_t ? (now - last_t) / 1e9 : 1.0 / 30;
    last_t = now;
    float ratio;
    bool paused = *(uint32_t *)(uintptr_t)PAUSE_STATE != 0;
    if (paused) {
        // paused (it updates twice a frame): the frame's length between presents, nothing owed
        ratio = g_step;
        real = given = 0;
        last_target = -1;
    } else {
        if (s > 0.25) s = 0.25;  // (a load or a pause in a debugger: not owed)
        double period = framepace_period_ns() / 1e9;
        int target = framepace_target();
        if (target != last_target) smooth = period, last_step = 0;  // (a new pace: start from it)
        last_target = target;
        smooth += (s - smooth) * 0.05;
        real += s;
        // The step changes slowly: the game reads some deltas a frame late, and a step that jumps between frames
        // moves the camera and Link apart (Link seemed to step back in place, seen from the side). One period once
        // the presents have kept the pace for half a second, else the average of ~20 frames; plus a fiftieth of what
        // is owed; and at most 3% away from the last step.
        static int measured = -1;  // (probe) SAVAGEWIND_STEP_MEASURED=1: the measured step even on the pace's grid
        if (measured < 0) measured = env_int("SAVAGEWIND_STEP_MEASURED", 0);
        double step = (g_grid_streak >= 30 && !measured ? period : smooth) + (real - given) * 0.02;
        if (last_step > 0 && step > last_step * 1.03) step = last_step * 1.03;
        if (last_step > 0 && step < last_step * 0.97) step = last_step * 0.97;
        static int floor_fps = -1;  // (probe) SAVAGEWIND_STEP_FLOOR_FPS=<fps>: the shortest step the game is given
        if (floor_fps < 0) floor_fps = env_int("SAVAGEWIND_STEP_FLOOR_FPS", 0);
        double floor_step = 1.0 / (floor_fps > 0 ? floor_fps : FPS_CEILING);
        if (step < floor_step) step = floor_step;
        if (step > 0.05) step = 0.05;
        last_step = step;
        ratio = (float)(step * 30);
    }
    bool held = VF(uint8_t, vfr, VFR_OVERRIDE) && ratio > 1;  // (stage loading: the game asks for single steps)
    if (held) ratio = 1;
    if (!paused) {
        given += ratio / 30.0;
        if (held || fabs(real - given) > 0.25) real = given;  // (nothing owed for a deliberate slow down)
    }
    // The interval switches some of the game's code between its 60 fps and 30 fps ways: only on a lasting change of
    // the frame rate (above ~43 fps: 1; below ~35: 2; below ~23: 3), never back and forth from one frame to the next.
    slow += (ratio - slow) * 0.05;
    static uint32_t iv = 2;
    if (iv == 1 && slow > 0.85) iv = 2;
    else if (iv == 2 && slow < 0.7) iv = 1;
    else if (iv == 2 && slow > 1.3) iv = 3;
    else if (iv == 3 && slow < 1.15) iv = 2;
    VF(float, vfr, VFR_RATIO) = ratio;
    VF(float, vfr, VFR_NEW_RATIO) = ratio;
    VF(uint32_t, vfr, VFR_INTERVAL) = iv;
    VF(uint8_t, vfr, VFR_CHANGED) = last_iv && iv != last_iv;
    last_iv = iv;
    input_cam_frame(ratio);
    if (log_every > 0 && nframe++ % (uint64_t)log_every == 0) {
        uint64_t pinfo = *(uint64_t *)(uintptr_t)g_site.player_info;  // (Link's position: PlayerInfo +0x88)
        LOG("[framepace] target %d fps, %.2f ms since the last update (avg %.2f), ratio %.3f, interval %u, owed "
            "%.2f ms, Link %.3f %.3f%s\n", framepace_target(), s * 1e3, smooth * 1e3, ratio, iv, (real - given) * 1e3,
            pinfo ? VF(float, pinfo, 0x88) : 0.0f, pinfo ? VF(float, pinfo, 0x90) : 0.0f,
            g_movie_playing > 0 ? ", movie" : "");
    }
}

// ------------------------------------------------------------------ test scripts, the clock probe, the screens
// SAVAGEWIND_SCRIPT_CLOCK=1: SAVAGEWIND_INPUT and SAVAGEWIND_SHOT_AT count 30ths of a second (the console's frames)
// instead of presented frames, so one script drives the game the same way at any frame rate: each presented frame
// adds its length x 30, at most 1.5 (a load's long frames count like the console's).
uint64_t framepace_script_frame(uint64_t presented) {
    static int on = -1;
    static double t;
    static uint64_t last;
    if (on < 0) on = env_int("SAVAGEWIND_SCRIPT_CLOCK", 0) != 0;
    if (!on) return presented;
    uint64_t now = plat_now_ns();
    double units = last ? (now - last) / 1e9 * 30 : 1;
    last = now;
    t += units > 1.5 ? 1.5 : units;
    {   // (debug) SAVAGEWIND_DUMP_DATA_AT=<frame>,...: main's .data and .bss -> build/data_<frame>.bin
        static const char *next;
        static int init;
        if (!init) init = 1, next = getenv("SAVAGEWIND_DUMP_DATA_AT");
        if (next && *next && (uint64_t)t >= strtoull(next, NULL, 10)) {
            char path[64];
            snprintf(path, sizeof path, "build/data_%llu.bin", (unsigned long long)t);
            FILE *f = fopen(path, "wb");
            if (f) fwrite((void *)(uintptr_t)g_site.main_data, 1, g_site.main_end - g_site.main_data, f), fclose(f);
            next = strchr(next, ',');
            if (next) next++;
        }
    }
    return (uint64_t)t;
}

// SAVAGEWIND_CLOCK_PROBE=<ms>: every <ms> of real time, a "[clock]" line with what the frame rate must not change:
// seconds since the probe started, frames, the game's own time (its delta frames summed, 1/30 s each, slow motion
// included), the time of day (TimeMgr, 0..360 = 24 h: one in-game minute is 0.25), Link's position and stamina
// (1000 a wheel): ksys::world::Manager [world_mgr] (+0x5c0 count, +0x5c8 its managers; [0] = TimeMgr, the time at
// +0xb8); uking::act::PlayerInfo [player_info] (+0x88 Link's position, +0xa4 the stamina).
static void clock_probe(uint64_t vfr) {
    static int every = -1;
    static uint64_t t0, next, frames;
    static double game;
    if (every < 0) every = env_int("SAVAGEWIND_CLOCK_PROBE", 0);
    if (every <= 0) return;
    frames++;
    game += rt_vfr_core0_delta_frame();  // (core 0's delta frame of the frame that just ended)
    uint64_t now = plat_now_ns();
    if (!t0) t0 = now, next = now;
    if (now < next) return;
    next += (uint64_t)every * 1000000ull;
    if (next < now) next = now + (uint64_t)every * 1000000ull;
    float tod = -1, pos[3] = {0, 0, 0}, stamina = -1;
    uint64_t wm = *(uint64_t *)(uintptr_t)g_site.world_mgr;
    if (wm && VF(uint32_t, wm, 0x5c0)) {
        uint64_t mgrs = VF(uint64_t, wm, 0x5c8), tm = mgrs ? *(uint64_t *)(uintptr_t)mgrs : 0;
        if (tm) tod = VF(float, tm, 0xb8);
    }
    uint64_t pinfo = *(uint64_t *)(uintptr_t)g_site.player_info;
    if (pinfo) {
        for (int i = 0; i < 3; i++) pos[i] = VF(float, pinfo, 0x88 + 4 * i);
        stamina = VF(float, pinfo, 0xa4);
    }
    LOG("[clock] t %.3f frames %llu game %.2f tod %.4f pos %.3f %.3f %.3f stamina %.1f fps %d at %.3f\n",
        (now - t0) / 1e9, (unsigned long long)frames, game, tod, pos[0], pos[1], pos[2], stamina, framepace_target(),
        now / 1e9);
}

// (research) SAVAGEWIND_DUMP_PLAYER=<bytes>: each game update, the player actor's first <bytes> (PlayerInfo +0x60,
// zeldaret's mPlayerActor) -> build/player.bin: records of u64 ns, f32 ratio, u32 interval, u64 actor, then the bytes
static void player_dump(uint64_t vfr) {
    static int bytes = -1;
    static FILE *f;
    if (bytes < 0) {
        bytes = env_int("SAVAGEWIND_DUMP_PLAYER", 0);
        if (bytes > 0) f = fopen("build/player.bin", "wb");
    }
    {   // (research) SAVAGEWIND_WATCH_PLAYER=<offset hex>: from 30 s on, the instructions that write the player actor's 4
        // bytes at that offset (runtime/sampler.c), reported every 10 s
        static long off = -2;
        static uint64_t t0, last;
        if (off == -2) { const char *e = getenv("SAVAGEWIND_WATCH_PLAYER"); off = e ? strtol(e, NULL, 16) : -1; }
        uint64_t pinfo = *(uint64_t *)(uintptr_t)g_site.player_info, now = plat_now_ns();
        uint64_t actor = pinfo ? VF(uint64_t, pinfo, 0x60) : 0;
#ifdef _WIN32
        if (off >= 0 && actor >= 0x7000000000ull) {
            extern void sampler_watch_write(uint64_t addr), sampler_watch_report(void);
            if (!t0) t0 = now;
            if (now - t0 > (uint64_t)env_int("SAVAGEWIND_WATCH_AFTER_MS", 30000) * 1000000ull)
                sampler_watch_write(actor + (uint64_t)off);
            if (now - last > 10000000000ull) last = now, sampler_watch_report();
        }
#endif
    }
    {   // (research) SAVAGEWIND_LOG_VFR=<n>: every n updates, VFR's ratio, interval, core 0's delta frame, the four
        // per-core deltas (runtime/hle/vfr.c) and the cold timer's step since the last update (player actor +0x268c)
        static int every = -1;
        static uint64_t n;
        static float last_timer;
        if (every < 0) every = env_int("SAVAGEWIND_LOG_VFR", 0);
        uint64_t pinfo = *(uint64_t *)(uintptr_t)g_site.player_info;
        uint64_t actor = pinfo ? VF(uint64_t, pinfo, 0x60) : 0;
        float timer = actor >= 0x7000000000ull ? VF(float, actor, 0x268c) : 0;
        if (every > 0 && n++ % (uint64_t)every == 0) {
            float core[4];
            for (int i = 0; i < 4; i++) core[i] = rt_vfr_core_delta_frame(i);
            LOG("[vfr] ratio %.4f interval %u delta0 %.4f cores %.4f %.4f %.4f %.4f; cold timer %.4f step %.5f life %d y %.3f\n",
                VF(float, vfr, VFR_RATIO), VF(uint32_t, vfr, VFR_INTERVAL), rt_vfr_core0_delta_frame(), core[0], core[1],
                core[2], core[3], timer, timer - last_timer, actor >= 0x7000000000ull ? VF(int32_t, actor, 0x848) : -1,
                pinfo ? VF(float, pinfo, 0x8c) : 0);
        }
        last_timer = timer;
    }
    if (bytes <= 0 || !f) return;
    uint64_t pinfo = *(uint64_t *)(uintptr_t)g_site.player_info;
    uint64_t actor = pinfo ? VF(uint64_t, pinfo, 0x60) : 0;
    if (actor < 0x7000000000ull || actor + (uint64_t)bytes > 0x7320000000ull) return;
    uint64_t t = plat_now_ns();
    float ratio = VF(float, vfr, VFR_RATIO);
    uint32_t iv = VF(uint32_t, vfr, VFR_INTERVAL);
    fwrite(&t, 8, 1, f), fwrite(&ratio, 4, 1, f), fwrite(&iv, 4, 1, f), fwrite(&actor, 8, 1, f);
    fwrite((const void *)(uintptr_t)actor, 1, (size_t)bytes, f);
}

// SAVAGEWIND_LOG_SCREENS=1: the pause state and the game's UI screens that exist (their ids), each time they change
static void log_screens(void) {
    static int on = -1;
    static char last[1024];
    if (on < 0) on = env_int("SAVAGEWIND_LOG_SCREENS", 0);
    if (!on) return;
    uint64_t mgr = *(uint64_t *)(uintptr_t)g_site.screen_mgr;
    if (!mgr) return;
    uint32_t n = VF(uint32_t, mgr, 0x28);
    uint64_t arr = VF(uint64_t, mgr, 0x30);
    char cur[1024];
    int k = snprintf(cur, sizeof cur, " (pause state %u):", *(uint32_t *)(uintptr_t)PAUSE_STATE);
    for (uint32_t i = 0; i < n && i < 256 && arr && k < (int)sizeof cur - 8; i++)
        if (*(uint64_t *)(uintptr_t)(arr + 8 * i)) k += snprintf(cur + k, sizeof cur - k, " %x", i);
    if (strcmp(cur, last)) {
        LOG("[screens]%s\n", cur);
        strcpy(last, cur);
    }
}
