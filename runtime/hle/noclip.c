// (debugging) SAVAGEWIND_NOCLIP=1: a noclip camera for trying places out (noclip-debug.bat). F6 turns it on and off
// (=2: on from the moment Link is in the world).
// While it is on, Link flies where you steer him, through walls and the ground: your move keys (W/A/S/D by default)
// or a controller's left stick move him relative to the camera, Space / right trigger up, Ctrl / left trigger down,
// the mouse wheel sets the speed (15 m/s at first, x1.25 a notch), Shift / left stick click 4x it. The camera stays
// the game's (mouse or right stick); the game's own movement input (left stick, buttons) is held back while flying so
// Link doesn't run or jump on the spot.
//
// Each game update (rt_vfr_update, the main thread at the start of the frame, before the physics step) the game's own
// physics API places him: his main rigid body (actor +0x578 ActorPhysics, +0x80 the body; its motion +8, or +0x2a0
// with flag 0x10000 at +0x120; the motion's accessor +0xb8 - the layout f_7001531040 and f_70012aa00c read) gets
// MotionAccessor::setPosition (vtable +0x18, f_70012e1ed0: Havok's setPositionAndRotation, linked motions too) and
// setLinearVelocity (+0x48) to the velocity he flies at, so gravity never builds up. Streaming, the world and the
// camera follow as for any move. Turning it off drops him where he is (a long fall still hurts).
//
// Camera-relative directions come from the scene camera the renderer sees (runtime/nvn/vulkan/exec.c
// vk_scene_camera: the game's camera constants; words 0-11 are the view matrix, rows right / up / back).
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../cpu.h"
#include "../game_sites.h"
#include "../hle.h"
#include "../kernel.h"
#include "../platform.h"
#include "../rt.h"

#define VF(T, base, off) (*(T *)(uintptr_t)((base) + (off)))

int g_noclip_enabled = -1;      // SAVAGEWIND_NOCLIP (read once)
volatile int g_noclip_toggle;   // F6 pressed (runtime/hle/input.c)
volatile int g_noclip_active;   // flying now: input.c holds the game's movement back
extern int g_noclip_camera;     // (exec.c: keep the scene camera)
extern bool vk_scene_camera(float out[48]);
extern void input_noclip_axes(float *right, float *forward, float *up, int *fast);  // (input.c)

static float pos[3];
static uint64_t last_ns;
static float g_speed = 15.0f;  // m/s, the mouse wheel's: x1.25 a notch, 2 .. 1000

void noclip_wheel(float notches) {  // (input.c, the mouse wheel) faster up, slower down - while flying
    if (!g_noclip_active || notches == 0) return;
    g_speed *= powf(1.25f, notches);
    g_speed = g_speed < 2 ? 2 : g_speed > 1000 ? 1000 : g_speed;
    LOG("[noclip] speed %.0f m/s\n", g_speed);
}

bool noclip_enabled(void) {
    if (g_noclip_enabled < 0) {
        const char *e = getenv("SAVAGEWIND_NOCLIP");
        g_noclip_enabled = e ? atoi(e) : 0;
        if (g_noclip_enabled) {
            g_noclip_camera = 1;
            LOG("[noclip] F6 turns noclip on and off: move keys / left stick fly, Space / RT up, Ctrl / LT down, mouse "
                "wheel speed, Shift / L3 4x\n");
        }
    }
    return g_noclip_enabled;
}

// Link's main rigid body's motion accessor, or 0
static uint64_t player_accessor(uint64_t *actor_out) {
    uint64_t pinfo = g_site.player_info ? *(uint64_t *)(uintptr_t)g_site.player_info : 0;
    if (pinfo < GUEST_LO || pinfo >= GUEST_HI) return 0;
    uint64_t actor = VF(uint64_t, pinfo, 0x60);
    if (actor < GUEST_LO || actor >= GUEST_HI) return 0;
    uint64_t phys = VF(uint64_t, actor, 0x578);
    if (phys < GUEST_LO || phys >= GUEST_HI) return 0;
    uint64_t body = VF(uint64_t, phys, 0x80);
    if (body < GUEST_LO || body >= GUEST_HI) return 0;
    uint64_t motion = VF(uint64_t, body, (VF(uint32_t, body, 0x120) & 0x10000) ? 0x2a0 : 8);
    if (motion < GUEST_LO || motion >= GUEST_HI) return 0;
    uint64_t acc = VF(uint64_t, motion, 0xb8);
    if (acc < GUEST_LO || acc >= GUEST_HI) return 0;
    *actor_out = actor;
    return acc;
}

static void norm2(float v[3]) {  // (horizontal: y is up in BOTW)
    v[1] = 0;
    float l = sqrtf(v[0] * v[0] + v[2] * v[2]);
    if (l > 1e-4f) v[0] /= l, v[2] /= l;
    else v[0] = 0, v[2] = 1;
}

void rt_noclip_update(void) {
    if (!noclip_enabled()) return;
    uint64_t now = plat_now_ns();
    float dt = last_ns ? (float)((now - last_ns) / 1e9) : 0;
    last_ns = now;
    if (dt > 0.1f) dt = 0.1f;
    uint64_t actor = 0, acc = player_accessor(&actor);
    // Link in the world: his actor already exists at the title screen (flying there would hold the menu's buttons
    // back), so PlayerInfo's position set too - it stays 0 0 0 until he is placed
    uint64_t pinfo = g_site.player_info ? *(uint64_t *)(uintptr_t)g_site.player_info : 0;
    bool in_world = acc && pinfo >= GUEST_LO && (VF(float, pinfo, 0x88) != 0 || VF(float, pinfo, 0x90) != 0);
    {   // (SAVAGEWIND_NOCLIP=2: flying from the moment Link is in the world, 2 s after he is placed: the landing)
        static bool auto_done;
        static uint64_t placed;
        if (!in_world) placed = 0;
        else if (!placed) placed = now;
        if (g_noclip_enabled == 2 && !auto_done && placed && now - placed > 2000000000ull && !g_noclip_active)
            auto_done = true, g_noclip_toggle = 1;
    }
    if (g_noclip_toggle) {
        g_noclip_toggle = 0;
        if (!g_noclip_active && in_world) {
            for (int i = 0; i < 3; i++) pos[i] = VF(float, actor, 0x398 + 0xc + 0x10 * i);  // (his matrix's translation)
            g_noclip_active = 1;
            LOG("[noclip] on at %.1f %.1f %.1f\n", pos[0], pos[1], pos[2]);
        } else if (g_noclip_active) {
            g_noclip_active = 0;
            LOG("[noclip] off at %.1f %.1f %.1f\n", pos[0], pos[1], pos[2]);
        }
    }
    if (!g_noclip_active) return;
    if (!acc) { g_noclip_active = 0; LOG("[noclip] off: Link has no body now (loading?)\n"); return; }
    // camera-relative directions (the view matrix's rows are the camera's right, up and back in the world)
    float cam[48], right[3] = {1, 0, 0}, fwd[3] = {0, 0, 1};
    if (vk_scene_camera(cam)) {
        right[0] = cam[0], right[2] = cam[2], fwd[0] = -cam[8], fwd[2] = -cam[10];
        norm2(right), norm2(fwd);
    }
    float ax, ay, az;
    int fast;
    input_noclip_axes(&ax, &ay, &az, &fast);
    float speed = g_speed * (fast ? 4.0f : 1.0f);  // m/s (a run is ~5, a gallop ~13); the wheel sets g_speed
    float vel[3] = {(right[0] * ax + fwd[0] * ay) * speed, az * speed, (right[2] * ax + fwd[2] * ay) * speed};
    for (int i = 0; i < 3; i++) pos[i] += vel[i] * dt;
    // the game's own calls, on this thread's guest context; every register and the stack come back as they were.
    // He is put where he flies to AND given the velocity he flies at: the physics step carries him on smoothly, and
    // the camera, which follows his motion, sees a body moving as it does instead of one standing still and jumping
    // (it trailed behind and Link wandered over the screen)
    Cpu *c = kernel_current_cpu();
    if (!c) return;
    static Cpu saved;
    saved = *c;
    uint64_t sp = (c->sp - 0x400) & ~15ull, vec = sp + 0x200;  // (scratch below the guest's frame; calls go below)
    uint64_t vt = VF(uint64_t, acc, 0);
    memcpy((void *)(uintptr_t)vec, pos, sizeof pos);
    c->sp = sp, c->x[0] = acc, c->x[1] = vec, c->x[2] = 1;  // MotionAccessor::setPosition(pos, propagate)
    guest_call(c, VF(uint64_t, vt, 0x18));
    static int zero_vel = -1;  // (SAVAGEWIND_NOCLIP_ZEROVEL=1: velocity 0, as the first version - to compare)
    if (zero_vel < 0) zero_vel = getenv("SAVAGEWIND_NOCLIP_ZEROVEL") != NULL;
    if (zero_vel) memset(vel, 0, sizeof vel);
    memcpy((void *)(uintptr_t)vec, vel, sizeof vel);
    c->sp = sp, c->x[0] = acc, c->x[1] = vec;  // MotionAccessor::setLinearVelocity(vel, epsilon 0)
    memset(&c->v[0], 0, sizeof c->v[0]);
    guest_call(c, VF(uint64_t, vt, 0x48));
    *c = saved;
    static uint64_t told;
    if (now - told > 2000000000ull) {
        told = now;
        LOG("[noclip] at %.1f %.1f %.1f, %.0f m/s (camera forward %.2f %.2f)\n", pos[0], pos[1], pos[2], g_speed, fwd[0],
            fwd[2]);
    }
}
