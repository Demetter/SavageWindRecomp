"""Per-instruction replacements applied by gen.py to the main module (address -> function(gen, addr) returning C).

Each entry replaces the single guest instruction at that address with hand-written C, for example
to redirect a hard-coded aspect-ratio load to the runtime. The addresses differ between game versions:
tools/game_sites.py finds them (and the functions and data the C uses) for the version being generated,
and table() maps them to these functions; gen.sites holds them all. Every patch still checks the
instruction word it expects before replacing it (assert on the decoded word) and fails loudly otherwise.
The 1.8.2 addresses in the comments are for reading along in a disassembly.

The Link's Awakening patches this file held in LASRecomp are kept in the archived copy
(reference/LASRecomp/tools/patches.py) as more examples of the pattern.
"""
import struct

from game_sites import adrp_page, sext


def _word(gen, a):
    return struct.unpack_from('<I', gen.nso.image, a - gen.base)[0]


def _fn(gen, site):  # the generated function at a site's address
    return f'f_{gen.sites[site]:x}'


# Savage Wind's own pages in the game's Options screen (runtime/ui/game_options.mm), opened from the title screen menu
# (screen 1, buttons 0x69..0x6f, FUN_7100d4c5e0) or the pause menu's System page (buttons 0x4d..0x53, FUN_7100cbc918).
# The screen builds one row per entry of a static option table every time it opens (the table is read through the
# pointers at opt_table, 1.8.2 0x7001cccd98..0x7001cccdb0), and every row goes through a handful of functions. The
# runtime points the table at a longer copy with its own entries (index >= the game's 14) and these hooks hand its rows
# to the runtime: values, texts and what A does. A page change closes the screen and opens it again with another row
# list, from whichever menu opened it. Quitting asks with the game's own two-button dialog (screen 0x39).

def _sub_sp(n):  # `sub sp, sp, #n`
    return 0xd10003ff | (n << 10)


def _quit_dialog(gen, ret, owner):
    # C: open the game's dialog (screen 0x39) as a two-button question (type 7, vtable +0x4e0) over the screen `owner`
    # (a C expression; its input is off while the question is up); the runtime gives it its texts (_dialog_texts) and
    # reads the answer
    req = _fn(gen, 'screen_open_request')
    return (f'{{ extern void {req}(Cpu *); extern uint64_t rt_quit_dialog_begin(uint64_t); '
            f'uint64_t own = {owner}; c->x[0] = 0x39; c->x[1] = 0; c->x[30] = {ret:#x}ull; {req}(c); '
            'uint64_t dlg = rt_quit_dialog_begin(own); '
            f'if (dlg) {{ c->x[0] = dlg; c->x[1] = 7; c->x[2] = 0; c->x[3] = 0; c->x[30] = {ret:#x}ull; '
            'c->pc = LD64(LD64(dlg) + 0x4e0); rt_dispatch(c); } }')


def _reopen_options(gen, ret):
    # C: open the Options screen as the menus' Options buttons do: open request (screen_open_request, 1.8.2
    # FUN_7100d6fd90, with 0x34), then vtable +0x28 (open: screen_open, FUN_71012d90e0) with 1
    req, op = _fn(gen, 'screen_open_request'), _fn(gen, 'screen_open')
    return (f'{{ extern void {req}(Cpu *); extern void {op}(Cpu *); extern uint64_t rt_opt_screen(void); '
            f'c->x[0] = 0x34; c->x[1] = 0; c->x[30] = {ret:#x}ull; {req}(c); uint64_t scr = rt_opt_screen(); '
            f'if (scr) {{ c->x[0] = scr; c->x[1] = 1; c->x[30] = {ret:#x}ull; {op}(c); }} }}')


def _opt_rows(gen, a):
    # createRows (Options vtable +0x2f0): the runtime installs the row list of the page about to be shown
    assert _word(gen, a) == _sub_sp(0xc0), hex(_word(gen, a))
    return ('{ extern void rt_opt_rows(void); rt_opt_rows(); c->sp -= 0xc0; }'
            f'  /* {a:#x}: sub sp, sp, #0xc0 (Options: build rows) */')


def _opt_flag_type(gen, a):
    # row init: the game data flag type of the entry (x1); 0 bool, 1 s32
    assert _word(gen, a) >> 26 == 0x25, hex(_word(gen, a))  # bl gdt_flag_type
    t, fn = gen.sites['gdt_flag_type'], _fn(gen, 'gdt_flag_type')
    return ('{ extern int rt_opt_flag_type(uint64_t); int t = rt_opt_flag_type(c->x[1]); '
            'if (t >= 0) c->x[0] = (uint32_t)t; '
            f'else {{ extern void {fn}(Cpu *); c->x[30] = {a + 4:#x}ull; {fn}(c); }} }}'
            f'  /* {a:#x}: bl {t:#x} (Options: flag type) */')


def _opt_read(gen, a):
    # row value read (x0 = row): fills +0x54 bool, +0x58 value, +0x5c max, +0x60 min, +0x64 button state
    assert _word(gen, a) == 0xa9bd7bfd, hex(_word(gen, a))  # stp x29, x30, [sp, #-0x30]!
    return ('{ extern int rt_opt_read(uint64_t); if (rt_opt_read(c->x[0])) return; '
            'uint64_t ea = c->sp - 0x30; ST64(ea, c->x[29]); ST64(ea + 8, c->x[30]); c->sp = ea; }'
            f'  /* {a:#x}: stp x29, x30, [sp, #-0x30]! (Options: read value) */')


def _opt_store(gen, a):
    # row value store (x0 = row)
    assert _word(gen, a) & 0x9f00001f == 0x90000008, hex(_word(gen, a))  # adrp x8, <the option table's page>
    page = adrp_page(_word(gen, a), a)
    return (f'{{ extern int rt_opt_store(uint64_t); if (rt_opt_store(c->x[0])) return; c->x[8] = {page:#x}ull; }}'
            f'  /* {a:#x}: adrp x8, {page:#x} (Options: store value) */')


def _opt_reset(gen, a):
    # row reset to the default (x0 = row), from "reset to defaults" (X)
    assert _word(gen, a) == 0xb9406808, hex(_word(gen, a))  # ldr w8, [x0, #0x68]
    return ('{ extern int rt_opt_reset(uint64_t); if (rt_opt_reset(c->x[0])) return; c->x[8] = LD32(c->x[0] + 0x68); }'
            f'  /* {a:#x}: ldr w8, [x0, #0x68] (Options: reset value) */')


def _opt_msg(gen, a):
    # every option text: message w21 of LayoutMsg/OptionWindow_00 into the {char16_t *, u32 length} at x2
    assert _word(gen, a) >> 26 == 0x25, hex(_word(gen, a))  # bl msg_lookup
    t, fn = gen.sites['msg_lookup'], _fn(gen, 'msg_lookup')
    return ('{ extern int rt_opt_msg(uint32_t, uint64_t); if (rt_opt_msg((uint32_t)c->x[21], c->x[2])) c->x[0] = 0; '
            f'else {{ extern void {fn}(Cpu *); c->x[30] = {a + 4:#x}ull; {fn}(c); }} }}'
            f'  /* {a:#x}: bl {t:#x} (Options: message text) */')


def _opt_decide(gen, a):
    # A on a row (x0 = row). 1: close the screen (a page change; the menu that opened it opens it again), 2: done,
    # 3: ask whether to quit
    assert _word(gen, a) == _sub_sp(0x30), hex(_word(gen, a))
    close = _fn(gen, 'screen_close')
    return ('{ extern int rt_opt_decide(uint64_t); int r = rt_opt_decide(c->x[0]); '
            f'if (r == 1) {{ extern void {close}(Cpu *); c->x[0] = LD64(c->x[0]); c->x[1] = 0xffffffffull; '
            f'c->x[30] = {a + 4:#x}ull; {close}(c); return; }} '
            f'if (r == 3) {_quit_dialog(gen, a + 4, "LD64(c->x[0])")} '
            'if (r) return; c->sp -= 0x30; }'
            f'  /* {a:#x}: sub sp, sp, #0x30 (Options: decide) */')


def _opt_tick(gen, a):
    # Options update (x0 = screen): the runtime may ask for one row to be redrawn (a binding changed); 1: skip the
    # update (the quit question is up: the screen's own dialog handling would take its answer)
    assert _word(gen, a) == _sub_sp(0x40), hex(_word(gen, a))
    redraw = _fn(gen, 'opt_row_redraw')
    return ('{ extern uint64_t rt_opt_tick(uint64_t); uint64_t row = rt_opt_tick(c->x[0]); if (row == 1) return; '
            f'if (row) {{ extern void {redraw}(Cpu *); uint64_t s0 = c->x[0], s30 = c->x[30]; '
            f'c->x[0] = row; c->x[1] = 0; c->x[30] = {a + 4:#x}ull; {redraw}(c); c->x[0] = s0; c->x[30] = s30; }} '
            'c->sp -= 0x40; }'
            f'  /* {a:#x}: sub sp, sp, #0x40 (Options: update) */')


def _opt_back(gen, a):
    # B on the screen: it closes (after the save); on one of Savage Wind's pages it opens again on the first page
    assert _word(gen, a) >> 26 == 0x25, hex(_word(gen, a))  # bl screen_back
    t, fn = gen.sites['screen_back'], _fn(gen, 'screen_back')
    return (f'{{ extern void rt_opt_back(void); rt_opt_back(); extern void {fn}(Cpu *); '
            f'c->x[30] = {a + 4:#x}ull; {fn}(c); }}'
            f'  /* {a:#x}: bl {t:#x} (Options: back) */')


def _title_tick(gen, a):
    # The title screen menu's update (screen 1, x0 = menu). While its Options button is active ([title_options_state]
    # == 0x6a; 1.8.2 +0x363c) it waits for the Options screen (id 0x34) to go away; after a page change the runtime
    # opens it again first. The quit question's answer is read here too.
    assert _word(gen, a) == 0xa9bb7bfd, hex(_word(gen, a))  # stp x29, x30, [sp, #-0x50]!
    return ('{ extern int rt_opt_title_tick(uint64_t); '
            f'if (rt_opt_title_tick(c->x[0])) {{ uint64_t s0 = c->x[0], s30 = c->x[30]; {_reopen_options(gen, a + 4)} '
            'c->x[0] = s0; c->x[30] = s30; } '
            'uint64_t ea = c->sp - 0x50; ST64(ea, c->x[29]); ST64(ea + 8, c->x[30]); c->sp = ea; }'
            f'  /* {a:#x}: stp x29, x30, [sp, #-0x50]! (title menu: update) */')


def _title_event(gen, a):
    # The title screen menu's button handler (x0 = menu, x1 = button, its id at +0x44): the amiibo settings button
    # (0x6c; Savage Wind has no amiibo support) is Quit, which asks where to
    assert _word(gen, a) == _sub_sp(0x60), hex(_word(gen, a))
    return ('{ extern int rt_title_event(uint32_t); '
            f'if (rt_title_event(LD32(c->x[1] + 0x44))) {{ {_quit_dialog(gen, a + 4, "c->x[0]")} return; }} '
            'c->sp -= 0x60; }'
            f'  /* {a:#x}: sub sp, sp, #0x60 (title menu: button) */')


def _title_msg(gen, a):
    # Message lookup (x0 = message file, x1 = label, x2 = {char16_t *, u32 length}): the title menu's file
    # (LayoutMsg/Title_00, title_msg_file) gives "0003" (amiibo settings) as Quit
    assert _word(gen, a) == _sub_sp(0x40), hex(_word(gen, a))
    title = gen.sites['title_msg_file']
    return ('{ extern int rt_title_msg(uint64_t, uint64_t); '
            f'if (c->x[0] == {title:#x}ull && rt_title_msg(c->x[1], c->x[2])) {{ c->x[0] = 0; return; }} '
            'c->sp -= 0x40; }'
            f'  /* {a:#x}: sub sp, sp, #0x40 (message lookup) */')


def _pause_option_state(gen, a):
    # The pause menu System page's Options state update (x0 = page): it leaves the state once the screen reports
    # closed. During a page change it waits instead, then opens it again.
    assert _word(gen, a) == 0xa9be7bfd, hex(_word(gen, a))  # stp x29, x30, [sp, #-0x20]!
    return ('{ extern int rt_opt_pause_tick(void); int r = rt_opt_pause_tick(); '
            f'if (r == 1) {_reopen_options(gen, a + 4)} '
            'if (r) return; '
            'uint64_t ea = c->sp - 0x20; ST64(ea, c->x[29]); ST64(ea + 8, c->x[30]); c->sp = ea; }'
            f'  /* {a:#x}: stp x29, x30, [sp, #-0x20]! (pause System page: Options state) */')


def _dialog_texts(gen, a):
    # The dialog's two-button layout (x0 = dialog, x1 = message, x2 / x3 = left / right button, {char16_t *, u32}
    # each): the quit question's texts
    assert _word(gen, a) == _sub_sp(0x50), hex(_word(gen, a))
    return ('{ extern void rt_quit_dialog_texts(uint64_t, uint64_t, uint64_t); '
            'rt_quit_dialog_texts(c->x[1], c->x[2], c->x[3]); c->sp -= 0x50; }'
            f'  /* {a:#x}: sub sp, sp, #0x50 (dialog: two-button texts) */')


def _dialog_back(gen, a):
    # B on the dialog presses its cancel button (one of the two); the quit question just closes instead (no answer)
    w = _word(gen, a)
    assert w & 0xffc003ff == 0xf9400100, hex(w)  # ldr x0, [x8, #off] (1.8.2 0x33b0, 1.9.0 0x33c8: the screen grew)
    off = ((w >> 10) & 0xfff) * 8
    return ('{ extern int rt_quit_dialog_back(void); '
            f'c->x[0] = rt_quit_dialog_back() ? 0 : LD64(c->x[8] + {off:#x}); }}'
            f'  /* {a:#x}: ldr x0, [x8, #{off:#x}] (dialog: B) */')


# Frame rate (runtime/hle/framepace.c): the runtime paces the frames at the present and sets the game's clock. The frame
# loop's vsync wait (sead framework, 1.8.2 0x7000d811fc, x0 = its display object; only the first frames after start-up
# come here) waits for the display's vsync, then for more vsyncs until interval / 60 s have passed: skipped (the game's
# own runs with SAVAGEWIND_FRAMEPACE=0). ksys::VFR::updateInterval (x19 = VFR) has just set the interval ratio the
# game's time step comes from (interval / 2): the runtime puts the frame's real length there, before the time
# multipliers and the deltas are worked out from it.

def _frame_wait(gen, a):
    assert _word(gen, a) == 0xa9bd7bfd, hex(_word(gen, a))  # stp x29, x30, [sp, #-0x30]!
    return ('{ extern int rt_frame_wait(void); if (rt_frame_wait()) return; '
            'uint64_t ea = c->sp - 0x30; ST64(ea, c->x[29]); ST64(ea + 8, c->x[30]); c->sp = ea; }'
            f'  /* {a:#x}: stp x29, x30, [sp, #-0x30]! (frame wait) */')


def _vfr_ratio(gen, a):
    assert _word(gen, a) == 0xb9407a68, hex(_word(gen, a))  # ldr w8, [x19, #0x78]
    return ('{ extern void rt_vfr_update(uint64_t); rt_vfr_update(c->x[19]); c->x[8] = LD32(c->x[19] + 0x78); }'
            f'  /* {a:#x}: ldr w8, [x19, #0x78] (VFR: interval ratio) */')


# The menus' speed (runtime/hle/framepace.c rt_gui_speed): once a frame the game stores its integer interval x 0.5 as
# the GUI speed (the float at +0xb20 of a UI object; 1.8.2 0x700116d0e8): 1 at 30 fps, 0.5 at 60 and never lower, so
# the menus' cursor and animations ran faster above 60 fps. The runtime gives the frame's real length instead, as Cemu's
# FPS++ does on the Wii U (v208 0x03415afc).

def _gui_speed(gen, a):
    assert _word(gen, a) == 0xbd0b2100, hex(_word(gen, a))  # str s0, [x8, #0xb20]
    return ('{ extern float rt_gui_speed(float); union { uint32_t u; float f; } s = { .u = (uint32_t)VS32(0) }; '
            's.f = rt_gui_speed(s.f); ST32(c->x[8] + 0xb20, s.u); }'
            f'  /* {a:#x}: str s0, [x8, #0xb20] (GUI speed) */')


# Perspective projections' matrix (sead's PerspectiveProjection, its vtable +0x58; 1.8.2 0x7000da13f4): it reads
# tan(fovy / 2) at +0xac and the aspect ratio at +0xb0 here. The runtime may change both before the matrix is built:
# a 16:9 view widened to the window's shape (ultrawide: more to the sides, not stretched) and the field of view
# (runtime/nvn/output_scale.c rt_persp).

def _persp_matrix(gen, a):
    assert _word(gen, a) == 0x2d558801, hex(_word(gen, a))  # ldp s1, s2, [x0, #0xac]
    return ('{ extern void rt_persp(uint64_t, uint32_t *, uint32_t *); uint32_t t = LD32(c->x[0] + 0xac), '
            'k = LD32(c->x[0] + 0xb0); rt_persp(c->x[0], &t, &k); c->v[1] = VV(t); c->v[2] = VV(k); }'
            f'  /* {a:#x}: ldp s1, s2, [x0, #0xac] (perspective matrix: tan(fovy / 2), aspect) */')


# Frustum projections' matrix (sead's FrustumProjection, its vtable +0x58; 1.8.2 0x7000da18d4): left / right at
# +0xa8 / +0xac and top / bottom at +0xa0 / +0xa4, each pair loaded twice. Some HUD layers are these (the menus'
# cursor): a 16:9 one is widened as the perspective ones are (runtime/nvn/output_scale.c rt_frustum).

def _frustum_load(word, regs, what):
    def patch(gen, a):
        assert _word(gen, a) == word, hex(_word(gen, a))
        r0, r1 = regs
        k = 0 if what == 'lr' else 2
        return ('{ extern void rt_frustum(uint64_t, float *); float f[4]; rt_frustum(c->x[0], f); '
                f'union {{ float f; uint32_t u; }} p0 = {{ f[{k}] }}, p1 = {{ f[{k + 1}] }}; '
                f'c->v[{r0}] = VV(p0.u); c->v[{r1}] = VV(p1.u); }}'
                f'  /* {a:#x}: ldp s{r0}, s{r1}, [x0, #{0xa8 if what == "lr" else 0xa0:#x}] (frustum matrix) */')
    return patch


# Projections' update (sead's Projection, 1.8.2 0x7000da0e18): every getter of the matrix, the device matrix and a
# point's projection makes the matrix again first when its dirty flag (+8) is set. It is taken as always set, so every
# matrix follows the window's shape and the field of view now (runtime/nvn/output_scale.c rt_persp): one made before
# the first frame, or before a change, stayed as it was (the menus' cursor is drawn by a HUD layer made at startup).

def _proj_update(gen, a):
    assert _word(gen, a) == 0x39402008, hex(_word(gen, a))  # ldrb w8, [x0, #8]
    return f'c->x[8] = 1;  /* {a:#x}: ldrb w8, [x0, #8] (projection update: dirty, always) */'


# ... and its load of the view's offset (+0xb4 x, +0xb8 y: a frustum moved by a fraction of its width and height; the
# inventory's Link is drawn to the right that way), which the widened view takes as a fraction of its 16:9 part
# (runtime/nvn/output_scale.c rt_persp_offset).

def _persp_matrix_offset(gen, a):
    assert _word(gen, a) == 0x2d569404, hex(_word(gen, a))  # ldp s4, s5, [x0, #0xb4]
    return ('{ extern void rt_persp_offset(uint64_t, uint32_t *, uint32_t *); uint32_t x = LD32(c->x[0] + 0xb4), '
            'y = LD32(c->x[0] + 0xb8); rt_persp_offset(c->x[0], &x, &y); c->v[4] = VV(x); c->v[5] = VV(y); }'
            f'  /* {a:#x}: ldp s4, s5, [x0, #0xb4] (perspective matrix: offset) */')


# Perspective projections' screen position to camera position (their vtable +0x68; 1.8.2 0x7000da14c0): the game
# places some things on screen with it (the menus' cursor corners), so it takes the same widened view as the matrix.

def _persp_unproject_y(gen, a):
    assert _word(gen, a) == 0xbd40ac01, hex(_word(gen, a))  # ldr s1, [x0, #0xac]
    return ('{ extern void rt_persp(uint64_t, uint32_t *, uint32_t *); uint32_t t = LD32(c->x[0] + 0xac), '
            'k = LD32(c->x[0] + 0xb0); rt_persp(c->x[0], &t, &k); c->v[1] = VV(t); }'
            f'  /* {a:#x}: ldr s1, [x0, #0xac] (screen to camera: tan(fovy / 2)) */')


def _persp_unproject_x(gen, a):
    assert _word(gen, a) == 0x2d558c02, hex(_word(gen, a))  # ldp s2, s3, [x0, #0xac]
    return ('{ extern void rt_persp(uint64_t, uint32_t *, uint32_t *); uint32_t t = LD32(c->x[0] + 0xac), '
            'k = LD32(c->x[0] + 0xb0); rt_persp(c->x[0], &t, &k); c->v[2] = VV(t); c->v[3] = VV(k); }'
            f'  /* {a:#x}: ldp s2, s3, [x0, #0xac] (screen to camera: tan(fovy / 2), aspect) */')


# The mouse camera (runtime/hle/input.c). Every camera action (CameraChase, CameraAiming, CameraHorse...) reads the
# right stick through one function (1.9.0 0x7000d57fe0), which applies a dead zone, squares the tilt (v |v|) and
# keeps it in the unit circle; the chase-like ones (CameraChase, CameraShieldSurfing, CameraWakeboard) square the
# vertical part once more (y |y|). The camera then turns by that x its speeds x the frame's length (VFR). Fine for a
# stick; from the mouse it crept on slow moves, sped up on fast ones and stopped at the stick's top speed. While the
# mouse has the camera the runtime answers the stick input itself (rt_cam_stick: the frame's mouse motion over the
# frame's length, linear, not clamped) and the vertical part goes through unsquared (rt_cam_pitch), so a frame's turn
# is the mouse's motion in it times the sensitivity, at any frame rate. The cameras turn the horizontal part 1.78
# times as fast as the vertical one for a stick, and the bow's (CameraAiming) and a held item's (CameraAiming2) scale
# them apart again (lngStickScale 0.7 and latStickScale 1; 0.8 and 1.5): the runtime gives the vertical part 1.78
# times the mouse's motion, and those two cameras' vertical part the horizontal part's scale (rt_cam_pitch_scale), so
# the mouse turns both alike. Optional sites (tools/game_sites.py OPTIONAL_GROUPS): without them the mouse moves the
# stick, as before.

def _cam_stick(gen, a):
    assert _word(gen, a) == _sub_sp(0x30), hex(_word(gen, a))
    return ('{ extern int rt_cam_stick(uint64_t, uint64_t); if (rt_cam_stick(c->x[0], c->x[30])) return; '
            'c->sp -= 0x30; }'
            f'  /* {a:#x}: sub sp, sp, #0x30 (camera stick input) */')


def _cam_pitch(gen, a):
    # fmul s1, s0, s0 just above, fnmul s3, s0, s0 here: the vertical part kept its sign squared, s1 or s3 by s0's sign
    assert _word(gen, a) == 0x1e208803 and _word(gen, a - 4) == 0x1e200801, hex(_word(gen, a))
    return ('{ extern int rt_cam_pitch(float *); union { uint32_t u; float f; } y = { .u = (uint32_t)VS32(0) }; '
            'if (rt_cam_pitch(&y.f)) c->v[1] = VV(y.u); else y.f = -(y.f * y.f); c->v[3] = VV(y.u); }'
            f'  /* {a:#x}: fnmul s3, s0, s0 (camera: the vertical stick part squared) */')


def _cam_pitch_aim(gen, a):
    # ldr s1, [x8]: the camera's latStickScale, through the parameter pointer loaded just above (ldr x8, [x19, #off]);
    # its lngStickScale's pointer is the next one (the parameters load in that order)
    assert _word(gen, a) == 0xbd400101, hex(_word(gen, a))
    w = next(_word(gen, a - 4 * i) for i in range(1, 5) if _word(gen, a - 4 * i) & 0xffc003ff == 0xf9400268)
    lng = ((w >> 10) & 0xfff) * 8 + 8
    return ('{ extern uint32_t rt_cam_pitch_scale(uint32_t, uint32_t); '
            f'c->v[1] = VV(rt_cam_pitch_scale(LD32(c->x[8]), LD32(LD64(c->x[19] + {lng:#x})))); }}'
            f'  /* {a:#x}: ldr s1, [x8] (camera: latStickScale) */')


# The game's Camera X-axis / Y-axis options (Inverted) are applied by the cameras' controller (the one the camera
# input reads, id 3 of the controller manager): every frame it reads the flags (CameraRLReverse, CameraUpDownReverse)
# and turns that axis of its right stick around. The mouse's camera input doesn't go through it, so the runtime is
# told the flags here and turns the mouse's motion around the same way (rt_cam_reverse).

def _cam_reverse(axis):
    def patch(gen, a):
        w = _word(gen, a)
        assert w >> 26 == 0x25, hex(w)  # bl <the flag's getter>
        t = a + sext(w & 0x3FFFFFF, 26) * 4
        return (f'{{ extern void f_{t:x}(Cpu *); extern void rt_cam_reverse(int, int); c->x[30] = {a + 4:#x}ull; '
                f'f_{t:x}(c); rt_cam_reverse({axis}, (int)(c->x[0] & 1)); }}'
                f'  /* {a:#x}: bl {t:#x} (camera controller: {"XY"[axis]}-axis inverted) */')
    return patch


# tools/game_sites.py site -> its patch
# BOTW's streaming Yaz0 decoder (1.9.0 0x700131d674; x0 its state, x1/w2 the input chunk): in native code
# (runtime/hle/yaz0.c rt_yaz0, which declines the first call of a stream, the one that reads its header). Its return
# instruction checks the native result against the game's code with SAVAGEWIND_YAZ0=2.
def _yaz0_stream(gen, a):
    assert _word(gen, a) == 0xa9bd7bfd, hex(_word(gen, a))  # stp x29, x30, [sp, #-0x30]!
    return ('{ extern int rt_yaz0(Cpu *); if (rt_yaz0(c)) return; '
            'uint64_t ea = c->sp - 0x30; ST64(ea, c->x[29]); ST64(ea + 8, c->x[30]); c->sp = ea; }'
            f'  /* {a:#x}: stp x29, x30, [sp, #-0x30]! (Yaz0 stream decoder) */')


def _yaz0_stream_ret(gen, a):
    assert _word(gen, a) == 0xd65f03c0, hex(_word(gen, a))  # ret
    return '{ extern void rt_yaz0_done(Cpu *); rt_yaz0_done(c); return; }' + f'  /* {a:#x}: ret (Yaz0 stream decoder) */'


# ksys::VFR per core with N slots (runtime/hle/vfr.c; BOTW-JOBS-NATIVE.md): its getters and per-core methods run in
# the runtime, which keeps a slot for every core the game has (sead's object has three); where the game's own code walks
# all three cores (copying buffer A to B, spreading core 0 to every core) a call does it for every core first. Each
# method is given sead's core-id TLS slot variable (core_id_tls), so it picks its core as the game's getters do.

def _vfr_method(fn, word, what):
    def patch(gen, a):
        assert _word(gen, a) == word, hex(_word(gen, a))
        tls = gen.sites['core_id_tls']
        return (f'{{ extern void {fn}(Cpu *, uint64_t); {fn}(c, {tls:#x}ull); return; }}'
                f'  /* {a:#x}: VFR {what} (runtime/hle/vfr.c) */')
    return patch


def _vfr_update_interval(gen, a):
    assert _word(gen, a) == 0xa9be7bfd, hex(_word(gen, a))  # stp x29, x30, [sp, #-0x20]!
    return ('{ extern void rt_vfr_copy_a_to_b(Cpu *); rt_vfr_copy_a_to_b(c); '
            'uint64_t ea = c->sp - 0x20; ST64(ea, c->x[29]); ST64(ea + 8, c->x[30]); c->sp = ea; }'
            f'  /* {a:#x}: stp x29, x30, [sp, #-0x20]! (VFR updateInterval: buffer A to B, every core) */')


def _vfr_spread(fn, what):
    def patch(gen, a):
        assert _word(gen, a) == 0xbd409a60, hex(_word(gen, a))  # ldr s0, [x19, #0x98]
        return (f'{{ extern void {fn}(Cpu *); {fn}(c); c->v[0] = VV(LD32(c->x[19] + 0x98)); }}'
                f'  /* {a:#x}: ldr s0, [x19, #0x98] (VFR {what}: core 0 to every core) */')
    return patch


# The command contexts (BOTW-JOBS-NATIVE.md #8; runtime/hle/percore.c): the game's four 0x288-byte contexts at +0x30
# of their singleton (three game cores and the system core's threads) live in a runtime block with a slot for every
# game core and the system threads' last. Every place that computes a context's address takes it from the block, the
# constructor builds every slot, and the walks over three or four contexts visit every one.
_CTX_DECL = ('extern uint64_t rt_percore_block(uint64_t, uint32_t, uint32_t, uint32_t, int); '
             'extern int rt_game_cores(void); ')


def _ctx(obj):
    return f'rt_percore_block({obj}, 0x30, 0x288, 4, 1)'


def _ctx_patch(word, body, what, structure='per-core contexts'):
    def patch(gen, a):
        w = _word(gen, a)
        assert (w == word) if word is not None else (w >> 26 == 0x25), hex(w)  # (None: a bl)
        return '{ ' + _CTX_DECL + body(gen, a) + ' }' + f'  /* {a:#x}: {what} ({structure}) */'
    return patch


def _ctx_get(reg):  # add x1, x8, #0x30 after x8 = this (reg) + core * 0x288
    return _ctx_patch(0x9100c101, lambda gen, a: f'c->x[1] = {_ctx(f"c->x[{reg}]")} + (c->x[8] - c->x[{reg}]);',
                      'add x1, x8, #0x30: the current core\'s context')


_CTX_SYS = _ctx_patch(0x911f2001, lambda gen, a: f'c->x[1] = {_ctx("c->x[0]")} + 0x288ull * (uint64_t)rt_game_cores();',
                      'add x1, x0, #0x7c8: the system core\'s context')


def _ctx_check_all(gen, a):  # f_7000ed70a0 (1.9.0): one context, or all the game cores' (all must pass)
    fn = _fn(gen, 'ctx_check_one')
    return _ctx_patch(0xa9be7bfd, lambda gen, a: (
        f'extern void {fn}(Cpu *); uint64_t self = c->x[0]; if (c->x[1]) {{ {fn}(c); return; }} '
        f'uint64_t blk = {_ctx("self")}; int n = rt_game_cores(); '
        f'for (int i = 0; i < n; i++) {{ c->x[0] = self; c->x[1] = blk + 0x288ull * (uint64_t)i; {fn}(c); '
        f'if (i < n - 1 && !(c->x[0] & 1)) {{ c->x[0] = 0; return; }} }} return;'),
        'stp x29, x30, [sp, #-0x20]!: check the game cores\' contexts')(gen, a)


def _ctx_reset_all(gen, a):  # the first of four calls f_7000ed7278(this, 0..3): every context, the system's last
    fn = _fn(gen, 'ctx_reset_fn')
    return _ctx_patch(None, lambda gen, a: (
        f'extern void {fn}(Cpu *); int n = rt_game_cores() + 1; '
        f'for (int i = 0; i < n; i++) {{ c->x[0] = c->x[20]; c->x[1] = (uint64_t)i; c->x[30] = {a + 4:#x}ull; {fn}(c); }}'),
        'bl: reset every context')(gen, a)


_CTX_RESET_DONE = _ctx_patch(None, lambda gen, a: '(void)0;', 'bl: (done with every context at the first call)')


# The command memory arenas (BOTW-JOBS-NATIVE.md #10; runtime/hle/percore.c): the per-core records at +0x88 of the one
# arena object live in a runtime block with a slot for every game core. The setup (memory for every core), the
# per-frame reset and the destructor's frees run in the runtime over every record; the two accessors take the current
# core's record from the block.
_ARENA_FNS = ('arena_malloc', 'arena_pool_init', 'arena_pool_addr', 'arena_pool_cpu', 'arena_pool_offset',
              'arena_blocks_alloc', 'arena_buffers_alloc')


def _arena_setup(gen, a):
    assert _word(gen, a) == 0xd10343ff, hex(_word(gen, a))  # sub sp, sp, #0xd0
    fns = ', '.join(f'{gen.sites[s]:#x}ull' for s in _ARENA_FNS)
    return (f'{{ extern void rt_arena_setup(Cpu *, const uint64_t *); static const uint64_t fn[] = {{{fns}}}; '
            'rt_arena_setup(c, fn); return; }'
            f'  /* {a:#x}: sub sp, sp, #0xd0 (command arenas setup: every core\'s memory and record) */')


def _arena_reset(gen, a):
    assert _word(gen, a) == 0xf9408c08, hex(_word(gen, a))  # ldr x8, [x0, #0x118]
    return ('{ extern void rt_arena_reset(Cpu *); rt_arena_reset(c); return; }'
            f'  /* {a:#x}: ldr x8, [x0, #0x118] (command arenas frame reset: every core\'s record) */')


def _arena_free(word, first):  # the destructor's three `ldr x8, [obj, #blocks]`: the first frees every core's
    def patch(gen, a):
        assert _word(gen, a) == word, hex(_word(gen, a))
        if not first:
            return f'{{ c->x[8] = 0; }}  /* {a:#x}: (every core\'s blocks freed at the first) */'
        fn = gen.sites['arena_free_fn']
        return ('{ extern void rt_arena_free_blocks(Cpu *, uint64_t, uint64_t); uint64_t self = c->x[0]; '
                f'rt_arena_free_blocks(c, self, {fn:#x}ull); c->x[0] = self; c->x[8] = 0; }}'
                f'  /* {a:#x}: ldr x8, [x0, #0x90] (command arenas: free every core\'s blocks) */')
    return patch


def _arena_rec(word, mov, dst, obj, core):  # smaddl x<dst>, w<core>, w<0x28>, x<obj>: the current core's record - 0x88
    def patch(gen, a):
        assert _word(gen, a) == word, hex(_word(gen, a))
        assert _word(gen, a + mov[0]) == mov[1], hex(_word(gen, a + mov[0]))  # (its `mov w, #0x28`)
        return ('{ extern uint64_t rt_percore_slot(uint64_t, uint32_t, uint32_t, uint32_t, int, int64_t); '
                f'c->x[{dst}] = rt_percore_slot(c->x[{obj}], 0x88, 0x28, 3, 0, (int32_t)c->x[{core}]) - 0x88; }}'
                f'  /* {a:#x}: smaddl x{dst}, w{core}, w, x{obj} (command arenas: the current core\'s record) */')
    return patch


# The per-core state (BOTW-JOBS-NATIVE.md #9): each core's choice of one of its singleton's 0x1e0-byte entries, three
# u32s at +0x1a68, lives in a runtime block with a slot for every game core. The accessors take the block for the
# game's array and bound the core by the game's core count (a core without a slot uses core 0's, as the game does past
# its three); the constructor zeroes every slot.
def _cstate(obj):
    return f'rt_percore_block({obj}, 0x1a68, 4, 3, 0)'


_cstate_ctor = _ctx_patch(0xb91a727f, lambda gen, a: (  # the first of the constructor's two stores: every core's
    f'uint64_t b = {_cstate("c->x[19]")}; for (int i = 0, n = rt_game_cores(); i < n; i++) ST32(b + 4ull * i, 0);'),
    'str wzr, [x19, #0x1a70]: zero every core\'s state', 'per-core state')
_cstate_ctor_done = _ctx_patch(0xf90d367f, lambda gen, a: '(void)0;',
                               'str xzr, [x19, #0x1a68]: (every core\'s zeroed at the first)', 'per-core state')


def _cstate_base(word, obj):  # add x8, x<obj>, x8 (x8 = 0x1a68): the game's array -> the block
    return _ctx_patch(word, lambda gen, a: f'c->x[8] = {_cstate(f"c->x[{obj}]")};',
                      f'add x8, x{obj}, x8: every core\'s state', 'per-core state')


_CSTATE_BOUND = _ctx_patch(0x71000d3f, lambda gen, a: '(void)(subs32(c, (uint32_t)c->x[9], (uint32_t)rt_game_cores()));',
                           'cmp w9, #3: a core with a slot', 'per-core state')


PATCHES = {
    'pause_options_update': _pause_option_state,
    'opt_create_rows': _opt_rows,
    'opt_row_init_flag_type': _opt_flag_type,
    'opt_read': _opt_read,
    'opt_store': _opt_store,
    'opt_reset': _opt_reset,
    'opt_msg_call': _opt_msg,
    'opt_decide': _opt_decide,
    'opt_update': _opt_tick,
    'opt_back_call': _opt_back,
    'dialog_texts': _dialog_texts,
    'dialog_back': _dialog_back,
    'title_update': _title_tick,
    'title_button': _title_event,
    'msg_lookup': _title_msg,
    'frame_wait': _frame_wait,
    'yaz0_stream': _yaz0_stream,
    'yaz0_stream_ret': _yaz0_stream_ret,
    'vfr_ratio': _vfr_ratio,
    'gui_speed': _gui_speed,
    'persp_matrix': _persp_matrix,
    'persp_matrix_offset': _persp_matrix_offset,
    'persp_unproject_y': _persp_unproject_y,
    'persp_unproject_x': _persp_unproject_x,
    'proj_update': _proj_update,
    'frustum_lr0': _frustum_load(0x2d550001, (1, 0), 'lr'),
    'frustum_lr1': _frustum_load(0x2d550c02, (2, 3), 'lr'),
    'frustum_tb0': _frustum_load(0x2d540800, (0, 2), 'tb'),
    'frustum_tb1': _frustum_load(0x2d540c02, (2, 3), 'tb'),
    'cam_stick': _cam_stick,
    'cam_pitch_chase': _cam_pitch,
    'cam_pitch_surf': _cam_pitch,
    'cam_pitch_wakeboard': _cam_pitch,
    'cam_pitch_aim': _cam_pitch_aim,
    'cam_pitch_item': _cam_pitch_aim,
    'cam_reverse_x': _cam_reverse(0),
    'cam_reverse_y': _cam_reverse(1),
    'vfr_get_delta_frame': _vfr_method('rt_vfr_delta_frame', 0xa9be7bfd, 'getDeltaFrame'),
    'vfr_get_raw_delta_frame': _vfr_method('rt_vfr_raw_delta_frame', 0xa9be7bfd, 'getRawDeltaFrame'),
    'vfr_get_delta_time': _vfr_method('rt_vfr_delta_time', 0xa9be7bfd, 'getDeltaTime'),
    'vfr_get_interval': _vfr_method('rt_vfr_interval', 0xa9be7bfd, 'getInterval'),
    'vfr_get_ratio': _vfr_method('rt_vfr_ratio', 0xa9be7bfd, 'getIntervalRatio'),
    'vfr_get_ratio_time': _vfr_method('rt_vfr_ratio_time', 0xa9be7bfd, 'getIntervalRatio x frame time'),
    'vfr_use_buffer_a': _vfr_method('rt_vfr_use_a', 0xa9be7bfd, 'useBufferA'),
    'vfr_use_buffer_b': _vfr_method('rt_vfr_use_b', 0xa9be7bfd, 'useBufferB'),
    'vfr_set_min_delta': _vfr_method('rt_vfr_set_min_delta', 0xb9407808, 'setMinDelta(core, mask)'),
    'vfr_get_and_set_min': _vfr_method('rt_vfr_get_and_set_min', 0xfc1c0fe8, 'getDeltaAndSetMin'),
    'vfr_set_delta': _vfr_method('rt_vfr_set_delta', 0xd100c3ff, 'setDelta'),
    'vfr_update_interval': _vfr_update_interval,
    'vfr_update_spread': _vfr_spread('rt_vfr_spread_core0', 'updateInterval'),
    'vfr_init_spread': _vfr_spread('rt_vfr_init_spread', 'init (and buffer A to B)'),
    'vfr_reset_spread': _vfr_spread('rt_vfr_spread_core0', 'resetTimeMultipliers'),
    'ctx_ctor_base': _ctx_patch(0x9100c278, lambda gen, a: f'c->x[24] = {_ctx("c->x[19]")};',
                                'add x24, x19, #0x30: the constructor\'s contexts'),
    'ctx_ctor_sys': _ctx_patch(0x71000f9f, lambda gen, a: '(void)(subs32(c, (uint32_t)c->x[28], (uint32_t)rt_game_cores()));',
                               'cmp w28, #3: the system core\'s context (no buffer)'),
    'ctx_ctor_count': _ctx_patch(0x7100139f,
                                 lambda gen, a: '(void)(subs32(c, (uint32_t)c->x[28], (uint32_t)rt_game_cores() + 1));',
                                 'cmp w28, #4: every context'),
    'ctx_set_all': _ctx_patch(0xb9403008, lambda gen, a: 'extern void rt_ctx_set_entries(Cpu *); rt_ctx_set_entries(c); return;',
                              'ldr w8, [x0, #0x30]: every context\'s entry'),
    'ctx_get_a': _ctx_get(25),
    'ctx_get_b': _ctx_get(25),
    'ctx_get_c': _ctx_get(25),
    'ctx_get_d': _ctx_get(23),
    'ctx_sys_a': _CTX_SYS,
    'ctx_sys_b': _CTX_SYS,
    'ctx_sys_c': _CTX_SYS,
    'ctx_sys_d': _CTX_SYS,
    'ctx_check_all': _ctx_check_all,
    'ctx_sys_count': _ctx_patch(0xb947ca6a,
                                lambda gen, a: f'c->x[10] = LD32({_ctx("c->x[19]")} + 0x288ull * (uint64_t)rt_game_cores());',
                                'ldr w10, [x19, #0x7c8]: the system core\'s context'),
    'ctx_sys_entries': _ctx_patch(0xf943ea6b,
                                  lambda gen, a: f'c->x[11] = LD64({_ctx("c->x[19]")} + 0x288ull * (uint64_t)rt_game_cores() + 8);',
                                  'ldr x11, [x19, #0x7d0]: the system core\'s context'),
    'ctx_reset_base': _ctx_patch(0x9b280037, lambda gen, a: (
        f'c->x[23] = {_ctx("c->x[0]")} - 0x30 + (uint64_t)((int64_t)(int32_t)c->x[1] * (int64_t)(int32_t)c->x[8]);'),
        'smaddl x23, w1, w8, x0: context w1'),
    'ctx_reset_0': _ctx_reset_all,
    'ctx_reset_1': _CTX_RESET_DONE,
    'ctx_reset_2': _CTX_RESET_DONE,
    'ctx_reset_3': _CTX_RESET_DONE,
    'ctx_size_base': _ctx_patch(0x9100c013, lambda gen, a: f'c->x[19] = {_ctx("c->x[0]")};', 'add x19, x0, #0x30: the contexts'),
    'ctx_flag_base': _ctx_patch(0x9b284d28, lambda gen, a: (
        f'c->x[8] = {_ctx("c->x[19]")} - 0x30 + (uint64_t)((int64_t)(int32_t)c->x[9] * (int64_t)(int32_t)c->x[8]);'),
        'smaddl x8, w9, w8, x19: the current core\'s context'),
    'cstate_ctor_a': _cstate_ctor,
    'cstate_ctor_b': _cstate_ctor_done,
    'cstate_set_base': _cstate_base(0x8b080288, 20),
    'cstate_set_bound': _CSTATE_BOUND,
    'cstate_get_base': _cstate_base(0x8b080268, 19),
    'cstate_get_bound': _CSTATE_BOUND,
    'cstate_slot_base': _cstate_base(0x8b080268, 19),
    'cstate_slot_bound': _CSTATE_BOUND,
    'arena_setup': _arena_setup,
    'arena_reset': _arena_reset,
    'arena_free_0': _arena_free(0xf9404808, True),
    'arena_free_1': _arena_free(0xf9405e68, False),
    'arena_free_2': _arena_free(0xf9407268, False),
    'arena_begin_rec': _arena_rec(0x9b285b28, (-0xc, 0x52800508), 8, 22, 25),
    'arena_end_rec': _arena_rec(0x9b294d13, (-0x4, 0x52800509), 19, 19, 8),
}


def table(sites):
    """address -> patch, for the main module whose sites these are (tools/game_sites.py; an optional site it lacks
    has no patch)"""
    return {sites[name]: fn for name, fn in PATCHES.items() if name in sites}
