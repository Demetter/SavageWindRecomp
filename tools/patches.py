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


# BOTW's job scheduler run natively (runtime/hle/jobs.c; BOTW-JOBS-NATIVE.md step 2): the entries of a job pool's push,
# kick and run-and-wait and of a worker's run go to the runtime, which takes them unless the pool is in the game's
# serial mode (or SAVAGEWIND_JOBS_NATIVE=0); then the game's code runs from its first instruction. Each gets the game
# functions the runtime calls: the group setup, the tick, the group event's clear and signal.
_JOBS_FNS = ('jobs_group_setup', 'jobs_tick', 'jobs_event_clear', 'jobs_event_signal')


def _jobs_entry(fn, word, frame, what):
    def patch(gen, a):
        assert _word(gen, a) == word, hex(_word(gen, a))
        fns = ', '.join(f'{gen.sites[s]:#x}ull' for s in _JOBS_FNS)
        first = (f'c->sp -= {frame:#x};' if word >> 24 == 0xd1 else  # sub sp, sp, #frame
                 f'uint64_t ea = c->sp - {frame:#x}; ST64(ea, c->x[29]); ST64(ea + 8, c->x[30]); c->sp = ea;')
        return (f'{{ static const uint64_t fns[4] = {{{fns}}}; extern int {fn}(Cpu *, const uint64_t *); '
                f'if ({fn}(c, fns)) return; {first} }}  /* {a:#x}: {what} */')
    return patch


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


# The job manager's per-core scratch (BOTW-JOBS-NATIVE.md #25; runtime/hle/percore.c): each core's 0x2800-byte scratch
# buffer (pointers at +0x458) and the most a job on it asked for (u32s at +0x448) live in runtime blocks with a slot
# for every game core. The constructor zeroes every slot at its first store of each array; the init makes every core's
# buffer at its first (the other two allocations and stores do nothing); the destructor frees every core's at its first
# load (the other two find nothing); the accessors take the blocks and bound the core by the game's core count.
def _jobs(obj, field):
    return f'rt_percore_block({obj}, {field:#x}, {4 if field == 0x448 else 8}, 3, 0)'


def _jobs_patch(word, body, what):
    return _ctx_patch(word, body, what, 'job manager scratch')


def _jobs_zero(word, field, what):  # the constructor's first store into an array: every core's slot
    st = 'ST32' if field == 0x448 else 'ST64'
    step = 4 if field == 0x448 else 8
    return _jobs_patch(word, lambda gen, a: (
        f'uint64_t b = {_jobs("c->x[19]", field)}; for (int i = 0, n = rt_game_cores(); i < n; i++) {st}(b + {step}ull * i, 0);'),
        what)


def _jobs_none(word, what):
    return _jobs_patch(word, lambda gen, a: '(void)0;', what)


def _jobs_alloc(gen, a):  # str x0, [x21, #0x458]: core 0's buffer, then every other core's
    return _jobs_patch(0xf9022ea0, lambda gen, a: (
        'extern void rt_jobs_scratch_alloc(Cpu *, uint64_t, uint64_t, uint64_t, uint64_t); '
        'rt_jobs_scratch_alloc(c, c->x[21], c->x[22], c->x[23], c->x[0]);'),
        'str x0, [x21, #0x458]: every core\'s scratch buffer')(gen, a)


def _jobs_free(gen, a):  # ldr x0, [x19, #0x458]: free every core's; x0 = 0 (nothing left for the game's frees)
    fn = gen.sites['jobs_free_fn']
    return _jobs_patch(0xf9422e60, lambda gen, a: (
        'extern void rt_jobs_scratch_free(Cpu *, uint64_t, uint64_t); '
        f'rt_jobs_scratch_free(c, c->x[19], {fn:#x}ull); c->x[0] = 0;'),
        'ldr x0, [x19, #0x458]: free every core\'s scratch buffer')(gen, a)


def _jobs_base(word, dst, obj, field, what):  # add x<dst>, x<obj>, #field: the game's array -> the block
    return _jobs_patch(word, lambda gen, a: f'c->x[{dst}] = {_jobs(f"c->x[{obj}]", field)};', what)


# #2 (BOTW-JOBS-NATIVE.md): a counter per core (u32s at +0x400 of a sub-object of the `*0x7001e0cd20` singleton; a job
# adds to its core's), summed and cleared once a frame (f_7000b1bfe0) and zeroed by the sub-object's init.
def _cnt(obj):
    return f'rt_percore_block({obj}, 0x400, 4, 3, 0)'


def _cnt_patch(word, body, what):
    return _ctx_patch(word, body, what, 'per-core counters')


_CNT_SUM = ('{ uint64_t b = ' + _cnt('c->x[19]') + '; uint32_t s = 0; '
            'for (int i = 0, n = rt_game_cores(); i < n; i++) s += LD32(b + 4ull * i); c->x[22] = s; }')
_CNT_ZERO = 'uint64_t b = {blk}; for (int i = 0, n = rt_game_cores(); i < n; i++) ST32(b + 4ull * i, 0);'


# #3 (BOTW-JOBS-NATIVE.md): a busy count per core in bytes at +0x1163 of the same singleton, the Switch's four: the
# three game cores', then the total - the per-frame step f_7000b32200 sums the game cores' into it, f_7000cd90a0 reads
# it, and the accessors' bound (< 4) lets a system-core thread (platform core 3) count into it too. The block has the
# game's cores and the total last.
def _busy(obj):
    return f'rt_percore_block({obj}, 0x1163, 1, 4, 1)'


def _busy_patch(word, body, what):
    return _ctx_patch(word, body, what, 'per-core busy counts')


_BUSY_BOUND = _busy_patch(0x7100111f, lambda gen, a: '(void)(subs32(c, (uint32_t)c->x[8], (uint32_t)rt_game_cores() + 1));',
                          'cmp w8, #4: a core with a slot (or the total)')
_BUSY_BASE = _busy_patch(0x8b080268, lambda gen, a: f'c->x[8] = {_busy("c->x[19]")} - 0x1163 + c->x[8];',
                         'add x8, x19, x8: the block (+ 0x1163 at the access)')


def _busy_zero(word, what):  # a 32-bit store of 0 over all four: every core's count and the total
    return _busy_patch(word, lambda gen, a: (
        f'uint64_t b = {_busy("c->x[19]")}; for (int i = 0, n = rt_game_cores() + 1; i < n; i++) ST8(b + i, 0);'), what)


# #7 (BOTW-JOBS-NATIVE.md; runtime/hle/percore.c): a 0xe8-byte object per core, pointers at +8 of their owner. The
# constructor runs in the runtime and makes one for every core; the destructor's first load frees every core's (the
# others find nothing, the game's own stores into its array do nothing); the accessor takes the block.
def _obj7_patch(word, body, what):
    return _ctx_patch(word, body, what, 'per-core 0xe8-byte objects')


def _obj7_ctor(gen, a):
    assert _word(gen, a) == 0xa9bd7bfd, hex(_word(gen, a))  # stp x29, x30, [sp, #-0x30]!
    fns = ', '.join(f'{gen.sites[s]:#x}ull' for s in ('obj7_vtable', 'obj7_heap', 'obj7_new', 'obj7_init'))
    return (f'{{ extern void rt_obj7_make(Cpu *, const uint64_t *); static const uint64_t fn[] = {{{fns}}}; '
            'rt_obj7_make(c, fn); return; }'
            f'  /* {a:#x}: stp x29, x30, [sp, #-0x30]! (per-core 0xe8-byte objects: one for every core) */')


def _obj7_free(gen, a):  # ldr x20, [x0, #8]: free every core's (x0 kept: the destructor goes on with it)
    fini, dele = gen.sites['obj7_fini'], gen.sites['obj7_delete']
    return _obj7_patch(0xf9400414, lambda gen, a: (
        'extern void rt_obj7_free(Cpu *, uint64_t, uint64_t, uint64_t); uint64_t self = c->x[0]; '
        f'rt_obj7_free(c, self, {fini:#x}ull, {dele:#x}ull); c->x[0] = self; c->x[20] = 0;'),
        'ldr x20, [x0, #8]: free every core\'s object')(gen, a)


# #20 (BOTW-JOBS-NATIVE.md): a per-core watchdog entry of the draw contexts' owner - the tick of the core's last change
# (+0xc3a8, 8 bytes), its state (+0xc3c0, 4) and a flag (+0xc3cc, 1) - indexed by the core, unbounded. Every accessor
# forms `obj + core x scale` and adds the field's offset from a register: that base becomes `block - offset + core x
# scale`. The constructor's three tick stores and the init's resets cover every core at their first store.
_WD = {'tick': (0xc3a8, 8), 'state': (0xc3c0, 4), 'flag': (0xc3cc, 1)}


def _wd(obj, field):
    off, size = _WD[field]
    return f'rt_percore_block({obj}, {off:#x}, {size}, 3, 0)'


def _wd_patch(word, body, what):
    return _ctx_patch(word, body, what, 'per-core watchdog')


def _wd_base(word, dst, obj, field, idx, what):  # add x<dst>, x<obj>, <idx> (core x scale): -> block - offset + idx
    off, size = _WD[field]
    return _wd_patch(word, lambda gen, a: f'c->x[{dst}] = {_wd(f"c->x[{obj}]", field)} - {off:#x}ull + ({idx});',
                     what + f': every core\'s {field}')


# #6 (BOTW-JOBS-NATIVE.md): a u32 per core (+0xe70) and a float per core (+0xe7c) of their owner; five accessors take
# the current core's (< 3), two per-frame sums add the first `sead core count` of them (unrolled up to three) and clear
# them, the constructor zeroes both. Blocks with a slot for every core; each sum over its count at its first load.
def _pair6(obj, field):
    return f'rt_percore_block({obj}, {field:#x}, 4, 3, 0)'


def _pair6_patch(word, body, what):
    return _ctx_patch(word, body, what, 'per-core pairs')


def _pair6_base(word, obj, field):
    return _pair6_patch(word, lambda gen, a: f'c->x[8] = {_pair6(f"c->x[{obj}]", field)};',
                        f'add x8, x{obj}, #{field:#x}: every core\'s')


_PAIR6_BOUND = _pair6_patch(0x71000d3f, lambda gen, a: '(void)(subs32(c, (uint32_t)c->x[9], (uint32_t)rt_game_cores()));',
                            'cmp w9, #3: a core with a slot')
def _PAIR6_NONE(word, what):
    return _pair6_patch(word, lambda gen, a: '(void)0;', what + ' at the first)')


_PAIR6_SUM_U = _pair6_patch(0xb94e7260, lambda gen, a: (  # w8: the sead core count
    f'uint64_t b = {_pair6("c->x[19]", 0xe70)}; uint32_t s = 0, n = (uint32_t)c->x[8], m = (uint32_t)rt_game_cores(); '
    'if (n > m) n = m; for (uint32_t i = 0; i < n; i++) { s += LD32(b + 4ull * i); ST32(b + 4ull * i, 0); } c->x[0] = s;'),
    'ldr w0, [x19, #0xe70]: the sum of every counted core\'s, cleared')
_PAIR6_SUM_F = _pair6_patch(0xbd4e7e60, lambda gen, a: (  # w8: the sead core count
    f'uint64_t b = {_pair6("c->x[19]", 0xe7c)}; union {{ float f; uint32_t u; }} s = {{ 0.0f }}, v; '
    'uint32_t n = (uint32_t)c->x[8], m = (uint32_t)rt_game_cores(); if (n > m) n = m; '
    'for (uint32_t i = 0; i < n; i++) { v.u = LD32(b + 4ull * i); s.f = i ? s.f + v.f : v.f; ST32(b + 4ull * i, 0); } '
    'c->v[0] = VV(s.u);'), 'ldr s0, [x19, #0xe7c]: the sum of every counted core\'s, cleared')
_PAIR6_ZERO = _pair6_patch(0xf907427f, lambda gen, a: (
    f'uint64_t a = {_pair6("c->x[19]", 0xe70)}, b = {_pair6("c->x[19]", 0xe7c)}; '
    'for (int i = 0, n = rt_game_cores(); i < n; i++) { ST32(a + 4ull * i, 0); ST32(b + 4ull * i, 0); }'),
    'str xzr, [x19, #0xe80]: zero every core\'s pair')


# #5 (BOTW-JOBS-NATIVE.md; runtime/hle/percore.c): per-slot results (slot = sead's core table entry of the core, < 3):
# 0x40-byte records at +0x1138, u32s at +0x1238, floats at +0x6f0, in blocks with a slot for every core. The accessors
# take the blocks and bound the slot by the game's core count; the merge (first valid record) and the float pick
# (largest magnitude) run in the runtime over every slot; the floats' two zeroings zero every slot.
def _slot5_patch(word, body, what):
    return _ctx_patch(word, body, what, 'per-slot results')


def _slot5_base(word, dst, obj, field, stride):
    return _slot5_patch(word, lambda gen, a: f'c->x[{dst}] = rt_percore_block(c->x[{obj}], {field:#x}, {stride:#x}, 3, 0);',
                        f'add x{dst}, x{obj}, #{field:#x}: every slot')


_SLOT5_BOUND = _slot5_patch(0x71000d1f, lambda gen, a: '(void)(subs32(c, (uint32_t)c->x[8], (uint32_t)rt_game_cores()));',
                            'cmp w8, #3: a slot with a block entry')


def _slot5_fzero(word, obj, what):
    return _slot5_patch(word, lambda gen, a: (
        f'uint64_t b = rt_percore_block(c->x[{obj}], 0x6f0, 4, 3, 0); '
        'for (int i = 0, n = rt_game_cores(); i < n; i++) ST32(b + 4ull * i, 0);'), what)


def _slot5_whole(word, fn, what):
    def patch(gen, a):
        assert _word(gen, a) == word, hex(_word(gen, a))
        return f'{{ extern void {fn}(Cpu *); {fn}(c); return; }}  /* {a:#x}: per-slot results: {what} */'
    return patch


# #12 (BOTW-JOBS-NATIVE.md; runtime/hle/percore.c): the graphics singleton's 0xfd8-byte objects per core, pointers at
# +0x70. The core-indexed accessors (through the core map f_7000fba16c) take them from a block with a slot for every
# core; other code picks the first three by a fixed offset (roles), so the game's array stays and the block's first
# three mirror it. The init makes every core's at its first store; its second and third makes take the block's.
def _obj12_patch(word, body, what):
    return _ctx_patch(word, body, what, 'per-core graphics objects')


_OBJ12 = 'rt_percore_block_live(c->x[22], 0x70, 8, 3)'
_OBJ12_DECL = 'extern uint64_t rt_percore_block_live(uint64_t, uint32_t, uint32_t, uint32_t); '


def _obj12_first(gen, a):  # str x25, [x22, #0x70]: core 0's (x25, just made); then every other core's
    alloc, ctor = gen.sites['obj12_alloc_fn'], gen.sites['obj12_ctor_fn']
    return _obj12_patch(0xf9003ad9, lambda gen, a: (
        _OBJ12_DECL + 'extern void rt_obj12_make(Cpu *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t); '
        f'ST64(c->x[22] + 0x70, c->x[25]); rt_obj12_make(c, c->x[22], c->x[25], c->x[21], {alloc:#x}ull, {ctor:#x}ull);'),
        'str x25, [x22, #0x70]: every core\'s object')(gen, a)


def _obj12_take(k):  # bl alloc: the block's object k (made at the first)
    return _obj12_patch(None, lambda gen, a: _OBJ12_DECL + f'c->x[0] = LD64({_OBJ12} + {8 * k});',
                        f'bl: object {k} (made at the first)')


def _obj12_base(word, reg):  # add x8, x<reg>, w0, uxtw #3 (then [x8, #0x70]): the core's object from the block
    return _obj12_patch(word, lambda gen, a: (
        _OBJ12_DECL + f'c->x[8] = rt_percore_block_live(c->x[{reg}], 0x70, 8, 3) - 0x70 + ((uint64_t)(uint32_t)c->x[0] << 3);'),
        f'add x8, x{reg}, w0, uxtw #3: the core\'s object')


# #27 (BOTW-JOBS-NATIVE.md; runtime/hle/percore.c): per-slot queues of a few 0x3f0-byte objects (slot = sead's core
# table entry of the core, < 3): buffers (+0x128), locks (+0x328) and cursors (+0x2a0) in blocks with a slot for every
# core. The append and the walker take the blocks and bound the slot by the game's core count; the init makes every
# slot's buffer at its first; the constructor's end makes every slot's lock and zeroes the rest.
def _q27(obj, field, stride):
    return f'rt_percore_block({obj}, {field:#x}, {stride:#x}, 3, 0)'


def _q27_patch(word, body, what):
    return _ctx_patch(word, body, what, 'per-slot queues')


def _q27_base(word, dst, obj, field, stride, what):
    return _q27_patch(word, lambda gen, a: f'c->x[{dst}] = {_q27(f"c->x[{obj}]", field, stride)};', what)


def _q27_cur(word, dst, obj, idx):  # add x<dst>, x<obj>, x<idx>, lsl #2 (then +0x2a0): the slot's cursor
    return _q27_patch(word, lambda gen, a: f'c->x[{dst}] = {_q27(f"c->x[{obj}]", 0x2a0, 4)} - 0x2a0 + (c->x[{idx}] << 2);',
                      f'add x{dst}, x{obj}, x{idx}, lsl #2: the slot\'s cursor')


_Q27_BOUND = _q27_patch(0x71000f3f, lambda gen, a: '(void)(subs32(c, (uint32_t)c->x[25], (uint32_t)rt_game_cores()));',
                        'cmp w25, #3: a slot with a block entry')


def _q27_init_first(gen, a):  # str w21, [x20, #0x128]: slot 0's buffer (x0, just made), then every other slot's
    alloc, tag = gen.sites['q27_alloc_fn'], gen.sites['q27_alloc_tag']
    return _q27_patch(0xb9012a95, lambda gen, a: (
        'extern void rt_q27_bufs(Cpu *, uint64_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t); uint64_t x0 = c->x[0]; '
        f'rt_q27_bufs(c, c->x[20], (uint32_t)c->x[21], x0, c->x[19], {alloc:#x}ull, LD64({tag:#x}ull)); c->x[0] = x0;'),
        'str w21, [x20, #0x128]: every slot\'s buffer')(gen, a)


def _q27_ctor_tail(gen, a):  # b memset(+0x2a0, 0, 0x84): then every slot's lock, buffer and cursor
    mset, lctor = gen.sites['q27_memset_fn'], gen.sites['q27_lock_ctor']
    w = _word(gen, a)
    assert w >> 26 == 0x05, hex(w)  # (a tail branch)
    return ('{ extern void guest_call(Cpu *, uint64_t); extern void rt_q27_ctor(Cpu *, uint64_t, uint64_t); '
            f'uint64_t obj = c->x[0] - 0x2a0; guest_call(c, {mset:#x}ull); rt_q27_ctor(c, obj, {lctor:#x}ull); return; }}'
            f'  /* {a:#x}: b memset (per-slot queues: the constructor\'s end, every slot\'s lock) */')


# #23 (BOTW-JOBS-NATIVE.md; runtime/hle/percore.c): per-core 0x160-byte records at +0x2a0 of a singleton (four 0x58-byte
# sub-entries each) in a block with a slot for every core. The constructor's end moves them to the block; the init sets
# every sub-entry\'s state at its first store; the walk goes over every record; the accessors take the block and bound
# the core by the game's core count.
def _rec23_patch(word, body, what):
    return _ctx_patch(word, body, what, 'per-core 0x160-byte records')


_REC23 = 'rt_percore_block({obj}, 0x2a0, 0x160, 3, 0)'


def _rec23_bound(word, reg):
    return _rec23_patch(word, lambda gen, a: f'(void)(subs32(c, (uint32_t)c->x[{reg}], (uint32_t)rt_game_cores()));',
                        f'cmp w{reg}, #3: a core with a slot')


def _rec23_base(word, dst, obj):  # add x<dst>, x<obj>, #0x2a8: the records (+8) in the block
    return _rec23_patch(word, lambda gen, a: f'c->x[{dst}] = {_REC23.format(obj=f"c->x[{obj}]")} + 8;',
                        f'add x{dst}, x{obj}, #0x2a8: every core\'s record')


def _rec23_none(word, what):
    return _rec23_patch(word, lambda gen, a: '(void)0;', what + ': (every sub-entry at the first)')


# #4 (BOTW-JOBS-NATIVE.md; runtime/hle/percore.c): arrays indexed by the slot of sead's core table (< 3), in blocks with
# a slot for every core. #4a: the 0x70-byte candidate records at +0xe00 of the #5 object - the constructor resets every
# slot's, the offer takes the block and bounds the slot, the per-frame pick (the best of every slot's) is the
# runtime's. #4b: the candidate lists of the object f_70010d4924 (1.9.0) makes - eight per slot at +0x28, one per slot
# at +0xa28 and at +0xb18: the constructor's end makes every slot's, the add takes the blocks and bounds the slot, the
# functions over the slots (unrolled for three) are the runtime's, and the per-frame update's clearings, checks and
# scans go over every slot.
def _rec4_patch(word, body, what):
    return _ctx_patch(word, body, what, 'per-slot candidates')


def _rec4a_ctor(word, slot):  # add x0, x19, #(0xe00 + 0x70 * slot): the slot's record in the block
    return _rec4_patch(word, lambda gen, a: f'c->x[0] = rt_percore_block(c->x[19], 0xe00, 0x70, 3, 0) + {0x70 * slot:#x};',
                       f'add x0, x19, #{0xe00 + 0x70 * slot:#x}: slot {slot}\'s record')


def _rec4a_rest(fn_site, obj, at, down, what):  # bl <fn> (slot 2's): slot 2's and every further slot's
    def patch(gen, a):
        fn = gen.sites[fn_site]
        return _rec4_patch(None, lambda gen, a: (
            'extern void rt_rec4a_rest(Cpu *, uint64_t, uint64_t, uint32_t, int); '
            f'rt_rec4a_rest(c, {obj}, {fn:#x}ull, {at:#x}, {down});'), f'bl: {what}, slot 2\'s and every further slot\'s')(gen, a)
    return patch


_REC4A_PART = 'rt_percore_block(c->x[19] - 0x860, 0xe00, 0x70, 3, 0)'  # (from the part at +0x860: x19)


def _rec4a_part(word, dst, add, what):  # add x<dst>, x19, #(0x5a0 + ...): a slot's record from the part
    return _rec4_patch(word, lambda gen, a: f'c->x[{dst}] = {_REC4A_PART} + {add:#x};', what)


def _rec4a_pick(gen, a):  # f_7000c2d894 (1.9.0): the best of every slot's records copied, every slot's reset
    copy, reset = gen.sites['rec4a_copy_fn'], gen.sites['rec4a_reset_fn']
    return _rec4_patch(0xa9bd7bfd, lambda gen, a: (
        f'extern void rt_rec4a_pick(Cpu *, uint64_t, uint64_t); rt_rec4a_pick(c, {copy:#x}ull, {reset:#x}ull); return;'),
        'stp x29, x30, [sp, #-0x30]!: the pick over every slot')(gen, a)


def _l4b(obj, field, stride):
    return f'rt_percore_block({obj}, {field:#x}, {stride:#x}, 3, 0)'


def _rec4b_whole(word, fn, what, *fns):  # a function over the slots' lists: the runtime's (fns: sites it calls)
    def patch(gen, a):
        types, args = ''.join(', uint64_t' for _ in fns), ''.join(f', {gen.sites[s]:#x}ull' for s in fns)
        return _rec4_patch(word, lambda gen, a: f'extern void {fn}(Cpu *{types}); {fn}(c{args}); return;', what)(gen, a)
    return patch


_REC4B_BOUND = _rec4_patch(0x71000f1f, lambda gen, a: '(void)(subs32(c, (uint32_t)c->x[24], (uint32_t)rt_game_cores()));',
                           'cmp w24, #3: a slot with a block entry')


def _rec4b_none(off):  # str wzr, [x19, #off]: a slot's list count (all cleared at the first store)
    return _rec4_patch(0xb900027f | (off >> 2) << 10, lambda gen, a: '(void)0;',
                       f'str wzr, [x19, #{off:#x}]: (every slot\'s cleared at the first)')


def _rec4b_clear(rest):  # the per-frame update's clearing of the slots' lists: {offset from the first store: field}
    first = _rec4_patch(0xb9002a7f, lambda gen, a: 'extern void rt_rec4b_clear(uint64_t); rt_rec4b_clear(c->x[19]);',
                        'str wzr, [x19, #0x28]: every slot\'s lists cleared')
    return {0: first, **{o: _rec4b_none(off) for o, off in rest.items()}}


_REC4B_COUNTS = [0x28 + 0x50 * j for j in range(24)] + [0xa28, 0xa78, 0xac8]  # (the kinds' lists, then A's, by slot)


def _rec4b_any(step, loads):  # three loads of the slots' kind-1 counts, `step` apart: any slot's at the first, 0 after
    (w0, d0), (w1, d1), (w2, d2) = loads
    return {0: _rec4_patch(w0, lambda gen, a: (
                'extern int rt_rec4b_any1(uint64_t); 'f'c->x[{d0}] = (uint32_t)rt_rec4b_any1(c->x[19]);'),
                'ldr: any slot\'s kind-1 candidates'),
            step: _rec4_patch(w1, lambda gen, a: f'c->x[{d1}] = 0;', 'ldr: (every slot\'s at the first)'),
            2 * step: _rec4_patch(w2, lambda gen, a: f'c->x[{d2}] = 0;', 'ldr: (every slot\'s at the first)')}


_REC4B_ANY8 = ((0xb9407a68, 8), (0xb942fa68, 8), (0xb9457a68, 8))  # (ldr w8, [x19, #0x78 / #0x2f8 / #0x578])


def _rec4b_keep_a(gen, a):  # the loop over the merged kind-1 list: a candidate no slot's A list holds out (643c next)
    assert _word(gen, a + 0xdc) == 0xf9460668, hex(_word(gen, a + 0xdc))  # (ldr x8, [x19, #0xc08])
    return _rec4_patch(0xb947fa69, lambda gen, a: (
        'extern void rt_rec4b_keep_a(uint64_t, uint64_t); rt_rec4b_keep_a(c->x[19], c->x[21]); '
        f'c->x[21] = 0; c->x[22] = 1; c->x[8] = LD64(c->x[19] + 0xc08); goto L_{a + 0xe0:x};'),
        'ldr w9, [x19, #0x7f8]: the loop, against every slot\'s A list')(gen, a)


def _rec4b_other1(gen, a):  # the scans of the slots' kind-1 lists for a candidate other than x8
    assert _word(gen, a + 0x3c) == 0xb9400288 and _word(gen, a + 0x100) == 0xaa1f03e2  # (their two ends)
    return _rec4_patch(0x93407d6b, lambda gen, a: (
        f'extern int rt_rec4b_other1(uint64_t, uint64_t); if (rt_rec4b_other1(c->x[19], c->x[8])) goto L_{a + 0x3c:x}; '
        f'goto L_{a + 0x100:x};'), 'sxtw x11, w11: every slot\'s kind-1 list scanned')(gen, a)


# #4c (BOTW-JOBS-NATIVE.md; runtime/hle/percore.c): the rings of the 0x3098-byte object at +0xbf0 of its owner - two
# groups (at +8 and +0xc50) of a 0x418-byte ring per slot, in a block per group (keyed by the group). The constructor's
# end makes every slot's rings; the push takes the block and bounds the slot; the drain's pops (the first non-empty
# ring of a group) are the runtime's, and its checks of the rings' counts go over every slot.
def _ring4c(group):
    return f'rt_percore_block({group}, 0, 0x418, 3, 0)'


def _rec4c_pop(gen, a):  # bl f_7001150ed4 / f_7001150f50 (byte-identical: patched at their one call each)
    return _rec4_patch(None, lambda gen, a: 'extern void rt_rec4c_pop(Cpu *); rt_rec4c_pop(c);',
                       'bl: the first non-empty ring of every slot\'s')(gen, a)


def _rec4c_any(group_off, loads):  # three loads of a group's ring counts: {offset: (word)}; any slot's at the first
    (o0, w0), *rest = loads
    out = {o0: _rec4_patch(w0, lambda gen, a: (
        f'extern int rt_rec4c_any(uint64_t); c->x[8] = (uint32_t)rt_rec4c_any(c->x[19] + {group_off:#x});'),
        'ldr w8: any slot\'s ring not empty')}
    for o, w in rest:
        out[o] = _rec4_patch(w, lambda gen, a: 'c->x[8] = 0;', 'ldr w8: (every slot\'s at the first)')
    return out


_REC4C_G1 = ((0xb9443268, 0x430), (0xb9484a68, 0x848))  # (ldr w8, [x19, #0x430 / #0x848]: rings 1 and 2 of group 1)
_REC4C_G2 = ((0xb94c6268, 0xc60), (0xb9507a68, 0x1078), (0xb9549268, 0x1490))


# #19 (BOTW-JOBS-NATIVE.md; runtime/hle/percore.c): per-core rings - 0x28-byte objects (the count of cores at +8, the
# capacity per core at +0xc, entries at +0x10 laid out [cursor][core], a cursor per core at +0x18). The makers size the
# entries and set the count for every core; the cursors live in a block per ring (the makers and the reset zero them);
# the appends take the block and bound the core; ring A's walker loops over every core, and rings B's and C's walkers
# (unrolled for three) go on from their third loop over every further core - their second and third loops keep the
# core on the stack as the game wrote them.
def _ring19_patch(word, body, what):
    return _ctx_patch(word, body, what, 'per-core rings')


def _cur19(ring):
    return f'rt_percore_block({ring}, 0x18, 4, 3, 0)'


def _ring19_n(word, dst, per_core, add, what):  # mov <dst>, #(3 x per_core + add): the same for every core
    return _ring19_patch(word, lambda gen, a: f'c->x[{dst}] = (uint64_t)({per_core:#x}ll * rt_game_cores() + {add:#x});',
                         what)


def _ring19_zero(ring, gap):  # str wzr, [xR, #0x20] and, gap later, str xzr, [xR, #0x18]: every core's cursor (second)
    return {0: _ring19_patch(0xb900201f | ring << 5, lambda gen, a: '(void)0;', f'str wzr, [x{ring}, #0x20]: (zeroed next)'),
            gap: _ring19_patch(0xf9000c1f | ring << 5, lambda gen, a: (
                f'extern void rt_ring19_zero(uint64_t); rt_ring19_zero(c->x[{ring}]);'),
                f'str xzr, [x{ring}, #0x18]: every core\'s cursor zeroed')}


_RING19_ADD = {  # add x8, x0, #0x18; cmp w2, #3: the ring's block, the core's bound
    0: _ring19_patch(0x91006008, lambda gen, a: f'c->x[8] = {_cur19("c->x[0]")};', 'add x8, x0, #0x18: every core\'s cursor'),
    4: _ring19_patch(0x71000c5f, lambda gen, a: '(void)(subs32(c, (uint32_t)c->x[2], (uint32_t)rt_game_cores()));',
                     'cmp w2, #3: a core with a cursor')}


def _ring19_c_rest(gen, a):  # bl make (ring C's 24th entry): it and every further core's
    make = gen.sites['ring19_c_make_fn']
    return _ring19_patch(None, lambda gen, a: (
        f'extern void rt_ring19_c_rest(Cpu *, uint64_t, uint64_t); rt_ring19_c_rest(c, c->x[20], {make:#x}ull);'),
        'bl: the 24th entry and every further core\'s')(gen, a)


def _ring19_c0(word, dst, ring):  # ldr w<dst>, [x<ring>, #0x18]: core 0's cursor
    return _ring19_patch(word, lambda gen, a: f'c->x[{dst}] = LD32({_cur19(f"c->x[{ring}]")});',
                         f'ldr w{dst}, [x{ring}, #0x18]: core 0\'s cursor')


def _ring19_c1(word, dst, ring, var):  # ldr w<dst>, [x<ring>, #0x1c]: core 1's cursor, its core kept at [sp + var]
    return _ring19_patch(word, lambda gen, a: (
        f'ST32(c->sp + {var:#x}, 1); c->x[{dst}] = LD32({_cur19(f"c->x[{ring}]")} + 4);'),
        f'ldr w{dst}, [x{ring}, #0x1c]: core 1\'s cursor (its core kept)')


def _b_target(gen, a, cond):  # a conditional branch's target (asserting its condition)
    w = _word(gen, a)
    assert w & 0xff00001f == 0x54000000 | cond, hex(w)
    return a + ((((w >> 5) & 0x7ffff) ^ 0x40000) - 0x40000) * 4


def _ring19_next(word, dst, ring, var, out, lt_off):
    """The third loop's head (ldr w<dst>, [x<ring>, #0x20]) and its `b.lt` (lt_off later) past the walker's end (`out`
    after the head): the next core after the one at [sp + var] - none left, the end; a core without entries, the next."""
    def head(gen, a):
        return _ring19_patch(word, lambda gen, a: (
            f'{{ uint32_t k = LD32(c->sp + {var:#x}) + 1; if (k >= (uint32_t)rt_game_cores()) goto L_{a + out:x}; '
            f'ST32(c->sp + {var:#x}, k); c->x[{dst}] = LD32({_cur19(f"c->x[{ring}]")} + 4ull * k); }}'),
            f'ldr w{dst}, [x{ring}, #0x20]: the next core\'s cursor')(gen, a)
    def lt(gen, a):
        assert _b_target(gen, a, 0xb) == a - lt_off + out
        return (f'{{ if (COND_LT(c)) goto L_{a - lt_off:x}; }}'
                f'  /* {a:#x}: b.lt: a core without entries, then the next (per-core rings) */')
    return {0: head, lt_off: lt}


def _ring19_mov(var):  # mov w8, #2: the loop's core
    return _ring19_patch(0x52800048, lambda gen, a: f'c->x[8] = LD32(c->sp + {var:#x});', 'mov w8, #2: the loop\'s core')


def _ring19_ccmp(var):  # ccmp w8, #2, #0, hi: against the loop's core
    return _ring19_patch(0x7a428900, lambda gen, a: (
        f'if (COND_HI(c)) (void)(subs32(c, (uint32_t)c->x[8], LD32(c->sp + {var:#x}))); else c->nzcv = 0;'),
        'ccmp w8, #2, #0, hi: against the loop\'s core')


def _ring19_end(head_back, loop_back):  # b.ne <loop>: the core's last entry done, then the next core
    def patch(gen, a):
        assert _b_target(gen, a, 0x1) == a - loop_back
        return (f'{{ if (!COND_EQ(c)) goto L_{a - loop_back:x}; goto L_{a - head_back:x}; }}'
                f'  /* {a:#x}: b.ne: the core\'s last entry, then the next core (per-core rings) */')
    return patch

# #28's second table (BOTW-JOBS-NATIVE.md; runtime/hle/percore.c): the object f_70010b6b94 makes keeps six 16-byte
# interfaces (+0x140) and six 0x58-byte polymorphic records (+0x1a0), one per core; f_70010b7184 picks the current
# core's. Blocks with a slot for every core and the system core's threads': the constructor makes every slot's, the
# destructor destroys every slot's, the pick takes the block.
def _obj28_patch(word, body, what):
    return _ctx_patch(word, body, what, 'per-core records (#28)')


def _obj28_rest(word, what):
    return _obj28_patch(word, lambda gen, a: '(void)0;', what + ' (every slot\'s at the first)')


def _obj28_call(fn, site):  # bl <site's function> (the first): every slot's
    def patch(gen, a):
        f = gen.sites[site]
        return _obj28_patch(None, lambda gen, a: (
            f'extern void {fn}(Cpu *, uint64_t, uint64_t); {fn}(c, c->x[19], {f:#x}ull);'), 'bl: every slot\'s')(gen, a)
    return patch


def _obj28_bl_rest(gen, a):  # bl: (every slot's at the first)
    return _obj28_patch(None, lambda gen, a: '(void)0;', 'bl: (every slot\'s at the first)')(gen, a)


def _obj28_pick(gen, a):  # cmp w8, #5 (then the switch): the current core's record in the block, to the end
    assert _word(gen, a + 0x24) == 0xf9400bf3  # (ldr x19, [sp, #0x10]: the switch's end)
    return _obj28_patch(0x7100151f, lambda gen, a: (
        '{ uint64_t b = rt_percore_block(c->x[19], 0x1a0, 0x58, 6, 1); uint32_t k = (uint32_t)c->x[8]; '
        f'c->x[0] = b + 0x58ull * (k <= (uint32_t)rt_game_cores() ? k : 0); goto L_{a + 0x24:x}; }}'),
        'cmp w8, #5: the current core\'s record')(gen, a)

# #26 (BOTW-JOBS-NATIVE.md): a 0x58-byte string per core (sead FixedSafeString<64>: vtable, pointer to the inline
# buffer at +0x14, capacity 0x40 at +0x10) in a function-local static at 0x7001e96c20 of f_7001594a84, indexed by the
# core (unbounded) and made by its guarded init, unrolled for three. A block keyed by the static's address: the init
# makes the first three in it, then (at its last store) every other core's the same way; the accessors take the block.
def _str26(reg, lo):
    return f'rt_percore_block(c->x[{reg}] + {lo:#x}, 0, 0x58, 3, 0)'


def _str26_patch(word, body, what):
    return _ctx_patch(word, body, what, 'per-core strings')


def _str26_rest(gen, a):  # str x26, [x22]: string 2's final vtable; then every further core's string (x24 / x26 vtables)
    fn = _fn(gen, 'str26_init_fn')
    return _str26_patch(0xf90002da, lambda gen, a: (
        f'extern void {fn}(Cpu *); ST64(c->x[22], c->x[26]); uint64_t b = c->x[22] - 0xb0, x0 = c->x[0]; '
        'for (uint64_t i = 3, n = (uint64_t)rt_game_cores(); i < n; i++) { uint64_t r = b + i * 0x58; '
        'ST64(r, c->x[24]); ST64(r + 8, r + 0x14); ST32(r + 0x10, 0x40); '
        f'c->x[0] = r; c->x[30] = {a + 4:#x}ull; {fn}(c); ST8(LD64(r + 8), 0); ST64(r, c->x[26]); }} c->x[0] = x0;'),
        'str x26, [x22]: string 2 done, then every further core\'s')(gen, a)


def _wd_ticks(gen, a):  # str x0, [x23]: core 0's tick (x0, just read), then every other core's
    return _wd_patch(0xf90002e0, lambda gen, a: (
        f'extern void {_fn(gen, "wd_tick_fn")}(Cpu *); uint64_t b = {_wd("c->x[19]", "tick")}; ST64(b, c->x[0]); '
        f'for (int i = 1, n = rt_game_cores(); i < n; i++) {{ c->x[30] = {a + 4:#x}ull; {_fn(gen, "wd_tick_fn")}(c); '
        'ST64(b + 8ull * i, c->x[0]); }'), 'str x0, [x23]: every core\'s tick')(gen, a)


# #11 (BOTW-JOBS-NATIVE.md; runtime/hle/percore.c): the per-core linear allocators' records (0x58 bytes at +0x38 of
# their object) live in a block with a slot for every core. The constructor, destructor, init and per-frame split run
# in the runtime over every record; the accessors' `obj + core x 0x58` (the game's field offsets on top) take the block.
def _lin_whole(word, fn, sites, what):  # a function replaced at its first instruction
    def patch(gen, a):
        assert _word(gen, a) == word, hex(_word(gen, a))
        args = ''
        if sites:
            args = ', fn'
            fns = ', '.join(f'{gen.sites[s]:#x}ull' for s in sites)
            decl, table = f'extern void {fn}(Cpu *, const uint64_t *); ', f'static const uint64_t fn[] = {{{fns}}}; '
        else:
            decl, table = f'extern void {fn}(Cpu *); ', ''
        return '{ ' + decl + table + f'{fn}(c{args}); return; }}' + f'  /* {a:#x}: per-core linear allocators: {what} */'
    return patch


def _lin_rec(word, obj, idx, mul):  # smaddl x8, w<idx>, w<mul (0x58)>, x<obj>
    return _ctx_patch(word, lambda gen, a: (
        f'c->x[8] = rt_percore_block(c->x[{obj}], 0x38, 0x58, 3, 0) - 0x38 + '
        f'(uint64_t)((int64_t)(int32_t)c->x[{idx}] * (int64_t)(int32_t)c->x[{mul}]);'),
        f'smaddl x8, w{idx}, w{mul}, x{obj}: the core\'s record', 'per-core linear allocators')


# #24 (BOTW-JOBS-NATIVE.md): per-core pointer lists in the game's data (0x810 bytes each at 0x7001e93918: u32 count,
# u32 capacity 0x100, a pointer to their inline entries) and a byte per core (0x7001e95148). Blocks keyed by the
# globals' addresses (as the code forms them: page + offset) with a slot for every core; the static init makes every
# core's list, every accessor takes the block and bounds the core by the game's core count.
def _list24_patch(word, body, what):
    return _ctx_patch(word, body, what, 'per-core lists')


def _list24_base(word, reg, lo, stride, what):  # add x<reg>, x<reg>, #lo: the global -> its block
    return _list24_patch(word, lambda gen, a: f'c->x[{reg}] = rt_percore_block(c->x[{reg}] + {lo:#x}, 0, {stride:#x}, 3, 0);',
                         what)


def _list24_bound(word, reg):
    return _list24_patch(word, lambda gen, a: f'(void)(subs32(c, (uint32_t)c->x[{reg}], (uint32_t)rt_game_cores()));',
                         f'cmp w{reg}, #3: a core with a slot')


_LIST24_INIT = _list24_patch(0x91246129, lambda gen, a: (
    'uint64_t b = rt_percore_block(c->x[9] + 0x918, 0, 0x810, 3, 0); '
    'for (uint64_t i = 0, n = (uint64_t)rt_game_cores(); i < n; i++) { '
    'ST64(b + i * 0x810, 0x10000000000ull); ST64(b + i * 0x810 + 8, b + i * 0x810 + 0x10); } c->x[9] = b;'),
    'add x9, x9, #0x918: every core\'s list (empty, 0x100 entries); the game\'s stores fill the first three again')


# #22 (BOTW-JOBS-NATIVE.md): a u16 counter per core at +0xb8 of its owner (unbounded; f_700118b52c adds to the
# core's), zeroed by f_700118b58c: a block with a slot for every core.
def _cnt22(obj):
    return f'rt_percore_block({obj}, 0xb8, 2, 3, 0)'


def _rec17_new(gen, a):
    # The per-core 0xd50-byte records (BOTW-JOBS-NATIVE.md #17): their maker, `new T[3]` with T's constructor inline,
    # runs in the runtime and makes the game's core count of them (every user goes by the holder's count).
    assert _word(gen, a) == 0xa9bc7bfd, hex(_word(gen, a))  # stp x29, x30, [sp, #-0x40]!
    fns = ', '.join(f'{gen.sites[s]:#x}ull' for s in ('rec17_alloc', 'rec17_alloc_tag', 'rec17_init_a', 'rec17_init_b',
                                                      'rec17_init_c', 'rec17_init_d'))
    return (f'{{ extern void rt_rec17_new(Cpu *, const uint64_t *); static const uint64_t fn[] = {{{fns}}}; '
            'rt_rec17_new(c, fn); return; }'
            f'  /* {a:#x}: stp x29, x30, [sp, #-0x40]! (per-core 0xd50-byte records: one for every core) */')


def _jobs_bound(word, reg):  # cmp w<reg>, #3: the core has a slot?
    return _jobs_patch(word, lambda gen, a: f'(void)(subs32(c, (uint32_t)c->x[{reg}], (uint32_t)rt_game_cores()));',
                       f'cmp w{reg}, #3: a core with a slot')


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
    'jobs_push': _jobs_entry('rt_jobs_push', 0xa9bc7bfd, 0x40, 'stp x29, x30, [sp, #-0x40]! (job pool push)'),
    'jobs_kick': _jobs_entry('rt_jobs_kick', 0xd10103ff, 0x40, 'sub sp, sp, #0x40 (job pool kick)'),
    'jobs_run': _jobs_entry('rt_jobs_run', 0xa9ba7bfd, 0x60, 'stp x29, x30, [sp, #-0x60]! (job worker run)'),
    'jobs_run_wait': _jobs_entry('rt_jobs_run_wait', 0xa9bc7bfd, 0x40, 'stp x29, x30, [sp, #-0x40]! (job pool run-and-wait)'),
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
    'jobs_ctor_marks': _jobs_zero(0xb904527f, 0x448, 'str wzr, [x19, #0x450]: zero every core\'s mark'),
    'jobs_ctor_marks_b': _jobs_none(0xf902267f, 'str xzr, [x19, #0x448]: (every core\'s zeroed)'),
    'jobs_ctor_bufs': _jobs_zero(0xf9022e7f, 0x458, 'str xzr, [x19, #0x458]: zero every core\'s buffer pointer'),
    'jobs_ctor_bufs_b': _jobs_none(0xf902327f, 'str xzr, [x19, #0x460]: (every core\'s zeroed)'),
    'jobs_ctor_bufs_c': _jobs_none(0xf902367f, 'str xzr, [x19, #0x468]: (every core\'s zeroed)'),
    'jobs_alloc_0': _jobs_alloc,
    'jobs_alloc_1': _jobs_none(0xd63f0100, 'blr x8: (every core\'s buffer made at the first)'),
    'jobs_alloc_1s': _jobs_none(0xf90232a0, 'str x0, [x21, #0x460]: (every core\'s buffer made at the first)'),
    'jobs_alloc_2': _jobs_none(0xd63f0100, 'blr x8: (every core\'s buffer made at the first)'),
    'jobs_alloc_2s': _jobs_none(0xf90236a0, 'str x0, [x21, #0x468]: (every core\'s buffer made at the first)'),
    'jobs_free_0': _jobs_free,
    'jobs_free_1': _jobs_patch(0xf9423260, lambda gen, a: 'c->x[0] = 0;', 'ldr x0, [x19, #0x460]: (freed at the first)'),
    'jobs_free_2': _jobs_patch(0xf9423660, lambda gen, a: 'c->x[0] = 0;', 'ldr x0, [x19, #0x468]: (freed at the first)'),
    'jobs_mark_base': _jobs_base(0x91112288, 8, 20, 0x448, 'add x8, x20, #0x448: every core\'s mark'),
    'jobs_mark_bound': _jobs_bound(0x71000d3f, 9),
    'jobs_buf_base': _jobs_base(0x91116013, 19, 0, 0x458, 'add x19, x0, #0x458: every core\'s buffer'),
    'jobs_buf_bound': _jobs_bound(0x71000d1f, 8),
    'rec17_new': _rec17_new,
    # the per-core draw contexts (BOTW-JOBS-NATIVE.md #16 / #18): their object (f_7000063ad4) sizes its per-core arrays
    # by the count in the descriptor f_700105bf6c builds, a literal 3 - the game's core count instead
    'drawctx_count': _ctx_patch(0x52800068, lambda gen, a: 'c->x[8] = (uint32_t)rt_game_cores();',
                                'mov w8, #3: a draw context for every core', 'per-core draw contexts'),
    'cnt_add_base': _cnt_patch(0x91100016, lambda gen, a: f'c->x[22] = {_cnt("c->x[0]")};',
                               'add x22, x0, #0x400: every core\'s counter'),
    'cnt_add_bound': _cnt_patch(0x71000d1f, lambda gen, a: '(void)(subs32(c, (uint32_t)c->x[8], (uint32_t)rt_game_cores()));',
                                'cmp w8, #3: a core with a slot'),
    'cnt_sum_a': _cnt_patch(0xb9440276, lambda gen, a: _CNT_SUM, 'ldr w22, [x19, #0x400]: the sum over every core'),
    'cnt_sum_b': _cnt_patch(0xb9440677, lambda gen, a: 'c->x[23] = 0;', 'ldr w23, [x19, #0x404]: (summed at the first)'),
    'cnt_sum_c': _cnt_patch(0xb9440a78, lambda gen, a: 'c->x[24] = 0;', 'ldr w24, [x19, #0x408]: (summed at the first)'),
    'cnt_clear_a': _cnt_patch(0xb9000abf, lambda gen, a: _CNT_ZERO.format(blk=_cnt('c->x[19]')),
                              'str wzr, [x21, #8]: clear every core\'s counter'),
    'cnt_clear_b': _cnt_patch(0xf90002bf, lambda gen, a: '(void)0;', 'str xzr, [x21]: (cleared at the first)'),
    'cnt_init_a': _cnt_patch(0xf902027f, lambda gen, a: _CNT_ZERO.format(blk=_cnt('c->x[19]')),
                             'str xzr, [x19, #0x400]: zero every core\'s counter'),
    'cnt_init_b': _cnt_patch(0xf902067f, lambda gen, a: 'ST32(c->x[19] + 0x40c, 0);',
                             'str xzr, [x19, #0x408]: (counters zeroed at the first) +0x40c'),
    'busy_init': _busy_patch(0xb800b2bf, lambda gen, a: (
        f'uint64_t b = {_busy("c->x[19]")}; for (int i = 0, n = rt_game_cores() + 1; i < n; i++) ST8(b + i, 0);'),
        'stur wzr, [x21, #0xb]: zero every core\'s busy count'),
    'obj7_ctor': _obj7_ctor,
    'obj7_free_0': _obj7_free,
    'obj7_free_1': _obj7_patch(0xf9400a74, lambda gen, a: 'c->x[20] = 0;', 'ldr x20, [x19, #0x10]: (freed at the first)'),
    'obj7_free_1s': _obj7_patch(0xf900067f, lambda gen, a: '(void)0;', 'str xzr, [x19, #8]: (cleared at the first)'),
    'obj7_free_2': _obj7_patch(0xf9400e74, lambda gen, a: 'c->x[20] = 0;', 'ldr x20, [x19, #0x18]: (freed at the first)'),
    'obj7_free_2s': _obj7_patch(0xf9000a7f, lambda gen, a: '(void)0;', 'str xzr, [x19, #0x10]: (cleared at the first)'),
    'obj7_free_3s': _obj7_patch(0xf9000e7f, lambda gen, a: '(void)0;', 'str xzr, [x19, #0x18]: (cleared at the first)'),
    'obj7_get_base': _obj7_patch(0x91002268, lambda gen, a: 'c->x[8] = rt_percore_block(c->x[19], 8, 8, 3, 0);',
                                 'add x8, x19, #8: every core\'s object'),
    'obj7_get_bound': _obj7_patch(0x71000d3f, lambda gen, a: '(void)(subs32(c, (uint32_t)c->x[9], (uint32_t)rt_game_cores()));',
                                  'cmp w9, #3: a core with a slot'),
    'wd_a_flag': _wd_base(0x8b204268, 8, 19, 'flag', '(uint64_t)(uint32_t)c->x[0]', 'add x8, x19, w0, uxtw'),
    'wd_a_tick': _wd_base(0x8b140e75, 21, 19, 'tick', 'c->x[20] << 3', 'add x21, x19, x20, lsl #3'),
    'wd_a_state': _wd_base(0x8b140a68, 8, 19, 'state', 'c->x[20] << 2', 'add x8, x19, x20, lsl #2'),
    'wd_b_flag': _wd_base(0x8b204268, 8, 19, 'flag', '(uint64_t)(uint32_t)c->x[0]', 'add x8, x19, w0, uxtw'),
    'wd_b_state': _wd_base(0x8b080a69, 9, 19, 'state', 'c->x[8] << 2', 'add x9, x19, x8, lsl #2'),
    'wd_b_tick': _wd_base(0x8b080e76, 22, 19, 'tick', 'c->x[8] << 3', 'add x22, x19, x8, lsl #3'),
    'wd_c_flag': _wd_base(0x8b204288, 8, 20, 'flag', '(uint64_t)(uint32_t)c->x[0]', 'add x8, x20, w0, uxtw'),
    'wd_c_state': _wd_base(0x8b080a8a, 10, 20, 'state', 'c->x[8] << 2', 'add x10, x20, x8, lsl #2'),
    'wd_c_tick': _wd_base(0x8b080e94, 20, 20, 'tick', 'c->x[8] << 3', 'add x20, x20, x8, lsl #3'),
    'wd_d_flag': _wd_base(0x8b204268, 8, 19, 'flag', '(uint64_t)(uint32_t)c->x[0]', 'add x8, x19, w0, uxtw'),
    'wd_d_state': _wd_base(0x8b080a69, 9, 19, 'state', 'c->x[8] << 2', 'add x9, x19, x8, lsl #2'),
    'wd_d_tick': _wd_base(0x8b080e7c, 28, 19, 'tick', 'c->x[8] << 3', 'add x28, x19, x8, lsl #3'),
    'wd_d_flag2': _wd_base(0x8b284269, 9, 19, 'flag', '(uint64_t)(uint32_t)c->x[8]', 'add x9, x19, w8, uxtw'),
    'wd_ctor_tick0': _wd_ticks,
    'wd_ctor_tick1': _wd_patch(0xf90006e0, lambda gen, a: '(void)0;', 'str x0, [x23, #8]: (every core\'s at the first)'),
    'wd_ctor_tick2': _wd_patch(0xf9000ae0, lambda gen, a: '(void)0;', 'str x0, [x23, #0x10]: (every core\'s at the first)'),
    'wd_init_state_base': _wd_patch(0x8b0802c8, lambda gen, a: f'c->x[8] = {_wd("c->x[22]", "state")};',
                                    'add x8, x22, x8: every core\'s state'),
    'wd_init_state_a': _wd_patch(0xb9000909, lambda gen, a: (
        'for (int i = 0, n = rt_game_cores(); i < n; i++) ST32(c->x[8] + 4ull * i, (uint32_t)c->x[9]);'),
        'str w9, [x8, #8]: every core\'s state to -1'),
    'wd_init_state_b': _wd_patch(0xf9000109, lambda gen, a: '(void)0;', 'str x9, [x8]: (every core\'s at the first)'),
    'wd_init_flag_base': _wd_patch(0x8b0802c8, lambda gen, a: f'c->x[8] = {_wd("c->x[22]", "flag")};',
                                   'add x8, x22, x8: every core\'s flag'),
    'wd_init_flag_a': _wd_patch(0x7900011f, lambda gen, a: (
        'for (int i = 0, n = rt_game_cores(); i < n; i++) ST8(c->x[8] + i, 0);'), 'strh wzr, [x8]: every core\'s flag to 0'),
    'wd_init_flag_b': _wd_patch(0x3900091f, lambda gen, a: '(void)0;', 'strb wzr, [x8, #2]: (every core\'s at the first)'),
    # the core -> index map f_7000fba16c (#28): its table is the identity for the Switch's six (0..5) - the core itself
    'core_index_map': _ctx_patch(0xb8687920, lambda gen, a: 'c->x[0] = (uint32_t)c->x[8];',
                                 'ldr w0, [x9, x8, lsl #2]: the core itself (the table is 0, 1, .. 5)', 'core index map'),
    'lin_ctor': _lin_whole(0xf0006b88, 'rt_lin_ctor', ('lin_vtable', 'lin_lock_ctor'), 'the constructor, every record'),
    'lin_dtor': _lin_whole(0xa9be7bfd, 'rt_lin_dtor', ('lin_vtable', 'lin_alloc_fini', 'lin_free', 'lin_lock_dtor'),
                           'the destructor, every record'),
    'lin_init': _lin_whole(0xd10183ff, 'rt_lin_init', ('lin_alloc_make', 'lin_alloc_init', 'lin_malloc', 'lin_split',
                                                       'lin_after'), 'the init, every record'),
    'lin_split': _lin_whole(0x3945000a, 'rt_lin_split', (), 'the split among every core'),
    'lin_rec_a': _lin_rec(0x9b280028, 0, 1, 8),
    'lin_rec_b': _lin_rec(0x9b280028, 0, 1, 8),
    'lin_rec_c': _lin_rec(0x9b295108, 20, 8, 9),
    'lin_rec_d': _lin_rec(0x9b280028, 0, 1, 8),
    'lin_rec_e': _ctx_patch(0x9100e008, lambda gen, a: 'c->x[8] = rt_percore_block(c->x[0], 0x38, 0x58, 3, 0);',
                            'add x8, x0, #0x38: every core\'s record', 'per-core linear allocators'),
    'list24_init': _LIST24_INIT,
    'list24_a_base': _list24_base(0x91246108, 8, 0x918, 0x810, 'add x8, x8, #0x918: every core\'s list'),
    'list24_a_bound': _list24_bound(0x71000c3f, 1),
    'list24_b_bytes': _list24_base(0x910522d6, 22, 0x148, 1, 'add x22, x22, #0x148: every core\'s byte'),
    'list24_b_base': _list24_base(0x91246129, 9, 0x918, 0x810, 'add x9, x9, #0x918: every core\'s list'),
    'list24_b_bound1': _list24_bound(0x71000d1f, 8),
    'list24_b_bound2': _list24_bound(0x71000d1f, 8),
    'list24_b_bound3': _list24_bound(0x71000d1f, 8),
    'list24_b_bound4': _list24_bound(0x71000d1f, 8),
    'list24_c_base': _list24_base(0x91246108, 8, 0x918, 0x810, 'add x8, x8, #0x918: every core\'s list'),
    'list24_c_bound': _list24_bound(0x71000c3f, 1),
    'list24_d_base': _list24_base(0x91246108, 8, 0x918, 0x810, 'add x8, x8, #0x918: every core\'s list'),
    'list24_d_bound': _list24_bound(0x71000c3f, 1),
    'cnt22_add_base': _ctx_patch(0x8b080688, lambda gen, a: f'c->x[8] = {_cnt22("c->x[20]")} - 0xb8 + (c->x[8] << 1);',
                                 'add x8, x20, x8, lsl #1: the core\'s counter', 'per-core u16 counters'),
    'cnt22_reset_a': _ctx_patch(0x7901781f, lambda gen, a: (
        f'uint64_t b = {_cnt22("c->x[0]")}; for (int i = 0, n = rt_game_cores(); i < n; i++) ST16(b + 2ull * i, 0);'),
        'strh wzr, [x0, #0xbc]: zero every core\'s counter', 'per-core u16 counters'),
    'cnt22_reset_b': _ctx_patch(0xb900b81f, lambda gen, a: '(void)0;', 'str wzr, [x0, #0xb8]: (zeroed at the first)',
                                'per-core u16 counters'),
    'pair6_a_base': _pair6_base(0x9139c288, 20, 0xe70), 'pair6_a_bound': _PAIR6_BOUND,
    'pair6_b_base': _pair6_base(0x9139c268, 19, 0xe70), 'pair6_b_bound': _PAIR6_BOUND,
    'pair6_c_base': _pair6_base(0x9139f268, 19, 0xe7c), 'pair6_c_bound': _PAIR6_BOUND,
    'pair6_d_base': _pair6_base(0x9139f268, 19, 0xe7c), 'pair6_d_bound': _PAIR6_BOUND,
    'pair6_e_base': _pair6_base(0x9139f268, 19, 0xe7c), 'pair6_e_bound': _PAIR6_BOUND,
    'pair6_sum_u0': _PAIR6_SUM_U,
    'pair6_sum_uz0': _PAIR6_NONE(0xb90e727f, 'str wzr, [x19, #0xe70]: (cleared'),
    'pair6_sum_u1': _pair6_patch(0xb94e7669, lambda gen, a: 'c->x[9] = 0;', 'ldr w9, [x19, #0xe74]: (summed at the first)'),
    'pair6_sum_uz1': _PAIR6_NONE(0xb90e767f, 'str wzr, [x19, #0xe74]: (cleared'),
    'pair6_sum_u2': _pair6_patch(0xb94e7a68, lambda gen, a: 'c->x[8] = 0;', 'ldr w8, [x19, #0xe78]: (summed at the first)'),
    'pair6_sum_uz2': _PAIR6_NONE(0xb90e7a7f, 'str wzr, [x19, #0xe78]: (cleared'),
    'pair6_sum_f0': _PAIR6_SUM_F,
    'pair6_sum_fz0': _PAIR6_NONE(0xb90e7e7f, 'str wzr, [x19, #0xe7c]: (cleared'),
    'pair6_sum_f1': _pair6_patch(0xbd4e8260, lambda gen, a: 'c->v[0] = VV((uint32_t)0);', 'ldr s0, [x19, #0xe80]: (summed at the first)'),
    'pair6_sum_fz1': _PAIR6_NONE(0xb90e827f, 'str wzr, [x19, #0xe80]: (cleared'),
    'pair6_sum_f2': _pair6_patch(0xbd4e8660, lambda gen, a: 'c->v[0] = VV((uint32_t)0);', 'ldr s0, [x19, #0xe84]: (summed at the first)'),
    'pair6_sum_fz2': _PAIR6_NONE(0xb90e867f, 'str wzr, [x19, #0xe84]: (cleared'),
    'pair6_zero_a': _PAIR6_ZERO,
    'pair6_zero_b': _PAIR6_NONE(0xf9073e7f, 'str xzr, [x19, #0xe78]: (zeroed'),
    'pair6_zero_c': _PAIR6_NONE(0xf9073a7f, 'str xzr, [x19, #0xe70]: (zeroed'),
    # #21 (BOTW-JOBS-NATIVE.md): a flip byte per core at +0x98 (unbounded; f_7001062ccc toggles the core's and picks a
    # table entry with it), zeroed by the constructor f_7001062940: a block with a slot for every core
    'flip21_base': _ctx_patch(0x91026288, lambda gen, a: 'c->x[8] = rt_percore_block(c->x[20], 0x98, 1, 3, 0);',
                              'add x8, x20, #0x98: every core\'s flip byte', 'per-core flip bytes'),
    'flip21_zero': _ctx_patch(0xb9009a9f, lambda gen, a: (
        'uint64_t b = rt_percore_block(c->x[20], 0x98, 1, 3, 0); '
        'for (int i = 0, n = rt_game_cores(); i < n; i++) ST8(b + i, 0); ST8(c->x[20] + 0x9b, 0);'),
        'str wzr, [x20, #0x98]: zero every core\'s flip byte (and +0x9b)', 'per-core flip bytes'),
    'slot5_rec_base': _slot5_base(0x8b0902a9, 9, 21, 0x1138, 0x40), 'slot5_rec_bound': _SLOT5_BOUND,
    'slot5_u_base': _slot5_base(0x8b090289, 9, 20, 0x1238, 4), 'slot5_u_bound': _SLOT5_BOUND,
    'slot5_f_base': _slot5_base(0x911bc269, 9, 19, 0x6f0, 4), 'slot5_f_bound': _SLOT5_BOUND,
    'slot5_merge': _slot5_whole(0x52822e88, 'rt_rec5_merge', 'the merge over every slot'),
    'slot5_pick': _slot5_whole(0xbd46f000, 'rt_f5_pick', 'the float pick over every slot'),
    'slot5_fz_a': _slot5_fzero(0xb906fa7f, 19, 'str wzr, [x19, #0x6f8]: zero every slot\'s float'),
    'slot5_fz_b': _slot5_patch(0xf9037a7f, lambda gen, a: '(void)0;', 'str xzr, [x19, #0x6f0]: (zeroed at the first)'),
    'slot5_fz_c': _slot5_fzero(0xf903781f, 0, 'str xzr, [x0, #0x6f0]: zero every slot\'s float'),
    'slot5_fz_d': _slot5_patch(0xb906f81f, lambda gen, a: '(void)0;', 'str wzr, [x0, #0x6f8]: (zeroed at the first)'),
    'obj12_first': _obj12_first,
    'obj12_alloc1': _obj12_take(1), 'obj12_ctor1': _obj12_patch(None, lambda gen, a: '(void)0;', 'bl: (made at the first)'),
    'obj12_alloc2': _obj12_take(2), 'obj12_ctor2': _obj12_patch(None, lambda gen, a: '(void)0;', 'bl: (made at the first)'),
    'obj12_a': _obj12_base(0x8b204e88, 20), 'obj12_b': _obj12_base(0x8b204ee8, 23), 'obj12_c': _obj12_base(0x8b204f48, 26),
    'obj12_d': _obj12_base(0x8b204e88, 20), 'obj12_e': _obj12_base(0x8b204e88, 20),
    'q27_add_bufs': _q27_base(0x9104a268, 8, 19, 0x128, 0x10, 'add x8, x19, #0x128: every slot\'s buffer'),
    'q27_add_bound1': _Q27_BOUND,
    'q27_add_locks': _q27_base(0x910ca268, 8, 19, 0x328, 0x40, 'add x8, x19, #0x328: every slot\'s lock'),
    'q27_add_bound2': _Q27_BOUND,
    'q27_add_cur': _q27_cur(0x8b190a69, 9, 19, 25),
    'q27_walk_locks': _q27_base(0x910ca289, 9, 20, 0x328, 0x40, 'add x9, x20, #0x328: every slot\'s lock'),
    'q27_walk_bufs': _q27_base(0x9104a288, 8, 20, 0x128, 0x10, 'add x8, x20, #0x128: every slot\'s buffer'),
    'q27_walk_cur': _q27_cur(0x8b1c0a88, 8, 20, 28),
    'q27_walk_bound': _q27_patch(0xf1000f9f, lambda gen, a: '(void)(subs64(c, c->x[28], (uint64_t)rt_game_cores()));',
                                 'cmp x28, #3: every slot'),
    'q27_init_first': _q27_init_first,
    'q27_init_first_b': _q27_patch(0xf9009a80, lambda gen, a: '(void)0;', 'str x0, [x20, #0x130]: (stored at the first)'),
    'q27_init_2': _q27_patch(None, lambda gen, a: 'c->x[0] = 0;', 'bl: (every slot\'s made at the first)'),
    'q27_init_3': _q27_patch(None, lambda gen, a: 'c->x[0] = 0;', 'bl: (every slot\'s made at the first)'),
    'q27_ctor_tail': _q27_ctor_tail,
    # the second append (f_700112e6d4 1.9.0, two places, without the lock): the buffers' base, the bound, the cursor
    'q27_add2_bufs': _q27_base(0x9104a269, 9, 19, 0x128, 0x10, 'add x9, x19, #0x128: every slot\'s buffer'),
    'q27_add2_bound': _q27_patch(0x71000ebf, lambda gen, a: (
        '(void)(subs32(c, (uint32_t)c->x[21], (uint32_t)rt_game_cores()));'), 'cmp w21, #3: a slot with a block entry'),
    'q27_add2_cur': _q27_cur(0x8b150a6a, 10, 19, 21),
    'q27_add3_bufs': _q27_base(0x9104a269, 9, 19, 0x128, 0x10, 'add x9, x19, #0x128: every slot\'s buffer'),
    'q27_add3_bound': _q27_patch(0x71000d5f, lambda gen, a: (
        '(void)(subs32(c, (uint32_t)c->x[10], (uint32_t)rt_game_cores()));'), 'cmp w10, #3: a slot with a block entry'),
    'q27_add3_cur': _q27_cur(0x8b0a0a6a, 10, 19, 10),
    'rec23_ctor_end': _rec23_patch(0xb908627f, lambda gen, a: (
        'extern void rt_rec23_ctor_end(uint64_t); ST32(c->x[19] + 0x860, 0); rt_rec23_ctor_end(c->x[19] - 0x28);'),
        'str wzr, [x19, #0x860]: then every core\'s records into the block (x19: this + 0x28, from the constructor\'s '
        '`str x8, [x19, #0x28]!`)'),
    'rec23_init_0': _rec23_patch(0xb902ca74, lambda gen, a: (
        f'uint64_t b = {_REC23.format(obj="c->x[19]")}; '
        'for (uint64_t j = 0, n = 4 * (uint64_t)rt_game_cores(); j < n; j++) ST32(b + 0x28 + 0x58 * j, (uint32_t)c->x[20]);'),
        'str w20, [x19, #0x2c8]: every sub-entry\'s state'),
    'rec23_init_1': _rec23_none(0xb9032274, 'str w20, [x19, #0x320]'),
    'rec23_init_2': _rec23_none(0xb9037a74, 'str w20, [x19, #0x378]'),
    'rec23_init_3': _rec23_none(0xb903d274, 'str w20, [x19, #0x3d0]'),
    'rec23_init_4': _rec23_none(0xb9042a74, 'str w20, [x19, #0x428]'),
    'rec23_init_5': _rec23_none(0xb9048274, 'str w20, [x19, #0x480]'),
    'rec23_init_6': _rec23_none(0xb904da74, 'str w20, [x19, #0x4d8]'),
    'rec23_init_7': _rec23_none(0xb9053274, 'str w20, [x19, #0x530]'),
    'rec23_init_8': _rec23_none(0xb9058a74, 'str w20, [x19, #0x588]'),
    'rec23_init_9': _rec23_none(0xb905e274, 'str w20, [x19, #0x5e0]'),
    'rec23_init_10': _rec23_none(0xb9063a74, 'str w20, [x19, #0x638]'),
    'rec23_init_11': _rec23_none(0xb9069274, 'str w20, [x19, #0x690]'),
    'rec23_walk_base': _rec23_patch(0x8b17027b, lambda gen, a: (
        f'c->x[27] = {_REC23.format(obj="c->x[19]")} - 0x2a0 + c->x[23];'), 'add x27, x19, x23: the record in the block'),
    'rec23_walk_end': _rec23_patch(0xf11082ff, lambda gen, a: (
        '(void)(subs64(c, c->x[23], 0x160ull * (uint64_t)rt_game_cores()));'), 'cmp x23, #0x420: every core\'s record'),
    'rec23_a_base': _rec23_base(0x910aa2a8, 8, 21), 'rec23_b_base': _rec23_base(0x910aa288, 8, 20),
    'rec23_a_b1': _rec23_bound(0x71000d3f, 9), 'rec23_b_b1': _rec23_bound(0x71000d3f, 9),
    'rec23_a_b2': _rec23_bound(0x71000d3f, 9), 'rec23_b_b2': _rec23_bound(0x71000d3f, 9),
    'rec23_a_b3': _rec23_bound(0x71000d3f, 9), 'rec23_b_b3': _rec23_bound(0x71000d3f, 9),
    'rec23_a_b4': _rec23_bound(0x71000d3f, 9), 'rec23_b_b4': _rec23_bound(0x71000d3f, 9),
    'rec23_a_b5': _rec23_bound(0x71000d5f, 10), 'rec23_b_b5': _rec23_bound(0x71000d5f, 10),
    'rec4a_ctor_0': _rec4a_ctor(0x91380260, 0), 'rec4a_ctor_1': _rec4a_ctor(0x9139c260, 1),
    'rec4a_ctor_2': _rec4a_ctor(0x913b8260, 2),
    'rec4a_ctor_rest': _rec4a_rest('rec4a_reset_fn', 'c->x[19]', 0, 0, 'the reset'),
    'rec4a_part_base': _rec4a_part(0x91168275, 21, 0, 'add x21, x19, #0x5a0: every slot\'s record'),
    'rec4a_part_size': _rec4_patch(0x52802a02, lambda gen, a: 'c->x[2] = 0x70ull * (uint64_t)rt_game_cores();',
                                   'mov w2, #0x150: every slot\'s record cleared'),
    'rec4a_part_1': _rec4a_part(0x91184260, 0, 0x70, 'add x0, x19, #0x610: slot 1\'s record'),
    'rec4a_part_2': _rec4a_part(0x911a0260, 0, 0xe0, 'add x0, x19, #0x680: slot 2\'s record'),
    'rec4a_part_rest': _rec4a_rest('rec4a_make_fn', 'c->x[19] - 0x860', 0, 0, 'the record\'s constructor'),
    'rec4a_drop_rest': _rec4a_rest('rec4a_release_fn', 'c->x[19] - 0x860', 0x60, 1, 'the handle\'s release'),
    'rec4a_drop_1': _rec4a_part(0x9119c260, 0, 0x70 + 0x60, 'add x0, x19, #0x670: slot 1\'s handle'),
    'rec4a_drop_0': _rec4a_part(0x91180260, 0, 0x60, 'add x0, x19, #0x600: slot 0\'s handle'),
    'rec4a_acc_base': _rec4_patch(0x91380289, lambda gen, a: 'c->x[9] = rt_percore_block(c->x[20], 0xe00, 0x70, 3, 0);',
                                  'add x9, x20, #0xe00: every slot\'s record'),
    'rec4a_acc_bound': _rec4_patch(0x71000d1f, lambda gen, a: (
        '(void)(subs32(c, (uint32_t)c->x[8], (uint32_t)rt_game_cores()));'), 'cmp w8, #3: a slot with a block entry'),
    'rec4a_pick': _rec4a_pick,
    'rec4b_ctor_end': _rec4_patch(0xb90a027f, lambda gen, a: (
        'extern void rt_rec4b_ctor(uint64_t); ST32(c->x[19] + 0xa00, 0); rt_rec4b_ctor(c->x[20]);'),
        'str wzr, [x19, #0xa00]: then every slot\'s lists (x19: this + 0xc8, x20: this)'),
    'rec4b_acc_base': _rec4_patch(0x9100a288, lambda gen, a: f'c->x[8] = {_l4b("c->x[20]", 0x28, 0x280)};',
                                  'add x8, x20, #0x28: every slot\'s lists'),
    'rec4b_acc_bound': _REC4B_BOUND,
    'rec4b_acc_a_base': _rec4_patch(0x9128a288, lambda gen, a: f'c->x[8] = {_l4b("c->x[20]", 0xa28, 0x50)};',
                                    'add x8, x20, #0xa28: every slot\'s A list'),
    'rec4b_acc_a_bound': _REC4B_BOUND,
    'rec4b_reset': _rec4b_whole(0xb900281f, 'rt_rec4b_reset', 'str wzr, [x0, #0x28]: every list emptied'),
    'rec4b_remove': _rec4b_whole(0xd103c3ff, 'rt_rec4b_remove', 'sub sp, sp, #0xf0: out of every slot\'s lists'),
    'rec4b_remove_a': _rec4b_whole(0xa9be7bfd, 'rt_rec4b_remove_a', 'stp x29, x30, [sp, #-0x20]!: out of every A list'),
    'rec4b_remove_b': _rec4b_whole(0xa9be7bfd, 'rt_rec4b_remove_b', 'stp x29, x30, [sp, #-0x20]!: out of every B list'),
    'rec4b_sweep': _rec4b_whole(0xa9bb7bfd, 'rt_rec4b_sweep', 'stp x29, x30, [sp, #-0x50]!: every slot\'s lists swept',
                                'rec4b_drop_fn'),
    'rec4b_pick': _rec4b_whole(0xfc1b0fe8, 'rt_rec4b_pick', 'str d8, [sp, #-0x50]!: the pick over every slot'),
    'rec4b_fill_b': _rec4b_whole(0xd101c3ff, 'rt_rec4b_fill_b', 'sub sp, sp, #0x70: every slot\'s B list'),
    'rec4b_filter': _rec4b_whole(0xa9bc7bfd, 'rt_rec4b_filter', 'stp x29, x30, [sp, #-0x40]!: every slot\'s kind 0'),
    'rec4b_states': _rec4b_whole(0xd10103ff, 'rt_rec4b_states', 'sub sp, sp, #0x40: every slot\'s kind 0'),
    'rec4b_pick_b': _rec4b_whole(0xfc1c0fe8, 'rt_rec4b_pick_b', 'str d8, [sp, #-0x40]!: the pick over every B list'),
    'rec4b_upd_clear1': _rec4b_clear({4 * j: off for j, off in enumerate(_REC4B_COUNTS) if j}),
    'rec4b_upd_clear2': _rec4b_clear(dict(zip([4, 8, 0x14, 0x18, 0x20, 0x28, 0x2c] + list(range(0x30, 0x70, 4)) +
                                              [0x70, 0x74, 0x7c], _REC4B_COUNTS[1:]))),
    'rec4b_upd_any_a': _rec4b_any(8, _REC4B_ANY8),
    'rec4b_upd_any_b': _rec4b_any(4, ((0xb9807a6a, 10), (0xb942fa6b, 11), (0xb9457a6c, 12))),
    'rec4b_upd_any_c': _rec4b_any(8, _REC4B_ANY8),
    'rec4b_upd_any_d': _rec4b_any(8, _REC4B_ANY8),
    'rec4b_upd_keep_a': _rec4b_keep_a,
    'rec4b_upd_other1': _rec4b_other1,
    'rec4c_ctor_end': _rec4_patch(0xf9400bf3, lambda gen, a: (
        'extern void rt_rec4c_ctor(uint64_t); rt_rec4c_ctor(c->x[19]); c->x[19] = LD64(c->sp + 0x10);'),
        'ldr x19, [sp, #0x10]: every slot\'s rings first'),
    'rec4c_push_base': _rec4_patch(0x9ba95109, lambda gen, a: (
        f'c->x[9] = {_ring4c("c->x[20]")} + 0x418ull * (uint32_t)c->x[8];'), 'umaddl x9, w8, w9, x20: the slot\'s ring'),
    'rec4c_push_bound': _rec4_patch(0x71000d1f, lambda gen, a: (
        '(void)(subs32(c, (uint32_t)c->x[8], (uint32_t)rt_game_cores()));'), 'cmp w8, #3: a slot with a ring'),
    'rec4c_push_pick': _rec4_patch(0x9a943129, lambda gen, a: (
        f'c->x[9] = (uint32_t)c->x[8] < (uint32_t)rt_game_cores() ? c->x[9] : {_ring4c("c->x[20]")};'),
        'csel x9, x9, x20, lo: the slot\'s ring (else the first)'),
    'rec4c_pop_1': _rec4c_pop, 'rec4c_pop_2': _rec4c_pop,
    'rec4c_any_1a': _rec4c_any(8, ((0, 0xb9401808), (0x10, _REC4C_G1[0][0]), (0x18, _REC4C_G1[1][0]))),
    'rec4c_any_1b': _rec4c_any(8, ((0, 0xb9401a68), (8, _REC4C_G1[0][0]), (0x10, _REC4C_G1[1][0]))),
    'rec4c_any_2a': _rec4c_any(0xc50, ((0, _REC4C_G2[0][0]), (8, _REC4C_G2[1][0]), (0x10, _REC4C_G2[2][0]))),
    'rec4c_any_2b': _rec4c_any(0xc50, ((0, _REC4C_G2[0][0]), (8, _REC4C_G2[1][0]), (0x10, _REC4C_G2[2][0]))),
    'ring19_a_size': _ring19_n(0x52908000, 0, 0x80 * 0x58, 0, 'mov w0, #0x8400: every core\'s entries'),
    'ring19_a_count': _ring19_n(0x52803008, 8, 0x80, 0, 'mov w8, #0x180: every core\'s entries made'),
    'ring19_a_cores': _ring19_n(0xd2800068, 8, 1, 0, 'mov x8, #3: the count of cores'),
    'ring19_b_size': _ring19_n(0x52842000, 0, 0x40 * 0x2c, 0, 'mov w0, #0x2100: every core\'s entries'),
    'ring19_b_start': {0: _ring19_n(0x92841fe8, 8, -0x40 * 0x2c, 0, 'mov x8, #-0x2100: from every core\'s first entry'),
                       4: _ring19_n(0x52842089, 9, 0x40 * 0x2c, 4, 'mov w9, #0x2104: (the same, +4)'),
                       8: _ring19_n(0x5284218a, 10, 0x40 * 0x2c, 0xc, 'mov w10, #0x210c: (the same, +0xc)')},
    'ring19_b_cores': _ring19_n(0xd2800068, 8, 1, 0, 'mov x8, #3: the count of cores'),
    'ring19_c_size': _ring19_n(0x52817400, 0, 8 * 0x7c, 0, 'mov w0, #0xba0: every core\'s entries'),
    'ring19_c_rest': _ring19_c_rest,
    'ring19_c_cores': _ring19_n(0xd2800068, 8, 1, 0, 'mov x8, #3: the count of cores'),
    'ring19_a_zero': _ring19_zero(23, 0xc), 'ring19_b_zero': _ring19_zero(22, 0xc), 'ring19_c_zero': _ring19_zero(22, 4),
    'ring19_reset': _ring19_patch(0xf941a408, lambda gen, a: (
        'extern void rt_ring19_reset(Cpu *); rt_ring19_reset(c); return;'), 'ldr x8, [x0, #0x348]: every ring\'s cursors'),
    'ring19_add_a': _RING19_ADD, 'ring19_add_c': _RING19_ADD,
    'ring19_wa_base': _ring19_patch(0x910062e9, lambda gen, a: f'c->x[9] = {_cur19("c->x[23]")};',
                                    'add x9, x23, #0x18: every core\'s cursor'),
    'ring19_wa_bound': _ring19_patch(0xf1000f7f, lambda gen, a: '(void)(subs64(c, c->x[27], (uint64_t)rt_game_cores()));',
                                     'cmp x27, #3: every core'),
    'ring19_wb_c0': _ring19_c0(0xb9401a96, 22, 20), 'ring19_wb_c1': _ring19_c1(0xb9401e88, 8, 20, 0xc),
    'ring19_wb_ck': _ring19_next(0xb9402288, 8, 20, 0xc, 0xf8, 0xc),
    'ring19_wb_mov': _ring19_mov(0xc), 'ring19_wb_ccmp': _ring19_ccmp(0xc), 'ring19_wb_end': _ring19_end(0xf4, 0xd8),
    'ring19_wc_c0': _ring19_c0(0xb940181b, 27, 0), 'ring19_wc_c1': _ring19_c1(0xb9401e9b, 27, 20, 4),
    'ring19_wc_ck': _ring19_next(0xb940229b, 27, 20, 4, 0x188, 8),
    'ring19_wc_mov': _ring19_mov(4), 'ring19_wc_ccmp': _ring19_ccmp(4), 'ring19_wc_end': _ring19_end(0x184, 0x168),
    'obj28_ifaces': {0: _obj28_patch(0xa9142668, lambda gen, a: (
                         'extern void rt_obj28_ifaces(uint64_t, uint64_t, uint64_t); '
                         'rt_obj28_ifaces(c->x[19], c->x[8], c->x[9]);'), 'stp x8, x9, [x19, #0x140]: every slot\'s interface'),
                     **{4 * i: _obj28_rest(0xa9142668 + (i << 16), f'stp x8, x9, [x19, #{0x140 + 0x10 * i:#x}]')
                        for i in range(1, 6)}},
    'obj28_make': {0: _obj28_call('rt_obj28_make', 'obj28_make_fn'), **{0xc * i: _obj28_bl_rest for i in range(1, 6)}},
    'obj28_destroy': {0: _obj28_call('rt_obj28_destroy', 'obj28_destroy_fn'), **{8 * i: _obj28_bl_rest for i in range(1, 6)}},
    'obj28_pick': _obj28_pick,
    'str26_init_base': _str26_patch(0x913082d6, lambda gen, a: f'c->x[22] = {_str26(22, 0xc20)};',
                                    'add x22, x22, #0xc20: every core\'s string'),
    'str26_init_rest': _str26_rest,
    'str26_a_base': _str26_patch(0x913082d6, lambda gen, a: f'c->x[22] = {_str26(22, 0xc20)};',
                                 'add x22, x22, #0xc20: every core\'s string'),
    'str26_b_base': _str26_patch(0x913082f7, lambda gen, a: f'c->x[23] = {_str26(23, 0xc20)};',
                                 'add x23, x23, #0xc20: every core\'s string'),
    'busy_inc_bound': _BUSY_BOUND,
    'busy_inc_base': _BUSY_BASE,
    'busy_dec_bound': _BUSY_BOUND,
    'busy_dec_base': _BUSY_BASE,
    'busy_reset_a': _busy_zero(0xb800a2bf, 'stur wzr, [x21, #0xa]: zero every core\'s busy count and the total'),
    'busy_reset_b': _busy_zero(0xb800229f, 'stur wzr, [x20, #2]: zero every core\'s busy count and the total'),
    'busy_sum_a': _busy_patch(0x39402ea8, lambda gen, a: (
        f'uint64_t b = {_busy("c->x[19]")}; uint32_t s = 0; '
        'for (int i = 0, n = rt_game_cores(); i < n; i++) s += LD8(b + i); c->x[8] = s;'),
        'ldrb w8, [x21, #0xb]: the sum of every core\'s busy count'),
    'busy_sum_b': _busy_patch(0x394032a9, lambda gen, a: 'c->x[9] = 0;', 'ldrb w9, [x21, #0xc]: (summed at the first)'),
    'busy_sum_c': _busy_patch(0x394036a9, lambda gen, a: 'c->x[9] = 0;', 'ldrb w9, [x21, #0xd]: (summed at the first)'),
    'busy_total': _busy_patch(0x39003aa8, lambda gen, a: (
        f'ST8({_busy("c->x[19]")} + (uint64_t)rt_game_cores(), (uint8_t)c->x[8]);'), 'strb w8, [x21, #0xe]: the total'),
    'busy_sys': _busy_patch(0x38e86a68, lambda gen, a: (
        f'c->x[8] = (uint32_t)(int32_t)(int8_t)LD8({_busy("c->x[19]")} + (uint64_t)rt_game_cores());'),
        'ldrsb w8, [x19, x8]: the total busy count'),
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
    has no patch). A patch given as a dict is several instructions of one site: {offset from it: patch}."""
    out = {}
    for name, fn in PATCHES.items():
        if name in sites:
            out.update({sites[name] + o: f for o, f in fn.items()} if isinstance(fn, dict) else {sites[name]: fn})
    return out
