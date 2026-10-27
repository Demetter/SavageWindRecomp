"""Places in the game's main module that Savage Wind patches or reads, found by pattern for each game version.

    game_sites.py find <exefs/main>      every site in that main module: what its pattern finds, against the
                                         addresses known for that build (a new version: the table to add)
    game_sites.py pattern <exefs/main> <address> [code|call|ref|got|imm]
                                         a pattern that finds that one instruction (for a new site)
    game_sites.py common <main A> <address in A> <main B> <address in B> [kind]
                                         a pattern that finds the same instruction in two versions (for a site whose
                                         pattern no longer finds it: the bits both versions share)

tools/patches.py replaces single instructions at some of these sites while gen.py generates the code. The runtime
reads or writes game data at the others (runtime/game_sites.h): gen.py puts every site into the game library (the
generated layout.c), so the runtime takes the addresses of the version it runs.

A site is found by a short pattern of instruction words, so a later build of the game (other addresses, the same
code) needs no new numbers. Patterns mask the fields that change when code or data moves: branch and call offsets,
ADRP pages, and the low 12 bits an ADD / LDR / STR adds to an ADRP page in the pattern. `at` picks the instruction
(in words from the pattern's start), and the site is
    code  that instruction's own address
    call  the function a BL / B there goes to
    ref   the address an ADD / LDR / STR there forms with its ADRP (the ADRP itself: its page)
    got   the pointer stored at that `ref` address (a GOT entry; the module's relocations make it absolute)
    imm   the instruction's immediate (an LDR / STR offset, an ADD / CMP / MOVZ value): a field offset in a game
          object that a version moved (1.9.0's UI screens are 0x18 bytes bigger from about +0x290 on)
and two have no pattern: data (where the module's .data starts) and end (where its image ends, .bss included).
One pattern serves every version while the code around a site stays the same. Where a version changed it, the site
takes a list of (pattern, at) instead: the first that finds exactly one place wins (each must find one place, or
none, in every version).

The versions Savage Wind can set up are VERSIONS (the same list as runtime/game_versions.h, which the configuration
screen checks: tools/test_game_sites.py compares them). The builds in KNOWN also list every site's address: there a
pattern has to find exactly that place (the self-check), and a pattern that no longer finds exactly one place falls
back to the listed address with a warning. A build not in KNOWN needs every pattern to find exactly one place.
docs/SAVAGE-WIND.md ("Versiones del juego") has the steps for a new version.
"""
import re
import struct
import sys
from collections import namedtuple

MAIN_BASE = 0x7000000000  # (tools/layout.py: main is the first module)

# name, version, main module build id (the first 8 bytes; tools/sw_probe.py prints it), supported. One that isn't
# supported yet is known but refused by the setup until its sites are checked.
Version = namedtuple('Version', 'name build_id supported')
VERSIONS = [
    Version('1.8.2', 'de0b6ac4efc1dda7', True),
    Version('1.9.0', 'cd57b23fa4bbad65', True),  # (2026-02-17: Thai text on Switch 2; SDK 21.4.1)
]

Site = namedtuple('Site', 'name kind pattern at')
SITES = [
    # The Options screen (tools/patches.py; runtime/ui/game_options.mm)
    Site('opt_create_rows', 'code', 'd10303ff a9067bfd a9076ffc a90867fa a9095ff8 a90a57f6 a90b4ff4 910183fd '
         '90000008/9f00001f b9400118/ffc003ff', 0),
    Site('opt_row_init_flag_type', 'code', '94000000/fc000000 b9006a60 aa1303e0 94000000/fc000000 aa1303e0 '
         '52800021 94000000/fc000000 a9424ff4', 0),
    Site('gdt_flag_type', 'call', '94000000/fc000000 b9006a60 aa1303e0 94000000/fc000000 aa1303e0 52800021 '
         '94000000/fc000000 a9424ff4', 0),
    Site('opt_row_redraw', 'call', 'aa1303e0 52800021 94000000/fc000000 a9424ff4 a9417bfd 9100c3ff d65f03c0', 2),
    Site('opt_read', 'code', 'a9bd7bfd f9000bf5 a9024ff4 910003fd 90000008/9f00001f 91000108/ffc003ff b9405009 '
         'b940010a', 0),
    Site('opt_store', 'code', '90000008/9f00001f 91000108/ffc003ff b9405009 b940010a 6b09015f f9400508 5280060a '
         '1a9f8129 9b2a2121', 0),
    Site('opt_table', 'ref', '90000008/9f00001f 91000108/ffc003ff b9405009 b940010a 6b09015f f9400508 5280060a '
         '1a9f8129 9b2a2121', 1),
    Site('opt_reset', 'code', 'b9406808 7100051f 54000000/ff00001f 35000008/ff00001f 90000008/9f00001f '
         '90000009/9f00001f 91000129/ffc003ff f9400108/ffc003ff', 0),
    Site('opt_msg_call', 'code', '94000000/fc000000 910023e2 aa1403e0 aa1303e1 94000000/fc000000 a9554ff4 a95457fc '
         'a9537bfd', 0),
    Site('msg_lookup', 'call', '94000000/fc000000 910023e2 aa1403e0 aa1303e1 94000000/fc000000 a9554ff4 a95457fc '
         'a9537bfd', 0),
    Site('opt_decide', 'code', 'd100c3ff a9017bfd a9024ff4 910043fd 90000009/9f00001f 91000129/ffc003ff b9405008 '
         'b940012a 5280060b', 0),
    Site('opt_update', 'code', 'd10103ff a9017bfd a90257f6 a9034ff4 910043fd 5286d288/fffff2ff aa0003f3 8b080015', 0),
    Site('opt_back_call', 'code', '94000000/fc000000 f9401e68 b9403909 121e7929 b9003909 90000008/9f00001f '
         '91000108/ffc003ff f9400108', 0),
    Site('screen_back', 'call', '94000000/fc000000 f9401e68 b9403909 121e7929 b9003909 90000008/9f00001f '
         '91000108/ffc003ff f9400108', 0),
    Site('screen_open_request', 'call', '94000000/fc000000 90000008/9f00001f f9400108/ffc003ff f9400108 b9402909 '
         'f9401908 7100d53f 52803509', 0),
    Site('screen_open', 'code', 'a9be7bfd a9014ff4 910003fd aa0003f4 390a001f/ffff1fff b940c000 2a0103f3 '
         '94000000/fc000000', 0),
    Site('screen_close', 'call', '94000000/fc000000 52800028 0a330100 a9414ff4 a8c27bfd d65f03c0', 0),
    Site('screen_mgr', 'ref', '90000008/9f00001f 91000108/ffc003ff f9400108 b9402909 f9401908 7100913f 52802409 '
         '9a9f8129', 1),
    Site('pause_options_update', 'code', 'a9be7bfd a9014ff4 910003fd 90000008/9f00001f aa0003f3 f9400108/ffc003ff '
         'f9400108 b9402909 f9401908 7100d13f 52803409 9a9f8129 f8696914 94000000/fc000000 b4000014/ff00001f '
         'f9400288 aa0003e1 aa1403e0 f9400908 d63f0100 36000000/fff8001f 394a0688/ffff1fff', 0),
    Site('opt_screen_rows', 'imm', 'b9400008/ffc003ff 7100051f 5400000b/ff00001f f95b5009/fffff3ff f9400120 '
         'b940500a 6b01015f 54000000/ff00001f 91002129 f1000508 54000001/ff00001f d65f03c0 52800021', 0),
    # the game's two-button dialog (Quit asks with it)
    Site('dialog_texts', 'code', 'd10143ff a9017bfd f90013f7 a90357f6 a9044ff4 910043fd d2800b49 5286d888/fffffcff', 0),
    Site('dialog_back', 'code', 'f959c100/ffffc3ff b4000000/ff00001f a9414ff4 2a1f03e1 a8c27bfd 14000000/fc000000 '
         'f9400268 aa1303e0', 0),
    Site('dialog_answer', 'ref', '90000008/9f00001f 528001a9 aa0003f3 b9000109/ffc003ff 94000000/fc000000 aa0003e1 '
         'aa1303e0 94000000/fc000000', 3),
    # the title screen menu
    Site('title_update', 'code', 'a9bb7bfd a90167fc 910003fd a9025ff8 a90357f6 a9044ff4 d107c3ff 5286c208/fffff2ff', 0),
    Site('title_button', 'code', 'd10183ff a9037bfd f90023f5 a9054ff4 9100c3fd b9404434 aa0003f3 94000000/fc000000', 0),
    Site('title_options_state', 'imm', 'b9400268/ffc003ff 5101a508 7100191f 54000008/ff00001f 90000009/9f00001f '
         '91000129/ffc003ff 1000000a/9f00001f 3868692b', 0),
    Site('title_msg_file', 'ref', 'f9000549 90000009/9f00001f 91000129/ffc003ff 9000000a/9f00001f '
         'f900015f/ffc003ff a9002d28 a8c17bfd d65f03c0', 2),
    Site('title_version_format', 'ref', '90000001/9f00001f 91000021/ffc003ff d103c3a0 94000000/fc000000 '
         'f9400294/ffffffdf b940da88 35000008/ff00001f 90000001/9f00001f', 1),
    # frame rate (runtime/hle/framepace.c, runtime/nvn/objects.c)
    # BOTW's streaming Yaz0 decoder: its entry and its return (tools/patches.py; runtime/hle/yaz0.c)
    Site('yaz0_stream', 'code', 'a9bd7bfd f9000bf5 a9024ff4 910003fd 39406808 2a0203f4 aa0103f5 aa0003f3', 0),
    Site('yaz0_stream_ret', 'code', 'd65f03c0 7100869f 54000003/ff00001f b9400e68 35000008/ff00001f 12800000 '
         '14000000/fc000000 12800020', 0),
    Site('frame_wait', 'code', 'a9bd7bfd f9000bf5 a9024ff4 910003fd 91078014 aa0003f3 aa1403e0 94000000/fc000000', 0),
    Site('frame_loop_interval_ret', 'code', 'f9400268 aa1303e0 f9409908 d63f0100 14000000/fc000000 1100052a '
         '14000000/fc000000 2a0903e1', 4),
    Site('vfr_ratio', 'code', 'b9407a68 7100051f 5400000b/ff00001f 2a1f03f4 f9404269 6b14011f 1a9f8288 bd403e60', 0),
    # the menus' speed (tools/patches.py _gui_speed): the frame interval x 0.5 stored as the GUI speed
    # perspective projections' matrix (tools/patches.py _persp_matrix): the load of tan(fovy / 2) and the aspect ratio
    Site('persp_matrix', 'code', 'bd409800 1e2c1003 52b7f008 2d558801 1e202800 2d569404', 3),
    Site('persp_matrix_offset', 'code', 'bd409800 1e2c1003 52b7f008 2d558801 1e202800 2d569404', 5),
    # frustum projections' matrix (tools/patches.py _frustum_matrix): its four loads of left / right and top / bottom
    Site('frustum_lr0', 'code', '2d550001 52b7f008 bd409802 b900043f 1e213800 1e2e1001 1e222842 1e201820', 0),
    Site('frustum_lr1', 'code', '2d550001 52b7f008 bd409802 b900043f 1e213800 1e2e1001 1e222842 1e201820', 10),
    Site('frustum_tb0', 'code', '2d550001 52b7f008 bd409802 b900043f 1e213800 1e2e1001 1e222842 1e201820', 15),
    Site('frustum_tb1', 'code', '2d550001 52b7f008 bd409802 b900043f 1e213800 1e2e1001 1e222842 1e201820', 23),
    # projections' update (tools/patches.py _proj_update): the load of the matrix's dirty flag
    Site('proj_update', 'code', '39402008 aa0003f3 34000268 f9400268 91003261 aa1303e0 f9402d08 d63f0100', 0),
    # perspective projections' screen position to camera position (tools/patches.py _persp_unproject): the loads of
    # tan(fovy / 2) (y) and of it with the aspect ratio (x)
    Site('persp_unproject_y', 'code', 'bd409800 f900003f 1e201004 1e214000 bd000820 bd409800 bd40ac01 bd400442', 6),
    Site('persp_unproject_x', 'code', 'bd409800 f900003f 1e201004 1e214000 bd000820 bd409800 bd40ac01 bd400442', 18),
    Site('gui_speed', 'code', 'bd0b2100 94000000/fc000000 f9400260 94000000/fc000000 52800020 94000000/fc000000 '
         'a9414ff4 a8c27bfd', 0),
    Site('pause_state', 'ref', '90000013/9f00001f b9400268/ffc003ff 35000008/ff00001f 90000008/9f00001f '
         'b9400108/ffc003ff 35000008/ff00001f b9400268/ffc003ff 34000008/ff00001f', 1),
    Site('event_mgr', 'ref', '9000000d/9f00001f 910001ad/ffc003ff f94001a0 aa1703e1 94000000/fc000000 aa1703e0 '
         '2a1f03e1 94000000/fc000000', 1),
    Site('controller_step', 'got', '90000008/9f00001f aa1f03f5 f9400108/ffc003ff 52800036 bd400108 b9411668 '
         '1ad522d8 8b150a77', 2),
    # (test tools only: the clock probe)
    Site('player_info', 'ref', '90000008/9f00001f 91000108/ffc003ff f9400100 b4000000/ff00001f 94000000/fc000000 '
         '7100001f 5400000c/ff00001f 91314260', 1),
    Site('world_mgr', 'ref', '90000008/9f00001f 91000108/ffc003ff f9400108 b4000008/ff00001f b946b908 7100051f '
         '54000001/ff00001f a9bf7bfd', 1),
    # the mouse camera (tools/patches.py _cam_stick, _cam_pitch, _cam_pitch_aim; runtime/hle/input.c): the camera's
    # stick input, which every camera action reads, and the square that the chase-like cameras apply to its vertical
    # part
    Site('cam_stick', 'code', 'd100c3ff fd000be8 f90017f3 a901fbfd 910063fd aa0003f3 f900001f 52800060', 0),
    Site('cam_pitch_chase', 'code', '1e208803 1e202008 bd41ce80 1e23cc2b bd41d281 1e212000 54000001/ff00001f '
         '52800039', 0),
    Site('cam_pitch_surf', 'code', '1e208803 1e202008 bd430260 1e23cc2d bd430661 1e212000 54000001/ff00001f '
         '2f00e400', 0),
    Site('cam_pitch_wakeboard', 'code', '1e208803 1e202008 bd433260 1e23cc2f bd433661 1e212000 54000001/ff00001f '
         '2f00e400', 0),
    # CameraAiming (the bow, its zoom, the scope): the load of its latStickScale for the vertical part
    Site('cam_pitch_aim', 'code', 'f9400268/ffc003ff 1e200923 910083e0 910063a1 bd400101 1e210860 bd4057e1 1e210809',
         4),
    # ... and CameraAiming2's (the camera for a held item)
    Site('cam_pitch_item', 'code', '94000000/fc000000 1e20400b 94000000/fc000000 f9400268/ffc003ff 1e200960 bd400101 '
         '90000008/9f00001f 1e210800', 5),
    # the cameras' controller turning its right stick around for the game's Camera X-axis / Y-axis options (Inverted:
    # the flags CameraRLReverse, CameraUpDownReverse): the calls that read them
    Site('cam_reverse_x', 'code', '94000000/fc000000 36000000/fff8001f bd413260 1e214000 bd013260 2a1f03e0 '
         '94000000/fc000000 36000000/fff8001f', 0),
    Site('cam_reverse_y', 'code', '94000000/fc000000 36000000/fff8001f bd413660 1e214000 bd013660 a9424ff4 a9417bfd '
         '9100c3ff', 0),
    # sead::CoreInfo::configure (runtime/hle/cores.c): the core count (a constant 3 on the console) and the core table,
    # replaced by the host's game cores (BOTW-JOBS-NATIVE.md)
    Site('core_info_configure', 'code', '90000008/9f00001f 52800069 b9000109/ffc003ff 9000000a/9f00001f '
         '9100014a/ffc003ff d2c0002b 52800048 f900014b', 0),
    # ksys::VFR per core with N slots (runtime/hle/vfr.c, tools/patches.py _vfr_*): its getters and per-core methods,
    # and where its own code walks all three cores; sead's core-id TLS slot variable (the getters' core)
    Site('vfr_get_delta_frame', 'code', 'a9be7bfd f9000bf3 910003fd 90000008/9f00001f aa0003f3 f9400108/ffc003ff b9400100 '
         '94000000/fc000000 b4000000/ff00001f 51000400 b9001fa0 91042268', 0),
    Site('vfr_get_raw_delta_frame', 'code', 'a9be7bfd f9000bf3 910003fd 90000008/9f00001f aa0003f3 f9400108/ffc003ff b9400100 '
         '94000000/fc000000 b4000000/ff00001f 51000400 b9001fa0 9103c268', 0),
    Site('vfr_get_delta_time', 'code', 'a9be7bfd f9000bf3 910003fd 90000008/9f00001f aa0003f3 f9400108/ffc003ff b9400100 '
         '94000000/fc000000 b4000000/ff00001f 51000400 b9001fa0 9104e268', 0),
    Site('vfr_get_interval', 'code', 'a9be7bfd f9000bf3 910003fd 90000008/9f00001f aa0003f3 f9400108/ffc003ff b9400100 '
         '94000000/fc000000 b4000000/ff00001f 51000400 b9001fa0 91012268', 0),
    Site('vfr_get_ratio', 'code', 'a9be7bfd f9000bf3 910003fd 90000008/9f00001f aa0003f3 f9400108/ffc003ff b9400100 '
         '94000000/fc000000 b4000000/ff00001f 51000400 b9001fa0 91018268 b9401fa9 f9400bf3', 0),
    Site('vfr_get_ratio_time', 'code', 'a9be7bfd f9000bf3 910003fd 90000008/9f00001f aa0003f3 f9400108/ffc003ff b9400100 '
         '94000000/fc000000 b4000000/ff00001f 51000400 b9001fa0 91018268 b9401fa9 bd415a61', 0),
    Site('vfr_use_buffer_b', 'code', 'a9be7bfd f9000bf3 910003fd 90000008/9f00001f aa0003f3 f9400108/ffc003ff b9400100 '
         '94000000/fc000000 b4000000/ff00001f 51000400 b9001fa0 9102f268', 0),
    Site('vfr_use_buffer_a', 'code', 'a9be7bfd f9000bf3 910003fd 90000008/9f00001f aa0003f3 f9400108/ffc003ff b9400100 '
         '94000000/fc000000 b4000000/ff00001f 51000400 b9001fa0 91023268', 0),
    Site('vfr_set_min_delta', 'code', 'b9407808 7100051f 5400000b/ff00001f f940400b 1e2e1000 aa1f03e9 b940004a 9100116b', 0),
    Site('vfr_get_and_set_min', 'code', 'fc1c0fe8 a9017bfd a90257f6 a9034ff4 910043fd 90000008/9f00001f aa0003f3 f9400108/ffc003ff '
         '2a0303f4', 0),
    Site('vfr_set_delta', 'code', 'd100c3ff fd000be8 f90017f3 a901fbfd 910063fd 90000008/9f00001f aa0003f3 1e204008 f9419108 '
         'b9400100 94000000/fc000000 b4000000/ff00001f 51000400 90000009/9f00001f', 0),
    Site('vfr_update_interval', 'code', 'a9be7bfd a9014ff4 910003fd 2d518400 aa0003f3 3940a408 bd00c001 7100011f', 0),
    Site('vfr_update_spread', 'code', 'bd409a60 bd009e60 bd40a661 bd40b262 bd408e63 bd00aa61 bd00b662 bd009263 bd00a260 bd00ae61 '
         'bd00ba62 bd009663 a9414ff4', 0),
    Site('vfr_init_spread', 'code', 'bd409a60 bd009e60 bd40a661 bd40b262 bd408e63 bd403e64 b9403668 bd00aa61', 0),
    Site('vfr_reset_spread', 'code', 'bd409a60 bd009e60 bd40a661 bd40b262 bd408e63 bd00aa61 bd00b662 bd009263 bd00a260 bd00ae61 '
         'bd00ba62 bd009663 f9400bf3', 0),
    Site('core_id_tls', 'got', '90000008/9f00001f aa0003f3 f9400108/ffc003ff b9400100 94000000/fc000000 b4000000/ff00001f '
         '51000400 b9001fa0 91042268', 2),
    # the command contexts per core (BOTW-JOBS-NATIVE.md #8; tools/patches.py _ctx_*, runtime/hle/percore.c): each
    # place that computes a context's address, the constructor's loop, the walks over three or four contexts
    Site('ctx_ctor_base', 'code', '9100c278 5280071a 5280141b 52805108 aa1403e1 9b2862f6 aa1603e0 94000000/fc000000', 0),
    Site('ctx_ctor_sys', 'code', '71000f9f 54000001/ff00001f 910006fc 92407f97 7100139f 54000001/ff00001f 14000000/fc000000 '
         'd10063a8', 0),
    Site('ctx_ctor_count', 'code', '7100139f 54000001/ff00001f 14000000/fc000000 d10063a8 52820000 aa1403e1 52800082 2a1f03e3', 0),
    Site('ctx_set_all', 'code', 'b9403008 f9057801 7100091f f9401c09 52802808 9a9f810a 8b0a0129 f9004521', 0),
    Site('ctx_get_a', 'code', '9100c101 12000388 3900c3e8 12000368 3900a3e8 94000000/fc000000 a9494ff4 a94857f6', 0),
    Site('ctx_get_b', 'code', '52800084 2a1503e5 2a1403e6 b90023fb f9000bfa b9000bf8 b90003ff 9b286528 52800029 390063e9 '
         '9100c101', 10),
    Site('ctx_get_c', 'code', '52800044 2a1503e5 2a1403e6 b90023fb f9000bfa b9000bf8 b90003ff 9b286528 52800029 390063e9 '
         '9100c101', 10),
    Site('ctx_get_d', 'code', '9100c101 12000368 3900c3e8 12000348 3900a3e8 94000000/fc000000 a9494ff4 f9402bfb', 0),
    Site('ctx_sys_a', 'code', '911f2001 3940a3ab f9000be7 52800024 2a0803e7 390063ff b90003ff 3900a3ea', 0),
    Site('ctx_sys_b', 'code', '911f2001 b9000be6 3940a3ab 52800084 2a0803e6 3940c3ac 390063ff b90003ff', 0),
    Site('ctx_sys_c', 'code', '911f2001 b90023e7 f9000be6 52800104 2a0503e6 2a0803e7 390063ff b90003ff', 0),
    Site('ctx_sys_d', 'code', '911f2001 1a9f1584 f9000be7 b90003e6 2a0803e6 3900a3ea 5280002a 52800027', 0),
    Site('ctx_check_all', 'code', 'a9be7bfd f9000bf3 910003fd aa0003f3 b5000001/ff00001f 9100c261 aa1303e0 94000000/fc000000', 0),
    Site('ctx_check_one', 'call', '94000000/fc000000 37000000/fff8001f f9400bf3 2a1f03e0 a8c27bfd d65f03c0 910ae261 aa1303e0', 0),
    Site('ctx_sys_count', 'code', 'b947ca6a 93407ee9 9b385920 f943ea6b 6b08015f 9a9f8128 9b182d01 94000000/fc000000', 0),
    Site('ctx_sys_entries', 'code', 'f943ea6b 6b08015f 9a9f8128 9b182d01 94000000/fc000000 910006e8 2a0002b5 92407d17', 0),
    Site('ctx_reset_base', 'code', '9b280037 910003e8 aa0003f3 91005108 910003e0 91004129 a90023e9 52808008', 0),
    Site('ctx_reset_0', 'code', '94000000/fc000000 aa1403e0 52800021 94000000/fc000000 aa1403e0 52800041 94000000/fc000000 '
         'aa1403e0 52800061 94000000/fc000000 aa1303e0', 0),
    Site('ctx_reset_1', 'code', '94000000/fc000000 aa1403e0 52800041 94000000/fc000000 aa1403e0 52800061 94000000/fc000000 '
         'aa1303e0', 0),
    Site('ctx_reset_2', 'code', '94000000/fc000000 aa1403e0 52800061 94000000/fc000000 aa1303e0 94000000/fc000000 79561a82 '
         'aa1603e0', 0),
    Site('ctx_reset_3', 'code', '94000000/fc000000 aa1303e0 94000000/fc000000 79561a82 aa1603e0 aa1a03e1 94000000/fc000000 '
         'aa1603e0', 0),
    Site('ctx_reset_fn', 'call', '94000000/fc000000 aa1403e0 52800021 94000000/fc000000 aa1403e0 52800041 94000000/fc000000 '
         'aa1403e0 52800061 94000000/fc000000 aa1303e0', 0),
    Site('ctx_size_base', 'code', '9100c013 f9419108 b9400100 94000000/fc000000 b4000000/ff00001f 51000400 b9001fa0 52805109', 0),
    Site('ctx_flag_base', 'code', '9b284d28 f9400bf3 b942ad08 12000100 a8c27bfd d65f03c0 94000000/fc000000 14000000/fc000000', 0),
    # the command memory arenas per core (BOTW-JOBS-NATIVE.md #10; tools/patches.py _arena_*, runtime/hle/percore.c):
    # the setup and its callees, the per-frame reset, the destructor's frees, the two accessors' record bases
    Site('arena_setup', 'code', 'd10343ff a9077bfd a9086ffc a90967fa a90a5ff8 a90b57f6 a90c4ff4 9101c3fd 91000c28', 0),
    Site('arena_malloc', 'call', '94000000/fc000000 f9003ee0 b5000013/ff00001f 7100069f 5400000a/ff00001f a94c4ff4 a94b57f6 '
         'a94a5ff8', 0),
    Site('arena_pool_init', 'call', '94000000/fc000000 d10063a0 aa1403e1 aa1f03e2 94000000/fc000000 90000008/9f00001f f94012e0 '
         'f9400108/ffc003ff', 0),
    Site('arena_pool_addr', 'call', '94000000/fc000000 90000008/9f00001f f94012e0 f9400108/ffc003ff f9400108 d63f0100 aa0003f9 '
         'aa1403e0', 0),
    Site('arena_pool_cpu', 'got', '90000008/9f00001f f94012e0 f9400108/ffc003ff f9400108 d63f0100 aa0003f9 aa1403e0 f90017f4', 2),
    Site('arena_pool_offset', 'call', '94000000/fc000000 f9400ee9 d342fd28 7100051f 5400000b/ff00001f 8b000328 d3428529 5285252a', 0),
    Site('arena_blocks_alloc', 'call', '94000000/fc000000 f9403ee8 2a1f03f8 aa1603f5 b9408afb 29156aff f9000bfa f94017f9', 0),
    Site('arena_buffers_alloc', 'call', 'aa1503e2 52800103 94000000/fc000000 14000000/fc000000', 2),
    Site('arena_reset', 'code', 'f9408c08 b9409809 eb09011f 54000002/ff00001f aa0903e8 f9008c09 f9409009 b900981f', 0),
    Site('arena_free_0', 'code', 'f9404808 aa0003f3 b4000008/ff00001f d1002100 94000000/fc000000 f9004a7f b9008a7f f9405e68', 0),
    Site('arena_free_1', 'code', 'f9405e68 b4000008/ff00001f d1002100 94000000/fc000000 f9005e7f b900b27f f9407268 '
         'b4000008/ff00001f', 0),
    Site('arena_free_2', 'code', 'f9407268 b4000008/ff00001f d1002100 94000000/fc000000 f900727f b900da7f f9408668 '
         'b5000008/ff00001f', 0),
    Site('arena_free_fn', 'call', '94000000/fc000000 f9004a7f b9008a7f f9405e68 b4000008/ff00001f d1002100 94000000/fc000000 '
         'f9005e7f', 0),
    Site('arena_begin_rec', 'code', '9b285b28 394446ca 6b18013f b940890b f940490c 1a9f8309 6b0a017f 5280030b', 0),
    Site('arena_end_rec', 'code', '9b294d13 aa1403e0 94000000/fc000000 b9409a68 0b080008 11000d08 121e7508 b9009a68', 0),
    # the per-core state picking a 0x1e0-byte entry (BOTW-JOBS-NATIVE.md #9; tools/patches.py _cstate_*):
    # the constructor's zeroing, and each accessor's base and bound
    Site('cstate_ctor_a', 'code', 'b91a727f f90d367f f9400bf3 a8c27bfd d65f03c0', 0),
    Site('cstate_ctor_b', 'code', 'f90d367f f9400bf3 a8c27bfd d65f03c0', 0),
    Site('cstate_set_base', 'code', '8b080288 a9417bfd 8b09090a 71000d3f 9a883148 b9000113 a9424ff4 9100c3ff', 0),
    Site('cstate_set_bound', 'code', '71000d3f 9a883148 b9000113 a9424ff4 9100c3ff d65f03c0 94000000/fc000000 14000000/fc000000', 0),
    Site('cstate_get_base', 'code', '8b080268 f9400bf3 8b09090a 71000d3f 9a883148 b9400100 a8c27bfd d65f03c0', 0),
    Site('cstate_get_bound', 'code', '71000d3f 9a883148 b9400100 a8c27bfd d65f03c0 94000000/fc000000 14000000/fc000000', 0),
    Site('cstate_slot_base', 'code', '8b080268 8b09090a 71000d3f 52800109 9a883148 5280014a b9400108 7100091f', 0),
    Site('cstate_slot_bound', 'code', '71000d3f 52800109 9a883148 5280014a b9400108 7100091f 9a890149 7100051f', 0),
    # the job manager's per-core scratch (BOTW-JOBS-NATIVE.md #25; tools/patches.py _jobs_*, runtime/hle/percore.c):
    # the constructor's zeroing, the init's allocations, the destructor's frees, both accessors' base and bound
    Site('jobs_ctor_marks', 'code', 'b904527f a9417bfd f902267f f9022e7f f902327f f902367f f9023a7f b9047a7f', 0),
    Site('jobs_ctor_marks_b', 'code', 'f902267f f9022e7f f902327f f902367f f9023a7f b9047a7f f94013f3 9100c3ff', 0),
    Site('jobs_ctor_bufs', 'code', 'f9022e7f f902327f f902367f f9023a7f b9047a7f f94013f3 9100c3ff d65f03c0', 0),
    Site('jobs_ctor_bufs_b', 'code', 'f902327f f902367f f9023a7f b9047a7f f94013f3 9100c3ff d65f03c0', 0),
    Site('jobs_ctor_bufs_c', 'code', 'f902367f f9023a7f b9047a7f f94013f3 9100c3ff d65f03c0', 0),
    Site('jobs_alloc_0', 'code', 'f9022ea0 aa1603e0 aa1703e1 f94002c8 52800202 f9401908 d63f0100 f90232a0', 0),
    Site('jobs_alloc_1', 'code', 'd63f0100 f90232a0 aa1603e0 aa1703e1 f94002c8 52800202 f9401908 d63f0100', 0),
    Site('jobs_alloc_1s', 'code', 'f90232a0 aa1603e0 aa1703e1 f94002c8 52800202 f9401908 d63f0100 f90236a0', 0),
    Site('jobs_alloc_2', 'code', 'd63f0100 f90236a0 b4000014/ff00001f f900dab4 90000008/9f00001f f9400108/ffc003ff b9400100 '
         '94000000/fc000000', 0),
    Site('jobs_alloc_2s', 'code', 'f90236a0 b4000014/ff00001f f900dab4 90000008/9f00001f f9400108/ffc003ff b9400100 '
         '94000000/fc000000 2a1f03e1', 0),
    Site('jobs_free_0', 'code', 'f9422e60 b4000000/ff00001f 94000000/fc000000 f9022e7f f9423260 b4000000/ff00001f '
         '94000000/fc000000 f902327f', 0),
    Site('jobs_free_1', 'code', 'f9423260 b4000000/ff00001f 94000000/fc000000 f902327f f9423660 b4000000/ff00001f '
         '94000000/fc000000 f902367f', 0),
    Site('jobs_free_2', 'code', 'f9423660 b4000000/ff00001f 94000000/fc000000 f902367f 9109c260 94000000/fc000000 '
         '90000008/9f00001f 9102a260', 0),
    Site('jobs_free_fn', 'call', '94000000/fc000000 f9022e7f f9423260 b4000000/ff00001f 94000000/fc000000 f902327f f9423660 '
         'b4000000/ff00001f', 0),
    Site('jobs_mark_base', 'code', '91112288 b85fc3a9 8b09090a 71000d3f 9a9f3129 9a88314a b940014a 6b13015f', 0),
    Site('jobs_mark_bound', 'code', '71000d3f 9a9f3129 9a88314a b940014a 6b13015f 5400000a/ff00001f b8297913 a9424ff4', 0),
    Site('jobs_buf_base', 'code', '91116013 f9419108 b9400100 94000000/fc000000 b4000000/ff00001f 51000400 b9001fa0 b9401fa8', 0),
    Site('jobs_buf_bound', 'code', '71000d1f 9a9f3108 f8687a60 f9400bf3 a8c27bfd d65f03c0 94000000/fc000000 14000000/fc000000', 0),
    # the per-core 0xd50-byte records (BOTW-JOBS-NATIVE.md #17; tools/patches.py _rec17_new, runtime/hle/percore.c):
    # their maker (`new T[3]`), the allocator and its tag's GOT slot (an SDK import: read at run time), the parts' constructors
    Site('rec17_new', 'code', 'a9bc7bfd f9000bf7 a90257f6 a9034ff4 910003fd 90000003/9f00001f aa0003f3 f9400063/ffc003ff '
         '5284ff00', 0),
    Site('rec17_alloc', 'call', '94000000/fc000000 b4000000/ff00001f 52800077 91002015 f9000017 aa0003f4 aa1503e0 '
         '94000000/fc000000', 0),
    Site('rec17_alloc_tag', 'ref', '90000003/9f00001f aa0003f3 f9400063/ffc003ff 5284ff00 52800102 94000000/fc000000 '
         'b4000000/ff00001f 52800077', 2),
    Site('rec17_init_a', 'call', '94000000/fc000000 9101c280 910ac296 94000000/fc000000 91064280 94000000/fc000000 aa1603e0 '
         '94000000/fc000000', 0),
    Site('rec17_init_b', 'call', '94000000/fc000000 91064280 94000000/fc000000 aa1603e0 94000000/fc000000 910f4280 '
         '94000000/fc000000 91140280', 0),
    Site('rec17_init_c', 'call', '94000000/fc000000 910f4280 94000000/fc000000 91140280 94000000/fc000000 9118c280 '
         '94000000/fc000000 911d8280', 0),
    Site('rec17_init_d', 'call', '94000000/fc000000 91140280 94000000/fc000000 9118c280 94000000/fc000000 911d8280 '
         '94000000/fc000000 91224280', 0),
    # the per-core draw contexts' count (BOTW-JOBS-NATIVE.md #16 / #18; tools/patches.py drawctx_count): a literal 3 in
    # the descriptor f_700105bf6c builds; the object sizes its per-core arrays (+0x10, +0x20, +0x1378) by it
    Site('drawctx_count', 'code', '52800068 b9005be8 94000000/fc000000 910043e1 aa0003f8 94000000/fc000000 f90016d8 d503201f', 0),
    # the per-core counters and busy bytes (BOTW-JOBS-NATIVE.md #2 / #3; tools/patches.py _cnt_*, _busy_*): #2's accessor,
    # its per-frame sum and clear, its init's zeroing; #3's two accessors, the constructor's zeroing, the system core's reader
    Site('cnt_add_base', 'code', '91100016 2a0203f4 b9400100 2a0103f5 94000000/fc000000 b4000000/ff00001f 51000400 b81fc3a0', 0),
    Site('cnt_add_bound', 'code', '71000d1f 9a9f3108 b8687ac9 0b150129 b8287ac9 36000014/fff8001f 52800028 39108268', 0),
    Site('cnt_sum_a', 'code', 'b9440276 b9440677 b9440a78 b9040e68 94000000/fc000000 b4000000/ff00001f b9440e69 0b1602e8', 0),
    Site('cnt_sum_b', 'code', 'b9440677 b9440a78 b9040e68 94000000/fc000000 b4000000/ff00001f b9440e69 0b1602e8 aa0003f4', 0),
    Site('cnt_sum_c', 'code', 'b9440a78 b9040e68 94000000/fc000000 b4000000/ff00001f b9440e69 0b1602e8 aa0003f4 0b180108', 0),
    Site('cnt_clear_a', 'code', 'b9000abf f90002bf a94757f6 3910827f a9484ff4 a9465ff8 a94567fa a9446ffc', 0),
    Site('cnt_clear_b', 'code', 'f90002bf a94757f6 3910827f a9484ff4 a9465ff8 a94567fa a9446ffc a9437bfd', 0),
    Site('cnt_init_a', 'code', 'f902027f f902067f f9020a7f b9041a7f 94000000/fc000000 a9414ff4 7100281f 1a9f17e8', 0),
    Site('cnt_init_b', 'code', 'f902067f f9020a7f b9041a7f 94000000/fc000000 a9414ff4 7100281f 1a9f17e8 90000009/9f00001f', 0),
    Site('busy_init', 'code', 'b800b2bf f9400bf5 b911da68 a9424ff4 a8c37bfd d65f03c0', 0),
    Site('busy_inc_bound', 'code', '7100111f 9a9f3108 8b080268 f9400bf3 3869690a 1100054a 3829690a a8c27bfd', 0),
    Site('busy_inc_base', 'code', '8b080268 f9400bf3 3869690a 1100054a 3829690a a8c27bfd d65f03c0 94000000/fc000000', 0),
    Site('busy_dec_bound', 'code', '7100111f 9a9f3108 8b080268 f9400bf3 3869690a 5100054a 3829690a a8c27bfd', 0),
    Site('busy_dec_base', 'code', '8b080268 f9400bf3 3869690a 5100054a 3829690a a8c27bfd d65f03c0 94000000/fc000000', 0),
    Site('busy_sys', 'code', '38e86a68 7100011f 5400000d/ff00001f f9400bf3 52800020 a8c27bfd d65f03c0 f9400bf3', 0),
    # the per-core 0xe8-byte objects (BOTW-JOBS-NATIVE.md #7; tools/patches.py _obj7_*, runtime/hle/percore.c): the
    # constructor (and its vtable and callees), the destructor's loads and stores, the accessor's base and bound
    Site('obj7_ctor', 'code', 'a9bd7bfd f9000bf5 a9024ff4 910003fd 90000008/9f00001f 91000108/ffc003ff f9000008 aa0003f3 '
         '94000000/fc000000 52801d14', 0),
    Site('obj7_vtable', 'ref', '90000008/9f00001f 91000108/ffc003ff f9000008 aa0003f3 94000000/fc000000 52801d14 aa0003e1 '
         '52840002', 1),
    Site('obj7_heap', 'call', '94000000/fc000000 52801d14 aa0003e1 52840002 aa1403e0 94000000/fc000000 aa0003f5 '
         '94000000/fc000000', 0),
    Site('obj7_new', 'call', '94000000/fc000000 aa0003f5 94000000/fc000000 f9000675 94000000/fc000000 aa0003e1 aa1403e0 '
         '52840002', 0),
    Site('obj7_init', 'call', '94000000/fc000000 f9000675 94000000/fc000000 aa0003e1 aa1403e0 52840002 94000000/fc000000 '
         'aa0003f5', 0),
    Site('obj7_free_0', 'code', 'f9400414 aa0003f3 90000008/9f00001f 91000108/ffc003ff f9000008 b4000014/ff00001f 91016280 '
         '94000000/fc000000', 0),
    Site('obj7_free_1', 'code', 'f9400a74 f900067f b4000014/ff00001f 91016280 94000000/fc000000 aa1403e0 94000000/fc000000 '
         'f9400e74', 0),
    Site('obj7_free_1s', 'code', 'f900067f b4000014/ff00001f 91016280 94000000/fc000000 aa1403e0 94000000/fc000000 f9400e74 '
         'f9000a7f', 0),
    Site('obj7_free_2', 'code', 'f9400e74 f9000a7f b4000014/ff00001f 91016280 94000000/fc000000 aa1403e0 94000000/fc000000 '
         'f9000e7f', 0),
    Site('obj7_free_2s', 'code', 'f9000a7f b4000014/ff00001f 91016280 94000000/fc000000 aa1403e0 94000000/fc000000 f9000e7f '
         'a9414ff4', 0),
    Site('obj7_free_3s', 'code', 'aa1403e0 94000000/fc000000 f9000e7f a9414ff4 a8c27bfd d65f03c0', 2),
    Site('obj7_fini', 'call', '94000000/fc000000 aa1403e0 94000000/fc000000 f9400a74 f900067f b4000014/ff00001f 91016280 '
         '94000000/fc000000', 0),
    Site('obj7_delete', 'call', '94000000/fc000000 f9400a74 f900067f b4000014/ff00001f 91016280 94000000/fc000000 aa1403e0 '
         '94000000/fc000000', 0),
    Site('obj7_get_base', 'code', '91002268 b9401fa9 f9400bf3 8b090d0a 71000d3f 9a883148 f9400100 a8c27bfd', 0),
    Site('obj7_get_bound', 'code', '71000d3f 9a883148 f9400100 a8c27bfd d65f03c0 94000000/fc000000 14000000/fc000000', 0),
    # the per-core watchdog entries (BOTW-JOBS-NATIVE.md #20; tools/patches.py _wd_*): every `obj + core x scale` the four
    # accessors form, the constructor's ticks, the init's resets; and the core -> index map (#28: identity up to 6)
    Site('wd_a_flag', 'code', '8b204268 52987989 38696908 35000008/ff00001f 2a0003f4 8b140e75 94000000/fc000000 8b140a68', 0),
    Site('wd_a_tick', 'code', '8b140e75 94000000/fc000000 8b140a68 52987809 5298750b 5298610c f86b6aab b8a9690a', 0),
    Site('wd_a_state', 'code', '8b140a68 52987809 5298750b 5298610c f86b6aab b8a9690a cb0b000b 8b0a0e6a', 0),
    Site('wd_b_flag', 'code', '8b204268 52987989 38696908 35000008/ff00001f 2a0003e8 5298780a 8b080a69 5280002b', 0),
    Site('wd_b_state', 'code', '8b080a69 5280002b 8b080e76 b82a692b 94000000/fc000000 52987508 f8286ac0 b4000014/ff00001f', 0),
    Site('wd_b_tick', 'code', '8b080e76 b82a692b 94000000/fc000000 52987508 f8286ac0 b4000014/ff00001f 394002a8 '
         '36000008/fff8001f', 0),
    Site('wd_c_flag', 'code', '8b204288 52987989 38696908 35000008/ff00001f 2a0003e8 110016a9 8b080a8a 5298780b', 0),
    Site('wd_c_state', 'code', '8b080a8a 5298780b 8b080e94 b82b6949 94000000/fc000000 52987508 f8286a80 94000000/fc000000', 0),
    Site('wd_c_tick', 'code', '8b080e94 b82b6949 94000000/fc000000 52987508 f8286a80 94000000/fc000000 b9402268 f9400e74', 0),
    Site('wd_d_flag', 'code', '8b204268 52987989 38696908 35000008/ff00001f 2a0003e8 5298780a 8b080a69 5280012b', 0),
    Site('wd_d_state', 'code', '8b080a69 5280012b 8b080e7c b82a692b 94000000/fc000000 52987508 f8286b80 3943e768', 0),
    Site('wd_d_tick', 'code', '8b080e7c b82a692b 94000000/fc000000 52987508 f8286b80 3943e768 7100051f 54000001/ff00001f', 0),
    Site('wd_d_flag2', 'code', '8b284269 52800028 aa1903e3 2a1a03e4 f90003e9 382a6928 94000000/fc000000 f94003e9', 0),
    Site('wd_ctor_tick0', 'code', 'f90002e0 94000000/fc000000 f90006e0 94000000/fc000000 52988288 92827fe9 f9000ae0 8b080268', 0),
    Site('wd_ctor_tick1', 'code', 'f90006e0 94000000/fc000000 52988288 92827fe9 f9000ae0 8b080268 8b09010a b1014129', 0),
    Site('wd_ctor_tick2', 'code', 'f9000ae0 8b080268 8b09010a b1014129 7928095f b914015f 54000001/ff00001f 52828608', 0),
    Site('wd_tick_fn', 'call', '94000000/fc000000 f90002e0 94000000/fc000000 f90006e0 94000000/fc000000 52988288 92827fe9 '
         'f9000ae0', 0),
    Site('wd_init_state_base', 'code', '8b0802c8 b9000909 f9000109 52987988 52800a02 2a1f03e1 8b0802c8 7900011f', 0),
    Site('wd_init_state_a', 'code', 'b9000909 f9000109 52987988 52800a02 2a1f03e1 8b0802c8 7900011f 3900091f', 0),
    Site('wd_init_state_b', 'code', 'f9000109 52987988 52800a02 2a1f03e1 8b0802c8 7900011f 3900091f a951a788', 0),
    Site('wd_init_flag_base', 'code', '8b0802c8 7900011f 3900091f a951a788 a916a788 a952a788 a917a788 a953a788', 0),
    Site('wd_init_flag_a', 'code', '7900011f 3900091f a951a788 a916a788 a952a788 a917a788 a953a788 a918a788', 0),
    Site('wd_init_flag_b', 'code', '3900091f a951a788 a916a788 a952a788 a917a788 a953a788 a918a788 a954a788', 0),
    Site('core_index_map', 'code', 'b8687920 910083ff d65f03c0 94000000/fc000000 14000000/fc000000', 0),
    # #3's other users (BOTW-JOBS-NATIVE.md #3; tools/patches.py _busy_*): two resets of all four bytes, and the per-frame
    # sum of the three game cores' counts into the fourth (the total f_7000cd90a0 reads)
    Site('busy_reset_a', 'code', 'b800a2bf a94257f6 a9415ff8 a8c47bfd d65f03c0', 0),
    Site('busy_reset_b', 'code', 'b800229f 94000000/fc000000 9100a260 94000000/fc000000 91114260 94000000/fc000000 3900029f '
         'a9414ff4', 0),
    Site('busy_sum_a', 'code', '39402ea8 394032a9 0b080128 394036a9 0b090108 39003aa8 394006a8 34000008/ff00001f', 0),
    Site('busy_sum_b', 'code', '394032a9 0b080128 394036a9 0b090108 39003aa8 394006a8 34000008/ff00001f b94cda68', 0),
    Site('busy_sum_c', 'code', '394036a9 0b090108 39003aa8 394006a8 34000008/ff00001f b94cda68 7100011f 5400000c/ff00001f', 0),
    Site('busy_total', 'code', '39003aa8 394006a8 34000008/ff00001f b94cda68 7100011f 5400000c/ff00001f 52800028 39000aa8', 0),
    # the per-core linear allocators (BOTW-JOBS-NATIVE.md #11; tools/patches.py _lin_*, runtime/hle/percore.c): the
    # constructor, destructor, init and split (and their callees), the accessors' record bases
    Site('lin_ctor', 'code', '90000008/9f00001f 91000108/ffc003ff f900141f f9000008 91052008 b900301f f900281f b900581f', 0),
    Site('lin_dtor', 'code', 'a9be7bfd f9000bf3 910003fd 90000008/9f00001f 91000108/ffc003ff f900141f aa0003f3 f9000008 '
         '91014000', 0),
    Site('lin_init', 'code', 'd10183ff a9027bfd f9001bf7 a90457f6 a9054ff4 910083fd aa0003f4 b9003003', 0),
    Site('lin_split', 'code', '3945000a 5280002b aa0003ec b9403008 52955569 72b55549 f100003f 4b0a016a', 0),
    Site('lin_vtable', 'ref', '90000008/9f00001f 91000108/ffc003ff f900141f f9000008 91052008 b900301f f900281f b900581f', 1),
    Site('lin_lock_ctor', 'call', 'f900881f aa0803e0 14000000/fc000000', 2),
    Site('lin_lock_dtor', 'call', '91052260 f9400bf3 a8c27bfd 14000000/fc000000', 3),
    Site('lin_alloc_make', 'call', '94000000/fc000000 a940a7e8 aa1403f7 52810001 a98526e8 f9400fe8 aa1703e0 f9000ae8', 0),
    Site('lin_alloc_init', 'call', '94000000/fc000000 b9403288 b9003af6 b81fc2e8 35000016/ff00001f 910062e8 a9017d1f a9007d1f', 0),
    Site('lin_alloc_fini', 'call', '94000000/fc000000 f9403660 b4000000/ff00001f 94000000/fc000000 f900367f f9403e60 '
         'b4000000/ff00001f 94000000/fc000000', 0),
    Site('lin_malloc', 'call', '94000000/fc000000 a9068280 b9408a80 aa1503e1 52802002 94000000/fc000000 a9078280 910023e8', 0),
    Site('lin_free', 'call', '94000000/fc000000 f900367f f9403e60 b4000000/ff00001f 94000000/fc000000 f9003e7f 9102a260 '
         '94000000/fc000000', 0),
    Site('lin_after', 'call', 'aa1303e3 94000000/fc000000 a9454ff4 f9401bf7 a94457f6 a9427bfd 910183ff d65f03c0', 1),
    Site('lin_rec_a', 'code', '9b280028 f9402109 91007d29 927be929 f9002109 14000000/fc000000', 0),
    Site('lin_rec_b', 'code', '52800b08 9b280028 91010100 d65f03c0', 1),
    Site('lin_rec_c', 'code', '9b295108 f9402909 b940590a f9403108 f9000269 b9000a6a f9000a68 a9424ff4', 0),
    Site('lin_rec_d', 'code', '9b280028 b9404909 b940410a b9403908 4b0a0108 0b080120 d65f03c0', 0),
    Site('lin_rec_e', 'code', '9100e008 52800b0a 9b2a202a 93407c29 b940094c b940014d b940114b 4b0c01ac', 0),
    # the per-core pointer lists and bytes in the game's data (BOTW-JOBS-NATIVE.md #24; tools/patches.py _list24_*): the
    # static init's base, every accessor's base and bound
    Site('list24_init', 'code', '91246129 f9048100 90000008/9f00001f f900011f/ffc003ff 90000008/9f00001f d2c0200a f904092a '
         'b900011f/ffc003ff', 0),
    Site('list24_a_base', 'code', '91246108 52810209 9ba92029 71000c3f 9a883128 b9400109 7100053f 5400000b/ff00001f', 0),
    Site('list24_a_bound', 'code', '71000c3f 9a883128 b9400109 7100053f 5400000b/ff00001f f940050b f940016a eb00015f', 0),
    Site('list24_b_bytes', 'code', '910522d6 90000009/9f00001f 91000129/ffc003ff 71000d1f 9a9f310a 386a6ad7 35000017/ff00001f '
         '5280002b', 0),
    Site('list24_b_base', 'code', '91246129 71000d1f 9a9f310a 386a6ad7 35000017/ff00001f 5280002b 71000d1f 382a6acb', 0),
    Site('list24_b_bound1', 'code', '71000d1f 9a9f310a 386a6ad7 35000017/ff00001f 5280002b 71000d1f 382a6acb 5281020a', 0),
    Site('list24_b_bound2', 'code', '71000d1f 382a6acb 5281020a 9baa250a 9a893148 b900011f b9400688 5281020a', 0),
    Site('list24_b_bound3', 'code', '71000d1f 9baa250a 9a893148 b9400109 7100053f 5400000b/ff00001f f940050a aa0903eb', 0),
    Site('list24_b_bound4', 'code', '71000d1f 9a9f3108 38286adf f940d268 b4000008/ff00001f f9400268 aa1303e0 2a1f03e1', 0),
    Site('list24_c_base', 'code', '91246108 b85fc3a1 52810209 91178302 aa1803e0 2a1703e3 9ba92029 2a1603e4', 0),
    Site('list24_c_bound', 'code', '71000c3f a9425ff8 9a883128 a9417bfd b900011f 910143ff 14000000/fc000000', 0),
    Site('list24_d_base', 'code', '91246108 52810209 9ba92029 71000c3f aa0703f3 aa0603f4 2a0503fa aa0003f9', 0),
    Site('list24_d_bound', 'code', '71000c3f aa0703f3 aa0603f4 2a0503fa aa0003f9 2a0403f5 2a0303f6 aa0203f7', 0),
    # the per-core u16 counters (BOTW-JOBS-NATIVE.md #22; tools/patches.py cnt22_*): the accessor's base, the reset
    Site('cnt22_add_base', 'code', '8b080688 a9417bfd 79417109 0b130129 a9424ff4 79017109 9100c3ff d65f03c0', 0),
    Site('cnt22_reset_a', 'code', '7901781f b900b81f b900c01f d65f03c0', 0),
    Site('cnt22_reset_b', 'code', 'b900b81f b900c01f d65f03c0', 0),
    # the per-core 0x58-byte strings in the game's data (BOTW-JOBS-NATIVE.md #26; tools/patches.py _str26_*): the guarded
    # init's base and its end (the other cores' strings), the string's init call, the two accessors' bases
    Site('str26_init_base', 'code', '913082d6 f941f108 910052c9 52800819 aa1603e0 b90012d9 91004118 a90026d8', 0),
    Site('str26_init_rest', 'code', 'f90002da 94000000/fc000000 14000000/fc000000', 0),
    Site('str26_init_fn', 'call', '94000000/fc000000 f94006c8 aa1603f7 3900011f f8058ef8 90000008/9f00001f f9400108/ffc003ff '
         'aa1703e0', 0),
    Site('str26_a_base', 'code', '913082d6 b9800be8 9b295917 b9404a88 f9402a89 6b13011f 1a9f8268 f868d920', 0),
    Site('str26_b_base', 'code', '913082f7 b9800be8 9b295d08 f9400508 3900011f b94002c0 94000000/fc000000 b4000000/ff00001f', 0),
    # the per-core u32 / float pairs (BOTW-JOBS-NATIVE.md #6; tools/patches.py _pair6_*): five accessors' bases and bounds,
    # the two read-and-clear sums over the sead core count, the constructor's zeroing
    Site('pair6_a_base', 'code', '9139c288 b85fc3a9 a9417bfd 8b09090a 71000d3f 9a883148 b9400109 0b130129', 0),
    Site('pair6_a_bound', 'code', '71000d3f 9a883148 b9400109 0b130129 a9424ff4 b9000109 9100c3ff d65f03c0', 0),
    Site('pair6_b_base', 'code', '9139c268 b85fc3a9 1e220280 8b09090a 71000d3f a9424ff4 a9417bfd 9a883148', 0),
    Site('pair6_b_bound', 'code', '71000d3f a9424ff4 a9417bfd 9a883148 bd400101 5e21d821 1e212800 1e380009', 0),
    Site('pair6_c_base', 'code', '9139f268 1e280920 b9401fa9 f94013f3 8b09090a 71000d3f a9417bfd 9a883148', 0),
    Site('pair6_c_bound', 'code', '71000d3f a9417bfd 9a883148 bd400101 1e212800 bd000100 6cc323e9 d65f03c0', 0),
    Site('pair6_d_base', 'code', '9139f268 b9400fe9 f94017f3 8b09090a 71000d3f a941fbfd 9a883148 bd400100 1e282800', 0),
    Site('pair6_d_bound', 'code', '71000d3f a941fbfd 9a883148 bd400100 1e282800 fd400be8 bd000100 9100c3ff', 0),
    Site('pair6_e_base', 'code', '9139f268 b9400fe9 f94017f3 8b09090a 71000d3f a941fbfd 9a883148 bd400100 1e202900', 0),
    Site('pair6_e_bound', 'code', '71000d3f a941fbfd 9a883148 bd400100 1e202900 fd400be8 bd000100 9100c3ff', 0),
    Site('pair6_sum_u0', 'code', 'b94e7260 7100051f b90e727f 54000000/ff00001f b94e7669 71000d1f b90e767f 0b000120', 0),
    Site('pair6_sum_uz0', 'code', 'b90e727f 54000000/ff00001f b94e7669 71000d1f b90e767f 0b000120 54000003/ff00001f b94e7a68', 0),
    Site('pair6_sum_u1', 'code', 'b94e7669 71000d1f b90e767f 0b000120 54000003/ff00001f b94e7a68 b90e7a7f 0b000100', 0),
    Site('pair6_sum_uz1', 'code', 'b90e767f 0b000120 54000003/ff00001f b94e7a68 b90e7a7f 0b000100 34000000/ff00001f '
         '94000000/fc000000', 0),
    Site('pair6_sum_u2', 'code', 'b94e7a68 b90e7a7f 0b000100 34000000/ff00001f 94000000/fc000000 14000000/fc000000 a9424ff4 '
         'f9400ff5', 0),
    Site('pair6_sum_uz2', 'code', 'b90e7a7f 0b000100 34000000/ff00001f 94000000/fc000000 14000000/fc000000 a9424ff4 f9400ff5 '
         'a940fbfd', 0),
    Site('pair6_sum_f0', 'code', 'bd4e7e60 2f00e401 7100051f b90e7e7f 1e212808 54000000/ff00001f bd4e8260 71000d1f', 0),
    Site('pair6_sum_fz0', 'code', 'b90e7e7f 1e212808 54000000/ff00001f bd4e8260 71000d1f b90e827f 1e202908 54000003/ff00001f', 0),
    Site('pair6_sum_f1', 'code', 'bd4e8260 71000d1f b90e827f 1e202908 54000003/ff00001f bd4e8660 b90e867f 1e202908', 0),
    Site('pair6_sum_fz1', 'code', 'b90e827f 1e202908 54000003/ff00001f bd4e8660 b90e867f 1e202908 1e202108 54000000/ff00001f', 0),
    Site('pair6_sum_f2', 'code', 'bd4e8660 b90e867f 1e202908 1e202108 54000000/ff00001f 14000000/fc000000 f9400268 aa1303e0', 0),
    Site('pair6_sum_fz2', 'code', 'b90e867f 1e202908 1e202108 54000000/ff00001f 14000000/fc000000 f9400268 aa1303e0 f9418508', 0),
    Site('pair6_zero_a', 'code', 'f907427f a900231f 52829a88 a94357f6 f9073e7f a9417bfd f9073a7f b915c277', 0),
    Site('pair6_zero_b', 'code', 'f9073e7f a9417bfd f9073a7f b915c277 b9160a77 a9425ff8 f8286a69 a9444ff4', 0),
    Site('pair6_zero_c', 'code', 'f9073a7f b915c277 b9160a77 a9425ff8 f8286a69 a9444ff4 910143ff d65f03c0', 0),
    # the per-core flip bytes (BOTW-JOBS-NATIVE.md #21; tools/patches.py flip21_*): the accessor's base, the constructor's zeroing
    Site('flip21_base', 'code', '91026288 5280002a b9801fa9 aa1303e0 38696917 b9801fa9 38696909 b9401fb6', 0),
    Site('flip21_zero', 'code', 'b9009a9f 91002280 aa1303e3 52800021 a9434ff4 2a1f03e2 a94257f6 a9415ff8', 0),
    # the per-slot results (BOTW-JOBS-NATIVE.md #5; tools/patches.py _slot5_*, runtime/hle/percore.c): three accessors' bases
    # and bounds, the merge and the float pick (in the runtime), the floats' zeroing
    Site('slot5_rec_base', 'code', '8b0902a9 8b08192a 71000d1f 9a893148 52800029 3d800100 3dc00680 3d800500', 0),
    Site('slot5_rec_bound', 'code', '71000d1f 9a893148 52800029 3d800100 3dc00680 3d800500 3dc00a80 3d800900', 0),
    Site('slot5_u_base', 'code', '8b090289 8b08092a 71000d1f 9a893148 f9000268 f100027f a9424ff4 a9417bfd', 0),
    Site('slot5_u_bound', 'code', '71000d1f 9a893148 f9000268 f100027f a9424ff4 a9417bfd 1a9f07e0 9100c3ff', 0),
    Site('slot5_f_base', 'code', '911bc269 a9424ff4 8b08092a 71000d1f 9a893148 bd000100 9100c3ff d65f03c0', 0),
    Site('slot5_f_bound', 'code', '71000d1f 9a893148 bd000100 9100c3ff d65f03c0 94000000/fc000000 14000000/fc000000', 0),
    Site('slot5_merge', 'code', '52822e88 8b080008 39400109 34000009/ff00001f aa1f03e9 5282270a 8b0a000a 8b091949', 0),
    Site('slot5_pick', 'code', 'bd46f000 1e3e1002 1e214001 1e202008 1e21cc01 1e226823 1e222020 2f00e401', 0),
    Site('slot5_fz_a', 'code', 'b906fa7f 52800602 f9037a7f 94000000/fc000000 911cb260 52801102 2a1f03e1 911cc275', 0),
    Site('slot5_fz_b', 'code', 'f9037a7f 94000000/fc000000 911cb260 52801102 2a1f03e1 911cc275 94000000/fc000000 aa1503e0', 0),
    Site('slot5_fz_c', 'code', 'f903781f b906f81f d65f03c0', 0),
    Site('slot5_fz_d', 'code', 'b906f81f d65f03c0', 0),
    # the graphics singleton's per-core objects (BOTW-JOBS-NATIVE.md #12; tools/patches.py _obj12_*, runtime/hle/percore.c):
    # the init's three makes (the first makes every core's), the core-indexed accessors' bases
    Site('obj12_first', 'code', 'f9003ad9 52800102 94000000/fc000000 2a1f03e1 aa0003f9 94000000/fc000000 aa1803e0 aa1503e1', 0),
    Site('obj12_alloc1', 'code', '94000000/fc000000 2a1f03e1 aa0003f9 94000000/fc000000 aa1803e0 aa1503e1 f9003ed9 52800102', 0),
    Site('obj12_ctor1', 'code', '94000000/fc000000 aa1803e0 aa1503e1 f9003ed9 52800102 94000000/fc000000 2a1f03e1 aa0003f8', 0),
    Site('obj12_alloc2', 'code', '94000000/fc000000 2a1f03e1 aa0003f8 94000000/fc000000 b94007e9 b2091be8 d503201f '
         '10000003/9f00001f', 0),
    Site('obj12_ctor2', 'code', '94000000/fc000000 b94007e9 b2091be8 d503201f 10000003/9f00001f f94016c0 2a1f03e1 a901a388', 0),
    Site('obj12_alloc_fn', 'call', '94000000/fc000000 2a1f03e1 aa0003f9 94000000/fc000000 aa1803e0 aa1503e1 f9003ad9 52800102', 0),
    Site('obj12_ctor_fn', 'call', '94000000/fc000000 aa1803e0 aa1503e1 f9003ad9 52800102 94000000/fc000000 2a1f03e1 aa0003f9', 0),
    Site('obj12_a', 'code', '8b204e88 f9403914 5281c308 9b285ae8 f941e101 aa1403e0 94000000/fc000000 f94002a0', 0),
    Site('obj12_b', 'code', '8b204ee8 f941e281 f9403917 aa1703e0 94000000/fc000000 f9400300 aa1303e1 aa1703e2', 0),
    Site('obj12_c', 'code', '8b204f48 f941e261 f940391a aa1a03e0 94000000/fc000000 29428be4 aa1303e0 aa1a03e1', 0),
    Site('obj12_d', 'code', '8b204e88 f941e261 f9403916 aa1603e0 94000000/fc000000 f9400308 b4000008/ff00001f '
         '90000009/9f00001f', 0),
    Site('obj12_e', 'code', '8b204e88 f941e261 f9403900 f90007e0 94000000/fc000000 f94006a8 3954a109 34000009/ff00001f', 0),
    # the per-slot queues (BOTW-JOBS-NATIVE.md #27; tools/patches.py _q27_*, runtime/hle/percore.c): the append's and the
    # walker's bases and bounds, the init's buffers, the constructor's end (and the functions they call)
    Site('q27_add_bufs', 'code', '9104a268 8b191109 71000f3f 9a883138 b9400308 7100051f 5400000b/ff00001f 910ca268', 0),
    Site('q27_add_bound1', 'code', '71000f3f 9a883138 b9400308 7100051f 5400000b/ff00001f 910ca268 71000f3f 8b191909', 0),
    Site('q27_add_locks', 'code', '910ca268 71000f3f 8b191909 9a883137 aa1703e0 94000000/fc000000 f9400288 aa1403e0', 0),
    Site('q27_add_bound2', 'code', '71000f3f 8b191909 9a883137 aa1703e0 94000000/fc000000 f9400288 aa1403e0 f9401108', 0),
    Site('q27_add_cur', 'code', '8b190a69 b9001ba0 aa1403e0 b9401ba8 b940030b f940070c b942a12a 6b0a017f', 0),
    Site('q27_walk_locks', 'code', '910ca289 9104a288 aa1f03fc a90027e8 91096269 9108e268 a90527e8 91086269', 0),
    Site('q27_walk_bufs', 'code', '9104a288 aa1f03fc a90027e8 91096269 9108e268 a90527e8 91086269 9107e268', 0),
    Site('q27_walk_cur', 'code', '8b1c0a88 910a8116 b942a108 7100051f 5400000b/ff00001f f94003e8 2a1f03f7 8b1c1118', 0),
    Site('q27_walk_bound', 'code', 'f1000f9f 54000001/ff00001f a95e4ff4 a95d57f6 a95c5ff8 a95b67fa a95a6ffc a9597bfd', 0),
    Site('q27_init_first', 'code', 'b9012a95 f9009a80 90000003/9f00001f aa1603e0 f9400063/ffc003ff aa1303e1 52800102 '
         '94000000/fc000000', 0),
    Site('q27_init_first_b', 'code', 'f9009a80 90000003/9f00001f aa1603e0 f9400063/ffc003ff aa1303e1 52800102 94000000/fc000000 '
         'b4000000/ff00001f', 0),
    Site('q27_init_2', 'code', '94000000/fc000000 b4000000/ff00001f b9013a95 f900a280 90000003/9f00001f aa1603e0 '
         'f9400063/ffc003ff aa1303e1', 0),
    Site('q27_init_3', 'code', '94000000/fc000000 b4000000/ff00001f b9014a95 f900aa80 aa1403e0 aa1303e1 94000000/fc000000 '
         'aa1403e0', 0),
    Site('q27_alloc_fn', 'call', '94000000/fc000000 b4000000/ff00001f b9012a95 f9009a80 90000003/9f00001f aa1603e0 '
         'f9400063/ffc003ff aa1303e1', 0),
    Site('q27_alloc_tag', 'ref', '90000003/9f00001f d37d7eb6 f9400063/ffc003ff aa1603e0 aa1303e1 52800102 94000000/fc000000 '
         'b4000000/ff00001f', 2),
    Site('q27_ctor_tail', 'code', '52801082 2a1f03e1 a8c37bfd 14000000/fc000000', 3),
    Site('q27_memset_fn', 'call', '52801082 2a1f03e1 a8c37bfd 14000000/fc000000', 3),
    Site('q27_lock_ctor', 'call', '94000000/fc000000 910da260 94000000/fc000000 910ea260 94000000/fc000000 390fa27f 910a8260 '
         'f9400bf5', 0),
    # the per-core 0x160-byte records (BOTW-JOBS-NATIVE.md #23; tools/patches.py _rec23_*, runtime/hle/percore.c): the
    # constructor's end, the init's twelve stores, the walk's base and end, both accessors' bases and bounds
    Site('rec23_ctor_end', 'code', 'b908627f a9444ff4 a94357f6 f9400bf9 a9425ff8 a8c57bfd d65f03c0', 0),
    Site('rec23_init_0', 'code', 'b902ca74 b9032274 1e204100 b9037a74 b903d274 b9042a74 b9048274 b904da74', 0),
    Site('rec23_init_1', 'code', 'b9032274 1e204100 b9037a74 b903d274 b9042a74 b9048274 b904da74 b9053274', 0),
    Site('rec23_init_2', 'code', 'b9037a74 b903d274 b9042a74 b9048274 b904da74 b9053274 b9058a74 b905e274', 0),
    Site('rec23_init_3', 'code', 'b903d274 b9042a74 b9048274 b904da74 b9053274 b9058a74 b905e274 b9063a74', 0),
    Site('rec23_init_4', 'code', 'b9042a74 b9048274 b904da74 b9053274 b9058a74 b905e274 b9063a74 b9069274', 0),
    Site('rec23_init_5', 'code', 'b9048274 b904da74 b9053274 b9058a74 b905e274 b9063a74 b9069274 b9001fbf', 0),
    Site('rec23_init_6', 'code', 'b904da74 b9053274 b9058a74 b905e274 b9063a74 b9069274 b9001fbf 94000000/fc000000', 0),
    Site('rec23_init_7', 'code', 'b9053274 b9058a74 b905e274 b9063a74 b9069274 b9001fbf 94000000/fc000000 f9440e60', 0),
    Site('rec23_init_8', 'code', 'b9058a74 b905e274 b9063a74 b9069274 b9001fbf 94000000/fc000000 f9440e60 2f00e40b', 0),
    Site('rec23_init_9', 'code', 'b905e274 b9063a74 b9069274 b9001fbf 94000000/fc000000 f9440e60 2f00e40b b4000000/ff00001f', 0),
    Site('rec23_init_10', 'code', 'b9063a74 b9069274 b9001fbf 94000000/fc000000 f9440e60 2f00e40b b4000000/ff00001f f9400008', 0),
    Site('rec23_init_11', 'code', 'b9069274 b9001fbf 94000000/fc000000 f9440e60 2f00e40b b4000000/ff00001f f9400008 f9400108', 0),
    Site('rec23_walk_base', 'code', '8b17027b b942cb68 71001d1f 5400000c/ff00001f b942fb68 7100011f 1a9f17e8 0a080288', 0),
    Site('rec23_walk_end', 'code', 'f11082ff 54000001/ff00001f 14000000/fc000000 395c5a68 71001d1f 54000008/ff00001f 395d9a68 '
         '71001d1f', 0),
    Site('rec23_a_base', 'code', '910aa2a8 52802c0a b9401fa9 b9401fa9 9baa212a 71000d3f 9a883149 b9402129', 0),
    Site('rec23_a_b1', 'code', '71000d3f 9a883149 b9402129 71001d3f 5400000c/ff00001f b9401fa9 52802c0a 9baa212a', 0),
    Site('rec23_a_b2', 'code', '71000d3f 9a883149 b9407929 71001d3f 5400000c/ff00001f b9401fa9 52802c0a 9baa212a', 0),
    Site('rec23_a_b3', 'code', '71000d3f 9a883149 b940d129 71001d3f 5400000c/ff00001f b9401fa9 52802c0a 9baa212a', 0),
    Site('rec23_a_b4', 'code', '71000d3f 9a883149 b9412929 7100213f 5400000a/ff00001f a9424ff4 f9400bf5 a8c37bfd', 0),
    Site('rec23_a_b5', 'code', '2a1303e2 9bab214b 71000d5f 52800b0a 9a883168 9baa2120 94000000/fc000000 14000000/fc000000', 2),
    Site('rec23_b_base', 'code', '910aa288 52802c0a b85fc3a9 b85fc3a9 9baa212a 71000d3f 9a883149 b9402129', 0),
    Site('rec23_b_b1', 'code', '71000d3f 9a883149 b9402129 71001d3f 5400000c/ff00001f b85fc3a9 52802c0a 9baa212a', 0),
    Site('rec23_b_b2', 'code', '71000d3f 9a883149 b9407929 71001d3f 5400000c/ff00001f b85fc3a9 52802c0a 9baa212a', 0),
    Site('rec23_b_b3', 'code', '71000d3f 9a883149 b940d129 71001d3f 5400000c/ff00001f b85fc3a9 52802c0a 9baa212a', 0),
    Site('rec23_b_b4', 'code', '71000d3f 9a883149 b9412929 7100213f 5400000b/ff00001f 52800069 14000000/fc000000 '
         '94000000/fc000000', 0),
    Site('rec23_b_b5', 'code', 'aa1303e1 9bab214b 71000d5f 52800b0a 9a883168 9baa2120 94000000/fc000000 14000000/fc000000', 2),
    # the per-slot candidates (BOTW-JOBS-NATIVE.md #4; tools/patches.py _rec4*, runtime/hle/percore.c): #4a's constructor,
    # offer and pick; #4b's constructor end, add, the functions over the slots and the per-frame update's places
    Site('rec4a_ctor_0', 'code', '91380260 94000000/fc000000 9139c260 94000000/fc000000 913b8260 94000000/fc000000 912b0260 '
         '94000000/fc000000', 0),
    Site('rec4a_ctor_1', 'code', '9139c260 94000000/fc000000 913b8260 94000000/fc000000 912b0260 94000000/fc000000 aa1403e0 '
         '94000000/fc000000', 0),
    Site('rec4a_ctor_2', 'code', '913b8260 94000000/fc000000 912b0260 94000000/fc000000 aa1403e0 94000000/fc000000 aa1603e0 '
         '94000000/fc000000', 0),
    Site('rec4a_ctor_rest', 'code', '94000000/fc000000 912b0260 94000000/fc000000 aa1403e0 94000000/fc000000 aa1603e0 '
         '94000000/fc000000 aa1503e0', 0),
    Site('rec4a_reset_fn', 'call', '94000000/fc000000 912b0260 94000000/fc000000 aa1403e0 94000000/fc000000 aa1603e0 '
         '94000000/fc000000 aa1503e0', 0),
    Site('rec4a_acc_base', 'code', '91380289 9baa250a 71000d1f 9a893140 b9405a69 b9405808 6b09011f 5400000a/ff00001f', 0),
    Site('rec4a_acc_bound', 'code', '71000d1f 9a893140 b9405a69 b9405808 6b09011f 5400000a/ff00001f aa1303e1 a9424ff4', 0),
    Site('rec4a_pick', 'code', 'a9bd7bfd f9000bf5 a9024ff4 910003fd b94e5808 91380013 9139c014 b94ec809', 0),
    Site('rec4a_copy_fn', 'call', '94000000/fc000000 aa1303e0 94000000/fc000000 aa1403e0 94000000/fc000000 aa1503e0 a9424ff4 '
         'f9400bf5', 0),
    Site('rec4b_ctor_end', 'code', 'b90a027f a9434ff4 a8c47bfd d65f03c0', 0),
    Site('rec4b_acc_base', 'code', '9100a288 52805009 9ba92309 71000f1f 2a0003f5 9a883128 52800a09 7100201f', 0),
    Site('rec4b_acc_bound', 'code', '71000f1f 2a0003f5 9a883128 52800a09 7100201f 9ba92009 9a883136 39788688', 0),
    Site('rec4b_acc_a_base', 'code', '9128a288 52800a09 9ba92309 71000f1f 9a883128 b9400109 7100053f 5400000b/ff00001f', 0),
    Site('rec4b_acc_a_bound', 'code', '71000f1f 9a883128 b9400109 7100053f 5400000b/ff00001f f940050a aa0903eb f940014c', 0),
    Site('rec4b_reset', 'code', 'b900281f b900781f b900c81f b901181f b901681f b901b81f b902081f b902581f', 0),
    Site('rec4b_remove', 'code', 'd103c3ff a90a7bfd a90b67fa a90c5ff8 a90d57f6 a90e4ff4 910283fd aa0003f4', 0),
    Site('rec4b_remove_a', 'code', 'a9be7bfd a9014ff4 910003fd b94a2808 aa0103f3 aa0003f4 7100051f 5400000b/ff00001f', 0),
    Site('rec4b_remove_b', 'code', 'a9be7bfd a9014ff4 910003fd b94b1808 aa0103f3 aa0003f4 7100051f 5400000b/ff00001f', 0),
    Site('rec4b_sweep', 'code', 'a9bb7bfd f9000bf9 a9025ff8 a90357f6 a9044ff4 910003fd aa0003f3 aa1f03f7', 0),
    Site('rec4b_drop_fn', 'call', '94000000/fc000000 14000000/fc000000 11000694 7100229f 54000001/ff00001f a9444ff4 f9400bf9 '
         'a94357f6', 0),
    Site('rec4b_pick', 'code', 'fc1b0fe8 a900fbfd f9000ff9 a9025ff8 a90357f6 a9044ff4 910023fd d37e7c28', 0),
    Site('rec4b_fill_b', 'code', 'd101c3ff a9057bfd f90033f3 910143fd aa0003f3 b90b181f 9101e001 b90b681f', 0),
    Site('rec4b_filter', 'code', 'a9bc7bfd a9015ff8 a90257f6 a9034ff4 910003fd aa0003f4 aa0003f3 529fff57', 0),
    Site('rec4b_states', 'code', 'd10103ff a9017bfd a90257f6 a9034ff4 910043fd aa0003f4 aa0003f3 b8828e88', 0),
    Site('rec4b_pick_b', 'code', 'fc1c0fe8 a9017bfd a90257f6 a9034ff4 910043fd b94b1808 1e3e1008 aa0003f3', 0),
    Site('rec4b_upd_clear1', 'code', 'b9002a7f b9007a7f b900ca7f b9011a7f b9016a7f b901ba7f b9020a7f b9025a7f', 0),
    Site('rec4b_upd_any_a', 'code', 'b9407a68 35000008/ff00001f b942fa68 35000008/ff00001f b9457a68 34000008/ff00001f 39788668 '
         '37180008/fff8001f 36080008/fff8001f', 0),
    Site('rec4b_upd_keep_a', 'code', 'b947fa69 aa1503f6 d10006b5 f9440268 eb0902bf 54000002/ff00001f f875790a 14000000/fc000000', 0),
    Site('rec4b_upd_any_b', 'code', 'b9807a6a b942fa6b b9457a6c 3500000a/ff00001f 3500000b/ff00001f 3400000c/ff00001f 3978866d '
         '3718000d/fff8001f', 0),
    Site('rec4b_upd_any_c', 'code', 'b9407a68 35000008/ff00001f b942fa68 35000008/ff00001f b9457a68 34000008/ff00001f 39788668 '
         '37180008/fff8001f 39788a68', 0),
    Site('rec4b_upd_any_d', 'code', 'b9407a68 35000008/ff00001f b942fa68 35000008/ff00001f b9457a68 35000008/ff00001f 3938827f '
         'b90c467f', 0),
    Site('rec4b_upd_clear2', 'code', 'b9002a7f b9007a7f b900ca7f 121c7909 531f7908 b9011a7f b9016a7f 121d0108', 0),
    Site('rec4b_upd_other1', 'code', '93407d6b 93407d89 3400000a/ff00001f f940426c d37df14a d100214d f840858a eb0a011f', 0),
    # #4a's records in their part at +0x860 (BOTW-JOBS-NATIVE.md #4; tools/patches.py _rec4a_*): its constructor makes
    # every slot's, its destructor releases every slot's handle
    Site('rec4a_part_base', 'code', '91168275 52802a02 2a1f03e1 aa1503e0 94000000/fc000000 aa1503e0 94000000/fc000000 91184260', 0),
    Site('rec4a_part_size', 'code', '52802a02 2a1f03e1 aa1503e0 94000000/fc000000 aa1503e0 94000000/fc000000 91184260 '
         '94000000/fc000000', 0),
    Site('rec4a_part_1', 'code', '91184260 94000000/fc000000 911a0260 94000000/fc000000 911bf260 aa1403e1 b906fa7f 52800602', 0),
    Site('rec4a_part_2', 'code', '911a0260 94000000/fc000000 911bf260 aa1403e1 b906fa7f 52800602 f9037a7f 94000000/fc000000', 0),
    Site('rec4a_part_rest', 'code', '94000000/fc000000 911bf260 aa1403e1 b906fa7f 52800602 f9037a7f 94000000/fc000000 911cb260', 0),
    Site('rec4a_make_fn', 'call', '94000000/fc000000 911bf260 aa1403e1 b906fa7f 52800602 f9037a7f 94000000/fc000000 911cb260', 0),
    Site('rec4a_drop_rest', 'code', '94000000/fc000000 9119c260 94000000/fc000000 91180260 94000000/fc000000 91164260 '
         '94000000/fc000000 91098260', 0),
    Site('rec4a_release_fn', 'call', '94000000/fc000000 9119c260 94000000/fc000000 91180260 94000000/fc000000 91164260 '
         '94000000/fc000000 91098260', 0),
    Site('rec4a_drop_1', 'code', '9119c260 94000000/fc000000 91180260 94000000/fc000000 91164260 94000000/fc000000 91098260 '
         '94000000/fc000000', 0),
    Site('rec4a_drop_0', 'code', '91180260 94000000/fc000000 91164260 94000000/fc000000 91098260 94000000/fc000000 91094260 '
         '94000000/fc000000', 0),
    # #27's second append (f_700112e6d4 1.9.0, without the lock; two places): the buffers' base, the bound, the cursor
    # (BOTW-JOBS-NATIVE.md #27; tools/patches.py _q27_*)
    Site('q27_add2_bufs', 'code', '9104a269 71000ebf 8b15112a 9a893149 8b150a6a b940012c f940052d b942a14b', 0),
    Site('q27_add2_bound', 'code', '71000ebf 8b15112a 9a893149 8b150a6a b940012c f940052d b942a14b 6b0b019f', 0),
    Site('q27_add2_cur', 'code', '8b150a6a b940012c f940052d b942a14b 6b0b019f 1a9f816b 937d7d6b 382b69a8', 0),
    Site('q27_add3_bufs', 'code', '9104a269 8b0a112b 71000d5f 8b0a0a6a 9a893169 b942a14b b940012c f940052d', 0),
    Site('q27_add3_bound', 'code', '71000d5f 8b0a0a6a 9a893169 b942a14b b940012c f940052d 6b0b019f 1a9f816b', 0),
    Site('q27_add3_cur', 'code', '8b0a0a6a 9a893169 b942a14b b940012c f940052d 6b0b019f 1a9f816b 937d7d6b', 0),
    # #4c's rings (BOTW-JOBS-NATIVE.md #4; tools/patches.py _rec4c_*, runtime/hle/percore.c): the constructor's end, the
    # push's base and bound, the drain's pops and its checks of the rings' counts
    Site('rec4c_ctor_end', 'code', 'b801051f 54000001/ff00001f f9400bf3 a8c27bfd d65f03c0', 2),
    Site('rec4c_push_base', 'code', '9ba95109 71000d1f 9a943129 b940112a b9400928 6b08015f 5400000a/ff00001f 1100054b', 0),
    Site('rec4c_push_bound', 'code', '71000d1f 9a943129 b940112a b9400928 6b08015f 5400000a/ff00001f 1100054b b900112b', 0),
    Site('rec4c_push_pick', 'code', '9a943129 b940112a b9400928 6b08015f 5400000a/ff00001f 1100054b b900112b f940012b', 0),
    Site('rec4c_pop_1', 'code', '94000000/fc000000 36000000/fff8001f b940026e 3400000e/ff00001f 2942abe9 aa1f03e8 b9401feb '
         '2a1f03ed', 0),
    Site('rec4c_pop_2', 'code', '94000000/fc000000 36000000/fff8001f b9400668 34000008/ff00001f bd400fe0 aa1f03e9 2a1f03ed '
         '2940afea', 0),
    Site('rec4c_any_1a', 'code', 'b9401808 390043ff b9001fff 35000008/ff00001f b9443268 35000008/ff00001f b9484a68 '
         '34000008/ff00001f', 0),
    Site('rec4c_any_1b', 'code', 'b9401a68 35000008/ff00001f b9443268 35000008/ff00001f b9484a68 35000008/ff00001f b2607fe8 '
         '390003ff', 0),
    Site('rec4c_any_2a', 'code', 'b94c6268 35000008/ff00001f b9507a68 35000008/ff00001f b9549268 34000008/ff00001f 52849308 '
         '91314274', 0),
    Site('rec4c_any_2b', 'code', 'b94c6268 35000008/ff00001f b9507a68 35000008/ff00001f b9549268 35000008/ff00001f a9444ff4 '
         'a94357f6', 0),
    # the per-core rings (BOTW-JOBS-NATIVE.md #19; tools/patches.py _ring19_*, runtime/hle/percore.c): the makers' sizes
    # and core counts, the cursors' zeroings, the reset, the appends' bounds and the walkers' per-core loops
    Site('ring19_a_size', 'code', '52908000 52800102 94000000/fc000000 b4000000/ff00001f 52803008 d2e7f009 aa0003ea a900255f', 0),
    Site('ring19_a_count', 'code', '52803008 d2e7f009 aa0003ea a900255f f1000508 b9003d5f a901255f 9101614a', 0),
    Site('ring19_a_cores', 'code', 'd2800068 f2c01008 a9000268 f9400bf3 a8c27bfd d65f03c0', 0),
    Site('ring19_b_size', 'code', '52842000 52800102 94000000/fc000000 b4000000/ff00001f 92841fe8 52842089 5284218a d2e7f00b', 0),
    Site('ring19_b_start', 'code', '92841fe8 52842089 5284218a d2e7f00b 8b08000c b100b108 f829699f f82a698b', 0),
    Site('ring19_b_cores', 'code', 'd2800068 f2c00808 a90082c8 90000008/9f00001f 91000108/ffc003ff b90022df 52800500 aa1503e1', 0),
    Site('ring19_c_size', 'code', '52817400 52800102 94000000/fc000000 b4000000/ff00001f aa0003f4 94000000/fc000000 9101f280 '
         'b900769f', 0),
    Site('ring19_c_rest', 'code', '94000000/fc000000 d2800068 b90b9a9f f2c00108 a9005268 a9414ff4 a8c27bfd d65f03c0', 0),
    Site('ring19_c_make_fn', 'call', '94000000/fc000000 d2800068 b90b9a9f f2c00108 a9005268 a9414ff4 a8c27bfd d65f03c0', 0),
    Site('ring19_c_cores', 'code', 'd2800068 b90b9a9f f2c00108 a9005268 a9414ff4 a8c27bfd d65f03c0', 0),
    Site('ring19_a_zero', 'code', 'b90022ff aa1603e0 aa1503e1 f9000eff 52800102 f90002e8 f901a697 94000000/fc000000', 0),
    Site('ring19_b_zero', 'code', 'b90022df 52800500 aa1503e1 f9000edf 52800102 f90002c8 f901aa96 94000000/fc000000', 0),
    Site('ring19_c_zero', 'code', 'b90022df f9000edf 90000009/9f00001f 91000129/ffc003ff f90002c8 90000008/9f00001f 910003e2 '
         'f901ae96', 0),
    Site('ring19_reset', 'code', 'f941a408 b900211f f9000d1f f941a808 b900211f f9000d1f f941ac08 b900211f', 0),
    Site('ring19_add_a', 'code', '91006008 71000c5f b940240a 8b224909 9a883128 b9400109 6b0a013f 5400000a/ff00001f', 0),
    Site('ring19_add_c', 'code', '91006008 71000c5f 8b224909 9a883134 b9402409 b9400288 6b09011f 5400000a/ff00001f', 0),
    Site('ring19_wa_base', 'code', '910062e9 a901dbe8 910e6268 52800b0c a900a3e9 f94007e8 b87b7908 7100051f', 0),
    Site('ring19_wa_bound', 'code', 'f1000f7f 54000001/ff00001f a9494ff4 a94857f6 a9475ff8 a94667fa a9456ffc a9447bfd', 0),
    Site('ring19_wb_c0', 'code', 'b9401a96 910c927b 91076268 b81f43b9 f81f83a8 710002df 5400000d/ff00001f b85f43a8', 0),
    Site('ring19_wb_c1', 'code', 'b9401e88 7100051f b90013e8 5400000b/ff00001f 2a1f03fc 52800028 b9000fe8 29412688', 0),
    Site('ring19_wb_ck', 'code', 'b9402288 7100051f b90013e8 5400000b/ff00001f 2a1f03fc 52800048 b9000fe8 29412688', 0),
    Site('ring19_wb_mov', 'code', '52800048 b9000fe8 29412688 6b1c013f b9400fe9 7a428900 1b1c2508 12800009', 0),
    Site('ring19_wb_ccmp', 'code', '7a428900 1b1c2508 12800009 2a0903e1 93407d08 9a9f8115 b9431a68 7100051f', 0),
    Site('ring19_wb_end', 'code', '54000001/ff00001f a9474ff4 a94657f6 a9455ff8 a94467fa a9436ffc a9427bfd 910203ff d65f03c0 '
         '2a1f03f6', 0),
    Site('ring19_wc_c0', 'code', 'b940181b b9424037 f90007e8 7100077f 5400000b/ff00001f 2a1f03f9 2a1f03fc 52800036', 0),
    Site('ring19_wc_c1', 'code', 'b9401e9b 7100077f 5400000b/ff00001f 2a1f03fc 52800028 b90007e8 52800035 29412688', 0),
    Site('ring19_wc_ck', 'code', 'b940229b 7100077f 5400000b/ff00001f 2a1f03fc 52800048 b90007e8 52800035 29412688', 0),
    Site('ring19_wc_mov', 'code', '52800048 b90007e8 52800035 29412688 6b1c013f b94007e9 f9400a96 7a428900', 0),
    Site('ring19_wc_ccmp', 'code', '7a428900 1b1c2508 93407d08 9a9f8118 52800f88 9b085b08 3941e109 34000009/ff00001f', 0),
    Site('ring19_wc_end', 'code', '6b1b039f 54000001/ff00001f a9464ff4 a94557f6 a9445ff8 a94367fa a9426ffc a9417bfd', 1),
    # #28's second table (BOTW-JOBS-NATIVE.md #28; tools/patches.py _obj28_*, runtime/hle/percore.c): the constructor's
    # interfaces and records, the destructor's records, the pick of the current core's
    Site('obj28_ifaces', 'code', 'a9142668 a9152668 a9162668 a9172668 a9182668 a9192668 94000000/fc000000 9107e260', 0),
    Site('obj28_make', 'code', '94000000/fc000000 9107e260 aa1403e1 94000000/fc000000 91094260 aa1503e1 94000000/fc000000 '
         '910aa260', 0),
    Site('obj28_make_fn', 'call', '94000000/fc000000 9107e260 aa1403e1 94000000/fc000000 91094260 aa1503e1 94000000/fc000000 '
         '910aa260', 0),
    Site('obj28_destroy', 'code', '94000000/fc000000 910c0260 94000000/fc000000 910aa260 94000000/fc000000 91094260 '
         '94000000/fc000000 9107e260', 0),
    Site('obj28_destroy_fn', 'call', '94000000/fc000000 910c0260 94000000/fc000000 910aa260 94000000/fc000000 91094260 '
         '94000000/fc000000 9107e260', 0),
    Site('obj28_pick', 'code', '7100151f 54000008/ff00001f 90000009/9f00001f 91000129/ffc003ff 1000000a/9f00001f 3868692b '
         '8b0b094a d61f0140 91068260', 0),
    # BOTW's job scheduler run natively (BOTW-JOBS-NATIVE.md step 2; runtime/hle/jobs.c; tools/patches.py _jobs_*): the
    # entries of push, kick, a worker's run and run-and-wait, and the group setup, tick and group-event calls they make
    Site('jobs_push', 'code', 'a9bc7bfd f9000bf7 a90257f6 a9034ff4 910003fd aa0203f4 aa0103f5 b9001ba3', 0),
    Site('jobs_kick', 'code', 'd10103ff a9017bfd a90257f6 a9034ff4 910043fd 39427008 aa0003f3 34000008/ff00001f', 0),
    Site('jobs_run', 'code', 'a9ba7bfd f9000bfb a90267fa a9035ff8 a90457f6 a9054ff4 910003fd b9414008', 0),
    Site('jobs_run_wait', 'code', 'a9bc7bfd f9000bf7 a90257f6 a9034ff4 910003fd 39427008 aa0003f3 35000008/ff00001f', 0),
    Site('jobs_group_setup', 'call', '94000000/fc000000 f9400288 aa1403e0 f9400508 d63f0100 b9407a68 7100051f 5400000b/ff00001f', 0),
    Site('jobs_tick', 'call', '94000000/fc000000 f900a660 52800057 aa1303e0 b9010277 94000000/fc000000 b940fe78 '
         'b4000000/ff00001f', 0),
    Site('jobs_event_clear', 'call', '94000000/fc000000 390073bf 2a1f03fb 394073a8 35000008/ff00001f 2a1f03fb f9400288 910063a2', 0),
    Site('jobs_event_signal', 'call', '94000000/fc000000 b9010279 aa1403e0 b940fe61 94000000/fc000000 aa1303e0 b901027a b9010277', 0),
    # sead's base heap constructor (BOTW-JOBS-NATIVE.md step 3; tools/patches.py _heap_ctor): every heap made with its
    # own lock, so heaps the Switch shared between threads of one core stay safe when threads run in parallel
    Site('heap_ctor', 'code', 'd10183ff a9037bfd a90457f6 a9054ff4 9100c3fd 90000008/9f00001f 90000015/9f00001f '
         'f9400108/ffc003ff', 0),
    # the job pool's thread loop (BOTW-JOBS-NATIVE.md step 4; tools/patches.py _jobs_thread_*): each worker's config entry
    # (past three, worker 2's) and its affinity in the kernel's platform numbering (3 = the system core)
    Site('jobs_thread_cfg', 'code', '8b160a88 b9401a82 aa1303e1 aa0003f9 b9400d03 b9400104 94000000/fc000000 f94003e8', 0),
    Site('jobs_thread_mask', 'code', '1ac82128 b9000be8 94000000/fc000000 f9400328 aa1903e0 f9401908 d63f0100 910006d6 d2c00028 '
         '8b08039c', 0),
    # a broadcast to every core (BOTW-JOBS-NATIVE.md #30; tools/patches.py _jobs_bcast_*): f_700127d4d0's per-core job
    # groups past three, made and finished by the runtime; the group constructor, set-grab-size and event finaliser
    Site('jobs_bcast_next', 'code', '910202d6 eb08037f 54000003/ff00001f aa1503e0 94000000/fc000000 aa1503e0 94000000/fc000000 '
         '90000008/9f00001f', 0),
    Site('jobs_bcast_fini', 'code', '90000008/9f00001f 9104a320 f9400108/ffc003ff 91004115 f90087f5 94000000/fc000000 9102a320 '
         'f90047f5', 0),
    Site('jobs_group_ctor', 'call', '94000000/fc000000 9000001a/9f00001f 9100035a/ffc003ff f90007fa 910023e0 52800021 '
         '94000000/fc000000 91020336', 0),
    Site('jobs_group_grab', 'call', '94000000/fc000000 91020336 aa1603e0 94000000/fc000000 aa1603e0 52800021 f90047fa '
         '94000000/fc000000', 0),
    Site('jobs_group_fini', 'call', '94000000/fc000000 9102a320 f90047f5 94000000/fc000000 9100a320 f90007f5 94000000/fc000000 '
         '90000000/9f00001f', 0),
    # the game's memory heap size (BOTW-JOBS-NATIVE.md step 4; tools/patches.py _init_heap_size): nninitStartup's
    # fixed request, the PC's application memory instead (runtime/hle/heaps.c rt_memory_heap_size)
    Site('init_heap_size', 'code', '52b86000 94000000/fc000000 910023e0 52a04001 94000000/fc000000 f94007e0 52a04001 '
         '94000000/fc000000', 0),
    # the other ways sead makes a heap (BOTW-JOBS-NATIVE.md step 3; tools/patches.py _heap_lock_arg, _heap_root_flags):
    # ExpHeap::create, two class constructors and the root heap, each made with its own lock
    Site('heap_create_exp', 'code', 'd10283ff a9067bfd f9003bf7 a90857f6 a9094ff4 910183fd 2a0503f4 2a0403e5', 0),
    Site('heap_ctor_b', 'code', 'd10183ff a9037bfd a90457f6 a9054ff4 9100c3fd 90000009/9f00001f 2a0603f4 f9400129/ffc003ff', 0),
    Site('heap_ctor_c', 'code', 'd10143ff a9017bfd f90013f7 a90357f6 a9044ff4 910043fd 927df02a f104c55f', 0),
    Site('heap_root_flags', 'code', '52800088 7901b66a 9000000a/9f00001f 7901b268 52800d09 52800208 f940014a/ffc003ff b9006669', 0),
    # sead's root heap size (BOTW-JOBS-NATIVE.md step 4; tools/patches.py _root_heap_size): the game framework's fixed
    # 0xC1000000, the PC's memory heap less the malloc block instead (runtime/hle/heaps.c rt_root_heap_size)
    Site('root_heap_size', 'code', '52b82008 52800020 a9027fe8 94000000/fc000000 52800020 94000000/fc000000 52800020 '
         '94000000/fc000000', 0),
    # the job system's threads (BOTW-JOBS-NATIVE.md #31; tools/patches.py _job_thread_proc, _job_thread_res): the pool's
    # GOT (BaseProcMgr::init's caller) and the two checks that compare against boot-time copies of three cores' threads
    Site('jobs_pool_got', 'code', '90000008/9f00001f 90000009/9f00001f f9400108/ffc003ff f9400129/ffc003ff 9000000a/9f00001f '
         '9100014a/ffc003ff a9027fea 5280800a', 0),
    Site('job_thread_proc', 'code', '7a481284 1a9f17e0 14000000/fc000000 2a1f03e0 a9414ff4 a8c27bfd d65f03c0', 1),
    Site('job_thread_res', 'code', 'b940d008 d37df129 f9410e8a f940014b b940016b 6b08017f 54000000/ff00001f 9100214a', 0),
    # the views' render recording (tools/patches.py _view_rec_jobs): f_7000fb7544 makes three jobs that each take the
    # next view item from a counter until none is left (f_7000fb5a20) - one per core of the Switch build's three
    Site('view_rec_jobs', 'code', '52800063 2a1f03e4 b828681f 14000000/fc000000', 0),
    # ... and the most parts it cuts each draw list into (f_7000fb98bc: min(draws / [+0x5e68], [+0x5e6c]) contiguous
    # ranges, each recorded into a command buffer of its own): the constructor's `movk x8, #3, lsl #32` (+0x5e6c = 3)
    Site('view_rec_parts', 'code', 'f2c00068 f92f267f f92f2a7f f92f2e7f f92f327f f92f3668 f901d273 a9424ff4', 0),
    # ... the item and order arrays those parts go into, sized for 61 lists x 3 parts a unit (f_7000fb2804:
    # `mov w9, #0xb7` before madd w1, w8, w9, w10 -> 183 n + 6), and each list's record of its parts' command buffers
    # (f_7000faa800: `mov w1, #0x20`, 32 each). Both clamp an overflow to slot 0 without a word.
    Site('view_rec_items', 'code', '528016e9 528000ca a909a2c8 910302c0 a90aa2c8 f94027e8 aa1503e2 91040277', 0),
    Site('view_rec_cbs', 'code', '52800401 aa1403e2 52800103 91016100 94000000/fc000000 910006e8 92407d17 6b0802bf', 0),
    # ... and the lists no view cut (a pass made at draw time): the consumer f_7000fab5e0 finds no record of parts for the
    # list's pass and records the whole list inline (bl f_7000fab328, range 0..-1) on the one thread drawing the layer.
    # tools/patches.py _view_rec_inline cuts it into parts there instead, recorded on every core (runtime/hle/jobs.c
    # rt_view_rec_inline) by the game's own part recorder f_7000fab13c with a draw context of its own (f_7000ebf0c4 /
    # f_7000ebf148, f_7000eb0380 after each part as f_7000fb5a20 does), the list's record lock f_7000eaff08 / 18
    Site('view_rec_inline', 'code', '94000000/fc000000 14000000/fc000000 52800e08 9ba86b88 b9405909 7100053f 5400000b/ff00001f '
         '90000001/9f00001f', 0),
    Site('view_rec_consume', 'code', 'b9403c08 7100051f b9400049 b9401c0a 0a090149 7a40a924 54000000/ff00001f d101c3ff', 0),
    Site('view_rec_part', 'call', '94000000/fc000000 885f7eb7 110006e8 88097ea8 35000009/ff00001f 94000000/fc000000 910023e0 '
         '94000000/fc000000', 0),
    Site('view_rec_ctx', 'call', '94000000/fc000000 f96f1e62 910062e5 910023e1 a9408ee0 d10073a6 f94002e4 94000000/fc000000', 0),
    Site('view_rec_ctx_fini', 'call', '94000000/fc000000 b9401a88 6b0802ff 5400000b/ff00001f 910743ff a9434ff4 a94257f6 '
         'a9415ffc', 0),
    Site('view_rec_item_end', 'call', '94000000/fc000000 910023e0 94000000/fc000000 b9401a88 6b0802ff 5400000b/ff00001f '
         '910743ff a9434ff4', 0),
    Site('view_rec_lock', 'call', '94000000/fc000000 b94052c8 7100051f 5400000b/ff00001f 2a1f03f8 52800e19 b94042ca 6b18015f', 0),
    Site('view_rec_unlock', 'call', '94000000/fc000000 aa1703e0 a9444ff4 a94357f6 f9400bf9 a9425ff8 a8c57bfd d65f03c0', 0),
    Site('view_rec_record', 'call', '94000000/fc000000 b94002e3 aa0003f3 9100e002 aa1503e0 aa1403e1 94000000/fc000000 '
         '2a0003fa', 0),
    Site('view_rec_begin', 'call', '94000000/fc000000 2a0003fa aa1903e0 aa1403e1 aa1303e2 aa1803e3 aa1703e4 52800025', 0),
    Site('view_rec_end', 'call', '94000000/fc000000 39409008 36000008/fff8001f 91016268 885f7d09 1100052a 880b7d0a '
         '3500000b/ff00001f', 0),
    Site('main_data', 'data', '', 0),
    Site('main_end', 'end', '', 0),
]

# Sites a version may lack, as a group: when one of them is neither found by its pattern nor listed in the version's
# KNOWN table, the whole group is left out (with its patches) instead of refusing the version. The mouse camera's
# sites go together: the stick input without the linear vertical part would square the mouse's (unclamped) motion.
# Without them the mouse turns the camera through the right stick, as before (runtime/hle/input.c); without the
# second group it turns it without the game's camera inversion options.
OPTIONAL_GROUPS = [
    ('yaz0_stream', 'yaz0_stream_ret'),
    ('cam_stick', 'cam_pitch_chase', 'cam_pitch_surf', 'cam_pitch_wakeboard', 'cam_pitch_aim', 'cam_pitch_item'),
    ('cam_reverse_x', 'cam_reverse_y'),
    # the game on N cores (BOTW-JOBS-NATIVE.md): all of these or none, so a version never runs partly on N (only
    # 1.9.0's checked so far; without them the game keeps sead's 3 cores and its own VFR)
    ('core_info_configure', 'vfr_get_delta_frame', 'vfr_get_raw_delta_frame',
     'vfr_get_delta_time', 'vfr_get_interval', 'vfr_get_ratio',
     'vfr_get_ratio_time', 'vfr_use_buffer_b', 'vfr_use_buffer_a',
     'vfr_set_min_delta', 'vfr_get_and_set_min', 'vfr_set_delta',
     'vfr_update_interval', 'vfr_update_spread', 'vfr_init_spread',
     'vfr_reset_spread', 'core_id_tls',
     'ctx_ctor_base', 'ctx_ctor_sys', 'ctx_ctor_count',
     'ctx_set_all', 'ctx_get_a', 'ctx_get_b',
     'ctx_get_c', 'ctx_get_d', 'ctx_sys_a',
     'ctx_sys_b', 'ctx_sys_c', 'ctx_sys_d',
     'ctx_check_all', 'ctx_check_one', 'ctx_sys_count',
     'ctx_sys_entries', 'ctx_reset_base', 'ctx_reset_0',
     'ctx_reset_1', 'ctx_reset_2', 'ctx_reset_3',
     'ctx_reset_fn', 'ctx_size_base', 'ctx_flag_base',
     'arena_setup', 'arena_malloc', 'arena_pool_init',
     'arena_pool_addr', 'arena_pool_cpu', 'arena_pool_offset',
     'arena_blocks_alloc', 'arena_buffers_alloc', 'arena_reset',
     'arena_free_0', 'arena_free_1', 'arena_free_2',
     'arena_free_fn', 'arena_begin_rec', 'arena_end_rec',
     'cstate_ctor_a', 'cstate_ctor_b', 'cstate_set_base',
     'cstate_set_bound', 'cstate_get_base', 'cstate_get_bound',
     'cstate_slot_base', 'cstate_slot_bound',
     'jobs_ctor_marks', 'jobs_ctor_marks_b', 'jobs_ctor_bufs',
     'jobs_ctor_bufs_b', 'jobs_ctor_bufs_c', 'jobs_alloc_0',
     'jobs_alloc_1', 'jobs_alloc_1s', 'jobs_alloc_2',
     'jobs_alloc_2s', 'jobs_free_0', 'jobs_free_1',
     'jobs_free_2', 'jobs_free_fn', 'jobs_mark_base',
     'jobs_mark_bound', 'jobs_buf_base', 'jobs_buf_bound',
     'rec17_new', 'rec17_alloc', 'rec17_alloc_tag',
     'rec17_init_a', 'rec17_init_b', 'rec17_init_c',
     'rec17_init_d',
     'drawctx_count',
     'cnt_add_base', 'cnt_add_bound', 'cnt_sum_a',
     'cnt_sum_b', 'cnt_sum_c', 'cnt_clear_a',
     'cnt_clear_b', 'cnt_init_a', 'cnt_init_b',
     'busy_init', 'busy_inc_bound', 'busy_inc_base',
     'busy_dec_bound', 'busy_dec_base', 'busy_sys',
     'obj7_ctor', 'obj7_vtable', 'obj7_heap',
     'obj7_new', 'obj7_init', 'obj7_free_0',
     'obj7_free_1', 'obj7_free_1s', 'obj7_free_2',
     'obj7_free_2s', 'obj7_free_3s', 'obj7_fini',
     'obj7_delete', 'obj7_get_base', 'obj7_get_bound',
     'wd_a_flag', 'wd_a_tick', 'wd_a_state',
     'wd_b_flag', 'wd_b_state', 'wd_b_tick',
     'wd_c_flag', 'wd_c_state', 'wd_c_tick',
     'wd_d_flag', 'wd_d_state', 'wd_d_tick',
     'wd_d_flag2', 'wd_ctor_tick0', 'wd_ctor_tick1',
     'wd_ctor_tick2', 'wd_tick_fn', 'wd_init_state_base',
     'wd_init_state_a', 'wd_init_state_b', 'wd_init_flag_base',
     'wd_init_flag_a', 'wd_init_flag_b', 'core_index_map',
     'busy_reset_a', 'busy_reset_b', 'busy_sum_a',
     'busy_sum_b', 'busy_sum_c', 'busy_total',
     'lin_ctor', 'lin_dtor', 'lin_init',
     'lin_split', 'lin_vtable', 'lin_lock_ctor',
     'lin_lock_dtor', 'lin_alloc_make', 'lin_alloc_init',
     'lin_alloc_fini', 'lin_malloc', 'lin_free',
     'lin_after', 'lin_rec_a', 'lin_rec_b',
     'lin_rec_c', 'lin_rec_d', 'lin_rec_e',
     'list24_init', 'list24_a_base', 'list24_a_bound',
     'list24_b_bytes', 'list24_b_base', 'list24_b_bound1',
     'list24_b_bound2', 'list24_b_bound3', 'list24_b_bound4',
     'list24_c_base', 'list24_c_bound', 'list24_d_base',
     'list24_d_bound',
     'cnt22_add_base', 'cnt22_reset_a', 'cnt22_reset_b',
     'str26_init_base', 'str26_init_rest', 'str26_init_fn',
     'str26_a_base', 'str26_b_base',
     'pair6_a_base', 'pair6_a_bound', 'pair6_b_base',
     'pair6_b_bound', 'pair6_c_base', 'pair6_c_bound',
     'pair6_d_base', 'pair6_d_bound', 'pair6_e_base',
     'pair6_e_bound', 'pair6_sum_u0', 'pair6_sum_uz0',
     'pair6_sum_u1', 'pair6_sum_uz1', 'pair6_sum_u2',
     'pair6_sum_uz2', 'pair6_sum_f0', 'pair6_sum_fz0',
     'pair6_sum_f1', 'pair6_sum_fz1', 'pair6_sum_f2',
     'pair6_sum_fz2', 'pair6_zero_a', 'pair6_zero_b',
     'pair6_zero_c',
     'flip21_base', 'flip21_zero',
     'slot5_rec_base', 'slot5_rec_bound', 'slot5_u_base',
     'slot5_u_bound', 'slot5_f_base', 'slot5_f_bound',
     'slot5_merge', 'slot5_pick', 'slot5_fz_a',
     'slot5_fz_b', 'slot5_fz_c', 'slot5_fz_d',
     'obj12_first', 'obj12_alloc1', 'obj12_ctor1',
     'obj12_alloc2', 'obj12_ctor2', 'obj12_alloc_fn',
     'obj12_ctor_fn', 'obj12_a', 'obj12_b',
     'obj12_c', 'obj12_d', 'obj12_e',
     'q27_add_bufs', 'q27_add_bound1', 'q27_add_locks',
     'q27_add_bound2', 'q27_add_cur', 'q27_walk_locks',
     'q27_walk_bufs', 'q27_walk_cur', 'q27_walk_bound',
     'q27_init_first', 'q27_init_first_b', 'q27_init_2',
     'q27_init_3', 'q27_alloc_fn', 'q27_alloc_tag',
     'q27_ctor_tail', 'q27_memset_fn', 'q27_lock_ctor',
     'rec23_ctor_end', 'rec23_init_0', 'rec23_init_1',
     'rec23_init_2', 'rec23_init_3', 'rec23_init_4',
     'rec23_init_5', 'rec23_init_6', 'rec23_init_7',
     'rec23_init_8', 'rec23_init_9', 'rec23_init_10',
     'rec23_init_11', 'rec23_walk_base', 'rec23_walk_end',
     'rec23_a_base', 'rec23_a_b1', 'rec23_a_b2',
     'rec23_a_b3', 'rec23_a_b4', 'rec23_a_b5',
     'rec23_b_base', 'rec23_b_b1', 'rec23_b_b2',
     'rec23_b_b3', 'rec23_b_b4', 'rec23_b_b5',
     'rec4a_ctor_0', 'rec4a_ctor_1', 'rec4a_ctor_2',
     'rec4a_ctor_rest', 'rec4a_reset_fn', 'rec4a_acc_base',
     'rec4a_acc_bound', 'rec4a_pick', 'rec4a_copy_fn',
     'rec4b_ctor_end', 'rec4b_acc_base', 'rec4b_acc_bound',
     'rec4b_acc_a_base', 'rec4b_acc_a_bound', 'rec4b_reset',
     'rec4b_remove', 'rec4b_remove_a', 'rec4b_remove_b',
     'rec4b_sweep', 'rec4b_drop_fn', 'rec4b_pick',
     'rec4b_fill_b', 'rec4b_filter', 'rec4b_states',
     'rec4b_pick_b', 'rec4b_upd_clear1', 'rec4b_upd_any_a',
     'rec4b_upd_keep_a', 'rec4b_upd_any_b', 'rec4b_upd_any_c',
     'rec4b_upd_any_d', 'rec4b_upd_clear2', 'rec4b_upd_other1',
     'rec4a_part_base', 'rec4a_part_size', 'rec4a_part_1',
     'rec4a_part_2', 'rec4a_part_rest', 'rec4a_make_fn',
     'rec4a_drop_rest', 'rec4a_release_fn', 'rec4a_drop_1',
     'rec4a_drop_0',
     'q27_add2_bufs', 'q27_add2_bound', 'q27_add2_cur',
     'q27_add3_bufs', 'q27_add3_bound', 'q27_add3_cur',
     'rec4c_ctor_end', 'rec4c_push_base', 'rec4c_push_bound',
     'rec4c_push_pick', 'rec4c_pop_1', 'rec4c_pop_2',
     'rec4c_any_1a', 'rec4c_any_1b', 'rec4c_any_2a',
     'rec4c_any_2b',
     'ring19_a_size', 'ring19_a_count', 'ring19_a_cores',
     'ring19_b_size', 'ring19_b_start', 'ring19_b_cores',
     'ring19_c_size', 'ring19_c_rest', 'ring19_c_make_fn',
     'ring19_c_cores', 'ring19_a_zero', 'ring19_b_zero',
     'ring19_c_zero', 'ring19_reset', 'ring19_add_a',
     'ring19_add_c', 'ring19_wa_base', 'ring19_wa_bound',
     'ring19_wb_c0', 'ring19_wb_c1', 'ring19_wb_ck',
     'ring19_wb_mov', 'ring19_wb_ccmp', 'ring19_wb_end',
     'ring19_wc_c0', 'ring19_wc_c1', 'ring19_wc_ck',
     'ring19_wc_mov', 'ring19_wc_ccmp', 'ring19_wc_end',
     'obj28_ifaces', 'obj28_make', 'obj28_make_fn',
     'obj28_destroy', 'obj28_destroy_fn', 'obj28_pick',
     'jobs_push', 'jobs_kick', 'jobs_run',
     'jobs_run_wait', 'jobs_group_setup', 'jobs_tick',
     'jobs_event_clear', 'jobs_event_signal',
     'heap_ctor',
     'jobs_thread_cfg', 'jobs_thread_mask',
     'jobs_bcast_next', 'jobs_bcast_fini', 'jobs_group_ctor',
     'jobs_group_grab', 'jobs_group_fini',
     'init_heap_size',
     'heap_create_exp', 'heap_ctor_b', 'heap_ctor_c',
     'heap_root_flags',
     'root_heap_size',
     'jobs_pool_got', 'job_thread_proc', 'job_thread_res',
     'view_rec_jobs', 'view_rec_parts', 'view_rec_items', 'view_rec_cbs',
     'view_rec_inline', 'view_rec_consume', 'view_rec_part', 'view_rec_ctx',
     'view_rec_ctx_fini', 'view_rec_item_end', 'view_rec_lock', 'view_rec_unlock',
     'view_rec_record', 'view_rec_begin', 'view_rec_end'),
]
OPTIONAL = {n for g in OPTIONAL_GROUPS for n in g}

KNOWN = {
    '1.8.2': {
        'opt_create_rows': 0x7000d1bd78,
        'opt_row_init_flag_type': 0x7000d1c208,
        'gdt_flag_type': 0x7001023038,
        'opt_row_redraw': 0x7000d1bba8,
        'opt_read': 0x7000d1cab4,
        'opt_store': 0x7000d1bb14,
        'opt_table': 0x7001cccd98,
        'opt_reset': 0x7000d1dde8,
        'opt_msg_call': 0x7000d1dc40,
        'msg_lookup': 0x7000d71210,
        'opt_decide': 0x7000d1d3d0,
        'opt_update': 0x7000d1c234,
        'opt_back_call': 0x7000d1c504,
        'screen_back': 0x70012d8928,
        'screen_open_request': 0x7000d6fd90,
        'screen_open': 0x70012d90e0,
        'screen_close': 0x70012d9124,
        'screen_mgr': 0x7001cd5c38,
        'pause_options_update': 0x7000cbd0b4,
        'opt_screen_rows': 0x3698,
        'dialog_texts': 0x7000d44748,
        'dialog_back': 0x7000d46364,
        'dialog_answer': 0x7001c4b308,
        'title_update': 0x7000d4bca0,
        'title_button': 0x7000d4c5e0,
        'title_options_state': 0x363c,
        'title_msg_file': 0x7001cd06d8,
        'title_version_format': 0x70014b854e,
        'frame_wait': 0x7000d811fc,
        'frame_loop_interval_ret': 0x7001012a34,
        'vfr_ratio': 0x70014092f4,
        'gui_speed': 0x700116d0e8,
        'persp_matrix': 0x7000da1400,
        'persp_matrix_offset': 0x7000da1408,
        'persp_unproject_y': 0x7000da14d8,
        'proj_update': 0x7000da0e24,
        'frustum_lr0': 0x7000da18d4,
        'frustum_lr1': 0x7000da18fc,
        'frustum_tb0': 0x7000da1910,
        'frustum_tb1': 0x7000da1930,
        'persp_unproject_x': 0x7000da1508,
        'pause_state': 0x7001cb48a4,
        'event_mgr': 0x7001cd8db8,
        'controller_step': 0x7001c4ba40,
        'player_info': 0x7001cb2658,
        'world_mgr': 0x7001cf5d00,
        'main_data': 0x7001a1f000,
        'main_end': 0x7001d37000,
    },
    '1.9.0': {
        'yaz0_stream': 0x700131d674,
        'yaz0_stream_ret': 0x700131d7e8,
        'opt_create_rows': 0x7000e2f0a8,
        'opt_row_init_flag_type': 0x7000e2f538,
        'gdt_flag_type': 0x7001136dc8,
        'opt_row_redraw': 0x7000e2eed8,
        'opt_read': 0x7000e2fde4,
        'opt_store': 0x7000e2ee44,
        'opt_table': 0x7001e2e610,
        'opt_reset': 0x7000e31118,
        'opt_msg_call': 0x7000e30f70,
        'msg_lookup': 0x7000e84870,
        'opt_decide': 0x7000e30700,
        'opt_update': 0x7000e2f564,
        'opt_back_call': 0x7000e2f834,
        'screen_back': 0x70013eca68,
        'screen_open_request': 0x7000e833f0,
        'screen_open': 0x70013ed220,
        'screen_close': 0x70013ed264,
        'screen_mgr': 0x7001e374b8,
        'pause_options_update': 0x7000dd0384,
        'opt_screen_rows': 0x36b0,
        'dialog_texts': 0x7000e57af8,
        'dialog_back': 0x7000e59714,
        'dialog_answer': 0x7001dac9a8,
        'title_update': 0x7000e5f070,
        'title_button': 0x7000e5f9b0,
        'title_options_state': 0x3654,
        'title_msg_file': 0x7001e31f50,
        'title_version_format': 0x70015cd092,
        'frame_wait': 0x7000e9487c,
        'frame_loop_interval_ret': 0x70011267c4,
        'vfr_ratio': 0x700151d494,
        'gui_speed': 0x7001280ee8,
        'persp_matrix': 0x7000eb4a50,
        'persp_matrix_offset': 0x7000eb4a58,
        'persp_unproject_y': 0x7000eb4b28,
        'proj_update': 0x7000eb4474,
        'frustum_lr0': 0x7000eb4f24,
        'frustum_lr1': 0x7000eb4f4c,
        'frustum_tb0': 0x7000eb4f60,
        'frustum_tb1': 0x7000eb4f80,
        'persp_unproject_x': 0x7000eb4b58,
        'pause_state': 0x7001e1611c,
        'event_mgr': 0x7001e3a638,
        'controller_step': 0x7001dad0e0,
        'player_info': 0x7001e13ed0,
        'world_mgr': 0x7001e57580,
        'cam_stick': 0x7000d57fe0,
        'cam_pitch_chase': 0x7000bf5334,
        'cam_pitch_surf': 0x7000c1a52c,
        'cam_pitch_wakeboard': 0x7000c22104,
        'cam_pitch_aim': 0x7000bf0bfc,
        'cam_pitch_item': 0x7000bf2c0c,
        'cam_reverse_x': 0x7000d17024,
        'cam_reverse_y': 0x7000d1703c,
        'core_info_configure': 0x7000eb92f0,
        'vfr_get_delta_frame': 0x700056fcd0,
        'vfr_get_raw_delta_frame': 0x70005b6fa8,
        'vfr_get_delta_time': 0x70005abcfc,
        'vfr_get_interval': 0x7000583118,
        'vfr_get_ratio': 0x700059c788,
        'vfr_get_ratio_time': 0x70007613c8,
        'vfr_use_buffer_b': 0x700151d628,
        'vfr_use_buffer_a': 0x700151d708,
        'vfr_set_min_delta': 0x700151d7e8,
        'vfr_get_and_set_min': 0x700151d8cc,
        'vfr_set_delta': 0x700151dc00,
        'vfr_update_interval': 0x700151d3d0,
        'vfr_update_spread': 0x700151d4dc,
        'vfr_init_spread': 0x700151d330,
        'vfr_reset_spread': 0x700151d9d4,
        'core_id_tls': 0x7001e348b0,
        'ctx_ctor_base': 0x7000ed60f8,
        'ctx_ctor_sys': 0x7000ed61a0,
        'ctx_ctor_count': 0x7000ed61b0,
        'ctx_set_all': 0x7000ed626c,
        'ctx_get_a': 0x7000ed6410,
        'ctx_get_b': 0x7000ed68e0,
        'ctx_get_c': 0x7000ed69c8,
        'ctx_get_d': 0x7000ed6aa8,
        'ctx_sys_a': 0x7000ed6b00,
        'ctx_sys_b': 0x7000ed6b5c,
        'ctx_sys_c': 0x7000ed6bb8,
        'ctx_sys_d': 0x7000ed6c20,
        'ctx_check_all': 0x7000ed70a0,
        'ctx_check_one': 0x7000ed71e0,
        'ctx_sys_count': 0x7000ed7220,
        'ctx_sys_entries': 0x7000ed722c,
        'ctx_reset_base': 0x7000ed72a4,
        'ctx_reset_0': 0x7000ed6674,
        'ctx_reset_1': 0x7000ed6680,
        'ctx_reset_2': 0x7000ed668c,
        'ctx_reset_3': 0x7000ed6698,
        'ctx_reset_fn': 0x7000ed7278,
        'ctx_size_base': 0x7000ed73f0,
        'ctx_flag_base': 0x7000eea510,
        'arena_setup': 0x7000ef3098,
        'arena_malloc': 0x7000007c20,
        'arena_pool_init': 0x7000ebf6e0,
        'arena_pool_addr': 0x7000ebf3e0,
        'arena_pool_cpu': 0x7001db2660,
        'arena_pool_offset': 0x7000ebf97c,
        'arena_blocks_alloc': 0x7000ef3728,
        'arena_buffers_alloc': 0x7000ef37a8,
        'arena_reset': 0x7000ef33e8,
        'arena_free_0': 0x7000ef2f6c,
        'arena_free_1': 0x7000ef2f88,
        'arena_free_2': 0x7000ef2fa0,
        'arena_free_fn': 0x7000000680,
        'arena_begin_rec': 0x7000ef350c,
        'arena_end_rec': 0x7000ef365c,
        'cstate_ctor_a': 0x7000eed9d4,
        'cstate_ctor_b': 0x7000eed9d8,
        'cstate_set_base': 0x7000eedeec,
        'cstate_set_bound': 0x7000eedef8,
        'cstate_get_base': 0x7000eedf4c,
        'cstate_get_bound': 0x7000eedf58,
        'cstate_slot_base': 0x7000eedfa8,
        'cstate_slot_bound': 0x7000eedfb0,
        'jobs_ctor_marks': 0x700156d720,
        'jobs_ctor_marks_b': 0x700156d728,
        'jobs_ctor_bufs': 0x700156d72c,
        'jobs_ctor_bufs_b': 0x700156d730,
        'jobs_ctor_bufs_c': 0x700156d734,
        'jobs_alloc_0': 0x700156dd70,
        'jobs_alloc_1': 0x700156dd88,
        'jobs_alloc_1s': 0x700156dd8c,
        'jobs_alloc_2': 0x700156dda4,
        'jobs_alloc_2s': 0x700156dda8,
        'jobs_free_0': 0x700156d8bc,
        'jobs_free_1': 0x700156d8cc,
        'jobs_free_2': 0x700156d8dc,
        'jobs_free_fn': 0x7000000680,
        'jobs_mark_base': 0x700156fb90,
        'jobs_mark_bound': 0x700156fb9c,
        'jobs_buf_base': 0x700156fc18,
        'jobs_buf_bound': 0x700156fc38,
        'rec17_new': 0x700148e9f4,
        'rec17_alloc': 0x7000e8c7f0,
        'rec17_alloc_tag': 0x7001d661a0,
        'rec17_init_a': 0x7000ec4924,
        'rec17_init_b': 0x7000ec53f0,
        'rec17_init_c': 0x7000ec54c4,
        'rec17_init_d': 0x7000ecbfc0,
        'drawctx_count': 0x700105c1b8,
        'cnt_add_base': 0x700057be78,
        'cnt_add_bound': 0x700057be9c,
        'cnt_sum_a': 0x7000b1c1e8,
        'cnt_sum_b': 0x7000b1c1ec,
        'cnt_sum_c': 0x7000b1c1f0,
        'cnt_clear_a': 0x7000b1c27c,
        'cnt_clear_b': 0x7000b1c280,
        'cnt_init_a': 0x7000b1bfac,
        'cnt_init_b': 0x7000b1bfb0,
        'busy_init': 0x7000b31a10,
        'busy_inc_bound': 0x7000aeea54,
        'busy_inc_base': 0x7000aeea5c,
        'busy_dec_bound': 0x7000aeead0,
        'busy_dec_base': 0x7000aeead8,
        'busy_sys': 0x7000cd90c8,
        'obj7_ctor': 0x7000ddfd70,
        'obj7_vtable': 0x7001ce5998,
        'obj7_heap': 0x7000e61350,
        'obj7_new': 0x7000e8c250,
        'obj7_init': 0x7000ec9a10,
        'obj7_free_0': 0x7000ddfe10,
        'obj7_free_1': 0x7000ddfe38,
        'obj7_free_1s': 0x7000ddfe3c,
        'obj7_free_2': 0x7000ddfe54,
        'obj7_free_2s': 0x7000ddfe58,
        'obj7_free_3s': 0x7000ddfe70,
        'obj7_fini': 0x7000ec3274,
        'obj7_delete': 0x7000e8c800,
        'obj7_get_base': 0x7000ddfed0,
        'obj7_get_bound': 0x7000ddfee0,
        'wd_a_flag': 0x700105c92c,
        'wd_a_tick': 0x700105c940,
        'wd_a_state': 0x700105c948,
        'wd_b_flag': 0x700105c9d4,
        'wd_b_state': 0x700105c9ec,
        'wd_b_tick': 0x700105c9f4,
        'wd_c_flag': 0x700105cb20,
        'wd_c_state': 0x700105cb38,
        'wd_c_tick': 0x700105cb40,
        'wd_d_flag': 0x700105ce0c,
        'wd_d_state': 0x700105ce24,
        'wd_d_tick': 0x700105ce2c,
        'wd_d_flag2': 0x700105ceec,
        'wd_ctor_tick0': 0x700105be30,
        'wd_ctor_tick1': 0x700105be38,
        'wd_ctor_tick2': 0x700105be48,
        'wd_tick_fn': 0x700159ff00,
        'wd_init_state_base': 0x700105c388,
        'wd_init_state_a': 0x700105c38c,
        'wd_init_state_b': 0x700105c390,
        'wd_init_flag_base': 0x700105c3a0,
        'wd_init_flag_a': 0x700105c3a4,
        'wd_init_flag_b': 0x700105c3a8,
        'core_index_map': 0x7000fba1a4,
        'busy_reset_a': 0x7000b31fa8,
        'busy_reset_b': 0x7000b3202c,
        'busy_sum_a': 0x7000b32270,
        'busy_sum_b': 0x7000b32274,
        'busy_sum_c': 0x7000b3227c,
        'busy_total': 0x7000b32284,
        'lin_ctor': 0x7000f9d7c8,
        'lin_dtor': 0x7000f9d80c,
        'lin_init': 0x7000f9d8d8,
        'lin_split': 0x7000f9dac0,
        'lin_vtable': 0x7001d10b98,
        'lin_lock_ctor': 0x700000151c,
        'lin_lock_dtor': 0x7000eafea0,
        'lin_alloc_make': 0x7000dded10,
        'lin_alloc_init': 0x7000ebf524,
        'lin_alloc_fini': 0x7000ebf440,
        'lin_malloc': 0x7000007c20,
        'lin_free': 0x7000e8c800,
        'lin_after': 0x7000f74a24,
        'lin_rec_a': 0x7000f9db64,
        'lin_rec_b': 0x7000f9dbc8,
        'lin_rec_c': 0x7000f9dc1c,
        'lin_rec_d': 0x7000f9dc60,
        'lin_rec_e': 0x7000f9dc7c,
        'list24_init': 0x700000e3b8,
        'list24_a_base': 0x70015306fc,
        'list24_a_bound': 0x7001530708,
        'list24_b_bytes': 0x70015307bc,
        'list24_b_base': 0x70015307c4,
        'list24_b_bound1': 0x70015307c8,
        'list24_b_bound2': 0x70015307dc,
        'list24_b_bound3': 0x70015307fc,
        'list24_b_bound4': 0x7001530a28,
        'list24_c_base': 0x7001531cb4,
        'list24_c_bound': 0x7001531ce8,
        'list24_d_base': 0x7001531d3c,
        'list24_d_bound': 0x7001531d48,
        'cnt22_add_base': 0x700118b564,
        'cnt22_reset_a': 0x700118b59c,
        'cnt22_reset_b': 0x700118b5a0,
        'str26_init_base': 0x7001594ef0,
        'str26_init_rest': 0x7001594f7c,
        'str26_init_fn': 0x7000ea6c84,
        'str26_a_base': 0x7001594b3c,
        'str26_b_base': 0x7001594cf0,
        'pair6_a_base': 0x7000cbf238,
        'pair6_a_bound': 0x7000cbf248,
        'pair6_b_base': 0x7000cbf2b4,
        'pair6_b_bound': 0x7000cbf2c4,
        'pair6_c_base': 0x7000cbf3ac,
        'pair6_c_bound': 0x7000cbf3c0,
        'pair6_d_base': 0x7000cbf420,
        'pair6_d_bound': 0x7000cbf430,
        'pair6_e_base': 0x7000cbf4a4,
        'pair6_e_bound': 0x7000cbf4b4,
        'pair6_sum_u0': 0x7000ce9558,
        'pair6_sum_uz0': 0x7000ce9560,
        'pair6_sum_u1': 0x7000ce9568,
        'pair6_sum_uz1': 0x7000ce9570,
        'pair6_sum_u2': 0x7000ce957c,
        'pair6_sum_uz2': 0x7000ce9580,
        'pair6_sum_f0': 0x7000cea038,
        'pair6_sum_fz0': 0x7000cea044,
        'pair6_sum_f1': 0x7000cea050,
        'pair6_sum_fz1': 0x7000cea058,
        'pair6_sum_f2': 0x7000cea064,
        'pair6_sum_fz2': 0x7000cea068,
        'pair6_zero_a': 0x7000cbdd0c,
        'pair6_zero_b': 0x7000cbdd1c,
        'pair6_zero_c': 0x7000cbdd24,
        'flip21_base': 0x7001062d2c,
        'flip21_zero': 0x7001062b50,
        'slot5_rec_base': 0x7000c311ac,
        'slot5_rec_bound': 0x7000c311b4,
        'slot5_u_base': 0x7000c2e210,
        'slot5_u_bound': 0x7000c2e218,
        'slot5_f_base': 0x7000c33064,
        'slot5_f_bound': 0x7000c33070,
        'slot5_merge': 0x7000c30e74,
        'slot5_pick': 0x7000c33098,
        'slot5_fz_a': 0x7000c32194,
        'slot5_fz_b': 0x7000c3219c,
        'slot5_fz_c': 0x7000c3308c,
        'slot5_fz_d': 0x7000c33090,
        'obj12_first': 0x700105c2c4,
        'obj12_alloc1': 0x700105c2cc,
        'obj12_ctor1': 0x700105c2d8,
        'obj12_alloc2': 0x700105c2ec,
        'obj12_ctor2': 0x700105c2f8,
        'obj12_alloc_fn': 0x7000e8c250,
        'obj12_ctor_fn': 0x700105b1cc,
        'obj12_a': 0x7000fb8454,
        'obj12_b': 0x7000fe6614,
        'obj12_c': 0x7000ff58dc,
        'obj12_d': 0x7000ff5cf4,
        'obj12_e': 0x7001036348,
        'q27_add_bufs': 0x7001138a5c,
        'q27_add_bound1': 0x7001138a64,
        'q27_add_locks': 0x7001138a78,
        'q27_add_bound2': 0x7001138a7c,
        'q27_add_cur': 0x7001138aa0,
        'q27_walk_locks': 0x700113e044,
        'q27_walk_bufs': 0x700113e048,
        'q27_walk_cur': 0x700113e0a8,
        'q27_walk_bound': 0x700113e9fc,
        'q27_init_first': 0x70011355a0,
        'q27_init_first_b': 0x70011355a4,
        'q27_init_2': 0x70011355bc,
        'q27_init_3': 0x70011355e0,
        'q27_alloc_fn': 0x7000e8c7f0,
        'q27_alloc_tag': 0x7001d661a0,
        'q27_ctor_tail': 0x7001132444,
        'q27_memset_fn': 0x70015a0330,
        'q27_lock_ctor': 0x700000151c,
        'rec23_ctor_end': 0x70014000cc,
        'rec23_init_0': 0x7001400538,
        'rec23_init_1': 0x700140053c,
        'rec23_init_2': 0x7001400544,
        'rec23_init_3': 0x7001400548,
        'rec23_init_4': 0x700140054c,
        'rec23_init_5': 0x7001400550,
        'rec23_init_6': 0x7001400554,
        'rec23_init_7': 0x7001400558,
        'rec23_init_8': 0x700140055c,
        'rec23_init_9': 0x7001400560,
        'rec23_init_10': 0x7001400564,
        'rec23_init_11': 0x7001400568,
        'rec23_walk_base': 0x7001400a6c,
        'rec23_walk_end': 0x7001400af4,
        'rec23_a_base': 0x70014001e4,
        'rec23_a_b1': 0x70014001f8,
        'rec23_a_b2': 0x7001400218,
        'rec23_a_b3': 0x7001400238,
        'rec23_a_b4': 0x7001400258,
        'rec23_a_b5': 0x70014002b4,
        'rec23_b_base': 0x7001400354,
        'rec23_b_b1': 0x7001400368,
        'rec23_b_b2': 0x7001400388,
        'rec23_b_b3': 0x70014003a8,
        'rec23_b_b4': 0x70014003c8,
        'rec23_b_b5': 0x7001400410,
        'rec4a_ctor_0': 0x7000c2acf0,
        'rec4a_ctor_1': 0x7000c2acf8,
        'rec4a_ctor_2': 0x7000c2ad00,
        'rec4a_ctor_rest': 0x7000c2ad04,
        'rec4a_reset_fn': 0x7000c2a760,
        'rec4a_acc_base': 0x7000c2d848,
        'rec4a_acc_bound': 0x7000c2d850,
        'rec4a_pick': 0x7000c2d894,
        'rec4a_copy_fn': 0x7000c2a778,
        'rec4b_ctor_end': 0x70010d4e54,
        'rec4b_acc_base': 0x70010d55b0,
        'rec4b_acc_bound': 0x70010d55bc,
        'rec4b_acc_a_base': 0x70010d5708,
        'rec4b_acc_a_bound': 0x70010d5714,
        'rec4b_reset': 0x70010d5fe8,
        'rec4b_remove': 0x70010d58c4,
        'rec4b_remove_a': 0x70010d5b68,
        'rec4b_remove_b': 0x70010d5c40,
        'rec4b_sweep': 0x70010d6a24,
        'rec4b_drop_fn': 0x70010d5d18,
        'rec4b_pick': 0x70010d6f60,
        'rec4b_fill_b': 0x70010d71ec,
        'rec4b_filter': 0x70010d7374,
        'rec4b_states': 0x70010d7610,
        'rec4b_pick_b': 0x70010d7930,
        'rec4b_upd_clear1': 0x70010d6204,
        'rec4b_upd_any_a': 0x70010d62fc,
        'rec4b_upd_keep_a': 0x70010d6360,
        'rec4b_upd_any_b': 0x70010d645c,
        'rec4b_upd_any_c': 0x70010d64c0,
        'rec4b_upd_any_d': 0x70010d651c,
        'rec4b_upd_clear2': 0x70010d66a4,
        'rec4b_upd_other1': 0x70010d681c,
        'rec4a_part_base': 0x7000c32160,
        'rec4a_part_size': 0x7000c32164,
        'rec4a_part_1': 0x7000c3217c,
        'rec4a_part_2': 0x7000c32184,
        'rec4a_part_rest': 0x7000c32188,
        'rec4a_make_fn': 0x7000c2a700,
        'rec4a_drop_rest': 0x7000c2ae64,
        'rec4a_release_fn': 0x7001517f8c,
        'rec4a_drop_1': 0x7000c2ae68,
        'rec4a_drop_0': 0x7000c2ae70,
        'q27_add2_bufs': 0x700112ec48,
        'q27_add2_bound': 0x700112ec4c,
        'q27_add2_cur': 0x700112ec58,
        'q27_add3_bufs': 0x700112f634,
        'q27_add3_bound': 0x700112f63c,
        'q27_add3_cur': 0x700112f640,
        'rec4c_ctor_end': 0x700112bd94,
        'rec4c_push_base': 0x7001150e74,
        'rec4c_push_bound': 0x7001150e78,
        'rec4c_push_pick': 0x7001150e7c,
        'rec4c_pop_1': 0x700115102c,
        'rec4c_pop_2': 0x7001151148,
        'rec4c_any_1a': 0x7001150ff0,
        'rec4c_any_1b': 0x70011510ec,
        'rec4c_any_2a': 0x7001151114,
        'rec4c_any_2b': 0x700115120c,
        'ring19_a_size': 0x7001060870,
        'ring19_a_count': 0x7001060880,
        'ring19_a_cores': 0x70010608a4,
        'ring19_b_size': 0x700106068c,
        'ring19_b_start': 0x700106069c,
        'ring19_b_cores': 0x70010606c0,
        'ring19_c_size': 0x7001060a24,
        'ring19_c_rest': 0x7001060b4c,
        'ring19_c_make_fn': 0x7001008c5c,
        'ring19_c_cores': 0x7001060b50,
        'ring19_a_zero': 0x7001060644,
        'ring19_b_zero': 0x70010606d4,
        'ring19_c_zero': 0x7001060720,
        'ring19_reset': 0x7001060774,
        'ring19_add_a': 0x7001059b58,
        'ring19_add_c': 0x700105a660,
        'ring19_wa_base': 0x7001059c5c,
        'ring19_wa_bound': 0x7001059e24,
        'ring19_wb_c0': 0x7001059f54,
        'ring19_wb_c1': 0x700105a058,
        'ring19_wb_ck': 0x700105a150,
        'ring19_wb_mov': 0x700105a164,
        'ring19_wb_ccmp': 0x700105a178,
        'ring19_wb_end': 0x700105a244,
        'ring19_wc_c0': 0x700105a744,
        'ring19_wc_c1': 0x700105a8d4,
        'ring19_wc_ck': 0x700105aa5c,
        'ring19_wc_mov': 0x700105aa6c,
        'ring19_wc_ccmp': 0x700105aa88,
        'ring19_wc_end': 0x700105abe0,
        'obj28_ifaces': 0x70010b6c98,
        'obj28_make': 0x70010b6cb0,
        'obj28_make_fn': 0x7001403f38,
        'obj28_destroy': 0x70010b6d98,
        'obj28_destroy_fn': 0x700140402c,
        'obj28_pick': 0x70010b71b4,
        'jobs_push': 0x7000f93ea0,
        'jobs_kick': 0x7000f93f78,
        'jobs_run': 0x7000f94580,
        'jobs_run_wait': 0x7000f94100,
        'jobs_group_setup': 0x7000ef6118,
        'jobs_tick': 0x700159ff00,
        'jobs_event_clear': 0x7000ef6648,
        'jobs_event_signal': 0x7000ef6640,
        'heap_ctor': 0x7000ea2a7c,
        'jobs_thread_cfg': 0x700154fa24,
        'jobs_thread_mask': 0x700154fae4,
        'jobs_bcast_next': 0x700127e58c,
        'jobs_bcast_fini': 0x700127e5a8,
        'jobs_group_ctor': 0x7000ef5e98,
        'jobs_group_grab': 0x7000ef6064,
        'jobs_group_fini': 0x7000ef65a4,
        'init_heap_size': 0x700000071c,
        'heap_create_exp': 0x7000003f10,
        'heap_ctor_b': 0x7000e9ef38,
        'heap_ctor_c': 0x7000e9fb18,
        'heap_root_flags': 0x7000004ba4,
        'root_heap_size': 0x7000c617c0,
        'jobs_pool_got': 0x700127f02c,
        'job_thread_proc': 0x70015198b4,
        'job_thread_res': 0x70015637f0,
        'view_rec_jobs': 0x7000fb754c,
        'view_rec_parts': 0x7000fb3ef4,
        'view_rec_items': 0x7000fb3d04,
        'view_rec_cbs': 0x7000faa864,
        'view_rec_inline': 0x7000fab6bc,
        'view_rec_consume': 0x7000fab5e0,
        'view_rec_part': 0x7000fab13c,
        'view_rec_ctx': 0x7000ebf0c4,
        'view_rec_ctx_fini': 0x7000ebf148,
        'view_rec_item_end': 0x7000eb0380,
        'view_rec_lock': 0x7000eaff08,
        'view_rec_unlock': 0x7000eaff18,
        'view_rec_record': 0x7000fab224,
        'view_rec_begin': 0x7000ef349c,
        'view_rec_end': 0x7000ef3608,
        'main_data': 0x7001b7a000,
        'main_end': 0x7001e98000,
    },
}


class SiteError(Exception):
    pass


def supported_version(build_id):
    """The VERSIONS entry of a main module build id (64 hex digits or the first 16), or None."""
    for v in VERSIONS:
        if v.build_id and build_id[:16] == v.build_id:
            return v
    return None


# ------------------------------------------------------------------ instructions

def sext(v, bits):
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


def is_adrp(w):
    return (w & 0x9F000000) == 0x90000000


def adrp_page(w, pc):
    return (pc & ~0xFFF) + (sext(((w >> 5) & 0x7FFFF) << 2 | ((w >> 29) & 3), 21) << 12)


def lo12(w):
    """(scaled low 12 bits, base register) of an ADD (immediate, 64-bit) or an LDR / STR (unsigned offset); else None"""
    if (w & 0xFFC00000) == 0x91000000:  # add xd, xn, #imm
        return (w >> 10) & 0xFFF, (w >> 5) & 31
    if (w & 0x3B000000) == 0x39000000:  # ldr / str (immediate, unsigned offset), integer or SIMD&FP
        scale = (w >> 30) & 3
        if w & 0x04000000 and (w >> 23) & 1:  # (a 128-bit SIMD&FP register)
            scale = 4
        return ((w >> 10) & 0xFFF) << scale, (w >> 5) & 31
    return None


def imm_value(w):
    """The immediate of an LDR / STR (unsigned offset, scaled), an ADD / SUB / CMP (immediate) or a MOVZ; else None"""
    if (w & 0x3B000000) == 0x39000000:
        return lo12(w)[0]
    if (w & 0x1F000000) == 0x11000000:
        return ((w >> 10) & 0xFFF) << (12 if (w >> 22) & 1 else 0)
    if (w & 0x7F800000) == 0x52800000:  # movz (32 / 64-bit)
        return ((w >> 5) & 0xFFFF) << (16 * ((w >> 21) & 3))
    return None


def mask(w, adrp_regs):
    """The bits of instruction w a pattern keeps (adrp_regs: registers an ADRP earlier in the pattern set)."""
    if (w & 0x7C000000) == 0x14000000:  # b, bl
        return 0xFC000000
    if (w & 0xFF000010) == 0x54000000 or (w & 0x7E000000) == 0x34000000:  # b.cond; cbz, cbnz
        return 0xFF00001F
    if (w & 0x7E000000) == 0x36000000:  # tbz, tbnz
        return 0xFFF8001F
    if (w & 0x1F000000) == 0x10000000:  # adr, adrp
        return 0x9F00001F
    if (w & 0x3B000000) == 0x18000000:  # ldr (literal)
        return 0xFF00001F
    lo = lo12(w)
    if lo and lo[1] in adrp_regs:
        return 0xFFC003FF
    return 0xFFFFFFFF


def make_pattern(words):
    """A pattern string for these instruction words: each word as hex, or value/mask where bits are masked."""
    out, adrp_regs = [], set()
    for w in words:
        m = mask(w, adrp_regs)
        if is_adrp(w):
            adrp_regs.add(w & 31)
        else:  # (most instructions write their bits 0-4 register: it holds no page any more)
            adrp_regs.discard(w & 31)
        out.append(f'{w:08x}' if m == 0xFFFFFFFF else f'{w & m:08x}/{m:08x}')
    return ' '.join(out)


def parse_pattern(p):
    words = []
    for t in p.split():
        v, _, m = t.partition('/')
        words.append((int(v, 16), int(m, 16) if m else 0xFFFFFFFF))
    return words


def compile_pattern(p):
    parts = []
    for v, m in parse_pattern(p):
        for i in range(4):
            vb, mb = (v >> 8 * i) & 0xFF, (m >> 8 * i) & 0xFF
            if mb == 0xFF:
                parts.append(re.escape(bytes([vb])))
            elif mb == 0:
                parts.append(b'.')
            else:
                parts.append(b'[' + b''.join(re.escape(bytes([b])) for b in range(256) if b & mb == vb & mb) + b']')
    return re.compile(b''.join(parts), re.DOTALL)


# ------------------------------------------------------------------ a main module

class Main:
    def __init__(self, nso, base=MAIN_BASE):
        self.nso, self.base = nso, base
        self.text_off, self.text_size = nso.text
        self.text = bytes(nso.image[self.text_off:self.text_off + self.text_size])

    def word(self, a):
        return struct.unpack_from('<I', self.nso.image, a - self.base)[0]

    def find(self, pattern, limit=2):
        """Addresses (word-aligned, in .text) where the pattern matches; at most `limit`."""
        rx, out, pos = compile_pattern(pattern), [], 0
        while len(out) < limit:
            m = rx.search(self.text, pos)
            if not m:
                break
            if m.start() % 4:
                pos = m.start() + 1
                continue
            out.append(self.base + self.text_off + m.start())
            pos = m.start() + 4
        return out

    def target(self, kind, a):
        """What the site of this kind at instruction a is (see the module docstring)."""
        w = self.word(a)
        if kind == 'code':
            return a
        if kind == 'call':
            if (w & 0x7C000000) != 0x14000000:
                raise SiteError(f'{a:#x} is not a branch ({w:08x})')
            return a + sext(w & 0x3FFFFFF, 26) * 4
        if kind == 'imm':
            v = imm_value(w)
            if v is None:
                raise SiteError(f'{a:#x} has no immediate ({w:08x})')
            return v
        if is_adrp(w):
            v = adrp_page(w, a)
        else:
            lo = lo12(w)
            if not lo:
                raise SiteError(f'{a:#x} does not complete an address ({w:08x})')
            for b in range(a - 4, a - 4 * 33, -4):  # its ADRP
                pw = self.word(b)
                if is_adrp(pw) and (pw & 31) == lo[1]:
                    v = adrp_page(pw, b) + lo[0]
                    break
            else:
                raise SiteError(f'{a:#x}: no ADRP for x{lo[1]} above it')
        if kind == 'got':
            return self.base + struct.unpack_from('<Q', self.nso.image, v - self.base)[0]
        return v

    def locate(self, s):
        """(address, or None when the pattern doesn't find exactly one place; how many places it matches) of a site"""
        if s.kind == 'data':
            return self.base + self.nso.data[0], 1
        if s.kind == 'end':
            return self.base + self.nso.size, 1
        hits = []
        for p, at in s.pattern if isinstance(s.pattern, list) else [(s.pattern, s.at)] if s.pattern else []:
            hits = self.find(p)
            if len(hits) == 1:
                return self.target(s.kind, hits[0] + 4 * at), 1
        return None, len(hits)

    def text_words(self, a, n):
        return [self.word(a + 4 * i) for i in range(n)]

    def function_of(self, a):
        """(start, end) of the function holding a, from the module's unwind info (None: none there)."""
        import bisect
        from analyze import eh_frame_extents
        if not hasattr(self, 'fdes'):
            self.fdes = sorted((self.base + s, self.base + e) for s, e in eh_frame_extents(self.nso) if e > s)
        i = bisect.bisect_right(self.fdes, (a, 1 << 64)) - 1
        return self.fdes[i] if i >= 0 and a < self.fdes[i][1] else None

    def pattern_for(self, a, kind='code', shortest=8, longest=24):
        """A pattern that finds only a: (pattern, at). It stays inside a's function (the code around it may change
        on its own), starts as close above a as it can - at the ADRP for a ref or got site, whose address bits it
        masks then - and has `shortest` words at least (or the whole function): short ones are likelier to find
        another place in a new build."""
        f0, f1 = self.function_of(a) or (a - 64, a + 128)
        first = 0
        if kind in ('ref', 'got') and not is_adrp(self.word(a)):
            reg = lo12(self.word(a))[1]
            first = next(i for i in range(1, 33) if is_adrp(self.word(a - 4 * i)) and self.word(a - 4 * i) & 31 == reg)
        for back in range(first, (a - f0) // 4 + 1):
            s = a - 4 * back
            room = (f1 - s) // 4
            for n in range(min(max(shortest, back + 1), room), min(longest, room) + 1):
                p = make_pattern(self.text_words(s, n))
                if self.find(p) == [s]:
                    return p, back
        raise SiteError(f'no pattern inside its function finds only {a:#x}')


def common_pattern(ma, a, mb, b, kind='code', shortest=8, longest=32):
    """A pattern that finds only a in main module ma and only b in mb, the same code in two versions: the bits both
    keep (and make_pattern keeps). For an imm site the immediate itself is always masked. (pattern, at)"""
    fa, fb = ma.function_of(a) or (a - 64, a + 128), mb.function_of(b) or (b - 64, b + 128)
    first = 0
    if kind in ('ref', 'got') and not is_adrp(ma.word(a)):
        reg = lo12(ma.word(a))[1]
        first = next(i for i in range(1, 33) if is_adrp(ma.word(a - 4 * i)) and ma.word(a - 4 * i) & 31 == reg)
    for back in range(first, min(a - fa[0], b - fb[0]) // 4 + 1):
        sa, sb = a - 4 * back, b - 4 * back
        room = min(fa[1] - sa, fb[1] - sb) // 4
        for n in range(min(max(shortest, back + 1), room), min(longest, room) + 1):
            out = []
            for i, ((va, ka), (vb, kb)) in enumerate(zip(parse_pattern(make_pattern(ma.text_words(sa, n))),
                                                         parse_pattern(make_pattern(mb.text_words(sb, n))))):
                k = ka & kb & ~(va ^ vb) & 0xFFFFFFFF
                if kind == 'imm' and i == back:
                    k &= 0xFFE0001F if (va & 0x7F800000) == 0x52800000 else 0xFFC003FF
                out.append(f'{va & k:08x}' if k == 0xFFFFFFFF else f'{va & k:08x}/{k:08x}')
            p = ' '.join(out)
            if ma.find(p) == [sa] and mb.find(p) == [sb]:
                return p, back
    raise SiteError(f'no pattern inside both functions finds only {a:#x} and {b:#x}')


def resolve(main, known=None, log=None):
    """{site name: address} for this main module. known: the addresses listed for its build (or None)."""
    out, problems = {}, []
    for s in SITES:
        try:
            found, hits = main.locate(s)
        except SiteError as e:
            problems.append(f'{s.name}: {e}')
            continue
        k = (known or {}).get(s.name)
        if found is not None and k is not None and found != k:
            raise SiteError(f'{s.name}: the pattern finds {found:#x}, the table says {k:#x}')
        if found is None and k is None:
            if s.name in OPTIONAL:
                continue
            problems.append(f'{s.name}: the pattern matches {hits} places' + (' or more' if hits > 1 else ''))
            continue
        if found is None and log:
            log(f'game_sites: {s.name}: the pattern matches {hits} places; {k:#x} from the table')
        out[s.name] = found if found is not None else k
    if problems:
        raise SiteError('; '.join(problems))
    for g in OPTIONAL_GROUPS:
        missing = [n for n in g if n not in out]
        if not missing:
            continue
        if log:
            log(f'game_sites: {", ".join(missing)} not found: {", ".join(g)} left out')
        for n in g:
            out.pop(n, None)
    return out


def for_module(nso, base, allow_unsupported=False, log=None):
    """(version name, sites) for the main module nso at base. Refuses a build that VERSIONS doesn't support, unless
    allow_unsupported (bringing up a new version: its sites must then all be found by pattern)."""
    v = supported_version(nso.build_id)
    if not v or not v.supported:
        what = f'version {v.name}, not supported yet' if v else 'not a version Savage Wind knows'
        if not allow_unsupported:
            raise SiteError(f'main module build {nso.build_id[:16]}: {what}')
        if log:
            log(f'game_sites: main module build {nso.build_id[:16]}: {what}; sites by pattern only')
    name = v.name if v else nso.build_id[:16]
    return name, resolve(Main(nso, base), KNOWN.get(name) if v and v.supported else None, log)


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ('find', 'pattern', 'common'):
        sys.exit(__doc__)
    from nso import Nso
    m = Main(Nso(sys.argv[2]))
    if sys.argv[1] == 'common':
        mb, a, b = Main(Nso(sys.argv[4])), int(sys.argv[3], 16), int(sys.argv[5], 16)
        kind = sys.argv[6] if len(sys.argv) > 6 else 'code'
        p, at = common_pattern(m, a, mb, b, kind)
        print(f"    Site('<name>', '{kind}', '{p}', {at}),  # -> {m.target(kind, a):#x} / {mb.target(kind, b):#x}")
        return
    if sys.argv[1] == 'pattern':
        a = int(sys.argv[3], 16)
        kind = sys.argv[4] if len(sys.argv) > 4 else 'code'
        p, at = m.pattern_for(a, kind)
        print(f"    Site('<name>', '{kind}', '{p}', {at}),  # -> {m.target(kind, a):#x}")
        return
    v = supported_version(m.nso.build_id)
    known = KNOWN.get(v.name, {}) if v else {}
    print(f'main module build {m.nso.build_id[:16]}: ' + (f'version {v.name}' + ('' if v.supported else
          ' (not supported yet)') if v else 'not in VERSIONS'))
    bad = 0
    for s in SITES:
        try:
            found, hits = m.locate(s)
        except SiteError as e:
            found, hits = None, str(e)
        k = known.get(s.name)
        state = (f'{hits} matches' if found is None else
                 'ok' if k == found else 'new' if k is None else f'TABLE SAYS {k:#x}')
        bad += state not in ('ok', 'new')
        print(f"        '{s.name}': {found:#x},  # {state}" if found is not None else
              f"        # '{s.name}': {state}" + (f' (table {k:#x})' if k is not None else ''))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
