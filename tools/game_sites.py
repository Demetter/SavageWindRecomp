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
     'cstate_slot_base', 'cstate_slot_bound'),
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
