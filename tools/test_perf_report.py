import unittest
from perf_report import report, trace_report


class PerfReportTests(unittest.TestCase):
    def test_mixed_modes_require_block_analysis(self):
        text = '[producer-material-ab] frame 1 block 0 enabled 0\n[ft] 999 1 0\n[ft] 1001 2 0\n[ft] 1006 3 0\n[ft] 1011 4 0\n'
        r = report(text, 1, 1.01)
        self.assertFalse(r['valid_uncapped_capture'])
        self.assertTrue(any('Mixed producer-material modes' in n for n in r['notes']))

    def test_stalled_run_cannot_pass_with_shorter_window(self):
        text = '[ft] 44000 1 0\n[ft] 45000 2 0\n[ft] 45005 3 0\n[ft] 52610 4 0\n'
        r = report(text, 45, 95)
        self.assertFalse(r['valid_uncapped_capture'])
        self.assertFalse(r['requested_window_covered'])

    def test_complete_quiet_window_passes(self):
        text = '[ft] 999 1 0\n[ft] 1001 2 0\n[ft] 1006 3 0\n[ft] 1011 4 0\n'
        r = report(text, 1, 1.01)
        self.assertTrue(r['valid_uncapped_capture'])

    def test_draw_batch_probe_is_not_quiet_throughput(self):
        text = '[ft] 999 1 0\n[ft] 1001 2 0\n[ft] 1006 3 0\n[ft] 1011 4 0\n[draw-batch-probe] frame 4 draws 10\n'
        self.assertFalse(report(text, 1, 1.01)['valid_uncapped_capture'])

    def test_missing_frame_or_later_crash_invalidates_capture(self):
        text = '[ft] 999 1 0\n[ft] 1001 2 0\n[ft] 1006 3 0\n[ft] 1011 4 0\n'
        for broken in (text.replace('1006 3', '1006 4'), text + 'FATAL: failed\n', text + '*** exception 80000001 accessing 0x1234\n'):
            self.assertFalse(report(broken, 1, 1.01)['valid_uncapped_capture'])

    def test_trace_queue_failures_invalidate_entire_capture(self):
        for marker in ('[trace-log] flush timeout or I/O error', '[trace-log] producer backpressure: 1 waits'):
            r = report('[ft] 1000 1 0\n[ft] 1005 2 0\n' + marker, 0, 2)
            self.assertFalse(r['valid_uncapped_capture'])

    def test_exact_frame_times_preserve_stutter(self):
        r = report("[ft] 10000.000 100 0\n[ft] 10005.000 101 0\n[ft] 10050.000 102 0", 10, 11)
        self.assertAlmostEqual(r["fps"], 40)
        self.assertAlmostEqual(r["max_frame_ms"], 45)

    def test_menu_makes_uncapped_comparison_invalid(self):
        r = report("[ft] 10000 100 0\n[ft] 10005 101 30\n[ft] 10040 102 0", 10, 11)
        self.assertFalse(r["valid_uncapped_capture"])
        self.assertEqual(r["frame_rate_targets"], [0, 30])

    def test_unequal_intervals_use_frame_deltas(self):
        text = """[vk] t=10.0 frame 1000 200.0 fps max 5ms
[vk] t=12.0 frame 1400 200.0 fps max 5ms
[vk] saved build/frames/frame_1.png
[vk] t=15.0 frame 1700 100.0 fps max 900ms
"""
        r = report(text, 10, 15)
        self.assertEqual(r["fps"], 140)  # neither 150 nor 166.7
        self.assertEqual(r["max_frame_ms"], 900)
        self.assertEqual(len(r["capture_events"]), 1)

    def test_excludes_partial_intervals_and_prior_capture(self):
        text = """[vk] saved before.png
[vk] t=9.9 frame 100 10 fps max 900ms
[vk] t=11.9 frame 500 200 fps max 5ms
[vk] t=13.9 frame 900 200 fps max 6ms
[vk] t=15.9 frame 950 25 fps max 900ms
"""
        r = report(text, 10, 15)
        self.assertAlmostEqual(r["fps"], 200)
        self.assertEqual(r["max_frame_ms"], 6)
        self.assertFalse(r["capture_events"])

    def test_restarted_capture_rejected(self):
        with self.assertRaises(ValueError):
            report("[vk] t=10 frame 100 1 fps max 1ms\n[ vk]\n"
                   "[vk] t=12 frame 90 1 fps max 1ms", 0, 20)

    def test_legacy_timeline_is_not_busy_time(self):
        text = "[vk] t=1 frame 10 10 fps max 1ms\n[gtl] 100 draws: pieces: 1>4\n[ vk]\n[vk] t=2 frame 20 10 fps max 1ms"
        r = report(text, 0, 3)
        self.assertIn("not GPU busy", r["notes"][-1])
        self.assertIsNone(trace_report(text))

    def test_frame_trace_is_not_a_throughput_benchmark(self):
        r = report('[frame-trace] rolling capture: enabled\n[ft] 1000 1 0\n[ft] 1005 2 0\n', 0, 2)
        self.assertFalse(r['valid_uncapped_capture'])
        self.assertIn('diagnostic only', r['notes'][-1])

    def test_draw_probes_are_not_throughput_benchmarks(self):
        for marker in ('[cube-verify]', '[small-pass]', '[cube-ab]', '[bindings-verify]', '[texture-bindings]', '[uniform-verify]', '[ubo-audit]',
                       '[ww-probe]', '[geometry-scan]', '[depth-verify]', '[ww-parallel]', '[texture-life]', '[compact-records]', '[geometry-flush]', '[geometry-guard]', '[compute-opt-audit]', '[feedback-probe]', '[pipeline-exec]', '[pipeline-stat]', '[hardware-profile]', '[stage-reuse-audit]', '[texture-hash-audit]', '[compute-path-audit]', '[range-query-audit]', '[producer-material-audit]'):
            r = report(marker + ' diagnostic\n[ft] 999 1 0\n[ft] 1000 2 0\n[ft] 1005 3 0\n[ft] 1006 4 0\n', 1, 1.005)
            self.assertFalse(r['valid_uncapped_capture'])
            self.assertTrue(r['requested_window_covered'])
            self.assertTrue(r['contiguous_frame_counters'])


if __name__ == "__main__":
    unittest.main()
