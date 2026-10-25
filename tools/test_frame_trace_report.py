import unittest
from frame_trace_report import capture_report, frame_stats, percentile


class FrameTraceTests(unittest.TestCase):
    def test_texture_work_is_counted_not_timed_and_missing_not_zero(self):
        log = '[ft] 0 0 0\n[ft] 5 1 0\n[ft] 10 2 0\n'
        log += '[frame-textures] 1 checks 11 samples 10 sampled_words 24000 full_bytes 4096 dynamic_bytes 2048 changed_uploads 1 changed_storage_bytes 1024\n'
        work = frame_stats(log, 0, 1)['texture_work']
        self.assertEqual(work['observations'], 1)
        self.assertEqual(work['missing_or_ambiguous_frames'], 1)
        self.assertEqual(work['mean_per_frame']['full_bytes'], 4096)
        self.assertEqual(work['mean_per_frame']['sampled_words'], 24000)
        self.assertNotIn('ms', work)

    def test_recorder_waits_are_separate_from_descriptor_waits(self):
        log = '[ft] 0 0 0\n[ft] 5 1 0\n[ft] 10 2 0\n'
        log += '[frame-work] 1 descriptor_ring_wait_ns 0 descriptor_ready_wait_ns 1500000\n'
        log += '[vkq-work] 1 recorder_ring_wait_ns 1000 recorder_drain_wait_ns 600000\n'
        log += '[vkq-work] 2 recorder_ring_wait_ns 0 recorder_drain_wait_ns 0\n' * 2
        result = frame_stats(log, 0, 1)
        self.assertEqual(result['descriptor_waits']['mean_ms']['descriptor_ready_wait_ms'], 1.5)
        self.assertEqual(result['recorder_waits']['observations'], 1)
        self.assertEqual(result['recorder_waits']['missing_or_ambiguous_frames'], 1)
        self.assertEqual(result['recorder_waits']['mean_ms'], {'recorder_ring_wait_ms': .001, 'recorder_drain_wait_ms': .6})

    def test_wait_totals_join_by_frame_and_missing_is_not_zero(self):
        log = '[ft] 0 0 0\n[ft] 5 1 0\n[ft] 10 2 0\n[ft] 15 3 0\n'
        log += '[frame-work] 1 descriptor_ring_wait_ns 800000 descriptor_ready_wait_ns 1500000\n'
        log += '[frame-work] 2 descriptor_ring_wait_ns 0 descriptor_ready_wait_ns 0\n' * 2
        waits = frame_stats(log, 0, 1)['descriptor_waits']
        self.assertEqual(waits['observations'], 1)
        self.assertEqual(waits['missing_or_ambiguous_frames'], 2)
        self.assertEqual(waits['mean_ms'], {'descriptor_ring_wait_ms': .8, 'descriptor_ready_wait_ms': 1.5})

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 3, 5, 7], .5), 4)
        self.assertEqual(percentile([1, 3, 5, 7], 1), 7)

    def test_microspike_is_relative_not_freeze_threshold(self):
        ms = 0
        lines = ['[ft] 0 0 0']
        for frame in range(1, 101):
            ms += 5.0 if frame != 80 else 6.1
            lines.append(f'[ft] {ms:.3f} {frame} 0')
        result = frame_stats('\n'.join(lines), 0, 1)
        self.assertEqual(result['microspike_count'], 1)
        self.assertEqual(result['largest_spikes'][0]['frame'], 80)

    def test_cap_and_counter_gap_are_not_stutters(self):
        log = '[ft] 0 0 0\n[ft] 5 1 0\n[ft] 38 2 30\n[ft] 71 3 30\n[ft] 100 4 0\n[ft] 105 5 0\n[ft] 200 50 0\n[ft] 205 51 0'
        result = frame_stats(log, 0, 1)
        self.assertEqual(result['frames'], 3)
        self.assertAlmostEqual(result['max_ms'], 5)

    def test_no_cross_clock_arithmetic_or_submission_double_count(self):
        def event(name, pid, tid, dur):
            return {'ph': 'X', 'pid': pid, 'tid': tid, 'name': name,
                    'ts': 900000 if pid == 1 else 0, 'dur': dur,
                    'args': {'frame': 12, 'seq': 90, 'draws': 1, 'first_shader': 'abcd'}}
        data = {'metadata': {'trigger_frame': 12}, 'traceEvents': [
            event('Frame interval', 1, 0, 6100), event('GPU wait', 1, 3, 1200),
            event('pass', 2, 10, 800), event('GPU submission', 2, 11, 1500)]}
        row = capture_report(data)['frames'][0]
        self.assertEqual(row['gpu_passes'], 1)
        self.assertEqual(row['largest_gpu_spans'][0]['ms'], .8)
        self.assertEqual(row['present_interval_ms'], 6.1)

    def test_overlapping_queue_and_operation_intervals_are_separate(self):
        events = []
        for lane, name in [(1, 'Translate'), (6, 'Operation'), (7, 'Queue'), (8, 'Fence'), (9, 'Present')]:
            events.append({'ph': 'X', 'pid': 1, 'tid': lane, 'name': name,
                           'ts': 12.5, 'dur': 2000, 'args': {'frame': 12, 'seq': 90}})
        data = {'metadata': {'trigger_frame': 12, 'cpu_origin_ns': 1000000000}, 'traceEvents': events}
        row = capture_report(data)['frames'][0]
        self.assertEqual([e['name'] for e in row['largest_cpu_spans']], ['Translate'])
        self.assertEqual(row['largest_command_queue_delays'][0]['name'], 'Queue')
        self.assertEqual(row['largest_cpu_operation_intervals'][0]['begin_ns'], 1000012500)
        self.assertEqual(row['largest_fence_wait_phases'][0]['end_ns'], 1002012500)


if __name__ == '__main__':
    unittest.main()
