import json
import unittest
from critical_path_report import gpu_to_cpu, gap_parts, report


class ClockMappingTests(unittest.TestCase):
    def test_signed_wrap_and_fractional_period(self):
        c = {'gpu_tick': 250, 'cpu_ns': 1000, 'period_ns': .5, 'valid_mask': 255}
        self.assertEqual(gpu_to_cpu(4, c), 1005)
        self.assertEqual(gpu_to_cpu(240, c), 995)

    def test_large_ticks_subtracted_before_scaling(self):
        c = {'gpu_tick': (1 << 64) - 4, 'cpu_ns': 1000, 'period_ns': 1, 'valid_mask': (1 << 64) - 1}
        self.assertEqual(gpu_to_cpu(2, c), 1006)

    def test_gap_classification_no_double_count(self):
        self.assertEqual(gap_parts(100, 200, 150, 170), (100, 50, 20, 30))
        self.assertEqual(gap_parts(100, 200, 50, 250), (100, 0, 100, 0))
        self.assertEqual(gap_parts(200, 100, 50, 70), (0, 0, 0, 0))

    def test_missing_calibration_fails(self):
        with self.assertRaises(ValueError):
            report('')

    def test_excludes_interframe_and_missing_sequence_gaps(self):
        c = {'gpu_tick': 0, 'cpu_ns': 1000, 'period_ns': 1, 'valid_mask': 65535, 'max_deviation_ns': 1}
        lines = ['[vkclock] ' + json.dumps(c)]
        for frame, seq, begin, end in ((1, 1, 100, 150), (1, 2, 200, 250), (2, 3, 500, 550), (2, 5, 800, 850)):
            r = {'frame': frame, 'seq': seq, 'enqueue_ns': 1000+begin-30, 'driver_ns': 1000+begin-20,
                 'return_ns': 1000+begin-10, 'gpu_begin_tick': begin, 'gpu_end_tick': end, 'reasons': 0}
            lines.append('[vktrace] ' + json.dumps(r))
        out = report('\n'.join(lines), 1, 2, 1)
        self.assertEqual(out['frames'], 1)
        self.assertEqual(out['mean']['gap_ms'], 50/1e6)
        self.assertEqual(out['rejected']['nonconsecutive_pair'], 1)


if __name__ == '__main__':
    unittest.main()
