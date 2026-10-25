import unittest
from cube_compare import compare


class CubeCompareTests(unittest.TestCase):
    def log(self, capped=False, gap=False):
        text, t = [], 0
        for f in range(60):
            if f in (10, 20, 30, 40):
                text.append(f'[cube-ab] frame {f} layered {int(f in (20, 40))}')
            t += 7 if f in (10, 30) else 5
            if gap and f == 9:
                continue
            text.append(f'[ft] {t} {f} {30 if capped and f == 10 else 0}')
        return '\n'.join(text)

    def test_pairs_and_cycle_cost(self):
        r = compare(self.log(), 0, 1)
        self.assertEqual(r['pairs'], 2)
        self.assertAlmostEqual(r['paired_mean_saved_ms'], 2)
        self.assertAlmostEqual(r['original']['mean_excess_over_neighbors_ms'], 2)
        self.assertAlmostEqual(r['ten_frame_cycles']['0']['mean_ms'], 52)
        self.assertAlmostEqual(r['ten_frame_cycles']['1']['mean_ms'], 50)

    def test_caps_and_counter_gaps_excluded(self):
        for kw in ({'capped': True}, {'gap': True}):
            r = compare(self.log(**kw), 0, 1)
            self.assertEqual(r['original']['updates'], 1)
            self.assertEqual(r['layered']['updates'], 2)


if __name__ == '__main__':
    unittest.main()
