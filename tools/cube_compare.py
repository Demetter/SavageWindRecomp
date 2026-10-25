"""Compare alternating cubemap-update paths inside one run (SAVAGEWIND_VK_CUBE_LAYERS=2).

Only eligible update frames in the requested uncapped interval are compared. Report neighboring non-update
frames as context; this diagnostic estimates update cost, not whole-game throughput or display scanout.
"""
import argparse
import json
import re
import statistics
from pathlib import Path
from frame_trace_report import percentile


def compare(text, start=45, end=65):
    modes = {int(f): int(m) for f, m in re.findall(r'\[cube-ab\] frame (\d+) layered ([01])', text)}
    intervals = {}
    previous = None
    for t, f, cap in re.findall(r'\[ft\] ([\d.]+) (\d+) (\d+)', text):
        t, f, cap = float(t) / 1000, int(f), int(cap)
        if previous:
            pt, pf, pc = previous
            if not cap and not pc and f == pf + 1 and start <= pt < t <= end:
                intervals[f] = (t - pt) * 1000
        previous = t, f, cap
    rows = {0: [], 1: []}
    normalized = {0: [], 1: []}
    for f, mode in sorted(modes.items()):
        if f not in intervals:
            continue
        nearby = [intervals[n] for n in range(f-4, f+5) if n in intervals and n not in modes]
        if len(nearby) < 4:
            continue
        rows[mode].append((f, intervals[f]))
        normalized[mode].append(intervals[f] - statistics.median(nearby))
    def summary(mode):
        values = [v for _, v in rows[mode]]
        if not values:
            return {'updates': 0}
        return {'updates': len(values), 'mean_ms': statistics.mean(values), 'median_ms': statistics.median(values),
                'p95_ms': percentile(values, .95), 'p99_ms': percentile(values, .99),
                'mean_excess_over_neighbors_ms': statistics.mean(normalized[mode]),
                'median_excess_over_neighbors_ms': statistics.median(normalized[mode])}
    # Pair consecutive old/new updates when separated by <= 12 frames; scene changes are bounded locally.
    ordered = sorted((f, m, intervals[f]) for m in rows for f, _ in rows[m])
    pairs = []
    for a, b in zip(ordered[::2], ordered[1::2]):
        if a[1] != b[1] and b[0] - a[0] <= 12:
            pairs.append(a[2] - b[2] if a[1] == 0 else b[2] - a[2])
    cycles = {0: [], 1: []}
    for f, mode in sorted(modes.items()):
        if all(n in intervals for n in range(f, f+10)) and not any(n in modes for n in range(f+1, f+10)):
            cycles[mode].append(sum(intervals[n] for n in range(f, f+10)))
    return {'original': summary(0), 'layered': summary(1), 'pairs': len(pairs),
            'paired_mean_saved_ms': statistics.mean(pairs) if pairs else None,
            'paired_median_saved_ms': statistics.median(pairs) if pairs else None,
            'ten_frame_cycles': {str(m): {'cycles': len(v), 'mean_ms': statistics.mean(v) if v else None}
                                 for m, v in cycles.items()},
            'note': 'Alternating eligible update frames, application present intervals. Positive saved time favors layered rendering. Not a throughput benchmark.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('log', type=Path)
    ap.add_argument('--start', type=float, default=45)
    ap.add_argument('--end', type=float, default=65)
    a = ap.parse_args()
    print(json.dumps(compare(a.log.read_text(errors='replace'), a.start, a.end), indent=2))
