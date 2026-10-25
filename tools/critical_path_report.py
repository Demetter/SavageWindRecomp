"""Correlate Vulkan submissions using explicit CPU/GPU timestamp calibration.

Never infer occupancy from timestamp spans. Gaps here are intervals on this queue,
not proof that the whole GPU is idle. Only consecutive submissions within one
present group are classified; inter-frame gaps are deliberately excluded.
"""
import argparse
import bisect
import collections
import json
from pathlib import Path
import re
import statistics


def gpu_to_cpu(tick, calibration):
    mask = calibration['valid_mask']
    delta = (tick - calibration['gpu_tick']) & mask
    if delta >= (mask + 1) // 2:
        delta -= mask + 1
    # Subtract device ticks before scaling, preserving precision near uint64 max.
    return calibration['cpu_ns'] + delta * calibration['period_ns']


def gap_parts(previous_end, begin, driver, returned):
    gap = max(0, begin - previous_end)
    late = min(gap, max(0, driver - previous_end))
    call = min(gap - late, max(0, returned - max(previous_end, driver)))
    return gap, late, call, max(0, gap - late - call)


def report(text, first=4000, last=8000, detail_frame=4500):
    clocks = sorted((json.loads(m[1]) for m in re.finditer(r'^\[vkclock\] (\{.*\})$', text, re.M)),
                    key=lambda c: c['cpu_ns'])
    clocks = [c for c in clocks if 0 < c['max_deviation_ns'] <= 100000 and
              c['valid_mask'] > 0 and not (c['valid_mask'] & (c['valid_mask'] + 1)) and c['period_ns'] > 0]
    if not clocks:
        raise ValueError('No usable calibrated timestamps; raw CPU/GPU clocks must not be subtracted.')
    times = [c['cpu_ns'] for c in clocks]
    frames = collections.defaultdict(list)
    rejected = collections.Counter()
    for m in re.finditer(r'^\[vktrace\] (\{.*\})$', text, re.M):
        row = json.loads(m[1])
        if not first <= row['frame'] <= last:
            continue
        at = bisect.bisect_left(times, row['enqueue_ns'])
        c = min(clocks[max(0, at-1):at+1], key=lambda c: abs(c['cpu_ns'] - row['enqueue_ns']))
        if abs(c['cpu_ns'] - row['enqueue_ns']) > 2000000000:
            rejected['calibration_older_than_2s'] += 1
            continue
        if row.get('driver_ns', 0) < row['enqueue_ns'] or row.get('return_ns', 0) < row['driver_ns']:
            rejected['missing_or_invalid_driver_times'] += 1
            continue
        row['gpu_begin_cpu_ns'] = gpu_to_cpu(row['gpu_begin_tick'], c)
        row['gpu_end_cpu_ns'] = gpu_to_cpu(row['gpu_end_tick'], c)
        row['calibration_deviation_ns'] = c['max_deviation_ns']
        if row['gpu_end_cpu_ns'] < row['gpu_begin_cpu_ns'] or row['gpu_begin_cpu_ns'] + c['max_deviation_ns'] < row['driver_ns']:
            rejected['inconsistent_clock_mapping'] += 1
            continue
        frames[row['frame']].append(row)
    per_frame = []
    for frame, rows in sorted(frames.items()):
        rows.sort(key=lambda r: r['seq'])
        totals = [0.0] * 4
        pairs = 0
        for a, b in zip(rows, rows[1:]):
            if b['seq'] != a['seq'] + 1:
                rejected['nonconsecutive_pair'] += 1
                continue
            parts = gap_parts(a['gpu_end_cpu_ns'], b['gpu_begin_cpu_ns'], b['driver_ns'], b['return_ns'])
            totals = [x+y for x, y in zip(totals, parts)]
            pairs += 1
        if pairs:
            per_frame.append({'frame': frame, 'consecutive_pairs': pairs,
                              **{k: v/1e6 for k, v in zip(('gap_ms', 'late_cpu_submission_ms', 'inside_submit_call_ms',
                                                          'after_submit_return_ms'), totals)}})
    if not per_frame:
        raise ValueError('No consecutive submissions in the requested frame window.')
    keys = ('gap_ms', 'late_cpu_submission_ms', 'inside_submit_call_ms', 'after_submit_return_ms')
    detail = frames.get(detail_frame, [])
    origin = detail[0]['enqueue_ns'] if detail else 0
    return {
        'frames': len(per_frame), 'frame_window': [first, last], 'rejected': dict(rejected),
        'calibrations': len(clocks), 'max_calibration_deviation_ns': max(c['max_deviation_ns'] for c in clocks),
        'mean': {k: statistics.mean(f[k] for f in per_frame) for k in keys},
        'median': {k: statistics.median(f[k] for f in per_frame) for k in keys},
        'detail_frame': detail_frame,
        'detail_submissions': [{'seq': r['seq'], 'reasons': r['reasons'],
            **{k: (r[v]-origin)/1e6 for k, v in (
                ('enqueue_ms', 'enqueue_ns'), ('driver_ms', 'driver_ns'), ('returned_ms', 'return_ns'),
                ('gpu_begin_ms', 'gpu_begin_cpu_ns'), ('gpu_end_ms', 'gpu_end_cpu_ns'))}} for r in detail],
        'notes': [
            'Diagnostic only; timestamp instrumentation changes performance.',
            'GPU spans and queue gaps are not whole-device utilization or active execution cycles.',
            'Only consecutive submissions inside each present group; excludes its first/inter-frame gap.',
            'Late CPU submission means the next driver call began after the preceding GPU end timestamp.',
            'Clock mapping uses the nearest calibration within 2 s and rejects deviations above 0.1 ms.',
            'Calibration deviation is not a bound on drift between calibration points.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--first-frame', type=int, default=4000)
    parser.add_argument('--last-frame', type=int, default=8000)
    parser.add_argument('--detail-frame', type=int, default=4500)
    args = parser.parse_args()
    print(json.dumps(report(args.log.read_text(errors='replace'), args.first_frame, args.last_frame, args.detail_frame), indent=2))


if __name__ == '__main__':
    main()
