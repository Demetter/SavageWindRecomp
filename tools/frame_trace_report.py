"""Summarize a frame-trace JSON capture and/or per-present [ft] log.

The Chrome/Perfetto JSON has separate CPU/GPU clock origins. Join by frame/seq,
never subtract their timestamps or add overlapping wait/translation intervals.
"""
import argparse
import collections
import json
import math
from pathlib import Path
import re
import statistics


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def frame_stats(log, start=45, end=65):
    samples = []
    selected_frames = []
    previous = None
    history = collections.deque(maxlen=64)
    spikes = []
    for match in re.finditer(r'^\[ft\] ([\d.]+) (\d+) (\d+)$', log, re.M):
        time, frame, target = float(match[1]) / 1000, int(match[2]), int(match[3])
        if previous:
            pt, pf, pc = previous
            if time <= pt or frame != pf + 1:
                history.clear()
            elif target == pc == 0:
                ms = (time - pt) * 1000
                if start <= pt and time <= end:
                    samples.append(ms)
                    selected_frames.append(frame)
                    if len(history) >= 32:
                        baseline = statistics.median(history)
                        if ms > baseline + max(0.5, baseline * 0.2):
                            spikes.append({'frame': frame, 'time_s': time, 'ms': ms, 'preceding_median_ms': baseline})
                history.append(ms)
            else:
                history.clear()
        previous = time, frame, target
    if not samples:
        raise ValueError('No contiguous uncapped frame intervals in the requested window.')
    result = {'frames': len(samples), 'mean_ms': statistics.mean(samples),
            'fps': 1000 / statistics.mean(samples), 'stddev_ms': statistics.pstdev(samples),
            **{name: percentile(samples, q) for name, q in [('p50_ms', .5), ('p90_ms', .9), ('p95_ms', .95), ('p99_ms', .99), ('p99_9_ms', .999)]},
            'max_ms': max(samples), 'microspike_count': len(spikes),
            'microspike_definition': 'More than preceding 64-frame median + max(0.5 ms, 20%); requires 32 prior uncapped frames.',
            'largest_spikes': sorted(spikes, key=lambda s: s['ms'], reverse=True)[:20],
            'note': 'Application present intervals, not display scanout; excludes caps, restarts, gaps, and intervals crossing window edges.'}
    for tag, prefix, result_key in (('frame-work', 'descriptor', 'descriptor_waits'), ('vkq-work', 'recorder', 'recorder_waits')):
        if f'[{tag}]' not in log:
            continue
        second = 'ready' if prefix == 'descriptor' else 'drain'
        work, duplicate = {}, set()
        pattern = rf'^\[{tag}\] (\d+) {prefix}_ring_wait_ns (\d+) {prefix}_{second}_wait_ns (\d+)$'
        for m in re.finditer(pattern, log, re.M):
            f = int(m[1])
            if f in work:
                duplicate.add(f)
            work[f] = {f'{prefix}_ring_wait_ms': int(m[2]) / 1e6, f'{prefix}_{second}_wait_ms': int(m[3]) / 1e6}
        for f in duplicate:
            work.pop(f, None)
        rows = [work[f] for f in selected_frames if f in work]
        result[result_key] = {
            'observations': len(rows), 'missing_or_ambiguous_frames': len(selected_frames) - len(rows),
            'mean_ms': {k: statistics.mean(r[k] for r in rows) for k in rows[0]} if rows else None,
            'note': 'Cumulative wait deltas between presents; producer and recorder waits overlap each other and other CPU spans. Do not add them to frame time. Long spans can cross interval boundaries.'}
        for spike in result['largest_spikes']:
            spike[result_key] = work.get(spike['frame'])
    if '[frame-textures]' in log:
        names = ('checks', 'samples', 'sampled_words', 'full_bytes', 'dynamic_bytes', 'changed_uploads', 'changed_storage_bytes')
        pattern = r'^\[frame-textures\] (\d+)' + ''.join(rf' {key} (\d+)' for key in names) + '$'
        work, duplicate = {}, set()
        for m in re.finditer(pattern, log, re.M):
            f = int(m[1])
            if f in work: duplicate.add(f)
            work[f] = {key: int(m[i + 2]) for i, key in enumerate(names)}
        for f in duplicate: work.pop(f, None)
        rows = [work[f] for f in selected_frames if f in work]
        result['texture_work'] = {'observations': len(rows), 'missing_or_ambiguous_frames': len(selected_frames) - len(rows),
            'mean_per_frame': {key: statistics.mean(r[key] for r in rows) for key in names} if rows else None,
            'note': 'Work counters, not durations. Sampled words exclude tail bytes/cache-line traffic. Upload counts cover hash-detected changes only, not every texture-transfer path.'}
        for spike in result['largest_spikes']: spike['texture_work'] = work.get(spike['frame'])
    return result


def capture_report(data):
    meta = data['metadata']
    center = meta['trigger_frame']
    by_frame = collections.defaultdict(list)
    for event in data['traceEvents']:
        if event.get('ph') == 'X':
            by_frame[event['args']['frame']].append(event)
    rows = []
    for frame, events in sorted(by_frame.items()):
        cpu = [e for e in events if e['pid'] == 1 and 1 <= e['tid'] <= 4]
        gpu = [e for e in events if e['pid'] == 2 and e['tid'] == 10]
        frame_times = [e['dur'] / 1000 for e in events if e['name'] == 'Frame interval']

        def details(e):
            row = {'name': e['name'], 'ms': e['dur'] / 1000, **e['args']}
            if e['pid'] == 1 and 'cpu_origin_ns' in meta:
                row['begin_ns'] = meta['cpu_origin_ns'] + round(e['ts'] * 1000)
                row['end_ns'] = meta['cpu_origin_ns'] + round((e['ts'] + e['dur']) * 1000)
            return row

        def lane_details(lane):
            selected = [e for e in events if e['pid'] == 1 and e['tid'] == lane]
            return [details(e) for e in sorted(selected, key=lambda e: e['dur'], reverse=True)[:6]]

        rows.append({'frame': frame, 'trigger': frame == center,
                     'present_interval_ms': frame_times[0] if frame_times else None,
                     'gpu_passes': len(gpu),
                     'largest_cpu_spans': [details(e) for e in sorted(cpu, key=lambda e: e['dur'], reverse=True)[:6]],
                     'largest_cpu_operation_intervals': lane_details(6),
                     'largest_command_queue_delays': lane_details(7),
                     'largest_fence_wait_phases': lane_details(8),
                     'largest_presentation_spans': lane_details(9),
                     'largest_gpu_spans': [details(e) for e in sorted(gpu, key=lambda e: e['dur'], reverse=True)[:6]]})
    return {'metadata': meta, 'frames': rows,
            'missing_frame_intervals': [f for f in range(max(0, center-8), center+9)
                                        if not any(e['name'] == 'Frame interval' for e in by_frame[f])],
            'note': 'GPU spans include idle/stalls. CPU spans, operation intervals, queue delays and fence phases overlap; do not add them. CPU operation intervals include following-operation setup. Rows are suspects, not proof of causation. Use neighboring frames and seq IDs; clocks are not aligned. Raw CPU timestamps can select stack sample rounds, whose sampling itself adds overhead.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture', type=Path, nargs='?')
    ap.add_argument('--log', type=Path)
    ap.add_argument('--start', type=float, default=45)
    ap.add_argument('--end', type=float, default=65)
    args = ap.parse_args()
    if not args.capture and not args.log:
        ap.error('Provide a capture and/or --log.')
    result = {}
    if args.capture:
        result['capture'] = capture_report(json.loads(args.capture.read_text()))
    if args.log:
        result['frame_times'] = frame_stats(args.log.read_text(errors='replace'), args.start, args.end)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
