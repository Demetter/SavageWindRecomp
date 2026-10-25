#!/usr/bin/env python3
"""Report throughput from cumulative frame counters, not averages of FPS bins.

Use a quiet capture for throughput, and a separate capture for GPU tracing. Old
[gtl] enqueue/wake intervals are deliberately never interpreted as GPU busy time.
"""
import argparse
import json
import re
from pathlib import Path

FPS = re.compile(r"\[vk\] t=([\d.]+) frame (\d+) ([\d.]+) fps max ([\d.]+)ms")
FRAME = re.compile(r"\[ft\] ([\d.]+) (\d+) (\d+)")
TRACE = re.compile(r"\[vktrace\] (\{.*\})")


def report(text, start=45.0, end=65.0):
    rows = []
    frame_rows = []
    targets = []
    events = []
    for line_no, line in enumerate(text.splitlines()):
        m = FPS.search(line)
        if m:
            t, frame, fps, maximum = m.groups()
            rows.append((float(t), int(frame), float(maximum), line_no))
        m = FRAME.search(line)
        if m:
            frame_rows.append((float(m[1]) / 1000, int(m[2]), 0, line_no))
            targets.append((line_no, int(m[3])))
        if "[vk] saved " in line or "[sample]" in line or "[gpuprof]" in line:
            events.append((line_no, line.strip()[:200]))
    precise = bool(frame_rows)
    readings = frame_rows if precise else rows
    chosen = [r for r in readings if start <= r[0] <= end]
    if len(chosen) < 2:
        raise ValueError("need two frame-counter readings inside the requested window")
    first, last = chosen[0], chosen[-1]
    if any(b[0] <= a[0] or b[1] <= a[1] for a, b in zip(chosen, chosen[1:])):
        raise ValueError("counters restarted or are not increasing; use one launch per log")
    complete = readings[0][0] <= start and readings[-1][0] >= end
    contiguous = not precise or all(b[1] == a[1] + 1 for a, b in zip(chosen, chosen[1:]))
    seconds = last[0] - first[0]
    frames = last[1] - first[1]
    result = {
        "start_s": first[0], "end_s": last[0], "frames": frames,
        "seconds": seconds, "fps": frames / seconds,
        "mean_frame_ms": seconds * 1000 / frames,
        "max_frame_ms": max((b[0] - a[0]) * 1000 for a, b in zip(chosen, chosen[1:])) if precise else max(r[2] for r in chosen[1:]),
        "capture_events": [e for n, e in events if first[3] < n <= last[3]],
        "frame_rate_targets": sorted({v for n, v in targets if first[3] <= n <= last[3]}),
        "requested_window_covered": complete,
        "contiguous_frame_counters": contiguous,
        "notes": ["Per-present timestamps, including stutters."] if precise else ["Wall-clock readings in legacy logs are rounded to 0.1 s."]
    }
    if result["capture_events"]:
        result["notes"].append("Capture work occurred in this window; do not use it as a clean FPS baseline.")
    frame_trace = "[frame-trace] rolling capture:" in text
    draw_probe = any(tag in text for tag in ("[cube-verify]", "[small-pass]", "[cube-ab]", "[bindings-verify]", "[texture-bindings]", "[uniform-verify]", "[ubo-audit]", "[ww-probe]", "[depth-verify]", "[geometry-scan]", "[ww-parallel]", "[texture-life]", "[compact-records]", "[geometry-flush]", "[geometry-guard]", "[compute-opt-audit]", "[feedback-probe]", "[pipeline-exec]", "[pipeline-stat]", "[hardware-profile]"))
    draw_probe = draw_probe or "[stage-reuse-audit]" in text
    draw_probe = draw_probe or "[producer-material-audit]" in text
    draw_probe = draw_probe or "[texture-hash-audit]" in text
    draw_probe = draw_probe or "[compute-path-audit]" in text
    draw_probe = draw_probe or "[range-query-audit]" in text
    draw_probe = draw_probe or "[draw-batch-probe]" in text
    draw_probe = draw_probe or "[native-multi-draw]" in text
    draw_probe = draw_probe or "[native-multi-draw-ab]" in text
    trace_failure = any(tag in text for tag in ("[trace-log] flush timeout", "[trace-log] producer backpressure"))
    runtime_failure = bool(re.search(r"(?m)^(?:FATAL:|\*\*\* exception )", text))
    mixed_modes = "[producer-material-ab]" in text or "[survey]" in text
    result["valid_uncapped_capture"] = not result["capture_events"] and result["frame_rate_targets"] == [0] and not frame_trace and not draw_probe and not trace_failure and not runtime_failure and not mixed_modes and complete and contiguous
    if mixed_modes:
        result["notes"].append("Mixed producer-material modes; compare individual controlled blocks, not whole-run throughput.")
    if not complete:
        result["notes"].append("Frame readings do not cover the requested window; a partial or stalled run is not a valid comparison.")
    if not contiguous:
        result["notes"].append("Per-present counters have gaps; stutters may be missing from this capture.")
    if runtime_failure:
        result["notes"].append("A fatal error or unhandled exception occurred; run rejected.")
    if trace_failure:
        result["notes"].append("Telemetry I/O failed or blocked a producer; incomplete/perturbed capture.")
    if frame_trace:
        result["notes"].append("Detailed frame tracing was enabled; diagnostic only, not a clean throughput baseline.")
    if draw_probe:
        result["notes"].append("Draw probes/GPU differential checks were enabled; diagnostic only, not a clean throughput baseline.")
    if any(result["frame_rate_targets"]):
        result["notes"].append("A frame cap/menu occurred in this window: invalid uncapped comparison.")
    if "[gtl]" in text:
        result["notes"].append("Legacy [gtl] is CPU enqueue/completion observation, not GPU busy time.")
    return result


def trace_report(text):
    rows = [json.loads(m.group(1)) for m in TRACE.finditer(text)]
    if not rows:
        return None
    # These are timestamp spans, which include stalls and bubbles. They are not
    # hardware occupancy or active cycles, and are not comparable to CPU clocks.
    spans = [r["gpu_ms"] for r in rows if "gpu_ms" in r]
    delays = [(r["driver_ns"] - r["enqueue_ns"]) / 1e6 for r in rows
              if r.get("driver_ns", 0) >= r.get("enqueue_ns", 1)]
    return {"submissions": len(rows),
            "mean_gpu_span_ms": sum(spans) / len(spans) if spans else None,
            "mean_recorder_delay_ms": sum(delays) / len(delays) if delays else None,
            "note": "GPU spans include uploads and stalls; they do not measure GPU utilization."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--start", type=float, default=45)
    ap.add_argument("--end", type=float, default=65)
    args = ap.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace")
    result = report(text, args.start, args.end)
    trace = trace_report(text)
    if trace:
        result["trace_whole_capture"] = trace
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
