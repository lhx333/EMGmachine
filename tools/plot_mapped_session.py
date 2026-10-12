from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot aligned EMG and IMU timelines from a session")
    parser.add_argument("session", type=Path, help="Session directory containing manifest.json")
    parser.add_argument("--output", type=Path, help="PNG output path (default: session/derived/mapped_signals.png)")
    args = parser.parse_args()

    session = args.session.resolve()
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    streams = []
    for name, key, count in (("EMG", "emg_timeline", 9), ("IMU", "imu_timeline", 6)):
        timeline = manifest.get(key, {})
        if timeline.get("status") != "complete":
            raise RuntimeError(f"{name} timeline is not complete: {timeline}")
        with h5py.File(session / timeline["path"], "r") as aligned:
            times = aligned["time_ns"][:].astype(np.float64) / 1e9
            values = aligned["data"][:]
        if values.ndim != 2 or values.shape[1] != count or len(values) != len(times):
            raise RuntimeError(f"Unexpected {name} aligned array shape: {values.shape}")
        # The mapping can yield non-monotonic sample times at packet boundaries.
        # Sort only for drawing and render an envelope so this doesn't imply a sample-order path.
        order = np.argsort(times, kind="stable")
        streams.append((name, times[order], values[order]))

    x_min = min(float(times.min()) for _, times, _ in streams)
    x_max = max(float(times.max()) for _, times, _ in streams)
    width, left, right = 2200, 130, 35
    panel_height, top, bottom = 112, 120, 80
    height = top + panel_height * sum(values.shape[1] for _, _, values in streams) + bottom
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "EMG + IMU mapped signals", (left, 48), font, 1.0, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(image, "Current session endpoint mapping | x: seconds from planned QPC start | y: raw SDK values",
                (left, 82), font, 0.58, (70, 70, 70), 1, cv2.LINE_AA)

    plot_width = width - left - right
    total_panels = sum(values.shape[1] for _, _, values in streams)
    panel_index = 0
    for name, times, values in streams:
        color = (176, 92, 29) if name == "EMG" else (38, 135, 66)  # BGR
        for channel in range(values.shape[1]):
            y_top = top + panel_index * panel_height
            y_bottom = y_top + panel_height - 18
            plot_top, plot_bottom = y_top + 15, y_bottom - 17
            series = values[:, channel].astype(np.float64)
            lo, hi = float(series.min()), float(series.max())
            if hi <= lo:
                hi, lo = hi + 1.0, lo - 1.0
            padding = (hi - lo) * 0.05
            lo -= padding
            hi += padding

            def x_pixel(t: float) -> int:
                return left + int(round((t - x_min) / (x_max - x_min) * plot_width))

            def y_pixel(v: float) -> int:
                return plot_bottom - int(round((v - lo) / (hi - lo) * (plot_bottom - plot_top)))

            cv2.line(image, (left, plot_bottom), (width - right, plot_bottom), (224, 224, 224), 1, cv2.LINE_AA)
            cv2.line(image, (left, plot_top), (width - right, plot_top), (238, 238, 238), 1, cv2.LINE_AA)
            if lo <= 0 <= hi:
                zero_y = y_pixel(0)
                cv2.line(image, (left, zero_y), (width - right, zero_y), (235, 235, 235), 1, cv2.LINE_AA)

            # Per-pixel time-bin envelope preserves peaks without connecting samples
            # across packet boundaries that map to overlapping or reversed times.
            bin_ids = np.clip(((times - x_min) / (x_max - x_min) * (plot_width - 1)).astype(np.int64),
                              0, plot_width - 1)
            min_bin = np.full(plot_width, np.inf)
            max_bin = np.full(plot_width, -np.inf)
            np.minimum.at(min_bin, bin_ids, series)
            np.maximum.at(max_bin, bin_ids, series)
            for col in np.flatnonzero(np.isfinite(min_bin)):
                px = left + int(col)
                cv2.line(image, (px, y_pixel(float(max_bin[col]))),
                         (px, y_pixel(float(min_bin[col]))), color, 1, cv2.LINE_AA)

            if panel_index == 0 or channel == 0:
                label = f"{name} channel_{channel}"
            else:
                label = f"channel_{channel}"
            cv2.putText(image, label, (12, y_top + 33), font, 0.48, color, 1, cv2.LINE_AA)
            cv2.putText(image, f"{hi-padding:.0f} / {lo+padding:.0f}", (12, y_top + 54), font, 0.36,
                        (80, 80, 80), 1, cv2.LINE_AA)
            panel_index += 1

    axis_y = height - 34
    for sec in range(int(x_min // 10 * 10), int(x_max) + 1, 10):
        if sec < x_min or sec > x_max:
            continue
        px = left + int(round((sec - x_min) / (x_max - x_min) * plot_width))
        cv2.line(image, (px, top - 7), (px, height - bottom + 5), (243, 243, 243), 1)
        cv2.putText(image, str(sec), (px - 12, axis_y), font, 0.38, (70, 70, 70), 1, cv2.LINE_AA)

    output = args.output.resolve() if args.output else session / "derived" / "mapped_signals.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Could not write plot to {output}")
    print(json.dumps({"output": str(output), "size_px": [width, height], "x_range_s": [x_min, x_max],
                      "panels": total_panels, "mapping": {
                          "emg": manifest["emg_timeline"]["algorithm"],
                          "imu": manifest["imu_timeline"]["algorithm"],
                      }}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
