from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import h5py


def validate_touchglove_runtime(model_path: Path) -> dict[str, Any]:
    """Load the bundled model during PREPARE and require DirectML as provider zero."""
    root = Path(__file__).resolve().parents[3]
    if not model_path.is_file():
        raise FileNotFoundError(f"TouchGlove Dense model not found: {model_path}")
    sys.path.insert(0, str(root / "vendor" / "touchglove" / "python"))
    from touch_glove_ptp import TouchGlove

    inferencer = TouchGlove(model=str(model_path), device="directml", auto_open=False)
    try:
        runtime = getattr(inferencer, "_ort_session", None)
        if runtime is None:
            raise RuntimeError("Could not initialize the TouchGlove Dense ONNX model")
        providers = runtime.get_providers()
        if not providers or providers[0] != "DmlExecutionProvider":
            raise RuntimeError(f"DirectML is not the active first provider: {providers}")
        return {"status": "ready", "provider_order": providers,
                "model": model_path.relative_to(root).as_posix()}
    finally:
        inferencer.close()


def run_force_inference(session_path: Path, output_path: Path,
                        model_path: Path, flush_frames: int = 8,
                        on_progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Add offline Dense outputs to the endpoint-mapped TouchGlove timeline."""
    if flush_frames < 1:
        raise ValueError("flush_frames must be positive")
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "vendor" / "touchglove" / "python"))
    from touch_glove_ptp import TouchGlove

    raw_path = session_path / "raw" / "force.h5"
    if not model_path.is_file():
        raise FileNotFoundError(f"TouchGlove Dense model not found: {model_path}")
    with h5py.File(raw_path, "r") as source:
        frames = source["frames"]
        active_channels = sorted(int(channel) for channel in set(frames["channel"][:].tolist()))
        baseline = source["baseline"]["image"][:]
        baseline_count = source["baseline"]["sample_count_by_channel"][:]
        frame_count = len(frames["channel"])
    if frame_count == 0:
        raise RuntimeError("No TouchGlove raw frames to infer")
    if len(baseline_count) != 5 or any(int(baseline_count[channel]) == 0 for channel in active_channels):
        raise RuntimeError(
            "Initial baseline interval did not receive samples for every channel present in the capture: "
            f"active_channels={active_channels}, baseline_samples={baseline_count.tolist()}"
        )

    inferencer = TouchGlove(model=str(model_path), device="directml", auto_open=False)
    try:
        # Provider selection is private in the vendored SDK; keep this check inside
        # the SDK adapter boundary so collection never silently completes on CPU.
        runtime = getattr(inferencer, "_ort_session", None)
        if runtime is None:
            raise RuntimeError("Could not initialize the TouchGlove Dense ONNX model")
        providers = runtime.get_providers()
        if not providers or providers[0] != "DmlExecutionProvider":
            raise RuntimeError(f"DirectML is not the active first provider: {providers}")
        if not callable(getattr(inferencer, "set_baseline_images", None)) or not callable(getattr(inferencer, "infer_frame", None)):
            raise RuntimeError("Vendored TouchGlove SDK does not expose offline inference helpers")
        inferencer.set_baseline_images(baseline)
        with h5py.File(raw_path, "r") as source, h5py.File(output_path, "r+") as output:
            frames = source["frames"]
            if "disp" in output:
                raise RuntimeError("Dense output already exists; refusing to overwrite")
            output.create_dataset("disp", shape=(frame_count, 32, 32, 3), dtype="float32",
                                  chunks=(min(flush_frames, frame_count), 32, 32, 3), compression="lzf")
            output.create_dataset("force_field", shape=(frame_count, 32, 32, 3), dtype="float32",
                                  chunks=(min(flush_frames, frame_count), 32, 32, 3), compression="lzf")
            for index in range(frame_count):
                disp, force_field = inferencer.infer_frame(
                    int(frames["channel"][index]), frames["image"][index]
                )
                if disp is None or force_field is None:
                    raise RuntimeError(f"TouchGlove inference failed for raw frame {index}")
                output["disp"][index] = disp
                output["force_field"][index] = force_field
                if index % flush_frames == flush_frames - 1:
                    output.flush()
                completed = index + 1
                if on_progress and (completed % max(flush_frames, 32) == 0 or completed == frame_count):
                    on_progress(completed, frame_count)
            output.attrs["inference_provider_order"] = json.dumps(providers)
            output.attrs["model_path"] = model_path.relative_to(root).as_posix()
            output.attrs["inference_status"] = "complete"
            output.flush()
        return {"status": "complete", "provider_order": providers,
                "model": model_path.relative_to(root).as_posix(),
                "raw_frame_count": frame_count,
                "active_channels": active_channels,
                "baseline_samples_by_channel": [int(value) for value in baseline_count]}
    finally:
        inferencer.close()
