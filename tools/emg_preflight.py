from __future__ import annotations

import argparse

from acquisition.devices.emg import EmgAdapter, detect_emg_ports


def main() -> int:
    parser = argparse.ArgumentParser(description="Connect to the ResearchKit EMG device without starting capture")
    parser.add_argument("--port", help="Override the auto-detected ResearchKit USB COM port")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    ports = detect_emg_ports()
    port = args.port or (ports[0][0] if len(ports) == 1 else None)
    if not port:
        parser.error(f"Expected exactly one detected EMG adapter or --port. Found: {ports}")

    adapter = EmgAdapter(lambda _record: None, lambda name, detail: print(name, detail, flush=True))
    try:
        info = adapter.prepare(port, timeout_s=args.timeout)
        print("EMG PREPARE succeeded; capture was not armed:", info, flush=True)
        return 0
    finally:
        adapter.stop()


if __name__ == "__main__":
    raise SystemExit(main())
