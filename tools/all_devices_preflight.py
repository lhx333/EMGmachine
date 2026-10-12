from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acquisition.devices.emg import DEFAULT_EMG_MAC, detect_emg_ports  # noqa: E402
from acquisition.devices.touchglove import detect_touchglove_ports  # noqa: E402
from acquisition.session.session import Session  # noqa: E402


def _select_port(requested: str | None, candidates: list[tuple[str, str]], name: str) -> str:
    if requested:
        return requested
    if len(candidates) != 1:
        raise RuntimeError(f"Select {name} port explicitly; detected: {candidates}")
    return candidates[0][0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and safely close all five acquisition streams; does not record")
    parser.add_argument("--emg-port")
    parser.add_argument("--emg-mac", default=DEFAULT_EMG_MAC,
                        help="target ResearchKit band MAC address")
    parser.add_argument("--touchglove-port")
    parser.add_argument("--nokov-server", default="10.1.1.198")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "preflight")
    args = parser.parse_args()

    emg_port = _select_port(args.emg_port, detect_emg_ports(), "ResearchKit")
    touchglove_port = _select_port(args.touchglove_port, detect_touchglove_ports(), "TouchGlove")
    session = Session(args.output_root, experiment={"purpose": "all_devices_preflight"})
    try:
        info = session.prepare_emg(emg_port, nokov_server_ip=args.nokov_server,
                                   touchglove_port=touchglove_port,
                                   emg_target_mac=args.emg_mac)
        session.stop()
        print(json.dumps({"status": "prepared_and_closed", "emg_port": emg_port,
                          "touchglove_port": touchglove_port,
                          "prepared_streams": sorted(info), "session": str(session.path),
                          "artifacts": "empty preflight session directory; no sensor recording was armed"},
                         ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        if session.state == "PREPARED":
            session.stop()
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                          "session": str(session.path)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
