from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acquisition.clock import WindowsQpcClock  # noqa: E402
from acquisition.devices.nokov import NokovAdapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Connect to XINGYING and count live NOKOV callback frames")
    parser.add_argument("--server-ip", default="10.1.1.198", help="XINGYING/NOKOV streaming server IPv4")
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    events = []
    adapter = NokovAdapter(lambda _record: None, lambda name, details: events.append({"event": name, **details}))
    try:
        info = adapter.prepare(args.server_ip)
        adapter.arm(adapter.clock.now_ns())
        time.sleep(args.seconds)
        print(json.dumps({"status": "connected", "device": info,
                          "frame_count": adapter.frame_count,
                          "quality_counts": adapter.quality_counts,
                          "events": events}, ensure_ascii=False, indent=2))
        return 0 if adapter.frame_count else 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "server_ip": args.server_ip,
                          "error": f"{type(exc).__name__}: {exc}", "events": events},
                         ensure_ascii=False, indent=2))
        return 1
    finally:
        adapter.stop()


if __name__ == "__main__":
    raise SystemExit(main())
