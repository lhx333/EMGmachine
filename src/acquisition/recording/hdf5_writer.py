from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class Hdf5Writer:
    """Single-writer append-only recorder for host-observed stream events."""

    def __init__(self, raw_dir: Path, queue_size: int = 8192) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue(queue_size)
        self._thread = threading.Thread(target=self._run, name="hdf5-writer", daemon=True)
        self._errors: list[str] = []
        self._error_counts: dict[str, int] = {}
        self._counts: dict[str, int] = {}
        self._closed = False
        self._thread.start()

    def append(self, stream: str, record: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        try:
            self._queue.put_nowait((stream, record))
        except queue.Full as exc:
            raise RuntimeError("record queue overflow") from exc

    def _run(self) -> None:
        files: dict[str, h5py.File] = {}
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        return
                    stream, record = item
                    handle = files.get(stream)
                    if handle is None:
                        handle = h5py.File(self.raw_dir / f"{stream}.h5", "a", libver="latest")
                        handle.attrs["schema_version"] = "0.2"
                        handle.attrs["stream"] = stream
                        handle.attrs["record_encoding"] = "utf-8 JSON per observation"
                        handle.create_dataset("record_json", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8"), chunks=True)
                        handle.create_dataset("host_receive_mono_ns", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
                        files[stream] = handle
                    encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                    for name, value in (("record_json", encoded), ("host_receive_mono_ns", int(record["host_receive_mono_ns"]))):
                        ds = handle[name]
                        ds.resize((len(ds) + 1,))
                        ds[-1] = value
                    self._counts[stream] = self._counts.get(stream, 0) + 1
                    if self._counts[stream] % 256 == 0:
                        handle.flush()
                except Exception as exc:
                    self._record_error(exc)
                finally:
                    self._queue.task_done()
        finally:
            for handle in files.values():
                try:
                    handle.flush()
                    handle.close()
                except Exception as exc:
                    self._record_error(exc)

    def _record_error(self, error: Exception) -> None:
        message = str(error)
        self._error_counts[message] = self._error_counts.get(message, 0) + 1
        if message not in self._errors:
            self._errors.append(message)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        if self._errors:
            summary = "; ".join(
                f"{message} (×{self._error_counts[message]})"
                if self._error_counts[message] > 1 else message
                for message in self._errors
            )
            raise RuntimeError("HDF5 writer errors: " + summary)

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)
