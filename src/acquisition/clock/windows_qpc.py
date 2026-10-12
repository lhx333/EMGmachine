from __future__ import annotations

import ctypes
import sys
import time


class WindowsQpcClock:
    """QPC-backed clock; non-Windows fallback exists only for development."""

    def __init__(self) -> None:
        self._kernel32 = None
        self._frequency = 1_000_000_000
        if sys.platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            frequency = ctypes.c_longlong()
            if not kernel32.QueryPerformanceFrequency(ctypes.byref(frequency)):
                raise ctypes.WinError(ctypes.get_last_error())
            self._kernel32 = kernel32
            self._frequency = int(frequency.value)

    @property
    def frequency_hz(self) -> int:
        return self._frequency

    @property
    def clock_id(self) -> str:
        return "windows_qpc" if self._kernel32 else "development_perf_counter"

    def counter(self) -> int:
        if self._kernel32:
            value = ctypes.c_longlong()
            if not self._kernel32.QueryPerformanceCounter(ctypes.byref(value)):
                raise ctypes.WinError(ctypes.get_last_error())
            return int(value.value)
        return time.perf_counter_ns()

    def to_ns(self, counter: int) -> int:
        return (int(counter) * 1_000_000_000) // self._frequency

    def now_ns(self) -> int:
        return self.to_ns(self.counter())
