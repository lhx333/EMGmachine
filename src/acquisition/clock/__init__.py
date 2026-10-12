"""Monotonic host clock implementation."""

from .windows_qpc import WindowsQpcClock

__all__ = ["WindowsQpcClock"]
