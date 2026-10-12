"""
touch_glove_ptp — Touch Glove SDK (with PyO3 native bindings)

High performance tactile glove data acquisition and physical inference.
All core logic is implemented in Rust.
"""
from __future__ import annotations

import os
import sys
import glob
import ctypes

_this_dir = os.path.dirname(os.path.abspath(__file__))

_RUNTIME_ORDER = {
    "linux": ["cuda", "npu", "cpu"],
    "win32": ["directml", "cuda", "npu", "cpu"],
    "darwin": ["coreml", "cpu"],
}

_RUNTIME_LIB_NAMES = {
    "linux": ["libonnxruntime.so.1.23.2", "libonnxruntime.so"],
    "win32": ["onnxruntime.dll"],
    "darwin": ["libonnxruntime.1.23.2.dylib", "libonnxruntime.dylib"],
}

_PROVIDER_PATTERNS = {
    "linux": ["libonnxruntime_providers*.so"],
    "win32": ["onnxruntime_providers*.dll"],
    "darwin": ["libonnxruntime_providers*.dylib"],
}

_DLOPEN_MODE = getattr(ctypes, "RTLD_GLOBAL", 0)
if hasattr(os, "RTLD_NOW"):
    _DLOPEN_MODE |= os.RTLD_NOW


def _platform_key() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("win"):
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    return sys.platform


def _try_load(path: str) -> bool:
    try:
        ctypes.CDLL(path, mode=_DLOPEN_MODE)
        return True
    except Exception:
        return False


def _add_dll_directory(path: str) -> None:
    if sys.platform.startswith("win") and hasattr(os, "add_dll_directory") and os.path.isdir(path):
        try:
            os.add_dll_directory(path)
        except OSError:
            pass


def _runtime_names() -> list[str]:
    requested = os.environ.get("TOUCH_GLOVE_RUNTIME", "auto").strip().lower()
    platform_key = _platform_key()
    if requested in ("", "auto"):
        return _RUNTIME_ORDER.get(platform_key, ["cpu"])
    aliases = {
        "npu": "npu",
        "openvino": "npu",
        "gpu": "cuda",
        "mac": "coreml",
    }
    return [aliases.get(requested, requested)]


def _runtime_dirs() -> list[str]:
    explicit = os.environ.get("TOUCH_GLOVE_RUNTIME_DIR")
    dirs: list[str] = []
    if explicit:
        dirs.extend([p for p in explicit.split(os.pathsep) if p])
    runtimes_root = os.path.join(_this_dir, "runtimes")
    for runtime in _runtime_names():
        dirs.append(os.path.join(runtimes_root, runtime))
    # Backward compatibility with the old Ubuntu SDK layout where ORT files were
    # placed flat next to _rust.* inside api/ or touch_glove_ptp/.
    dirs.append(_this_dir)
    seen = set()
    unique = []
    for d in dirs:
        if d not in seen:
            unique.append(d)
            seen.add(d)
    return unique


def _find_ort_library(runtime_dir: str) -> str | None:
    for libname in _RUNTIME_LIB_NAMES.get(_platform_key(), []):
        candidate = os.path.join(runtime_dir, libname)
        if os.path.exists(candidate):
            return candidate
    return None


def _preload_cuda_libs() -> None:
    if not sys.platform.startswith("linux"):
        return
    cuda_roots = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/targets/aarch64-linux/lib",
    ]
    cuda_roots.extend(glob.glob("/usr/local/cuda-*/lib64"))
    cuda_roots.extend(glob.glob("/usr/local/cuda-*/targets/*/lib"))

    roots = []

    def add_site_packages_nvidia_libs(site_packages: str):
        roots.extend(glob.glob(os.path.join(site_packages, "nvidia", "*", "lib")))

    for base in [sys.prefix, sys.base_prefix, os.path.expanduser("~/.local")]:
        for site_packages in glob.glob(os.path.join(base, "lib", "python*", "site-packages")):
            add_site_packages_nvidia_libs(site_packages)
        roots.append(os.path.join(base, "lib"))
    for p in sys.path:
        if p.endswith("site-packages"):
            add_site_packages_nvidia_libs(p)
    for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if p:
            roots.append(p)

    roots = sorted(set(cuda_roots + roots))
    cuda_patterns = [
        "libnvJitLink.so.12",
        "libcudart.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libnvrtc.so.12",
        "libnvrtc-builtins.so.*",
        "libcurand.so.10",
        "libcufft.so.11",
    ]
    cudnn_patterns = [
        "libcudnn.so.9",
        "libcudnn_graph.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
    ]
    for pattern in cuda_patterns + cudnn_patterns:
        for root in roots:
            loaded = False
            for path in sorted(glob.glob(os.path.join(root, pattern))):
                if _try_load(path):
                    loaded = True
                    break
            if loaded:
                break


def _configure_runtime() -> None:
    platform_key = _platform_key()
    if any(name in _runtime_names() for name in ("cuda", "auto")):
        _preload_cuda_libs()

    provider_patterns = _PROVIDER_PATTERNS.get(platform_key, [])
    selected_ort = None
    selected_dir = None
    for runtime_dir in _runtime_dirs():
        if not os.path.isdir(runtime_dir):
            continue
        _add_dll_directory(runtime_dir)
        ort_lib = _find_ort_library(runtime_dir)
        if not ort_lib:
            continue
        selected_ort = ort_lib
        selected_dir = runtime_dir
        break

    if selected_ort and not os.environ.get("ORT_DYLIB_PATH"):
        os.environ["ORT_DYLIB_PATH"] = selected_ort

    if selected_dir:
        for pattern in provider_patterns:
            for path in sorted(glob.glob(os.path.join(selected_dir, pattern))):
                _try_load(path)
        _try_load(selected_ort)


# Pre-load ORT/provider libraries before importing the PyO3 module.
_configure_runtime()

try:
    from ._rust import (  # type: ignore[import]
        TouchGlove,
        TouchGlovePTP,
        Frame,
        FrameBatch,
        list_ports,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        IMAGE_CHANNELS,
        IMAGE_BYTES,
    )
except ImportError:
    from ._fallback import (
        TouchGlove,
        TouchGlovePTP,
        Frame,
        FrameBatch,
        list_ports,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        IMAGE_CHANNELS,
        IMAGE_BYTES,
    )

__all__ = [
    "TouchGlove",
    "TouchGlovePTP",
    "Frame",
    "FrameBatch",
    "list_ports",
    "IMAGE_WIDTH",
    "IMAGE_HEIGHT",
    "IMAGE_CHANNELS",
    "IMAGE_BYTES",
]

__version__ = "1.1.0"
