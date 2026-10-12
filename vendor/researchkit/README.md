# Bundled ResearchKit SDK

`ResearchKit_SDK/` is copied from the Windows CPython 3.10 ResearchKit SDK used by
the legacy acquisition host. Its Python package and `.cp310-win_amd64.pyd`
extensions are loaded directly by `src/acquisition/devices/emg.py`; no path
outside this project is searched at runtime.

The Python ABI is intentional: `full_system` supports CPython 3.10 on 64-bit
Windows because these compiled extensions are built for that interpreter.
Replace the package as a whole when updating the vendor SDK. The upstream SDK
license and redistribution terms should be retained with any external release.
