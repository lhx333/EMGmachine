# Bundled NOKOV SDK runtime

The Windows binding and 64-bit runtime in `nokov/` were extracted from the
workspace's vendor wheel `nokovpy-3.0.1-py3-none-any.whl` distributed with XING
Python SDK 2.4.0.5428. Only the Windows Python binding and `nokov_sdk.dll` are
included. Runtime loading uses `ctypes`; project Python dependencies are declared
in `../../pyproject.toml`.

The vendor SDK DLL and binding are kept together because the binding resolves the
DLL relative to its own `lib/` directory. Update both from a matching vendor SDK
release.
