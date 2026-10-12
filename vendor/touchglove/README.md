# Bundled TouchGlove Windows SDK

`python/touch_glove_ptp/` contains the SDK Python entry point, ctypes fallback,
type stub, and Windows `touch_glove.dll`. The package's PyO3 extension is not
needed by this host: acquisition explicitly requires the DLL backend and rejects
the Python serial fallback because that fallback fabricates `timestamp_us` from
the host wall clock.

`model/dense_3ch.onnx` is the Dense inference model. Raw frames are written to
the session before inference; post-capture inference requires the DirectML
provider. The Python runtime dependencies (`numpy`, `h5py`, and
`onnxruntime-directml`) are declared in the project `pyproject.toml`.
