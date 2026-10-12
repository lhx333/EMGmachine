from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
import uuid

import h5py
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = 0
        time_values: dict[str, list[Decimal]] = {
            field: [] for field in fields if "time" in field.lower() or "timestamp" in field.lower()
        }
        parse_errors = {field: 0 for field in time_values}
        for row in reader:
            rows += 1
            for field in time_values:
                raw = row.get(field, "")
                if raw in (None, ""):
                    continue
                try:
                    time_values[field].append(Decimal(raw))
                except (InvalidOperation, ValueError, TypeError):
                    parse_errors[field] += 1
    time_checks: dict[str, Any] = {}
    for field, values in time_values.items():
        increasing = all(right > left for left, right in zip(values, values[1:]))
        nondecreasing = all(right >= left for left, right in zip(values, values[1:]))
        time_checks[field] = {
            "numeric_values": len(values),
            "parse_errors": parse_errors[field],
            "strictly_increasing": increasing if len(values) > 1 else None,
            "nondecreasing": nondecreasing if len(values) > 1 else None,
        }
    return {"type": "csv", "rows": rows, "fields": fields, "time_checks": time_checks}


def _inspect_hdf5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        attrs = {str(k): v.item() if isinstance(v, np.generic) else str(v) for k, v in handle.attrs.items()}
        datasets: dict[str, Any] = {}
        def visit(name: str, item: Any) -> None:
            if isinstance(item, h5py.Dataset):
                datasets[name] = {"shape": list(item.shape), "dtype": str(item.dtype)}
        handle.visititems(visit)
    return {"type": "hdf5", "attributes": attrs, "datasets": datasets}


class LegacyImporter:
    """Copies a legacy dataset byte-for-byte and reports its discoverable schema."""

    def import_directory(self, source: Path, output_root: Path) -> Path:
        source = source.resolve()
        if not source.is_dir():
            raise ValueError(f"legacy sample directory does not exist: {source}")
        files = sorted(path for path in source.rglob("*") if path.is_file())
        if not files:
            raise ValueError("legacy sample directory contains no files")

        source_metadata = source / "metadata.json"
        metadata: dict[str, Any] = {}
        if source_metadata.exists():
            metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
        base_id = str(metadata.get("session_id") or source.name)
        session_id = f"{base_id}_legacy_{uuid.uuid4().hex[:6]}"
        target = output_root.resolve() / session_id
        target.mkdir(parents=True, exist_ok=False)
        raw_root = target / "raw" / "legacy_source"
        report_files: dict[str, Any] = {}

        for source_file in files:
            relative = source_file.relative_to(source)
            destination = raw_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
            details: dict[str, Any] = {
                "sha256": _sha256(destination),
                "bytes": destination.stat().st_size,
            }
            suffix = destination.suffix.lower()
            try:
                if suffix == ".csv":
                    details.update(_inspect_csv(destination))
                elif suffix in {".h5", ".hdf5"}:
                    details.update(_inspect_hdf5(destination))
                else:
                    details["type"] = "opaque_file"
            except Exception as exc:
                details["inspection_error"] = str(exc)
            report_files[relative.as_posix()] = details

        time_warnings = []
        for filename, details in report_files.items():
            for field, check in details.get("time_checks", {}).items():
                if check.get("strictly_increasing") is False:
                    time_warnings.append(f"{filename}:{field} is not strictly increasing")
        if metadata.get("alignment"):
            time_warnings.append("legacy alignment metadata is retained as provenance; first-frame matching does not establish full-record synchronization")

        report = {
            "imported_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_directory": str(source),
            "source_metadata": metadata,
            "source_alignment_report": json.loads((source / "alignment_report.json").read_text(encoding="utf-8")) if (source / "alignment_report.json").exists() else None,
            "source_files": report_files,
            "time_quality": "WARN" if time_warnings else "UNKNOWN",
            "time_warnings": time_warnings,
            "interpretation": "Original files were copied without rewriting. Legacy timestamps and alignment results retain their original semantics and are not promoted to the new QPC time axis.",
        }
        (target / "legacy_import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = {
            "schema_version": "0.1",
            "session_id": session_id,
            "state": "IMPORTED_LEGACY",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_session_id": metadata.get("session_id"),
            "source_directory": str(source),
            "raw_data_path": "raw/legacy_source",
            "time_quality": report["time_quality"],
            "time_warnings": time_warnings,
            "import_report": "legacy_import_report.json",
        }
        (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return target
