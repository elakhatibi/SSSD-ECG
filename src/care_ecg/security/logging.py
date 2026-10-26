from __future__ import annotations

import fcntl
import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import AttackResult
from .defenses import hash_waveform

_ZERO_HASH = "0" * 64


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {item.name: _json_ready(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            _json_ready(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Security log values must be finite and JSON-serializable") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LogReceipt:
    sequence: int
    event_id: str
    record_sha256: str
    waveform_path: str | None = None
    waveform_sha256: str | None = None
    npz_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LogVerification:
    valid: bool
    records: int
    waveform_artifacts: int
    errors: tuple[str, ...]
    complete: bool | None = None
    expected_records: int | None = None
    terminal_record_sha256: str | None = None
    manifest_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AppendOnlySecurityLogger:
    """Hash-chained JSONL events with separately hashed raw-mV NPZ waveforms."""

    def __init__(
        self,
        path: str | Path,
        waveform_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
    ):
        self.path = Path(path)
        self.waveform_dir = (
            Path(waveform_dir)
            if waveform_dir is not None
            else self.path.parent / f"{self.path.stem}_waveforms"
        )
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else Path(f"{self.path}.manifest.json")
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.waveform_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _last_record(handle) -> tuple[int, str]:
        handle.seek(0)
        sequence = 0
        previous = _ZERO_HASH
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line in append-only log at line {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid existing JSONL at line {line_number}") from exc
            if record.get("sequence") != sequence + 1:
                raise ValueError("Existing JSONL sequence is not contiguous")
            if record.get("previous_record_sha256") != previous:
                raise ValueError("Existing JSONL hash chain is broken")
            supplied = record.pop("record_sha256", None)
            observed = hashlib.sha256(_canonical_json(record)).hexdigest()
            if supplied != observed:
                raise ValueError("Existing JSONL record hash is invalid")
            sequence += 1
            previous = observed
        return sequence, previous

    def _store_waveform(
        self,
        waveform_mv: np.ndarray,
        event_id: str,
        sequence: int,
        attributes: dict[str, Any],
    ) -> dict[str, Any]:
        waveform = np.asarray(waveform_mv, dtype=np.float32)
        if waveform.ndim != 2 or not np.isfinite(waveform).all():
            raise ValueError("Logged waveform must be finite with shape [leads, samples]")
        destination = self.waveform_dir / f"{sequence:08d}_{event_id}.npz"
        temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
        with temporary.open("xb") as handle:
            np.savez_compressed(
                handle,
                waveform_mv=waveform,
                units=np.asarray("mV"),
                attributes_json=np.asarray(_canonical_json(attributes).decode("utf-8")),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return {
            "path": os.path.relpath(destination, self.path.parent),
            "waveform_sha256": hash_waveform(waveform),
            "npz_sha256": _file_sha256(destination),
            "shape": list(waveform.shape),
            "dtype": str(waveform.dtype),
            "units": "mV",
        }

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        waveform_mv: np.ndarray | None = None,
    ) -> LogReceipt:
        if not event_type.strip():
            raise ValueError("event_type must not be empty")
        clean_payload = _json_ready(payload)
        _canonical_json(clean_payload)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                sequence, previous_hash = self._last_record(handle)
                sequence += 1
                event_id = uuid.uuid4().hex
                waveform = (
                    None
                    if waveform_mv is None
                    else self._store_waveform(
                        waveform_mv,
                        event_id,
                        sequence,
                        {"event_type": event_type, "sequence": sequence},
                    )
                )
                record = {
                    "schema_version": "care-ecg-security-log.v1",
                    "sequence": sequence,
                    "event_id": event_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "event_type": event_type,
                    "payload": clean_payload,
                    "waveform": waveform,
                    "previous_record_sha256": previous_hash,
                }
                record_hash = hashlib.sha256(_canonical_json(record)).hexdigest()
                record["record_sha256"] = record_hash
                handle.seek(0, os.SEEK_END)
                handle.write(_canonical_json(record).decode("utf-8") + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return LogReceipt(
            sequence=sequence,
            event_id=event_id,
            record_sha256=record_hash,
            waveform_path=None if waveform is None else waveform["path"],
            waveform_sha256=None if waveform is None else waveform["waveform_sha256"],
            npz_sha256=None if waveform is None else waveform["npz_sha256"],
        )

    def append_attack_result(
        self,
        result: AttackResult,
        *,
        patient_id: str,
        record_id: str,
        target_id: str,
        experiment_id: str,
        context: Mapping[str, Any] | None = None,
    ) -> list[LogReceipt]:
        common = {
            "experiment_id": experiment_id,
            "patient_id": patient_id,
            "record_id": record_id,
            "target_id": target_id,
            "target_label": result.target_label,
            "family": result.family,
            "experiment_context": {} if context is None else dict(context),
        }
        receipts: list[LogReceipt] = []
        for attempt in result.attempts:
            payload = {
                **common,
                "query_index": attempt.query_index,
                "objective": attempt.objective,
                "target_probability": attempt.target_probability,
                "diagnosis": attempt.diagnosis,
                "released": attempt.released,
                "target_claim_present": attempt.target_claim_present,
                "admissible": attempt.admissible,
                "success": attempt.success,
                "metadata": attempt.metadata,
            }
            receipts.append(
                self.append("attack_attempt", payload, waveform_mv=attempt.waveform)
            )
        receipts.append(
            self.append(
                "attack_result",
                {
                    **common,
                    "success": result.success,
                    "queries_used": result.queries_used,
                    "stopped_reason": result.stopped_reason,
                    "attempt_event_ids": [item.event_id for item in receipts],
                    "metadata": result.metadata,
                },
            )
        )
        return receipts

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def write_terminal_manifest(self, path: str | Path | None = None) -> dict[str, Any]:
        """Seal the current log length and terminal hash in an independently stored manifest."""

        destination = self.manifest_path if path is None else Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Security log does not exist: {self.path}")
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                records, terminal_hash = self._last_record(handle)
                if records < 1:
                    raise ValueError("Cannot seal an empty security log")
                manifest = {
                    "schema_version": "care-ecg-security-manifest.v1",
                    "log_file": self.path.name,
                    "expected_records": records,
                    "terminal_record_sha256": terminal_hash,
                    "log_sha256": _file_sha256(self.path),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                manifest["manifest_sha256"] = hashlib.sha256(
                    _canonical_json(manifest)
                ).hexdigest()
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(
                    f"{destination.suffix}.{uuid.uuid4().hex}.tmp"
                )
                with temporary.open("xb") as output:
                    output.write(_canonical_json(manifest) + b"\n")
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, destination)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return manifest

    seal = write_terminal_manifest

    def _resolve_waveform_path(self, relative: str) -> Path:
        resolved = (self.path.parent / relative).resolve()
        try:
            resolved.relative_to(self.waveform_dir.resolve())
        except ValueError as exc:
            raise ValueError("Waveform artifact escapes the configured directory") from exc
        return resolved

    def load_waveform(self, relative: str) -> np.ndarray:
        path = self._resolve_waveform_path(relative)
        with np.load(path, allow_pickle=False) as artifact:
            if set(artifact.files) != {"waveform_mv", "units", "attributes_json"}:
                raise ValueError("Waveform NPZ has an unexpected schema")
            waveform = np.asarray(artifact["waveform_mv"], dtype=np.float32)
            if str(artifact["units"].item()) != "mV":
                raise ValueError("Waveform NPZ units are not raw mV")
            if waveform.ndim != 2 or not np.isfinite(waveform).all():
                raise ValueError("Waveform NPZ must contain a finite [leads, samples] array")
        return waveform

    @staticmethod
    def _verify_attack_transactions(
        records: list[dict[str, Any]], errors: list[str]
    ) -> None:
        event_records: dict[str, dict[str, Any]] = {}
        event_positions: dict[str, int] = {}
        attempt_ids: set[str] = set()
        referenced_attempts: set[str] = set()
        for position, record in enumerate(records, start=1):
            event_id = record.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                errors.append(f"record {position}: missing event_id")
                continue
            if event_id in event_records:
                errors.append(f"record {position}: duplicate event_id")
            event_records[event_id] = record
            event_positions[event_id] = position
            if record.get("event_type") == "attack_attempt":
                attempt_ids.add(event_id)

        identity_fields = (
            "experiment_id",
            "patient_id",
            "record_id",
            "target_id",
            "target_label",
            "family",
        )
        for position, record in enumerate(records, start=1):
            if record.get("event_type") != "attack_result":
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                errors.append(f"record {position}: attack_result payload is not an object")
                continue
            references = payload.get("attempt_event_ids")
            if not isinstance(references, list) or any(
                not isinstance(item, str) for item in references
            ):
                errors.append(f"record {position}: invalid attempt_event_ids")
                continue
            if len(references) != len(set(references)):
                errors.append(f"record {position}: duplicate attempt_event_ids")
            if payload.get("queries_used") != len(references):
                errors.append(f"record {position}: queries_used does not match attempts")
            attempts: list[dict[str, Any]] = []
            for reference in references:
                referenced = event_records.get(reference)
                if referenced is None or referenced.get("event_type") != "attack_attempt":
                    errors.append(
                        f"record {position}: missing attack_attempt reference {reference}"
                    )
                    continue
                if event_positions[reference] >= position:
                    errors.append(
                        f"record {position}: attack_result references a non-prior attempt"
                    )
                attempt_payload = referenced.get("payload")
                if not isinstance(attempt_payload, dict):
                    errors.append(
                        f"record {position}: referenced attack_attempt payload is invalid"
                    )
                    continue
                if any(
                    attempt_payload.get(field) != payload.get(field)
                    for field in identity_fields
                ):
                    errors.append(
                        f"record {position}: attack_result identity differs from its attempts"
                    )
                attempts.append(attempt_payload)
                referenced_attempts.add(reference)
            indices = [attempt.get("query_index") for attempt in attempts]
            if indices != list(range(1, len(attempts) + 1)):
                errors.append(f"record {position}: attack query indices are not contiguous")
            successes = [bool(attempt.get("success")) for attempt in attempts]
            aggregate_success = bool(payload.get("success"))
            if aggregate_success != any(successes):
                errors.append(f"record {position}: aggregate attack success is inconsistent")
            if any(successes) and successes != [False] * (len(successes) - 1) + [True]:
                errors.append(f"record {position}: attempts continue after first success")
        dangling = sorted(attempt_ids - referenced_attempts)
        if dangling:
            errors.append(
                f"uncommitted attack attempts are not referenced by a result: {len(dangling)}"
            )

    @staticmethod
    def _read_manifest(path: Path, errors: list[str]) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"terminal manifest unavailable: {exc}")
            return None
        if not isinstance(value, dict):
            errors.append("terminal manifest is not an object")
            return None
        supplied = value.get("manifest_sha256")
        unsigned = dict(value)
        unsigned.pop("manifest_sha256", None)
        try:
            observed = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        except ValueError as exc:
            errors.append(f"terminal manifest is invalid: {exc}")
            return None
        if supplied != observed:
            errors.append("terminal manifest hash mismatch")
        if value.get("schema_version") != "care-ecg-security-manifest.v1":
            errors.append("terminal manifest schema is unsupported")
        return value

    def verify(
        self,
        *,
        manifest_path: str | Path | None = None,
        require_manifest: bool = False,
        expected_records: int | None = None,
        expected_terminal_sha256: str | None = None,
    ) -> LogVerification:
        errors: list[str] = []
        records: list[dict[str, Any]] = []
        if not self.path.exists():
            errors.append(f"log unavailable: {self.path} does not exist")
        else:
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            errors.append(f"record {line_number}: blank JSONL line")
                            continue
                        try:
                            value = json.loads(line)
                        except json.JSONDecodeError as exc:
                            errors.append(f"record {line_number}: invalid JSON: {exc.msg}")
                            continue
                        if not isinstance(value, dict):
                            errors.append(f"record {line_number}: JSON value is not an object")
                            continue
                        records.append(value)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (OSError, UnicodeError) as exc:
                errors.append(f"log unavailable: {exc}")
        if self.path.exists() and not records:
            errors.append("security log is empty")
        previous = _ZERO_HASH
        waveform_count = 0
        for expected_sequence, original in enumerate(records, start=1):
            record = dict(original)
            supplied = record.pop("record_sha256", None)
            try:
                observed = hashlib.sha256(_canonical_json(record)).hexdigest()
            except ValueError as exc:
                errors.append(f"record {expected_sequence}: invalid payload: {exc}")
                continue
            if record.get("sequence") != expected_sequence:
                errors.append(f"record {expected_sequence}: non-contiguous sequence")
            if record.get("previous_record_sha256") != previous:
                errors.append(f"record {expected_sequence}: previous hash mismatch")
            if supplied != observed:
                errors.append(f"record {expected_sequence}: record hash mismatch")
            previous = supplied if isinstance(supplied, str) else observed
            waveform = record.get("waveform")
            if waveform is None:
                continue
            waveform_count += 1
            try:
                path = self._resolve_waveform_path(str(waveform["path"]))
                if _file_sha256(path) != waveform["npz_sha256"]:
                    errors.append(f"record {expected_sequence}: NPZ hash mismatch")
                    continue
                array = self.load_waveform(str(waveform["path"]))
                if hash_waveform(array) != waveform["waveform_sha256"]:
                    errors.append(f"record {expected_sequence}: waveform hash mismatch")
                if list(array.shape) != waveform["shape"]:
                    errors.append(f"record {expected_sequence}: waveform shape mismatch")
            except (KeyError, OSError, TypeError, ValueError) as exc:
                errors.append(f"record {expected_sequence}: waveform unavailable: {exc}")
        self._verify_attack_transactions(records, errors)

        selected_manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self.manifest_path
            if self.manifest_path.exists()
            else None
        )
        manifest: dict[str, Any] | None = None
        if selected_manifest_path is not None:
            manifest = self._read_manifest(selected_manifest_path, errors)
        elif require_manifest:
            errors.append("terminal manifest is required but unavailable")

        completeness_requested = bool(
            manifest_path is not None
            or require_manifest
            or manifest is not None
            or expected_records is not None
            or expected_terminal_sha256 is not None
        )
        if manifest is not None:
            if manifest.get("log_file") != self.path.name:
                errors.append("terminal manifest names a different log file")
            manifest_records = manifest.get("expected_records")
            if not isinstance(manifest_records, int) or manifest_records < 1:
                errors.append("terminal manifest expected_records is invalid")
            elif expected_records is None:
                expected_records = manifest_records
            elif expected_records != manifest_records:
                errors.append("explicit expected_records conflicts with terminal manifest")
            manifest_terminal = manifest.get("terminal_record_sha256")
            if not isinstance(manifest_terminal, str) or len(manifest_terminal) != 64:
                errors.append("terminal manifest record hash is invalid")
            elif expected_terminal_sha256 is None:
                expected_terminal_sha256 = manifest_terminal
            elif expected_terminal_sha256 != manifest_terminal:
                errors.append("explicit terminal hash conflicts with terminal manifest")
            if self.path.exists():
                try:
                    if _file_sha256(self.path) != manifest.get("log_sha256"):
                        errors.append("terminal manifest log hash mismatch")
                except OSError as exc:
                    errors.append(f"log unavailable during manifest verification: {exc}")

        if expected_records is not None:
            if (
                isinstance(expected_records, bool)
                or not isinstance(expected_records, int)
                or expected_records < 1
            ):
                errors.append("expected_records must be a positive integer")
            elif len(records) != expected_records:
                errors.append(
                    f"terminal record count mismatch: expected {expected_records}, "
                    f"observed {len(records)}"
                )
        if expected_terminal_sha256 is not None:
            if (
                not isinstance(expected_terminal_sha256, str)
                or len(expected_terminal_sha256) != 64
            ):
                errors.append("expected terminal record hash is invalid")
            elif not records or previous != expected_terminal_sha256:
                errors.append("terminal record hash mismatch")

        complete = None if not completeness_requested else not errors
        return LogVerification(
            not errors,
            len(records),
            waveform_count,
            tuple(errors),
            complete=complete,
            expected_records=expected_records,
            terminal_record_sha256=None if not records else previous,
            manifest_path=(
                None if selected_manifest_path is None else str(selected_manifest_path)
            ),
        )


SecurityEventLogger = AppendOnlySecurityLogger
