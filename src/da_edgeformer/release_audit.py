from __future__ import annotations

import re
from pathlib import Path
from typing import Any

IGNORED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "prepared",
    "checkpoints",
    "outputs",
    "reports",
}
FORBIDDEN_SUFFIXES = {".npz", ".npy", ".parquet", ".pt", ".pth", ".ckpt", ".h5", ".hdf5"}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".cff", ".txt", ".sh"}
SENSITIVE_PATTERNS = {
    "unfinished marker": re.compile(r"\b(?:TODO|FIXME|CHANGEME|TBD)\b", re.IGNORECASE),
    "private absolute path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "credential assignment": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[:=]\s*['\"]?[^\s'\"]+"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def release_audit(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    failures: list[str] = []
    reviewed: list[str] = []
    for path in sorted(root_path.rglob("*")):
        relative = path.relative_to(root_path)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            failures.append(f"symbolic link requires manual review: {relative}")
            continue
        if not path.is_file():
            continue
        reviewed.append(str(relative))
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"data/model artifact is release-visible: {relative}")
        if path.name == ".env" or path.stat().st_size > 5_000_000:
            failures.append(f"sensitive or oversized file requires review: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if relative.as_posix() == "src/da_edgeformer/release_audit.py":
                continue
            for label, pattern in SENSITIVE_PATTERNS.items():
                if pattern.search(text):
                    failures.append(f"{label}: {relative}")
    return {
        "ready": not failures,
        "root": root_path.name,
        "reviewed_files": len(reviewed),
        "failures": failures,
    }
