"""Append newly documented environment variables without replacing user values."""

from __future__ import annotations

import re
from pathlib import Path

_ASSIGNMENT = re.compile(r"^\s*(?:#\s*)?(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def sync_env(
    example_path: Path = Path(".env.example"),
    env_path: Path = Path(".env"),
) -> list[str]:
    """Copy missing assignments, preserving comments and all existing file bytes."""
    example_bytes = example_path.read_bytes()
    example = example_bytes.decode("utf-8")
    try:
        existing_bytes = env_path.read_bytes()
    except FileNotFoundError:
        existing_bytes = None
    existing = existing_bytes.decode("utf-8") if existing_bytes is not None else ""
    known = {match.group(1) for line in existing.splitlines() if (match := _ASSIGNMENT.match(line))}
    added: list[str] = []
    lines: list[str] = []
    for line in example.splitlines():
        match = _ASSIGNMENT.match(line)
        if match is not None and match.group(1) not in known:
            name = match.group(1)
            known.add(name)
            added.append(name)
            lines.append(line)

    if existing_bytes is None:
        env_path.write_bytes(example_bytes)
    elif lines:
        newline = "\r\n" if b"\r\n" in existing_bytes else "\n"
        separator = "" if not existing or existing.endswith(("\n", "\r")) else newline
        addition = separator + newline.join(lines) + newline
        with env_path.open("ab") as target:
            target.write(addition.encode("utf-8"))
    return added
