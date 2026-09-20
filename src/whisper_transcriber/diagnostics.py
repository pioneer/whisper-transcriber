"""Diagnostics: report environment, dependency versions and CUDA usability.

Deliberately avoids importing torch; CUDA availability is checked through
CTranslate2 directly, which is the runtime faster-whisper actually uses.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import config as cfg


def cuda_library_dirs() -> list[str]:
    """Return ``lib`` directories provided by the ``nvidia-cublas-cu12`` and
    ``nvidia-cudnn-cu12`` wheels, if they are installed in the current
    environment. Computed dynamically so no machine-specific path is ever
    hardcoded.
    """
    dirs: list[str] = []
    for module_name in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError):
            spec = None
        if spec and spec.submodule_search_locations:
            for location in spec.submodule_search_locations:
                lib_dir = Path(location) / "lib"
                if lib_dir.is_dir():
                    dirs.append(str(lib_dir))
    return dirs


@dataclass(slots=True)
class DiagnosticsReport:
    python_version: str
    platform: str
    faster_whisper_version: str
    ctranslate2_version: str
    cuda_device_count: int | None
    cuda_error: str | None
    nvidia_smi_output: str | None
    cuda_lib_dirs: list[str] = field(default_factory=list)
    cuda_compute_types: list[str] = field(default_factory=list)
    cpu_compute_types: list[str] = field(default_factory=list)
    configuration: dict[str, str] = field(default_factory=dict)

    @property
    def cuda_usable(self) -> bool:
        return self.cuda_error is None and (self.cuda_device_count or 0) > 0


def _get_ctranslate2_cuda_info() -> tuple[int | None, str | None]:
    """Return (cuda_device_count, error_message)."""
    try:
        import ctranslate2
    except Exception as exc:  # pragma: no cover - import failure is environmental
        return None, f"could not import ctranslate2: {exc}"

    try:
        count = ctranslate2.get_cuda_device_count()
    except Exception as exc:
        return None, str(exc)
    return count, None


def _get_supported_compute_types(device: str) -> list[str]:
    try:
        import ctranslate2

        return sorted(ctranslate2.get_supported_compute_types(device))
    except Exception:
        return []


def _get_nvidia_smi_output() -> str | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return f"nvidia-smi failed to run: {exc}"
    if result.returncode != 0:
        return f"nvidia-smi exited with code {result.returncode}: {result.stderr.strip()}"
    return result.stdout.strip()


def collect_diagnostics() -> DiagnosticsReport:
    try:
        import faster_whisper

        fw_version = getattr(faster_whisper, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover
        fw_version = f"import failed: {exc}"

    try:
        import ctranslate2

        ct2_version = getattr(ctranslate2, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover
        ct2_version = f"import failed: {exc}"

    cuda_device_count, cuda_error = _get_ctranslate2_cuda_info()
    nvidia_smi_output = _get_nvidia_smi_output()

    return DiagnosticsReport(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        faster_whisper_version=fw_version,
        ctranslate2_version=ct2_version,
        cuda_device_count=cuda_device_count,
        cuda_error=cuda_error,
        nvidia_smi_output=nvidia_smi_output,
        cuda_lib_dirs=cuda_library_dirs(),
        cuda_compute_types=_get_supported_compute_types("cuda"),
        cpu_compute_types=_get_supported_compute_types("cpu"),
        configuration={
            "model": cfg.DEFAULT_MODEL,
            "device": cfg.DEFAULT_DEVICE,
            "compute_type": cfg.DEFAULT_COMPUTE_TYPE,
            "beam_size": str(cfg.DEFAULT_BEAM_SIZE),
            "vad_filter": str(cfg.DEFAULT_VAD_FILTER),
            "language": cfg.DEFAULT_LANGUAGE or "auto",
        },
    )


def format_report(report: DiagnosticsReport) -> str:
    lines = [
        "=== whisper-transcriber diagnostics ===",
        f"Python version:       {report.python_version}",
        f"Platform:             {report.platform}",
        f"faster-whisper:       {report.faster_whisper_version}",
        f"CTranslate2:          {report.ctranslate2_version}",
    ]

    if report.cuda_error is not None:
        lines.append(f"CUDA (CTranslate2):   NOT usable ({report.cuda_error})")
    else:
        lines.append(f"CUDA (CTranslate2):   usable ({report.cuda_device_count} device(s) found)")

    lines.append("")
    lines.append("--- nvidia-smi ---")
    lines.append(report.nvidia_smi_output or "nvidia-smi not available")

    lines.append("")
    lines.append("--- bundled CUDA libraries (nvidia-cublas-cu12 / nvidia-cudnn-cu12) ---")
    if report.cuda_lib_dirs:
        for lib_dir in report.cuda_lib_dirs:
            lines.append(lib_dir)
        lines.append("Run `uv run inv cuda-env` to print an export command for these paths.")
    else:
        lines.append("not found (are nvidia-cublas-cu12 / nvidia-cudnn-cu12 installed?)")

    lines.append("")
    lines.append("--- supported compute types (CTranslate2) ---")
    lines.append(f"cuda: {', '.join(report.cuda_compute_types) or 'unknown'}")
    lines.append(f"cpu:  {', '.join(report.cpu_compute_types) or 'unknown'}")

    lines.append("")
    lines.append("--- default configuration ---")
    for key, value in report.configuration.items():
        lines.append(f"{key:<15} = {value}")

    return "\n".join(lines)
