# whisper-transcriber

Local, offline video/audio transcription powered by
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2).
No cloud APIs, no PyTorch dependency, no artificial duration limits — built
for long recordings (multi-hour) on a modest GPU (tested on a 4 GB GTX 1050
Ti Mobile).

## Features

- Transcribes `mp4`, `mkv`, `webm`, `mp3`, `wav`, `m4a` directly.
- Writes `<file>.txt` (human-readable, timestamped) and `<file>.srt`
  (standard subtitles) next to the source file.
- Streams segments as they are produced — never buffers the whole
  transcript in memory, so hours-long files are fine.
- Live progress: detected language, model/device, media position / total
  duration, percentage complete, and real (wall-clock) time elapsed.
- Voice activity detection (VAD) enabled by default.
- Automatic language detection, or pass `--language=uk` / `--language=ru`
  explicitly.
- Clean Ctrl+C handling — partial output is kept, not corrupted.
- Fails loudly (non-zero exit + clear message) if CUDA was requested but
  isn't usable at all — never silently falls back to CPU. If CUDA runs out
  of VRAM partway through, it loudly resumes on CPU from where it left off
  by default (pass `--no-cpu-fallback` to fail instead).
- `uv run inv diagnose` reports Python/OS/library/CUDA state without
  depending on PyTorch.

## Requirements

- Linux, Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- An NVIDIA GPU + driver, if you want to use CUDA (CPU also works, just
  slower)

## Installation

```bash
git clone <this-repo>
cd whisper-transcriber
uv sync
```

`uv sync` creates `.venv/` and installs everything from `pyproject.toml` /
`uv.lock`, including the CUDA 12 runtime libraries
(`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) needed by CTranslate2 — you do
not need a system-wide CUDA toolkit install.

## Usage

```bash
# Basic transcription (auto language detection, model=medium, device=cuda)
uv run inv transcribe path/to/video.mp4

# Force Ukrainian
uv run inv transcribe path/to/video.mp4 --language=uk

# Force Russian
uv run inv transcribe path/to/video.mp4 --language=ru

# Use a smaller/faster model
uv run inv transcribe path/to/video.mp4 --model=small

# Run on CPU (no GPU / CUDA not available)
uv run inv transcribe path/to/video.mp4 --device=cpu

# Use CUDA, but retry on CPU instead of failing if the GPU runs out of VRAM
uv run inv transcribe path/to/video.mp4 --cpu-fallback

# Disable the default CPU fallback and fail immediately on a GPU OOM instead
uv run inv transcribe path/to/video.mp4 --no-cpu-fallback

# Full override
uv run inv transcribe path/to/video.mp4 --model=small --device=cpu \
    --compute-type=int8 --language=uk --beam-size=1

# Diagnostics
uv run inv diagnose

# Tests
uv run inv test
```

This produces `video.txt` and `video.srt` next to `video.mp4`.

### Transcribing from a video URL

Pass a URL instead of a local path, and the video is downloaded first with
[`yt-dlp`](https://github.com/yt-dlp/yt-dlp) (must be installed separately,
e.g. `pip install yt-dlp` or via your system package manager):

```bash
uv run inv transcribe https://example.com/watch?v=abc123

# Change the download folder (default: ~/Video)
uv run inv transcribe https://example.com/watch?v=abc123 --video-download-dir=/data/videos

# Delete the downloaded video once transcription finishes (default: kept)
uv run inv transcribe https://example.com/watch?v=abc123 --delete-video

# Use a different download tool/command; {url} and {output} are substituted
uv run inv transcribe https://example.com/watch?v=abc123 \
    --video-download-command="yt-dlp --format best -o {output} {url}"
```

The TXT/SRT outputs are written next to the downloaded video file inside the
download folder, named after the video's title (e.g. `My Video.a1b2c3d4e5.mp4`,
with a short tag derived from the URL). Re-running the same URL reuses that
same destination path, so a finished download is skipped and an interrupted
one is resumed instead of starting over — and if transcription itself fails
(e.g. an out-of-memory error), the downloaded video is kept (unless
`--delete-video` was passed and transcription succeeded), so the next run
doesn't need to re-download it.

### TXT output format

```
[00:00:03 → 00:00:09] Вітаю всіх сьогодні...
[00:00:09 → 00:00:15] Сьогодні ми поговоримо...
```

### SRT output format

Standard SRT, with millisecond-precision timestamps:

```
1
00:00:03,000 --> 00:00:09,000
Вітаю всіх сьогодні...

2
00:00:09,000 --> 00:00:15,000
Сьогодні ми поговоримо...
```

## Model recommendations for 4 GB VRAM

| Model    | VRAM (approx, int8/float16) | Notes                                   |
|----------|------------------------------|------------------------------------------|
| `small`  | ~1 GB                        | Fast, good for quick drafts / long files |
| `medium` | ~2.5–3 GB                    | Default. Best accuracy that reliably fits in 4 GB |
| `large-v3` | ~4.5 GB+ (int8)             | Usually too tight on a 4 GB card; expect OOM or very slow fallback |

**`small` vs `medium`**: `medium` has roughly 3x more parameters than
`small` and is noticeably more accurate, especially for accents,
background noise, and non-English speech (Ukrainian/Russian included) — at
the cost of being several times slower and using more VRAM. `small` is a
reasonable choice when you need speed (e.g. quick previews, near
real-time use) and can tolerate more transcription errors. On a 4 GB card,
`medium` with `compute_type=int8_float32` (the default here) is the sweet
spot; drop to `small` if you hit VRAM pressure with other GPU workloads
running, or need faster turnaround.

> **Compute type note:** `int8_float16` is only efficient on GPUs with
> compute capability ≥ 7.0 (Turing and newer). Pascal-generation cards
> like the GTX 1050 Ti (compute capability 6.1) don't support it
> efficiently, so the default here is `int8_float32`, which CTranslate2
> confirms is supported (`uv run inv diagnose` lists supported compute
> types per device). If you have a newer GPU, override with
> `--compute-type=int8_float16` for extra speed.

If you hit out-of-memory errors with `medium`, try:

```bash
uv run inv transcribe file.mp4 --model=small
# or
uv run inv transcribe file.mp4 --compute-type=int8
```

### Keeping `large-v3` quality despite VRAM limits

VRAM usage isn't fixed for the whole run — it can grow during decoding, so
an OOM can happen partway through a long file even if the model loaded
fine. By default this resumes on CPU automatically when that happens (see
above), so you don't need to do anything. If you want to avoid the CPU
slowdown and stay on GPU, try, in order:

```bash
# Lightest CUDA compute type (uses less VRAM than the default int8_float32)
uv run inv transcribe file.mp4 --model=large-v3 --compute-type=int8

# Greedy decoding uses less memory than beam search, at a small accuracy cost
uv run inv transcribe file.mp4 --model=large-v3 --compute-type=int8 --beam-size=1

# No VRAM limit at all (uses system RAM instead), just slower
uv run inv transcribe file.mp4 --model=large-v3 --device=cpu
```

Also close other GPU-using programs before starting a long transcription.

## How models are downloaded/cached

The first time you use a given model name, `faster-whisper` downloads it
from the Hugging Face Hub and caches it under `~/.cache/huggingface/`
(controlled by the standard `HF_HOME` / `HUGGINGFACE_HUB_CACHE`
environment variables). Subsequent runs reuse the cached copy — no
network access is needed afterwards. There is no cloud transcription API
involved; only the model *weights* are downloaded once, all inference
runs locally.

## CUDA troubleshooting (Linux)

Modern `faster-whisper` / CTranslate2 builds link against CUDA 12
libraries (cuBLAS, cuDNN 9) at the *shared library* level, independent of
what your NVIDIA driver reports as the max supported CUDA version. This
project depends on the `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` PyPI
wheels so you don't need a system CUDA toolkit — but the dynamic linker
still needs to find those `.so` files via `LD_LIBRARY_PATH`.

`uv run inv transcribe ...` handles this automatically: it detects the
CUDA libraries bundled in `.venv` and re-executes itself with
`LD_LIBRARY_PATH` set correctly before importing faster-whisper, so you
normally don't have to do anything.

If you want to run something else (a script, a Python REPL, `python -m
whisper_transcriber...`) against the same libraries, print the required
export with:

```bash
uv run inv cuda-env
# prints, e.g.:
# export LD_LIBRARY_PATH=".../whisper-transcriber/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:.../nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

Then evaluate it in your shell:

```bash
eval "$(uv run inv cuda-env)"
```

**Common CUDA problems and fixes:**

- **`libcudnn.so.9: cannot open shared object file`** or similar for
  `libcublas` — `LD_LIBRARY_PATH` isn't set. Use `uv run inv transcribe`
  (handles it automatically) or `eval "$(uv run inv cuda-env)"` first.
- **`CUDA driver version is insufficient`** / no CUDA device found — your
  NVIDIA driver is missing, too old, or the GPU isn't visible (check
  `nvidia-smi`). Run `uv run inv diagnose` for details. As a workaround,
  pass `--device=cpu`.
- **Out of memory / `CUDA_ERROR_OUT_OF_MEMORY`** — close other
  GPU-hungry programs, use a smaller `--model` (e.g. `small`), or a
  lighter `--compute-type` (e.g. `int8`).
- **Model download failures** — check your internet connection; once a
  model is cached under `~/.cache/huggingface/`, no network is required.

By design, this tool **never silently falls back from CUDA to CPU** — if
you asked for `--device=cuda` and it can't be used *at all* (no device,
missing libraries, etc.), you get a clear error and a non-zero exit code,
not a slow surprise. The one exception is running out of VRAM mid-transcription:
by default (`--cpu-fallback`, on unless you pass `--no-cpu-fallback`) it prints
a clear message and resumes on CPU from wherever the GPU attempt left off
(already-written segments are kept, not re-transcribed) instead of failing —
keeping the same model/quality, just slower.

## Project layout

```
whisper-transcriber/
├── pyproject.toml
├── uv.lock
├── tasks.py                     # Invoke tasks (CLI entry points)
├── README.md
├── src/
│   └── whisper_transcriber/
│       ├── __init__.py
│       ├── config.py            # centralized defaults
│       ├── transcriber.py       # faster-whisper integration
│       ├── downloader.py        # video URL download (yt-dlp)
│       ├── output.py            # TXT/SRT formatting + incremental writer
│       ├── cli.py               # orchestration, progress printing, errors
│       └── diagnostics.py       # environment/CUDA diagnostics
└── tests/
    ├── test_cli.py
    ├── test_downloader.py
    ├── test_output.py
    ├── test_timestamps.py
    └── test_transcriber.py
```

## Configuration defaults

Centralized in `src/whisper_transcriber/config.py`:

```python
model = "medium"
device = "cuda"
compute_type = "int8_float32"
beam_size = 5
vad_filter = True
language = None  # automatic detection
video_download_dir = "~/Video"
video_download_command = "yt-dlp -o {output} {url}"
delete_video = False
cpu_fallback = True
```

Override any of these per-run via CLI flags (see Usage above).
