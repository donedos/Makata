# Makata

Zero-shot voice-cloning TTS server for **Tagalog / Taglish / English** with a
pluggable engine system:

| Engine | Tagalog | Notes |
|---|---|---|
| **OmniVoice** (k2-fsa) — default | ✅ native `fil` (646 languages) | 0.6B diffusion-LM; weights CC-BY-NC (code Apache-2.0) |
| XTTS v2 (Coqui) | ⚠️ via fallback phonemizer (`en`/`es`) | 17 languages; checkpoint CPML (non-commercial) |

```
[mic/upload ref] → [preprocess: trim·denoise·-23 LUFS] → [engine prompt cache (.pt/.npz)]
[text TL/EN]     → [normalizer: numbers·₱/% ·dates]    → [OmniVoice / XTTS]  → WAV
```

## Honest limitations (read first)

- **OmniVoice Filipino is trained on ~8h of data** — pronunciation of rare words may
  drift. A/B it against the XTTS route (both ship enabled) and pick per voice.
- Engines are **mutually exclusive in VRAM**: loading one unloads the other
  (~10–20s swap). On <8GB GPUs OmniVoice's audio codec runs on CPU automatically.
- OmniVoice clones best from **3–20s references that end on a complete
  sentence**. Without a transcript, Makata cuts the clip at a real pause
  (~8–12s) and keeps it only if the Whisper transcription ends sentence-
  complete — otherwise the model continues the unfinished reference clause,
  which is heard as a stray voice *before* your text. If you supply a
  **transcript** it must match the audio word-for-word (misalignment corrupts
  the clone) and should end with punctuation.
- Both checkpoints are **non-commercial licensed**.
- **VRAM:** both engines fit a 4GB GPU; set `MAKATA_DEVICE=cpu` to force CPU.

## Start Makata

Makata requires **Python 3.10+** and **ffmpeg** on `PATH`. Choose the setup
for your operating system below, then use one of the Docker, pip, or dev
checkout options.

### Platform setup

#### macOS

Install [Homebrew](https://brew.sh/) if it is not already installed, then:

```bash
brew install python@3.11 ffmpeg
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
MAKATA_DEVICE=cpu makata check-setup
MAKATA_DEVICE=cpu makata serve
```

Docker Desktop also works on Intel and Apple Silicon Macs:

```bash
docker compose up --build
```

macOS does not provide NVIDIA CUDA. The native setup therefore uses CPU mode;
Apple Silicon GPU/MPS is not currently selected by Makata.

#### Ubuntu Linux

Install the system prerequisites:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
makata check-setup
makata serve
```

On an NVIDIA GPU, install a compatible NVIDIA driver and
[`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
for Docker GPU support. Linux PyTorch wheels include the CUDA runtime; use
`MAKATA_DEVICE=cpu` if you want to force CPU mode.

#### Windows

Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/windows/)
and select **Add Python to PATH** during installation. Install `ffmpeg` and
add it to `PATH`, for example with [winget](https://learn.microsoft.com/windows/package-manager/winget/):

```powershell
winget install Gyan.FFmpeg
```

From PowerShell in the Makata checkout:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
makata check-setup
makata serve
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser
RemoteSigned` once, or invoke the environment directly with
`.venv\Scripts\python.exe`. The standard Windows PyTorch install is CPU-only.
For NVIDIA CUDA, install the matching PyTorch wheel before installing Makata:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
makata serve
```

Docker Desktop is another supported Windows option and runs CPU-only unless a
separate compatible GPU configuration is available.

### Option A — Docker (easiest)

```bash
docker compose up --build      # build + run in the foreground
docker compose up -d           # or detached; then open http://localhost:8300
```

Works CPU-only out of the box. For NVIDIA GPU acceleration, add the GPU overlay
(requires the driver + `nvidia-container-toolkit` on Linux; Docker Desktop handles
it on Windows/Mac):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

#### Stopping Makata

```bash
docker compose stop            # stop the server (container + data kept)
docker compose start           # start it again later
docker compose down            # remove container (the makata-data volume survives)
docker compose down -v         # ⚠ also deletes voices, samples, outputs, model cache
docker compose logs -f makata  # follow server logs (Ctrl+C to detach)
```

(On hosts where the docker daemon needs root, prefix with `sudo`.)

Voices, samples, outputs and downloaded model weights survive rebuilds via the
`makata-data` volume.

### Option B — pip / pipx

```bash
pipx install .                 # or: pip install .
makata check-setup             # verifies python/torch/cuda/ffmpeg/deps
makata serve                   # http://127.0.0.1:8300  (--port 8000 to change)
```

- **Linux:** PyPI torch wheels bundle CUDA 12 runtime libs — CUDA works out of the box.
- **Windows:** PyPI torch is CPU-only; for CUDA first run
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`
  inside the same venv before installing makata.
- Data lives beside a source checkout (`data/`) or in
  `~/.local/share/makata` when installed; override either with `MAKATA_DATA_DIR`.

### Option C — dev checkout

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .               # editable install → gives you the makata command
makata check-setup             # same checks as python scripts/check_setup.py
makata serve                   # http://127.0.0.1:8300
```

Equivalent classic form, if you prefer not to use the CLI:
`uvicorn app.main:app --host 127.0.0.1 --port 8300`

Then open `http://127.0.0.1:8300` (or `localhost:8300` with Docker) and:

### Accessing from another device (phone, tablet, other PC)

The server binds to `127.0.0.1` (localhost only) by default. To access it from
another device on the same network, bind to `0.0.0.0`:

**Docker** — already exposed; just find your PC's local IP (`ip addr` on Linux,
`ifconfig` on macOS, or check your router admin page) and open
`http://<your-pc-ip>:8300` on the other device.

**pip / dev checkout:**

```bash
makata serve --host 0.0.0.0 --port 8300
```

Or directly with uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8300
```

Make sure your firewall allows incoming connections on port 8300. Both devices
must be on the same local network (e.g. the same Wi-Fi).

### Stopping the server

Press **Ctrl+C** in the terminal where `makata serve` (or `uvicorn`) is running.
For Docker, see the [Stopping Makata](#option-a--docker-easiest) section above.

- record or upload a reference clip (**3–20s, ending on a complete sentence**;
  paste its transcript if you have it — it improves cloning)
- type text (Tagalog / Taglish / English), pick a language or leave Auto,
  and synthesize

### Command-line interface

All installs get a single `makata` command:

| Command | Purpose |
|---|---|
| `makata serve [--host H] [--port P] [--engine omni\|xtts] [--device cuda\|cpu] [--data-dir DIR]` | start the web server + UI |
| `makata check-setup [--model]` | verify python / torch / CUDA / ffmpeg / deps; `--model` also downloads & loads XTTS v2 |

Running `makata` with no subcommand prints help. Flags set the matching
`MAKATA_*` env vars for that invocation only.

First-run downloads from HuggingFace (cached after that):

| Model | Size | Trigger |
|---|---|---|
| OmniVoice | ~1.2 GB | first synthesis |
| Whisper-large-v3-turbo | ~1.6 GB | first transcript-less reference upload |
| XTTS v2 | ~1.8 GB | only when an XTTS voice is used |

Run headless/CPU-only with `MAKATA_DEVICE=cpu` (much slower).

## API

| Endpoint | Description |
|---|---|
| `POST /api/upload-reference` | multipart `file`, optional `name` + `transcript`. Denoises, trims, loudness-normalizes, caches engine prompt. Returns `{voice_id, name, engine}`. |
| `POST /api/voices/{voice_id}/samples` | append an extra reference clip (`file`, optional `transcript`) and **rebuild the clone prompt from all samples** — see [Improving a registered voice](#improving-a-registered-voice-multi-sample). |
| `GET /api/voices/{voice_id}/samples[/{index}]` | list stored samples (with transcripts) / fetch a sample WAV. |
| `PATCH /api/voices/{voice_id}` | rename: `{"name": "My Voice"}` (`<data dir>/voices.json`). |
| `POST /api/synthesize` | `{text, voice_id, language}` — auto-routes to the voice's owning engine. Returns `{audio_url, engine, normalized_text, seconds, rtf}`. |
| `GET /api/outputs` · `DELETE /api/outputs/{file}` | generation history (text kept per clip). |
| `GET /api/voices` · `DELETE /api/voices/{id}` | list/delete registered voices. |
| `GET /api/languages` · `GET /api/health` | introspection. |

```bash
curl -F file=@my_voice.wav -F name=Me -F transcript="Nagsalita ako ng Tagalog dito." \
  localhost:8300/api/upload-reference
curl -X POST localhost:8300/api/synthesize -H 'Content-Type: application/json' \
  -d '{"text":"Kumusta! Ang halaga ay ₱125.50 ngayong 2026-08-22.","voice_id":"<id>","language":"auto"}'
```

`auto` detects Tagalog vs English per request via stopword scoring (Taglish-friendly);
numbers/currency/dates are expanded to words before synthesis. For OmniVoice voices,
Tagalog text is synthesized natively; for XTTS voices it routes through
`MAKATA_TL_FALLBACK`.

## Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `MAKATA_ENGINE` | `omni` | engine for newly registered voices (`omni`\|`xtts`) |
| `MAKATA_DEVICE` | `auto` | `cuda` / `cpu` |
| `MAKATA_DATA_DIR` | checkout `data/`, else `~/.local/share/makata` | where voices, samples, outputs and the HF cache live (Docker sets this to `/data`) |
| `MAKATA_OMNI_NUM_STEP` | `32` | diffusion steps (16 = faster, lower fidelity) |
| `MAKATA_OMNI_TARGET_REF` | `12` | preferred pause-cut length (seconds) for auto-transcribed references |
| `MAKATA_OMNI_MAX_PROMPT` | `20` | total stitched clone-prompt length cap (seconds) when rebuilding from multiple samples |
| `MAKATA_TL_FALLBACK` | `en` | XTTS phonemizer for Tagalog text |
| `MAKATA_XTTS_MODEL` / `MAKATA_OMNI_MODEL` | xtts_v2 / k2-fsa/OmniVoice | swap checkpoints |

In Docker all of the above can be set via `environment:` in `docker-compose.yml`
or `docker run -e`; the `makata-data` volume keeps them across rebuilds.

## Improving a registered voice (multi-sample)

Makata's engines are **zero-shot cloners — there is no fine-tuning step**. A
registered voice is a cached clone prompt built from its reference clip(s).
You cannot retrain it, but you *can* sharpen pronunciation — especially
Tagalog — by giving the model more/better evidence: append additional clips
of the same speaker via `POST /api/voices/{voice_id}/samples` (or the
**✎ Edit voice → Extra samples** panel in the UI). Each append preprocesses
the new clip and rebuilds the clone prompt from **all** stored samples.

How each engine combines samples:

| Engine | Rebuild strategy |
|---|---|
| OmniVoice | Clips **with transcripts** are stitched into one prompt — audio concatenated at 250 ms silences, texts joined in the same order — capped at `MAKATA_OMNI_MAX_PROMPT` (20 s; earlier clips win). With **no** transcripts anywhere it falls back to the single longest clip + Whisper auto-transcribe. |
| XTTS v2 | Conditioning latents are averaged across every clip (`transcript` unused by this engine). More clips = steadier timbre. |

### Recording guidelines (Tagalog focus)

- **Same session setup as the original clip**: same mic, same distance
  (~15–20 cm), same room. Samples recorded under different conditions can
  blur the clone instead of improving it.
- **3–20 s per clip**, ending on a complete sentence with final punctuation.
- **Target the sounds that fail.** Record clips deliberately packed with the
  phonemes that drift: `ng`/`nang`, `mga`, `ñ` (*Niño*, *señorita*), the
  glottal stop in *bata' vs batá*, rolled/tapped `r`, and loanwords the
  speaker pronounces the Filipino way (*eroplano*, *kompyuter*).
- **Transcripts must match word-for-word** — including fillers ("*eh*",
  "*ano*", "*kung baga*"), repeated words, and false starts. Misalignment
  corrupts the clone more than a missing transcript.
- Write numbers, currency, dates, and abbreviations **as spoken**: audio says
  "*isang daan at dalawampung piso*" → transcribe exactly that, not "₱120".
- Keep proper nouns/technical terms in there if they mispronounce — that's
  precisely the vocabulary you want anchored.
- 2–4 good transcript-backed samples usually beat 10 sloppy ones.

```bash
curl -F file=@sample2.wav -F transcript="Ang mga bata ay naglalaro sa hardin ngayong hapon." \
  localhost:8300/api/voices/<voice_id>/samples
```

The response reports how the prompt was rebuilt:
`"rebuild": {"mode": "stitched", "used": 2, "dropped": 0}`
(`stitched` = multi-clip prompt, `single`/`single-auto` = one clip).

## Project layout

```
app/
  api/routes.py     FastAPI endpoints
  audio/            preprocess (trim · denoise · LUFS)
  text/             Taglish normalizer, language detect
  tts/engine.py     engine loader + clone-prompt cache
  frontend/         single-file web UI (ships inside the wheel)
  cli.py            makata serve / check-setup
  config.py         paths + MAKATA_* env vars
scripts/check_setup.py   thin wrapper over `makata check-setup`
Dockerfile, docker-compose.yml
```

## Roadmap

- VoxCPM2 as a third engine (explicit Tagalog, Apache-2.0 — needs >4GB VRAM)
- Sentence-level Taglish mixing (route clauses to their best engine)
- Streaming output, MP3 encoding
