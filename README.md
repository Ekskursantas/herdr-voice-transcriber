# Herdr voice transcriber

Dictate a prompt into the focused Herdr pane.

## Use

1. Press `F9` in Herdr. A small pill appears at the bottom of the terminal window. Speak.
   The red bars follow your voice.
2. Press `F9` again. The pill shows `Transcribing`, then `Cleaning up`, in amber.
3. The pill shows `Pasted` in green. The clean text is in the pane that had focus when you started.

The text is not submitted. Read it, then press Enter.
A red message in the pill means an error. See `state/voice.log`.

## How it works

```
F9 -> voice.py toggle -> voice.py record (background)
        microphone (sounddevice, 16 kHz)
     -> faster-whisper large-v3-turbo on the GPU   (local, free)
     -> claude -p --model haiku                    (cleanup: fillers, repeats, punctuation)
     -> herdr pane send-text <pane> <text>
```

The key binding is in `%APPDATA%\herdr\config.toml` (`[[keys.command]]`).
Settings are in `config.toml`: model, language, cleanup on/off, vocabulary.

## Test without a microphone

```
.venv\Scripts\python.exe voice.py file some-audio.wav
```

A recording stops by itself after 10 minutes (`max_seconds` in `config.toml`). The text is then transcribed and pasted as usual.

## Requirements

- Windows, with [Herdr](https://herdr.dev) 0.9 or later
- Python 3.11 or later
- An NVIDIA GPU (or set `device = "cpu"` and `compute_type = "int8"` in `config.toml`)
- Claude Code (`claude`) with a login, for the cleanup step

## Install

```
git clone https://github.com/Ekskursantas/herdr-voice-transcriber
cd herdr-voice-transcriber
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Add the key to `%APPDATA%\herdr\config.toml`. Replace `C:\path\to` with your folder:

```toml
[[keys.command]]
key = "f9"
type = "shell"
command = 'C:\path\to\herdr-voice-transcriber\.venv\Scripts\pythonw.exe C:\path\to\herdr-voice-transcriber\voice.py toggle'
```

Then run `herdr server reload-config`. The first recording downloads the Whisper model (about 1.6 GB).

## Privacy

The audio stays on your PC. Only the transcript text goes to Claude for the cleanup. Set `cleanup = false` to keep the text local too. `state/voice.log` holds your transcripts; it is not in git.

## Decisions

- 2026-11-11: Local Whisper does the transcription. Reason: Claude does not accept audio input.
- 2026-11-11: Claude Haiku (`claude -p`) does the cleanup. Reason: best text quality for a small usage cost; about 2.3 s for each call.
- 2026-11-11: No second transcription run. Reason: the cleanup pass fixes wrong words from context; a second run adds time and little quality.
- 2026-11-11: The key is a toggle, not hold-to-talk. Reason: Herdr key commands fire on key press only.
- 2026-11-11: The model loads for each recording, while the user speaks (about 2.7 s). Reason: no background service to manage, and no VRAM in use between recordings.
- 2026-11-11: `claude -p` runs with `--safe-mode --tools ""`. Reason: the cleanup must not load CLAUDE.md, plugins or MCP servers, and must not run tools.
- 2026-11-11: A floating overlay window (tkinter) shows the state, and the beeps are off by default. Reason: Herdr 0.9.x has no API for a colored status indicator; its tab bar status is plain text with a refresh of 1 s or more. The overlay does not take focus and passes clicks through.
- 2026-11-11: The `claude` cleanup process starts when the recording starts (`--input-format stream-json`). Reason: the process start is 2 to 4 s of the cleanup time; a started process answers in about 1 s. A one-shot `claude -p` call is the fallback.
- 2026-11-11: The vocabulary goes to Whisper as `hotwords`, not as `initial_prompt`. Reason: with near-silent audio, Whisper returned the prompt text itself ("Vocabulary, ...").
- 2026-11-11: A recording stops after 10 minutes at most. Reason: a forgotten recording must not run without end; the limit was 5 minutes before.
- 2026-11-11: The script sends text without Enter. Reason: the user checks the text before it goes to an agent.
