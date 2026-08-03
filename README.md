# Vuho

Local-first, fully private speech-to-text for macOS. Dictate with your keyboard — no cloud, no account, no internet required.

**Apple Silicon only.** Runs entirely on-device using the Apple Neural Engine.

## Features

- **Global hotkey** — press a key (Caps Lock by default, or a chord) to start listening
- **Live transcription** — semi-transparent overlay shows partial results and a waveform
- **One-key confirm** — press the hotkey again to stop; cleaned text is injected at your cursor
- **Filler-word cleanup** — rule-based removal of fillers ("um", "uh"), spacing normalization, and newline cleanup
- **Always offline** — models live locally; nothing ever leaves your Mac

## Requirements

- macOS 14.0+ on Apple Silicon (M1 or later)
- ~500 MB disk space for the model files

## Quick Start

### 1. Fetch the model

```bash
./scripts/fetch-model.sh
```

This downloads the Parakeet TDT v3 CoreML model and Silero VAD weights into `models/`.
Run once — the script is idempotent. No cleanup model is fetched (rule-based post-processing is built in).

> **No `huggingface-cli`?** The script falls back to `curl` automatically.

### 2. Build and run

```bash
cargo run -p vuho-ui --features demo   # overlay demo (no mic)
cargo build --release -p vuho-ui       # full build
```

### 3. Grant permissions

On first launch, macOS will prompt for:

- **Microphone** — to capture audio
- **Accessibility** — to inject text into other apps
- **Input Monitoring** — to detect the hotkey globally

## Building a distributable `.app`

```bash
SIGN_ID="Vuho Dev" ./scripts/package.sh
```

This produces `Vuho.app` in the repository root, fully code-signed and ready to distribute.

## Settings

Settings are stored at `~/.config/vuho/settings.json` (or `$XDG_CONFIG_HOME/vuho/settings.json`):

```json
{
  "hotkey": {
    "key": "CapsLock",
    "modifiers": []
  },
  "microphone": null
}
```

Change the hotkey in the Settings window — it takes effect immediately. The microphone setting applies at the next session start.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design documentation, including 20 architecture decision records (ADRs) and the target system design.

## Model

Vuho uses the [Parakeet TDT v3](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml) model (0.6B parameters, FastConformer architecture) converted to CoreML by FluidInference. The model runs on the Apple Neural Engine for real-time transcription.

Voice activity detection uses [Silero VAD](https://github.com/snakers4/silero-vad) (ONNX, FP16) via the `voice_activity_detector` Rust crate, which embeds Silero v5 directly — the fetched `models/silero-vad/` ONNX weights are provisioned for a future direct-`ort` swap and are not loaded by the app today.

## Models & licenses

- **Parakeet TDT-0.6b-v3 Speech Recognition Model** — NVIDIA Corporation, CC-BY-4.0 (https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- **Parakeet TDT-0.6b-v3 CoreML Conversion** — FluidInference, CC-BY-4.0 (https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml)
- **Silero VAD & voice_activity_detector** — Silero.AI & Nicholas Keenan, MIT license (https://github.com/snakers4/silero-vad, https://github.com/nkeenan38/voice_activity_detector)

## License

MIT — see [LICENSE](LICENSE) in the repository root.

Third-party attributions are included in the app bundle at `Contents/Resources/ATTRIBUTION.txt` (copied there by `scripts/bundle-macos.sh`) and in this repo at [packaging/ATTRIBUTION.txt](packaging/ATTRIBUTION.txt).
