# Vuho

## CAPSLOCK to shout at your favorite LLM: ヽ(°〇°)ﾉ  °.ིྀ🤖.⚡︎

Speech-to-text dictation typing for macOS.
CapsLock light is on == vuho is listening & transcribing...
CapsLock again to paste at the cursor.

**Apple Silicon only. macOS 14.0+** 

Runs entirely **on-device** using the Apple Neural Engine (ANE).

## Quickstart

Homebrew:

```bash
brew tap aka-rider/tap
brew install --cask vuho
```

On first launch, download models, grant **permissions**:

- **Microphone** — to capture audio
- **Input Monitoring** — to detect the hotkey globally
- **Accessibility** — to paste text into other apps


## Why not built-in?

1. 100% On-device
2. Good Ukrainian language support
3. Hours long non-stop sessions

## Why not XXX?

XXX is probably better and more polished.
Vuho works for its author.

## Models

Vuho ships two speech models. Pick one in **Settings → Speech Model**; the list shows every
model, downloads the one you choose, and deletes one you no longer want.

| Model | Size | Needs | Notes |
|---|---|---|---|
| [Parakeet TDT v3](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml) | 496 MB | macOS 14.0+ | The default. Fastest, runs partly on the Neural Engine. |
| [Canary 1B v2](https://huggingface.co/FluidInference/canary-1b-v2-coreml) | 569 MB | macOS 15+ | 25 languages. Transcription only. Runs on the CPU, so it is slower (about 0.8 s per 15 s of speech on an M-series Mac). |

Both are CoreML conversions by FluidInference of NVIDIA models. Vuho tells the model which
language you are speaking, from your keyboard input source — if the chosen model does not
support that language, Vuho says so instead of guessing.

Voice activity detection uses [Silero VAD](https://github.com/snakers4/silero-vad) (ONNX, FP16).

## Licenses

- **Parakeet TDT-0.6b-v3 Speech Recognition Model** — NVIDIA Corporation, CC-BY-4.0 (https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- **Parakeet TDT-0.6b-v3 CoreML Conversion** — FluidInference, CC-BY-4.0 (https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml)
- **Canary-1B-v2 Speech Recognition Model** — NVIDIA Corporation, CC-BY-4.0 (https://huggingface.co/nvidia/canary-1b-v2)
- **Canary-1B-v2 CoreML Conversion** — FluidInference, CC-BY-4.0 (https://huggingface.co/FluidInference/canary-1b-v2-coreml)
- **Silero VAD & voice_activity_detector** — Silero.AI & Nicholas Keenan, MIT license (https://github.com/snakers4/silero-vad, https://github.com/nkeenan38/voice_activity_detector)
- Vuho is licensed under [MIT license](LICENSE).
