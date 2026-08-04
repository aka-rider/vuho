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

Vuho uses the [Parakeet TDT v3](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml) model converted to CoreML by FluidInference.
Voice activity detection uses [Silero VAD](https://github.com/snakers4/silero-vad) (ONNX, FP16).

## Licenses

- **Parakeet TDT-0.6b-v3 Speech Recognition Model** — NVIDIA Corporation, CC-BY-4.0 (https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- **Parakeet TDT-0.6b-v3 CoreML Conversion** — FluidInference, CC-BY-4.0 (https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml)
- **Silero VAD & voice_activity_detector** — Silero.AI & Nicholas Keenan, MIT license (https://github.com/snakers4/silero-vad, https://github.com/nkeenan38/voice_activity_detector)
- Vuho is licensed under [MIT license](LICENSE).
