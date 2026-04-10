"""
TTS client — sends text to a running vLLM-Omni TTS server and saves the audio.

Usage:
    python -m inference.tasks.tts.client "Hello, this is a test."

    python -m inference.tasks.tts.client "Hello, this is a test." \
        --output output.wav --base-url http://localhost:8091

    # With voice cloning (base64-encoded reference audio)
    python -m inference.tasks.tts.client "Hello, this is a cloned voice." \
        --ref-audio /path/to/reference.wav \
        --ref-text "Transcript of the reference audio."

    # With uploaded voice name (use demo UI to upload first)
    python -m inference.tasks.tts.client "Hello!" --voice my_uploaded_voice

    # With emotion tags
    python -m inference.tasks.tts.client "[excited] Wow, this is amazing!"
"""

from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

import requests


DEFAULT_BASE_URL = "http://localhost:8091"


def _audio_to_base64(file_path: str) -> str:
    """Read a local audio file and return a base64 data URL."""
    path = Path(file_path)
    suffix = path.suffix.lower().lstrip(".")
    mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
                "ogg": "audio/ogg", "webm": "audio/webm", "m4a": "audio/mp4"}
    mime = mime_map.get(suffix, "audio/wav")
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def synthesize(
    text: str,
    base_url: str = DEFAULT_BASE_URL,
    voice: str = "default",
    response_format: str = "wav",
    ref_audio: str | None = None,
    ref_text: str | None = None,
    language: str = "Auto",
    instructions: str = "",
    seed: int | None = 42,
) -> bytes:
    url = f"{base_url}/v1/audio/speech"

    body: dict = {
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "max_new_tokens": 4096,
        "language": language,
    }

    if seed is not None:
        body["seed"] = seed
    if instructions:
        body["instructions"] = instructions
    if ref_audio:
        if ref_audio.startswith(("http://", "https://", "data:")):
            body["ref_audio"] = ref_audio
        else:
            body["ref_audio"] = _audio_to_base64(ref_audio)
    if ref_text:
        body["ref_text"] = ref_text

    r = requests.post(url, json=body, timeout=300)
    r.raise_for_status()
    return r.content


def main():
    parser = argparse.ArgumentParser(description="TTS client for vLLM-Omni")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--output", "-o", default="output.wav", help="Output WAV path")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Server base URL")
    parser.add_argument("--voice", default="default", help="Voice name (use uploaded name for cloning)")
    parser.add_argument("--ref-audio", default=None,
                        help="Reference audio file path, HTTP URL, or data: URL for inline cloning")
    parser.add_argument("--ref-text", default=None, help="Transcript of reference audio")
    parser.add_argument("--language", default="Auto",
                        choices=["Auto", "English", "Chinese", "Japanese", "Korean",
                                 "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
                        help="Language hint")
    parser.add_argument("--instructions", default="", help="Voice style instructions")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible voice (default: 42)")
    args = parser.parse_args()

    print(f"Sending to {args.base_url}...")
    print(f"  Text:   {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
    print(f"  Voice:  {args.voice}")
    if args.ref_audio:
        print(f"  Clone:  {args.ref_audio}")
    if args.instructions:
        print(f"  Style:  {args.instructions}")
    print(f"  Seed:   {args.seed}")

    t0 = time.perf_counter()
    audio_bytes = synthesize(
        text=args.text,
        base_url=args.base_url,
        voice=args.voice,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        language=args.language,
        instructions=args.instructions,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0

    out = Path(args.output)
    out.write_bytes(audio_bytes)

    size_kb = len(audio_bytes) / 1024
    print(f"\n  Saved:    {out}")
    print(f"  Size:     {size_kb:.1f} KB")
    print(f"  Latency:  {elapsed:.2f}s")


if __name__ == "__main__":
    main()
