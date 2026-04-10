"""
TTS client — sends text to a running vLLM-Omni TTS server and saves the audio.

Usage:
    python -m inference.tasks.tts.client "Hello, this is a test."

    python -m inference.tasks.tts.client "Hello, this is a test." \
        --output output.wav --base-url http://localhost:8091

    # With voice cloning
    python -m inference.tasks.tts.client "Hello, this is a cloned voice." \
        --ref-audio /path/to/reference.wav \
        --ref-text "Transcript of the reference audio."

    # With emotion tags
    python -m inference.tasks.tts.client "[excited] Wow, this is amazing!"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests


DEFAULT_BASE_URL = "http://localhost:8091"


def synthesize(
    text: str,
    base_url: str = DEFAULT_BASE_URL,
    voice: str = "default",
    response_format: str = "wav",
    ref_audio: str | None = None,
    ref_text: str | None = None,
) -> bytes:
    url = f"{base_url}/v1/audio/speech"

    body: dict = {
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "max_new_tokens": 4096,
    }

    if ref_audio:
        body["ref_audio"] = ref_audio
    if ref_text:
        body["ref_text"] = ref_text

    r = requests.post(url, json=body, timeout=120)
    r.raise_for_status()
    return r.content


def main():
    parser = argparse.ArgumentParser(description="TTS client for vLLM-Omni")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--output", "-o", default="output.wav", help="Output WAV path")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Server base URL")
    parser.add_argument("--voice", default="default", help="Voice name")
    parser.add_argument("--ref-audio", default=None, help="Reference audio URL/path for cloning")
    parser.add_argument("--ref-text", default=None, help="Transcript of reference audio")
    args = parser.parse_args()

    print(f"Sending to {args.base_url}...")
    print(f"  Text:   {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
    if args.ref_audio:
        print(f"  Clone:  {args.ref_audio}")

    t0 = time.perf_counter()
    audio_bytes = synthesize(
        text=args.text,
        base_url=args.base_url,
        voice=args.voice,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
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
