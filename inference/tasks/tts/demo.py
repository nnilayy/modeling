"""
TTS Demo — Gradio UI for Fish Speech S2 Pro via vLLM-Omni.

Usage:
    python -m inference.tasks.tts.demo
    python -m inference.tasks.tts.demo --base-url http://localhost:8091 --port 7860
"""

from __future__ import annotations

import argparse
import base64
import time
from datetime import datetime
from pathlib import Path

import requests

OUTPUT_DIR = Path("output_audio")
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


def _mime_for(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower().lstrip(".")
    mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
                "ogg": "audio/ogg", "webm": "audio/webm", "m4a": "audio/mp4",
                "aac": "audio/aac", "mp4": "audio/mp4"}
    return mime_map.get(suffix, "audio/wav")


def upload_voice(
    name: str,
    audio_path: str,
    ref_text: str,
    base_url: str,
) -> dict:
    """Upload a voice sample via POST /v1/audio/voices for persistent cloning."""
    url = f"{base_url}/v1/audio/voices"
    mime = _mime_for(audio_path)
    with open(audio_path, "rb") as f:
        files = {"audio_sample": (Path(audio_path).name, f, mime)}
        data = {"consent": "self", "name": name}
        if ref_text:
            data["ref_text"] = ref_text
        r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()


def list_voices(base_url: str) -> list[str]:
    """Fetch available voice names from GET /v1/audio/voices."""
    try:
        r = requests.get(f"{base_url}/v1/audio/voices", timeout=10)
        r.raise_for_status()
        return r.json().get("voices", [])
    except Exception:
        return ["default"]


def synthesize(
    text: str,
    base_url: str,
    voice: str = "default",
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
    language: str = "Auto",
    instructions: str = "",
    seed: int | None = 42,
) -> tuple[bytes, float]:
    url = f"{base_url}/v1/audio/speech"
    body: dict = {
        "input": text,
        "voice": voice,
        "response_format": "wav",
        "max_new_tokens": 4096,
        "language": language,
    }
    if seed is not None:
        body["seed"] = seed
    if instructions:
        body["instructions"] = instructions
    if ref_audio_path:
        body["ref_audio"] = _audio_to_base64(ref_audio_path)
    if ref_text:
        body["ref_text"] = ref_text

    t0 = time.perf_counter()
    r = requests.post(url, json=body, timeout=300)
    r.raise_for_status()
    elapsed = time.perf_counter() - t0
    return r.content, elapsed


def build_ui(base_url: str):
    import gradio as gr

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
        button_primary_text_color="white",
        block_border_width="0px",
        block_shadow="0 1px 3px 0 rgb(0 0 0 / 0.1)",
        input_background_fill="*neutral_50",
    )

    css = """
    .main-header { text-align: center; margin-bottom: 0.5rem; }
    .main-header h1 { font-size: 1.8rem; font-weight: 700; margin: 0; }
    .main-header p { color: #64748b; font-size: 0.9rem; margin: 0.25rem 0 0 0; }
    .status-bar { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
    footer { display: none !important; }
    """

    with gr.Blocks(theme=theme, css=css, title="Fish Speech S2 Pro — TTS Demo") as demo:

        gr.HTML("""
            <div class="main-header">
                <h1>Fish Speech S2 Pro</h1>
                <p>Text-to-Speech via vLLM-Omni</p>
            </div>
        """)

        with gr.Row():
            # ── Left column: inputs ──
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text to synthesize",
                    placeholder="Type or paste text here... Use [whisper], [excited], [sad], [angry] for emotions.",
                    lines=4,
                    max_lines=12,
                )

                with gr.Row():
                    language_input = gr.Dropdown(
                        label="Language",
                        choices=["Auto", "English", "Chinese", "Japanese", "Korean",
                                 "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
                        value="Auto",
                        scale=1,
                    )
                    instructions_input = gr.Textbox(
                        label="Voice style (optional)",
                        placeholder="e.g. warm female voice, calm pace",
                        scale=2,
                    )
                    seed_input = gr.Number(label="Seed", value=42, precision=0, scale=1)

                # ── Voice Cloning ──
                with gr.Accordion("Voice Cloning", open=False):
                    ref_audio_input = gr.Audio(
                        label="Record or upload reference audio (10-30s)",
                        type="filepath",
                        sources=["microphone", "upload"],
                    )
                    ref_text_input = gr.Textbox(
                        label="Reference text (type exactly what was said in the audio)",
                        placeholder="Provide the exact transcript of your reference audio...",
                        lines=3,
                    )

                    with gr.Row():
                        voice_name_input = gr.Textbox(
                            label="Voice name",
                            placeholder="my_voice",
                            scale=2,
                        )
                        upload_btn = gr.Button("Upload Voice", variant="secondary", scale=1)

                    upload_status = gr.Textbox(
                        label="Upload status",
                        interactive=False,
                        elem_classes=["status-bar"],
                    )

                # ── Voice selection & generate ──
                with gr.Row():
                    initial_voices = list_voices(base_url)
                    voice_input = gr.Dropdown(
                        label="Voice",
                        choices=initial_voices,
                        value=initial_voices[0] if initial_voices else "default",
                        allow_custom_value=True,
                        scale=3,
                    )
                    refresh_voices_btn = gr.Button("Refresh", size="sm", scale=1)

                with gr.Row():
                    submit_btn = gr.Button("Generate", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)

            # ── Right column: output ──
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Generated audio",
                    type="filepath",
                    autoplay=True,
                    show_download_button=True,
                )
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes=["status-bar"],
                )

        gr.Markdown("### History")
        history_container = gr.Dataframe(
            headers=["Time", "Text", "Voice", "Latency", "Size", "File"],
            datatype=["str", "str", "str", "str", "str", "str"],
            label="Generation history",
            interactive=False,
            wrap=True,
        )
        history_state = gr.State([])

        # ── Callbacks ──

        def do_upload(audio_path, ref_text, voice_name):
            if not audio_path:
                return "Record or upload audio first."
            if not voice_name or not voice_name.strip():
                return "Enter a voice name."
            if not ref_text or not ref_text.strip():
                return "Provide the reference text (what was said in the audio)."
            try:
                result = upload_voice(
                    name=voice_name.strip(),
                    audio_path=audio_path,
                    ref_text=ref_text.strip(),
                    base_url=base_url,
                )
                if result.get("success"):
                    return f"Uploaded '{voice_name.strip()}' — click Refresh and select it."
                return f"Server response: {result}"
            except requests.HTTPError as e:
                return f"Upload failed ({e.response.status_code}): {e.response.text[:300]}"
            except Exception as e:
                return f"Upload error: {e}"

        def do_refresh_voices():
            voices = list_voices(base_url)
            return gr.update(choices=voices, value=voices[0] if voices else "default")

        def generate(text, voice, language, instructions, seed, history):
            if not text or not text.strip():
                return None, "Enter some text.", history, history

            seed_val = int(seed) if seed is not None else None
            try:
                audio_bytes, elapsed = synthesize(
                    text=text.strip(),
                    base_url=base_url,
                    voice=voice or "default",
                    language=language or "Auto",
                    instructions=instructions or "",
                    seed=seed_val,
                )
            except requests.ConnectionError:
                return None, "Connection refused — is the TTS server running?", history, history
            except requests.HTTPError as e:
                return None, f"Server error: {e.response.status_code} — {e.response.text[:200]}", history, history
            except Exception as e:
                return None, f"Error: {e}", history, history

            ts = datetime.now().strftime("%H:%M:%S")
            filename = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            out_path = OUTPUT_DIR / filename
            out_path.write_bytes(audio_bytes)

            size_kb = len(audio_bytes) / 1024
            status = f"Done in {elapsed:.2f}s  |  {size_kb:.0f} KB  |  {filename}"
            preview = text[:60] + "..." if len(text) > 60 else text

            history = history + [[ts, preview, voice or "default",
                                  f"{elapsed:.2f}s", f"{size_kb:.0f} KB", filename]]
            return str(out_path), status, history, history

        def clear_all():
            return "", "Auto", "", 42, None, "", [], []

        # ── Wiring ──

        upload_btn.click(
            fn=do_upload,
            inputs=[ref_audio_input, ref_text_input, voice_name_input],
            outputs=[upload_status],
        )

        refresh_voices_btn.click(fn=do_refresh_voices, outputs=[voice_input])

        submit_btn.click(
            fn=generate,
            inputs=[text_input, voice_input, language_input, instructions_input,
                    seed_input, history_state],
            outputs=[audio_output, status_output, history_container, history_state],
        )

        text_input.submit(
            fn=generate,
            inputs=[text_input, voice_input, language_input, instructions_input,
                    seed_input, history_state],
            outputs=[audio_output, status_output, history_container, history_state],
        )

        clear_btn.click(
            fn=clear_all,
            outputs=[text_input, language_input, instructions_input, seed_input,
                     audio_output, status_output, history_container, history_state],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="TTS Demo UI")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vLLM-Omni server URL")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    args = parser.parse_args()

    demo = build_ui(args.base_url)
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=True,
    )


if __name__ == "__main__":
    main()
