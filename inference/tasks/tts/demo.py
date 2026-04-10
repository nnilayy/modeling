"""
TTS Demo — Gradio UI for Fish Speech S2 Pro via vLLM-Omni.

Usage:
    python -m inference.tasks.tts.demo
    python -m inference.tasks.tts.demo --base-url http://localhost:8091 --port 7860
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import requests

OUTPUT_DIR = Path("output_audio")

DEFAULT_BASE_URL = "http://localhost:8091"
DEFAULT_EXAMPLES = [
    "Hello, this is a test of Fish Speech text to speech.",
    "[excited] Wow, this is absolutely incredible!",
    "[whisper] Let me tell you a secret.",
    "[sad] I can't believe it's already over.",
    "[angry] This is completely unacceptable!",
    "The quick brown fox jumps over the lazy dog.",
]


def synthesize(
    text: str,
    base_url: str,
    voice: str = "default",
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
) -> tuple[bytes, float]:
    url = f"{base_url}/v1/audio/speech"
    body: dict = {
        "input": text,
        "voice": voice,
        "response_format": "wav",
    }
    if ref_audio_path:
        body["ref_audio"] = f"file://{ref_audio_path}"
    if ref_text:
        body["ref_text"] = ref_text

    t0 = time.perf_counter()
    r = requests.post(url, json=body, timeout=120)
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
    .history-item { border-left: 3px solid #3b82f6; padding-left: 0.75rem; margin-bottom: 0.5rem; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="Fish Speech S2 Pro — TTS Demo") as demo:

        gr.HTML("""
            <div class="main-header">
                <h1>Fish Speech S2 Pro</h1>
                <p>Text-to-Speech via vLLM-Omni</p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text",
                    placeholder="Type text to synthesize... Use [whisper], [excited], [sad], [angry] for emotions.",
                    lines=3,
                    max_lines=10,
                )

                with gr.Row():
                    voice_input = gr.Textbox(
                        label="Voice",
                        value="default",
                        scale=1,
                    )
                    ref_text_input = gr.Textbox(
                        label="Reference transcript (for cloning)",
                        placeholder="Transcript of the reference audio...",
                        scale=2,
                    )

                ref_audio_input = gr.Audio(
                    label="Reference audio (for voice cloning)",
                    type="filepath",
                    sources=["upload"],
                )

                with gr.Row():
                    submit_btn = gr.Button("Generate", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)

                gr.Examples(
                    examples=[[ex] for ex in DEFAULT_EXAMPLES],
                    inputs=[text_input],
                    label="Try these",
                )

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
            headers=["Time", "Text", "Latency", "Size", "File"],
            datatype=["str", "str", "str", "str", "str"],
            label="Generation history",
            interactive=False,
            wrap=True,
        )
        history_state = gr.State([])

        def generate(text, voice, ref_audio, ref_text, history):
            if not text or not text.strip():
                return None, "Please enter some text.", history, history

            try:
                audio_bytes, elapsed = synthesize(
                    text=text.strip(),
                    base_url=base_url,
                    voice=voice or "default",
                    ref_audio_path=ref_audio,
                    ref_text=ref_text or None,
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

            history = history + [[ts, preview, f"{elapsed:.2f}s", f"{size_kb:.0f} KB", filename]]

            return str(out_path), status, history, history

        def clear_all():
            return "", "default", None, "", None, "", [], []

        submit_btn.click(
            fn=generate,
            inputs=[text_input, voice_input, ref_audio_input, ref_text_input, history_state],
            outputs=[audio_output, status_output, history_container, history_state],
        )

        text_input.submit(
            fn=generate,
            inputs=[text_input, voice_input, ref_audio_input, ref_text_input, history_state],
            outputs=[audio_output, status_output, history_container, history_state],
        )

        clear_btn.click(
            fn=clear_all,
            outputs=[text_input, voice_input, ref_audio_input, ref_text_input,
                     audio_output, status_output, history_container, history_state],
        )

    return demo, theme, css


def main():
    parser = argparse.ArgumentParser(description="TTS Demo UI")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vLLM-Omni server URL")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    args = parser.parse_args()

    demo, theme, css = build_ui(args.base_url)
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=True,
        theme=theme,
        css=css,
    )


if __name__ == "__main__":
    main()
