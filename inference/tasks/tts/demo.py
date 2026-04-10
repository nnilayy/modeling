"""
TTS Demo — Gradio UI for Fish Speech S2 Pro via vLLM-Omni.

Usage:
    python -m inference.tasks.tts.demo
    python -m inference.tasks.tts.demo --base-url http://localhost:8091 --port 7860
"""

from __future__ import annotations

import argparse
import base64
import random
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

READING_PROMPTS = [
    "The old lighthouse keeper watched the storm roll in from the west. Dark clouds gathered "
    "over the harbor as fishing boats hurried back to shore. He climbed the spiral staircase "
    "one last time, his weathered hands gripping the iron railing. The beam of light cut through "
    "the rain, a steady pulse guiding sailors home through the darkness.",

    "Every morning, Clara walked through the garden before sunrise. The dew on the roses caught "
    "the first light like tiny diamonds scattered across green velvet. She carried a notebook "
    "where she sketched each new bloom. The quiet hours before the world woke up belonged entirely "
    "to her and the flowers that seemed to lean toward her gentle voice.",

    "The bookshop on Maple Street had been there longer than anyone could remember. Its shelves "
    "reached from floor to ceiling, packed with stories waiting to be discovered. The owner, "
    "a soft-spoken man with silver hair, knew exactly where every book lived. He believed that "
    "the right story always found the right reader at exactly the right moment.",

    "Rain tapped against the window as the train pulled out of the station. The passenger in "
    "seat fourteen opened a leather journal and began to write. Outside, green hills rolled past "
    "like waves frozen in time. The rhythm of the tracks became a kind of music, steady and "
    "comforting, carrying thoughts forward into the unknown distance ahead.",

    "The marketplace buzzed with energy on Saturday mornings. Vendors called out prices while "
    "children ran between the stalls chasing each other. A woman selling fresh bread smiled at "
    "every customer who stopped to smell the warm loaves. The air was thick with spices, roasted "
    "coffee, and the sound of a dozen conversations happening all at once.",

    "High above the valley, an eagle circled slowly in the warm afternoon air. Below, a river "
    "wound through the forest like a silver ribbon untangling itself from the trees. The hiker "
    "paused on the ridge to catch her breath and take in the view. She had walked twelve miles "
    "that day and every step had been worth this single perfect moment.",

    "Professor Chen adjusted his glasses and looked out at the lecture hall. Three hundred faces "
    "stared back at him, some eager, some half asleep. He cleared his throat and began with a "
    "question that had no easy answer. The best lectures, he always said, were the ones that "
    "left students with more questions than they started with.",

    "The jazz club opened its doors at nine, but the real music never started before midnight. "
    "A pianist with long fingers warmed up in the corner, playing scales that turned into melodies "
    "that turned into something nobody had heard before. The bartender polished glasses and nodded "
    "along. In this room, time moved differently, measured in notes instead of minutes.",
]


def random_reading_prompt() -> str:
    return random.choice(READING_PROMPTS)


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


def upload_voice(
    name: str,
    audio_path: str,
    ref_text: str,
    base_url: str,
    description: str = "",
) -> dict:
    """Upload a voice sample via POST /v1/audio/voices for persistent cloning."""
    url = f"{base_url}/v1/audio/voices"
    with open(audio_path, "rb") as f:
        files = {"audio_sample": (Path(audio_path).name, f)}
        data = {"consent": "self", "name": name}
        if ref_text:
            data["ref_text"] = ref_text
        if description:
            data["speaker_description"] = description
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
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text",
                    placeholder="Type text to synthesize... Use [whisper], [excited], [sad], [angry] for emotions.",
                    lines=3,
                    max_lines=10,
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
                        label="Voice instructions (optional)",
                        placeholder="e.g. warm female voice, natural conversational tone",
                        scale=2,
                    )
                    seed_input = gr.Number(
                        label="Seed",
                        value=42,
                        precision=0,
                        scale=1,
                    )

                # ── Voice Cloning: Upload & Register ──
                with gr.Accordion("Voice Cloning", open=False):
                    gr.Markdown(
                        "**Step 1** — Record yourself reading the prompt (10-30s, quiet room, natural pace).  \n"
                        "**Step 2** — Give the voice a name and click **Upload Voice** to register it on the server.  \n"
                        "**Step 3** — Select your uploaded voice from the dropdown and generate."
                    )

                    clone_prompt = gr.Textbox(
                        label="Read this aloud",
                        value=random_reading_prompt(),
                        lines=4,
                        interactive=False,
                    )
                    new_prompt_btn = gr.Button("New prompt", size="sm")

                    ref_audio_input = gr.Audio(
                        label="Record or upload your voice (10-30s)",
                        type="filepath",
                        sources=["microphone", "upload"],
                    )

                    ref_text_input = gr.Textbox(
                        label="Reference transcript (auto-filled when you stop recording)",
                        placeholder="Transcript of the reference audio...",
                        lines=2,
                    )

                    with gr.Row():
                        voice_name_input = gr.Textbox(
                            label="Voice name",
                            placeholder="my_voice",
                            value="",
                            scale=2,
                        )
                        voice_desc_input = gr.Textbox(
                            label="Voice description (optional)",
                            placeholder="warm male narrator",
                            value="",
                            scale=2,
                        )
                        upload_btn = gr.Button("Upload Voice", variant="secondary", scale=1)

                    upload_status = gr.Textbox(label="Upload status", interactive=False,
                                               elem_classes=["status-bar"])

                    gr.Markdown("---")
                    gr.Markdown("**Or use inline ref_audio** — skip upload and pass reference audio + transcript directly per request.")

                    use_inline_ref = gr.Checkbox(
                        label="Use inline reference (don't upload, send base64 each request)",
                        value=False,
                    )

                # ── Voice selection ──
                initial_voices = list_voices(base_url)
                voice_input = gr.Dropdown(
                    label="Voice",
                    choices=initial_voices,
                    value=initial_voices[0] if initial_voices else "default",
                    allow_custom_value=True,
                )
                refresh_voices_btn = gr.Button("Refresh voices", size="sm")

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
            headers=["Time", "Text", "Voice", "Latency", "Size", "File"],
            datatype=["str", "str", "str", "str", "str", "str"],
            label="Generation history",
            interactive=False,
            wrap=True,
        )
        history_state = gr.State([])

        # ── Callbacks ──

        def do_upload(audio_path, ref_text, voice_name, voice_desc):
            if not audio_path:
                return "No audio recorded/uploaded."
            if not voice_name or not voice_name.strip():
                return "Please enter a voice name."
            try:
                result = upload_voice(
                    name=voice_name.strip(),
                    audio_path=audio_path,
                    ref_text=ref_text or "",
                    base_url=base_url,
                    description=voice_desc or "",
                )
                if result.get("success"):
                    return f"Uploaded '{voice_name.strip()}'. Select it from the Voice dropdown (click Refresh)."
                return f"Server response: {result}"
            except requests.HTTPError as e:
                return f"Upload failed ({e.response.status_code}): {e.response.text[:300]}"
            except Exception as e:
                return f"Upload error: {e}"

        def do_refresh_voices():
            voices = list_voices(base_url)
            return gr.update(choices=voices, value=voices[0] if voices else "default")

        def generate(text, voice, ref_audio, ref_text, use_inline,
                     language, instructions, seed, history):
            if not text or not text.strip():
                return None, "Please enter some text.", history, history

            ref_path = ref_audio if use_inline else None
            ref_t = ref_text if use_inline else None
            seed_val = int(seed) if seed is not None else None

            try:
                audio_bytes, elapsed = synthesize(
                    text=text.strip(),
                    base_url=base_url,
                    voice=voice or "default",
                    ref_audio_path=ref_path,
                    ref_text=ref_t or None,
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
            return ("", None, "", False, "Auto", "", 42, None, "", [], [])

        # ── Wiring ──

        upload_btn.click(
            fn=do_upload,
            inputs=[ref_audio_input, ref_text_input, voice_name_input, voice_desc_input],
            outputs=[upload_status],
        )

        refresh_voices_btn.click(
            fn=do_refresh_voices,
            outputs=[voice_input],
        )

        submit_btn.click(
            fn=generate,
            inputs=[text_input, voice_input, ref_audio_input, ref_text_input,
                    use_inline_ref, language_input, instructions_input, seed_input,
                    history_state],
            outputs=[audio_output, status_output, history_container, history_state],
        )

        text_input.submit(
            fn=generate,
            inputs=[text_input, voice_input, ref_audio_input, ref_text_input,
                    use_inline_ref, language_input, instructions_input, seed_input,
                    history_state],
            outputs=[audio_output, status_output, history_container, history_state],
        )

        clear_btn.click(
            fn=clear_all,
            outputs=[text_input, ref_audio_input, ref_text_input, use_inline_ref,
                     language_input, instructions_input, seed_input,
                     audio_output, status_output, history_container, history_state],
        )

        new_prompt_btn.click(
            fn=lambda: random_reading_prompt(),
            outputs=[clone_prompt],
        )

        ref_audio_input.stop_recording(
            fn=lambda prompt: prompt,
            inputs=[clone_prompt],
            outputs=[ref_text_input],
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
