#!/usr/bin/env python3
"""
html_to_scrolling_video.py
--------------------------
100% Autonomous HTML-to-Scrolling-Video Pipeline with Whisper Forced Alignment & Subtitles.

Features:
- Fixes audio repeats: ignores child elements whose parents were already extracted (e.g. p inside blockquote).
- Cleans technical characters (e.g. // to dashes) so Kokoro reads naturally without saying 'slash slash'.
- Generates precise word/sentence timestamps and an .srt subtitle track via local Whisper.
- Calibrates camera scroll directly against Whisper's speech timestamps (zero drift).
- Reads exact content bottom from DOM (footer/body) to prevent scrolling into blank dark space.
- Burns in elegant, high-contrast semi-transparent subtitles at the bottom.
- 100% turnkey CLI tool.

Usage:
    python3 html_to_scrolling_video.py <page.html> [--voice VOICE] [--output video.mp4]
"""

import os
import sys
import re
import json
import base64
import argparse
import subprocess
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
import whisper

KOKORO_BIN = "/home/jack/Desktop/Kokoro-TTS-Pause/venv/bin/kokoro-tts"
KOKORO_MODEL = "/home/jack/Desktop/Kokoro-TTS-Pause/kokoro-v1.0.onnx"
KOKORO_VOICES = "/home/jack/Desktop/Kokoro-TTS-Pause/voices-v1.0.bin"

def extract_story_text(html_path):
    """
    Extracts clean text paragraphs, titles, and quotes from HTML.
    Eliminates audio repeats by ignoring child elements whose ancestors were already extracted.
    Cleans up // to dashes so Kokoro doesn't say 'slash slash'.
    """
    print(f"[*] [1/5] Extracting story text from {html_path}...")
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_html = f.read()

    # Pre-clean LLM tag glitches
    clean_html = re.sub(r"</<+/?p>", "</p>", raw_html)
    clean_html = re.sub(r"</\s*(?=\n|\r)", "</p>", clean_html)

    soup = BeautifulSoup(clean_html, "html.parser")

    title = soup.find("h1")
    title_text = title.get_text(" ", strip=True) if title else ""
    subtitle = soup.find(class_=lambda c: c and "subtitle" in c) or (soup.header.find("p") if soup.header else None)
    sub_text = subtitle.get_text(" ", strip=True) if subtitle else ""

    # Elements to ignore from story body extraction to prevent repeats
    ignore_nodes = set()
    if title:
        ignore_nodes.add(title)
        for d in title.descendants:
            ignore_nodes.add(d)
    if subtitle:
        ignore_nodes.add(subtitle)
        for d in subtitle.descendants:
            ignore_nodes.add(d)
    if soup.header:
        for d in soup.header.descendants:
            ignore_nodes.add(d)
        ignore_nodes.add(soup.header)

    root = (
        soup.find("main")
        or soup.find("article")
        or soup.find(class_=lambda c: c and any(k in c for k in ["story-content", "story-card", "content", "container"]))
        or soup.body
        or soup
    )

    extracted_elements = []
    # Collect candidate tags in DOM order
    for elem in root.find_all(["blockquote", "p", "h2", "h3", "h4", "li", "div"]):
        if elem in ignore_nodes:
            continue
        # If any ancestor has already been extracted, skip this element! (e.g. p inside blockquote)
        if any(anc in extracted_elements for anc in elem.parents):
            continue

        if elem.name == "div":
            cls = " ".join(elem.get("class", []))
            # Leaf callout divs (quote, terminal output, AI dialogue)
            if any(k in cls for k in ["quote", "terminal-output", "ai-voice"]) and not elem.find(["div", "p", "li", "h2", "h3", "h4"]):
                extracted_elements.append(elem)
        elif elem.name in ["blockquote", "p", "h2", "h3", "h4", "li"]:
            extracted_elements.append(elem)

    paragraphs = []
    seen_texts = set()
    for el in extracted_elements:
        txt = el.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt or len(txt) <= 2:
            continue

        # Prevent exact or substring duplicates (e.g. quote block vs quote text)
        if txt in seen_texts:
            continue
        if paragraphs:
            prev = paragraphs[-1]
            if txt in prev or prev in txt:
                if len(txt) > len(prev):
                    paragraphs[-1] = txt
                continue

        seen_texts.add(txt)
        paragraphs.append(txt)

    # Format header text cleanly
    header_part = f"{title_text}.\n{sub_text}\n\n" if sub_text else f"{title_text}.\n\n"
    raw_full_text = header_part + "\n\n".join(paragraphs)

    # Clean technical characters: replace // or /// with — so Kokoro doesn't say "slash slash"
    clean_full_text = re.sub(r"\s*/{2,}\s*", " — ", raw_full_text)
    clean_full_text = re.sub(r"\s*//\s*", " — ", clean_full_text)

    words = len(clean_full_text.split())
    print(f"[+] Extracted {len(paragraphs)} clean blocks ({words:,} words, repeats eliminated).")
    return clean_full_text, paragraphs, words

def generate_tts_narration(text, output_wav, voice="am_michael"):
    """Generates continuous TTS narration using local Kokoro TTS."""
    print(f"[*] [2/5] Synthesizing speech with Kokoro TTS ({voice})...")
    temp_text = "temp_tts_input.txt"
    with open(temp_text, "w", encoding="utf-8") as f:
        f.write(text)

    lang = "en-gb" if voice.startswith("bf_") or voice.startswith("bm_") else "en-us"

    cmd = [
        KOKORO_BIN,
        temp_text,
        output_wav,
        "--voice", voice,
        "--lang", lang,
        "--format", "wav",
        "--model", KOKORO_MODEL,
        "--voices", KOKORO_VOICES,
    ]
    subprocess.run(cmd, check=True)

    ffprobe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        output_wav
    ]
    res = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
    duration = float(res.stdout.strip())
    print(f"[+] Audio generated: {duration:.2f} seconds ({duration/60:.2f} minutes).")

    if os.path.exists(temp_text):
        os.remove(temp_text)
    return duration

def generate_subtitles_and_timestamps(audio_wav, output_srt, model_size="tiny"):
    """Uses local Whisper to transcribe speech and generate an accurate .srt subtitle file with timestamps."""
    print(f"[*] [3/5] Transcribing audio with Whisper ({model_size}) for subtitles & alignment...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_wav, language="en")

    segments = result.get("segments", [])

    def format_timestamp(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        start_str = format_timestamp(seg["start"])
        end_str = format_timestamp(seg["end"])
        txt = seg["text"].strip()
        srt_lines.append(f"{i}\n{start_str} --> {end_str}\n{txt}\n")

    with open(output_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    print(f"[+] Generated subtitle file: {output_srt} ({len(segments)} timed segments).")
    return segments

def capture_page_and_layout(html_path, output_png, width=1280):
    """
    Sanitizes CSS, captures full-page screenshot, and extracts layout bounding boxes and content bottom via Base64.
    """
    print("[*] [4/5] Capturing page screenshot and measuring DOM coordinates...")
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    cleaned = re.sub(r"</<+/?p>", "</p>", html)
    cleaned = re.sub(r"</\s*(?=\n|\r)", "</p>", cleaned)
    cleaned = re.sub(r'min-height:\s*\d+vh;?', 'min-height: 550px;', cleaned)
    cleaned = re.sub(r'height:\s*100vh;?', 'height: 600px;', cleaned)

    # Injected script: measures all content elements and exact footer position, encoded via Base64
    script = """
    <script>
    window.addEventListener('DOMContentLoaded', () => {
        const targets = Array.from(document.querySelectorAll('h1, h2, h3, h4, p, blockquote, .tech-specs, .terminal, .chapter-title, footer'));
        const f = document.querySelector('footer');
        const contentBottom = f ? Math.round(f.getBoundingClientRect().top + window.scrollY + f.offsetHeight + 60) : Math.round(document.body.scrollHeight);
        const metrics = targets.map(s => {
            const rect = s.getBoundingClientRect();
            return {
                tag: s.tagName,
                top: Math.round(rect.top + window.scrollY),
                height: Math.round(s.offsetHeight),
                text: (s.innerText || '').trim().replace(/\\s+/g, ' ')
            };
        }).filter(m => m.height > 0 && m.text.length > 3);
        const payload = JSON.stringify({contentBottom, metrics});
        console.log('UNASSISTED_B64:' + btoa(unescape(encodeURIComponent(payload))));
    });
    </script>
    """
    if "</body>" in cleaned:
        injected_html = cleaned.replace("</body>", f"{script}</body>")
    else:
        injected_html = cleaned + script

    temp_render = "temp_auto_render.html"
    raw_png = "temp_auto_tall.png"
    with open(temp_render, "w", encoding="utf-8") as f:
        f.write(injected_html)

    cmd = [
        "google-chrome",
        "--headless=new",
        "--enable-logging=stderr",
        f"--screenshot={raw_png}",
        f"--window-size={width},9000",
        "--default-background-color=050a14",
        "--hide-scrollbars",
        f"file://{os.path.abspath(temp_render)}"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    metrics = []
    content_bottom = 0
    for line in proc.stderr.splitlines():
        if "UNASSISTED_B64:" in line:
            b64_str = line.split("UNASSISTED_B64:")[1].split('"')[0].strip()
            try:
                data = json.loads(base64.b64decode(b64_str).decode("utf-8"))
                metrics = data.get("metrics", [])
                content_bottom = data.get("contentBottom", 0)
                break
            except Exception as e:
                print(f"[!] Warning: failed to parse B64 metrics: {e}")

    im = Image.open(raw_png)
    if content_bottom > 720:
        crop_h = min(im.size[1], content_bottom)
        cropped = im.crop((0, 0, width, crop_h))
    else:
        # Fallback to visual pixel difference
        arr = np.array(im)
        bg_color = arr[0, 0, :3]
        diff = np.abs(arr[:, :, :3] - bg_color).sum(axis=2)
        content_rows = np.where((diff > 15).sum(axis=1) > 5)[0]
        if len(content_rows) > 0:
            last_row = min(im.size[1], content_rows[-1] + 120)
            cropped = im.crop((0, 0, width, last_row))
        else:
            cropped = im

    cropped.save(output_png)
    print(f"[+] Page captured: {cropped.size[0]}x{cropped.size[1]}px, measured {len(metrics)} DOM milestones (contentBottom: {content_bottom}px).")

    for f_temp in [temp_render, raw_png]:
        if os.path.exists(f_temp):
            os.remove(f_temp)

    return cropped.size[0], cropped.size[1], metrics

def build_whisper_synchronized_scroll_expr(dom_metrics, segments, total_duration, max_scroll):
    """
    Aligns DOM elements to Whisper speech segments using keyword matching.
    Builds a monotonic piecewise camera curve locked directly to the narrator's voice.
    """
    if max_scroll <= 0:
        return "0"

    if not dom_metrics or not segments:
        return f"if(lt(t,3), 0, if(gt(t,{total_duration-3:.2f}), {max_scroll}, {max_scroll} * (t - 3) / ({total_duration - 6:.2f})))"

    # Sequentially match DOM blocks to Whisper segments
    seg_idx = 0
    keyframes = [(0.0, 0)]  # Start at t=0, Y=0

    for block in dom_metrics:
        block_words = [w.lower() for w in re.findall(r"\w+", block["text"]) if len(w) > 2][:4]
        if not block_words:
            continue

        for si in range(seg_idx, len(segments)):
            s_text = segments[si]["text"].lower()
            matched_cnt = sum(1 for w in block_words if w in s_text)
            if matched_cnt >= 2 or (len(block_words) <= 2 and matched_cnt >= 1):
                t_speech = max(0.5, segments[si]["start"])
                # Center the block comfortably on screen
                target_y = min(max_scroll, max(0, block["top"] - 140))
                # Ensure monotonic downward scroll
                if target_y >= keyframes[-1][1]:
                    keyframes.append((t_speech, target_y))
                seg_idx = si
                break

    # Add final hold point at max_scroll
    keyframes.append((total_duration, max_scroll))

    # Remove duplicates
    deduped = [keyframes[0]]
    for t, y in keyframes[1:]:
        last_t, last_y = deduped[-1]
        if t > last_t + 0.5:
            deduped.append((t, max(last_y, y)))

    # Ensure ending hits max_scroll
    if deduped[-1][0] < total_duration:
        deduped.append((total_duration, max_scroll))
    else:
        deduped[-1] = (total_duration, max_scroll)

    # Build nested FFmpeg conditional expression
    expr = str(max_scroll)
    for i in range(len(deduped) - 2, -1, -1):
        t0, y0 = deduped[i]
        t1, y1 = deduped[i + 1]
        dt = max(0.1, t1 - t0)
        dy = y1 - y0
        seg_expr = f"{y0} + ({dy}) * (t - {t0:.2f}) / {dt:.2f}"
        expr = f"if(lt(t,{t1:.2f}), {seg_expr}, {expr})"

    print(f"[+] Synchronized camera curve calibrated with {len(deduped)} Whisper voice-locked keyframes.")
    return expr

def render_scrolling_video(screenshot_png, audio_wav, srt_file, output_mp4, duration, y_expr, win_w=1280, win_h=720):
    """Encodes the final synchronized MP4 with FFmpeg and burns in subtitles."""
    print(f"[*] [5/5] Encoding synchronized MP4 with burned-in Whisper subtitles...")

    safe_srt = os.path.basename(srt_file)
    sub_filter = f"subtitles={safe_srt}:force_style='FontSize=18,FontName=Liberation Sans,PrimaryColour=&H00FFFFFF,BackColour=&HA0000000,BorderStyle=4,MarginV=25'"

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", screenshot_png,
        "-i", audio_wav,
        "-filter_complex", f"[0:v]crop=w={win_w}:h={win_h}:x=0:y='{y_expr}'[v0];[v0]{sub_filter}[v]",
        "-map", "[v]",
        "-map", "1:a",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_mp4
    ]
    subprocess.run(cmd, check=True)
    print(f"\n[🎉] SUCCESS! Synchronized video generated: {output_mp4}")
    print(f"     Preview with: mpv {output_mp4} or vlc {output_mp4}\n")

def main():
    parser = argparse.ArgumentParser(description="Autonomous HTML-to-Scrolling-Video Pipeline with Whisper Subtitles & Alignment.")
    parser.add_argument("html_file", help="Path to input HTML page")
    parser.add_argument("--voice", default="am_michael", help="Kokoro voice (default: am_michael)")
    parser.add_argument("--output", default=None, help="Output MP4 filename")
    parser.add_argument("--whisper-model", default="tiny", help="Whisper model size (default: tiny)")
    args = parser.parse_args()

    if not os.path.exists(args.html_file):
        print(f"[-] HTML file not found: {args.html_file}")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(args.html_file))[0]
    out_mp4 = args.output or f"{base_name}_{args.voice}.mp4"
    temp_wav = f"temp_{base_name}_{args.voice}.wav"
    temp_png = f"temp_{base_name}_page.png"
    temp_srt = f"temp_{base_name}_{args.voice}.srt"

    try:
        # Step 1: Text extraction (no repeats, cleaned symbols)
        clean_text, paragraphs, words = extract_story_text(args.html_file)

        # Step 2: Speech synthesis
        duration = generate_tts_narration(clean_text, temp_wav, voice=args.voice)

        # Step 3: Whisper transcription and subtitles
        segments = generate_subtitles_and_timestamps(temp_wav, temp_srt, model_size=args.whisper_model)

        # Step 4: Layout measurement & page screenshot
        img_w, img_h, dom_metrics = capture_page_and_layout(args.html_file, temp_png, width=1280)

        max_scroll = max(0, img_h - 720)
        y_expr = build_whisper_synchronized_scroll_expr(dom_metrics, segments, duration, max_scroll)

        # Step 5: Video compilation with subtitles
        render_scrolling_video(temp_png, temp_wav, temp_srt, out_mp4, duration, y_expr, win_w=1280, win_h=720)

    finally:
        for f in [temp_wav, temp_png]:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    main()
