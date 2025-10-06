"""
dagen_video_reels.py
====================

This module provides a framework for converting online news articles into
vertical video reels with Norwegian audio, branded colours and fonts for
Dagen.  It contains functions for fetching article content, dividing
the article into scenes, generating speech for each scene, producing
subtitles, and assembling the final video using MoviePy.

The script is designed to be run from a Python environment with
third‑party dependencies installed.  At a minimum you will need:

* requests
* beautifulsoup4
* moviepy==1.0.3
* pillow
* pysrt
* numpy
* Any Text‑to‑Speech (TTS) library or API you prefer (see notes below)

Example usage:

```bash
python dagen_video_reels.py \
    --url https://www.dagen.no/nyheter/2025/08/25/viktig-nyhet/ \
    --max-scenes 5 \
    --output-file output.mp4
```

The script will fetch the article, summarise it into scenes, call your
configured TTS provider to synthesise speech for each scene, generate
subtitles, assemble the vertical video (1080×1920) with Dagen’s
branding, and save it to the specified output file.  See the README
string at the bottom of this file for further configuration options.

Author: OpenAI ChatGPT agent
Date: 2025‑09‑26
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None  # type: ignore
    BeautifulSoup = None  # type: ignore

# These imports require external dependencies (moviepy, pillow, numpy, pysrt).
try:
    from moviepy.editor import (
        ColorClip,
        CompositeVideoClip,
        CompositeAudioClip,
        TextClip,
        AudioFileClip,
        ImageClip,
    )
except Exception:
    # moviepy is unavailable in the environment; the script will still load,
    # but video creation will fail at runtime.  Install moviepy to enable
    # functionality: pip install moviepy==1.0.3
    ColorClip = CompositeVideoClip = CompositeAudioClip = TextClip = None  # type: ignore
    AudioFileClip = ImageClip = None  # type: ignore

try:
    import pysrt
except ImportError:
    pysrt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None  # type: ignore

import numpy as np


@dataclass
class BrandConfig:
    """Configuration for Dagen branding.

    Colours are taken from the Dagen brand palette obtained via
    Brandfetch (2025):

    * Bay of Many (dark):   #204977
    * Rajah (light):        #f6a37b
    * Thunderbird (accent): #c4151c

    Fonts are taken from Google Fonts as specified on Brandfetch:
    Merriweather (body) and Schibsted Grotesk (title).  You will need
    these fonts installed locally or specify paths via the environment.
    """

    dark_colour: str = "#204977"
    light_colour: str = "#f6a37b"
    accent_colour: str = "#c4151c"
    body_font: str = "Merriweather"
    title_font: str = "Schibsted Grotesk"
    logo_path: Optional[str] = None  # path to Dagen logo PNG if available


@dataclass
class Scene:
    """Represents a single scene in the video."""

    text: str
    audio_path: str
    duration: float
    image_path: Optional[str] = None


def fetch_article_text(url: str) -> str:
    """Fetch article text from a web page.

    This function attempts to download the HTML and extract paragraphs.
    It uses BeautifulSoup if available; otherwise it raises an error.

    Args:
        url: Link to the article on dagen.no or another site.

    Returns:
        The concatenated textual content of the article.
    """
    if requests is None or BeautifulSoup is None:
        raise RuntimeError(
            "The requests and beautifulsoup4 packages must be installed to fetch articles."
        )

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract headlines and paragraphs.  Dagen articles often use <h1> for
    # title and <p> for paragraphs.  Adjust selectors as needed.
    parts: List[str] = []
    title_tag = soup.find("h1")
    if title_tag:
        parts.append(title_tag.get_text(strip=True))
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if text:
            parts.append(text)
    return "\n".join(parts)


def summarise_text(text: str, max_scenes: int = 5) -> List[str]:
    """Naïvely summarise text into a given number of scenes.

    This function splits the text into sentences and groups them into roughly
    equal chunks.  For a more sophisticated summary, integrate a library
    such as transformers or nltk summarisation.

    Args:
        text: Full article text.
        max_scenes: Desired number of scenes.

    Returns:
        A list of scene texts.
    """
    # Split on period, exclamation and question marks
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return [text]
    n = max(1, min(max_scenes, len(sentences)))
    chunk_size = int(np.ceil(len(sentences) / n))
    scenes = []
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i : i + chunk_size]
        scenes.append(" ".join(chunk))
    return scenes


def generate_silent_audio(text: str, idx: int, output_dir: str) -> Tuple[str, float]:
    """Create a silent audio file as a placeholder.

    Replace this function with a call to your preferred TTS provider.  The
    duration is estimated based on word count (roughly 3 words per second).

    Args:
        text: Scene text.
        idx: Scene index used to name the file.
        output_dir: Directory for audio files.

    Returns:
        Tuple of audio file path and duration in seconds.
    """
    words = text.split()
    duration = max(3.0, len(words) / 3.0)
    import wave
    import struct

    filepath = os.path.join(output_dir, f"scene_{idx:02d}.wav")
    sample_rate = 44100
    n_frames = int(duration * sample_rate)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Write silence (all zeroes)
        silence = struct.pack("<h", 0)
        for _ in range(n_frames):
            wf.writeframes(silence)
    return filepath, duration


def generate_srt(scenes: List[Scene], srt_path: str) -> None:
    """Generate an SRT subtitle file from scene timings.

    Args:
        scenes: List of scenes with durations.
        srt_path: Output path for the .srt file.
    """
    if pysrt is None:
        raise RuntimeError("pysrt is required to generate subtitle files.")
    subs = pysrt.SubRipFile()
    t0 = 0.0
    for idx, scene in enumerate(scenes, 1):
        start = t0
        end = t0 + scene.duration
        # Wrap lines at ~40 characters for readability
        import textwrap
        lines = textwrap.wrap(scene.text, width=40)
        subs.append(
            pysrt.SubRipItem(
                index=idx,
                start=pysrt.SubRipTime(seconds=start),
                end=pysrt.SubRipTime(seconds=end),
                text="\n".join(lines),
            )
        )
        t0 = end
    subs.save(srt_path, encoding="utf-8")


def load_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font.  Falls back to default if not available."""
    if ImageFont is None:
        raise RuntimeError("Pillow must be installed to render text onto images.")
    try:
        return ImageFont.truetype(font_name, size)
    except IOError:
        # Fallback to a basic PIL font if the custom font cannot be found.
        return ImageFont.load_default()


def create_video(
    scenes: List[Scene],
    brand: BrandConfig,
    output_path: str,
    fps: int = 30,
) -> None:
    """Assemble the final vertical video from scene data.

    This function uses MoviePy to stack background colour clips, overlay
    scene text, and composite the audio tracks.  If a Dagen logo path
    is provided in the brand configuration, it will be placed in the
    bottom right corner.

    Args:
        scenes: List of Scene objects with audio and durations.
        brand: Brand configuration for colours and fonts.
        output_path: Path where the final MP4 will be saved.
        fps: Frames per second for the output video.
    """
    if ColorClip is None:
        raise RuntimeError(
            "moviepy is not installed. Install moviepy to enable video generation."
        )
    video_clips = []
    audio_clips = []
    width, height = 1080, 1920

    # Preload logo if provided
    logo_clip: Optional[ImageClip] = None
    if brand.logo_path and os.path.exists(brand.logo_path):
        logo_clip = ImageClip(brand.logo_path).set_duration(0)
        # Resize logo to 10% of video width
        w = width * 0.2
        h = logo_clip.h * (w / logo_clip.w)
        logo_clip = logo_clip.resize(newsize=(int(w), int(h)))

    # Preload fonts if Pillow is available
    body_font = load_font(brand.body_font, 48)
    title_font = load_font(brand.title_font, 64)

    cumulative_time = 0.0
    for scene in scenes:
        # Create base colour clip
        bg = ColorClip(size=(width, height), color=hex_to_rgb(brand.light_colour)).set_duration(scene.duration)

        # Create an image with text overlay using Pillow
        if Image is None:
            raise RuntimeError("Pillow is required for drawing text overlays.")
        img = Image.new("RGB", (width, height), color=brand.light_colour)
        draw = ImageDraw.Draw(img)
        # Draw headline (first sentence) in accent colour
        first_sentence = scene.text.split(".")[0]
        draw.text((50, 200), first_sentence, font=title_font, fill=brand.accent_colour)
        # Draw remaining text in dark colour below
        remaining = scene.text[len(first_sentence) + 1 :].strip()
        y_offset = 300
        for line in wrap_text(remaining, body_font, width - 100):
            draw.text((50, y_offset), line, font=body_font, fill=brand.dark_colour)
            y_offset += body_font.getsize(line)[1] + 8
        # Convert to numpy array for MoviePy
        np_img = np.array(img)
        txt_clip = ImageClip(np_img).set_duration(scene.duration)
        # If there's a logo, overlay it at bottom right
        if logo_clip:
            pos = (
                width - logo_clip.w - 20,
                height - logo_clip.h - 20,
            )
            txt_clip = CompositeVideoClip([txt_clip, logo_clip.set_start(0).set_pos(pos)], size=(width, height)).set_duration(scene.duration)
        video_clips.append(txt_clip)
        # Load audio
        audio_clips.append(AudioFileClip(scene.audio_path))
        cumulative_time += scene.duration

    # Concatenate video and audio
    final_video = CompositeVideoClip(video_clips, size=(width, height)).set_audio(CompositeAudioClip(audio_clips))
    final_video.write_videofile(output_path, fps=fps, codec="libx264", audio_codec="aac")


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Wrap text to fit within a given pixel width."""
    if ImageDraw is None:
        return [text]
    words = text.split()
    lines: List[str] = []
    current_line: List[str] = []
    for word in words:
        test_line = " ".join(current_line + [word])
        w, _ = font.getsize(test_line)
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def hex_to_rgb(hexstr: str) -> Tuple[int, int, int]:
    """Convert hex colour (e.g., '#c4151c') to an RGB tuple."""
    hexstr = hexstr.lstrip("#")
    return tuple(int(hexstr[i : i + 2], 16) for i in (0, 2, 4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a news article to a vertical Dagen‑branded video reel.")
    parser.add_argument("--url", type=str, required=True, help="URL of the article to process")
    parser.add_argument("--max-scenes", type=int, default=5, help="Maximum number of scenes to generate")
    parser.add_argument("--output-file", type=str, default="dagen_reel.mp4", help="Output video filename")
    parser.add_argument("--audio-dir", type=str, default="audio", help="Directory for storing generated audio")
    parser.add_argument("--logo", type=str, default=None, help="Path to Dagen logo (PNG)")
    args = parser.parse_args()

    os.makedirs(args.audio_dir, exist_ok=True)
    brand = BrandConfig(logo_path=args.logo)

    # Fetch and summarise article
    text = fetch_article_text(args.url)
    scene_texts = summarise_text(text, max_scenes=args.max_scenes)

    scenes: List[Scene] = []
    for idx, scene_text in enumerate(scene_texts, 1):
        audio_path, duration = generate_silent_audio(scene_text, idx, args.audio_dir)
        scenes.append(Scene(text=scene_text, audio_path=audio_path, duration=duration))

    # Generate subtitle file
    srt_path = os.path.splitext(args.output_file)[0] + ".srt"
    generate_srt(scenes, srt_path)

    # Assemble video
    create_video(scenes, brand, args.output_file)

    print(f"Video saved to {args.output_file} and subtitles saved to {srt_path}")


if __name__ == "__main__":
    main()
