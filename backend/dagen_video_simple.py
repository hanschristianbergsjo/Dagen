from dataclasses import dataclass
from typing import List, Tuple
import os
import requests
from bs4 import BeautifulSoup
from moviepy.editor import ColorClip, CompositeVideoClip, CompositeAudioClip, TextClip, AudioFileClip, AudioClip, concatenate_videoclips
import pysrt

@dataclass
class Scene:
    text: str
    audio_path: str
    duration: float

@dataclass
class BrandConfig:
    primary_color: str = '#005bb7'
    secondary_color: str = '#f48c2a'
    tertiary_color: str = '#7f001b'
    font_body: str = 'Merriweather'
    font_title: str = 'Merriweather'

def fetch_article_text(url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, 'html.parser')
    paragraphs = soup.find_all('p')
    return ' '.join(p.get_text().strip() for p in paragraphs)

def summarise_text(text: str, max_scenes: int = 5) -> List[str]:
    words = text.split()
    n = len(words)
    if n == 0:
        return []
    size = max(1, n // max_scenes)
    scenes = [' '.join(words[i:i+size]) for i in range(0, n, size)]
    return scenes[:max_scenes]

def generate_audio_elevenlabs(text: str, idx: int, output_dir: str) -> Tuple[str, float]:
    except Exception:
    duration = max(3.0, len(text.split()) / 3.0)
    silence = AudioClip(lambda t: 0 * t, duration=duration)
    silence.write_audiofile(filepath, fps=44100)
    # return the fallback path and duration
    return filepath, duration

    os.makedirs(output_dir, exist_ok=True)
    filename = f'scene_{idx}.mp3'
    filepath = os.path.join(output_dir, filename)
    api_key = os.environ.get('ELEVENLABS_API_KEY')
    voice_id = os.environ.get('ELEVENLABS_VOICE_ID', '21m00Tcm4TlvDq8ikWAM')
    model_id = os.environ.get('ELEVENLABS_MODEL_ID', 'eleven_monolingual_v1')
    if api_key:
        url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}'
        headers = {
            'xi-api-key': api_key,
            'Content-Type': 'application/json',
            'Accept': 'audio/mpeg'
        }
        payload = {'text': text, 'model_id': model_id}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(resp.content)
            clip = AudioFileClip(filepath)
            duration = float(clip.duration)
            clip.close()
            return filepath, duration
        except Exception:
            pass
    duration = max(3.0, len(text.split()) / 3.0)
    silence = AudioClip(lambda t: 0 * t, duration=duration)
    silence.write_audiofile(filepath, fps=44100)
   
  

def generate_srt(scenes: List[Scene], srt_path: str) -> None:
    subs = pysrt.SubRipFile()
    start = 0.0
    for i, scene in enumerate(scenes, 1):
        end = start + scene.duration
        subs.append(pysrt.SubRipItem(
            index=i,
            start=pysrt.SubRipTime(seconds=start),
            end=pysrt.SubRipTime(seconds=end),
            text=scene.text
        ))
        start = end
    subs.save(srt_path, encoding='utf-8')


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_video(scenes: List[Scene], brand: BrandConfig, output_path: str) -> None:
    clips = []
    audio_clips = []
    width, height = 1080, 1920
    for scene in scenes:
        bg = ColorClip(size=(width, height), color=hex_to_rgb(brand.primary_color)).set_duration(scene.duration)
        txt = TextClip(scene.text, fontsize=40, font=brand.font_body, color='white', size=(int(width*0.8), int(height*0.8)), method='caption')
        txt = txt.set_position(('center','center')).set_duration(scene.duration)
        clip = CompositeVideoClip([bg, txt])
        clips.append(clip)
        audio_clips.append(AudioFileClip(scene.audio_path))
    video = concatenate_videoclips(clips)
    audio = CompositeAudioClip(audio_clips)
    final = video.set_audio(audio)
    final.write_videofile(output_path, fps=30, codec='libx264', audio_codec='aac')
  # #   return filepath, duration
