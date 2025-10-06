from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from .dagen_video_reels import fetch_article_text, summarise_text

app = FastAPI(title="Dagen Reels API", description="Convert Dagen articles into video reels")

class ConvertResponse(BaseModel):
    url: str
    scenes: list[str]

@app.get('/convert', response_model=ConvertResponse)
async def convert(url: str):
    """
    Accept a Dagen article URL, summarise it into scenes, and return the scene texts.
    This stub does not yet call the full video pipeline; it demonstrates the API shape.
    """
    try:
        article = fetch_article_text(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch article: {e}")
    scenes = summarise_text(article, max_scenes=5)
    return ConvertResponse(url=url, scenes=scenes)
