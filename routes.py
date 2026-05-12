from __future__ import annotations

import re
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from typing import Any

from flask import Flask, jsonify, render_template, request

from .constants import EMOTION_MAP, SONG_TAG_MAP
from .services.cakechat_service import CakeChatService
from .services.lastfm_service import LastFMService
from .services.tone_service import ToneService


def _latest_text(context: list[Any]) -> str:
    # Mood should reflect what the user most recently said.
    for item in reversed(context):
        text = str(item).strip()
        if text:
            return text
    return ""


def _youtube_video_ids(query: str, limit: int = 8) -> list[str]:
    cleaned = str(query or "").strip()
    if not cleaned:
        return []

    search_url = f"https://www.youtube.com/results?search_query={quote_plus(cleaned)}"
    request = Request(
        search_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        },
    )

    try:
        with urlopen(request, timeout=8) as response:
            payload = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    matches = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', payload)
    unique: list[str] = []
    seen = set()
    for video_id in matches:
        if video_id in seen:
            continue
        seen.add(video_id)
        unique.append(video_id)
        if len(unique) >= limit:
            break

    return unique


def _is_embeddable_video(video_id: str) -> bool:
    value = str(video_id or "").strip()
    if not value:
        return False

    oembed_url = (
        "https://www.youtube.com/oembed"
        f"?url=https://www.youtube.com/watch?v={quote_plus(value)}&format=json"
    )
    request = Request(
        oembed_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        },
    )

    try:
        with urlopen(request, timeout=6):
            return True
    except Exception:
        return False


def register_routes(app: Flask) -> None:
    @app.get("/")
    def home() -> str:
        return render_template("index.html")

    @app.post("/tone")
    def tone() -> tuple[Any, int]:
        payload = request.get_json(silent=True) or {}
        context = payload.get("context", [])
        if not isinstance(context, list):
            return jsonify({"error": "context must be a list of strings"}), 400

        tone_result = ToneService.analyze_text(_latest_text(context))

        return jsonify(
            {
                "tone": tone_result.top_tone,
                "tone_for_chatbot": EMOTION_MAP.get(tone_result.top_tone, "neutral"),
                "tone_analysis": tone_result.raw,
            }
        ), 200

    @app.post("/response")
    def response() -> tuple[Any, int]:
        payload = request.get_json(silent=True) or {}
        context = payload.get("context", [])
        tone = payload.get("tone", "neutral")

        if not isinstance(context, list) or not context:
            return jsonify({"error": "context must be a non-empty list of strings"}), 400

        mapped_tone = EMOTION_MAP.get(str(tone).lower(), "neutral")
        reply = CakeChatService.get_response([str(item) for item in context], mapped_tone)

        return jsonify({"emotion": mapped_tone, "response": reply}), 200

    @app.get("/songs")
    def songs() -> tuple[Any, int]:
        tag = request.args.get("tag", "neutral").lower()
        mapped_tag = SONG_TAG_MAP.get(tag, tag)
        limit = app.config.get("SONG_RECOMMENDATION_LIMIT", 10)

        try:
            tracks = LastFMService.top_songs_by_tag(mapped_tag, limit=limit)
        except Exception as exc:
            return jsonify({"error": f"failed to fetch songs: {exc}"}), 500

        return jsonify({"tag": mapped_tag, "songs": tracks}), 200

    @app.get("/similarsongs")
    def similar_songs() -> tuple[Any, int]:
        track = request.args.get("track", "")
        artist = request.args.get("artist", "")
        limit = app.config.get("SONG_RECOMMENDATION_LIMIT", 10)
        if not track or not artist:
            return jsonify({"error": "track and artist query params are required"}), 400

        try:
            tracks = LastFMService.similar_songs(track, artist, limit=limit)
        except Exception as exc:
            return jsonify({"error": f"failed to fetch similar songs: {exc}"}), 500

        return jsonify({"track": track, "artist": artist, "songs": tracks}), 200

    @app.get("/youtube/search")
    def youtube_search() -> tuple[Any, int]:
        query = request.args.get("query", "").strip()
        if not query:
            return jsonify({"error": "query is required"}), 400

        ids = _youtube_video_ids(query, limit=16)
        playable: list[str] = []
        for video_id in ids:
            if _is_embeddable_video(video_id):
                playable.append(video_id)
            if len(playable) >= 8:
                break

        return jsonify({"query": query, "video_ids": playable}), 200

    @app.post("/chat")
    def chat() -> tuple[Any, int]:
        payload = request.get_json(silent=True) or {}
        context = payload.get("context", [])
        limit = app.config.get("SONG_RECOMMENDATION_LIMIT", 10)
        if not isinstance(context, list) or not context:
            return jsonify({"error": "context must be a non-empty list of strings"}), 400

        tone_result = ToneService.analyze_text(_latest_text(context))
        mapped_tone = EMOTION_MAP.get(tone_result.top_tone, "neutral")

        bot_reply = CakeChatService.get_response([str(item) for item in context], mapped_tone)
        songs = []
        try:
            songs = LastFMService.top_songs_by_tag(SONG_TAG_MAP.get(mapped_tone, mapped_tone), limit=limit)
        except Exception:
            songs = []

        return jsonify(
            {
                "tone": tone_result.top_tone,
                "tone_for_chatbot": mapped_tone,
                "response": bot_reply,
                "songs": songs,
            }
        ), 200
