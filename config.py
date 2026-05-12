import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret")
    SONG_RECOMMENDATION_LIMIT = int(os.getenv("SONG_RECOMMENDATION_LIMIT", "10"))
    LOCAL_SONGS_DATASET_PATH = os.getenv("LOCAL_SONGS_DATASET_PATH", "data/local_songs.csv")

    EMOTION_MODEL_ENABLED = os.getenv("EMOTION_MODEL_ENABLED", "1") == "1"
    EMOTION_MODEL_PATH = os.getenv("EMOTION_MODEL_PATH", "model/emotion_model.pkl")
    EMOTION_MODEL_MIN_CONFIDENCE = float(os.getenv("EMOTION_MODEL_MIN_CONFIDENCE", "0.50"))

    RESPONSE_MODEL_ENABLED = os.getenv("RESPONSE_MODEL_ENABLED", "1") == "1"
    RESPONSE_MODEL_PATH = os.getenv("RESPONSE_MODEL_PATH", "model/response_model.pkl")

    SONG_MODEL_ENABLED = os.getenv("SONG_MODEL_ENABLED", "1") == "1"
    SONG_MODEL_PATH = os.getenv("SONG_MODEL_PATH", "model/song_model.pkl")
