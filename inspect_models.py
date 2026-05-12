from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def _safe(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


def inspect_emotion_model(model_path: Path) -> None:
    model = joblib.load(model_path)
    print(f"\n=== {model_path.name} ===")
    print(f"Type: {type(model).__name__}")

    if isinstance(model, dict) and model.get("type") == "emotion_kmeans":
        print("Artifact kind: TF-IDF + KMeans clustering")
        print(f"Keys: {sorted(model.keys())}")
        labels = model.get("labels", [])
        print(f"Labels: {labels}")
        mapping = model.get("cluster_to_tone", {})
        print(f"Cluster to tone map: {mapping}")

        vectorizer = model.get("vectorizer")
        if vectorizer is not None:
            print(f"TF-IDF max_features: {_safe(getattr(vectorizer, 'max_features', None))}")
            print(f"TF-IDF ngram_range: {_safe(getattr(vectorizer, 'ngram_range', None))}")

        clusterer = model.get("clusterer")
        if clusterer is not None:
            print(f"Clusterer: {type(clusterer).__name__}")
            print(f"n_clusters: {_safe(getattr(clusterer, 'n_clusters', None))}")
        return

    classes = getattr(model, "classes_", None)
    if classes is not None:
        print(f"Classes: {list(classes)}")

    named_steps = getattr(model, "named_steps", None)
    if named_steps is not None:
        print(f"Pipeline steps: {list(named_steps.keys())}")
        tfidf = named_steps.get("tfidf")
        clf = named_steps.get("clf")
        if tfidf is not None:
            print(f"TF-IDF max_features: {_safe(getattr(tfidf, 'max_features', None))}")
            print(f"TF-IDF ngram_range: {_safe(getattr(tfidf, 'ngram_range', None))}")
        if clf is not None:
            print(f"Classifier: {type(clf).__name__}")


def inspect_response_model(model_path: Path) -> None:
    model = joblib.load(model_path)
    print(f"\n=== {model_path.name} ===")
    print(f"Type: {type(model).__name__}")

    if isinstance(model, dict):
        print(f"Keys: {sorted(model.keys())}")
        responses = model.get("responses", [])
        print(f"Responses in corpus: {len(responses)}")
        if responses:
            print(f"Sample response: {responses[0]}")

        vectorizer = model.get("vectorizer")
        if vectorizer is not None:
            print(f"Vectorizer: {type(vectorizer).__name__}")
            print(f"Vectorizer max_features: {_safe(getattr(vectorizer, 'max_features', None))}")

        matrix = model.get("matrix")
        if matrix is not None:
            print(f"Matrix shape: {getattr(matrix, 'shape', 'N/A')}")


def inspect_song_model(model_path: Path) -> None:
    model = joblib.load(model_path)
    print(f"\n=== {model_path.name} ===")
    print(f"Type: {type(model).__name__}")

    if isinstance(model, dict):
        print(f"Keys: {sorted(model.keys())}")

        songs = model.get("songs", [])
        print(f"Songs in corpus: {len(songs)}")
        if songs:
            sample = songs[0]
            print(f"Sample song: {sample.get('name', 'N/A')} - {sample.get('artist', 'N/A')} ({sample.get('tag', 'N/A')})")

        anchors = model.get("anchors", {})
        print(f"Mood anchors: {sorted(anchors.keys()) if isinstance(anchors, dict) else 'N/A'}")

        vectorizer = model.get("vectorizer")
        if vectorizer is not None:
            print(f"Vectorizer: {type(vectorizer).__name__}")

        matrix = model.get("matrix")
        if matrix is not None:
            print(f"Matrix shape: {getattr(matrix, 'shape', 'N/A')}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / "model"

    emotion_model_path = model_dir / "emotion_model.pkl"
    response_model_path = model_dir / "response_model.pkl"
    song_model_path = model_dir / "song_model.pkl"

    required = [emotion_model_path, response_model_path, song_model_path]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Missing model files:")
        for p in missing:
            print(f"- {p}")
        print("\nRun: python scripts/train_emotion_model.py")
        return

    inspect_emotion_model(emotion_model_path)
    inspect_response_model(response_model_path)
    inspect_song_model(song_model_path)


if __name__ == "__main__":
    main()
