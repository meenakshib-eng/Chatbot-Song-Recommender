from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report


ALLOWED_LABELS = {"joy", "sadness", "anger", "neutral"}

RAW_TO_NORMALIZED = {
    "joy": "joy",
    "happiness": "joy",
    "love": "joy",
    "sadness": "sadness",
    "sad": "sadness",
    "fear": "sadness",
    "anger": "anger",
    "angry": "anger",
    "neutral": "neutral",
    "surprise": "neutral",
    "disgust": "neutral",
    "analytical": "neutral",
    "confident": "neutral",
    "tentative": "neutral",
}


DEFAULT_RESPONSE_CORPUS = [
    {"emotion": "joy", "prompt": "I am feeling great today", "response": "That is amazing. Tell me one highlight from today."},
    {"emotion": "joy", "prompt": "I had a wonderful day", "response": "Love that energy. Want upbeat tracks to keep it going?"},
    {"emotion": "joy", "prompt": "I am excited and happy", "response": "That sounds awesome. What made you smile the most?"},
    {"emotion": "joy", "prompt": "Everything is going well", "response": "Great momentum. Let us ride this positive mood."},
    {"emotion": "sadness", "prompt": "I feel sad and low", "response": "I hear you. We can take this slowly together."},
    {"emotion": "sadness", "prompt": "Today was hard and I feel down", "response": "That sounds heavy. What feels most difficult right now?"},
    {"emotion": "sadness", "prompt": "I am exhausted emotionally", "response": "Thanks for sharing that. I am here to listen."},
    {"emotion": "sadness", "prompt": "I feel lonely tonight", "response": "You are not alone here. Want to talk through it step by step?"},
    {"emotion": "anger", "prompt": "I am really angry right now", "response": "That frustration makes sense. Want to vent first or solve it?"},
    {"emotion": "anger", "prompt": "This situation is so annoying", "response": "I can feel that intensity. What triggered it most?"},
    {"emotion": "anger", "prompt": "I am upset about what happened", "response": "That sounds rough. Let us channel that energy into a next step."},
    {"emotion": "anger", "prompt": "Everything is making me mad", "response": "I hear you. What is one thing we can control right now?"},
    {"emotion": "neutral", "prompt": "How are you", "response": "I am here with you. How is your day going so far?"},
    {"emotion": "neutral", "prompt": "I want music recommendations", "response": "Absolutely. Tell me your mood and I will tune recommendations."},
    {"emotion": "neutral", "prompt": "I do not know how I feel", "response": "That is okay. Want to describe what your day looked like?"},
    {"emotion": "neutral", "prompt": "Can you help me", "response": "Yes. Share a little context and we will work through it together."},
]


def normalize_label(label: str) -> str:
    value = str(label).strip().lower()
    mapped = RAW_TO_NORMALIZED.get(value, value)
    if mapped in ALLOWED_LABELS:
        return mapped
    return "neutral"


def load_split(csv_path: Path) -> tuple[pd.Series, pd.Series]:
    frame = pd.read_csv(csv_path)

    if "text" not in frame.columns or "label" not in frame.columns:
        raise ValueError(f"{csv_path} must contain text and label columns")

    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].map(normalize_label)

    return frame["text"], frame["label"]


def build_emotion_cluster_artifact(x_train: pd.Series, y_train: pd.Series) -> dict[str, Any]:
    return build_emotion_cluster_artifact_with_params(
        x_train,
        y_train,
        {
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 1.0,
            "max_features": 20000,
            "sublinear_tf": False,
            "n_init": 20,
        },
    )


def build_emotion_cluster_artifact_with_params(
    x_train: pd.Series,
    y_train: pd.Series,
    params: dict[str, Any],
) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=params.get("ngram_range", (1, 2)),
        min_df=params.get("min_df", 1),
        max_df=params.get("max_df", 1.0),
        max_features=params.get("max_features", 20000),
        sublinear_tf=bool(params.get("sublinear_tf", False)),
    )
    train_matrix = vectorizer.fit_transform(x_train)

    labels = sorted(ALLOWED_LABELS)
    init_strategy = str(params.get("init_strategy", "k-means++"))
    init_matrix: np.ndarray | None = None
    if init_strategy == "label_centroids":
        centroids = []
        train_labels = y_train.values
        for label in labels:
            idx = np.where(train_labels == label)[0]
            if len(idx) == 0:
                centroids = []
                break
            centroids.append(np.asarray(train_matrix[idx].mean(axis=0)).ravel())
        if centroids and len(centroids) == len(labels):
            init_matrix = np.vstack(centroids)

    if init_matrix is not None:
        clusterer = KMeans(
            n_clusters=len(ALLOWED_LABELS),
            random_state=42,
            init=init_matrix,
            n_init=1,
        )
    else:
        clusterer = KMeans(
            n_clusters=len(ALLOWED_LABELS),
            random_state=42,
            n_init=int(params.get("n_init", 20)),
        )
    cluster_ids = clusterer.fit_predict(train_matrix)

    train_frame = pd.DataFrame({"label": y_train.values, "cluster": cluster_ids})
    cluster_to_tone: dict[int, str] = {}
    for cluster_id, group in train_frame.groupby("cluster"):
        cluster_to_tone[int(cluster_id)] = str(group["label"].value_counts().idxmax())

    return {
        "type": "emotion_kmeans",
        "vectorizer": vectorizer,
        "clusterer": clusterer,
        "cluster_to_tone": cluster_to_tone,
        "labels": sorted(ALLOWED_LABELS),
        "tuning": dict(params),
    }


def predict_emotions(artifact: dict[str, Any], x: pd.Series) -> list[str]:
    vectorizer = artifact["vectorizer"]
    clusterer = artifact["clusterer"]
    cluster_to_tone = artifact["cluster_to_tone"]

    matrix = vectorizer.transform(x)
    cluster_ids = clusterer.predict(matrix)
    return [cluster_to_tone.get(int(cluster_id), "neutral") for cluster_id in cluster_ids]


def evaluate(name: str, artifact: dict[str, Any], x: pd.Series, y: pd.Series) -> None:
    predictions = predict_emotions(artifact, x)
    accuracy = accuracy_score(y, predictions)
    print(f"\n{name} accuracy: {accuracy:.4f}")
    print(classification_report(y, predictions, digits=4, zero_division=0))


def score_accuracy(artifact: dict[str, Any], x: pd.Series, y: pd.Series) -> float:
    predictions = predict_emotions(artifact, x)
    return float(accuracy_score(y, predictions))


def tune_emotion_cluster_artifact(
    x_train: pd.Series,
    y_train: pd.Series,
    x_val: pd.Series,
    y_val: pd.Series,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float, float]:
    baseline_params = {
        "ngram_range": (1, 2),
        "min_df": 1,
        "max_df": 1.0,
        "max_features": 20000,
        "sublinear_tf": False,
        "n_init": 20,
        "init_strategy": "k-means++",
    }

    search_space = [
        baseline_params,
        {"ngram_range": (1, 2), "min_df": 1, "max_df": 1.0, "max_features": 20000, "sublinear_tf": False, "n_init": 20, "init_strategy": "label_centroids"},
        {"ngram_range": (1, 2), "min_df": 1, "max_df": 0.98, "max_features": 12000, "sublinear_tf": True, "n_init": 30},
        {"ngram_range": (1, 2), "min_df": 1, "max_df": 0.95, "max_features": 8000, "sublinear_tf": True, "n_init": 40},
        {"ngram_range": (1, 1), "min_df": 1, "max_df": 1.0, "max_features": 6000, "sublinear_tf": True, "n_init": 30},
        {"ngram_range": (1, 3), "min_df": 1, "max_df": 0.98, "max_features": 16000, "sublinear_tf": True, "n_init": 30},
        {"ngram_range": (1, 2), "min_df": 2, "max_df": 1.0, "max_features": 10000, "sublinear_tf": False, "n_init": 30},
        {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.98, "max_features": 12000, "sublinear_tf": True, "n_init": 50},
        {"ngram_range": (1, 1), "min_df": 1, "max_df": 0.95, "max_features": 4000, "sublinear_tf": False, "n_init": 20},
    ]

    baseline_artifact = build_emotion_cluster_artifact_with_params(x_train, y_train, baseline_params)
    baseline_val_acc = score_accuracy(baseline_artifact, x_val, y_val)

    best_artifact = baseline_artifact
    best_params = dict(baseline_params)
    best_val_acc = baseline_val_acc

    for params in search_space[1:]:
        artifact = build_emotion_cluster_artifact_with_params(x_train, y_train, params)
        val_acc = score_accuracy(artifact, x_val, y_val)
        if val_acc > best_val_acc:
            best_artifact = artifact
            best_params = dict(params)
            best_val_acc = val_acc

    return best_artifact, best_params, baseline_artifact, baseline_val_acc, best_val_acc


def train_emotion_model(project_root: Path) -> Path:
    data_dir = project_root / "data"
    model_dir = project_root / "model"

    train_csv = data_dir / "emotion_train.csv"
    val_csv = data_dir / "emotion_val.csv"
    test_csv = data_dir / "emotion_test.csv"

    missing = [p for p in [train_csv, val_csv, test_csv] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing}")

    x_train, y_train = load_split(train_csv)
    x_val, y_val = load_split(val_csv)
    x_test, y_test = load_split(test_csv)

    artifact, best_params, baseline_artifact, baseline_val_acc, tuned_val_acc = tune_emotion_cluster_artifact(
        x_train,
        y_train,
        x_val,
        y_val,
    )

    baseline_test_acc = score_accuracy(baseline_artifact, x_test, y_test)
    tuned_test_acc = score_accuracy(artifact, x_test, y_test)

    print("\nEmotion clustering tuning summary:")
    print(f"- Baseline validation accuracy: {baseline_val_acc:.4f}")
    print(f"- Tuned validation accuracy: {tuned_val_acc:.4f}")
    print(f"- Baseline test accuracy: {baseline_test_acc:.4f}")
    print(f"- Tuned test accuracy: {tuned_test_acc:.4f}")
    print(f"- Selected params: {best_params}")

    evaluate("Validation", artifact, x_val, y_val)
    evaluate("Test", artifact, x_test, y_test)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "emotion_model.pkl"
    joblib.dump(artifact, model_path)
    print(f"\nSaved emotion model to: {model_path}")
    return model_path


def train_response_model(project_root: Path) -> Path:
    docs = []
    responses = []

    for row in DEFAULT_RESPONSE_CORPUS:
        emotion = normalize_label(row.get("emotion", "neutral"))
        prompt = str(row.get("prompt", "")).strip()
        response = str(row.get("response", "")).strip()
        if not prompt or not response:
            continue
        docs.append(f"{prompt} emotion:{emotion}")
        responses.append(response)

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=12000)
    matrix = vectorizer.fit_transform(docs)

    artifact = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "responses": responses,
    }

    model_dir = project_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "response_model.pkl"
    joblib.dump(artifact, model_path)
    print(f"Saved response model to: {model_path}")
    return model_path


def train_song_model(project_root: Path) -> Path:
    songs_path = project_root / "data" / "local_songs.csv"
    if not songs_path.exists():
        raise FileNotFoundError(f"Song dataset not found: {songs_path}")

    rows: list[dict[str, str]] = []
    with songs_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = str(row.get("name", "")).strip()
            artist = str(row.get("artist", "")).strip()
            tag = str(row.get("tag", "")).strip().lower()
            if not name or not artist or not tag:
                continue

            rows.append(
                {
                    "name": name,
                    "artist": artist,
                    "tag": tag,
                    "mbid": str(row.get("mbid", "")).strip(),
                    "url": str(row.get("url", "")).strip(),
                }
            )

    docs = [f"{row['name']} {row['artist']} mood:{row['tag']}" for row in rows]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=18000)
    matrix = vectorizer.fit_transform(docs)

    anchors = {
        "pop": "happy energetic dance bright uplifting mood:pop",
        "rock": "angry intense loud strong energetic mood:rock",
        "acoustic": "sad soft calm intimate emotional mood:acoustic",
        "chill": "neutral mellow ambient calm focus relax mood:chill",
    }

    artifact = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "songs": rows,
        "anchors": anchors,
    }

    model_dir = project_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "song_model.pkl"
    joblib.dump(artifact, model_path)
    print(f"Saved song model to: {model_path}")
    return model_path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    emotion_path = train_emotion_model(project_root)
    response_path = train_response_model(project_root)
    song_path = train_song_model(project_root)

    print("\nAll local ML artifacts are ready:")
    print(f"- {emotion_path}")
    print(f"- {response_path}")
    print(f"- {song_path}")


if __name__ == "__main__":
    main()
