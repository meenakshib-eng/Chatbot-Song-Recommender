# Chatbot Song Recommender (Fully ML-Based, Flask)

This project is now fully ML-based and offline-first.

What is model-driven now:
- Tone detection (`/tone`, `/chat`) via local text clustering model
- Chatbot reply generation (`/response`, `/chat`) via local retrieval model
- Song recommendation (`/songs`, `/chat`) via local ranking model
- Similar songs (`/similarsongs`) via local vector similarity model

No external CakeChat/Last.fm/Hugging Face calls are required.

## Project Structure

- run.py: Flask app entrypoint
- app/routes.py: API and page routes
- app/services/tone_service.py: local emotion inference
- app/services/cakechat_service.py: local response retrieval model
- app/services/lastfm_service.py: local song ranking + similarity model
- scripts/train_emotion_model.py: trains all ML artifacts (emotion clustering + retrieval/ranking)
- data/emotion_train.csv: training split (`text,label`)
- data/emotion_val.csv: validation split (`text,label`)
- data/emotion_test.csv: test split (`text,label`)
- data/local_songs.csv: local song corpus
- model/: saved model artifacts

## Endpoints

### POST /tone
Input:
```json
{ "context": ["Hey there", "How are you?"] }
```
Output:
```json
{
  "tone": "joy",
  "tone_for_chatbot": "joy",
  "tone_analysis": { "document_tone": { "tones": [] }, "sentences_tone": [] }
}
```

### POST /response
Input:
```json
{ "context": ["Hey there"], "tone": "joy" }
```
Output:
```json
{ "emotion": "joy", "response": "..." }
```

### GET /songs?tag=joy
Output:
```json
{ "tag": "pop", "songs": [{ "name": "...", "artist": "..." }] }
```

### GET /similarsongs?track=Believer&artist=Imagine%20Dragons
Output:
```json
{ "track": "Believer", "artist": "Imagine Dragons", "songs": [...] }
```

### POST /chat
Input:
```json
{ "context": ["Hey there!! What's up? How's the day?"] }
```
Output:
```json
{
  "tone": "joy",
  "tone_for_chatbot": "joy",
  "response": "...",
  "songs": [{ "name": "...", "artist": "..." }]
}
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create environment file:
```bash
copy .env.example .env
```

3. Train all local models:
```bash
python scripts/train_emotion_model.py
```
This produces:
- model/emotion_model.pkl
- model/response_model.pkl
- model/song_model.pkl

4. Start app:
```bash
python run.py
```

5. Open:
- http://127.0.0.1:5000/

## Notes

- If model files are missing, services attempt safe runtime model initialization.
- Emotion inference uses TF-IDF embeddings + KMeans clustering, then maps clusters to the four app tones (`joy`, `sadness`, `anger`, `neutral`).
- local_songs.csv is the recommendation corpus used by the song model.
- API routes are unchanged, so the current frontend keeps working.
