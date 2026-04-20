from pathlib import Path
import pandas as pd

MODALITY_MAP = {
    "01": "full-AV",
    "02": "video-only",
    "03": "audio-only"
}

VOCAL_CHANNEL_MAP = {
    "01": "speech",
    "02": "song",
}

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

INTENSITY_MAP = {
    "01": "normal",
    "02": "strong"
}

STATEMENT_MAP = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door"
}

REPETITION_MAP = {
    "01": "1st",
    "02": "2nd"
}

def parse_filename(filepath: Path) -> dict:
    parts = filepath.stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Invalid filename format: {filepath}")
    modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts
    return {
        "filename": filepath.name,
        "filepath": str(filepath),

        "modality_code": modality,
        "vocal_channel_code": vocal_channel,
        "emotion_code": emotion,
        "intensity_code": intensity,
        "statement_code": statement,
        "repetition_code": repetition,
        "actor_code": actor,
        
        "modality": MODALITY_MAP.get(modality),
        "vocal_channel": VOCAL_CHANNEL_MAP.get(vocal_channel),
        "emotion": EMOTION_MAP.get(emotion),
        "intensity": INTENSITY_MAP.get(intensity),
        "statement": STATEMENT_MAP.get(statement),
        "repetition": REPETITION_MAP.get(repetition),
        "actor_gender": "female" if int(actor) % 2 == 0 else "male"
    }

files = Path("data").rglob("*.wav")

rows = []
for file in files:
    rows.append(parse_filename(file))
df = pd.DataFrame(rows)

print(df.head())
print(df["emotion"].value_counts())
print(df["vocal_channel"].value_counts())