from pathlib import Path
import pandas as pd
import librosa
import numpy as np

def add_vector_stats(feature_dict, prefix, values):
    for i, val in enumerate(values, start=1):
        feature_dict[f"{prefix}_{i}_mean"] = float(val[0]) if np.ndim(val) > 0 else float(val)

def add_mean_std_features(feature_dict, prefix, feature_matrix):
    mean_vals = np.mean(feature_matrix, axis=1)
    std_vals = np.std(feature_matrix, axis=1)
    for i, (mean_val, std_val) in enumerate(zip(mean_vals, std_vals), start=1):
        feature_dict[f"{prefix}_{i}_mean"] = float(mean_val)
        feature_dict[f"{prefix}_{i}_std"] = float(std_val)

def load_features(filepath: Path) -> dict:
    y, sr = librosa.load(filepath, sr=48000)

    features = {
        "filename": filepath.name,
    }

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    add_mean_std_features(features, "mfcc", mfcc)

    mfcc_delta = librosa.feature.delta(mfcc)
    add_mean_std_features(features, "mfcc_delta", mfcc_delta)

    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    add_mean_std_features(features, "mfcc_delta2", mfcc_delta2)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    add_mean_std_features(features, "chroma", chroma)

    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    add_mean_std_features(features, "mel", melspectrogram)

    zcr = librosa.feature.zero_crossing_rate(y=y)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["centroid_mean"] = float(np.mean(centroid))
    features["centroid_std"] = float(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["bandwidth_mean"] = float(np.mean(bandwidth))
    features["bandwidth_std"] = float(np.std(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["rolloff_mean"] = float(np.mean(rolloff))
    features["rolloff_std"] = float(np.std(rolloff))

    return features

files = Path("data").rglob("*.wav")

rows = []
for file in files:
    rows.append(load_features(file))

df = pd.DataFrame(rows)
print(df.head())
df.to_csv("ravdess_features.csv", index=False)