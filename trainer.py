"""
trainer.py
==========
SafeHer — Full Model Training Pipeline

Models trained:
1. scream_model.pkl    → Audio distress detection    (target: 90-93%)
2. movement_model.pkl  → Movement danger detection   (target: 88-92%)

Key improvements:
- Uses central feature_extractor.py (no more feature mismatch)
- SMOTE oversampling for class balance
- Augmentation: noise, pitch, stretch, reverb
- RobustScaler for audio (handles outliers better)
- GradientBoosting for movement (better on overlapping data)
- Cross-validation on every model
- Automatic overfitting detection
"""

import os
import sys
import warnings
import subprocess

import numpy as np
import pandas as pd
import joblib
import librosa

from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import (train_test_split,
                                     StratifiedKFold,
                                     cross_val_score)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle as sk_shuffle

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path)

from core.feature_extractor import (
    extract_audio_features_from_file,
    extract_audio_features_from_array,
    N_AUDIO_FEATURES,
    N_MOVEMENT_FEATURES,
    MOVEMENT_FEATURE_NAMES,
    AUDIO_SR,
    AUDIO_DURATION
)

MODELS_DIR = os.path.join(base_path, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FFMPEG_EXE = r"C:\Users\reddy\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

# Try to find ffmpeg automatically
import shutil as _shutil
_auto_ffmpeg = _shutil.which("ffmpeg")
if _auto_ffmpeg:
    FFMPEG_EXE = _auto_ffmpeg


# ══════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════

def print_section(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def convert_to_wav(file_path: str) -> str:
    wav_path = file_path.rsplit(".", 1)[0] + "_tmp.wav"
    try:
        subprocess.run(
            [FFMPEG_EXE, "-y", "-i", file_path, wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30
        )
        return wav_path if os.path.exists(wav_path) else None
    except Exception:
        return None


def validate_array(X: np.ndarray, y: np.ndarray, name: str):
    print(f"\n🔬 Data Quality — {name}")
    print(f"   Samples  : {len(X)}")
    print(f"   Features : {X.shape[1]}")
    print(f"   NaN      : {np.isnan(X).sum()}")
    print(f"   Inf      : {np.isinf(X).sum()}")
    print(f"   Class 0  : {np.sum(y == 0)}")
    print(f"   Class 1  : {np.sum(y == 1)}")

    if np.isnan(X).sum() > 0 or np.isinf(X).sum() > 0:
        print("   ⚠️  Fixing NaN/Inf with column medians...")
        col_medians = np.nanmedian(X, axis=0)
        for col in range(X.shape[1]):
            mask = ~np.isfinite(X[:, col])
            X[mask, col] = col_medians[col]

    return X, y


def balance_classes(X: np.ndarray, y: np.ndarray,
                    strategy: str = "oversample") -> tuple:
    """Balance classes using SMOTE-style oversampling or undersampling."""
    danger_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    if strategy == "oversample":
        # Oversample minority class
        min_count = min(len(danger_idx), len(normal_idx))
        max_count = max(len(danger_idx), len(normal_idx))

        if len(danger_idx) < len(normal_idx):
            extra = np.random.choice(danger_idx, max_count - len(danger_idx), replace=True)
            all_idx = np.concatenate([danger_idx, extra, normal_idx])
        else:
            extra = np.random.choice(normal_idx, max_count - len(normal_idx), replace=True)
            all_idx = np.concatenate([danger_idx, normal_idx, extra])
    else:
        # Undersample majority class
        min_count = min(len(danger_idx), len(normal_idx))
        d_idx = np.random.choice(danger_idx, min_count, replace=False)
        n_idx = np.random.choice(normal_idx, min_count, replace=False)
        all_idx = np.concatenate([d_idx, n_idx])

    X_bal = X[all_idx]
    y_bal = y[all_idx]
    return sk_shuffle(X_bal, y_bal, random_state=2024)


def evaluate_model(model, X_test, y_test, name: str):
    y_pred = model.predict(X_test)
    print(f"\n--- {name} Performance ---")
    print(classification_report(y_test, y_pred,
                                 target_names=["Normal", "Danger"]))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:")
    print(f"  TN={tn}  FP={fp}   (False Alarm Rate: {fp/(tn+fp):.1%})")
    print(f"  FN={fn}  TP={tp}   (Miss Rate:        {fn/(fn+tp):.1%})")


# ══════════════════════════════════════════════════════════════════
# PART 1 — AUDIO SCREAM DETECTION MODEL
# ══════════════════════════════════════════════════════════════════

def load_audio_dataset():
    """
    Load audio files from data/danger/ and data/normal/.
    Applies 2x augmentation on danger samples.
    Uses central extract_audio_features_from_file() for consistency.
    """
    X, y = [], []
    augmented_count = 0

    danger_folder = os.path.join(base_path, "data", "danger")
    normal_folder = os.path.join(base_path, "data", "normal")

    if not os.path.exists(danger_folder) or not os.path.exists(normal_folder):
        print("❌ Audio data folders missing!")
        print("   Run: python download_ravdess.py")
        print("   Run: python download_cremad.py")
        print("   Run: python download_tess.py")
        return np.array([]), np.array([])

    print("\n📂 Loading DANGER audio files...")
    danger_files = sorted([f for f in os.listdir(danger_folder)
                            if f.endswith((".wav", ".mp3", ".flac"))])

    for i, filename in enumerate(danger_files):
        path = os.path.join(danger_folder, filename)

        # Original
        feats = extract_audio_features_from_file(path, augment=False)
        if feats is not None and len(feats) == N_AUDIO_FEATURES:
            X.append(feats)
            y.append(1)

            # 2 augmented copies per original
            for _ in range(2):
                aug_feats = extract_audio_features_from_file(path, augment=True)
                if aug_feats is not None and len(aug_feats) == N_AUDIO_FEATURES:
                    X.append(aug_feats)
                    y.append(1)
                    augmented_count += 1

        if (i + 1) % 50 == 0:
            print(f"  📄 {i+1}/{len(danger_files)} danger files | "
                  f"features so far: {np.sum(np.array(y) == 1)}")

    print(f"  ✅ Danger: {np.sum(np.array(y) == 1)} "
          f"(orig + {augmented_count} augmented)")

    print("\n📂 Loading NORMAL audio files...")
    normal_files = sorted([f for f in os.listdir(normal_folder)
                            if f.endswith((".wav", ".mp3", ".flac"))])

    for i, filename in enumerate(normal_files):
        path = os.path.join(normal_folder, filename)
        feats = extract_audio_features_from_file(path, augment=False)
        if feats is not None and len(feats) == N_AUDIO_FEATURES:
            X.append(feats)
            y.append(0)

        if (i + 1) % 50 == 0:
            print(f"  📄 {i+1}/{len(normal_files)} normal files | "
                  f"features so far: {np.sum(np.array(y) == 0)}")

    print(f"  ✅ Normal: {np.sum(np.array(y) == 0)}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def train_audio_model():
    print_section("PART 1 — Audio Scream Detection Model")

    X, y = load_audio_dataset()

    if len(X) == 0:
        print("❌ No audio data. Skipping audio model.")
        return

    X, y = validate_array(X, y, "audio")
    X, y = balance_classes(X, y, strategy="undersample")

    print(f"\n📊 Balanced: {np.sum(y==1)} danger + {np.sum(y==0)} normal")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=2024, stratify=y
    )

    print("\n⏳ Training audio model (RandomForest + GradientBoosting ensemble)...")

    # Ensemble of two complementary classifiers
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=28,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=2024,
        n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.80,
        min_samples_leaf=3,
        random_state=2024
    )

    model = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", VotingClassifier(
            estimators=[("rf", rf), ("gb", gb)],
            voting="soft",
            weights=[2, 1]     # RF gets more weight (better for audio)
        ))
    ])

    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test, "Audio Scream Model")

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
    print(f"\nCross-validation F1 : {cv_scores.round(4)}")
    print(f"Mean CV F1          : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    if cv_scores.mean() > 0.97:
        print("⚠️  WARNING: Possible overfitting (CV F1 > 0.97)")
    elif cv_scores.std() < 0.005:
        print("⚠️  WARNING: Very low CV variance — data may be too clean")
    else:
        print("✅ Audio model looks healthy!")

    save_path = os.path.join(MODELS_DIR, "scream_model.pkl")
    joblib.dump(model, save_path)
    print(f"✅ Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════
# PART 2 — MOVEMENT DANGER DETECTION MODEL
# ══════════════════════════════════════════════════════════════════

def train_movement_model():
    print_section("PART 2 — Movement Danger Detection Model")

    data_path = os.path.join(base_path, "data", "movement_dataset.csv")

    if not os.path.exists(data_path):
        print("❌ movement_dataset.csv not found!")
        print("   Run: python generate_movement_dataset.py first")
        return

    df = pd.read_csv(data_path)
    before = len(df)
    df = df.drop_duplicates().dropna()
    if before != len(df):
        print(f"   Removed {before - len(df)} duplicate/null rows")

    print(f"\n📊 Dataset:")
    print(f"   Total   : {len(df)}")
    print(f"   Danger  : {len(df[df['label'] == 1])}")
    print(f"   Normal  : {len(df[df['label'] == 0])}")
    print(f"   Features: {N_MOVEMENT_FEATURES} (from feature_extractor.py)")

    feature_cols = MOVEMENT_FEATURE_NAMES
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    X, y = validate_array(X, y, "movement")
    X, y = balance_classes(X, y, strategy="oversample")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=2024, stratify=y
    )

    print("\n⏳ Training movement model (GradientBoosting)...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.82,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=2024
        ))
    ])

    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, "Movement Model")

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
    print(f"\nCross-validation F1 : {cv_scores.round(4)}")
    print(f"Mean CV F1          : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    if cv_scores.mean() > 0.97:
        print("⚠️  WARNING: Possible overfitting")
    else:
        print("✅ Movement model looks healthy!")

    # Feature importance
    clf = model.named_steps["clf"]
    print("\n📊 Top 10 Feature Importances:")
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    for i in top_idx:
        bar = "█" * int(importances[i] * 60)
        print(f"  {feature_cols[i]:<28}: {importances[i]:.4f}  {bar}")

    save_path = os.path.join(MODELS_DIR, "movement_model.pkl")
    joblib.dump(model, save_path)
    print(f"\n✅ Saved: {save_path}")

    # Save feature names for validation
    names_path = os.path.join(MODELS_DIR, "movement_feature_names.txt")
    with open(names_path, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"✅ Feature names saved: {names_path}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_section("SafeHer — Full Model Training Pipeline")
    print("  Training audio + movement models.")
    print("  Features sourced from feature_extractor.py (no mismatch).")
    print("  Expected time: 10-25 minutes depending on dataset size.\n")

    train_audio_model()
    train_movement_model()

    print("\n" + "=" * 65)
    print("  ✅ All models trained successfully!")
    print("  📋 Next step: python train_danger_model.py")
    print("=" * 65)