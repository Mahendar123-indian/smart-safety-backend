"""
audio_processor.py
==================
SafeHer — Audio Distress Detector

FIXED VERSION:
- Uses central feature_extractor.py (230 features — matches trainer exactly)
- Supports file upload AND real-time mic stream
- No more feature mismatch with trainer
"""

import os
import sys
import subprocess
import numpy as np
import joblib
import warnings
from typing import Dict, Optional

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.feature_extractor import (
    extract_audio_features_from_file,
    extract_audio_features_from_array,
    N_AUDIO_FEATURES,
    validate_features,
    AUDIO_SR,
    AUDIO_DURATION
)

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(base_path, "models", "scream_model.pkl")

# Find ffmpeg automatically, fall back to known path
import shutil as _sh
FFMPEG_EXE = _sh.which("ffmpeg") or \
    r"C:\Users\reddy\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"


class ScreamDetector:
    """
    Detects distress audio (screams, cries, fearful speech).

    Supports:
    - analyze_audio(file_path) — from uploaded file
    - analyze_array(y, sr)     — from mic stream / numpy array
    """

    def __init__(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print(f"✅ Audio model loaded: {N_AUDIO_FEATURES} features expected")
        else:
            self.model = None
            print(f"⚠️  Audio model not found. Run trainer.py first.")

    def analyze_audio(self, file_path: str) -> Dict:
        """Analyze an audio file for distress signals."""
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}",
                        "status": "Failed"}

            if os.path.getsize(file_path) == 0:
                return {"error": "File is empty", "status": "Failed"}

            # Convert to WAV if needed
            load_path = file_path
            tmp_wav   = None

            if not file_path.lower().endswith(".wav"):
                tmp_wav   = self._convert_to_wav(file_path)
                load_path = tmp_wav or file_path

            # Extract 230 features using central extractor
            features = extract_audio_features_from_file(load_path, augment=False)

            # Cleanup temp file
            if tmp_wav and os.path.exists(tmp_wav):
                os.remove(tmp_wav)

            if features is None:
                return {"error": "Feature extraction failed", "status": "Failed"}

            if not validate_features(features, N_AUDIO_FEATURES, "audio"):
                return {"error": "Feature validation failed", "status": "Failed"}

            return self._predict(features)

        except Exception as e:
            return {"error": str(e), "status": "Failed"}

    def analyze_array(self, y: np.ndarray, sr: int = AUDIO_SR) -> Dict:
        """
        Analyze audio from a numpy array (mic stream).
        Flutter → capture audio chunk → send as float array → call this.
        """
        try:
            if y is None or len(y) == 0:
                return {"error": "Empty audio array", "status": "Failed"}

            features = extract_audio_features_from_array(y, sr)

            if features is None:
                return {"error": "Feature extraction failed", "status": "Failed"}

            if not validate_features(features, N_AUDIO_FEATURES, "audio"):
                return {"error": "Feature validation failed", "status": "Failed"}

            return self._predict(features)

        except Exception as e:
            return {"error": str(e), "status": "Failed"}

    def _predict(self, features: np.ndarray) -> Dict:
        """Run model prediction on extracted features."""
        if self.model is None:
            return {"error": "Model not loaded. Run trainer.py first.",
                    "status": "Failed"}

        prediction = self.model.predict([features])[0]
        proba      = self.model.predict_proba([features])[0]
        danger_prob = round(float(proba[1]) * 100, 1)
        is_danger   = bool(prediction == 1)

        danger_level, action = self._get_danger_level(danger_prob)

        if is_danger and danger_prob >= 60:
            status          = "Distress Detected"
            alert_triggered = True
        elif is_danger and danger_prob >= 40:
            status          = "Possible Distress"
            alert_triggered = False
        else:
            status          = "Normal Audio"
            alert_triggered = False

        return {
            "is_scream":         is_danger,
            "danger_probability": danger_prob,
            "danger_level":       danger_level,
            "action":             action,
            "alert_triggered":    alert_triggered,
            "status":             status,
            "features_used":      N_AUDIO_FEATURES
        }

    def _convert_to_wav(self, file_path: str) -> Optional[str]:
        """Convert audio file to WAV using ffmpeg."""
        wav_path = file_path.rsplit(".", 1)[0] + "_tmp.wav"
        try:
            result = subprocess.run(
                [FFMPEG_EXE, "-y", "-i", file_path, wav_path],
                capture_output=True, text=True, timeout=20
            )
            return wav_path if os.path.exists(wav_path) else None
        except Exception as e:
            print(f"⚠️  ffmpeg conversion failed: {e}")
            return None

    @staticmethod
    def _get_danger_level(probability: float):
        if probability >= 90:
            return "CRITICAL DANGER", "🔴 Immediate SOS Required"
        elif probability >= 75:
            return "HIGH DANGER",     "🟠 Distress Detected — Alert Sent"
        elif probability >= 60:
            return "MODERATE DANGER", "🟡 Possible Distress — Monitoring"
        elif probability >= 40:
            return "LOW RISK",        "🟢 Slight Anomaly — No Action"
        else:
            return "SAFE",            "✅ Normal Audio"