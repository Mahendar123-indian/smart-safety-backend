"""
feature_extractor.py
====================
SafeHer — Central Feature Extraction Engine

SINGLE SOURCE OF TRUTH for all feature extraction.
Both trainer.py and all detectors import from here.
This eliminates feature mismatch bugs permanently.

Movement Features : 28 features (window-based, 50 samples)
Audio Features    : 223 features (librosa-based)

FIXES vs old version:
  - N_AUDIO_FEATURES corrected to 223 (spectral_contrast returns 7 bands,
    not 10 — so total is 160+24+20+7+6+6 = 223, not 230)
  - import os moved to top (was incorrectly placed after first use)
"""

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

GRAVITY        = 9.81
SAMPLE_RATE    = 50
WINDOW_SIZE    = 50
AUDIO_SR       = 22050
AUDIO_DURATION = 5


# ══════════════════════════════════════════════════════════════════
# MOVEMENT FEATURE EXTRACTION  (28 features)
# ══════════════════════════════════════════════════════════════════

def extract_movement_features(
    ax_window: list, ay_window: list, az_window: list,
    gx_window: list, gy_window: list, gz_window: list
) -> np.ndarray:
    """
    Extract 28 movement features from a 1-second sensor window.
    Parameters:
        ax_window, ay_window, az_window : accelerometer readings (list of 50)
        gx_window, gy_window, gz_window : gyroscope readings (list of 50)
    Returns:
        np.ndarray of shape (28,)
    """
    ax = np.array(ax_window, dtype=np.float32)
    ay = np.array(ay_window, dtype=np.float32)
    az = np.array(az_window, dtype=np.float32)
    gx = np.array(gx_window, dtype=np.float32)
    gy = np.array(gy_window, dtype=np.float32)
    gz = np.array(gz_window, dtype=np.float32)

    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag  = np.sqrt(gx**2 + gy**2 + gz**2)

    # ── Accelerometer features ────────────────────────────────────
    a_mean   = float(np.mean(accel_mag))
    a_std    = float(np.std(accel_mag))
    a_peak   = float(np.max(accel_mag))
    a_energy = float(np.sum(accel_mag**2) / len(accel_mag))

    a_jerk      = np.abs(np.diff(accel_mag))
    a_jerk_mean = float(np.mean(a_jerk))
    a_jerk_peak = float(np.max(a_jerk))

    a_grav_dev = float(abs(a_mean - GRAVITY))
    a_iqr      = float(np.percentile(accel_mag, 75) - np.percentile(accel_mag, 25))
    a_skew     = float(_safe_skew(accel_mag))
    a_kurt     = float(_safe_kurt(accel_mag))

    # ── Gyroscope features ────────────────────────────────────────
    g_mean   = float(np.mean(gyro_mag))
    g_std    = float(np.std(gyro_mag))
    g_peak   = float(np.max(gyro_mag))
    g_energy = float(np.sum(gyro_mag**2) / len(gyro_mag))

    g_jerk      = np.abs(np.diff(gyro_mag))
    g_jerk_mean = float(np.mean(g_jerk))
    g_jerk_peak = float(np.max(g_jerk))

    g_iqr = float(np.percentile(gyro_mag, 75) - np.percentile(gyro_mag, 25))
    g_var = float(np.var(gyro_mag))

    # ── Cross-sensor features ─────────────────────────────────────
    combined_intensity = float(a_mean * g_mean)
    phone_shake        = float(np.var(ax) + np.var(ay) + np.var(az))
    impact_force       = float(max(0.0, a_peak - GRAVITY))

    if a_std > 1e-6 and g_std > 1e-6:
        accel_gyro_corr = float(np.corrcoef(accel_mag, gyro_mag)[0, 1])
    else:
        accel_gyro_corr = 0.0

    sma           = float((np.sum(np.abs(ax)) + np.sum(np.abs(ay)) + np.sum(np.abs(az))) / len(ax))
    dynamic_accel = float(np.mean(np.abs(accel_mag - GRAVITY)))

    # ── Temporal/Frequency features ───────────────────────────────
    mean_accel     = np.mean(accel_mag)
    zero_crossings = float(np.sum(np.diff(np.sign(accel_mag - mean_accel)) != 0))
    signal_entropy = float(_signal_entropy(accel_mag))
    dom_freq_accel = float(_dominant_freq(accel_mag, SAMPLE_RATE))
    dom_freq_gyro  = float(_dominant_freq(gyro_mag, SAMPLE_RATE))

    features = np.array([
        a_mean, a_std, a_peak, a_energy,
        a_jerk_mean, a_jerk_peak, a_grav_dev, a_iqr, a_skew, a_kurt,
        g_mean, g_std, g_peak, g_energy,
        g_jerk_mean, g_jerk_peak, g_iqr, g_var,
        combined_intensity, phone_shake, impact_force,
        accel_gyro_corr, sma, dynamic_accel,
        zero_crossings, signal_entropy, dom_freq_accel, dom_freq_gyro
    ], dtype=np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
    return features


MOVEMENT_FEATURE_NAMES = [
    "accel_mean", "accel_std", "accel_peak", "accel_energy",
    "accel_jerk_mean", "accel_jerk_peak", "gravity_deviation", "accel_iqr",
    "accel_skewness", "accel_kurtosis",
    "gyro_mean", "gyro_std", "gyro_peak", "gyro_energy",
    "gyro_jerk_mean", "gyro_jerk_peak", "gyro_iqr", "gyro_variance",
    "combined_intensity", "phone_shake", "impact_force",
    "accel_gyro_correlation", "sma", "dynamic_accel",
    "zero_crossings", "signal_entropy", "dominant_freq_accel", "dominant_freq_gyro"
]

N_MOVEMENT_FEATURES = 28


# ══════════════════════════════════════════════════════════════════
# AUDIO FEATURE EXTRACTION  (223 features)
#
# Feature breakdown (verified by counting):
#   mfcc_mean  : 40
#   mfcc_std   : 40
#   mfcc_max   : 40
#   mfcc_min   : 40   → subtotal: 160
#   chroma_mean: 12
#   chroma_std : 12   → subtotal: 24
#   mel_mean   : 20   → subtotal: 20
#   contrast   :  7   ← spectral_contrast default = 6 bands + 1 = 7
#   tonnetz    :  6   → subtotal: 13
#   scalars    :  6   (centroid, bandwidth, rolloff, zcr, rms, rms_max)
#                     ─────────────────────────────────────────────
#   TOTAL      : 223
# ══════════════════════════════════════════════════════════════════

def extract_audio_features_from_array(y: np.ndarray, sr: int = AUDIO_SR) -> np.ndarray:
    """Extract 223 audio features from a loaded audio array."""
    import librosa
    import librosa.effects

    if y is None or len(y) == 0:
        return None

    # Normalize
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    # MFCC (160) — 40 coefficients × 4 statistics
    mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfccs, axis=1)   # shape (40,)
    mfcc_std  = np.std(mfccs, axis=1)    # shape (40,)
    mfcc_max  = np.max(mfccs, axis=1)    # shape (40,)
    mfcc_min  = np.min(mfccs, axis=1)    # shape (40,)

    # Chroma (24) — 12 chroma × 2 statistics
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)   # shape (12,)
    chroma_std  = np.std(chroma, axis=1)    # shape (12,)

    # Mel Spectrogram (20) — first 20 mel band means
    mel      = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel, axis=1)[:20]    # shape (20,)

    # Spectral contrast (7) — librosa default: 6 bands + 1 = 7 values
    contrast = np.mean(
        librosa.feature.spectral_contrast(y=y, sr=sr), axis=1
    )   # shape (7,)

    # Tonnetz (6)
    harmonic = librosa.effects.harmonic(y)
    tonnetz  = np.mean(
        librosa.feature.tonnetz(y=harmonic, sr=sr), axis=1
    )   # shape (6,)

    # Scalar features (6)
    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms       = float(np.mean(librosa.feature.rms(y=y)))
    rms_max   = float(np.max(librosa.feature.rms(y=y)))

    # Concatenate → 40+40+40+40 + 12+12 + 20 + 7 + 6 + 6 = 223
    features = np.concatenate([
        mfcc_mean, mfcc_std, mfcc_max, mfcc_min,          # 160
        chroma_mean, chroma_std,                            # 24
        mel_mean,                                           # 20
        contrast,                                           # 7
        tonnetz,                                            # 6
        [centroid, bandwidth, rolloff, zcr, rms, rms_max]  # 6
    ])

    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    return features.astype(np.float32)


def extract_audio_features_from_file(file_path: str, augment: bool = False) -> np.ndarray:
    """
    Load audio file and extract 223 features.
    Tries librosa first, then soundfile+resampy as fallback.
    """
    import librosa
    import soundfile as sf

    try:
        # ── Check file exists and is not empty ────────────────────
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None

        if os.path.getsize(file_path) == 0:
            print(f"❌ Empty file: {file_path}")
            return None

        # ── Guard against git-lfs stub files (<1KB) ───────────────
        if os.path.getsize(file_path) < 1000:
            print(f"❌ File too small (git-lfs stub?): {os.path.basename(file_path)}")
            return None

        # ── Try loading with librosa ──────────────────────────────
        try:
            y, sr = librosa.load(file_path, duration=AUDIO_DURATION, sr=AUDIO_SR)
        except Exception as load_err:
            # Fallback: soundfile + resampy
            try:
                y_raw, sr_raw = sf.read(file_path)
                if len(y_raw.shape) > 1:
                    y_raw = y_raw[:, 0]   # mono
                import resampy
                y  = resampy.resample(y_raw.astype(np.float32), sr_raw, AUDIO_SR)
                sr = AUDIO_SR
                y  = y[:AUDIO_SR * AUDIO_DURATION]
            except Exception as sf_err:
                print(f"❌ Cannot load {os.path.basename(file_path)}: {load_err}")
                return None

        # ── Validate loaded audio ─────────────────────────────────
        if y is None or len(y) == 0:
            return None
        if len(y) < sr * 0.3:     # reject clips shorter than 0.3s
            return None

        # ── Augment if requested ──────────────────────────────────
        if augment:
            y = _augment_audio(y, sr)

        # ── Extract features ──────────────────────────────────────
        features = extract_audio_features_from_array(y, sr)

        if features is None:
            return None

        # ── Validate feature count ────────────────────────────────
        if len(features) != N_AUDIO_FEATURES:
            print(f"❌ Wrong feature count: {len(features)} (expected {N_AUDIO_FEATURES})")
            return None

        return features

    except Exception as e:
        print(f"❌ Feature extraction error [{os.path.basename(file_path)}]: {e}")
        return None


def _augment_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Apply random real-world augmentation to audio."""
    import librosa.effects

    try:
        aug_type = np.random.choice(["noise", "pitch", "stretch", "combined", "reverb"])

        if aug_type == "noise":
            noise_level = np.random.uniform(0.002, 0.018)
            y = y + noise_level * np.random.randn(len(y)).astype(np.float32)

        elif aug_type == "pitch":
            semitones = np.random.uniform(-2.5, 2.5)
            y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitones)

        elif aug_type == "stretch":
            rate = np.random.uniform(0.80, 1.20)
            y    = librosa.effects.time_stretch(y=y, rate=rate)
            y    = y[:sr * AUDIO_DURATION] if len(y) > sr * AUDIO_DURATION else y

        elif aug_type == "combined":
            noise_level = np.random.uniform(0.001, 0.010)
            y = y + noise_level * np.random.randn(len(y)).astype(np.float32)
            semitones = np.random.uniform(-1.5, 1.5)
            y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitones)

        elif aug_type == "reverb":
            decay          = np.random.uniform(0.3, 0.7)
            reverb_signal  = np.convolve(
                y, np.array([1.0, 0, 0, 0, decay * 0.3]), mode="same"
            )
            y = reverb_signal[:len(y)]

        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

    except Exception:
        pass   # return original if augmentation fails

    return y.astype(np.float32)


# ── CORRECTED constant ────────────────────────────────────────────
# 160 (MFCC) + 24 (chroma) + 20 (mel) + 7 (contrast) + 6 (tonnetz) + 6 (scalars)
N_AUDIO_FEATURES = 223


# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def _safe_skew(arr: np.ndarray) -> float:
    std = np.std(arr)
    if std < 1e-9:
        return 0.0
    return float(np.mean(((arr - np.mean(arr)) / std) ** 3))


def _safe_kurt(arr: np.ndarray) -> float:
    std = np.std(arr)
    if std < 1e-9:
        return 0.0
    return float(np.mean(((arr - np.mean(arr)) / std) ** 4) - 3)


def _signal_entropy(arr: np.ndarray, bins: int = 10) -> float:
    hist, _ = np.histogram(arr, bins=bins, density=True)
    hist    = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-10)))


def _dominant_freq(arr: np.ndarray, fs: int) -> float:
    n = len(arr)
    if n < 4:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(arr - np.mean(arr)))
    freqs    = np.fft.rfftfreq(n, d=1.0 / fs)
    idx      = np.argmax(fft_vals[1:]) + 1
    return float(freqs[idx])


def pad_or_trim_window(window: list, target_size: int = WINDOW_SIZE) -> list:
    if len(window) >= target_size:
        return window[-target_size:]
    pad_val = window[-1] if window else 0.0
    return window + [pad_val] * (target_size - len(window))


def validate_features(features: np.ndarray, expected_size: int,
                       name: str = "") -> bool:
    if features is None:
        return False
    if len(features) != expected_size:
        print(f"⚠️  {name} feature size mismatch: "
              f"got {len(features)}, expected {expected_size}")
        return False
    if np.isnan(features).any() or np.isinf(features).any():
        print(f"⚠️  {name} features contain NaN/Inf")
        return False
    return True