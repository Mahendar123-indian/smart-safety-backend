import os
import shutil
import requests
import zipfile
import librosa
import warnings

warnings.filterwarnings("ignore")

base_path = os.path.dirname(os.path.abspath(__file__))
danger_folder = os.path.join(base_path, "data", "danger")
normal_folder = os.path.join(base_path, "data", "normal")

os.makedirs(danger_folder, exist_ok=True)
os.makedirs(normal_folder, exist_ok=True)

# ── Auto-detect ffmpeg (no hardcoded path) ────────────────────────
FFMPEG_EXE = shutil.which("ffmpeg")
if FFMPEG_EXE:
    print(f"✅ ffmpeg found: {FFMPEG_EXE}")
else:
    print("⚠️  ffmpeg not found in PATH.")
    print("   Install it: https://ffmpeg.org/download.html")
    print("   Or: winget install Gyan.FFmpeg")

# ================================================================
# RAVDESS EMOTION CODES
# ================================================================
# Filename format: 03-01-[EMOTION]-[INTENSITY]-[STATEMENT]-[REPETITION]-[ACTOR].wav
# Emotions: 01=neutral, 02=calm, 03=happy, 04=sad,
#           05=angry, 06=fearful, 07=disgust, 08=surprised
# Intensity: 01=normal, 02=strong
# Actor: 01-24 (odd=male, even=female)
# ================================================================

DANGER_EMOTIONS = [
    "05",  # angry  - being attacked/threatened
    "06",  # fearful - in danger/scared
    "07",  # disgust - distress situation
    "08",  # surprised - sudden danger
]

NORMAL_EMOTIONS = [
    "01",  # neutral  - everyday talking
    "02",  # calm     - relaxed
    "03",  # happy    - normal positive
    "04",  # sad      - not danger
]

# Only FEMALE actors (even numbers 02, 04 ... 24)
FEMALE_ACTORS = ["02", "04", "06", "08", "10", "12",
                 "14", "16", "18", "20", "22", "24"]


def download_ravdess():
    """Download RAVDESS dataset from Zenodo (official source)."""
    print("⬇️  Downloading RAVDESS Speech Dataset...")
    print("    Source: Zenodo (Official Research Repository)")
    print("    This is the HIGHEST QUALITY emotional speech dataset!\n")

    urls = [
        {
            "url": "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip",
            "filename": "ravdess_speech.zip"
        }
    ]

    extract_path = os.path.join(base_path, "data", "ravdess_raw")
    os.makedirs(extract_path, exist_ok=True)

    for item in urls:
        zip_path = os.path.join(base_path, "data", item["filename"])

        print(f"📥 Downloading: {item['filename']}")
        print("    (This is ~215MB — may take 3-5 minutes)\n")

        try:
            response = requests.get(item["url"], stream=True, timeout=300)
            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent  = int(downloaded * 100 / total)
                        mb_done  = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        print(f"\r  📦 {percent}% ({mb_done:.1f}MB / {mb_total:.1f}MB)", end="")

            print(f"\n✅ Downloaded: {item['filename']}")

            print("📂 Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_path)
            print("✅ Extracted!")

            os.remove(zip_path)

        except Exception as e:
            print(f"\n❌ Error downloading: {e}")
            return False

    return extract_path


def sort_ravdess_files(extract_path):
    """Sort RAVDESS files into danger/normal folders (female actors only)."""
    print("\n🔍 Sorting RAVDESS files by emotion and gender...")

    # Clear old RAVDESS files
    print("🗑️  Clearing old RAVDESS files...")
    for f in os.listdir(danger_folder):
        if f.startswith("ravdess_"):
            os.remove(os.path.join(danger_folder, f))
    for f in os.listdir(normal_folder):
        if f.startswith("ravdess_"):
            os.remove(os.path.join(normal_folder, f))

    danger_count = 0
    normal_count = 0
    skipped      = 0

    for root, dirs, files in os.walk(extract_path):
        for filename in files:
            if not filename.endswith(".wav"):
                continue

            # Format: 03-01-EMOTION-INTENSITY-STATEMENT-REPETITION-ACTOR.wav
            parts = filename.replace(".wav", "").split("-")
            if len(parts) != 7:
                continue

            modality  = parts[0]
            emotion   = parts[2]
            intensity = parts[3]
            actor     = parts[6]

            # Speech files only (modality 03)
            if modality != "03":
                continue

            # Female actors only
            if actor not in FEMALE_ACTORS:
                skipped += 1
                continue

            src = os.path.join(root, filename)

            if emotion in DANGER_EMOTIONS:
                dst_name = f"ravdess_danger_{danger_count}_emo{emotion}_int{intensity}.wav"
                shutil.copy(src, os.path.join(danger_folder, dst_name))
                danger_count += 1

            elif emotion in NORMAL_EMOTIONS:
                dst_name = f"ravdess_normal_{normal_count}_emo{emotion}_int{intensity}.wav"
                shutil.copy(src, os.path.join(normal_folder, dst_name))
                normal_count += 1

            else:
                skipped += 1

    print(f"\n✅ Sorting Complete!")
    print(f"   🔴 Danger samples (fearful/angry/disgust/surprised) : {danger_count}")
    print(f"   🟢 Normal samples (neutral/calm/happy/sad)          : {normal_count}")
    print(f"   ⏭️  Skipped (male actors / other)                   : {skipped}")

    return danger_count, normal_count


def verify_audio_quality():
    """Verify downloaded files are valid and readable by librosa."""
    print("\n🔬 Verifying audio quality...")

    valid_danger   = 0
    invalid_danger = 0
    valid_normal   = 0
    invalid_normal = 0

    for filename in os.listdir(danger_folder):
        if not filename.startswith("ravdess_"):
            continue
        file_path = os.path.join(danger_folder, filename)
        try:
            y, sr = librosa.load(file_path, duration=3)
            if len(y) > 0:
                valid_danger += 1
            else:
                invalid_danger += 1
                os.remove(file_path)
        except Exception:
            invalid_danger += 1

    for filename in os.listdir(normal_folder):
        if not filename.startswith("ravdess_"):
            continue
        file_path = os.path.join(normal_folder, filename)
        try:
            y, sr = librosa.load(file_path, duration=3)
            if len(y) > 0:
                valid_normal += 1
            else:
                invalid_normal += 1
                os.remove(file_path)
        except Exception:
            invalid_normal += 1

    print(f"   ✅ Valid danger files  : {valid_danger}")
    print(f"   ✅ Valid normal files  : {valid_normal}")
    print(f"   ❌ Invalid/removed     : {invalid_danger + invalid_normal}")

    return valid_danger, valid_normal


def main():
    print("=" * 55)
    print("  Smart Women Safety - RAVDESS Dataset Downloader")
    print("  High Quality Emotional Speech Dataset")
    print("=" * 55)

    # Step 1: Download
    extract_path = download_ravdess()
    if not extract_path:
        print("❌ Download failed! Check your internet connection.")
        return

    # Step 2: Sort files
    danger_count, normal_count = sort_ravdess_files(extract_path)

    # Step 3: Verify quality
    valid_danger, valid_normal = verify_audio_quality()

    # Step 4: Cleanup
    print("\n🗑️  Cleaning up temporary files...")
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    print("✅ Cleanup done!")

    # Final summary
    print("\n" + "=" * 55)
    print("✅ RAVDESS Dataset Ready!")
    print(f"   🔴 Total Danger samples : {valid_danger}")
    print(f"   🟢 Total Normal samples : {valid_normal}")
    print(f"   📊 Total dataset size   : {valid_danger + valid_normal}")
    print("\n📌 RAVDESS contains REAL professional actors")
    print("   expressing genuine emotions — highest accuracy!")
    print("\n▶️  Next: python download_cremad.py")
    print("=" * 55)


if __name__ == "__main__":
    main()