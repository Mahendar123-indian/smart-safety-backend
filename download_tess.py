import os
import requests
import zipfile
import shutil
import warnings

warnings.filterwarnings("ignore")

base_path = os.path.dirname(os.path.abspath(__file__))
danger_folder = os.path.join(base_path, "data", "danger")
normal_folder = os.path.join(base_path, "data", "normal")

os.makedirs(danger_folder, exist_ok=True)
os.makedirs(normal_folder, exist_ok=True)

DANGER_EMOTIONS = ["angry", "fear", "disgust"]
NORMAL_EMOTIONS = ["neutral", "happy", "ps", "sad"]


def download_tess():
    print("=" * 55)
    print("  Smart Women Safety - TESS Dataset Downloader")
    print("  Toronto Emotional Speech Set - Female Only!")
    print("=" * 55)

    # TESS from Kaggle (public dataset)
    url = "https://www.kaggle.com/api/v1/datasets/download/ejlok1/toronto-emotional-speech-set-tess"

    zip_path = os.path.join(base_path, "data", "tess.zip")
    extract_path = os.path.join(base_path, "data", "tess_raw")

    print("\n⬇️  Downloading TESS from Kaggle...")
    print("    (This is ~124MB — may take 2-3 minutes)\n")

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, stream=True, timeout=300, headers=headers)

        if response.status_code != 200:
            print(f"❌ Download failed! Status: {response.status_code}")
            print("   Trying alternative source...")
            # Alternative: direct GitHub mirror
            url2 = "https://github.com/ejlok1/Toronto-Emotional-Speech-Set-TESS/archive/refs/heads/master.zip"
            response = requests.get(url2, stream=True, timeout=300)

        total = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = int(downloaded * 100 / total)
                    mb_done = downloaded / (1024 * 1024)
                    print(f"\r  📦 {percent}% ({mb_done:.1f}MB)", end="")

        print(f"\n✅ Downloaded!")

        print("📂 Extracting...")
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("✅ Extracted!")
        os.remove(zip_path)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return

    # Clear old TESS files
    print("\n🗑️  Clearing old TESS files...")
    for f in os.listdir(danger_folder):
        if f.startswith("tess_"):
            os.remove(os.path.join(danger_folder, f))
    for f in os.listdir(normal_folder):
        if f.startswith("tess_"):
            os.remove(os.path.join(normal_folder, f))

    danger_count = 0
    normal_count = 0

    print("\n📂 Sorting TESS files by emotion...")
    for root, dirs, files in os.walk(extract_path):
        for filename in files:
            if not filename.endswith(".wav"):
                continue

            # TESS filename: OAF_back_angry.wav or YAF_back_fear.wav
            filename_lower = filename.lower()
            src = os.path.join(root, filename)

            # Detect emotion from filename
            emotion_detected = None
            for emotion in DANGER_EMOTIONS + NORMAL_EMOTIONS:
                if f"_{emotion}" in filename_lower or filename_lower.endswith(f"{emotion}.wav"):
                    emotion_detected = emotion
                    break

            if emotion_detected in DANGER_EMOTIONS:
                dst = os.path.join(danger_folder, f"tess_danger_{danger_count}.wav")
                shutil.copy(src, dst)
                danger_count += 1
            elif emotion_detected in NORMAL_EMOTIONS:
                dst = os.path.join(normal_folder, f"tess_normal_{normal_count}.wav")
                shutil.copy(src, dst)
                normal_count += 1

    # Cleanup
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)

    print("\n" + "=" * 55)
    print("✅ TESS Dataset Ready!")
    print(f"   🔴 Danger samples added : {danger_count}")
    print(f"   🟢 Normal samples added : {normal_count}")
    print("=" * 55)


if __name__ == "__main__":
    download_tess()