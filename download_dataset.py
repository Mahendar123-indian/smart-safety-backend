import os
import requests
import zipfile
import shutil

base_path = os.path.dirname(os.path.abspath(__file__))
danger_folder = os.path.join(base_path, "data", "danger")
normal_folder = os.path.join(base_path, "data", "normal")

os.makedirs(danger_folder, exist_ok=True)
os.makedirs(normal_folder, exist_ok=True)


def download_esc50():
    print("⬇️ Downloading ESC-50 dataset (this may take 2-3 minutes)...")

    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = os.path.join(base_path, "data", "esc50.zip")
    extract_path = os.path.join(base_path, "data", "esc50")

    # Download
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            percent = int(downloaded * 100 / total) if total else 0
            print(f"\r  Downloading... {percent}%", end="")

    print("\n✅ Download complete!")

    # Extract
    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Extracted!")

    # ESC-50 ALL categories:
    # https://github.com/karoldvl/ESC-50#dataset-details

    DANGER_CATEGORIES = [
        "40",  # screaming
        "38",  # crying baby
        "37",  # glass breaking
    ]

    NORMAL_CATEGORIES = [
        # Human sounds
        "10",  # crying baby (soft)
        "11",  # sneezing
        "12",  # clapping
        "13",  # breathing
        "14",  # coughing
        # Indoor sounds
        "00",  # dog bark (normal)
        "01",  # rooster
        "02",  # pig
        "03",  # cow
        # Outdoor/crowd
        "20",  # crying baby outdoor
        "21",  # footsteps
        "22",  # door knock
        "23",  # mouse click
        "24",  # keyboard typing
        # Nature
        "30",  # rain
        "31",  # sea waves
        "32",  # crackling fire
        "33",  # crickets
        "34",  # chirping birds
        # Urban/crowd
        "41",  # laughing
        "42",  # drinking/sipping
        "43",  # door knock
        "44",  # mouse click
        "45",  # clock tick
    ]

    audio_folder = os.path.join(extract_path, "ESC-50-master", "audio")

    # Clear old files first
    print("\n🗑️ Clearing old dataset files...")
    for f in os.listdir(danger_folder):
        if f.startswith("esc50_"):
            os.remove(os.path.join(danger_folder, f))
    for f in os.listdir(normal_folder):
        if f.startswith("esc50_"):
            os.remove(os.path.join(normal_folder, f))

    danger_count = 0
    normal_count = 0

    print("📂 Sorting files into danger/normal folders...")
    for filename in os.listdir(audio_folder):
        if not filename.endswith(".wav"):
            continue

        parts = filename.replace(".wav", "").split("-")
        if len(parts) < 4:
            continue

        category = parts[-1]
        src = os.path.join(audio_folder, filename)

        if category in DANGER_CATEGORIES:
            dst = os.path.join(danger_folder, f"esc50_danger_{danger_count}.wav")
            shutil.copy(src, dst)
            danger_count += 1
        elif category in NORMAL_CATEGORIES:
            dst = os.path.join(normal_folder, f"esc50_normal_{normal_count}.wav")
            shutil.copy(src, dst)
            normal_count += 1

    # Cleanup
    os.remove(zip_path)
    shutil.rmtree(extract_path)

    print(f"\n✅ Done!")
    print(f"   Danger samples : {danger_count}")
    print(f"   Normal samples : {normal_count}")
    print(f"\nNow run: python trainer.py")


if __name__ == "__main__":
    print("=" * 50)
    print("  Smart Women Safety - Dataset Downloader")
    print("=" * 50)
    download_esc50()