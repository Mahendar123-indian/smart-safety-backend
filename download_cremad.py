"""
download_cremad.py
==================
SafeHer — CREMA-D Dataset Downloader (FIXED VERSION)

WHY THE OLD VERSION FAILED:
  The CREMA-D GitHub repo uses Git Large File Storage (git-lfs).
  Downloading the ZIP from GitHub only gives you ~24MB of text pointer
  stubs — NOT the actual audio files. That's why all 4589 files failed.

THIS VERSION uses 3 strategies in order:
  1. Kaggle API  (best  — requires kaggle.json credentials)
  2. git-lfs clone (good — requires git + git-lfs installed)
  3. HuggingFace datasets (fallback — no credentials needed)
"""

import os
import sys
import shutil
import subprocess
import warnings

warnings.filterwarnings("ignore")

base_path     = os.path.dirname(os.path.abspath(__file__))
danger_folder = os.path.join(base_path, "data", "danger")
normal_folder = os.path.join(base_path, "data", "normal")

os.makedirs(danger_folder, exist_ok=True)
os.makedirs(normal_folder, exist_ok=True)

# ── CREMA-D emotion codes ─────────────────────────────────────────
# Filename format: 1001_DFA_ANG_XX.wav
# Emotions: ANG=angry, DIS=disgust, FEA=fear, HAP=happy, NEU=neutral, SAD=sad
DANGER_EMOTIONS = ["ANG", "DIS", "FEA"]
NORMAL_EMOTIONS = ["HAP", "NEU", "SAD"]

# Female actors only (43 female actors in CREMA-D)
FEMALE_ACTORS = [
    "1002", "1003", "1004", "1006", "1007", "1008", "1009", "1010",
    "1012", "1013", "1014", "1015", "1016", "1018", "1020", "1021",
    "1024", "1025", "1028", "1029", "1030", "1037", "1043", "1046",
    "1047", "1049", "1052", "1053", "1054", "1055", "1056", "1058",
    "1060", "1061", "1063", "1064", "1065", "1066", "1067", "1068",
    "1069", "1070", "1091"
]


# ══════════════════════════════════════════════════════════════════
# STRATEGY 1 — Kaggle API download
# ══════════════════════════════════════════════════════════════════

def try_kaggle_download():
    """
    Download CREMA-D from Kaggle using the kaggle API.

    Setup (one-time):
      1. Go to https://www.kaggle.com → Account → Create API Token
      2. Save kaggle.json to C:/Users/<YourName>/.kaggle/kaggle.json
      3. pip install kaggle
    """
    print("\n📦 STRATEGY 1: Trying Kaggle API download...")

    try:
        import kaggle  # noqa
    except ImportError:
        print("   ⚠️  kaggle package not installed.")
        print("   Run: pip install kaggle")
        return None

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("   ⚠️  ~/.kaggle/kaggle.json not found.")
        print("   Go to kaggle.com → Account → Create API Token → save kaggle.json")
        return None

    extract_path = os.path.join(base_path, "data", "cremad_raw")
    os.makedirs(extract_path, exist_ok=True)

    try:
        print("   Downloading ejlok1/cremad from Kaggle (~900MB)...")
        subprocess.run(
            [
                sys.executable, "-m", "kaggle",
                "datasets", "download",
                "-d", "ejlok1/cremad",
                "-p", extract_path,
                "--unzip"
            ],
            check=True
        )
        print("   ✅ Kaggle download complete!")
        return extract_path

    except subprocess.CalledProcessError as e:
        print(f"   ❌ Kaggle download failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# STRATEGY 2 — git-lfs clone (gets real audio files)
# ══════════════════════════════════════════════════════════════════

def try_git_lfs_clone():
    """
    Clone only the AudioWAV folder using git sparse-checkout + git-lfs.
    Requires: git and git-lfs installed.
      Install git-lfs: https://git-lfs.com
      Or: winget install Git.LFS
    """
    print("\n📦 STRATEGY 2: Trying git-lfs sparse clone...")

    # Check git is available
    if not shutil.which("git"):
        print("   ❌ git not found in PATH.")
        return None

    # Check git-lfs is available
    try:
        result = subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise FileNotFoundError
        print(f"   ✅ git-lfs found: {result.stdout.strip()}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("   ❌ git-lfs not installed.")
        print("   Install: winget install Git.LFS")
        print("   Then run: git lfs install")
        return None

    extract_path = os.path.join(base_path, "data", "cremad_raw")
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path, exist_ok=True)

    repo_url = "https://github.com/CheyneyComputerScience/CREMA-D.git"

    try:
        print("   Initializing sparse clone (AudioWAV only — ~800MB)...")
        print("   This may take 10-20 minutes...\n")

        # Step 1: init empty repo
        subprocess.run(["git", "init", extract_path], check=True)

        # Step 2: add remote
        subprocess.run(
            ["git", "-C", extract_path, "remote", "add", "origin", repo_url],
            check=True
        )

        # Step 3: enable sparse checkout
        subprocess.run(
            ["git", "-C", extract_path, "config",
             "core.sparseCheckout", "true"],
            check=True
        )

        # Step 4: set sparse path to AudioWAV only
        sparse_file = os.path.join(
            extract_path, ".git", "info", "sparse-checkout"
        )
        with open(sparse_file, "w") as f:
            f.write("AudioWAV/\n")

        # Step 5: install lfs
        subprocess.run(
            ["git", "-C", extract_path, "lfs", "install"],
            check=True
        )

        # Step 6: pull only AudioWAV
        subprocess.run(
            ["git", "-C", extract_path, "pull",
             "--depth=1", "origin", "master"],
            check=True
        )

        print("\n   ✅ git-lfs clone complete!")
        return extract_path

    except subprocess.CalledProcessError as e:
        print(f"   ❌ git-lfs clone failed: {e}")
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        return None


# ══════════════════════════════════════════════════════════════════
# STRATEGY 3 — HuggingFace datasets library (no credentials)
# ══════════════════════════════════════════════════════════════════

def try_huggingface_download():
    """
    Download CREMA-D via HuggingFace datasets library.
    No credentials needed but requires: pip install datasets soundfile
    """
    print("\n📦 STRATEGY 3: Trying HuggingFace datasets download...")

    try:
        from datasets import load_dataset  # noqa
    except ImportError:
        print("   ⚠️  datasets package not installed.")
        print("   Run: pip install datasets soundfile")
        return None

    try:
        import soundfile  # noqa
    except ImportError:
        print("   ⚠️  soundfile not installed.")
        print("   Run: pip install soundfile")
        return None

    extract_path = os.path.join(base_path, "data", "cremad_hf")
    os.makedirs(extract_path, exist_ok=True)

    try:
        print("   Loading CREMA-D from HuggingFace (myleslinder/crema-d)...")
        print("   This downloads ~800MB. Please wait...\n")

        from datasets import load_dataset
        import soundfile as sf
        import numpy as np

        dataset = load_dataset("myleslinder/crema-d", split="train",
                               trust_remote_code=True)

        print(f"   ✅ Dataset loaded: {len(dataset)} samples")
        print("   💾 Saving audio files to disk...")

        saved = 0
        for i, sample in enumerate(dataset):
            audio    = sample["audio"]
            array    = np.array(audio["array"], dtype=np.float32)
            sr       = audio["sampling_rate"]
            actor_id = str(sample.get("actor_id", "0000"))
            emotion  = sample.get("emotion", "NEU").upper()[:3]

            # Map emotion label to code
            emotion_map = {
                "ANGER": "ANG", "ANG": "ANG",
                "DISGUST": "DIS", "DIS": "DIS",
                "FEAR": "FEA", "FEA": "FEA",
                "HAPPY": "HAP", "HAP": "HAP",
                "NEUTRAL": "NEU", "NEU": "NEU",
                "SAD": "SAD"
            }
            emotion_code = emotion_map.get(emotion, None)
            if emotion_code is None:
                continue

            # Female actors only
            if actor_id not in FEMALE_ACTORS:
                continue

            # Save to correct folder
            fname = f"{actor_id}_HF_{emotion_code}_{i:04d}.wav"
            if emotion_code in DANGER_EMOTIONS:
                out_path = os.path.join(extract_path, "danger", fname)
            else:
                out_path = os.path.join(extract_path, "normal", fname)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sf.write(out_path, array, sr)
            saved += 1

            if saved % 100 == 0:
                print(f"   💾 Saved {saved} files...")

        print(f"   ✅ Saved {saved} CREMA-D files via HuggingFace!")
        return extract_path

    except Exception as e:
        print(f"   ❌ HuggingFace download failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# SORT FILES INTO danger/ AND normal/ FOLDERS
# ══════════════════════════════════════════════════════════════════

def sort_cremad_files(extract_path: str):
    """
    Walk extracted CREMA-D folder and sort files into:
      data/danger/cremad_danger_N.wav
      data/normal/cremad_normal_N.wav
    Handles both standard AudioWAV structure and HuggingFace structure.
    """
    print("\n📂 Sorting CREMA-D files by emotion and gender...")

    # Clear old CREMA-D files
    print("🗑️  Clearing old cremad_* files...")
    for fname in os.listdir(danger_folder):
        if fname.startswith("cremad_"):
            os.remove(os.path.join(danger_folder, fname))
    for fname in os.listdir(normal_folder):
        if fname.startswith("cremad_"):
            os.remove(os.path.join(normal_folder, fname))

    danger_count = 0
    normal_count = 0
    skipped      = 0

    for root, dirs, files in os.walk(extract_path):
        for filename in files:
            if not filename.lower().endswith(".wav"):
                continue

            # ── Parse filename ────────────────────────────────────
            # Standard CREMA-D: 1001_DFA_ANG_XX.wav
            # HuggingFace saved: 1002_HF_ANG_0042.wav
            parts = filename.replace(".wav", "").split("_")
            if len(parts) < 3:
                skipped += 1
                continue

            actor_id = parts[0]
            emotion  = parts[2].upper()

            # Female actors only
            if actor_id not in FEMALE_ACTORS:
                skipped += 1
                continue

            src = os.path.join(root, filename)

            # Validate file is not a git-lfs pointer (pointer files are tiny)
            try:
                file_size = os.path.getsize(src)
                if file_size < 1000:   # real WAV files are >1KB
                    skipped += 1
                    continue
            except OSError:
                skipped += 1
                continue

            if emotion in DANGER_EMOTIONS:
                dst = os.path.join(
                    danger_folder, f"cremad_danger_{danger_count}.wav"
                )
                shutil.copy(src, dst)
                danger_count += 1

            elif emotion in NORMAL_EMOTIONS:
                dst = os.path.join(
                    normal_folder, f"cremad_normal_{normal_count}.wav"
                )
                shutil.copy(src, dst)
                normal_count += 1

            else:
                skipped += 1

    print(f"\n✅ Sorting Complete!")
    print(f"   🔴 Danger files copied : {danger_count}")
    print(f"   🟢 Normal files copied : {normal_count}")
    print(f"   ⏭️  Skipped             : {skipped}")
    return danger_count, normal_count


# ══════════════════════════════════════════════════════════════════
# VERIFY FILES ARE REAL (not git-lfs stubs)
# ══════════════════════════════════════════════════════════════════

def verify_files():
    """Quick sanity check — make sure copied files are real audio."""
    print("\n🔬 Verifying audio files are real (not git-lfs stubs)...")

    import librosa

    valid   = 0
    invalid = 0
    checked = 0

    for folder in [danger_folder, normal_folder]:
        for fname in os.listdir(folder):
            if not fname.startswith("cremad_"):
                continue
            if checked >= 20:   # check first 20 as sample
                break
            fpath = os.path.join(folder, fname)
            try:
                y, sr = librosa.load(fpath, duration=1.0)
                if len(y) > 0:
                    valid += 1
                else:
                    invalid += 1
                    os.remove(fpath)
            except Exception:
                invalid += 1
                try:
                    os.remove(fpath)
                except Exception:
                    pass
            checked += 1

    if invalid > 0:
        print(f"   ⚠️  {invalid} invalid files removed (were git-lfs stubs)")
        print("   This means the download strategy didn't get real files.")
        print("   Try a different strategy above.")
    else:
        print(f"   ✅ All {valid} sampled files are real audio!")

    return invalid == 0


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def download_cremad():
    print("=" * 60)
    print("  SafeHer — CREMA-D Dataset Downloader (FIXED)")
    print("  7,442 Real Emotional Speech Samples")
    print("=" * 60)

    print("""
⚠️  IMPORTANT — Why the old downloader failed:
   The CREMA-D GitHub repo uses Git Large File Storage (git-lfs).
   Downloading as a ZIP gives you ~24MB of stub files, NOT audio.
   This script uses 3 alternative strategies to get real files.
""")

    extract_path = None

    # ── Try each strategy ─────────────────────────────────────────
    extract_path = try_kaggle_download()

    if extract_path is None:
        extract_path = try_git_lfs_clone()

    if extract_path is None:
        extract_path = try_huggingface_download()

    if extract_path is None:
        print("\n" + "=" * 60)
        print("❌ All automatic strategies failed.")
        print("\n📋 MANUAL DOWNLOAD INSTRUCTIONS:")
        print("-" * 60)
        print("Option A — Kaggle (Recommended):")
        print("  1. Go to: https://www.kaggle.com/datasets/ejlok1/cremad")
        print("  2. Click Download (~900MB ZIP)")
        print("  3. Extract ZIP to: data/cremad_manual/")
        print("  4. Re-run this script — it will auto-detect and sort files")
        print()
        print("Option B — git-lfs:")
        print("  1. Install git-lfs: winget install Git.LFS")
        print("  2. Run: git lfs install")
        print("  3. Run: git clone https://github.com/CheyneyComputerScience/CREMA-D.git data/cremad_raw")
        print("  4. Re-run this script")
        print()
        print("Option C — HuggingFace:")
        print("  pip install datasets soundfile")
        print("  Then re-run this script")
        print("=" * 60)

        # Check if manual download exists
        manual_path = os.path.join(base_path, "data", "cremad_manual")
        if os.path.exists(manual_path):
            print(f"\n📂 Found manual folder: {manual_path}")
            print("   Attempting to sort files from it...")
            extract_path = manual_path
        else:
            return

    # ── Sort files ────────────────────────────────────────────────
    danger_count, normal_count = sort_cremad_files(extract_path)

    if danger_count == 0 and normal_count == 0:
        print("\n❌ No files were sorted! The downloaded files may be git-lfs stubs.")
        print("   Please use Manual Option A (Kaggle) above.")
        return

    # ── Verify ────────────────────────────────────────────────────
    files_ok = verify_files()

    # ── Cleanup temp folder ───────────────────────────────────────
    temp_paths = [
        os.path.join(base_path, "data", "cremad_raw"),
        os.path.join(base_path, "data", "cremad_hf"),
    ]
    for p in temp_paths:
        if os.path.exists(p) and p != extract_path:
            shutil.rmtree(p)
            print(f"🗑️  Cleaned up: {p}")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if files_ok:
        print("✅ CREMA-D Dataset Ready!")
    else:
        print("⚠️  CREMA-D Download had issues — see warnings above")
    print(f"   🔴 Danger samples added : {danger_count}")
    print(f"   🟢 Normal samples added : {normal_count}")
    print(f"   📊 Total CREMA-D added  : {danger_count + normal_count}")
    print()

    # Show total dataset size
    total_danger = len([f for f in os.listdir(danger_folder)
                        if f.endswith(".wav")])
    total_normal = len([f for f in os.listdir(normal_folder)
                        if f.endswith(".wav")])
    print(f"   📁 Total danger/ files  : {total_danger}")
    print(f"   📁 Total normal/ files  : {total_normal}")
    print()
    print("▶️  Next step: python trainer.py")
    print("=" * 60)


if __name__ == "__main__":
    download_cremad()