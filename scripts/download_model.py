"""
Download PPO model artifacts from Google Drive.
Run before paper_trader.py in CI or any fresh environment.

Usage:
    python scripts/download_model.py

Required env vars:
    GDRIVE_MODEL_FILE_ID     — Google Drive file ID of v10_bro_ppo_allocation_agent.zip
    GDRIVE_METADATA_FILE_ID  — Google Drive file ID of v10_bro_ppo_allocation_agent_metadata.json
"""

import os
import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("gdown not installed. Run: pip install gdown")
    sys.exit(1)


PROJECT_ROOT  = Path(__file__).parent.parent
ARTIFACT_DIR  = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE_ID    = os.environ.get("GDRIVE_MODEL_FILE_ID",    "1qPvDVLU68AiSWDDo9lvoB1rfGZ7WAkQM")
METADATA_FILE_ID = os.environ.get("GDRIVE_METADATA_FILE_ID", "1j5HWvyTY7O7XW2mc0NhrKtP09QiM1nnt")

MODEL_DST    = ARTIFACT_DIR / "v10_bro_ppo_allocation_agent.zip"
METADATA_DST = ARTIFACT_DIR / "v10_bro_ppo_allocation_agent_metadata.json"


def download(file_id: str, dest: Path, label: str):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"✓ {label} already exists at {dest} — skipping download.")
        return

    print(f"⬇  Downloading {label} ...")
    print(f"   File ID : {file_id}")
    print(f"   Dest    : {dest}")

    # Use id= kwarg (works with all recent gdown versions, no fuzzy needed)
    try:
        gdown.download(id=file_id, output=str(dest), quiet=False)
    except TypeError:
        # Fallback for older gdown that only accepts positional url arg
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(dest), quiet=False)

    if dest.exists() and dest.stat().st_size > 0:
        size_mb = dest.stat().st_size / 1_048_576
        print(f"✓ Downloaded {label} ({size_mb:.2f} MB)")
    else:
        print(f"✗ Download FAILED for {label}. Check that:")
        print(f"  1. File ID is correct: {file_id}")
        print(f"  2. Sharing is set to 'Anyone with the link can view' in Google Drive")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("BR-PPO Model Downloader")
    print("=" * 60)

    download(MODEL_FILE_ID,    MODEL_DST,    "PPO model")
    download(METADATA_FILE_ID, METADATA_DST, "PPO metadata")

    print("\n✓ All artifacts ready.")
    print(f"  Model    : {MODEL_DST}")
    print(f"  Metadata : {METADATA_DST}")
