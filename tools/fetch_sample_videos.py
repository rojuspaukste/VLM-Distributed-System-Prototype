"""Download a few sample robot videos for multi-device tests.

Lists the files of the Hugging Face dataset lerobot/libero and downloads the first --count MP4
files of the main (third-person) camera into data/ as libero-1.mp4, libero-2.mp4, ... Each file
holds several minutes of robot episodes (256x256, 10 fps). The rest of the dataset is not fetched.

    python tools/fetch_sample_videos.py [--count 4] [--out data]
"""
import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem

REPO_ID = "lerobot/libero"
CAMERA = "observation.images.image"  # main third-person camera; observation.images.image2 is the wrist camera
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download sample robot videos from lerobot/libero.")
    parser.add_argument("--count", type=int, default=4, help="number of videos (default 4)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output folder (default {DEFAULT_OUT})")
    args = parser.parse_args()

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    videos = sorted(f for f in files if f.startswith(f"videos/{CAMERA}/") and f.endswith(".mp4"))
    print(f"{REPO_ID}: {len(files)} files, {len(videos)} videos from camera {CAMERA}")
    if not videos:
        print("No videos found; any MP4 file can be used with device.py --source instead.")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    for number, path in enumerate(videos[:args.count], start=1):
        target = args.out / f"libero-{number}.mp4"
        if target.exists():
            print(f"{target.name}: already downloaded")
            continue
        start = time.monotonic()
        partial = target.with_suffix(".part")  # renamed when complete, so a broken download is never used
        fs.get_file(f"datasets/{REPO_ID}/{path}", str(partial))
        partial.rename(target)
        print(f"{target.name}: {path} ({target.stat().st_size / 2**20:.1f} MB, {time.monotonic() - start:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
