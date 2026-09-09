"""
Scans a cache folder for corrupted .npz files (e.g. from an interrupted
run) and deletes them, so the next pipeline.py run regenerates them
instead of crashing on them again.

Usage:
    python verify_cache.py ./Cache_n4cropped
    python verify_cache.py ./Cache
"""

import sys
import numpy as np
from pathlib import Path

def check_folder(folder):
    folder = Path(folder)
    bad_files = []

    files = sorted(folder.glob("*.npz"))
    print(f"Checking {len(files)} files in {folder}...")

    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            for key in data.files:
                _ = data[key]  # actually read each array, not just list keys
        except Exception as e:
            print(f"  CORRUPTED: {f.name} — {e}")
            bad_files.append(f)

    if not bad_files:
        print("No corrupted files found.")
        return

    print(f"\n{len(bad_files)} corrupted file(s) found. Delete them? (y/n): ", end="")
    if input().strip().lower() == "y":
        for f in bad_files:
            f.unlink()
            print(f"  Deleted {f.name}")
        print("Done. Re-run pipeline.py to regenerate these cases.")
    else:
        print("Left in place — fix manually before re-running pipeline.py.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_cache.py <folder>")
        sys.exit(1)
    check_folder(sys.argv[1])