#!/usr/bin/env python3
import os
import sys
from collections import defaultdict

def dedupe_filenames(directory: str):
    # 1) List all regular files
    files = [f for f in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, f))]
    print("▶ Files in", directory, ":", files)

    # 2) Group by lowercase “base name” (no suffix, no ext)
    groups = defaultdict(list)
    for f in files:
        base, ext = os.path.splitext(f)
        groups[base.lower()].append((f, ext))

    # 3) Show any groups with more than one file
    collisions = {k: v for k, v in groups.items() if len(v) > 1}
    if not collisions:
        print("✅ No name collisions found.")
        return
    print("⚠️ Collisions detected:")
    for key, lst in collisions.items():
        print(f"  • {key} → {[name for name, _ in lst]}")

    # 4) Rename the 2nd, 3rd… file in each collision group
    for key, lst in collisions.items():
        # lst is list of (orig_name, ext); skip first, rename rest
        for idx, (orig, ext) in enumerate(lst[1:], start=1):
            new_name = f"{os.path.splitext(orig)[0]}_{idx}{ext}"
            print(f"Renaming: {orig} → {new_name}")
            os.rename(
                os.path.join(directory, orig),
                os.path.join(directory, new_name)
            )

    # 5) Final listing
    print("▶ Files now:", sorted(os.listdir(directory)))


if __name__ == "__main__":
    # use first argv or default to current dir
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    dedupe_filenames(target)
