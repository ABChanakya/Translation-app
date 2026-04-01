#!/usr/bin/env python3
import os

# ——— CONFIG ———
DIR = "/home/chanakya/Downloads/task_1333840_annotations_2025_04_19_16_40_04_yolo 1.1/obj_train_data/run1"   # ← change this
# —————————

def dedupe_filenames(directory: str):
    # track how many times we've seen each base name
    counts = {}

    # sort so we rename in a stable order (helps avoid conflicts)
    for fname in sorted(os.listdir(directory)):
        src = os.path.join(directory, fname)
        if not os.path.isfile(src):
            continue

        base, ext = os.path.splitext(fname)
        key = base.lower()            # case‑insensitive grouping

        # how many times have we already seen this base?
        seen = counts.get(key, 0)

        if seen > 0:
            # already exists → append suffix
            new_name = f"{base}_{seen}{ext}"
            dst = os.path.join(directory, new_name)
            print(f"Renaming: {fname} → {new_name}")
            os.rename(src, dst)

        # record that we've now handled one more of this base
        counts[key] = seen + 1

if __name__ == "__main__":
    dedupe_filenames(DIR)
