#!/usr/bin/env python3
# rename_seq.py
import os
import sys
import re
import argparse
"python rename_seq.py --prefix 01"


# Default directory for files
DEFAULT_DIR = "/home/chanakya/chanakya/UNI/translation_tool/data/labels/train"

def natural_sort_key(s):
    # Split into text and numeric components for human-friendly sorting
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def rename_sequential(directory, prefix, ext, start, dry_run, verbose):
    # 1) List & filter files
    try:
        all_files = sorted(os.listdir(directory), key=natural_sort_key)
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        sys.exit(1)

    files = []
    for f in all_files:
        path = os.path.join(directory, f)
        if not os.path.isfile(path):
            continue
        if ext and not f.lower().endswith(ext.lower()):
            continue
        files.append(f)

    if not files:
        print("No matching files found.")
        return

    # 2) Compute zero‑padding width
    width = len(str(len(files) + start - 1))

    # 3) Rename in order
    for idx, old in enumerate(files, start):
        base_ext = os.path.splitext(old)[1]
        new_name = f"{prefix}{idx:0{width}d}{base_ext}"
        old_path = os.path.join(directory, old)
        new_path = os.path.join(directory, new_name)

        if os.path.exists(new_path):
            print(f"Skipping '{old}': target name '{new_name}' already exists.")
            continue

        print(f"{old} → {new_name}")
        if not dry_run:
            os.rename(old_path, new_path)
        elif verbose:
            print("Dry-run mode: no files renamed.")


def main():
    parser = argparse.ArgumentParser(
        description="Sequentially rename files in /home/chanakya/.../data/needs with optional prefix, extension filter, and dry-run."
    )
    parser.add_argument(
        "directory", nargs="?", default=DEFAULT_DIR,
        help="Directory containing files to rename (default: /home/chanakya/chanakya/UNI/translation_tool/data/needs)"
    )
    parser.add_argument(
        "--prefix", default="",
        help="String prefix for new filenames (default: empty)"
    )
    parser.add_argument(
        "--ext", help="Filter by file extension (e.g. .jpg)"
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="Starting index for numbering (default: 1)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show renames without applying changes"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output in dry-run mode"
    )
    args = parser.parse_args()

    rename_sequential(
        directory=args.directory,
        prefix=args.prefix,
        ext=args.ext,
        start=args.start,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
