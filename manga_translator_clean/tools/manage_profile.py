#!/usr/bin/env python3
"""
CLI tool for managing manga translation profiles.

Usage:
    python tools/manage_profile.py create "Berserk"
    python tools/manage_profile.py add-term "Berserk" "グリフィス" "Griffith" character
    python tools/manage_profile.py add-character "Berserk" "ガッツ" "Guts" "he/him" "protagonist"
    python tools/manage_profile.py show "Berserk"
    python tools/manage_profile.py list
    python tools/manage_profile.py set-style "Berserk" natural
    python tools/manage_profile.py set-honorifics "Berserk" true
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.translation.manga_profile import MangaProfile


PROFILES_DIR = str(Path(__file__).resolve().parents[1] / "profiles")


def cmd_create(args):
    profile = MangaProfile(args.series, profiles_dir=PROFILES_DIR)
    profile.save()
    print(f"✅ Created profile: {profile.profile_path}")


def cmd_add_term(args):
    profile = MangaProfile(args.series, profiles_dir=PROFILES_DIR)
    profile.add_glossary_term(args.japanese, args.english, args.category)
    print(f"✅ Added glossary term: {args.japanese} → {args.english} [{args.category}]")


def cmd_add_character(args):
    profile = MangaProfile(args.series, profiles_dir=PROFILES_DIR)
    profile.add_character(args.japanese, args.english, args.pronouns, args.role)
    print(f"✅ Added character: {args.japanese} = {args.english} ({args.pronouns})")


def cmd_show(args):
    profile = MangaProfile(args.series, profiles_dir=PROFILES_DIR)
    data = profile.data

    print(f"\n{'='*60}")
    print(f"  MANGA PROFILE: {data['series_name']}")
    print(f"{'='*60}")
    print(f"  Created: {data['created_at']}")
    print(f"  Style: {data['settings']['translation_style']}")
    print(f"  Honorifics: {'preserved' if data['settings']['preserve_honorifics'] else 'removed'}")

    # Characters
    if data["character_names"]:
        print(f"\n  CHARACTERS ({len(data['character_names'])}):")
        for jp, info in data["character_names"].items():
            role = f" — {info['role']}" if info.get("role") else ""
            print(f"    {jp} = {info['english']} ({info['pronouns']}){role}")

    # Glossary (non-character entries)
    non_char = {
        k: v for k, v in data["glossary"].items()
        if v["category"] != "character"
    }
    if non_char:
        print(f"\n  GLOSSARY ({len(non_char)} non-character terms):")
        for jp, entry in non_char.items():
            print(f"    {jp} → {entry['english']} [{entry['category']}]")

    # Chapter summaries
    if data["chapter_summaries"]:
        print(f"\n  CHAPTER SUMMARIES ({len(data['chapter_summaries'])}):")
        for ch, info in sorted(data["chapter_summaries"].items(), key=lambda x: int(x[0])):
            summary = info["summary"]
            if len(summary) > 80:
                summary = summary[:77] + "..."
            print(f"    Ch.{ch}: {summary}")

    # Recent translations
    recent = data["recent_translations"]
    if recent:
        print(f"\n  RECENT TRANSLATIONS ({len(recent)} stored, showing last 5):")
        for entry in recent[-5:]:
            jp = entry["japanese"]
            en = entry["english"]
            if len(jp) > 30:
                jp = jp[:27] + "..."
            if len(en) > 30:
                en = en[:27] + "..."
            print(f"    {jp} → {en}")

    print(f"\n{'='*60}\n")


def cmd_list(args):
    profiles_dir = Path(PROFILES_DIR)
    if not profiles_dir.exists():
        print("No profiles directory found.")
        return

    files = sorted(profiles_dir.glob("*.json"))
    if not files:
        print("No profiles found.")
        return

    print(f"\n{'='*60}")
    print(f"  MANGA PROFILES ({len(files)})")
    print(f"{'='*60}")

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            g = len(data.get("glossary", {}))
            c = len(data.get("character_names", {}))
            s = len(data.get("chapter_summaries", {}))
            r = len(data.get("recent_translations", []))
            print(
                f"  {data['series_name']:30s}  "
                f"glossary={g:3d}  chars={c:2d}  "
                f"chapters={s:2d}  recent={r:3d}"
            )
        except Exception as e:
            print(f"  {f.name}: ⚠️  Error reading: {e}")

    print(f"{'='*60}\n")


def cmd_set_style(args):
    profile = MangaProfile(args.series, profiles_dir=PROFILES_DIR)
    profile.set_translation_style(args.style)
    print(f"✅ Translation style set to: {args.style}")


def cmd_set_honorifics(args):
    preserve = args.value.lower() in ("true", "yes", "1", "on")
    profile = MangaProfile(args.series, profiles_dir=PROFILES_DIR)
    profile.set_preserve_honorifics(preserve)
    print(f"✅ Honorifics: {'preserved' if preserve else 'removed'}")


def cmd_review_auto(args):
    profile = MangaProfile(args.series, profiles_dir=PROFILES_DIR)
    auto_terms = {
        jp: entry for jp, entry in profile.data["glossary"].items()
        if entry.get("auto_detected")
    }

    if not auto_terms:
        print("No auto-detected terms to review.")
        return

    print(f"\nReviewing {len(auto_terms)} auto-detected term(s):\n")
    for jp, entry in list(auto_terms.items()):
        print(f"  {jp} → {entry['english']}  [{entry['category']}]")
        choice = input("  [A]pprove / [R]eject / [S]kip? ").strip().lower()
        if choice == "a":
            # Remove auto_detected flag
            profile.data["glossary"][jp].pop("auto_detected", None)
            print(f"  ✅ Approved: {jp} → {entry['english']}")
        elif choice == "r":
            del profile.data["glossary"][jp]
            print(f"  ❌ Rejected: {jp}")
        else:
            print(f"  ⏭️  Skipped")

    profile.save()
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Manage manga translation profiles")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    p = subparsers.add_parser("create", help="Create a new profile")
    p.add_argument("series", help="Series name")
    p.set_defaults(func=cmd_create)

    # add-term
    p = subparsers.add_parser("add-term", help="Add glossary term")
    p.add_argument("series", help="Series name")
    p.add_argument("japanese", help="Japanese term")
    p.add_argument("english", help="English translation")
    p.add_argument("category", nargs="?", default="general",
                   help="Category: character|place|technique|honorific|general")
    p.set_defaults(func=cmd_add_term)

    # add-character
    p = subparsers.add_parser("add-character", help="Add character")
    p.add_argument("series", help="Series name")
    p.add_argument("japanese", help="Japanese name")
    p.add_argument("english", help="English name")
    p.add_argument("pronouns", nargs="?", default="they/them", help="Pronouns")
    p.add_argument("role", nargs="?", default="", help="Character role")
    p.set_defaults(func=cmd_add_character)

    # show
    p = subparsers.add_parser("show", help="Show profile details")
    p.add_argument("series", help="Series name")
    p.set_defaults(func=cmd_show)

    # list
    p = subparsers.add_parser("list", help="List all profiles")
    p.set_defaults(func=cmd_list)

    # set-style
    p = subparsers.add_parser("set-style", help="Set translation style")
    p.add_argument("series", help="Series name")
    p.add_argument("style", choices=["natural", "literal", "localized"])
    p.set_defaults(func=cmd_set_style)

    # set-honorifics
    p = subparsers.add_parser("set-honorifics", help="Set honorifics preservation")
    p.add_argument("series", help="Series name")
    p.add_argument("value", help="true/false")
    p.set_defaults(func=cmd_set_honorifics)

    # review-auto
    p = subparsers.add_parser("review-auto", help="Review auto-detected glossary terms")
    p.add_argument("series", help="Series name")
    p.set_defaults(func=cmd_review_auto)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
