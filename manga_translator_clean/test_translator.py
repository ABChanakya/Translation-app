#!/usr/bin/env python3
"""Lightweight smoke tests for the translation stack."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def test_google_translator() -> bool:
    """Smoke-test the Google translator wrapper without requiring network by default."""
    print("=" * 60)
    print("Testing Google Translator")
    print("=" * 60)

    try:
        from src.translators.base import TranslatorFactory

        translator = TranslatorFactory.create("google", "ja", "en")
        print("✓ Google translator created via factory")
        print(f"✓ Name: {translator.name}")
        print(f"✓ Available: {translator.is_available()}")

        if os.getenv("RUN_LIVE_TRANSLATION_TESTS") == "1":
            result = translator.translate("こんにちは")
            print(f"Live translation result: {result!r}")
        else:
            print("ℹ️ Skipping live translation request (set RUN_LIVE_TRANSLATION_TESTS=1 to enable)")

        print("✅ Google translator interface working!")
        return True
    except Exception as exc:
        print(f"❌ Error: {exc}")
        import traceback

        traceback.print_exc()
        return False


def test_yolo_model() -> bool:
    """Test default YOLO model loading."""
    print("\n" + "=" * 60)
    print("Testing YOLO Model")
    print("=" * 60)

    try:
        from ultralytics import YOLO
        from config.settings import YOLO_MODEL_PATH

        model_path = Path(YOLO_MODEL_PATH)
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return False

        print(f"✓ Model file exists: {model_path}")
        print("Loading model...")
        model = YOLO(str(model_path))
        print(f"✓ Model loaded: {type(model)}")
        print(f"✓ Model names: {model.names}")
        print("✅ YOLO model working!")
        return True
    except Exception as exc:
        print(f"❌ Error: {exc}")
        import traceback

        traceback.print_exc()
        return False


def main() -> int:
    print("\n🧪 Running System Tests...\n")

    results = {
        "Google Translator": test_google_translator(),
        "YOLO Model": test_yolo_model(),
    }

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<40} {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All tests passed!")
        return 0

    print("\n❌ Some tests failed - check the output above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
