from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.translators.google import GoogleTranslator  # noqa: E402


class GoogleTranslatorTests(unittest.TestCase):
    @patch.dict("sys.modules", {"deep_translator": MagicMock()})
    def test_translates_single_text(self) -> None:
        import deep_translator

        client = MagicMock()
        client.translate.return_value = "hello"
        deep_translator.GoogleTranslator.return_value = client

        translator = GoogleTranslator("ja", "en")
        result = translator.translate("こんにちは")

        client.translate.assert_called_once_with("こんにちは")
        self.assertEqual(result, "hello")
        self.assertEqual(translator.name, "Google")
        self.assertTrue(translator.is_available())


if __name__ == "__main__":
    unittest.main()
