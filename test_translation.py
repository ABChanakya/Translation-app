import logging
from test_translate import translate

def test_translations():
    text = "こんにちは"  # "Hello" in Japanese
    
    print("\n1. Testing Gemma3 translation (with Ollama running):")
    result = translate(text, "ja", "en", "Gemma3")
    print(f"Result: {result}")

    input("\nPlease stop Ollama now (pkill ollama) and press Enter...")
    
    print("\n2. Testing fallback path (Ollama stopped):")
    result = translate(text, "ja", "en", "Gemma3")
    print(f"Result: {result}")

if __name__ == "__main__":
    test_translations()
