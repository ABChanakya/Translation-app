from __future__ import annotations
import logging
import traceback
import ollama
import argostranslate.package
import argostranslate.translate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def translate(text: str, src: str, tgt: str, engine: str) -> str:
    """
    Translate text from src → tgt using the chosen engine.
    Falls back to Argos if the chosen engine fails.
    """
    text = text.strip()
    if not text:
        return ""

    try:
        if engine == "Gemma3":
            try:
                system_prompt = (
                    f"You are a world-class translator with deep expertise in {src} and {tgt}. "
                    f"Translate the following {src} text into fluent, idiomatic {tgt}, preserving nuance and tone. "
                    "Output ONLY the translated text, with no commentary or formatting."
                )
                user_prompt = (
                    f"=== Begin {src} text ===\n"
                    f"{text}\n"
                    f"=== End {src} text ==="
                )
                reply = ollama.chat(
                    model="gemma3:4b",
                    messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}]
                )["message"]["content"]
                logging.info("Gemma3 translation succeeded: %s", reply[:120])
                return reply
            except Exception as e:
                logging.error("Gemma3 translation failed", exc_info=True)
                return translate(text, src, tgt, "Argos")

        elif engine == "Argos":
            try:
                if _ensure_argos_pkg(src, tgt):
                    result = argostranslate.translate.translate(text, src, tgt)
                    logging.info("Argos translation succeeded: %s", result[:120])
                    return result
                else:
                    raise RuntimeError(f"No Argos package available for {src}->{tgt}")
            except Exception as e:
                logging.error("Argos translation failed", exc_info=True)
                return text  # Return original text if all fails

        else:
            raise ValueError(f"Unknown translation engine: {engine!r}")

    except Exception as e:
        logging.error("Translation failed", exc_info=True)
        return text

def _ensure_argos_pkg(src: str, tgt: str):
    """Download and install Argos language packages if needed."""
    try:
        # Download available packages
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        
        # Find the right package for this language pair
        package_to_install = next(
            (pkg for pkg in available_packages
            if pkg.from_code == src and pkg.to_code == tgt),
            None
        )
        
        if package_to_install:
            argostranslate.package.install_from_path(package_to_install.download())
            logging.info(f"Installed Argos package for {src}->{tgt}")
            return True
        else:
            logging.warning(f"No Argos package available for {src}->{tgt}")
            return False
    except Exception as e:
        logging.error(f"Failed to install Argos package: {e}")
        return False
