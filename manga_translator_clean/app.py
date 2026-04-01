"""
Main entry point for the Manga Translator application.
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Run the application"""
    
    # Determine which UI to use
    ui_choice = os.getenv("WEB_UI", "streamlit").lower()
    
    print("\n" + "="*80)
    print("                    🎌 MANGA TRANSLATOR 🎌")
    print("="*80)
    print(f"UI: {ui_choice.upper()}")
    print("="*80 + "\n")
    
    if ui_choice == "gradio":
        print("🚀 Starting Gradio interface...")
        from src.ui.gradio_app import build_gradio_app
        app = build_gradio_app()
        app.launch(share=False)
    else:
        print("🚀 Starting Streamlit interface...")
        from src.ui.streamlit_app import build_streamlit_app
        build_streamlit_app()


if __name__ == "__main__":
    main()
