"""
Streamlit web interface for manga translation.
"""

import streamlit as st
from PIL import Image

from config.settings import (
    STREAMLIT_PAGE_TITLE,
    STREAMLIT_PAGE_ICON,
    STREAMLIT_LAYOUT,
    AVAILABLE_ENGINES,
    DEFAULT_ENGINE,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_TEXT_COLOR,
    DEVICE,
    YOLO_MODEL_PATH
)
from src.pipeline import MangaTranslationPipeline


def build_streamlit_app():
    """Build and run the Streamlit interface"""
    
    st.set_page_config(
        page_title=STREAMLIT_PAGE_TITLE,
        page_icon=STREAMLIT_PAGE_ICON,
        layout=STREAMLIT_LAYOUT
    )
    
    st.title(f"{STREAMLIT_PAGE_ICON} Manga Translator")
    st.markdown("### Powered by YOUR Custom YOLO Model + AI Translation")
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("🌐 Languages")
        source_lang = st.text_input(
            "Source language",
            value="ja",
            help="ISO 639-1 code (e.g., 'ja' for Japanese)"
        )
        target_lang = st.text_input(
            "Target language",
            value="en",
            help="ISO 639-1 code (e.g., 'en' for English)"
        )
        
        st.subheader("🤖 Translation Engine")
        translation_engine = st.selectbox(
            "Choose translator",
            AVAILABLE_ENGINES,
            index=AVAILABLE_ENGINES.index(DEFAULT_ENGINE),
            help="Gemma3 provides best quality but requires Ollama"
        )
        
        st.subheader("🎯 Detection Settings")
        confidence = st.slider(
            "YOLO confidence",
            0.1, 1.0, DEFAULT_CONFIDENCE, 0.05,
            help="Higher = fewer but more confident detections"
        )
        iou_threshold = st.slider(
            "NMS IoU threshold",
            0.1, 1.0, DEFAULT_IOU_THRESHOLD, 0.05,
            help="Higher = more overlapping boxes allowed"
        )
        
        st.subheader("🎨 Appearance")
        text_color = st.color_picker(
            "Translation text color",
            DEFAULT_TEXT_COLOR,
            help="Color for translated text"
        )
        
        st.divider()
        st.caption(f"🖥️ Device: {DEVICE.upper()}")
        st.caption(f"📦 Model: {YOLO_MODEL_PATH.split('/')[-3]}")
    
    # Main area
    st.header("📤 Upload Manga Page")
    
    uploaded_file = st.file_uploader(
        "Choose a manga page image",
        type=["png", "jpg", "jpeg"],
        help="Upload a manga page to translate"
    )
    
    if uploaded_file:
        # Show original
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Original")
            st.image(uploaded_file, use_container_width=True)
        
        # Process
        input_image = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner("🔄 Processing... (Detecting → OCR → Translating → Rendering)"):
            try:
                pipeline = MangaTranslationPipeline(
                    source_lang=source_lang,
                    target_lang=target_lang,
                    translation_engine=translation_engine,
                    detection_confidence=confidence,
                    nms_iou_threshold=iou_threshold,
                    text_color=text_color
                )
                
                output_image, logs = pipeline.process(input_image)
                
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")
                return
        
        # Show result
        with col2:
            st.subheader("✨ Translated")
            st.image(output_image, use_container_width=True)
        
        # Show detailed logs
        st.header("📊 Translation Details")
        st.caption(f"Processed {len(logs)} text regions")
        
        for i, log in enumerate(logs, 1):
            with st.expander(
                f"#{i} - {log['class']} [confidence: {log['confidence']:.2f}]"
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Original ({source_lang}):**")
                    st.text(log['src_text'] or "—")
                with col_b:
                    st.markdown(f"**Translation ({target_lang}):**")
                    st.text(log['tgt_text'] or "—")
                
                st.caption(f"Bounding box: {log['bbox']}")
    
    else:
        st.info("👆 Upload a manga page image to get started!")
    
    # Footer
    st.divider()
    st.caption(
        "🚀 This tool uses YOUR custom YOLO model trained to detect manga text regions. "
        "All processing happens locally for privacy."
    )
