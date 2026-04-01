"""
Gradio web interface for manga translation.
"""

import json
import numpy as np
import gradio as gr
from PIL import Image

from config.settings import (
    AVAILABLE_ENGINES,
    DEFAULT_ENGINE,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_TEXT_COLOR
)
from src.pipeline import MangaTranslationPipeline


def build_gradio_app():
    """Build and run the Gradio interface"""
    
    def process_image(
        img,
        src_lang,
        tgt_lang,
        engine,
        conf,
        iou,
        text_color
    ):
        """Process image and return result"""
        if img is None:
            return None, "No image uploaded"
        
        try:
            input_image = Image.fromarray(img).convert("RGB")
            
            pipeline = MangaTranslationPipeline(
                source_lang=src_lang,
                target_lang=tgt_lang,
                translation_engine=engine,
                detection_confidence=conf,
                nms_iou_threshold=iou,
                text_color=text_color
            )
            
            output_image, logs = pipeline.process(input_image)
            
            logs_json = json.dumps(logs, ensure_ascii=False, indent=2)
            return np.array(output_image), logs_json
        
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    with gr.Blocks(title="Manga Translator") as app:
        gr.Markdown("# 📖 Manga Translator")
        gr.Markdown("### Powered by YOUR Custom YOLO Model + AI Translation")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="📤 Upload Manga Page")
            with gr.Column():
                image_output = gr.Image(type="numpy", label="✨ Translated Result")
        
        with gr.Row():
            src_lang = gr.Textbox(value="ja", label="Source Language (ISO code)")
            tgt_lang = gr.Textbox(value="en", label="Target Language (ISO code)")
            engine = gr.Dropdown(
                AVAILABLE_ENGINES,
                value=DEFAULT_ENGINE,
                label="Translation Engine"
            )
        
        with gr.Row():
            conf = gr.Slider(
                0.1, 1.0, DEFAULT_CONFIDENCE,
                label="YOLO Confidence"
            )
            iou = gr.Slider(
                0.1, 1.0, DEFAULT_IOU_THRESHOLD,
                label="NMS IoU Threshold"
            )
            text_color = gr.ColorPicker(
                value=DEFAULT_TEXT_COLOR,
                label="Text Color"
            )
        
        logs_output = gr.Textbox(label="📊 Processing Logs (JSON)", lines=10)
        
        translate_btn = gr.Button("🚀 Translate", variant="primary")
        translate_btn.click(
            process_image,
            inputs=[image_input, src_lang, tgt_lang, engine, conf, iou, text_color],
            outputs=[image_output, logs_output],
        )
        
        gr.Markdown(
            "---\n"
            "🚀 This tool uses YOUR custom YOLO model for text detection. "
            "All processing happens locally for privacy."
        )
    
    return app
