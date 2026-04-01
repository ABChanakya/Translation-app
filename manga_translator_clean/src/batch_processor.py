"""
Batch Processing Module for Manga Translation
Handles multi-page processing with ZIP/PDF output
"""
import os
import io
import zipfile
from pathlib import Path
from typing import List, Dict, Callable, Optional
from PIL import Image
import json
from datetime import datetime

# Optional PDF support
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  ReportLab not installed. PDF generation disabled.")
    print("   Install with: pip install reportlab")


class BatchProcessor:
    """Process multiple manga pages and generate ZIP/PDF outputs"""
    
    def __init__(self, output_dir: str = "batch_outputs"):
        """
        Initialize batch processor
        
        Args:
            output_dir: Directory to save batch outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_batch(
        self,
        input_paths: List[str],
        process_func: Callable,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        chunk_size: int = 8,
        story_context: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Process multiple images with a given processing function
        
        Args:
            input_paths: List of input image paths
            process_func: Function that processes a single image (input_path, output_path, **kwargs) -> dict
            progress_callback: Optional callback(current, total, status_message)
            chunk_size: Number of images per chunk
            story_context: Optional global story context (characters, plot, glossary, etc.)
                          Passed to all pages for consistent naming and translation
            **kwargs: Additional arguments passed to process_func
            
        Returns:
            Dictionary with batch results and output paths
        """
        # Current design is intentionally chunk-aware and disk-backed for lower-RAM
        # use. Future queue/storage-backed execution should extend this structure
        # instead of replacing it with an in-memory whole-batch flow.
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_pages': len(input_paths),
            'processed': 0,
            'failed': 0,
            'pages': [],
            'errors': [],
            'chunks': [],
            'story_context': story_context  # Store for reference
        }
        
        # Create temporary directory for processed images
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = self.output_dir / f"batch_{batch_id}"
        temp_dir.mkdir(exist_ok=True)
        
        chunk_size = max(1, int(chunk_size))
        # Context buffer lives outside the chunk loop so narrative continuity
        # is preserved across chunk boundaries, not just within a single chunk.
        previous_page_translations = []
        for chunk_index, chunk_start in enumerate(range(0, len(input_paths), chunk_size), start=1):
            chunk_paths = input_paths[chunk_start:chunk_start + chunk_size]
            results['chunks'].append({
                'chunk_index': chunk_index,
                'start_index': chunk_start,
                'count': len(chunk_paths),
            })

            for offset, input_path in enumerate(chunk_paths):
                idx = chunk_start + offset
                try:
                    if progress_callback:
                        progress_callback(
                            idx + 1,
                            len(input_paths),
                            f"Processing page {idx + 1}/{len(input_paths)} (chunk {chunk_index})"
                        )
                    
                    filename = Path(input_path).name
                    output_path = temp_dir / f"translated_{filename}"
                    
                    # Pass context from previous pages to this page's processing
                    # This provides narrative continuity so Gemma3 understands the story
                    # Also pass global story context so it's consistent across all pages
                    page_result = process_func(
                        input_path,
                        str(output_path),
                        previous_page_context=previous_page_translations.copy(),
                        story_context=story_context,
                        **kwargs
                    )
                    
                    results['pages'].append({
                        'index': idx,
                        'input': input_path,
                        'output': str(output_path),
                        'filename': filename,
                        'stats': page_result
                    })
                    results['processed'] += 1
                    
                    # Extract translated strings from this page and add to context buffer.
                    # Only the translated text (not the full dict) is passed to the next
                    # page so Gemma3 gets plain narrative lines it can actually use.
                    if isinstance(page_result, dict) and 'translations' in page_result:
                        for t in page_result['translations']:
                            txt = t.get('translated', '') if isinstance(t, dict) else str(t)
                            if txt and txt.strip():
                                previous_page_translations.append(txt.strip())
                        # Keep buffer to last 20 translated lines (~2 pages of dialogue)
                        if len(previous_page_translations) > 20:
                            previous_page_translations = previous_page_translations[-20:]
                    
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append({
                        'page': idx,
                        'input': input_path,
                        'error': str(e)
                    })
                    print(f"❌ Failed to process {input_path}: {e}")
        
        # Save metadata
        metadata_path = temp_dir / "batch_info.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        results['temp_dir'] = str(temp_dir)
        results['batch_id'] = batch_id
        
        return results
    
    @staticmethod
    def _zip_compression(path: Path) -> int:
        """Return ZIP_STORED for already-compressed formats, ZIP_DEFLATED for text/JSON."""
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            return zipfile.ZIP_STORED
        return zipfile.ZIP_DEFLATED

    def create_zip(self, batch_result: Dict, include_originals: bool = True) -> str:
        """
        Create ZIP archive from batch results.

        Writes to a .tmp file first then atomically renames to the final path so
        that a mid-write process kill never leaves a corrupt .zip behind.
        JPEG/PNG files use ZIP_STORED (they are already compressed) which
        drastically reduces memory pressure for large manga batches.
        """
        batch_id = batch_result['batch_id']
        zip_path = self.output_dir / f"manga_translation_{batch_id}.zip"
        tmp_path = zip_path.with_suffix(".zip.tmp")
        temp_dir = Path(batch_result['temp_dir'])

        # Remove any stale tmp from a previous crashed run
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            with zipfile.ZipFile(tmp_path, 'w') as zipf:
                # Translated images
                for page in batch_result['pages']:
                    output_file = Path(page['output'])
                    if output_file.exists():
                        zipf.write(output_file,
                                   f"translated/{page['filename']}",
                                   compress_type=self._zip_compression(output_file))

                # Original images (optional)
                if include_originals:
                    for page in batch_result['pages']:
                        input_file = Path(page['input'])
                        if input_file.exists():
                            zipf.write(input_file,
                                       f"originals/{page['filename']}",
                                       compress_type=self._zip_compression(input_file))

                # Metadata
                metadata_file = temp_dir / "batch_info.json"
                if metadata_file.exists():
                    zipf.write(metadata_file, "batch_info.json",
                               compress_type=zipfile.ZIP_DEFLATED)
        except Exception:
            # Clean up the partial tmp file before re-raising
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        # Atomic rename: only the complete, valid ZIP gets the final name
        tmp_path.replace(zip_path)
        return str(zip_path)
    
    def create_pdf(
        self,
        batch_result: Dict,
        page_size: str = "A4",
        include_originals: bool = False
    ) -> Optional[str]:
        """
        Create PDF from batch results
        
        Args:
            batch_result: Result dictionary from process_batch
            page_size: Page size ('A4' or 'letter')
            include_originals: Whether to include original images side-by-side
            
        Returns:
            Path to created PDF file, or None if PDF generation not available
        """
        if not PDF_AVAILABLE:
            print("⚠️  PDF generation not available. Install reportlab: pip install reportlab")
            return None
        
        batch_id = batch_result['batch_id']
        pdf_path = self.output_dir / f"manga_translation_{batch_id}.pdf"
        
        # Select page size
        pagesize = A4 if page_size.upper() == "A4" else letter
        page_width, page_height = pagesize
        
        # Create PDF
        c = canvas.Canvas(str(pdf_path), pagesize=pagesize)
        
        # Add cover page
        c.setFont("Helvetica-Bold", 24)
        c.drawCentredString(page_width / 2, page_height - 100, "Manga Translation")
        c.setFont("Helvetica", 12)
        c.drawCentredString(page_width / 2, page_height - 130, f"Batch ID: {batch_id}")
        c.drawCentredString(page_width / 2, page_height - 150, f"Total Pages: {batch_result['processed']}")
        c.drawCentredString(page_width / 2, page_height - 170, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.showPage()
        
        # Add each translated page
        for page in batch_result['pages']:
            output_file = Path(page['output'])
            
            if not output_file.exists():
                continue
            
            try:
                # Load and scale image
                img = Image.open(output_file)
                img_width, img_height = img.size
                
                # Calculate scaling to fit page with margins
                margin = 50
                max_width = page_width - 2 * margin
                max_height = page_height - 2 * margin
                
                scale = min(max_width / img_width, max_height / img_height)
                scaled_width = img_width * scale
                scaled_height = img_height * scale
                
                # Center image on page
                x = (page_width - scaled_width) / 2
                y = (page_height - scaled_height) / 2
                
                # Draw translated image
                c.drawImage(
                    str(output_file),
                    x, y,
                    width=scaled_width,
                    height=scaled_height,
                    preserveAspectRatio=True
                )
                
                # Add page number
                c.setFont("Helvetica", 10)
                c.drawString(margin, margin / 2, f"Page {page['index'] + 1}")
                
                c.showPage()
                
                # If including originals, add original on next page
                if include_originals:
                    input_file = Path(page['input'])
                    if input_file.exists():
                        orig_img = Image.open(input_file)
                        orig_width, orig_height = orig_img.size
                        
                        scale = min(max_width / orig_width, max_height / orig_height)
                        scaled_width = orig_width * scale
                        scaled_height = orig_height * scale
                        
                        x = (page_width - scaled_width) / 2
                        y = (page_height - scaled_height) / 2
                        
                        c.drawImage(
                            str(input_file),
                            x, y,
                            width=scaled_width,
                            height=scaled_height,
                            preserveAspectRatio=True
                        )
                        
                        c.setFont("Helvetica", 10)
                        c.drawString(margin, margin / 2, f"Page {page['index'] + 1} (Original)")
                        
                        c.showPage()
                
            except Exception as e:
                print(f"⚠️  Failed to add {output_file} to PDF: {e}")
                continue
        
        # Save PDF
        c.save()
        
        return str(pdf_path)
    
    def cleanup_temp_files(self, batch_result: Dict):
        """
        Clean up temporary files after batch processing
        
        Args:
            batch_result: Result dictionary from process_batch
        """
        import shutil
        
        temp_dir = Path(batch_result.get('temp_dir', ''))
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"⚠️  Failed to clean up {temp_dir}: {e}")
