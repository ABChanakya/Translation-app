"""
Context & Metadata Layer for Manga Translation
Parses and manages series/chapter/page metadata for translation consistency
"""
import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any
from datetime import datetime


@dataclass
class PageMeta:
    """Metadata for a single manga page"""
    series: str
    chapter: int
    page: int
    file_name: str
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PageMeta':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ChapterContext:
    """Context information for a chapter to improve translation consistency"""
    series: str
    chapter: int
    summary: str = ""
    characters: List[str] = None
    glossary: Dict[str, str] = None  # JP term -> EN term
    
    def __post_init__(self):
        if self.characters is None:
            self.characters = []
        if self.glossary is None:
            self.glossary = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChapterContext':
        """Create from dictionary"""
        return cls(**data)
    
    def save(self, output_path: Path):
        """Save chapter context to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, input_path: Path) -> 'ChapterContext':
        """Load chapter context from JSON file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class MetadataParser:
    """Parse metadata from manga file names"""
    
    # Common naming patterns
    PATTERNS = [
        # MyManga_ch01_p05.png
        r'(?P<series>.+?)_ch(?P<chapter>\d+)_p(?P<page>\d+)',
        # MyManga_c01_005.png
        r'(?P<series>.+?)_c(?P<chapter>\d+)_(?P<page>\d+)',
        # MyManga Chapter 01 Page 05.png
        r'(?P<series>.+?)\s+[Cc]hapter\s+(?P<chapter>\d+)\s+[Pp]age\s+(?P<page>\d+)',
        # MyManga - 01 - 05.png
        r'(?P<series>.+?)\s*-\s*(?P<chapter>\d+)\s*-\s*(?P<page>\d+)',
        # [MyManga] Chapter 01 - Page 05.png
        r'\[(?P<series>.+?)\]\s+[Cc]hapter\s+(?P<chapter>\d+)\s*-\s*[Pp]age\s+(?P<page>\d+)',
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(p) for p in self.PATTERNS]
    
    def parse_filename(self, filename: str, fallback_series: str = "Unknown") -> PageMeta:
        """
        Parse metadata from filename
        
        Args:
            filename: File name or path
            fallback_series: Default series name if parsing fails
            
        Returns:
            PageMeta object with extracted metadata
        """
        # Extract just the filename without extension
        base_name = Path(filename).stem
        
        # Try each pattern
        for pattern in self.compiled_patterns:
            match = pattern.search(base_name)
            if match:
                groups = match.groupdict()
                return PageMeta(
                    series=groups['series'].strip(),
                    chapter=int(groups['chapter']),
                    page=int(groups['page']),
                    file_name=filename,
                    file_path=filename
                )
        
        # Fallback: try to extract any numbers
        numbers = re.findall(r'\d+', base_name)
        if len(numbers) >= 2:
            return PageMeta(
                series=fallback_series,
                chapter=int(numbers[0]),
                page=int(numbers[1]),
                file_name=filename,
                file_path=filename
            )
        elif len(numbers) == 1:
            return PageMeta(
                series=fallback_series,
                chapter=1,
                page=int(numbers[0]),
                file_name=filename,
                file_path=filename
            )
        
        # Complete fallback
        return PageMeta(
            series=fallback_series,
            chapter=1,
            page=1,
            file_name=filename,
            file_path=filename
        )
    
    def parse_directory_structure(self, file_path: str) -> PageMeta:
        """
        Parse metadata from directory structure
        Example: MyManga/ch01/005.png or MyManga/Chapter 01/Page 05.png
        
        Args:
            file_path: Full path to file
            
        Returns:
            PageMeta object with extracted metadata
        """
        path = Path(file_path)
        parts = path.parts
        
        # Try to find series, chapter, and page from directory structure
        series = "Unknown"
        chapter = 1
        page = 1
        
        # Look for chapter directory
        for part in reversed(parts[:-1]):  # Exclude filename
            # Check for chapter patterns
            ch_match = re.search(r'[Cc]h(?:apter)?\s*(\d+)', part)
            if ch_match:
                chapter = int(ch_match.group(1))
                # Series is likely the parent of chapter directory
                idx = parts.index(part)
                if idx > 0:
                    series = parts[idx - 1]
                break
        
        # Try to get page from filename
        page_match = re.search(r'[Pp](?:age)?\s*(\d+)', path.stem)
        if page_match:
            page = int(page_match.group(1))
        else:
            # Try any number in filename
            numbers = re.findall(r'\d+', path.stem)
            if numbers:
                page = int(numbers[-1])  # Use last number as page
        
        return PageMeta(
            series=series,
            chapter=chapter,
            page=page,
            file_name=path.name,
            file_path=file_path
        )


class TranslationLogger:
    """Extended logging with metadata for translation consistency"""
    
    def __init__(self, output_dir: str = "translation_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs = []
    
    def log_bubble(
        self,
        page_meta: PageMeta,
        class_name: str,
        src_text: str,
        tgt_text: str,
        src_lang: str = "ja",
        tgt_lang: str = "en",
        confidence: Optional[float] = None,
        bbox: Optional[List[float]] = None
    ):
        """
        Log a translated bubble with full context
        
        Args:
            page_meta: Page metadata
            class_name: Detection class (dialogue, sfx, etc.)
            src_text: Original text
            tgt_text: Translated text
            src_lang: Source language
            tgt_lang: Target language
            confidence: Detection confidence
            bbox: Bounding box [x1, y1, x2, y2]
        """
        log_entry = {
            "series": page_meta.series,
            "chapter": page_meta.chapter,
            "page": page_meta.page,
            "file_name": page_meta.file_name,
            "class": class_name,
            "src_lang": src_lang,
            "src_text": src_text,
            "tgt_lang": tgt_lang,
            "tgt_text": tgt_text,
            "timestamp": datetime.now().isoformat()
        }
        
        if confidence is not None:
            log_entry["confidence"] = confidence
        if bbox is not None:
            log_entry["bbox"] = bbox
        
        self.logs.append(log_entry)
    
    def save_chapter_logs(self, series: str, chapter: int) -> Path:
        """
        Save all logs for a specific chapter
        
        Args:
            series: Series name
            chapter: Chapter number
            
        Returns:
            Path to saved log file
        """
        # Filter logs for this chapter
        chapter_logs = [
            log for log in self.logs
            if log["series"] == series and log["chapter"] == chapter
        ]
        
        if not chapter_logs:
            return None
        
        # Create filename
        safe_series = re.sub(r'[^\w\s-]', '', series).strip().replace(' ', '_')
        filename = f"{safe_series}_ch{chapter:02d}_logs.json"
        output_path = self.output_dir / filename
        
        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chapter_logs, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def create_chapter_summary(self, series: str, chapter: int) -> Dict[str, Any]:
        """
        Create a chapter summary for context tracking
        
        Args:
            series: Series name
            chapter: Chapter number
            
        Returns:
            Dictionary with chapter summary
        """
        chapter_logs = [
            log for log in self.logs
            if log["series"] == series and log["chapter"] == chapter
        ]
        
        # Group by page
        pages = {}
        for log in chapter_logs:
            page_num = log["page"]
            if page_num not in pages:
                pages[page_num] = {
                    "page": page_num,
                    "file_name": log["file_name"],
                    "bubble_count": 0,
                    "dialogues": []
                }
            pages[page_num]["bubble_count"] += 1
            if log.get("tgt_text"):
                pages[page_num]["dialogues"].append(log["tgt_text"])
        
        return {
            "series": series,
            "chapter": chapter,
            "total_pages": len(pages),
            "total_bubbles": len(chapter_logs),
            "pages": sorted(pages.values(), key=lambda x: x["page"]),
            "logs_file": f"{series}_ch{chapter:02d}_logs.json"
        }
    
    def clear(self):
        """Clear all logs"""
        self.logs = []
