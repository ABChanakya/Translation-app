"""
Progress Tracking for Translation Pipeline
Provides real-time updates via Server-Sent Events (SSE)
"""
import time
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum


class ProcessingStage(Enum):
    """Processing stages for manga translation"""
    UPLOADING = "uploading"
    DETECTING = "detecting"
    OCR = "ocr"
    TRANSLATING = "translating"
    INPAINTING = "inpainting"
    RENDERING = "rendering"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ProgressUpdate:
    """Progress update data structure"""
    stage: str
    progress: float  # 0-100
    message: str
    current_page: Optional[int] = None
    total_pages: Optional[int] = None
    eta_seconds: Optional[float] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_sse(self) -> str:
        """Convert to Server-Sent Events format"""
        data = json.dumps(self.to_dict())
        return f"data: {data}\n\n"


class ProgressTracker:
    """
    Track and emit progress updates for manga translation pipeline
    """
    
    def __init__(self):
        """Initialize progress tracker"""
        self.subscribers = []
        self.current_stage = None
        self.start_time = None
        self.stage_times = {}
        
    def subscribe(self, callback):
        """
        Subscribe to progress updates
        
        Args:
            callback: Function that receives ProgressUpdate objects
        """
        self.subscribers.append(callback)
        
    def unsubscribe(self, callback):
        """Unsubscribe from progress updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def emit(self, update: ProgressUpdate):
        """
        Emit progress update to all subscribers
        
        Args:
            update: ProgressUpdate object
        """
        for callback in self.subscribers:
            try:
                callback(update)
            except Exception as e:
                print(f"⚠️  Error in progress callback: {e}")
    
    def start_tracking(self):
        """Start tracking progress"""
        self.start_time = time.time()
        self.stage_times = {}
        
    def update_stage(
        self,
        stage: ProcessingStage,
        progress: float,
        message: str,
        current_page: Optional[int] = None,
        total_pages: Optional[int] = None
    ):
        """
        Update current processing stage
        
        Args:
            stage: Current processing stage
            progress: Progress percentage (0-100)
            message: User-friendly status message
            current_page: Current page number (for batch processing)
            total_pages: Total number of pages (for batch processing)
        """
        # Track stage start time
        stage_name = stage.value
        if self.current_stage != stage_name:
            self.stage_times[stage_name] = time.time()
            self.current_stage = stage_name
        
        # Calculate ETA if we have historical data
        eta_seconds = None
        if self.start_time and progress > 0:
            elapsed = time.time() - self.start_time
            if progress < 100:
                eta_seconds = (elapsed / progress) * (100 - progress)
        
        # Create and emit update
        update = ProgressUpdate(
            stage=stage_name,
            progress=progress,
            message=message,
            current_page=current_page,
            total_pages=total_pages,
            eta_seconds=eta_seconds
        )
        
        self.emit(update)
    
    def complete(self, message: str = "Processing complete!"):
        """Mark processing as complete"""
        self.update_stage(
            ProcessingStage.COMPLETE,
            100.0,
            message
        )
    
    def error(self, message: str):
        """Report an error"""
        update = ProgressUpdate(
            stage=ProcessingStage.ERROR.value,
            progress=0.0,
            message=message
        )
        self.emit(update)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since tracking started"""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def get_stage_duration(self, stage: ProcessingStage) -> Optional[float]:
        """Get duration of a specific stage"""
        stage_name = stage.value
        if stage_name in self.stage_times:
            start = self.stage_times[stage_name]
            # If it's the current stage, calculate duration until now
            if self.current_stage == stage_name:
                return time.time() - start
            # Otherwise, find when the next stage started
            stages = list(ProcessingStage)
            try:
                current_idx = stages.index(stage)
                if current_idx + 1 < len(stages):
                    next_stage = stages[current_idx + 1].value
                    if next_stage in self.stage_times:
                        return self.stage_times[next_stage] - start
            except ValueError:
                pass
        return None


# Global progress tracker instance for Flask routes
_global_tracker = ProgressTracker()


def get_global_tracker() -> ProgressTracker:
    """Get the global progress tracker instance"""
    return _global_tracker
