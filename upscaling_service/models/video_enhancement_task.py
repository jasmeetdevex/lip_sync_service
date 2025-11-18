# models/videoEnhancementTask.py
from datetime import datetime
from bson import ObjectId

class VideoEnhancementTask:
    def __init__(self, task_id, input_video_url, lip_sync_task_id=None):
        self.task_id = task_id
        self.input_video_url = input_video_url
        self.lip_sync_task_id = lip_sync_task_id  # Reference to lip sync task
        self.status = "pending"  # pending, processing, completed, failed
        self.enhancement_type = "gfpgan"  # gfpgan, realesrgan, codeformer
        self.upscale_factor = 2
        self.output_url = None
        self.error_log = None
        self.performance_snapshot = None
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        
    def mark_processing(self):
        self.status = "processing"
        self.started_at = datetime.utcnow()
    
    def mark_completed(self, output_url, performance_data=None):
        self.status = "completed"
        self.output_url = output_url
        self.performance_snapshot = performance_data
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error_message):
        self.status = "failed"
        self.error_log = error_message
        self.completed_at = datetime.utcnow()
    
    def to_dict(self):
        return {
            "task_id": self.task_id,
            "input_video_url": self.input_video_url,
            "lip_sync_task_id": self.lip_sync_task_id,
            "status": self.status,
            "enhancement_type": self.enhancement_type,
            "upscale_factor": self.upscale_factor,
            "output_url": self.output_url,
            "error_log": self.error_log,
            "performance_snapshot": self.performance_snapshot,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }