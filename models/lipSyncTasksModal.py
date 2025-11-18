from datetime import datetime

class LipSyncTask:
    def __init__(self, task_id=None, video_url=None, audio_url=None,
                 status='queued', s3_keys=None, output_s3_urls=None,
                 models=None, error_log=None, performance_snapshot=None,
                 created_at=None, completed_at=None):
        """
        Enhanced with upscaling support
        """
        self.task_id = task_id
        self.video_url = video_url
        self.audio_url = audio_url
        self.status = status
        self.s3_keys = s3_keys or []
        self.output_s3_urls = output_s3_urls or []
        self.models = models or []
        self.error_log = error_log
        self.performance_snapshot = performance_snapshot or {}
        self.created_at = created_at or datetime.utcnow()
        self.completed_at = completed_at
        
        # Upscaling fields
        self.upscaling_status = "pending"  # pending, processing, completed, failed
        self.original_output_urls = []  # Store original URLs before upscaling
        self.upscaled_output_urls = []  # Store upscaled URLs
        self.upscaling_started_at = None
        self.upscaling_completed_at = None
        self.upscaling_error = None

    # ---- Existing lifecycle methods ----
    def mark_downloading(self):
        self.status = "downloading"

    def mark_processing(self):
        self.status = "processing"

    def mark_completed(self, s3_keys=None, output_s3_urls=None, models=None):
        self.status = "completed"
        
        if s3_keys:
            self.s3_keys = s3_keys if isinstance(s3_keys, list) else [s3_keys]
        if output_s3_urls:
            self.output_s3_urls = output_s3_urls if isinstance(output_s3_urls, list) else [output_s3_urls]
            # Store original URLs before upscaling
            self.original_output_urls = self.output_s3_urls.copy()
        if models:
            self.models = models if isinstance(models, list) else [models]
        
        self.completed_at = datetime.utcnow()

    def mark_failed(self, error_message):
        self.status = "failed"
        self.error_log = error_message
        self.completed_at = datetime.utcnow()

    # ---- Upscaling lifecycle methods ----
    def mark_upscaling_started(self):
        """Mark upscaling as started"""
        self.upscaling_status = "processing"
        self.upscaling_started_at = datetime.utcnow()

    def mark_upscaling_completed(self, upscaled_urls):
        """Mark upscaling as completed with new URLs"""
        self.upscaling_status = "completed"
        self.upscaled_output_urls = upscaled_urls
        # Replace original output URLs with upscaled ones
        self.output_s3_urls = upscaled_urls
        self.upscaling_completed_at = datetime.utcnow()

    def mark_upscaling_failed(self, error_message):
        """Mark upscaling as failed"""
        self.upscaling_status = "failed"
        self.upscaling_error = error_message
        self.upscaling_completed_at = datetime.utcnow()
        # Keep original output URLs (no upscaling applied)

    # ---- Database operations ----
    def save(self, collection):
        result = collection.update_one(
            {"task_id": self.task_id},
            {"$set": self.to_dict()},
            upsert=True
        )
        return result

    # ---- Utility ----
    def to_dict(self):
        """Convert to dictionary with upscaling fields"""
        return {
            "task_id": self.task_id,
            "video_url": self.video_url,
            "audio_url": self.audio_url,
            "status": self.status,
            "s3_keys": self.s3_keys,
            "output_s3_urls": self.output_s3_urls,
            "models": self.models,
            "error_log": self.error_log,
            "performance_snapshot": self.performance_snapshot,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            # Upscaling fields
            "upscaling_status": self.upscaling_status,
            "original_output_urls": self.original_output_urls,
            "upscaled_output_urls": self.upscaled_output_urls,
            "upscaling_started_at": self.upscaling_started_at,
            "upscaling_completed_at": self.upscaling_completed_at,
            "upscaling_error": self.upscaling_error
        }

    @classmethod
    def from_dict(cls, data):
        """Create from MongoDB record with upscaling support"""
        task = cls(
            task_id=data.get("task_id"),
            video_url=data.get("video_url"),
            audio_url=data.get("audio_url"),
            status=data.get("status", "queued"),
            s3_keys=data.get("s3_keys", []),
            output_s3_urls=data.get("output_s3_urls", []),
            models=data.get("models", []),
            error_log=data.get("error_log"),
            created_at=data.get("created_at"),
            completed_at=data.get("completed_at")
        )
        
        # Load upscaling fields
        task.upscaling_status = data.get("upscaling_status", "pending")
        task.original_output_urls = data.get("original_output_urls", [])
        task.upscaled_output_urls = data.get("upscaled_output_urls", [])
        task.upscaling_started_at = data.get("upscaling_started_at")
        task.upscaling_completed_at = data.get("upscaling_completed_at")
        task.upscaling_error = data.get("upscaling_error")
        
        return task

    def get_output_by_model(self, model_name):
        if model_name in self.models:
            idx = self.models.index(model_name)
            return self.output_s3_urls[idx] if idx < len(self.output_s3_urls) else None
        return None

    def get_all_outputs(self):
        return {
            model: url
            for model, url in zip(self.models, self.output_s3_urls)
        }

    def __repr__(self):
        models_str = ", ".join(self.models) if self.models else "none"
        upscaling_status = f", upscaling={self.upscaling_status}" if hasattr(self, 'upscaling_status') else ""
        return f"<LipSyncTask task_id={self.task_id} status={self.status} models=[{models_str}]{upscaling_status}>"