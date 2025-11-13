from datetime import datetime

class LipSyncTask:
    def __init__(self, task_id=None, video_url=None, audio_url=None,
                 status='queued', s3_keys=None, output_s3_urls=None,
                 models=None, error_log=None, created_at=None, completed_at=None):
        """
        Represents a lip-sync generation task with support for multiple models.

        :param task_id: Celery task ID
        :param video_url: Input video URL
        :param audio_url: Input audio URL
        :param status: Task status — 'queued', 'downloading', 'processing', 'completed', 'failed'
        :param s3_keys: List of S3 object keys for output videos (one per model)
        :param output_s3_urls: List of public URLs for output videos (one per model)
        :param models: List of model names used (e.g., ['Wav2Lip', 'Wav2Lip-GAN'])
        :param error_log: Error logs/stdout/stderr captured during processing
        :param created_at: Timestamp when task was created
        :param completed_at: Timestamp when task finished
        """
        self.task_id = task_id
        self.video_url = video_url
        self.audio_url = audio_url
        self.status = status
        self.s3_keys = s3_keys or []
        self.output_s3_urls = output_s3_urls or []
        self.models = models or []
        self.error_log = error_log
        self.created_at = created_at or datetime.utcnow()
        self.completed_at = completed_at

    # ---- Lifecycle methods ----
    def mark_downloading(self):
        """Mark task as downloading input files"""
        self.status = "downloading"

    def mark_processing(self):
        """Mark task as currently processing"""
        self.status = "processing"

    def mark_completed(self, s3_keys=None, output_s3_urls=None, models=None):
        """
        Mark task as completed with output information.
        
        :param s3_keys: List of S3 keys for outputs (can also be single key for backward compat)
        :param output_s3_urls: List of S3 URLs for outputs (can also be single URL for backward compat)
        :param models: List of model names used
        """
        self.status = "completed"
        
        # Support both single and multiple outputs
        if s3_keys:
            self.s3_keys = s3_keys if isinstance(s3_keys, list) else [s3_keys]
        if output_s3_urls:
            self.output_s3_urls = output_s3_urls if isinstance(output_s3_urls, list) else [output_s3_urls]
        if models:
            self.models = models if isinstance(models, list) else [models]
        
        self.completed_at = datetime.utcnow()

    def mark_failed(self, error_message):
        """Mark task as failed with error message"""
        self.status = "failed"
        self.error_log = error_message
        self.completed_at = datetime.utcnow()

    # ---- Database operations ----
    def save(self, collection):
        """
        Save task to MongoDB collection.
        Updates if exists, inserts if new.
        
        Args:
            collection: PyMongo collection object
            
        Returns:
            MongoDB UpdateResult object
        """
        result = collection.update_one(
            {"task_id": self.task_id},
            {"$set": self.to_dict()},
            upsert=True  # Insert if doesn't exist, update if does
        )
        return result

    # ---- Utility ----
    def to_dict(self):
        """Convert this object to a dictionary (for MongoDB or JSON)."""
        return {
            "task_id": self.task_id,
            "video_url": self.video_url,
            "audio_url": self.audio_url,
            "status": self.status,
            "s3_keys": self.s3_keys,
            "output_s3_urls": self.output_s3_urls,
            "models": self.models,
            "error_log": self.error_log,
            "created_at": self.created_at,
            "completed_at": self.completed_at
        }

    @classmethod
    def from_dict(cls, data):
        """Create a LipSyncTask object from MongoDB record."""
        return cls(
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

    def get_output_by_model(self, model_name):
        """
        Get the S3 URL for a specific model output.
        
        :param model_name: Name of the model (e.g., 'Wav2Lip', 'Wav2Lip-GAN')
        :return: S3 URL for that model, or None if not found
        """
        if model_name in self.models:
            idx = self.models.index(model_name)
            return self.output_s3_urls[idx] if idx < len(self.output_s3_urls) else None
        return None

    def get_all_outputs(self):
        """
        Get all model outputs as a dictionary.
        
        :return: Dictionary mapping model names to S3 URLs
        """
        return {
            model: url
            for model, url in zip(self.models, self.output_s3_urls)
        }

    def __repr__(self):
        models_str = ", ".join(self.models) if self.models else "none"
        return f"<LipSyncTask task_id={self.task_id} status={self.status} models=[{models_str}]>"