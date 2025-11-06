from datetime import datetime

class LipSyncTask:
    def __init__(self, task_id=None, video_url=None, audio_url=None,
                 status='queued', s3_key=None, output_s3_url=None,
                 error_log=None, created_at=None, completed_at=None):
        """
        Represents a lip-sync generation task.

        :param task_id: Celery task ID
        :param video_url: Input video URL
        :param audio_url: Input audio URL
        :param status: Task status — 'queued', 'downloading', 'processing', 'completed', 'failed'
        :param s3_key: The object key of the output video in S3
        :param output_s3_url: The public URL of the output video
        :param error_log: Any error message captured during processing
        :param created_at: Timestamp when task was created
        :param completed_at: Timestamp when task finished
        """
        self.task_id = task_id
        self.video_url = video_url
        self.audio_url = audio_url
        self.status = status
        self.s3_key = s3_key
        self.output_s3_url = output_s3_url
        self.error_log = error_log
        self.created_at = created_at or datetime.utcnow()
        self.completed_at = completed_at

    # ---- Lifecycle methods ----
    def mark_downloading(self):
        self.status = "downloading"

    def mark_processing(self):
        self.status = "processing"

    def mark_completed(self, s3_key, output_s3_url):
        self.status = "completed"
        self.s3_key = s3_key
        self.output_s3_url = output_s3_url
        self.completed_at = datetime.utcnow()

    def mark_failed(self, error_message):
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
            "s3_key": self.s3_key,
            "output_s3_url": self.output_s3_url,
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
            s3_key=data.get("s3_key"),
            output_s3_url=data.get("output_s3_url"),
            error_log=data.get("error_log"),
            created_at=data.get("created_at"),
            completed_at=data.get("completed_at")
        )

    def __repr__(self):
        return f"<LipSyncTask task_id={self.task_id} status={self.status}>"