from flask import jsonify
from flask_pymongo import PyMongo
import uuid
from datetime import datetime
from tasks import run_wav2lip_task
from models.lipSyncTasksModal import LipSyncTask
from extensions import mongo
import logging

is_testing  =False
logger = logging.getLogger(__name__)


def run_task_directly(video_url, audio_url, use_hd=False):
    """
    Runs a Wav2Lip task synchronously (bypasses Celery).
    Ideal for local testing/debugging.

    Args:
        video_url (str): URL or path of the video file
        audio_url (str): URL or path of the audio file
        use_hd (bool): If True, use HD processing
    Returns:
        dict: Task result (success/failure + output path)
    """
    import uuid
    task_id = f"test_{uuid.uuid4()}"
    
    # Call the Celery task function directly without .apply_async
    # Pass None as 'self' because Celery binds the task to 'self'
    result = run_wav2lip_task(None, task_id, video_url, audio_url, use_hd)
    
    return result

def submit_task(video_url, audio_url):
    """
    Submit a new Wav2Lip lip-sync task for processing.
    
    Args:
        video_url (str): URL of the video file
        audio_url (str): URL of the audio file
        
    Returns:
        dict: Response containing task_id or error message
    """
    task_id = None
    
    try:
        # Validate inputs
        if not video_url or not video_url.strip():
            return {"error": "video_url is required and cannot be empty"}
        
        if not audio_url or not audio_url.strip():
            return {"error": "audio_url is required and cannot be empty"}
        
        # Get database collection
        tasks_collection = mongo.db.lip_sync_tasks
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create task instance with metadata
        task = LipSyncTask(
            task_id=task_id,
            video_url=video_url.strip(),
            audio_url=audio_url.strip(),
            status="queued",
            created_at=datetime.utcnow(),
            completed_at=None,
            output_s3_urls=None,
            error_log=None,
        )
        if is_testing:
            task.mark_completed(
                s3_keys="testing key",
            output_s3_urls=['https://narratix-media.s3.us-east-1.amazonaws.com/upscaled_outputs/5dc7a319-ce4b-4dc5-8a3c-bf40b5c8797e/model_1_sync_0p00s.mp4'],
            models=[f"Wav2Lip"]
            )
            task.mark_upscaling_completed(["https://narratix-media.s3.us-east-1.amazonaws.com/upscaled_outputs/5dc7a319-ce4b-4dc5-8a3c-bf40b5c8797e/model_1_sync_0p00s.mp4"])
        
        # Save to MongoDB
        result = tasks_collection.insert_one(task.to_dict())
        
        if not result.inserted_id:
            raise Exception("Failed to insert task into database")
        
        logger.info(f"Task {task_id} created successfully")
        
        # Submit background Celery task
        # Use .apply_async for more control over task submission
        if not is_testing:
            celery_result = run_wav2lip_task.apply_async(   
                args=[task_id, video_url.strip(), audio_url.strip()],
                task_id=task_id,  # Use same ID for tracking
                retry=True,
                retry_policy={
                    'max_retries': 3,
                    'interval_start': 1,
                    'interval_step': 0.2,
                    'interval_max': 0.2,
                }
            )
            logger.info(f"Celery task queued with ID: {celery_result.id}")
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Lip-sync generation task submitted successfully",
            "status": "queued"
        }
    
    except Exception as e:
        logger.error(f"Task submission failed: {str(e)}", exc_info=True)
        
        # Attempt to mark task as failed in database
        if task_id:
            try:
                tasks_collection = mongo.db.lip_sync_tasks
                tasks_collection.update_one(
                    {"task_id": task_id},
                    {
                        "$set": {
                            "status": "failed",
                            "error_log": str(e),
                            "completed_at": datetime.utcnow()
                        }
                    }
                )
                logger.info(f"Task {task_id} marked as failed in database")
            except Exception as db_err:
                logger.error(f"Failed to update task {task_id} status in DB: {db_err}")
        
        return {
            "success": False,
            "error": f"Task submission failed: {str(e)}"
        }


def get_task_status(task_id):
    """
    Retrieve the status of a lip-sync task.
    
    Args:
        task_id (str): The ID of the task to retrieve
        
    Returns:
        dict: Task details or error message
    """
    try:
        if not task_id or not task_id.strip():
            return {"error": "task_id is required"}
        
        tasks_collection = mongo.db.lip_sync_tasks
        
        # Fetch task from database
        task_data = tasks_collection.find_one({"task_id": task_id})
        
        if not task_data:
            return {
                "success": False,
                "error": f"No task found with ID: {task_id}"
            }
        
        # Convert to model instance
        task = LipSyncTask.from_dict(task_data)
        
        return {
            "success": True,
            "task_id": task.task_id,
            "status": task.status,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "output_s3_urls": task.output_s3_urls,
            "error_log": task.error_log,
        }
    
    except Exception as e:
        logger.error(f"Failed to retrieve task {task_id}: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to retrieve task: {str(e)}"
        }