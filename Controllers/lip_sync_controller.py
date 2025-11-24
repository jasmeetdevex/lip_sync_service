from botocore.exceptions import ClientError
import subprocess
import ffmpeg
import boto3
import tempfile
import requests
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import jsonify
from flask_pymongo import PyMongo
import uuid
from datetime import datetime
from tasks import run_wav2lip_task
from models.lipSyncTasksModal import LipSyncTask
from extensions import mongo
import logging
import os
is_testing  =True
logger = logging.getLogger(__name__)
s3_client = boto3.client(
's3',
aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
region_name=os.getenv("AWS_REGION")
)

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
        print("🔍 [DEBUG] submit_task called")
        print(f"   - video_url: {video_url[:50]}..." if video_url else "   - video_url: None")
        print(f"   - audio_url: {audio_url[:50]}..." if audio_url else "   - audio_url: None")
        
        # Validate inputs
        if not video_url or not video_url.strip():
            print("❌ [DEBUG] Validation failed: video_url is empty")
            return {"error": "video_url is required and cannot be empty"}
        
        if not audio_url or not audio_url.strip():
            print("❌ [DEBUG] Validation failed: audio_url is empty")
            return {"error": "audio_url is required and cannot be empty"}
        
        print("✅ [DEBUG] Input validation passed")
        
        # Get database collection
        print("🔍 [DEBUG] Accessing MongoDB collection...")
        tasks_collection = mongo.db.lip_sync_tasks
        print("✅ [DEBUG] MongoDB collection accessed successfully")
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        print(f"🆔 [DEBUG] Generated task_id: {task_id}")
        
        # Create task instance with metadata
        print("🔍 [DEBUG] Creating LipSyncTask instance...")
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
        print(f"✅ [DEBUG] LipSyncTask instance created with status: {task.status}")
        
        if is_testing:
            print("🧪 [DEBUG] TESTING MODE ENABLED - Merging audio/video locally...")
            merged_url = merge_audio_video_ffmpeg(video_url, audio_url, task_id)
            print(f"✅ [DEBUG] Merge completed. URL: {merged_url[:50]}...")

            print("🔍 [DEBUG] Marking task as completed (testing mode)...")
            task.mark_completed(
                s3_keys="testing key",
                output_s3_urls=[merged_url],
                models=["Wav2Lip"]
            )
            print(f"✅ [DEBUG] Task marked as completed. Status: {task.status}")

            print("🔍 [DEBUG] Marking upscaling as completed (testing mode)...")
            task.mark_upscaling_completed([merged_url])
            print("✅ [DEBUG] Upscaling marked as completed")
        
        # Save to MongoDB
        print("🔍 [DEBUG] Inserting task into MongoDB...")
        result = tasks_collection.insert_one(task.to_dict())
        print(f"✅ [DEBUG] Task inserted. Inserted ID: {result.inserted_id}")
        
        if not result.inserted_id:
            print("❌ [DEBUG] MongoDB insertion returned no inserted_id")
            raise Exception("Failed to insert task into database")
        
        logger.info(f"Task {task_id} created successfully")
        print(f"✅ [DEBUG] Task {task_id} created successfully in database")
        
        # Submit background Celery task
        if not is_testing:
            print("🔍 [DEBUG] NOT IN TESTING MODE - Queuing Celery task...")
            print(f"   - task_id: {task_id}")
            print(f"   - video_url: {video_url[:50]}...")
            print(f"   - audio_url: {audio_url[:50]}...")
            
            celery_result = run_wav2lip_task(   
                args=[task_id, video_url.strip(), audio_url.strip()],
                task_id=task_id,
                retry=True,
                retry_policy={
                    'max_retries': 3,
                    'interval_start': 1,
                    'interval_step': 0.2,
                    'interval_max': 0.2,
                }
            )
            print(f"✅ [DEBUG] Celery task queued successfully")
            print(f"   - Celery task ID: {celery_result.id}")
            logger.info(f"Celery task queued with ID: {celery_result.id}")
        else:
            print("🧪 [DEBUG] TESTING MODE - Skipping Celery task submission")
        
        print(f"✅ [DEBUG] Returning success response for task {task_id}")
        return {
            "success": True,
            "task_id": task_id,
            "message": "Lip-sync generation task submitted successfully",
            "status": "queued"
        }
    
    except Exception as e:
        print(f"❌ [DEBUG] Exception caught: {str(e)}")
        print(f"❌ [DEBUG] Exception type: {type(e).__name__}")
        logger.error(f"Task submission failed: {str(e)}", exc_info=True)
        
        # Attempt to mark task as failed in database
        if task_id:
            try:
                print(f"🔍 [DEBUG] Attempting to mark task {task_id} as failed in database...")
                tasks_collection = mongo.db.lip_sync_tasks
                result = tasks_collection.update_one(
                    {"task_id": task_id},
                    {
                        "$set": {
                            "status": "failed",
                            "error_log": str(e),
                            "completed_at": datetime.utcnow()
                        }
                    }
                )
                print(f"✅ [DEBUG] Task {task_id} update result:")
                print(f"   - Modified count: {result.modified_count}")
                print(f"   - Matched count: {result.matched_count}")
                logger.info(f"Task {task_id} marked as failed in database")
            except Exception as db_err:
                print(f"❌ [DEBUG] Failed to update task status in DB: {str(db_err)}")
                print(f"❌ [DEBUG] DB error type: {type(db_err).__name__}")
                logger.error(f"Failed to update task {task_id} status in DB: {db_err}")
        else:
            print("⚠️ [DEBUG] No task_id available to update database")
        
        print(f"❌ [DEBUG] Returning error response")
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
    

def merge_audio_video_ffmpeg(video_url, audio_url, task_id):
    """Downloads video/audio → merges using system ffmpeg → uploads to S3 → returns S3 URL"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")
        audio_path = os.path.join(tmpdir, "audio.wav")
        output_path = os.path.join(tmpdir, "merged_output.mp4")

        try:
            # 1. Download video with error handling
            response = requests.get(video_url, timeout=30)
            response.raise_for_status()
            with open(video_path, "wb") as f:
                f.write(response.content)

            # 2. Download audio with error handling
            response = requests.get(audio_url, timeout=30)
            response.raise_for_status()
            with open(audio_path, "wb") as f:
                f.write(response.content)

            # 3. Use ffmpeg command to merge
            command = [
                "ffmpeg",
                "-y",                      # overwrite output file
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",            # copy video track without re-encoding
                "-c:a", "aac",             # encode audio to AAC
                "-shortest",               # match duration to shortest stream
                output_path
            ]

            # Run the command with error handling
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8')
                raise Exception(f"FFmpeg failed: {error_msg}")

            # 4. Upload merged file to S3 (pass file path, not file object)
            s3_upload_response = upload_from_file_path(output_path)
            if not s3_upload_response.get("file_url"):
                raise Exception("File upload failed")
            
            return s3_upload_response.get("file_url")

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download video/audio: {str(e)}")
        except ClientError as e:
            raise Exception(f"S3 upload failed: {str(e)}")
        except subprocess.TimeoutExpired:
            raise Exception("FFmpeg process timed out (exceeded 5 minutes)")
        except Exception as e:
            raise Exception(f"Unexpected error during merge: {str(e)}")
        

def upload_from_file_path(file_path):
    """Upload a file from disk to S3"""
    BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
    
    if not file_path or not os.path.exists(file_path):
        print(f"File path does not exist: {file_path}")
        raise Exception({"error": "File path does not exist"})

    try:
        # Get filename from path
        filename = os.path.basename(file_path)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        print(f"Generated unique filename: {unique_filename}")

        # Detect content type based on file extension
        content_type = "video/mp4"  # Since we're uploading merged video
        print(f"Content type: {content_type}")

        # Upload to S3
        print("Starting upload to S3...")
        s3_client.upload_file(
            file_path,
            BUCKET_NAME,
            unique_filename,
            ExtraArgs={
                "ContentType": content_type
            }
        )
        print("Upload successful.")

        file_url = f"https://{BUCKET_NAME}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{unique_filename}"
        print(f"File accessible at: {file_url}")

        return {"file_url": file_url, "file_key": unique_filename}

    except Exception as e:
        print(f"Error during upload: {str(e)}")
        raise Exception({"error": str(e)})
    

def upload_from_file(file):
    """Upload a file object (from request) to S3"""
    BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
    if not file:
        print("No file provided in request.")
        raise Exception({"error": "image file is required"})

    if file.filename == '':
        print("Empty filename received.")
        raise Exception({"error": "No file selected"})

    try:
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        print(f"Generated unique filename: {unique_filename}")

        # Detect content type or use a fallback
        content_type = file.content_type or "image/jpeg"
        print(f"Detected content type: {content_type}")

        # Upload to S3
        print("Starting upload to S3...")
        s3_client.upload_fileobj(
            file,
            BUCKET_NAME,
            unique_filename,
            ExtraArgs={
                "ContentType": content_type
            }
        )
        print("Upload successful.")

        file_url = f"https://{BUCKET_NAME}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{unique_filename}"
        print(f"File accessible at: {file_url}")

        return {"file_url": file_url, "file_key": unique_filename}

    except Exception as e:
        print(f"Error during upload: {str(e)}")
        raise Exception({"error": str(e)})