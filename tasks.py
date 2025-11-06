# tasks.py
import os
import requests
from celery_config import celery
from extensions import mongo
from datetime import datetime
import subprocess
import boto3
from models.lipSyncTasksModal import LipSyncTask
import logging

logger = logging.getLogger(__name__)

# S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)


@celery.task(name="run_wav2lip_task", bind=True)
def run_wav2lip_task(self, task_id, video_url, audio_url):
    """
    Downloads video/audio, runs Wav2Lip, uploads to S3, and logs stdout/stderr.
    
    ⭐ This task automatically runs within Flask app context thanks to ContextTask.
    """
    
    logger.info(f"Starting task {task_id}")
    
    try:
        # ✅ mongo.db now works because we're in Flask app context
        tasks_collection = mongo.db.lip_sync_tasks
        logger.info(f"Connected to MongoDB: {tasks_collection}")
        
        bucket_name = os.getenv("S3_BUCKET_NAME")
        
        # Create task model instance
        task = LipSyncTask(task_id=task_id, video_url=video_url, audio_url=audio_url)
        
        # 1️⃣ --- DOWNLOADING ---
        logger.info(f"Task {task_id}: Starting download phase")
        task.mark_downloading()
        task.save(tasks_collection)
        
        os.makedirs("inputs", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        
        video_path = f"inputs/{task_id}_video.mp4"
        audio_path = f"inputs/{task_id}_audio.wav"
        output_path = f"outputs/{task_id}_result.mp4"
        
        try:
            logger.info(f"Downloading video from {video_url}")
            with open(video_path, "wb") as f:
                f.write(requests.get(video_url, timeout=60).content)
            
            logger.info(f"Downloading audio from {audio_url}")
            with open(audio_path, "wb") as f:
                f.write(requests.get(audio_url, timeout=60).content)
                
            logger.info("Download completed successfully")
        except Exception as e:
            raise Exception(f"Failed to download input files: {e}")
        
        # 2️⃣ --- PROCESSING ---
        logger.info(f"Task {task_id}: Starting processing phase")
        task.mark_processing()
        task.save(tasks_collection)
        
        # ⭐ Get the Python executable from the current environment
        import sys
        python_executable = sys.executable
        logger.info(f"Using Python executable: {python_executable}")
        
        # UPDATE THIS PATH to your Wav2Lip repository location
        # wav2lip_dir = os.getenv(
        #     "WAV2LIP_DIR",
        #     "D:\\Wav2Lip_implementation\\lip_sync_service\\Wav2Lip-HD"
        # )
        wav2lip_dir = os.getenv(
            "WAV2LIP_DIR",
            "D:\\Wav2Lip_implementation\\lip_sync_service\\Wav2Lip"
        )
        inference_script = os.path.join(wav2lip_dir, "inference.py")
        
        # Convert relative paths to absolute paths
        video_path_abs = os.path.abspath(video_path)
        audio_path_abs = os.path.abspath(audio_path)
        output_path_abs = os.path.abspath(output_path)
        
        command = [
            python_executable, inference_script,
            "--checkpoint_path", os.path.join(wav2lip_dir, "checkpoints/wav2lip_gan.pth"),
            "--face", video_path_abs,
            "--audio", audio_path_abs,
            # "--segmentation_path", os.path.join(wav2lip_dir, "checkpoints/face_segmentation.pth"),
            # "--sr_path", os.path.join(wav2lip_dir, "checkpoints/esrgan_yunying.pth")
        ]
        
        logger.info(f"Running command: {' '.join(command)}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Capture stdout AND stderr separately for better debugging
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        stdout_output = ""
        stderr_output = ""
        
        # Read stdout
        for line in process.stdout:
            line_stripped = line.strip()
            if line_stripped:
                logger.info(f"[Wav2Lip STDOUT] {line_stripped}")
                stdout_output += line
        
        # Read stderr
        for line in process.stderr:
            line_stripped = line.strip()
            if line_stripped:
                logger.warning(f"[Wav2Lip STDERR] {line_stripped}")
                stderr_output += line
        
        # Wait for process to complete
        process.wait()
        
        # Combine logs
        log_output = stdout_output + "\n" + stderr_output
        
        # Check return code
        if process.returncode != 0:
            error_msg = f"Wav2Lip process failed with exit code {process.returncode}"
            logger.error(f"{error_msg}")
            
            if stderr_output.strip():
                logger.error(f"STDERR Output:\n{stderr_output}")
                error_msg += f"\n\nSTDERR:\n{stderr_output}"
            
            if stdout_output.strip():
                logger.error(f"STDOUT Output:\n{stdout_output}")
            
            raise Exception(error_msg)
        
        # Check if output file exists
        if not os.path.exists(output_path):
            raise Exception(f"Wav2Lip output file not created at {output_path}")
        
        logger.info("Wav2Lip processing completed successfully")
        
        # 3️⃣ --- UPLOAD TO S3 ---
        logger.info(f"Task {task_id}: Starting S3 upload phase")
        s3_key = f"wav2lip_outputs/{task_id}_result.mp4"
        
        try:
            logger.info(f"Uploading to S3: s3://{bucket_name}/{s3_key}")
            s3.upload_file(output_path, bucket_name, s3_key)
            s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
            logger.info(f"Upload completed: {s3_url}")
        except Exception as e:
            raise Exception(f"Failed to upload to S3: {e}")
        
        # 4️⃣ --- MARK COMPLETED ---
        logger.info(f"Task {task_id}: Marking as completed")
        task.mark_completed(s3_key=s3_key, output_s3_url=s3_url)
        task.error_log = log_output
        task.save(tasks_collection)
        
        # Cleanup
        logger.info("Cleaning up temporary files")
        for path in [video_path, audio_path, output_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Deleted {path}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to cleanup {path}: {cleanup_err}")
        
        logger.info(f"✅ Task {task_id} completed successfully")
        return {"success": True, "s3_url": s3_url}
    
    except Exception as e:
        logger.error(f"❌ Task {task_id} failed: {str(e)}", exc_info=True)
        
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
        except Exception as db_err:
            logger.error(f"Failed to update task status in DB: {db_err}")
        
        return {"success": False, "error": str(e)}