# tasks.py
import os
import sys
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

def clean_up(audio_path, video_path, output_path):
    for f in [video_path, audio_path, output_path]:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception as e:
            logger.warning(f"Cleanup failed for {f}: {e}")


@celery.task(name="run_wav2lip_task", bind=True)
def run_wav2lip_task(self, task_id, video_url, audio_url, use_hd=False):
    """
    Celery Task:
    Runs Wav2Lip based on `use_hd` flag,
    uploads the output video to S3, and returns the S3 URL.
    """

    logger.info(f"🎬 Starting Wav2Lip task: {task_id} | use_hd={use_hd}")
    
    # Get the lip_sync_service directory (current directory of this file)
    SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    video_path = os.path.join(SERVICE_DIR, f"inputs/{task_id}_video.mp4")
    audio_path = os.path.join(SERVICE_DIR, f"inputs/{task_id}_audio.wav")
    output_path = os.path.join(SERVICE_DIR, f"outputs/{task_id}_result.mp4")
    
    try:
        tasks_collection = mongo.db.lip_sync_tasks
        bucket_name = os.getenv("S3_BUCKET_NAME")

        # Initialize task
        task = LipSyncTask(task_id=task_id, video_url=video_url, audio_url=audio_url)
        task.mark_downloading()
        task.save(tasks_collection)

        # Ensure directories exist
        inputs_dir = os.path.join(SERVICE_DIR, "inputs")
        outputs_dir = os.path.join(SERVICE_DIR, "outputs")
        temp_dir = os.path.join(SERVICE_DIR, "temp")
        
        os.makedirs(inputs_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        # --- 1️⃣ DOWNLOAD INPUT FILES ---
        logger.info("📥 Downloading input files...")
        try:
            with open(video_path, "wb") as f:
                f.write(requests.get(video_url, timeout=60).content)
            with open(audio_path, "wb") as f:
                f.write(requests.get(audio_url, timeout=60).content)
            logger.info("✅ Input files downloaded successfully")
        except Exception as e:
            raise Exception(f"Failed to download input files: {e}")

        # --- 2️⃣ PROCESSING PHASE ---
        task.mark_processing()
        task.save(tasks_collection)

        python_executable = sys.executable
        
        # Determine Wav2Lip directory (sibling of tasks.py inside lip_sync_service)
        wav2lip_dir = os.path.join(SERVICE_DIR, "Wav2Lip")
        inference_script = os.path.join(wav2lip_dir, "inference.py")
        
        # Validate inference script exists
        if not os.path.exists(inference_script):
            raise Exception(f"Inference script not found at: {inference_script}")
        
        # Use absolute paths for checkpoint
        checkpoint_path = os.path.abspath(os.path.join(wav2lip_dir, "checkpoints/wav2lip.pth"))
        
        # Validate checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise Exception(f"Checkpoint not found at: {checkpoint_path}")
        
        # Build command with absolute paths for all inputs/outputs
        command = [
            python_executable,
            inference_script,
            "--checkpoint_path", checkpoint_path,
            "--face", os.path.abspath(video_path),
            "--audio", os.path.abspath(audio_path),
            "--outfile", os.path.abspath(output_path)
        ]
        
        logger.info(f"⚙️ Executing command: {' '.join(command)}")
        logger.info(f"Wav2Lip directory: {wav2lip_dir}")
        logger.info(f"Checkpoint: {checkpoint_path} (exists: {os.path.exists(checkpoint_path)})")
        logger.info(f"Video input: {video_path} (exists: {os.path.exists(video_path)})")
        logger.info(f"Audio input: {audio_path} (exists: {os.path.exists(audio_path)})")

        # Run subprocess from Wav2Lip directory
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=wav2lip_dir  # Important: set working directory to Wav2Lip folder
        )

        stdout_output, stderr_output = "", ""
        
        for line in process.stdout:
            logger.info(f"[Wav2Lip STDOUT] {line.strip()}")
            stdout_output += line
            
        for line in process.stderr:
            logger.warning(f"[Wav2Lip STDERR] {line.strip()}")
            stderr_output += line

        process.wait()
        
        if process.returncode != 0:
            raise Exception(f"Wav2Lip failed. Exit code: {process.returncode}\nStderr: {stderr_output}")
        
        if not os.path.exists(output_path):
            raise Exception(f"Output file not generated at: {output_path}")

        logger.info("✅ Processing completed successfully.")

        # --- 3️⃣ UPLOAD TO S3 ---
        s3_key = f"wav2lip_outputs/{task_id}_result.mp4"
        logger.info(f"📤 Uploading output to S3: s3://{bucket_name}/{s3_key}")

        try:
            s3.upload_file(output_path, bucket_name, s3_key)
            logger.info(f"✅ Uploaded successfully to S3")
        except Exception as e:
            raise Exception(f"Failed to upload to S3: {e}")

        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"

        # --- 4️⃣ MARK TASK COMPLETED ---
        task.mark_completed(s3_key=s3_key, output_s3_url=s3_url)
        task.error_log = stdout_output + "\n" + stderr_output
        task.save(tasks_collection)

        # --- CLEANUP ---
        clean_up(audio_path, video_path, output_path)

        logger.info(f"🏁 Task {task_id} completed successfully.")
        return {
            "success": True,
            "s3_url": s3_url,
            "model_used": "Wav2Lip"
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Task {task_id} failed: {error_msg}", exc_info=True)
        
        # --- CLEANUP ---
        clean_up(audio_path, video_path, output_path)
        
        # Update failure status
        try:
            mongo.db.lip_sync_tasks.update_one(
                {"task_id": task_id},
                {"$set": {
                    "status": "failed",
                    "error_log": error_msg,
                    "completed_at": datetime.utcnow()
                }}
            )
        except Exception as db_err:
            logger.error(f"Failed to update MongoDB: {db_err}")

        return {"success": False, "error": error_msg}