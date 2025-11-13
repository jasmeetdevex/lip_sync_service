# tasks.py
import os
import sys
import requests
from celery_config import celery
from dotenv import load_dotenv
from extensions import mongo
from datetime import datetime
import subprocess
import boto3
from models.lipSyncTasksModal import LipSyncTask
import logging
import threading

load_dotenv()
logger = logging.getLogger(__name__)

# S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

def clean_up(file_paths):
    """Clean up multiple files"""
    for f in file_paths:
        try:
            if f and os.path.exists(f):
                os.remove(f)
        except Exception as e:
            logger.warning(f"Cleanup failed for {f}: {e}")


def run_inference_with_retry(python_executable, inference_script, checkpoint_path, video_path, audio_path, output_path, wav2lip_dir, model_type="non-gan", max_retries=2):
    """
    Run Wav2Lip inference with retry logic and progressive memory optimization.
    Retries with increased resize_factor if memory errors occur.
    """
    
    resize_factors = [1, 2, 4]  # Start with better quality, fall back to more memory efficient
    
    for attempt, resize_factor in enumerate(resize_factors[:max_retries + 1]):
        try:
            logger.info(f"🔄 Attempt {attempt + 1}/{len(resize_factors[:max_retries + 1])} with resize_factor={resize_factor}")
            stdout, stderr = run_inference(
                python_executable, inference_script, checkpoint_path,
                video_path, audio_path, output_path, wav2lip_dir,
                model_type, resize_factor
            )
            return stdout, stderr
        except Exception as e:
            error_str = str(e)
            if "ArrayMemoryError" in error_str or "out of memory" in error_str.lower():
                if attempt < len(resize_factors[:max_retries + 1]) - 1:
                    logger.warning(f"⚠️ Memory error on attempt {attempt + 1}. Retrying with higher resize_factor...")
                    # Clean up partial output if it exists
                    try:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                    except:
                        pass
                    continue
            raise


def run_inference(python_executable, inference_script, checkpoint_path, video_path, audio_path, output_path, wav2lip_dir, model_type="non-gan", resize_factor=1):
    """
    Run Wav2Lip inference with enhanced parameters and memory optimization
    
    Args:
        model_type: "non-gan" or "gan"
    """
    
    logger.info(f"🎬 Starting {model_type.upper()} inference...")
    
    # Enhanced parameters for better output with memory optimization
    command = [
        python_executable,
        inference_script,
        "--checkpoint_path", checkpoint_path,
        "--face", os.path.abspath(video_path),
        "--audio", os.path.abspath(audio_path),
        "--fps", "25",
        "--resize_factor", str(resize_factor),
        "--pads", "0", "10", "0", "0",
        "--nosmooth",
        "--outfile", os.path.abspath(output_path)
    ]
    
    logger.info(f"⚙️ Executing command: {' '.join(command)}")
    logger.info(f"Memory optimization: resize_factor=2, batch_size=64")
    
    # Set environment variables for memory optimization
    env = os.environ.copy()
    env['NUMEXPR_MAX_THREADS'] = '2'  # Limit thread usage
    env['OMP_NUM_THREADS'] = '2'
    
    # Run subprocess from Wav2Lip directory with memory optimization
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=wav2lip_dir,
        env=env
    )

    stdout_output = ""
    stderr_output = ""

    def read_output(pipe, output_list, prefix):
        try:
            for line in pipe:
                line = line.strip()
                if line:
                    logger.info(f"[{prefix}] {line}")
                    output_list.append(line + "\n")
        except Exception as e:
            logger.error(f"Error reading {prefix}: {e}")

    # Read stdout and stderr simultaneously with threading
    stdout_list = []
    stderr_list = []
    
    stdout_thread = threading.Thread(
        target=read_output, 
        args=(process.stdout, stdout_list, "Wav2Lip STDOUT"), 
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=read_output, 
        args=(process.stderr, stderr_list, "Wav2Lip STDERR"), 
        daemon=True
    )
    
    stdout_thread.start()
    stderr_thread.start()
    process.wait()
    
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    
    stdout_output = "".join(stdout_list)
    stderr_output = "".join(stderr_list)
    
    if process.returncode != 0:
        raise Exception(f"Wav2Lip {model_type} failed. Exit code: {process.returncode}\nStderr: {stderr_output}")
    
    if not os.path.exists(output_path):
        raise Exception(f"Output file not generated at: {output_path}")
    
    logger.info(f"✅ {model_type.upper()} inference completed successfully.")
    return stdout_output, stderr_output


def enhance_gan_output(python_executable, input_video, output_video):
    """
    Enhance GAN output with higher-quality encoding
    Applies CRF 17 for better crispness
    """
    logger.info("🎥 Enhancing GAN output with H.264 encoding...")
    
    command = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-c:v", "libx264",
        "-crf", "17",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "160k",
        output_video
    ]
    
    logger.info(f"⚙️ Executing ffmpeg: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    stdout_output = ""
    stderr_output = ""
    
    def read_output(pipe, output_list):
        try:
            for line in pipe:
                line = line.strip()
                if line:
                    logger.debug(f"[FFmpeg] {line}")
                    output_list.append(line + "\n")
        except Exception as e:
            logger.error(f"Error reading ffmpeg output: {e}")
    
    stdout_list = []
    stderr_list = []
    
    stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_list), daemon=True)
    stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_list), daemon=True)
    
    stdout_thread.start()
    stderr_thread.start()
    process.wait()
    
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    
    if process.returncode != 0:
        raise Exception(f"FFmpeg encoding failed. Exit code: {process.returncode}")
    
    if not os.path.exists(output_video):
        raise Exception(f"Enhanced output file not generated at: {output_video}")
    
    logger.info("✅ GAN output enhanced successfully.")


def upload_to_s3(local_file, s3_key, bucket_name):
    """Upload file to S3 and return the URL"""
    logger.info(f"📤 Uploading to S3: s3://{bucket_name}/{s3_key}")
    
    try:
        s3.upload_file(local_file, bucket_name, s3_key)
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        logger.info(f"✅ Uploaded successfully: {s3_url}")
        return s3_url
    except Exception as e:
        raise Exception(f"Failed to upload to S3: {e}")


@celery.task(name="run_wav2lip_task", bind=True)
def run_wav2lip_task(self, task_id, video_url, audio_url):
    """
    Celery Task:
    Runs both Wav2Lip (Non-GAN) and Wav2Lip-GAN models,
    uploads both outputs to S3, and stores URLs in database.
    
    Returns both S3 URLs for comparison.
    """
    
    logger.info(f"🎬 Starting Dual Wav2Lip task: {task_id}")
    
    # Get the lip_sync_service directory
    SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    video_path = os.path.join(SERVICE_DIR, f"inputs/{task_id}_video.mp4")
    audio_path = os.path.join(SERVICE_DIR, f"inputs/{task_id}_audio.wav")
    output_w2l = os.path.join(SERVICE_DIR, f"outputs/{task_id}_w2l.mp4")
    output_gan = os.path.join(SERVICE_DIR, f"outputs/{task_id}_gan.mp4")
    output_gan_enhanced = os.path.join(SERVICE_DIR, f"outputs/{task_id}_gan_q17.mp4")
    
    results = {}
    files_to_cleanup = []
    
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
        
        os.makedirs(inputs_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)
        
        # --- 1️⃣ DOWNLOAD INPUT FILES ---
        logger.info("📥 Downloading input files...")
        try:
            with open(video_path, "wb") as f:
                f.write(requests.get(video_url, timeout=60).content)
            with open(audio_path, "wb") as f:
                f.write(requests.get(audio_url, timeout=60).content)
            
            files_to_cleanup.extend([video_path, audio_path])
            logger.info("✅ Input files downloaded successfully")
        except Exception as e:
            raise Exception(f"Failed to download input files: {e}")
        
        # --- 2️⃣ PROCESSING PHASE ---
        task.mark_processing()
        task.save(tasks_collection)
        
        python_executable = sys.executable
        wav2lip_dir = os.path.join(SERVICE_DIR, "Wav2Lip")
        inference_script = os.path.join(wav2lip_dir, "inference.py")
        checkpoint_w2l = os.path.abspath(os.path.join(wav2lip_dir, "checkpoints/wav2lip.pth"))
        checkpoint_gan = os.path.abspath(os.path.join(wav2lip_dir, "checkpoints/wav2lip_gan.pth"))
        
        # Validate paths
        if not os.path.exists(inference_script):
            raise Exception(f"Inference script not found at: {inference_script}")
        if not os.path.exists(checkpoint_w2l):
            raise Exception(f"Wav2Lip checkpoint not found at: {checkpoint_w2l}")
        if not os.path.exists(checkpoint_gan):
            raise Exception(f"Wav2Lip-GAN checkpoint not found at: {checkpoint_gan}")
        
        logger.info(f"Using Wav2Lip directory: {wav2lip_dir}")
        
        # --- 3️⃣ RUN NON-GAN MODEL ---
        logger.info("=" * 60)
        logger.info("RUNNING NON-GAN MODEL (Wav2Lip)")
        logger.info("=" * 60)
        
        stdout_w2l, stderr_w2l = run_inference_with_retry(
            python_executable, inference_script, checkpoint_w2l,
            video_path, audio_path, output_w2l, wav2lip_dir, "non-gan", max_retries=2
        )
        files_to_cleanup.append(output_w2l)
        
        # --- 4️⃣ RUN GAN MODEL ---
        logger.info("=" * 60)
        logger.info("RUNNING GAN MODEL (Wav2Lip-GAN)")
        logger.info("=" * 60)
        
        stdout_gan, stderr_gan = run_inference_with_retry(
            python_executable, inference_script, checkpoint_gan,
            video_path, audio_path, output_gan, wav2lip_dir, "gan", max_retries=2
        )
        files_to_cleanup.append(output_gan)
        
        # --- 5️⃣ ENHANCE GAN OUTPUT ---
        logger.info("=" * 60)
        logger.info("ENHANCING GAN OUTPUT WITH HIGH-QUALITY ENCODING")
        logger.info("=" * 60)
        
        try:
            enhance_gan_output(python_executable, output_gan, output_gan_enhanced)
            files_to_cleanup.append(output_gan_enhanced)
            gan_output_for_upload = output_gan_enhanced
        except Exception as e:
            logger.warning(f"GAN enhancement failed: {e}. Using non-enhanced version.")
            gan_output_for_upload = output_gan
        
        # --- 6️⃣ UPLOAD BOTH OUTPUTS TO S3 ---
        logger.info("=" * 60)
        logger.info("UPLOADING OUTPUTS TO S3")
        logger.info("=" * 60)
        
        s3_key_w2l = f"wav2lip_outputs/{task_id}/w2l_output.mp4"
        s3_key_gan = f"wav2lip_outputs/{task_id}/gan_output.mp4"
        
        s3_url_w2l = upload_to_s3(output_w2l, s3_key_w2l, bucket_name)
        s3_url_gan = upload_to_s3(gan_output_for_upload, s3_key_gan, bucket_name)
        
        results = {
            "non_gan": {
                "s3_key": s3_key_w2l,
                "s3_url": s3_url_w2l,
                "model": "Wav2Lip"
            },
            "gan": {
                "s3_key": s3_key_gan,
                "s3_url": s3_url_gan,
                "model": "Wav2Lip-GAN"
            }
        }
        
        # --- 7️⃣ UPDATE DATABASE ---
        logger.info("💾 Updating database with results...")
        
        task.mark_completed(
            s3_keys=[s3_key_w2l, s3_key_gan],
            output_s3_urls=[s3_url_w2l, s3_url_gan],
            models=["Wav2Lip", "Wav2Lip-GAN"]
        )
        task.error_log = {
            "w2l_stdout": stdout_w2l,
            "w2l_stderr": stderr_w2l,
            "gan_stdout": stdout_gan,
            "gan_stderr": stderr_gan
        }
        task.save(tasks_collection)
        
        logger.info(f"🏁 Task {task_id} completed successfully with both models!")
        
        return {
            "success": True,
            "task_id": task_id,
            "results": results,
            "message": "Both Wav2Lip and Wav2Lip-GAN outputs generated and uploaded"
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Task {task_id} failed: {error_msg}", exc_info=True)
        
        # Cleanup on failure
        clean_up(files_to_cleanup)
        
        # Update failure status
        try:
            tasks_collection = mongo.db.lip_sync_tasks
            tasks_collection.update_one(
                {"task_id": task_id},
                {"$set": {
                    "status": "failed",
                    "error_log": error_msg,
                    "completed_at": datetime.utcnow()
                }}
            )
        except Exception as db_err:
            logger.error(f"Failed to update MongoDB: {db_err}")
        
        return {
            "success": False,
            "task_id": task_id,
            "error": error_msg
        }
    
    finally:
        # Clean up input files
        clean_up([video_path, audio_path])


# Optional: Add a task to run only specific model if needed
@celery.task(name="run_single_wav2lip_task", bind=True)
def run_single_wav2lip_task(self, task_id, video_url, audio_url, model_type="non-gan"):
    """
    Run a single Wav2Lip model (non-gan or gan)
    
    Args:
        model_type: "non-gan" or "gan"
    """
    
    logger.info(f"🎬 Starting Single {model_type.upper()} task: {task_id}")
    
    SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    video_path = os.path.join(SERVICE_DIR, f"inputs/{task_id}_video.mp4")
    audio_path = os.path.join(SERVICE_DIR, f"inputs/{task_id}_audio.wav")
    output_path = os.path.join(SERVICE_DIR, f"outputs/{task_id}_result.mp4")
    
    files_to_cleanup = []
    
    try:
        tasks_collection = mongo.db.lip_sync_tasks
        bucket_name = os.getenv("S3_BUCKET_NAME")
        
        task = LipSyncTask(task_id=task_id, video_url=video_url, audio_url=audio_url)
        task.mark_downloading()
        task.save(tasks_collection)
        
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download inputs
        logger.info("📥 Downloading input files...")
        with open(video_path, "wb") as f:
            f.write(requests.get(video_url, timeout=60).content)
        with open(audio_path, "wb") as f:
            f.write(requests.get(audio_url, timeout=60).content)
        
        files_to_cleanup.extend([video_path, audio_path])
        
        task.mark_processing()
        task.save(tasks_collection)
        
        # Prepare paths
        python_executable = sys.executable
        wav2lip_dir = os.path.join(SERVICE_DIR, "Wav2Lip")
        inference_script = os.path.join(wav2lip_dir, "inference.py")
        
        checkpoint = os.path.abspath(os.path.join(
            wav2lip_dir,
            "checkpoints/wav2lip.pth" if model_type == "non-gan" else "checkpoints/wav2lip_gan.pth"
        ))
        
        if not os.path.exists(checkpoint):
            raise Exception(f"Checkpoint not found: {checkpoint}")
        
        # Run inference
        stdout_output, stderr_output = run_inference(
            python_executable, inference_script, checkpoint,
            video_path, audio_path, output_path, wav2lip_dir, model_type
        )
        
        files_to_cleanup.append(output_path)
        
        # Optional GAN enhancement
        if model_type == "gan":
            enhanced_path = output_path.replace(".mp4", "_q17.mp4")
            try:
                enhance_gan_output(python_executable, output_path, enhanced_path)
                output_for_upload = enhanced_path
                files_to_cleanup.append(enhanced_path)
            except Exception as e:
                logger.warning(f"Enhancement failed: {e}")
                output_for_upload = output_path
        else:
            output_for_upload = output_path
        
        # Upload to S3
        s3_key = f"wav2lip_outputs/{task_id}/output.mp4"
        s3_url = upload_to_s3(output_for_upload, s3_key, bucket_name)
        
        # Update database
        task.mark_completed(s3_key=s3_key, output_s3_url=s3_url)
        task.error_log = {"stdout": stdout_output, "stderr": stderr_output}
        task.save(tasks_collection)
        
        logger.info(f"🏁 Task {task_id} completed successfully!")
        
        return {
            "success": True,
            "task_id": task_id,
            "s3_url": s3_url,
            "model": model_type
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Task {task_id} failed: {error_msg}", exc_info=True)
        
        clean_up(files_to_cleanup)
        
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
        
        return {"success": False, "task_id": task_id, "error": error_msg}
    
    finally:
        clean_up([video_path, audio_path])