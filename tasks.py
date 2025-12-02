import os
import sys
import requests
from celery_config import celery
from dotenv import load_dotenv
from extensions import mongo
from datetime import datetime
import subprocess
import shutil
import boto3
from models.lipSyncTasksModal import LipSyncTask
import logging
import threading
import time
import json

load_dotenv()
logger = logging.getLogger(__name__)

# Global variables for performance monitoring
peak_vram = 0
vram_samples = []
monitoring = False
vram_lock = threading.Lock()

# S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

# ========== VRAM MONITORING ==========
def monitor_vram_usage():
    """Monitor VRAM usage in background thread"""
    global peak_vram, vram_samples, monitoring, vram_lock
    
    while monitoring:
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                vram_used = float(result.stdout.strip().split('\n')[0].strip())
                
                with vram_lock:
                    vram_samples.append(vram_used)
                    peak_vram = max(peak_vram, vram_used)
            else:
                logger.debug("nvidia-smi returned no output or failed")
        except Exception as e:
            logger.debug(f"VRAM monitoring error: {e}")
        
        time.sleep(0.1)


def get_total_gpu_vram():
    """Get total GPU VRAM available"""
    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=memory.total --format=csv,nounits,noheader",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip().split('\n')[0].strip())
        else:
            logger.warning("nvidia-smi returned no output for total VRAM")
            return 0
    except Exception as e:
        logger.warning(f"Failed to get total GPU VRAM: {e}")
        return 0


def start_vram_monitoring():
    """Start background VRAM monitoring"""
    global peak_vram, vram_samples, monitoring
    
    peak_vram = 0
    vram_samples = []
    monitoring = True
    
    monitor_thread = threading.Thread(target=monitor_vram_usage, daemon=True)
    monitor_thread.start()
    time.sleep(0.2)
    
    return monitor_thread


def stop_vram_monitoring():
    """Stop VRAM monitoring and return statistics"""
    global peak_vram, vram_samples, monitoring, vram_lock
    
    monitoring = False
    time.sleep(0.2)
    
    with vram_lock:
        avg_vram = sum(vram_samples) / len(vram_samples) if vram_samples else 0
        peak_vram_value = peak_vram
        sample_count = len(vram_samples)
    
    return {
        "peak_vram_mb": peak_vram_value,
        "avg_vram_mb": avg_vram,
        "samples": sample_count
    }

# ========== FILE OPERATIONS ==========
def clean_up(file_paths):
    """Clean up multiple files"""
    for f in file_paths:
        try:
            if f and os.path.exists(f):
                os.remove(f)
                logger.debug(f"Cleaned up: {f}")
        except Exception as e:
            logger.warning(f"Cleanup failed for {f}: {e}")


def get_video_framerate_info(video_path):
    """Get video frame rate information using ffprobe"""
    logger.info(f"📊 Probing video frame rate info: {video_path}")
    
    # Preflight checks
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None

    # Ensure ffprobe is available
    ffprobe_exe = os.getenv("FFPROBE_PATH") or shutil.which("ffprobe")
    if not ffprobe_exe:
        logger.error("ffprobe executable not found. Please install FFmpeg and make sure 'ffprobe' is on PATH or set FFPROBE_PATH environment variable.")
        return None

    try:
        result = subprocess.run([
            ffprobe_exe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate,codec_time_base",
            "-of", "default=nw=1",
            video_path
        ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            logger.info(f"FFprobe output:\n{result.stdout}")
            return result.stdout
        else:
            logger.warning(f"FFprobe failed: {result.stderr}")
            return None
    except Exception as e:
        logger.warning(f"Failed to probe video: {e}")
        return None


# ========== VIDEO/AUDIO PREPROCESSING ==========
def convert_video_to_cfr25(input_video, output_video):
    """Convert face video to constant frame rate 25fps with yuv420p pixel format"""
    logger.info("🎥 Converting video to constant frame rate 25fps with CFR encoding...")
    
    # preflight: check input exists
    if not os.path.exists(input_video):
        raise Exception(f"Input video file not found: {input_video}")

    # find ffmpeg
    ffmpeg_exe = os.getenv("FFMPEG_PATH") or shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise Exception("'ffmpeg' not found. Please install FFmpeg and add it to PATH or set FFMPEG_PATH environment variable.")

    command = [
        ffmpeg_exe,
        "-y",
        "-i", input_video,
        "-r", "25",
        "-vsync", "1",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-crf", "18",
        "-an",
        output_video
    ]
    
    logger.info(f"⚙️ Executing: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        stdout_list = []
        stderr_list = []
        
        def read_output(pipe, output_list):
            try:
                for line in iter(pipe.readline, ''):
                    line = line.strip()
                    if line:
                        logger.debug(f"[FFmpeg-Video] {line}")
                        output_list.append(line + "\n")
            except Exception as e:
                logger.error(f"Error reading ffmpeg output: {e}")
        
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_list), daemon=True)
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_list), daemon=True)
        
        stdout_thread.start()
        stderr_thread.start()
        
        process.wait(timeout=600)
        
        if process.returncode != 0:
            raise Exception(f"Video conversion failed. Exit code: {process.returncode}")
        
        if not os.path.exists(output_video):
            raise Exception(f"Output video not generated at: {output_video}")
        
        logger.info("✅ Video converted to CFR25 successfully.")
        return True
        
    except subprocess.TimeoutExpired:
        process.kill()
        raise Exception("Video conversion timed out after 10 minutes")
    except Exception as e:
        logger.error(f"Video conversion failed: {e}")
        raise


def resample_audio_to_16k(input_audio, output_audio):
    """Resample audio to 16 kHz mono (Wav2Lip expects this)"""
    logger.info("🔊 Resampling audio to 16 kHz mono...")
    
    if not os.path.exists(input_audio):
        raise Exception(f"Input audio file not found: {input_audio}")

    ffmpeg_exe = os.getenv("FFMPEG_PATH") or shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise Exception("'ffmpeg' not found. Please install FFmpeg and add it to PATH or set FFMPEG_PATH environment variable.")

    command = [
        ffmpeg_exe,
        "-y",
        "-i", input_audio,
        "-ac", "1",
        "-ar", "16000",
        output_audio
    ]
    
    logger.info(f"⚙️ Executing: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        stdout_list = []
        stderr_list = []
        
        def read_output(pipe, output_list):
            try:
                for line in iter(pipe.readline, ''):
                    line = line.strip()
                    if line:
                        logger.debug(f"[FFmpeg-Audio] {line}")
                        output_list.append(line + "\n")
            except Exception as e:
                logger.error(f"Error reading ffmpeg output: {e}")
        
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_list), daemon=True)
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_list), daemon=True)
        
        stdout_thread.start()
        stderr_thread.start()
        
        process.wait(timeout=300)
        
        if process.returncode != 0:
            raise Exception(f"Audio resampling failed. Exit code: {process.returncode}")
        
        if not os.path.exists(output_audio):
            raise Exception(f"Output audio not generated at: {output_audio}")
        
        logger.info("✅ Audio resampled to 16 kHz mono successfully.")
        return True
        
    except subprocess.TimeoutExpired:
        process.kill()
        raise Exception("Audio resampling timed out after 5 minutes")
    except Exception as e:
        logger.error(f"Audio resampling failed: {e}")
        raise

# ========== S3 OPERATIONS ==========
def upload_to_s3(local_file, s3_key, bucket_name):
    """Upload file to S3 and return the URL"""
    logger.info(f"📤 Uploading to S3: s3://{bucket_name}/{s3_key}")
    
    try:
        s3.upload_file(local_file, bucket_name, s3_key)
        
        bucket_location = s3.get_bucket_location(Bucket=bucket_name)
        region = bucket_location['LocationConstraint']
        
        if region is None:
            region = 'us-east-1'
        elif region == 'EU':
            region = 'eu-west-1'
            
        s3_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
        logger.info(f"✅ Uploaded successfully: {s3_url}")
        return s3_url
    except Exception as e:
        raise Exception(f"Failed to upload to S3: {e}")

# ========== WAV2LIP INFERENCE ==========
def run_inference_with_retry(python_executable, inference_script, checkpoint_path, video_path, audio_path, output_path, wav2lip_dir, model_type="non-gan", max_retries=2):
    """Run Wav2Lip inference with retry logic and progressive memory optimization"""
    
    resize_factors = [1, 2, 4]
    
    for attempt, resize_factor in enumerate(resize_factors[:max_retries + 1]):
        try:
            logger.info(f"🔄 Attempt {attempt + 1}/{len(resize_factors[:max_retries + 1])} with resize_factor={resize_factor}")
            stdout, stderr, wall_time, vram_stats = run_inference(
                python_executable, inference_script, checkpoint_path,
                video_path, audio_path, output_path, wav2lip_dir,
                model_type, resize_factor
            )
            return stdout, stderr, wall_time, vram_stats
        except Exception as e:
            error_str = str(e)
            if "ArrayMemoryError" in error_str or "out of memory" in error_str.lower() or "CUDA out of memory" in error_str:
                if attempt < len(resize_factors[:max_retries + 1]) - 1:
                    logger.warning(f"⚠️ Memory error on attempt {attempt + 1}. Retrying with higher resize_factor...")
                    try:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                    except:
                        pass
                    continue
            raise e


def run_inference(python_executable, inference_script, checkpoint_path, video_path, audio_path, output_path, wav2lip_dir, model_type="non-gan", resize_factor=1):
    """Run Wav2Lip inference with model-specific parameters and memory optimization"""
    
    logger.info(f"🎬 Starting {model_type.upper()} inference with resize_factor={resize_factor}...")
    
    monitor_thread = start_vram_monitoring()
    inference_start = time.time()
    
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
        "--wav2lip_batch_size", "16",
        "--face_det_batch_size", "4",
        "--outfile", os.path.abspath(output_path)
    ]
    
    logger.info(f"⚙️ Executing command: {' '.join(command)}")
    logger.info(f"Memory optimization: resize_factor={resize_factor}, limiting threads")
    
    env = os.environ.copy()
    env['NUMEXPR_MAX_THREADS'] = '2'
    env['OMP_NUM_THREADS'] = '2'
    
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
            for line in iter(pipe.readline, ''):
                line = line.strip()
                if line:
                    logger.info(f"[{prefix}] {line}")
                    output_list.append(line + "\n")
        except Exception as e:
            logger.error(f"Error reading {prefix}: {e}")

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
    
    try:
        process.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        process.kill()
        raise Exception(f"Wav2Lip {model_type} inference timed out after 1 hour")
    
    vram_stats = stop_vram_monitoring()
    inference_end = time.time()
    wall_clock_time = inference_end - inference_start
    
    stdout_output = "".join(stdout_list)
    stderr_output = "".join(stderr_list)
    
    if process.returncode != 0:
        raise Exception(f"Wav2Lip {model_type} failed. Exit code: {process.returncode}\nStderr: {stderr_output}")
    
    if not os.path.exists(output_path):
        raise Exception(f"Output file not generated at: {output_path}")
    
    logger.info(f"✅ {model_type.upper()} inference completed successfully.")
    logger.info(f"⏱️  Wall-clock time: {wall_clock_time:.2f}s | Peak VRAM: {vram_stats['peak_vram_mb']:.0f}MB")
    
    return stdout_output, stderr_output, wall_clock_time, vram_stats


def enhance_gan_output(python_executable, input_video, output_video):
    """Enhance GAN output with higher-quality encoding (CRF 17)"""
    logger.info("🎥 Enhancing GAN output with H.264 encoding...")
    
    if not os.path.exists(input_video):
        raise Exception(f"GAN input video not found: {input_video}")

    ffmpeg_exe = os.getenv("FFMPEG_PATH") or shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise Exception("'ffmpeg' not found. Please install FFmpeg and add it to PATH or set FFMPEG_PATH environment variable.")

    command = [
        ffmpeg_exe,
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
            for line in iter(pipe.readline, ''):
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
    
    try:
        process.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        process.kill()
        raise Exception("FFmpeg encoding timed out after 30 minutes")
    
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    
    if process.returncode != 0:
        raise Exception(f"FFmpeg encoding failed. Exit code: {process.returncode}")
    
    if not os.path.exists(output_video):
        raise Exception(f"Enhanced output file not generated at: {output_video}")
    
    logger.info("✅ GAN output enhanced successfully.")

# ========== UPSCALING INTEGRATION ==========
def trigger_upscaling_with_output(task_id, output_video_path, model_type="wav2lip"):
    """
    Direct integration with upscaling service.
    Calls the upscaling function directly (no HTTP endpoint).
    
    Returns: success boolean
    """
    try:
        logger.info(f"🚀 Preparing to upscale task {task_id}...")
        
        # Get task from database
        task_data = mongo.db.lip_sync_tasks.find_one({"task_id": task_id})
        if not task_data:
            logger.error(f"Task {task_id} not found in database")
            return False
        
        task = LipSyncTask.from_dict(task_data)
        
        if not os.path.exists(output_video_path):
            logger.error(f"Output video file not found: {output_video_path}")
            return False
        
        file_size_mb = os.path.getsize(output_video_path) / (1024**2)
        logger.info(f"📤 Sending to upscaling service: {file_size_mb:.2f} MB")
        
        # Import upscaling service
        from upscaling_service.upscaling_app import run_upscale_task, app as upscale_app
        
        # Update task status
        task.mark_upscaling_started()
        task.save(mongo.db.lip_sync_tasks)
        
        # Read the entire file content and create a BytesIO object
        with open(output_video_path, 'rb') as f:
            video_content = f.read()
        
        # Create a BytesIO object that can be properly handled
        from io import BytesIO
        video_file_obj = BytesIO(video_content)
        
        # Call upscaling directly in background thread
        thread = threading.Thread(
            target=run_upscale_task, 
            args=(task_id, video_file_obj, model_type, upscale_app)
        )
        thread.daemon = False
        thread.start()
        
        logger.info(f"✅ Upscaling triggered directly for task {task_id}")
        return True
            
    except Exception as e:
        logger.error(f"❌ Failed to trigger upscaling: {e}")
        return False

# ========== CELERY TASKS ==========
# Register this function as a Celery task so it can be queued via apply_async
@celery.task(name="run_wav2lip_task", bind=True)
def run_wav2lip_task(self, task_id, video_url, audio_url, enable_upscaling=True, enable_gan=False):
    """
    Celery Task: Complete Wav2Lip pipeline with direct upscaling integration
    
    Flow:
    1. Download video & audio
    2. Preprocess (CFR25, 16k audio)
    3. Run Wav2Lip (non-GAN, and optionally GAN)
    4. Upload outputs to S3
    5. Update database with S3 URLs
    6. Trigger upscaling service with output URLs
    7. Upscaling service handles further enhancement and re-uploads
    """
    logger.info(f"🎬 Starting Wav2Lip task: {task_id} | GAN Enabled: {enable_gan}")
    
    SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    video_path = os.path.join(SERVICE_DIR, f"inputs/{task_id}_video.mp4")
    audio_path = os.path.join(SERVICE_DIR, f"inputs/{task_id}_audio.wav")
    video_cfr_path = os.path.join(SERVICE_DIR, f"inputs/{task_id}_video_cfr25.mp4")
    audio_16k_path = os.path.join(SERVICE_DIR, f"inputs/{task_id}_audio_16k.wav")
    output_w2l = os.path.join(SERVICE_DIR, f"outputs/{task_id}_w2l.mp4")
    output_gan = os.path.join(SERVICE_DIR, f"outputs/{task_id}_gan.mp4")
    output_gan_enhanced = os.path.join(SERVICE_DIR, f"outputs/{task_id}_gan_q17.mp4")
    
    results = {}
    files_to_cleanup = []
    
    try:
        tasks_collection = mongo.db.lip_sync_tasks
        bucket_name = os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            raise Exception("S3_BUCKET_NAME environment variable is not set")
        
        task = LipSyncTask(task_id=task_id, video_url=video_url, audio_url=audio_url)
        task.mark_downloading()
        task.save(tasks_collection)
        
        os.makedirs(os.path.join(SERVICE_DIR, "inputs"), exist_ok=True)
        os.makedirs(os.path.join(SERVICE_DIR, "outputs"), exist_ok=True)
        
        # --- DOWNLOAD INPUT FILES ---
        logger.info("📥 Downloading input files...")
        for path, url in [(video_path, video_url), (audio_path, audio_url)]:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            files_to_cleanup.append(path)
        logger.info("✅ Input files downloaded")
        
        # --- PREPROCESS INPUT FILES ---
        logger.info("=" * 60)
        logger.info("PREPROCESSING VIDEO AND AUDIO")
        logger.info("=" * 60)
        get_video_framerate_info(video_path)
        convert_video_to_cfr25(video_path, video_cfr_path)
        resample_audio_to_16k(audio_path, audio_16k_path)
        files_to_cleanup.extend([video_cfr_path, audio_16k_path])
        logger.info("✅ Preprocessing completed")
        
        task.mark_processing()
        task.save(tasks_collection)
        
        python_executable = sys.executable
        wav2lip_dir = os.path.join(SERVICE_DIR, "Wav2Lip")
        inference_script = os.path.join(wav2lip_dir, "inference.py")
        checkpoint_w2l = os.path.abspath(os.path.join(wav2lip_dir, "checkpoints/wav2lip.pth"))
        checkpoint_gan = os.path.abspath(os.path.join(wav2lip_dir, "checkpoints/wav2lip_gan.pth"))
        
        # --- RUN NON-GAN MODEL ---
        logger.info("=" * 60)
        logger.info("RUNNING WAV2LIP NON-GAN MODEL")
        logger.info("=" * 60)
        stdout_w2l, stderr_w2l, time_w2l, vram_w2l = run_inference_with_retry(
            python_executable, inference_script, checkpoint_w2l,
            video_cfr_path, audio_16k_path, output_w2l, wav2lip_dir, "non-gan", max_retries=2
        )
        files_to_cleanup.append(output_w2l)
        
        results['non_gan'] = {
            "s3_key": f"wav2lip_outputs/{task_id}/w2l_output.mp4",
            "model": "Wav2Lip",
            "local_path": output_w2l
        }
        
        # --- RUN GAN MODEL IF ENABLED ---
        if enable_gan:
            logger.info("=" * 60)
            logger.info("RUNNING WAV2LIP GAN MODEL")
            logger.info("=" * 60)
            stdout_gan, stderr_gan, time_gan, vram_gan = run_inference_with_retry(
                python_executable, inference_script, checkpoint_gan,
                video_cfr_path, audio_16k_path, output_gan, wav2lip_dir, "gan", max_retries=2
            )
            files_to_cleanup.append(output_gan)
            
            try:
                logger.info("=" * 60)
                logger.info("ENHANCING GAN OUTPUT")
                logger.info("=" * 60)
                enhance_gan_output(python_executable, output_gan, output_gan_enhanced)
                gan_output_for_upload = output_gan_enhanced
                files_to_cleanup.append(output_gan_enhanced)
            except Exception as e:
                logger.warning(f"GAN enhancement failed: {e}. Using raw GAN output.")
                gan_output_for_upload = output_gan
            
            results['gan'] = {
                "s3_key": f"wav2lip_outputs/{task_id}/gan_output.mp4",
                "model": "Wav2Lip-GAN",
                "local_path": gan_output_for_upload
            }
        
        # --- TRIGGER UPSCALING ---
        if enable_upscaling:
            logger.info("=" * 60)
            logger.info("TRIGGERING UPSCALING SERVICE")
            logger.info("=" * 60)
            
            # Use the non-GAN output for upscaling by default
            primary_output = output_w2l
            trigger_upscaling_with_output(task_id, primary_output, model_type="wav2lip")
            
            # Now upload Wav2Lip outputs to S3 after upscaling is triggered
            logger.info("Uploading Wav2Lip outputs to S3...")
            s3_url_w2l = upload_to_s3(output_w2l, results['non_gan']['s3_key'], bucket_name)
            results['non_gan']['s3_url'] = s3_url_w2l
        #     output_urls.append(s3_url_w2l)
            
        #     if enable_gan:
        #         s3_url_gan = upload_to_s3(gan_output_for_upload, results['gan']['s3_key'], bucket_name)
        #         results['gan']['s3_url'] = s3_url_gan
        #         output_urls.append(s3_url_gan)
        # else:
        #     # Just upload without triggering upscaling
        #     s3_url_w2l = upload_to_s3(output_w2l, results['non_gan']['s3_key'], bucket_name)
        #     results['non_gan']['s3_url'] = s3_url_w2l
        #     output_urls.append(s3_url_w2l)
            
        #     if enable_gan:
        #         s3_url_gan = upload_to_s3(gan_output_for_upload, results['gan']['s3_key'], bucket_name)
        #         results['gan']['s3_url'] = s3_url_gan
        #         output_urls.append(s3_url_gan)
        
        # # --- UPDATE DATABASE ---
        # logger.info("=" * 60)
        # logger.info("UPDATING DATABASE")
        # logger.info("=" * 60)
        
        # task.mark_completed(
        #     s3_keys=[r['s3_key'] for r in results.values()],
        #     output_s3_urls=output_urls,
        #     models=[r['model'] for r in results.values()]
        # )
        
        # logger.info(f"🏁 Task {task_id} completed successfully!")
        # logger.info(f"Generated {len(output_urls)} output(s)")
        
        return {
            "success": True,
            "task_id": task_id,
            "results": results,
            "upscaling_triggered": enable_upscaling,
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Task {task_id} failed: {error_msg}", exc_info=True)
        clean_up(files_to_cleanup)
        
        try:
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
        
        return {"success": False, "task_id": task_id, "error": error_msg}
    
    finally:
        clean_up([video_path, audio_path, video_cfr_path, audio_16k_path])


@celery.task(name="run_single_wav2lip_task", bind=True)
def run_single_wav2lip_task(self, task_id, video_url, audio_url, model_type="non-gan", enable_upscaling=True):
    """
    Run a single Wav2Lip model with preprocessing and direct upscaling integration
    
    Args:
        task_id: Unique task identifier
        video_url: URL to download video from
        audio_url: URL to download audio from
        model_type: "non-gan" or "gan"
        enable_upscaling: Whether to trigger upscaling after completion
    
    Flow:
    1. Download and preprocess video/audio
    2. Run single Wav2Lip model (non-gan or gan)
    3. Upload output to S3
    4. Update database
    5. Trigger upscaling if enabled
    """
    
    logger.info(f"🎬 Starting Single {model_type.upper()} task: {task_id}")
    
    SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    video_path = os.path.join(SERVICE_DIR, f"inputs/{task_id}_video.mp4")
    audio_path = os.path.join(SERVICE_DIR, f"inputs/{task_id}_audio.wav")
    video_cfr_path = os.path.join(SERVICE_DIR, f"inputs/{task_id}_video_cfr25.mp4")
    audio_16k_path = os.path.join(SERVICE_DIR, f"inputs/{task_id}_audio_16k.wav")
    output_path = os.path.join(SERVICE_DIR, f"outputs/{task_id}_result.mp4")
    output_enhanced = os.path.join(SERVICE_DIR, f"outputs/{task_id}_result_q17.mp4")
    
    files_to_cleanup = []
    
    try:
        tasks_collection = mongo.db.lip_sync_tasks
        bucket_name = os.getenv("S3_BUCKET_NAME")
        
        if not bucket_name:
            raise Exception("S3_BUCKET_NAME environment variable is not set")
        
        task = LipSyncTask(task_id=task_id, video_url=video_url, audio_url=audio_url)
        task.mark_downloading()
        task.save(tasks_collection)
        
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # --- DOWNLOAD INPUT FILES ---
        logger.info("=" * 60)
        logger.info("DOWNLOADING INPUT FILES")
        logger.info("=" * 60)
        logger.info("📥 Downloading video and audio...")
        
        video_response = requests.get(video_url, timeout=60)
        video_response.raise_for_status()
        with open(video_path, "wb") as f:
            f.write(video_response.content)
        logger.info(f"✅ Video downloaded: {os.path.getsize(video_path) / (1024**2):.2f} MB")
            
        audio_response = requests.get(audio_url, timeout=60)
        audio_response.raise_for_status()
        with open(audio_path, "wb") as f:
            f.write(audio_response.content)
        logger.info(f"✅ Audio downloaded: {os.path.getsize(audio_path) / (1024**2):.2f} MB")
        
        files_to_cleanup.extend([video_path, audio_path])
        
        # --- PREPROCESS INPUT FILES ---
        logger.info("=" * 60)
        logger.info("PREPROCESSING VIDEO AND AUDIO")
        logger.info("=" * 60)
        
        get_video_framerate_info(video_path)
        convert_video_to_cfr25(video_path, video_cfr_path)
        resample_audio_to_16k(audio_path, audio_16k_path)
        files_to_cleanup.extend([video_cfr_path, audio_16k_path])
        
        logger.info("✅ Input preprocessing completed successfully")
        
        task.mark_processing()
        task.save(tasks_collection)
        
        # --- PREPARE INFERENCE ---
        python_executable = sys.executable
        wav2lip_dir = os.path.join(SERVICE_DIR, "Wav2Lip")
        inference_script = os.path.join(wav2lip_dir, "inference.py")
        
        checkpoint = os.path.abspath(os.path.join(
            wav2lip_dir,
            "checkpoints/wav2lip.pth" if model_type == "non-gan" else "checkpoints/wav2lip_gan.pth"
        ))
        
        if not os.path.exists(checkpoint):
            raise Exception(f"Checkpoint not found: {checkpoint}")
        
        # --- RUN INFERENCE ---
        logger.info("=" * 60)
        logger.info(f"RUNNING WAV2LIP {model_type.upper()} MODEL")
        logger.info("=" * 60)
        
        stdout_output, stderr_output, wall_time, vram_stats = run_inference_with_retry(
            python_executable, inference_script, checkpoint,
            video_cfr_path, audio_16k_path, output_path, wav2lip_dir, model_type
        )
        
        files_to_cleanup.append(output_path)
        
        # --- OPTIONAL ENHANCEMENT FOR GAN ---
        output_for_upload = output_path
        
        if model_type == "gan":
            try:
                logger.info("=" * 60)
                logger.info("ENHANCING GAN OUTPUT")
                logger.info("=" * 60)
                enhance_gan_output(python_executable, output_path, output_enhanced)
                output_for_upload = output_enhanced
                files_to_cleanup.append(output_enhanced)
            except Exception as e:
                logger.warning(f"⚠️ Enhancement failed: {e}. Using raw output.")
                output_for_upload = output_path
        
        # --- UPLOAD TO S3 ---
        logger.info("=" * 60)
        logger.info("UPLOADING TO S3")
        logger.info("=" * 60)
        
        s3_key = f"wav2lip_outputs/{task_id}/{model_type}_output.mp4"
        s3_url = upload_to_s3(output_for_upload, s3_key, bucket_name)
        
        # --- UPLOAD TO S3 ---
        logger.info("=" * 60)
        logger.info("UPLOADING TO S3")
        logger.info("=" * 60)
        
        s3_key = f"wav2lip_outputs/{task_id}/{model_type}_output.mp4"
        s3_url = upload_to_s3(output_for_upload, s3_key, bucket_name)
        
        # --- TRIGGER UPSCALING ---
        if enable_upscaling:
            logger.info("=" * 60)
            logger.info("TRIGGERING UPSCALING SERVICE")
            logger.info("=" * 60)
            
            trigger_upscaling_with_output(task_id, output_for_upload, model_type=model_type)
        
        # --- UPDATE DATABASE ---
        logger.info("=" * 60)
        logger.info("UPDATING DATABASE")
        logger.info("=" * 60)
        
        total_gpu_vram = get_total_gpu_vram()
        
        task.mark_completed(
            s3_keys=[s3_key],
            output_s3_urls=[s3_url],
            models=[f"Wav2Lip-{model_type.upper()}"]
        )
        
        task.performance_snapshot = {
            "model": model_type,
            "wall_clock_time_seconds": wall_time,
            "peak_vram_mb": vram_stats['peak_vram_mb'],
            "avg_vram_mb": vram_stats['avg_vram_mb'],
            "total_gpu_vram_mb": total_gpu_vram,
            "vram_usage_percent": (vram_stats['peak_vram_mb'] / total_gpu_vram * 100) if total_gpu_vram > 0 else 0,
            "timestamp": datetime.utcnow()
        }
        
        task.save(tasks_collection)
        logger.info(f"✅ Task saved to database with S3 URL: {s3_url}")
        
        logger.info(f"🏁 Task {task_id} completed successfully!")
        
        return {
            "success": True,
            "task_id": task_id,
            "model": model_type,
            "s3_url": s3_url,
            "upscaling_triggered": enable_upscaling,
            "performance": task.performance_snapshot
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Task {task_id} failed: {error_msg}", exc_info=True)
        
        clean_up(files_to_cleanup)
        
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
        
        return {"success": False, "task_id": task_id, "error": error_msg}
    
    finally:
        clean_up([video_path, audio_path, video_cfr_path, audio_16k_path])