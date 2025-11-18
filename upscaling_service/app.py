from flask import Flask, request, jsonify
from flask_cors import CORS
from extensions import mongo, init_mongo
from models.lipSyncTasksModal import LipSyncTask
import subprocess
import os
import logging
from datetime import datetime
import threading
from dotenv import load_dotenv
import json
import time
from io import BytesIO

load_dotenv()
print("🔍 [DEBUG] app.py starting...")

app = Flask(__name__)
CORS(app)
init_mongo(app)

print("🔍 [DEBUG] Flask app initialized")
print(f"🔍 [DEBUG] MongoDB initialized: {mongo}")

# Get the upscaling_service directory
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"🔍 [DEBUG] SERVICE_DIR: {SERVICE_DIR}")

# Track active background threads
active_threads = set()
threads_lock = threading.Lock()

@app.teardown_appcontext
def teardown_mongo(exception):
    """Close MongoDB connection only if no background threads are active"""
    try:
        with threads_lock:
            if len(active_threads) > 0:
                print(f"🔍 [DEBUG] {len(active_threads)} background thread(s) still active - keeping MongoDB connection alive")
                return
        
        print("🔍 [DEBUG] No background threads active - MongoDB connection remains open")
    except Exception as e:
        print(f"🔍 [DEBUG] Error in teardown_mongo: {e}")

logger = logging.getLogger(__name__)

# Upscaling service configuration
UPSCALING_SERVICE_PORT = 5002
# Default 0.00s offset as configurable default (client request)
DEFAULT_AUDIO_OFFSET = 0.0
AUDIO_SYNC_OFFSETS = [-0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12]
IS_TESTING = os.getenv("IS_TESTING", "False").lower() == "true"
IS_BATCH_JOB = os.getenv("AWS_BATCH_JOB_ID") is not None

print(f"🔍 [DEBUG] Upscaling service port: {UPSCALING_SERVICE_PORT}")
print(f"🔍 [DEBUG] Testing mode: {IS_TESTING}")
print(f"🔍 [DEBUG] Running as AWS Batch Job: {IS_BATCH_JOB}")
print(f"🔍 [DEBUG] AWS Batch Job ID: {os.getenv('AWS_BATCH_JOB_ID', 'N/A')}")

# ========== MANIFEST & LOGGING ==========
def create_processing_manifest(task_id, input_info, model_info, output_urls, processing_time, vram_stats):
    """
    Create processing manifest for cost visibility and audit trail.
    Includes: inputs, outputs, model used, offset, timing, VRAM usage.
    """
    manifest = {
        "task_id": task_id,
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline": "Wav2Lip v1 + GFPGAN x2 + CRF17",
        "execution_environment": {
            "batch_job_id": os.getenv("AWS_BATCH_JOB_ID", "local"),
            "compute_environment": os.getenv("AWS_BATCH_COMPUTE_ENV", "local"),
            "instance_type": os.getenv("AWS_BATCH_INSTANCE_TYPE", "g6.xlarge"),
            "is_spot": os.getenv("AWS_BATCH_SPOT_INSTANCE", "unknown"),
        },
        "input": {
            "file_size_mb": input_info.get("file_size_mb", 0),
            "model": model_info.get("model_type", "wav2lip"),
            "checkpoint": model_info.get("checkpoint", "wav2lip.pth"),
        },
        "processing": {
            "step_1": {
                "name": "GFPGAN Face Enhancement",
                "upscale_factor": 2,
                "enabled": True,
            },
            "step_2": {
                "name": "FFmpeg Quality Encoding",
                "codec": "libx264",
                "crf": 17,
                "pixel_format": "yuv420p",
                "audio_codec": "aac",
                "audio_bitrate": "160k",
            },
            "step_3": {
                "name": "Audio Sync Variations",
                "default_offset": DEFAULT_AUDIO_OFFSET,
                "offsets_generated": AUDIO_SYNC_OFFSETS,
                "total_variations": len(output_urls),
            },
        },
        "output": {
            "total_files": len(output_urls),
            "urls": output_urls,
            "bucket": os.getenv("S3_BUCKET_NAME"),
            "prefix": f"upscaled_outputs/{task_id}",
        },
        "performance_metrics": {
            "total_processing_time_seconds": processing_time,
            "peak_vram_mb": vram_stats.get("peak_vram_mb", 0),
            "avg_vram_mb": vram_stats.get("avg_vram_mb", 0),
            "vram_sample_count": vram_stats.get("samples", 0),
            "estimated_gpu_minutes": round(processing_time / 60, 2),
        },
        "cost_tracking": {
            "gpu_type": "NVIDIA TESLA T4",
            "estimated_cost_usd": round((processing_time / 3600) * 0.35, 4),
            "resource_type": "AWS Batch Spot Instance",
        },
    }
    
    return manifest


def save_manifest_to_s3(manifest, task_id):
    """Save processing manifest to S3 for audit trail"""
    try:
        import boto3
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        
        bucket_name = os.getenv("S3_BUCKET_NAME")
        manifest_key = f"upscaled_outputs/{task_id}/manifest.json"
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2),
            ContentType="application/json"
        )
        
        manifest_url = f"https://{bucket_name}.s3.{os.getenv('AWS_REGION', 'us-east-1')}.amazonaws.com/{manifest_key}"
        print(f"🔍 [DEBUG] Manifest saved to S3: {manifest_url}")
        logger.info(f"✅ Processing manifest saved: {manifest_url}")
        
        return manifest_url
    except Exception as e:
        logger.warning(f"⚠️ Failed to save manifest to S3: {e}")
        return None


def log_batch_job_info():
    """Log AWS Batch job information for resource tracking"""
    batch_info = {
        "job_id": os.getenv("AWS_BATCH_JOB_ID"),
        "job_queue": os.getenv("AWS_BATCH_JOB_QUEUE"),
        "compute_environment": os.getenv("AWS_BATCH_COMPUTE_ENV"),
        "array_job_id": os.getenv("AWS_BATCH_ARRAY_JOB_ID"),
        "instance_type": os.getenv("AWS_BATCH_INSTANCE_TYPE", "g6.xlarge"),
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    logger.info(f"🖥️ AWS Batch Job Info: {json.dumps(batch_info, indent=2)}")
    print(f"🔍 [DEBUG] Batch Job: {batch_info['job_id']}")
    
    return batch_info

# ========== VRAM MONITORING ==========
peak_vram = 0
vram_samples = []
monitoring = False
vram_lock = threading.Lock()

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
def cleanup_files(file_paths):
    """Clean up local files to minimize disk usage (important for Batch jobs)"""
    print(f"🔍 [DEBUG] cleanup_files: {len(file_paths)} files")
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                print(f"🔍 [DEBUG] Deleted: {file_path} ({file_size / (1024**2):.2f} MB)")
        except Exception as e:
            print(f"🔍 [DEBUG] Failed to delete {file_path}: {str(e)}")
            logger.warning(f"Failed to cleanup {file_path}: {e}")


# ========== API ROUTES ==========




def get_upscaling_status(task_id):
    """
    Get upscaling status for a task including cost/resource info.
    Used internally for status checks.
    """
    print(f"🔍 [DEBUG] get_upscaling_status called with task_id: {task_id}")
    try:
        task_data = mongo.db.lip_sync_tasks.find_one({"task_id": task_id})
        print(f"🔍 [DEBUG] Task data found: {task_data is not None}")
        
        if not task_data:
            print(f"🔍 [DEBUG] Task {task_id} not found")
            return None
        
        task = LipSyncTask.from_dict(task_data)
        print(f"🔍 [DEBUG] Upscaling status: {task.upscaling_status}")
        
        return {
            "success": True,
            "task_id": task_id,
            "upscaling_status": task.upscaling_status,
            "upscaled_outputs": task.upscaled_output_urls,
            "error": task.upscaling_error,
            "started_at": task.upscaling_started_at,
            "completed_at": task.upscaling_completed_at,
            "batch_job_id": os.getenv("AWS_BATCH_JOB_ID", "local"),
            "pipeline_version": "v1-standard-gfpgan-crf17"
        }
        
    except Exception as e:
        print(f"🔍 [DEBUG] Exception in get_upscaling_status: {str(e)}")
        return None

# ========== PROCESSING ENGINE ==========

def run_upscale_task(task_id, video_file, model_type, app_context):
    """
    Run upscaling task with video data passed directly.
    Pipeline: GFPGAN (2x) → FFmpeg CRF17 → Audio Sync Variations → Upload to S3
    
    Optimized for Batch/Spot execution with cost tracking.
    No downloading - file is processed directly from request.
    """
    current_thread = threading.current_thread()
    overall_start_time = time.time()
    
    print(f"🔍 [DEBUG] run_upscale_task called for task_id: {task_id}")
    print(f"🔍 [DEBUG] Running in thread: {current_thread.name}")
    
    # Set up logging for this task
    logs_dir = os.path.join(SERVICE_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = os.path.join(logs_dir, f"upscale_{task_id}.log")
    print(f"🔍 [DEBUG] Log file: {log_filename}")

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    task_logger = logging.getLogger(f"upscale_{task_id}")
    task_logger.setLevel(logging.INFO)
    task_logger.addHandler(file_handler)
    task_logger.propagate = False

    task_logger.info(f"🎬 Starting upscaling task for {task_id}")
    task_logger.info(f"Pipeline: Wav2Lip v1 + GFPGAN x2 + CRF17 + Audio Sync Variations")
    task_logger.info(f"Model Type: {model_type}")
    task_logger.info(f"Batch Job ID: {os.getenv('AWS_BATCH_JOB_ID', 'local')}")
    task_logger.info(f"Default Audio Offset: {DEFAULT_AUDIO_OFFSET}s")
    print(f"🔍 [DEBUG] Task logger initialized")

    # Start VRAM monitoring
    monitor_thread = start_vram_monitoring()

    files_to_cleanup = []

    with app_context.app_context():
        try:
            # Create temp directories
            temp_dir = os.path.join(SERVICE_DIR, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save uploaded video file to temp location
            local_input_path = os.path.join(temp_dir, f"{task_id}_input.mp4")
            print(f"🔍 [DEBUG] Saving video to: {local_input_path}")
            
            # FIX: Handle different file object types properly
            if hasattr(video_file, 'save'):
                # Flask FileStorage object
                video_file.save(local_input_path)
            elif hasattr(video_file, 'read'):
                # File-like object (BufferedReader, BytesIO, etc.)
                # Reset to beginning if it's seekable
                if hasattr(video_file, 'seek'):
                    video_file.seek(0)
                
                # Read the file content and write to disk
                with open(local_input_path, 'wb') as f:
                    # Read in chunks to handle large files efficiently
                    chunk_size = 8192
                    while True:
                        chunk = video_file.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
            else:
                raise ValueError(f"Unsupported file object type: {type(video_file)}")
            
            input_size_mb = os.path.getsize(local_input_path) / (1024**2)
            print(f"🔍 [DEBUG] Video saved. Size: {input_size_mb:.2f} MB")
            task_logger.info(f"📥 Video received and saved: {input_size_mb:.2f} MB")
            
            # Prepare output paths (nested in upscaling_service)
            local_gfpgan_path = os.path.join(temp_dir, f"{task_id}_gfpgan.mp4")
            local_q17_path = os.path.join(temp_dir, f"{task_id}_q17.mp4")
            
            files_to_cleanup = [local_input_path, local_gfpgan_path, local_q17_path]

            print(f"🔍 [DEBUG] Starting upscaling pipeline...")
            
            # Run GFPGAN enhancement (2x upscale)
            print(f"🔍 [DEBUG] Starting GFPGAN enhancement (2x)...")
            task_logger.info(f"⚙️ GFPGAN Enhancement: 2x upscale factor")
            upscale_video_with_gfpgan(local_input_path, local_gfpgan_path, upscale_factor=2, logger=task_logger)
            print(f"🔍 [DEBUG] GFPGAN enhancement completed")

            # Apply FFmpeg quality enhancement (CRF 17)
            print(f"🔍 [DEBUG] Applying FFmpeg quality enhancement (CRF 17)...")
            task_logger.info(f"📹 FFmpeg Quality Encoding: CRF 17, libx264, yuv420p")
            apply_ffmpeg_quality(local_gfpgan_path, local_q17_path, task_logger)
            print(f"🔍 [DEBUG] FFmpeg quality enhancement completed")

            # Generate audio sync variations with default offset
            print(f"🔍 [DEBUG] Generating audio sync variations...")
            task_logger.info(f"🔊 Audio Sync Variations: Generating {len(AUDIO_SYNC_OFFSETS)} offsets")
            upscaled_urls = apply_audio_sync_offsets(local_q17_path, task_id, task_logger)
            print(f"🔍 [DEBUG] Generated {len(upscaled_urls)} sync variations")

            if not upscaled_urls:
                raise Exception("No upscaled videos were generated")

            # Calculate processing time and VRAM stats
            overall_end_time = time.time()
            total_processing_time = overall_end_time - overall_start_time
            vram_stats = stop_vram_monitoring()

            # Create processing manifest
            model_info = {
                "model_type": model_type,
                "checkpoint": f"{model_type}.pth",
            }
            input_info = {
                "file_size_mb": input_size_mb
            }
            manifest = create_processing_manifest(
                task_id, 
                input_info,
                model_info,
                upscaled_urls,
                total_processing_time,
                vram_stats
            )

            # Save manifest to S3
            manifest_url = save_manifest_to_s3(manifest, task_id)

            # Update task with upscaled URLs
            print(f"🔍 [DEBUG] Marking task as upscaling completed with {len(upscaled_urls)} URLs")
            task_data = mongo.db.lip_sync_tasks.find_one({"task_id": task_id})
            task = LipSyncTask.from_dict(task_data)
            task.mark_upscaling_completed(upscaled_urls)
            
            # Store manifest URL in task
            if manifest_url:
                task.manifest_url = manifest_url
            
            task.save(mongo.db.lip_sync_tasks)
            print(f"🔍 [DEBUG] Task saved to database")

            # Log summary
            task_logger.info(f"🎉 Upscaling completed successfully!")
            task_logger.info(f"📊 Summary:")
            task_logger.info(f"  - Input size: {input_size_mb:.2f} MB")
            task_logger.info(f"  - Output files: {len(upscaled_urls)}")
            task_logger.info(f"  - Processing time: {total_processing_time:.2f} seconds ({total_processing_time/60:.2f} minutes)")
            task_logger.info(f"  - Peak VRAM: {vram_stats['peak_vram_mb']:.0f} MB")
            task_logger.info(f"  - Estimated GPU cost: ${manifest['cost_tracking']['estimated_cost_usd']}")
            task_logger.info(f"  - Batch Job ID: {os.getenv('AWS_BATCH_JOB_ID', 'local')}")
            task_logger.info(f"  - Manifest: {manifest_url}")
            
            print(f"🔍 [DEBUG] Upscaling completed successfully")

        except Exception as e:
            print(f"🔍 [DEBUG] Exception in run_upscale_task: {str(e)}")
            task_logger.error(f"❌ Upscaling failed: {e}", exc_info=True)

            # Update task with failure
            try:
                task_data = mongo.db.lip_sync_tasks.find_one({"task_id": task_id})
                if task_data:
                    task = LipSyncTask.from_dict(task_data)
                    task.mark_upscaling_failed(str(e))
                    task.save(mongo.db.lip_sync_tasks)
                    print(f"🔍 [DEBUG] Task failure recorded in database")
            except Exception as db_error:
                print(f"🔍 [DEBUG] Failed to update task failure in DB: {str(db_error)}")
                task_logger.error(f"Failed to record failure in database: {db_error}")
        
        finally:
            # Stop VRAM monitoring
            stop_vram_monitoring()
            
            # Cleanup local files
            cleanup_files(files_to_cleanup)
            
            # Remove thread from active list
            with threads_lock:
                active_threads.discard(current_thread)
                print(f"🔍 [DEBUG] Thread removed from active threads. Total active: {len(active_threads)}")
                print(f"🔍 [DEBUG] Upscaling task for {task_id} completed")

# ========== HELPER FUNCTIONS ==========

def upscale_video_with_gfpgan(input_path, output_path, upscale_factor=2, logger=None):
    """
    Run GFPGAN enhancement on video using enhance.py script.
    enhance.py is located in SERVICE_DIR
    Standard pipeline: 2x upscale factor
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    print(f"🔍 [DEBUG] upscale_video_with_gfpgan: input={input_path}, upscale={upscale_factor}")
    try:
        enhance_script = os.path.join(SERVICE_DIR, "enhance.py")
        print(f"🔍 [DEBUG] Using enhance script: {enhance_script}")
        print(f"🔍 [DEBUG] Script exists: {os.path.exists(enhance_script)}")

        command = [
            "python", enhance_script,
            "--input", input_path,
            "--output", output_path,
            "--model", "gfpgan",
            "--upscale", str(upscale_factor),
            "--gpu-id", "0"
        ]

        logger.info(f"Running GFPGAN with upscale factor {upscale_factor}x...")
        print(f"🔍 [DEBUG] Running command: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=SERVICE_DIR
        )

        print(f"🔍 [DEBUG] Process started with PID: {process.pid}")
        
        # Read output line by line
        try:
            for line in process.stdout:
                line = line.rstrip('\n')
                if line:
                    print(f"🔍 [GFPGAN] {line}")
                    logger.debug(f"[GFPGAN] {line}")
        except:
            pass
        
        process.wait(timeout=3600)
        
        print(f"🔍 [DEBUG] Process completed with return code: {process.returncode}")

        if process.returncode != 0:
            raise Exception(f"GFPGAN enhancement failed with return code {process.returncode}")

        print(f"🔍 [DEBUG] Checking if output file exists: {output_path}")
        if not os.path.exists(output_path):
            raise Exception("Upscaled video not created")

        file_size = os.path.getsize(output_path)
        print(f"🔍 [DEBUG] Output file size: {file_size / (1024**2):.2f} MB")
        logger.info(f"✅ GFPGAN completed: {file_size / (1024**2):.2f} MB")

    except subprocess.TimeoutExpired:
        print(f"🔍 [DEBUG] GFPGAN process timeout, killing...")
        process.kill()
        logger.error("GFPGAN timeout after 1 hour")
        raise Exception("GFPGAN enhancement timed out")
    except Exception as e:
        print(f"🔍 [DEBUG] Exception in upscale_video_with_gfpgan: {str(e)}")
        logger.error(f"GFPGAN failed: {e}")
        raise

def apply_ffmpeg_quality(input_path, output_path, logger=None):
    """
    Apply FFmpeg quality enhancement (CRF 17, libx264, yuv420p).
    Standard production encoding parameters.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    print(f"🔍 [DEBUG] apply_ffmpeg_quality: input={input_path}, output={output_path}")
    try:
        command = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-crf", "17",
            "-preset", "slow",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "160k",
            output_path
        ]

        logger.info("Encoding with FFmpeg: libx264, CRF 17, yuv420p")
        print(f"🔍 [DEBUG] Running FFmpeg quality command")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(timeout=1800)
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg failed: {stderr}")

        if not os.path.exists(output_path):
            raise Exception("Quality-enhanced video not created")

        file_size = os.path.getsize(output_path)
        print(f"🔍 [DEBUG] FFmpeg completed. Size: {file_size / (1024**2):.2f} MB")
        logger.info(f"✅ FFmpeg CRF17 completed: {file_size / (1024**2):.2f} MB")

    except subprocess.TimeoutExpired:
        print(f"🔍 [DEBUG] FFmpeg timeout, killing...")
        process.kill()
        logger.error("FFmpeg timeout after 30 minutes")
        raise Exception("FFmpeg quality enhancement timed out")
    except Exception as e:
        print(f"🔍 [DEBUG] Exception in apply_ffmpeg_quality: {str(e)}")
        logger.error(f"FFmpeg failed: {e}")
        raise

def apply_audio_sync_offsets(input_path, task_id, logger=None):
    """
    Generate audio sync variations with different offsets.
    Default offset is 0.00s (as per client request).
    All offsets uploaded to S3.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    print(f"🔍 [DEBUG] apply_audio_sync_offsets: task_id={task_id}")
    
    upscaled_urls = []
    bucket_name = os.getenv("S3_BUCKET_NAME")
    
    if not bucket_name:
        raise Exception("S3_BUCKET_NAME environment variable not set")
    
    temp_dir = os.path.join(SERVICE_DIR, "temp")
    
    for idx, offset in enumerate(AUDIO_SYNC_OFFSETS):
        try:
            # Format offset for filename
            offset_str = f"{offset:.2f}".replace("-", "m").replace(".", "p") + "s"
            local_sync_path = os.path.join(temp_dir, f"{task_id}_sync_{idx}_{offset_str}.mp4")
            
            is_default = abs(offset - DEFAULT_AUDIO_OFFSET) < 0.001
            offset_label = f"{offset}s (DEFAULT)" if is_default else f"{offset}s"
            
            print(f"🔍 [DEBUG] Processing offset {offset_label}: {local_sync_path}")
            
            command = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-itsoffset", str(offset),
                "-i", input_path,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "160k",
                local_sync_path
            ]

            logger.info(f"Creating audio sync variation: {offset_label}")
            print(f"🔍 [DEBUG] Running FFmpeg sync for offset {offset_label}")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stdout, stderr = process.communicate(timeout=600)
            
            if process.returncode != 0:
                logger.warning(f"Audio sync offset {offset_label} failed: {stderr}")
                print(f"🔍 [DEBUG] Sync variation failed, continuing...")
                continue

            if not os.path.exists(local_sync_path):
                logger.warning(f"Sync file not created for offset {offset_label}")
                continue

            # Upload to S3
            s3_key = f"upscaled_outputs/{task_id}/model_0_sync_{offset_str}.mp4"
            upscaled_url = upload_to_s3(local_sync_path, s3_key, logger)
            upscaled_urls.append(upscaled_url)
            
            file_size = os.path.getsize(local_sync_path)
            print(f"🔍 [DEBUG] Uploaded {offset_label}: {upscaled_url}")
            logger.info(f"✅ Audio sync {offset_label} uploaded: {file_size / (1024**2):.2f} MB")

            # Cleanup sync file immediately after upload
            try:
                os.remove(local_sync_path)
                print(f"🔍 [DEBUG] Cleaned up: {local_sync_path}")
            except:
                pass

        except subprocess.TimeoutExpired:
            print(f"🔍 [DEBUG] Timeout for offset {offset}, skipping...")
            logger.warning(f"Audio sync offset {offset}s timed out")
        except Exception as e:
            print(f"🔍 [DEBUG] Error processing offset {offset}: {str(e)}")
            logger.warning(f"Error processing offset {offset}s: {e}")

    print(f"🔍 [DEBUG] Generated {len(upscaled_urls)} sync variations")
    return upscaled_urls

def upload_to_s3(local_path, s3_key, logger=None):
    """Upload file to S3 and return public URL"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    print(f"🔍 [DEBUG] upload_to_s3: s3_key={s3_key}")
    
    try:
        import boto3
        
        # Get AWS credentials
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        bucket_name = os.getenv("S3_BUCKET_NAME")
        
        if not all([aws_access_key, aws_secret_key, bucket_name]):
            raise Exception("Missing AWS credentials in environment variables")
        
        # Create S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        # Validate file exists
        if not os.path.exists(local_path):
            raise Exception(f"Local file does not exist: {local_path}")
        
        file_size = os.path.getsize(local_path)
        print(f"🔍 [DEBUG] Uploading {file_size / (1024**2):.2f} MB to S3...")
        
        # Upload file
        s3_client.upload_file(
            local_path,
            bucket_name,
            s3_key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        
        print(f"🔍 [DEBUG] Upload completed successfully")
        
        # Generate public URL
        url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_key}"
        print(f"🔍 [DEBUG] Generated S3 URL: {url}")
        logger.debug(f"S3 URL: {url}")
        
        return url
    
    except Exception as e:
        print(f"🔍 [DEBUG] S3 upload error: {str(e)}")
        logger.error(f"S3 upload failed: {e}")
        raise



# ========== STARTUP ==========

if __name__ == '__main__':
    print("🔍 [DEBUG] Starting Flask app...")
    print("\n" + "="*80)
    print("📦 WAV2LIP UPSCALING SERVICE v1 - PRODUCTION SETUP")
    print("="*80)
    print("\n🖥️ DEPLOYMENT INFO:")
    print("   Compute Environment: AWS Batch")
    print("   Instance Type: g6.xlarge (NVIDIA T4 GPU)")
    print("   Execution Mode: Job-based (Spot instances only)")
    print("   Always-on Server: NO")
    print("\n📁 SERVICE STRUCTURE:")
    print(f"   Service Directory: {SERVICE_DIR}")
    print(f"   enhance.py location: {os.path.join(SERVICE_DIR, 'enhance.py')}")
    print(f"   Logs directory: {os.path.join(SERVICE_DIR, 'logs')}")
    print(f"   Temp directory: {os.path.join(SERVICE_DIR, 'temp')}")
    print("\n🔧 PIPELINE:")
    print("   1. Receive video file directly from tasks.py")
    print("   2. GFPGAN Enhancement (2x upscale)")
    print("   3. FFmpeg Quality Encoding (CRF 17)")
    print("   4. Audio Sync Variations (7 offsets)")
    print("   5. Upload to S3 + Generate Manifest")
    print("\n⚙️ CONFIGURATION:")
    print(f"   Default Audio Offset: {DEFAULT_AUDIO_OFFSET}s")
    print(f"   GFPGAN Upscale Factor: 2x")
    print(f"   FFmpeg CRF: 17")
    print(f"   Audio Sync Variations: {len(AUDIO_SYNC_OFFSETS)}")
    print("\n📊 COST TRACKING:")
    print("   Manifest saved to: S3 upscaled_outputs/{task_id}/manifest.json")
    print("   Includes: inputs, outputs, model, offset, timing, VRAM, cost")
    print("\n🔐 ENVIRONMENT VARIABLES REQUIRED:")
    print("   AWS_ACCESS_KEY_ID")
    print("   AWS_SECRET_ACCESS_KEY")
    print("   AWS_REGION (default: us-east-1)")
    print("   S3_BUCKET_NAME")
    print("   MONGODB_URI")
    print("   AWS_BATCH_JOB_ID (auto-populated by Batch)")
    print("   AWS_BATCH_COMPUTE_ENV (auto-populated by Batch)")
    print("\n🔗 INTEGRATION:")
    print("   Called directly from: tasks.py (parent folder)")
    print("   Method: trigger_upscaling_with_output()")
    print("   No HTTP endpoints - direct function call")
    print("\n" + "="*80)
    print(f"✅ Service initialized (not running Flask server)")
    print("="*80 + "\n")