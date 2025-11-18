"""
Video Enhancement Script for Wav2Lip Output
Enhances blurry lip-synced videos using face restoration models
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import os
import logging
import sys
from datetime import datetime

# You'll need to install these packages:
# pip install opencv-python
# pip install basicsr
# pip install facexlib
# pip install gfpgan
# pip install realesrgan
# pip install boto3 (optional, for S3 upload)

from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import torch

# Optional boto3 import for S3 uploads
try:
    import boto3
    from botocore.exceptions import ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


# Configure logging
def setup_logging():
    """Setup logging to both console and file"""
    log_format = '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    
    # Force UTF-8 encoding on Windows
    if hasattr(console_handler, 'stream'):
        import io
        console_handler.stream = io.TextIOWrapper(
            sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else sys.stdout,
            encoding='utf-8',
            errors='replace'
        )
    
    # Root logger
    logger = logging.getLogger('enhance')
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


class S3Uploader:
    """Handle S3 uploads for enhanced videos"""
    
    def __init__(self):
        """Initialize S3 client with credentials from environment"""
        try:
            if not HAS_BOTO3:
                logger.warning("[S3] boto3 not installed - S3 upload disabled")
                logger.warning("[S3] Install with: pip install boto3")
                self.s3_client = None
                return
            
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            self.bucket_name = os.getenv("S3_BUCKET_NAME")
            self.aws_region = aws_region
            
            logger.info("-" * 80)
            logger.info("[S3] Initializing S3 Uploader")
            logger.info("[S3] AWS Region: " + aws_region)
            logger.info("[S3] S3 Bucket: " + (self.bucket_name or "NOT SET"))
            
            if not all([aws_access_key, aws_secret_key, self.bucket_name]):
                logger.warning("[S3] WARNING: Missing AWS credentials - S3 upload disabled")
                self.s3_client = None
                return
            
            logger.info("[S3] Creating S3 client...")
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            logger.info("[S3] S3 client initialized successfully")
            logger.info("-" * 80)
        
        except Exception as e:
            logger.error(f"[S3] Failed to initialize S3 client: {str(e)}")
            self.s3_client = None
    
    def upload_file(self, local_path, s3_key):
        """
        Upload file to S3 and return the public URL
        
        Args:
            local_path: Path to local file
            s3_key: S3 object key (path in bucket)
        
        Returns:
            S3 URL if successful, None otherwise
        """
        if self.s3_client is None:
            logger.warning("[S3] S3 client not available, skipping upload")
            return None
        
        try:
            if not os.path.exists(local_path):
                raise Exception(f"Local file does not exist: {local_path}")
            
            file_size = os.path.getsize(local_path)
            file_size_mb = file_size / (1024 ** 2)
            
            logger.info("[S3] Starting S3 upload")
            logger.info(f"[S3] Local file: {local_path}")
            logger.info(f"[S3] File size: {file_size_mb:.2f} MB")
            logger.info(f"[S3] S3 key: {s3_key}")
            logger.info(f"[S3] S3 bucket: {self.bucket_name}")
            
            # Upload to S3
            logger.info("[S3] Uploading to S3...")
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={'ContentType': 'video/mp4'}
            )
            
            logger.info("[S3] Upload completed successfully")
            
            # Generate public URL
            url = f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{s3_key}"
            logger.info(f"[S3] S3 URL: {url}")
            
            return url
        
        except Exception as e:
            error_name = type(e).__name__
            if error_name == 'ClientError':
                logger.error(f"[S3] AWS error during upload: {e}")
            else:
                logger.error(f"[S3] Error uploading to S3: {str(e)}")
            return None


class VideoEnhancer:
    def __init__(self, model_type='gfpgan', upscale=2, gpu_id=0):
        """
        Initialize the video enhancer
        
        Args:
            model_type: 'gfpgan', 'codeformer', or 'realesrgan'
            upscale: upscale factor (1, 2, or 4)
            gpu_id: GPU device ID (default: 0)
        """
        logger.info("=" * 80)
        logger.info("[INIT] Initializing VideoEnhancer")
        logger.info("=" * 80)
        
        self.model_type = model_type
        self.upscale = upscale
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.restorer = None
        self.s3_uploader = S3Uploader()
        
        logger.info(f"[MODEL] Model Type: {model_type}")
        logger.info(f"[SCALE] Upscale Factor: {upscale}x")
        logger.info(f"[DEVICE] Device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"[GPU] GPU Name: {torch.cuda.get_device_name(gpu_id)}")
            logger.info(f"[CUDA] CUDA Version: {torch.version.cuda}")
            logger.info(f"[CUDA] CUDA Available: Yes")
        else:
            logger.warning("[CUDA] WARNING - CUDA not available - using CPU (slower)")
        
        self._load_model()
    
    def _load_model(self):
        """Load the enhancement model"""
        logger.info("-" * 80)
        logger.info(f"[LOAD] Loading {self.model_type.upper()} model...")
        logger.info("-" * 80)
        
        try:
            # Use FP16 (half precision) only if CUDA is available
            use_half = torch.cuda.is_available()
            logger.info(f"[FP16] Using FP16 (half precision): {use_half}")
            
            if self.model_type == 'gfpgan':
                logger.info("[INIT] Initializing GFPGAN model...")
                model_path = 'weights/GFPGANv1.4.pth'
                
                # Check if model exists
                if not os.path.exists(model_path):
                    logger.error(f"[ERROR] Model not found at {model_path}")
                    raise FileNotFoundError(f"Model weights not found at {model_path}")
                
                logger.info(f"[OK] Model weights found at {model_path}")
                
                logger.info("[INIT] Initializing RealESRGAN background upsampler...")
                # Create RealESRGAN upsampler as background enhancer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                              num_block=23, num_grow_ch=32, scale=2)
                
                bg_model_path = 'weights/RealESRGAN_x2plus.pth'
                if not os.path.exists(bg_model_path):
                    logger.warning(f"[WARN] Background upsampler not found at {bg_model_path}, skipping...")
                    bg_upsampler = None
                else:
                    logger.info(f"[OK] Background upsampler found at {bg_model_path}")
                    bg_upsampler = RealESRGANer(
                        scale=2,
                        model_path=bg_model_path,
                        model=model,
                        tile=400,
                        tile_pad=10,
                        pre_pad=0,
                        half=use_half,
                        device=self.device
                    )
                    logger.info("[OK] Background upsampler initialized")
                
                self.restorer = GFPGANer(
                    model_path=model_path,
                    upscale=self.upscale,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=bg_upsampler,
                    device=self.device
                )
                logger.info("[OK] GFPGAN model initialized successfully")
            
            elif self.model_type == 'realesrgan':
                logger.info("[INIT] Initializing RealESRGAN model...")
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_block=23, num_grow_ch=32, scale=self.upscale)
                
                model_path = f'weights/RealESRGAN_x{self.upscale}plus.pth'
                
                if not os.path.exists(model_path):
                    logger.error(f"[ERROR] Model not found at {model_path}")
                    raise FileNotFoundError(f"Model weights not found at {model_path}")
                
                self.restorer = RealESRGANer(
                    scale=self.upscale,
                    model_path=model_path,
                    model=model,
                    tile=400,
                    tile_pad=10,
                    pre_pad=0,
                    half=use_half,
                    device=self.device
                )
                logger.info("[OK] RealESRGAN model initialized successfully")
            
            logger.info("=" * 80)
            logger.info("[SUCCESS] Model loaded successfully!")
            logger.info("=" * 80)
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model: {str(e)}", exc_info=True)
            raise
    
    def enhance_frame(self, frame):
        """
        Enhance a single frame
        
        Args:
            frame: input frame (numpy array)
        
        Returns:
            enhanced frame
        """
        if self.model_type == 'gfpgan':
            # GFPGAN returns: cropped_faces, restored_faces, restored_img
            _, _, output = self.restorer.enhance(
                frame, 
                has_aligned=False, 
                only_center_face=False, 
                paste_back=True,
                weight=0.5  # Adjust blend weight (0-1)
            )
        else:
            # Real-ESRGAN
            output, _ = self.restorer.enhance(frame, outscale=self.upscale)
        
        return output
    
    def enhance_video(self, input_path, output_path, start_frame=0, end_frame=None):
        """
        Enhance entire video
        
        Args:
            input_path: path to input video
            output_path: path to save enhanced video
            start_frame: frame to start from (default: 0)
            end_frame: frame to end at (default: None = end of video)
        """
        logger.info("=" * 80)
        logger.info("[VIDEO] Starting video enhancement")
        logger.info("=" * 80)
        logger.info(f"[INPUT] {input_path}")
        logger.info(f"[OUTPUT] {output_path}")
        
        try:
            # Open input video
            logger.info("[LOAD] Opening input video...")
            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
                logger.error(f"[ERROR] Failed to open video file: {input_path}")
                raise Exception(f"Cannot open video file: {input_path}")
            
            logger.info("[OK] Video file opened successfully")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info("-" * 80)
            logger.info("[PROPS] Video Properties:")
            logger.info(f"[PROPS] Resolution: {width}x{height}")
            logger.info(f"[PROPS] FPS: {fps}")
            logger.info(f"[PROPS] Total Frames: {total_frames}")
            logger.info(f"[PROPS] Duration: {total_frames/fps:.2f}s")
            logger.info("-" * 80)
            
            if end_frame is None:
                end_frame = total_frames
            
            logger.info(f"[PROCESS] Processing frames {start_frame} to {end_frame}")
            
            # Prepare output video writer
            output_width = width * self.upscale
            output_height = height * self.upscale
            
            logger.info(f"[OUTPUT] Output Resolution: {output_width}x{output_height} (upscaled {self.upscale}x)")
            
            logger.info("[INIT] Initializing video writer...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (output_width, output_height))
            
            if not out.isOpened():
                logger.error("[ERROR] Failed to create output video writer")
                raise Exception("Cannot create output video writer")
            
            logger.info("[OK] Video writer initialized")
            logger.info("-" * 80)
            
            # Process frames
            frame_count = 0
            processed_count = 0
            start_time = datetime.now()
            
            logger.info("[PROCESS] Processing frames...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_count < start_frame:
                    frame_count += 1
                    continue
                
                if frame_count >= end_frame:
                    break
                
                # Enhance frame
                enhanced_frame = self.enhance_frame(frame)
                
                # Write enhanced frame
                out.write(enhanced_frame)
                
                processed_count += 1
                
                # Log progress every 10 frames
                if processed_count % 10 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    fps_actual = processed_count / elapsed if elapsed > 0 else 0
                    remaining_frames = (end_frame - start_frame) - processed_count
                    eta_seconds = remaining_frames / fps_actual if fps_actual > 0 else 0
                    
                    progress_pct = 100 * processed_count / (end_frame - start_frame)
                    logger.info(f"[PROGRESS] {processed_count}/{end_frame - start_frame} frames ({progress_pct:.1f}%) | "
                               f"Speed: {fps_actual:.2f} fps | ETA: {eta_seconds:.0f}s")
                
                frame_count += 1
            
            # Release resources
            cap.release()
            out.release()
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("-" * 80)
            logger.info("[SUCCESS] Video enhancement completed!")
            logger.info(f"[STATS] Total frames processed: {processed_count}")
            logger.info(f"[STATS] Total time: {elapsed_time:.2f}s")
            logger.info(f"[STATS] Average speed: {processed_count/elapsed_time:.2f} fps")
            logger.info(f"[OUTPUT] Output saved to: {output_path}")
            logger.info(f"[OUTPUT] Output size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
            logger.info("=" * 80)
        
        except Exception as e:
            logger.error(f"[ERROR] Error during video enhancement: {str(e)}", exc_info=True)
            raise
    
    def enhance_with_audio(self, input_video, output_video, audio_path=None, s3_key=None):
        """
        Enhance video and preserve/add audio, optionally upload to S3
        
        Args:
            input_video: path to input video
            output_video: path to save enhanced video
            audio_path: optional separate audio file (uses input video audio if None)
            s3_key: optional S3 key to upload final video
        """
        logger.info("=" * 80)
        logger.info("[AUDIO] Audio processing mode detected")
        logger.info("=" * 80)
        
        # First enhance the video
        temp_output = output_video.replace('.mp4', '_no_audio.mp4')
        self.enhance_video(input_video, temp_output)
        
        # Then add audio using ffmpeg
        if audio_path is None:
            audio_source = input_video
            logger.info("[AUDIO] Using audio from input video")
        else:
            audio_source = audio_path
            logger.info(f"[AUDIO] Using audio from: {audio_path}")
        
        logger.info("-" * 80)
        logger.info("[AUDIO] Adding audio to enhanced video...")
        cmd = f'ffmpeg -i "{temp_output}" -i "{audio_source}" -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 "{output_video}" -y'
        logger.info("[AUDIO] Running FFmpeg audio merge...")
        
        result = os.system(cmd)
        
        if result == 0:
            logger.info("[AUDIO] Audio added successfully")
        else:
            logger.error(f"[AUDIO] Failed to add audio (exit code: {result})")
        
        # Clean up temp file
        if os.path.exists(temp_output):
            logger.info(f"[CLEANUP] Removing temporary file: {temp_output}")
            os.remove(temp_output)
        
        # Upload to S3 if key provided
        s3_url = None
        if s3_key:
            logger.info("-" * 80)
            logger.info("[S3] Uploading enhanced video to S3...")
            s3_url = self.s3_uploader.upload_file(output_video, s3_key)
            if s3_url:
                logger.info(f"[S3] S3 upload successful")
                logger.info(f"[S3] Video URL: {s3_url}")
            else:
                logger.warning("[S3] S3 upload failed or skipped")
            logger.info("-" * 80)
        
        logger.info("=" * 80)
        logger.info("[COMPLETE] Final video with audio saved")
        logger.info(f"[OUTPUT] Local path: {output_video}")
        logger.info(f"[OUTPUT] File size: {os.path.getsize(output_video) / (1024**2):.2f} MB")
        if s3_url:
            logger.info(f"[OUTPUT] S3 URL: {s3_url}")
        logger.info("=" * 80)
        
        return s3_url


def download_models():
    """Download required model weights if not present"""
    import requests
    os.makedirs('weights', exist_ok=True)

    models = {
        'GFPGANv1.4.pth': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
        'RealESRGAN_x2plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    }

    for model_name, url in models.items():
        model_path = f'weights/{model_name}'
        if not os.path.exists(model_path):
            logger.info(f"[DOWNLOAD] Downloading {model_name}...")
            response = requests.get(url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"[DOWNLOAD] Downloaded {model_name}")
        else:
            logger.info(f"[OK] {model_name} already exists")

    logger.info("[DOWNLOAD] All models downloaded!")


def main():
    logger.info("=" * 80)
    logger.info("[START] Video Enhancement Script Started")
    logger.info(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    parser = argparse.ArgumentParser(description='Enhance Wav2Lip output video')
    parser.add_argument('--input', '-i', help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--model', '-m', default='gfpgan', 
                       choices=['gfpgan', 'realesrgan'],
                       help='Enhancement model to use')
    parser.add_argument('--upscale', '-u', type=int, default=2,
                       choices=[1, 2, 4], help='Upscale factor')
    parser.add_argument('--audio', '-a', default=None,
                       help='Optional audio file to add (uses input video audio if not specified)')
    parser.add_argument('--s3-key', '-s', default=None,
                       help='Optional S3 key for uploading enhanced video')
    parser.add_argument('--download-models', action='store_true',
                       help='Download model weights before processing')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Start frame for processing')
    parser.add_argument('--end-frame', type=int, default=None,
                       help='End frame for processing')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Download models if requested
    if args.download_models:
        download_models()
        logger.info("[OK] Models downloaded successfully!")
        if not args.input:
            logger.info("[INFO] To enhance a video, run:")
            logger.info("[INFO] python enhance.py -i input_video.mp4 -o output_video.mp4")
            return
    
    # Check if input/output are provided
    if not args.input or not args.output:
        parser.error("--input and --output are required for video enhancement")
    
    try:
        # Create enhancer
        enhancer = VideoEnhancer(model_type=args.model, upscale=args.upscale, gpu_id=args.gpu_id)
        
        # Enhance video
        if args.audio is not None or os.path.exists(args.input):
            s3_url = enhancer.enhance_with_audio(args.input, args.output, args.audio, args.s3_key)
        else:
            enhancer.enhance_video(args.input, args.output, args.start_frame, args.end_frame)
        
        logger.info("=" * 80)
        logger.info("[COMPLETE] Enhancement completed successfully!")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"[FAILED] Enhancement failed: {str(e)}")
        logger.error("=" * 80)
        raise


if __name__ == '__main__':
    main()