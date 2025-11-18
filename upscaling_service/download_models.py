import os
import boto3
import requests

S3_BUCKET = "my-ml-models"   # <-- CHANGE THIS
UPLOAD_TO_S3 = False         # Set True if you want to upload models after download


# ---------------------------
# Model download registry
# ---------------------------

models = {
    # GFPGAN + RealESRGAN
    'upscaling_service/weights/GFPGANv1.4.pth':
        'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',

    'upscaling_service/weights/RealESRGAN_x2plus.pth':
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',

    'upscaling_service/weights/RealESRGAN_x4plus.pth':
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',

    'upscaling_service/gfpgan/weights/detection_Resnet50_Final.pth':
        'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',

    'upscaling_service/gfpgan/weights/parsing_parsenet.pth':
        'https://huggingface.co/gmk123/GFPGAN/resolve/main/parsing_parsenet.pth?download=true',

    # Wav2Lip
    'Wav2Lip/checkpoints/wav2lip.pth':
        'https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip.pth?download=true'
}


def download_file(url, dest_path):
    """ Download a file with streaming mode (handles HuggingFace redirects). """
    print(f"➡️ Downloading: {url}")
    
    resp = requests.get(url, stream=True, allow_redirects=True)
    resp.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"✔ Download complete: {dest_path}")


def ensure_directory(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def upload_to_s3(local_path, s3_key):
    s3 = boto3.client("s3")
    print(f"⬆ Uploading to S3: s3://{S3_BUCKET}/{s3_key}")
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print("✔ Upload complete")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for rel_path, url in models.items():
        abs_path = os.path.join(base_dir, rel_path)

        ensure_directory(abs_path)

        # Skip if file already exists
        if os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
            print(f"✔ Already exists, skipping: {rel_path}")
        else:
            download_file(url, abs_path)

        # Upload to S3 if enabled
        if UPLOAD_TO_S3:
            upload_to_s3(abs_path, rel_path)



if __name__ == "__main__":
    main()
