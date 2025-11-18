#!/usr/bin/env python3
"""
Script to download pretrained models for the video upscaling service.
This script downloads the necessary model weights for GFPGAN and Real-ESRGAN.
"""

import os
import requests
from pathlib import Path


def download_file(url, destination):
    """
    Download a file from URL to destination with progress indication.

    Args:
        url (str): URL to download from
        destination (str): Local path to save the file
    """
    print(f"Downloading {os.path.basename(destination)}...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get total file size
        total_size = int(response.headers.get('content-length', 0))

        # Download with progress
        downloaded = 0
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"{progress:.1f}%", end='', flush=True)

        print(" ✓ Done")
    except Exception as e:
        print(f" ✗ Failed to download {os.path.basename(destination)}: {e}")
        return False

    return True


def download_models():
    """Download all required model weights."""

    # Create weights directory if it doesn't exist
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)

    gfpgan_weights_dir = Path('gfpgan/weights')
    gfpgan_weights_dir.mkdir(parents=True, exist_ok=True)

    # Model URLs and destinations
    models = {
        'weights/GFPGANv1.4.pth': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
        'weights/RealESRGAN_x2plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        'weights/RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'gfpgan/weights/detection_Resnet50_Final.pth': 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        'gfpgan/weights/parsing_parsenet.pth': 'https://huggingface.co/gmk123/GFPGAN/resolve/main/parsing_parsenet.pth?download=true',
    }

    print("Starting model downloads...\n")

    success_count = 0
    total_count = len(models)

    for model_path, url in models.items():
        if os.path.exists(model_path):
            print(f"✓ {os.path.basename(model_path)} already exists")
            success_count += 1
            continue

        if download_file(url, model_path):
            success_count += 1

    print(f"\nDownloaded {success_count}/{total_count} models successfully!")

    if success_count == total_count:
        print("All models are ready!")
        return True
    else:
        print(f"Failed to download {total_count - success_count} models.")
        return False


if __name__ == '__main__':
    print("Video Upscaling Service - Model Downloader")
    print("=" * 50)

    success = download_models()

    if success:
        print("\nYou can now run the video enhancement service!")
        print("Example: python enhance.py -i input.mp4 -o output.mp4")
    else:
        print("\nSome models failed to download. Please check your internet connection and try again.")
        exit(1)
