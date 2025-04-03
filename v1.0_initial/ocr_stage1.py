#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML OCR System - Docker Container Execution Version for Vertex AI Notebook (Final Version)

- PDF Input from Host: /home/jupyter/Google Drive/Study Materials/
- GCS Upload: eju-ocr-results/Chemistry/stage1/[pdf_name]/page_{n}.json
"""

import os
import json
import logging
import subprocess
import argparse
import glob
from datetime import datetime
from dotenv import load_dotenv
load_dotenv('/home/jupyter/Your_Folder_Name/.env')

# ----------------------------
# [1] Log Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# [2] Docker Container Execution Function
# ----------------------------
def run_docker_container(input_dir, output_dir, credentials_dir, image_name="shit"):
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    """
    Run Docker container to perform OCR processing.
    
    Args:
        input_dir (str): Host-side PDF file directory path
        output_dir (str): Host-side OCR results/logs storage directory path
        credentials_dir (str): Host-side Google Cloud credentials directory
        image_name (str): Docker image name to use
    
    Returns:
        bool: Success status
    """
    try:
        # Convert to absolute paths
        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)
        credentials_dir = os.path.abspath(credentials_dir)
        
        # Check and create directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Check input directory
        if not os.path.exists(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return False
        
        # Check PDF files
        pdf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
        logger.info(f"Number of PDF files found in input directory: {len(pdf_files)}")
        if len(pdf_files) > 0:
            logger.info(f"PDF file list (max 20): {pdf_files[:20]}")
        else:
            logger.warning(f"No PDF files in input directory: {input_dir}")
            # Continue even if no PDF files
        
        # Check if Docker image exists
        result = subprocess.run(
            ["docker", "images", "-q", image_name], 
            capture_output=True, 
            text=True
        )
        
        if not result.stdout.strip():
            logger.info(f"Docker image '{image_name}' not found. Starting build.")
            # Build Docker image (Dockerfile location example)
            docker_dir = "/home/jupyter/YOUR_DOCKER_DIRECTORY"
            subprocess.run(
                ["docker", "build", "-t", image_name, docker_dir],
                check=True
            )
            logger.info(f"Docker image '{image_name}' build complete")
        
        # Create command string to handle paths with spaces (added GPU usage)
        cmd_str = " ".join([
            "docker", "run", "--gpus", "all", "--rm",
            "--runtime=nvidia",
            "-e NVIDIA_VISIBLE_DEVICES=all",
            "-e NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            f"-v \"{input_dir}\":/app/input",
            f"-v \"{output_dir}\":/app/output",
            f"-v \"{credentials_dir}\":/app/credentials",
            "-e PDF_FOLDER=/app/input",
            "-e OUTPUT_FOLDER=/app/output",
            "-e PYTHONUNBUFFERED=1",
            f"-e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/Google_Vision_S.Account.json",
            "-e PYTHONPATH=/app:/app/DocLayout-YOLO",
            f"-e GEMINI_API_KEY={gemini_api_key}",
            image_name,
            "python /app/advanced_ocr.py"
        ])
        
        logger.info(f"Running Docker container: {cmd_str}")
        
        # Use shell=True to handle paths with spaces
        subprocess.run(cmd_str, shell=True, check=True)
        logger.info("Docker container execution complete")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Docker container execution failed: {e}")
        return False

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return False

# ----------------------------
# [3] Main Function
# ----------------------------
def main():
    
    parser = argparse.ArgumentParser(description="OCR System - Docker (Final Version)")
    
    # Dummy argument to ignore -f argument automatically added by Jupyter/Colab
    parser.add_argument("-f", "--somefile", help="(Jupyter) ignore this argument", default=None)
    
    # Existing arguments
    parser.add_argument("--input-dir", default="/home/jupyter/Google Drive/Study Materials",
                        help="Host-side PDF directory for OCR processing (default: /home/jupyter/Google Drive/Study Materials)")
    parser.add_argument("--output-dir", default="/home/jupyter/ocr_output",
                        help="Host-side OCR results/logs directory (default: /home/jupyter/ocr_output)")
    parser.add_argument("--credentials-dir", default="/home/jupyter/credentials",
                        help="Google Cloud credentials directory (default: /home/jupyter/credentials)")
    parser.add_argument("--image-name", default="cantaloupe", #You have to change the image name
                        help="Docker image name to use (default: cantaloupe)")
    
    # Use parse_known_args() to ignore unknown arguments like -f
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.info(f"Ignored arguments: {unknown}")
    
    logger.info("=== OCR System (Docker) Starting ===")
    logger.info(f"Input directory (host): {args.input_dir}")
    logger.info(f"Output directory (host): {args.output_dir}")
    logger.info(f"Credentials directory (host): {args.credentials_dir}")
    logger.info(f"Docker image name: {args.image_name}")
    
    success = run_docker_container(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        credentials_dir=args.credentials_dir,
        image_name=args.image_name
    )
    
    if success:
        logger.info("=== OCR System Complete ===")
    else:
        logger.error("=== OCR System Failed ===")

if __name__ == "__main__":
    main()

# To customize output language, modify the log messages in this file.
# Environment variables are kept as is since they are configuration paths.
# If you need to change the input directory path, modify the default value in the
# --input-dir argument in the main() function.
