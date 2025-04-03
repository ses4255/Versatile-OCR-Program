# OCR System Setup Guide

This guide provides step-by-step instructions for setting up the EJU OCR system, including environment configuration, NVIDIA setup, API key requirements, and file organization.

By default,the files are stored in the user’s directory (/home/jupyter), but you should modify the path according to your own environment.

**Important update**
If you are using the v2.0_initial version, please enter the following bash code in your terminal.
```bash
sudo usermod -aG docker jupyter
  sudo reboot
  

## 1. Environment File Setup

Create a `.env` file in your project directory with the following content. Replace the placeholder values with your actual API keys and credentials:

```
OPENAI_API_KEY=your_openai_api_key_here
MATHPIX_APP_ID=your_mathpix_app_id_here
MATHPIX_APP_KEY=your_mathpix_app_key_here
GOOGLE_SHEETS_SPREADSHEET_ID=your_google_sheets_id_here
GOOGLE_APPLICATION_CREDENTIALS=/home/jupyter/credentials/Vision_S.Account.json
GEMINI_API_KEY=your_gemini_api_key_here
```

## 2. Required Python Packages

Install the required Python packages:

```bash
pip install google-genai
pip install openai
```

## 3. NVIDIA Setup

Follow these steps to set up NVIDIA for GPU acceleration:

### 3.1. Install NVIDIA Container Toolkit

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 3.2. Configure Docker to Use NVIDIA Runtime

Check if the Docker daemon configuration file exists:

```bash
cat /etc/docker/daemon.json
```

If the file doesn't exist or doesn't contain NVIDIA runtime configuration, create or edit it:

```bash
sudo nano /etc/docker/daemon.json
```

Add the following content (make sure to maintain proper indentation):

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

### 3.3. Verify GPU Recognition

Test if Docker can access the GPU:

```bash
docker run --gpus all nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 nvidia-smi
```

### 3.4. Check CUDA Version

Verify the CUDA version:

```bash
docker run --gpus all --rm nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 nvcc --version
```

If both commands display output without errors, your NVIDIA setup is complete!

## 4. API Key Requirements

You need to obtain API keys from the following services:

1. **OpenAI API Key**: Register at [OpenAI Platform](https://platform.openai.com/) to get your API key.
2. **Gemini API Key**: Get your API key from [Google AI Studio](https://makersuite.google.com/).
3. **MathPix API Key and App ID**: Register at [MathPix](https://mathpix.com/) to get your API key and App ID.
4. **Google Cloud Service Account**: Create a service account with Vision API and Storage permissions in the [Google Cloud Console](https://console.cloud.google.com/).

## 5. File Organization

The following files must be in the same directory (e.g., in a `docker` folder):

- `Dockerfile`
- `advanced_ocr.py`
- `custom_doclayout_yolo.py`

## 6. Google Cloud Storage (GCS) Bucket Setup

1. Create a GCS bucket in the [Google Cloud Console](https://console.cloud.google.com/storage/browser).
2. Make sure your service account has the necessary permissions to access this bucket.
3. Update the `GCS_BUCKET_NAME` environment variable in your `.env` file with your bucket name.

## 7. Credentials Setup

Create a `credentials` directory to store your Google service account JSON files:

```bash
mkdir -p /home/jupyter/credentials 
```

Place your service account JSON files in this directory:
- `Vision_S.Account.json` - For Google Vision API
- `Sheets_S.Account.json` - For Google Sheets API

## 8. Running the OCR System

After completing all the setup steps, you can run the OCR system using the Docker container:

```bash
python docker_runner.py
```

This will:
1. Build the Docker image if it doesn't exist
2. Mount the input, output, and credentials directories
3. Run the OCR processing on your PDF files

## Troubleshooting

- If you encounter GPU-related errors, make sure your NVIDIA drivers are properly installed and compatible with the CUDA version.
- If API calls fail, verify that your API keys are correctly set in the `.env` file.
- For Docker-related issues, check that the Docker daemon is running and properly configured for NVIDIA runtime.

## Additional Notes

- The OCR system processes PDF files from the input directory specified in the `OCR_stage1.py` script.
- Results are saved to the output directory and also uploaded to your GCS bucket.
- To customize the output language, modify the prompt templates in the OCR scripts.
