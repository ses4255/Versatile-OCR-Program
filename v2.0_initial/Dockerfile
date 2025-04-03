###############################################################################
# Dockerfile for GPU-based Python environment with DocLayout-YOLO (HEAD)
# - CUDA 11.8 + cuDNN 8 + Ubuntu 20.04
# - Python 3.9 (via deadsnakes)
# - Timezone: Asia/Seoul (can be changed)
# - NumPy <2.0 (1.24.3)
# - Patched DocLayout-YOLO (latest HEAD) to remove 'init_subclass' keyword argument
###############################################################################
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# NVIDIA settings
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 1) Install Packages
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    git \
    build-essential \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tzdata \
    python3.9 \
    python3.9-distutils \
    python3.9-dev && \
    ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    echo "Asia/Seoul" > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

# 2) Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py && \
    python3.9 /tmp/get-pip.py && \
    rm /tmp/get-pip.py

# 3) Create symbolic links for python3 and pip
RUN ln -sf /usr/bin/python3.9 /usr/local/bin/python && \
    ln -sf /usr/local/bin/pip /usr/local/bin/pip3

# 4) Set working directory
WORKDIR /app

# 5) Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 6) Install PyTorch & TorchVision (e.g., 2.0.1 + cu118)
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 7) Install NumPy and other Python dependencies
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    Pillow==9.4.0 \
    opencv-python==4.7.0.72 \
    pdf2image==1.16.3 \
    requests==2.31.0 \
    huggingface_hub==0.19.4 \
    google-cloud-storage==2.9.0 \
    google-cloud-vision==3.4.0 \
    PyYAML==6.0.1 \
    ultralytics==8.0.196 \
    protobuf==3.20.3

RUN pip install google-genai

# 8) Clone the latest HEAD version of DocLayout-YOLO
RUN git clone https://github.com/opendatalab/DocLayout-YOLO.git /app/doclayout-yolo
WORKDIR /app/doclayout-yolo
RUN git checkout main
RUN pip install --no-cache-dir -e .

# 9) Patch: Remove 'init_subclass' keyword argument from YOLOv10
RUN sed -i \
    's/class YOLOv10(Model, PyTorchModelHubMixin, repo_url=.*$/class YOLOv10(Model, PyTorchModelHubMixin):/' \
    /app/doclayout-yolo/doclayout_yolo/models/yolov10/model.py

# 10) Switch back to /app directory
WORKDIR /app

# 11) Copy custom_doclayout_yolo.py and advanced_ocr.py
COPY custom_doclayout_yolo.py /app/custom_doclayout_yolo.py
COPY advanced_ocr.py /app/advanced_ocr.py

# 12) Define mountable volumes
VOLUME ["/app/input", "/app/output", "/app/credentials"]

# 13) Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/YOUR_Google_Vision_S.Account.json
ENV PDF_FOLDER=/app/input
ENV OUTPUT_FOLDER=/app/output
ENV GCS_BUCKET_NAME=YOUR_GCS_BUCKET_NAME
ENV MATHPIX_APP_ID="YOUR_MATHPIX_APP_ID"
ENV MATHPIX_APP_KEY="YOUR_MATHPIX_APP_KEY"
ENV PYTHONPATH=/app:/app/doclayout-yolo

# 14) CMD: Run advanced_ocr.py with --input /app/input to process all PDFs in that directory
CMD ["python", "/app/advanced_ocr.py", "--input", "/app/input"]
