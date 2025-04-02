#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
custom_doclayout_yolo.py
- Performs document layout analysis using the DocLayout-YOLO model.
- Updated for compatibility with PyTorch 2.0.1 or higher.
- Loads the model using the officially recommended method (hf_hub_download or from_pretrained).
"""

import os
import torch
import logging
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10

logger = logging.getLogger(__name__)

class DocLayoutYOLO:
    """DocLayout-YOLO model wrapper class"""

    def __init__(self, model_path=None):
        """
        Initialize the DocLayout-YOLO model
        
        Args:
            model_path (str, optional): Local model file path.
              If not provided, the pre-trained model will be loaded from Hugging Face Hub.
        """
        self.model_path = model_path
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.init_model()

    def init_model(self):
        """Initialize the model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Use the local model file if available
                logger.info(f"Loading local model file: {self.model_path}")
                self.model = YOLOv10(self.model_path)
            else:
                # If a local file is not available, download and load the pre-trained model from Hugging Face Hub
                logger.info("Loading pre-trained model from Hugging Face (using hf_hub_download)")
                filepath = hf_hub_download(
                    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
                )
                self.model = YOLOv10(filepath)
                # Alternatively, you can use the from_pretrained method as follows:
                # self.model = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-DocStructBench")
            logger.info("DocLayout-YOLO model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize DocLayout-YOLO model: {e}")
            try:
                from ultralytics import YOLO
                if self.model_path and os.path.exists(self.model_path):
                    self.model = YOLO(self.model_path)
                else:
                    self.model = YOLO("yolov8n.pt")
                logger.info("Successfully loaded ultralytics YOLO model as an alternative")
                return True
            except Exception as e2:
                logger.error(f"Alternative initialization failed: {e2}")
                self.model = None
                return False

    def predict(self, image_path, imgsz=1024, conf=0.25, device=None):
        """
        Perform layout prediction on the image.
        
        Args:
            image_path (str): Path to the image file.
            imgsz (int): Input image size.
            conf (float): Confidence threshold.
            device (str, optional): Device to use (if None, automatically selected).
            
        Returns:
            list: List of prediction results.
        """
        if self.model is None:
            logger.error("The model is not initialized")
            return []
        
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return []
        
        if device is None:
            device = self.device
        
        try:
            results = self.model.predict(
                source=image_path,
                imgsz=imgsz,
                conf=conf,
                device=device
            )
            return results
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []