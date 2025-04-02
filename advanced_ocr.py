import os
import cv2
import numpy as np
import json
import time
import hashlib
import base64
import requests
import io
import tempfile
from datetime import datetime
from google.cloud import storage
from google import genai
from google.genai import types
from PIL import Image

class AdvancedOCR:
    def __init__(self, model_path=None, confidence_threshold=0.5, use_cache=True, cache_dir='cache'):
        """
        Initialize advanced OCR processing class
        
        Args:
            model_path (str): DocLayout-YOLO model path
            confidence_threshold (float): Detection confidence threshold
            use_cache (bool): Whether to use caching
            cache_dir (str): Cache directory path
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        # Create cache directory
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Load DocLayout-YOLO model
        try:
            from custom_doclayout_yolo import DocLayoutYOLO
            self.doc_layout_model = DocLayoutYOLO(model_path=self.model_path)
            print("DocLayout-YOLO model loaded successfully")
        except Exception as e:
            print(f"Failed to load DocLayout-YOLO model: {e}")
            self.doc_layout_model = None
        
        # Set up Gemini API
        self._setup_gemini_api()
        
        # Initialize Google Cloud Storage client
        self._setup_gcs_client()
    
    def _setup_gemini_api(self):
        """Set up Gemini API"""
        # Get API key from environment variable
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            # Initialize latest Gemini API client
            self.gemini_client = genai.Client(api_key=api_key)
            print("Gemini API client initialized successfully")
        else:
            self.gemini_client = None
            print("Warning: GEMINI_API_KEY environment variable not set")
    
    def _setup_gcs_client(self):
        """Initialize Google Cloud Storage client"""
        try:
            # Get service account info from environment variable
            SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            self.BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "YOUR_GCS_BUCKET_NAME")
            
            if SERVICE_ACCOUNT_JSON:
                from google.oauth2.service_account import Credentials
                creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON)
                self.storage_client = storage.Client(credentials=creds, project=creds.project_id)
                print("Google Cloud Storage client initialized successfully")
            else:
                self.storage_client = None
                print("Warning: GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        except Exception as e:
            self.storage_client = None
            print(f"Failed to initialize Google Cloud Storage client: {e}")
    
    def _calculate_image_hash(self, image):
        """
        Calculate image hash
        
        Args:
            image (numpy.ndarray): Image to calculate hash for
            
        Returns:
            str: Image hash string
        """
        # Convert image to bytes
        _, buffer = cv2.imencode('.png', image)
        # Calculate hash
        image_hash = hashlib.md5(buffer).hexdigest()
        return image_hash
    
    def _get_cached_result(self, image_hash, cache_type):
        """
        Get cached result
        
        Args:
            image_hash (str): Image hash
            cache_type (str): Cache type (e.g., 'ocr', 'layout')
            
        Returns:
            dict or None: Cached result or None (cache miss)
        """
        if not self.use_cache:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"{cache_type}_{image_hash}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache file: {e}")
        
        return None
    
    def _save_to_cache(self, image_hash, cache_type, result):
        """
        Save result to cache
        
        Args:
            image_hash (str): Image hash
            cache_type (str): Cache type (e.g., 'ocr', 'layout')
            result (dict): Result to save
        """
        if not self.use_cache:
            return
        
        cache_file = os.path.join(self.cache_dir, f"{cache_type}_{image_hash}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving to cache: {e}")
    
    def _detect_with_doclayout_yolo(self, image_np):
        """
        Detect document layout using DocLayout-YOLO
        
        Args:
            image_np (numpy.ndarray): Input image
            
        Returns:
            list: List of detected regions
        """
        # Calculate image hash
        image_hash = self._calculate_image_hash(image_np)
        
        # Check cache
        cached_result = self._get_cached_result(image_hash, 'layout')
        if cached_result is not None:
            return cached_result
        
        # Return empty result if DocLayout-YOLO model is not initialized
        if self.doc_layout_model is None:
            print("DocLayout-YOLO model not initialized")
            return []
        
        # Detect with DocLayout-YOLO
        try:
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, image_np)
            
            # Use predict method
            results = self.doc_layout_model.predict(temp_path, conf=0.25)
            
            # Filter and format results
            regions = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    class_names = result.names
                    
                    for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confs)):
                        x1, y1, x2, y2 = map(int, box)
                        cls_name = class_names[int(cls_id)]
                        
                        if conf >= self.confidence_threshold:
                            regions.append({
                                'type': cls_name,
                                'coords': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                'confidence': float(conf)
                            })
            
            # Delete temporary file
            os.unlink(temp_path)
            
            # Merge overlapping regions
            regions = self._merge_overlapping_regions(regions)
            
            # Save to cache
            self._save_to_cache(image_hash, 'layout', regions)
            
            return regions
        except Exception as e:
            print(f"DocLayout-YOLO detection error: {e}")
            return []
    
    def _merge_overlapping_regions(self, regions):
        """
        Merge duplicate or overlapping regions
        
        Args:
            regions (list): List of regions to merge
            
        Returns:
            list: List of merged regions
        """
        if len(regions) <= 1:
            return regions
        
        # Function to calculate IoU
        def calculate_iou(box1, box2):
            # Extract box coordinates
            x1, y1, w1, h1 = box1['coords']
            x2, y2, w2, h2 = box2['coords']
            
            # Calculate box endpoints
            x1_end, y1_end = x1 + w1, y1 + h1
            x2_end, y2_end = x2 + w2, y2 + h2
            
            # Calculate intersection area
            x_inter = max(0, min(x1_end, x2_end) - max(x1, x2))
            y_inter = max(0, min(y1_end, y2_end) - max(y1, y2))
            area_inter = x_inter * y_inter
            
            # Calculate union area
            area1 = w1 * h1
            area2 = w2 * h2
            area_union = area1 + area2 - area_inter
            
            # Calculate IoU
            if area_union == 0:
                return 0
            return area_inter / area_union
        
        # Mark regions to keep
        to_keep = [True] * len(regions)
        
        # Check for duplicate regions
        for i in range(len(regions)):
            if not to_keep[i]:
                continue
                
            for j in range(i+1, len(regions)):
                if not to_keep[j]:
                    continue
                
                # Consider as duplicate if same class and IoU above threshold
                if regions[i]['type'] == regions[j]['type'] and calculate_iou(regions[i], regions[j]) > 0.5:
                    # Remove the one with lower confidence
                    if regions[i]['confidence'] < regions[j]['confidence']:
                        to_keep[i] = False
                        break
                    else:
                        to_keep[j] = False
        
        # Return only non-duplicate regions
        filtered_regions = []
        for i in range(len(regions)):
            if to_keep[i]:
                filtered_regions.append(regions[i])
        
        return filtered_regions
    
    def _detect_regions(self, image_np):
        """
        Detect special regions in image
        
        Args:
            image_np (numpy.ndarray): Input image
            
        Returns:
            list: List of detected regions
        """
        # Detect regions with DocLayout-YOLO
        regions = self._detect_with_doclayout_yolo(image_np)
        
        # If no regions, treat entire image as text region
        if not regions:
            height, width = image_np.shape[:2]
            regions = [{
                'type': 'text',
                'coords': [0, 0, width, height],
                'confidence': 1.0
            }]
        
        # Sort regions by Y coordinate
        regions.sort(key=lambda r: r['coords'][1])
        
        return regions
    
    def _crop_region(self, image, region):
        """
        Extract region from image
        
        Args:
            image (numpy.ndarray): Original image
            region (dict): Region information
            
        Returns:
            numpy.ndarray: Extracted region image
        """
        x, y, w, h = region['coords']
        # Adjust coordinates if they exceed image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        return image[y:y+h, x:x+w]
    
    def _process_text_region(self, region_img, region_info):
        """
        Process text region
        
        Args:
            region_img (numpy.ndarray): Region image
            region_info (dict): Region information
            
        Returns:
            dict: Processed region information
        """
        # Calculate image hash
        image_hash = self._calculate_image_hash(region_img)
        
        # Check cache
        cached_result = self._get_cached_result(image_hash, 'text_ocr')
        if cached_result is not None:
            cached_result['coords'] = region_info['coords']
            return cached_result
        
        # Call Google Vision OCR API
        try:
            # Encode image as base64
            _, buffer = cv2.imencode('.png', region_img)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare API request data
            request_data = {
                'requests': [
                    {
                        'image': {
                            'content': encoded_image
                        },
                        'features': [
                            {
                                'type': 'TEXT_DETECTION'
                            }
                        ],
                        'imageContext': {
                            'languageHints': ['ja', 'en', 'ko']
                        }
                    }
                ]
            }
            
            # API call (using service account credentials)
            from google.cloud import vision
            from google.oauth2.service_account import Credentials
            
            SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if SERVICE_ACCOUNT_JSON:
                creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON)
                vision_client = vision.ImageAnnotatorClient(credentials=creds)
                
                image = vision.Image(content=buffer.tobytes())
                context = vision.ImageContext(language_hints=['ja', 'en', 'ko'])
                response = vision_client.text_detection(image=image, image_context=context)
                
                text = ''
                if response.text_annotations:
                    text = response.text_annotations[0].description
                
                processed_result = {
                    'type': 'text',
                    'coords': region_info['coords'],
                    'text': text
                }
                
                # Save to cache
                self._save_to_cache(image_hash, 'text_ocr', processed_result)
                
                return processed_result
            else:
                # API key method (alternative)
                response = requests.post(
                    'https://vision.googleapis.com/v1/images:annotate',
                    params={'key': os.environ.get('GOOGLE_VISION_API_KEY', '')},
                    json=request_data
                )
                
                # Process response
                if response.status_code == 200:
                    result = response.json()
                    text = ''
                    
                    # Extract text
                    if 'responses' in result and result['responses'] and 'fullTextAnnotation' in result['responses'][0]:
                        text = result['responses'][0]['fullTextAnnotation']['text']
                    
                    processed_result = {
                        'type': 'text',
                        'coords': region_info['coords'],
                        'text': text
                    }
                    
                    # Save to cache
                    self._save_to_cache(image_hash, 'text_ocr', processed_result)
                    
                    return processed_result
                else:
                    print(f"Google Vision API error: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Text region processing error: {e}")
        
        # Return empty result on error
        return {
            'type': 'text',
            'coords': region_info['coords'],
            'text': ''
        }
    
    def _process_table_region(self, region_img, region_info):
        """
        Process table region (using Gemini API)
        
        Args:
            region_img (numpy.ndarray): Region image
            region_info (dict): Region information
            
        Returns:
            dict: Processed region information
        """
        # Calculate image hash
        image_hash = self._calculate_image_hash(region_img)
        
        # Check cache
        cached_result = self._get_cached_result(image_hash, 'table_ocr')
        if cached_result is not None:
            cached_result['coords'] = region_info['coords']
            return cached_result
        
        # Process table with Gemini API
        try:
            # Process as text region if Gemini client is not initialized
            if self.gemini_client is None:
                print("Gemini API client not initialized. Processing as text region.")
                return self._process_text_region(region_img, region_info)
            
            # Convert image to PIL format
            pil_image = Image.fromarray(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB))
            
            # Create prompt
            prompt = """
            Analyze this table and respond in the following format:

            1. Accurately reproduce the table structure in markdown format. Clearly distinguish each column and row, and use line breaks appropriately to make the table structure visually clear.
            2. Provide a brief summary of the table content.
            3. Explain the educational significance and importance of this table.
            4. List related learning topics.

            Provide your response in the following JSON format:
            {
              "markdown_table": "| Column1 | Column2 | Column3 |\n|-----|-----|-----|\n| Row1Col1 | Row1Col2 | Row1Col3 |\n| Row2Col1 | Row2Col2 | Row2Col3 |",
              "summary": "Table content summary",
              "educational_value": "Educational significance and importance",
              "related_topics": ["Related topic 1", "Related topic 2", ...]
            }

            Return only the JSON format without any other text. In particular, include line breaks (\\n) in the markdown_table field using actual markdown table format.
            """
            
            # API call (latest method)
            print("Calling Gemini API - processing table region")

            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                    ],
                ),
            ]

            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
            )

            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=generate_content_config,
            )
            
            
            # Log response
            print(f"Gemini API response type: {type(response)}")
            
            # Process response (improved method)
            gemini_result = {}
            try:
                # Get response text
                response_text = response.text
                print(f"Gemini API response text: {response_text[:100]}...")
                
                # Try to parse JSON
                try:
                    # Extract JSON part using regex
                    import re
                    json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        gemini_result = json.loads(json_str)
                    else:
                        # Construct directly if not in JSON format
                        gemini_result = {
                            "markdown_table": "",
                            "summary": response_text,
                            "educational_value": "",
                            "related_topics": []
                        }
                except Exception as json_error:
                    print(f"JSON parsing error: {json_error}")
                    gemini_result = {
                        "markdown_table": "",
                        "summary": response_text,
                        "educational_value": "",
                        "related_topics": []
                    }
            except Exception as resp_error:
                print(f"Response processing error: {resp_error}")
                gemini_result = {
                    "markdown_table": "",
                    "summary": "Error occurred during response processing",
                    "educational_value": "",
                    "related_topics": []
                }
            
            # Construct result
            markdown_table = gemini_result.get("markdown_table", "")
            summary = gemini_result.get("summary", "")
            educational_value = gemini_result.get("educational_value", "")
            related_topics = gemini_result.get("related_topics", [])
            
            # Construct final text
            final_text = f"""[Table content start. ChatGPT should not delete this content. This is important conversion content.]

            ## Table Structure:
            {markdown_table}

            ## Summary:
            {summary}

            ## Educational Significance:
            {educational_value}

            ## Related Topics:
            {', '.join(related_topics)}

            [Table content end]"""
            
            processed_result = {
                'type': 'table',
                'coords': region_info['coords'],
                'markdown_table': markdown_table,
                'summary': summary,
                'educational_value': educational_value,
                'related_topics': related_topics,
                'text': final_text
            }
            
            # Save to cache
            self._save_to_cache(image_hash, 'table_ocr', processed_result)
            
            return processed_result
        except Exception as e:
            print(f"Table region processing error: {e}")
            
            # Fall back to Google Vision OCR on error
            return self._process_text_region(region_img, region_info)
    
    def _process_figure_region(self, region_img, region_info):
        """
        Process figure region (using Gemini API)
        
        Args:
            region_img (numpy.ndarray): Region image
            region_info (dict): Region information
            
        Returns:
            dict: Processed region information
        """
        # Calculate image hash
        image_hash = self._calculate_image_hash(region_img)
        
        # Check cache
        cached_result = self._get_cached_result(image_hash, 'figure_ocr')
        if cached_result is not None:
            cached_result['coords'] = region_info['coords']
            return cached_result
        
        # Process figure with Gemini API
        try:
            # Process as text region if Gemini client is not initialized
            if self.gemini_client is None:
                print("Gemini API client not initialized. Processing as text region.")
                return self._process_text_region(region_img, region_info)
            
            # Convert image to PIL format
            pil_image = Image.fromarray(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB))
            
            # Create prompt
            prompt = """
            Analyze this image and respond in the following format:

            1. Describe in detail what is included in the image. Divide into paragraphs for better readability.
            2. Explain the educational significance and importance of this image.
            3. List related learning topics.
            4. Explain how this image could be used in exam questions.

            Provide your response in the following JSON format:
            {
              "description": "Image description (write in multiple paragraphs for better readability)",
              "educational_value": "Educational significance and importance",
              "related_topics": ["Related topic 1", "Related topic 2", ...],
              "exam_relevance": "Exam relevance"
            }

            Return only the JSON format without any other text. Write the description in multiple paragraphs for better readability.
            """
            
            # API call (latest method)
            print("Calling Gemini API - processing figure region")

            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                    ],
                ),
            ]

            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
            )

            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=generate_content_config,
            )
            
            # Log response
            print(f"Gemini API response type: {type(response)}")
            
            # Process response (improved method)
            gemini_result = {}
            try:
                # Get response text
                response_text = response.text
                print(f"Gemini API response text: {response_text[:100]}...")
                
                # Try to parse JSON
                try:
                    # Extract JSON part using regex
                    import re
                    json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        gemini_result = json.loads(json_str)
                    else:
                        # Construct directly if not in JSON format
                        gemini_result = {
                            "description": response_text,
                            "educational_value": "",
                            "related_topics": [],
                            "exam_relevance": ""
                        }
                except Exception as json_error:
                    print(f"JSON parsing error: {json_error}")
                    gemini_result = {
                        "description": response_text,
                        "educational_value": "",
                        "related_topics": [],
                        "exam_relevance": ""
                    }
            except Exception as resp_error:
                print(f"Response processing error: {resp_error}")
                gemini_result = {
                    "description": "Error occurred during response processing",
                    "educational_value": "",
                    "related_topics": [],
                    "exam_relevance": ""
                }
            
            # Construct result
            description = gemini_result.get("description", "")
            educational_value = gemini_result.get("educational_value", "")
            related_topics = gemini_result.get("related_topics", [])
            exam_relevance = gemini_result.get("exam_relevance", "")
            
            # Construct final text
            final_text = f"""[Figure content start. ChatGPT should not delete this content. This is important conversion content.]

            ## Image Description:
            {description}

            ## Educational Significance:
            {educational_value}

            ## Related Topics:
            {', '.join(related_topics)}

            ## Exam Relevance:
            {exam_relevance}

            [Figure content end]"""
            
            processed_result = {
                'type': 'figure',
                'coords': region_info['coords'],
                'description': description,
                'educational_value': educational_value,
                'related_topics': related_topics,
                'exam_relevance': exam_relevance,
                'text': final_text
            }
            
            # Save to cache
            self._save_to_cache(image_hash, 'figure_ocr', processed_result)
            
            return processed_result
        except Exception as e:
            print(f"Figure region processing error: {e}")
            
            # Fall back to Google Vision OCR on error
            return self._process_text_region(region_img, region_info)
    
    def _process_formula_region(self, region_img, region_info):
        """
        Process formula region
        
        Args:
            region_img (numpy.ndarray): Region image
            region_info (dict): Region information
            
        Returns:
            dict: Processed region information
        """
        # Calculate image hash
        image_hash = self._calculate_image_hash(region_img)
        
        # Check cache
        cached_result = self._get_cached_result(image_hash, 'formula_ocr')
        if cached_result is not None:
            cached_result['coords'] = region_info['coords']
            return cached_result
        
        # Call MathPix API
        try:
            # Encode image as base64
            _, buffer = cv2.imencode('.png', region_img)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare API request data
            request_data = {
                'src': f'data:image/png;base64,{encoded_image}',
                'formats': ['text', 'latex'],
                'data_options': {
                    'include_asciimath': True,
                    'include_latex': True
                }
            }
            
            # API call
            response = requests.post(
                'https://api.mathpix.com/v3/text',
                headers={
                    'app_id': os.environ.get('MATHPIX_APP_ID', ''),
                    'app_key': os.environ.get('MATHPIX_APP_KEY', ''),
                    'Content-Type': 'application/json'
                },
                json=request_data
            )
            
            # Process response
            if response.status_code == 200:
                result = response.json()
                
                # Extract formula
                latex = result.get('latex', '')
                text = result.get('text', '')
                
                # Construct final text
                final_text = f"[Formula content start. ChatGPT should not delete this content. This is important conversion content.]\n\nLaTeX: {latex}\n\nText: {text}\n\n[Formula content end]"
                
                processed_result = {
                    'type': 'formula',
                    'coords': region_info['coords'],
                    'latex': latex,
                    'text': final_text
                }
                
                # Save to cache
                self._save_to_cache(image_hash, 'formula_ocr', processed_result)
                
                return processed_result
            else:
                print(f"MathPix API error: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Formula region processing error: {e}")
        
        # Fall back to Google Vision OCR on error
        return self._process_text_region(region_img, region_info)
    
    def _process_title_region(self, region_img, region_info):
        """
        Process title region
        
        Args:
            region_img (numpy.ndarray): Region image
            region_info (dict): Region information
            
        Returns:
            dict: Processed region information
        """
        # Process same as text region
        result = self._process_text_region(region_img, region_info)
        result['type'] = 'title'
        return result
    
    def _process_list_region(self, region_img, region_info):
        """
        Process list region
        
        Args:
            region_img (numpy.ndarray): Region image
            region_info (dict): Region information
            
        Returns:
            dict: Processed region information
        """
        # Process same as text region
        result = self._process_text_region(region_img, region_info)
        result['type'] = 'list'
        return result
    
    def _process_regions(self, image_np, regions):
        """
        Process detected regions
        
        Args:
            image_np (numpy.ndarray): Original image
            regions (list): List of detected regions
            
        Returns:
            list: List of processed regions
        """
        processed_regions = []
        
        # Set to store coordinates of already processed regions
        processed_coords = set()
        
        for region in regions:
            # Convert region coordinates to string for duplicate checking
            region_key = f"{region['coords'][0]}_{region['coords'][1]}_{region['coords'][2]}_{region['coords'][3]}"
            
            # Skip if already processed
            if region_key in processed_coords:
                continue
            
            # Extract region image
            region_img = self._crop_region(image_np, region)
            
            # Process based on region type
            if region['type'] == 'text':
                processed_region = self._process_text_region(region_img, region)
            elif region['type'] == 'title':
                processed_region = self._process_title_region(region_img, region)
            elif region['type'] == 'list':
                processed_region = self._process_list_region(region_img, region)
            elif region['type'] == 'table':
                processed_region = self._process_table_region(region_img, region)
            elif region['type'] == 'figure':
                processed_region = self._process_figure_region(region_img, region)
            elif region['type'] == 'formula':
                processed_region = self._process_formula_region(region_img, region)
            else:
                # Process unknown types as text
                processed_region = self._process_text_region(region_img, region)
            
            # Add processed region
            processed_regions.append(processed_region)
            
            # Store processed region coordinates
            processed_coords.add(region_key)
        
        # Sort by Y coordinate
        processed_regions.sort(key=lambda r: r['coords'][1])
        
        return processed_regions
    
    def _combine_processed_regions(self, processed_regions):
        """
        Combine processed regions to generate final text
        
        Args:
            processed_regions (list): List of processed regions
            
        Returns:
            str: Combined text
        """
        combined_text = ""
        
        for region in processed_regions:
            if 'text' in region and region['text']:
                combined_text += region['text'] + "\n\n"
        
        return combined_text.strip()
    
    def _upload_to_gcs(self, data, gcs_path):
        """
        Upload results to GCS
        
        Args:
            data (dict): Data to upload
            gcs_path (str): GCS path
            
        Returns:
            bool: Upload success status
        """
        if not self.storage_client:
            print(f"GCS client not initialized, skipping upload: {gcs_path}")
            return False
        
        try:
            bucket = self.storage_client.bucket(self.BUCKET_NAME)
            blob = bucket.blob(gcs_path)
            
            # Serialize JSON data
            json_data = json.dumps(data, ensure_ascii=False, indent=2)
            
            # Upload
            blob.upload_from_string(json_data, content_type="application/json")
            print(f"GCS upload complete: gs://{self.BUCKET_NAME}/{gcs_path}")
            return True
        
        except Exception as e:
            print(f"GCS upload error: {e}")
            return False
    
    def process_image(self, image_path):
        """
        Main image processing function
        
        Args:
            image_path (str): Path to image to process
            
        Returns:
            dict: Processing results
        """
        start_time = time.time()
        
        # Load image
        image_np = cv2.imread(image_path)
        if image_np is None:
            return {'error': f"Cannot load image: {image_path}"}
        
        # Get image dimensions
        height, width = image_np.shape[:2]
        
        # Detect regions
        regions = self._detect_regions(image_np)
        
        # Process regions
        processed_regions = self._process_regions(image_np, regions)
        
        # Combine text
        text = self._combine_processed_regions(processed_regions)
        
        # Calculate processing time
        processed_time = time.time() - start_time
        
        # Return results
        return {
            'width': width,
            'height': height,
            'regions': regions,
            'processed_regions': processed_regions,
            'text': text,
            'region_positions': [region['coords'] for region in processed_regions],
            'processed_time': datetime.now().isoformat()
        }
    
    def process_pdf(self, pdf_path, output_folder=None):
        """
        Process PDF file
        
        Args:
            pdf_path (str): PDF file path
            output_folder (str): Output folder path
            
        Returns:
            dict: Processing results summary
        """
        try:
            from pdf2image import convert_from_path, pdfinfo_from_path
            
            # Extract PDF filename
            pdf_file = os.path.basename(pdf_path)
            
            # Extract subject name (from filename or use default)
            subject = pdf_file.replace(".pdf", "").split("_")[-1] if "_" in pdf_file else "Unknown"
            
            print(f"Starting PDF processing: {pdf_file}, Subject: {subject}")
            
            # Read PDF info
            pdf_info = pdfinfo_from_path(pdf_path)
            num_pages = pdf_info["Pages"]
            print(f"PDF page count: {num_pages}")
            
            # Set output folder
            if output_folder is None:
                output_folder = os.path.join(os.path.dirname(pdf_path), "output")
            
            # Create output folder
            os.makedirs(output_folder, exist_ok=True)
            
            # Create subject folder
            subject_folder = os.path.join(output_folder, subject)
            os.makedirs(subject_folder, exist_ok=True)
            
            # Create PDF name folder
            pdf_name = pdf_file.replace(".pdf", "")
            pdf_folder = os.path.join(subject_folder, pdf_name)
            os.makedirs(pdf_folder, exist_ok=True)
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            print(f"PDF converted to {len(images)} images")
            
            # Process each page
            results = []
            for i, image in enumerate(images):
                print(f"Processing page {i+1}/{len(images)}...")
                
                # Save image
                image_path = os.path.join(pdf_folder, f"page_{i+1}.jpg")
                image.save(image_path, "JPEG")
                
                # Process image
                page_result = self.process_image(image_path)
                results.append(page_result)
                
                # Save results
                output_path = os.path.join(pdf_folder, f"page_{i+1}.json")
                self.save_result(page_result, output_path)
                
                # Upload page results to GCS
                gcs_path = f"{subject}/stage1/{pdf_name}/page_{i+1}.json"
                self._upload_to_gcs(page_result, gcs_path)
            
            # Create summary results
            summary = {
                "pdf_name": pdf_name,
                "num_pages": num_pages,
                "processed_time": datetime.now().isoformat(),
                "pages": [{"page": i+1, "status": "processed"} for i in range(len(images))]
            }
            
            # Save summary results
            summary_path = os.path.join(pdf_folder, "summary_stage1.json")
            self.save_result(summary, summary_path)
            
            # Upload summary results to GCS
            gcs_summary_path = f"{subject}/stage1/{pdf_name}/summary_stage1.json"
            self._upload_to_gcs(summary, gcs_summary_path)
            
            print(f"PDF processing complete: {pdf_file}")
            return summary
        
        except Exception as e:
            print(f"PDF processing error: {e}")
            return {"error": str(e)}
    
    def save_result(self, result, output_path):
        """
        Save processing results to JSON file
        
        Args:
            result (dict): Processing results
            output_path (str): Path to save file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)



# (AdvancedOCR class and other code parts use the definitions above)

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Advanced OCR Processing')
    # Required argument: --input (accepts both single file or directory)
    parser.add_argument('--input', default='/app/input', help='Input file or directory path (image or PDF)')
    # Optional argument: --output
    parser.add_argument('--output', help='Output JSON file path (for image) or output folder (for PDF)')
    parser.add_argument('--model', default=None, help='DocLayout-YOLO model path')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--cache-dir', default='cache', help='Cache directory path')
    
    args = parser.parse_args()
    
    # Create OCR processing object
    ocr = AdvancedOCR(
        model_path=args.model,
        confidence_threshold=args.confidence,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir
    )
    
    input_path = args.input
    # If input is a directory, process all PDF files in that directory
    if os.path.isdir(input_path):
        import glob
        pdf_files = glob.glob(os.path.join(input_path, "*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in input directory {input_path}.")
        for pdf_file in pdf_files:
            # Output file path: create output folder in the directory containing the PDF file
            output_folder = args.output if args.output else os.path.join(os.path.dirname(pdf_file), "output")
            result = ocr.process_pdf(pdf_file, output_folder)
            ocr.save_result(result, os.path.join(output_folder, os.path.splitext(os.path.basename(pdf_file))[0] + ".json"))
            print(f"PDF processing complete: {pdf_file} -> {output_folder}")
    else:
        # For single file: process as PDF or image based on extension
        if input_path.lower().endswith('.pdf'):
            output_folder = args.output if args.output else os.path.join(os.path.dirname(input_path), "output")
            result = ocr.process_pdf(input_path, output_folder)
            print(f"PDF processing complete: {input_path} -> {output_folder}")
        else:
            result = ocr.process_image(input_path)
            output_path = args.output if args.output else input_path.rsplit('.', 1)[0] + '.json'
            ocr.save_result(result, output_path)
            print(f"Image processing complete: {input_path} -> {output_path}")


# To customize output language, modify the prompt strings in the following methods:
# 1. _process_table_region(): Change the prompt text to your desired language
# 2. _process_figure_region(): Change the prompt text to your desired language
# 3. _process_formula_region(): Change the final_text format to your desired language
# 4. Change the language hints in _process_text_region() from ['ja', 'en', 'ko'] to your preferred languages

# IMPORTANT: Do not make any omissions or changes to the content, abbreviations, or code modifications. This code is already working correctly.
