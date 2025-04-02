#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ocr_stage2_final_fixed.py - ML OCR System Stage 2 (ChatGPT Correction)

Features:
1) Load stage1 results from all folders in GCS bucket
2) Use ChatGPT for context-based text correction
   - Mark uncertain text with [?]
   - Simplify special content tags (formulas, figures, tables, etc.)
   - Only correct special content when high error probability
   - Remove unnecessary content
3) Save corrected results to stage2 folder at the same level as stage1
4) Skip folders that already have stage2 folder
"""

import os
import re
import json
import logging
import argparse
import difflib
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Set

# OpenAI API
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv("/home/jupyter/Your_Folder_Name/.env")

# Google Cloud Storage
from google.cloud import storage

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ocr_stage2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "YOUR_GCS_BUCKET_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully")
else:
    logger.warning("OPENAI_API_KEY is not set. ChatGPT calls may fail.")

# Initialize Google Cloud Storage client
try:
    storage_client = storage.Client()
    logger.info("Google Cloud Storage client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Google Cloud Storage client: {e}")
    storage_client = None

# Special content tag patterns (regex)
SPECIAL_CONTENT_PATTERNS = {
    "formula": r"\[Formula content start\. ChatGPT should not delete this content\. This is important conversion content\.\](.*?)\[Formula content end\]",
    "figure": r"\[Figure content start\. ChatGPT should not delete this content\. This is important conversion content\.\](.*?)\[Figure content end\]",
    "chart": r"\[Chart content start\. ChatGPT should not delete this content\. This is important conversion content\.\](.*?)\[Chart content end\]",
    "chemical_structure": r"\[Chemical structure start\. ChatGPT should not delete this content\. This is important conversion content\.\](.*?)\[Chemical structure end\]",
    "math_graph": r"\[Math graph start\. ChatGPT should not delete this content\. This is important conversion content\.\](.*?)\[Math graph end\]",
    "table": r"\[Table content start\. ChatGPT should not delete this content\. This is important conversion content\.\](.*?)\[Table content end\]"
}

# Simplified tag format
SIMPLIFIED_TAGS = {
    "formula": ("[FormulaStart]", "[FormulaEnd]"),
    "figure": ("[FigureStart]", "[FigureEnd]"),
    "chart": ("[ChartStart]", "[ChartEnd]"),
    "chemical_structure": ("[ChemicalStructureStart]", "[ChemicalStructureEnd]"),
    "math_graph": ("[MathGraphStart]", "[MathGraphEnd]"),
    "table": ("[TableStart]", "[TableEnd]")
}


def parse_gcs_prefix(gcs_url: str) -> Tuple[str, str]:
    """
    Separate bucket and prefix parts from gs://bucket/folder/... format
    
    Args:
        gcs_url: GCS URL (gs://bucket/folder/...)
        
    Returns:
        Tuple[str, str]: (bucket_name, prefix)
    """
    no_scheme = gcs_url.replace("gs://", "")
    parts = no_scheme.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix


def load_json_from_gcs(gcs_url: str) -> Optional[Dict]:
    """
    Download JSON file from GCS path and return as Python dict
    
    Args:
        gcs_url: GCS URL (gs://bucket/blob_path)
        
    Returns:
        Optional[Dict]: Loaded JSON data or None (on error)
    """
    try:
        if not gcs_url.startswith("gs://"):
            logger.error(f"Invalid GCS URL format: {gcs_url}")
            return None

        bucket_name, blob_path = parse_gcs_prefix(gcs_url)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        if not blob.exists():
            logger.error(f"Blob not found: {gcs_url}")
            return None

        data_str = blob.download_as_text(encoding="utf-8")
        data = json.loads(data_str)
        logger.info(f"JSON loaded successfully: {gcs_url}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from GCS: {e}")
        return None


def save_json_to_gcs(data: Dict, gcs_path: str) -> Optional[str]:
    """
    Serialize data to JSON and upload to GCS
    
    Args:
        data: Data to save (dict)
        gcs_path: GCS path (excluding bucket, e.g., "biology/stage2/2010_1_B/page_1_stage2.json")
        
    Returns:
        Optional[str]: Saved GCS URL or None (on error)
    """
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        if not bucket.exists():
            bucket.create()

        blob = bucket.blob(gcs_path)
        json_data = json.dumps(data, ensure_ascii=False, indent=2)
        blob.upload_from_string(json_data, content_type="application/json")

        logger.info(f"JSON saved successfully: gs://{BUCKET_NAME}/{gcs_path}")
        return f"gs://{BUCKET_NAME}/{gcs_path}"
    except Exception as e:
        logger.error(f"Error saving JSON to GCS: {e}")
        return None


def check_folder_exists(folder_path: str) -> bool:
    """
    Check if GCS folder exists
    
    Args:
        folder_path: GCS folder path (excluding bucket, e.g., "biology/stage2/")
        
    Returns:
        bool: Whether folder exists
    """
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        # GCS doesn't actually have folder concept, so check if any blob with this prefix exists
        blobs = list(bucket.list_blobs(prefix=folder_path, max_results=1))
        return len(blobs) > 0
    except Exception as e:
        logger.error(f"Error checking if GCS folder exists: {e}")
        return False


def simplify_special_content_tags(text: str) -> str:
    """
    Simplify special content tags
    
    Args:
        text: Original text
        
    Returns:
        str: Text with simplified tags
    """
    simplified_text = text
    
    for content_type, pattern in SPECIAL_CONTENT_PATTERNS.items():
        start_tag, end_tag = SIMPLIFIED_TAGS[content_type]
        
        def replace_tags(match):
            content = match.group(1).strip()
            # Add line breaks between label and content, and between content and end label
            return f"{start_tag}\n\n{content}\n\n{end_tag}"
        
        simplified_text = re.sub(pattern, replace_tags, simplified_text, flags=re.DOTALL)
    
    return simplified_text


def extract_special_content(text: str) -> Tuple[str, Dict[str, List[Dict[str, str]]]]:
    """
    Extract special content (formulas, figures, tables, etc.) from text and replace with placeholders
    
    Args:
        text: Original text
        
    Returns:
        Tuple[str, Dict]: (Text with placeholders, special content information)
    """
    placeholder_text = text
    special_contents = {}
    
    for content_type, pattern in SPECIAL_CONTENT_PATTERNS.items():
        special_contents[content_type] = []
        
        # Find special content
        matches = list(re.finditer(pattern, text, re.DOTALL))
        
        # Process from end to avoid index changes
        for i, match in enumerate(reversed(matches)):
            # Use clearer placeholder format (easier for ChatGPT to recognize)
            placeholder_id = f"___SPECIAL_CONTENT_{content_type}_{len(matches) - i - 1}_DO_NOT_REMOVE_THIS_PLACEHOLDER___"
            content = match.group(1).strip()
            
            # Replace special content with placeholder in original text
            start, end = match.span()
            placeholder_text = placeholder_text[:start] + placeholder_id + placeholder_text[end:]
            
            # Save special content information
            special_contents[content_type].append({
                "id": placeholder_id,
                "content": content,
                "original_tag": match.group(0)
            })
    
    return placeholder_text, special_contents


def restore_special_content(text: str, special_contents: Dict[str, List[Dict[str, str]]]) -> str:
    """
    Restore placeholders to simplified special content tags
    
    Args:
        text: Text with placeholders
        special_contents: Special content information
        
    Returns:
        str: Text with restored special content
    """
    restored_text = text
    
    # Process all special content types
    for content_type, contents in special_contents.items():
        start_tag, end_tag = SIMPLIFIED_TAGS[content_type]
        
        for content_info in contents:
            placeholder_id = content_info["id"]
            content = content_info["content"]
            
            # Replace placeholder with simplified tag
            if placeholder_id in restored_text:
                # Replace if placeholder exists
                restored_text = restored_text.replace(
                    placeholder_id, 
                    f"{start_tag}\n\n{content}\n\n{end_tag}"
                )
            else:
                # Try to restore original position if placeholder was deleted
                logger.warning(f"Placeholder '{placeholder_id}' was deleted in ChatGPT response. Preserving original tag.")
                
                # Add special content to end of text
                if not restored_text.endswith("\n"):
                    restored_text += "\n"
                restored_text += f"\n{start_tag}\n\n{content}\n\n{end_tag}\n"
    
    # Line break processing - ensure proper display in JSON output
    # This part doesn't affect JSON storage so no modification needed here
    
    return restored_text


def chatgpt_correct_text(original_text: str) -> Dict[str, Any]:
    """
    Use ChatGPT to correct OCR text
    
    Args:
        original_text: Original OCR text
        
    Returns:
        Dict: Correction results (corrected_text, confidence, special_content_corrections)
    """
    if not client:
        logger.error("OpenAI client not initialized. Check OPENAI_API_KEY.")
        return {"corrected_text": original_text, "confidence": 0.0, "special_content_corrections": {}}

    if not original_text:
        return {"corrected_text": "", "confidence": 0.0, "special_content_corrections": {}}

    # First simplify special content tags
    simplified_text = simplify_special_content_tags(original_text)
    
    # Extract special content and replace with placeholders
    placeholder_text, special_contents = extract_special_content(simplified_text)
    
    # Log: Original text length
    logger.info(f"Sending text to ChatGPT (length={len(placeholder_text)}).")

    # System prompt - correction guidelines (enhanced version)
    system_prompt = """You are an expert in accurately correcting Japanese OCR results. Please strictly follow these guidelines:

1. Identify and correct clear OCR errors based on context.
2. Mark text that is difficult to infer from context or where corrections might significantly alter content as [?text?].
3. Never change the original language of any text:
   - Keep Korean text in Korean.
   - Keep Japanese text in Japanese.
   - Keep English text in English.
   - Do not translate any language to another language.
4. Never modify or translate special area tags and content enclosed in brackets:
   - Special area tag formats: "[XXStart]", "[XXEnd]" or placeholders starting with "___SPECIAL_CONTENT_..."
   - These tags and placeholders contain important content that must be preserved exactly as is.
   - Within special areas, only correct obvious typos without deleting or omitting any content.
5. Delete content that is completely unnecessary in context (e.g., duplicate text, page numbers).
6. Add empty lines between paragraphs to improve readability.
7. Improve alignment of Markdown format tables and charts for better readability.
8. Return only the corrected text without explanations or comments.

Important: Maintain the original language of all text, and never delete or translate special area tags and content enclosed in brackets! This information is essential for ML training!
"""

    # User prompt - OCR text
    user_prompt = f"""The following is a Japanese OCR result. Please correct errors according to the guidelines above:

-----------
{placeholder_text}
-----------

Return only the corrected text without additional explanations or comments.
Never change the original language of any text. Keep Korean in Korean, Japanese in Japanese, and English in English.
Never delete or translate special area tags and content enclosed in brackets! This information is essential for ML training!
Do not delete or omit any content, only correct obvious typos.
"""

    try:
        # Call ChatGPT
        completion = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4" or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=4096
        )

        # Extract corrected text
        corrected_placeholder_text = completion.choices[0].message.content.strip()
        
        # Restore special content
        corrected_text = restore_special_content(corrected_placeholder_text, special_contents)
        
        # Calculate similarity
        sm = difflib.SequenceMatcher(None, original_text, corrected_text)
        confidence = sm.ratio()

        logger.info(f"ChatGPT response length={len(corrected_text)}, similarity={confidence:.3f}")
        
        return {
            "corrected_text": corrected_text,
            "confidence": confidence,
            "special_content_corrections": {}  # Can add special content correction info in future
        }
    except Exception as e:
        logger.error(f"ChatGPT error: {e}")
        return {
            "corrected_text": original_text, 
            "confidence": 0.0,
            "special_content_corrections": {}
        }


def chatgpt_correct_special_content(content_type: str, content: str) -> Dict[str, Any]:
    """
    Use ChatGPT to correct special content (formulas, figures, tables, etc.)
    
    Args:
        content_type: Content type (formula, figure, table, etc.)
        content: Original content
        
    Returns:
        Dict: Correction results (corrected_content, confidence)
    """
    # Return original content without correction
    logger.info(f"{content_type} content is kept as is without correction.")
    return {"corrected_content": content, "confidence": 1.0}


def extract_page_number_from_filename(filename: str) -> Optional[int]:
    """
    Extract page number from filename
    
    Args:
        filename: Filename (e.g., "page_7.json")
        
    Returns:
        Optional[int]: Extracted page number or None
    """
    match = re.search(r'page_(\d+)\.json', filename)
    if match:
        return int(match.group(1))
    return None


def process_page_stage2(page_data: Dict, original_blob_name: str, folder_name: str, subfolder: str) -> Dict[str, Any]:
    """
    Correct page OCR results with ChatGPT and save
    
    Args:
        page_data: Page OCR result data
        original_blob_name: Original blob name (e.g., "TOEFL/stage1/2010_1_B/page_7.json")
        folder_name: Parent folder name (e.g., "TOEFL")
        subfolder: Subfolder name (e.g., "2010_1_B")
        
    Returns:
        Dict: Processing results
    """
    # Extract page number from original filename
    filename = original_blob_name.split("/")[-1]
    page_number = extract_page_number_from_filename(filename)
    
    if page_number is None:
        # If page number can't be extracted, get from page data or use default
        page_number = page_data.get("page", 0)
        logger.warning(f"Could not extract page number from filename {filename}. Using page data or default value {page_number}.")
    
    # Extract original text - use text field already collected in stage1
    original_text = page_data.get("text", "")
    logger.info(f"Processing page {page_number} (folder: '{folder_name}', subfolder: '{subfolder}', original text length={len(original_text)})")

    # Correct text
    corrected = chatgpt_correct_text(original_text)
    corrected_text = corrected["corrected_text"]
    confidence = corrected["confidence"]
    special_content_corrections = corrected.get("special_content_corrections", {})

    # Construct result data - remove text_original field and change text_corrected to text
    result_data = {
        "page": page_number,
        "text": corrected_text,  # Save as text instead of text_corrected
        "confidence": confidence,
        "special_content_corrections": special_content_corrections,
        "processing_date": datetime.now().isoformat(),
        "stage": "stage2",
        "original_blob_name": original_blob_name
    }

    # Save result - maintain original page number
    page_filename = f"page_{page_number}_stage2.json"
    gcs_path = f"{folder_name}/stage2/{subfolder}/{page_filename}"
    output_url = save_json_to_gcs(result_data, gcs_path)

    if output_url:
        logger.info(f"Page {page_number} correction results saved: {output_url}")
    
    return {
        "page_number": page_number,
        "gcs_url": output_url,
        "confidence": confidence,
        "original_blob_name": original_blob_name
    }


def list_top_level_folders() -> List[str]:
    """
    List top-level folders in GCS bucket (improved version)
    
    Returns:
        List[str]: List of top-level folders
    """
    top_folders = set()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    logger.info(f"Listing top-level folders in bucket '{BUCKET_NAME}'")
    
    # List all blobs in bucket
    blobs = list(bucket.list_blobs())
    
    # Extract top-level folder from each blob path
    for blob in blobs:
        parts = blob.name.split('/')
        if len(parts) > 0 and parts[0]:  # Not empty string
            top_folders.add(parts[0])
    
    top_folders_list = list(top_folders)
    logger.info(f"Top-level folders found: {top_folders_list}")
    return top_folders_list


def check_stage1_exists(folder_name: str) -> bool:
    """
    Check if stage1 folder exists in folder
    
    Args:
        folder_name: Folder name
        
    Returns:
        bool: Whether stage1 folder exists
    """
    return check_folder_exists(f"{folder_name}/stage1/")


def check_stage2_exists(folder_name: str) -> bool:
    """
    Check if stage2 folder exists in folder
    
    Args:
        folder_name: Folder name
        
    Returns:
        bool: Whether stage2 folder exists
    """
    return check_folder_exists(f"{folder_name}/stage2/")


def list_stage1_subfolders(folder_name: str) -> List[str]:
    """
    Extract list of subfolders under stage1 in folder
    
    Args:
        folder_name: Folder name
        
    Returns:
        List[str]: List of subfolders
    """
    subfolders = set()
    bucket = storage_client.bucket(BUCKET_NAME)
    prefix = f"{folder_name}/stage1/"

    logger.info(f"Listing subfolders under prefix '{prefix}'")
    
    # List all blobs
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    # Extract subfolder from each blob path
    for blob in blobs:
        parts = blob.name.split("/")
        # Example: "TOEFL/stage1/2010_1_B/page_1.json" -> parts = ["TOEFL","stage1","2010_1_B","page_1.json"]
        if len(parts) >= 3 and parts[2]:  # Not empty string
            subfolders.add(parts[2])  # "2010_1_B"

    subfolders_list = list(subfolders)
    logger.info(f"Subfolders found: {subfolders_list}")
    return subfolders_list


def list_page_blobs(folder_name: str, subfolder: str) -> List[Any]:
    """
    List page_n.json files in specific subfolder
    
    Args:
        folder_name: Folder name
        subfolder: Subfolder name
        
    Returns:
        List[Any]: List of blobs
    """
    folder_prefix = f"{folder_name}/stage1/{subfolder}/"
    bucket = storage_client.bucket(BUCKET_NAME)

    logger.info(f"Listing page blobs under subfolder '{subfolder}' (prefix='{folder_prefix}')")
    
    # List all blobs
    all_blobs = list(bucket.list_blobs(prefix=folder_prefix))
    
    # Filter for page_n.json files
    page_blobs = [
        blob for blob in all_blobs
        if blob.name.endswith(".json") and "summary_stage1" not in blob.name
    ]
    
    # Sort by filename (maintain page order)
    page_blobs.sort(key=lambda b: b.name)
    logger.info(f"Found {len(page_blobs)} page blobs in subfolder '{subfolder}'")
    return page_blobs


def process_folder(folder_name: str) -> Dict[str, Any]:
    """
    Process stage1 data in folder to create stage2
    
    Args:
        folder_name: Folder name
        
    Returns:
        Dict: Processing results
    """
    results = {}
    
    # Check if stage1 folder exists
    if not check_stage1_exists(folder_name):
        logger.warning(f"No stage1 folder in folder '{folder_name}'. Skipping.")
        return results
    
    # Check if stage2 folder exists (skip if already exists)
    if check_stage2_exists(folder_name):
        logger.warning(f"Folder '{folder_name}' already has stage2 folder. Skipping.")
        return results
    
    # List stage1 subfolders
    subfolders = list_stage1_subfolders(folder_name)
    if not subfolders:
        logger.error(f"Could not find subfolders under stage1 in folder '{folder_name}'.")
        return results

    for subfolder in subfolders:
        logger.info(f"[Stage2] Folder: {folder_name}, Processing subfolder: {subfolder}")
        page_blobs = list_page_blobs(folder_name, subfolder)
        stage2_pages = []

        for blob in page_blobs:
            logger.info(f"  - Loading {blob.name}")
            try:
                page_json = json.loads(blob.download_as_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Error loading blob {blob.name}: {e}")
                continue

            # Pass original blob name to maintain page number
            page_result = process_page_stage2(page_json, blob.name, folder_name, subfolder)
            if page_result and page_result.get("gcs_url"):
                stage2_pages.append(page_result)

        # Sort by page number
        stage2_pages.sort(key=lambda p: p["page_number"])

        # Create summary_stage2.json for each subfolder
        summary = {
            "folder": folder_name,
            "subfolder": subfolder,
            "processing_date": datetime.now().isoformat(),
            "stage": "stage2",
            "pages": stage2_pages
        }
        summary_path = f"{folder_name}/stage2/{subfolder}/summary_stage2.json"
        summary_url = save_json_to_gcs(summary, summary_path)

        results[subfolder] = {
            "summary_url": summary_url,
            "pages": stage2_pages
        }
        logger.info(f"Folder: {folder_name}, Subfolder {subfolder} processing complete: {summary_url} (total pages={len(stage2_pages)})")

    return results


def process_all_folders() -> Dict[str, Dict[str, Any]]:
    """
    Process all top-level folders in GCS bucket
    
    Returns:
        Dict: Processing results
    """
    all_results = {}
    
    # List all top-level folders
    top_folders = list_top_level_folders()
    if not top_folders:
        logger.error(f"Could not find folders in bucket '{BUCKET_NAME}'.")
        return all_results
    
    for folder_name in top_folders:
        logger.info(f"Starting processing folder '{folder_name}'")
        
        # Process folder
        folder_results = process_folder(folder_name)
        
        if folder_results:
            all_results[folder_name] = folder_results
            logger.info(f"Folder '{folder_name}' processing complete")
        else:
            logger.info(f"No results for folder '{folder_name}' (no stage1 or stage2 already exists)")
    
    return all_results


def main():
    """
    Main function
    """
    global BUCKET_NAME
    parser = argparse.ArgumentParser(description="OCR System - Stage2 (ChatGPT Correction)")
    parser.add_argument("--bucket", type=str, default=BUCKET_NAME,
                        help=f"GCS bucket name (default: {BUCKET_NAME})")
    parser.add_argument("--folder", type=str, default=None,
                        help="Process specific folder only (processes all folders if not specified)")
    
    # Use parse_known_args() to ignore unknown arguments
    args, unknown = parser.parse_known_args()
    
    # Modify global variable BUCKET_NAME
    BUCKET_NAME = args.bucket
    
    logger.info(f"Starting OCR Stage2 - Bucket: {BUCKET_NAME}")
    
    if args.folder:
        # Process specific folder only
        logger.info(f"Starting processing folder '{args.folder}'")
        results = process_folder(args.folder)
        
        if results:
            logger.info(f"Folder '{args.folder}' processing complete. The following subfolders were processed:")
            for subfolder, info in results.items():
                logger.info(f"  {subfolder}: summary -> {info['summary_url']}")
        else:
            logger.info(f"No results for folder '{args.folder}' (no stage1 or stage2 already exists)")
    else:
        # Process all folders
        logger.info("Starting processing all folders")
        all_results = process_all_folders()
        
        if all_results:
            logger.info("All folders processing complete. The following folders were processed:")
            for folder, results in all_results.items():
                logger.info(f"Folder '{folder}':")
                for subfolder, info in results.items():
                    logger.info(f"  {subfolder}: summary -> {info['summary_url']}")
        else:
            logger.info("No folders were processed.")

if __name__ == "__main__":
    main()


# To customize output language, modify the system_prompt and user_prompt strings in the 
# chatgpt_correct_text() function, and update the SPECIAL_CONTENT_PATTERNS and SIMPLIFIED_TAGS
# dictionaries to match your desired language.
