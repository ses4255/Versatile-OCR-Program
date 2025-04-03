# File Organization Guide

This guide explains how to organize the project files into version-specific folders for better management.

### 1. Create the `v1.0_initial` Folder

In your project directory, create the folder to store the initial version files:

```bash
mkdir -p v1.0_initial/04-02-2025

# Move your Python code files
mv advanced_ocr.py v1.0_initial/04-02-2025/
mv OCR_stage1.py v1.0_initial/04-02-2025/
mv custom_doclayout_yolo.py v1.0_initial/04-02-2025/
mv ocr_stage2.py v1.0_initial/04-02-2025/


mkdir -p sample_outputs

mv Biology_Converted.jpeg sample_outputs/
mv Math_Converted.jpeg sample_outputs/
mv Biology_Original.jpeg sample_outputs/
mv Math_Original.jpeg sample_outputs/

git add v1.0_initial/ sample_outputs/
git commit -m "Reorganize project files into v1.0_initial and sample_outputs folders"
git push origin main  # Push to your GitHub repository
