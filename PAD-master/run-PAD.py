"""
PAD Defense - Main Execution Script

Author: Maya Geva
Date: April 2026
Based on: PAD: Patch-Agnostic Defense Against Adversarial Patch Attacks

Key Changes from Original:
- Resized images to 2048x2048 to fit GPU memory (11GB VRAM)
- Added adaptive path handling for URI HPC cluster
- Used temporary files for resized images
- Added detailed logging for debugging

Input: Attacked aerial images from INRIA dataset
Output: Defended images with patches removed
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import glob
import os
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

import math
from PIL import Image

from heatmap_MI import img_heatmap_mi
from heatmap_CD import img_heatmap_cd
from fuse_filter import fuse_heatmap, heatmap_filter

# ============= CONFIGURATION =============
iou_thre = 0.5 # Intersection-over-union threshold for patch detection
ratio_mi = 0.5 # Ratio between MI and CD heatmaps (ratio_cd = 1-ratio_mi)
kernel_pram = 80 # Morphological kernel size for heatmap processing
thresh_pram = 80  # Percentile threshold (80th percentile = top 20% anomalies)

# INPUT: Where attacked images are stored
input_path = "./attacked_images/"
# OUTPUT: Where defended images will be saved
save_path = "./defended_images/"

# ============= HELPER FUNCTIONS =============

def get_mask(image, mask_generator):
    """
    Generate object segmentation masks using Segment Anything Model (SAM)
    
    SAM is a large vision transformer that can segment any object in an image
    without fine-tuning. We use it to get precise object boundaries.
    
    Args:
        image (np.ndarray): Input image (H x W x 3, RGB format)
        mask_generator: SAM mask generator instance
        
    Returns:
        list: List of mask dictionaries, each containing:
            - 'segmentation': boolean mask array
            - 'area': pixel count
            - 'bbox': bounding box
            - 'predicted_iou': SAM's confidence score
            
    slow (~30-60 sec per image) but very accurate
    """
    masks = mask_generator.generate(image.astype(np.uint8))
    return masks

# ============= MAIN EXECUTION =============

if __name__ == "__main__":
    """
    Main PAD Defense Pipeline
    
    Steps:
    1. Load image from disk
    2. Resize if too large (GPU memory constraint)
    3. Generate MI and CD heatmaps
    4. Fuse heatmaps to locate anomalies
    5. Generate SAM segmentation masks
    6. Match heatmap regions with SAM masks
    7. Remove detected patches
    8. Save defended image
    """

    device = "cuda:0" # Use first GPU (gypsum-gpu nodes have 1-2 GPUs)
    #Load pre-trained SAM ViT-L model
    # ViT-L: Better quality but slower (1.2GB)
    # Could use ViT-B for faster processing (375MB)
    sam = sam_model_registry["vit_l"](checkpoint="segment-anything/models/sam_vit_l_0b3195.pth")
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    print(save_path)
    folder = os.path.exists(save_path)

    # Create output directory if it doesn't exist
    if not folder:
        os.makedirs(save_path)

    # Create output directory if it doesn't exist

    with torch.no_grad(): # Don't compute gradients (not training)
        data_dir = input_path
        data_files = os.listdir(data_dir)
        for data_file in data_files:
            print(data_file)
            name = data_file.split(".")[0] # Get filename without extension
            impath = data_dir + data_file
            
            # ===== STEP 1: LOAD IMAGE =====
            ori_img = Image.open(impath).convert('RGB')
            ori_width, ori_height = ori_img.size
            print("ori_height , ori_width", ori_height, ori_width)

            # ===== STEP 2: RESIZE FOR GPU MEMORY =====
            # Original images are 5000x5000, but GPU can't handle this
            # Resizing to 2048x2048 reduces memory by ~16x
            MAX_SIZE = 2048
            if ori_height > MAX_SIZE or ori_width > MAX_SIZE:
                scale = MAX_SIZE / max(ori_height, ori_width)
                new_h = int(ori_height * scale)
                new_w = int(ori_width * scale)
                ori_img = ori_img.resize((new_w, new_h))
                ori_height, ori_width = new_h, new_w
                print(f"Resized to {ori_height}x{ori_width}")
                
                # Save resized image temporarily (fuse_heatmap reads from disk)
                resized_impath = f"/tmp/{name}_resized.png"
                ori_img.save(resized_impath)
                impath = resized_impath  # ← Use resized path
            
            # Now compute heatmaps on RESIZED image
            mi_img, cd_img, fuse_img = fuse_heatmap(impath, ori_height, ori_width)
            
            # ===== STEP 4: FILTER HEATMAP =====
            # Apply threshold to get regions of interest
            threshold = np.percentile(fuse_img, thresh_pram)
            h_t, h_t_o, h_t_o_c, h_t_o_c_o = heatmap_filter(fuse_img, threshold, ori_height, ori_width)

            # Convert to binary mask (1 = anomaly, 0 = normal)
            gray = np.where(h_t_o_c_o > 0, 1, 0)
            
            # ===== STEP 5: LOAD IMAGE FOR SAM =====
            rgb_color = cv2.imread(impath)  # ← This will load resized version
            image = cv2.cvtColor(rgb_color, cv2.COLOR_BGR2RGB)

            h = image.shape[0]
            w = image.shape[1]

            # ===== STEP 6: SEGMENT WITH SAM =====            
            mask = get_mask(image, mask_generator)
            print(len(mask))
        
            # ===== STEP 7: DETECT & REMOVE PATCHES =====
            result_mask = np.zeros((h, w)) # Track which regions already processed
            
            # For each SAM-generated mask, check if it matches heatmap anomaly
            for k in range(len(mask)):
                mask_k = mask[k].get('segmentation')
                # Calculate how much overlap between mask and heatmap anomaly
                n = mask_k & gray # Intersection
                u = mask_k # Union
                iou = np.sum(n) / (np.sum(u))
                print("iou", iou)

                # Check spatial consistency (don't remove overlapping masks)
                n_1 = mask_k & result_mask.astype(np.uint8)
                u_1 = mask_k
                iou1 = np.sum(n_1) / (np.sum(u_1))
                print("iou1", iou1)

                # Accept mask if:
                # 1. High IOU with heatmap anomaly (IOU > 0.5 = patch detected)
                # 2. Low overlap with previously detected patches (IOU1 < 0.1 = no spatial collision)               
                if(iou > iou_thre and iou1 < 0.1):

                    # Remove patch by masking it out (set to 0 = black)

                    mask_k_save = np.expand_dims(mask_k, axis=2)
                    mask_k_save = np.tile(mask_k_save, 3)
                    rgb_color = rgb_color * (~mask_k_save) # Zero out patch pixels
                    result_mask = result_mask.astype(np.uint8) | mask_k 

            # Save defended image
            cv2.imwrite(save_path + name + ".png", rgb_color)