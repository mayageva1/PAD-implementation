"""
PAD: Patch-Agnostic Defense with Color Detection

IMPROVEMENT OVER ORIGINAL:
Original PAD only detects noise/entropy-based patches
This improved version ALSO detects solid color patches

Two detection methods combined:
1. Heatmap analysis (MI + CD) - catches noise patches
2. Color saturation analysis - catches solid color patches

Author: Maya Geva
Date: April 2026
"""

import numpy as np
import torch
import cv2
import os
from pathlib import Path
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import time

# ============= CONFIGURATION =============

iou_thre = 0.5
ratio_mi = 0.5
kernel_pram = 80
thresh_pram = 80

input_path = "./attacked_images/"
save_path = "./defended_images2/"
MAX_SIZE = 2048

# ============= COLOR DETECTION =============

def detect_color_patches(image):
    """
    IMPROVEMENT: Detect solid color patches
    
    Key insight: Patches are PURE COLORS (high saturation)
    Natural scenes have MIXED colors (low saturation)
    
    Args:
        image (np.ndarray): BGR image
        
    Returns:
        np.ndarray: Binary mask of color patches
    """
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract saturation channel (0-255)
    # High saturation = pure colors = likely patches
    saturation = hsv[:, :, 1]
    
    # Threshold: anything above 150 is considered a pure color
    # This catches red, blue, green, yellow, etc.
    color_mask = saturation > 150
    
    # Count detected pixels
    num_pixels = np.sum(color_mask)
    
    if num_pixels > 0:
        percentage = 100 * num_pixels / (image.shape[0] * image.shape[1])
        print(f"[Color Detection] Found {num_pixels} color anomaly pixels ({percentage:.2f}%)")
    else:
        print(f"[Color Detection] No color anomalies detected")
    
    return color_mask


# ============= HELPER FUNCTIONS (Original PAD) =============

def get_mask(image, mask_generator):
    """Generate SAM segmentation masks"""
    print(f"[SAM] Generating object masks...")
    masks = mask_generator.generate(image.astype(np.uint8))
    print(f"[SAM] Generated {len(masks)} masks")
    return masks


def heatmap_filter(fuse_img, threshold, ori_height, ori_width):
    """Filter heatmap using threshold and morphological operations"""
    h_t = np.where(fuse_img > threshold, 1, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_pram, kernel_pram))
    h_t_o = cv2.morphologyEx(h_t.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    h_t_o_c = cv2.morphologyEx(h_t_o, cv2.MORPH_CLOSE, kernel)
    h_t_o_c_o = cv2.dilate(h_t_o_c, kernel, iterations=1)
    return h_t, h_t_o, h_t_o_c, h_t_o_c_o


# ============= MAIN IMPROVED PAD CLASS =============

class PADDefenseImproved:
    """
    Improved PAD with Color Detection
    
    Combines two detection methods:
    1. Heatmap-based (MI + CD) - for entropy/noise anomalies
    2. Color-based (saturation) - for solid color patches
    
    Result: Better detection of diverse patch types
    """
    
    def __init__(self, checkpoint_path="segment-anything/models/sam_vit_l_0b3195.pth",
                 device="cuda:0", iou_threshold=0.5):
        """
        Initialize improved PAD
        
        Args:
            checkpoint_path: Path to SAM model
            device: GPU device
            iou_threshold: Detection threshold
        """
        
        self.device = device
        self.iou_threshold = iou_threshold
        
        print(f"[PAD Improved] Loading SAM model...")
        self.sam = sam_model_registry["vit_l"](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        print(f"[PAD Improved] Model loaded successfully")
    
    def defend(self, image_path, output_dir=save_path):
        """
        Full improved defense pipeline
        
        Steps:
        1. Load and resize image
        2. Generate heatmaps (MI + CD)
        3. Generate SAM masks
        4. Heatmap-based detection
        5. COLOR-BASED DETECTION (improvement)
        6. Combine both detections
        7. Remove detected patches
        8. Save result
        
        Args:
            image_path: Path to attacked image
            output_dir: Directory to save defended image
            
        Returns:
            dict: Results with detection metrics
        """
        
        start_time = time.time()
        name = Path(image_path).stem
        
        print(f"\n{'='*70}")
        print(f"[PAD Improved] Processing: {name}")
        print(f"{'='*70}")
        
        # ===== STEP 1: LOAD AND RESIZE =====
        print(f"\n[Step 1/8] Loading image...")
        ori_img = Image.open(image_path).convert('RGB')
        ori_width, ori_height = ori_img.size
        print(f"  Original size: {ori_height} x {ori_width}")
        
        # Resize for GPU memory constraints
        if ori_height > MAX_SIZE or ori_width > MAX_SIZE:
            scale = MAX_SIZE / max(ori_height, ori_width)
            new_h = int(ori_height * scale)
            new_w = int(ori_width * scale)
            ori_img = ori_img.resize((new_w, new_h))
            ori_height, ori_width = new_h, new_w
            print(f"  Resized to: {ori_height} x {ori_width}")
            
            resized_impath = f"/tmp/{name}_resized.png"
            ori_img.save(resized_impath)
            impath = resized_impath
        else:
            impath = image_path
        
        # ===== STEP 2: GENERATE HEATMAPS =====
        print(f"\n[Step 2/8] Generating heatmaps (MI + CD)...")
        from fuse_filter import fuse_heatmap
        
        mi_img, cd_img, fuse_img = fuse_heatmap(impath, ori_height, ori_width)
        print(f"  Heatmap range: {np.min(fuse_img):.4f} - {np.max(fuse_img):.4f}")
        
        # ===== STEP 3: FILTER HEATMAP =====
        print(f"\n[Step 3/8] Applying heatmap threshold...")
        threshold = np.percentile(fuse_img, thresh_pram)
        h_t, h_t_o, h_t_o_c, h_t_o_c_o = heatmap_filter(fuse_img, threshold, ori_height, ori_width)
        gray = np.where(h_t_o_c_o > 0, 1, 0)
        print(f"  Heatmap anomalies: {np.sum(gray)} pixels")
        
        # ===== STEP 4: LOAD IMAGE FOR PROCESSING =====
        print(f"\n[Step 4/8] Loading image for SAM...")
        rgb_color = cv2.imread(impath)
        image = cv2.cvtColor(rgb_color, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # ===== STEP 5: COLOR-BASED DETECTION (IMPROVEMENT!) =====
        print(f"\n[Step 5/8] Detecting color-based patches...")
        color_mask = detect_color_patches(image)
        
        # ===== STEP 6: GENERATE SAM MASKS =====
        print(f"\n[Step 6/8] Running SAM segmentation...")
        masks = get_mask(image, self.mask_generator)
        
        # ===== STEP 7: IMPROVED DETECTION =====
        print(f"\n[Step 7/8] Detecting patches via combined method...")
        
        result_mask = np.zeros((h, w))
        patches_detected_heatmap = 0
        patches_detected_color = 0
        patches_detected_total = 0
        patch_ious = []
        
        for k in range(len(masks)):
            mask_k = masks[k].get('segmentation')
            
            # METHOD 1: Heatmap-based detection (original PAD)
            n_heat = mask_k & gray
            u_heat = mask_k
            iou_heat = np.sum(n_heat) / (np.sum(u_heat) + 1e-8)
            
            # METHOD 2: Color-based detection (improvement)
            n_color = mask_k & color_mask
            u_color = mask_k
            iou_color = np.sum(n_color) / (np.sum(u_color) + 1e-8)
            
            # Spatial consistency check
            n_1 = mask_k & result_mask.astype(np.uint8)
            u_1 = mask_k
            iou1 = np.sum(n_1) / (np.sum(u_1) + 1e-8)
            
            # IMPROVED DECISION: Accept if EITHER method detects it
            detected_by_heatmap = (iou_heat > self.iou_threshold)
            detected_by_color = (iou_color > 0.3)  # Lower threshold for color
            
            if (detected_by_heatmap or detected_by_color) and iou1 < 0.1:
                # PATCH DETECTED - Remove it
                rgb_color[mask_k, :] = [0, 0, 0]  # Set to black
                
                result_mask = result_mask.astype(np.uint8) | mask_k
                patches_detected_total += 1
                patch_ious.append(max(iou_heat, iou_color))
                
                if detected_by_heatmap:
                    patches_detected_heatmap += 1
                
                if detected_by_color:
                    patches_detected_color += 1
                
                if k < 10:  # Print first 10
                    detection_method = "Heatmap" if detected_by_heatmap else "Color"
                    print(f"  Mask {k}: Heatmap={iou_heat:.3f}, Color={iou_color:.3f} → REMOVED ({detection_method})")
        
        # ===== STEP 8: SAVE RESULTS =====
        print(f"\n[Step 8/8] Saving results...")
        
        output_file = os.path.join(output_dir, name + ".png")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_file, rgb_color)
        
        elapsed_time = time.time() - start_time
        
        # ===== RESULTS =====
        print(f"\n{'='*70}")
        print(f"[RESULTS]")
        print(f"{'='*70}")
        print(f"Detection Method Breakdown:")
        print(f"  Detected by heatmap: {patches_detected_heatmap} patches")
        print(f"  Detected by color: {patches_detected_color} patches")
        print(f"  Total detected: {patches_detected_total} patches")
        
        if patch_ious:
            print(f"\nDetection Quality:")
            print(f"  Mean IOU: {np.mean(patch_ious):.3f}")
            print(f"  Max IOU: {np.max(patch_ious):.3f}")
            print(f"  Min IOU: {np.min(patch_ious):.3f}")
        
        print(f"\nProcessing time: {elapsed_time:.2f} seconds")
        print(f"✓ Saved: {output_file}\n")
        
        return {
            'image': name,
            'patches_heatmap': patches_detected_heatmap,
            'patches_color': patches_detected_color,
            'patches_total': patches_detected_total,
            'mean_iou': np.mean(patch_ious) if patch_ious else 0,
            'max_iou': np.max(patch_ious) if patch_ious else 0,
            'time': elapsed_time
        }


# ============= BATCH PROCESSING =============

def process_batch(input_dir, output_dir):
    """Process multiple images"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pad = PADDefenseImproved(device="cuda:0")
    
    image_files = sorted(Path(input_dir).glob("*.png"))
    print(f"\nProcessing {len(image_files)} images...\n")
    
    all_results = []
    
    for image_file in image_files:
        result = pad.defend(str(image_file), output_dir=output_dir)
        all_results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    
    total_heatmap = sum(r['patches_heatmap'] for r in all_results)
    total_color = sum(r['patches_color'] for r in all_results)
    total_patches = sum(r['patches_total'] for r in all_results)
    
    print(f"Total images processed: {len(all_results)}")
    print(f"\nDetection Breakdown:")
    print(f"  Total patches detected by heatmap: {total_heatmap}")
    print(f"  Total patches detected by color: {total_color}")
    print(f"  Total patches detected: {total_patches}")
    
    if total_patches > 0:
        avg_iou = np.mean([r['mean_iou'] for r in all_results if r['mean_iou'] > 0])
        print(f"  Average IOU: {avg_iou:.3f}")
    
    total_time = sum(r['time'] for r in all_results)
    print(f"\nTotal processing time: {total_time:.1f}s")
    print(f"Average per image: {total_time/len(all_results):.1f}s")
    
    return all_results


# ============= MAIN EXECUTION =============

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("PAD: Improved Patch-Agnostic Defense")
    print("With Color Detection")
    print("="*70)
    
    with torch.no_grad():
        results = process_batch(
            input_dir=input_path,
            output_dir=save_path
        )
        
        print(f"\n✓ Defense complete!")
        print(f"✓ Defended images saved to: {save_path}")