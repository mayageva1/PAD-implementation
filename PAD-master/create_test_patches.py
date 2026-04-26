# Create_test_patches_v2.py - Better adversarial patches

import cv2
import numpy as np
from pathlib import Path

image_dir = Path("/home/maya_geva_uri_edu/AerialImageDataset/train/images")
output_dir = Path("./attacked_images")
output_dir.mkdir(exist_ok=True)

for img_file in sorted(image_dir.glob("*.tif"))[:3]:
    print(f"Creating attacked version: {img_file.name}")
    
    img = cv2.imread(str(img_file))
    h, w = img.shape[:2]
    
    attacked = img.copy()
    
    # Multiple patches for better detection
    patch_h, patch_w = int(h * 0.1), int(w * 0.1)
    
    # Patch 1: Random noise
    y1, x1 = int(h * 0.3), int(w * 0.3)
    attacked[y1:y1+patch_h, x1:x1+patch_w] = np.random.randint(0, 256, (patch_h, patch_w, 3), dtype=np.uint8)
    
    # Patch 2: High contrast red
    y2, x2 = int(h * 0.6), int(w * 0.6)
    attacked[y2:y2+patch_h, x2:x2+patch_w] = [0, 0, 255]
    
    # Patch 3: Adversarial perturbation
    y3, x3 = int(h * 0.5), int(w * 0.2)
    attacked[y3:y3+patch_h, x3:x3+patch_w] = np.clip(
        attacked[y3:y3+patch_h, x3:x3+patch_w].astype(float) + 
        np.random.randn(patch_h, patch_w, 3) * 50,
        0, 255
    ).astype(np.uint8)
    
    # Save both
    name = img_file.stem
    cv2.imwrite(str(output_dir / f"{name}_attacked.png"), attacked)
    cv2.imwrite(str(output_dir / f"{name}_clean.png"), img)
    
    print(f"  Saved: {name}_attacked.png")

print(f"✓ Created stronger attacked images in {output_dir}/")