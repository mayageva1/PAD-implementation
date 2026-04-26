import cv2
import numpy as np
from pathlib import Path

# Get your images
image_dir = Path("/home/maya_geva_uri_edu/AerialImageDataset/train/images")
output_dir = Path("./attacked_images")
output_dir.mkdir(exist_ok=True)

# Create attacked versions for first 5 images
for img_file in sorted(image_dir.glob("*.tif"))[:5]:
    print(f"Creating attacked version: {img_file.name}")
    
    # Load image
    img = cv2.imread(str(img_file))
    
    # Create adversarial patch (red square)
    h, w = img.shape[:2]
    patch_h, patch_w = int(h * 0.15), int(w * 0.15)
    y, x = int(h * 0.4), int(w * 0.4)
    
    # Add patch
    attacked = img.copy()
    attacked[y:y+patch_h, x:x+patch_w] = [0, 0, 255]  # Red patch
    
    # Save
    name = img_file.stem
    cv2.imwrite(str(output_dir / f"{name}_attacked.png"), attacked)
    
    # Also save clean for comparison
    cv2.imwrite(str(output_dir / f"{name}_clean.png"), img)

print(f"✓ Created attacked images in {output_dir}/")