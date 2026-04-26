#!/bin/bash

# fix_paths.sh - Replace all hardcoded paths in PAD code
# Updated to include attacked images path

echo "=========================================="
echo "PAD Path Replacement Script"
echo "=========================================="

# Backup original files
echo "[1/5] Creating backups..."
mkdir -p backups/
cp *.py backups/
echo "✓ Backups saved to ./backups/"

# Create output directories
echo "[2/5] Creating output directories..."
mkdir -p ./output
mkdir -p ./defended_images/
mkdir -p ./heatmap_visualizations/
mkdir -p ./mask_visualizations/
mkdir -p ./input_images/
mkdir -p ./attacked_images/
echo "✓ Output directories created"

# Replace paths - MAIN CONVERSIONS
echo "[3/5] Replacing hardcoded paths..."

# Replace /home/dell paths with ./output
sed -i 's|/home/dell/jlh/my_patch_defense/code/[^"'\'']*|./output|g' *.py

# Replace defended image paths
sed -i 's|/home/dell/jlh/ultralytics/ultralytics/datasets/inria/[^"'\'']*defended[^"'\'']*|./defended_images|g' *.py

# Replace attacked image paths
sed -i 's|/home/dell/jlh/[^"'\'']*attacked[^"'\'']*|./attacked_images|g' *.py

# Replace input paths (clean images)
sed -i 's|/home/dell/jlh/ultralytics/ultralytics/datasets/inria/images/inria_P[0-9]/|./input_images/|g' *.py

# Replace any remaining /home/dell paths
sed -i 's|/home/dell/[^"'\'']*|./output|g' *.py

echo "✓ Path replacements complete"

# Verify replacements
echo "[4/5] Verifying replacements..."
echo ""
echo "Remaining suspicious paths (if any):"
grep -r "/home/" *.py || echo "✓ No /home/ paths found"
echo ""
echo "savefig calls:"
grep "savefig" *.py | head -5
echo ""
echo "imwrite calls:"
grep "imwrite" *.py | head -5

# Print configuration
echo ""
echo "[5/5] Configuration Summary:"
echo "=========================================="
echo "Input Images:        ./input_images/"
echo "Attacked Images:     ./attacked_images/"
echo "Defended Images:     ./defended_images/"
echo "Heatmap Viz:         ./heatmap_visualizations/"
echo "Mask Viz:            ./mask_visualizations/"
echo "Output Files:        ./output/"
echo "=========================================="
echo "✓ DONE! Paths have been replaced."
echo "=========================================="