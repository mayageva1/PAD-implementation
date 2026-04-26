# Changes from the original code

# 1. Image resizing for the URI HCP Cluster GPU memory
- Original images (5000×5000) caused a GPU OOM error. GPU has 11.91GB, needs 17.88GB
- I changed it to resize images to 2048x2048 before processesing, reduced memory usage by 16x

Added:
if ori_height > MAX_SIZE or ori_width > 2048:
    scale = MAX_SIZE / max(ori_height, ori_width)
    new_h, new_w = int(ori_height * scale), int(ori_width * scale)
    ori_img = ori_img.resize((new_w, new_h))
    ori_height, ori_width = new_h, new_w

# 2. Path Handling
Original (doesn't exist on my machine):
input_path = "/home/dell/jlh/ultralytics/ultralytics/datasets/inria/images/inria_P6/"

My version:
input_path = "./attacked_images/"
save_path = "./defended_images/"

# Files Modified
run-PAD.py - Added comments + changes
fix_paths.sh - Created path replacement script
create_test_patches.py - Creates adversarial patches