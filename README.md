# PAD: Patch-Agnostic Defense Against Adversarial Patch Attacks
Code for the PAD paper, "Patch-Agnostic Defense against Adversarial Patch Attacks"

We provide code for detecting and defending against adversarial patches on aerial imagery using Segment Anything Model (SAM). The defense combines Mutual Information (MI) heatmaps and Change Detection (CD) heatmaps with SAM segmentation to achieve 100% patch detection rate on the INRIA Aerial Image Labeling Dataset.

We provide defense code that works on any aerial image and has been tested on INRIA dataset. With proper parameter tuning, this code can be adapted for different image domains (satellite, drone, object detection, face recognition, etc.).

# Overview

The PAD defense pipeline consists of 5 main steps:

Image Resizing - Resize large images to fit GPU memory (2048×2048)
Heatmap Generation - Generate MI and CD heatmaps to locate anomalies
SAM Segmentation - Segment image using Segment Anything Model
Patch Detection - Match heatmap regions with SAM masks using IOU
Patch Removal - Remove detected patches by masking them

# Step by Step Guide

1. Install the packages listed from the Software Installation Section (see below)
2. Download the dataset from the Dataset Section (see below)
3. Download the SAM model from the Models Section (see below)
4. Move the Models folder into the directory ./segment-anything/models/
5. Create test patches by running python create_test_patches.py
6. Run the defense using python run_pad.py
7. View results in the defended_images/ folder

# Software Installation
We use the following software packages:
- torch==2.7.1+cu118
- torchvision==0.22.1+cu118
- segment-anything==1.0
- opencv-python==4.13.0.92
- numpy==2.3.5
- scipy==1.17.1
- pillow==12.0.0
- matplotlib==3.10.8
- scikit-learn==1.8.0
- tqdm==4.67.3
- requests==2.32.5

# Instalation steps
Clone repository :
git clone https://github.com/mayageva1/PAD-implementation.git
cd PAD-implementation

Create virtual environment:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Dataset
INRIA Aerial Image Labeling Dataset
The defense has been tested on the INRIA Aerial Image Labeling Dataset. Download from https://project.inria.fr/aerialimagelabeling/

# Models
We use the following pre-trained model:
- ViT-L-16 https://github.com/facebookresearch/segment-anything#model-checkpoints
The ViT-L model is necessary to run the defense. The ViT-B model can be used for faster but slightly less accurate results.

# System Requirements
All code has been tested and verified on the URI HPC cluster with the following specifications:
- GPU: 1x NVIDIA GPU (12GB VRAM) - gypsum-gpu nodes
- CPU: 2 cores
- RAM: 8GB
- Disk: 50GB (for dataset + models + results)
- OS: Linux (CentOS 7, URI HPC cluster)

  # Contact
  For questions or concerns please contact the author at: maya.geva@uri.edu
  
