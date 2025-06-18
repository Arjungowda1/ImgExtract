import os
import torch


# Set device based on CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

SCORE_THRESHOLD = 0.9 # Threshold for filtering out low-confidence predictions
NMS_THRESHOLD = 0.5 # Threshold for Non-Maximum Suppression (reducing overlapping boxes)
MODEL_NAME = "model_v2.pth" # Name of the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_NAME) # Path to the model file
BASE_CONFIG_PATH = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" # Base configuration file for the model
NUM_CLASSES = 1 # Number of classes in the COCO dataset (currently 1 - drawing class)

# Test configurations
THIS_DIR = os.path.realpath(__file__).rpartition('/')[0]
TEST_DIR = os.path.join(THIS_DIR, "test")
TEST_IMAGE_DIR = os.path.join(TEST_DIR, "samples")  # Directory containing test images
TEST_JSON_PATH = os.path.join(TEST_DIR, "ground_truth.json")  # Path to ground truth data

# Test thresholds
SCORE_DIFF_THRESH = 0.05  # Maximum allowable difference in confidence scores
BBOX_DIFF_THRESH = 0.05   # Maximum allowable difference in bounding box