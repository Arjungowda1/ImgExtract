import json
import os
import random
import sys
from PIL import Image

# python does not automatically find parent directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from inference.inference import inference_image
from utility.utils import compare_detections
from utility.config import TEST_IMAGE_DIR, TEST_JSON_PATH, BBOX_DIFF_THRESH, SCORE_DIFF_THRESH

def test_all_entries():
    """
    Test all images in the test directory against ground truth
    """
    with open(TEST_JSON_PATH, 'r') as fp:
        TEST_DATA = json.load(fp)
    
    for image_fname in TEST_DATA:
        image_path = os.path.join(TEST_IMAGE_DIR, image_fname)
        image = Image.open(image_path).convert("RGB")
        
        # Get predictions
        results = inference_image(image, draw=False)
        if results is None:
            results = []
        
        # Extract boxes and scores from results
        boxes1 = [result["box"] for result in results]
        scores1 = [result["score"] for result in results]
        
        # Get ground truth
        boxes2 = TEST_DATA[image_fname]['boxes']
        scores2 = TEST_DATA[image_fname]['scores']
        
        assert compare_detections(
            boxes1, boxes2, 
            scores1, scores2,
            bbox_diff_th=BBOX_DIFF_THRESH,
            score_diff_th=SCORE_DIFF_THRESH
        ), f"Detections for {image_fname} don't match ground truth"
        print(f"Test passed for {image_fname}")

def test_single_entry():
    """
    Test a single random image from the test directory
    """
    with open(TEST_JSON_PATH, 'r') as fp:
        TEST_DATA = json.load(fp)
    
    image_fname = random.choice(sorted(TEST_DATA.keys()))
    image_path = os.path.join(TEST_IMAGE_DIR, image_fname)
    image = Image.open(image_path).convert("RGB")
    
    # Get predictions
    results = inference_image(image, draw=False)
    if results is None:
        results = []
    
    # Extract boxes and scores from results
    boxes1 = [result["box"] for result in results]
    scores1 = [result["score"] for result in results]
    
    # Get ground truth
    boxes2 = TEST_DATA[image_fname]['boxes']
    scores2 = TEST_DATA[image_fname]['scores']
    
    assert compare_detections(
        boxes1, boxes2,
        scores1, scores2,
        bbox_diff_th=BBOX_DIFF_THRESH,
        score_diff_th=SCORE_DIFF_THRESH
    ), f"Detections for {image_fname} don't match ground truth"
    print(f"Test passed for {image_fname}")

def generate_ground_truth():
    """
    Generate ground truth data from test images
    """
    data = {}
    for image_fname in os.listdir(TEST_IMAGE_DIR):

        image_path = os.path.join(TEST_IMAGE_DIR, image_fname)
        image = Image.open(image_path).convert("RGB")
        
        results = inference_image(image, draw=False)
        if results is None:
            results = []
        
        boxes = [result["box"] for result in results]
        scores = [result["score"] for result in results]
        
        data[image_fname] = {
            "boxes": boxes,
            "scores": scores
        }
    
    with open(TEST_JSON_PATH, 'w') as fp:
        json.dump(data, fp, indent=4)
    print(f"Generated ground truth data for {len(data)} images")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate_ground_truth()
    else:
        test_single_entry() 