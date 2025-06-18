import json
import os
import sys
from PIL import Image
import numpy as np
from typing import List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from inference.inference import inference_image
from utility.config import TEST_IMAGE_DIR, TEST_JSON_PATH
from utility.utils import bbox_iou

def calculate_metrics(predictions: List[dict], ground_truth: List[dict], iou_threshold: float = 0.5) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score for object detection.
    
    Args:
        predictions: List of predicted boxes and scores
        ground_truth: List of ground truth boxes
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    if not predictions and not ground_truth:
        return 1.0, 1.0, 1.0  # Perfect score if both are empty
    if not predictions or not ground_truth:
        return 0.0, 0.0, 0.0  # Zero score if one is empty
    
    # Sort predictions by confidence score
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
    
    # Initialize arrays to track matched predictions and ground truth
    matched_preds = [False] * len(predictions)
    matched_gt = [False] * len(ground_truth)
    
    # Match predictions to ground truth
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            if not matched_gt[j] and bbox_iou(pred["box"], gt["box"]) >= iou_threshold:
                matched_preds[i] = True
                matched_gt[j] = True
                break
    
    # Calculate metrics
    true_positives = sum(matched_preds)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - sum(matched_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def calculate_map(predictions: List[dict], ground_truth: List[dict], iou_threshold: float = 0.5) -> float:
    """
    Calculate mean Average Precision (mAP) for object detection.
    
    Args:
        predictions: List of predicted boxes and scores
        ground_truth: List of ground truth boxes
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        mAP score
    """
    if not predictions and not ground_truth:
        return 1.0  # Perfect score if both are empty
    if not predictions or not ground_truth:
        return 0.0  # Zero score if one is empty
    
    # Sort predictions by confidence score
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
    
    # Initialize arrays to track matched predictions and ground truth
    matched_preds = [False] * len(predictions)
    matched_gt = [False] * len(ground_truth)
    
    # Calculate precision and recall at each threshold
    precisions = []
    recalls = []
    
    for i, pred in enumerate(predictions):
        # Match prediction to ground truth
        for j, gt in enumerate(ground_truth):
            if not matched_gt[j] and bbox_iou(pred["box"], gt["box"]) >= iou_threshold:
                matched_preds[i] = True
                matched_gt[j] = True
                break
        
        # Calculate precision and recall at current threshold
        true_positives = sum(matched_preds[:i+1])
        false_positives = (i + 1) - true_positives
        false_negatives = len(ground_truth) - sum(matched_gt)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP using 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if len([r for r in recalls if r >= t]) == 0:
            p = 0
        else:
            p = max([p for r, p in zip(recalls, precisions) if r >= t])
        ap += p / 11.0
    
    return ap

def main():
    # Load ground truth data
    with open(TEST_JSON_PATH, 'r') as fp:
        ground_truth_data = json.load(fp)
    
    # Initialize metrics
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_map = 0
    num_images = len(ground_truth_data)
    
    # Process each image
    for image_fname, gt_data in ground_truth_data.items():
        image_path = os.path.join(TEST_IMAGE_DIR, image_fname)
        image = Image.open(image_path).convert("RGB")
        
        # Get predictions
        predictions = inference_image(image, draw=False)
        if predictions is None:
            predictions = []
        
        # Calculate metrics for this image
        precision, recall, f1 = calculate_metrics(predictions, gt_data)
        map_score = calculate_map(predictions, gt_data)
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_map += map_score
    
    # Calculate average metrics
    avg_precision = total_precision / num_images
    avg_recall = total_recall / num_images
    avg_f1 = total_f1 / num_images
    avg_map = total_map / num_images
    
    print("\nModel Performance Metrics:")
    print(f"Precision: {avg_precision:.3f}")
    print(f"Recall: {avg_recall:.3f}")
    print(f"F1 Score: {avg_f1:.3f}")
    print(f"mAP: {avg_map:.3f}")
    
    return avg_precision, avg_recall, avg_f1, avg_map

if __name__ == "__main__":
    main() 