import logging
import io
import zipfile
import warnings
from PIL import Image

def configure_warnings():
    """
    Configure warning filters for the application.
    This function should be called at the start of the application.
    """
    warnings.filterwarnings(
        "ignore",
        message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument"
    )

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name (str): The name of the logger.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.propagate = False  # Prevent propagation to parent loggers
    
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # Filter out warnings from detectron2 library
        handler.addFilter(lambda record: record.levelno != logging.WARNING) 
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def create_zip(images: list) -> io.BytesIO:
    """
    Create a zip file containing images with bounding boxes drawn on them.
    
    Args:
        images (list): List of tuples containing image filename and image object.
        
    Returns:
        io.BytesIO: A BytesIO object containing the zip file.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, image in images:
            if image is not None:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                zip_file.writestr(filename, img_byte_arr.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

def get_images(image: Image, bbox: list) -> list:
    """
    Extract images from the original image based on bounding boxes.
    
    Args:
        image (Image): The original image.
        bbox (list): List of bounding boxes to extract images from.
        
    Returns:
        list: List of extracted images.
    """
    extracted_images = []
    for box in bbox:
        xmin, ymin, xmax, ymax = map(int, box)
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        extracted_images.append(cropped_image)
    return extracted_images

def compare_detections(boxes1, boxes2, scores1, scores2, bbox_diff_th, score_diff_th):
    """
    Compare two sets of detections to see if they match within thresholds
    Args:
        boxes1, boxes2: List of bounding boxes [x1, y1, x2, y2]
        scores1, scores2: List of confidence scores
        bbox_diff_th: Threshold for bounding box difference
        score_diff_th: Threshold for score difference
    Returns:
        bool: True if detections match within thresholds
    """
    assert len(boxes1) == len(boxes2), "number of boxes not matching"
    for box1, box2, score1, score2 in zip(boxes1, boxes2, scores1, scores2):
        bbox_diff = 1.0 - bbox_iou(box1, box2)
        score_diff = abs(score1 - score2)
        print(f"bbox iou diff: {bbox_diff:.3f}, score diff: {score_diff:.3f}")
        if (bbox_diff > bbox_diff_th) or (score_diff > score_diff_th):
            return False
    return True

def bbox_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]
    Returns:
        float: IoU score
    """
    # Calculate intersection area
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    
    intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0