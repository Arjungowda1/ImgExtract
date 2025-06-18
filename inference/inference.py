from utility.utils import configure_warnings, get_logger
from inference.load_model import get_predictor
import numpy as np
import time
from PIL import ImageDraw, ImageFont, Image

configure_warnings()
logger = get_logger(__name__)

def inference_image(image: Image, draw: bool) -> list:
    """
    Performs inference on a single image using a model trained with detectron2.

    Args:
        image (Image): The input image to be processed.

    Returns:
        list: A list containing bbox of drawings, scores and class[currently one].
    """
    predictor = get_predictor()
    
    logger.info("[Inference] Starting inference on the image...")

    start_time = time.perf_counter()
    outputs = predictor(np.array(image))
    end_time = time.perf_counter()

    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()
    classes = outputs["instances"].pred_classes.cpu().numpy()

    results = []
    for box, score, cls in zip(boxes, scores, classes):
        results.append({
            "box": box.tolist(),  # [xmin, ymin, xmax, ymax]
            "score": float(score),
            "class": int(cls)
        })

    logger.info(f"[Inference] Processed in {end_time - start_time:.2f} seconds.")
    if not results:
        logger.info("[Inference] No drawings detected in the image.")
        return None
    if not draw:
        return results
    else:
        logger.info("[Inference] Drawing boxes on the image...")
        draw_image = image.copy()
        draw_obj = ImageDraw.Draw(draw_image)
        font = ImageFont.load_default()

        for result in results:
            box = result["box"]
            score = result["score"]
            class_id = result["class"]
            label = f"Class {class_id} ({score:.2f})"
            draw_obj.rectangle(box, outline="red", width=2)
            draw_obj.text((box[0]+5, box[1]-15), label, fill="red", font=font)

        logger.info("[Inference] Completed drawing boxes on the image.")
        return draw_image