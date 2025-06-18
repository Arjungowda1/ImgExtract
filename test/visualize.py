import os
import sys

# python does not automatically find parent directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from PIL import Image
import matplotlib.pyplot as plt

from inference.load_model import get_predictor
from inference.inference import inference_image

def visualize_inference(image_dir: str = "test/samples"):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png'))]
    get_predictor()
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        
        result_image = inference_image(image, draw=True)
        
        if result_image is None:
            print(f"No objects detected in {image_file}")
            continue
        
        plt.figure(figsize=(12, 8))
        plt.imshow(result_image)
        plt.title(f"Detection Results - {image_file}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    visualize_inference()
