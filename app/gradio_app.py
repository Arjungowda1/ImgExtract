import gradio as gr
from inference.load_model import get_predictor
from PIL import Image
import zipfile
import io
import os
import numpy as np
import tempfile
import atexit
import time
import threading

predictor = get_predictor()

temp_files = []
temp_files_lock = threading.Lock()

def cleanup_temp_files():
    """Clean up all temporary files created by this app"""
    with temp_files_lock:
        files_to_remove = temp_files.copy()
        temp_files.clear()
    
    for temp_file in files_to_remove:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                print(f"Cleaned up temp file: {temp_file}")
        except Exception as e:
            print(f"Failed to delete temp file {temp_file}: {e}")

def periodic_cleanup():
    """Periodically clean up old temporary files"""
    while True:
        time.sleep(300)  # Clean up every 5 minutes
        cleanup_temp_files()

atexit.register(cleanup_temp_files)

cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

def extract_and_zip(image: Image.Image):
    image_np = np.array(image)
    results = predictor(image_np) 
    
    pil_images = []
    if 'instances' in results:
        instances = results['instances']
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            region = image.crop((x1, y1, x2, y2))
            pil_images.append(region)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    temp_file_path = temp_file.name
    temp_file.close()
    
    with temp_files_lock:
        temp_files.append(temp_file_path)
    
    with zipfile.ZipFile(temp_file_path, "w") as zf:
        for i, img in enumerate(pil_images):
            img_io = io.BytesIO()
            img.save(img_io, format="PNG")
            zf.writestr(f"image_{i}.png", img_io.getvalue())
    
    return temp_file_path

demo = gr.Interface(
    fn=extract_and_zip,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.File(label="Download ZIP of Extracted Images"),
    title="Image Extractor",
    description="Upload a image and download all detected image regions as a zip file.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()