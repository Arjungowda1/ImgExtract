# Image Extract

CHECK OUT LIVE DEMO: [here](https://huggingface.co/spaces/aaarjunnnnnnn/ImgExtract) 

A FastAPI service that detects and extracts drawings, math equations, and tables from images and PDFs using Detectron2. 
- Trained on patent documents.
- Extracted images are all classified as drawing 

## Table of Contents
- [Model Details](#model-details)
- [Performance Metrics](#performance-metrics)
- [Dataset Information](#dataset-information)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [API Documentation](#api-documentation)
- [Known Issues](#known-issues)
- [License](#license)
- [Author](#author)

## Model Details

- **Architecture**: Detectron2 with Mask R-CNN
- **Input**: Images
- **Output**: Extracted drawings, math equations, and tables (classified as drawing)
- **Inference Time**: 
  - CPU: ~1s per image (tested on MacOS Sequoia, input size 400x600)
  - GPU: ~200ms per image(tested on Nvidia RTX A500, input size 400x600)
- **Memory Usage**: upto 2GB RAM during inference

## Performance Metrics
 This model was evaluated using COCO metrics standard. 

| Metric                              | Value  |
|-------------------------------------|--------|
| mAP@[0.50:0.95]                     | 57.3%  |
| mAP@0.50                            | 88.5%  |
| mAP@0.75                            | 66.7%  |
| AP (Large objects only)             | 57.3%  |
| AR@[0.50:0.95] (max 100 detections) | 70.3%  |

 *This model performs well for larger bounding boxes.*

## Dataset Information

- **Training Data**: Patent documents
- **Dataset Size**: 4000 (3500 + 500 hard negative samples)

## Requirements

- Python 3.10+
- Poetry (package manager)

## Quick Start

1. Clone the repository
2. For Windows:

   ```ps1
   #if you encounter permission issue:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

   #then run:
   .\setup.ps1 
   # if detectron2 installation fails, clone from git manually

   # To start the service
   poetry run service
   ```
3. For Linux/Mac:
   ```bash
   #create virtual env
   python -m venv venv
   source venv/bin/activate

   # install poetry and run
   source $(poetry env info --path)/bin/activate

   # Make the script executable
   chmod +x setup.sh
   
   # Run the setup script
   ./setup.sh

   # To start the service
   poetry run service
   ```

4. After setup, run the service:
   ```bash
   uvicorn app:app
   ```

## Usage Examples

### Example API Request (using curl)
```bash
curl --location 'http://127.0.0.1:8000/inference/image?mode=bbox' \
--form 'images=@"/C:/Projects/detectron/ImgExtract/test/samples/0.png"'
```

### Example API Request (using Python requests)
```python
import requests

url = "http://127.0.0.1:8000/inference/image?mode=bbox"
files = {"images": open("/C:/Projects/detectron/ImgExtract/test/samples/0.png", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## API Documentation

API documentation is available at:
1. Swagger : localhost:port/docs
2. ReDoc : localhost:port/redoc

## Known Issues

- If Detectron2 installation fails, please clone it manually from the official repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Arjun C E <arjunce15@gmail.com>
