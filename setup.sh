# setup.sh
#!/bin/bash

set -e -x

python -m pip install --upgrade pip
python -m pip install poetry

# Install project dependencies
echo "Installing project dependencies..."
poetry install
echo "Base libraries installed sucessfully!"

# Install dependencies needed for detectron2
echo "Installing detectron2 dependencies..."
poetry run pip install setuptools
poetry run pip install pycocotools cython numpy
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo "detectron2 dependency installed sucessfully!"

# Install detectron2
echo "Installing detectron2..."
poetry run pip install --no-build-isolation "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
poetry run pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# Install Poppler - a dependency for pdf2image
echo "Installing Poppler..."
brew install poppler
echo "Poppler installed successfully!"

echo "Setup complete!"


# if detectron2 is not installed 
# clone it from github directly
# https://github.com/facebookresearch/detectron2

# THIS SCRIPT INSTALLS TORCH with CPU support only