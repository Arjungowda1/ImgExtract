# setup.ps1

function Install-Poppler {
    Write-Host "Installing Poppler..."
    
    # Create temp directory for download
    $tempDir = Join-Path $env:TEMP "poppler_install"
    New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
    
    # Download latest Poppler release
    $popplerUrl = "https://github.com/oschwartz10612/poppler-windows/releases/download/v24.02.0-0/Release-24.02.0-0.zip"
    $zipPath = Join-Path $tempDir "poppler.zip"
    Write-Host "Downloading Poppler..."
    Invoke-WebRequest -Uri $popplerUrl -OutFile $zipPath
    
    # Extract to Local AppData
    $installPath = Join-Path $env:LOCALAPPDATA "poppler"
    Write-Host "Extracting Poppler to $installPath..."
    Expand-Archive -Path $zipPath -DestinationPath $installPath -Force
    
    # Add to PATH
    $binPath = Join-Path $installPath "poppler-24.02.0\Library\bin"
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $currentPath.Contains($binPath)) {
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$binPath", "User")
        $env:Path = "$env:Path;$binPath"
    }
    
    # Cleanup
    Remove-Item -Path $tempDir -Recurse -Force
    Write-Host "Poppler installed successfully!"
}

# Install Poppler
Install-Poppler

Write-Host "Installing poetry.."
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

$env:Path = "$env:APPDATA\Python\Scripts;$env:Path"

# Install project dependencies
Write-Host "Installing project dependencies..."
poetry install
Write-Host "Base libraries installed successfully!"

# Install dependencies needed for detectron2
Write-Host "Installing detectron2 dependencies..."
poetry run pip install setuptools
poetry run pip install pycocotools cython numpy
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
Write-Host "detectron2 dependency installed successfully!"

# Install detectron2
Write-Host "Installing detectron2..."
poetry run pip install --no-build-isolation "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
poetry run pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
Write-Host "Setup complete!" 


# if detectron2 is not installed 
# clone it from github directly
# https://github.com/facebookresearch/detectron2

# THIS SCRIPT INSTALLS TORCH with CPU support only