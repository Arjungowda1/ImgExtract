from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import utility.utils as utils
from inference import inference
from pdf2image import convert_from_bytes

app = APIRouter()

logger = utils.get_logger(__name__)

@app.get("/health", include_in_schema=False)
async def health_check():
    """
    Health check endpoint to verify if the service is running.
    ** Internal Use Only **
    """
    return JSONResponse(content={"status": "ok"}, status_code=200)

@app.post("/image")
async def inference_image(
    images: list[UploadFile] = File(...),
    mode: str = Query("bbox", enum = ["bbox", "draw", "extract"]),
):
    """
    Performs inference on a image using a model trained using detectron2.

    ## Description:
    This accepts an image as input and performs 3 kinds of operations based on chosen `mode` :
        - `"bounding_box"` : returns the bounding box of detected image along with its confidence
        - `"draw"` : draws identified images as boxes in the input image along with its confidence
        - `"extract"` : creates a zip file and extracts all identified images; each image will have its score in the file name

    ## Parameters:
        - `images` (UploadFile): List of images to be processed.
        - `mode` string: operation to perform

    ## Returns:
        - `JSONResponse` : if mode is `bounding_box`
        - `StreamingResponse` : otherwise
    
    ## Raises: 
        - HTTPException : For any errors while processing
    
    ## Example:
    ```
        curl --location 'http://localhost:8000/inference/image?mode=draw' \
        --form 'images=@"/path/to/file/input.png"'
    ```
    """

    try: 
        if mode not in ["bbox", "draw", "extract"]:
            raise HTTPException(status_code=400, detail="Invalid mode specified. Choose from 'bbox', 'draw', or 'extract'.")
        logger.info(f"[Inference] Received {len(images)} images for processing in mode '{mode}'")
        if mode == "bbox":
            results = []
            for image_file in images:
                image = Image.open(image_file.file).convert("RGB")
                logger.info(f"[Inference] Processing image: {image_file.filename}")
                result = inference.inference_image(image, draw=False)
                results.append({"filename": image_file.filename, "results": result if result else []})
            return JSONResponse(content=results, status_code=200)
        elif mode == "draw":
            images_with_boxes = []
            for image_file in images:
                image = Image.open(image_file.file).convert("RGB")
                logger.info(f"[Inference] Processing image: {image_file.filename}")
                result_image = inference.inference_image(image, draw=True)
                images_with_boxes.append((image_file.filename, result_image if isinstance(result_image, Image.Image) else image))
            return StreamingResponse(
                utils.create_zip(images_with_boxes),
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=images_with_boxes.zip"}
            )
        elif mode == "extract":
            images = []
            for image_file in images:
                image = Image.open(image_file.file).convert("RGB")
                logger.info(f"[Inference] Processing image: {image_file.filename}")
                bbox = inference.inference_image(image, draw=False)
                if bbox is None or not bbox:
                    logger.warning(f"[Inference] No drawings found in image: {image_file.filename}")
                    continue
                extracted_images = utils.get_images(image, bbox)
                images.extend([(f"{image_file.filename}_extracted_{i}.png", img) for i, img in enumerate(extracted_images)])
            return StreamingResponse(
                utils.create_zip(images),
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=extracted_images.zip"}
            )
    except Exception as e:
        logger.error(f"[Inference] Error during inference: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/pdf")
async def inference_pdf(
    pdf: UploadFile = File(...),
    mode: str = Query("bbox", enum=["bbox", "draw", "extract"]),
):
    """
    Performs inference on a PDF file using a model trained using detectron2.

    ## Description:
    This accepts a PDF file as input and performs 3 kinds of operations based on chosen `mode` :
        - `"bounding_box"` : returns the bounding box of detected images along with their confidence
        - `"draw"` : draws identified images as boxes in the input PDF along with their confidence
        - `"extract"` : creates a zip file and extracts all identified images; each image will have its score in the file name

    ## Parameters:
        - `pdf` (UploadFile): PDF file to be processed.
        - `mode` string: operation to perform

    ## Returns:
        - `JSONResponse` : if mode is `bounding_box`
        - `StreamingResponse` : otherwise

    ## Example:
    ```
        curl --location 'http://localhost:8000/inference/pdf?mode=draw' \
        --form 'pdf=@"/path/to/file/input.pdf"'
    ```
    """
    try:
        if mode not in ["bbox", "draw", "extract"]:
            raise HTTPException(status_code=400, detail="Invalid mode specified. Choose from 'bbox', 'draw', or 'extract'.")
        
        logger.info(f"[Inference] Received PDF file '{pdf.filename}' for processing in mode '{mode}'")
        pdf_bytes = await pdf.read()
        images = convert_from_bytes(pdf_bytes, dpi=300, fmt="png")
        logger.info(f"[Inference] Converted PDF to {len(images)} images for processing")

        if mode == "bbox":
            results = []
            for i, image in enumerate(images):
                logger.info(f"[Inference] Processing page {i+1} of PDF")
                result = inference.inference_image(image, draw=False)
                results.append({"page": i+1, "results": result if result else []})
            return JSONResponse(content=results, status_code=200)
        elif mode == "draw":
            images_with_boxes = []
            for i, image in enumerate(images):
                logger.info(f"[Inference] Processing page {i+1} of PDF")
                result_image = inference.inference_image(image, draw=True)
                images_with_boxes.append((f"page_{i+1}.png", result_image if isinstance(result_image, Image.Image) else image))
            return StreamingResponse(
                utils.create_zip(images_with_boxes),
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=pdf_with_boxes.zip"}
            )
        elif mode == "extract":
            extracted_images = []
            for i, image in enumerate(images):
                logger.info(f"[Inference] Processing page {i+1} of PDF")
                bbox = inference.inference_image(image, draw=False)
                if bbox is None or not bbox:
                    logger.warning(f"[Inference] No drawings found in page {i+1}")
                    continue
                page_images = utils.get_images(image, bbox)
                extracted_images.extend([(f"page_{i+1}_extracted_{j}.png", img) for j, img in enumerate(page_images)])
            return StreamingResponse(
                utils.create_zip(extracted_images),
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=extracted_images.zip"}
            )
    
    except Exception as e:
        logger.error(f"[Inference] Error during PDF inference: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")