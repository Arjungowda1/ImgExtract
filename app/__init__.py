from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import main
from contextlib import asynccontextmanager
from inference.load_model import get_predictor
import time
import utility.utils as utils

logger = utils.get_logger(__name__)

# Load the predictor at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Startup] Initializing model predictor...")
    start_time = time.perf_counter()
    get_predictor()
    end_time = time.perf_counter()
    logger.info(f"[Startup] Model predictor initialized in {end_time - start_time:.2f} seconds.")
    yield


# Initialize FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="Image Detection API",
    description="""
    A powerful API for detecting and extracting images from documents, particularly optimized for patent PDFs.
    
    ## Features
    * Image detection in PDFs and images
    * detectron2-based model trained on patent documents.
    
    ## Usage
    upload images or PDFs to the respective endpoints to receive detected images.
    """,
    version="1.0.0",
    docs_url="/docs",  
    redoc_url="/redoc",  
    openapi_url="/openapi.json", 
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}  
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# adding a base route and including the main router
app.include_router(main.app, prefix="/inference", tags=["Image Detection API"])