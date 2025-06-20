from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch
import time
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class CorrectionRequest(BaseModel):
    text: str
    max_length: Optional[int] = 64

class CorrectionResponse(BaseModel):
    original_text: str
    corrected_text: str
    processing_time: float
    confidence: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

class ModelInfo(BaseModel):
    model_type: str
    model_size: str
    max_length: int
    supported_language: str

class GECServer:
    def __init__(self, model_path="./models/trained_models/checkpoint-3500/"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        
    def load_model(self):
        """Load PyTorch model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            logger.info(f"Using device: {self.device}")
            
            self.model = MT5ForConditionalGeneration.from_pretrained(self.model_path)
            self.tokenizer = MT5Tokenizer.from_pretrained(self.model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise
    
    def correct_text(self, text: str, max_length: int = 64) -> tuple:
        """Correct German text using PyTorch model"""
        start_time = time.time()
        
        try:
            # Prepare input
            input_text = f"Korrigiere: {text}"
            inputs = self.tokenizer(
                input_text,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate correction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=2,
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode output
            corrected_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            ).strip()
            
            processing_time = time.time() - start_time
            
            # Clean up output (remove any residual "Korrigiere:" prefix)
            if corrected_text.startswith("Korrigiere:"):
                corrected_text = corrected_text[11:].strip()
            
            if not corrected_text:
                corrected_text = text
                
            return corrected_text, processing_time
            
        except Exception as e:
            logger.error(f"Correction failed: {str(e)}")
            return text, time.time() - start_time

# Initialize server
app = FastAPI(
    title="German Grammar Correction API (PyTorch)",
    description="API for German text correction using mT5 PyTorch model",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize GEC model
try:
    gec_model = GECServer()
    model_loaded = True
    logger.info("🚀 Server initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    gec_model = None
    model_loaded = False

@app.get("/", response_model=dict)
async def root():
    return {"message": "German Grammar Correction API (PyTorch)", "status": "running"}

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        version="1.0.0"
    )

@app.get("/api/v1/models/info", response_model=ModelInfo)
async def model_info():
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    device_info = str(gec_model.device)
    return ModelInfo(
        model_type=f"mT5-small (PyTorch on {device_info})",
        model_size="~300MB",
        max_length=64,
        supported_language="German"
    )

@app.post("/api/v1/correct", response_model=CorrectionResponse)
async def correct_text(request: CorrectionRequest):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(request.text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long (max 1000 chars)")
    
    try:
        corrected_text, processing_time = gec_model.correct_text(
            request.text, 
            request.max_length
        )
        
        return CorrectionResponse(
            original_text=request.text,
            corrected_text=corrected_text,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Correction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "gec_server_pytorch:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )