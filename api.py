from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os
from dotenv import load_dotenv
from datetime import datetime
import uvicorn

# Import your chatbot class from chatbot.py
from chatbot import RiphahChatbot

# Load environment variables from .env
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Riphah University Chatbot API",
    description="AI-powered chatbot for Riphah International University information",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot = None

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="User's question")
    include_sources: bool = Field(default=False, description="Include source URLs in response")

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    timestamp: str
    success: bool = True

class HealthResponse(BaseModel):
    status: str
    chatbot_initialized: bool
    timestamp: str

class RefreshResponse(BaseModel):
    message: str
    success: bool
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str
    success: bool = False


# Startup event to initialize chatbot
@app.on_event("startup")
async def startup_event():
    """Initialize chatbot when API starts"""
    global chatbot
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file!")
        return
    
    print("Initializing Riphah chatbot...")
    chatbot = RiphahChatbot(api_key)
    
    # Initialize with cache if available
    success = chatbot.initialize(use_cache=True)
    
    if success:
        print("✓ Chatbot initialized successfully")
    else:
        print("✗ Chatbot initialization failed")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Riphah University Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "refresh": "/refresh",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and chatbot health status"""
    return HealthResponse(
        status="healthy" if chatbot and chatbot.retriever else "unhealthy",
        chatbot_initialized=chatbot is not None and chatbot.retriever is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint to ask questions about Riphah University
    
    - **question**: Your question about Riphah (required)
    - **include_sources**: Whether to include source URLs (optional)
    """
    global chatbot
    
    # Check if chatbot is initialized
    if not chatbot or not chatbot.retriever:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not initialized. Please try again later."
        )
    
    try:
        # Get response with or without sources
        if request.include_sources:
            answer, sources = chatbot.chat_with_sources(request.question)
            return ChatResponse(
                answer=answer,
                sources=sources,
                timestamp=datetime.now().isoformat()
            )
        else:
            answer = chatbot.chat(request.question)
            return ChatResponse(
                answer=answer,
                timestamp=datetime.now().isoformat()
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.post("/refresh", response_model=RefreshResponse, tags=["Admin"])
async def refresh_data(background_tasks: BackgroundTasks):
    """
    Refresh chatbot data by re-scraping Riphah website
    (Admin endpoint - should be protected in production)
    """
    global chatbot
    
    if not chatbot:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not initialized"
        )
    
    # Run refresh in background to avoid timeout
    def refresh_task():
        chatbot.refresh_data()
    
    background_tasks.add_task(refresh_task)
    
    return RefreshResponse(
        message="Data refresh started in background",
        success=True,
        timestamp=datetime.now().isoformat()
    )


@app.get("/campuses", tags=["Information"])
async def get_campuses():
    """Get list of Riphah campuses with links"""
    return {
        "campuses": [
            {
                "name": "Islamabad Campus",
                "url": "https://riphah.edu.pk/islamabad-campus/"
            },
            {
                "name": "Lahore Campus",
                "url": "https://riphah.edu.pk/lahore-campus/"
            },
            {
                "name": "Faisalabad Campus",
                "url": "https://riphahfsd.edu.pk/"
            },
            {
                "name": "Sahiwal Campus",
                "url": "https://riphahsahiwal.edu.pk/"
            }
        ],
        "main_website": "https://riphah.edu.pk/"
    }


@app.get("/quick-links", tags=["Information"])
async def get_quick_links():
    """Get important quick links for Riphah University"""
    return {
        "links": {
            "main_website": "https://riphah.edu.pk/",
            "programs": "https://riphah.edu.pk/programs/",
            "admissions": "https://riphah.edu.pk/admissions/",
            "fee_structure": "https://riphah.edu.pk/fee-structure/",
            "scholarships": "https://riphah.edu.pk/scholarships/",
            "contact": "https://riphah.edu.pk/contact/"
        }
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return ErrorResponse(
        error=exc.detail,
        detail=str(exc.status_code),
        timestamp=datetime.now().isoformat()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return ErrorResponse(
        error="Internal server error",
        detail=str(exc),
        timestamp=datetime.now().isoformat()
    )


# Run the API
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
 