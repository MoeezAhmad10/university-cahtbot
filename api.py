from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import uvicorn
import logging
import time
from collections import defaultdict
import uuid
import asyncio

# Import your chatbot class from chatbot.py
from chatbot import RiphahChatbot

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with enhanced metadata
app = FastAPI(
    title="Riphah University Chatbot API",
    description="""AI-powered chatbot for Riphah International University information.
    
    Features:
    - Intelligent question answering with RAG
    - Multi-stage response generation
    - Conversation context tracking
    - Source citation
    - Session management
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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

# Session management - stores conversation history per session
sessions: Dict[str, Dict[str, Any]] = {}
SESSION_TIMEOUT = timedelta(hours=2)  # Sessions expire after 2 hours

# Rate limiting - simple in-memory rate limiting
rate_limit_data: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 10  # requests per window
RATE_LIMIT_WINDOW = 60  # seconds

# API metrics
metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "average_response_time": 0.0,
    "total_sessions": 0
}

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="User's question about Riphah University"
    )
    session_id: Optional[str] = Field(
        None, 
        description="Session ID for conversation context (auto-generated if not provided)"
    )
    include_sources: bool = Field(
        default=False, 
        description="Include source URLs in response"
    )
    
    @validator('question')
    def validate_question(cls, v):
        # Strip whitespace and check for actual content
        v = v.strip()
        if not v:
            raise ValueError('Question cannot be empty or only whitespace')
        # Check for potentially malicious input
        suspicious_patterns = ['<script>', 'javascript:', 'onclick=']
        if any(pattern in v.lower() for pattern in suspicious_patterns):
            raise ValueError('Invalid input detected')
        return v

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    session_id: str
    timestamp: str
    response_time: float
    conversation_length: int
    success: bool = True

class HealthResponse(BaseModel):
    status: str
    chatbot_initialized: bool
    active_sessions: int
    total_requests: int
    uptime: str
    timestamp: str

class RefreshResponse(BaseModel):
    message: str
    success: bool
    refresh_id: str
    timestamp: str

class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    message_count: int
    last_activity: str

class ClearSessionResponse(BaseModel):
    message: str
    session_id: str
    cleared_messages: int
    success: bool

class MetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    active_sessions: int
    total_sessions: int
    success_rate: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str
    success: bool = False


# Store startup time for uptime calculation
startup_time = None

# Middleware for request logging and metrics
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and track metrics"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Update metrics
        metrics["total_requests"] += 1
        if response.status_code < 400:
            metrics["successful_requests"] += 1
        else:
            metrics["failed_requests"] += 1
        
        # Update average response time
        total = metrics["total_requests"]
        metrics["average_response_time"] = (
            (metrics["average_response_time"] * (total - 1) + process_time) / total
        )
        
        # Add response time header
        response.headers["X-Process-Time"] = str(process_time)
        
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        metrics["failed_requests"] += 1
        raise

# Startup event to initialize chatbot
@app.on_event("startup")
async def startup_event():
    """Initialize chatbot when API starts"""
    global chatbot, startup_time
    startup_time = datetime.now()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in .env file!")
        return
    
    logger.info("Initializing Riphah chatbot...")
    try:
        chatbot = RiphahChatbot(api_key)
        
        # Initialize with cache if available
        success = chatbot.initialize(use_cache=True)
        
        if success:
            logger.info("✓ Chatbot initialized successfully")
        else:
            logger.warning("✗ Chatbot initialization failed")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")
    # Clear all sessions
    sessions.clear()
    logger.info("Cleanup completed")


# Helper functions
def check_rate_limit(client_id: str) -> bool:
    """Check if client has exceeded rate limit"""
    now = time.time()
    # Clean old entries
    rate_limit_data[client_id] = [
        req_time for req_time in rate_limit_data[client_id]
        if now - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check limit
    if len(rate_limit_data[client_id]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Add current request
    rate_limit_data[client_id].append(now)
    return True

def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create new one"""
    # Clean expired sessions
    now = datetime.now()
    expired_sessions = [
        sid for sid, data in sessions.items()
        if now - data["last_activity"] > SESSION_TIMEOUT
    ]
    for sid in expired_sessions:
        del sessions[sid]
        logger.info(f"Session {sid} expired and removed")
    
    # Create new session if needed
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "created_at": now,
            "last_activity": now,
            "messages": [],
            "conversation_history": []
        }
        metrics["total_sessions"] += 1
        logger.info(f"Created new session: {session_id}")
    else:
        sessions[session_id]["last_activity"] = now
    
    return session_id

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Riphah University Chatbot API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)",
            "session": "/session/{session_id} (GET)",
            "clear_session": "/session/{session_id}/clear (DELETE)",
            "refresh": "/refresh (POST)",
            "metrics": "/metrics (GET)",
            "campuses": "/campuses (GET)",
            "quick_links": "/quick-links (GET)",
            "docs": "/docs"
        },
        "features": [
            "Intelligent question answering",
            "Conversation context tracking",
            "Session management",
            "Source citation",
            "Rate limiting"
        ]
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and chatbot health status with detailed metrics"""
    uptime = str(datetime.now() - startup_time) if startup_time else "unknown"
    
    return HealthResponse(
        status="healthy" if chatbot and chatbot.retriever else "unhealthy",
        chatbot_initialized=chatbot is not None and chatbot.retriever is not None,
        active_sessions=len(sessions),
        total_requests=metrics["total_requests"],
        uptime=uptime,
        timestamp=datetime.now().isoformat()
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest, http_request: Request):
    """
    Main chat endpoint to ask questions about Riphah University
    
    Features:
    - Intelligent multi-stage response generation
    - Conversation context tracking via sessions
    - Optional source citation
    - Rate limiting protection
    
    Parameters:
    - **question**: Your question about Riphah (required, 1-1000 chars)
    - **session_id**: Session ID for context (optional, auto-generated if not provided)
    - **include_sources**: Include source URLs in response (optional, default: false)
    """
    start_time = time.time()
    global chatbot
    
    # Rate limiting check
    client_id = http_request.client.host
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds."
        )
    
    # Check if chatbot is initialized
    if not chatbot or not chatbot.retriever:
        logger.error("Chat request received but chatbot not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized. Please try again later."
        )
    
    # Get or create session
    session_id = get_or_create_session(request.session_id)
    session_data = sessions[session_id]
    
    try:
        # Store question in session
        session_data["messages"].append({
            "role": "user",
            "content": request.question,
            "timestamp": datetime.now().isoformat()
        })
        
        # Sync chatbot conversation history with session
        if hasattr(chatbot, 'conversation_history'):
            chatbot.conversation_history = session_data.get("conversation_history", [])
        
        # Get response with or without sources
        if request.include_sources:
            answer, sources = chatbot.chat_with_sources(request.question)
        else:
            answer = chatbot.chat(request.question)
            sources = None
        
        # Update session with response
        session_data["messages"].append({
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().isoformat(),
            "sources": sources
        })
        
        # Save updated conversation history
        if hasattr(chatbot, 'conversation_history'):
            session_data["conversation_history"] = chatbot.conversation_history
        
        response_time = time.time() - start_time
        
        logger.info(f"Successfully answered question in session {session_id[:8]}... ({response_time:.2f}s)")
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            response_time=round(response_time, 3),
            conversation_length=len(session_data["messages"]) // 2
        )
            
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )


@app.get("/session/{session_id}", response_model=SessionResponse, tags=["Session"])
async def get_session_info(session_id: str):
    """Get information about a specific session"""
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired"
        )
    
    session_data = sessions[session_id]
    return SessionResponse(
        session_id=session_id,
        created_at=session_data["created_at"].isoformat(),
        message_count=len(session_data["messages"]),
        last_activity=session_data["last_activity"].isoformat()
    )

@app.delete("/session/{session_id}/clear", response_model=ClearSessionResponse, tags=["Session"])
async def clear_session(session_id: str):
    """Clear conversation history for a specific session"""
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    message_count = len(sessions[session_id]["messages"])
    sessions[session_id]["messages"] = []
    sessions[session_id]["conversation_history"] = []
    
    logger.info(f"Cleared session {session_id[:8]}... ({message_count} messages)")
    
    return ClearSessionResponse(
        message="Session cleared successfully",
        session_id=session_id,
        cleared_messages=message_count,
        success=True
    )

@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Get API metrics and statistics"""
    total = metrics["total_requests"]
    success_rate = (
        (metrics["successful_requests"] / total * 100) if total > 0 else 0.0
    )
    
    return MetricsResponse(
        total_requests=metrics["total_requests"],
        successful_requests=metrics["successful_requests"],
        failed_requests=metrics["failed_requests"],
        average_response_time=round(metrics["average_response_time"], 3),
        active_sessions=len(sessions),
        total_sessions=metrics["total_sessions"],
        success_rate=round(success_rate, 2)
    )

@app.post("/refresh", response_model=RefreshResponse, tags=["Admin"])
async def refresh_data(background_tasks: BackgroundTasks):
    """
    Refresh chatbot data by re-scraping Riphah website
    (Admin endpoint - should be protected with authentication in production)
    
    This operation runs in the background to avoid timeout.
    """
    global chatbot
    
    if not chatbot:
        logger.error("Refresh requested but chatbot not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized"
        )
    
    refresh_id = str(uuid.uuid4())
    
    # Run refresh in background to avoid timeout
    def refresh_task():
        logger.info(f"Starting data refresh (ID: {refresh_id[:8]}...)")
        try:
            chatbot.refresh_data()
            logger.info(f"Data refresh completed (ID: {refresh_id[:8]}...)")
        except Exception as e:
            logger.error(f"Data refresh failed (ID: {refresh_id[:8]}...): {str(e)}")
    
    background_tasks.add_task(refresh_task)
    
    return RefreshResponse(
        message="Data refresh started in background. Check logs for progress.",
        success=True,
        refresh_id=refresh_id,
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
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured error response"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "detail": f"Status code: {exc.status_code}",
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "success": False
        }
    )


# Run the API
if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting Riphah Chatbot API on {host}:{port}")
    logger.info(f"Documentation available at http://{host}:{port}/docs")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
 