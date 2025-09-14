from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api import upload, chat, quiz, flashcards, config
from app.services.persistent_langgraph_chat_service import get_chat_service
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Study Material Platform API",
    description="AI-powered platform for generating study materials from documents",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize persistent services on server startup."""
    print("🚀 SERVER STARTUP - Initializing services...")
    
    # Initialize the persistent chat service (singleton)
    chat_service = get_chat_service()
    print(f"✅ Persistent LangGraph Chat Service initialized")
    print(f"   Current config: {chat_service.get_current_config()}")
    
    print("🎉 All services initialized successfully!")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

upload_dir = os.getenv("UPLOAD_DIR", "./data/documents")
os.makedirs(upload_dir, exist_ok=True)

app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(quiz.router, prefix="/api/quiz", tags=["quiz"])
app.include_router(flashcards.router, prefix="/api/flashcards", tags=["flashcards"])
app.include_router(config.router, prefix="/api/config", tags=["config"])

@app.get("/")
async def root():
    return {"message": "Study Material Platform API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    # Include chat service health in health check
    try:
        chat_service = get_chat_service()
        chat_config = chat_service.get_current_config()
        return {
            "status": "healthy",
            "chat_service": "online",
            "current_model": f"{chat_config.get('provider', 'unknown')}/{chat_config.get('model_name', 'unknown')}"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "chat_service": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEBUG", "true").lower() == "true"
    )