from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import ChatRequest, ChatResponse
from app.services.persistent_langgraph_chat_service import get_chat_service, PersistentLangGraphChatService
import uuid

router = APIRouter()

async def get_persistent_chat_service() -> PersistentLangGraphChatService:
    """Get the singleton persistent chat service."""
    return get_chat_service()

@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    try:
        print("=" * 60)
        print("🔵 CHAT API CALLED - Using Persistent LangGraph Service")
        print(f"📧 Message: {request.message[:50]}..." if len(request.message) > 50 else f"📧 Message: {request.message}")
        
        session_id = request.session_id or str(uuid.uuid4())
        print(f"🆔 Session ID: {session_id}")
        
        # Get the singleton persistent chat service
        chat_service = get_chat_service()
        
        # Prepare model configuration if provided
        model_config = None
        if request.model_configuration:
            print("🤖 Using CUSTOM model configuration:")
            print(f"   Provider: {request.model_configuration.provider.value}")
            print(f"   Model: {request.model_configuration.model_name}")
            print(f"   Temperature: {request.model_configuration.temperature}")
            print(f"   Base URL: {request.model_configuration.base_url}")
            print(f"   Max Tokens: {request.model_configuration.max_tokens}")
            
            model_config = {
                "provider": request.model_configuration.provider.value,
                "model_name": request.model_configuration.model_name,
                "temperature": request.model_configuration.temperature,
                "base_url": request.model_configuration.base_url,
                "max_tokens": request.model_configuration.max_tokens
            }
        else:
            print("🤖 Using EXISTING persistent service configuration")
            current_config = chat_service.get_current_config()
            print(f"   Current: {current_config}")
        
        print("⚡ Processing message with persistent LangGraph service...")
        
        # Process message using persistent service
        response = await chat_service.process_message(
            message=request.message,
            session_id=session_id,
            model_config=model_config
        )
        
        print(f"✅ Chat response generated successfully")
        print(f"📤 Response length: {len(response['response'])} characters")
        print(f"📚 Sources: {response.get('sources', [])}")
        print("=" * 60)
        
        return ChatResponse(
            response=response["response"],
            sources=response.get("sources", []),
            session_id=session_id
        )
        
    except Exception as e:
        print(f"❌ Chat API Error: {str(e)}")
        print("=" * 60)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@router.get("/sessions/{session_id}/history")
async def get_chat_history(
    session_id: str,
    chat_service: PersistentLangGraphChatService = Depends(get_persistent_chat_service)
):
    try:
        print(f"📜 Getting chat history for session: {session_id}")
        history = await chat_service.get_chat_history(session_id)
        print(f"📋 Retrieved {len(history)} messages from persistent memory")
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        print(f"❌ Failed to retrieve chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")

@router.delete("/sessions/{session_id}")
async def clear_chat_session(
    session_id: str,
    chat_service: PersistentLangGraphChatService = Depends(get_persistent_chat_service)
):
    try:
        print(f"🗑️ Clearing chat session: {session_id}")
        await chat_service.clear_session(session_id)
        print(f"✅ Session cleared from persistent memory")
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        print(f"❌ Failed to clear session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")

@router.get("/status")
async def chat_status():
    try:
        chat_service = get_chat_service()
        current_config = chat_service.get_current_config()
        return {
            "status": "Persistent LangGraph Chat service is running",
            "service_type": "PersistentLangGraphChatService",
            "current_model": f"{current_config.get('provider', 'unknown')}/{current_config.get('model_name', 'unknown')}",
            "memory_type": "InMemorySaver with persistent graph",
            "graph_persistence": "Singleton - created once, reused for all conversations"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@router.post("/sessions/{session_id}/test-memory")
async def test_memory(
    session_id: str,
    chat_service: PersistentLangGraphChatService = Depends(get_persistent_chat_service)
):
    """Test endpoint to verify memory persistence."""
    try:
        # Send a test message to establish conversation
        response1 = await chat_service.process_message(
            "My name is Test User and my favorite color is blue.",
            session_id
        )
        
        # Send follow-up to test memory
        response2 = await chat_service.process_message(
            "What is my name and favorite color?",
            session_id
        )
        
        # Get full history
        history = await chat_service.get_chat_history(session_id)
        
        return {
            "test_result": "success",
            "conversation_established": response1["response"],
            "memory_test_response": response2["response"],
            "total_messages_in_history": len(history),
            "full_history": history
        }
        
    except Exception as e:
        return {
            "test_result": "failed",
            "error": str(e)
        }