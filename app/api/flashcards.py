from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import FlashCardRequest, FlashCardResponse
from app.services.flashcard_service import FlashCardService
from app.services.service_manager import service_manager

router = APIRouter()

async def get_flashcard_service():
    return await service_manager.get_flashcard_service()

@router.post("/generate", response_model=FlashCardResponse)
async def generate_flashcards(request: FlashCardRequest):
    try:
        print("=" * 60)
        print("🟠 FLASHCARD API CALLED")
        print(f"🃏 Cards: {request.num_cards}")
        print(f"🏷️ Topics: {request.topics}")
        
        # Create temporary service instance with specific model config if provided
        if request.model_configuration:
            print("🤖 Using CUSTOM model configuration:")
            print(f"   Provider: {request.model_configuration.provider.value}")
            print(f"   Model: {request.model_configuration.model_name}")
            print(f"   Temperature: {request.model_configuration.temperature}")
            print(f"   Base URL: {request.model_configuration.base_url}")
            print(f"   Max Tokens: {request.model_configuration.max_tokens}")
            
            from app.services.flashcard_service import FlashCardService
            config = {
                "provider": request.model_configuration.provider.value,
                "model_name": request.model_configuration.model_name,
                "temperature": request.model_configuration.temperature,
                "base_url": request.model_configuration.base_url,
                "max_tokens": request.model_configuration.max_tokens
            }
            flashcard_service = FlashCardService(initial_config=config)
        else:
            print("🤖 Using DEFAULT model configuration (ollama with llama3.1:8b)")
            # Use default config (ollama with llama3.1:8b)
            from app.utils.defaults import get_default_model_config
            from app.services.flashcard_service import FlashCardService
            flashcard_service = FlashCardService(initial_config=get_default_model_config())
        
        print("⚡ Generating flashcards...")
        cards = await flashcard_service.generate_flashcards(
            num_cards=request.num_cards,
            topics=request.topics
        )
        
        print(f"✅ Flashcards generated successfully")
        print(f"📤 Generated {len(cards)} flashcards")
        print("=" * 60)
        
        return FlashCardResponse(
            cards=cards,
            total_cards=len(cards)
        )
    except Exception as e:
        print(f"❌ Flashcard API Error: {str(e)}")
        print("=" * 60)
        raise HTTPException(status_code=500, detail=f"Flashcard generation failed: {str(e)}")

@router.get("/topics")
async def get_available_topics(flashcard_service: FlashCardService = Depends(get_flashcard_service)):
    try:
        topics = await flashcard_service.get_available_topics()
        return {"topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve topics: {str(e)}")

@router.get("/status")
async def flashcard_status():
    return {"status": "Flashcard service is running"}