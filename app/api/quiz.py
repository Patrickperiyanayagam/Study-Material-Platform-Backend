from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import QuizRequest, QuizResponse
from app.services.quiz_service import QuizService
from app.services.service_manager import service_manager

router = APIRouter()

async def get_quiz_service():
    return await service_manager.get_quiz_service()

@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    try:
        print("=" * 60)
        print("🟡 QUIZ API CALLED")
        print(f"📝 Questions: {request.num_questions}")
        print(f"📊 Difficulty: {request.difficulty}")
        print(f"🏷️ Topics: {request.topics}")
        
        # Create temporary service instance with specific model config if provided
        if request.model_configuration:
            print("🤖 Using CUSTOM model configuration:")
            print(f"   Provider: {request.model_configuration.provider.value}")
            print(f"   Model: {request.model_configuration.model_name}")
            print(f"   Temperature: {request.model_configuration.temperature}")
            print(f"   Base URL: {request.model_configuration.base_url}")
            print(f"   Max Tokens: {request.model_configuration.max_tokens}")
            
            from app.services.quiz_service import QuizService
            config = {
                "provider": request.model_configuration.provider.value,
                "model_name": request.model_configuration.model_name,
                "temperature": request.model_configuration.temperature,
                "base_url": request.model_configuration.base_url,
                "max_tokens": request.model_configuration.max_tokens
            }
            quiz_service = QuizService(initial_config=config)
        else:
            print("🤖 Using DEFAULT model configuration (ollama with llama3.1:8b)")
            # Use default config (ollama with llama3.1:8b)
            from app.utils.defaults import get_default_model_config
            from app.services.quiz_service import QuizService
            quiz_service = QuizService(initial_config=get_default_model_config())
        
        print("⚡ Generating quiz questions...")
        questions = await quiz_service.generate_quiz(
            num_questions=request.num_questions,
            difficulty=request.difficulty,
            topics=request.topics
        )
        
        print(f"✅ Quiz generated successfully")
        print(f"📤 Generated {len(questions)} questions")
        print("=" * 60)
        
        return QuizResponse(
            questions=questions,
            total_questions=len(questions)
        )
    except Exception as e:
        print(f"❌ Quiz API Error: {str(e)}")
        print("=" * 60)
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")

@router.get("/topics")
async def get_available_topics(quiz_service: QuizService = Depends(get_quiz_service)):
    try:
        topics = await quiz_service.get_available_topics()
        return {"topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve topics: {str(e)}")

@router.get("/status")
async def quiz_status():
    return {"status": "Quiz service is running"}