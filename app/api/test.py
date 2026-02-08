from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import TestRequest, TestResponse, GradingRequest, GradingResponse
from app.services.test_service import TestService
from app.services.service_manager import service_manager

router = APIRouter()

async def get_test_service():
    return await service_manager.get_test_service()

@router.post("/generate", response_model=TestResponse)
async def generate_test(request: TestRequest):
    try:
        print("=" * 60)
        print("🟡 TEST API CALLED - GENERATE")
        print(f"📝 Questions: {request.num_questions}")
        print(f"📊 Difficulty: {request.difficulty}")
        print(f"🎯 Mark Distribution: {request.mark_distribution}")
        print(f"🏷️ Topics: {request.topics}")
        
        # Create temporary service instance with specific model config if provided
        if request.model_configuration:
            print("🤖 Using CUSTOM model configuration:")
            print(f"   Provider: {request.model_configuration.provider.value}")
            print(f"   Model: {request.model_configuration.model_name}")
            print(f"   Temperature: {request.model_configuration.temperature}")
            print(f"   Base URL: {request.model_configuration.base_url}")
            print(f"   Max Tokens: {request.model_configuration.max_tokens}")
            
            from app.services.test_service import TestService
            config = {
                "provider": request.model_configuration.provider.value,
                "model_name": request.model_configuration.model_name,
                "temperature": request.model_configuration.temperature,
                "base_url": request.model_configuration.base_url,
                "max_tokens": request.model_configuration.max_tokens
            }
            test_service = TestService(initial_config=config)
        else:
            print("🤖 Using DEFAULT model configuration (ollama with llama3.1:8b)")
            # Use default config (ollama with llama3.1:8b)
            from app.utils.defaults import get_default_model_config
            from app.services.test_service import TestService
            test_service = TestService(initial_config=get_default_model_config())
        
        print("⚡ Generating test questions...")
        questions = await test_service.generate_test(
            num_questions=request.num_questions,
            difficulty=request.difficulty,
            mark_distribution=request.mark_distribution,
            topics=request.topics
        )
        
        total_marks = sum(q.marks for q in questions)
        
        print(f"✅ Test generated successfully")
        print(f"📤 Generated {len(questions)} questions")
        print(f"💯 Total marks: {total_marks}")
        print("=" * 60)
        
        return TestResponse(
            questions=questions,
            total_questions=len(questions),
            total_marks=total_marks
        )
    except Exception as e:
        print(f"❌ Test API Error: {str(e)}")
        print("=" * 60)
        raise HTTPException(status_code=500, detail=f"Test generation failed: {str(e)}")

@router.post("/grade", response_model=GradingResponse)
async def grade_test(request: GradingRequest):
    try:
        print("=" * 60)
        print("🟡 TEST API CALLED - GRADE")
        print(f"📝 Questions to grade: {len(request.questions)}")
        print(f"📝 Answers submitted: {len(request.answers)}")
        
        # Create temporary service instance with specific model config if provided
        if request.model_configuration:
            print("🤖 Using CUSTOM model configuration:")
            print(f"   Provider: {request.model_configuration.provider.value}")
            print(f"   Model: {request.model_configuration.model_name}")
            print(f"   Temperature: {request.model_configuration.temperature}")
            print(f"   Base URL: {request.model_configuration.base_url}")
            print(f"   Max Tokens: {request.model_configuration.max_tokens}")
            
            from app.services.test_service import TestService
            config = {
                "provider": request.model_configuration.provider.value,
                "model_name": request.model_configuration.model_name,
                "temperature": request.model_configuration.temperature,
                "base_url": request.model_configuration.base_url,
                "max_tokens": request.model_configuration.max_tokens
            }
            test_service = TestService(initial_config=config)
        else:
            print("🤖 Using DEFAULT model configuration (ollama with llama3.1:8b)")
            # Use default config (ollama with llama3.1:8b)
            from app.utils.defaults import get_default_model_config
            from app.services.test_service import TestService
            test_service = TestService(initial_config=get_default_model_config())
        
        print("⚡ Grading test answers...")
        grading_result = await test_service.grade_test(
            questions=request.questions,
            answers=request.answers
        )
        
        print(f"✅ Test graded successfully")
        print(f"💯 Total score: {grading_result.overall_percentage:.1f}%")
        print("=" * 60)
        
        return grading_result
    except Exception as e:
        print(f"❌ Test grading API Error: {str(e)}")
        print("=" * 60)
        raise HTTPException(status_code=500, detail=f"Test grading failed: {str(e)}")

@router.get("/topics")
async def get_available_topics(test_service: TestService = Depends(get_test_service)):
    try:
        topics = await test_service.get_available_topics()
        return {"topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve topics: {str(e)}")

@router.get("/status")
async def test_status():
    return {"status": "Test service is running"}