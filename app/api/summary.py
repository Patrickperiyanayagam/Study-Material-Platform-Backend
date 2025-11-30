from fastapi import APIRouter, HTTPException
from app.models.schemas import SummaryRequest, SummaryResponse
from app.services.summary_service import SummaryService
from app.services.service_manager import get_service_manager
from app.services.document_processor import DocumentProcessor

router = APIRouter()

async def get_document_processor():
    return DocumentProcessor()

@router.post("/generate", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    try:
        print("=" * 60)
        print("📄 SUMMARY API CALLED - Generating summary")
        print(f"📊 Parameters: {request.length} length, {request.type} type")
        print(f"🎯 Topics: {request.topics}")
        
        # Get service manager and create summary service
        service_manager = get_service_manager()
        
        # Create summary service with custom config if provided
        if request.model_configuration:
            print("🤖 Using CUSTOM model configuration:")
            print(f"   Provider: {request.model_configuration.provider.value}")
            print(f"   Model: {request.model_configuration.model_name}")
            print(f"   Temperature: {request.model_configuration.temperature}")
            
            model_config = {
                "provider": request.model_configuration.provider.value,
                "model_name": request.model_configuration.model_name,
                "temperature": request.model_configuration.temperature,
                "base_url": request.model_configuration.base_url,
                "max_tokens": request.model_configuration.max_tokens
            }
            summary_service = SummaryService(model_config)
        else:
            print("🤖 Using DEFAULT model configuration")
            summary_service = service_manager.get_summary_service()
        
        print("⚡ Generating summary...")
        
        # Generate summary using summary service
        result = await summary_service.generate_summary(
            length=request.length,
            summary_type=request.type,
            topics=request.topics
        )
        
        print(f"✅ Summary generated successfully")
        print(f"📤 Summary length: {len(result['content'])} characters")
        print(f"📚 Sources used: {result.get('sources_used', 0)}")
        print("=" * 60)
        
        return SummaryResponse(**result)
        
    except Exception as e:
        print(f"❌ Summary generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

@router.get("/topics")
async def get_summary_topics():
    try:
        document_processor = DocumentProcessor()
        topics = await document_processor.get_all_sources()
        return {"topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get topics: {str(e)}")

@router.get("/status")
async def summary_status():
    return {"status": "Summary service is running"}