from fastapi import APIRouter, HTTPException, Depends
import requests
from app.models.schemas import ConfigRequest, ConfigResponse, ProvidersResponse, ProviderInfo, ProviderModel
from app.services.config_service import ConfigService
from app.services.service_manager import service_manager
import os

router = APIRouter()

def get_config_service():
    return service_manager.get_config_service()

@router.post("/models", response_model=ConfigResponse)
async def update_model_config(
    request: ConfigRequest,
    config_service: ConfigService = Depends(get_config_service)
):
    try:
        print("=" * 60)
        print("⚙️ CONFIG UPDATE API CALLED")
        print("🤖 New Model Configuration:")
        print(f"   Chat Model: {request.chat_model.provider.value} - {request.chat_model.model_name}")
        print(f"   Quiz Model: {request.quiz_model.provider.value} - {request.quiz_model.model_name}")
        print(f"   Flashcard Model: {request.flashcard_model.provider.value} - {request.flashcard_model.model_name}")
        print(f"   Summary Model: {request.summary_model.provider.value} - {request.summary_model.model_name}")
        print(f"   Test Model: {request.test_model.provider.value} - {request.test_model.model_name}")
        
        print("⚡ Updating configuration...")
        await config_service.update_configuration(
            chat_model=request.chat_model,
            quiz_model=request.quiz_model,
            flashcard_model=request.flashcard_model,
            summary_model=request.summary_model,
            test_model=request.test_model
        )
        
        print("🔄 Updating all services...")
        # Update all services with new configuration
        await service_manager.update_all_services()
        
        current_config = await config_service.get_current_config()
        
        print("✅ Configuration updated successfully")
        print("=" * 60)
        
        return ConfigResponse(
            message="Model configuration updated successfully",
            current_config=current_config
        )
    except Exception as e:
        print(f"❌ Config Update API Error: {str(e)}")
        print("=" * 60)
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

@router.get("/models")
async def get_model_config(config_service: ConfigService = Depends(get_config_service)):
    try:
        config = await config_service.get_current_config()
        return {"current_config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve configuration: {str(e)}")

async def get_ollama_models():
    """Fetch available models from local Ollama instance."""
    try:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get("models", []):
                model_name = model.get("name", "")
                if model_name:
                    # Clean up model name for display
                    display_name = model_name.split(":")[0]  # Remove tag part
                    models.append(ProviderModel(name=model_name, display_name=display_name))
            return models
    except Exception as e:
        print(f"Failed to fetch Ollama models: {e}")
    
    # Fallback models if Ollama is not available
    return [
        ProviderModel(name="llama2", display_name="Llama 2"),
        ProviderModel(name="codellama", display_name="Code Llama"),
        ProviderModel(name="mistral", display_name="Mistral")
    ]

@router.get("/providers", response_model=ProvidersResponse)
async def get_available_providers():
    # Get dynamic Ollama models
    ollama_models = await get_ollama_models()
    
    providers = [
        ProviderInfo(
            name="ollama",
            display_name="Ollama",
            requires_api_key=False,
            models=ollama_models
        ),
        ProviderInfo(
            name="groq",
            display_name="Groq",
            requires_api_key=True,
            models=[
                ProviderModel(name="qwen/qwen3-32b", display_name="Qwen 3 32B"),
                ProviderModel(name="meta-llama/llama-prompt-guard-2-86m", display_name="Llama Prompt Guard 2 86M"),
                ProviderModel(name="meta-llama/llama-4-maverick-17b-128e-instruct", display_name="Llama 4 Maverick 17B"),
                ProviderModel(name="llama-3.3-70b-versatile", display_name="Llama 3.3 70B Versatile"),
                ProviderModel(name="llama-3.1-8b-instant", display_name="Llama 3.1 8B Instant")
            ]
        ),
        ProviderInfo(
            name="openrouter",
            display_name="OpenRouter",
            requires_api_key=True,
            models=[
                ProviderModel(name="qwen/qwen3-235b-a22b:free", display_name="Qwen 3 235B A22B (Free)"),
                ProviderModel(name="mistralai/mistral-small-3.1-24b-instruct:free", display_name="Mistral Small 3.1 24B (Free)"),
                ProviderModel(name="google/gemma-3-12b-it:free", display_name="Google Gemma 3 12B IT (Free)"),
                ProviderModel(name="google/gemma-3-27b-it:free", display_name="Google Gemma 3 27B IT (Free)"),
                ProviderModel(name="meta-llama/llama-3.3-70b-instruct:free", display_name="Llama 3.3 70B Instruct (Free)"),
                ProviderModel(name="mistralai/mistral-7b-instruct:free", display_name="Mistral 7B Instruct (Free)")
            ]
        ),
        ProviderInfo(
            name="gemini",
            display_name="Google Gemini",
            requires_api_key=True,
            models=[
                ProviderModel(name="gemini-2.5-pro", display_name="Gemini 2.5 Pro"),
                ProviderModel(name="gemini-2.5-flash", display_name="Gemini 2.5 Flash"),
                ProviderModel(name="gemini-2.5-flash-lite", display_name="Gemini 2.5 Flash Lite")
            ]
        )
    ]
    
    return ProvidersResponse(providers=providers)

@router.get("/status")
async def config_status():
    return {"status": "Config service is running"}