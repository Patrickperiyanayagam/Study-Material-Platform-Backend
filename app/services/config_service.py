import json
import os
from typing import Dict, Any
from app.models.schemas import ModelConfig

class ConfigService:
    def __init__(self):
        upload_dir = os.getenv("UPLOAD_DIR", "./data/documents")
        config_dir = os.path.dirname(upload_dir)
        self.config_file = os.path.join(config_dir, "model_config.json")
        self.current_config = self._load_config()
        
        print(f"⚙️ CONFIG SERVICE - Initializing")
        print(f"   Config file: {self.config_file}")
        
        # Service references for updating models
        self.chat_service = None
        self.quiz_service = None
        self.flashcard_service = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults."""
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        default_config = {
            "chat_model": {
                "provider": "ollama",
                "model_name": "llama3.2",
                "temperature": 0.7,
                "base_url": ollama_base_url
            },
            "quiz_model": {
                "provider": "ollama",
                "model_name": "llama3.2",
                "temperature": 0.7,
                "base_url": ollama_base_url
            },
            "flashcard_model": {
                "provider": "ollama",
                "model_name": "llama3.2",
                "temperature": 0.7,
                "base_url": ollama_base_url
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                for key, value in default_config.items():
                    if key not in loaded_config:
                        loaded_config[key] = value
                return loaded_config
            else:
                return default_config
        except Exception:
            return default_config
    
    def _save_config(self):
        """Save current configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.current_config, f, indent=2)
        except Exception as e:
            raise Exception(f"Failed to save configuration: {str(e)}")
    
    async def update_configuration(
        self, 
        chat_model: ModelConfig, 
        quiz_model: ModelConfig, 
        flashcard_model: ModelConfig
    ):
        """Update the model configuration and rebuild the processing pipeline."""
        try:
            # Update configuration (API keys come from environment)
            self.current_config = {
                "chat_model": {
                    "provider": chat_model.provider.value,
                    "model_name": chat_model.model_name,
                    "temperature": chat_model.temperature,
                    "base_url": chat_model.base_url,
                    "max_tokens": chat_model.max_tokens
                },
                "quiz_model": {
                    "provider": quiz_model.provider.value,
                    "model_name": quiz_model.model_name,
                    "temperature": quiz_model.temperature,
                    "base_url": quiz_model.base_url,
                    "max_tokens": quiz_model.max_tokens
                },
                "flashcard_model": {
                    "provider": flashcard_model.provider.value,
                    "model_name": flashcard_model.model_name,
                    "temperature": flashcard_model.temperature,
                    "base_url": flashcard_model.base_url,
                    "max_tokens": flashcard_model.max_tokens
                }
            }
            
            # Save to file
            self._save_config()
            
            # Update service instances if they exist
            await self._update_service_models()
            
        except Exception as e:
            raise Exception(f"Failed to update configuration: {str(e)}")
    
    def _get_api_key_for_provider(self, provider: str) -> str:
        """Get API key from environment variables based on provider."""
        provider_keys = {
            "groq": os.getenv("GROQ_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY"), 
            "openai": os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY")
        }
        api_key = provider_keys.get(provider)
        return api_key or ""
    
    async def _update_service_models(self):
        """Update the model configurations in all services."""
        try:
            # Update chat service
            if self.chat_service:
                config = self.current_config["chat_model"]
                api_key = self._get_api_key_for_provider(config["provider"])
                self.chat_service.update_model_config(
                    provider=config["provider"],
                    model_name=config["model_name"],
                    temperature=config["temperature"],
                    base_url=config.get("base_url"),
                    api_key=api_key,
                    max_tokens=config.get("max_tokens")
                )
            
            # Update quiz service
            if self.quiz_service:
                config = self.current_config["quiz_model"]
                api_key = self._get_api_key_for_provider(config["provider"])
                self.quiz_service.update_model_config(
                    provider=config["provider"],
                    model_name=config["model_name"],
                    temperature=config["temperature"],
                    base_url=config.get("base_url"),
                    api_key=api_key,
                    max_tokens=config.get("max_tokens")
                )
            
            # Update flashcard service
            if self.flashcard_service:
                config = self.current_config["flashcard_model"]
                api_key = self._get_api_key_for_provider(config["provider"])
                self.flashcard_service.update_model_config(
                    provider=config["provider"],
                    model_name=config["model_name"],
                    temperature=config["temperature"],
                    base_url=config.get("base_url"),
                    api_key=api_key,
                    max_tokens=config.get("max_tokens")
                )
                
        except Exception as e:
            # Log the error but don't fail the configuration update
            print(f"Warning: Failed to update service models: {str(e)}")
    
    async def get_current_config(self) -> Dict[str, Any]:
        """Get the current model configuration."""
        # Return config without API keys (they are managed via environment)
        safe_config = {}
        for key, value in self.current_config.items():
            safe_config[key] = {k: v for k, v in value.items()}
            # Indicate if API key is available in environment
            provider = value.get("provider")
            if provider and provider != "ollama":
                api_key = self._get_api_key_for_provider(provider)
                safe_config[key]["has_api_key"] = bool(api_key)
        
        return safe_config
    
    def set_service_references(
        self, 
        chat_service = None,
        quiz_service = None, 
        flashcard_service = None
    ):
        """Set references to services for model updates."""
        if chat_service:
            self.chat_service = chat_service
        if quiz_service:
            self.quiz_service = quiz_service
        if flashcard_service:
            self.flashcard_service = flashcard_service
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        provider_configs = {
            "ollama": {
                "requires_api_key": False,
                "supports_base_url": True,
                "default_base_url": "http://localhost:11434",
"default_models": []  # Dynamic models from Ollama API
            },
            "groq": {
                "requires_api_key": True,
                "supports_base_url": False,
"default_models": ["qwen/qwen3-32b", "meta-llama/llama-prompt-guard-2-86m", "meta-llama/llama-4-maverick-17b-128e-instruct", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
            },
            "openrouter": {
                "requires_api_key": True,
                "supports_base_url": False,
"default_models": ["deepseek/deepseek-chat-v3.1:free", "qwen/qwen3-coder:free", "mistralai/mistral-small-3.2-24b-instruct:free", "meta-llama/llama-3.3-8b-instruct:free", "qwen/qwen3-235b-a22b:free", "deepseek/deepseek-r1-distill-llama-70b:free"]
            },
            "openai": {
                "requires_api_key": True,
                "supports_base_url": False,
"default_models": []  # OpenAI removed from requirements
            },
            "gemini": {
                "requires_api_key": True,
                "supports_base_url": False,
"default_models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
            }
        }
        
        return provider_configs.get(provider, {})