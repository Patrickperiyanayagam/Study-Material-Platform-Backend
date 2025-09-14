from typing import Dict, Any, Optional
import os
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class ModelFactory:
    """Factory class to create LLM instances based on provider and configuration."""
    
    @staticmethod
    def create_llm(provider: str, model_name: str, temperature: float = 0.7, 
                   base_url: Optional[str] = None, api_key: Optional[str] = None, 
                   max_tokens: Optional[int] = None, **kwargs):
        """Create an LLM instance based on the provider and configuration."""
        
        print(f"🏭 MODEL FACTORY - Creating LLM instance")
        print(f"   Provider: {provider}")
        print(f"   Model: {model_name}")
        print(f"   Temperature: {temperature}")
        print(f"   Base URL: {base_url}")
        print(f"   Max Tokens: {max_tokens}")
        
        if provider == "ollama":
            ollama_base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            print(f"🦙 Creating Ollama instance:")
            print(f"   URL: {ollama_base_url}")
            print(f"   Model: {model_name}")
            
            instance = ChatOllama(
                base_url=ollama_base_url,
                model=model_name,
                temperature=temperature,
                num_predict=max_tokens
            )
            print(f"✅ Ollama instance created successfully")
            return instance
        
        elif provider == "groq":
            if not api_key:
                api_key = os.getenv("GROQ_API_KEY")
            print(f"🚀 Creating Groq instance:")
            print(f"   API Key present: {'Yes' if api_key else 'No'}")
            print(f"   API Key (masked): {api_key[:10]}...{api_key[-4:] if api_key else 'None'}")
            print(f"   Model: {model_name}")
            
            if not api_key:
                error_msg = "Groq API key not found in environment"
                print(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            instance = ChatGroq(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"✅ Groq instance created successfully")
            return instance
        
        elif provider == "openrouter":
            if not api_key:
                api_key = os.getenv("OPENROUTER_API_KEY")
            print(f"🌐 Creating OpenRouter instance:")
            print(f"   API Key present: {'Yes' if api_key else 'No'}")
            print(f"   API Key (masked): {api_key[:10]}...{api_key[-4:] if api_key else 'None'}")
            print(f"   Model: {model_name}")
            print(f"   Base URL: https://openrouter.ai/api/v1")
            
            if not api_key:
                error_msg = "OpenRouter API key not found in environment"
                print(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            instance = ChatOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": "http://localhost:5173",
                    "X-Title": "Study Material Platform"
                }
            )
            print(f"✅ OpenRouter instance created successfully")
            return instance
        
        elif provider == "gemini":
            if not api_key:
                api_key = os.getenv("GEMINI_API_KEY")
            print(f"🤖 Creating Gemini instance:")
            print(f"   API Key present: {'Yes' if api_key else 'No'}")
            print(f"   API Key (masked): {api_key[:10]}...{api_key[-4:] if api_key else 'None'}")
            print(f"   Model: {model_name}")
            
            if not api_key:
                error_msg = "Gemini API key not found in environment"
                print(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            instance = ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            print(f"✅ Gemini instance created successfully")
            return instance
        
        elif provider == "openai":
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            print(f"🤖 Creating OpenAI instance:")
            print(f"   API Key present: {'Yes' if api_key else 'No'}")
            print(f"   API Key (masked): {api_key[:10]}...{api_key[-4:] if api_key else 'None'}")
            print(f"   Model: {model_name}")
            
            if not api_key:
                error_msg = "OpenAI API key not found in environment"
                print(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            instance = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"✅ OpenAI instance created successfully")
            return instance
        
        else:
            error_msg = f"Unsupported provider: {provider}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
    
    @staticmethod
    def get_api_key_for_provider(provider: str) -> str:
        """Get API key from environment variables based on provider."""
        print(f"🔑 Getting API key for provider: {provider}")
        
        provider_keys = {
            "groq": os.getenv("GROQ_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY")
        }
        
        api_key = provider_keys.get(provider)
        key_present = "Yes" if api_key else "No"
        key_masked = f"{api_key[:10]}...{api_key[-4:]}" if api_key else "None"
        
        print(f"   API Key found: {key_present}")
        print(f"   API Key (masked): {key_masked}")
        
        return api_key or ""
    
    @staticmethod
    def validate_provider_config(provider: str, model_name: str, api_key: Optional[str] = None) -> bool:
        """Validate if a provider configuration is complete and valid."""
        print(f"✅ Validating provider config:")
        print(f"   Provider: {provider}")
        print(f"   Model: {model_name}")
        
        if provider == "ollama":
            is_valid = bool(model_name)
            print(f"   Ollama validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
            return is_valid
        
        # For API-based providers, check if API key is available
        if provider in ["groq", "openrouter", "openai", "gemini"]:
            if not api_key:
                api_key = ModelFactory.get_api_key_for_provider(provider)
            is_valid = bool(api_key and model_name)
            print(f"   {provider.title()} validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
            return is_valid
        
        print(f"   ❌ Unsupported provider: {provider}")
        return False