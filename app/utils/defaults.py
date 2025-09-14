from typing import Dict, Any


def get_default_model_config() -> Dict[str, Any]:
    """Get default model configuration when none is provided."""
    return {
        "provider": "ollama",
        "model_name": "llama3.1:8b",
        "temperature": 0.7,
        "base_url": "http://localhost:11434",
        "max_tokens": None
    }