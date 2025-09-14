from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GEMINI = "gemini"

class ModelConfig(BaseModel):
    provider: ModelProvider
    model_name: str
    base_url: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)

class DocumentUploadResponse(BaseModel):
    message: str
    file_count: int
    processed_chunks: int
    file_names: List[str]

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's chat message")
    session_id: Optional[str] = None
    model_configuration: Optional[ModelConfig] = None

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    sources: List[str] = Field(default=[], description="Source documents used")
    session_id: str

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: int = Field(..., ge=0, le=3, description="Index of correct answer (0-3)")
    explanation: str

class QuizRequest(BaseModel):
    num_questions: int = Field(default=10, ge=1, le=50)
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")
    topics: Optional[List[str]] = None
    model_configuration: Optional[ModelConfig] = None

class QuizResponse(BaseModel):
    questions: List[QuizQuestion]
    total_questions: int

class FlashCard(BaseModel):
    front: str
    back: str
    topic: str
    difficulty: str

class FlashCardRequest(BaseModel):
    num_cards: int = Field(default=10, ge=1, le=50)
    topics: Optional[List[str]] = None
    model_configuration: Optional[ModelConfig] = None

class FlashCardResponse(BaseModel):
    cards: List[FlashCard]
    total_cards: int

class ConfigRequest(BaseModel):
    chat_model: ModelConfig
    quiz_model: ModelConfig
    flashcard_model: ModelConfig

class ConfigResponse(BaseModel):
    message: str
    current_config: Dict[str, Any]

class ProviderModel(BaseModel):
    name: str
    display_name: str

class ProviderInfo(BaseModel):
    name: str
    display_name: str
    requires_api_key: bool
    models: List[ProviderModel]

class ProvidersResponse(BaseModel):
    providers: List[ProviderInfo]