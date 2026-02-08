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

class SummaryRequest(BaseModel):
    length: str = Field(default="medium", pattern="^(short|medium|long)$")
    type: str = Field(default="overview", pattern="^(overview|key_points|detailed|bullet_points)$")
    topics: Optional[List[str]] = None
    model_configuration: Optional[ModelConfig] = None

class SummaryResponse(BaseModel):
    content: str = Field(..., description="The generated summary content")
    length: str
    type: str
    topics: Optional[List[str]] = None
    word_count: int
    reading_time: int = Field(..., description="Estimated reading time in minutes")
    sources_used: int = Field(..., description="Number of source documents used")
    confidence_score: str = Field(..., description="Confidence score as percentage")

class TestQuestion(BaseModel):
    question: str = Field(..., description="The test question")
    marks: int = Field(..., ge=2, le=8, description="Marks for this question (2, 4, or 8)")
    difficulty: str = Field(..., description="Question difficulty level")
    topic: str = Field(..., description="Topic/subject area of the question")

class TestRequest(BaseModel):
    num_questions: int = Field(default=10, ge=1, le=20, description="Number of questions to generate")
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")
    mark_distribution: Dict[int, int] = Field(
        default={2: 5, 4: 3, 8: 2}, 
        description="Distribution of marks: {marks: count}"
    )
    topics: Optional[List[str]] = None
    model_configuration: Optional[ModelConfig] = None

class TestResponse(BaseModel):
    questions: List[TestQuestion]
    total_questions: int
    total_marks: int

class AnswerSubmission(BaseModel):
    question_index: int = Field(..., ge=0, description="Index of the question")
    answer: str = Field(..., description="User's answer to the question")

class GradingRequest(BaseModel):
    questions: List[TestQuestion]
    answers: List[AnswerSubmission]
    model_configuration: Optional[ModelConfig] = None

class QuestionGrade(BaseModel):
    question_index: int
    question: str
    user_answer: str
    marks_awarded: float = Field(..., ge=0, description="Marks awarded for this question")
    max_marks: int = Field(..., description="Maximum marks possible for this question")
    percentage: float = Field(..., ge=0, le=100, description="Percentage score for this question")
    feedback: str = Field(..., description="Detailed feedback on the answer")
    strengths: List[str] = Field(default=[], description="Strong points in the answer")
    improvements: List[str] = Field(default=[], description="Areas for improvement")

class GradingResponse(BaseModel):
    grades: List[QuestionGrade]
    total_marks_awarded: float
    total_marks_possible: int
    overall_percentage: float
    overall_feedback: str
    weak_topics: List[str] = Field(default=[], description="Topics that need more focus")
    study_plan: List[str] = Field(default=[], description="Recommended study plan")

class ConfigRequest(BaseModel):
    chat_model: ModelConfig
    quiz_model: ModelConfig
    flashcard_model: ModelConfig
    summary_model: ModelConfig
    test_model: ModelConfig

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