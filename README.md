# Study Material Platform - Backend

FastAPI-based backend with advanced AI orchestration using LangGraph for intelligent document processing and conversational memory.

## 🏗️ Architecture Overview

The backend is built using a modern, scalable architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                     │
├─────────────────────────────────────────────────────────────┤
│                      API Layer                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────┐  │
│  │ Upload  │ │  Chat   │ │  Quiz   │ │Flashcard│ │ Summary │ │Config│  │
│  │   API   │ │   API   │ │   API   │ │   API   │ │   API   │ │ API  │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └──────┘  │
│  ┌─────────┐                                                           │
│  │  Test   │                                                           │
│  │   API   │                                                           │
│  └─────────┘                                                           │
├─────────────────────────────────────────────────────────────┤
│                    Service Layer                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           PersistentLangGraphChatService                │ │
│  │              (Singleton Pattern)                       │ │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │ │
│  │ │   Memory    │ │    Graph    │ │    Model Factory    │ │ │
│  │ │ Management  │ │  Execution  │ │   (Multi-Provider)  │ │ │
│  │ └─────────────┘ └─────────────┘ └─────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              DocumentProcessor                          │ │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │ │
│  │ │   File      │ │    Text     │ │     Embedding       │ │ │
│  │ │ Processing  │ │  Chunking   │ │    Generation       │ │ │
│  │ └─────────────┘ └─────────────┘ └─────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     Data Layer                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  ChromaDB                               │ │
│  │        Vector Database & Document Storage               │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Core Components

### 1. Persistent LangGraph Chat Service (Singleton)

**Location**: `app/services/persistent_langgraph_chat_service.py`

**Architecture Features:**
- **Singleton Pattern**: One service instance per server lifecycle
- **Persistent Graph**: Created once on startup, reused for all conversations
- **Session-based Threading**: Isolated memory per user session
- **Dynamic Configuration**: Graph rebuilds only when model config changes

**Key Methods:**
```python
class PersistentLangGraphChatService:
    def __init__(self)                           # Singleton initialization
    def _create_persistent_graph(self)           # Graph creation
    def process_message(self, message, session_id) # Message processing
    def get_chat_history(self, session_id)       # History retrieval
    def clear_session(self, session_id)          # Session cleanup
    def update_model_config(self, new_config)    # Model switching
```

**Memory Architecture:**
- **InMemorySaver**: Maintains conversation state
- **Thread-based Sessions**: Each user gets isolated memory
- **Persistent Context**: Full conversation history preserved
- **Automatic Cleanup**: Sessions cleared on demand

### 2. Document Processing Pipeline

**Location**: `app/services/document_processor.py`

**Processing Flow:**
```
Document Upload → File Validation → Text Extraction → 
Text Chunking → Embedding Generation → Vector Storage
```

**Supported Formats:**
- **PDF**: PyPDFLoader for text extraction
- **DOC/DOCX**: UnstructuredWordDocumentLoader for Word documents  
- **PPT/PPTX**: UnstructuredPowerPointLoader for PowerPoint presentations
- **TXT**: Direct text processing
- **Size Limit**: 50MB per file

**Chunking Strategy:**
- **Method**: RecursiveCharacterTextSplitter
- **Chunk Size**: 1000 characters
- **Overlap**: 200 characters
- **Separators**: Optimized for academic content

**Embedding Pipeline:**
- **Model**: mxbai-embed-large:335m (via Ollama)
- **Dimensions**: 335 dimensions
- **Storage**: ChromaDB with automatic persistence
- **Search**: Semantic search with relevance scoring

### 3. Model Factory System

**Location**: `app/services/model_factory.py`

**Supported AI Providers:**

| Provider | Models | Use Case |
|----------|--------|----------|
| **Ollama** | llama3.1:8b, mistral, etc. | Local processing, privacy |
| **Groq** | llama3-70b, mixtral-8x7b | High-speed inference |
| **OpenRouter** | qwen3-235b, mistral-small-3.1, gemma-3, llama-3.3-70b | Model diversity |
| **Google Gemini** | gemini-pro, gemini-flash | Latest Google AI |
| **OpenAI** | gpt-4, gpt-3.5-turbo | Industry standard |

**Configuration Management:**
```python
{
    "provider": "ollama",
    "model_name": "llama3.1:8b", 
    "temperature": 0.7,
    "base_url": "http://localhost:11434",
    "max_tokens": 4096
}
```

### 4. Summary Service

**Location**: `app/services/summary_service.py`

**Summary Types:**
- **Overview**: General summary covering main themes and concepts
- **Key Points**: Focus on most important points and takeaways
- **Detailed**: Comprehensive analysis with supporting details
- **Bullet Points**: Structured list format with clear organization

**Length Options:**
- **Short**: 1-2 paragraphs (150-300 words)
- **Medium**: 3-5 paragraphs (300-600 words)
- **Long**: Detailed analysis (600+ words)

**Features:**
- **Document-level Summarization**: Generate summaries from all uploaded content
- **Topic-specific Summaries**: Focus on selected documents or topics
- **Semantic Content Analysis**: Intelligent content grouping and organization
- **Metadata Generation**: Word count, reading time, confidence scoring
- **Export Functionality**: Download summaries as text files
- **Clean Markdown Output**: Well-formatted responses with proper headers and bullet points

### 5. Test Service - Auto-Exam & AI Grading

**Location**: `app/services/test_service.py`

**Core Functionality:**
- **Test Generation**: Creates syllabus-aligned tests from uploaded documents with configurable question types
- **Question Configuration**: Supports different mark distributions (2/4/8-mark questions) with customizable difficulty levels
- **AI Grading**: Evaluates student answers against comprehensive rubrics with detailed feedback
- **Performance Analysis**: Provides per-question scoring, overall performance metrics, and targeted improvement guidance

**Technical Implementation:**
```python
class TestService:
    def __init__(self):
        self.vector_store = ChromaService()
    
    async def generate_test(self, num_questions: int, difficulty: str, 
                           mark_distribution: dict, model_config: dict) -> TestResponse:
        # Generate contextual test questions using vector DB and AI
        
    async def grade_test_answers(self, questions: List[TestQuestion], 
                               answers: List[str], model_config: dict) -> GradingResponse:
        # AI-powered grading with rubric-based evaluation
```

**Key Features:**
- **Contextual Generation**: Uses vector database retrieval for relevant content-based questions
- **Rubric-Based Grading**: Comprehensive evaluation criteria for objective and consistent scoring
- **Detailed Feedback**: Per-question analysis with strengths, weaknesses, and improvement suggestions
- **Study Plan Generation**: AI-generated personalized study recommendations based on performance
- **Multi-Model Support**: Compatible with all configured AI providers (Ollama, Groq, OpenRouter, etc.)

**API Endpoints:**
- `POST /api/test/generate` - Generate new test with custom parameters
- `POST /api/test/grade` - Grade submitted test answers with AI evaluation

### 6. Enhanced Response Formatting

**Location**: All chat and summary services

**Markdown Support Features:**
- **Source Citation**: Source filenames are displayed in bold formatting (e.g., **filename.pdf**)
- **Clean Markdown**: Responses use proper Markdown formatting with headers (##, ###), bullet points (-), and emphasis (**bold**)
- **Semantic Search**: Enhanced similarity search with relevance scoring for better content retrieval
- **Professional Layout**: Structured responses with logical flow and clear organization

**Response Enhancement:**
- Chat responses include bold source references for easy identification
- Summary responses use clean Markdown structure with proper sections
- No document numbering - only clean filename references
- Improved readability with consistent formatting standards

## 📡 API Endpoints

### Document Management (`/api/upload/`)

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| POST | `/documents` | Upload multiple files | `multipart/form-data` |
| DELETE | `/documents` | Clear all documents | None |
| GET | `/status` | Service health check | None |

**Upload Response:**
```json
{
    "message": "Documents uploaded successfully",
    "file_names": ["document1.pdf", "document2.txt"],
    "processed_count": 2
}
```

### Chat Interface (`/api/chat/`)

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| POST | `/message` | Send chat message | `ChatRequest` |
| GET | `/sessions/{id}/history` | Get conversation history | None |
| DELETE | `/sessions/{id}` | Clear session memory | None |
| GET | `/status` | Chat service status | None |
| POST | `/sessions/{id}/test-memory` | Test memory persistence | None |

**Chat Request/Response:**
```json
// Request
{
    "message": "What are the main topics in the documents?",
    "session_id": "user-session-123",
    "model_configuration": {
        "provider": "ollama",
        "model_name": "llama3.1:8b",
        "temperature": 0.7
    }
}

// Response
{
    "response": "Based on the documents, the main topics include...",
    "sources": ["document1.pdf", "document2.txt"],
    "session_id": "user-session-123"
}
```

### Quiz Generation (`/api/quiz/`)

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| POST | `/generate` | Generate MCQ quiz | `QuizRequest` |
| GET | `/status` | Quiz service status | None |

**Quiz Request/Response:**
```json
// Request
{
    "num_questions": 5,
    "difficulty": "medium",
    "topic": "machine learning",
    "model_configuration": {...}
}

// Response
{
    "questions": [
        {
            "question": "What is machine learning?",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "Machine learning is...",
            "difficulty": "medium",
            "topic": "machine learning"
        }
    ],
    "total_questions": 5,
    "generated_at": "2024-01-01T12:00:00Z"
}
```

### Flashcard Generation (`/api/flashcards/`)

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| POST | `/generate` | Generate flashcards | `FlashCardRequest` |
| GET | `/status` | Flashcard service status | None |

**Flashcard Response:**
```json
{
    "cards": [
        {
            "front": "What is the definition of AI?",
            "back": "Artificial Intelligence is...",
            "topic": "artificial intelligence",
            "difficulty": "easy"
        }
    ],
    "total_cards": 10,
    "generated_at": "2024-01-01T12:00:00Z"
}
```

### Test Generation & Grading (`/api/test/`)

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| POST | `/generate` | Generate syllabus-aligned test | `TestRequest` |
| POST | `/grade` | Grade test answers with AI | `GradingRequest` |

**Test Generation Request/Response:**
```json
// Request
{
    "num_questions": 5,
    "difficulty": "medium",
    "mark_distribution": {"2": 2, "4": 2, "8": 1},
    "topics": ["machine learning", "data structures"],
    "model_configuration": {
        "provider": "groq",
        "model_name": "llama-3.3-70b-versatile",
        "temperature": 0.7
    }
}

// Response
{
    "questions": [
        {
            "question": "Explain the concept of supervised learning...",
            "marks": 4,
            "rubric": "Clear definition (2 marks), Examples (1 mark), Applications (1 mark)"
        }
    ],
    "total_questions": 5,
    "total_marks": 20,
    "difficulty": "medium"
}
```

**Test Grading Request/Response:**
```json
// Request
{
    "questions": [...],
    "answers": ["Supervised learning is...", "Data structures are..."],
    "model_configuration": {...}
}

// Response
{
    "question_grades": [
        {
            "question_number": 1,
            "marks_awarded": 3.5,
            "total_marks": 4,
            "feedback": "Good understanding but missing examples",
            "strengths": ["Clear definition", "Accurate concepts"],
            "improvements": ["Include practical examples"]
        }
    ],
    "total_score": 17.5,
    "total_possible": 20,
    "percentage": 87.5,
    "overall_feedback": "Strong performance with room for improvement in examples",
    "study_plan": "Focus on practical applications and real-world examples"
}
```

### Summary Generation (`/api/summary/`)

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| POST | `/generate` | Generate document summary | `SummaryRequest` |
| GET | `/topics` | Get available topics | None |
| GET | `/status` | Summary service status | None |

**Summary Request/Response:**
```json
// Request
{
    "length": "medium",
    "type": "overview",
    "topics": ["document1.pdf", "document2.txt"],
    "model_configuration": {
        "provider": "ollama",
        "model_name": "llama3.1:8b",
        "temperature": 0.7
    }
}

// Response
{
    "content": "This summary provides an overview of...",
    "length": "medium",
    "type": "overview", 
    "topics": ["document1.pdf", "document2.txt"],
    "word_count": 450,
    "reading_time": 2,
    "sources_used": 5,
    "confidence_score": "85%"
}
```

### Configuration (`/api/config/`)

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| POST | `/models` | Update model config | `ModelConfig` |
| GET | `/models` | Get current config | None |
| GET | `/providers` | Available providers | None |
| GET | `/status` | Config service status | None |

## 🗃️ Data Models

### Core Schemas (`app/models/schemas.py`)

```python
class ModelConfig(BaseModel):
    provider: ModelProvider
    model_name: str
    temperature: float = 0.7
    base_url: Optional[str] = None
    max_tokens: Optional[int] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model_configuration: Optional[ModelConfig] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    session_id: str

class QuizRequest(BaseModel):
    num_questions: int = 5
    difficulty: str = "medium"
    topic: Optional[str] = None
    model_configuration: Optional[ModelConfig] = None

class FlashCardRequest(BaseModel):
    num_cards: int = 10
    topic: Optional[str] = None
    model_configuration: Optional[ModelConfig] = None

class SummaryRequest(BaseModel):
    length: str = "medium"  # short, medium, long
    type: str = "overview"  # overview, key_points, detailed, bullet_points
    topics: Optional[List[str]] = None
    model_configuration: Optional[ModelConfig] = None

class SummaryResponse(BaseModel):
    content: str
    length: str
    type: str
    topics: Optional[List[str]] = None
    word_count: int
    reading_time: int
    sources_used: int
    confidence_score: str
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **Ollama** (recommended for local AI)
- **ChromaDB** (automatically installed)

### Installation

1. **Clone and Setup Environment**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit with your settings
   OLLAMA_BASE_URL=http://localhost:11434
   GROQ_API_KEY=your_groq_key_here
   OPENAI_API_KEY=your_openai_key_here
   ```

3. **Install Ollama Models**
   ```bash
   ollama serve
   ollama pull llama3.1:8b
   ollama pull mxbai-embed-large:335m
   ```

### Running the Server

**Development Mode:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
```bash
python3 -m app.main
```

**Production Mode:**
```bash. 
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### API Documentation

Once running, access interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🧪 Testing

### Manual Testing

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Chat Test:**
```bash
curl -X POST "http://localhost:8000/api/chat/message" \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello", "session_id": "test-123"}'
```

**Memory Test:**
```bash
curl -X POST "http://localhost:8000/api/chat/sessions/test-123/test-memory"
```

### Unit Testing

```bash
pytest tests/ -v
pytest tests/test_chat_service.py -v
pytest tests/test_document_processor.py -v
```

## 📊 Performance Metrics

### LangGraph Performance
- **First Message**: ~10-15 seconds (graph creation + processing)
- **Subsequent Messages**: ~3-5 seconds (graph reuse)
- **Memory Overhead**: ~50MB per active session
- **Concurrent Sessions**: Supports 100+ simultaneous users

### Document Processing
- **PDF Processing**: ~2-5 seconds per MB
- **Text Chunking**: ~500ms per document
- **Embedding Generation**: ~1-3 seconds per chunk
- **Vector Storage**: ~100ms per chunk

## 🔧 Configuration

### Environment Variables

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Ollama Configuration  
OLLAMA_BASE_URL=http://localhost:11434

# AI Provider API Keys
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
OPENROUTER_API_KEY=your_openrouter_key

# Storage Configuration
UPLOAD_DIR=./data/documents
CHROMA_PERSIST_DIR=./data/chroma

# Model Configuration
DEFAULT_MODEL_PROVIDER=ollama
DEFAULT_MODEL_NAME=llama3.1:8b
DEFAULT_TEMPERATURE=0.7
```

### Logging Configuration

The application uses structured logging with different levels:

```python
# Development
LOG_LEVEL=DEBUG

# Production  
LOG_LEVEL=INFO
```

## 🔒 Security Features

### Input Validation
- **File Type Validation**: Whitelist approach for uploads
- **Size Limits**: Configurable file size restrictions
- **Content Sanitization**: Text cleaning and validation
- **SQL Injection Protection**: Parameterized queries

### API Security
- **CORS Configuration**: Configurable allowed origins
- **Rate Limiting**: Request throttling (optional)
- **Input Sanitization**: Pydantic model validation
- **Error Handling**: Secure error responses

### Data Privacy
- **Local Processing**: Ollama models run locally
- **Session Isolation**: User data separation
- **Memory Cleanup**: Automatic session clearing
- **File Storage**: Local document storage

## 📈 Monitoring and Observability

### Health Checks
- **Service Health**: `/health` endpoint
- **Component Status**: Individual service status endpoints
- **Memory Usage**: Session and graph monitoring
- **Model Availability**: AI provider connectivity

### Logging
- **Structured Logs**: JSON formatted logs
- **Request Tracking**: Request/response logging
- **Error Tracking**: Detailed error information
- **Performance Metrics**: Response time tracking

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations
- **Process Management**: Use gunicorn with multiple workers
- **Reverse Proxy**: Nginx for static files and load balancing  
- **Database**: Configure persistent ChromaDB storage
- **Monitoring**: Prometheus/Grafana for metrics
- **Logging**: Centralized logging with ELK stack

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install dependencies with `pip install -r requirements.txt`
4. Run tests with `pytest`
5. Follow PEP 8 style guidelines

### Code Structure
- **API Routes**: Add new endpoints in `app/api/`
- **Services**: Business logic in `app/services/`
- **Models**: Data schemas in `app/models/`
- **Tests**: Unit tests in `tests/`

---

**Built with FastAPI, LangGraph, and modern AI technologies for scalable document intelligence.**