import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.document_processor import DocumentProcessor
from app.services.model_factory import ModelFactory
from app.models.schemas import QuizQuestion
# Environment variables accessed directly with os.getenv()

class QuizService:
    def __init__(self, initial_config: Dict[str, Any] = None):
        self.document_processor = DocumentProcessor()
        
        # Load initial configuration or use defaults
        if initial_config:
            config = initial_config
        else:
            config = {
                "provider": "ollama",
                "model_name": "llama3.2",
                "temperature": 0.7,
                "base_url": settings.OLLAMA_BASE_URL
            }
        
        # Create LLM instance using factory
        self.current_config = config
        self.llm = self._create_llm_instance(config)
        self._initialize_prompt()
    
    def _initialize_prompt(self):
        """Initialize the quiz generation prompt."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert quiz generator. Generate multiple-choice questions based on the provided context from documents.

Instructions:
1. Create {num_questions} multiple-choice questions with 4 options each (A, B, C, D)
2. Make sure questions test understanding of the material, not just memorization
3. Difficulty level: {difficulty}
4. Each question should have exactly one correct answer
5. Provide a brief explanation for the correct answer

Context from documents:
{context}

Return the questions in the following JSON format:
{{
  "questions": [
    {{
      "question": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": 0,
      "explanation": "Explanation of why this is correct"
    }}
  ]
}}"""),
            ("human", "Generate {num_questions} {difficulty} difficulty quiz questions based on the provided context.")
        ])
    
    async def generate_quiz(
        self, 
        num_questions: int = 10, 
        difficulty: str = "medium", 
        topics: Optional[List[str]] = None
    ) -> List[QuizQuestion]:
        """Generate quiz questions from the document corpus."""
        try:
            # Get relevant content from documents
            if topics:
                # If specific topics are requested, search for them
                all_content = []
                for topic in topics:
                    docs = await self.document_processor.search_documents(topic, k=3)
                    all_content.extend(docs)
            else:
                # Get diverse content from all documents
                sample_queries = [
                    "main concepts", "key points", "important information",
                    "definitions", "examples", "processes", "methods"
                ]
                all_content = []
                for query in sample_queries:
                    docs = await self.document_processor.search_documents(query, k=2)
                    all_content.extend(docs)
            
            if not all_content:
                raise Exception("No content available for quiz generation. Please upload documents first.")
            
            # Format context
            context = self._format_context_for_quiz(all_content, num_questions)
            
            # Generate quiz using LLM
            chain = self.prompt | self.llm | StrOutputParser()
            
            response = await asyncio.to_thread(
                chain.invoke,
                {
                    "num_questions": num_questions,
                    "difficulty": difficulty,
                    "context": context
                }
            )
            
            # Parse the JSON response
            quiz_data = self._parse_quiz_response(response)
            
            # Convert to QuizQuestion objects
            questions = []
            for q_data in quiz_data.get("questions", []):
                try:
                    question = QuizQuestion(
                        question=q_data["question"],
                        options=q_data["options"][:4],  # Ensure exactly 4 options
                        correct_answer=int(q_data["correct_answer"]),
                        explanation=q_data.get("explanation", "")
                    )
                    questions.append(question)
                except (KeyError, ValueError) as e:
                    # Skip malformed questions
                    continue
            
            if not questions:
                raise Exception("Failed to generate valid quiz questions")
            
            return questions[:num_questions]  # Ensure we don't return more than requested
            
        except Exception as e:
            raise Exception(f"Quiz generation failed: {str(e)}")
    
    def _format_context_for_quiz(self, documents: List[Dict], num_questions: int) -> str:
        """Format documents into context suitable for quiz generation."""
        # Remove duplicates and get diverse content
        seen_content = set()
        unique_docs = []
        
        for doc in documents:
            content = doc["content"]
            if content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)
        
        # Limit context length to avoid token limits
        context_parts = []
        total_length = 0
        max_length = 4000  # Rough token limit
        
        for doc in unique_docs:
            source = doc["metadata"].get("source", "Document")
            content = doc["content"]
            part = f"From {source}:\n{content}\n\n"
            
            if total_length + len(part) > max_length:
                break
                
            context_parts.append(part)
            total_length += len(part)
        
        return "".join(context_parts)
    
    def _parse_quiz_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract quiz questions."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                return json.loads(response)
                
        except json.JSONDecodeError:
            # Fallback: manual parsing if JSON parsing fails
            return self._manual_parse_quiz_response(response)
    
    def _manual_parse_quiz_response(self, response: str) -> Dict[str, Any]:
        """Manually parse quiz response if JSON parsing fails."""
        questions = []
        lines = response.split('\n')
        
        current_question = {}
        options = []
        
        for line in lines:
            line = line.strip()
            
            # Look for question patterns
            if line.endswith('?') and 'question' not in current_question:
                current_question['question'] = line
                options = []
            
            # Look for options (A), (B), (C), (D) or A., B., C., D.
            elif re.match(r'^[A-D][\.)]\s*', line):
                option = re.sub(r'^[A-D][\.)]\s*', '', line)
                options.append(option)
            
            # Look for correct answer indication
            elif 'correct' in line.lower() and 'answer' in line.lower():
                # Try to extract the correct answer
                match = re.search(r'[A-D]', line)
                if match:
                    correct_letter = match.group()
                    current_question['correct_answer'] = ord(correct_letter) - ord('A')
            
            # If we have a complete question, add it
            if len(options) == 4 and 'question' in current_question and 'correct_answer' in current_question:
                current_question['options'] = options
                current_question['explanation'] = "Generated from document content"
                questions.append(current_question)
                current_question = {}
                options = []
        
        return {"questions": questions}
    
    async def get_available_topics(self) -> List[str]:
        """Get available topics from uploaded documents."""
        try:
            # Get all document sources
            sources = await self.document_processor.get_all_sources()
            
            # For now, return document names as topics
            # In a more advanced implementation, this could extract actual topics
            return sources
            
        except Exception as e:
            return []
    
    def _create_llm_instance(self, config: Dict[str, Any]):
        """Create LLM instance from configuration."""
        try:
            api_key = None
            if config["provider"] != "ollama":
                api_key = ModelFactory.get_api_key_for_provider(config["provider"])
            
            return ModelFactory.create_llm(
                provider=config["provider"],
                model_name=config["model_name"],
                temperature=config.get("temperature", 0.7),
                base_url=config.get("base_url"),
                api_key=api_key,
                max_tokens=config.get("max_tokens")
            )
        except Exception as e:
            # Fallback to default Ollama configuration
            print(f"Failed to create LLM with config {config}: {e}")
            return ModelFactory.create_llm(
                provider="ollama",
                model_name="llama3.2",
                temperature=0.7,
                base_url=settings.OLLAMA_BASE_URL
            )
    
    def update_model_config(self, provider: str, model_name: str, **kwargs):
        """Update the quiz generation model configuration."""
        try:
            config = {
                "provider": provider,
                "model_name": model_name,
                "temperature": kwargs.get("temperature", 0.7),
                "base_url": kwargs.get("base_url"),
                "api_key": kwargs.get("api_key"),
                "max_tokens": kwargs.get("max_tokens")
            }
            
            self.current_config = config
            self.llm = self._create_llm_instance(config)
            print(f"Quiz service updated to use {provider} with model {model_name}")
                
        except Exception as e:
            raise Exception(f"Failed to update model config: {str(e)}")