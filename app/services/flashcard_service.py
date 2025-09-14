import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.document_processor import DocumentProcessor
from app.services.model_factory import ModelFactory
from app.models.schemas import FlashCard
# Environment variables accessed directly with os.getenv()

class FlashCardService:
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
        """Initialize the flashcard generation prompt."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating educational flashcards. Generate flashcards based on the provided context from documents.

Instructions:
1. Create {num_cards} flashcards with key information from the context
2. Each flashcard should have a clear, concise question/term on the front
3. The back should have a comprehensive but focused answer/explanation
4. Focus on the most important concepts, definitions, processes, and facts
5. Make flashcards that help with active recall and understanding

Context from documents:
{context}

Return the flashcards in the following JSON format:
{{
  "flashcards": [
    {{
      "front": "Question or term here",
      "back": "Detailed answer or explanation here",
      "topic": "Main topic category",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}"""),
            ("human", "Generate {num_cards} flashcards based on the most important concepts from the provided context.")
        ])
    
    async def generate_flashcards(
        self, 
        num_cards: int = 10, 
        topics: Optional[List[str]] = None
    ) -> List[FlashCard]:
        """Generate flashcards from the document corpus."""
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
                    "key concepts", "important definitions", "main points",
                    "processes", "methods", "principles", "facts"
                ]
                all_content = []
                for query in sample_queries:
                    docs = await self.document_processor.search_documents(query, k=2)
                    all_content.extend(docs)
            
            if not all_content:
                raise Exception("No content available for flashcard generation. Please upload documents first.")
            
            # Format context
            context = self._format_context_for_flashcards(all_content, num_cards)
            
            # Generate flashcards using LLM
            chain = self.prompt | self.llm | StrOutputParser()
            
            response = await asyncio.to_thread(
                chain.invoke,
                {
                    "num_cards": num_cards,
                    "context": context
                }
            )
            
            # Parse the JSON response
            flashcard_data = self._parse_flashcard_response(response)
            
            # Convert to FlashCard objects
            flashcards = []
            for card_data in flashcard_data.get("flashcards", []):
                try:
                    flashcard = FlashCard(
                        front=card_data["front"],
                        back=card_data["back"],
                        topic=card_data.get("topic", "General"),
                        difficulty=card_data.get("difficulty", "medium")
                    )
                    flashcards.append(flashcard)
                except (KeyError, ValueError) as e:
                    # Skip malformed flashcards
                    continue
            
            if not flashcards:
                raise Exception("Failed to generate valid flashcards")
            
            return flashcards[:num_cards]  # Ensure we don't return more than requested
            
        except Exception as e:
            raise Exception(f"Flashcard generation failed: {str(e)}")
    
    def _format_context_for_flashcards(self, documents: List[Dict], num_cards: int) -> str:
        """Format documents into context suitable for flashcard generation."""
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
    
    def _parse_flashcard_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract flashcards."""
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
            return self._manual_parse_flashcard_response(response)
    
    def _manual_parse_flashcard_response(self, response: str) -> Dict[str, Any]:
        """Manually parse flashcard response if JSON parsing fails."""
        flashcards = []
        lines = response.split('\n')
        
        current_card = {}
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Look for front/back indicators
            if line.lower().startswith('front:') or 'question:' in line.lower():
                current_section = 'front'
                content = re.sub(r'^(front:|question:)\s*', '', line, flags=re.IGNORECASE)
                if content:
                    current_card['front'] = content
            elif line.lower().startswith('back:') or 'answer:' in line.lower():
                current_section = 'back'
                content = re.sub(r'^(back:|answer:)\s*', '', line, flags=re.IGNORECASE)
                if content:
                    current_card['back'] = content
            elif line.lower().startswith('topic:'):
                content = re.sub(r'^topic:\s*', '', line, flags=re.IGNORECASE)
                if content:
                    current_card['topic'] = content
            elif line.lower().startswith('difficulty:'):
                content = re.sub(r'^difficulty:\s*', '', line, flags=re.IGNORECASE)
                if content:
                    current_card['difficulty'] = content.lower()
            elif line and current_section:
                # Continue adding to the current section
                if current_section in current_card:
                    current_card[current_section] += ' ' + line
                else:
                    current_card[current_section] = line
            
            # If we have a complete card, add it
            if 'front' in current_card and 'back' in current_card:
                if 'topic' not in current_card:
                    current_card['topic'] = "General"
                if 'difficulty' not in current_card:
                    current_card['difficulty'] = "medium"
                    
                flashcards.append(current_card)
                current_card = {}
                current_section = None
        
        return {"flashcards": flashcards}
    
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
        """Update the flashcard generation model configuration."""
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
            print(f"Flashcard service updated to use {provider} with model {model_name}")
                
        except Exception as e:
            raise Exception(f"Failed to update model config: {str(e)}")