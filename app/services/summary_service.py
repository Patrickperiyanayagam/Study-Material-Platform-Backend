import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.document_processor import DocumentProcessor
from app.services.model_factory import ModelFactory

class SummaryService:
    """Service for generating document summaries with different lengths and types."""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """Initialize summary service with model configuration."""
        self.document_processor = DocumentProcessor()
        self.model_config = model_config or {
            "provider": "ollama",
            "model_name": "llama3.1:8b",
            "temperature": 0.7,
            "base_url": "http://localhost:11434"
        }
        
        print(f"📄 SUMMARY SERVICE - Initializing with config: {self.model_config}")
        
        try:
            # Initialize LLM using model factory
            self.llm = ModelFactory.create_llm(
                provider=self.model_config["provider"],
                model_name=self.model_config["model_name"],
                temperature=self.model_config.get("temperature", 0.7),
                base_url=self.model_config.get("base_url"),
                api_key=self.model_config.get("api_key"),
                max_tokens=self.model_config.get("max_tokens")
            )
            self.output_parser = StrOutputParser()
            print("✅ Summary service initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize summary service: {str(e)}")
            # Fallback to default Ollama configuration
            fallback_config = {
                "provider": "ollama",
                "model_name": "llama3.1:8b",
                "temperature": 0.7,
                "base_url": "http://localhost:11434"
            }
            self.llm = ModelFactory.create_llm(
                provider=fallback_config["provider"],
                model_name=fallback_config["model_name"],
                temperature=fallback_config.get("temperature", 0.7),
                base_url=fallback_config.get("base_url"),
                api_key=fallback_config.get("api_key"),
                max_tokens=fallback_config.get("max_tokens")
            )
            self.output_parser = StrOutputParser()
            print("⚠️ Using fallback configuration")

    async def generate_summary(self, length: str = "medium", summary_type: str = "overview", topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate summary from uploaded documents."""
        try:
            print(f"📄 SUMMARY SERVICE - Generating {length} {summary_type} summary")
            
            # Get document content
            documents = await self._get_document_content(topics)
            if not documents:
                raise Exception("No documents found to summarize")
            
            # Format content for summarization
            content = self._format_content_for_summary(documents)
            print(f"📊 Content length: {len(content)} characters from {len(documents)} documents")
            
            # Generate summary using LLM
            summary_content = await self._generate_summary_content(content, length, summary_type)
            
            # Calculate metadata
            word_count = len(summary_content.split())
            reading_time = max(1, word_count // 200)  # ~200 words per minute
            sources_used = len(set(doc.get('metadata', {}).get('source', '') for doc in documents))
            
            # Calculate confidence score based on source diversity and content length
            confidence_score = min(100, int((sources_used * 20) + (len(content) / 100)))
            
            result = {
                "content": summary_content,
                "length": length,
                "type": summary_type,
                "topics": topics,
                "word_count": word_count,
                "reading_time": reading_time,
                "sources_used": sources_used,
                "confidence_score": f"{confidence_score}%"
            }
            
            print(f"✅ Summary generated: {word_count} words, {sources_used} sources")
            return result
            
        except Exception as e:
            print(f"❌ Summary generation failed: {str(e)}")
            raise Exception(f"Summary generation failed: {str(e)}")

    async def _get_document_content(self, topics: Optional[List[str]] = None) -> List[Dict]:
        """Retrieve relevant document content for summarization."""
        try:
            if topics:
                # Search for specific topics
                all_docs = []
                for topic in topics:
                    topic_docs = await self.document_processor.search_documents(topic, k=10)
                    all_docs.extend(topic_docs)
                
                # Remove duplicates based on content
                seen_content = set()
                unique_docs = []
                for doc in all_docs:
                    content_hash = hash(doc['content'][:100])  # Use first 100 chars as hash
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        unique_docs.append(doc)
                
                return unique_docs[:20]  # Limit to 20 documents
            else:
                # Get documents from all sources
                all_sources = await self.document_processor.get_all_sources()
                all_docs = []
                
                for source in all_sources:
                    source_docs = await self.document_processor.search_documents(source, k=5)
                    all_docs.extend(source_docs)
                
                return all_docs[:25]  # Limit to 25 documents for overview
                
        except Exception as e:
            print(f"❌ Failed to get document content: {str(e)}")
            return []

    def _format_content_for_summary(self, documents: List[Dict]) -> str:
        """Format document content for summary generation."""
        content_parts = []
        
        # Group documents by source
        source_groups = {}
        for doc in documents:
            source = doc.get('metadata', {}).get('source', 'Unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc['content'])
        
        # Format content by source
        for source, contents in source_groups.items():
            content_parts.append(f"\n--- From {source} ---")
            for content in contents:
                # Clean and limit content length
                clean_content = re.sub(r'\s+', ' ', content.strip())
                if len(clean_content) > 500:
                    clean_content = clean_content[:500] + "..."
                content_parts.append(clean_content)
        
        full_content = "\n".join(content_parts)
        
        # Limit total content length to prevent context overflow
        max_content_length = 8000  # Conservative limit for most models
        if len(full_content) > max_content_length:
            full_content = full_content[:max_content_length] + "\n\n[Content truncated for length...]"
        
        return full_content

    async def _generate_summary_content(self, content: str, length: str, summary_type: str) -> str:
        """Generate summary content using LLM."""
        
        # Define length guidelines
        length_guidelines = {
            "short": "1-2 paragraphs (150-300 words)",
            "medium": "3-5 paragraphs (300-600 words)", 
            "long": "detailed analysis (600+ words)"
        }
        
        # Define type-specific instructions
        type_instructions = {
            "overview": "Provide a general overview covering the main themes and concepts",
            "key_points": "Focus on the most important points and key takeaways",
            "detailed": "Provide comprehensive analysis with supporting details and examples",
            "bullet_points": "Present information in clear bullet points with headings"
        }
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
You are an expert content summarizer. Your task is to create a {summary_type} summary of the provided documents.

INSTRUCTIONS:
- Length: {length_guideline}
- Type: {type_instruction}
- Format: Use clean Markdown formatting with proper headers, bullet points, and emphasis
- Be accurate and faithful to the source material
- Use clear, professional language
- Maintain logical flow and structure
- Include specific details when relevant
- Use ## for main sections, ### for subsections
- Use **bold** for important terms and concepts
- Use bullet points (-) for lists
- Avoid excessive symbols or special characters

CONTENT TO SUMMARIZE:
{content}

Generate a well-formatted markdown summary:
""")
        
        # Create chain
        chain = prompt | self.llm | self.output_parser
        
        # Generate summary
        try:
            summary = await asyncio.to_thread(
                chain.invoke,
                {
                    "summary_type": summary_type.replace('_', ' '),
                    "length_guideline": length_guidelines.get(length, "medium length"),
                    "type_instruction": type_instructions.get(summary_type, "general summary"),
                    "content": content
                }
            )
            
            # Clean up the summary
            summary = summary.strip()
            
            # Ensure minimum quality
            if len(summary) < 50:
                raise Exception("Generated summary is too short")
            
            return summary
            
        except Exception as e:
            print(f"❌ LLM generation failed: {str(e)}")
            raise Exception(f"Failed to generate summary content: {str(e)}")

    async def get_available_topics(self) -> List[str]:
        """Get list of available topics (document sources)."""
        try:
            return await self.document_processor.get_all_sources()
        except Exception as e:
            print(f"❌ Failed to get topics: {str(e)}")
            return []

    def update_model_config(self, new_config: Dict[str, Any]):
        """Update the model configuration."""
        try:
            print(f"🔄 SUMMARY SERVICE - Updating model config: {new_config}")
            self.model_config = new_config
            self.llm = ModelFactory.create_llm(
                provider=new_config["provider"],
                model_name=new_config["model_name"],
                temperature=new_config.get("temperature", 0.7),
                base_url=new_config.get("base_url"),
                api_key=new_config.get("api_key"),
                max_tokens=new_config.get("max_tokens")
            )
            print("✅ Model configuration updated successfully")
        except Exception as e:
            print(f"❌ Failed to update model config: {str(e)}")
            raise Exception(f"Failed to update model configuration: {str(e)}")

    def get_current_config(self) -> Dict[str, Any]:
        """Get current model configuration."""
        return self.model_config.copy()