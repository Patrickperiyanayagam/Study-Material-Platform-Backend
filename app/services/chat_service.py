import asyncio
import os
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from app.services.document_processor import DocumentProcessor
from app.services.model_factory import ModelFactory

class ChatService:
    def __init__(self, initial_config: Dict[str, Any] = None):
        self.document_processor = DocumentProcessor()
        self.sessions = {}  # Store conversation memory per session
        
        # Load initial configuration or use defaults
        if initial_config:
            print("💬 CHAT SERVICE - Received custom config:", initial_config)
            config = initial_config
        else:
            default_config = {
                "provider": "ollama",
                "model_name": "llama3.2",
                "temperature": 0.7,
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            }
            print("💬 CHAT SERVICE - Using default config:", default_config)
            config = default_config
        
        # Create LLM instance using factory
        self.current_config = config
        self.llm = self._create_llm_instance(config)
        self._initialize_prompt()
    
    def _initialize_prompt(self):
        """Initialize the chat prompt template."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context from uploaded documents. 

IMPORTANT: Provide only the final answer. Do not include any thinking process, reasoning steps, or tags like <think>. Give a direct, clear answer to the user's question.

Use the following context to answer the user's question. If the question cannot be answered using the provided context, say so clearly.

Context:
{context}

Provide accurate, helpful responses based on the context. If you reference specific information, mention which document it came from when possible. Answer directly without showing your thought process."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
    
    def _get_or_create_memory(self, session_id: str) -> ConversationBufferMemory:
        """Get or create conversation memory for a session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        return self.sessions[session_id]
    
    async def process_message(self, message: str, session_id: str) -> Dict[str, Any]:
        """Process a chat message and return the response."""
        try:
            # Get conversation memory
            memory = self._get_or_create_memory(session_id)
            
            # Search for relevant documents
            relevant_docs = await self.document_processor.search_documents(message, k=2)
            
            # Prepare context from retrieved documents
            context = self._format_context(relevant_docs)
            
            # Get chat history
            chat_history = memory.chat_memory.messages
            
            # Create the chain
            chain = (
                {
                    "context": lambda x: context,
                    "input": RunnablePassthrough(),
                    "chat_history": lambda x: chat_history
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Generate response
            print(f"🔮 Making LLM API call...")
            print(f"   Provider: {self.current_config['provider']}")
            print(f"   Model: {self.current_config['model_name']}")
            print(f"   Message: {message[:100]}..." if len(message) > 100 else f"   Message: {message}")
            print(f"   Context length: {len(context)} characters")
            
            # Time the API call
            import time
            start_time = time.time()
            
            response = await asyncio.to_thread(chain.invoke, {"input": message})
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"🎯 LLM API call completed:")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Response length: {len(response)} characters")
            print(f"   Response preview: {response[:150]}..." if len(response) > 150 else f"   Response: {response}")
            
            # Update memory
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(response)
            
            # Extract sources
            sources = [doc["metadata"].get("source", "Unknown") for doc in relevant_docs]
            unique_sources = list(set(sources))
            
            return {
                "response": response,
                "sources": unique_sources,
                "session_id": session_id
            }
            
        except Exception as e:
            raise Exception(f"Chat processing failed: {str(e)}")
    
    def _format_context(self, relevant_docs: List[Dict]) -> str:
        """Format retrieved documents into context string."""
        if not relevant_docs:
            return "No relevant documents found."
        
        context_parts = []
        for doc in relevant_docs:
            source = doc["metadata"].get("source", "Unknown")
            content = doc["content"]
            context_parts.append(f"From {source}:\n{content}\n")
        
        return "\n".join(context_parts)
    
    async def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get chat history for a session."""
        try:
            if session_id not in self.sessions:
                return []
            
            memory = self.sessions[session_id]
            messages = memory.chat_memory.messages
            
            formatted_messages = []
            for message in messages:
                if isinstance(message, HumanMessage):
                    formatted_messages.append({
                        "role": "user",
                        "content": message.content
                    })
                elif isinstance(message, AIMessage):
                    formatted_messages.append({
                        "role": "assistant",
                        "content": message.content
                    })
            
            return formatted_messages
            
        except Exception as e:
            raise Exception(f"Failed to retrieve chat history: {str(e)}")
    
    async def clear_session(self, session_id: str):
        """Clear a chat session."""
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
        except Exception as e:
            raise Exception(f"Failed to clear session: {str(e)}")
    
    def _create_llm_instance(self, config: Dict[str, Any]):
        """Create LLM instance from configuration."""
        print("💬 CHAT SERVICE - Creating LLM instance with config:", config)
        
        try:
            api_key = None
            if config["provider"] != "ollama":
                print(f"📡 Getting API key for {config['provider']}")
                api_key = ModelFactory.get_api_key_for_provider(config["provider"])
            
            print(f"🔧 Calling ModelFactory.create_llm with:")
            print(f"   provider={config['provider']}")
            print(f"   model_name={config['model_name']}")
            print(f"   temperature={config.get('temperature', 0.7)}")
            print(f"   base_url={config.get('base_url')}")
            print(f"   max_tokens={config.get('max_tokens')}")
            
            llm_instance = ModelFactory.create_llm(
                provider=config["provider"],
                model_name=config["model_name"],
                temperature=config.get("temperature", 0.7),
                base_url=config.get("base_url"),
                api_key=api_key,
                max_tokens=config.get("max_tokens")
            )
            
            print("✅ LLM instance created successfully in ChatService")
            return llm_instance
            
        except Exception as e:
            # Fallback to default Ollama configuration
            print(f"❌ Failed to create LLM with config {config}: {e}")
            print("🔄 Falling back to default Ollama configuration")
            
            return ModelFactory.create_llm(
                provider="ollama",
                model_name="llama3.2",
                temperature=0.7,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
    
    def update_model_config(self, provider: str, model_name: str, **kwargs):
        """Update the chat model configuration."""
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
            print(f"Chat service updated to use {provider} with model {model_name}")
                
        except Exception as e:
            raise Exception(f"Failed to update model config: {str(e)}")
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.sessions.keys())