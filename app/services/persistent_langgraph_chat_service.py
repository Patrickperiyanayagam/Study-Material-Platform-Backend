import asyncio
import os
from typing import List, Dict, Any, Annotated, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from app.services.document_processor import DocumentProcessor
from app.services.model_factory import ModelFactory

class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    session_id: str

class PersistentLangGraphChatService:
    """
    Singleton LangGraph Chat Service that maintains persistent graph and memory.
    Graph is created once and reused for all conversations.
    Only recreated when model configuration changes.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PersistentLangGraphChatService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        print("🌟 PERSISTENT LANGGRAPH CHAT SERVICE - Initializing singleton")
        
        self.document_processor = DocumentProcessor()
        self.graph = None
        self.memory = InMemorySaver()
        self.current_config = None
        self.llm = None
        self.prompt = None
        
        # Initialize with default configuration
        self._initialize_default_config()
        self._create_persistent_graph()
        
        self._initialized = True
        print("✅ Persistent LangGraph Chat Service initialized successfully")
    
    def _initialize_default_config(self):
        """Initialize with default model configuration."""
        self.current_config = {
            "provider": "ollama",
            "model_name": "llama3.1:8b",
            "temperature": 0.7,
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        }
        print(f"🔧 PERSISTENT LANGGRAPH - Default config: {self.current_config}")
        
        # Create LLM instance
        self.llm = self._create_llm_instance(self.current_config)
        self._initialize_prompt()
    
    def _initialize_prompt(self):
        """Initialize the chat prompt template with strict no-thinking instructions."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context from uploaded documents.

CRITICAL INSTRUCTIONS:
- NEVER include any thinking process, reasoning steps, or internal thoughts
- NEVER use <think> tags or similar markers  
- NEVER show your work or explain your reasoning process
- Give ONLY the final answer
- Be direct, clear, and concise
- You have access to full conversation history - use it to answer follow-up questions
- If asked about previous questions, refer to the conversation history

Use the following context to answer the user's question. If the question cannot be answered using the provided context, say so clearly.

Context:
{context}

Provide accurate, helpful responses based on the context and conversation history. When referencing specific information, cite the source filename using bold formatting (e.g., "according to **filename.pptx**" or "from **filename.pdf**"). Do not use document numbers. Use proper markdown formatting in your responses."""),
            MessagesPlaceholder("messages"),
        ])
        print("✅ Prompt initialized with strict no-thinking instructions")
    
    def _create_persistent_graph(self):
        """Create the persistent LangGraph that will be reused for all conversations."""
        print("🕸️ PERSISTENT LANGGRAPH - Creating persistent state graph")
        
        # Create state graph
        graph_builder = StateGraph(State)
        
        def chatbot_node(state: State):
            """Main chatbot processing node - this gets called for each message."""
            print(f"🤖 PERSISTENT LANGGRAPH - Processing in chatbot node")
            print(f"   Session: {state.get('session_id', 'unknown')}")
            print(f"   Total messages in conversation: {len(state['messages'])}")
            print(f"   Context length: {len(state.get('context', ''))} characters")
            
            # Show last few messages for context tracking
            messages = state['messages']
            for i, msg in enumerate(messages[-3:]):  # Show last 3 messages
                role = "Human" if isinstance(msg, HumanMessage) else "AI"
                content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                print(f"   Message {i+1} [{role}]: {content_preview}")
            
            # Create chain with context and full message history
            chain = (
                {
                    "context": lambda x: state.get("context", ""),
                    "messages": lambda x: state["messages"]  # Full conversation history
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Generate response
            response = chain.invoke({})
            
            # Clean any remaining thinking tags
            response = self._clean_response(response)
            
            print(f"   Generated response length: {len(response)} characters")
            print(f"   Response preview: {response[:150]}..." if len(response) > 150 else f"   Response: {response}")
            
            return {"messages": [AIMessage(content=response)]}
        
        # Add the chatbot node
        graph_builder.add_node("chatbot", chatbot_node)
        graph_builder.set_entry_point("chatbot")
        graph_builder.set_finish_point("chatbot")
        
        # Compile the graph with persistent memory
        self.graph = graph_builder.compile(checkpointer=self.memory)
        
        print("✅ Persistent state graph created successfully")
    
    def _clean_response(self, response: str) -> str:
        """Remove any thinking tags or reasoning processes from the response."""
        import re
        
        # Remove <think> tags and their content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove thinking patterns
        thinking_patterns = [
            r'let me think.*?(?=\n\n|\n[A-Z]|$)',
            r'hmm.*?(?=\n\n|\n[A-Z]|$)',
            r'i need to.*?(?=\n\n|\n[A-Z]|$)',
            r'first.*?(?=\n\n|\n[A-Z]|$)',
            r'the user is asking.*?(?=\n\n|\n[A-Z]|$)',
        ]
        
        for pattern in thinking_patterns:
            response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra whitespace
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = response.strip()
        
        return response
    
    def update_model_config(self, new_config: Dict[str, Any]) -> bool:
        """Update model configuration and recreate graph if needed."""
        print(f"🔄 PERSISTENT LANGGRAPH - Checking model config update")
        print(f"   Current: {self.current_config}")
        print(f"   New: {new_config}")
        
        # Check if configuration actually changed
        if (self.current_config and 
            self.current_config.get("provider") == new_config.get("provider") and
            self.current_config.get("model_name") == new_config.get("model_name") and
            self.current_config.get("temperature") == new_config.get("temperature")):
            print("   No significant changes detected, keeping existing graph")
            return False
        
        print("   Significant changes detected, recreating graph")
        
        # Update configuration
        self.current_config = new_config
        
        # Create new LLM instance
        self.llm = self._create_llm_instance(new_config)
        
        # Recreate the graph with new LLM
        self._create_persistent_graph()
        
        print("✅ Graph recreated with new model configuration")
        return True
    
    async def process_message(self, message: str, session_id: str, model_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a chat message using the persistent LangGraph."""
        try:
            print(f"🔗 PERSISTENT LANGGRAPH - Processing message")
            print(f"   Session ID: {session_id}")
            print(f"   Message: {message[:100]}..." if len(message) > 100 else f"   Message: {message}")
            
            # Update model configuration if provided
            if model_config:
                graph_recreated = self.update_model_config(model_config)
                if graph_recreated:
                    print("   Graph was recreated due to model config change")
            
            # Search for relevant documents
            print(f"🔍 PERSISTENT LANGGRAPH - Searching documents")
            relevant_docs = await self.document_processor.search_documents(message, k=3)
            
            # Prepare context from retrieved documents
            context = self._format_context(relevant_docs)
            print(f"   Found {len(relevant_docs)} relevant documents")
            print(f"   Context length: {len(context)} characters")
            
            # Create thread configuration for this session
            config = {"configurable": {"thread_id": session_id}}
            
            # Prepare state for this message
            # The graph will automatically maintain conversation history via checkpointer
            new_state = {
                "messages": [HumanMessage(content=message)],
                "context": context,
                "session_id": session_id
            }
            
            # Time the API call
            import time
            start_time = time.time()
            
            print(f"🚀 PERSISTENT LANGGRAPH - Invoking persistent graph")
            
            # Invoke the persistent graph - this maintains all conversation history
            result = await asyncio.to_thread(
                self.graph.invoke,
                new_state,
                config
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"🎯 PERSISTENT LANGGRAPH - Graph invocation completed:")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Total messages after processing: {len(result.get('messages', []))}")
            
            # Extract the AI response (last message should be the AI response)
            messages = result.get("messages", [])
            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            
            if ai_messages:
                response = ai_messages[-1].content
            else:
                raise Exception("No AI response found in result")
            
            # Extract sources
            sources = [doc["metadata"].get("source", "Unknown") for doc in relevant_docs]
            unique_sources = list(set(sources))
            
            print(f"   Final response length: {len(response)} characters")
            print(f"   Sources: {unique_sources}")
            
            return {
                "response": response,
                "sources": unique_sources,
                "session_id": session_id
            }
            
        except Exception as e:
            print(f"❌ PERSISTENT LANGGRAPH - Error processing message: {str(e)}")
            raise Exception(f"Persistent LangGraph chat processing failed: {str(e)}")
    
    async def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get chat history for a session from persistent memory."""
        try:
            print(f"📜 PERSISTENT LANGGRAPH - Getting chat history for session: {session_id}")
            
            config = {"configurable": {"thread_id": session_id}}
            
            # Get the current state from persistent memory
            current_state = await asyncio.to_thread(
                self.graph.get_state,
                config
            )
            
            if not current_state or not current_state.values or not current_state.values.get("messages"):
                print("   No chat history found in persistent memory")
                return []
            
            messages = current_state.values["messages"]
            print(f"   Found {len(messages)} messages in persistent memory")
            
            formatted_messages = []
            for i, message in enumerate(messages):
                print(f"   Message {i+1}: {type(message).__name__} - {message.content[:50]}...")
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
            
            print(f"   Returning {len(formatted_messages)} formatted messages")
            return formatted_messages
            
        except Exception as e:
            print(f"❌ PERSISTENT LANGGRAPH - Error retrieving chat history: {str(e)}")
            return []
    
    async def clear_session(self, session_id: str):
        """Clear a chat session from persistent memory."""
        try:
            print(f"🧹 PERSISTENT LANGGRAPH - Clearing session: {session_id}")
            
            config = {"configurable": {"thread_id": session_id}}
            
            # Clear the thread by updating state to empty
            empty_state = {
                "messages": [],
                "context": "",
                "session_id": session_id
            }
            
            await asyncio.to_thread(
                self.graph.update_state,
                config,
                empty_state
            )
            
            print("✅ Session cleared from persistent memory")
            
        except Exception as e:
            print(f"❌ PERSISTENT LANGGRAPH - Error clearing session: {str(e)}")
            raise Exception(f"Failed to clear session: {str(e)}")
    
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
    
    def _create_llm_instance(self, config: Dict[str, Any]):
        """Create LLM instance from configuration."""
        print("🔧 PERSISTENT LANGGRAPH - Creating LLM instance")
        print(f"   Config: {config}")
        
        try:
            api_key = None
            if config["provider"] != "ollama":
                api_key = ModelFactory.get_api_key_for_provider(config["provider"])
            
            llm_instance = ModelFactory.create_llm(
                provider=config["provider"],
                model_name=config["model_name"],
                temperature=config.get("temperature", 0.7),
                base_url=config.get("base_url"),
                api_key=api_key,
                max_tokens=config.get("max_tokens")
            )
            
            print("✅ LLM instance created successfully")
            return llm_instance
            
        except Exception as e:
            print(f"❌ Failed to create LLM: {e}")
            print("🔄 Falling back to default Ollama configuration")
            
            return ModelFactory.create_llm(
                provider="ollama",
                model_name="llama3.1:8b",
                temperature=0.7,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs from persistent memory."""
        # Note: InMemorySaver doesn't provide direct access to all thread IDs
        # This would need to be tracked separately if needed
        return []
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current model configuration."""
        return self.current_config.copy() if self.current_config else {}

# Global singleton instance
chat_service_instance = None

def get_chat_service() -> PersistentLangGraphChatService:
    """Get the singleton chat service instance."""
    global chat_service_instance
    if chat_service_instance is None:
        chat_service_instance = PersistentLangGraphChatService()
    return chat_service_instance