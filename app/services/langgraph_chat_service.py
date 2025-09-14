import asyncio
import os
from typing import List, Dict, Any, Annotated
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

class LangGraphChatService:
    def __init__(self, initial_config: Dict[str, Any] = None):
        print("🔗 LANGGRAPH CHAT SERVICE - Initializing")
        
        self.document_processor = DocumentProcessor()
        
        # Load initial configuration or use defaults
        if initial_config:
            print("💬 LANGGRAPH CHAT - Received custom config:", initial_config)
            config = initial_config
        else:
            default_config = {
                "provider": "ollama",
                "model_name": "llama3.2",
                "temperature": 0.7,
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            }
            print("💬 LANGGRAPH CHAT - Using default config:", default_config)
            config = default_config
        
        # Create LLM instance using factory
        self.current_config = config
        self.llm = self._create_llm_instance(config)
        self._initialize_prompt()
        
        # Initialize LangGraph
        self._initialize_graph()
        
        print("✅ LangGraph Chat Service initialized successfully")
    
    def _initialize_prompt(self):
        """Initialize the chat prompt template."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context from uploaded documents. 

You have access to the full conversation history and should remember previous questions and answers in this conversation.

IMPORTANT: Provide only the final answer. Do not include any thinking process, reasoning steps, or tags like <think>. Give a direct, clear answer to the user's question.

If the user asks about previous questions or refers to earlier parts of the conversation, use the conversation history to provide accurate context.

Use the following context to answer the user's question. If the question cannot be answered using the provided context, say so clearly.

Context:
{context}

Provide accurate, helpful responses based on the context and conversation history. If you reference specific information, mention which document it came from when possible. Answer directly without showing your thought process."""),
            MessagesPlaceholder("messages"),
        ])
    
    def _initialize_graph(self):
        """Initialize the LangGraph state graph."""
        print("🕸️ LANGGRAPH CHAT - Setting up state graph")
        
        # Create state graph
        graph_builder = StateGraph(State)
        
        # Add chatbot node
        def chatbot(state: State):
            print(f"🤖 LANGGRAPH - Processing message in chatbot node")
            print(f"   Session: {state.get('session_id', 'unknown')}")
            print(f"   Messages count: {len(state['messages'])}")
            print(f"   Context length: {len(state.get('context', ''))} characters")
            
            # Create chain with context and messages
            chain = (
                {
                    "context": lambda x: state.get("context", ""),
                    "messages": lambda x: state["messages"]
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Generate response
            response = chain.invoke({})
            
            print(f"   Generated response length: {len(response)} characters")
            print(f"   Response preview: {response[:150]}..." if len(response) > 150 else f"   Response: {response}")
            
            return {"messages": [AIMessage(content=response)]}
        
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.set_entry_point("chatbot")
        graph_builder.set_finish_point("chatbot")
        
        # Create memory checkpointer
        memory = InMemorySaver()
        
        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=memory)
        
        print("✅ LangGraph state graph initialized successfully")
    
    async def process_message(self, message: str, session_id: str, model_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a chat message using LangGraph and return the response."""
        try:
            print(f"🔗 LANGGRAPH CHAT - Processing message")
            print(f"   Session ID: {session_id}")
            print(f"   Message: {message[:100]}..." if len(message) > 100 else f"   Message: {message}")
            
            # Update model configuration if provided
            if model_config:
                print(f"🔧 LANGGRAPH CHAT - Updating model config:")
                print(f"   Provider: {model_config.get('provider')}")
                print(f"   Model: {model_config.get('model_name')}")
                
                self.current_config = model_config
                self.llm = self._create_llm_instance(model_config)
                # Reinitialize graph with new model
                self._initialize_graph()
            
            # Search for relevant documents
            print(f"🔍 LANGGRAPH CHAT - Searching documents")
            relevant_docs = await self.document_processor.search_documents(message, k=2)
            
            # Prepare context from retrieved documents
            context = self._format_context(relevant_docs)
            print(f"   Found {len(relevant_docs)} relevant documents")
            print(f"   Context length: {len(context)} characters")
            
            # Create thread configuration for this session
            config = {"configurable": {"thread_id": session_id}}
            
            # Get existing conversation state to preserve history
            try:
                current_state = await asyncio.to_thread(
                    self.graph.get_state,
                    config
                )
                existing_messages = current_state.values.get("messages", []) if current_state.values else []
                print(f"📜 Found {len(existing_messages)} existing messages in conversation")
            except Exception as e:
                print(f"⚠️ Could not retrieve existing state: {e}")
                existing_messages = []
            
            # Prepare new state with conversation history + new message
            new_state = {
                "messages": existing_messages + [HumanMessage(content=message)],
                "context": context,
                "session_id": session_id
            }
            
            print(f"💬 Total messages in conversation: {len(new_state['messages'])}")
            
            # Time the API call
            import time
            start_time = time.time()
            
            print(f"🚀 LANGGRAPH CHAT - Invoking graph")
            
            # Invoke the graph with full conversation history
            result = await asyncio.to_thread(
                self.graph.invoke,
                new_state,
                config
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"🎯 LANGGRAPH CHAT - Graph invocation completed:")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Messages after processing: {len(result['messages'])}")
            
            # Extract the response (last message should be the AI response)
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
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
            print(f"❌ LANGGRAPH CHAT - Error processing message: {str(e)}")
            raise Exception(f"LangGraph chat processing failed: {str(e)}")
    
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
        """Get chat history for a session from LangGraph memory."""
        try:
            print(f"📜 LANGGRAPH CHAT - Getting chat history for session: {session_id}")
            
            config = {"configurable": {"thread_id": session_id}}
            
            # Get the current state to access message history
            current_state = await asyncio.to_thread(
                self.graph.get_state,
                config
            )
            
            if not current_state or not current_state.values or not current_state.values.get("messages"):
                print("   No chat history found")
                return []
            
            messages = current_state.values["messages"]
            print(f"   Found {len(messages)} messages in history")
            
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
            print(f"❌ LANGGRAPH CHAT - Error retrieving chat history: {str(e)}")
            return []
    
    async def clear_session(self, session_id: str):
        """Clear a chat session from LangGraph memory."""
        try:
            print(f"🧹 LANGGRAPH CHAT - Clearing session: {session_id}")
            
            config = {"configurable": {"thread_id": session_id}}
            
            # Clear the thread by creating a new empty state
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
            
            print("✅ Session cleared successfully")
            
        except Exception as e:
            print(f"❌ LANGGRAPH CHAT - Error clearing session: {str(e)}")
            raise Exception(f"Failed to clear session: {str(e)}")
    
    def _create_llm_instance(self, config: Dict[str, Any]):
        """Create LLM instance from configuration."""
        print("💬 LANGGRAPH CHAT - Creating LLM instance with config:", config)
        
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
            
            print("✅ LLM instance created successfully in LangGraph ChatService")
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
            
            print(f"🔧 LANGGRAPH CHAT - Updating model config to {provider}/{model_name}")
            
            self.current_config = config
            self.llm = self._create_llm_instance(config)
            # Reinitialize graph with new model
            self._initialize_graph()
            
            print(f"✅ LangGraph chat service updated successfully")
                
        except Exception as e:
            print(f"❌ Failed to update model config: {str(e)}")
            raise Exception(f"Failed to update model config: {str(e)}")
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        # Note: InMemorySaver doesn't provide direct access to all thread IDs
        # This would need to be tracked separately if needed
        return []