import os
import asyncio
from typing import List, Dict
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
# Environment variables accessed directly

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large:335m")
        
        print(f"📄 DOCUMENT PROCESSOR - Initializing embeddings:")
        print(f"   Ollama URL: {ollama_base_url}")
        print(f"   Embedding Model: {ollama_embedding_model}")
        
        self.embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=ollama_embedding_model
        )
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load existing Chroma vector store."""
        chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
        os.makedirs(chroma_persist_dir, exist_ok=True)
        
        print(f"📄 DOCUMENT PROCESSOR - Initializing vector store:")
        print(f"   Persist Directory: {chroma_persist_dir}")
        
        try:
            self.vector_store = Chroma(
                persist_directory=chroma_persist_dir,
                embedding_function=self.embeddings
            )
            print("✅ Vector store initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing vector store: {e}")
            print("🔄 Creating new vector store...")
            # Create new vector store if loading fails
            self.vector_store = Chroma(
                persist_directory=chroma_persist_dir,
                embedding_function=self.embeddings
            )
    
    async def process_document(self, file_path: str, original_filename: str) -> List[Document]:
        """Process a document and add it to the vector store."""
        try:
            # Load document based on file type
            documents = await self._load_document(file_path, original_filename)
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata to chunks
            for chunk in chunks:
                chunk.metadata.update({
                    "source": original_filename,
                    "file_path": file_path,
                    "chunk_id": str(uuid.uuid4())
                })
            
            # Add to vector store
            if chunks:
                await asyncio.to_thread(
                    self.vector_store.add_documents,
                    chunks
                )
                # Note: ChromaDB v1.0+ automatically persists data, no need to call persist()
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Document processing failed: {str(e)}")
    
    async def _load_document(self, file_path: str, original_filename: str) -> List[Document]:
        """Load document based on file extension."""
        file_extension = os.path.splitext(original_filename)[1].lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension in ['.pptx', '.ppt']:
                loader = UnstructuredPowerPointLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Load documents in a separate thread
            documents = await asyncio.to_thread(loader.load)
            return documents
            
        except Exception as e:
            raise Exception(f"Failed to load document {original_filename}: {str(e)}")
    
    async def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Perform semantic search for relevant documents."""
        try:
            if not self.vector_store:
                return []
            
            # Perform enhanced semantic search with relevance scoring
            # Using similarity_search_with_relevance_scores for better semantic understanding
            results = await asyncio.to_thread(
                self.vector_store.similarity_search_with_relevance_scores,
                query, k=k, score_threshold=0.1  # Filter out very low relevance results
            )
            
            # Format results with semantic relevance scores
            formatted_results = []
            for doc, relevance_score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": relevance_score,  # Higher score = more relevant
                    "semantic_similarity": 1.0 - relevance_score  # For backward compatibility
                })
            
            return formatted_results
            
        except AttributeError:
            # Fallback to standard similarity search if relevance_scores not available
            print("⚠️ Relevance scores not available, falling back to similarity search")
            results = await asyncio.to_thread(
                self.vector_store.similarity_search_with_score,
                query, k=k
            )
            
            formatted_results = []
            for doc, distance_score in results:
                # Convert distance to relevance (lower distance = higher relevance)
                relevance_score = max(0.0, 1.0 - distance_score)
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": relevance_score,
                    "semantic_similarity": distance_score
                })
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Semantic document search failed: {str(e)}")
    
    async def get_all_sources(self) -> List[str]:
        """Get list of all document sources in the vector store."""
        try:
            if not self.vector_store:
                return []
            
            # Get all documents to extract unique sources
            all_docs = await asyncio.to_thread(
                self.vector_store.similarity_search,
                "", k=1000  # Large number to get all docs
            )
            
            sources = set()
            for doc in all_docs:
                if "source" in doc.metadata:
                    sources.add(doc.metadata["source"])
            
            return list(sources)
            
        except Exception as e:
            return []
    
    async def clear_vector_store(self):
        """Clear all documents from the vector store."""
        try:
            print("🧹 DOCUMENT PROCESSOR - Clearing vector store")
            
            # Delete the persist directory
            import shutil
            chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
            if os.path.exists(chroma_persist_dir):
                shutil.rmtree(chroma_persist_dir)
                print(f"   Deleted directory: {chroma_persist_dir}")
            
            # Reinitialize vector store
            print("   Reinitializing vector store...")
            self._initialize_vector_store()
            print("✅ Vector store cleared successfully")
            
        except Exception as e:
            print(f"❌ Failed to clear vector store: {str(e)}")
            raise Exception(f"Failed to clear vector store: {str(e)}")
    
    def get_vector_store(self):
        """Get the vector store instance."""
        return self.vector_store