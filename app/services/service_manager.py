from typing import Dict, Any, Optional
from app.services.config_service import ConfigService
from app.services.chat_service import ChatService
from app.services.quiz_service import QuizService
from app.services.flashcard_service import FlashCardService
from app.services.summary_service import SummaryService


class ServiceManager:
    """Singleton service manager to handle service initialization and configuration updates."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.config_service = ConfigService()
            self.chat_service = None
            self.quiz_service = None
            self.flashcard_service = None
            self.summary_service = None
            self._initialized = True
    
    async def initialize_services(self):
        """Initialize all services with current configuration."""
        if self.chat_service is None:
            # Load current configuration
            current_config = await self.config_service.get_current_config()
            
            # Initialize services with configuration
            self.chat_service = ChatService(initial_config=current_config.get("chat_model"))
            self.quiz_service = QuizService(initial_config=current_config.get("quiz_model"))
            self.flashcard_service = FlashCardService(initial_config=current_config.get("flashcard_model"))
            self.summary_service = SummaryService(model_config=current_config.get("summary_model"))
            
            # Set service references in config service
            self.config_service.set_service_references(
                chat_service=self.chat_service,
                quiz_service=self.quiz_service,
                flashcard_service=self.flashcard_service,
                summary_service=self.summary_service
            )
            
            print("All services initialized with current configuration")
    
    async def get_chat_service(self) -> ChatService:
        """Get the chat service instance, initializing if needed."""
        if self.chat_service is None:
            await self.initialize_services()
        return self.chat_service
    
    async def get_quiz_service(self) -> QuizService:
        """Get the quiz service instance, initializing if needed."""
        if self.quiz_service is None:
            await self.initialize_services()
        return self.quiz_service
    
    async def get_flashcard_service(self) -> FlashCardService:
        """Get the flashcard service instance, initializing if needed."""
        if self.flashcard_service is None:
            await self.initialize_services()
        return self.flashcard_service
    
    async def get_summary_service(self) -> SummaryService:
        """Get the summary service instance, initializing if needed."""
        if self.summary_service is None:
            await self.initialize_services()
        return self.summary_service
    
    def get_config_service(self) -> ConfigService:
        """Get the config service instance."""
        return self.config_service
    
    async def update_all_services(self):
        """Update all services with current configuration."""
        await self.config_service._update_service_models()


# Global service manager instance
service_manager = ServiceManager()

def get_service_manager() -> ServiceManager:
    """Get the global service manager instance."""
    return service_manager