"""
YouTube Live Chat Collector Module

Collects live chat messages from YouTube livestreams using pytchat.
"""

import pytchat
from datetime import datetime
from typing import Generator, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeChattCollector:
    """Handles collection of live chat messages from YouTube livestreams."""
    
    def __init__(self, video_id: str):
        """
        Initialize the chat collector.
        
        Args:
            video_id: YouTube video ID of the livestream
        """
        self.video_id = video_id
        self.chat = None
        
    def connect(self) -> bool:
        """
        Connect to the YouTube livestream chat.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.chat = pytchat.create(video_id=self.video_id, interruptable=False)
            logger.info(f"Successfully connected to livestream {self.video_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to livestream: {str(e)}")
            return False
    
    def get_chat_stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yield chat messages as they arrive from the livestream.
        
        Uses pytchat's sync_items() method which is thread-safe.
        
        Yields:
            Dictionary containing:
            - author: Username of the message sender
            - message: The chat message content
            - timestamp: When the message was posted
        """
        if not self.chat:
            logger.error("Not connected to chat. Call connect() first.")
            return
        
        try:
            while self.chat.is_alive():
                for c in self.chat.get().sync_items():
                    yield {
                        'author': c.author.name,
                        'message': c.message,
                        'timestamp': datetime.now().isoformat()
                    }
        except Exception as e:
            logger.error(f"Error in chat stream: {str(e)}")
            return
    
    def disconnect(self):
        """Clean up connection resources."""
        if self.chat:
            try:
                self.chat.terminate()
                logger.info("Disconnected from chat")
            except Exception as e:
                logger.error(f"Error disconnecting: {str(e)}")
