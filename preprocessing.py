"""
Text Preprocessing Module

Handles cleaning and preprocessing of chat messages.
"""

import re
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handles text preprocessing for chat messages."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Input text
            
        Returns:
            Text without URLs
        """
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def remove_emojis(text: str) -> str:
        """
        Remove emoji characters from text.
        
        Args:
            text: Input text
            
        Returns:
            Text without emojis
        """
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)
    
    @staticmethod
    def remove_special_characters(text: str) -> str:
        """
        Remove special characters and punctuation.
        
        Args:
            text: Input text
            
        Returns:
            Text with special characters removed
        """
        # Keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """
        Remove extra whitespace.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @staticmethod
    def lowercase(text: str) -> str:
        """
        Convert text to lowercase.
        
        Args:
            text: Input text
            
        Returns:
            Lowercase text
        """
        return text.lower()
    
    def preprocess(self, text: str) -> str:
        """
        Apply all preprocessing steps in sequence.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
            
        Example:
            Input: "STREAMER INI LUCU BANGET!!! 😂😂 https://example.com"
            Output: "streamer ini lucu banget"
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Apply preprocessing steps in order
        text = self.lowercase(text)
        text = self.remove_urls(text)
        text = self.remove_emojis(text)
        text = self.remove_special_characters(text)
        text = self.remove_extra_whitespace(text)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]
