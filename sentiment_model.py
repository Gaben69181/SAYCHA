"""
Sentiment Analysis Model Module

Handles sentiment classification using HuggingFace Transformers.
"""

import logging
from typing import Dict, List, Tuple, Any
import torch
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Handles sentiment analysis using pre-trained transformers models."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        
        try:
            logger.info(f"Loading model: {model_name}")
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=self.device
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.pipeline = None
    
    def classify_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Classify sentiment of a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment classification:
            {
                'label': 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL',
                'score': float (0-1),
                'raw_label': original model label,
                'raw_score': original model score
            }
        """
        if not text or not self.pipeline:
            return {
                'label': 'NEUTRAL',
                'score': 0.5,
                'raw_label': 'UNKNOWN',
                'raw_score': 0.0
            }
        
        try:
            # Get model prediction
            result = self.pipeline(text, truncation=True)[0]
            raw_label = result['label']
            raw_score = result['score']
            
            # Map model output to three-class classification
            sentiment = self._map_sentiment(raw_label, raw_score)
            
            return {
                'label': sentiment,
                'score': raw_score,
                'raw_label': raw_label,
                'raw_score': raw_score
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {
                'label': 'NEUTRAL',
                'score': 0.5,
                'raw_label': 'ERROR',
                'raw_score': 0.0
            }
    
    @staticmethod
    def _map_sentiment(label: str, score: float) -> str:
        """
        Map model output to three-class sentiment classification.
        
        Args:
            label: Original model label (POSITIVE/NEGATIVE)
            score: Model confidence score (0-1)
            
        Returns:
            Sentiment label: POSITIVE, NEGATIVE, or NEUTRAL
        """
        if label.upper() == 'POSITIVE':
            if score > 0.7:
                return 'POSITIVE'
            else:
                return 'NEUTRAL'
        else:  # NEGATIVE
            if score > 0.7:
                return 'NEGATIVE'
            else:
                return 'NEUTRAL'
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify sentiment for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of sentiment classifications
        """
        return [self.classify_sentiment(text) for text in texts]
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model': self.model_name,
            'device': 'GPU' if self.device == 0 else 'CPU'
        }
