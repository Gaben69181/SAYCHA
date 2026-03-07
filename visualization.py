"""
Visualization Module

Handles visualization of sentiment analysis results using Streamlit and Matplotlib.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentVisualizer:
    """Handles visualization of sentiment analysis results."""
    
    @staticmethod
    def display_sentiment_counters(sentiment_counts: Dict[str, int]):
        """
        Display sentiment counters in Streamlit.
        
        Args:
            sentiment_counts: Dictionary with sentiment counts
            {
                'POSITIVE': int,
                'NEGATIVE': int,
                'NEUTRAL': int
            }
        """
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="✅ Positive",
                value=sentiment_counts.get('POSITIVE', 0),
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                label="😐 Neutral",
                value=sentiment_counts.get('NEUTRAL', 0),
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                label="❌ Negative",
                value=sentiment_counts.get('NEGATIVE', 0),
                delta_color="inverse"
            )
    
    @staticmethod
    def display_bar_chart(sentiment_counts: Dict[str, int]):
        """
        Display bar chart of sentiment distribution.
        
        Args:
            sentiment_counts: Dictionary with sentiment counts
        """
        if not sentiment_counts or sum(sentiment_counts.values()) == 0:
            st.info("No data to display yet.")
            return
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        
        sentiments = list(sentiment_counts.keys())
        counts = list(sentiment_counts.values())
        colors = ['#00CC96', '#FFA15A', '#EF553B']  # Green, Orange, Red
        
        bars = ax.bar(sentiments, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold'
            )
        
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sentiment', fontsize=12, fontweight='bold')
        ax.set_title('Sentiment Distribution - Bar Chart', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    @staticmethod
    def display_pie_chart(sentiment_counts: Dict[str, int]):
        """
        Display pie chart of sentiment distribution.
        
        Args:
            sentiment_counts: Dictionary with sentiment counts
        """
        if not sentiment_counts or sum(sentiment_counts.values()) == 0:
            st.info("No data to display yet.")
            return
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        
        sentiments = list(sentiment_counts.keys())
        counts = list(sentiment_counts.values())
        colors = ['#00CC96', '#FFA15A', '#EF553B']  # Green, Orange, Red
        
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=sentiments,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        
        # Enhance percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax.set_title('Sentiment Distribution - Pie Chart', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    @staticmethod
    def display_recent_messages(messages: List[Dict[str, str]], max_display: int = 10):
        """
        Display recent analyzed messages in a table.
        
        Args:
            messages: List of message dictionaries with sentiment
            max_display: Maximum number of messages to display
        """
        if not messages:
            st.info("No messages analyzed yet.")
            return
        
        # Show recent messages
        recent_messages = messages[-max_display:]
        
        # Create DataFrame
        df_data = []
        for msg in recent_messages:
            df_data.append({
                'Author': msg.get('author', 'Unknown'),
                'Message': msg.get('message', ''),
                'Sentiment': msg.get('sentiment', 'UNKNOWN'),
                'Confidence': f"{msg.get('confidence', 0):.2%}"
            })
        
        df = pd.DataFrame(df_data)
        
        # Display with styling
        st.subheader("📝 Recent Messages")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
    
    @staticmethod
    def display_statistics(sentiment_counts: Dict[str, int], total_messages: int):
        """
        Display overall statistics.
        
        Args:
            sentiment_counts: Dictionary with sentiment counts
            total_messages: Total number of messages analyzed
        """
        st.subheader("📊 Overall Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        positive_pct = (sentiment_counts.get('POSITIVE', 0) / total_messages * 100) if total_messages > 0 else 0
        negative_pct = (sentiment_counts.get('NEGATIVE', 0) / total_messages * 100) if total_messages > 0 else 0
        neutral_pct = (sentiment_counts.get('NEUTRAL', 0) / total_messages * 100) if total_messages > 0 else 0
        
        with col1:
            st.metric("Total Messages", total_messages)
        
        with col2:
            st.metric("Positive %", f"{positive_pct:.1f}%")
        
        with col3:
            st.metric("Negative %", f"{negative_pct:.1f}%")
        
        with col4:
            st.metric("Neutral %", f"{neutral_pct:.1f}%")
