"""
YouTube Sentiment Analysis Dashboard

Main Streamlit application for real-time sentiment analysis of YouTube live chat messages.
"""

import streamlit as st
import logging
from typing import Dict, List
from chat_collector import YouTubeChattCollector
from preprocessing import TextPreprocessor
from sentiment_model import SentimentAnalyzer
from visualization import SentimentVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="YouTube Chat Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chat_collector' not in st.session_state:
        st.session_state.chat_collector = None
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = TextPreprocessor()
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SentimentAnalyzer()
    if 'sentiment_counts' not in st.session_state:
        st.session_state.sentiment_counts = {
            'POSITIVE': 0,
            'NEGATIVE': 0,
            'NEUTRAL': 0
        }
    if 'messages_analyzed' not in st.session_state:
        st.session_state.messages_analyzed = []
    if 'is_collecting' not in st.session_state:
        st.session_state.is_collecting = False
    if 'total_messages' not in st.session_state:
        st.session_state.total_messages = 0


def display_header():
    """Display the application header."""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(
            "<h1 style='text-align: center; color: #FF0000;'>📺 YouTube Live Chat Sentiment Analysis</h1>",
            unsafe_allow_html=True
        )
    
    st.markdown("---")


def display_sidebar():
    """Display the sidebar with input controls."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model information
        with st.expander("ℹ️ Model Info"):
            model_info = st.session_state.analyzer.get_model_info()
            st.json(model_info)
        
        st.markdown("---")
        
        # YouTube Video ID input
        st.subheader("📹 Livestream Details")
        video_id = st.text_input(
            "Enter YouTube Video ID:",
            placeholder="e.g., dQw4w9WgXcQ",
            help="The video ID from the YouTube URL (after 'v=')"
        )
        
        st.markdown("---")
        
        # Control buttons
        st.subheader("🎮 Controls")
        
        col_start, col_stop = st.columns(2)
        
        with col_start:
            start_button = st.button("▶️ Start Collecting", use_container_width=True)
        
        with col_stop:
            stop_button = st.button("⏹️ Stop Collecting", use_container_width=True)
        
        if start_button and video_id:
            with st.spinner("🔄 Connecting to livestream..."):
                st.session_state.is_collecting = True
                st.session_state.chat_collector = YouTubeChattCollector(video_id)
                
                try:
                    if st.session_state.chat_collector.connect():
                        st.success("✅ Connected to livestream!")
                    else:
                        st.error("❌ Failed to connect to livestream. Check if the video ID is correct and the livestream is active.")
                        st.session_state.is_collecting = False
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")
                    st.session_state.is_collecting = False
        
        if stop_button:
            st.session_state.is_collecting = False
            if st.session_state.chat_collector:
                st.session_state.chat_collector.disconnect()
            st.info("⏹️ Chat collection stopped.")
        
        st.markdown("---")
        
        # Display current status
        st.subheader("📊 Collection Status")
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            status_text = "🔴 Inactive"
            if st.session_state.is_collecting:
                status_text = "🟢 Active"
            st.metric("Status", status_text)
        
        with status_col2:
            st.metric("Messages Analyzed", st.session_state.total_messages)


def process_chat_stream():
    """Process incoming chat messages from the livestream."""
    if not st.session_state.is_collecting or not st.session_state.chat_collector:
        return
    
    try:
        # Create placeholders for real-time updates
        status_placeholder = st.empty()
        counter_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        message_count = 0
        
        for message in st.session_state.chat_collector.get_chat_stream():
            if not st.session_state.is_collecting:
                break
            
            # Preprocess the message
            raw_text = message.get('message', '')
            cleaned_text = st.session_state.preprocessor.preprocess(raw_text)
            
            # Skip empty messages
            if not cleaned_text:
                continue
            
            try:
                # Analyze sentiment
                result = st.session_state.analyzer.classify_sentiment(cleaned_text)
                sentiment = result['label']
                confidence = result['score']
                
                # Update counts
                st.session_state.sentiment_counts[sentiment] += 1
                st.session_state.total_messages += 1
                message_count += 1
                
                # Store message with sentiment
                st.session_state.messages_analyzed.append({
                    'author': message.get('author', 'Unknown'),
                    'original_message': raw_text,
                    'message': cleaned_text,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'timestamp': message.get('timestamp', '')
                })
                
                # Update status text
                status_placeholder.info(
                    f"🟢 **Active** | Processed: **{st.session_state.total_messages}** messages | "
                    f"Latest: {message.get('author', 'Unknown')} - **{sentiment}**"
                )
                
                # Update counters every 5 messages
                if message_count % 5 == 0:
                    with counter_placeholder.container():
                        SentimentVisualizer.display_sentiment_counters(st.session_state.sentiment_counts)
                    
                    with chart_placeholder.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            SentimentVisualizer.display_bar_chart(st.session_state.sentiment_counts)
                        with col2:
                            SentimentVisualizer.display_pie_chart(st.session_state.sentiment_counts)
            
            except Exception as msg_error:
                logger.error(f"Error analyzing message from {message.get('author', 'Unknown')}: {str(msg_error)}")
                continue
    
    except Exception as e:
        st.error(f"Error processing chat stream: {str(e)}")
        logger.error(f"Error processing chat stream: {str(e)}")


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    if st.session_state.total_messages > 0 or st.session_state.is_collecting:
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "👁️ Watch", "📝 Messages"])
        
        with tab1:
            st.subheader("📊 Sentiment Analysis Dashboard")
            
            # Display sentiment counters
            st.markdown("### Sentiment Counters")
            SentimentVisualizer.display_sentiment_counters(st.session_state.sentiment_counts)
            
            # Display statistics
            st.markdown("---")
            SentimentVisualizer.display_statistics(
                st.session_state.sentiment_counts,
                st.session_state.total_messages
            )
            
            # Display charts
            st.markdown("---")
            col_bar, col_pie = st.columns(2)
            
            with col_bar:
                st.markdown("### Bar Chart")
                SentimentVisualizer.display_bar_chart(st.session_state.sentiment_counts)
            
            with col_pie:
                st.markdown("### Pie Chart")
                SentimentVisualizer.display_pie_chart(st.session_state.sentiment_counts)
        
        with tab2:
            st.subheader("📺 Live Monitoring")
            
            if st.session_state.is_collecting:
                st.warning("� **Chat collection is ACTIVE** - Messages are being analyzed")
                
                # Add auto-refresh button
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Refresh Now"):
                        st.rerun()
                
                with col2:
                    st.metric("Messages Collected", st.session_state.total_messages)
                
                st.markdown("---")
                
                # Start processing - but only process new messages
                if st.session_state.chat_collector and st.session_state.chat_collector.chat:
                    try:
                        # Get new messages and process them
                        messages_before = st.session_state.total_messages
                        
                        for message in st.session_state.chat_collector.get_chat_stream():
                            if not st.session_state.is_collecting:
                                break
                            
                            # Preprocess the message
                            raw_text = message.get('message', '')
                            cleaned_text = st.session_state.preprocessor.preprocess(raw_text)
                            
                            # Skip empty messages
                            if not cleaned_text:
                                continue
                            
                            try:
                                # Analyze sentiment
                                result = st.session_state.analyzer.classify_sentiment(cleaned_text)
                                sentiment = result['label']
                                confidence = result['score']
                                
                                # Update counts
                                st.session_state.sentiment_counts[sentiment] += 1
                                st.session_state.total_messages += 1
                                
                                # Store message with sentiment
                                st.session_state.messages_analyzed.append({
                                    'author': message.get('author', 'Unknown'),
                                    'original_message': raw_text,
                                    'message': cleaned_text,
                                    'sentiment': sentiment,
                                    'confidence': confidence,
                                    'timestamp': message.get('timestamp', '')
                                })
                                
                                # Display processed message
                                sentiment_emoji = {'POSITIVE': '✅', 'NEGATIVE': '❌', 'NEUTRAL': '😐'}
                                emoji = sentiment_emoji.get(sentiment, '❓')
                                
                                st.success(
                                    f"{emoji} **{message.get('author', 'Unknown')}**: {raw_text[:100]}... → {sentiment} ({confidence:.1%})"
                                )
                                
                            except Exception as msg_error:
                                logger.error(f"Error analyzing message: {str(msg_error)}")
                                continue
                    
                    except Exception as stream_error:
                        logger.error(f"Chat stream error: {str(stream_error)}")
            else:
                st.info("⏹️ Chat collection is not active. Click 'Start Collecting' in the sidebar.")
        
        with tab3:
            st.subheader("📝 Analyzed Messages Feed")
            
            # Display recent messages
            SentimentVisualizer.display_recent_messages(
                st.session_state.messages_analyzed,
                max_display=20
            )
            
            # Option to download data
            if st.session_state.messages_analyzed:
                import pandas as pd
                df_export = pd.DataFrame(st.session_state.messages_analyzed)
                csv = df_export.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download analyzed messages as CSV",
                    data=csv,
                    file_name="youtube_sentiment_analysis.csv",
                    mime="text/csv"
                )
    
    else:
        # Welcome screen when no data
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ### 👋 Welcome to YouTube Chat Sentiment Analysis!
            
            This application analyzes the sentiment of live chat messages from YouTube livestreams.
            
            **How to use:**
            
            1. 📝 Enter a YouTube video ID in the sidebar (not the full URL, just the video ID)
            2. ▶️ Click "Start Collecting" to begin listening to the livestream chat
            3. 📊 Watch the sentiment analysis results update in real-time
            4. 👁️ View charts and statistics
            5. 📥 Download the analyzed data as CSV
            
            **Sentiment Categories:**
            - ✅ **Positive**: Messages expressing positive emotions
            - 😐 **Neutral**: Messages without clear sentiment
            - ❌ **Negative**: Messages expressing negative emotions
            
            ---
            
            **Example Video ID:** `dQw4w9WgXcQ`
            
            Get Started Now! 🚀
            """)


if __name__ == "__main__":
    main()
