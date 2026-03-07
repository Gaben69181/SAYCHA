# YouTube Live Chat Sentiment Analysis

A real-time sentiment analysis application for YouTube live chat messages using Python, transformers, and Streamlit.

## 📋 Project Overview

This project demonstrates a complete machine learning pipeline that:

1. Collects live chat messages from YouTube livestreams
2. Preprocesses text data (removing URLs, emojis, special characters)
3. Performs sentiment analysis using HuggingFace Transformers
4. Classifies messages as Positive, Negative, or Neutral
5. Displays real-time analytics in an interactive Streamlit dashboard

**Use Cases:**

- Monitor audience sentiment during live events
- Understand viewer reactions in real-time
- Analyze content engagement and reception
- Academic prototype for NLP/ML projects

---

## 🏗️ System Architecture

```
YouTube Live Chat Source
        ↓
Chat Collection Module (pytchat)
        ↓
Live Message Stream
        ↓
Text Preprocessing
  ├─ Remove URLs
  ├─ Remove Emojis
  ├─ Remove Special Characters
  ├─ Normalize Whitespace
  └─ Lowercase
        ↓
Sentiment Analysis Model
  (HuggingFace Transformers)
        ↓
Sentiment Classification
  ├─ POSITIVE (score > 0.7)
  ├─ NEUTRAL (0.4 ≤ score ≤ 0.7)
  └─ NEGATIVE (score < 0.4)
        ↓
Statistics Aggregation
  ├─ Count by sentiment
  ├─ Percentages
  └─ Confidence scores
        ↓
Visualization Dashboard (Streamlit)
  ├─ Sentiment counters
  ├─ Bar chart
  ├─ Pie chart
  ├─ Message feed
  └─ Statistics
```

---

## 📁 Project Structure

```
youtube-chat-sentiment/
├── app.py                 # Main Streamlit application
├── chat_collector.py      # YouTube live chat collection
├── preprocessing.py       # Text preprocessing module
├── sentiment_model.py     # Sentiment analysis with transformers
├── visualization.py       # Dashboard visualizations
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Module Descriptions

#### `app.py`

Main Streamlit application that orchestrates the entire pipeline. Features:

- Sidebar controls for video ID input and chat collection
- Three main tabs: Dashboard, Watch, Messages
- Real-time sentiment counters and statistics
- Interactive charts (bar and pie)
- Message feed display
- Data export functionality (CSV)

#### `chat_collector.py`

Handles YouTube live chat collection using pytchat library:

- `YouTubeChattCollector` class with connect/disconnect methods
- Stream-based message retrieval
- Error handling and logging
- Message format: `{author, message, timestamp}`

#### `preprocessing.py`

Text preprocessing pipeline:

- `TextPreprocessor` class with modular preprocessing steps
- URL removal
- Emoji removal
- Special character removal
- Whitespace normalization
- Lowercase conversion
- Batch processing support

#### `sentiment_model.py`

Sentiment analysis using pre-trained transformers:

- `SentimentAnalyzer` class using HuggingFace models
- GPU/CPU device management
- Three-class sentiment mapping (POSITIVE/NEGATIVE/NEUTRAL)
- Batch processing capability
- Confidence scores

#### `visualization.py`

Streamlit-based visualization components:

- Sentiment counters with icons
- Bar chart visualization
- Pie chart visualization
- Recent messages table
- Overall statistics display
- CSV export functionality

---

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Internet connection (for downloading models)
- YouTube livestream URL/Video ID

### Step 1: Clone or Download the Project

```bash
cd "/path/to/youtube-chat-sentiment"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:

- **pytchat** (1.5.7): YouTube live chat scraping
- **streamlit** (1.28.1): Interactive dashboard
- **transformers** (4.38.1): Pre-trained NLP models
- **torch** (2.1.2): PyTorch deep learning framework
- **pandas** (2.1.4): Data manipulation and analysis
- **matplotlib** (3.8.2): Data visualization
- **regex** (2023.12.25): Regular expression library

> Note: First download will take a few minutes as the transformer model is cached locally (~500MB).

---

## 🚀 How to Run

### Option 1: Local Development

```bash
# Ensure you're in the virtual environment
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Option 2: From Command Line

```bash
python -m streamlit run app.py
```

---

## 📖 Usage Guide

### Getting a YouTube Video ID

1. Open a YouTube livestream URL
2. Copy the video ID from the URL:
   ```
   https://www.youtube.com/watch?v=dQw4w9WgXcQ
                                    ^^^^^^^^^^^
                                    This is the Video ID
   ```

### Running the Application

1. **Open Streamlit App**
   - Run: `streamlit run app.py`
   - Browser opens automatically at `http://localhost:8501`

2. **Enter YouTube Video ID**
   - Paste the video ID in the sidebar input field
   - Example: `dQw4w9WgXcQ`

3. **Start Collecting Messages**
   - Click the "▶️ Start Collecting" button in the sidebar
   - Wait for connection confirmation (✅ or ❌)

4. **Monitor in Real-Time**
   - Watch sentiment counters update automatically
   - View live statistics on the Dashboard tab
   - Check the Watch tab for status updates
   - Browse individual messages on the Messages tab

5. **Export Analysis**
   - Click "📥 Download analyzed messages as CSV" to export results
   - Data includes: Author, Original Message, Cleaned Text, Sentiment, Confidence, Timestamp

---

## 💡 Example Output

### Input Message

```
"STREAMER INI LUCU BANGET!!! 😂😂 https://youtube.com"
```

### Processing Steps

1. **Preprocessing:**
   - Input: "STREAMER INI LUCU BANGET!!! 😂😂 https://youtube.com"
   - Output: "streamer ini lucu banget"

2. **Sentiment Analysis:**
   - Model Output: POSITIVE (confidence: 0.85)
   - Classification: POSITIVE (score > 0.7)

3. **Dashboard Update:**
   - Positive counter: +1
   - Statistics: Updated percentages
   - Charts: Regenerated visualizations

### Dashboard Metrics

```
Positive: 120    (54%)
Negative: 35     (16%)
Neutral:  50     (22%)
─────────────────────
Total:    205
```

---

## 🔬 Technical Details

### Sentiment Analysis Model

**Model Used:** `distilbert-base-uncased-finetuned-sst-2-english`

- **Type:** DistilBERT (Distilled BERT)
- **Task:** Binary sentiment classification (originally) → mapped to 3 classes
- **Dimension:** 768 hidden dimensions
- **Parameters:** ~66 million
- **Size:** ~250 MB
- **Speed:** ~100-500 messages/second (depending on hardware)
- **Accuracy:** 91% on SST-2 dataset

### Classification Logic

```python
Model Output: {label: "POSITIVE"/"NEGATIVE", score: 0.0-1.0}

Mapping to 3-class:
├─ POSITIVE (model) + score > 0.7 → POSITIVE
├─ POSITIVE (model) + score ≤ 0.7 → NEUTRAL
├─ NEGATIVE (model) + score > 0.7 → NEGATIVE
└─ NEGATIVE (model) + score ≤ 0.7 → NEUTRAL
```

### Preprocessing Pipeline

```python
Input: "STREAMER INI LUCU BANGET!!! 😂😂 https://youtube.com"
  ↓ [lowercase]
"streamer ini lucu banget!!! 😂😂 https://youtube.com"
  ↓ [remove_urls]
"streamer ini lucu banget!!! 😂😂"
  ↓ [remove_emojis]
"streamer ini lucu banget!!!"
  ↓ [remove_special_characters]
"streamer ini lucu banget"
  ↓ [remove_extra_whitespace]
"streamer ini lucu banget"
```

---

## 🔋 Performance Considerations

- **GPU Acceleration:** Automatically uses CUDA if available
- **Batch Processing:** Can process multiple messages simultaneously
- **Memory:** ~2-3 GB RAM for models + message buffer
- **Network:** Requires persistent connection to YouTube
- **Scalability:** Can analyze ~100-500 messages/second

---

## 🐛 Troubleshooting

### Issue: "Failed to connect to livestream"

**Solution:**

- Verify the YouTube video ID is correct
- Ensure the video is an active livestream (not a regular video)
- Check your internet connection

### Issue: "ModuleNotFoundError: No module named 'pytchat'"

**Solution:**

```bash
pip install pytchat==1.5.7
```

### Issue: "CUDA out of memory" (GPU users)

**Solution:**

- Use CPU instead (automatic fallback available)
- Reduce batch size in code

### Issue: "No messages appearing"

**Solution:**

- Verify the livestream is active and has chat enabled
- Check browser console for errors
- Try a different livestream video ID

### Issue: Slow sentiment analysis

**Solution:**

- First run downloads the model (~500MB) - this is normal
- Subsequent runs are much faster
- Ensure sufficient RAM (4GB+ recommended)

---

## 📊 Analysis Use Cases

### Real-Time Monitoring

- Track audience sentiment during live events
- Identify negative feedback immediately
- Monitor engagement levels

### Content Analysis

- Understand which topics trigger positive/negative responses
- Evaluate content quality through sentiment trends
- A/B test different content approaches

### Community Management

- Identify toxic comments early
- Respond to negative sentiment promptly
- Celebrate positive audience reactions

### Research

- Collect datasets for NLP research
- Study sentiment patterns in live streaming
- Analyze multilingual sentiment (with appropriate models)

---

## 🔒 Privacy & Ethics

- **Data Collection:** Only collects publicly available chat messages from livestreams
- **Data Storage:** Messages stored locally in session state (not persisted to disk by default)
- **Model Bias:** DistilBERT may have biases from training data - use with awareness
- **Usage:** Comply with YouTube's Terms of Service when using pytchat

---

## 🔄 Future Enhancements

- [ ] Multi-language support with multilingual models
- [ ] Real-time alerts for negative sentiment spikes
- [ ] User authentication and session persistence
- [ ] Database integration for historical analysis
- [ ] Advanced NLP features (topic modeling, emotion detection)
- [ ] Custom model fine-tuning
- [ ] API endpoint for integration with other services
- [ ] Mobile app support

---

## 📚 References

- **pytchat Documentation:** https://github.com/taizan-hokuto/pytchat
- **HuggingFace Transformers:** https://huggingface.co/docs/transformers
- **DistilBERT Model Card:** https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
- **Streamlit Documentation:** https://docs.streamlit.io
- **sentencepiece Tokenizer:** https://github.com/google/sentencepiece

---

## 📄 License

This project is provided as an educational prototype. Use freely for learning and academic purposes.

---

## 🤝 Contributing

Contributions welcome! Feel free to:

- Report bugs and issues
- Suggest new features
- Improve documentation
- Optimize code performance

---

## 📞 Support

For issues or questions:

1. Check the Troubleshooting section above
2. Review code comments in individual modules
3. Verify all dependencies are correctly installed
4. Check console logs for detailed error messages

---

## ✨ Highlights

- **Complete ML Pipeline:** Data collection → preprocessing → analysis → visualization
- **Production-Ready:** Error handling, logging, and robust design
- **User-Friendly:** Intuitive Streamlit interface with real-time updates
- **Scalable Architecture:** Modular design allows easy customization
- **Educational Value:** Well-documented code suitable for learning ML/NLP concepts

---

**Built with ❤️ for sentiment analysis and real-time insights**

---

**Last Updated:** March 2026  
**Python Version:** 3.10+  
**Status:** ✅ Production Ready
