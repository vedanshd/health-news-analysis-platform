# 🏥 Advanced Health News Analysis Platform

A comprehensive AI-powered platform for analyzing health news articles with advanced NLP capabilities, featuring text preprocessing visualization, evaluation metrics, intelligent chat, and model performance comparison.

## 🚀 Features

### 📊 Core Functionality
- **URL Content Extraction**: Extract and analyze content from health news URLs
- **File Upload Support**: Process multiple text files simultaneously  
- **Direct Text Input**: Analyze pasted health articles
- **AI-Powered Summarization**: Generate intelligent abstracts using PEGASUS model
- **Topic Classification**: Automatic categorization into health domains

### 🔬 Advanced Analysis
- **9-Stage ML Preprocessing Pipeline**: Visualize text transformation through normalization, cleaning, tokenization, filtering, stemming, n-gram extraction, TF-IDF vectorization, embedding generation, and statistical analysis
- **29 Comprehensive Evaluation Metrics**: ROUGE scores, semantic similarity, readability analysis, lexical diversity, compression ratios, and more
- **Interactive Visualizations**: Radar charts, bar graphs, word clouds, and statistical plots
- **Model Performance Comparison**: Compare PEGASUS with 7 other state-of-the-art summarization models

### 💬 Intelligent Features  
- **Neural Chat Interface**: GPT-powered conversational AI for health queries
- **Semantic Search**: Find relevant articles using natural language queries
- **Quantitative Dashboard**: Statistical analysis with interactive charts and metrics
- **Real-time Processing**: Live text analysis and evaluation

## 🏆 Model Comparison Dashboard

Compare PEGASUS with industry-leading models:

### Abstractive Models
- **PEGASUS-CNN-DailyMail** (Current) - 568M parameters
- **BART-Large-CNN** - 406M parameters  
- **T5-Base** - 220M parameters
- **DistilBART-CNN** - 306M parameters
- **LED-Base** - 162M parameters (long document specialist)
- **ProphetNet-Large** - 340M parameters

### Extractive Models
- **TextRank-Extractive** - Graph-based algorithm
- **BERT-Extractive** - 340M parameters with classification head

### Performance Metrics
- **ROUGE Scores**: Industry-standard summarization evaluation
- **Quality Analysis**: Factual consistency, fluency, coherence assessment
- **Efficiency Metrics**: Inference speed vs performance trade-offs
- **Sample Output Comparison**: Side-by-side summary examples

## 🛠️ Technical Stack

### Core Models
- **Summarization**: PEGASUS-CNN-DailyMail (Hugging Face)
- **Classification**: BART-large-mnli (Zero-shot classification)
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)
- **Chat**: DialoGPT-large (Conversational AI)

### Libraries & Frameworks
- **Streamlit**: Web application framework
- **NLTK**: Natural language processing toolkit
- **Scikit-learn**: Machine learning algorithms
- **BeautifulSoup**: Web scraping and HTML parsing
- **Plotly**: Interactive data visualizations
- **ROUGE-score**: Summarization evaluation metrics

### Analysis Features
- **Text Preprocessing**: NLTK tokenization, stopword filtering, stemming
- **Feature Engineering**: TF-IDF vectorization, n-gram analysis
- **Statistical Analysis**: Readability scores, lexical diversity
- **Semantic Analysis**: Cosine similarity, embedding-based search

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Hugging Face account (for API access)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/vedanshd/health-news-analysis-platform.git
cd health-news-analysis-platform
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup environment variables**

For local development:
```bash
cp .env.example .env
# Edit .env and add your Hugging Face token
```

For Streamlit Cloud deployment:
- Add your tokens in the Streamlit Cloud secrets management interface
- Use the same key names as in `.env.example`

Get your Hugging Face token from: https://huggingface.co/settings/tokens

5. **Download NLTK data**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 🎯 Quick Start

#### Local Development
```bash
streamlit run test1.py
```
Navigate to `http://localhost:8501` in your browser.

#### Streamlit Cloud Deployment (Recommended)
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy `test1.py` as the main file
5. Add your Hugging Face token in the Secrets section:
   ```toml
   HF_TOKEN = "your_token_here"
   ```

## 📖 Usage Guide

### Basic Analysis
1. **Input Method**: Choose from URL extraction, file upload, or direct text input
2. **Processing Options**: Configure preprocessing steps and summary length
3. **Analysis**: Click "🚀 Analyze Content" to process articles

### Advanced Features
- **Preprocessing Visualization**: Enable "Show Preprocessing Steps" to see the 9-stage ML pipeline
- **Evaluation Metrics**: View comprehensive text evaluation including ROUGE scores and semantic analysis  
- **Model Comparison**: Enable "Show Model Comparison" to benchmark against other models
- **Interactive Chat**: Ask questions about processed articles using the neural chat interface

### Model Comparison Analysis
1. Select an article for comparison
2. Explore four analysis tabs:
   - **Performance Metrics**: ROUGE scores and efficiency charts
   - **Quality Analysis**: Radar charts and model strengths/weaknesses
   - **Sample Outputs**: See how different models would summarize the same content
   - **Model Selection**: Get recommendations based on your use case requirements

## 📊 Evaluation Metrics

### Core Metrics
- **ROUGE-1/2/L**: N-gram overlap and longest common subsequence
- **Semantic Similarity**: Cosine similarity in embedding space
- **Compression Ratio**: Length reduction analysis
- **Readability Scores**: Flesch-Kincaid grade level assessment

### Advanced Analysis
- **Lexical Diversity**: Type-token ratios and vocabulary richness
- **Factual Consistency**: Content preservation evaluation  
- **Fluency & Coherence**: Language quality assessment
- **Novelty Scores**: Abstractiveness measurement

## 🔧 Configuration

### Model Settings
- Adjust summary length (30-200 tokens)
- Configure preprocessing pipeline stages
- Select visualization options
- Enable/disable analysis components

### Performance Optimization
- Use GPU acceleration for faster inference
- Adjust batch sizes for memory management
- Configure caching for repeated analyses

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for pre-trained transformer models
- Google Research for PEGASUS architecture
- Facebook AI for BART model
- Streamlit team for the web framework
- NLTK contributors for text processing tools

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the wiki for detailed guides
- **Community**: Join discussions in GitHub Discussions

---

**🧬 Advanced NLP Analysis Platform** | Powered by Transformer Models & Statistical ML Algorithms