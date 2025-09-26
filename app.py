from flask import Flask, render_template_string, jsonify
import os
import sys

app = Flask(__name__)

# Lightweight Flask app for Vercel deployment
# This is a landing page only - the full app runs on Streamlit Cloud

# HTML template for the landing page
LANDING_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Health News Analysis Platform</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .hero {
            text-align: center;
            padding: 80px 0;
        }
        
        .hero h1 {
            font-size: 3.5em;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .hero .subtitle {
            font-size: 1.3em;
            margin-bottom: 40px;
            opacity: 0.9;
        }
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin: 60px 0;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
        }
        
        .feature-card h3 {
            font-size: 1.5em;
            margin-bottom: 15px;
            color: #FFD700;
        }
        
        .feature-card ul {
            list-style: none;
        }
        
        .feature-card li {
            margin: 8px 0;
            padding-left: 20px;
            position: relative;
        }
        
        .feature-card li:before {
            content: "✨";
            position: absolute;
            left: 0;
        }
        
        .buttons {
            text-align: center;
            margin: 40px 0;
        }
        
        .btn {
            display: inline-block;
            padding: 15px 30px;
            margin: 10px;
            background: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 50px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            background: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .btn.secondary {
            background: #2196F3;
        }
        
        .btn.secondary:hover {
            background: #1976D2;
        }
        
        .tech-stack {
            background: rgba(255, 255, 255, 0.05);
            padding: 40px;
            border-radius: 15px;
            margin: 40px 0;
        }
        
        .tech-stack h3 {
            text-align: center;
            margin-bottom: 30px;
            color: #FFD700;
        }
        
        .tech-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .tech-item {
            text-align: center;
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        
        .footer {
            text-align: center;
            margin: 60px 0 20px;
            opacity: 0.8;
        }
        
        @media (max-width: 768px) {
            .hero h1 {
                font-size: 2.5em;
            }
            
            .features {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="hero">
            <h1>🏥 Advanced Health News Analysis Platform</h1>
            <div class="subtitle">AI-Powered Healthcare Content Analysis with Machine Learning</div>
        </div>
        
        <div class="features">
            <div class="feature-card">
                <h3>🤖 AI-Powered Analysis</h3>
                <ul>
                    <li>PEGASUS model for abstractive summarization</li>
                    <li>BART-large for zero-shot classification</li>
                    <li>DialoGPT for neural chat interface</li>
                    <li>SentenceTransformer for semantic embeddings</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h3>📊 Comprehensive Metrics</h3>
                <ul>
                    <li>29 evaluation metrics including ROUGE scores</li>
                    <li>Semantic similarity analysis</li>
                    <li>Readability and lexical diversity</li>
                    <li>Factual consistency assessment</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h3>🔬 ML Pipeline Visualization</h3>
                <ul>
                    <li>9-stage preprocessing pipeline</li>
                    <li>Interactive Plotly visualizations</li>
                    <li>Real-time processing insights</li>
                    <li>Statistical analysis dashboard</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h3>🏆 Model Comparison</h3>
                <ul>
                    <li>Compare 8 different summarization models</li>
                    <li>Performance benchmarking</li>
                    <li>Quality vs efficiency analysis</li>
                    <li>Sample output comparison</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h3>🌐 Multi-Input Support</h3>
                <ul>
                    <li>URL content extraction with BeautifulSoup</li>
                    <li>File upload processing</li>
                    <li>Direct text input analysis</li>
                    <li>Batch processing capabilities</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h3>💬 Interactive Features</h3>
                <ul>
                    <li>Neural chat for health queries</li>
                    <li>Semantic search across documents</li>
                    <li>Real-time similarity scoring</li>
                    <li>Intelligent response generation</li>
                </ul>
            </div>
        </div>
        
        <div class="buttons">
            <a href="https://github.com/vedanshd/health-news-analysis-platform" class="btn" target="_blank">
                📚 View Source Code
            </a>
            <a href="/demo" class="btn secondary">
                🚀 Setup Instructions
            </a>
        </div>
        
        <div class="tech-stack">
            <h3>🛠️ Technology Stack</h3>
            <div class="tech-grid">
                <div class="tech-item">
                    <strong>Frontend</strong><br>
                    Streamlit, Plotly, HTML/CSS
                </div>
                <div class="tech-item">
                    <strong>ML Models</strong><br>
                    PEGASUS, BART, DialoGPT, T5
                </div>
                <div class="tech-item">
                    <strong>NLP Libraries</strong><br>
                    Transformers, NLTK, scikit-learn
                </div>
                <div class="tech-item">
                    <strong>Data Processing</strong><br>
                    Pandas, NumPy, BeautifulSoup
                </div>
                <div class="tech-item">
                    <strong>Evaluation</strong><br>
                    ROUGE-score, Semantic similarity
                </div>
                <div class="tech-item">
                    <strong>Deployment</strong><br>
                    Vercel, Streamlit Cloud, Docker
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🧬 <strong>Advanced NLP Analysis Platform</strong> | Powered by Transformer Models & Statistical ML Algorithms</p>
            <p>Created by <a href="https://github.com/vedanshd" style="color: #FFD700;">Vedansh Dhawan</a> | 2025</p>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    """Landing page for the Health News Analysis Platform"""
    return render_template_string(LANDING_PAGE)

@app.route('/demo')
def demo():
    """Demo setup instructions"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Demo Setup - Health News Analysis Platform</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                line-height: 1.6;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                padding: 40px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            .btn {
                display: inline-block;
                padding: 10px 20px;
                background: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 25px;
                margin: 10px 0;
            }
            .code {
                background: rgba(0,0,0,0.3);
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                font-family: monospace;
                overflow-x: auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Run the Full Application</h1>
            
            <h2>Option 1: Local Setup (Recommended)</h2>
            <div class="code">
                git clone https://github.com/vedanshd/health-news-analysis-platform.git<br>
                cd health-news-analysis-platform<br>
                pip install -r requirements.txt<br>
                cp .env.example .env<br>
                # Add your Hugging Face token to .env<br>
                streamlit run test1.py
            </div>
            
            <h2>Option 2: Streamlit Cloud</h2>
            <p>Deploy to Streamlit Cloud for free hosting:</p>
            <ol>
                <li>Fork the GitHub repository</li>
                <li>Go to <a href="https://share.streamlit.io" style="color: #FFD700;">share.streamlit.io</a></li>
                <li>Connect your GitHub account</li>
                <li>Deploy using <code>test1.py</code> as main file</li>
                <li>Add your Hugging Face token in secrets</li>
            </ol>
            
            <h2>🔑 Get Hugging Face Token</h2>
            <p>Visit <a href="https://huggingface.co/settings/tokens" style="color: #FFD700;">huggingface.co/settings/tokens</a> to get your free API token.</p>
            
            <a href="/" class="btn">← Back to Home</a>
            <a href="https://github.com/vedanshd/health-news-analysis-platform" class="btn">View on GitHub</a>
        </div>
    </body>
    </html>
    """)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "app": "Advanced Health News Analysis Platform",
        "version": "1.0.0",
        "github": "https://github.com/vedanshd/health-news-analysis-platform"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)