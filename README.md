# FinMedia Sentiment Processing

Real-time financial news sentiment analysis with vector database storage.

## What it does

1. **Fetches** RSS feeds for financial news
2. **Processes** text and extracts financial assets 
3. **Analyzes** sentiment using FinBERT ML model
4. **Calculates** trading signals with technical analysis
5. **Stores** results in Qdrant vector database

## Quick Start

```bash
# Dowload FinBERT
cd inference/FinBERT
python3 -m venv venv
source venv/bin/activate
(venv) pip install -r requirements.txt
(venv) python3 quantize_FinBERT.py
(venv) python3 test_setup.py

# Setup environment
cp .env.example .env
# Edit .env with your Qdrant cloud credentials

# Run pipeline
cd orchestrator
cargo run --release
```

## Requirements

- Rust 1.70+
- FinBERT ONNX model
- Qdrant cloud instance
- RSS feed configuration

## Pipeline Flow

```
RSS Feeds → Text Processing → FinBERT Analysis → Signals Processing → Vector Storage
```

## Key Components

- **orchestrator/**: Main pipeline coordinator
- **preprocessing/**: Text processing and asset extraction
- **vdatabase/**: Vector database integration with Qdrant
- **finmedia-signals/**: Technical analysis and scoring
- **inference/**: FinBERT sentiment analysis

## Configuration

Edit `.env` file:
```
QDRANT_URL=your-cloud-url
QDRANT_API_KEY=your-api-key
FINBERT_MODEL_PATH=path/to/model.onnx
```

## Output

The pipeline produces unified events with:
- Multi-layer sentiment analysis
- Asset detection and confidence scores
- Trading signals and risk assessment
- Vector embeddings for similarity search

Each event is stored in the vector database for retrieval and analysis.

## Future Development

A web frontend and backend API will be developed for:
- Visual dashboards and analytics
- Asset-based news querying and filtering  
- Real-time sentiment monitoring
- Historical trend analysis
- Interactive vector similarity search