# FinMedia Orchestrator

Central orchestrator for the FinMedia sentiment analysis pipeline that coordinates all services and components.

## Architecture

```
Go RSS Service → Preprocessing (gRPC) → Orchestrator → [Inference + Signals] → Terminal Output
```

## Components Integrated

- **Go RSS Ingestion** - Fetches RSS feeds and detects financial assets
- **Rust Preprocessing** - gRPC server for text processing and tokenization  
- **ONNX Inference** - FinBERT sentiment analysis using ONNX runtime
- **Signals Processing** - Mathematical transforms and scoring algorithms

## Usage

### Prerequisites

1. **Go service binary built:**
   ```bash
   cd ../finmedia
   go build -o bin/finmedia cmd/ingest/main.go
   ```

2. **FinBERT model available:**
   ```bash
   # Ensure FinBERT model exists at:
   # ../inference/FinBERT/finbert_config.json
   # ../inference/FinBERT/FinBERT_tone_int8.onnx
   ```

### Run Commands

```bash
# Build the orchestrator
cargo build

# Single pipeline run with REAL RSS data (recommended)
cargo run -- --once

# Continuous mode with REAL RSS data
cargo run

# With custom config paths
cargo run -- --config ../config.json --finbert-config ../inference/FinBERT/finbert_config.json

# Debug mode (shows detailed logs)
RUST_LOG=debug cargo run -- --once
```

### Command Line Options

- `--once` - Run pipeline once and exit (default: continuous mode)
- `--config` - Path to config.json file (default: ../config.json)  
- `--finbert-config` - Path to FinBERT config (default: ../inference/FinBERT/finbert_config.json)
- `--preprocessing-addr` - Preprocessing service address (default: http://127.0.0.1:50051)
- `--go-binary` - Go service binary path (default: ../finmedia/bin/finmedia)
- `--feed` - Specific RSS feed URL to process

## Output Format

The orchestrator processes **real RSS feed data** and displays comprehensive analysis results:

```
================================================================================
🔍 FINMEDIA ANALYSIS RESULT #1
================================================================================
📰 Event ID: reuters-1735689600          ← REAL RSS EVENT ID
🕐 Processed: 2025-01-08T10:30:00Z
📝 Text: Federal Reserve signals potential rate cuts as inflation moderates... ← REAL RSS CONTENT

🏷️  Categories: monetary, economic

----------------------------------------
📊 SENTIMENT ANALYSIS  
----------------------------------------
🔵 Go Service:
   Sentiment: -0.120 📉                   ← REAL ASSET DETECTION
   Confidence: 0.850

🤖 ML Inference (FinBERT):
   Sentiment: -0.089 📉                   ← REAL ML ANALYSIS 
   Confidence: 0.923

📡 Signals Processing:
   Final Score: -0.095 📉                ← REAL SIGNALS PROCESSING

💹 Market Impact: MEDIUM ⚡

🎯 Detected Assets:                      ← REAL FINANCIAL ASSETS
   • USD (monetary) - 89.50% confidence
   • FEDERAL_RESERVE (economic) - 95.20% confidence
```

**Key Difference**: All data comes from live RSS feeds, not mock/simulated events!

## Integration Details

### Preprocessing Service Management
- **Automatically builds and starts** preprocessing service
- Waits for service to be ready with retry logic
- Health checks and gRPC connection management  
- **Automatic cleanup** when orchestrator exits

### Go Service Management
- Automatically builds Go binary if missing
- Manages Go service lifecycle (start/stop)
- Handles graceful shutdown with SIGTERM

### gRPC Communication  
- Connects to preprocessing service on port 50051
- Health checks and retry logic
- Batch processing support

### ML Enhancement
- Integrates FinBERT inference as library
- Fallback behavior if ML unavailable
- Confidence scoring and error handling

### Signals Processing
- Mathematical transforms on sentiment scores  
- Filtering and normalization
- Future: FFT analysis, pattern recognition

## Development

### Adding New Features

1. **Extend pipeline.rs** for new processing steps
2. **Update output.rs** for new display formats  
3. **Modify client.rs** for additional gRPC calls
4. **Enhance go_service.rs** for new Go integration modes

### Testing

```bash
# Run tests
cargo test

# Test with mock data (bypasses Go service)
cargo run -- --once --mock-data

# Debug logging
RUST_LOG=debug cargo run -- --once
```

## Troubleshooting

### Common Issues

1. **Preprocessing service not running:**
   ```
   Error: Connection refused (port 50051)
   Solution: Start preprocessing service first
   ```

2. **Go binary missing:**
   ```
   Error: Go binary not found
   Solution: Run 'go build' in finmedia directory
   ```

3. **FinBERT model missing:**
   ```
   Warning: FinBERT initialization failed
   Solution: Ensure ONNX model files exist
   ```

### Dependencies

The orchestrator automatically manages all services:
- **Preprocessing service** - Built and started automatically
- **Go service** - Built and started automatically
- **No manual service management required**