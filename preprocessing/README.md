# FinMedia Preprocessing Service

High-performance Rust preprocessing service for financial news sentiment analysis.

## Project Status

This project is being developed in phases following a module-by-module approach.

**Current Phase**: Phase 1 - Foundation  
**Current Module**: Module 1 - Project Structure & Dependencies ✅

## Development Strategy

See `../claude-log/preprocessing_development_strategy.txt` for the complete development roadmap.

## Quick Start

```bash
# Build the project
cargo build

# Run with default settings
cargo run

# Show help
cargo run -- --help

# Run health check
cargo run -- health

# Generate sample config
cargo run -- gen-config -o config.toml
```

## Next Steps

1. **Phase 1, Module 2**: Implement core data models
2. **Phase 1, Module 3**: Build configuration system
3. **Phase 2**: Core processing logic
4. **Phase 3**: External communication (gRPC)
5. **Phase 4**: Production features (metrics, monitoring)

## Architecture

```
src/
├── lib.rs          # Library entry point
├── main.rs         # CLI application
├── models.rs       # Data structures (placeholder)
├── config.rs       # Configuration management (placeholder)
└── error.rs        # Error handling (placeholder)
```

## Integration

This service is designed to integrate with the Go finmedia service for preprocessing financial news before sentiment analysis.