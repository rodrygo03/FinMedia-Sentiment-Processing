# Configuration Rules for finmedia News Ingestion Service

This document outlines the structure and rules for the configuration files used by the finmedia news ingestion service.

## File Structure

The configuration system uses two separate files:
- `config.json` - Main application configuration
- `assets.json` - Financial asset definitions for detection

## config.json Rules

### Overview
The main configuration file that defines RSS feeds, server settings, pipeline configuration, and asset detection parameters.

### Required Fields

#### RSS Feeds
```json
{
  "rss_feeds": [
    "https://news.google.com/rss/search?q=bitcoin&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=ethereum&hl=en-US&gl=US&ceid=US:en"
  ]
}
```

**Rules:**
- **Type:** Array of strings
- **Required:** Yes
- **Minimum:** 1 feed URL
- **Format:** Must be valid HTTP/HTTPS URLs
- **Purpose:** RSS feeds to monitor for news content

#### Server Configuration
```json
{
  "server": {
    "port": 8080
  }
}
```

**Rules:**
- **Type:** Object
- **Required:** Yes
- **port:** Integer between 1-65535
- **Default:** 8080 if not specified

#### Pipeline Configuration
```json
{
  "pipeline": {
    "batch_size": 10,
    "flush_timeout": "30s",
    "output_queue": "news_events"
  }
}
```

**Rules:**
- **Type:** Object
- **Required:** Yes
- **batch_size:** Integer > 0 (default: 10)
- **flush_timeout:** Valid duration string (e.g., "30s", "1m", "5m")
- **output_queue:** Non-empty string (default: "news_events")

#### Asset Detection Configuration
```json
{
  "asset_detection": {
    "assets_file": "assets.json",
    "case_sensitive": false,
    "min_confidence": 0.3,
    "context_words": ["price", "trading", "market"],
    "exclude_patterns": ["bitcoin pizza", "apple fruit"],
    "category_filters": ["crypto", "stock", "etf"]
  }
}
```

**Rules:**
- **Type:** Object
- **Required:** Yes
- **assets_file:** Path to external assets file (relative to config.json)
- **case_sensitive:** Boolean (default: false)
- **min_confidence:** Float between 0.0-1.0 (default: 0.3)
- **context_words:** Array of strings for context matching
- **exclude_patterns:** Array of strings to exclude from detection
- **category_filters:** Array of valid asset types

### Environment Variable Overrides

The following environment variables can override config.json values:

- `RSS_FEEDS` - JSON array of feed URLs
- `SERVER_PORT` - Integer port number
- `PIPELINE_BATCH_SIZE` - Integer batch size
- `PIPELINE_FLUSH_TIMEOUT` - Duration string
- `PIPELINE_OUTPUT_QUEUE` - Queue name string

**Example:**
```bash
export RSS_FEEDS='["https://example.com/feed1", "https://example.com/feed2"]'
export SERVER_PORT=9090
export PIPELINE_BATCH_SIZE=20
```

## assets.json Rules

### Overview
Contains financial asset definitions for detection in news content.

### Required Structure
```json
{
  "assets": [
    {
      "symbol": "BTC",
      "name": "Bitcoin",
      "type": "crypto",
      "patterns": ["bitcoin", "btc", "₿"],
      "exchanges": ["coinbase", "binance"],
      "priority": 10,
      "enabled": true
    }
  ]
}
```

### Asset Object Rules

#### Required Fields
- **symbol:** String, unique identifier (e.g., "BTC", "AAPL")
- **name:** String, full name (e.g., "Bitcoin", "Apple Inc.")
- **type:** String, must be one of: `crypto`, `stock`, `etf`, `forex`
- **patterns:** Array of strings, keywords for detection
- **priority:** Integer 1-10, higher = more important
- **enabled:** Boolean, whether to use this asset

#### Optional Fields
- **exchanges:** Array of strings, relevant exchanges

### Asset Types

#### Cryptocurrency (`crypto`)
```json
{
  "symbol": "BTC",
  "name": "Bitcoin",
  "type": "crypto",
  "patterns": ["bitcoin", "btc", "₿", "btc/usd", "btc-usd"],
  "exchanges": ["coinbase", "binance", "kraken"],
  "priority": 10,
  "enabled": true
}
```

#### Stock (`stock`)
```json
{
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "type": "stock",
  "patterns": ["apple", "aapl", "apple inc", "apple stock", "iphone"],
  "exchanges": ["nasdaq", "nyse"],
  "priority": 9,
  "enabled": true
}
```

#### ETF (`etf`)
```json
{
  "symbol": "SPY",
  "name": "SPDR S&P 500 ETF",
  "type": "etf",
  "patterns": ["spy", "s&p 500", "sp500", "spdr"],
  "exchanges": ["nyse"],
  "priority": 8,
  "enabled": true
}
```

### Asset Rules

#### Symbol Rules
- Must be unique across all assets
- Typically 2-5 characters for stocks/crypto
- Use standard market symbols (e.g., AAPL, BTC, SPY)

#### Pattern Rules
- Include common variations of the asset name
- Consider both full names and abbreviations
- Include trading pairs for crypto (e.g., "btc/usd", "eth-usd")
- Avoid overly generic terms that could cause false positives

#### Priority Rules
- **10:** Most important assets (e.g., Bitcoin, major stocks)
- **7-9:** Important assets (e.g., Ethereum, FAANG stocks)
- **4-6:** Moderate importance (e.g., altcoins, mid-cap stocks)
- **1-3:** Lower priority assets

#### Exchange Rules
- Use lowercase exchange names
- Common exchanges:
  - **Crypto:** coinbase, binance, kraken, gemini
  - **Stock:** nasdaq, nyse, amex
  - **ETF:** nyse, nasdaq

## Configuration Loading Order

1. Load `config.json` from application directory
2. Parse main configuration sections
3. Load external `assets.json` file if `assets_file` is specified
4. Apply environment variable overrides
5. Set default values for missing optional fields
6. Validate all required fields are present

## Error Handling

### config.json Errors
- Missing required fields → Fatal error
- Invalid JSON syntax → Fatal error
- Invalid RSS URLs → Warning, skip invalid URLs
- Invalid duration format → Use default value

### assets.json Errors
- File not found → Fatal error
- Invalid JSON syntax → Fatal error
- Missing required asset fields → Skip asset, log warning
- Duplicate symbols → Use first occurrence, log warning

## Best Practices

### config.json
- Keep RSS feeds focused on relevant financial news
- Set appropriate batch sizes based on expected volume
- Use reasonable flush timeouts (30s-5m)
- Include relevant context words for your use case

### assets.json
- Use comprehensive pattern lists for better detection
- Set priorities based on your trading focus
- Disable assets you don't need to improve performance
- Group similar assets with consistent priority levels
- Test pattern effectiveness with sample news content

## Validation Commands

```bash
# Validate JSON syntax
cat config.json | jq '.'
cat assets.json | jq '.'

# Test configuration loading
go run main.go --validate-config
```

## Example Files

### Minimal config.json
```json
{
  "rss_feeds": [
    "https://news.google.com/rss/search?q=bitcoin&hl=en-US&gl=US&ceid=US:en"
  ],
  "server": {
    "port": 8080
  },
  "pipeline": {
    "batch_size": 10,
    "flush_timeout": "30s",
    "output_queue": "news_events"
  },
  "asset_detection": {
    "assets_file": "assets.json",
    "case_sensitive": false,
    "min_confidence": 0.3,
    "context_words": ["price", "trading"],
    "exclude_patterns": [],
    "category_filters": ["crypto", "stock"]
  }
}
```

### Minimal assets.json
```json
{
  "assets": [
    {
      "symbol": "BTC",
      "name": "Bitcoin",
      "type": "crypto",
      "patterns": ["bitcoin", "btc"],
      "exchanges": ["coinbase"],
      "priority": 10,
      "enabled": true
    }
  ]
}
```