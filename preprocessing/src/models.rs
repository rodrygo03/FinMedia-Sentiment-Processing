use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Asset match information from Go asset detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetMatch {
    pub symbol: String,        // BTC, AAPL, etc.
    pub name: String,          // Bitcoin, Apple Inc.
    pub asset_type: String,    // crypto, stock, etf, monetary, economic, geopolitical
    pub confidence: f64,       // Detection confidence 0.0-1.0
    pub contexts: Vec<String>, // Context snippets around matches
}

/// Enhanced news event from Go finmedia service with complete asset detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsEvent {
    pub id: String,
    pub title: String,
    pub content: String,
    pub published_at: DateTime<Utc>,
    pub source: String,
    pub url: String,
    pub assets: Vec<AssetMatch>,         // Asset detection results
    pub categories: Vec<String>,         // crypto, stock, forex, monetary, economic, geopolitical
    pub sentiment: f64,                  // Sentiment score -1 to 1
    pub confidence: f64,                 // Overall detection confidence
    pub news_type: String,               // financial, political, geopolitical
    pub market_impact: String,           // high, medium, low
}

/// Processed event output for trading analysis - retains ALL Go service information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedEvent {
    pub id: String,
    pub original_event: NewsEvent,
    pub processed_text: String,
    pub tokens: Vec<String>,              // ML service tokens
    
    // Enhanced fields from Go service - TOP LEVEL ACCESS
    pub assets: Vec<AssetMatch>,          // Asset detection results from Go
    pub categories: Vec<String>,          // Asset type classifications from Go
    pub sentiment_score: f64,             // Sentiment from Go service
    pub confidence: f64,                  // Overall confidence from Go service
    pub news_type: String,                // News classification from Go service
    pub market_impact: String,            // Market impact from Go service
    
    // Additional ML processing fields (future use)
    pub ml_sentiment_score: f64,          // ML-enhanced sentiment (-1.0 to 1.0)
    pub ml_confidence: f64,               // ML processing confidence (0.0 to 1.0)
    pub asset_mentions: Vec<String>,      // ML-detected additional mentions
    
    pub processed_at: DateTime<Utc>,
}

impl ProcessedEvent {
    pub fn new(id: String, original_event: NewsEvent) -> Self {
        Self {
            id,
            // Extract enhanced fields from Go service to top level
            assets: original_event.assets.clone(),
            categories: original_event.categories.clone(),
            sentiment_score: original_event.sentiment,
            confidence: original_event.confidence,
            news_type: original_event.news_type.clone(),
            market_impact: original_event.market_impact.clone(),
            
            original_event,
            processed_text: String::new(),
            tokens: Vec::new(),
            
            // ML processing fields (empty initially)
            ml_sentiment_score: 0.0,
            ml_confidence: 0.0,
            asset_mentions: Vec::new(),
            
            processed_at: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_news_event_creation() {
        let asset_match = AssetMatch {
            symbol: "BTC".to_string(),
            name: "Bitcoin".to_string(),
            asset_type: "crypto".to_string(),
            confidence: 0.95,
            contexts: vec!["Bitcoin surges to new high".to_string()],
        };

        let event = NewsEvent {
            id: "test-123".to_string(),
            title: "Test News".to_string(),
            content: "This is test content".to_string(),
            published_at: Utc::now(),
            source: "TestSource".to_string(),
            url: "https://example.com".to_string(),
            assets: vec![asset_match],
            categories: vec!["crypto".to_string()],
            sentiment: 0.5,
            confidence: 0.8,
            news_type: "financial".to_string(),
            market_impact: "high".to_string(),
        };

        assert_eq!(event.id, "test-123");
        assert_eq!(event.title, "Test News");
        assert_eq!(event.assets.len(), 1);
        assert_eq!(event.assets[0].symbol, "BTC");
    }

    #[test]
    fn test_processed_event_creation() {
        let news_event = NewsEvent {
            id: "test-123".to_string(),
            title: "Test News".to_string(),
            content: "This is test content".to_string(),
            published_at: Utc::now(),
            source: "TestSource".to_string(),
            url: "https://example.com".to_string(),
            assets: vec![],
            categories: vec![],
            sentiment: 0.0,
            confidence: 0.0,
            news_type: String::new(),
            market_impact: String::new(),
        };

        let processed_event = ProcessedEvent::new("processed-123".to_string(), news_event);

        assert_eq!(processed_event.id, "processed-123");
        assert_eq!(processed_event.original_event.id, "test-123");
        assert_eq!(processed_event.sentiment_score, 0.0);
        assert_eq!(processed_event.confidence, 0.0);
    }
}