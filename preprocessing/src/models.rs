use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// News event from Go finmedia service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsEvent {
    pub id: String,
    pub title: String,
    pub content: String,
    pub published_at: DateTime<Utc>,
    pub source: String,
    pub url: String,
}

/// Processed event output for trading analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedEvent {
    pub id: String,
    pub original_event: NewsEvent,
    pub processed_text: String,
    pub tokens: Vec<String>,
    pub asset_mentions: Vec<String>,
    pub sentiment_score: f64,      // -1.0 to 1.0
    pub confidence: f64,           // 0.0 to 1.0
    pub processed_at: DateTime<Utc>,
}

impl ProcessedEvent {
    pub fn new(id: String, original_event: NewsEvent) -> Self {
        Self {
            id,
            original_event,
            processed_text: String::new(),
            tokens: Vec::new(),
            asset_mentions: Vec::new(),
            sentiment_score: 0.0,
            confidence: 0.0,
            processed_at: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_news_event_creation() {
        let event = NewsEvent {
            id: "test-123".to_string(),
            title: "Test News".to_string(),
            content: "This is test content".to_string(),
            published_at: Utc::now(),
            source: "TestSource".to_string(),
            url: "https://example.com".to_string(),
        };

        assert_eq!(event.id, "test-123");
        assert_eq!(event.title, "Test News");
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
        };

        let processed_event = ProcessedEvent::new("processed-123".to_string(), news_event);

        assert_eq!(processed_event.id, "processed-123");
        assert_eq!(processed_event.original_event.id, "test-123");
        assert_eq!(processed_event.sentiment_score, 0.0);
        assert_eq!(processed_event.confidence, 0.0);
    }
}