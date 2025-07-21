use crate::{models::{NewsEvent, ProcessedEvent}, Result};
use regex::Regex;
use uuid::Uuid;

pub struct TextProcessor {
    punctuation_regex: Regex,
}

impl TextProcessor {
    pub fn new() -> Self {
        let punctuation_regex = Regex::new(r"[^\w\s]").unwrap();
        
        Self {
            punctuation_regex,
        }
    }
    
    pub async fn process_event(&self, event: NewsEvent) -> Result<ProcessedEvent> {
        let processed_text = self.clean_text(&format!("{} {}", event.title, event.content));
        
        Ok(ProcessedEvent {
            id: Uuid::new_v4().to_string(),
            original_event: event,
            processed_text,
            tokens: vec![], // Will be populated by ML service later
            asset_mentions: vec![], // Not needed - Go service handles routing
            sentiment_score: 0.0, // Will be calculated by ML service later
            confidence: 0.0, // Will be calculated by ML service later
            processed_at: chrono::Utc::now(),
        })
    }
    
    pub async fn process_batch(&self, events: Vec<NewsEvent>) -> Vec<Result<ProcessedEvent>> {
        let mut results = Vec::with_capacity(events.len());
        
        for event in events {
            results.push(self.process_event(event).await);
        }
        
        results
    }
    
    fn clean_text(&self, text: &str) -> String {
        let mut cleaned = text.to_lowercase();
        cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
        cleaned = self.punctuation_regex.replace_all(&cleaned, " ").to_string();
        cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
        cleaned.trim().to_string()
    }
    
}

impl Default for TextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_event() -> NewsEvent {
        NewsEvent {
            id: "test-123".to_string(),
            title: "Bitcoin Surges to New Heights!".to_string(),
            content: "Bitcoin has reached a new all-time high as bullish sentiment drives the market.".to_string(),
            published_at: Utc::now(),
            source: "CryptoNews".to_string(),
            url: "https://example.com/news".to_string(),
        }
    }

    #[tokio::test]
    async fn test_process_event() {
        let processor = TextProcessor::new();
        let event = create_test_event();
        
        let result = processor.process_event(event).await;
        assert!(result.is_ok());
        
        let processed = result.unwrap();
        assert!(!processed.processed_text.is_empty());
        assert!(processed.tokens.is_empty()); // Should be empty until ML service processes
        assert!(processed.asset_mentions.is_empty()); // Go service handles routing
        assert_eq!(processed.sentiment_score, 0.0); // Should be 0.0 until ML service processes
        assert_eq!(processed.confidence, 0.0); // Should be 0.0 until ML service processes
    }

    #[test]
    fn test_clean_text() {
        let processor = TextProcessor::new();
        let dirty_text = "Bitcoin!!! Surges   to  $45,000!!!";
        let cleaned = processor.clean_text(dirty_text);
        
        assert_eq!(cleaned, "bitcoin surges to 45 000");
    }


    #[tokio::test]
    async fn test_batch_processing() {
        let processor = TextProcessor::new();
        let events = vec![
            create_test_event(),
            NewsEvent {
                id: "test-456".to_string(),
                title: "Ethereum Update".to_string(),
                content: "Ethereum network upgrade successful".to_string(),
                published_at: Utc::now(),
                source: "EthNews".to_string(),
                url: "https://example.com/eth".to_string(),
            }
        ];
        
        let results = processor.process_batch(events).await;
        assert_eq!(results.len(), 2);
        
        for result in results {
            assert!(result.is_ok());
        }
    }
}