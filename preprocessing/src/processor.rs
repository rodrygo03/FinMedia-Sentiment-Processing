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
        let combined_text = format!("{} {}", event.title, event.content);
        let processed_text = self.clean_text(&combined_text);
        
        // Use the constructor that properly extracts Go service fields
        let mut processed_event = ProcessedEvent::new(Uuid::new_v4().to_string(), event);
        processed_event.processed_text = processed_text;
        
        // TODO: COMPREHENSIVE TOKENIZATION & NLP IMPLEMENTATION
        // ============================================================
        // 
        // PHASE 1: Basic Tokenization Foundation (Week 1-2)
        // --------------------------------------------------
        // - [ ] Implement TokenizationEngine struct with:
        //   - [ ] Financial-aware stop word removal (preserve "up", "down", "high", "low")
        //   - [ ] Number normalization: "$50,000" → "50000_USD", "5%" → "5_percent"
        //   - [ ] Compound financial term preservation: "interest-rate", "market-cap"
        //   - [ ] Case preservation for entities: "Fed" vs "fed"
        //   - [ ] Token quality scoring by financial relevance (0.0-1.0)
        //
        // PHASE 2: Advanced NLP Features (Week 3-4)
        // ------------------------------------------
        // - [ ] Named Entity Recognition (NER):
        //   - [ ] FinancialNER struct with regex patterns for:
        //     - [ ] Companies: "Apple Inc", "Microsoft Corp"
        //     - [ ] People: "Jerome Powell", "Elon Musk"
        //     - [ ] Financial instruments: "bonds", "derivatives", "futures"
        //     - [ ] Locations: "Wall Street", "Silicon Valley"
        //   - [ ] NamedEntity struct with confidence scoring
        //
        // - [ ] Financial Term Extraction:
        //   - [ ] TermCategory enum: MonetaryPolicy, TradingTerms, CorporateActions, etc.
        //   - [ ] Comprehensive financial vocabulary HashMap
        //   - [ ] Context validation for financial relevance
        //
        // - [ ] Sentiment-Bearing Word Detection:
        //   - [ ] SentimentWordExtractor with financial context:
        //     - [ ] positive_financial: "surge" → 0.8, "growth" → 0.6
        //     - [ ] negative_financial: "crash" → -0.9, "decline" → -0.4
        //     - [ ] context_modifiers: "might" → 0.5, "definitely" → 1.0
        //
        // PHASE 3: ML-Enhanced Analysis (Week 5-6)
        // -----------------------------------------
        // - [ ] MLAssetExtractor for implicit asset references:
        //   - [ ] "the crypto market" → ["CRYPTO_MARKET"]
        //   - [ ] "tech giants" → ["AAPL", "MSFT", "GOOGL", "AMZN"]
        //   - [ ] "energy sector" → ["XLE", "OIL", "GAS"]
        //   - [ ] Word embeddings or similarity matching
        //
        // - [ ] Context-Aware Tokenization:
        //   - [ ] "Apple" in tech vs fruit context
        //   - [ ] "Bear"/"Bull" in market vs animal context
        //   - [ ] TokenSentiment with context influence scoring
        //
        // PHASE 4: Production Optimization (Week 7-8)
        // --------------------------------------------
        // - [ ] TokenizationCache with LRU for performance
        // - [ ] StreamingTokenizer for real-time processing
        // - [ ] Parallel processing with Rayon
        // - [ ] Comprehensive testing and benchmarks
        //
        // ENHANCED DATA STRUCTURES TO IMPLEMENT:
        // =====================================
        // ```rust
        // pub struct Token {
        //     pub text: String,
        //     pub stem: Option<String>,
        //     pub pos_tag: Option<String>,
        //     pub financial_relevance: f64,
        //     pub position: TextPosition,
        //     pub frequency: usize,
        // }
        //
        // pub struct NamedEntity {
        //     pub text: String,
        //     pub entity_type: EntityType,
        //     pub confidence: f64,
        //     pub start_pos: usize,
        //     pub end_pos: usize,
        // }
        //
        // pub struct AssetMention {
        //     pub mention_text: String,
        //     pub inferred_assets: Vec<String>,
        //     pub confidence: f64,
        //     pub extraction_method: ExtractionMethod,
        // }
        // ```
        //
        // INTEGRATION POINTS:
        // ==================
        // - [ ] Update display in orchestrator/src/pipeline.rs print_unified_event()
        // - [ ] Enhance ML inference with token context in inference crate
        // - [ ] Use tokenization quality for signals processing confidence
        // - [ ] Add rich NLP metrics to UnifiedFinancialEvent
        //
        // SUCCESS METRICS TO TRACK:
        // ========================
        // - Token Relevance: % financially relevant tokens
        // - Entity Accuracy: % correctly identified entities  
        // - Asset Mention Precision: % accurate ML detections
        // - Processing Speed: tokens/second throughput
        // - Signal Quality: improvement in sentiment analysis
        //
        // Keep tokens empty for now - implement above roadmap when ready
        processed_event.tokens = Vec::new();
        
        Ok(processed_event)
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
            assets: vec![],
            categories: vec![],
            sentiment: 0.0,
            confidence: 0.0,
            news_type: String::new(),
            market_impact: String::new(),
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
                assets: vec![],
                categories: vec![],
                sentiment: 0.0,
                confidence: 0.0,
                news_type: String::new(),
                market_impact: String::new(),
            }
        ];
        
        let results = processor.process_batch(events).await;
        assert_eq!(results.len(), 2);
        
        for result in results {
            assert!(result.is_ok());
        }
    }
}