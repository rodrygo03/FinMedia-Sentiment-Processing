use crate::pipeline::EnhancedResult;
use crate::signals_processing::SignalsAnalysis;
use chrono::{DateTime, Utc};
use tracing::debug;

/// Unified event structure for signals and systems processing
/// Combines all data from Go service, preprocessing, ML inference, and orchestrator
pub struct UnifiedFinancialEvent {
    // Core Event Identity
    pub id: String,
    pub title: String,
    pub content: String,
    pub processed_text: String,
    pub published_at: DateTime<Utc>,
    pub processed_at: DateTime<Utc>,
    pub source: String,
    pub url: String,
    
    // Asset Detection
    pub assets: Vec<AssetInfo>,
    pub categories: Vec<String>,
    pub news_type: String,
    pub market_impact: String,
    
    // Multi-Layer Sentiment Analysis
    pub sentiment_layers: SentimentLayers,
    
    // Tokenization and NLP
    pub tokens: Vec<String>,
    pub asset_mentions: Vec<String>,
    
    // Vector database integration
    pub event_embedding: Option<Vec<f32>>,
    pub stored_in_vdb: bool,
    pub vdb_event_id: Option<String>,
    
    // TODO: ENHANCED NLP DATA STRUCTURES
    // =================================
    // When implementing comprehensive tokenization in preprocessing service,
    // enhance these fields with rich NLP data structures:
    //
    // PROPOSED ENHANCEMENTS:
    // ```rust
    // // Rich tokenization results
    // pub enhanced_tokens: Vec<EnhancedToken>,
    // pub token_statistics: TokenStatistics,
    // 
    // // Named entity recognition
    // pub named_entities: Vec<NamedEntity>,
    // pub entity_categories: HashMap<EntityType, usize>,
    // 
    // // Financial analysis
    // pub financial_terms: Vec<FinancialTerm>,
    // pub sentiment_words: Vec<SentimentWord>, 
    // pub ml_asset_mentions: Vec<MLAssetMention>,
    // 
    // // Quality metrics
    // pub tokenization_quality: f64,
    // pub financial_relevance_score: f64,
    // pub text_complexity_score: f64,
    // pub entity_recognition_confidence: f64,
    // ```
    //
    // SUPPORTING STRUCTS TO ADD:
    // ```rust
    // pub struct EnhancedToken {
    //     pub text: String,
    //     pub stem: Option<String>,
    //     pub pos_tag: Option<String>, 
    //     pub financial_relevance: f64,
    //     pub sentiment_score: Option<f64>,
    //     pub entity_type: Option<EntityType>,
    //     pub frequency: usize,
    // }
    //
    // pub struct TokenStatistics {
    //     pub total_tokens: usize,
    //     pub unique_tokens: usize,
    //     pub financial_tokens: usize,
    //     pub average_relevance: f64,
    // }
    //
    // pub struct NamedEntity {
    //     pub text: String,
    //     pub entity_type: EntityType,
    //     pub confidence: f64,
    //     pub start_position: usize,
    //     pub end_position: usize,
    // }
    //
    // pub struct MLAssetMention {
    //     pub mention_text: String,
    //     pub inferred_assets: Vec<String>,
    //     pub confidence: f64,
    //     pub extraction_method: String,
    // }
    // ```
    
    // Signals Processing Results
    pub signals_analysis: SignalsAnalysis,
}

#[derive(Debug, Clone)]
pub struct AssetInfo {
    pub symbol: String,
    pub name: String,
    pub asset_type: String,
    pub confidence: f64,
    pub contexts: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SentimentLayers {
    // Go service analysis
    pub go_sentiment: f64,
    pub go_confidence: f64,
    
    // ML inference (FinBERT)
    pub ml_sentiment: f64,
    pub ml_confidence: f64,
    
    // Final processed score
    pub signals_score: f64,
}

impl UnifiedFinancialEvent {
    pub fn from_enhanced_result(enhanced: EnhancedResult) -> Self {
        let event = &enhanced.event;
        
        let published_at = chrono::DateTime::parse_from_rfc3339(&event.original_event.as_ref()
            .map(|e| e.published_at.clone())
            .unwrap_or_else(|| chrono::Utc::now().to_rfc3339()))
            .unwrap_or_else(|_| chrono::Utc::now().into())
            .with_timezone(&chrono::Utc);
            
        let processed_at = chrono::DateTime::parse_from_rfc3339(&event.processed_at)
            .unwrap_or_else(|_| chrono::Utc::now().into())
            .with_timezone(&chrono::Utc);
        
        let assets: Vec<AssetInfo> = event.assets.iter().map(|asset| AssetInfo {
            symbol: asset.symbol.clone(),
            name: asset.name.clone(),
            asset_type: asset.r#type.clone(),
            confidence: asset.confidence,
            contexts: asset.contexts.clone(),
        }).collect();
        
        let (title, content, source, url) = if let Some(original) = &event.original_event {
            (original.title.clone(), original.content.clone(), 
             original.source.clone(), original.url.clone())
        } else {
            (String::new(), String::new(), String::new(), String::new())
        };
        
        Self {
            id: event.id.clone(),
            title,
            content,
            processed_text: event.processed_text.clone(),
            published_at,
            processed_at,
            source,
            url,
            assets,
            categories: event.categories.clone(),
            news_type: event.news_type.clone(),
            market_impact: event.market_impact.clone(),
            sentiment_layers: SentimentLayers {
                go_sentiment: event.sentiment_score,
                go_confidence: event.confidence,
                ml_sentiment: enhanced.ml_sentiment,
                ml_confidence: enhanced.ml_confidence,
                signals_score: enhanced.signals_score,
            },
            tokens: event.tokens.clone(),
            asset_mentions: event.asset_mentions.clone(),
            event_embedding: None,
            stored_in_vdb: false,
            vdb_event_id: None,
            signals_analysis: enhanced.signals_analysis, 
        }
    }
        
    pub fn get_trading_signal(&self) -> f64 {
        use crate::signals_processing::SignalsProcessingEngine;
        SignalsProcessingEngine::calculate_trading_signal(
            self.sentiment_layers.go_sentiment,
            self.sentiment_layers.ml_sentiment,
            self.sentiment_layers.signals_score,
        )
    }
    
    pub fn is_high_impact(&self) -> bool {
        use crate::signals_processing::SignalsProcessingEngine;
        SignalsProcessingEngine::is_high_impact_event(
            &self.market_impact,
            self.sentiment_layers.signals_score,
            self.signals_analysis.volatility_index,
        )
    }
    
    pub fn primary_asset(&self) -> Option<&AssetInfo> {
        self.assets.iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
    }
}