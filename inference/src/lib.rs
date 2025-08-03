pub mod error;
pub mod model;
pub mod tokenizer;
pub mod inference;

pub use error::{InferenceError, Result};
pub use inference::FinBertInference;
pub use tokenizer::{FinBertTokenizer, TokenizerOutput};

#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub sentiment_score: f64,    // -1.0 (negative) to 1.0 (positive)
    pub confidence: f64,         // 0.0 to 1.0
    pub label: String,           // "negative", "neutral", "positive"
    pub raw_scores: Vec<f64>,    // Raw model outputs [neg, neu, pos]
    
    // TODO: TOKEN-LEVEL SENTIMENT ANALYSIS ENHANCEMENT
    // ===============================================
    // When tokenization is implemented in preprocessing service,
    // enhance SentimentResult with token-level analysis:
    //
    // PROPOSED ENHANCEMENTS:
    // ```rust
    // // Token-level sentiment analysis
    // pub token_sentiments: Option<Vec<TokenSentiment>>,
    // pub attention_weights: Option<Vec<f64>>,
    // 
    // // Enhanced explainability  
    // pub sentiment_contributors: Option<Vec<SentimentContributor>>,
    // pub confidence_factors: Option<ConfidenceBreakdown>,
    // 
    // // Financial context analysis
    // pub financial_sentiment_score: Option<f64>,  // Weighted by financial relevance
    // pub asset_specific_sentiment: Option<HashMap<String, f64>>,
    // ```
    //
    // SUPPORTING STRUCTS:
    // ```rust
    // pub struct TokenSentiment {
    //     pub token: String,
    //     pub sentiment_score: f64,
    //     pub confidence: f64,
    //     pub attention_weight: f64,
    //     pub financial_relevance: f64,
    // }
    //
    // pub struct SentimentContributor {
    //     pub token: String,
    //     pub contribution: f64,  // How much this token affected overall sentiment
    //     pub reasoning: String,  // "Strong positive financial term"
    // }
    //
    // pub struct ConfidenceBreakdown {
    //     pub text_quality: f64,      // Clear, unambiguous text
    //     pub financial_context: f64, // Financial relevance of content
    //     pub model_certainty: f64,   // Model's internal confidence
    //     pub token_consistency: f64, // Agreement between token sentiments
    // }
    // ```
    //
    // INTEGRATION WITH TOKENIZATION:
    // - Use tokenization results to provide token-level sentiment
    // - Weight sentiment by financial relevance of tokens
    // - Provide explainable AI features for trading decisions
    // - Improve overall sentiment accuracy with context awareness
}

// Configuration structure matching finbert_config.json
#[derive(Debug, Clone, serde::Deserialize)]
pub struct FinBertConfig {
    pub model_path: String,
    pub tokenizer_path: String,
    pub max_length: usize,
    pub labels: Vec<String>,
    pub quantized: bool,
    pub model_type: String,
}