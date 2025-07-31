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