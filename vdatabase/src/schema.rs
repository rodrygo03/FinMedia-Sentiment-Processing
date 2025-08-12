use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Datelike, Timelike};
use uuid::Uuid;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialEvent {
    pub id: String,
    pub title: String,
    pub content: String,
    pub url: Option<String>,
    pub source: String,
    pub published_at: DateTime<Utc>,
    pub processed_at: DateTime<Utc>,
    pub assets: Vec<AssetInfo>,
    pub primary_asset: Option<String>,
    pub sentiment: SentimentInfo,
    
    // Multi-vector embeddings (each 768-dimensional)
    pub embedding: Vec<f32>,           // Primary content embedding
    pub sentiment_embedding: Vec<f32>, 
    pub asset_embedding: Vec<f32>,     
    
    // Metadata for filtering and search
    pub metadata: EventMetadata,
    
    // Raw event data for reconstruction
    pub raw_data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetInfo {
    pub symbol: String,
    pub name: Option<String>,
    pub asset_type: AssetType,
    pub confidence: f64,
    pub context: String,
    pub exchange: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetType {
    Crypto,
    Stock,
    ETF,
    Commodity,
    Currency,
    Bond,
    Index,
    Monetary,
    Economic,
    Geopolitical,
}

impl AssetType {
    pub fn as_str(&self) -> &'static str {
        match self {
            AssetType::Crypto => "crypto",
            AssetType::Stock => "stock",
            AssetType::ETF => "etf",
            AssetType::Commodity => "commodity",
            AssetType::Currency => "currency",
            AssetType::Bond => "bond",
            AssetType::Index => "index",
            AssetType::Monetary => "monetary",
            AssetType::Economic => "economic",
            AssetType::Geopolitical => "geopolitical",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentInfo {
    pub score: f64,
    pub label: SentimentLabel,
    pub confidence: f64,
    pub positive_score: f64,
    pub negative_score: f64,
    pub neutral_score: f64,
    pub trading_signal: f64,
    pub signal_strength: SignalStrength,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SentimentLabel {
    VeryPositive,
    Positive,
    Neutral,
    Negative,
    VeryNegative,
}

impl SentimentLabel {
    pub fn as_str(&self) -> &'static str {
        match self {
            SentimentLabel::VeryPositive => "very_positive",
            SentimentLabel::Positive => "positive",
            SentimentLabel::Neutral => "neutral",
            SentimentLabel::Negative => "negative",
            SentimentLabel::VeryNegative => "very_negative",
        }
    }
    
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s > 0.5 => SentimentLabel::VeryPositive,
            s if s > 0.2 => SentimentLabel::Positive,
            s if s > -0.2 => SentimentLabel::Neutral,
            s if s > -0.5 => SentimentLabel::Negative,
            _ => SentimentLabel::VeryNegative,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SignalStrength {
    StrongBuy,
    Buy,
    Neutral,
    Sell,
    StrongSell,
}

impl SignalStrength {
    pub fn as_str(&self) -> &'static str {
        match self {
            SignalStrength::StrongBuy => "strong_buy",
            SignalStrength::Buy => "buy",
            SignalStrength::Neutral => "neutral",
            SignalStrength::Sell => "sell",
            SignalStrength::StrongSell => "strong_sell",
        }
    }
    
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s > 0.6 => SignalStrength::StrongBuy,
            s if s > 0.2 => SignalStrength::Buy,
            s if s > -0.2 => SignalStrength::Neutral,
            s if s > -0.6 => SignalStrength::Sell,
            _ => SignalStrength::StrongSell,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    // Temporal metadata
    pub timestamp_bucket: String,
    pub hour_bucket: u8,
    pub day_of_week: u8,
    pub unix_timestamp: i64,
    
    // Asset metadata
    pub asset_types: Vec<String>,
    pub asset_count: u32,
    pub primary_asset_type: Option<String>,
    pub confidence_tier: String,
    
    // Content metadata
    pub content_length: u32,
    pub content_tier: String,
    pub language: String,
    pub source_domain: String,
    pub categories: Vec<String>,
    
    // Sentiment metadata
    pub sentiment_tier: String,
    pub impact_level: String,
    pub volatility_tier: String,
    pub trading_signal_tier: String,
    pub momentum_direction: String,
    
    // Additional metadata
    pub custom_fields: HashMap<String, serde_json::Value>,
}

impl FinancialEvent {
    pub const EMBEDDING_DIM: usize = 768;
    
    pub fn new(
        title: String,
        content: String,
        source: String,
        published_at: DateTime<Utc>,
    ) -> Self {
        let id = Uuid::new_v4().to_string();
        let processed_at = Utc::now();
        
        Self {
            id,
            title,
            content,
            url: None,
            source,
            published_at,
            processed_at,
            assets: Vec::new(),
            primary_asset: None,
            sentiment: SentimentInfo::default(),
            embedding: vec![0.0; Self::EMBEDDING_DIM],
            sentiment_embedding: vec![0.0; Self::EMBEDDING_DIM],
            asset_embedding: vec![0.0; Self::EMBEDDING_DIM],
            metadata: EventMetadata::default(),
            raw_data: String::new(),
        }
    }
    
    pub fn validate_embedding(&self) -> crate::Result<()> {
        if self.embedding.len() != Self::EMBEDDING_DIM {
            return Err(crate::VectorDBError::InvalidEmbeddingDimension {
                expected: Self::EMBEDDING_DIM,
                actual: self.embedding.len(),
            });
        }
        if self.sentiment_embedding.len() != Self::EMBEDDING_DIM {
            return Err(crate::VectorDBError::InvalidEmbeddingDimension {
                expected: Self::EMBEDDING_DIM,
                actual: self.sentiment_embedding.len(),
            });
        }
        if self.asset_embedding.len() != Self::EMBEDDING_DIM {
            return Err(crate::VectorDBError::InvalidEmbeddingDimension {
                expected: Self::EMBEDDING_DIM,
                actual: self.asset_embedding.len(),
            });
        }
        Ok(())
    }
    
    pub fn add_asset(&mut self, asset: AssetInfo) {
        if self.primary_asset.is_none() || asset.confidence > 0.8 {
            self.primary_asset = Some(asset.symbol.clone());
        }
        self.assets.push(asset);
        self.update_metadata();
    }
    
    pub fn set_sentiment(&mut self, sentiment: SentimentInfo) {
        self.sentiment = sentiment;
        self.update_metadata();
    }
    
    pub fn set_embedding(&mut self, embedding: Vec<f32>) -> crate::Result<()> {
        if embedding.len() != Self::EMBEDDING_DIM {
            return Err(crate::VectorDBError::InvalidEmbeddingDimension {
                expected: Self::EMBEDDING_DIM,
                actual: embedding.len(),
            });
        }
        self.embedding = embedding;
        Ok(())
    }
    
    pub fn set_multi_embeddings(
        &mut self, 
        content_embedding: Vec<f32>,
        sentiment_embedding: Vec<f32>,
        asset_embedding: Vec<f32>
    ) -> crate::Result<()> {
        if content_embedding.len() != Self::EMBEDDING_DIM 
            || sentiment_embedding.len() != Self::EMBEDDING_DIM 
            || asset_embedding.len() != Self::EMBEDDING_DIM {
            return Err(crate::VectorDBError::InvalidEmbeddingDimension {
                expected: Self::EMBEDDING_DIM,
                actual: content_embedding.len().max(sentiment_embedding.len()).max(asset_embedding.len()),
            });
        }
        
        self.embedding = content_embedding;
        self.sentiment_embedding = sentiment_embedding;
        self.asset_embedding = asset_embedding;
        Ok(())
    }
    
    /// Get embedding by type for multi-vector search
    pub fn get_embedding_by_type(&self, embedding_type: &str) -> Option<&Vec<f32>> {
        match embedding_type {
            "content" => Some(&self.embedding),
            "sentiment" => Some(&self.sentiment_embedding),
            "asset" => Some(&self.asset_embedding),
            _ => None,
        }
    }
    
    fn update_metadata(&mut self) {
        let asset_types: Vec<String> = self.assets
            .iter()
            .map(|a| a.asset_type.as_str().to_string())
            .collect();
        
        let confidence_tier = if self.assets.iter().any(|a| a.confidence > 0.8) {
            "high"
        } else if self.assets.iter().any(|a| a.confidence > 0.5) {
            "medium"
        } else {
            "low"
        }.to_string();
        
        let content_tier = match self.content.len() {
            0..=200 => "short",
            201..=800 => "medium",
            _ => "long",
        }.to_string();
        
        self.metadata = EventMetadata {
            timestamp_bucket: self.published_at.format("%Y-%m").to_string(),
            hour_bucket: self.published_at.hour() as u8,
            day_of_week: self.published_at.weekday().num_days_from_monday() as u8,
            unix_timestamp: self.published_at.timestamp(),
            
            asset_types,
            asset_count: self.assets.len() as u32,
            primary_asset_type: self.assets.first().map(|a| a.asset_type.as_str().to_string()),
            confidence_tier,
            
            content_length: self.content.len() as u32,
            content_tier,
            language: "en".to_string(),
            source_domain: self.source.clone(),
            categories: Vec::new(),
            
            sentiment_tier: self.sentiment.label.as_str().to_string(),
            impact_level: "medium".to_string(),
            volatility_tier: "unknown".to_string(),
            trading_signal_tier: self.sentiment.signal_strength.as_str().to_string(),
            momentum_direction: if self.sentiment.trading_signal > 0.1 {
                "bullish"
            } else if self.sentiment.trading_signal < -0.1 {
                "bearish"
            } else {
                "sideways"
            }.to_string(),
            
            custom_fields: HashMap::new(),
        };
    }
}

impl SentimentInfo {
    pub fn default() -> Self {
        Self {
            score: 0.0,
            label: SentimentLabel::Neutral,
            confidence: 0.0,
            positive_score: 0.0,
            negative_score: 0.0,
            neutral_score: 1.0,
            trading_signal: 0.0,
            signal_strength: SignalStrength::Neutral,
        }
    }
    
    pub fn new(score: f64, confidence: f64, positive: f64, negative: f64, neutral: f64) -> Self {
        let label = SentimentLabel::from_score(score);
        let trading_signal = score * confidence;
        let signal_strength = SignalStrength::from_score(trading_signal);
        
        Self {
            score,
            label,
            confidence,
            positive_score: positive,
            negative_score: negative,
            neutral_score: neutral,
            trading_signal,
            signal_strength,
        }
    }
}

impl EventMetadata {
    pub fn default() -> Self {
        Self {
            timestamp_bucket: String::new(),
            hour_bucket: 0,
            day_of_week: 0,
            unix_timestamp: 0,
            asset_types: Vec::new(),
            asset_count: 0,
            primary_asset_type: None,
            confidence_tier: "low".to_string(),
            content_length: 0,
            content_tier: "short".to_string(),
            language: "en".to_string(),
            source_domain: String::new(),
            categories: Vec::new(),
            sentiment_tier: "neutral".to_string(),
            impact_level: "low".to_string(),
            volatility_tier: "unknown".to_string(),
            trading_signal_tier: "neutral".to_string(),
            momentum_direction: "sideways".to_string(),
            custom_fields: HashMap::new(),
        }
    }
}

/// Signals processing results for storage in vector database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalsResult {
    pub id: String,
    pub event_id: String,  // References parent FinancialEvent
    pub processed_at: DateTime<Utc>,
    pub asset_symbols: Vec<String>,
    pub time_series: Option<TimeSeriesAnalysis>,
    pub frequency_analysis: Option<FrequencyAnalysis>,
    pub correlation_analysis: Option<CorrelationAnalysis>,
    pub scoring_results: ScoringResults,
    pub processing_metadata: SignalsMetadata,
    // Embeddings for similarity search
    pub signals_embedding: Vec<f32>,  // 768-dimensional embedding of signals features
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesAnalysis {
    pub trend_direction: String,
    pub trend_strength: f64,
    pub volatility_measure: f64,
    pub seasonal_patterns: Vec<SeasonalPattern>,
    pub statistical_measures: StatisticalMeasures,
    pub anomaly_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    pub pattern_type: String,  // "daily", "weekly", "monthly"
    pub strength: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalMeasures {
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub autocorrelation: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyAnalysis {
    pub dominant_frequencies: Vec<DominantFrequency>,
    pub power_spectrum: Vec<f64>,
    pub spectral_entropy: f64,
    pub peak_to_average_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominantFrequency {
    pub frequency: f64,
    pub magnitude: f64,
    pub phase: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysis {
    pub asset_correlations: HashMap<String, f64>,
    pub cross_correlations: Vec<CrossCorrelation>,
    pub correlation_matrix: Vec<Vec<f64>>,
    pub lead_lag_relationships: Vec<LeadLagRelation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCorrelation {
    pub asset_pair: (String, String),
    pub correlation_value: f64,
    pub lag: i32,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadLagRelation {
    pub leading_asset: String,
    pub lagging_asset: String,
    pub lag_time: f64,
    pub strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringResults {
    pub final_score: f64,
    pub confidence_level: f64,
    pub signal_strength: SignalStrength,
    pub trading_recommendation: TradingRecommendation,
    pub risk_assessment: RiskAssessment,
    pub component_scores: ComponentScores,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingRecommendation {
    pub action: String,  // "buy", "sell", "hold"
    pub strength: f64,
    pub time_horizon: String,  // "short", "medium", "long"
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub volatility_risk: f64,
    pub correlation_risk: f64,
    pub market_impact_risk: f64,
    pub overall_risk_score: f64,
    pub risk_category: String,  // "low", "medium", "high"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentScores {
    pub sentiment_component: f64,
    pub technical_component: f64,
    pub correlation_component: f64,
    pub volatility_component: f64,
    pub momentum_component: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalsMetadata {
    pub processing_duration_ms: u64,
    pub algorithm_version: String,
    pub model_parameters: HashMap<String, f64>,
    pub data_quality_score: f64,
    pub signal_to_noise_ratio: f64,
    pub feature_importance: HashMap<String, f64>,
}

impl SignalsResult {
    pub const EMBEDDING_DIM: usize = 768;
    
    pub fn new(event_id: String, asset_symbols: Vec<String>) -> Self {
        let id = Uuid::new_v4().to_string();
        let processed_at = Utc::now();
        
        Self {
            id,
            event_id,
            processed_at,
            asset_symbols,
            time_series: None,
            frequency_analysis: None,
            correlation_analysis: None,
            scoring_results: ScoringResults::default(),
            processing_metadata: SignalsMetadata::default(),
            signals_embedding: vec![0.0; Self::EMBEDDING_DIM],
        }
    }
    
    pub fn set_signals_embedding(&mut self, embedding: Vec<f32>) -> crate::Result<()> {
        if embedding.len() != Self::EMBEDDING_DIM {
            return Err(crate::VectorDBError::InvalidEmbeddingDimension {
                expected: Self::EMBEDDING_DIM,
                actual: embedding.len(),
            });
        }
        self.signals_embedding = embedding;
        Ok(())
    }
    
    pub fn validate_embedding(&self) -> crate::Result<()> {
        if self.signals_embedding.len() != Self::EMBEDDING_DIM {
            return Err(crate::VectorDBError::InvalidEmbeddingDimension {
                expected: Self::EMBEDDING_DIM,
                actual: self.signals_embedding.len(),
            });
        }
        Ok(())
    }
}

impl ScoringResults {
    pub fn default() -> Self {
        Self {
            final_score: 0.0,
            confidence_level: 0.0,
            signal_strength: SignalStrength::Neutral,
            trading_recommendation: TradingRecommendation::default(),
            risk_assessment: RiskAssessment::default(),
            component_scores: ComponentScores::default(),
        }
    }
}

impl TradingRecommendation {
    pub fn default() -> Self {
        Self {
            action: "hold".to_string(),
            strength: 0.0,
            time_horizon: "medium".to_string(),
            confidence: 0.0,
        }
    }
}

impl RiskAssessment {
    pub fn default() -> Self {
        Self {
            volatility_risk: 0.0,
            correlation_risk: 0.0,
            market_impact_risk: 0.0,
            overall_risk_score: 0.0,
            risk_category: "medium".to_string(),
        }
    }
}

impl ComponentScores {
    pub fn default() -> Self {
        Self {
            sentiment_component: 0.0,
            technical_component: 0.0,
            correlation_component: 0.0,
            volatility_component: 0.0,
            momentum_component: 0.0,
        }
    }
}

impl SignalsMetadata {
    pub fn default() -> Self {
        Self {
            processing_duration_ms: 0,
            algorithm_version: "1.0.0".to_string(),
            model_parameters: HashMap::new(),
            data_quality_score: 0.0,
            signal_to_noise_ratio: 0.0,
            feature_importance: HashMap::new(),
        }
    }
}