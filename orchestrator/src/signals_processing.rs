use anyhow::Result;
use tracing::{debug, warn};
use finmedia_signals::SignalsProcessor;
use crate::client::preprocessing::ProcessedEvent;

#[derive(Debug, Clone)]
pub struct SignalsAnalysis {
    /// Primary processed signals score (weighted combination of all analysis)
    pub final_score: f64,
    
    /// Frequency domain analysis results (FFT-based features)
    /// Currently includes: peak_frequency, spectral_centroid, zero_crossing_rate
    /// Future: Advanced frequency analysis, harmonics detection, spectral rolloff
    pub frequency_domain: Option<Vec<f64>>,
    
    /// Pattern recognition and statistical analysis scores
    /// Currently includes: mean, variance, quality_score
    /// Future: Advanced pattern matching, regime detection, anomaly scores
    pub pattern_scores: Option<Vec<f64>>,
    
    /// Market volatility index calculated from signal variance
    /// Represents short-term price movement volatility prediction
    pub volatility_index: Option<f64>,
    
    /// Momentum indicator showing directional trend strength
    /// Calculated as difference between recent and historical sentiment
    pub momentum_indicator: Option<f64>,
    
    /// Cross-asset correlation analysis (future enhancement)
    /// Will include correlations with major market indices and assets
    pub correlation_scores: Option<Vec<(String, f64)>>,
}

/// Enhanced result structure combining ML inference with comprehensive signals analysis
#[derive(Clone)]
pub struct SignalsEnhancedResult {
    pub event: ProcessedEvent,
    pub ml_sentiment: f64,
    pub ml_confidence: f64,
    pub signals_score: f64,
    pub signals_analysis: SignalsAnalysis,
}

pub struct SignalsProcessingEngine<'a> {
    signals_processor: &'a SignalsProcessor,
}

impl<'a> SignalsProcessingEngine<'a> {
    /// Create a new signals processing engine with a reference to the signals processor
    pub fn new(signals_processor: &'a SignalsProcessor) -> Self {
        Self { signals_processor }
    }
    
    pub async fn comprehensive_signals_analysis(
        &self,
        sentiment_score: f64,
        event: &ProcessedEvent,
    ) -> Result<(f64, SignalsAnalysis)> {
        debug!("Applying comprehensive signals analysis to event: {}", event.id);
        
        let sentiment_scores = vec![
            event.sentiment_score,  // Go service sentiment (asset detection + basic analysis)
            sentiment_score,        // ML inference sentiment (FinBERT ONNX model)
        ];
        
        match self.signals_processor.process_sentiment_signal(&sentiment_scores) {
            Ok(processed_signal) => {
                let final_score = processed_signal.features.mean.max(-1.0).min(1.0);
                
                let volatility_index = processed_signal.features.variance.sqrt();
                let momentum_indicator = if sentiment_scores.len() >= 2 {
                    sentiment_scores.last().unwrap() - sentiment_scores.first().unwrap()
                } else {
                    0.0
                };
                
                let signals_analysis = SignalsAnalysis {
                    final_score,
                    
                    // Frequency domain features from FFT analysis
                    frequency_domain: Some(vec![
                        processed_signal.features.peak_frequency,      // Dominant frequency component
                        processed_signal.features.spectral_centroid,   // Center of mass of spectrum
                        processed_signal.features.zero_crossing_rate,  // Signal oscillation rate
                    ]),
                    
                    // Statistical and quality analysis features
                    pattern_scores: Some(vec![
                        processed_signal.features.mean,        // Central tendency of sentiment
                        processed_signal.features.variance,    // Sentiment volatility measure
                        processed_signal.quality_score,        // Processing quality indicator
                    ]),
                    
                    // Market analysis indicators
                    volatility_index: Some(volatility_index),        // Market volatility prediction
                    momentum_indicator: Some(momentum_indicator),    // Trend direction strength
                    
                    // Future enhancement placeholder
                    correlation_scores: None, // Will include cross-asset correlations
                };
                
                debug!(
                    "Comprehensive signals analysis complete: input=[{:.3}, {:.3}] -> features(mean={:.3}, variance={:.3}, quality={:.3}) -> final={:.3}, volatility={:.3}, momentum={:.3}",
                    event.sentiment_score,
                    sentiment_score,
                    processed_signal.features.mean,
                    processed_signal.features.variance,
                    processed_signal.quality_score,
                    final_score,
                    volatility_index,
                    momentum_indicator
                );
                
                if processed_signal.quality_score < 0.5 {
                    warn!(
                        "Low quality signals processing for event {}: quality_score={:.3}",
                        event.id, processed_signal.quality_score
                    );
                }
                
                Ok((final_score, signals_analysis))
            }
            
            Err(e) => {
                warn!(
                    "Comprehensive signals analysis failed for event {}: {}. Using fallback analysis.",
                    event.id, e
                );
                // Generate fallback analysis using basic statistical methods
                let fallback_result = self.fallback_signals_analysis(&sentiment_scores, sentiment_score, event);
                Ok(fallback_result)
            }
        }
    }
    
    fn fallback_signals_analysis(
        &self,
        sentiment_scores: &[f64],
        sentiment_score: f64,
        event: &ProcessedEvent,
    ) -> (f64, SignalsAnalysis) {
        let fallback_score = (sentiment_score * 0.7 + event.sentiment_score * 0.3)
            .max(-1.0)
            .min(1.0);
        let mean = sentiment_scores.iter().sum::<f64>() / sentiment_scores.len() as f64;
        let variance = sentiment_scores.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / sentiment_scores.len() as f64;
        let volatility_index = variance.sqrt();
        let momentum_indicator = if sentiment_scores.len() >= 2 {
            sentiment_scores.last().unwrap() - sentiment_scores.first().unwrap()
        } else {
            0.0
        };
        
        let fallback_analysis = SignalsAnalysis {
            final_score: fallback_score,
            frequency_domain: None, // Not available in fallback mode
            pattern_scores: Some(vec![mean, variance, 0.5]), // Default quality score
            volatility_index: Some(volatility_index),
            momentum_indicator: Some(momentum_indicator),
            correlation_scores: None,
        };
        
        debug!(
            "Fallback signals analysis: {} -> final={:.3}, volatility={:.3}, momentum={:.3}",
            sentiment_score, fallback_score, volatility_index, momentum_indicator
        );
        
        (fallback_score, fallback_analysis)
    }
    
    pub fn calculate_trading_signal(
        go_sentiment: f64,
        ml_sentiment: f64,
        signals_score: f64,
    ) -> f64 {
        // Weighted combination of all sentiment layers
        // Weights can be adjusted based on empirical performance
        let weights = [0.3, 0.4, 0.3]; // go, ml, signals
        let scores = [go_sentiment, ml_sentiment, signals_score];
        
        weights.iter().zip(scores.iter())
            .map(|(w, s)| w * s)
            .sum()
    }

    pub fn is_high_impact_event(
        market_impact: &str,
        signals_score: f64,
        volatility_index: Option<f64>,
    ) -> bool {
        market_impact == "high" || 
        signals_score.abs() > 0.5 ||
        volatility_index.unwrap_or(0.0) > 0.3
    }
}

/// Utility functions for signals processing display and formatting
pub mod display {
    use super::SignalsAnalysis;
    
    pub fn format_signals_summary(analysis: &SignalsAnalysis) -> String {
        let mut summary = Vec::new();
        
        summary.push(format!("Final Score: {:.3}", analysis.final_score));
        
        if let Some(volatility) = analysis.volatility_index {
            summary.push(format!("Volatility: {:.3}", volatility));
        }
        
        if let Some(momentum) = analysis.momentum_indicator {
            summary.push(format!("Momentum: {:.3}", momentum));
        }
        
        if let Some(ref freq_domain) = analysis.frequency_domain {
            summary.push(format!("Freq Features: {} components", freq_domain.len()));
        }
        
        if let Some(ref pattern_scores) = analysis.pattern_scores {
            summary.push(format!("Pattern Quality: {:.3}", 
                pattern_scores.get(2).unwrap_or(&0.0)));
        }
        
        summary.join(" | ")
    }
    
    pub fn sentiment_emoji(score: f64) -> &'static str {
        match score {
            s if s > 0.5 => "🚀",
            s if s > 0.2 => "📈",
            s if s > -0.2 => "➡️",
            s if s > -0.5 => "📉",
            _ => "🔻",
        }
    }
}