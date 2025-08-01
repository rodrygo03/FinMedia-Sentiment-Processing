use crate::pipeline::EnhancedResult;
use chrono::{DateTime, Utc};

pub struct ResultFormatter {
    event_count: u64,
    start_time: DateTime<Utc>,
}

impl ResultFormatter {
    pub fn new() -> Self {
        Self {
            event_count: 0,
            start_time: Utc::now(),
        }
    }

    pub fn display_result(&mut self, result: &EnhancedResult) {
        self.event_count += 1;
        
        println!("\n{}", "=".repeat(80));
        println!("🔍 FINMEDIA ANALYSIS RESULT #{}", self.event_count);
        println!("{}", "=".repeat(80));

        println!("📰 Event ID: {}", result.event.id);
        println!("🕐 Processed: {}", result.event.processed_at);
        println!("📝 Text: {}", self.truncate_text(&result.event.processed_text, 100));

        if !result.event.categories.is_empty() {
            println!("🏷️  Categories: {}", result.event.categories.join(", "));
        }

        println!("\n{}", "-".repeat(40));
        println!("📊 SENTIMENT ANALYSIS");
        println!("{}", "-".repeat(40));

        // Original Go service sentiment
        println!("🔵 Go Service:");
        println!("   Sentiment: {:.3} {}", 
            result.event.sentiment_score, 
            self.sentiment_emoji(result.event.sentiment_score));
        println!("   Confidence: {:.3}", result.event.confidence);

        // ML inference results
        println!("🤖 ML Inference (FinBERT):");
        if result.ml_confidence > 0.0 {
            println!("   Sentiment: {:.3} {}", 
                result.ml_sentiment, 
                self.sentiment_emoji(result.ml_sentiment));
            println!("   Confidence: {:.3}", result.ml_confidence);
        } else {
            println!("   Status: Not available");
        }

        // Signals processing
        println!("📡 Signals Processing:");
        println!("   Final Score: {:.3} {}", 
            result.signals_score, 
            self.sentiment_emoji(result.signals_score));

        // Market impact
        if !result.event.market_impact.is_empty() {
            println!("\n💹 Market Impact: {} {}", 
                result.event.market_impact.to_uppercase(),
                self.impact_emoji(&result.event.market_impact));
        }

        // Asset detection summary
        if !result.event.assets.is_empty() {
            println!("\n🎯 Detected Assets:");
            for asset in &result.event.assets {
                println!("   • {} ({}) - {:.2}% confidence", 
                    asset.symbol, 
                    asset.r#type,
                    asset.confidence * 100.0);
            }
        }

        // Processing summary
        self.display_summary();
    }

    fn display_summary(&self) {
        let runtime = Utc::now().signed_duration_since(self.start_time);
        println!("\n{}", "-".repeat(40));
        println!("📈 PROCESSING SUMMARY");
        println!("{}", "-".repeat(40));
        println!("Events processed: {}", self.event_count);
        println!("Runtime: {}s", runtime.num_seconds());
        if runtime.num_seconds() > 0 {
            println!("Throughput: {:.2} events/sec", 
                self.event_count as f64 / runtime.num_seconds() as f64);
        }
    }

    pub fn display_pipeline_start(&self) {
        println!("\n{}", "-".repeat(20));
        println!("FINMEDIA SENTIMENT PROCESSING PIPELINE");
        println!("{}", "-".repeat(20));
        println!("Pipeline Flow:");
        println!("   Go RSS Ingestion → Rust Preprocessing → ML Inference → Signals → Terminal");
        println!("Started: {}", self.start_time.format("%Y-%m-%d %H:%M:%S UTC"));
        println!("{}", "=".repeat(80));
    }

    pub fn display_pipeline_complete(&self) {
        let runtime = Utc::now().signed_duration_since(self.start_time);
        
        println!("\n{}", "-".repeat(20));
        println!("PIPELINE EXECUTION COMPLETE");
        println!("{}", "-".repeat(20));
        println!("Final Statistics:");
        println!("   Total Events: {}", self.event_count);
        println!("   Total Runtime: {}s", runtime.num_seconds());
        println!("   Average Throughput: {:.2} events/sec", 
            if runtime.num_seconds() > 0 { 
                self.event_count as f64 / runtime.num_seconds() as f64 
            } else { 
                0.0 
            });
        println!("{}", "=".repeat(80));
    }

    pub fn display_error(&self, error: &anyhow::Error) {
        println!("\n{}", "❌".repeat(20));
        println!("💥 PIPELINE ERROR");
        println!("{}", "❌".repeat(20));
        println!("Error: {}", error);
        println!("Events processed before error: {}", self.event_count);
        println!("{}", "=".repeat(80));
    }

    fn sentiment_emoji(&self, score: f64) -> &'static str {
        match score {
            s if s > 0.5 => "🚀", // Very positive
            s if s > 0.2 => "📈", // Positive
            s if s > -0.2 => "➡️", // Neutral
            s if s > -0.5 => "📉", // Negative
            _ => "🔻" // Very negative
        }
    }

    fn impact_emoji(&self, impact: &str) -> &'static str {
        match impact.to_lowercase().as_str() {
            "high" => "🔥",
            "medium" => "⚡",
            "low" => "💧",
            _ => "❓"
        }
    }

    fn truncate_text(&self, text: &str, max_len: usize) -> String {
        if text.len() <= max_len {
            text.to_string()
        } else {
            format!("{}...", &text[..max_len])
        }
    }
}

impl Default for ResultFormatter {
    fn default() -> Self {
        Self::new()
    }
}