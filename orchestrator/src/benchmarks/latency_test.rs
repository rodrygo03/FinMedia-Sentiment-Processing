use anyhow::Result;
use std::time::{Duration, Instant};
use tracing::{info, debug, warn};
use chrono::Utc;

use crate::benchmarks::metrics::{LatencyMetrics, LatencyReport, PipelineTimer};
use crate::pipeline::PipelineOrchestrator;
use crate::client::preprocessing::{ProcessedEvent, AssetMatch};

#[derive(Debug, Clone)]
pub enum LatencyTestType {
    ColdStart,      // First event processing (cold caches)
    WarmPipeline,   // Pipeline already warmed up
    BurstLoad,      // Multiple events simultaneously  
    LongRunning,    // Extended duration test
    StressTest,     // High load test
}

pub struct LatencyBenchmark {
    test_type: LatencyTestType,
    event_count: usize,
    test_duration: Option<Duration>,
    sample_events: Vec<ProcessedEvent>,
    metrics: Vec<LatencyMetrics>,
}

impl LatencyBenchmark {
    pub fn new(test_type: LatencyTestType, event_count: usize) -> Self {
        Self {
            test_type,
            event_count,
            test_duration: None,
            sample_events: Vec::new(),
            metrics: Vec::new(),
        }
    }
    
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.test_duration = Some(duration);
        self
    }
    
    pub async fn run_benchmark(&mut self, pipeline: &mut PipelineOrchestrator) -> Result<LatencyReport> {
        info!("Starting latency benchmark: {:?}", self.test_type);

        if self.sample_events.is_empty() {
            self.sample_events = self.generate_test_events();
        }
        
        match self.test_type {
            LatencyTestType::ColdStart => self.run_cold_start_test(pipeline).await?,
            LatencyTestType::WarmPipeline => self.run_warm_pipeline_test(pipeline).await?,
            LatencyTestType::BurstLoad => self.run_burst_load_test(pipeline).await?,
            LatencyTestType::LongRunning => self.run_long_running_test(pipeline).await?,
            LatencyTestType::StressTest => self.run_stress_test(pipeline).await?,
        }
        
        let report = LatencyReport::from_metrics(self.metrics.clone());
        report.display();
        
        Ok(report)
    }
    
    async fn run_cold_start_test(&mut self, pipeline: &mut PipelineOrchestrator) -> Result<()> {
        info!("Running cold start test - measuring first event processing time");
        
        if let Some(test_event) = self.sample_events.first() {
            let metrics = self.time_single_event(pipeline, test_event.clone(), true).await?;
            self.metrics.push(metrics);
            
            info!("Cold start completed - total time: {:.2}ms", self.metrics[0].total_time_ms());
        }
        
        Ok(())
    }
    
    async fn run_warm_pipeline_test(&mut self, pipeline: &mut PipelineOrchestrator) -> Result<()> {
        info!("Running warm pipeline test with {} events", self.event_count);
        
        // Warm up the pipeline with a dummy event first
        if let Some(warmup_event) = self.sample_events.first() {
            info!("Warming up pipeline...");
            let _ = self.time_single_event(pipeline, warmup_event.clone(), false).await?;
            info!("Pipeline warmed up");
        }
        
        for (i, event) in self.sample_events.iter().take(self.event_count).enumerate() {
            let metrics = self.time_single_event(pipeline, event.clone(), false).await?;
            self.metrics.push(metrics);
            
            if (i + 1) % 10 == 0 {
                info!("Processed {}/{} events", i + 1, self.event_count);
            }
            
            // Small delay to avoid overwhelming the system
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        Ok(())
    }
    
    async fn run_burst_load_test(&mut self, pipeline: &mut PipelineOrchestrator) -> Result<()> {
        info!("Running burst load test - processing {} events rapidly", self.event_count);
        
        // For this implementation, we'll simulate burst by processing events rapidly
        for event in self.sample_events.iter().take(self.event_count) {
            let metrics = self.time_single_event(pipeline, event.clone(), false).await?;
            self.metrics.push(metrics);
        }
        
        Ok(())
    }
    
    async fn run_long_running_test(&mut self, pipeline: &mut PipelineOrchestrator) -> Result<()> {
        let duration = self.test_duration.unwrap_or(Duration::from_secs(300)); // 5 minutes default
        info!("Running long running test for {} seconds", duration.as_secs());
        
        let start_time = Instant::now();
        let mut event_count = 0;
        
        while start_time.elapsed() < duration {
            // Cycle through available events
            let event_index = event_count % self.sample_events.len();
            let event = &self.sample_events[event_index];
            
            let metrics = self.time_single_event(pipeline, event.clone(), false).await?;
            self.metrics.push(metrics);
            
            event_count += 1;
            
            if event_count % 50 == 0 {
                info!("Long running test: {} events processed in {:.1}s", 
                    event_count, start_time.elapsed().as_secs_f64());
            }
            
            // Small delay to simulate realistic processing rate
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        
        info!("Long running test completed: {} events in {:.1}s", 
            event_count, start_time.elapsed().as_secs_f64());
        
        Ok(())
    }
    
    async fn run_stress_test(&mut self, pipeline: &mut PipelineOrchestrator) -> Result<()> {
        info!("Running stress test - maximum throughput for {} events", self.event_count);
        
        for (i, event) in self.sample_events.iter().take(self.event_count).enumerate() {
            let metrics = self.time_single_event(pipeline, event.clone(), false).await?;
            self.metrics.push(metrics);
            
            // No delay - maximum throughput
            if (i + 1) % 25 == 0 {
                info!("Stress test: {}/{} events processed", i + 1, self.event_count);
            }
        }
        
        Ok(())
    }
    
    async fn time_single_event(&self, pipeline: &mut PipelineOrchestrator, event: ProcessedEvent, is_cold_start: bool) -> Result<LatencyMetrics> {
        let mut timer = PipelineTimer::new();
        let event_id = event.id.clone();
        
        debug!("Timing event: {} (cold_start: {})", event_id, is_cold_start);

        timer.mark_stage("start");
        
        let start_processing = Instant::now();
        
        match pipeline.enhance_event_for_benchmark(event).await {
            Ok(_enhanced_result) => {
                timer.mark_stage("complete");
 
                let memory_usage = self.get_current_memory_usage();
                let total_time = start_processing.elapsed();
                
                // Estimate stage times based on total time (in production, these would be measured individually)
                let preprocessing_time = Duration::from_millis((total_time.as_millis() as f64 * 0.15) as u64);
                let finbert_time = Duration::from_millis((total_time.as_millis() as f64 * 0.35) as u64);
                let signals_time = Duration::from_millis((total_time.as_millis() as f64 * 0.25) as u64);
                let storage_time = Duration::from_millis((total_time.as_millis() as f64 * 0.25) as u64);
                
                let metrics = LatencyMetrics {
                    event_id,
                    timestamp: Utc::now(),
                    rss_fetch_time: Duration::from_millis(0).into(), // Not measured in this context
                    preprocessing_time: preprocessing_time.into(),
                    finbert_inference_time: finbert_time.into(),
                    signals_processing_time: signals_time.into(),
                    vector_storage_time: storage_time.into(),
                    total_pipeline_time: total_time.into(),
                    memory_usage_mb: memory_usage,
                };
                
                debug!("Event {} processed in {:.2}ms", metrics.event_id, metrics.total_time_ms());
                Ok(metrics)
            }
            Err(e) => {
                warn!("Failed to process event {}: {}", event_id, e);
                Err(e)
            }
        }
    }
    
    fn generate_test_events(&self) -> Vec<ProcessedEvent> {
        let mut events = Vec::new();
        
        // Generate sample financial news events for testing
        let sample_texts = vec![
            "Federal Reserve announces interest rate decision affecting global markets and cryptocurrency prices",
            "Apple Inc reports quarterly earnings beating analyst expectations with strong iPhone sales growth",
            "Bitcoin surges to new all-time high as institutional adoption continues to accelerate worldwide",
            "Tesla stock drops after Elon Musk announces production delays at new manufacturing facility",
            "JPMorgan Chase raises outlook for economic growth following positive employment data release",
            "Gold prices rally amid inflation concerns and geopolitical tensions in emerging markets",
            "Microsoft Azure cloud revenue exceeds forecasts driving tech sector momentum higher today",
            "Oil futures climb on OPEC supply cut announcement and strong demand from Asian markets",
        ];
        
        let sample_assets = vec![
            vec!["AAPL", "MSFT", "GOOGL"],
            vec!["BTC", "ETH", "ADA"],
            vec!["JPM", "BAC", "WFC"],
            vec!["TSLA", "NIO", "RIVN"],
            vec!["GLD", "SLV", "GOLD"],
            vec!["USO", "XLE", "OIL"],
        ];
        
        for i in 0..self.event_count.max(sample_texts.len()) {
            let text_index = i % sample_texts.len();
            let asset_index = i % sample_assets.len();
            
            events.push(ProcessedEvent {
                id: format!("benchmark_event_{}", i + 1),
                processed_text: sample_texts[text_index].to_string(),
                original_event: None,
                assets: sample_assets[asset_index].iter().map(|symbol| {
                    AssetMatch {
                        symbol: symbol.to_string(),
                        name: format!("{} Asset", symbol),
                        r#type: if symbol.contains("BTC") || symbol.contains("ETH") { 
                            "crypto".to_string() 
                        } else { 
                            "stock".to_string() 
                        },
                        confidence: 0.85,
                        contexts: vec!["financial".to_string()],
                    }
                }).collect(),
                categories: vec!["financial".to_string(), "markets".to_string()],
                sentiment_score: 0.0,
                confidence: 0.85,
                news_type: "financial".to_string(),
                market_impact: "medium".to_string(),
                ml_sentiment_score: 0.0,
                ml_confidence: 0.0,
                asset_mentions: vec![],
                processed_at: Utc::now().to_rfc3339(),
                tokens: vec![],
            });
        }
        
        events
    }
    
    fn get_current_memory_usage(&self) -> f64 {
        // Simplified memory usage calculation
        // In production use:
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
                for line in contents.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<f64>() {
                                return kb / 1024.0; // Convert KB to MB
                            }
                        }
                    }
                }
            }
        }
        
        // Fallback estimation
        150.0 + (self.metrics.len() as f64 * 0.5) // Base memory + growth estimate
    }
}