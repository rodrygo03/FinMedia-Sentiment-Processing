use anyhow::Result;
use std::path::PathBuf;
use std::time::Duration;
use tracing::{info, debug, error, warn};
use tokio::time::sleep;
use tokio::process::{Child, Command};
use std::collections::VecDeque;
use std::sync::Arc;

use crate::client::{PreprocessingClient, preprocessing::*};
use crate::go_service::GoServiceManager;
use crate::output::ResultFormatter;
use crate::interceptor_server::InterceptorServer;

// Import our analysis crates
use inference::FinBertInference;

pub struct PipelineOrchestrator {
    config_path: PathBuf,
    finbert_config_path: PathBuf,
    preprocessing_addr: String,
    interceptor_port: u16,
    go_service: GoServiceManager,
    preprocessing_service: Option<Child>,
    preprocessing_client: Option<PreprocessingClient>,
    interceptor_server: Option<Arc<InterceptorServer>>,
    inference_engine: Option<FinBertInference>,
    formatter: ResultFormatter,
    processed_events_queue: VecDeque<ProcessedEvent>,
    last_health_check: std::time::Instant,
}

impl PipelineOrchestrator {
    pub async fn new(config_path: PathBuf,
        finbert_config_path: PathBuf,
        preprocessing_addr: String,
        go_binary_path: PathBuf,
    ) -> Result<Self> {
        info!("Initializing Pipeline Orchestrator...");

        let go_service = GoServiceManager::new(go_binary_path);
        Ok(Self {
            config_path,
            finbert_config_path,
            preprocessing_addr,
            interceptor_port: 50052, // Use different port for interceptor
            go_service,
            preprocessing_service: None,
            preprocessing_client: None,
            interceptor_server: None,
            inference_engine: None,
            formatter: ResultFormatter::new(),
            processed_events_queue: VecDeque::new(),
            last_health_check: std::time::Instant::now(),
        })
    }

    pub async fn run_once(&mut self, _feed_url: Option<String>) -> Result<()> {
        info!("Starting single pipeline run...");
        self.formatter.display_pipeline_start();

        self.initialize_services().await?;
        self.start_preprocessing_service().await?;
        self.start_interceptor_server().await?;

        let run_duration = Duration::from_secs(5);
        let interceptor_addr = format!("localhost:{}", self.interceptor_port);
        info!("Connecting Go service to interceptor at {}", interceptor_addr);
        
        self.go_service.start_with_preprocessing_addr(&self.config_path, &interceptor_addr).await?;
        
        info!("Waiting {}s for RSS feeds to be fetched and processed...", run_duration.as_secs());
        tokio::time::sleep(run_duration).await;
        
        info!("Stopping Go service...");
        self.go_service.stop().await?;
        sleep(Duration::from_secs(5)).await;

        self.process_real_events().await?;

        info!("***| Single pipeline run completed |***");
        self.formatter.display_pipeline_complete();
        self.cleanup_services().await?;
        
        Ok(())
    }

    pub async fn run_continuous(&mut self) -> Result<()> {
        info!("Starting continuous pipeline...");
        
        self.formatter.display_pipeline_start();
        self.initialize_services().await?;
        self.start_preprocessing_service().await?;
        self.start_interceptor_server().await?;

        // Start Go service (it will connect to our interceptor)
        let interceptor_addr = format!("localhost:{}", self.interceptor_port);
        self.go_service.start_with_preprocessing_addr(&self.config_path, &interceptor_addr).await?;

        loop {
            // Check if Go service is still running
            if !self.go_service.is_running() {
                warn!("Go service stopped, restarting...");
                println!("Go service stopped, restarting...");
                self.go_service.start(&self.config_path).await?;
            }

            // Process any available results
            debug!("Processing available results - Queue size: {}", self.queue_size());
            if let Err(e) = self.process_available_results().await {
                error!("Error processing results: {}", e);
                println!("Error processing results: {}", e);
            }

            // Wait before next check
            sleep(Duration::from_secs(5)).await;
        }
    }

    async fn initialize_services(&mut self) -> Result<()> {
        info!("Initializing services...");

        match FinBertInference::new(&self.finbert_config_path) {
            Ok(engine) => {
                info!("FinBERT inference engine initialized");
                self.inference_engine = Some(engine);
            }
            Err(e) => {
                warn!("FinBERT initialization failed: {}", e);
                info!("Continuing without ML inference...");
            }
        }

        Ok(())
    }

    async fn start_preprocessing_service(&mut self) -> Result<()> {
        info!("Starting preprocessing service...");
        let preprocessing_dir = PathBuf::from("../preprocessing");
        
        info!("Building preprocessing service...");
        let build_output = Command::new("cargo")
            .args(&["build", "--release"])
            .current_dir(&preprocessing_dir)
            .output()
            .await?;

        if !build_output.status.success() {
            let stderr = String::from_utf8_lossy(&build_output.stderr);
            anyhow::bail!("Failed to build preprocessing service: {}", stderr);
        }

        info!("Starting preprocessing service binary...");
        let child = Command::new("cargo")
            .args(&["run", "--release"])
            .current_dir(&preprocessing_dir)
            .spawn()?;
        self.preprocessing_service = Some(child);

        info!("Waiting for preprocessing service to be ready...");
        tokio::time::sleep(Duration::from_secs(3)).await;


        let mut attempts = 0;
        loop {
            match PreprocessingClient::new(&self.preprocessing_addr).await {
                Ok(client) => {
                    self.preprocessing_client = Some(client);
                    info!("Preprocessing client init");
                    break;
                }
                Err(e) if attempts < 10 => {
                    attempts += 1;
                    info!("Attempt {}/10: Waiting for preprocessing service: {}", attempts, e);
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
                Err(e) => return Err(e),
            }
        }

        if let Some(ref mut client) = self.preprocessing_client {
            client.wait_for_service(5).await?;
        }

        info!("Preprocessing service started successfully");
        Ok(())
    }

    async fn start_interceptor_server(&mut self) -> Result<()> {
        info!("Starting interceptor server...");
        let interceptor = InterceptorServer::new();
        
        if let Some(ref _preprocessing_client) = self.preprocessing_client {
            println!("Connecting interceptor to real preprocessing service at {}", self.preprocessing_addr);
            let interceptor_client = PreprocessingClient::new(&self.preprocessing_addr).await?;
            interceptor.set_preprocessing_client(interceptor_client).await;
            println!("Interceptor connected to preprocessing service");
        } else {
            println!("No preprocessing client available to connect interceptor");
        }

        let interceptor_arc = Arc::new(interceptor);
        let server_interceptor = Arc::clone(&interceptor_arc);
        let port = self.interceptor_port;
        
        tokio::spawn(async move {
            if let Err(e) = server_interceptor.start_server(port).await {
                error!("Interceptor server failed: {}", e);
            }
        });

        // Give the server time to start
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Keep the Arc reference instead of trying to unwrap it
        self.interceptor_server = Some(Arc::clone(&interceptor_arc));

        info!("Interceptor server started on port {}", self.interceptor_port);
        Ok(())
    }

    async fn process_real_events(&mut self) -> Result<()> {
        info!("Processing real events from RSS feeds...");

        if let Err(_) = self.try_fetch_real_events().await {
            warn!("  No real events captured, this may be due to:");
            warn!("   - RSS feeds not returning data");
            warn!("   - Network connectivity issues");
            warn!("   - Go service configuration issues");
            println!("  No real events captured - check RSS feeds and network connectivity");
            
            return Ok(());
        }

        let mut processed_count = 0;
        while let Some(event) = self.processed_events_queue.pop_front() {
            info!("Processing real RSS event: {}", event.id);
            
            match self.enhance_event(event).await {
                Ok(enhanced_result) => {
                    self.formatter.display_result(&enhanced_result);
                    processed_count += 1;
                }
                Err(e) => {
                    error!("Failed to conduct inference on event: {}", e);
                }
            }
        }

        if processed_count > 0 {
            info!("Successfully processed {} RSS events", processed_count);
        } else {
            info!(" No events were ready for processing at this time");
        }

        Ok(())
    }

    // async fn process_mock_results(&mut self) -> Result<()> {
    //     info!("📊 Processing mock results for demonstration...");

    //     // Create some mock processed events
    //     let mock_events = self.create_mock_events();

    //     for event in mock_events {
    //         let enhanced_result = self.enhance_event(event).await?;
    //         self.formatter.display_result(&enhanced_result);
    //     }

    //     Ok(())
    // }

    async fn process_available_results(&mut self) -> Result<()> {
        debug!("Processing available results...");

        // Check health of preprocessing service periodically
        if self.last_health_check.elapsed() > Duration::from_secs(30) {
            if let Some(ref mut client) = self.preprocessing_client {
                match client.health_check().await {
                    Ok(true) => {
                        debug!("Preprocessing service health check: OK");
                    }
                    Ok(false) => {
                        warn!("Preprocessing service health check: UNHEALTHY");
                    }
                    Err(e) => {
                        error!("Preprocessing service health check failed: {}", e);
                        return Err(e.into());
                    }
                }
                self.last_health_check = std::time::Instant::now();
            }
        }

        // Process any queued events
        let mut processed_count = 0;
        while let Some(event) = self.processed_events_queue.pop_front() {
            match self.enhance_event(event).await {
                Ok(enhanced_result) => {
                    self.formatter.display_result(&enhanced_result);
                    processed_count += 1;
                }
                Err(e) => {
                    error!("Failed to enhance event: {}", e);
                }
            }
        }

        // Try to fetch real events from preprocessing service (if available)
        // Otherwise, generate mock events for demonstration
        if processed_count == 0 && self.processed_events_queue.is_empty() {
            if let Err(_) = self.try_fetch_real_events().await {
                // Fallback to simulation if real events unavailable
                println!("fellback")
            }
        }

        if processed_count > 0 {
            info!("Processed {} events", processed_count);
        }

        Ok(())
    }

    async fn try_fetch_real_events(&mut self) -> Result<()> {
        debug!("Attempting to fetch events from interceptor server...");
        
        if let Some(ref interceptor_arc) = self.interceptor_server {
            let captured_events = interceptor_arc.get_captured_events().await;
            
            info!("Interceptor has {} captured events", captured_events.len());
            
            if !captured_events.is_empty() {
                info!("📦 Retrieved {} real events from interceptor", captured_events.len());
                
                // Add captured events to our processing queue
                for (i, event) in captured_events.into_iter().enumerate() {
                    println!("   Event {}: {} ({})", i + 1, event.id, event.processed_text.chars().take(50).collect::<String>());
                    self.processed_events_queue.push_back(event);
                }
                
                return Ok(());
            } else {
                println!("No events captured by interceptor");
            }
        } else {
            println!("No interceptor server available");
        }
        
        Err(anyhow::anyhow!("No revents available from interceptor"))
    }

    async fn simulate_incoming_events(&mut self) -> Result<()> {
        // Simulate receiving processed events from the preprocessing service
        // In a real implementation, this would be replaced by actual event polling/streaming
        
        static mut SIMULATION_COUNTER: u32 = 0;
        unsafe {
            SIMULATION_COUNTER += 1;
            
            // // Only generate mock events every 5th call to avoid spam
            // if SIMULATION_COUNTER % 5 == 0 {
            //     debug!("Generating mock processed events...");
            //     let mock_events = self.create_mock_events_with_variation(SIMULATION_COUNTER);
                
            //     for event in mock_events {
            //         self.processed_events_queue.push_back(event);
            //     }
                
            //     info!("Added {} mock events to processing queue", self.processed_events_queue.len());
            // }
        }
        
        Ok(())
    }

    // fn create_mock_events_with_variation(&self, counter: u32) -> Vec<ProcessedEvent> {
    //     let event_templates = vec![
    //         ("Bitcoin reaches new price milestone", "Bitcoin price surges past $50,000 amid institutional adoption and growing retail interest", vec!["crypto".to_string()], 0.7),
    //         ("Federal Reserve policy update", "Federal Reserve maintains current interest rates while signaling potential future adjustments based on economic indicators", vec!["monetary".to_string()], -0.1),
    //         ("Tech stock earnings report", "Major technology companies report mixed quarterly earnings with some beating expectations while others fall short", vec!["stock".to_string()], 0.2),
    //         ("Economic uncertainty rises", "Market volatility increases as investors react to geopolitical tensions and inflation concerns", vec!["economic".to_string()], -0.4),
    //         ("Cryptocurrency regulation news", "New regulatory framework proposed for digital assets aims to provide clarity while maintaining innovation", vec!["crypto".to_string()], 0.1),
    //     ];

    //     let template_index = (counter as usize) % event_templates.len();
    //     let (title, content, categories, sentiment) = &event_templates[template_index];

    //     vec![ProcessedEvent {
    //         id: format!("event-{}-{}", counter, template_index),
    //         original_event: None,
    //         processed_text: format!("{} {}", title, content),
    //         tokens: content.split_whitespace().take(5).map(|s| s.to_lowercase()).collect(),
    //         assets: vec![],
    //         categories: categories.clone(),
    //         sentiment_score: *sentiment,
    //         confidence: 0.75 + (counter as f64 * 0.01) % 0.2, // Vary confidence between 0.75-0.95
    //         news_type: "financial".to_string(),
    //         market_impact: if sentiment.abs() > 0.3 { "high" } else { "medium" }.to_string(),
    //         ml_sentiment_score: 0.0,
    //         ml_confidence: 0.0,
    //         asset_mentions: vec![],
    //         processed_at: chrono::Utc::now().to_rfc3339(),
    //     }]
    // }

    /// Add events to the processing queue (useful for testing or manual injection)
    pub fn add_events_to_queue(&mut self, events: Vec<ProcessedEvent>) {
        for event in events {
            self.processed_events_queue.push_back(event);
        }
        info!("Added {} events to processing queue", self.processed_events_queue.len());
    }

    /// Get current queue size for monitoring
    pub fn queue_size(&self) -> usize {
        self.processed_events_queue.len()
    }

    async fn enhance_event(&mut self, event: ProcessedEvent) -> Result<EnhancedResult> {
        debug!("Enhancing event: {}", event.id);

        let mut ml_sentiment = 0.0;
        let mut ml_confidence = 0.0;

        // Apply ML inference if available
        if let Some(ref mut inference) = self.inference_engine {
            match inference.enhance_processed_event(&event.processed_text) {
                Ok((sentiment, confidence)) => {
                    ml_sentiment = sentiment;
                    ml_confidence = confidence;
                    debug!("ML analysis: sentiment={:.3}, confidence={:.3}", sentiment, confidence);
                }
                Err(e) => {
                    warn!("ML inference failed: {}", e);
                }
            }
        }

        // Apply signals processing
        let signals_score = self.apply_signals_processing(ml_sentiment, &event).await?;

        Ok(EnhancedResult {
            event,
            ml_sentiment,
            ml_confidence,
            signals_score,
        })
    }

    async fn apply_signals_processing(&self, sentiment_score: f64, event: &ProcessedEvent) -> Result<f64> {
        // Apply signals and systems processing to the sentiment score
        // Since the signals crate functions are not implemented yet, we'll do basic processing
        
        // Simple smoothing (placeholder for future filter implementation)
        let smoothed_score = sentiment_score * 0.9 + event.sentiment_score * 0.1;
        
        // Basic normalization (placeholder for future scoring implementation)
        let final_score = smoothed_score.max(-1.0).min(1.0);
        
        debug!("Signals processing: {} -> {} -> {}", sentiment_score, smoothed_score, final_score);
        
        Ok(final_score)
    }

    async fn cleanup_services(&mut self) -> Result<()> {
        info!("Cleaning up services...");
        self.go_service.stop().await?;
        
        if let Some(mut preprocessing_service) = self.preprocessing_service.take() {
            info!("Stopping preprocessing service...");
            let _ = preprocessing_service.kill().await;
            let _ = preprocessing_service.wait().await;
            info!("Preprocessing service stopped");
        }
        
        Ok(())
    }

    // fn create_mock_events(&self) -> Vec<ProcessedEvent> {
    //     vec![
    //         ProcessedEvent {
    //             id: "mock-1".to_string(),
    //             original_event: None,
    //             processed_text: "Bitcoin price surges amid institutional adoption and positive market sentiment".to_string(),
    //             tokens: vec!["bitcoin".to_string(), "price".to_string(), "surges".to_string()],
    //             assets: vec![],
    //             categories: vec!["crypto".to_string()],
    //             sentiment_score: 0.7,
    //             confidence: 0.85,
    //             news_type: "financial".to_string(),
    //             market_impact: "high".to_string(),
    //             ml_sentiment_score: 0.0,
    //             ml_confidence: 0.0,
    //             asset_mentions: vec![],
    //             processed_at: chrono::Utc::now().to_rfc3339(),
    //         },
    //         ProcessedEvent {
    //             id: "mock-2".to_string(),
    //             original_event: None,
    //             processed_text: "Federal Reserve maintains interest rates amid economic uncertainty and inflation concerns".to_string(),
    //             tokens: vec!["federal".to_string(), "reserve".to_string(), "interest".to_string()],
    //             assets: vec![],
    //             categories: vec!["monetary".to_string()],
    //             sentiment_score: -0.2,
    //             confidence: 0.75,
    //             news_type: "economic".to_string(),
    //             market_impact: "medium".to_string(),
    //             ml_sentiment_score: 0.0,
    //             ml_confidence: 0.0,
    //             asset_mentions: vec![],
    //             processed_at: chrono::Utc::now().to_rfc3339(),
    //         },
    //     ]
    // }
}

pub struct EnhancedResult {
    pub event: ProcessedEvent,
    pub ml_sentiment: f64,
    pub ml_confidence: f64,
    pub signals_score: f64,
}

impl Drop for PipelineOrchestrator {
    fn drop(&mut self) {
        // Best effort cleanup on drop
        if let Some(mut preprocessing_service) = self.preprocessing_service.take() {
            let _ = preprocessing_service.start_kill();
        }
    }
}