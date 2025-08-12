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
use crate::unified_event::UnifiedFinancialEvent;
use crate::signals_processing::SignalsProcessingEngine;

use inference::FinBertInference;
use finmedia_signals::{SignalsProcessor, ProcessorConfig};
use vdatabase::{QdrantVectorClient, VectorStorage, FinancialEvent, SignalsResult};

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
    signals_processor: SignalsProcessor,
    formatter: ResultFormatter,
    processed_events_queue: VecDeque<ProcessedEvent>,
    last_health_check: std::time::Instant,
    vector_storage: Option<VectorStorage>,
}

impl PipelineOrchestrator {
    pub async fn new(config_path: PathBuf,
        finbert_config_path: PathBuf,
        preprocessing_addr: String,
        go_binary_path: PathBuf,
    ) -> Result<Self> {
        info!("Initializing Pipeline Orchestrator...");

        let go_service = GoServiceManager::new(go_binary_path);
        let signals_config = ProcessorConfig::default();
        let signals_processor = SignalsProcessor::new(signals_config);
        
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
            signals_processor,
            formatter: ResultFormatter::new(),
            processed_events_queue: VecDeque::new(),
            last_health_check: std::time::Instant::now(),
            vector_storage: None,
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

        let interceptor_addr = format!("localhost:{}", self.interceptor_port);
        self.go_service.start_with_preprocessing_addr(&self.config_path, &interceptor_addr).await?;

        loop {
            if !self.go_service.is_running() {
                warn!("Go service stopped, restarting...");
                println!("Go service stopped, restarting...");
                self.go_service.start(&self.config_path).await?;
            }

            debug!("Processing available results - Queue size: {}", self.queue_size());
            if let Err(e) = self.process_available_results().await {
                error!("Error processing results: {}", e);
                println!("Error processing results: {}", e);
            }

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

        match self.initialize_vector_database().await {
            Ok(()) => {
                info!("Vector database initialized successfully");
            }
            Err(e) => {
                warn!("Vector database initialization failed: {}", e);
                info!("Continuing without vector storage...");
            }
        }

        Ok(())
    }

    async fn initialize_vector_database(&mut self) -> Result<()> {
        info!("Initializing vector database connection...");

        let config = match vdatabase::QdrantConfig::from_env() {
            Ok(config) => {
                info!("Loaded Qdrant config from environment");
                config
            }
            Err(e) => {
                warn!("Failed to load Qdrant config from environment: {}, using defaults", e);
                vdatabase::QdrantConfig::default()
            }
        };
        
        info!("Connecting to Qdrant at: {}", config.url);
        match QdrantVectorClient::new(config).await {
            Ok(client) => {
                let storage = VectorStorage::new(client).await?;
                
                // Ensure collections exist
                if let Err(e) = storage.ensure_collections().await {
                    warn!("Failed to ensure collections exist: {}", e);
                    return Err(e.into());
                }
                
                self.vector_storage = Some(storage);
                info!("Vector database connection established");
                Ok(())
            }
            Err(e) => {
                warn!("Failed to connect to Qdrant: {}", e);
                Err(e.into())
            }
        }
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

        tokio::time::sleep(Duration::from_secs(2)).await;
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
                    let mut unified_event = UnifiedFinancialEvent::from_enhanced_result(enhanced_result.clone());
                    self.print_unified_event(&unified_event, processed_count + 1);
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

    async fn process_available_results(&mut self) -> Result<()> {
        debug!("Processing available results...");

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

        if processed_count == 0 && self.processed_events_queue.is_empty() {
            if let Err(_) = self.try_fetch_real_events().await {
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

    /// Add events to the processing queue (useful for testing or manual injection)
    pub fn add_events_to_queue(&mut self, events: Vec<ProcessedEvent>) {
        for event in events {
            self.processed_events_queue.push_back(event);
        }
        info!("Added {} events to processing queue", self.processed_events_queue.len());
    }

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

        // Apply comprehensive signals processing using dedicated engine
        let signals_engine = SignalsProcessingEngine::new(&self.signals_processor);
        let (signals_score, signals_analysis) = signals_engine.comprehensive_signals_analysis(ml_sentiment, &event).await?;

        // Store both financial event and signals results if vector storage is available
        if self.vector_storage.is_some() {
            info!("Vector storage available - storing event and signals");
            if let Err(e) = self.store_financial_event_with_signals(&event, ml_sentiment, ml_confidence, &signals_analysis, signals_score).await {
                warn!("Failed to store event and signals: {}", e);
            }
        } else {
            warn!("Vector storage not available - skipping database storage");
        }

        Ok(EnhancedResult {
            event,
            ml_sentiment,
            ml_confidence,
            signals_score,
            signals_analysis, // Add signals analysis to result
        })
    }

    async fn store_financial_event(
        &mut self,
        event: &ProcessedEvent,
        ml_sentiment: f64,
        ml_confidence: f64,
        storage: &VectorStorage
    ) -> Result<()> {
        debug!("Storing financial event: {}", event.id);

        // Create FinancialEvent from ProcessedEvent
        let published_at = if let Some(original) = &event.original_event {
            chrono::DateTime::parse_from_rfc3339(&original.published_at)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now())
        } else {
            chrono::Utc::now()
        };

        let mut financial_event = FinancialEvent::new(
            event.original_event.as_ref()
                .map(|e| e.title.clone())
                .unwrap_or_else(|| "Untitled".to_string()),
            event.processed_text.clone(),
            event.original_event.as_ref()
                .map(|e| e.source.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            published_at,
        );

        if let Some(original) = &event.original_event {
            financial_event.url = Some(original.url.clone());
        }

        financial_event.assets = event.assets.iter().map(|asset| {
            vdatabase::AssetInfo {
                symbol: asset.symbol.clone(),
                name: Some(asset.name.clone()),
                asset_type: match asset.r#type.as_str() {
                    "crypto" => vdatabase::AssetType::Crypto,
                    "stock" => vdatabase::AssetType::Stock,
                    "etf" => vdatabase::AssetType::ETF,
                    "commodity" => vdatabase::AssetType::Commodity,
                    "currency" => vdatabase::AssetType::Currency,
                    "bond" => vdatabase::AssetType::Bond,
                    "index" => vdatabase::AssetType::Index,
                    "monetary" => vdatabase::AssetType::Monetary,
                    "economic" => vdatabase::AssetType::Economic,
                    _ => vdatabase::AssetType::Economic,
                },
                confidence: asset.confidence,
                context: asset.contexts.join(", "),
                exchange: None,
            }
        }).collect();

        financial_event.sentiment = vdatabase::SentimentInfo::new(
            ml_sentiment,
            ml_confidence,
            if ml_sentiment > 0.0 { ml_sentiment } else { 0.0 },
            if ml_sentiment < 0.0 { -ml_sentiment } else { 0.0 },
            if ml_sentiment.abs() < 0.1 { 1.0 - ml_sentiment.abs() } else { 0.0 },
        );

        if let Some(ref mut inference) = self.inference_engine {
            if let Ok(embedding) = inference.generate_embedding(&event.processed_text) {
                let _ = financial_event.set_embedding(embedding);
            }
        }

        storage.store_financial_event(&financial_event).await?;
        info!("Stored financial event: {}", financial_event.id);
        
        Ok(())
    }

    async fn store_signals_results(
        &mut self,
        event: &ProcessedEvent,
        signals_analysis: &crate::signals_processing::SignalsAnalysis,
        signals_score: f64,
        storage: &VectorStorage
    ) -> Result<()> {
        debug!("Storing signals results for event: {}", event.id);

        let asset_symbols: Vec<String> = event.assets.iter()
            .map(|a| a.symbol.clone())
            .collect();

        let mut signals_result = SignalsResult::new(event.id.clone(), asset_symbols);

        // Map signals analysis to storage format
        signals_result.scoring_results.final_score = signals_score;
        signals_result.scoring_results.confidence_level = 0.75; // Default confidence
        
        if let Some(volatility) = signals_analysis.volatility_index {
            signals_result.scoring_results.risk_assessment.volatility_risk = volatility;
        }
        
        if let Some(momentum) = signals_analysis.momentum_indicator {
            signals_result.scoring_results.component_scores.momentum_component = momentum;
        }

        // Set trading recommendation based on signals score
        signals_result.scoring_results.trading_recommendation = vdatabase::TradingRecommendation {
            action: if signals_score > 0.2 {
                "buy".to_string()
            } else if signals_score < -0.2 {
                "sell".to_string()
            } else {
                "hold".to_string()
            },
            strength: signals_score.abs(),
            time_horizon: "medium".to_string(),
            confidence: 0.75, // Default confidence
        };

        // Generate signals embedding (simplified - in practice would use signal features)
        let signals_features = format!("score:{} volatility:{} momentum:{}", 
            signals_score, 
            signals_analysis.volatility_index.unwrap_or(0.0),
            signals_analysis.momentum_indicator.unwrap_or(0.0)
        );
        
        if let Some(ref mut inference) = self.inference_engine {
            if let Ok(embedding) = inference.generate_embedding(&signals_features) {
                let _ = signals_result.set_signals_embedding(embedding);
            }
        }

        storage.store_signals_result(&signals_result).await?;
        info!("Stored signals result: {}", signals_result.id);
        
        Ok(())
    }

    async fn store_financial_event_with_signals(
        &mut self,
        event: &ProcessedEvent,
        ml_sentiment: f64,
        ml_confidence: f64,
        signals_analysis: &crate::signals_processing::SignalsAnalysis,
        signals_score: f64,
    ) -> Result<()> {
        // Create storage objects directly without complex borrowing
        self.store_event_and_signals_directly(event, ml_sentiment, ml_confidence, signals_analysis, signals_score).await
    }

    async fn store_event_and_signals_directly(
        &mut self,
        event: &ProcessedEvent,
        ml_sentiment: f64,
        ml_confidence: f64,
        _signals_analysis: &crate::signals_processing::SignalsAnalysis,
        signals_score: f64,
    ) -> Result<()> {
        if self.vector_storage.is_some() {
            if let Err(e) = self.create_and_store_financial_event(event, ml_sentiment, ml_confidence).await {
                warn!("Failed to store financial event: {}", e);
            }

            if let Err(e) = self.create_and_store_signals_results(event, signals_score).await {
                warn!("Failed to store signals results: {}", e);
            }
        }
        
        Ok(())
    }

    async fn create_and_store_financial_event(
        &mut self,
        event: &ProcessedEvent,
        ml_sentiment: f64,
        ml_confidence: f64,
    ) -> Result<()> {
        debug!("Creating and storing financial event: {}", event.id);

        // Create FinancialEvent from ProcessedEvent (simplified version)
        let published_at = if let Some(original) = &event.original_event {
            chrono::DateTime::parse_from_rfc3339(&original.published_at)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now())
        } else {
            chrono::Utc::now()
        };

        let mut financial_event = FinancialEvent::new(
            event.original_event.as_ref()
                .map(|e| e.title.clone())
                .unwrap_or_else(|| "Untitled".to_string()),
            event.processed_text.clone(),
            event.original_event.as_ref()
                .map(|e| e.source.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            published_at,
        );

        // Set basic sentiment
        financial_event.sentiment = vdatabase::SentimentInfo::new(
            ml_sentiment,
            ml_confidence,
            if ml_sentiment > 0.0 { ml_sentiment } else { 0.0 },
            if ml_sentiment < 0.0 { -ml_sentiment } else { 0.0 },
            if ml_sentiment.abs() < 0.1 { 1.0 - ml_sentiment.abs() } else { 0.0 },
        );

        if let Some(ref storage) = self.vector_storage {
            storage.store_financial_event(&financial_event).await?;
            info!("Stored financial event: {}", financial_event.id);
        }
        
        Ok(())
    }

    async fn create_and_store_signals_results(
        &mut self,
        event: &ProcessedEvent,
        signals_score: f64,
    ) -> Result<()> {
        debug!("Creating and storing signals results for event: {}", event.id);

        let asset_symbols: Vec<String> = event.assets.iter()
            .map(|a| a.symbol.clone())
            .collect();

        let mut signals_result = SignalsResult::new(event.id.clone(), asset_symbols);

        signals_result.scoring_results.final_score = signals_score;
        signals_result.scoring_results.confidence_level = 0.75;

        if let Some(ref storage) = self.vector_storage {
            storage.store_signals_result(&signals_result).await?;
            info!("Stored signals result: {}", signals_result.id);
        }
        
        Ok(())
    }

    fn print_unified_event(&self, unified_event: &UnifiedFinancialEvent, event_number: usize) {
        println!("\n================================================================================");
        println!("🔍 UNIFIED FINANCIAL EVENT #{}", event_number);
        println!("================================================================================");
        
        println!("📰 Event ID: {}", unified_event.id);
        println!("📰 Title: {}", unified_event.title);
        println!("📝 Content: {}", unified_event.content.chars().take(100).collect::<String>() + "...");
        println!("📝 Processed Text: {}", unified_event.processed_text.chars().take(100).collect::<String>() + "...");
        println!("🕐 Published: {}", unified_event.published_at.format("%Y-%m-%d %H:%M:%S UTC"));
        println!("🕐 Processed: {}", unified_event.processed_at.format("%Y-%m-%d %H:%M:%S UTC"));
        println!("📰 Source: {}", unified_event.source);
        println!("🔗 URL: {}", unified_event.url);
        
        println!("\n----------------------------------------");
        println!("🏷️ CLASSIFICATION");
        println!("----------------------------------------");
        println!("📊 Categories: {:?}", unified_event.categories);
        println!("📊 News Type: {}", unified_event.news_type);
        println!("💹 Market Impact: {}", unified_event.market_impact);
        

        println!("\n----------------------------------------");
        println!("🎯 DETECTED ASSETS ({} total)", unified_event.assets.len());
        println!("----------------------------------------");
        for (i, asset) in unified_event.assets.iter().enumerate() {
            println!("  {}. {} ({}) - {:.2}% confidence", 
                i + 1, asset.symbol, asset.asset_type, asset.confidence * 100.0);
        }
        
        if let Some(primary) = unified_event.primary_asset() {
            println!("🥇 Primary Asset: {} ({:.2}% confidence)", primary.symbol, primary.confidence * 100.0);
        }
        
        println!("\n----------------------------------------");
        println!("📊 SENTIMENT ANALYSIS (Multi-Layer)");
        println!("----------------------------------------");
        println!("🔵 Go Service:");
        println!("   Sentiment: {:.3} {}", 
            unified_event.sentiment_layers.go_sentiment,
            self.sentiment_emoji(unified_event.sentiment_layers.go_sentiment)
        );
        println!("   Confidence: {:.3}", unified_event.sentiment_layers.go_confidence);
        
        println!("🤖 ML Inference (FinBERT):");
        println!("   Sentiment: {:.3} {}", 
            unified_event.sentiment_layers.ml_sentiment,
            self.sentiment_emoji(unified_event.sentiment_layers.ml_sentiment)
        );
        println!("   Confidence: {:.3}", unified_event.sentiment_layers.ml_confidence);
        
        println!("📡 Signals Processing:");
        println!("   Final Score: {:.3} {}", 
            unified_event.sentiment_layers.signals_score,
            self.sentiment_emoji(unified_event.sentiment_layers.signals_score)
        );
        
        println!("\n----------------------------------------");
        println!("💰 TRADING SIGNALS");
        println!("----------------------------------------");
        println!("📈 Trading Signal: {:.3} {}", 
            unified_event.get_trading_signal(),
            self.sentiment_emoji(unified_event.get_trading_signal())
        );
        println!("🚨 High Impact Event: {}", if unified_event.is_high_impact() { "YES 🔥" } else { "NO 💧" });
        
        if let Some(volatility) = unified_event.signals_analysis.volatility_index {
            println!("📊 Volatility Index: {:.3}", volatility);
        }
        if let Some(momentum) = unified_event.signals_analysis.momentum_indicator {
            println!("📈 Momentum Indicator: {:.3} {}", momentum, self.sentiment_emoji(momentum));
        }
        
        // Tokenization and NLP
        println!("\n----------------------------------------");
        println!("🔤 TOKENIZATION & NLP");
        println!("----------------------------------------");
        
        // TODO: ENHANCED NLP DISPLAY IMPLEMENTATION
        // =========================================
        // Current implementation shows basic empty token arrays.
        // When preprocessing/src/processor.rs tokenization is implemented,
        // enhance this display to show:
        //
        // RICH NLP ANALYSIS DISPLAY:
        // -------------------------
        // ```
        // 📊 Token Statistics:
        //    Total: 45 | Unique: 32 | Financial: 18 | Relevance: 87.3%
        // 
        // 🔤 Top Financial Tokens: ["federal", "reserve", "interest", "rates", "monetary", "policy"]
        //    General Tokens: ["announced", "today", "economic", "growth"]
        //
        // 🏷️ Named Entities:
        //    • Federal Reserve (Organization): 95.2% confidence
        //    • Jerome Powell (Person): 89.1% confidence
        //    • Wall Street (Location): 76.4% confidence
        //
        // 💰 Financial Terms by Category:
        //    • Monetary Policy: ["interest rates", "quantitative easing", "fed funds"]
        //    • Market Terms: ["bull market", "resistance level", "volatility"]
        //    • Economic Indicators: ["GDP", "inflation", "unemployment"]
        //
        // 🎯 ML-Detected Asset Mentions:
        //    • "tech giants" → [AAPL, MSFT, GOOGL] (confidence: 0.87)
        //    • "crypto market" → [CRYPTO_MARKET, BTC, ETH] (confidence: 0.92)
        //
        // 📈 Sentiment Keywords:
        //    • Positive: ["surge", "growth", "optimistic"] (3 tokens, avg: +0.71)
        //    • Negative: ["concerns", "decline", "uncertainty"] (3 tokens, avg: -0.58)
        //    • Neutral: ["announced", "reported", "stated"] (3 tokens)
        //
        // 📊 NLP Quality Metrics:
        //    • Tokenization Quality: 94.2%
        //    • Financial Relevance: 87.3%  
        //    • Text Complexity: 72.1%
        //    • Entity Recognition: 91.7%
        // ```
        //
        // CURRENT BASIC DISPLAY (will be enhanced):
        println!("🔤 Tokens ({}): {:?}", unified_event.tokens.len(), 
            unified_event.tokens.iter().take(10).collect::<Vec<_>>());
        if unified_event.tokens.len() > 10 {
            println!("   ... and {} more tokens", unified_event.tokens.len() - 10);
        }
        
        if !unified_event.asset_mentions.is_empty() {
            println!("💰 Asset Mentions: {:?}", unified_event.asset_mentions);
        }
        
        // TODO: When tokens are populated, add:
        // - Token quality and relevance statistics
        // - Named entity extraction results  
        // - Financial term categorization
        // - Sentiment-bearing word analysis
        // - ML-enhanced asset mention detection
        // - NLP processing quality metrics
        
        println!("================================================================================\n");
    }

    fn sentiment_emoji(&self, score: f64) -> &'static str {
        match score {
            s if s > 0.5 => "🚀",
            s if s > 0.2 => "📈",
            s if s > -0.2 => "➡️",
            s if s > -0.5 => "📉",
            _ => "🔻",
        }
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

    pub async fn initialize_for_benchmark(&mut self) -> Result<()> {
        info!("Initializing services for benchmarking...");
        self.initialize_services().await?;
        info!("Services initialized for benchmark");
        Ok(())
    }

    pub async fn enhance_event_for_benchmark(&mut self, event: ProcessedEvent) -> Result<EnhancedResult> {
        self.enhance_event(event).await
    }
}

#[derive(Clone)]
pub struct EnhancedResult {
    pub event: ProcessedEvent,
    pub ml_sentiment: f64,
    pub ml_confidence: f64,
    pub signals_score: f64,
    pub signals_analysis: crate::signals_processing::SignalsAnalysis,
}

impl Drop for PipelineOrchestrator {
    fn drop(&mut self) {
        if let Some(mut preprocessing_service) = self.preprocessing_service.take() {
            let _ = preprocessing_service.start_kill();
        }
    }
}