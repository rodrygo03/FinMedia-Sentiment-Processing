use crate::{config::Config, models::{NewsEvent, ProcessedEvent}, processor::TextProcessor, Result};
use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{error, info, instrument};

// Generated protobuf code will be included here
pub mod preprocessing {
    tonic::include_proto!("preprocessing");
}

use preprocessing::{
    preprocessing_service_server::{PreprocessingService, PreprocessingServiceServer},
    *,
};

/// gRPC server for preprocessing service
pub struct PreprocessingServer {
    config: Config,
    processor: Arc<TextProcessor>,
    start_time: std::time::Instant,
}

impl PreprocessingServer {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            processor: Arc::new(TextProcessor::new()),
            start_time: std::time::Instant::now(),
        }
    }

    pub async fn serve(self) -> Result<()> {
        let addr = self.config.server_address().parse()?;
        
        info!("Starting gRPC server on {}", addr);

        Server::builder()
            .add_service(PreprocessingServiceServer::new(self))
            .serve(addr)
            .await?;

        Ok(())
    }
}

#[tonic::async_trait]
impl PreprocessingService for PreprocessingServer {
    #[instrument(skip(self, request))]
    async fn process_news_event(
        &self,
        request: Request<NewsEventRequest>,
    ) -> std::result::Result<Response<ProcessedEventResponse>, Status> {
        let req = request.into_inner();
        
        let news_event = match req.event {
            Some(event) => convert_proto_to_news_event(event),
            None => {
                return Ok(Response::new(ProcessedEventResponse {
                    processed_event: None,
                    success: false,
                    error_message: "No event provided".to_string(),
                }));
            }
        };

        match self.processor.process_event(news_event).await {
            Ok(processed_event) => {
                let proto_event = convert_processed_event_to_proto(processed_event);
                Ok(Response::new(ProcessedEventResponse {
                    processed_event: Some(proto_event),
                    success: true,
                    error_message: String::new(),
                }))
            }
            Err(e) => {
                error!("Failed to process event: {}", e);
                Ok(Response::new(ProcessedEventResponse {
                    processed_event: None,
                    success: false,
                    error_message: e.to_string(),
                }))
            }
        }
    }

    #[instrument(skip(self, request))]
    async fn process_batch(
        &self,
        request: Request<BatchRequest>,
    ) -> std::result::Result<Response<BatchResponse>, Status> {
        let req = request.into_inner();
        
        let news_events: Vec<NewsEvent> = req
            .events
            .into_iter()
            .map(convert_proto_to_news_event)
            .collect();

        let results = self.processor.process_batch(news_events).await;
        
        let mut processed_events = Vec::new();
        let mut error_messages = Vec::new();
        let mut total_failed = 0;

        for result in results {
            match result {
                Ok(processed_event) => {
                    processed_events.push(convert_processed_event_to_proto(processed_event));
                }
                Err(e) => {
                    total_failed += 1;
                    error_messages.push(e.to_string());
                }
            }
        }

        let total_processed = processed_events.len() as i32;
        
        Ok(Response::new(BatchResponse {
            processed_events,
            total_processed,
            total_failed,
            error_messages,
        }))
    }

    #[instrument(skip(self, _request))]
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> std::result::Result<Response<HealthResponse>, Status> {
        let uptime = self.start_time.elapsed().as_secs() as i64;
        
        Ok(Response::new(HealthResponse {
            status: "healthy".to_string(),
            version: crate::VERSION.to_string(),
            uptime_seconds: uptime,
        }))
    }
}

/// Convert protobuf NewsEvent to internal NewsEvent
fn convert_proto_to_news_event(proto_event: preprocessing::NewsEvent) -> NewsEvent {
    let published_at = chrono::DateTime::parse_from_rfc3339(&proto_event.published_at)
        .unwrap_or_else(|_| chrono::Utc::now().into())
        .with_timezone(&chrono::Utc);

    // Convert protobuf AssetMatch to internal AssetMatch
    let assets: Vec<crate::models::AssetMatch> = proto_event.assets
        .into_iter()
        .map(|proto_asset| crate::models::AssetMatch {
            symbol: proto_asset.symbol,
            name: proto_asset.name,
            asset_type: proto_asset.r#type,
            confidence: proto_asset.confidence,
            contexts: proto_asset.contexts,
        })
        .collect();

    NewsEvent {
        id: proto_event.id,
        title: proto_event.title,
        content: proto_event.content,
        published_at,
        source: proto_event.source,
        url: proto_event.url,
        assets,
        categories: proto_event.categories,
        sentiment: proto_event.sentiment,
        confidence: proto_event.confidence,
        news_type: proto_event.news_type,
        market_impact: proto_event.market_impact,
    }
}

/// Convert internal ProcessedEvent to protobuf ProcessedEvent
fn convert_processed_event_to_proto(processed_event: ProcessedEvent) -> preprocessing::ProcessedEvent {
    // Convert AssetMatch from internal model to protobuf
    let proto_assets: Vec<preprocessing::AssetMatch> = processed_event.original_event.assets
        .into_iter()
        .map(|asset| preprocessing::AssetMatch {
            symbol: asset.symbol,
            name: asset.name,
            r#type: asset.asset_type, // Use raw identifier for 'type' keyword
            confidence: asset.confidence,
            contexts: asset.contexts,
        })
        .collect();

    let original_proto = preprocessing::NewsEvent {
        id: processed_event.original_event.id.clone(),
        title: processed_event.original_event.title.clone(),
        content: processed_event.original_event.content.clone(),
        published_at: processed_event.original_event.published_at.to_rfc3339(),
        source: processed_event.original_event.source.clone(),
        url: processed_event.original_event.url.clone(),
        assets: proto_assets,
        categories: processed_event.original_event.categories,
        sentiment: processed_event.original_event.sentiment,
        confidence: processed_event.original_event.confidence,
        news_type: processed_event.original_event.news_type,
        market_impact: processed_event.original_event.market_impact,
    };

    // Convert AssetMatch for top-level ProcessedEvent fields
    let processed_assets: Vec<preprocessing::AssetMatch> = processed_event.assets
        .into_iter()
        .map(|asset| preprocessing::AssetMatch {
            symbol: asset.symbol,
            name: asset.name,
            r#type: asset.asset_type,
            confidence: asset.confidence,
            contexts: asset.contexts,
        })
        .collect();

    preprocessing::ProcessedEvent {
        id: processed_event.id,
        original_event: Some(original_proto),
        processed_text: processed_event.processed_text,
        tokens: processed_event.tokens,
        
        // Top-level Go service fields - NO INFORMATION LOSS
        assets: processed_assets,
        categories: processed_event.categories,
        sentiment_score: processed_event.sentiment_score,
        confidence: processed_event.confidence,
        news_type: processed_event.news_type,
        market_impact: processed_event.market_impact,
        
        // ML processing fields
        ml_sentiment_score: processed_event.ml_sentiment_score,
        ml_confidence: processed_event.ml_confidence,
        asset_mentions: processed_event.asset_mentions,
        
        processed_at: processed_event.processed_at.to_rfc3339(),
    }
}