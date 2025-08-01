use anyhow::Result;
use tonic::transport::Channel;
use tracing::{info, debug, error};

// Import the generated protobuf types from preprocessing service
pub mod preprocessing {
    tonic::include_proto!("preprocessing");
}

use preprocessing::{
    preprocessing_service_client::PreprocessingServiceClient,
    NewsEvent, ProcessedEvent, NewsEventRequest, BatchRequest,
    HealthRequest,
};

pub struct PreprocessingClient {
    client: PreprocessingServiceClient<Channel>,
}

impl PreprocessingClient {
    pub async fn new(address: &str) -> Result<Self> {
        info!("Connecting to preprocessing service at {}", address);
        
        let channel = Channel::from_shared(address.to_string())?
            .connect()
            .await?;

        let client = PreprocessingServiceClient::new(channel);

        info!("Connected to preprocessing service");
        Ok(Self { client })
    }

    pub async fn health_check(&mut self) -> Result<bool> {
        debug!("Performing health check...");
        
        let request = tonic::Request::new(HealthRequest {});
        
        match self.client.health(request).await {
            Ok(response) => {
                let health = response.into_inner();
                debug!("Health check response: {} (uptime: {}s)", 
                    health.status, health.uptime_seconds);
                Ok(health.status == "healthy")
            }
            Err(e) => {
                error!("Health check failed: {}", e);
                Ok(false)
            }
        }
    }

    pub async fn process_event(&mut self, event: NewsEvent) -> Result<ProcessedEvent> {
        debug!("Processing single event: {}", event.id);

        let request = tonic::Request::new(NewsEventRequest {
            event: Some(event),
        });

        let response = self.client.process_news_event(request).await?;
        let processed_response = response.into_inner();

        if !processed_response.success {
            anyhow::bail!("Processing failed: {}", processed_response.error_message);
        }

        processed_response
            .processed_event
            .ok_or_else(|| anyhow::anyhow!("No processed event in response"))
    }

    pub async fn process_batch(&mut self, events: Vec<NewsEvent>) -> Result<Vec<ProcessedEvent>> {
        debug!("Processing batch of {} events", events.len());

        let request = tonic::Request::new(BatchRequest { events });

        let response = self.client.process_batch(request).await?;
        let batch_response = response.into_inner();

        if batch_response.total_failed > 0 {
            error!("Batch processing had {} failures: {:?}", 
                batch_response.total_failed, batch_response.error_messages);
        }

        info!("Batch processed: {}/{} events successful", 
            batch_response.total_processed, 
            batch_response.total_processed + batch_response.total_failed);

        Ok(batch_response.processed_events)
    }

    pub async fn wait_for_service(&mut self, max_attempts: u32) -> Result<()> {
        info!("Waiting for preprocessing service to be ready...");
        
        for attempt in 1..=max_attempts {
            match self.health_check().await {
                Ok(true) => {
                    info!("Preprocessing service is ready");
                    return Ok(());
                }
                Ok(false) => {
                    debug!("Attempt {}/{}: Service not healthy", attempt, max_attempts);
                }
                Err(e) => {
                    debug!("Attempt {}/{}: Connection failed: {}", attempt, max_attempts, e);
                }
            }

            if attempt < max_attempts {
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            }
        }

        anyhow::bail!("Preprocessing service did not become ready after {} attempts", max_attempts)
    }
}