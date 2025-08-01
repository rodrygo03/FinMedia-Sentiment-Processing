use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, debug, error, warn};
use std::collections::VecDeque;

use crate::client::{PreprocessingClient, preprocessing::*};
use crate::client::preprocessing::preprocessing_service_server;

/// gRPC server that intercepts events from Go service and forwards them to real preprocessing service
pub struct InterceptorServer {
    preprocessing_client: Arc<Mutex<Option<PreprocessingClient>>>,
    captured_events: Arc<Mutex<VecDeque<ProcessedEvent>>>,
    start_time: std::time::Instant,
}

impl InterceptorServer {
    pub fn new() -> Self {
        Self {
            preprocessing_client: Arc::new(Mutex::new(None)),
            captured_events: Arc::new(Mutex::new(VecDeque::new())),
            start_time: std::time::Instant::now(),
        }
    }

    pub async fn set_preprocessing_client(&self, client: PreprocessingClient) {
        let mut guard = self.preprocessing_client.lock().await;
        *guard = Some(client);
        info!("Interceptor server connected to real preprocessing service");
    }

    pub async fn get_captured_events(&self) -> Vec<ProcessedEvent> {
        let mut guard = self.captured_events.lock().await;
        let events: Vec<ProcessedEvent> = guard.drain(..).collect();
        events
    }

    pub async fn start_server(self: Arc<Self>, port: u16) -> Result<()> {
        let addr = format!("127.0.0.1:{}", port).parse()?;
        
        info!("Starting interceptor gRPC server on {}", addr);

        // Use the same Arc references from self
        let server_instance = InterceptorServerImpl {
            preprocessing_client: Arc::clone(&self.preprocessing_client),
            captured_events: Arc::clone(&self.captured_events),
            start_time: self.start_time,
        };

        Server::builder()
            .add_service(preprocessing_service_server::PreprocessingServiceServer::new(server_instance))
            .serve(addr)
            .await?;

        Ok(())
    }
}

/// Implementation of the gRPC service that intercepts and forwards events
struct InterceptorServerImpl {
    preprocessing_client: Arc<Mutex<Option<PreprocessingClient>>>,
    captured_events: Arc<Mutex<VecDeque<ProcessedEvent>>>,
    start_time: std::time::Instant,
}

#[tonic::async_trait]
impl preprocessing_service_server::PreprocessingService for InterceptorServerImpl {
    async fn process_news_event(
        &self,
        request: Request<NewsEventRequest>,
    ) -> std::result::Result<Response<ProcessedEventResponse>, Status> {
        info!("Intercepted news event from Go service");

        let mut client_guard = self.preprocessing_client.lock().await;
        if let Some(ref mut client) = client_guard.as_mut() {
            match client.process_event(request.into_inner().event.unwrap()).await {
                Ok(processed_event) => {
                    info!("Event processed successfully, capturing for orchestrator");
                    {// Capture the processed event for our queue
                        let mut events_guard = self.captured_events.lock().await;
                        events_guard.push_back(processed_event.clone());
                        debug!("📦 Captured event queue size: {}", events_guard.len());
                    }// Return response to Go service
                    Ok(Response::new(ProcessedEventResponse {
                        processed_event: Some(processed_event),
                        success: true,
                        error_message: String::new(),
                    }))
                }
                Err(e) => {
                    error!("Failed to process event: {}", e);
                    Err(Status::internal(format!("Processing failed: {}", e)))
                }
            }
        } else {
            warn!("No preprocessing client available");
            Err(Status::unavailable("Preprocessing service not available"))
        }
    }

    async fn process_batch(
        &self,
        request: Request<BatchRequest>,
    ) -> std::result::Result<Response<BatchResponse>, Status> {
        let batch_size = request.get_ref().events.len();
        info!("Intercepted batch of {} events from Go service", batch_size);
        println!("Intercepted batch of {} events from Go service", batch_size);

        let mut client_guard = self.preprocessing_client.lock().await;
        
        if let Some(ref mut client) = client_guard.as_mut() {
            let events = request.into_inner().events;
            println!("Forwarding {} events to real preprocessing service...", events.len());
            
            match client.process_batch(events).await {
                Ok(processed_events) => {
                    info!("Batch of {} events processed successfully", processed_events.len());
                    println!("Batch of {} events processed successfully", processed_events.len());
                    
                    // Capture all processed events
                    {
                        let mut events_guard = self.captured_events.lock().await;
                        for event in &processed_events {
                            events_guard.push_back(event.clone());
                        }
                        let queue_size = events_guard.len();
                        info!("Captured {} events, queue size: {}", processed_events.len(), queue_size);
                        println!("Captured {} events, total queue size: {}", processed_events.len(), queue_size);
                    }

                    // Return response to Go service
                    let total_processed = processed_events.len() as i32;
                    Ok(Response::new(BatchResponse {
                        processed_events,
                        total_processed,
                        total_failed: 0,
                        error_messages: vec![],
                    }))
                }
                Err(e) => {
                    error!("Failed to process batch: {}", e);
                    println!("Failed to process batch: {}", e);
                    Err(Status::internal(format!("Batch processing failed: {}", e)))
                }
            }
        } else {
            warn!("No preprocessing client available for batch processing");
            println!("No preprocessing client available - cannot forward batch to real preprocessing service");
            Err(Status::unavailable("Preprocessing service not available"))
        }
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> std::result::Result<Response<HealthResponse>, Status> {
        let uptime = self.start_time.elapsed().as_secs() as i64;
        
        // Check if we can reach the real preprocessing service
        let status = {
            let mut client_guard = self.preprocessing_client.lock().await;
            if let Some(ref mut client) = client_guard.as_mut() {
                match client.health_check().await {
                    Ok(true) => "healthy",
                    Ok(false) => "unhealthy",
                    Err(_) => "preprocessing_unavailable",
                }
            } else {
                "no_preprocessing_client"
            }
        };

        Ok(Response::new(HealthResponse {
            status: status.to_string(),
            version: "orchestrator-0.1.0".to_string(),
            uptime_seconds: uptime,
        }))
    }
}