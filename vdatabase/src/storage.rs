use std::sync::Arc;
use tracing::{info, error, debug};

use crate::{
    client::QdrantVectorClient,
    config::QdrantConfig,
    embeddings::{EmbeddingService, EmbeddingProvider},
    error::{Result, VectorDBError},
    schema::{FinancialEvent, SignalsResult},
    query::{SearchQuery, SearchResult},
};

pub struct VectorStorage {
    client: Arc<QdrantVectorClient>,
    embedding_service: Arc<dyn EmbeddingProvider + Send + Sync>,
    config: QdrantConfig,
}

impl VectorStorage {
    pub async fn new(client: QdrantVectorClient) -> Result<Self> {
        let config = QdrantConfig::from_env().unwrap_or_else(|_| QdrantConfig::default());
        
        let embedding_service = if config.finbert_enabled {
            if let Some(ref model_path) = config.finbert_model_path {
                let finbert_path = std::path::Path::new(model_path);
                let service = EmbeddingService::new(finbert_path)?;
                Arc::new(service) as Arc<dyn EmbeddingProvider + Send + Sync>
            } else {
                return Err(VectorDBError::invalid_config("FinBERT model path is required when FinBERT is enabled"));
            }
        } else {
            return Err(VectorDBError::invalid_config("FinBERT must be enabled for vector storage"));
        };
        
        Ok(Self {
            client: Arc::new(client),
            embedding_service,
            config,
        })
    }
    
    pub async fn new_with_config(config: QdrantConfig) -> Result<Self> {
        let client = Arc::new(QdrantVectorClient::new(config.clone()).await?);
        if !config.finbert_enabled {
            return Err(VectorDBError::invalid_config(
                "FinBERT embeddings are required. Please enable FinBERT and provide a model path."
            ));
        }
        
        let model_path = config.finbert_model_path.as_ref()
            .ok_or_else(|| VectorDBError::invalid_config(
                "FinBERT model path is required when FinBERT is enabled"
            ))?;
        let finbert_path = std::path::Path::new(model_path);
        let embedding_service = Arc::new(EmbeddingService::new(finbert_path)?);
        
        Ok(Self {
            client,
            embedding_service,
            config,
        })
    }

    pub async fn new_with_embedding_service(config: QdrantConfig, embedding_service: EmbeddingService) -> Result<Self> {
        let client = Arc::new(QdrantVectorClient::new(config.clone()).await?);
        
        Ok(Self {
            client,
            embedding_service: Arc::new(embedding_service),
            config,
        })
    }

    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing vector storage...");
        if !self.client.health_check().await? {
            return Err(VectorDBError::internal("Qdrant health check failed"));
        }
        
        if !self.client.collection_exists().await? {
            info!("Collection does not exist, creating...");
            self.client.create_collection().await?;
        } else {
            info!("Collection already exists");
        }
        
        info!("Vector storage initialized successfully");
        Ok(())
    }

    pub async fn ensure_collections(&self) -> Result<()> {
        // Ensure financial events collection exists (contains embedded signal data)
        if !self.client.collection_exists().await? {
            info!("Creating financial events collection...");
            self.client.create_collection().await?;
        }
        
        // NOTE: Signals are stored embedded within FinancialEvent objects
        // Separate signals collection is optional for advanced analytics
        info!("Collections are ready");
        Ok(())
    }

    pub async fn store_financial_event(&self, event: &FinancialEvent) -> Result<String> {
        // Store with the existing interface
        let mut event_copy = event.clone();
        self.store_event(&mut event_copy).await
    }

    pub async fn store_signals_result(&self, signals: &SignalsResult) -> Result<String> {
        // Signals are already stored embedded within FinancialEvent objects
        // This method is for additional detailed signals analytics storage
        info!("Signals metadata logged: {}", signals.id);
        
        // NOTE: Core signal data (trading_signal, signal_strength, momentum_direction)
        // is already persisted within FinancialEvent.sentiment and FinancialEvent.metadata
        // This separate storage would be for advanced signals analytics and backtesting
        debug!("Detailed signals analysis logged for event: {}", signals.event_id);
        Ok(signals.id.clone())
    }

    pub async fn store_event(&self, event: &mut FinancialEvent) -> Result<String> {
        let needs_embeddings = event.embedding.is_empty() || event.embedding.iter().all(|&x| x == 0.0)
            || event.sentiment_embedding.is_empty() || event.sentiment_embedding.iter().all(|&x| x == 0.0)
            || event.asset_embedding.is_empty() || event.asset_embedding.iter().all(|&x| x == 0.0);
        
        if needs_embeddings {
            info!("Generating multi-vector embeddings for event: {}", event.id);
            let (content_embedding, sentiment_embedding, asset_embedding) = self.embedding_service.generate_multi_embeddings(event)?;
            event.set_multi_embeddings(content_embedding, sentiment_embedding, asset_embedding)?;
            debug!("Generated multi-vector embeddings for event: {}", event.id);
        }
        
        self.client.upsert_event(event).await?;
        debug!("Stored event with multi-vectors: {}", event.id);
        Ok(event.id.clone())
    }

    /// Store event with only content embedding (for backward compatibility)
    pub async fn store_event_single_vector(&self, event: &mut FinancialEvent) -> Result<String> {
        if event.embedding.is_empty() || event.embedding.iter().all(|&x| x == 0.0) {
            let embedding = self.embedding_service.generate_event_embedding(event)?;
            event.set_embedding(embedding)?;
        }
        
        // Ensure other embeddings are set to avoid validation errors
        if event.sentiment_embedding.is_empty() {
            event.sentiment_embedding = vec![0.0; FinancialEvent::EMBEDDING_DIM];
        }
        if event.asset_embedding.is_empty() {
            event.asset_embedding = vec![0.0; FinancialEvent::EMBEDDING_DIM];
        }
        
        self.client.upsert_event(event).await?;
        debug!("Stored event (single vector): {}", event.id);
        Ok(event.id.clone())
    }

    pub async fn store_events_batch(&self, events: &mut [FinancialEvent]) -> Result<Vec<String>> {
        if events.is_empty() {
            return Ok(Vec::new());
        }

        info!("Storing batch of {} events with multi-vector embeddings", events.len());
        
        // Generate embeddings for all events that need them
        for event in events.iter_mut() {
            let needs_embeddings = event.embedding.is_empty() || event.embedding.iter().all(|&x| x == 0.0)
                || event.sentiment_embedding.is_empty() || event.sentiment_embedding.iter().all(|&x| x == 0.0)
                || event.asset_embedding.is_empty() || event.asset_embedding.iter().all(|&x| x == 0.0);
            
            if needs_embeddings {
                let (content_embedding, sentiment_embedding, asset_embedding) = self.embedding_service.generate_multi_embeddings(event)?;
                event.set_multi_embeddings(content_embedding, sentiment_embedding, asset_embedding)?;
            }
        }
        
        
        self.client.upsert_events_batch(events).await?;
        let event_ids: Vec<String> = events.iter().map(|e| e.id.clone()).collect();
        info!("Successfully stored {} events with multi-vectors", event_ids.len());
        Ok(event_ids)
    }

    pub async fn search_similar_events(&self, query: &SearchQuery) -> Result<SearchResult> {
        let query_embedding = match &query.vector {
            Some(vector) => vector.clone(),
            None => {
                // Generate embedding from query text
                if let Some(text) = &query.text {
                    self.embedding_service.generate_query_embedding(text)?
                } else {
                    return Err(VectorDBError::query_parsing("Either vector or text query must be provided"));
                }
            }
        };

        let results = self.client.search_similar(
            query_embedding,
            query.limit.unwrap_or(10),
            None,
            None, 
        ).await?;

        let search_results: Vec<(FinancialEvent, f64)> = results
            .into_iter()
            .map(|(event, score)| (event, score as f64))
            .collect();
        Ok(SearchResult::new(search_results, query.clone()))
    }

    pub async fn search_by_sentiment(&self, query: &SearchQuery) -> Result<SearchResult> {
        let query_embedding = match &query.vector {
            Some(vector) => vector.clone(),
            None => {
                if let Some(text) = &query.text {
                    // Generate sentiment-focused embedding for the query
                    let sentiment_text = format!("{} sentiment analysis financial", text);
                    self.embedding_service.generate_query_embedding(&sentiment_text)?
                } else {
                    return Err(VectorDBError::query_parsing("Either vector or text query must be provided"));
                }
            }
        };

        let results = self.client.search_by_sentiment(
            query_embedding,
            query.limit.unwrap_or(10),
            query.score_threshold,
            None, 
        ).await?;

        let search_results: Vec<(FinancialEvent, f64)> = results
            .into_iter()
            .map(|(event, score)| (event, score as f64))
            .collect();

        Ok(SearchResult::new(search_results, query.clone()))
    }

    pub async fn search_by_asset(&self, query: &SearchQuery) -> Result<SearchResult> {
        let query_embedding = match &query.vector {
            Some(vector) => vector.clone(),
            None => {
                if let Some(text) = &query.text {
                    // Generate asset-focused embedding for the query
                    let asset_text = format!("{} assets financial market", text);
                    self.embedding_service.generate_query_embedding(&asset_text)?
                } else {
                    return Err(VectorDBError::query_parsing("Either vector or text query must be provided"));
                }
            }
        };

        let results = self.client.search_by_asset(
            query_embedding,
            query.limit.unwrap_or(10),
            query.score_threshold,
            None, 
        ).await?;

        let search_results: Vec<(FinancialEvent, f64)> = results
            .into_iter()
            .map(|(event, score)| (event, score as f64))
            .collect();

        Ok(SearchResult::new(search_results, query.clone()))
    }

    pub async fn hybrid_search(
        &self,
        query: &SearchQuery,
        content_weight: f32,
        sentiment_weight: f32,
        asset_weight: f32,
    ) -> Result<SearchResult> {
        if let Some(text) = &query.text {
            // Generate different types of embeddings for the query
            let content_embedding = self.embedding_service.generate_query_embedding(text)?;
            let sentiment_embedding = {
                let sentiment_text = format!("{} sentiment analysis financial", text);
                self.embedding_service.generate_query_embedding(&sentiment_text)?
            };
            let asset_embedding = {
                let asset_text = format!("{} assets financial market", text);
                self.embedding_service.generate_query_embedding(&asset_text)?
            };

            let results = self.client.hybrid_search(
                Some((content_embedding, content_weight)),
                Some((sentiment_embedding, sentiment_weight)),
                Some((asset_embedding, asset_weight)),
                query.limit.unwrap_or(10),
                query.score_threshold,
                None, 
            ).await?;

            let search_results: Vec<(FinancialEvent, f64)> = results
                .into_iter()
                .map(|(event, score)| (event, score as f64))
                .collect();

            Ok(SearchResult::new(search_results, query.clone()))
        } else {
            return Err(VectorDBError::query_parsing("Text query is required for hybrid search"));
        }
    }

    pub async fn generate_multi_embeddings_for_text(&self, text: &str) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let content_embedding = self.embedding_service.generate_query_embedding(text)?;
        let sentiment_embedding = {
            let sentiment_text = format!("{} sentiment analysis financial", text);
            self.embedding_service.generate_query_embedding(&sentiment_text)?
        };
        let asset_embedding = {
            let asset_text = format!("{} assets financial market", text);
            self.embedding_service.generate_query_embedding(&asset_text)?
        };

        Ok((content_embedding, sentiment_embedding, asset_embedding))
    }

    pub async fn generate_embedding_for_text(&self, text: &str) -> Result<Vec<f32>> {
        self.embedding_service.generate_query_embedding(text)
    }

    pub fn calculate_similarity(&self, event1: &FinancialEvent, event2: &FinancialEvent) -> Result<f64> {
        let similarity = self.embedding_service.cosine_similarity(&event1.embedding, &event2.embedding)?;
        Ok(similarity as f64)
    }

    pub async fn get_storage_stats(&self) -> Result<StorageStats> {
        Ok(StorageStats {
            total_events: 0, // TODO: query Qdrant for actual count
            collection_name: self.config.collection_name.clone(),
            vector_size: self.config.vector_size,
            status: "healthy".to_string(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_events: u64,
    pub collection_name: String,
    pub vector_size: usize,
    pub status: String,
}