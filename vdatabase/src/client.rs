use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollection, VectorParams, Distance,
    PointStruct, UpsertPoints, SearchPoints, GetPoints,
    DeletePoints, ScrollPoints, Filter, ScoredPoint, RetrievedPoint,
    Value as QdrantValue
};

use std::collections::HashMap;
use tracing::{info, error, debug};

use crate::{
    config::QdrantConfig,
    error::{Result, VectorDBError},
    schema::FinancialEvent,
};

pub struct QdrantVectorClient {
    client: Qdrant,
    config: QdrantConfig,
}

impl QdrantVectorClient {
    pub async fn new(config: QdrantConfig) -> Result<Self> {
        config.validate()?;
        
        let client = if let Some(api_key) = &config.api_key {
            Qdrant::from_url(&config.url)
                .api_key(api_key.clone())
                .build()
                .map_err(|e| VectorDBError::QdrantClient(e.into()))?
        } else {
            Qdrant::from_url(&config.url)
                .build()
                .map_err(|e| VectorDBError::QdrantClient(e.into()))?
        };

        Ok(Self { client, config })
    }

    pub async fn health_check(&self) -> Result<bool> {
        match self.client.health_check().await {
            Ok(_) => {
                info!("Qdrant health check passed");
                Ok(true)
            }
            Err(e) => {
                error!("Qdrant health check failed: {}", e);
                Err(VectorDBError::QdrantClient(e.into()))
            }
        }
    }

    pub async fn create_collection(&self) -> Result<()> {
        info!("Creating collection with multi-vector support: {}", self.config.collection_name);

        // Create collection with named vectors for multi-vector support
        let mut named_vectors = std::collections::HashMap::new();
        
        // Content vector (primary)
        named_vectors.insert("content".to_string(), VectorParams {
            size: self.config.vector_size as u64,
            distance: Distance::Cosine.into(),
            ..Default::default()
        });
        
        // Sentiment vector
        named_vectors.insert("sentiment".to_string(), VectorParams {
            size: self.config.vector_size as u64,
            distance: Distance::Cosine.into(),
            ..Default::default()
        });
        
        // Asset vector
        named_vectors.insert("asset".to_string(), VectorParams {
            size: self.config.vector_size as u64,
            distance: Distance::Cosine.into(),
            ..Default::default()
        });

        let create_collection = CreateCollection {
            collection_name: self.config.collection_name.clone(),
            vectors_config: Some(qdrant_client::qdrant::VectorsConfig {
                config: Some(qdrant_client::qdrant::vectors_config::Config::ParamsMap(
                    qdrant_client::qdrant::VectorParamsMap {
                        map: named_vectors,
                    }
                ))
            }),
            ..Default::default()
        };

        match self.client.create_collection(create_collection).await {
            Ok(_) => {
                info!("Collection '{}' created successfully", self.config.collection_name);
                Ok(())
            }
            Err(e) => {
                let error_msg = e.to_string();
                if error_msg.contains("already exists") {
                    info!("Collection '{}' already exists", self.config.collection_name);
                    Ok(())
                } else {
                    error!("Failed to create collection: {}", e);
                    Err(VectorDBError::QdrantClient(e.into()))
                }
            }
        }
    }

    pub async fn collection_exists(&self) -> Result<bool> {
        match self.client.collection_info(&self.config.collection_name).await {
            Ok(_) => Ok(true),
            Err(e) => {
                let error_msg = e.to_string();
                if error_msg.contains("doesn't exist") || error_msg.contains("not found") {
                    Ok(false)
                } else {
                    Err(VectorDBError::QdrantClient(e.into()))
                }
            }
        }
    }

    pub async fn get_collection_info(&self) -> Result<()> {
        self.client
            .collection_info(&self.config.collection_name)
            .await
            .map_err(|e| VectorDBError::QdrantClient(e.into()))?;
        Ok(())
    }

    pub async fn upsert_event(&self, event: &FinancialEvent) -> Result<()> {
        event.validate_embedding()?;

        let payload = self.event_to_payload(event)?;
        
        // Create named vectors map for multi-vector storage
        let mut vectors_map = std::collections::HashMap::new();
        vectors_map.insert("content".to_string(), event.embedding.clone().into());
        vectors_map.insert("sentiment".to_string(), event.sentiment_embedding.clone().into());
        vectors_map.insert("asset".to_string(), event.asset_embedding.clone().into());
        
        let point = PointStruct {
            id: Some(event.id.clone().into()),
            vectors: Some(qdrant_client::qdrant::Vectors {
                vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vectors(
                    qdrant_client::qdrant::NamedVectors {
                        vectors: vectors_map,
                    }
                )),
            }),
            payload,
        };

        let upsert_points = UpsertPoints {
            collection_name: self.config.collection_name.clone(),
            points: vec![point],
            wait: Some(true),
            ..Default::default()
        };

        self.client
            .upsert_points(upsert_points)
            .await
            .map_err(|e| VectorDBError::QdrantClient(e.into()))?;

        debug!("Upserted event with multi-vectors: {}", event.id);
        Ok(())
    }

    pub async fn upsert_events_batch(&self, events: &[FinancialEvent]) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }

        for event in events {
            event.validate_embedding()?;
        }

        let points: Result<Vec<PointStruct>> = events
            .iter()
            .map(|event| {
                let payload = self.event_to_payload(event)?;
                
                let mut vectors_map = std::collections::HashMap::new();
                vectors_map.insert("content".to_string(), event.embedding.clone().into());
                vectors_map.insert("sentiment".to_string(), event.sentiment_embedding.clone().into());
                vectors_map.insert("asset".to_string(), event.asset_embedding.clone().into());
                
                Ok(PointStruct {
                    id: Some(event.id.clone().into()),
                    vectors: Some(qdrant_client::qdrant::Vectors {
                        vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vectors(
                            qdrant_client::qdrant::NamedVectors {
                                vectors: vectors_map,
                            }
                        )),
                    }),
                    payload,
                })
            })
            .collect();

        let points = points?;

        let upsert_points = UpsertPoints {
            collection_name: self.config.collection_name.clone(),
            points,
            wait: Some(true),
            ..Default::default()
        };

        self.client
            .upsert_points(upsert_points)
            .await
            .map_err(|e| VectorDBError::QdrantClient(e.into()))?;

        info!("Upserted {} events in batch", events.len());
        Ok(())
    }

    pub async fn get_event(&self, id: &str) -> Result<Option<FinancialEvent>> {
        let get_points = GetPoints {
            collection_name: self.config.collection_name.clone(),
            ids: vec![id.into()],
            with_payload: Some(true.into()),
            with_vectors: Some(true.into()),
            ..Default::default()
        };

        let response = self.client
            .get_points(get_points)
            .await
            .map_err(|e| VectorDBError::QdrantClient(e.into()))?;

        if let Some(point) = response.result.first() {
            self.point_to_event(point)
        } else {
            Ok(None)
        }
    }

    pub async fn delete_event(&self, id: &str) -> Result<()> {
        let delete_points = DeletePoints {
            collection_name: self.config.collection_name.clone(),
            points: Some(qdrant_client::qdrant::PointsSelector {
                points_selector_one_of: Some(
                    qdrant_client::qdrant::points_selector::PointsSelectorOneOf::Points(
                        qdrant_client::qdrant::PointsIdsList {
                            ids: vec![id.into()],
                        }
                    )
                )
            }),
            wait: Some(true),
            ..Default::default()
        };

        self.client
            .delete_points(delete_points)
            .await
            .map_err(|e| VectorDBError::QdrantClient(e.into()))?;

        debug!("Deleted event: {}", id);
        Ok(())
    }

    pub async fn search_similar(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
        score_threshold: Option<f32>,
        filter: Option<Filter>,
    ) -> Result<Vec<(FinancialEvent, f32)>> {
        if query_vector.len() != self.config.vector_size {
            return Err(VectorDBError::InvalidEmbeddingDimension {
                expected: self.config.vector_size,
                actual: query_vector.len(),
            });
        }

        // Default to content vector for backward compatibility
        let search_points = SearchPoints {
            collection_name: self.config.collection_name.clone(),
            vector: query_vector,
            vector_name: Some("content".to_string()),
            limit: limit as u64,
            with_payload: Some(true.into()),
            with_vectors: Some(true.into()),
            score_threshold: score_threshold,
            filter,
            ..Default::default()
        };

        let response = self.client
            .search_points(search_points)
            .await
            .map_err(|e| VectorDBError::QdrantClient(e.into()))?;

        let mut results = Vec::new();
        for scored_point in response.result {
            if let Some(event) = self.scored_point_to_event(&scored_point)? {
                results.push((event, scored_point.score));
            }
        }

        Ok(results)
    }

    pub async fn scroll_events(
        &self,
        limit: Option<usize>,
        offset: Option<String>,
        filter: Option<Filter>,
    ) -> Result<(Vec<FinancialEvent>, Option<String>)> {
        let scroll_points = ScrollPoints {
            collection_name: self.config.collection_name.clone(),
            limit: Some(limit.unwrap_or(100) as u32),
            with_payload: Some(true.into()),
            with_vectors: Some(true.into()),
            offset: offset.map(|o| o.into()),
            filter,
            ..Default::default()
        };

        let response = self.client
            .scroll(scroll_points)
            .await
            .map_err(|e| VectorDBError::QdrantClient(e.into()))?;

        let mut events = Vec::new();
        for point in response.result {
            if let Some(event) = self.point_to_event(&point)? {
                events.push(event);
            }
        }

        let next_offset = response.next_page_offset.map(|point_id| {
            match &point_id.point_id_options {
                Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => num.to_string(),
                Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => uuid.to_string(),
                None => String::new(),
            }
        });

        Ok((events, next_offset))
    }

    pub async fn search_by_vector_type(
        &self,
        query_vector: Vec<f32>,
        vector_type: &str,
        limit: usize,
        score_threshold: Option<f32>,
        filter: Option<Filter>,
    ) -> Result<Vec<(FinancialEvent, f32)>> {
        if query_vector.len() != self.config.vector_size {
            return Err(VectorDBError::InvalidEmbeddingDimension {
                expected: self.config.vector_size,
                actual: query_vector.len(),
            });
        }

        let search_points = SearchPoints {
            collection_name: self.config.collection_name.clone(),
            vector: query_vector,
            vector_name: Some(vector_type.to_string()),
            limit: limit as u64,
            with_payload: Some(true.into()),
            with_vectors: Some(true.into()),
            score_threshold,
            filter,
            ..Default::default()
        };

        let response = self.client
            .search_points(search_points)
            .await
            .map_err(|e| VectorDBError::QdrantClient(e.into()))?;

        let mut results = Vec::new();
        for scored_point in response.result {
            if let Some(event) = self.scored_point_to_event(&scored_point)? {
                results.push((event, scored_point.score));
            }
        }

        debug!("Found {} similar events using {} vector", results.len(), vector_type);
        Ok(results)
    }

    pub async fn search_by_sentiment(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
        score_threshold: Option<f32>,
        filter: Option<Filter>,
    ) -> Result<Vec<(FinancialEvent, f32)>> {
        self.search_by_vector_type(query_vector, "sentiment", limit, score_threshold, filter).await
    }

    pub async fn search_by_asset(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
        score_threshold: Option<f32>,
        filter: Option<Filter>,
    ) -> Result<Vec<(FinancialEvent, f32)>> {
        self.search_by_vector_type(query_vector, "asset", limit, score_threshold, filter).await
    }

    pub async fn hybrid_search(
        &self,
        content_vector: Option<(Vec<f32>, f32)>,  // (vector, weight)
        sentiment_vector: Option<(Vec<f32>, f32)>,
        asset_vector: Option<(Vec<f32>, f32)>,
        limit: usize,
        score_threshold: Option<f32>,
        filter: Option<Filter>,
    ) -> Result<Vec<(FinancialEvent, f32)>> {
        let mut all_results: std::collections::HashMap<String, (FinancialEvent, f32)> = std::collections::HashMap::new();

        if let Some((vector, weight)) = content_vector {
            let results = self.search_by_vector_type(vector, "content", limit * 2, score_threshold, filter.clone()).await?;
            for (event, score) in results {
                let weighted_score = score * weight;
                all_results.entry(event.id.clone())
                    .and_modify(|(_, current_score)| *current_score = (*current_score + weighted_score).min(1.0))
                    .or_insert((event, weighted_score));
            }
        }

        if let Some((vector, weight)) = sentiment_vector {
            let results = self.search_by_vector_type(vector, "sentiment", limit * 2, score_threshold, filter.clone()).await?;
            for (event, score) in results {
                let weighted_score = score * weight;
                all_results.entry(event.id.clone())
                    .and_modify(|(_, current_score)| *current_score = (*current_score + weighted_score).min(1.0))
                    .or_insert((event, weighted_score));
            }
        }

        if let Some((vector, weight)) = asset_vector {
            let results = self.search_by_vector_type(vector, "asset", limit * 2, score_threshold, filter).await?;
            for (event, score) in results {
                let weighted_score = score * weight;
                all_results.entry(event.id.clone())
                    .and_modify(|(_, current_score)| *current_score = (*current_score + weighted_score).min(1.0))
                    .or_insert((event, weighted_score));
            }
        }

        // Sort by combined score and take top results
        let mut final_results: Vec<(FinancialEvent, f32)> = all_results.into_values().collect();
        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        final_results.truncate(limit);

        debug!("Hybrid search returned {} results", final_results.len());
        Ok(final_results)
    }

    fn event_to_payload(&self, event: &FinancialEvent) -> Result<HashMap<String, QdrantValue>> {
        let mut payload = HashMap::new();

        payload.insert("title".to_string(), event.title.clone().into());
        payload.insert("content".to_string(), event.content.clone().into());
        payload.insert("source".to_string(), event.source.clone().into());
        payload.insert("published_at".to_string(), event.published_at.timestamp().into());
        payload.insert("processed_at".to_string(), event.processed_at.timestamp().into());
        
        if let Some(url) = &event.url {
            payload.insert("url".to_string(), url.clone().into());
        }
        
        if let Some(primary_asset) = &event.primary_asset {
            payload.insert("primary_asset".to_string(), primary_asset.clone().into());
        }

        payload.insert("sentiment_score".to_string(), event.sentiment.score.into());
        payload.insert("sentiment_label".to_string(), event.sentiment.label.as_str().into());
        payload.insert("sentiment_confidence".to_string(), event.sentiment.confidence.into());
        payload.insert("trading_signal".to_string(), event.sentiment.trading_signal.into());
        payload.insert("signal_strength".to_string(), event.sentiment.signal_strength.as_str().into());

        let asset_symbols: Vec<QdrantValue> = event.assets
            .iter()
            .map(|a| QdrantValue::from(a.symbol.clone()))
            .collect();
        let asset_types: Vec<QdrantValue> = event.assets
            .iter()
            .map(|a| QdrantValue::from(a.asset_type.as_str()))
            .collect();
        
        payload.insert("asset_symbols".to_string(), QdrantValue::from(asset_symbols));
        payload.insert("asset_types".to_string(), QdrantValue::from(asset_types));

        // Metadata
        payload.insert("timestamp_bucket".to_string(), event.metadata.timestamp_bucket.clone().into());
        payload.insert("hour_bucket".to_string(), (event.metadata.hour_bucket as i64).into());
        payload.insert("day_of_week".to_string(), (event.metadata.day_of_week as i64).into());
        payload.insert("unix_timestamp".to_string(), event.metadata.unix_timestamp.into());
        payload.insert("asset_count".to_string(), (event.metadata.asset_count as i64).into());
        payload.insert("confidence_tier".to_string(), event.metadata.confidence_tier.clone().into());
        payload.insert("content_length".to_string(), (event.metadata.content_length as i64).into());
        payload.insert("content_tier".to_string(), event.metadata.content_tier.clone().into());
        payload.insert("language".to_string(), event.metadata.language.clone().into());
        payload.insert("source_domain".to_string(), event.metadata.source_domain.clone().into());
        payload.insert("sentiment_tier".to_string(), event.metadata.sentiment_tier.clone().into());
        payload.insert("impact_level".to_string(), event.metadata.impact_level.clone().into());
        payload.insert("trading_signal_tier".to_string(), event.metadata.trading_signal_tier.clone().into());
        payload.insert("momentum_direction".to_string(), event.metadata.momentum_direction.clone().into());

        // Raw data for perfect reconstruction
        payload.insert("raw_data".to_string(), event.raw_data.clone().into());

        Ok(payload)
    }

    fn point_to_event(&self, point: &RetrievedPoint) -> Result<Option<FinancialEvent>> {
        let payload = &point.payload;
        
        // Extract raw data and deserialize if possible
        if let Some(raw_data) = payload.get("raw_data") {
            if let Some(raw_str) = raw_data.as_str() {
                if !raw_str.is_empty() {
                    match serde_json::from_str::<FinancialEvent>(raw_str) {
                        Ok(event) => return Ok(Some(event)),
                        Err(e) => {
                            debug!("Failed to deserialize raw event data: {}", e);
                        }
                    }
                }
            }
        }

        // Fallback: reconstruct from payload fields
        let id = point.id.as_ref()
            .map(|point_id| {
                match &point_id.point_id_options {
                    Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => num.to_string(),
                    Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => uuid.to_string(),
                    None => String::new(),
                }
            })
            .unwrap_or_default();
        
        let title = payload.get("title")
            .and_then(|v| v.as_str())
            .map_or("".to_string(), |s| s.to_string());
            
        let content = payload.get("content")
            .and_then(|v| v.as_str())
            .map_or("".to_string(), |s| s.to_string());
            
        let source = payload.get("source")
            .and_then(|v| v.as_str())
            .map_or("".to_string(), |s| s.to_string());

        let published_at = payload.get("published_at")
            .and_then(|v| v.as_integer())
            .and_then(|ts| chrono::DateTime::from_timestamp(ts, 0))
            .unwrap_or_else(chrono::Utc::now);

        let processed_at = payload.get("processed_at")
            .and_then(|v| v.as_integer())
            .and_then(|ts| chrono::DateTime::from_timestamp(ts, 0))
            .unwrap_or_else(chrono::Utc::now);

        let mut event = FinancialEvent::new(title, content, source, published_at);
        event.id = id;
        event.processed_at = processed_at;

        // Set embedding from vectors if available
        if let Some(vectors) = &point.vectors {
            if let Some(vector_data) = vectors.vectors_options.as_ref() {
                match vector_data {
                    qdrant_client::qdrant::vectors_output::VectorsOptions::Vector(v) => {
                        event.embedding = v.data.clone();
                    }
                    _ => {}
                }
            }
        }

        Ok(Some(event))
    }

    fn scored_point_to_event(&self, scored_point: &ScoredPoint) -> Result<Option<FinancialEvent>> {
        // Create a RetrievedPoint-like structure for reuse
        let retrieved_point = RetrievedPoint {
            id: scored_point.id.clone(),
            payload: scored_point.payload.clone(),
            vectors: scored_point.vectors.clone(),
            order_value: None,
            shard_key: None,
        };
        self.point_to_event(&retrieved_point)
    }
}

pub fn create_asset_filter(asset_symbols: &[String]) -> Filter {
    use qdrant_client::qdrant::{Condition, FieldCondition, Match};
    
    Filter {
        must: vec![Condition {
            condition_one_of: Some(
                qdrant_client::qdrant::condition::ConditionOneOf::Field(
                    FieldCondition {
                        key: "asset_symbols".to_string(),
                        r#match: Some(Match {
                            match_value: Some(qdrant_client::qdrant::r#match::MatchValue::Keywords(
                                qdrant_client::qdrant::RepeatedStrings {
                                    strings: asset_symbols.to_vec(),
                                }
                            )),
                        }),
                        range: None,
                        geo_bounding_box: None,
                        geo_radius: None,
                        values_count: None,
                        geo_polygon: None,
                        datetime_range: None,
                        is_empty: None,
                        is_null: None,
                    }
                )
            ),
        }],
        ..Default::default()
    }
}

pub fn create_sentiment_filter(min_score: f64, max_score: f64) -> Filter {
    use qdrant_client::qdrant::{Condition, FieldCondition, Range};
    
    Filter {
        must: vec![Condition {
            condition_one_of: Some(
                qdrant_client::qdrant::condition::ConditionOneOf::Field(
                    FieldCondition {
                        key: "sentiment_score".to_string(),
                        r#match: None,
                        range: Some(Range {
                            gte: Some(min_score),
                            lte: Some(max_score),
                            gt: None,
                            lt: None,
                        }),
                        geo_bounding_box: None,
                        geo_radius: None,
                        values_count: None,
                        geo_polygon: None,
                        datetime_range: None,
                        is_empty: None,
                        is_null: None,
                    }
                )
            ),
        }],
        ..Default::default()
    }
}

pub fn create_time_filter(start_timestamp: i64, end_timestamp: i64) -> Filter {
    use qdrant_client::qdrant::{Condition, FieldCondition, Range};
    
    Filter {
        must: vec![Condition {
            condition_one_of: Some(
                qdrant_client::qdrant::condition::ConditionOneOf::Field(
                    FieldCondition {
                        key: "unix_timestamp".to_string(),
                        r#match: None,
                        range: Some(Range {
                            gte: Some(start_timestamp as f64),
                            lte: Some(end_timestamp as f64),
                            gt: None,
                            lt: None,
                        }),
                        geo_bounding_box: None,
                        geo_radius: None,
                        values_count: None,
                        geo_polygon: None,
                        datetime_range: None,
                        is_empty: None,
                        is_null: None,
                    }
                )
            ),
        }],
        ..Default::default()
    }
}