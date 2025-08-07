use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

use crate::{
    error::{Result, VectorDBError},
    schema::{FinancialEvent, AssetType, SentimentLabel, SignalStrength},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub text: Option<String>,
    pub vector: Option<Vec<f32>>,
    pub limit: Option<usize>,
    pub score_threshold: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub events: Vec<(FinancialEvent, f64)>,
    pub query: SearchQuery,
}

#[derive(Debug, Clone)]
pub struct SimilaritySearch {
    pub reference_event_id: String,
    pub similarity_threshold: f64,
    pub max_results: usize,
}

impl SearchQuery {
    pub fn new() -> Self {
        Self {
            text: None,
            vector: None,
            limit: Some(10),
            score_threshold: None,
        }
    }

    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    pub fn with_vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(vector);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn with_score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = Some(threshold);
        self
    }
}

impl SearchResult {
    pub fn new(events: Vec<(FinancialEvent, f64)>, query: SearchQuery) -> Self {
        Self { events, query }
    }

    pub fn events(&self) -> &[(FinancialEvent, f64)] {
        &self.events
    }

    pub fn into_events(self) -> Vec<FinancialEvent> {
        self.events.into_iter().map(|(event, _)| event).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn count(&self) -> usize {
        self.events.len()
    }
}

// Query builder for common patterns
pub struct QueryBuilder;
impl QueryBuilder {
    pub fn semantic_search(query_text: &str) -> SearchQuery {
        SearchQuery::new()
            .with_text(query_text)
            .with_limit(20)
            .with_score_threshold(0.7)
    }

    pub fn similar_to_vector(reference_vector: Vec<f32>, threshold: f32) -> SearchQuery {
        SearchQuery::new()
            .with_vector(reference_vector)
            .with_score_threshold(threshold)
            .with_limit(15)
    }
}