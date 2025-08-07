pub mod client;
pub mod schema;
pub mod embeddings;
pub mod storage;
pub mod query;
pub mod error;
pub mod config;


pub use error::{VectorDBError, Result};
pub use schema::{FinancialEvent, EventMetadata, AssetInfo, SentimentInfo, AssetType, SentimentLabel, SignalStrength};
pub use client::QdrantVectorClient;
pub use embeddings::EmbeddingService;
pub use storage::VectorStorage;
pub use query::{SearchQuery, SearchResult, SimilaritySearch, QueryBuilder};
pub use config::QdrantConfig;

// Re-export common types
pub use chrono::{DateTime, Utc};
pub use uuid::Uuid;