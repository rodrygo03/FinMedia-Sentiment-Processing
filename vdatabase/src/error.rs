use thiserror::Error;

pub type Result<T> = std::result::Result<T, VectorDBError>;

#[derive(Error, Debug)]
pub enum VectorDBError {
    #[error("Qdrant client error: {0}")]
    QdrantClient(#[from] anyhow::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidEmbeddingDimension { expected: usize, actual: usize },

    #[error("Collection not found: {name}")]
    CollectionNotFound { name: String },

    #[error("Document not found: {id}")]
    DocumentNotFound { id: String },

    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Embedding generation failed: {reason}")]
    EmbeddingGeneration { reason: String },

    #[error("Query parsing error: {message}")]
    QueryParsing { message: String },

    #[error("Timeout error: operation took too long")]
    Timeout,

    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl VectorDBError {
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }

    pub fn embedding_generation(reason: impl Into<String>) -> Self {
        Self::EmbeddingGeneration {
            reason: reason.into(),
        }
    }

    pub fn query_parsing(message: impl Into<String>) -> Self {
        Self::QueryParsing {
            message: message.into(),
        }
    }
}