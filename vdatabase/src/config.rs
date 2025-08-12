use serde::{Deserialize, Serialize};
use std::env;
use crate::error::{Result, VectorDBError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub collection_name: String,
    pub vector_size: usize,
    pub timeout_seconds: u64,
    pub max_retries: usize,
    pub batch_size: usize,
    pub finbert_enabled: bool,
    pub finbert_model_path: Option<String>,
    pub finbert_batch_size: usize,
}


impl QdrantConfig {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let url = env::var("QDRANT_URL")
            .map_err(|_| VectorDBError::invalid_config("QDRANT_URL not set"))?;

        let api_key = env::var("QDRANT_API_KEY").ok();

        let collection_name = env::var("QDRANT_COLLECTION_NAME")
            .unwrap_or_else(|_| "finmedia_events".to_string());

        let vector_size = env::var("QDRANT_VECTOR_SIZE")
            .unwrap_or_else(|_| "768".to_string())
            .parse()
            .map_err(|_| VectorDBError::invalid_config("Invalid QDRANT_VECTOR_SIZE"))?;

        let timeout_seconds = env::var("QDRANT_TIMEOUT_SECONDS")
            .unwrap_or_else(|_| "30".to_string())
            .parse()
            .map_err(|_| VectorDBError::invalid_config("Invalid QDRANT_TIMEOUT_SECONDS"))?;

        let max_retries = env::var("QDRANT_MAX_RETRIES")
            .unwrap_or_else(|_| "3".to_string())
            .parse()
            .map_err(|_| VectorDBError::invalid_config("Invalid QDRANT_MAX_RETRIES"))?;

        let batch_size = env::var("QDRANT_BATCH_SIZE")
            .unwrap_or_else(|_| "100".to_string())
            .parse()
            .map_err(|_| VectorDBError::invalid_config("Invalid QDRANT_BATCH_SIZE"))?;

        let finbert_enabled = env::var("FINBERT_ENABLED")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .map_err(|_| VectorDBError::invalid_config("Invalid FINBERT_ENABLED (must be true/false)"))?;

        let finbert_model_path = env::var("FINBERT_MODEL_PATH").ok();

        let finbert_batch_size = env::var("FINBERT_BATCH_SIZE")
            .unwrap_or_else(|_| "32".to_string())
            .parse()
            .map_err(|_| VectorDBError::invalid_config("Invalid FINBERT_BATCH_SIZE"))?;


        Ok(Self {
            url,
            api_key,
            collection_name,
            vector_size,
            timeout_seconds,
            max_retries,
            batch_size,
            finbert_enabled,
            finbert_model_path,
            finbert_batch_size,
        })
    }

    pub fn new(
        url: impl Into<String>,
        api_key: Option<String>,
        collection_name: impl Into<String>,
    ) -> Self {
        Self {
            url: url.into(),
            api_key,
            collection_name: collection_name.into(),
            vector_size: 768,
            timeout_seconds: 30,
            max_retries: 3,
            batch_size: 100,
            finbert_enabled: true,
            finbert_model_path: None,
            finbert_batch_size: 32,
        }
    }

    pub fn new_finbert(
        url: impl Into<String>,
        api_key: Option<String>,
        collection_name: impl Into<String>,
        finbert_model_path: impl Into<String>,
    ) -> Self {
        Self {
            url: url.into(),
            api_key,
            collection_name: collection_name.into(),
            vector_size: 768,
            timeout_seconds: 30,
            max_retries: 3,
            batch_size: 100,
            finbert_enabled: true,
            finbert_model_path: Some(finbert_model_path.into()),
            finbert_batch_size: 32,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.url.is_empty() {
            return Err(VectorDBError::invalid_config("URL cannot be empty"));
        }

        if self.collection_name.is_empty() {
            return Err(VectorDBError::invalid_config("Collection name cannot be empty"));
        }

        if self.vector_size == 0 {
            return Err(VectorDBError::invalid_config("Vector size must be greater than 0"));
        }

        if self.timeout_seconds == 0 {
            return Err(VectorDBError::invalid_config("Timeout must be greater than 0"));
        }

        if self.batch_size == 0 {
            return Err(VectorDBError::invalid_config("Batch size must be greater than 0"));
        }

        if !self.finbert_enabled {
            return Err(VectorDBError::invalid_config(
                "FinBERT must be enabled. This system requires FinBERT embeddings to function."
            ));
        }
        
        if self.finbert_model_path.is_none() {
            return Err(VectorDBError::invalid_config(
                "FinBERT model path is required"
            ));
        }
        
        if let Some(ref path) = self.finbert_model_path {
            if path.is_empty() {
                return Err(VectorDBError::invalid_config(
                    "FinBERT model path cannot be empty"
                ));
            }
        }

        if self.finbert_batch_size == 0 {
            return Err(VectorDBError::invalid_config(
                "FinBERT batch size must be greater than 0"
            ));
        }

        if self.vector_size != 768 {
            return Err(VectorDBError::invalid_config(
                "Vector size must be 768 for FinBERT embeddings"
            ));
        }

        Ok(())
    }
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_string(),
            api_key: None,
            collection_name: "finmedia_events".to_string(),
            vector_size: 768,
            timeout_seconds: 30,
            max_retries: 3,
            batch_size: 100,
            finbert_enabled: false, // Disabled by default for simplified setup
            finbert_model_path: None,
            finbert_batch_size: 32,
        }
    }
}