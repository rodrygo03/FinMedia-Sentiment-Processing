use std::path::Path;
use std::cell::RefCell;
use tracing::{debug, error};

use inference::FinBertInference;

use crate::{
    error::{Result, VectorDBError},
    schema::FinancialEvent,
};

pub struct EmbeddingService {
    embedding_dim: usize,
    finbert: RefCell<FinBertInference>,
}

impl EmbeddingService {
    pub fn new(config_path: &Path) -> Result<Self> {
        let finbert = FinBertInference::new(config_path)
            .map_err(|e| {
                error!("Failed to initialize FinBERT model at path '{}': {}", config_path.display(), e);
                VectorDBError::embedding_generation(
                    format!("Failed to initialize FinBERT model. Embeddings are required for this system to function. Error: {}", e)
                )
            })?;
        
        Ok(Self {
            embedding_dim: 768, // FinBERT uses 768-dimensional embeddings
            finbert: RefCell::new(finbert),
        })
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn generate_event_embedding(&self, event: &FinancialEvent) -> Result<Vec<f32>> {
        let combined_text = format!("{} {} {} {}", 
            event.title, 
            event.content, 
            event.source,
            event.assets.iter().map(|a| &a.symbol).cloned().collect::<Vec<_>>().join(" ")
        );
        
        debug!("Generating FinBERT embedding for event: {}", event.id);
        let mut finbert = self.finbert.borrow_mut();
        let embedding = finbert.generate_embedding(&combined_text)
            .map_err(|e| VectorDBError::embedding_generation(
                format!("Failed to generate FinBERT embedding: {}", e)
            ))?;
        
        Ok(embedding)
    }
    
    /// multi-vector embeddings for specialized search
    pub fn generate_multi_embeddings(&self, event: &FinancialEvent) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        debug!("Generating multi-vector FinBERT embeddings for event: {}", event.id);
        let mut finbert = self.finbert.borrow_mut();
        
        // 1. Content-focused embedding (title + content)
        let content_text = format!("{} {}", event.title, event.content);
        let content_embedding = finbert.generate_embedding(&content_text)
            .map_err(|e| VectorDBError::embedding_generation(
                format!("Failed to generate content embedding: {}", e)
            ))?;
        
        // 2. Sentiment-focused embedding (sentiment + trading signal context)
        let sentiment_text = format!("{} {} sentiment: {} confidence: {:.2} signal: {:.2}", 
            event.title, 
            event.content,
            event.sentiment.label.as_str(),
            event.sentiment.confidence,
            event.sentiment.trading_signal
        );
        let sentiment_embedding = finbert.generate_embedding(&sentiment_text)
            .map_err(|e| VectorDBError::embedding_generation(
                format!("Failed to generate sentiment embedding: {}", e)
            ))?;
        
        // 3. Asset-focused embedding (assets + context)
        let asset_text = if !event.assets.is_empty() {
            let asset_info = event.assets.iter()
                .map(|a| format!("{} {} {} {}", 
                    a.symbol, 
                    a.name.as_deref().unwrap_or(""), 
                    a.asset_type.as_str(),
                    a.context
                ))
                .collect::<Vec<_>>()
                .join(" ");
            format!("{} {} assets: {}", event.title, event.content, asset_info)
        } else {
            format!("{} {} general financial", event.title, event.content)
        };
        let asset_embedding = finbert.generate_embedding(&asset_text)
            .map_err(|e| VectorDBError::embedding_generation(
                format!("Failed to generate asset embedding: {}", e)
            ))?;
        
        Ok((content_embedding, sentiment_embedding, asset_embedding))
    }

    pub fn generate_query_embedding(&self, query: &str) -> Result<Vec<f32>> {
        debug!("Generating query embedding for: {}", query);
        let mut finbert = self.finbert.borrow_mut();
        let embedding = finbert.generate_embedding(query)
            .map_err(|e| VectorDBError::embedding_generation(
                format!("Failed to generate query embedding: {}", e)
            ))?;
        Ok(embedding)
    }


    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VectorDBError::InvalidEmbeddingDimension {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (magnitude_a * magnitude_b))
    }
}