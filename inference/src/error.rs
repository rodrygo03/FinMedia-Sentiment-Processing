use thiserror::Error;
use anyhow::Error as AnyhowError;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model loading failed: {0}")]
    ModelLoad(String),
    
    #[error("Tokenization failed: {0}")]
    Tokenization(String),
    
    #[error("ONNX inference failed: {0}")]
    OnnxInference(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Input validation failed: {0}")]
    InvalidInput(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Other error: {0}")]
    Other(#[from] AnyhowError),
}

pub type Result<T> = std::result::Result<T, InferenceError>;