use thiserror::Error;

#[derive(Error, Debug)]
pub enum PreprocessingError {
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Processing error: {0}")]
    Processing(String),
    
    #[error("Network error: {0}")]
    Network(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Generic error: {0}")]
    Generic(#[from] anyhow::Error),
}

impl From<std::net::AddrParseError> for PreprocessingError {
    fn from(err: std::net::AddrParseError) -> Self {
        PreprocessingError::Network(err.to_string())
    }
}

impl From<tonic::transport::Error> for PreprocessingError {
    fn from(err: tonic::transport::Error) -> Self {
        PreprocessingError::Network(err.to_string())
    }
}

/// Result type alias for preprocessing operations
pub type Result<T> = std::result::Result<T, PreprocessingError>;