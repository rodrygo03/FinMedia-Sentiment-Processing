use crate::Result;

#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub max_text_length: usize,
    pub batch_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 50051,
            max_text_length: 10000,
            batch_size: 32,
        }
    }
}

impl Config {
    /// Load configuration from environment variables with fallback to defaults
    pub fn from_env_or_default() -> Self {
        let mut config = Self::default();
        
        // Override with environment variables if present
        if let Ok(host) = std::env::var("PREPROCESSING_HOST") {
            config.host = host;
        }
        
        if let Ok(port_str) = std::env::var("PREPROCESSING_PORT") {
            if let Ok(port) = port_str.parse::<u16>() {
                config.port = port;
            }
        }
        
        if let Ok(max_length_str) = std::env::var("PREPROCESSING_MAX_TEXT_LENGTH") {
            if let Ok(max_length) = max_length_str.parse::<usize>() {
                config.max_text_length = max_length;
            }
        }
        
        if let Ok(batch_size_str) = std::env::var("PREPROCESSING_BATCH_SIZE") {
            if let Ok(batch_size) = batch_size_str.parse::<usize>() {
                config.batch_size = batch_size;
            }
        }
        
        config
    }
    
    pub fn server_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
    
    pub fn validate(&self) -> Result<()> {
        if self.host.is_empty() {
            return Err(crate::error::PreprocessingError::Config(
                "Host cannot be empty".to_string()
            ));
        }
        
        if self.port == 0 {
            return Err(crate::error::PreprocessingError::Config(
                "Port cannot be zero".to_string()
            ));
        }
        
        if self.max_text_length == 0 {
            return Err(crate::error::PreprocessingError::Config(
                "Max text length must be greater than zero".to_string()
            ));
        }
        
        if self.batch_size == 0 {
            return Err(crate::error::PreprocessingError::Config(
                "Batch size must be greater than zero".to_string()
            ));
        }
        
        Ok(())
    }
}