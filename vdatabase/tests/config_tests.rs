use vdatabase::config::QdrantConfig;

#[test]
fn test_config_creation() {
    let config = QdrantConfig::new("http://localhost:6333", None, "test");
    assert_eq!(config.url, "http://localhost:6333");
    assert_eq!(config.collection_name, "test");
    assert_eq!(config.vector_size, 768);
    assert!(config.finbert_enabled); // Should now be enabled by default
}

#[test]
fn test_config_validation_requires_finbert() {
    let mut config = QdrantConfig::new("http://localhost:6333", None, "test");
    config.finbert_enabled = false;
    
    // Should fail validation when FinBERT is disabled
    let result = config.validate();
    assert!(result.is_err());
    
    // Should pass when FinBERT is enabled with model path
    config.finbert_enabled = true;
    config.finbert_model_path = Some("/path/to/model".to_string());
    let result = config.validate();
    assert!(result.is_ok());
}

#[test]
fn test_config_with_finbert() {
    let config = QdrantConfig::new_finbert(
        "http://localhost:6333",
        None,
        "test",
        "/path/to/finbert/config.json"
    );
    
    assert_eq!(config.url, "http://localhost:6333");
    assert_eq!(config.collection_name, "test");
    assert!(config.finbert_enabled);
    assert_eq!(config.finbert_model_path, Some("/path/to/finbert/config.json".to_string()));
    assert_eq!(config.vector_size, 768);
}

#[test]
fn test_config_validation_vector_size() {
    let mut config = QdrantConfig::new("http://localhost:6333", None, "test");
    config.finbert_model_path = Some("/path/to/model".to_string());
    
    // Valid vector size for FinBERT
    config.vector_size = 768;
    assert!(config.validate().is_ok());
    
    // Invalid vector size for FinBERT
    config.vector_size = 512;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_empty_fields() {
    let mut config = QdrantConfig::new("", None, "");
    config.finbert_model_path = Some("".to_string());
    
    // Empty URL should fail
    assert!(config.validate().is_err());
    
    // Fix URL but keep empty collection name
    config.url = "http://localhost:6333".to_string();
    assert!(config.validate().is_err());
    
    // Fix collection name but keep empty model path
    config.collection_name = "test".to_string();
    assert!(config.validate().is_err());
    
    // Fix model path - should now pass
    config.finbert_model_path = Some("/path/to/model".to_string());
    assert!(config.validate().is_ok());
}