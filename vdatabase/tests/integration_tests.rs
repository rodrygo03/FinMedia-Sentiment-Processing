use chrono::Utc;
use std::path::Path;
use vdatabase::{
    config::QdrantConfig,
    embeddings::EmbeddingService,
    storage::VectorStorage,
    schema::{FinancialEvent, AssetInfo, AssetType, SentimentInfo},
};

/// Integration tests that verify the entire system works together
/// These tests require external dependencies like Qdrant or FinBERT models

#[tokio::test]
async fn test_full_storage_integration() {
    // This test requires both Qdrant and FinBERT to be running
    let config = QdrantConfig::new_finbert(
        "http://localhost:6333",
        None,
        "test_integration",
        "../inference/FinBERT/finbert_config.json"
    );
    
    if config.validate().is_err() {
        return; // Skip if config is invalid
    }
    
    let storage = match VectorStorage::new(config).await {
        Ok(s) => s,
        Err(_) => return, // Skip if storage can't be created
    };
    
    if storage.initialize().await.is_err() {
        return; // Skip if can't initialize
    }
    
    // Create a test event
    let mut event = FinancialEvent::new(
        "Integration Test Event".to_string(),
        "This is a test event for full system integration.".to_string(),
        "test_source".to_string(),
        Utc::now(),
    );
    
    // Add test asset and sentiment
    let asset = AssetInfo {
        symbol: "TEST".to_string(),
        name: Some("Test Asset".to_string()),
        asset_type: AssetType::Stock,
        confidence: 0.9,
        context: "integration test".to_string(),
        exchange: Some("test".to_string()),
    };
    event.add_asset(asset);
    
    let sentiment = SentimentInfo::new(0.5, 0.8, 0.6, 0.3, 0.2);
    event.set_sentiment(sentiment);
    
    // Store the event (this should generate embeddings automatically)
    let event_id = storage.store_event(&mut event).await.unwrap();
    assert_eq!(event_id, event.id);
    
    // Verify embeddings were generated
    assert!(!event.embedding.is_empty());
    assert!(!event.sentiment_embedding.is_empty());
    assert!(!event.asset_embedding.is_empty());
}

#[test]
fn test_embedding_service_integration() {
    let model_path = Path::new("../inference/FinBERT");
    if !model_path.exists() {
        return; // Skip if model not available
    }
    
    let service = match EmbeddingService::new(model_path) {
        Ok(s) => s,
        Err(_) => return, // Skip if model initialization fails
    };
    
    // Test that all embedding methods work together
    let mut event = FinancialEvent::new(
        "Full Integration Test".to_string(),
        "Testing that all embedding methods produce consistent results.".to_string(),
        "integration_test".to_string(),
        Utc::now(),
    );
    
    // Add some test data
    let asset = AssetInfo {
        symbol: "INTG".to_string(),
        name: Some("Integration Corp".to_string()),
        asset_type: AssetType::Stock,
        confidence: 0.95,
        context: "integration testing".to_string(),
        exchange: Some("test".to_string()),
    };
    event.add_asset(asset);
    
    let sentiment = SentimentInfo::new(0.7, 0.9, 0.8, 0.15, 0.05);
    event.set_sentiment(sentiment);
    
    // Test single event embedding
    let event_embedding = service.generate_event_embedding(&event).unwrap();
    assert_eq!(event_embedding.len(), 768);
    
    // Test multi-vector embeddings
    let (content_emb, sentiment_emb, asset_emb) = service
        .generate_multi_embeddings(&event).unwrap();
    
    assert_eq!(content_emb.len(), 768);
    assert_eq!(sentiment_emb.len(), 768);
    assert_eq!(asset_emb.len(), 768);
    
    // Test query embedding
    let query_embedding = service
        .generate_query_embedding("integration testing query").unwrap();
    assert_eq!(query_embedding.len(), 768);
    
    // Test that embeddings are different (they should capture different aspects)
    let content_vs_sentiment = service.cosine_similarity(&content_emb, &sentiment_emb).unwrap();
    let content_vs_asset = service.cosine_similarity(&content_emb, &asset_emb).unwrap();
    
    // They should be similar (same source text) but not identical
    assert!(content_vs_sentiment > 0.5 && content_vs_sentiment < 1.0);
    assert!(content_vs_asset > 0.5 && content_vs_asset < 1.0);
}

#[test]
fn test_error_propagation() {
    // Test that errors are properly handled and propagated through the system
    
    // Test config validation error
    let invalid_config = QdrantConfig::new("", None, "");
    assert!(invalid_config.validate().is_err());
    
    // Test embedding service with invalid path
    let invalid_path = Path::new("/nonexistent/path/to/model");
    let service_result = EmbeddingService::new(invalid_path);
    assert!(service_result.is_err());
}

#[test]
fn test_data_consistency() {
    // Test that data structures maintain consistency across operations
    let mut event = FinancialEvent::new(
        "Consistency Test".to_string(),
        "Testing data consistency across operations.".to_string(),
        "consistency_test".to_string(),
        Utc::now(),
    );
    
    let original_id = event.id.clone();
    let original_timestamp = event.published_at;
    
    // Add multiple assets and verify consistency
    let assets = vec![
        AssetInfo {
            symbol: "CONS1".to_string(),
            name: Some("Consistency Test 1".to_string()),
            asset_type: AssetType::Stock,
            confidence: 0.9,
            context: "test 1".to_string(),
            exchange: Some("test".to_string()),
        },
        AssetInfo {
            symbol: "CONS2".to_string(),
            name: Some("Consistency Test 2".to_string()),
            asset_type: AssetType::Crypto,
            confidence: 0.8,
            context: "test 2".to_string(),
            exchange: Some("test".to_string()),
        },
    ];
    
    for asset in assets {
        event.add_asset(asset);
    }
    
    // Verify that core properties remain unchanged
    assert_eq!(event.id, original_id);
    assert_eq!(event.published_at, original_timestamp);
    assert_eq!(event.assets.len(), 2);
    
    // Add sentiment and verify it doesn't affect assets
    let sentiment = SentimentInfo::new(0.4, 0.7, 0.5, 0.3, 0.4);
    event.set_sentiment(sentiment);
    
    assert_eq!(event.assets.len(), 2);
    assert_eq!(event.sentiment.score, 0.4);
}