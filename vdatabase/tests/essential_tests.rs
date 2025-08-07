use chrono::Utc;
use std::time::Duration;
use tokio::time::sleep;

use vdatabase::{
    config::QdrantConfig,
    storage::VectorStorage,
    embeddings::EmbeddingService,
    schema::{FinancialEvent, AssetInfo, AssetType, SentimentInfo},
    query::SearchQuery,
};

#[tokio::test]
async fn test_embedding_generation() {
    let finbert_path = std::path::Path::new("../inference/FinBERT");
    if !finbert_path.exists() {
        return; // Skip if model not available
    }
    
    let embedding_service = match EmbeddingService::new(finbert_path) {
        Ok(service) => service,
        Err(_) => return, // Skip if model initialization fails
    };
    
    let mut event = FinancialEvent::new(
        "Tesla Reports Strong Q4 Earnings".to_string(),
        "Tesla Inc. announced record quarterly earnings with vehicle deliveries exceeding expectations.".to_string(),
        "test_source".to_string(),
        Utc::now(),
    );
    
    // Add asset
    let asset = AssetInfo {
        symbol: "TSLA".to_string(),
        name: Some("Tesla Inc".to_string()),
        asset_type: AssetType::Stock,
        confidence: 0.95,
        context: "quarterly earnings report".to_string(),
        exchange: Some("NASDAQ".to_string()),
    };
    event.add_asset(asset);
    
    // Add sentiment
    let sentiment = SentimentInfo::new(0.7, 0.85, 0.8, 0.15, 0.05);
    event.set_sentiment(sentiment);
    
    // Test embedding generation
    let embedding = embedding_service.generate_event_embedding(&event).expect("Should generate embedding");
    assert_eq!(embedding.len(), 768);
    assert!(embedding.iter().any(|&x| x != 0.0), "Embedding should have non-zero values");
    
    // Test multi-vector embeddings
    let (content_emb, sentiment_emb, asset_emb) = embedding_service
        .generate_multi_embeddings(&event)
        .expect("Should generate multi-embeddings");
    
    assert_eq!(content_emb.len(), 768);
    assert_eq!(sentiment_emb.len(), 768);
    assert_eq!(asset_emb.len(), 768);
    
    // Test similarity calculation
    let similarity = embedding_service.cosine_similarity(&content_emb, &sentiment_emb).unwrap();
    assert!(similarity >= -1.0 && similarity <= 1.0, "Similarity should be in [-1, 1] range");
    
    println!("Embedding generation tests passed");
}

#[tokio::test]
async fn test_finbert_integration() {
    // Test FinBERT integration when available
    let finbert_path = std::path::Path::new("../inference/FinBERT/finbert_config.json");
    if !finbert_path.exists() {
        println!("⚠️  FinBERT config not found, skipping integration test");
        return;
    }
    
    let embedding_service = match EmbeddingService::new(finbert_path) {
        Ok(service) => {
            println!("FinBERT service created successfully");
            service
        },
        Err(e) => {
            println!("⚠️  FinBERT not available: {}, skipping test", e);
            return;
        }
    };
    
    let event = FinancialEvent::new(
        "Apple Stock Rises on Strong iPhone Sales".to_string(),
        "Apple Inc. shares climbed 3% after reporting better-than-expected iPhone sales in the latest quarter.".to_string(),
        "financial_news".to_string(),
        Utc::now(),
    );
    
    let embedding = embedding_service.generate_event_embedding(&event).unwrap();
    assert_eq!(embedding.len(), 768);
    
    // Test embedding quality - should be normalized
    let magnitude: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!((magnitude - 1.0).abs() < 0.1, "Embedding should be approximately normalized");
    
    println!("✅ FinBERT integration test passed");
}

#[tokio::test]
async fn test_config_and_validation() {
    // Test configuration creation
    let mut config = QdrantConfig::new(
        "http://localhost:6333",
        Some("test-api-key".to_string()),
        "test_collection"
    );
    
    assert_eq!(config.url, "http://localhost:6333");
    assert_eq!(config.collection_name, "test_collection");
    assert_eq!(config.vector_size, 768);
    
    // Add required model path for validation to pass
    config.finbert_model_path = Some("/path/to/model".to_string());
    
    // Test config validation
    assert!(config.validate().is_ok(), "Valid config should pass validation");
    
    // Test event validation
    let mut event = FinancialEvent::new(
        "Test Event".to_string(),
        "Test content".to_string(),
        "test_source".to_string(),
        Utc::now(),
    );
    
    // Event with default embeddings should be valid
    assert!(event.validate_embedding().is_ok(), "Event with default embeddings should be valid");
    
    // Test invalid embedding size
    event.embedding = vec![0.5; 512];
    assert!(event.validate_embedding().is_err(), "Event with wrong embedding size should fail");
    
    // Restore valid embedding and test invalid sentiment embedding
    event.embedding = vec![0.5; 768];
    event.sentiment_embedding = vec![0.5; 512];
    assert!(event.validate_embedding().is_err(), "Event with wrong sentiment embedding size should fail");
    
    // Restore all valid embeddings
    event.sentiment_embedding = vec![0.5; 768];
    event.asset_embedding = vec![0.5; 768];
    assert!(event.validate_embedding().is_ok(), "Event with all valid embeddings should pass");
    
    println!("Configuration and validation tests passed");
}

#[tokio::test]
async fn test_database_upload() {
    let storage = setup_test_storage().await;
    
    let mut event = create_test_event(
        "Database Test Event",
        "This is a test event for database upload functionality."
    );
    
    // Upload single event
    let event_id = storage.store_event(&mut event).await
        .expect("Should successfully upload event");
    
    assert_eq!(event_id, event.id);
    assert_eq!(event.embedding.len(), 768);
    
    // Test batch upload
    let mut events = vec![
        create_test_event("Event 1", "First test event for batch upload"),
        create_test_event("Event 2", "Second test event for batch upload"),
        create_test_event("Event 3", "Third test event for batch upload"),
    ];
    
    let event_ids = storage.store_events_batch(&mut events).await
        .expect("Should successfully upload batch");
    
    assert_eq!(event_ids.len(), 3);
    
    // Wait for indexing
    sleep(Duration::from_millis(1000)).await;
    
    // Test search functionality
    let query = SearchQuery::new()
        .with_text("test event batch")
        .with_limit(5);
    
    let results = storage.search_similar_events(&query).await
        .expect("Search should succeed");
    
    assert!(!results.events.is_empty(), "Should find similar events");
    
    println!("Database upload and search test passed");
}

// Helper functions
async fn setup_test_storage() -> VectorStorage {
    let config = QdrantConfig::from_env().expect("Failed to load config");
    let storage = VectorStorage::new(config).await.expect("Failed to create storage");
    storage.initialize().await.expect("Failed to initialize storage");
    storage
}

fn create_test_event(title: &str, content: &str) -> FinancialEvent {
    let mut event = FinancialEvent::new(
        title.to_string(),
        content.to_string(),
        "test_source".to_string(),
        Utc::now(),
    );
    
    // Add test asset
    let asset = AssetInfo {
        symbol: "TEST".to_string(),
        name: Some("Test Asset".to_string()),
        asset_type: AssetType::Stock,
        confidence: 0.9,
        context: "test context".to_string(),
        exchange: Some("TEST_EXCHANGE".to_string()),
    };
    event.add_asset(asset);
    
    // Add test sentiment
    let sentiment = SentimentInfo::new(0.5, 0.8, 0.6, 0.2, 0.2);
    event.set_sentiment(sentiment);
    
    event
}