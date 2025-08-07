use chrono::Utc;
use std::time::Duration;
use tokio::time::sleep;

use vdatabase::{
    config::QdrantConfig,
    storage::VectorStorage,
    schema::{FinancialEvent, AssetInfo, AssetType, SentimentInfo},
    query::SearchQuery,
};

async fn setup_real_storage() -> Result<VectorStorage, Box<dyn std::error::Error>> {
    let config = QdrantConfig::from_env()?;
    let storage = VectorStorage::new(config).await?;
    storage.initialize().await?;
    
    // Give Qdrant time to initialize
    sleep(Duration::from_millis(500)).await;
    
    Ok(storage)
}

fn create_test_event(title: &str, content: &str, symbol: &str, asset_type: AssetType, sentiment: f64) -> FinancialEvent {
    let mut event = FinancialEvent::new(
        title.to_string(),
        content.to_string(),
        "test_source".to_string(),
        Utc::now(),
    );
    
    let asset = AssetInfo {
        symbol: symbol.to_string(),
        name: Some(format!("{} Asset", symbol)),
        asset_type,
        confidence: 0.9,
        context: format!("{} financial context", symbol),
        exchange: Some("TEST_EXCHANGE".to_string()),
    };
    event.add_asset(asset);
    
    let confidence = 0.85;
    let positive = if sentiment > 0.0 { sentiment } else { 0.0 };
    let negative = if sentiment < 0.0 { sentiment.abs() } else { 0.0 };
    let neutral = 1.0 - positive - negative;
    
    let sentiment_info = SentimentInfo::new(sentiment, confidence, positive, negative, neutral);
    event.set_sentiment(sentiment_info);
    
    event
}

#[tokio::test]
async fn test_upload_financial_data_to_database() {
    println!("Testing database upload with financial events...");
    
    let storage = setup_real_storage().await.expect("Failed to connect to database");
    
    // Create sample financial events
    let mut events = vec![
        create_test_event("Tesla Reports Strong Q4 Earnings", 
            "Tesla Inc. reported record quarterly earnings with strong vehicle deliveries.",
            "TSLA", AssetType::Stock, 0.75),
        
        create_test_event("Bitcoin Reaches New High",
            "Bitcoin surged past $50,000 as institutional adoption accelerates.",
            "BTC", AssetType::Crypto, 0.85),
        
        create_test_event("Apple iPhone Sales Decline",
            "Apple reported a decline in iPhone sales in China market.",
            "AAPL", AssetType::Stock, -0.35),
    ];
    
    println!("Created {} financial events", events.len());
    
    // Upload events
    let start_time = std::time::Instant::now();
    let event_ids = storage.store_events_batch(&mut events).await
        .expect("Failed to upload events");
    let upload_duration = start_time.elapsed();
    
    assert_eq!(event_ids.len(), events.len());
    
    // Verify all events have embeddings
    for (i, event) in events.iter().enumerate() {
        assert_eq!(event.embedding.len(), 768);
        assert!(event.embedding.iter().any(|&x| x != 0.0));
        println!("   Event {}: {} (ID: {})", 
            i + 1, 
            event.title.chars().take(50).collect::<String>(), 
            event_ids[i]
        );
    }
    
    // Wait for indexing
    println!("Waiting for indexing...");
    sleep(Duration::from_secs(2)).await;
    
    // Test search functionality
    let query = SearchQuery::new()
        .with_text("Tesla earnings performance")
        .with_limit(5);
    
    let results = storage.search_similar_events(&query).await
        .expect("Search should succeed");
    
    assert!(!results.events.is_empty(), "Should find similar events");
    
    println!("Search found {} results", results.events.len());
    for (i, (event, score)) in results.events.iter().take(2).enumerate() {
        println!("   {}. {} (Score: {:.3})", 
            i + 1, 
            event.title.chars().take(50).collect::<String>(), 
            score
        );
    }
    
    let throughput = events.len() as f64 / upload_duration.as_secs_f64();
    println!("\nPerformance:");
    println!("   Upload time: {:?}", upload_duration);
    println!("   Throughput: {:.1} events/second", throughput);
    
    println!("Database upload test completed successfully!");
}