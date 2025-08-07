use chrono::Utc;
use vdatabase::schema::{FinancialEvent, AssetInfo, AssetType, SentimentInfo};

#[test]
fn test_financial_event_creation() {
    let event = FinancialEvent::new(
        "Bitcoin price surges".to_string(),
        "Bitcoin reached new heights today...".to_string(),
        "crypto_news".to_string(),
        Utc::now(),
    );
    
    assert!(!event.id.is_empty());
    assert_eq!(event.title, "Bitcoin price surges");
    assert_eq!(event.source, "crypto_news");
    assert_eq!(event.embedding.len(), FinancialEvent::EMBEDDING_DIM);
}

#[test]
fn test_asset_and_sentiment_integration() {
    let mut event = FinancialEvent::new(
        "Microsoft cloud revenue grows".to_string(),
        "Microsoft reported strong Azure growth in latest earnings.".to_string(),
        "tech_earnings".to_string(),
        Utc::now(),
    );
    
    // Add asset
    let asset = AssetInfo {
        symbol: "MSFT".to_string(),
        name: Some("Microsoft Corporation".to_string()),
        asset_type: AssetType::Stock,
        confidence: 0.95,
        context: "cloud revenue earnings".to_string(),
        exchange: Some("nasdaq".to_string()),
    };
    event.add_asset(asset);
    
    // Add sentiment
    let sentiment = SentimentInfo::new(0.6, 0.88, 0.7, 0.2, 0.1);
    event.set_sentiment(sentiment);
    
    assert_eq!(event.assets.len(), 1);
    assert_eq!(event.assets[0].symbol, "MSFT");
    assert_eq!(event.sentiment.score, 0.6);
    assert_eq!(event.sentiment.confidence, 0.88);
}

#[test]
fn test_error_handling() {
    // Test invalid embedding dimensions
    let mut event = FinancialEvent::new(
        "Test".to_string(),
        "Test content".to_string(),
        "test".to_string(),
        Utc::now(),
    );
    
    // Invalid embedding size
    event.embedding = vec![0.5; 512]; // Wrong size
    let result = event.validate_embedding();
    assert!(result.is_err());
    
    // Valid embedding size
    event.embedding = vec![0.5; 768]; // Correct size
    let result = event.validate_embedding();
    assert!(result.is_ok());
}

#[test]
fn test_asset_types() {
    let crypto_asset = AssetInfo {
        symbol: "BTC".to_string(),
        name: Some("Bitcoin".to_string()),
        asset_type: AssetType::Crypto,
        confidence: 0.9,
        context: "crypto trading".to_string(),
        exchange: Some("coinbase".to_string()),
    };
    
    let stock_asset = AssetInfo {
        symbol: "AAPL".to_string(),
        name: Some("Apple Inc".to_string()),
        asset_type: AssetType::Stock,
        confidence: 0.95,
        context: "technology stock".to_string(),
        exchange: Some("nasdaq".to_string()),
    };
    
    assert_eq!(crypto_asset.asset_type.as_str(), "crypto");
    assert_eq!(stock_asset.asset_type.as_str(), "stock");
}

#[test]
fn test_sentiment_info_creation() {
    let sentiment = SentimentInfo::new(0.7, 0.85, 0.15, 0.05, 0.8);
    
    assert_eq!(sentiment.score, 0.7);
    assert_eq!(sentiment.confidence, 0.85);
    assert_eq!(sentiment.trading_signal, 0.595); // 0.7 * 0.85
    assert_eq!(sentiment.positive_score, 0.15);
    assert_eq!(sentiment.negative_score, 0.05);
    
    // Verify the label is set correctly based on score (0.7 > 0.5 = very_positive)
    assert_eq!(sentiment.label.as_str(), "very_positive");
}

#[test]
fn test_multi_asset_event() {
    let mut event = FinancialEvent::new(
        "Tech stocks rally on earnings".to_string(),
        "Apple and Microsoft both reported strong quarterly results.".to_string(),
        "market_news".to_string(),
        Utc::now(),
    );
    
    // Add multiple assets
    let apple_asset = AssetInfo {
        symbol: "AAPL".to_string(),
        name: Some("Apple Inc".to_string()),
        asset_type: AssetType::Stock,
        confidence: 0.95,
        context: "quarterly earnings".to_string(),
        exchange: Some("nasdaq".to_string()),
    };
    
    let msft_asset = AssetInfo {
        symbol: "MSFT".to_string(),
        name: Some("Microsoft Corporation".to_string()),
        asset_type: AssetType::Stock,
        confidence: 0.90,
        context: "strong results".to_string(),
        exchange: Some("nasdaq".to_string()),
    };
    
    event.add_asset(apple_asset);
    event.add_asset(msft_asset);
    
    assert_eq!(event.assets.len(), 2);
    assert_eq!(event.assets[0].symbol, "AAPL");
    assert_eq!(event.assets[1].symbol, "MSFT");
    
    // Test that both assets are properly stored
    let symbols: Vec<&String> = event.assets.iter().map(|a| &a.symbol).collect();
    assert!(symbols.contains(&&"AAPL".to_string()));
    assert!(symbols.contains(&&"MSFT".to_string()));
}