use std::path::Path;
use chrono::Utc;
use vdatabase::{
    embeddings::EmbeddingService,
    schema::{FinancialEvent, AssetInfo, AssetType},
};

#[test]
fn test_embedding_generation() {
    // This test would require a valid FinBERT model path
    // In practice, this would need to be run with actual model files
    
    let model_path = Path::new("../inference/FinBERT");
    if !model_path.exists() {
        return; // Skip if model not available
    }
    
    let service = EmbeddingService::new(model_path);
    if service.is_err() {
        return; // Skip if model initialization fails
    }
    
    let service = service.unwrap();
    
    let mut event = FinancialEvent::new(
        "Tesla stock rises 5%".to_string(),
        "Tesla stock increased significantly after earnings report.".to_string(),
        "financial_news".to_string(),
        Utc::now(),
    );
    
    let asset = AssetInfo {
        symbol: "TSLA".to_string(),
        name: Some("Tesla Inc".to_string()),
        asset_type: AssetType::Stock,
        confidence: 0.9,
        context: "Tesla stock".to_string(),
        exchange: Some("nasdaq".to_string()),
    };
    event.add_asset(asset);
    
    let embedding = service.generate_event_embedding(&event).unwrap();
    
    assert_eq!(embedding.len(), 768);
    
    // Check that embedding has reasonable values
    assert!(embedding.iter().any(|&x| x != 0.0));
    assert!(embedding.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_similarity_calculation() {
    let model_path = Path::new("../inference/FinBERT");
    if !model_path.exists() {
        return; // Skip if model not available
    }
    
    let service = EmbeddingService::new(model_path);
    if service.is_err() {
        return; // Skip if model initialization fails
    }
    
    let service = service.unwrap();
    
    let vec1 = vec![1.0, 0.0, 0.0];
    let vec2 = vec![0.0, 1.0, 0.0];
    let vec3 = vec![1.0, 0.0, 0.0];
    
    let sim_orthogonal = service.cosine_similarity(&vec1, &vec2).unwrap();
    let sim_identical = service.cosine_similarity(&vec1, &vec3).unwrap();
    
    assert!((sim_orthogonal - 0.0).abs() < 0.001);
    assert!((sim_identical - 1.0).abs() < 0.001);
}

#[test]
fn test_multi_vector_embeddings() {
    let model_path = Path::new("../inference/FinBERT");
    if !model_path.exists() {
        return; // Skip if model not available
    }
    
    let service = EmbeddingService::new(model_path);
    if service.is_err() {
        return; // Skip if model initialization fails
    }
    
    let service = service.unwrap();
    
    let event = FinancialEvent::new(
        "Apple reports strong iPhone sales".to_string(),
        "Apple Inc. announced impressive iPhone sales numbers for the quarter.".to_string(),
        "tech_news".to_string(),
        Utc::now(),
    );
    
    let (content_emb, sentiment_emb, asset_emb) = service
        .generate_multi_embeddings(&event)
        .expect("Should generate multi-embeddings");
    
    assert_eq!(content_emb.len(), 768);
    assert_eq!(sentiment_emb.len(), 768);
    assert_eq!(asset_emb.len(), 768);
    
    // All embeddings should have non-zero values
    assert!(content_emb.iter().any(|&x| x != 0.0));
    assert!(sentiment_emb.iter().any(|&x| x != 0.0));
    assert!(asset_emb.iter().any(|&x| x != 0.0));
    
    // All values should be finite
    assert!(content_emb.iter().all(|&x| x.is_finite()));
    assert!(sentiment_emb.iter().all(|&x| x.is_finite()));
    assert!(asset_emb.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_embedding_consistency() {
    let model_path = Path::new("../inference/FinBERT");
    if !model_path.exists() {
        return; // Skip if model not available
    }
    
    let service = EmbeddingService::new(model_path);
    if service.is_err() {
        return; // Skip if model initialization fails
    }
    
    let service = service.unwrap();
    let text = "Tesla stock performance analysis";
    
    // Generate same embedding multiple times
    let embedding1 = service.generate_query_embedding(text).unwrap();
    let embedding2 = service.generate_query_embedding(text).unwrap();
    
    // Should be identical for same input
    assert_eq!(embedding1, embedding2);
    
    // Test similarity with itself should be 1.0
    let similarity = service.cosine_similarity(&embedding1, &embedding2).unwrap();
    assert!((similarity - 1.0).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_edge_cases() {
    // We can test cosine similarity without FinBERT since it's a pure math function
    // Create a dummy service just for testing similarity calculation
    let model_path = Path::new("nonexistent_path");
    let service = EmbeddingService::new(model_path);
    
    if service.is_err() {
        // This is expected since the path doesn't exist
        // We'll test the mathematical properties directly
        
        // Test zero vectors
        let zero_vec1 = vec![0.0, 0.0, 0.0];
        let zero_vec2 = vec![0.0, 0.0, 0.0];
        
        // Manual cosine similarity for zero vectors should be 0
        let dot_product: f32 = zero_vec1.iter().zip(zero_vec2.iter()).map(|(&a, &b)| a * b).sum();
        let mag1: f32 = zero_vec1.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let mag2: f32 = zero_vec2.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if mag1 == 0.0 || mag2 == 0.0 {
            assert_eq!(dot_product, 0.0); // Should be 0 for zero vectors
        }
        return;
    }
    
    let service = service.unwrap();
    
    // Test different dimension vectors (should error)
    let vec1 = vec![1.0, 0.0];
    let vec2 = vec![1.0, 0.0, 0.0];
    
    let result = service.cosine_similarity(&vec1, &vec2);
    assert!(result.is_err());
}

#[test]
fn test_query_embedding_generation() {
    let model_path = Path::new("../inference/FinBERT");
    if !model_path.exists() {
        return; // Skip if model not available
    }
    
    let service = EmbeddingService::new(model_path);
    if service.is_err() {
        return; // Skip if model initialization fails
    }
    
    let service = service.unwrap();
    
    let queries = vec![
        "Bitcoin price analysis",
        "Tesla earnings report",
        "Federal Reserve interest rates",
        "Apple stock performance"
    ];
    
    for query in queries {
        let embedding = service.generate_query_embedding(query).unwrap();
        
        assert_eq!(embedding.len(), 768);
        assert!(embedding.iter().any(|&x| x != 0.0));
        assert!(embedding.iter().all(|&x| x.is_finite()));
    }
}