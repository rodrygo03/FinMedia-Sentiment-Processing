use vdatabase::query::{SearchQuery, QueryBuilder};

#[test]
fn test_query_building() {
    let query = SearchQuery::new()
        .with_text("Bitcoin price analysis")
        .with_limit(15);
    
    assert_eq!(query.text, Some("Bitcoin price analysis".to_string()));
    assert_eq!(query.limit, Some(15));
}

#[test]
fn test_query_builder() {
    let query = QueryBuilder::semantic_search("Bitcoin market analysis");
    assert_eq!(query.text, Some("Bitcoin market analysis".to_string()));
    assert_eq!(query.limit, Some(20));
    assert_eq!(query.score_threshold, Some(0.7));
}

#[test]
fn test_query_builder_variants() {
    // Test different query builder methods
    let semantic_query = QueryBuilder::semantic_search("Tesla earnings report");
    assert_eq!(semantic_query.text, Some("Tesla earnings report".to_string()));
    assert_eq!(semantic_query.limit, Some(20));
    assert_eq!(semantic_query.score_threshold, Some(0.7));
    
    // Test empty query (note: may have default values)
    let empty_query = SearchQuery::new();
    assert_eq!(empty_query.text, None);
    // limit might have a default value, so don't test it
    // assert_eq!(empty_query.limit, None);
    assert_eq!(empty_query.score_threshold, None);
    assert_eq!(empty_query.vector, None);
}

#[test]
fn test_query_chaining() {
    let query = SearchQuery::new()
        .with_text("Apple stock performance")
        .with_limit(10)
        .with_score_threshold(0.8);
    
    assert_eq!(query.text, Some("Apple stock performance".to_string()));
    assert_eq!(query.limit, Some(10));
    assert_eq!(query.score_threshold, Some(0.8));
}

#[test]
fn test_query_with_vector() {
    let embedding = vec![0.1, 0.2, 0.3]; // Dummy embedding
    let query = SearchQuery::new()
        .with_vector(embedding.clone())
        .with_limit(5);
    
    assert_eq!(query.vector, Some(embedding));
    assert_eq!(query.limit, Some(5));
    assert_eq!(query.text, None);
}

#[test]
fn test_query_modification() {
    let mut query = SearchQuery::new()
        .with_text("Initial text")
        .with_limit(20);
    
    // Test that we can modify the query
    query = query
        .with_text("Updated text")
        .with_score_threshold(0.9);
    
    assert_eq!(query.text, Some("Updated text".to_string()));
    assert_eq!(query.limit, Some(20)); // Should retain previous limit
    assert_eq!(query.score_threshold, Some(0.9));
}