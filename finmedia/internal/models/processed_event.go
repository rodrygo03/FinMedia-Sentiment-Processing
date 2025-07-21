package models

import (
	"time"
)

// ProcessedEvent represents a news event after preprocessing by the Rust service
type ProcessedEvent struct {
	ID               string    `json:"id"`
	OriginalEvent    NewsEvent `json:"original_event"`
	ProcessedText    string    `json:"processed_text"`    // Cleaned text ready for ML inference
	Tokens           []string  `json:"tokens"`            // Will be populated by ML service later
	AssetMentions    []string  `json:"asset_mentions"`    // Will be populated by ML service later
	SentimentScore   float64   `json:"sentiment_score"`   // Will be calculated by ML service later
	Confidence       float64   `json:"confidence"`        // Will be calculated by ML service later
	ProcessedAt      time.Time `json:"processed_at"`
}