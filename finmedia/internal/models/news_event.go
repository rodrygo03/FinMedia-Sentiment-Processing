package models

import (
	"time"
	
	"finmedia/internal/assets"
)

type NewsEvent struct {
	Title        string                `json:"title"`
	Content      string                `json:"content"`
	PublishedAt  time.Time             `json:"published_at"`
	Source       string                `json:"source"`
	URL          string                `json:"url"`
	Assets       []assets.AssetMatch   `json:"assets"`       // Enhanced from AssetTags
	Categories   []string              `json:"categories"`   // crypto, stock, forex, monetary, economic, geopolitical
	Sentiment    float64               `json:"sentiment"`    // -1 to 1 (future use)
	Confidence   float64               `json:"confidence"`   // Overall detection confidence
	NewsType     string                `json:"news_type"`    // financial, political, geopolitical
	MarketImpact string                `json:"market_impact"` // high, medium, low
}
