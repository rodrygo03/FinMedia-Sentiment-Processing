package assets

import (
	"sort"
	"strings"
)

// represents a financial instrument with detection patterns
type Asset struct {
	Symbol      string   `json:"symbol"`       // BTC, AAPL, etc.
	Name        string   `json:"name"`         // Bitcoin, Apple Inc.
	Type        string   `json:"type"`         // crypto, stock, forex
	Patterns    []string `json:"patterns"`     // Detection keywords
	Exchanges   []string `json:"exchanges"`    // NYSE, NASDAQ, Binance
	Priority    int      `json:"priority"`     // Higher = more important
	Enabled     bool     `json:"enabled"`      // Can disable specific assets
}

type AssetDetection struct {
	Assets              []Asset           `json:"assets"`                // Direct asset list (legacy)
	AssetsFile          string           `json:"assets_file"`           // Path to external assets file
	CaseSensitive       bool             `json:"case_sensitive"`
	MinConfidence       float64          `json:"min_confidence"`
	ContextWords        []string         `json:"context_words"`
	ExcludePatterns     []string         `json:"exclude_patterns"`
	CategoryFilters     []string         `json:"category_filters"`
}

//  represents a detected asset with metadata
type AssetMatch struct {
	Symbol     string    `json:"symbol"`
	Name       string    `json:"name"`
	Type       string    `json:"type"`
	Confidence float64   `json:"confidence"`
	Contexts   []string  `json:"contexts"`
}

// handles configurable asset detection
type AssetDetector struct {
	config   *AssetDetection
	assets   map[string]Asset
	patterns map[string][]string
}

// creates a new asset detector with configuration
func NewAssetDetector(config *AssetDetection) *AssetDetector {
	detector := &AssetDetector{
		config:   config,
		assets:   make(map[string]Asset),
		patterns: make(map[string][]string),
	}
	
	// Build lookup maps for efficient detection
	for _, asset := range config.Assets {
		if asset.Enabled {
			detector.assets[asset.Symbol] = asset
			detector.patterns[asset.Symbol] = asset.Patterns
		}
	}
	
	return detector
}

// performs configurable asset detection
func (d *AssetDetector) ExtractAssetTags(title, content string) []AssetMatch {
	text := title + " " + content
	if !d.config.CaseSensitive {
		text = strings.ToLower(text)
	}
	
	var matches []AssetMatch
	
	for symbol, asset := range d.assets {
		confidence := d.calculateConfidence(text, asset)
		if confidence >= d.config.MinConfidence {
			matches = append(matches, AssetMatch{
				Symbol:     symbol,
				Name:       asset.Name,
				Type:       asset.Type,
				Confidence: confidence,
				Contexts:   d.extractContexts(text, asset.Patterns),
			})
		}
	}
	
	// Sort by priority and confidence
	sort.Slice(matches, func(i, j int) bool {
		if matches[i].Confidence != matches[j].Confidence {
			return matches[i].Confidence > matches[j].Confidence
		}
		return d.assets[matches[i].Symbol].Priority > d.assets[matches[j].Symbol].Priority
	})
	
	return matches
}

// calculateConfidence determines how confident we are about an asset match
func (d *AssetDetector) calculateConfidence(text string, asset Asset) float64 {
	var confidence float64
	patternMatches := 0
	contextMatches := 0
	
	// Check for pattern matches
	for _, pattern := range asset.Patterns {
		if !d.config.CaseSensitive {
			pattern = strings.ToLower(pattern)
		}
		
		if strings.Contains(text, pattern) {
			patternMatches++
			// Longer patterns get higher confidence
			confidence += float64(len(pattern)) * 0.1
		}
	}
	
	// Check for context words
	for _, contextWord := range d.config.ContextWords {
		if !d.config.CaseSensitive {
			contextWord = strings.ToLower(contextWord)
		}
		
		if strings.Contains(text, contextWord) {
			contextMatches++
			confidence += 0.2
		}
	}
	
	// Check for exclude patterns
	for _, excludePattern := range d.config.ExcludePatterns {
		if !d.config.CaseSensitive {
			excludePattern = strings.ToLower(excludePattern)
		}
		
		if strings.Contains(text, excludePattern) {
			confidence -= 0.5
		}
	}
	
	// Base confidence calculation
	if patternMatches > 0 {
		confidence = confidence / float64(len(asset.Patterns))
		
		// Bonus for multiple pattern matches
		if patternMatches > 1 {
			confidence += 0.3
		}
		
		// Bonus for context matches
		if contextMatches > 0 {
			confidence += 0.2
		}
		
		// Priority bonus
		confidence += float64(asset.Priority) * 0.05
	}
	
	// Normalize confidence to 0-1 range
	if confidence > 1.0 {
		confidence = 1.0
	}
	if confidence < 0.0 {
		confidence = 0.0
	}
	
	return confidence
}

// extractContexts finds relevant context around pattern matches
func (d *AssetDetector) extractContexts(text string, patterns []string) []string {
	var contexts []string
	words := strings.Fields(text)
	
	for _, pattern := range patterns {
		if !d.config.CaseSensitive {
			pattern = strings.ToLower(pattern)
		}
		
		// Find pattern in text and extract surrounding context
		for i, word := range words {
			if strings.Contains(word, pattern) {
				// Extract 3 words before and after
				start := i - 3
				if start < 0 {
					start = 0
				}
				end := i + 4
				if end > len(words) {
					end = len(words)
				}
				
				context := strings.Join(words[start:end], " ")
				contexts = append(contexts, context)
			}
		}
	}
	
	return contexts
}

// GetAssetsByType returns assets filtered by type
func (d *AssetDetector) GetAssetsByType(assetType string) []Asset {
	var assets []Asset
	for _, asset := range d.assets {
		if asset.Type == assetType {
			assets = append(assets, asset)
		}
	}
	return assets
}

// GetEnabledAssets returns all enabled assets
func (d *AssetDetector) GetEnabledAssets() []Asset {
	var assets []Asset
	for _, asset := range d.assets {
		if asset.Enabled {
			assets = append(assets, asset)
		}
	}
	return assets
}

// UpdateAsset updates an existing asset or adds a new one
func (d *AssetDetector) UpdateAsset(asset Asset) {
	if asset.Enabled {
		d.assets[asset.Symbol] = asset
		d.patterns[asset.Symbol] = asset.Patterns
	} else {
		delete(d.assets, asset.Symbol)
		delete(d.patterns, asset.Symbol)
	}
}