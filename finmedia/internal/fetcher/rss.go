package fetcher

import (
	"context"
	"finmedia/internal/models"
	"finmedia/internal/assets"
	"fmt"
	"log"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/mmcdole/gofeed"
)

//  wraps HTTP operations for RSS fetching
type HTTPClient struct {
	client  *http.Client
	timeout time.Duration
}

//  handles concurrent RSS feed fetching
type RSSFetcher struct {
	feeds         []string
	client        *HTTPClient
	workers       int
	interval      time.Duration
	assetDetector *assets.AssetDetector
}

// creates a new RSS fetcher instance
func NewRSSFetcher(feeds []string, workers int, interval time.Duration, assetDetector *assets.AssetDetector) *RSSFetcher {
	if workers <= 0 {
		workers = 3
	}
	if interval <= 0 {
		interval = 5 * time.Minute
	}

	httpClient := &HTTPClient{
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		timeout: 30 * time.Second,
	}

	// Validate feed URLs before storing
	validFeeds := make([]string, 0, len(feeds))
	for _, feed := range feeds {
		if strings.HasPrefix(feed, "http://") || strings.HasPrefix(feed, "https://") {
			validFeeds = append(validFeeds, feed)
		}
	}

	return &RSSFetcher{
		feeds:         validFeeds,
		client:        httpClient,
		workers:       workers,
		interval:      interval,
		assetDetector: assetDetector,
	}
}

// begins concurrent RSS feed fetching
func (r *RSSFetcher) Start(ctx context.Context, output chan<- models.NewsEvent) error {
	feedQueue := make(chan string, len(r.feeds))
	
	// Create worker goroutines for concurrent fetching
	var wg sync.WaitGroup
	for i := 0; i < r.workers; i++ {
		wg.Add(1)
		go r.ProcessFeedWorker(ctx, feedQueue, output, &wg)
	}
	
	// Use ticker for periodic fetching at specified interval
	ticker := time.NewTicker(r.interval)
	defer ticker.Stop()
	
	go func() {
		for {
			select {
			case <-ctx.Done():
				close(feedQueue)
				return
			case <-ticker.C:
				// Distribute feeds across workers evenly
				for _, feed := range r.feeds {
					select {
					case feedQueue <- feed:
					case <-ctx.Done():
						close(feedQueue)
						return
					default:
						// Skip if queue is full
						log.Printf("Feed queue full, skipping: %s", feed)
					}
				}
			}
		}
	}()
	
	wg.Wait()
	return nil
}

// fetches and parses a single RSS feed
func (r *RSSFetcher) FetchFeed(ctx context.Context, feedURL string) ([]models.NewsEvent, error) {
	// Create HTTP request with context
	req, err := http.NewRequestWithContext(ctx, "GET", feedURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	// Make HTTP GET request to RSS feed URL
	resp, err := r.client.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch feed: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("feed returned status: %d", resp.StatusCode)
	}
	
	// Parse XML response using gofeed
	fp := gofeed.NewParser()
	feed, err := fp.Parse(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to parse feed: %w", err)
	}
	
	// Convert RSS items to NewsEvent structs
	var events []models.NewsEvent
	for _, item := range feed.Items {
		event, err := NormalizeNewsEvent(map[string]interface{}{
			"title":       item.Title,
			"description": item.Description,
			"link":        item.Link,
			"published":   item.PublishedParsed,
			"source":      feed.Title,
		})
		if err != nil {
			log.Printf("Failed to normalize news event: %v", err)
			continue
		}
		
		// Use configurable asset detection
		event.Assets = r.assetDetector.ExtractAssetTags(event.Title, event.Content)
		event.Categories = r.categorizeEvent(event.Assets)
		event.Confidence = r.calculateOverallConfidence(event.Assets)
		event.NewsType = r.determineNewsType(event.Assets)
		event.MarketImpact = r.assessMarketImpact(event.Assets, event.Confidence)
		
		events = append(events, *event)
	}
	
	return events, nil
}

// processes RSS feeds from a work queue
func (r *RSSFetcher) ProcessFeedWorker(ctx context.Context, feedQueue <-chan string, output chan<- models.NewsEvent, wg *sync.WaitGroup) {
	defer wg.Done()
	
	// Listen for feed URLs from feedQueue channel
	for {
		select {
		case <-ctx.Done():
			// Handle worker shutdown on context cancellation
			return
		case feedURL, ok := <-feedQueue:
			if !ok {
				// Channel closed, worker should exit
				return
			}
			
			// Call FetchFeed for each URL received
			events, err := r.FetchFeed(ctx, feedURL)
			if err != nil {
				log.Printf("Failed to fetch feed %s: %v", feedURL, err)
				continue
			}
			
			// Send parsed NewsEvents to output channel
			for _, event := range events {
				select {
				case output <- event:
				case <-ctx.Done():
					return
				}
			}
		}
	}
}

// categorizeEvent determines categories based on detected assets
func (r *RSSFetcher) categorizeEvent(assets []assets.AssetMatch) []string {
	categoryMap := make(map[string]bool)
	
	for _, asset := range assets {
		categoryMap[asset.Type] = true
	}
	
	var categories []string
	for category := range categoryMap {
		categories = append(categories, category)
	}
	
	return categories
}

// calculateOverallConfidence calculates the overall confidence for the news event
func (r *RSSFetcher) calculateOverallConfidence(assets []assets.AssetMatch) float64 {
	if len(assets) == 0 {
		return 0.0
	}
	
	var totalConfidence float64
	var maxConfidence float64
	
	for _, asset := range assets {
		totalConfidence += asset.Confidence
		if asset.Confidence > maxConfidence {
			maxConfidence = asset.Confidence
		}
	}
	
	// Use weighted average: 70% max confidence, 30% average confidence
	avgConfidence := totalConfidence / float64(len(assets))
	return (maxConfidence * 0.7) + (avgConfidence * 0.3)
}

// determineNewsType categorizes news based on detected asset types
func (r *RSSFetcher) determineNewsType(assets []assets.AssetMatch) string {
	if len(assets) == 0 {
		return "financial"
	}
	
	typeMap := make(map[string]int)
	for _, asset := range assets {
		typeMap[asset.Type]++
	}
	
	// Priority order: geopolitical > political > economic > financial
	if typeMap["geopolitical"] > 0 {
		return "geopolitical"
	}
	if typeMap["monetary"] > 0 {
		return "political"
	}
	if typeMap["economic"] > 0 {
		return "political"
	}
	
	return "financial"
}

// assessMarketImpact determines potential market impact based on assets and confidence
func (r *RSSFetcher) assessMarketImpact(assets []assets.AssetMatch, confidence float64) string {
	if len(assets) == 0 || confidence < 0.3 {
		return "low"
	}
	
	// Check for high-impact asset types
	highImpactTypes := map[string]bool{
		"monetary":     true,
		"geopolitical": true,
	}
	
	var maxPriority int
	var hasHighImpactType bool
	
	for _, asset := range assets {
		if maxPriority < 10 { // Assuming priority max is 10
			if priority := r.getAssetPriority(asset.Symbol); priority > maxPriority {
				maxPriority = priority
			}
		}
		if highImpactTypes[asset.Type] {
			hasHighImpactType = true
		}
	}
	
	// High impact: high confidence + (high priority asset OR high-impact type)
	if confidence >= 0.7 && (maxPriority >= 9 || hasHighImpactType) {
		return "high"
	}
	
	// Medium impact: moderate confidence + decent priority
	if confidence >= 0.5 && maxPriority >= 6 {
		return "medium"
	}
	
	return "low"
}

// getAssetPriority retrieves the priority of an asset by symbol
func (r *RSSFetcher) getAssetPriority(symbol string) int {
	// This would need access to asset detector's asset map
	// For now, return a default priority
	switch symbol {
	case "FED", "BTC", "NVDA":
		return 10
	case "INFLATION", "ETH", "META":
		return 9
	case "CHN_TRADE", "GDP", "TAO", "TSM":
		return 8
	case "WAR_UKR", "OIL_CRISIS", "GLD":
		return 7
	case "UNEMPLOYMENT", "SLV":
		return 6
	default:
		return 5
	}
}

// standardizes and validates a news event
func NormalizeNewsEvent(rawEvent map[string]interface{}) (*models.NewsEvent, error) {
	// Extract from raw RSS item
	title, _ := rawEvent["title"].(string)
	description, _ := rawEvent["description"].(string)
	link, _ := rawEvent["link"].(string)
	source, _ := rawEvent["source"].(string)
	
	// Clean and normalize text content (remove HTML, extra whitespace)
	title = strings.TrimSpace(regexp.MustCompile(`<[^>]*>`).ReplaceAllString(title, ""))
	description = strings.TrimSpace(regexp.MustCompile(`<[^>]*>`).ReplaceAllString(description, ""))
	
	// Parse and validate published date format
	var publishedAt time.Time
	if published, ok := rawEvent["published"].(*time.Time); ok && published != nil {
		publishedAt = *published
	} else {
		publishedAt = time.Now()
	}
	
	if title == "" {
		return nil, fmt.Errorf("title is required")
	}
	if link == "" {
		return nil, fmt.Errorf("link is required")
	}
	
	return &models.NewsEvent{
		Title:       title,
		Content:     description,
		PublishedAt: publishedAt,
		Source:      source,
		URL:          link,
		Assets:       []assets.AssetMatch{}, // Will be populated by AssetDetector
		Categories:   []string{},
		Sentiment:    0.0,
		Confidence:   0.0,
		NewsType:     "financial", // Default type
		MarketImpact: "low",       // Default impact
	}, nil
}