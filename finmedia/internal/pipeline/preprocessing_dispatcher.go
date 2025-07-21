package pipeline

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"finmedia/internal/config"
	"finmedia/internal/models"
	"finmedia/internal/preprocessing"
)

//  handles batching and preprocessing of news events using the Rust gRPC service
type PreprocessingDispatcher struct {
	batchSize          int
	flushTimeout       time.Duration
	outputQueue        string
	buffer             []models.NewsEvent
	mu                 sync.Mutex
	preprocessingClient *preprocessing.Client
	config             *config.Config
}

func NewPreprocessingDispatcher(cfg *config.Config) (*PreprocessingDispatcher, error) {
	// Validate batch size and flush timeout
	batchSize := cfg.Pipeline.BatchSize
	if batchSize <= 0 {
		batchSize = 10
	}
	
	flushTimeout, err := time.ParseDuration(cfg.Pipeline.FlushTimeout)
	if err != nil || flushTimeout <= 0 {
		flushTimeout = 60 * time.Second
	}
	
	outputQueue := cfg.Pipeline.OutputQueue
	if outputQueue == "" {
		outputQueue = "processed_news_events"
	}
	
	var preprocessingClient *preprocessing.Client
	if cfg.Preprocessing.Enabled {
		client, err := preprocessing.NewClient(cfg.Preprocessing.Address)
		if err != nil {
			return nil, fmt.Errorf("failed to create preprocessing client: %w", err)
		}
		preprocessingClient = client
		
		// Test connection
		if err := client.Health(); err != nil {
			log.Printf("Warning: Preprocessing service health check failed: %v", err)
		} else {
			log.Printf("Connected to preprocessing service at %s", cfg.Preprocessing.Address)
		}
	}
	
	return &PreprocessingDispatcher{
		batchSize:           batchSize,
		flushTimeout:        flushTimeout,
		outputQueue:         outputQueue,
		buffer:              make([]models.NewsEvent, 0, batchSize),
		preprocessingClient: preprocessingClient,
		config:              cfg,
	}, nil
}

func (d *PreprocessingDispatcher) Start(ctx context.Context, input <-chan models.NewsEvent) error {
	ticker := time.NewTicker(d.flushTimeout)
	defer ticker.Stop()
	
	// Ensure cleanup on shutdown
	defer func() {
		if d.preprocessingClient != nil {
			d.preprocessingClient.Close()
		}
	}()
	
	for {
		select {
		case <-ctx.Done():
			if len(d.buffer) > 0 {
				if err := d.FlushBatch(); err != nil {
					log.Printf("Failed to flush final batch: %v", err)
				}
			}
			return nil
			
		case event, ok := <-input:
			if !ok {
				// Channel closed, flush remaining events
				if len(d.buffer) > 0 {
					if err := d.FlushBatch(); err != nil {
						log.Printf("Failed to flush final batch: %v", err)
					}
				}
				return nil
			}
			
			// Process incoming event
			if err := d.Dispatch(event); err != nil {
				log.Printf("Failed to dispatch event: %v", err)
			}
			
		case <-ticker.C:
			// Periodic flush based on timeout
			if len(d.buffer) > 0 {
				if err := d.FlushBatch(); err != nil {
					log.Printf("Failed to flush batch: %v", err)
				}
			}
		}
	}
}

func (d *PreprocessingDispatcher) Dispatch(event models.NewsEvent) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	
	d.buffer = append(d.buffer, event)
	
	if len(d.buffer) >= d.batchSize {
		// If batch is full, trigger immediate flush
		return d.flushBatchUnsafe()
	}
	
	return nil
}

func (d *PreprocessingDispatcher) FlushBatch() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.flushBatchUnsafe()
}

// performs flush without locking (assumes caller has lock)
func (d *PreprocessingDispatcher) flushBatchUnsafe() error {
	if len(d.buffer) == 0 {
		return nil
	}
	
	start := time.Now()
	eventCount := len(d.buffer)
	
	// Copy buffer to avoid holding lock during processing
	events := make([]models.NewsEvent, len(d.buffer))
	copy(events, d.buffer)
	d.buffer = d.buffer[:0] // Clear buffer
	
	// Process events
	if err := d.processEvents(events); err != nil {
		log.Printf("Failed to process events: %v", err)
		// TODO: retry or send to dead queue
	}
	
	d.LogMetrics(eventCount, time.Since(start))
	return nil
}

func (d *PreprocessingDispatcher) processEvents(events []models.NewsEvent) error {
	if d.preprocessingClient == nil || !d.config.Preprocessing.Enabled {
		log.Printf("Preprocessing disabled, processing %d events without preprocessing", len(events))
		for _, event := range events {
			d.logOriginalEvent(&event)
		}
		return nil
	}
	
	if len(events) == 1 {
		// Single event processing
		processedEvent, err := d.preprocessingClient.ProcessNewsEvent(&events[0])
		if err != nil {
			return fmt.Errorf("failed to preprocess single event: %w", err)
		}
		d.logProcessedEvent(processedEvent)
	} else {
		// Batch processing - convert to pointer slice
		eventPointers := make([]*models.NewsEvent, len(events))
		for i := range events {
			eventPointers[i] = &events[i]
		}
		
		processedEvents, err := d.preprocessingClient.ProcessBatch(eventPointers)
		if err != nil {
			return fmt.Errorf("failed to preprocess batch: %w", err)
		}
		
		for _, processedEvent := range processedEvents {
			d.logProcessedEvent(processedEvent)
		}
	}
	
	return nil
}

func (d *PreprocessingDispatcher) logOriginalEvent(event *models.NewsEvent) {
	log.Printf("ORIGINAL EVENT:")
	log.Printf("  Title: %s", event.Title)
	log.Printf("  Content: %s", truncateString(event.Content, 100))
	log.Printf("  Source: %s", event.Source)
	log.Printf("  Published: %s", event.PublishedAt.Format(time.RFC3339))
	log.Printf("---")
}

func (d *PreprocessingDispatcher) logProcessedEvent(processedEvent *models.ProcessedEvent) {
	log.Printf("PREPROCESSED EVENT:")
	log.Printf("  ID: %s", processedEvent.ID)
	log.Printf("  ORIGINAL: %s", truncateString(processedEvent.OriginalEvent.Title+" "+processedEvent.OriginalEvent.Content, 150))
	log.Printf("  PROCESSED: %s", truncateString(processedEvent.ProcessedText, 150))
	log.Printf("  Source: %s", processedEvent.OriginalEvent.Source)
	log.Printf("  Processed At: %s", processedEvent.ProcessedAt.Format(time.RFC3339))
	log.Printf("---")
}

func (d *PreprocessingDispatcher) LogMetrics(eventCount int, processingTime time.Duration) {
	// Log throughput metrics (events/second)
	throughput := float64(eventCount) / processingTime.Seconds()
	
	log.Printf("Processed batch: %d events in %v (%.2f events/sec)", 
		eventCount, processingTime, throughput)
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}