package pipeline

import (
	"bytes"
	"context"
	"encoding/json"
	"finmedia/internal/models"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"
)

// handles batching and forwarding of news events to downstream systems
type Dispatcher struct {
	batchSize    int
	flushTimeout time.Duration
	outputQueue  string
	buffer       []models.NewsEvent
	mu           sync.Mutex
}

//  creates a new pipeline dispatcher
func NewDispatcher(batchSize int, flushTimeoutStr string, outputQueue string) *Dispatcher {
	// Validate batchSize and flushTimeout are reasonable
	if batchSize <= 0 {
		batchSize = 10
	}
	
	// Parse flush timeout string
	flushTimeout, err := time.ParseDuration(flushTimeoutStr)
	if err != nil || flushTimeout <= 0 {
		flushTimeout = 60 * time.Second
	}
	
	if outputQueue == "" {
		outputQueue = "news_events"
	}
	
	return &Dispatcher{
		batchSize:    batchSize,
		flushTimeout: flushTimeout,
		outputQueue:  outputQueue,
		buffer:       make([]models.NewsEvent, 0, batchSize),
	}
}

//  begins the dispatcher processing loop
func (d *Dispatcher) Start(ctx context.Context, input <-chan models.NewsEvent) error {
	ticker := time.NewTicker(d.flushTimeout)
	defer ticker.Stop()
	
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

// sends a news event to the pipeline
func (d *Dispatcher) Dispatch(event models.NewsEvent) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	
	d.buffer = append(d.buffer, event)
	
	if len(d.buffer) >= d.batchSize {
		// If batch is full, trigger immediate flush
		return d.flushBatchUnsafe()
	}
	
	return nil
}

// sends accumulated events to downstream processing
func (d *Dispatcher) FlushBatch() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.flushBatchUnsafe()
}

// performs flush without locking (assumes caller has lock)
func (d *Dispatcher) flushBatchUnsafe() error {
	if len(d.buffer) == 0 {
		return nil
	}
	
	start := time.Now()
	eventCount := len(d.buffer)
	
	// Send batch to message queue or HTTP endpoint
	if err := d.SendToQueue(d.outputQueue, d.buffer); err != nil {
		// Fallback to Rust service if queue fails
		if err := d.SendToRust(d.buffer); err != nil {
			return fmt.Errorf("failed to send batch: %w", err)
		}
	}
	
	d.buffer = d.buffer[:0]
	d.LogMetrics(eventCount, time.Since(start))
	
	return nil
}

// sends processed news events to Rust preprocessing layer
func (d *Dispatcher) SendToRust(events []models.NewsEvent) error {
	// Format events as JSON payload
	payload := map[string]interface{}{
		"events":    events,
		"timestamp": time.Now(),
		"count":     len(events),
		"source":    "finmedia-ingestion",
	}
	
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal events: %w", err)
	}
	
	// Make HTTP POST request to Rust service endpoint
	resp, err := http.Post("http://localhost:8081/events", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to send to Rust service: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("rust service returned status: %d", resp.StatusCode)
	}
	
	return nil
}

// sends events to a message queue for async processing
func (d *Dispatcher) SendToQueue(queueName string, events []models.NewsEvent) error {
	// For now, implement as simple JSON file output
	// In production, this would connect to Redis, RabbitMQ, etc.
	
	payload := map[string]interface{}{
		"queue":     queueName,
		"events":    events,
		"timestamp": time.Now(),
		"count":     len(events),
	}
	
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal queue payload: %w", err)
	}
	
	// Log successful deliveries for monitoring
	log.Printf("Queued %d events to %s", len(events), queueName)
	// For demonstration, just log the payload
	log.Printf("Queue payload: %s", string(jsonData))
	
	return nil
}

// records processing metrics for monitoring
func (d *Dispatcher) LogMetrics(eventCount int, processingTime time.Duration) {
	// Log throughput metrics (events/second)
	throughput := float64(eventCount) / processingTime.Seconds()
	
	// Track batch processing latency
	log.Printf("Processed batch: %d events in %v (%.2f events/sec)", 
		eventCount, processingTime, throughput)
	
	// In production, this would export metrics to Prometheus, etc.
	// For now, just log the metrics
	log.Printf("Metrics - Count: %d, Latency: %v, Throughput: %.2f/sec", 
		eventCount, processingTime, throughput)
}