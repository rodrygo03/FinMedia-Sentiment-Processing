package preprocessing

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"finmedia/internal/models"
	pb "finmedia/proto"
)

type Client struct {
	conn   *grpc.ClientConn
	client pb.PreprocessingServiceClient
}

func NewClient(address string) (*Client, error) {
	// Create connection with insecure credentials for development
	conn, err := grpc.Dial(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to preprocessing service at %s: %w", address, err)
	}

	return &Client{
		conn:   conn,
		client: pb.NewPreprocessingServiceClient(conn),
	}, nil
}

func (c *Client) Close() error {
	return c.conn.Close()
}

func (c *Client) ProcessNewsEvent(event *models.NewsEvent) (*models.ProcessedEvent, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Convert Go NewsEvent to protobuf NewsEvent
	protoEvent := &pb.NewsEvent{
		Id:          generateEventID(event), // Generate ID since Go model doesn't have one
		Title:       event.Title,
		Content:     event.Content,
		PublishedAt: event.PublishedAt.Format(time.RFC3339),
		Source:      event.Source,
		Url:         event.URL,
	}

	req := &pb.NewsEventRequest{
		Event: protoEvent,
	}

	// Call the Rust preprocessing service
	resp, err := c.client.ProcessNewsEvent(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to process news event: %w", err)
	}

	if !resp.Success {
		return nil, fmt.Errorf("preprocessing failed: %s", resp.ErrorMessage)
	}

	// Convert protobuf ProcessedEvent back to Go ProcessedEvent
	processedAt, err := time.Parse(time.RFC3339, resp.ProcessedEvent.ProcessedAt)
	if err != nil {
		processedAt = time.Now() // fallback to current time if parsing fails
	}

	processedEvent := &models.ProcessedEvent{
		ID:               resp.ProcessedEvent.Id,
		OriginalEvent:    *event,
		ProcessedText:    resp.ProcessedEvent.ProcessedText,
		Tokens:           resp.ProcessedEvent.Tokens,
		AssetMentions:    resp.ProcessedEvent.AssetMentions,
		SentimentScore:   resp.ProcessedEvent.SentimentScore,
		Confidence:       resp.ProcessedEvent.Confidence,
		ProcessedAt:      processedAt,
	}

	return processedEvent, nil
}

func (c *Client) ProcessBatch(events []*models.NewsEvent) ([]*models.ProcessedEvent, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	protoEvents := make([]*pb.NewsEvent, len(events))
	for i, event := range events {
		protoEvents[i] = &pb.NewsEvent{
			Id:          generateEventID(event),
			Title:       event.Title,
			Content:     event.Content,
			PublishedAt: event.PublishedAt.Format(time.RFC3339),
			Source:      event.Source,
			Url:         event.URL,
		}
	}

	req := &pb.BatchRequest{
		Events: protoEvents,
	}

	resp, err := c.client.ProcessBatch(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to process batch: %w", err)
	}

	processedEvents := make([]*models.ProcessedEvent, len(resp.ProcessedEvents))
	for i, protoProcessed := range resp.ProcessedEvents {
		processedAt, err := time.Parse(time.RFC3339, protoProcessed.ProcessedAt)
		if err != nil {
			processedAt = time.Now()
		}

		processedEvents[i] = &models.ProcessedEvent{
			ID:               protoProcessed.Id,
			OriginalEvent:    *events[i], // Use original event from input
			ProcessedText:    protoProcessed.ProcessedText,
			Tokens:           protoProcessed.Tokens,
			AssetMentions:    protoProcessed.AssetMentions,
			SentimentScore:   protoProcessed.SentimentScore,
			Confidence:       protoProcessed.Confidence,
			ProcessedAt:      processedAt,
		}
	}

	return processedEvents, nil
}

func (c *Client) Health() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &pb.HealthRequest{}
	resp, err := c.client.Health(ctx, req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}

	if resp.Status != "healthy" {
		return fmt.Errorf("service is unhealthy: %s", resp.Status)
	}

	return nil
}

func generateEventID(event *models.NewsEvent) string {
	return fmt.Sprintf("%s-%d", event.Source, event.PublishedAt.Unix())
}