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

func (c *Client) ProcessNewsEvent(event *models.NewsEvent) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Convert Go AssetMatch to protobuf AssetMatch
	protoAssets := make([]*pb.AssetMatch, len(event.Assets))
	for i, asset := range event.Assets {
		protoAssets[i] = &pb.AssetMatch{
			Symbol:     asset.Symbol,
			Name:       asset.Name,
			Type:       asset.Type,
			Confidence: asset.Confidence,
			Contexts:   asset.Contexts,
		}
	}

	// Convert Go NewsEvent to protobuf NewsEvent with complete data
	protoEvent := &pb.NewsEvent{
		Id:           generateEventID(event), // Generate ID since Go model doesn't have one
		Title:        event.Title,
		Content:      event.Content,
		PublishedAt:  event.PublishedAt.Format(time.RFC3339),
		Source:       event.Source,
		Url:          event.URL,
		Assets:       protoAssets,
		Categories:   event.Categories,
		Sentiment:    event.Sentiment,
		Confidence:   event.Confidence,
		NewsType:     event.NewsType,
		MarketImpact: event.MarketImpact,
	}

	req := &pb.NewsEventRequest{
		Event: protoEvent,
	}

	// Fire-and-forget: Send to Rust preprocessing service
	resp, err := c.client.ProcessNewsEvent(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to send news event to preprocessing: %w", err)
	}

	if !resp.Success {
		return fmt.Errorf("preprocessing failed: %s", resp.ErrorMessage)
	}

	return nil
}

func (c *Client) ProcessBatch(events []*models.NewsEvent) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	protoEvents := make([]*pb.NewsEvent, len(events))
	for i, event := range events {
		// Convert Go AssetMatch to protobuf AssetMatch
		protoAssets := make([]*pb.AssetMatch, len(event.Assets))
		for j, asset := range event.Assets {
			protoAssets[j] = &pb.AssetMatch{
				Symbol:     asset.Symbol,
				Name:       asset.Name,
				Type:       asset.Type,
				Confidence: asset.Confidence,
				Contexts:   asset.Contexts,
			}
		}

		protoEvents[i] = &pb.NewsEvent{
			Id:           generateEventID(event),
			Title:        event.Title,
			Content:      event.Content,
			PublishedAt:  event.PublishedAt.Format(time.RFC3339),
			Source:       event.Source,
			Url:          event.URL,
			Assets:       protoAssets,
			Categories:   event.Categories,
			Sentiment:    event.Sentiment,
			Confidence:   event.Confidence,
			NewsType:     event.NewsType,
			MarketImpact: event.MarketImpact,
		}
	}

	req := &pb.BatchRequest{
		Events: protoEvents,
	}

	// Fire-and-forget: Send batch to Rust preprocessing service
	resp, err := c.client.ProcessBatch(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to send batch to preprocessing: %w", err)
	}

	if resp.TotalFailed > 0 {
		return fmt.Errorf("batch processing failed for %d events: %v", resp.TotalFailed, resp.ErrorMessages)
	}

	return nil
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