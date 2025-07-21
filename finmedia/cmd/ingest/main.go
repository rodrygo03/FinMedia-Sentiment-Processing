package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"finmedia/internal/assets"
	"finmedia/internal/config"
	"finmedia/internal/fetcher"
	"finmedia/internal/models"
	"finmedia/internal/pipeline"
)

func main() {
	fmt.Println("Ingesting...")

	cfg, err := config.LoadConfig("config.json")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	assetDetector := assets.NewAssetDetector(&cfg.AssetDetection)
	if assetDetector == nil {
		log.Fatalf("Failed to load asset config: %v", err)
	}
	
	// Producer-Consumer, shared buff
	rssFetcher := fetcher.NewRSSFetcher(
		cfg.GetRSSFeeds(),
		5,                    // workers
		5*time.Second,        // fetch interval
		assetDetector,        // asset detector
	)
	// Use the new preprocessing dispatcher instead of the old one
	dispatcher, err := pipeline.NewPreprocessingDispatcher(cfg)
	if err != nil {
		log.Fatalf("Failed to create preprocessing dispatcher: %v", err)
	}
	newsEventChan := make(chan models.NewsEvent, 100)

	// App's "top level" context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	var wg sync.WaitGroup

	// start RSS fetcher in background goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := rssFetcher.Start(ctx, newsEventChan); err != nil {
			log.Printf("RSS fetcher error: %v", err)
		}
	}()
	// start pipeline dispatcher in background goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := dispatcher.Start(ctx, newsEventChan); err != nil {
			log.Printf("Pipeline dispatcher error: %v", err)
		}
	}()

	<-sigChan
	fmt.Println("Shutdown signal received, stopping...")
	cancel()
	wg.Wait()
	
	fmt.Println("News Ingestion Service stopped")
}
