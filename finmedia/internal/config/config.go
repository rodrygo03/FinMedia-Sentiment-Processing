package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	
	"finmedia/internal/assets"
)

// Server configuration for the ingestion service
type Server struct {
	Port int `json:"port"`
}

// Pipeline configuration for downstream processing
type Pipeline struct {
	BatchSize    int    `json:"batch_size"`
	FlushTimeout string `json:"flush_timeout"`
	OutputQueue  string `json:"output_queue"`
}

// Preprocessing configuration for the Rust gRPC service
type Preprocessing struct {
	Enabled bool   `json:"enabled"`
	Address string `json:"address"`
	Timeout string `json:"timeout"`
}

// represents the application configuration
type Config struct {
	RSSFeeds        []string              `json:"rss_feeds"`
	Server          Server                `json:"server"`
	Pipeline        Pipeline              `json:"pipeline"`
	Preprocessing   Preprocessing         `json:"preprocessing"`
	AssetDetection  assets.AssetDetection `json:"asset_detection"`
}

// loads configuration from JSON/YAML/env
func LoadConfig(configPath string) (*Config, error) {
	config := &Config{}

	if _, err := os.Stat(configPath); err == nil {
		file, err := os.ReadFile(configPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
		
		if err := json.Unmarshal(file, config); err != nil {
			return nil, fmt.Errorf("failed to parse config file: %w", err)
		}
	}
	
	// Override with environment variables if present
	if feeds := os.Getenv("RSS_FEEDS"); feeds != "" {
		var feedList []string
		if err := json.Unmarshal([]byte(feeds), &feedList); err == nil {
			config.RSSFeeds = feedList
		}
	}
	if port := os.Getenv("SERVER_PORT"); port != "" {
		if p, err := strconv.Atoi(port); err == nil {
			config.Server.Port = p
		}
	}
	if batchSize := os.Getenv("PIPELINE_BATCH_SIZE"); batchSize != "" {
		if bs, err := strconv.Atoi(batchSize); err == nil {
			config.Pipeline.BatchSize = bs
		}
	}
	if flushTimeout := os.Getenv("PIPELINE_FLUSH_TIMEOUT"); flushTimeout != "" {
		config.Pipeline.FlushTimeout = flushTimeout
	}
	if outputQueue := os.Getenv("PIPELINE_OUTPUT_QUEUE"); outputQueue != "" {
		config.Pipeline.OutputQueue = outputQueue
	}
	if preprocessingEnabled := os.Getenv("PREPROCESSING_ENABLED"); preprocessingEnabled != "" {
		if enabled, err := strconv.ParseBool(preprocessingEnabled); err == nil {
			config.Preprocessing.Enabled = enabled
		}
	}
	if preprocessingAddress := os.Getenv("PREPROCESSING_ADDRESS"); preprocessingAddress != "" {
		config.Preprocessing.Address = preprocessingAddress
	}
	if preprocessingTimeout := os.Getenv("PREPROCESSING_TIMEOUT"); preprocessingTimeout != "" {
		config.Preprocessing.Timeout = preprocessingTimeout
	}
	
	// Set defaults if not configured
	if config.Server.Port == 0 {
		config.Server.Port = 8080
	}
	if config.Pipeline.BatchSize == 0 {
		config.Pipeline.BatchSize = 10
	}
	if config.Pipeline.FlushTimeout == "" {
		config.Pipeline.FlushTimeout = "30s"
	}
	if config.Pipeline.OutputQueue == "" {
		config.Pipeline.OutputQueue = "news_events"
	}
	if config.Preprocessing.Address == "" {
		config.Preprocessing.Address = "localhost:50051"
	}
	if config.Preprocessing.Timeout == "" {
		config.Preprocessing.Timeout = "5s"
	}
	// Enable preprocessing by default
	if !config.Preprocessing.Enabled {
		config.Preprocessing.Enabled = true
	}
	
	// Load assets from external file if specified
	if config.AssetDetection.AssetsFile != "" {
		if err := loadAssetsFromFile(&config.AssetDetection, configPath); err != nil {
			return nil, fmt.Errorf("failed to load assets from file: %w", err)
		}
	}
	
	if len(config.RSSFeeds) == 0 {
		return nil, fmt.Errorf("no RSS feeds configured")
	}
	
	if len(config.AssetDetection.Assets) == 0 {
		return nil, fmt.Errorf("no assets configured")
	}
	
	return config, nil
}

// returns the list of RSS feeds to monitor
func (c *Config) GetRSSFeeds() []string {
	return c.RSSFeeds
}

// returns the server port
func (c *Config) GetServerPort() int {
	return c.Server.Port
}

// loadAssetsFromFile loads assets from an external JSON file
func loadAssetsFromFile(assetDetection *assets.AssetDetection, configPath string) error {
	// Resolve relative path from config directory
	configDir := filepath.Dir(configPath)
	assetsFilePath := filepath.Join(configDir, assetDetection.AssetsFile)
	
	// Check if file exists
	if _, err := os.Stat(assetsFilePath); os.IsNotExist(err) {
		return fmt.Errorf("assets file not found: %s", assetsFilePath)
	}
	
	// Read the assets file
	assetsData, err := os.ReadFile(assetsFilePath)
	if err != nil {
		return fmt.Errorf("failed to read assets file: %w", err)
	}
	
	// Parse the assets file
	var assetsFile struct {
		Assets []assets.Asset `json:"assets"`
	}
	
	if err := json.Unmarshal(assetsData, &assetsFile); err != nil {
		return fmt.Errorf("failed to parse assets file: %w", err)
	}
	
	// Set the assets in the detection config
	assetDetection.Assets = assetsFile.Assets
	
	return nil
}