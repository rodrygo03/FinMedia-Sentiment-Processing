use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing::info;

mod pipeline;
mod go_service;
mod client;
mod output;
mod interceptor_server;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    once: bool,

    #[arg(long, default_value = "../config.json")]
    config: PathBuf,

    #[arg(long, default_value = "../inference/FinBERT/finbert_config.json")]
    finbert_config: PathBuf,

    #[arg(long, default_value = "http://127.0.0.1:50051")]
    preprocessing_addr: String,

    // pwd: finmedia/
    //      go build -o finmedia ./cmd/ingest/main.go
    #[arg(long, default_value = "../finmedia/bin/finmedia")]
    go_binary: PathBuf,

    /// Specific RSS feed URL to process (optional)
    #[arg(long)]
    feed: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    info!("Starting FinMedia Sentiment Processing Orchestrator");
    info!("Config: {:?}", args.config);
    info!("FinBERT Config: {:?}", args.finbert_config);
    info!("Mode: {}", if args.once { "Single run" } else { "Continuous" });

    let mut orchestrator = pipeline::PipelineOrchestrator::new(
        args.config,
        args.finbert_config,
        args.preprocessing_addr,
        args.go_binary,
    ).await?;
    if args.once {
        info!("Running pipeline once...");
        orchestrator.run_once(args.feed).await?;
    } else {
        info!("Starting continuous pipeline...");
        orchestrator.run_continuous().await?;
    }

    info!("Orchestrator completed successfully");
    Ok(())
}