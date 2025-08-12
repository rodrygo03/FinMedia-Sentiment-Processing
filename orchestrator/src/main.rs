use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::time::Duration;
use tracing::info;

use benchmarks::{LatencyBenchmark, LatencyTestType};

mod pipeline;
mod go_service;
mod client;
mod output;
mod interceptor_server;
mod unified_event;
mod signals_processing;
mod benchmarks;

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

    /// Run latency benchmark
    #[arg(long)]
    benchmark: bool,

    /// Benchmark type (cold-start, warm, burst, long-running, stress)
    #[arg(long, default_value = "warm")]
    benchmark_type: String,

    /// Number of events for benchmark
    #[arg(long, default_value = "100")]
    benchmark_events: usize,

    /// Duration for long-running benchmark (seconds)
    #[arg(long, default_value = "300")]
    benchmark_duration: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::from_filename("../.env").ok();
    
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    info!("Starting FinMedia Sentiment Processing Orchestrator");
    info!("Config: {:?}", args.config);
    info!("FinBERT Config: {:?}", args.finbert_config);
    info!("Mode: {}", if args.once { "Single run" } else { "Continuous" });

    let mut orchestrator = pipeline::PipelineOrchestrator::new(
        args.config.clone(),
        args.finbert_config.clone(),
        args.preprocessing_addr.clone(),
        args.go_binary.clone(),
    ).await?;

    if args.benchmark {
        info!("Running latency benchmark...");
        run_benchmark(&mut orchestrator, &args).await?;
    } else if args.once {
        info!("Running pipeline once...");
        orchestrator.run_once(args.feed).await?;
    } else {
        info!("Starting continuous pipeline...");
        orchestrator.run_continuous().await?;
    }

    info!("Orchestrator completed successfully");
    Ok(())
}

async fn run_benchmark(orchestrator: &mut pipeline::PipelineOrchestrator, args: &Args) -> Result<()> {
    let test_type = match args.benchmark_type.as_str() {
        "cold-start" => LatencyTestType::ColdStart,
        "warm" => LatencyTestType::WarmPipeline,
        "burst" => LatencyTestType::BurstLoad,
        "long-running" => LatencyTestType::LongRunning,
        "stress" => LatencyTestType::StressTest,
        _ => {
            eprintln!("Invalid benchmark type: {}. Valid options: cold-start, warm, burst, long-running, stress", args.benchmark_type);
            return Ok(());
        }
    };

    info!("Initializing services for benchmark...");
    orchestrator.initialize_for_benchmark().await?;

    let mut benchmark = LatencyBenchmark::new(test_type, args.benchmark_events);    
    if args.benchmark_type == "long-running" {
        benchmark = benchmark.with_duration(Duration::from_secs(args.benchmark_duration));
    }

    let report = benchmark.run_benchmark(orchestrator).await?;
    let report_filename = format!("benchmark_report_{}.json", chrono::Utc::now().format("%Y%m%d_%H%M%S"));
    if let Ok(report_json) = serde_json::to_string_pretty(&report) {
        std::fs::write(&report_filename, report_json)?;
        info!("Benchmark report saved to: {}", report_filename);
    }

    Ok(())
}