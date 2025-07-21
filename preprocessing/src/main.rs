use anyhow::Result;
use finmedia_preprocessing::{config::Config, server::PreprocessingServer};
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init(); // formatted subscriber that outputs structured logs to stdout.
    let config = Config::from_env_or_default();

    finmedia_preprocessing::init().await?;    
    info!(
        "Starting finmedia-preprocessing v{} on {}",
        finmedia_preprocessing::VERSION,
        config.server_address()
    );
    
    let server = PreprocessingServer::new(config);
    server.serve().await?;
    
    Ok(())
}