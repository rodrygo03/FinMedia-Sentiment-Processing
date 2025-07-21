// Financial Media Preprocessing Library

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod models;
pub mod config;
pub mod error;
pub mod server;
pub mod processor;

pub use models::{NewsEvent, ProcessedEvent};
pub use config::Config;
pub use error::{PreprocessingError, Result};
pub use server::PreprocessingServer;

pub async fn init() -> Result<()> {
    tracing::info!("Finmedia preprocessing library initialized (version: {})", VERSION);
    Ok(())
}

pub fn health_check() -> Result<()> {
    tracing::info!("Performing health check");
    Ok(())
}