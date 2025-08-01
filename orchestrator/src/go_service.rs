use anyhow::Result;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::{Child, Command};
use tokio::time::{timeout, Duration};
use tracing::{info, error, warn};

pub struct GoServiceManager {
    binary_path: PathBuf,
    process: Option<Child>,
}

impl GoServiceManager {
    pub fn new(binary_path: PathBuf) -> Self {
        Self {
            binary_path,
            process: None,
        }
    }

    pub async fn ensure_binary_exists(&self) -> Result<()> {
        if !self.binary_path.exists() {
            info!("Go binary not found, building...");
            self.build_go_service().await?;
        }
        Ok(())
    }

    async fn build_go_service(&self) -> Result<()> {
        let finmedia_dir = self.binary_path
            .parent()
            .and_then(|p| p.parent())
            .ok_or_else(|| anyhow::anyhow!("Invalid binary path"))?;

        info!("Building Go service in: {:?}", finmedia_dir);

        let output = Command::new("go")
            .args(&["build", "-o", "bin/finmedia", "cmd/ingest/main.go"])
            .current_dir(finmedia_dir)
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Failed to build Go service: {}", stderr);
        }

        info!("✅ Go service built successfully");
        Ok(())
    }

    pub async fn start(&mut self, _config_path: &PathBuf) -> Result<()> {
        self.start_with_preprocessing_addr(_config_path, "localhost:50051").await
    }

    pub async fn start_with_preprocessing_addr(&mut self, _config_path: &PathBuf, preprocessing_addr: &str) -> Result<()> {
        self.ensure_binary_exists().await?;

        info!("Starting Go service: {:?}", self.binary_path);
        info!("Go service will connect to preprocessing at: {}", preprocessing_addr);

        let mut cmd = Command::new(&self.binary_path);
        if let Some(finmedia_dir) = self.binary_path.parent().and_then(|p| p.parent()) {
            cmd.current_dir(finmedia_dir);
        }

        // Set environment variable for preprocessing service address
        cmd.env("PREPROCESSING_ADDRESS", preprocessing_addr);
        // Redirect stdout and stderr for logging
        cmd.stdout(Stdio::piped())
           .stderr(Stdio::piped());

        let child = cmd.spawn()?;
        self.process = Some(child);

        info!("Go service started");

        // Give it a moment to initialize
        tokio::time::sleep(Duration::from_secs(2)).await;

        Ok(())
    }

    pub async fn start_with_timeout(&mut self, config_path: &PathBuf, run_duration: Duration) -> Result<()> {
        self.start_with_timeout_and_addr(config_path, run_duration, "localhost:50051").await
    }

    pub async fn start_with_timeout_and_addr(&mut self, config_path: &PathBuf, run_duration: Duration, preprocessing_addr: &str) -> Result<()> {
        self.start_with_preprocessing_addr(config_path, preprocessing_addr).await?;

        info!("Running Go service for {:?}...", run_duration);
        tokio::time::sleep(run_duration).await;

        self.stop().await?;
        Ok(())
    }

    pub async fn stop(&mut self) -> Result<()> {
        if let Some(mut process) = self.process.take() {
            info!("Stopping Go service...");

            // Try graceful shutdown first
            #[cfg(unix)]
            {
                if let Some(id) = process.id() {
                    let _ = signal::kill(
                        Pid::from_raw(id as i32),
                        signal::Signal::SIGTERM,
                    );
                }
            }

            match timeout(Duration::from_secs(5), process.wait()).await {
                Ok(Ok(status)) => {
                    info!("Go service stopped gracefully with status: {}", status);
                }
                Ok(Err(e)) => {
                    error!("Error waiting for Go service: {}", e);
                }
                Err(_) => {
                    warn!("Go service didn't stop gracefully, killing...");
                    let _ = process.kill().await;
                }
            }
        }
        Ok(())
    }

    pub fn is_running(&mut self) -> bool {
        if let Some(ref mut process) = self.process {
            match process.try_wait() {
                Ok(None) => true,  // Still running
                Ok(Some(_)) => {
                    self.process = None;
                    false  // Process has exited
                }
                Err(_) => {
                    self.process = None;
                    false  // Error checking status
                }
            }
        } else {
            false
        }
    }

    pub async fn run_once(&mut self, config_path: &PathBuf, duration: Duration) -> Result<()> {
        info!("Running Go service once for {:?}...", duration);
        
        self.start_with_timeout(config_path, duration).await?;
        
        // Additional wait to ensure all events are processed
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        Ok(())
    }
}

impl Drop for GoServiceManager {
    fn drop(&mut self) {
        if let Some(mut process) = self.process.take() {
            let _ = process.start_kill();
        }
    }
}

#[cfg(unix)]
use nix::{sys::signal, unistd::Pid};