use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableDuration {
    pub secs: u64,
    pub nanos: u32,
}

impl From<Duration> for SerializableDuration {
    fn from(duration: Duration) -> Self {
        Self {
            secs: duration.as_secs(),
            nanos: duration.subsec_nanos(),
        }
    }
}

impl From<SerializableDuration> for Duration {
    fn from(duration: SerializableDuration) -> Self {
        Duration::new(duration.secs, duration.nanos)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    pub event_id: String,
    pub timestamp: DateTime<Utc>,
    pub rss_fetch_time: SerializableDuration,
    pub preprocessing_time: SerializableDuration,
    pub finbert_inference_time: SerializableDuration,
    pub signals_processing_time: SerializableDuration,
    pub vector_storage_time: SerializableDuration,
    pub total_pipeline_time: SerializableDuration,
    pub memory_usage_mb: f64,
}

impl LatencyMetrics {
    pub fn new(event_id: String) -> Self {
        Self {
            event_id,
            timestamp: Utc::now(),
            rss_fetch_time: Duration::from_millis(0).into(),
            preprocessing_time: Duration::from_millis(0).into(),
            finbert_inference_time: Duration::from_millis(0).into(),
            signals_processing_time: Duration::from_millis(0).into(),
            vector_storage_time: Duration::from_millis(0).into(),
            total_pipeline_time: Duration::from_millis(0).into(),
            memory_usage_mb: 0.0,
        }
    }

    pub fn total_time_ms(&self) -> f64 {
        let duration: Duration = self.total_pipeline_time.clone().into();
        duration.as_secs_f64() * 1000.0
    }

    pub fn finbert_time_ms(&self) -> f64 {
        let duration: Duration = self.finbert_inference_time.clone().into();
        duration.as_secs_f64() * 1000.0
    }

    pub fn signals_time_ms(&self) -> f64 {
        let duration: Duration = self.signals_processing_time.clone().into();
        duration.as_secs_f64() * 1000.0
    }

    pub fn storage_time_ms(&self) -> f64 {
        let duration: Duration = self.vector_storage_time.clone().into();
        duration.as_secs_f64() * 1000.0
    }

    pub fn preprocessing_time_ms(&self) -> f64 {
        let duration: Duration = self.preprocessing_time.clone().into();
        duration.as_secs_f64() * 1000.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyReport {
    pub total_events: usize,
    pub test_duration_secs: f64,
    pub throughput_events_per_sec: f64,
    pub memory_peak_mb: f64,
    pub memory_avg_mb: f64,
    
    // Stage-specific metrics
    pub total_pipeline: StageMetrics,
    pub finbert_inference: StageMetrics,
    pub signals_processing: StageMetrics,
    pub vector_storage: StageMetrics,
    pub preprocessing: StageMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageMetrics {
    pub avg_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub std_dev_ms: f64,
}

impl StageMetrics {
    pub fn from_durations(mut times: Vec<f64>) -> Self {
        if times.is_empty() {
            return Self::zero();
        }
        
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = times.len();
        
        let avg = times.iter().sum::<f64>() / len as f64;
        let min = times[0];
        let max = times[len - 1];
        let p50 = times[len / 2];
        let p95 = times[(len as f64 * 0.95) as usize];
        let p99 = times[(len as f64 * 0.99) as usize];
        
        // Calculate standard deviation
        let variance = times.iter()
            .map(|x| (x - avg).powi(2))
            .sum::<f64>() / len as f64;
        let std_dev = variance.sqrt();
        
        Self {
            avg_ms: avg,
            min_ms: min,
            max_ms: max,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
            std_dev_ms: std_dev,
        }
    }
    
    fn zero() -> Self {
        Self {
            avg_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            p50_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            std_dev_ms: 0.0,
        }
    }
}

impl LatencyReport {
    pub fn from_metrics(metrics: Vec<LatencyMetrics>) -> Self {
        if metrics.is_empty() {
            return Self::empty();
        }
        
        let total_events = metrics.len();
        let test_duration = metrics.last().unwrap().timestamp - metrics.first().unwrap().timestamp;
        let test_duration_secs = test_duration.num_seconds() as f64;
        let throughput = total_events as f64 / test_duration_secs;
        
        // Extract memory metrics
        let memory_values: Vec<f64> = metrics.iter().map(|m| m.memory_usage_mb).collect();
        let memory_peak = memory_values.iter().fold(0.0_f64, |acc, &x| acc.max(x));
        let memory_avg = memory_values.iter().sum::<f64>() / memory_values.len() as f64;
        
        // Extract timing data for each stage
        let total_times: Vec<f64> = metrics.iter().map(|m| m.total_time_ms()).collect();
        let finbert_times: Vec<f64> = metrics.iter().map(|m| m.finbert_time_ms()).collect();
        let signals_times: Vec<f64> = metrics.iter().map(|m| m.signals_time_ms()).collect();
        let storage_times: Vec<f64> = metrics.iter().map(|m| m.storage_time_ms()).collect();
        let preprocessing_times: Vec<f64> = metrics.iter().map(|m| m.preprocessing_time_ms()).collect();
        
        Self {
            total_events,
            test_duration_secs,
            throughput_events_per_sec: throughput,
            memory_peak_mb: memory_peak,
            memory_avg_mb: memory_avg,
            total_pipeline: StageMetrics::from_durations(total_times),
            finbert_inference: StageMetrics::from_durations(finbert_times),
            signals_processing: StageMetrics::from_durations(signals_times),
            vector_storage: StageMetrics::from_durations(storage_times),
            preprocessing: StageMetrics::from_durations(preprocessing_times),
        }
    }
    
    fn empty() -> Self {
        Self {
            total_events: 0,
            test_duration_secs: 0.0,
            throughput_events_per_sec: 0.0,
            memory_peak_mb: 0.0,
            memory_avg_mb: 0.0,
            total_pipeline: StageMetrics::zero(),
            finbert_inference: StageMetrics::zero(),
            signals_processing: StageMetrics::zero(),
            vector_storage: StageMetrics::zero(),
            preprocessing: StageMetrics::zero(),
        }
    }
    
    pub fn display(&self) {
        println!("\n=== LATENCY BENCHMARK RESULTS ===");
        println!("Events processed: {}", self.total_events);
        println!("Duration: {:.1} seconds", self.test_duration_secs);
        println!("Throughput: {:.2} events/second", self.throughput_events_per_sec);
        println!("Memory peak: {:.1}MB, avg: {:.1}MB", self.memory_peak_mb, self.memory_avg_mb);
        
        println!("\nStage Breakdown (ms):");
        println!("├─ Total Pipeline:   {}", self.format_stage_metrics(&self.total_pipeline));
        println!("├─ Preprocessing:    {}", self.format_stage_metrics(&self.preprocessing));
        println!("├─ FinBERT:          {}", self.format_stage_metrics(&self.finbert_inference));
        println!("├─ Signals:          {}", self.format_stage_metrics(&self.signals_processing));
        println!("└─ Vector Storage:   {}", self.format_stage_metrics(&self.vector_storage));
        
        println!("\nPerformance Summary:");
        println!("• Average total latency: {:.1}ms", self.total_pipeline.avg_ms);
        println!("• 95th percentile: {:.1}ms", self.total_pipeline.p95_ms);
        println!("• 99th percentile: {:.1}ms", self.total_pipeline.p99_ms);
        println!("• Processing efficiency: {:.1}%", self.calculate_efficiency());
    }
    
    fn format_stage_metrics(&self, metrics: &StageMetrics) -> String {
        format!("avg: {:.1}  p95: {:.1}  p99: {:.1}  max: {:.1}", 
            metrics.avg_ms, metrics.p95_ms, metrics.p99_ms, metrics.max_ms)
    }
    
    fn calculate_efficiency(&self) -> f64 {
        // Simple efficiency metric: how close we are to theoretical max throughput
        let ideal_latency = 50.0; // Assume 50ms ideal per event
        ((ideal_latency / self.total_pipeline.avg_ms) * 100.0).min(100.0)
    }
}

#[derive(Debug)]
pub struct PipelineTimer {
    start_time: Instant,
    stage_times: Vec<(String, Instant)>,
}

impl PipelineTimer {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            stage_times: Vec::new(),
        }
    }
    
    pub fn mark_stage(&mut self, stage_name: &str) {
        self.stage_times.push((stage_name.to_string(), Instant::now()));
    }
    
    pub fn get_stage_duration(&self, stage_name: &str) -> Option<Duration> {
        let mut prev_time = self.start_time;
        
        for (name, time) in &self.stage_times {
            if name == stage_name {
                return Some(time.duration_since(prev_time));
            }
            prev_time = *time;
        }
        None
    }
    
    pub fn total_duration(&self) -> Duration {
        self.start_time.elapsed()
    }
}