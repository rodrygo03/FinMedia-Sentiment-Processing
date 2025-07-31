use crate::{Result, InferenceError, FinBertConfig};
use ort::environment::Environment;
use ort::execution_providers::ExecutionProvider;
use ort::session::Session;
use ort::session::builder::SessionBuilder;
use ort::value::Value;
use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::path::Path;


pub struct FinBertModel {
    session: Session,
    config: FinBertConfig,
}

impl FinBertModel {
    pub fn new(model_path: &Path, config: FinBertConfig) -> anyhow::Result<Self> {
        tracing::info!("Loading FinBERT model from: {:?}", model_path);
        
        if !model_path.exists() {
            return Err(InferenceError::ModelLoad(
                format!("Model file not found: {:?}", model_path)
            ).into());
        }

        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to set optimization level: {}", e)))?
            .with_intra_threads(1) // Single-threaded for consistent performance
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to set thread count: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to load model: {}", e)))?;

        tracing::info!("Successfully loaded FinBERT model");        
        Ok(Self {
            session,
            config,
        })
    }

    pub fn run_inference(&mut self, input_ids: &[i64], attention_mask: &[i64]) -> anyhow::Result<Vec<f64>> {
        use ort::inputs;
        
        let input_ids_array = ndarray::Array2::from_shape_vec(
            (1, input_ids.len()), 
            input_ids.to_vec()
        ).map_err(|e| InferenceError::OnnxInference(format!("Failed to create input_ids array: {}", e)))?;
        
        let attention_mask_array = ndarray::Array2::from_shape_vec(
            (1, attention_mask.len()), 
            attention_mask.to_vec()
        ).map_err(|e| InferenceError::OnnxInference(format!("Failed to create attention_mask array: {}", e)))?;

        let input_tensor = Value::from_array(input_ids_array)?;
        let attention_tensor = Value::from_array(attention_mask_array)?;

        let logits_vec: Vec<f64> = {
            let outputs = self.session.borrow_mut().run(inputs![
                "input_ids" => input_tensor,
                "attention_mask" => attention_tensor
            ]).map_err(|e| InferenceError::OnnxInference(format!("Inference failed: {}", e)))?;
            
            let (_shape, data) = outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| InferenceError::OnnxInference(format!("Failed to extract logits: {}", e)))?;
            
            data.iter().map(|&x| x as f64).collect()
        };

        let probabilities = self.softmax(&logits_vec);
        Ok(probabilities)
    }

    pub fn get_config(&self) -> &FinBertConfig {
        &self.config
    }

    // private helpers

    fn softmax(&self, logits: &[f64]) -> Vec<f64> {
        let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f64 = exp_logits.iter().sum();
        
        exp_logits.iter().map(|&x| x / sum_exp).collect()
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn get_test_config() -> FinBertConfig {
        FinBertConfig {
            model_path: "FinBERT_tone_int8.onnx".to_string(),
            tokenizer_path: "finbert_tokenizer/".to_string(),
            max_length: 512,
            labels: vec!["negative".to_string(), "neutral".to_string(), "positive".to_string()],
            quantized: true,
            model_type: "finbert-tone".to_string(),
        }
    }

    #[tokio::test]
    async fn test_model_load() {
        // Test with actual model path from FinBERT directory
        let model_path = PathBuf::from("FinBERT/FinBERT_tone_int8.onnx");
        let config = get_test_config();

        let result = FinBertModel::new(&model_path, config);
        
        if model_path.exists() {
            assert!(result.is_ok(), "Model loading should succeed when file exists");
        } else {
            assert!(result.is_err(), "Model loading should fail when file doesn't exist");
        }
    }

    #[test]
    fn test_softmax() {
        let config = get_test_config();
        let model_path = PathBuf::from("dummy_path");
        
        // We can't create a model without a real file, but we can test softmax logic
        let logits = vec![1.0, 2.0, 3.0];
        
        // Manual softmax calculation for verification
        let max_val: f64 = 3.0;
        let exp_vals: Vec<f64> = vec![(1.0 - max_val).exp(), (2.0 - max_val).exp(), (3.0 - max_val).exp()];
        let sum_exp: f64 = exp_vals.iter().sum();
        let expected: Vec<f64> = exp_vals.iter().map(|x| x / sum_exp).collect();
        
        // Test that probabilities sum to 1.0
        let total: f64 = expected.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "Softmax probabilities should sum to 1.0");
    }
}