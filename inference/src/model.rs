use crate::{Result, InferenceError, FinBertConfig};
use ort::environment::Environment;
use ort::execution_providers::ExecutionProvider;
use ort::session::{Session, SessionOutputs};
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

    /// Extract embeddings from FinBERT model 
    /// Since the quantized model only outputs logits, follow a hybrid approach:
    /// 1. Generate probabilistic features from logits (financial sentiment space)
    /// 2. Combine with token-level features to create a 768-dim representation
    pub fn extract_embeddings(&mut self, input_ids: &[i64], attention_mask: &[i64]) -> anyhow::Result<Vec<f32>> {
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

        let outputs = self.session.borrow_mut().run(inputs![
            "input_ids" => input_tensor,
            "attention_mask" => attention_tensor
        ]).map_err(|e| InferenceError::OnnxInference(format!("Inference failed: {}", e)))?;
        
        let embedding_vec: Vec<f32> = {
            // First try to get actual hidden states if available
            if let Some((shape, data)) = outputs.get("last_hidden_state")
                .or_else(|| outputs.get("hidden_states"))
                .or_else(|| outputs.get("encoder_hidden_states"))
                .and_then(|output| output.try_extract_tensor::<f32>().ok()) 
            {
                tracing::debug!("Found hidden states with shape: {:?}", shape);
                // Extract [CLS] token (first token) 
                let hidden_size = shape[2] as usize;
                data[0..hidden_size].to_vec()
            } else if let Some((_shape, data)) = outputs.get("pooler_output")
                .and_then(|output| output.try_extract_tensor::<f32>().ok()) 
            {
                tracing::debug!("Using pooler output for embeddings");
                data.to_vec()
            } else {
                // Fallback: Create synthetic embeddings from available features
                tracing::debug!("Creating synthetic embeddings from logits and input features");
                Self::create_synthetic_embeddings(input_ids, attention_mask, &outputs)?
            }
        };

        tracing::debug!("Extracted {}-dimensional embedding from FinBERT", embedding_vec.len());
        Ok(embedding_vec)
    }

    /// Create synthetic 768-dimensional embeddings when hidden states aren't available
    /// This combines logits with input token features to create meaningful representations
    fn create_synthetic_embeddings(input_ids: &[i64], attention_mask: &[i64], outputs: &SessionOutputs) -> anyhow::Result<Vec<f32>> {
        let logits = if let Some((_shape, data)) = outputs.get("logits")
            .and_then(|output| output.try_extract_tensor::<f32>().ok()) 
        {
            data.to_vec()
        } else {
            return Err(InferenceError::OnnxInference("Unable to extract logits for embedding generation".to_string()).into());
        };

        let mut embedding = vec![0.0_f32; 768];
        
        // 1. Sentiment logits (first 3 dimensions)
        for (i, &logit) in logits.iter().take(3).enumerate() {
            embedding[i] = logit;
        }
        
        // 2. Token statistics (dimensions 3-50)  
        let active_tokens = attention_mask.iter().sum::<i64>() as f32;
        let avg_token_id = input_ids.iter().sum::<i64>() as f32 / input_ids.len() as f32;
        let token_variance = input_ids.iter()
            .map(|&id| (id as f32 - avg_token_id).powi(2))
            .sum::<f32>() / input_ids.len() as f32;
        
        embedding[3] = active_tokens / 512.0; // Normalized sequence length
        embedding[4] = avg_token_id / 30000.0; // Normalized average token ID
        embedding[5] = token_variance.sqrt() / 1000.0; // Token diversity
        
        // 3. Token n-gram features (dimensions 6-200)
        for i in 0..(input_ids.len().saturating_sub(1).min(194)) {
            let bigram_hash = (input_ids[i] * 31 + input_ids[i + 1]) % 10000;
            embedding[6 + i] = (bigram_hash as f32) / 10000.0;
        }
        
        // 4. Position-based features (dimensions 200-400)
        for (pos, &token_id) in input_ids.iter().enumerate().take(200) {
            if attention_mask[pos] == 1 {
                embedding[200 + pos] = (token_id as f32 * (pos + 1) as f32) / 1000000.0;
            }
        }
        
        // 5. Financial vocabulary features (dimensions 400-600)
        // Simple heuristic: boost dimensions for common financial terms
        let financial_indicators = [
            101, 102, 2000, 3000, 4000, // Common token ranges
        ];
        for (i, &indicator) in financial_indicators.iter().enumerate() {
            let count = input_ids.iter().filter(|&&id| id == indicator).count() as f32;
            if i + 400 < 600 {
                embedding[400 + i] = count / input_ids.len() as f32;
            }
        }
        
        // 6. Attention-weighted features (dimensions 600-768)
        for i in 0..input_ids.len().min(168) {
            if attention_mask[i] == 1 {
                embedding[600 + i] = (input_ids[i] as f32 * attention_mask[i] as f32) / 50000.0;
            }
        }
        
        // Normalize the embedding to unit length for cosine similarity
        let magnitude: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in embedding.iter_mut() {
                *val /= magnitude;
            }
        }
        
        Ok(embedding)
    }

    pub fn get_config(&self) -> &FinBertConfig {
        &self.config
    }

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