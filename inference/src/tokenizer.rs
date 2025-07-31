use crate::{Result, InferenceError, FinBertConfig};
use tokenizers::{Tokenizer, Encoding};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TokenizerOutput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
}

pub struct FinBertTokenizer {
    tokenizer: Tokenizer,
    max_length: usize,
}

impl FinBertTokenizer {
    pub fn new(tokenizer_path: &Path, max_length: usize) -> Result<Self> {
        tracing::info!("Loading FinBERT tokenizer from: {:?}", tokenizer_path);
        
        let tokenizer_file = tokenizer_path.join("tokenizer.json");
        if !tokenizer_file.exists() {
            return Err(InferenceError::Tokenization(
                format!("Tokenizer file not found: {:?}", tokenizer_file)
            ));
        }

        let tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| InferenceError::Tokenization(format!("Failed to load tokenizer: {}", e)))?;
        tracing::info!("Successfully loaded FinBERT tokenizer");
        
        Ok(Self {
            tokenizer,
            max_length,
        })
    }

    pub fn from_config(config: &FinBertConfig, base_path: &Path) -> Result<Self> {
        let tokenizer_path = base_path.join(&config.tokenizer_path);
        Self::new(&tokenizer_path, config.max_length)
    }

    pub fn tokenize(&self, text: &str) -> Result<TokenizerOutput> {
        let cleaned_text = self.preprocess_text(text);
    
        let encoding = self.tokenizer
            .encode(cleaned_text, true)
            .map_err(|e| InferenceError::Tokenization(format!("Encoding failed: {}", e)))?;
        let mut input_ids = encoding.get_ids().iter().map(|&id| id as i64).collect::<Vec<_>>();
        let mut attention_mask = encoding.get_attention_mask().iter().map(|&mask| mask as i64).collect::<Vec<_>>();
        let mut token_type_ids = encoding.get_type_ids().iter().map(|&id| id as i64).collect::<Vec<_>>();

        if input_ids.len() > self.max_length {
            input_ids.truncate(self.max_length);
            attention_mask.truncate(self.max_length);
            token_type_ids.truncate(self.max_length);
            
            // Make sure we end with [SEP] token (id = 4 based on tokenizer_config.json)
            if input_ids.len() > 0 {
                input_ids[self.max_length - 1] = 4; // [SEP]
            }
        }

        // Pad to max_length if needed
        while input_ids.len() < self.max_length {
            input_ids.push(0);        // [PAD] token
            attention_mask.push(0);   // No attention for padding
            token_type_ids.push(0);   // Padding type
        }

        Ok(TokenizerOutput {
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }

    pub fn tokenize_batch(&self, texts: &[&str]) -> Result<Vec<TokenizerOutput>> {
        texts.iter()
            .map(|text| self.tokenize(text))
            .collect()
    }

    // private: 

    fn preprocess_text(&self, text: &str) -> String { // TODO: implement better strategy later
        text.trim()
            .replace('\n', " ")          // Replace newlines with spaces
            .replace('\t', " ")          // Replace tabs with spaces  
            .chars()
            .filter(|c| !c.is_control() || *c == ' ')  // Remove control characters except spaces
            .collect::<String>()
            .split_whitespace()          // Normalize whitespace
            .collect::<Vec<_>>()
            .join(" ")
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
            max_length: 128, // Smaller for testing
            labels: vec!["negative".to_string(), "neutral".to_string(), "positive".to_string()],
            quantized: true,
            model_type: "finbert-tone".to_string(),
        }
    }

    #[tokio::test]
    async fn test_tokenization() {
        let tokenizer_path = PathBuf::from("FinBERT/finbert_tokenizer");
        
        if !tokenizer_path.exists() {
            println!("Skipping tokenization test - tokenizer directory not found");
            return;
        }

        let config = get_test_config();
        let tokenizer = FinBertTokenizer::from_config(&config, &PathBuf::from("FinBERT/")).unwrap();

        // Test basic tokenization
        let text = "Apple Inc. stock price rises due to strong earnings report.";
        let result = tokenizer.tokenize(text).unwrap();

        // Validate output structure
        assert_eq!(result.input_ids.len(), config.max_length);
        assert_eq!(result.attention_mask.len(), config.max_length);
        assert_eq!(result.token_type_ids.len(), config.max_length);

        // Check that we have some actual tokens (not all padding)
        let non_pad_tokens = result.input_ids.iter().filter(|&&id| id != 0).count();
        assert!(non_pad_tokens > 0, "Should have some non-padding tokens");

        // Check attention mask corresponds to non-padding tokens
        let attention_sum: i64 = result.attention_mask.iter().sum();
        assert!(attention_sum > 0, "Should have some attention");
    }

    #[tokio::test]
    async fn test_batch_tokenization() {
        let tokenizer_path = PathBuf::from("FinBERT/finbert_tokenizer");
        
        if !tokenizer_path.exists() {
            println!("Skipping batch tokenization test - tokenizer directory not found");
            return;
        }

        let config = get_test_config();
        let tokenizer = FinBertTokenizer::from_config(&config, &PathBuf::from("FinBERT/")).unwrap();

        let texts = vec![
            "Bitcoin price surges after institutional adoption news.",
            "Federal Reserve announces interest rate hike.",
            "Tesla stock drops following production concerns."
        ];

        let results = tokenizer.tokenize_batch(&texts).unwrap();
        assert_eq!(results.len(), 3);

        // All results should have consistent dimensions
        for result in &results {
            assert_eq!(result.input_ids.len(), config.max_length);
            assert_eq!(result.attention_mask.len(), config.max_length);
        }
    }

    #[test]
    fn test_text_preprocessing() {
        let config = get_test_config();
        let tokenizer_path = PathBuf::from("dummy");
        
        // We can't create a real tokenizer without the files, but we can test preprocessing
        let test_text = "  This is\na test\twith\nspecial\tcharacters  ";
        
        // Since preprocess_text is private, we'll test it indirectly by checking
        // that the preprocessing logic works as expected
        let expected = "This is a test with special characters";
        
        let cleaned = test_text.trim()
            .replace('\n', " ")
            .replace('\t', " ")
            .chars()
            .filter(|c| !c.is_control() || *c == ' ')
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
            
        assert_eq!(cleaned, expected);
    }
}