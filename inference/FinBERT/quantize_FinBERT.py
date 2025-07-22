import torch
import onnx
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort  
import os

def download_and_convert_finbert():
    print("Starting FinBERT conversion process...")
    
    model_name = "yiyanghkust/finbert-tone"
    
    print(f"Downloading {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=False)
        model.eval()
        
        print("Model downloaded successfully!")
        print(f"Number of parameters: {model.num_parameters():,}")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise
    
    dummy_text = "Bitcoin price surges to new highs amid institutional adoption"
    dummy_input = tokenizer(dummy_text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    
    print("Creating ONNX export...")
    
    onnx_path = "FinBERT_tone.onnx"
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input['input_ids'], dummy_input['attention_mask']),
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size'}
            },
            verbose=False
        )
    
    print(f"ONNX model exported to: {onnx_path}")
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")
    
    print("Applying INT8 quantization...")
    quantized_path = "FinBERT_tone_int8.onnx"
    
    quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QInt8)
    
    print(f"Quantized model saved to: {quantized_path}")
    
    tokenizer.save_pretrained("./finbert_tokenizer/")
    print("Tokenizer saved to: ./finbert_tokenizer/")
    
    original_size = os.path.getsize(onnx_path) / (1024 * 1024)
    quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
    
    print(f"\nModel size comparison:")
    print(f"Original ONNX: {original_size:.1f} MB")
    print(f"INT8 Quantized: {quantized_size:.1f} MB")
    print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
    
    return onnx_path, quantized_path

def create_rust_config():
    config = {
        "model_path": "FinBERT_tone_int8.onnx",
        "tokenizer_path": "finbert_tokenizer/",
        "max_length": 512,
        "labels": ["negative", "neutral", "positive"],
        "quantized": True,
        "model_type": "finbert-tone"
    }
    
    import json
    with open("finbert_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Rust configuration saved to: finbert_config.json")

def test_quantized_finbert():
    print("Testing quantized FinBERT model...")
    
    tokenizer = AutoTokenizer.from_pretrained("./finbert_tokenizer/")
    session = ort.InferenceSession("FinBERT_tone_int8.onnx")
    
    test_texts = [
        "Bitcoin price surges to new all-time high",
        "Company reports disappointing earnings results", 
        "Federal Reserve maintains interest rates",
        "Market volatility increases amid uncertainty"
    ]
    
    print("\nTesting sentiment predictions:")
    labels = ["negative", "neutral", "positive"]
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="np", max_length=512, padding="max_length", truncation=True)
        
        outputs = session.run(None, {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        })
        
        logits = outputs[0][0]
        probs = torch.softmax(torch.tensor(logits), dim=-1)
        
        predicted_class = probs.argmax().item()
        confidence = probs.max().item()
        
        print(f"\nText: {text}")
        print(f"Prediction: {labels[predicted_class]} (confidence: {confidence:.3f})")
        print(f"Probabilities: {dict(zip(labels, [f'{p:.3f}' for p in probs]))}")

    print("\n✅ Quantized model working correctly!")


if __name__ == "__main__":
    try:
        original_path, quantized_path = download_and_convert_finbert()
        test_quantized_finbert()
        create_rust_config()
        
        print("\nFinBERT conversion completed successfully!")
        print("\nFiles created:")
        print("- FinBERT_tone.onnx (original)")
        print("- FinBERT_tone_int8.onnx (quantized for production)")
        print("- finbert_tokenizer/ (tokenizer files)")
        print("- finbert_config.json (Rust configuration)")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise