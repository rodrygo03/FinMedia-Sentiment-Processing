#!/usr/bin/env python3
"""
Simple test script to verify FinBERT setup and identify issues
"""

import sys
import traceback

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'torch',
        'transformers', 
        'onnx',
        'onnxruntime',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def test_model_download():
    """Test downloading the FinBERT model"""
    print("\nTesting model download...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = "yiyanghkust/finbert-tone"
        print(f"Downloading {model_name}...")
        
        # Test tokenizer download
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Tokenizer downloaded - vocab size: {len(tokenizer)}")
        
        # Test model download  
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        print(f"✅ Model downloaded - parameters: {model.num_parameters():,}")
        print(f"✅ Model config: {model.config.num_labels} labels")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        traceback.print_exc()
        return None, None

def test_basic_inference(tokenizer, model):
    """Test basic inference with the model"""
    print("\nTesting basic inference...")
    
    if not tokenizer or not model:
        print("❌ Skipping inference test - model not available")
        return False
        
    try:
        import torch
        
        # Test text
        test_text = "Bitcoin price surges to new all-time high"
        
        # Tokenize
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True
        )
        
        print(f"✅ Tokenization successful - input shape: {inputs['input_ids'].shape}")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
        print(f"✅ Inference successful - output shape: {logits.shape}")
        print(f"✅ Probabilities: {probabilities[0].tolist()}")
        
        # Check labels
        labels = ["negative", "neutral", "positive"]
        predicted_class = probabilities.argmax().item()
        confidence = probabilities.max().item()
        
        print(f"✅ Prediction: {labels[predicted_class]} (confidence: {confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        traceback.print_exc()
        return False

def test_onnx_conversion(tokenizer, model):
    """Test ONNX conversion"""
    print("\nTesting ONNX conversion...")
    
    if not tokenizer or not model:
        print("❌ Skipping ONNX test - model not available")
        return False
        
    try:
        import torch
        import onnx
        
        # Create dummy input
        dummy_text = "Test financial sentiment analysis"
        dummy_input = tokenizer(
            dummy_text,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True
        )
        
        # Export to ONNX
        onnx_path = "test_finbert.onnx"
        
        print("Exporting to ONNX...")
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
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"✅ ONNX export successful - saved to {onnx_path}")
        
        # Test ONNX inference
        import onnxruntime as ort
        import numpy as np
        
        session = ort.InferenceSession(onnx_path)
        
        # Convert inputs to numpy
        onnx_inputs = {
            "input_ids": dummy_input["input_ids"].numpy().astype(np.int64),
            "attention_mask": dummy_input["attention_mask"].numpy().astype(np.int64)
        }
        
        # Run ONNX inference
        onnx_outputs = session.run(None, onnx_inputs)
        
        print(f"✅ ONNX inference successful - output shape: {onnx_outputs[0].shape}")
        
        # Clean up test file
        import os
        os.remove(onnx_path)
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX conversion failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("FinBERT Setup Test")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Setup incomplete - fix dependencies first")
        return 1
    
    # Test model download
    tokenizer, model = test_model_download()
    
    # Test basic inference
    if not test_basic_inference(tokenizer, model):
        print("\n❌ Basic inference failed")
        return 1
    
    # Test ONNX conversion
    if not test_onnx_conversion(tokenizer, model):
        print("\n❌ ONNX conversion failed") 
        return 1
    
    print("\n✅ All tests passed! Your setup is working correctly.")
    print("You can now run quantize_FinBERT.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())