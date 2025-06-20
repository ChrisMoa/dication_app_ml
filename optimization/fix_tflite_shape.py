import tensorflow as tf
import numpy as np
import json
import time
from pathlib import Path

class TFLiteShapeFixer:
    def __init__(self, model_path="./models_mobile/german_gec_optimized.tflite"):
        self.model_path = Path(model_path)
        self.output_dir = Path("./models_mobile")
        
    def inspect_model(self):
        """Inspect TFLite model details"""
        print("🔍 Inspecting TFLite model...")
        
        try:
            interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"📊 Model: {self.model_path}")
            print(f"📦 Size: {self.model_path.stat().st_size / (1024*1024):.2f} MB")
            
            print(f"\n📥 Input Details:")
            for i, detail in enumerate(input_details):
                print(f"  Input {i}:")
                print(f"    Name: {detail['name']}")
                print(f"    Shape: {detail['shape']}")
                print(f"    Dtype: {detail['dtype']}")
                print(f"    Index: {detail['index']}")
            
            print(f"\n📤 Output Details:")
            for i, detail in enumerate(output_details):
                print(f"  Output {i}:")
                print(f"    Name: {detail['name']}")
                print(f"    Shape: {detail['shape']}")
                print(f"    Dtype: {detail['dtype']}")
                print(f"    Index: {detail['index']}")
            
            return input_details, output_details
            
        except Exception as e:
            print(f"❌ Failed to inspect model: {e}")
            return None, None
    
    def test_with_correct_shape(self):
        """Test model with correct input shape"""
        print("\n🧪 Testing with correct shapes...")
        
        try:
            interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Get expected input shape
            expected_shape = input_details[0]['shape']
            print(f"Expected input shape: {expected_shape}")
            
            # Create input that matches exactly
            if expected_shape[1] == 1:
                # Model expects shape [batch, 1] - single token?
                test_input = np.array([[42]], dtype=np.int32)
                print(f"Using single token input: {test_input.shape}")
            elif expected_shape[1] is None or expected_shape[1] == -1:
                # Dynamic shape - try different lengths
                for seq_len in [1, 8, 16, 32, 64]:
                    test_input = np.random.randint(0, 1000, size=(1, seq_len), dtype=np.int32)
                    print(f"Trying sequence length {seq_len}...")
                    try:
                        result = self._run_inference(interpreter, test_input, input_details, output_details)
                        if result:
                            print(f"✅ Success with length {seq_len}")
                            return result
                    except Exception as e:
                        print(f"❌ Failed with length {seq_len}: {e}")
                        continue
            else:
                # Fixed shape
                seq_len = expected_shape[1]
                test_input = np.random.randint(0, 1000, size=(1, seq_len), dtype=np.int32)
                print(f"Using fixed length {seq_len}: {test_input.shape}")
                return self._run_inference(interpreter, test_input, input_details, output_details)
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return None
    
    def _run_inference(self, interpreter, test_input, input_details, output_details):
        """Run inference with given input"""
        # Set input
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        # Measure time
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        inference_time = (end_time - start_time) * 1000
        
        print(f"✅ Inference successful!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Time: {inference_time:.2f} ms")
        
        return {
            'success': True,
            'input_shape': test_input.shape,
            'output_shape': output.shape,
            'inference_time_ms': inference_time,
            'test_input': test_input.tolist(),
            'sample_output': output[:, :5, :10].tolist() if len(output.shape) == 3 else output.tolist()
        }
    
    def create_inference_example(self):
        """Create working inference example"""
        print("\n📝 Creating inference example...")
        
        try:
            # Load vocabulary
            vocab_path = self.output_dir / "vocab_optimized.json"
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            
            reverse_vocab = {v: k for k, v in vocab.items()}
            
            # Test sentence
            test_sentence = "Das ist ein fehler."
            print(f"Test sentence: '{test_sentence}'")
            
            # Simple tokenization
            tokens = self.simple_tokenize(test_sentence, vocab)
            print(f"Tokens: {tokens}")
            print(f"Token words: {[reverse_vocab.get(t, f'<{t}>') for t in tokens]}")
            
            # Test inference
            interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            expected_shape = input_details[0]['shape']
            
            # Prepare input
            if expected_shape[1] == 1:
                # Single token model - use first token
                model_input = np.array([[tokens[0]]], dtype=np.int32)
            else:
                # Sequence model
                seq_len = expected_shape[1] if expected_shape[1] > 0 else 64
                
                # Pad or truncate tokens
                if len(tokens) < seq_len:
                    # Pad with <pad> token
                    pad_token = vocab.get('<pad>', 0)
                    tokens.extend([pad_token] * (seq_len - len(tokens)))
                else:
                    tokens = tokens[:seq_len]
                
                model_input = np.array([tokens], dtype=np.int32)
            
            print(f"Model input shape: {model_input.shape}")
            
            # Run inference
            result = self._run_inference(interpreter, model_input, input_details, interpreter.get_output_details())
            
            if result:
                # Decode output
                output_logits = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
                predicted_tokens = np.argmax(output_logits[0], axis=-1)
                
                predicted_words = []
                for token_id in predicted_tokens[:10]:  # First 10 tokens
                    word = reverse_vocab.get(int(token_id), f'<{token_id}>')
                    if word not in ['<pad>', '__']:
                        predicted_words.append(word)
                
                print(f"Predicted tokens: {predicted_tokens[:10]}")
                print(f"Predicted words: {predicted_words}")
                
                result['prediction_example'] = {
                    'input_sentence': test_sentence,
                    'input_tokens': tokens[:10],
                    'predicted_tokens': predicted_tokens[:10].tolist(),
                    'predicted_words': predicted_words
                }
            
            return result
            
        except Exception as e:
            print(f"❌ Inference example failed: {e}")
            return None
    
    def simple_tokenize(self, text, vocab):
        """Simple tokenization for testing"""
        # Basic preprocessing
        text = text.lower().strip()
        words = text.split()
        
        tokens = []
        for word in words:
            if word in vocab:
                tokens.append(vocab[word])
            elif f'▁{word}' in vocab:
                tokens.append(vocab[f'▁{word}'])
            else:
                # Character fallback
                for char in word:
                    if char in vocab:
                        tokens.append(vocab[char])
                    else:
                        tokens.append(vocab.get('<unk>', 1))
        
        return tokens
    
    def create_flutter_integration_code(self, working_example):
        """Create Flutter integration code"""
        if not working_example or not working_example['success']:
            print("❌ No working example to create Flutter code")
            return
        
        flutter_code = f'''// Flutter TFLite Integration Example
// Based on working model test

import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';

class GermanGECTFLite {{
  static const String modelAsset = 'assets/models/german_gec_optimized.tflite';
  static const String vocabAsset = 'assets/models/vocab_optimized.json';
  
  Interpreter? _interpreter;
  Map<String, int>? _vocab;
  
  // Model expects input shape: {working_example['input_shape']}
  static const int maxLength = {working_example['input_shape'][1]};
  
  Future<void> initialize() async {{
    try {{
      _interpreter = await Interpreter.fromAsset(modelAsset);
      
      // Load vocabulary
      final vocabJson = await rootBundle.loadString(vocabAsset);
      final Map<String, dynamic> vocabData = json.decode(vocabJson);
      _vocab = Map<String, int>.from(vocabData);
      
      print('✅ German GEC TFLite initialized');
      print('   Model size: ~16.6 MB');
      print('   Expected inference time: ~{working_example['inference_time_ms']:.0f}ms');
      
    }} catch (e) {{
      print('❌ Failed to initialize GEC: $e');
      throw Exception('GEC initialization failed');
    }}
  }}
  
  Future<String> correctText(String text) async {{
    if (_interpreter == null || _vocab == null) {{
      throw Exception('Model not initialized');
    }}
    
    try {{
      // Tokenize input
      final tokens = _tokenize(text);
      
      // Prepare input tensor
      final inputData = Int32List(maxLength);
      for (int i = 0; i < maxLength; i++) {{
        inputData[i] = i < tokens.length ? tokens[i] : 0; // 0 = <pad>
      }}
      
      // Reshape for model: [1, maxLength]
      final input = inputData.reshape([1, maxLength]);
      
      // Prepare output tensor  
      final output = List.generate(1, (_) => 
        List.generate(maxLength, (_) => 
          List.filled(32000, 0.0) // vocab_size = 32000
        )
      );
      
      // Run inference
      final stopwatch = Stopwatch()..start();
      _interpreter!.run(input, output);
      stopwatch.stop();
      
      // Get predictions (argmax)
      final predictions = <int>[];
      for (int i = 0; i < maxLength; i++) {{
        double maxLogit = output[0][i][0];
        int maxIndex = 0;
        
        for (int j = 1; j < output[0][i].length; j++) {{
          if (output[0][i][j] > maxLogit) {{
            maxLogit = output[0][i][j];
            maxIndex = j;
          }}
        }}
        
        predictions.add(maxIndex);
      }}
      
      // Detokenize
      final correctedText = _detokenize(predictions);
      
      print('Inference time: ${{stopwatch.elapsedMilliseconds}}ms');
      
      return correctedText;
      
    }} catch (e) {{
      print('❌ Correction failed: $e');
      return text; // Return original on error
    }}
  }}
  
  List<int> _tokenize(String text) {{
    // Simple tokenization (implement your logic here)
    final words = text.toLowerCase().split(' ');
    final tokens = <int>[];
    
    for (final word in words) {{
      final tokenId = _vocab![word] ?? _vocab!['<unk>'] ?? 1;
      tokens.add(tokenId);
    }}
    
    return tokens;
  }}
  
  String _detokenize(List<int> tokens) {{
    // Convert token IDs back to text
    final reverseVocab = Map.fromEntries(
      _vocab!.entries.map((e) => MapEntry(e.value, e.key))
    );
    
    final words = <String>[];
    for (final tokenId in tokens) {{
      if (tokenId == 0) break; // Stop at padding
      
      final word = reverseVocab[tokenId];
      if (word != null && !word.startsWith('__')) {{
        words.add(word.replaceAll('▁', ''));
      }}
    }}
    
    return words.join(' ');
  }}
  
  void dispose() {{
    _interpreter?.close();
  }}
}}

// Usage Example:
/*
final gec = GermanGECTFLite();
await gec.initialize();

final corrected = await gec.correctText("Das ist ein fehler.");
print("Corrected: $corrected");
*/
'''
        
        # Save Flutter code
        flutter_path = self.output_dir / "flutter_integration_example.dart"
        with open(flutter_path, 'w') as f:
            f.write(flutter_code)
        
        print(f"✅ Flutter integration code saved: {flutter_path}")
    
    def fix_and_test(self):
        """Main fix and test function"""
        print("🔧 TFLite Model Shape Fix & Test")
        print("="*50)
        
        # 1. Inspect model
        input_details, output_details = self.inspect_model()
        
        if not input_details:
            return
        
        # 2. Test with correct shapes
        working_example = self.test_with_correct_shape()
        
        # 3. Create inference example
        if not working_example:
            working_example = self.create_inference_example()
        
        # 4. Create Flutter code
        if working_example:
            self.create_flutter_integration_code(working_example)
        
        # 5. Save test results
        results = {
            'model_path': str(self.model_path),
            'model_size_mb': self.model_path.stat().st_size / (1024*1024),
            'input_details': input_details,
            'output_details': output_details,
            'working_example': working_example
        }
        
        results_path = self.output_dir / "tflite_test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.print_summary(results)
        
        return working_example is not None
    
    def print_summary(self, results):
        """Print test summary"""
        print("\n" + "="*50)
        print("TFLITE MODEL TEST SUMMARY")
        print("="*50)
        
        print(f"📱 Model: {results['model_path']}")
        print(f"📦 Size: {results['model_size_mb']:.2f} MB")
        
        if results['working_example'] and results['working_example']['success']:
            example = results['working_example']
            print(f"✅ Status: WORKING!")
            print(f"📥 Input shape: {example['input_shape']}")
            print(f"📤 Output shape: {example['output_shape']}")
            print(f"⚡ Latency: {example['inference_time_ms']:.2f} ms")
            
            if 'prediction_example' in example:
                pred = example['prediction_example']
                print(f"🧪 Test: '{pred['input_sentence']}' → {pred['predicted_words'][:3]}...")
            
            print(f"\n🎉 Ready for Flutter integration!")
            print(f"   See: {self.output_dir}/flutter_integration_example.dart")
        else:
            print(f"❌ Status: NOT WORKING")
            print(f"   Model has shape compatibility issues")
        
        print("="*50)

def main():
    model_path = "./models_mobile/german_gec_optimized.tflite"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please run: python optimized_vocab_converter.py first")
        return
    
    fixer = TFLiteShapeFixer(model_path)
    success = fixer.fix_and_test()
    
    if success:
        print("\n🚀 TFLite model is ready for mobile deployment!")
    else:
        print("\n❌ Model needs further debugging.")

if __name__ == "__main__":
    main()
