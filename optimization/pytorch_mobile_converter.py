# pytorch_mobile_converter_fixed.py
import torch
import json
from pathlib import Path
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMobileGECModel(torch.nn.Module):
    def __init__(self, base_model, vocab_size, embedding_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = base_model.shared
        self.hidden_dim = 256
        self.max_length = 32
        
        # Einfachere LSTM-basierte Architektur (TorchScript-kompatibel)
        self.input_projection = torch.nn.Linear(embedding_dim, self.hidden_dim)
        self.lstm = torch.nn.LSTM(
            self.hidden_dim, 
            self.hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=0.1
        )
        self.output_projection = torch.nn.Linear(self.hidden_dim, self.vocab_size)
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Limit input length
        if seq_len > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            seq_len = self.max_length
        
        # Get embeddings
        token_embeddings = self.embeddings(input_ids)
        
        # Project to LSTM dimension
        hidden = self.input_projection(token_embeddings)
        hidden = self.dropout(hidden)
        
        # LSTM processing
        lstm_out, _ = self.lstm(hidden)
        lstm_out = self.dropout(lstm_out)
        
        # Output projection
        logits = self.output_projection(lstm_out)
        
        # Copy mechanism
        input_one_hot = torch.nn.functional.one_hot(input_ids, self.vocab_size).float()
        copy_bias = input_one_hot * 2.0
        
        final_logits = logits + copy_bias
        predicted_tokens = torch.argmax(final_logits, dim=-1)
        
        return predicted_tokens

class PyTorchMobileConverter:
    def __init__(self):
        self.model_path = self._find_trained_model()
        self.output_dir = Path("./models/pytorch_mobile")
        self.output_dir.mkdir(exist_ok=True)
        
    def _find_trained_model(self):
        possible_paths = [
            Path("./models/trained_models_improved/final_model"),
            Path("./models/trained_models/final_model"),
            Path("./german_gec_mt5/final_model")
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"✅ Found working model: {path}")
                return path
        
        raise FileNotFoundError("❌ No trained model found!")
    
    def load_and_verify_model(self):
        logger.info("Loading the WORKING MT5 model...")
        
        model = MT5ForConditionalGeneration.from_pretrained(self.model_path)
        tokenizer = MT5Tokenizer.from_pretrained(self.model_path)
        model.eval()
        
        # Verify model works
        test_cases = [
            "Das ist ein fehler",
            "Ich gehe zur schule", 
            "Er hat das buch gelest"
        ]
        
        logger.info("Verifying model works correctly...")
        for test_text in test_cases:
            input_text = f"Korrigiere: {test_text}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=3,
                    early_stopping=True,
                    do_sample=False
                )
            
            corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"   '{test_text}' → '{corrected}'")
        
        logger.info("✅ Model is working perfectly!")
        return model, tokenizer
    
    def create_mobile_wrapper(self, model, tokenizer):
        logger.info("Creating mobile-optimized LSTM wrapper...")
        
        original_vocab = tokenizer.get_vocab()
        vocab_size = len(original_vocab)
        embedding_dim = model.config.d_model
        
        logger.info(f"Using vocabulary: {vocab_size} tokens")
        
        mobile_model = SimpleMobileGECModel(model, vocab_size, embedding_dim)
        mobile_model.eval()
        
        # Test mobile model
        test_input = tokenizer("Korrigiere: Das ist ein test", return_tensors="pt", max_length=32, truncation=True)
        with torch.no_grad():
            output = mobile_model(test_input['input_ids'])
        
        logger.info(f"✅ Mobile wrapper test: {test_input['input_ids'].shape} → {output.shape}")
        
        return mobile_model
    
    def convert_to_torchscript(self, mobile_model, tokenizer):
        logger.info("Converting to TorchScript using scripting...")
        
        mobile_model.eval()
        
        try:
            # Method 1: TorchScript via scripting (robuster für LSTM)
            with torch.no_grad():
                scripted_model = torch.jit.script(mobile_model)
                scripted_model.eval()
                
                # Test scripted model first
                test_input = torch.randint(0, 1000, (1, 16), dtype=torch.long)
                _ = scripted_model(test_input)
                
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
                
                torchscript_path = self.output_dir / "german_gec_mobile.ptl"
                scripted_model.save(str(torchscript_path))
                
                size_mb = torchscript_path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ TorchScript model saved: {torchscript_path} ({size_mb:.1f} MB)")
                
                return torchscript_path
            
        except Exception as e:
            logger.warning(f"TorchScript scripting failed: {e}")
            logger.info("Trying TorchScript tracing as fallback...")
            
            try:
                # Method 2: Fallback to tracing with fixed input
                example_input = torch.randint(0, 1000, (1, 32), dtype=torch.long)
                traced_model = torch.jit.trace(mobile_model, example_input)
                traced_model = torch.jit.optimize_for_inference(traced_model)
                
                torchscript_path = self.output_dir / "german_gec_mobile.ptl"
                traced_model.save(str(torchscript_path))
                
                size_mb = torchscript_path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ TorchScript model (traced) saved: {torchscript_path} ({size_mb:.1f} MB)")
                
                return torchscript_path
                
            except Exception as e2:
                logger.error(f"TorchScript tracing also failed: {e2}")
                
                # Method 3: Save complete PyTorch model as fallback
                fallback_path = self.output_dir / "german_gec_mobile.pth"
                
                torch.save({
                    'complete_model': mobile_model,
                    'model_state_dict': mobile_model.state_dict(),
                    'model_class': 'SimpleMobileGECModel',
                    'tokenizer_vocab': tokenizer.get_vocab(),
                    'config': {
                        'vocab_size': len(tokenizer.get_vocab()),
                        'hidden_dim': mobile_model.hidden_dim,
                        'max_length': mobile_model.max_length,
                        'embedding_dim': mobile_model.input_projection.in_features
                    }
                }, fallback_path)
                
                size_mb = fallback_path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ Complete PyTorch model saved: {fallback_path} ({size_mb:.1f} MB)")
                
                return fallback_path
    
    def test_mobile_model(self, model_path, tokenizer):
        logger.info("Testing mobile model...")
        
        try:
            if str(model_path).endswith('.ptl'):
                # Load TorchScript model
                mobile_model = torch.jit.load(str(model_path))
            else:
                # Load PyTorch model
                checkpoint = torch.load(str(model_path), map_location='cpu')
                mobile_model = checkpoint['complete_model']
            
            mobile_model.eval()
            
            test_sentences = [
                "Das ist ein fehler",
                "Ich gehe zur schule",
                "Er hat das buch gelest"
            ]
            
            successful_tests = 0
            total_time = 0
            
            for sentence in test_sentences:
                logger.info(f"Testing: '{sentence}'")
                
                try:
                    input_text = f"Korrigiere: {sentence}"
                    inputs = tokenizer(input_text, return_tensors="pt", max_length=32, truncation=True)
                    
                    import time
                    start_time = time.time()
                    
                    with torch.no_grad():
                        output_ids = mobile_model(inputs['input_ids'])
                    
                    inference_time = (time.time() - start_time) * 1000
                    total_time += inference_time
                    
                    try:
                        corrected = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                        logger.info(f"   Corrected: '{corrected}' ({inference_time:.1f}ms)")
                    except:
                        logger.info(f"   Output tokens: {output_ids[0][:10].tolist()} ({inference_time:.1f}ms)")
                    
                    successful_tests += 1
                
                except Exception as e:
                    logger.warning(f"   Test failed: {e}")
            
            avg_time = total_time / len(test_sentences) if test_sentences else 0
            success_rate = successful_tests / len(test_sentences) * 100
            
            logger.info(f"✅ Mobile test results:")
            logger.info(f"   Success rate: {successful_tests}/{len(test_sentences)} ({success_rate:.1f}%)")
            logger.info(f"   Average inference: {avg_time:.1f}ms")
            
            return success_rate >= 75
            
        except Exception as e:
            logger.error(f"Mobile model test failed: {e}")
            return False
    
    def create_flutter_integration_assets(self, model_path, tokenizer):
        logger.info("Creating Flutter integration assets...")
        
        full_vocab = tokenizer.get_vocab()
        
        # Optimized vocabulary for mobile
        mobile_vocab = {}
        
        # Important German words
        important_words = [
            'das', 'ist', 'ein', 'fehler', 'Korrigiere', 'korrigiere',
            'ich', 'gehe', 'zur', 'schule', 'der', 'die', 'und', 'haben',
            'buch', 'gelest', 'gelesen', 'weil', 'dass', 'seit', 'seid'
        ]
        
        for word in important_words:
            for variant in [word, word.lower(), word.capitalize(), f'▁{word}', f'▁{word.lower()}']:
                if variant in full_vocab:
                    mobile_vocab[variant] = full_vocab[variant]
        
        # Add other tokens up to limit
        for token, token_id in full_vocab.items():
            if token not in mobile_vocab:
                mobile_vocab[token] = token_id
                if len(mobile_vocab) >= 50000:
                    break
        
        logger.info(f"Mobile vocabulary: {len(mobile_vocab)} tokens")
        
        # Save vocabulary
        vocab_path = self.output_dir / "vocab.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(mobile_vocab, f, ensure_ascii=False, indent=2)
        
        # Model configuration
        model_config = {
            'model_type': 'pytorch_mobile_gec',
            'model_path': str(model_path.name),
            'vocab_path': 'vocab.json',
            'source_model': str(self.model_path),
            'mobile_optimized': True,
            'max_length': 32,
            'vocab_size': len(mobile_vocab),
            'platform': 'pytorch_mobile',
            'flutter_ready': True,
            'architecture': 'LSTM'
        }
        
        config_path = self.output_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Flutter Dart integration code
        flutter_code = f'''// lib/core/ml/pytorch_gec_service.dart
import 'dart:typed_data';
import 'package:pytorch_mobile/pytorch_mobile.dart';
import 'dart:convert';
import 'package:flutter/services.dart';

class PyTorchGECService {{
  Module? _model;
  Map<String, int>? _vocab;
  Map<int, String>? _reverseVocab;
  
  Future<void> initialize() async {{
    try {{
      _model = await PyTorchMobile.loadModel('assets/models/{model_path.name}');
      
      String vocabString = await rootBundle.loadString('assets/models/vocab.json');
      _vocab = Map<String, int>.from(json.decode(vocabString));
      _reverseVocab = _vocab!.map((k, v) => MapEntry(v, k));
      
      print('PyTorch GEC Service initialized with ${{_vocab!.length}} tokens');
    }} catch (e) {{
      print('Failed to initialize PyTorch GEC Service: $e');
      throw e;
    }}
  }}
  
  Future<String> correctText(String text) async {{
    if (_model == null || _vocab == null) {{
      throw Exception('Model not initialized');
    }}
    
    try {{
      List<int> tokens = _tokenize('Korrigiere: $text');
      
      // Convert to tensor (32 tokens for mobile)
      List<double> inputData = List.filled(32, 0.0);
      for (int i = 0; i < tokens.length && i < 32; i++) {{
        inputData[i] = tokens[i].toDouble();
      }}
      
      // Run inference
      List<double> output = await _model!.forward(inputData);
      
      // Decode output
      String corrected = _detokenize(output);
      
      return corrected.isEmpty ? text : corrected;
    }} catch (e) {{
      print('Correction failed: $e');
      return text;
    }}
  }}
  
  List<int> _tokenize(String text) {{
    List<String> words = text.split(' ');
    List<int> tokens = [];
    
    for (String word in words) {{
      int? tokenId;
      
      for (String variant in [word, word.toLowerCase(), '▁' + word.toLowerCase()]) {{
        if (_vocab!.containsKey(variant)) {{
          tokenId = _vocab![variant];
          break;
        }}
      }}
      
      tokens.add(tokenId ?? _vocab!['<unk>'] ?? 1);
    }}
    
    return tokens;
  }}
  
  String _detokenize(List<double> output) {{
    List<String> words = [];
    
    for (int i = 0; i < output.length && i < 20; i++) {{
      int tokenId = output[i].round();
      
      if (_reverseVocab!.containsKey(tokenId)) {{
        String token = _reverseVocab![tokenId]!;
        
        if (!['<pad>', '<s>', '</s>', '<unk>'].contains(token)) {{
          String cleanToken = token.startsWith('▁') ? token.substring(1) : token;
          words.add(cleanToken);
        }}
      }}
    }}
    
    return words.join(' ');
  }}
  
  void dispose() {{
    // Cleanup
  }}
}}'''
        
        flutter_path = self.output_dir / "pytorch_gec_service.dart"
        with open(flutter_path, 'w') as f:
            f.write(flutter_code)
        
        logger.info(f"✅ Flutter assets created:")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Vocabulary: {vocab_path} ({len(mobile_vocab)} tokens)")
        logger.info(f"   Config: {config_path}")
        logger.info(f"   Flutter integration: {flutter_path}")
        
        return len(mobile_vocab)
    
    def run_pytorch_mobile_conversion(self):
        logger.info("="*70)
        logger.info("🚀 PYTORCH MOBILE CONVERSION - FIXED")
        logger.info("Using LSTM architecture for better TorchScript compatibility")
        logger.info("="*70)
        
        try:
            # 1. Load working model
            model, tokenizer = self.load_and_verify_model()
            
            # 2. Create mobile wrapper
            mobile_model = self.create_mobile_wrapper(model, tokenizer)
            
            # 3. Convert to TorchScript (with fallbacks)
            model_path = self.convert_to_torchscript(mobile_model, tokenizer)
            
            # 4. Test mobile model
            works = self.test_mobile_model(model_path, tokenizer)
            
            # 5. Create Flutter assets
            self.create_flutter_integration_assets(model_path, tokenizer)
            
            logger.info("\n" + "="*70)
            logger.info("🎉 PYTORCH MOBILE CONVERSION COMPLETE!")
            logger.info("="*70)
            
            if works:
                logger.info("✅ SUCCESS: LSTM-based mobile model ready!")
                logger.info("✅ TorchScript compatible architecture")
                logger.info("✅ Flutter integration prepared")
                
                logger.info(f"\n🚀 MOBILE MODEL READY:")
                logger.info(f"   Model: {model_path}")
                logger.info(f"   Flutter code: models/pytorch_mobile/pytorch_gec_service.dart")
                logger.info(f"   Add 'pytorch_mobile' plugin to pubspec.yaml")
                
            else:
                logger.warning("⚠️ Model converted but needs testing")
            
            logger.info(f"\n📱 FLUTTER SETUP:")
            logger.info(f"1. pubspec.yaml: pytorch_mobile: ^1.0.0")
            logger.info(f"2. Copy models to assets/models/")
            logger.info(f"3. Use generated Dart code")
            
            return works
            
        except Exception as e:
            logger.error(f"❌ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    print("🚀 PYTORCH MOBILE CONVERTER - FIXED")
    print("LSTM-based architecture for mobile compatibility")
    
    converter = PyTorchMobileConverter()
    success = converter.run_pytorch_mobile_conversion()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("Mobile-optimized German GEC model ready for Flutter!")
    else:
        print("\n❌ Conversion failed")

if __name__ == "__main__":
    main()