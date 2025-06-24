import torch
import json
from pathlib import Path
import time

# Fix für PyTorch 2.6 - Definiere alle möglichen Modell-Klassen
import sys
sys.path.append('.')

# Definiere die Modell-Klassen hier nochmal (für Pickle-Kompatibilität)
class FixedMobileGECModel(torch.nn.Module):
    """Dummy-Klasse für das Laden des alten Modells"""
    def __init__(self, base_model, vocab_size, embedding_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = base_model.shared if hasattr(base_model, 'shared') else torch.nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = 256
        self.max_length = 32
        
        self.input_projection = torch.nn.Linear(embedding_dim, self.hidden_dim)
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.output_projection = torch.nn.Linear(self.hidden_dim, self.vocab_size)
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_length:
            input_ids = input_ids[:, :self.max_length]
        
        token_embeddings = self.embeddings(input_ids)
        hidden = self.input_projection(token_embeddings)
        transformed = self.transformer(hidden)
        logits = self.output_projection(transformed)
        
        input_one_hot = torch.nn.functional.one_hot(input_ids, self.vocab_size).float()
        copy_bias = input_one_hot * 2.0
        final_logits = logits + copy_bias
        
        return torch.argmax(final_logits, dim=-1)

class SimpleMobileGECModel(torch.nn.Module):
    """LSTM-basierte Mobile-Version"""
    def __init__(self, base_model, vocab_size, embedding_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = base_model.shared if hasattr(base_model, 'shared') else torch.nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = 256
        self.max_length = 32
        
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
        
        if seq_len > self.max_length:
            input_ids = input_ids[:, :self.max_length]
        
        token_embeddings = self.embeddings(input_ids)
        hidden = self.input_projection(token_embeddings)
        hidden = self.dropout(hidden)
        
        lstm_out, _ = self.lstm(hidden)
        lstm_out = self.dropout(lstm_out)
        
        logits = self.output_projection(lstm_out)
        
        input_one_hot = torch.nn.functional.one_hot(input_ids, self.vocab_size).float()
        copy_bias = input_one_hot * 2.0
        final_logits = logits + copy_bias
        
        return torch.argmax(final_logits, dim=-1)

# Registriere beide Klassen für sicheres Laden
torch.serialization.add_safe_globals([FixedMobileGECModel, SimpleMobileGECModel])

# Versuche auch Import aus dem originalen Converter
try:
    from pytorch_mobile_converter_fixed import SimpleMobileGECModel as ImportedSimpleMobileGECModel
    torch.serialization.add_safe_globals([ImportedSimpleMobileGECModel])
except ImportError:
    print("⚠️ Could not import from pytorch_mobile_converter_fixed - using local definitions")

class PyTorchMobileModelTester:
    def __init__(self):
        self.model_path = Path("./models/pytorch_mobile/german_gec_mobile.pth")
        self.vocab_path = Path("./models/pytorch_mobile/vocab.json")
        self.config_path = Path("./models/pytorch_mobile/model_config.json")
        
        if not self.model_path.exists():
            print(f"❌ Model not found: {self.model_path}")
            print("Please run: python pytorch_mobile_converter_fixed.py first")
            return
        
        self.load_model_and_assets()
    
    def load_model_and_assets(self):
        print("🔧 Loading PyTorch Mobile model...")
        
        try:
            # PyTorch 2.6 Fix: Verwende weights_only=False für trusted source
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            print(f"✅ Model loaded: {self.model_path.stat().st_size / (1024*1024):.1f} MB")
            print(f"   Keys in checkpoint: {list(checkpoint.keys())}")
            
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            print(f"✅ Vocabulary loaded: {len(self.vocab)} tokens")
            
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            print(f"✅ Config loaded: {self.config['model_type']}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def simple_tokenize(self, text):
        words = text.split()
        tokens = []
        
        for word in words:
            found = False
            for variant in [word, word.lower(), word.capitalize(), f'▁{word}', f'▁{word.lower()}']:
                if variant in self.vocab:
                    tokens.append(self.vocab[variant])
                    found = True
                    break
            
            if not found:
                tokens.append(self.vocab.get('<unk>', 1))
        
        return tokens
    
    def simple_detokenize(self, token_ids, max_tokens=20):
        words = []
        for token_id in token_ids[:max_tokens]:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                
                if token in ['<pad>', '</s>', '<eos>']:
                    break
                
                if token not in ['<s>', '<unk>', 'Korrigiere:', 'korrigiere']:
                    clean_token = token[1:] if token.startswith('▁') else token
                    words.append(clean_token)
        
        return ' '.join(words)
    
    def test_model_direct(self):
        print("\n" + "="*60)
        print("🧪 TESTING PYTORCH MOBILE MODEL DIRECTLY")
        print("="*60)
        
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            if 'complete_model' in checkpoint:
                print("✅ Found complete model in checkpoint!")
                mobile_model = checkpoint['complete_model']
                mobile_model.eval()
                
                test_sentences = [
                    "Das ist ein fehler",
                    "Ich gehe zur schule", 
                    "Er hat das buch gelest"
                ]
                
                successful_tests = 0
                
                for sentence in test_sentences:
                    print(f"\nTesting: '{sentence}'")
                    
                    try:
                        tokens = self.simple_tokenize(f"Korrigiere: {sentence}")
                        
                        while len(tokens) < 32:
                            tokens.append(0)
                        tokens = tokens[:32]
                        
                        input_tensor = torch.tensor([tokens], dtype=torch.long)
                        
                        with torch.no_grad():
                            start_time = time.time()
                            output = mobile_model(input_tensor)
                            inference_time = (time.time() - start_time) * 1000
                        
                        predicted_text = self.simple_detokenize(output[0].tolist())
                        
                        print(f"   Input: '{sentence}'")
                        print(f"   Output: '{predicted_text}'")
                        print(f"   Inference: {inference_time:.1f}ms")
                        
                        if len(predicted_text.strip()) > 0:
                            successful_tests += 1
                            print("   ✅ Success")
                        else:
                            print("   ⚠️  Empty output")
                        
                    except Exception as e:
                        print(f"   ❌ Failed: {e}")
                
                success_rate = successful_tests / len(test_sentences) * 100
                print(f"\n📊 Direct Model Test Results:")
                print(f"   Success rate: {successful_tests}/{len(test_sentences)} ({success_rate:.1f}%)")
                
                return success_rate >= 75
                
            else:
                print("❌ Complete model not found in checkpoint")
                print("Available keys:", list(checkpoint.keys()))
                return False
                
        except Exception as e:
            print(f"❌ Direct model test failed: {e}")
            return False
    
    def test_vocabulary_coverage(self):
        print("\n" + "="*60)
        print("📚 VOCABULARY COVERAGE TEST")
        print("="*60)
        
        test_words = [
            'Das', 'ist', 'ein', 'fehler', 'Korrigiere',
            'ich', 'gehe', 'zur', 'schule', 'der', 'die', 'das'
        ]
        
        found_count = 0
        
        for word in test_words:
            variants = [word, word.lower(), word.capitalize(), f'▁{word}', f'▁{word.lower()}']
            found = False
            found_as = None
            
            for variant in variants:
                if variant in self.vocab:
                    found = True
                    found_as = variant
                    found_count += 1
                    break
            
            if found:
                print(f"✅ '{word}' found as '{found_as}' (ID: {self.vocab[found_as]})")
            else:
                print(f"❌ '{word}' not found")
        
        coverage = found_count / len(test_words) * 100
        print(f"\n📊 Coverage: {found_count}/{len(test_words)} ({coverage:.1f}%)")
        
        return coverage >= 75
    
    def test_tokenization_pipeline(self):
        print("\n" + "="*60)
        print("🔧 TOKENIZATION PIPELINE TEST")
        print("="*60)
        
        test_sentences = [
            "Das ist ein fehler",
            "Korrigiere: Ich gehe zur schule",
            "Er hat das buch gelest"
        ]
        
        for sentence in test_sentences:
            print(f"\nTesting: '{sentence}'")
            
            tokens = self.simple_tokenize(sentence)
            print(f"   Tokens: {tokens[:10]}...")
            
            reconstructed = self.simple_detokenize(tokens)
            print(f"   Reconstructed: '{reconstructed}'")
            
            original_words = set(sentence.lower().split())
            reconstructed_words = set(reconstructed.lower().split())
            
            overlap = len(original_words.intersection(reconstructed_words))
            total = len(original_words)
            quality = overlap / total * 100 if total > 0 else 0
            
            print(f"   Quality: {overlap}/{total} words preserved ({quality:.1f}%)")
    
    def analyze_model_structure(self):
        print("\n" + "="*60)
        print("🔍 MODEL STRUCTURE ANALYSIS")
        print("="*60)
        
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model_state_dict', {})
            
            print(f"Model parameters:")
            for key in sorted(state_dict.keys()):
                tensor = state_dict[key]
                print(f"   {key}: {tensor.shape}")
            
            if 'embeddings.weight' in state_dict:
                emb_shape = state_dict['embeddings.weight'].shape
                print(f"\n📊 Embeddings: {emb_shape[0]} vocab × {emb_shape[1]} dims")
            
            if 'output_projection.weight' in state_dict:
                out_shape = state_dict['output_projection.weight'].shape
                print(f"📊 Output projection: {out_shape[1]} → {out_shape[0]} vocab")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    def test_model_reconstruction(self):
        """Teste ob das Modell korrekt rekonstruiert werden kann"""
        print("\n" + "="*60)
        print("🔧 MODEL RECONSTRUCTION TEST")
        print("="*60)
        
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            if 'complete_model' in checkpoint:
                print("✅ Complete model found - testing direct loading")
                model = checkpoint['complete_model']
                print(f"   Model type: {type(model)}")
                print(f"   Model modules: {list(model._modules.keys())}")
                return True
                
            elif 'model_state_dict' in checkpoint and 'config' in checkpoint:
                print("✅ State dict found - testing reconstruction")
                
                # Versuche Modell zu rekonstruieren
                config = checkpoint['config']
                print(f"   Config: {config}")
                
                if SimpleMobileGECModel is not None:
                    # Mock embeddings für Test
                    class MockModel:
                        def __init__(self):
                            self.shared = torch.nn.Embedding(config['vocab_size'], config.get('embedding_dim', 512))
                    
                    mock_base = MockModel()
                    reconstructed_model = SimpleMobileGECModel(
                        mock_base, 
                        config['vocab_size'], 
                        config.get('embedding_dim', 512)
                    )
                    
                    reconstructed_model.load_state_dict(checkpoint['model_state_dict'])
                    print("✅ Model reconstruction successful")
                    return True
                else:
                    print("❌ Cannot reconstruct - trying with FixedMobileGECModel")
                    
                    # Versuche mit FixedMobileGECModel
                    try:
                        class MockModel:
                            def __init__(self):
                                self.shared = torch.nn.Embedding(config['vocab_size'], config.get('embedding_dim', 512))
                        
                        mock_base = MockModel()
                        reconstructed_model = FixedMobileGECModel(
                            mock_base, 
                            config['vocab_size'], 
                            config.get('embedding_dim', 512)
                        )
                        
                        reconstructed_model.load_state_dict(checkpoint['model_state_dict'])
                        print("✅ Model reconstruction successful with FixedMobileGECModel")
                        return True
                    except Exception as e:
                        print(f"❌ FixedMobileGECModel reconstruction failed: {e}")
                        return False
            
            else:
                print("❌ Neither complete model nor state dict found")
                return False
                
        except Exception as e:
            print(f"❌ Reconstruction test failed: {e}")
            return False
    
    def run_all_tests(self):
        print("🔧 PYTORCH MOBILE MODEL TESTER")
        print("="*60)
        
        print(f"Model: {self.model_path}")
        print(f"Size: {self.model_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"Config: {self.config}")
        
        self.analyze_model_structure()
        vocab_ok = self.test_vocabulary_coverage()
        self.test_tokenization_pipeline()
        reconstruct_ok = self.test_model_reconstruction()
        model_ok = self.test_model_direct()
        
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        if vocab_ok:
            print("✅ Vocabulary coverage: GOOD")
        else:
            print("❌ Vocabulary coverage: POOR")
            
        if reconstruct_ok:
            print("✅ Model reconstruction: WORKING")
        else:
            print("❌ Model reconstruction: FAILED")
        
        if model_ok:
            print("✅ Model functionality: WORKING")
        else:
            print("❌ Model functionality: NEEDS FIXING")
        
        print("\n🎯 CONCLUSION:")
        if vocab_ok and reconstruct_ok and model_ok:
            print("🎉 Model is ready for Flutter integration!")
        elif vocab_ok and reconstruct_ok and not model_ok:
            print("Model loads correctly but inference needs debugging.")
        elif not vocab_ok:
            print("Vocabulary coverage insufficient - need better tokenization.")
        elif not reconstruct_ok:
            print("Model cannot be loaded - need to fix the converter.")
        else:
            print("Multiple issues detected - need comprehensive fixes.")

def main():
    try:
        tester = PyTorchMobileModelTester()
        tester.run_all_tests()
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()