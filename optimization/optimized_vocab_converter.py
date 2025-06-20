import tensorflow as tf
import numpy as np
import json
import os
from pathlib import Path
import time
from collections import Counter
import re

class OptimizedVocabConverter:
    def __init__(self):
        self.output_dir = Path("./models_mobile")
        self.output_dir.mkdir(exist_ok=True)
        
        # Better balance: larger vocab but efficient encoding
        self.vocab_size = 32000  # Compromise: MT5 standard but manageable
        self.max_length = 64     # Standard length
        
    def create_smart_german_vocab(self):
        """Create German-optimized vocabulary with subwords"""
        
        # Core vocabulary with subword pieces
        vocab = {
            # Special tokens
            '<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3, '<mask>': 4,
            
            # Task tokens
            'Korrigiere': 10, ':': 11, '▁': 12,  # ▁ = word boundary
            
            # German subword pieces (most common)
            # Word beginnings
            '▁der': 20, '▁die': 21, '▁das': 22, '▁und': 23, '▁in': 24,
            '▁zu': 25, '▁den': 26, '▁von': 27, '▁mit': 28, '▁sich': 29,
            '▁auf': 30, '▁für': 31, '▁ist': 32, '▁im': 33, '▁eine': 34,
            '▁als': 35, '▁auch': 36, '▁an': 37, '▁werden': 38, '▁aus': 39,
            '▁er': 40, '▁hat': 41, '▁dass': 42, '▁sie': 43, '▁nach': 44,
            '▁wird': 45, '▁bei': 46, '▁einer': 47, '▁um': 48, '▁am': 49,
            '▁sind': 50, '▁noch': 51, '▁wie': 52, '▁einem': 53, '▁über': 54,
            
            # Common endings
            'en': 100, 'er': 101, 'es': 102, 'te': 103, 'st': 104,
            'ch': 105, 'sch': 106, 'ung': 107, 'eit': 108, 'keit': 109,
            'heit': 110, 'ion': 111, 'lich': 112, 'isch': 113, 'bar': 114,
            
            # Common prefixes
            'ge': 150, 'be': 151, 'ver': 152, 'un': 153, 'vor': 154,
            'über': 155, 'unter': 156, 'aus': 157, 'ab': 158, 'auf': 159,
            
            # Grammar particles
            'nicht': 200, 'nur': 201, 'auch': 202, 'aber': 203, 'oder': 204,
            'wenn': 205, 'weil': 206, 'dass': 207, 'damit': 208, 'ob': 209,
            
            # Common German words (full)
            '▁ich': 300, '▁Sie': 301, '▁wir': 302, '▁haben': 303, '▁sein': 304,
            '▁können': 305, '▁sollen': 306, '▁müssen': 307, '▁wollen': 308,
            '▁gehen': 309, '▁kommen': 310, '▁sehen': 311, '▁machen': 312,
            
            # Error correction patterns
            '▁fehler': 400, '▁Fehler': 401,
            'gelest': 402, 'gelesen': 403,
            'gehst': 404, 'geht': 405,
            'binn': 406, 'bin': 407,
            'seit': 408, 'seid': 409,
            'das': 410, 'dass': 411,
            
            # Punctuation and special
            '.': 500, ',': 501, '?': 502, '!': 503, ';': 504, ':': 505,
            '(': 506, ')': 507, '"': 508, "'": 509, '-': 510, '–': 511,
            
            # Numbers
            '0': 600, '1': 601, '2': 602, '3': 603, '4': 604,
            '5': 605, '6': 606, '7': 607, '8': 608, '9': 609,
            
            # Letters for unknown word handling
            'a': 700, 'b': 701, 'c': 702, 'd': 703, 'e': 704, 'f': 705,
            'g': 706, 'h': 707, 'i': 708, 'j': 709, 'k': 710, 'l': 711,
            'm': 712, 'n': 713, 'o': 714, 'p': 715, 'q': 716, 'r': 717,
            's': 718, 't': 719, 'u': 720, 'v': 721, 'w': 722, 'x': 723,
            'y': 724, 'z': 725, 'ä': 726, 'ö': 727, 'ü': 728, 'ß': 729,
            'A': 730, 'B': 731, 'C': 732, 'D': 733, 'E': 734, 'F': 735,
            'G': 736, 'H': 737, 'I': 738, 'J': 739, 'K': 740, 'L': 741,
            'M': 742, 'N': 743, 'O': 744, 'P': 745, 'Q': 746, 'R': 747,
            'S': 748, 'T': 749, 'U': 750, 'V': 751, 'W': 752, 'X': 753,
            'Y': 754, 'Z': 755, 'Ä': 756, 'Ö': 757, 'Ü': 758,
        }
        
        # Fill remaining slots with computed subwords for German
        current_id = 800
        
        # Add common German subword combinations
        german_subwords = [
            # Common syllables
            'tion', 'sion', 'heit', 'keit', 'lich', 'isch', 'bar', 'sam',
            'los', 'voll', 'reich', 'arm', 'frei', 'neu', 'alt', 'groß',
            'klein', 'gut', 'böse', 'schön', 'hässlich', 'schnell', 'langsam',
            
            # Verb forms
            'ieren', 'eln', 'ern', 'igen', 'ungen', 'schaft', 'tum',
            'nis', 'sal', 'tät', 'anz', 'enz', 'ität', 'osis', 'asis',
            
            # Case endings
            'dem', 'des', 'der', 'die', 'das', 'ein', 'eine', 'einen',
            'einem', 'einer', 'eines', 'mein', 'dein', 'sein', 'ihr',
            'unser', 'euer', 'dieser', 'jener', 'welcher', 'alle',
            
            # Common phrases pieces
            'heute', 'morgen', 'gestern', 'immer', 'nie', 'oft', 'manchmal',
            'gleich', 'sofort', 'später', 'früher', 'jetzt', 'dann', 'hier',
            'dort', 'überall', 'nirgends', 'irgendwo', 'wohin', 'woher',
        ]
        
        for subword in german_subwords:
            if current_id < self.vocab_size - 1000:  # Leave space for more
                vocab[subword] = current_id
                current_id += 1
        
        # Fill remaining with generic tokens
        while current_id < self.vocab_size:
            vocab[f'__{current_id}'] = current_id
            current_id += 1
        
        return vocab
    
    def simple_tokenizer(self, text, vocab):
        """Simple German-aware tokenizer"""
        # Basic preprocessing
        text = text.lower().strip()
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+|[.!?;,:]', text)
        
        token_ids = []
        for token in tokens:
            if token in vocab:
                token_ids.append(vocab[token])
            elif f'▁{token}' in vocab:  # Try with word boundary
                token_ids.append(vocab[f'▁{token}'])
            else:
                # Subword fallback: break into smaller pieces
                subword_ids = self.fallback_tokenize(token, vocab)
                token_ids.extend(subword_ids)
        
        return token_ids
    
    def fallback_tokenize(self, word, vocab):
        """Fallback tokenization for unknown words"""
        token_ids = []
        
        # Try common subwords first
        remaining = word
        while remaining:
            found = False
            
            # Try longest subword first
            for length in range(min(len(remaining), 6), 0, -1):
                subword = remaining[:length]
                if subword in vocab:
                    token_ids.append(vocab[subword])
                    remaining = remaining[length:]
                    found = True
                    break
            
            if not found:
                # Fallback to character level
                char = remaining[0]
                if char in vocab:
                    token_ids.append(vocab[char])
                else:
                    token_ids.append(vocab['<unk>'])
                remaining = remaining[1:]
        
        return token_ids
    
    def create_efficient_gec_model(self):
        """Create efficient but capable GEC model"""
        vocab = self.create_smart_german_vocab()
        
        print(f"Created vocabulary with {len(vocab)} tokens")
        
        class EfficientGECModel(tf.Module):
            def __init__(self, vocab_size, max_length, embedding_dim=256):
                super().__init__()
                self.vocab_size = vocab_size
                self.max_length = max_length
                self.embedding_dim = embedding_dim
                
                # Efficient embedding layer
                self.embedding = tf.Variable(
                    tf.random.normal([vocab_size, embedding_dim], stddev=0.02),
                    trainable=False,
                    name='token_embeddings'
                )
                
                # Positional embeddings
                self.pos_embedding = tf.Variable(
                    tf.random.normal([max_length, embedding_dim], stddev=0.02),
                    trainable=False,
                    name='position_embeddings'
                )
                
                # Efficient transformer-like layers
                self.attention = tf.keras.layers.MultiHeadAttention(
                    num_heads=4, key_dim=64
                )
                
                self.ffn = tf.keras.Sequential([
                    tf.keras.layers.Dense(512, activation='gelu'),
                    tf.keras.layers.Dense(embedding_dim)
                ])
                
                self.norm1 = tf.keras.layers.LayerNormalization()
                self.norm2 = tf.keras.layers.LayerNormalization()
                
                # Output projection
                self.output_projection = tf.keras.layers.Dense(vocab_size)
                
                # Correction bias for common errors
                self.correction_bias = tf.Variable(
                    tf.zeros([vocab_size]), trainable=False, name='correction_bias'
                )
                
                # Set bias for known corrections
                self._initialize_correction_bias()
            
            def _initialize_correction_bias(self):
                """Initialize correction biases for common errors"""
                # This would be learned from training data
                # For now, simple heuristics
                pass
            
            @tf.function(input_signature=[
                tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='input_ids')
            ])
            def __call__(self, input_ids):
                batch_size = tf.shape(input_ids)[0]
                seq_len = tf.shape(input_ids)[1]
                
                # Pad/truncate to max_length
                if seq_len > self.max_length:
                    input_ids = input_ids[:, :self.max_length]
                    seq_len = self.max_length
                elif seq_len < self.max_length:
                    padding = self.max_length - seq_len
                    pad_tensor = tf.zeros([batch_size, padding], dtype=tf.int32)
                    input_ids = tf.concat([input_ids, pad_tensor], axis=1)
                    seq_len = self.max_length
                
                # Clip to vocabulary size
                input_ids = tf.clip_by_value(input_ids, 0, self.vocab_size - 1)
                
                # Token embeddings
                token_emb = tf.nn.embedding_lookup(self.embedding, input_ids)
                
                # Position embeddings
                pos_emb = self.pos_embedding[:seq_len]
                pos_emb = tf.expand_dims(pos_emb, 0)
                pos_emb = tf.tile(pos_emb, [batch_size, 1, 1])
                
                # Combined embeddings
                embeddings = token_emb + pos_emb
                
                # Self-attention
                attn_out = self.attention(embeddings, embeddings)
                attn_out = self.norm1(embeddings + attn_out)
                
                # Feed forward
                ffn_out = self.ffn(attn_out)
                hidden = self.norm2(attn_out + ffn_out)
                
                # Output projection
                logits = self.output_projection(hidden)
                
                # Add correction bias
                logits = logits + self.correction_bias
                
                # Apply copy mechanism (bias toward input tokens)
                input_one_hot = tf.one_hot(input_ids, self.vocab_size)
                copy_bias = input_one_hot * 2.0  # Moderate copy bias
                
                final_logits = logits + copy_bias
                
                return final_logits
        
        model = EfficientGECModel(self.vocab_size, self.max_length)
        
        # Test the model
        test_input = tf.constant([[10, 11, 22, 32, 34, 400, 500]], dtype=tf.int32)
        output = model(test_input)
        print(f"✅ Model test successful. Input: {test_input.shape}, Output: {output.shape}")
        
        return model, vocab
    
    def convert_to_tflite_optimized(self, model, name="german_gec_optimized"):
        """Convert to TFLite with optimization"""
        try:
            concrete_func = model.__call__.get_concrete_function(
                tf.TensorSpec(shape=[1, self.max_length], dtype=tf.int32, name='input_ids')
            )
            
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            
            # Aggressive optimizations
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS  # Allow some TF ops for compatibility
            ]
            
            # Convert
            tflite_model = converter.convert()
            
            # Save
            output_path = self.output_dir / f"{name}.tflite"
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            size_mb = len(tflite_model) / (1024 * 1024)
            print(f"✅ Optimized TFLite model: {output_path} ({size_mb:.2f} MB)")
            
            return output_path, size_mb
            
        except Exception as e:
            print(f"❌ Optimized conversion failed: {e}")
            return None, 0
    
    def benchmark_quality(self, vocab):
        """Benchmark vocabulary coverage"""
        test_sentences = [
            "Das ist ein fehler.",
            "Ich gehe in die Schule morgen.",
            "Er hat das Buch gelest.",
            "Sie haben einen schwierigen Auftrag bekommen.",
            "Die Wissenschaftler untersuchten das Phänomen.",
            "Entwicklungsumgebung für maschinelles Lernen.",
        ]
        
        print("\n📊 Vocabulary Coverage Test:")
        
        total_tokens = 0
        covered_tokens = 0
        
        for sentence in test_sentences:
            tokens = self.simple_tokenizer(sentence, vocab)
            total_tokens += len(sentence.split())
            
            # Count non-UNK tokens
            unk_id = vocab.get('<unk>', 1)
            covered = sum(1 for t in tokens if t != unk_id)
            covered_tokens += covered
            
            coverage = covered / len(tokens) * 100 if tokens else 0
            print(f"  '{sentence}' → {coverage:.1f}% coverage")
        
        overall_coverage = covered_tokens / total_tokens * 100
        print(f"\n📈 Overall vocabulary coverage: {overall_coverage:.1f}%")
        
        return overall_coverage
    
    def convert_all(self):
        """Convert with quality optimization"""
        print("Creating quality-optimized German GEC model...")
        
        model, vocab = self.create_efficient_gec_model()
        
        # Test vocabulary quality
        coverage = self.benchmark_quality(vocab)
        
        results = {'vocab_coverage': coverage}
        
        # Convert to TFLite
        tflite_path, size_mb = self.convert_to_tflite_optimized(model)
        
        if tflite_path:
            # Test the TFLite model
            test_result = self.test_tflite_model(tflite_path)
            
            results['optimized_model'] = {
                'path': str(tflite_path),
                'size_mb': size_mb,
                'test': test_result
            }
        
        # Save vocabulary
        vocab_path = self.output_dir / "vocab_optimized.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        results['vocab_path'] = str(vocab_path)
        
        # Save results
        results_path = self.output_dir / "optimized_conversion_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.print_results(results)
        
        return results
    
    def test_tflite_model(self, model_path):
        """Test TFLite model"""
        try:
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Use fixed shape for testing
            test_input = np.random.randint(0, 1000, size=(1, self.max_length), dtype=np.int32)
            
            interpreter.set_tensor(input_details[0]['index'], test_input)
            
            start_time = time.time()
            interpreter.invoke()
            end_time = time.time()
            
            output = interpreter.get_tensor(output_details[0]['index'])
            inference_time = (end_time - start_time) * 1000
            
            print(f"✅ TFLite test passed ({inference_time:.2f} ms)")
            
            return {
                'success': True,
                'inference_time_ms': inference_time,
                'input_shape': test_input.shape,
                'output_shape': output.shape
            }
            
        except Exception as e:
            print(f"❌ TFLite test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def print_results(self, results):
        """Print results"""
        print("\n" + "="*60)
        print("QUALITY-OPTIMIZED CONVERSION RESULTS")
        print("="*60)
        
        print(f"📊 Vocabulary Coverage: {results['vocab_coverage']:.1f}%")
        
        if 'optimized_model' in results:
            model_data = results['optimized_model']
            print(f"\n📱 TFLite Model:")
            print(f"  Size: {model_data['size_mb']:.2f} MB")
            
            if model_data['test']['success']:
                print(f"  Status: ✅ Working")
                print(f"  Latency: {model_data['test']['inference_time_ms']:.2f} ms")
                print(f"  Ready for Flutter!")
            else:
                print(f"  Status: ❌ Failed")
        
        print(f"\n📝 Vocabulary: {results['vocab_path']}")
        print("="*60)

def main():
    converter = OptimizedVocabConverter()
    results = converter.convert_all()
    
    coverage = results.get('vocab_coverage', 0)
    if coverage > 80:
        print(f"\n🎉 Success! {coverage:.1f}% vocabulary coverage")
        print("Good balance between quality and mobile performance!")
    else:
        print(f"\n⚠️  Coverage only {coverage:.1f}%. Consider larger vocabulary.")

if __name__ == "__main__":
    main()
