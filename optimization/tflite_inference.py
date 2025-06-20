import tensorflow as tf
import numpy as np
from transformers import MT5Tokenizer
import time

class TFLiteGECInference:
    def __init__(self, model_path, tokenizer_name="google/mt5-small"):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.tokenizer = MT5Tokenizer.from_pretrained(tokenizer_name)
        
        print(f"Loaded TFLite model: {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def correct_text(self, text, max_length=64):
        """Correct German text using TFLite model"""
        start_time = time.time()
        
        # Prepare input
        input_text = f"Korrigiere: {text}"
        inputs = self.tokenizer(
            input_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        # Run inference
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            inputs['input_ids'].astype(np.int32)
        )
        
        if len(self.input_details) > 1:  # If attention_mask is needed
            self.interpreter.set_tensor(
                self.input_details[1]['index'],
                inputs['attention_mask'].astype(np.int32)
            )
        
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Decode
        predicted_ids = np.argmax(output_data, axis=-1)
        corrected_text = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'original': text,
            'corrected': corrected_text,
            'inference_time_ms': inference_time
        }
    
    def batch_correct(self, texts, max_length=64):
        """Correct multiple texts"""
        results = []
        total_time = 0
        
        for text in texts:
            result = self.correct_text(text, max_length)
            results.append(result)
            total_time += result['inference_time_ms']
        
        return {
            'results': results,
            'total_time_ms': total_time,
            'avg_time_ms': total_time / len(texts) if texts else 0
        }

def benchmark_all_models():
    """Benchmark all available TFLite models"""
    test_sentences = [
        "Das ist ein fehler.",
        "Ich gehe in die Schule morgen.",
        "Er hat das Buch gelest.",
        "Die Katze sitzt auf dem Stuhl.",
        "Wir sind nach Hause gegangen gestern."
    ]
    
    models = [
        "./models_mobile/german_gec_fp32.tflite",
        "./models_mobile/german_gec_dynamic.tflite",
        "./models_mobile/german_gec_int8.tflite"
    ]
    
    results = {}
    
    for model_path in models:
        try:
            model_name = model_path.split('/')[-1].replace('.tflite', '')
            print(f"\nTesting {model_name}...")
            
            corrector = TFLiteGECInference(model_path)
            batch_result = corrector.batch_correct(test_sentences)
            
            results[model_name] = {
                'avg_latency_ms': batch_result['avg_time_ms'],
                'total_time_ms': batch_result['total_time_ms'],
                'corrections': [(r['original'], r['corrected']) for r in batch_result['results']]
            }
            
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            results[model_path] = {'error': str(e)}
    
    return results

def main():
    # Test individual model
    model_path = "./models_mobile/german_gec_dynamic.tflite"
    
    try:
        corrector = TFLiteGECInference(model_path)
        
        test_text = "Das ist ein fehler."
        result = corrector.correct_text(test_text)
        
        print(f"\nOriginal: {result['original']}")
        print(f"Corrected: {result['corrected']}")
        print(f"Time: {result['inference_time_ms']:.2f} ms")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run convert_to_tflite.py first!")

if __name__ == "__main__":
    main()