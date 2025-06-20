# convert_to_tflite.py
import tensorflow as tf
import numpy as np
import os
import json
from pathlib import Path
import time
from transformers import MT5Tokenizer

class TFLiteConverter:
    def __init__(self, tf_model_path, output_dir="./models_mobile"):
        self.tf_model_path = tf_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load tokenizer for representative dataset
        self.tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
        
    def create_representative_dataset(self, num_samples=100):
        """Create representative dataset for quantization"""
        german_texts = [
            "Das ist ein fehler.",
            "Ich gehe in die Schule morgen.",
            "Er hat das Buch gelest.",
            "Die Katze sitzt auf dem Stuhl.",
            "Wir sind nach Hause gegangen gestern.",
            "Das Auto ist sehr schnell gefahren.",
            "Sie hat ihren Freund angeruft.",
            "Der Hund bellt laut in der Nacht.",
            "Ich habe meine Hausaufgaben gemacht.",
            "Das Wetter ist heute sehr schön."
        ]
        
        def representative_data_gen():
            for i in range(num_samples):
                text = german_texts[i % len(german_texts)]
                # Tokenize like in training
                input_text = f"Korrigiere: {text}"
                inputs = self.tokenizer(
                    input_text,
                    max_length=64,
                    padding='max_length',
                    truncation=True,
                    return_tensors='tf'
                )
                
                # Yield as float32 for quantization
                yield [
                    tf.cast(inputs['input_ids'], tf.float32),
                    tf.cast(inputs['attention_mask'], tf.float32)
                ]
        
        return representative_data_gen
    
    def convert_float32(self):
        """Convert to standard TFLite (FP32)"""
        print("Converting to TFLite FP32...")
        
        converter = tf.lite.TFLiteConverter.from_saved_model(self.tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        output_path = self.output_dir / "german_gec_fp32.tflite"
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        return output_path, len(tflite_model)
    
    def convert_int8_quantized(self):
        """Convert with INT8 quantization"""
        print("Converting to TFLite INT8 quantized...")
        
        converter = tf.lite.TFLiteConverter.from_saved_model(self.tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set representative dataset for full integer quantization
        converter.representative_dataset = self.create_representative_dataset()
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int32
        converter.inference_output_type = tf.int32
        
        try:
            tflite_model = converter.convert()
            
            output_path = self.output_dir / "german_gec_int8.tflite"
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            return output_path, len(tflite_model)
        except Exception as e:
            print(f"INT8 quantization failed: {e}")
            return None, 0
    
    def convert_dynamic_range(self):
        """Convert with dynamic range quantization"""
        print("Converting to TFLite dynamic range quantized...")
        
        converter = tf.lite.TFLiteConverter.from_saved_model(self.tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        output_path = self.output_dir / "german_gec_dynamic.tflite"
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        return output_path, len(tflite_model)
    
    def benchmark_model(self, tflite_path, num_runs=10):
        """Benchmark TFLite model performance"""
        if not tflite_path or not os.path.exists(tflite_path):
            return None
        
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create dummy input
        input_shape = input_details[0]['shape']
        input_data = np.random.randint(0, 1000, size=input_shape, dtype=np.int32)
        
        # Warm up
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'avg_latency_ms': np.mean(times),
            'min_latency_ms': np.min(times),
            'max_latency_ms': np.max(times),
            'std_latency_ms': np.std(times)
        }
    
    def convert_all(self):
        """Convert model to all TFLite variants"""
        results = {}
        
        # FP32 conversion
        fp32_path, fp32_size = self.convert_float32()
        results['fp32'] = {
            'path': str(fp32_path),
            'size_bytes': fp32_size,
            'size_mb': fp32_size / (1024 * 1024),
            'benchmark': self.benchmark_model(fp32_path)
        }
        
        # Dynamic range quantization
        dynamic_path, dynamic_size = self.convert_dynamic_range()
        results['dynamic'] = {
            'path': str(dynamic_path),
            'size_bytes': dynamic_size,
            'size_mb': dynamic_size / (1024 * 1024),
            'benchmark': self.benchmark_model(dynamic_path)
        }
        
        # INT8 quantization (may fail for complex models)
        int8_path, int8_size = self.convert_int8_quantized()
        if int8_path:
            results['int8'] = {
                'path': str(int8_path),
                'size_bytes': int8_size,
                'size_mb': int8_size / (1024 * 1024),
                'benchmark': self.benchmark_model(int8_path)
            }
        
        # Save results
        results_path = self.output_dir / "conversion_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.print_summary(results)
        return results
    
    def print_summary(self, results):
        """Print conversion summary"""
        print("\n" + "="*60)
        print("TFLite Conversion Summary")
        print("="*60)
        
        for model_type, data in results.items():
            print(f"\n{model_type.upper()} Model:")
            print(f"  Size: {data['size_mb']:.2f} MB")
            
            if data['benchmark']:
                bench = data['benchmark']
                print(f"  Avg Latency: {bench['avg_latency_ms']:.2f} ms")
                print(f"  Min Latency: {bench['min_latency_ms']:.2f} ms")
                print(f"  Max Latency: {bench['max_latency_ms']:.2f} ms")
        
        print("\n" + "="*60)

def main():
    # Check if TF SavedModel exists
    tf_model_path = "./models/german_gec_tf"
    
    if not os.path.exists(tf_model_path):
        print(f"Error: TensorFlow SavedModel not found at {tf_model_path}")
        print("Please run ONNX conversion first!")
        return
    
    # Convert to TFLite
    converter = TFLiteConverter(tf_model_path)
    results = converter.convert_all()
    
    print("\nTFLite models saved to ./models_mobile/")
    print("Use 'german_gec_dynamic.tflite' for best size/performance balance")

if __name__ == "__main__":
    main()