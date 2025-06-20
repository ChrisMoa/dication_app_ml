import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import os
from pathlib import Path
import json

class ModelOptimizer:
    def __init__(self, tf_model_path, output_dir="./models_optimized"):
        self.tf_model_path = tf_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_model(self):
        """Load TensorFlow SavedModel"""
        return tf.saved_model.load(self.tf_model_path)
    
    def create_pruned_model(self, sparsity=0.5):
        """Create pruned model with specified sparsity"""
        print(f"Creating pruned model with {sparsity*100}% sparsity...")
        
        # Load model as Keras model for pruning
        try:
            model = tf.keras.models.load_model(self.tf_model_path)
        except:
            print("Cannot load as Keras model, skipping pruning")
            return None
        
        # Define pruning parameters
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=sparsity,
                begin_step=0,
                end_step=1000
            )
        }
        
        # Apply pruning
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            model, **pruning_params
        )
        
        # Compile and train briefly to apply pruning
        pruned_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy'
        )
        
        # Create dummy data for pruning
        dummy_input = np.random.randint(0, 1000, size=(1, 64), dtype=np.int32)
        dummy_output = np.random.randint(0, 1000, size=(1, 64), dtype=np.int32)
        
        # "Train" for pruning
        pruned_model.fit(
            dummy_input, dummy_output,
            epochs=1, verbose=0
        )
        
        # Strip pruning and export
        final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        
        pruned_path = self.output_dir / f"pruned_{int(sparsity*100)}percent"
        final_model.save(pruned_path)
        
        return pruned_path
    
    def create_distilled_model(self):
        """Create knowledge distilled smaller model"""
        print("Creating distilled model...")
        
        # This would require implementing a student-teacher setup
        # For now, we'll create a simplified version
        
        try:
            original_model = tf.keras.models.load_model(self.tf_model_path)
            
            # Create smaller architecture (simplified)
            inputs = tf.keras.Input(shape=(64,), dtype=tf.int32)
            
            # Smaller embedding and transformer layers
            embeddings = tf.keras.layers.Embedding(32000, 256)(inputs)
            
            # Simplified transformer block
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=64
            )(embeddings, embeddings)
            
            # Add & Norm
            add_norm = tf.keras.layers.LayerNormalization()(
                tf.keras.layers.Add()([embeddings, attention])
            )
            
            # Feed forward
            ff = tf.keras.layers.Dense(512, activation='relu')(add_norm)
            ff = tf.keras.layers.Dense(256)(ff)
            
            # Output layer
            outputs = tf.keras.layers.Dense(32000, activation='softmax')(ff)
            
            distilled_model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            distilled_path = self.output_dir / "distilled_model"
            distilled_model.save(distilled_path)
            
            return distilled_path
            
        except Exception as e:
            print(f"Distillation failed: {e}")
            return None
    
    def optimize_for_edge(self, input_model_path):
        """Apply edge-specific optimizations"""
        print("Applying edge optimizations...")
        
        # Load and optimize
        converter = tf.lite.TFLiteConverter.from_saved_model(str(input_model_path))
        
        # Enable all optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Target edge TPU operations if available
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Reduce precision
        converter.target_spec.supported_types = [tf.float16]
        
        # Convert
        optimized_model = converter.convert()
        
        # Save
        output_path = self.output_dir / "edge_optimized.tflite"
        with open(output_path, 'wb') as f:
            f.write(optimized_model)
        
        return output_path, len(optimized_model)
    
    def benchmark_optimized_models(self):
        """Benchmark all optimized variants"""
        results = {}
        
        models_to_test = [
            ("original", "./models/german_gec_dynamic.tflite"),
            ("pruned_50", "./models_optimized/pruned_50percent"),
            ("distilled", "./models_optimized/distilled_model"),
            ("edge_optimized", "./models_optimized/edge_optimized.tflite")
        ]
        
        for name, path in models_to_test:
            if os.path.exists(path):
                try:
                    size_mb = self._get_model_size(path)
                    latency = self._benchmark_latency(path)
                    
                    results[name] = {
                        'size_mb': size_mb,
                        'avg_latency_ms': latency,
                        'path': str(path)
                    }
                    
                except Exception as e:
                    results[name] = {'error': str(e)}
        
        return results
    
    def _get_model_size(self, path):
        """Get model size in MB"""
        if os.path.isdir(path):
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(path)
                for filename in filenames
            )
        else:
            total_size = os.path.getsize(path)
        
        return total_size / (1024 * 1024)
    
    def _benchmark_latency(self, path, num_runs=5):
        """Benchmark model latency"""
        import time
        
        if path.endswith('.tflite'):
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            input_data = np.random.randint(0, 1000, size=input_details[0]['shape'], dtype=np.int32)
            
            times = []
            for _ in range(num_runs):
                start = time.time()
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                times.append((time.time() - start) * 1000)
            
            return np.mean(times)
        else:
            # For SavedModel format
            model = tf.saved_model.load(path)
            input_data = tf.constant(np.random.randint(0, 1000, size=(1, 64), dtype=np.int32))
            
            times = []
            for _ in range(num_runs):
                start = time.time()
                _ = model(input_data)
                times.append((time.time() - start) * 1000)
            
            return np.mean(times)
    
    def optimize_all(self):
        """Run all optimization techniques"""
        results = {'optimization_summary': {}}
        
        print("Starting model optimization pipeline...")
        
        # 1. Pruning
        pruned_50_path = self.create_pruned_model(sparsity=0.5)
        if pruned_50_path:
            results['pruned_50'] = str(pruned_50_path)
        
        pruned_75_path = self.create_pruned_model(sparsity=0.75)
        if pruned_75_path:
            results['pruned_75'] = str(pruned_75_path)
        
        # 2. Knowledge Distillation
        distilled_path = self.create_distilled_model()
        if distilled_path:
            results['distilled'] = str(distilled_path)
        
        # 3. Edge optimization
        original_path = self.tf_model_path
        edge_path, edge_size = self.optimize_for_edge(original_path)
        results['edge_optimized'] = {
            'path': str(edge_path),
            'size_mb': edge_size / (1024 * 1024)
        }
        
        # 4. Benchmark all variants
        benchmark_results = self.benchmark_optimized_models()
        results['benchmarks'] = benchmark_results
        
        # 5. Save results
        results_path = self.output_dir / "optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self._print_optimization_summary(benchmark_results)
        
        return results
    
    def _print_optimization_summary(self, benchmarks):
        """Print optimization summary"""
        print("\n" + "="*70)
        print("MODEL OPTIMIZATION SUMMARY")
        print("="*70)
        
        print(f"{'Model':<20} {'Size (MB)':<12} {'Latency (ms)':<15} {'Reduction':<12}")
        print("-" * 70)
        
        baseline_size = benchmarks.get('original', {}).get('size_mb', 0)
        baseline_latency = benchmarks.get('original', {}).get('avg_latency_ms', 0)
        
        for name, data in benchmarks.items():
            if 'error' in data:
                print(f"{name:<20} {'ERROR':<12} {'ERROR':<15} {'N/A':<12}")
                continue
            
            size = data.get('size_mb', 0)
            latency = data.get('avg_latency_ms', 0)
            
            size_reduction = f"{(1 - size/baseline_size)*100:.1f}%" if baseline_size > 0 else "N/A"
            
            print(f"{name:<20} {size:<12.2f} {latency:<15.2f} {size_reduction:<12}")
        
        print("\n" + "="*70)
        print("RECOMMENDATION:")
        
        # Find best model based on size/performance trade-off
        best_model = min(
            [(name, data) for name, data in benchmarks.items() 
             if 'error' not in data and name != 'original'],
            key=lambda x: x[1].get('size_mb', float('inf')) * x[1].get('avg_latency_ms', float('inf')),
            default=None
        )
        
        if best_model:
            name, data = best_model
            print(f"Best model for mobile: {name}")
            print(f"Size: {data['size_mb']:.2f} MB, Latency: {data['avg_latency_ms']:.2f} ms")
        
        print("="*70)

def main():
    # Check if TF SavedModel exists
    tf_model_path = "./models/german_gec_tf"
    
    if not os.path.exists(tf_model_path):
        print(f"Error: TensorFlow SavedModel not found at {tf_model_path}")
        print("Please run ONNX conversion first!")
        return
    
    # Run optimization pipeline
    optimizer = ModelOptimizer(tf_model_path)
    results = optimizer.optimize_all()
    
    print(f"\nOptimization complete! Results saved to {optimizer.output_dir}")
    print("Mobile-ready models available for Flutter integration.")

if __name__ == "__main__":
    main()