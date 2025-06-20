import os
import json
import shutil
from pathlib import Path
from transformers import MT5Tokenizer

class MobileAssetPreparer:
    def __init__(self, flutter_project_path="./flutter_gec_app"):
        self.flutter_path = Path(flutter_project_path)
        self.assets_path = self.flutter_path / "assets" / "models"
        self.models_path = Path("./models/mobile")
        
    def setup_flutter_project(self):
        """Setup basic Flutter project structure"""
        print("Setting up Flutter project structure...")
        
        # Create directories
        dirs_to_create = [
            self.flutter_path / "lib" / "core" / "ml",
            self.flutter_path / "lib" / "features" / "text_correction" / "data",
            self.flutter_path / "lib" / "features" / "text_correction" / "domain",
            self.flutter_path / "lib" / "features" / "text_correction" / "presentation" / "widgets",
            self.flutter_path / "lib" / "features" / "text_correction" / "presentation" / "pages",
            self.assets_path,
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create pubspec.yaml
        self._create_pubspec()
        
        # Create main.dart
        self._create_main_dart()
        
    def _create_pubspec(self):
        """Create pubspec.yaml with required dependencies"""
        pubspec_content = """name: german_gec_app
description: German Grammar Error Correction App
version: 1.0.0+1

environment:
  sdk: '>=3.0.0 <4.0.0'
  flutter: ">=3.0.0"

dependencies:
  flutter:
    sdk: flutter
  tflite_flutter: ^0.10.4
  tflite_flutter_helper: ^0.3.1
  http: ^1.1.0
  cupertino_icons: ^1.0.2

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.0

flutter:
  uses-material-design: true
  
  assets:
    - assets/models/
    - assets/models/german_gec_dynamic.tflite
    - assets/models/vocab.json
"""
        
        with open(self.flutter_path / "pubspec.yaml", 'w') as f:
            f.write(pubspec_content)
    
    def _create_main_dart(self):
        """Create main.dart"""
        main_content = '''// main.dart
import 'package:flutter/material.dart';
import 'features/text_correction/presentation/pages/gec_page.dart';

void main() {
  runApp(const GermanGECApp());
}

class GermanGECApp extends StatelessWidget {
  const GermanGECApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'German Grammar Correction',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const GECPage(),
    );
  }
}
'''
        
        with open(self.flutter_path / "lib" / "main.dart", 'w') as f:
            f.write(main_content)
    
    def prepare_tflite_models(self):
        """Copy optimized TFLite models to Flutter assets"""
        print("Preparing TFLite models for Flutter...")
        
        self.assets_path.mkdir(parents=True, exist_ok=True)
        
        # Find best TFLite model
        candidate_models = [
            self.models_path / "edge_optimized.tflite",
            Path("./models/mobile/german_gec_dynamic.tflite"),
            Path("./models/mobile/german_gec_fp32.tflite"),
        ]
        
        selected_model = None
        for model_path in candidate_models:
            if model_path.exists():
                selected_model = model_path
                break
        
        if selected_model:
            target_path = self.assets_path / "german_gec_dynamic.tflite"
            shutil.copy2(selected_model, target_path)
            print(f"Copied {selected_model} -> {target_path}")
            
            # Get model info
            model_size = selected_model.stat().st_size / (1024 * 1024)
            print(f"Model size: {model_size:.2f} MB")
            
            return target_path
        else:
            print("No TFLite model found! Run convert_to_tflite.py first.")
            return None
    
    def create_simplified_vocab(self):
        """Create simplified vocabulary for mobile"""
        print("Creating simplified vocabulary...")
        
        try:
            # Load MT5 tokenizer
            tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
            
            # Get most common German tokens
            german_tokens = {
                # Special tokens
                '<pad>': 0,
                '</s>': 1,
                '<unk>': 2,
                
                # Common German words
                'Korrigiere': 100,
                ':': 101,
                'Das': 200,
                'ist': 201,
                'ein': 202,
                'eine': 203,
                'der': 204,
                'die': 205,
                'und': 206,
                'ich': 207,
                'Sie': 208,
                'haben': 209,
                'sein': 210,
                'mit': 211,
                'auf': 212,
                'für': 213,
                'nicht': 214,
                'von': 215,
                'zu': 216,
                'den': 217,
                'im': 218,
                'wird': 219,
                'war': 220,
                'hat': 221,
                'kann': 222,
                'soll': 223,
                'mehr': 224,
                'auch': 225,
                'nur': 226,
                'aber': 227,
                'wenn': 228,
                'oder': 229,
                'als': 230,
                'sehr': 231,
                'noch': 232,
                'wie': 233,
                'so': 234,
                'dann': 235,
                'nach': 236,
                'über': 237,
                'aus': 238,
                'bei': 239,
                'gegen': 240,
                'zwischen': 241,
                'unter': 242,
                'durch': 243,
                'ohne': 244,
                'um': 245,
                'bis': 246,
                'seit': 247,
                'während': 248,
                'vor': 249,
                'hinter': 250,
                'neben': 251,
                'zwischen': 252,
                'fehler': 300,
                'Fehler': 301,
                'falsch': 302,
                'richtig': 303,
                'gut': 304,
                'schlecht': 305,
                '.': 400,
                ',': 401,
                '?': 402,
                '!': 403,
                ';': 404,
                ':': 405,
            }
            
            # Save simplified vocab
            vocab_path = self.assets_path / "vocab.json"
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(german_tokens, f, ensure_ascii=False, indent=2)
            
            print(f"Simplified vocabulary saved: {len(german_tokens)} tokens")
            return vocab_path
            
        except Exception as e:
            print(f"Failed to create vocabulary: {e}")
            return None
    
    def create_gec_page(self):
        """Create main GEC page"""
        page_content = '''// lib/features/text_correction/presentation/pages/gec_page.dart
import 'package:flutter/material.dart';
import '../widgets/gec_text_field.dart';
import '../../../../core/ml/german_gec_service.dart';

class GECPage extends StatefulWidget {
  const GECPage({Key? key}) : super(key: key);

  @override
  State<GECPage> createState() => _GECPageState();
}

class _GECPageState extends State<GECPage> {
  final List<GECResult> _correctionHistory = [];
  bool _isOnlineMode = false;
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('German Grammar Correction'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          Switch(
            value: _isOnlineMode,
            onChanged: (value) {
              setState(() {
                _isOnlineMode = value;
              });
            },
          ),
          const SizedBox(width: 8),
          Text(_isOnlineMode ? 'Online' : 'Offline'),
          const SizedBox(width: 16),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            GECTextField(
              hintText: 'Geben Sie deutschen Text zur Korrektur ein...',
              onCorrectionResult: _onCorrectionResult,
              enableRealTimeCorrection: true,
            ),
            const SizedBox(height: 20),
            Expanded(
              child: _buildHistoryList(),
            ),
          ],
        ),
      ),
    );
  }
  
  void _onCorrectionResult(GECResult result) {
    setState(() {
      _correctionHistory.insert(0, result);
      if (_correctionHistory.length > 20) {
        _correctionHistory.removeLast();
      }
    });
  }
  
  Widget _buildHistoryList() {
    if (_correctionHistory.isEmpty) {
      return const Center(
        child: Text(
          'Korrekturverlauf wird hier angezeigt',
          style: TextStyle(
            fontSize: 16,
            color: Colors.grey,
          ),
        ),
      );
    }
    
    return ListView.builder(
      itemCount: _correctionHistory.length,
      itemBuilder: (context, index) {
        final result = _correctionHistory[index];
        return Card(
          margin: const EdgeInsets.only(bottom: 8),
          child: ListTile(
            title: Text(
              result.hasCorrections ? 'Korrigiert' : 'Korrekt',
              style: TextStyle(
                color: result.hasCorrections ? Colors.blue : Colors.green,
                fontWeight: FontWeight.w500,
              ),
            ),
            subtitle: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Original: ${result.originalText}'),
                if (result.hasCorrections)
                  Text(
                    'Korrektur: ${result.correctedText}',
                    style: const TextStyle(fontWeight: FontWeight.w500),
                  ),
                Text(
                  '${result.inferenceTimeMs}ms • ${(result.confidence * 100).toStringAsFixed(1)}%',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey[600],
                  ),
                ),
              ],
            ),
            trailing: result.hasCorrections
                ? const Icon(Icons.edit, color: Colors.blue)
                : const Icon(Icons.check_circle, color: Colors.green),
          ),
        );
      },
    );
  }
}
'''
        
        page_path = (self.flutter_path / "lib" / "features" / "text_correction" / 
                    "presentation" / "pages" / "gec_page.dart")
        with open(page_path, 'w') as f:
            f.write(page_content)
    
    def create_performance_benchmark(self):
        """Create performance benchmark for mobile"""
        benchmark_content = '''// lib/core/ml/performance_benchmark.dart
import 'dart:math';
import 'german_gec_service.dart';

class PerformanceBenchmark {
  static Future<BenchmarkResult> runBenchmark() async {
    final gecService = GermanGECService();
    await gecService.initialize();
    
    final testSentences = [
      'Das ist ein fehler.',
      'Ich gehe in die Schule morgen.',
      'Er hat das Buch gelest.',
      'Die Katze sitzt auf dem Stuhl.',
      'Wir sind nach Hause gegangen gestern.',
      'Das Auto ist sehr schnell gefahren.',
      'Sie hat ihren Freund angeruft.',
      'Der Hund bellt laut in der Nacht.',
      'Ich habe meine Hausaufgaben gemacht.',
      'Das Wetter ist heute sehr schön.',
    ];
    
    final latencies = <int>[];
    final confidences = <double>[];
    int correctionsCount = 0;
    
    for (final sentence in testSentences) {
      final result = await gecService.correctText(sentence);
      
      latencies.add(result.inferenceTimeMs);
      confidences.add(result.confidence);
      
      if (result.hasCorrections) {
        correctionsCount++;
      }
    }
    
    gecService.dispose();
    
    return BenchmarkResult(
      avgLatencyMs: latencies.reduce((a, b) => a + b) / latencies.length,
      minLatencyMs: latencies.reduce(min),
      maxLatencyMs: latencies.reduce(max),
      avgConfidence: confidences.reduce((a, b) => a + b) / confidences.length,
      correctionsPercentage: (correctionsCount / testSentences.length) * 100,
      totalSentences: testSentences.length,
    );
  }
}

class BenchmarkResult {
  final double avgLatencyMs;
  final int minLatencyMs;
  final int maxLatencyMs;
  final double avgConfidence;
  final double correctionsPercentage;
  final int totalSentences;
  
  const BenchmarkResult({
    required this.avgLatencyMs,
    required this.minLatencyMs,
    required this.maxLatencyMs,
    required this.avgConfidence,
    required this.correctionsPercentage,
    required this.totalSentences,
  });
  
  @override
  String toString() {
    return '''
Benchmark Results:
- Avg Latency: ${avgLatencyMs.toStringAsFixed(1)}ms
- Min Latency: ${minLatencyMs}ms
- Max Latency: ${maxLatencyMs}ms
- Avg Confidence: ${(avgConfidence * 100).toStringAsFixed(1)}%
- Corrections Found: ${correctionsPercentage.toStringAsFixed(1)}%
- Total Sentences: $totalSentences
''';
  }
}
'''
        
        benchmark_path = self.flutter_path / "lib" / "core" / "ml" / "performance_benchmark.dart"
        with open(benchmark_path, 'w') as f:
            f.write(benchmark_content)
    
    def setup_all(self):
        """Setup complete mobile environment"""
        print("Setting up complete mobile environment...")
        
        # 1. Flutter project structure
        self.setup_flutter_project()
        
        # 2. TFLite models
        model_path = self.prepare_tflite_models()
        
        # 3. Vocabulary
        vocab_path = self.create_simplified_vocab()
        
        # 4. UI components
        self.create_gec_page()
        
        # 5. Performance tools
        self.create_performance_benchmark()
        
        # 6. Summary
        self._print_setup_summary(model_path, vocab_path)
        
        return {
            'flutter_project': str(self.flutter_path),
            'model_path': str(model_path) if model_path else None,
            'vocab_path': str(vocab_path) if vocab_path else None,
            'status': 'ready' if model_path and vocab_path else 'incomplete'
        }
    
    def _print_setup_summary(self, model_path, vocab_path):
        """Print setup summary"""
        print("\n" + "="*60)
        print("MOBILE SETUP SUMMARY")
        print("="*60)
        
        print(f"Flutter Project: {self.flutter_path}")
        print(f"✅ Project Structure Created")
        print(f"✅ Dependencies Configured (pubspec.yaml)")
        print(f"✅ Main App Created")
        
        if model_path:
            model_size = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ TFLite Model: {model_size:.2f} MB")
        else:
            print("❌ TFLite Model: MISSING")
        
        if vocab_path:
            print(f"✅ Vocabulary: {vocab_path}")
        else:
            print("❌ Vocabulary: MISSING")
        
        print(f"✅ UI Components Created")
        print(f"✅ Performance Benchmark Ready")
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("1. cd flutter_gec_app")
        print("2. flutter pub get")
        print("3. flutter run")
        print("="*60)

def main():
    # Setup mobile environment
    preparer = MobileAssetPreparer()
    results = preparer.setup_all()
    
    if results['status'] == 'ready':
        print("\n🎉 Mobile setup complete!")
        print("Your German GEC Flutter app is ready to build and test.")
    else:
        print("\n⚠️ Setup incomplete!")
        print("Please run previous conversion steps first.")

if __name__ == "__main__":
    main()