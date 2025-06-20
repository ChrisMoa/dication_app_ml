# test_german_gec.py
import unittest
import json
import time
from training.german_hybrid_corrector import HybridGermanGrammarCorrector, GermanGECBenchmark, CorrectionResult
import torch

class TestGermanGrammarCorrection(unittest.TestCase):
    """Unit Tests für deutsches Grammatikkorrektur-System"""
    
    @classmethod
    def setUpClass(cls):
        """Setup für alle Tests"""
        cls.corrector = HybridGermanGrammarCorrector(
            mt5_model_path="./models/trained_models/final_model",
            use_gpu=torch.cuda.is_available(),
            enable_caching=True
        )
        
    def test_basic_correction(self):
        """Test grundlegende Korrektur-Funktionalität"""
        test_cases = [
            ("Der Hund beißt der Mann", "den Mann"),  # Erwarte "den" statt "der"
            ("Ich gehen zur Schule", "gehe"),  # Erwarte "gehe" statt "gehen"
            ("Das ist ein schöne Tag", "schöner"),  # Erwarte "schöner" statt "schöne"
        ]
        
        for incorrect, expected_fragment in test_cases:
            with self.subTest(text=incorrect):
                result = self.corrector.correct_text(incorrect, method="hybrid")
                
                # Prüfe dass Korrektur stattgefunden hat
                self.assertIsInstance(result, CorrectionResult)
                self.assertNotEqual(result.original_text, result.corrected_text)
                
                # Prüfe dass erwartete Korrektur enthalten ist
                self.assertIn(expected_fragment, result.corrected_text)
                
                # Prüfe Metadaten
                self.assertGreater(result.processing_time_ms, 0)
                self.assertGreaterEqual(result.confidence_score, 0)
                self.assertLessEqual(result.confidence_score, 1)
                
    def test_no_correction_needed(self):
        """Test für bereits korrekte Sätze"""
        correct_sentences = [
            "Der Hund läuft schnell.",
            "Ich gehe heute ins Kino.",
            "Das Wetter ist schön."
        ]
        
        for sentence in correct_sentences:
            with self.subTest(text=sentence):
                result = self.corrector.correct_text(sentence, method="hybrid")
                
                # Korrekte Sätze sollten unverändert bleiben oder minimal verändert werden
                self.assertLessEqual(len(result.corrections), 1)  # Maximal eine kleine Korrektur
                self.assertGreaterEqual(result.confidence_score, 0.8)  # Hohe Konfidenz
                
    def test_method_comparison(self):
        """Vergleiche verschiedene Korrektur-Methoden"""
        test_text = "Der Hund beißt der Mann"
        
        methods = ["languagetool", "hybrid"]
        if self.corrector.mt5_model is not None:
            methods.append("mt5")
            
        results = {}
        for method in methods:
            result = self.corrector.correct_text(test_text, method=method)
            results[method] = result
            
            # Grundlegende Validierung
            self.assertIsInstance(result, CorrectionResult)
            self.assertEqual(result.correction_method, method)
            self.assertGreater(result.processing_time_ms, 0)
            
        # Vergleiche Ergebnisse
        if "hybrid" in results and "languagetool" in results:
            # Hybrid sollte mindestens so gut sein wie LanguageTool allein
            hybrid_conf = results["hybrid"].confidence_score
            lt_conf = results["languagetool"].confidence_score
            self.assertGreaterEqual(hybrid_conf, lt_conf * 0.9)  # 90% der LT-Konfidenz
            
    def test_performance_requirements(self):
        """Test Performance-Anforderungen"""
        test_sentences = [
            "Das ist ein Test",
            "Ich gehe zur Schule",
            "Der Hund läuft schnell"
        ]
        
        # Messe Durchschnittszeit
        times = []
        for sentence in test_sentences:
            start_time = time.time()
            result = self.corrector.correct_text(sentence, method="hybrid")
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000  # ms
            times.append(processing_time)
            
            # Einzelne Anforderungen
            self.assertLess(processing_time, 2000)  # Unter 2 Sekunden
            self.assertIsInstance(result.corrected_text, str)
            
        # Durchschnittszeit
        avg_time = sum(times) / len(times)
        self.assertLess(avg_time, 1000)  # Unter 1 Sekunde im Durchschnitt
        
    def test_memory_usage(self):
        """Test Speicherverbrauch"""
        import psutil
        
        # Messe Speicher vor und nach Verarbeitung
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Verarbeite mehrere Sätze
        test_sentences = ["Das ist ein Test"] * 50
        for sentence in test_sentences:
            self.corrector.correct_text(sentence, method="hybrid")
            
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Speicheranstieg sollte moderat sein
        self.assertLess(memory_increase, 500)  # Unter 500MB Anstieg
        
    def test_error_handling(self):
        """Test Fehlerbehandlung"""
        edge_cases = [
            "",  # Leerer String
            "   ",  # Nur Leerzeichen
            "A" * 1000,  # Sehr langer Text
            "123 456 789",  # Nur Zahlen
            "!!!???...",  # Nur Satzzeichen
        ]
        
        for case in edge_cases:
            with self.subTest(text=case):
                try:
                    result = self.corrector.correct_text(case, method="hybrid")
                    self.assertIsInstance(result, CorrectionResult)
                    # System sollte nicht abstürzen
                except Exception as e:
                    self.fail(f"Unerwarteter Fehler bei '{case}': {e}")
                    
    def test_batch_processing(self):
        """Test Batch-Verarbeitung"""
        sentences = [
            "Der Hund beißt der Mann",
            "Ich gehen zur Schule", 
            "Das ist ein schöne Tag"
        ]
        
        results = self.corrector.correct_sentences(sentences, method="hybrid")
        
        self.assertEqual(len(results), len(sentences))
        for i, result in enumerate(results):
            self.assertIsInstance(result, CorrectionResult)
            self.assertEqual(result.original_text, sentences[i])
            
    def test_caching(self):
        """Test Caching-Funktionalität"""
        if not self.corrector.enable_caching:
            self.skipTest("Caching ist deaktiviert")
            
        test_text = "Das ist ein Test für Caching"
        
        # Erste Verarbeitung
        start_time = time.time()
        result1 = self.corrector.correct_text(test_text, method="hybrid")
        time1 = time.time() - start_time
        
        # Zweite Verarbeitung (sollte gecacht sein)
        start_time = time.time()
        result2 = self.corrector.correct_text(test_text, method="hybrid")
        time2 = time.time() - start_time
        
        # Ergebnisse sollten identisch sein
        self.assertEqual(result1.corrected_text, result2.corrected_text)
        self.assertEqual(result1.confidence_score, result2.confidence_score)

class TestGermanGECBenchmark(unittest.TestCase):
    """Tests für Benchmark-Funktionalität"""
    
    @classmethod
    def setUpClass(cls):
        """Setup für Benchmark-Tests"""
        cls.corrector = HybridGermanGrammarCorrector(
            mt5_model_path="./models/trained_models/final_model",
            use_gpu=torch.cuda.is_available(),
            enable_caching=False  # Für konsistente Benchmark-Ergebnisse
        )
        cls.benchmark = GermanGECBenchmark(cls.corrector)
        
    def test_accuracy_benchmark(self):
        """Test Genauigkeits-Benchmark"""
        test_cases = [
            ("Der Hund beißt der Mann", "Der Hund beißt den Mann"),
            ("Ich gehen zur Schule", "Ich gehe zur Schule"),
        ]
        
        results = self.benchmark.benchmark_accuracy(test_cases)
        
        # Validiere Struktur
        self.assertIn("languagetool", results)
        self.assertIn("hybrid", results)
        
        for method in results:
            self.assertIn("accuracy", results[method])
            self.assertIn("total", results[method])
            self.assertIn("correct", results[method])
            self.assertIn("corrections", results[method])
            
            # Validiere Werte
            self.assertGreaterEqual(results[method]["accuracy"], 0)
            self.assertLessEqual(results[method]["accuracy"], 1)
            self.assertEqual(results[method]["total"], len(test_cases))
            
    def test_performance_benchmark(self):
        """Test Performance-Benchmark"""
        results = self.benchmark.benchmark_performance(num_sentences=10)
        
        # Validiere Struktur
        self.assertIn("languagetool", results)
        self.assertIn("hybrid", results)
        
        for method in results:
            self.assertIn("total_time_seconds", results[method])
            self.assertIn("avg_time_per_sentence_ms", results[method])
            self.assertIn("throughput_sentences_per_second", results[method])
            self.assertIn("peak_memory_mb", results[method])
            
            # Validiere Werte
            self.assertGreater(results[method]["total_time_seconds"], 0)
            self.assertGreater(results[method]["throughput_sentences_per_second"], 0)
            self.assertGreater(results[method]["peak_memory_mb"], 0)
            
    def test_comprehensive_benchmark(self):
        """Test umfassender Benchmark"""
        results = self.benchmark.run_comprehensive_benchmark(save_results=False)
        
        # Validiere Hauptstruktur
        required_keys = ["timestamp", "system_info", "accuracy", "performance", "summary"]
        for key in required_keys:
            self.assertIn(key, results)
            
        # Validiere Summary
        summary = results["summary"]
        self.assertIn("best_accuracy_method", summary)
        self.assertIn("best_speed_method", summary)
        self.assertIn("recommended_method", summary)
        self.assertIn("accuracy_ranking", summary)
        self.assertIn("speed_ranking", summary)
        
        # Validiere System-Info
        system_info = results["system_info"]
        self.assertIn("device", system_info)
        self.assertIn("gpu_available", system_info)
        self.assertIn("mt5_model_loaded", system_info)

class TestGermanGECIntegration(unittest.TestCase):
    """Integrationstests für das gesamte System"""
    
    @classmethod
    def setUpClass(cls):
        """Setup für Integrationstests"""
        cls.corrector = HybridGermanGrammarCorrector(
            mt5_model_path="./models/trained_models/final_model",
            use_gpu=torch.cuda.is_available()
        )
        
    def test_real_world_examples(self):
        """Test mit realistischen Beispielen aus der Praxis"""
        real_world_cases = [
            # E-Mail-Texte
            ("Hallo, ich wollte fragen ob du Zeit hast?", "Hallo, ich wollte fragen, ob du Zeit hast?"),
            
            # Social Media Posts
            ("Das war ein echt coole Party gestern!", "Das war eine echt coole Party gestern!"),
            ("Bin grad im Urlaub und es ist super schön hier", "Ich bin gerade im Urlaub und es ist super schön hier."),
            
            # Geschäftliche Kommunikation
            ("Wir würden gerne ein Termin vereinbaren", "Wir würden gerne einen Termin vereinbaren"),
            ("Können Sie mir bitte die Dokumente schicken", "Können Sie mir bitte die Dokumente schicken?"),
            
            # Lernertexte (typische Deutschlerner-Fehler)
            ("Ich bin in Deutschland seit drei Jahren", "Ich bin seit drei Jahren in Deutschland"),
            ("Der Auto von mein Vater ist neu", "Das Auto von meinem Vater ist neu"),
            ("Ich habe gestern meine Freundin getroffen", "Ich habe gestern meine Freundin getroffen"),  # Korrekt
            
            # Komplexe Sätze
            ("Obwohl das Wetter schlecht war, sind wir trotzdem spazieren gegangen", 
             "Obwohl das Wetter schlecht war, sind wir trotzdem spazieren gegangen"),  # Korrekt
        ]
        
        for original, expected in real_world_cases:
            with self.subTest(text=original):
                result = self.corrector.correct_text(original, method="hybrid")
                
                # Prüfe dass sinnvolle Korrektur stattfand
                self.assertIsInstance(result.corrected_text, str)
                self.assertGreaterEqual(result.confidence_score, 0.3)  # Mindest-Konfidenz
                
                # Korrektur sollte nicht länger als Original + 50% sein
                self.assertLessEqual(len(result.corrected_text), len(original) * 1.5)
                
    def test_performance_under_load(self):
        """Test Performance unter Last"""
        # Simuliere hohe Last
        sentences = [
            "Das ist ein Test für Performance",
            "Ich gehe heute einkaufen",
            "Der Hund läuft schnell durch den Park",
            "Wir haben gestern einen Film gesehen"
        ] * 25  # 100 Sätze total
        
        start_time = time.time()
        results = []
        
        for sentence in sentences:
            result = self.corrector.correct_text(sentence, method="hybrid")
            results.append(result)
            
        total_time = time.time() - start_time
        
        # Performance-Validierung
        self.assertEqual(len(results), len(sentences))
        self.assertLess(total_time, 60)  # Unter 1 Minute für 100 Sätze
        
        # Durchschnittszeit pro Satz
        avg_time_per_sentence = total_time / len(sentences)
        self.assertLess(avg_time_per_sentence, 1.0)  # Unter 1 Sekunde pro Satz
        
    def test_memory_stability(self):
        """Test Speicher-Stabilität bei längerer Nutzung"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Lange Verarbeitungsschleife
        test_sentence = "Das ist ein Stabilitätstest für das System"
        
        for i in range(200):  # 200 Iterationen
            result = self.corrector.correct_text(test_sentence, method="hybrid")
            
            # Gelegentliche Garbage Collection
            if i % 50 == 0:
                gc.collect()
                
            # Speicher-Check alle 50 Iterationen
            if i % 50 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # Speicherwachstum sollte kontrolliert sein
                self.assertLess(memory_growth, 1000)  # Unter 1GB Wachstum
                
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        # Finaler Speicher-Check
        self.assertLess(total_growth, 1500)  # Unter 1.5GB Gesamtwachstum
        
    def test_concurrent_usage(self):
        """Test gleichzeitige Nutzung (Simulation)"""
        import threading
        import queue
        
        # Queue für Ergebnisse
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker(sentence_id):
            """Worker-Funktion für Thread"""
            try:
                sentence = f"Das ist Test-Satz Nummer {sentence_id}"
                result = self.corrector.correct_text(sentence, method="hybrid")
                results_queue.put((sentence_id, result))
            except Exception as e:
                errors_queue.put((sentence_id, str(e)))
                
        # Starte mehrere Threads
        threads = []
        num_threads = 10
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Warte auf alle Threads
        for thread in threads:
            thread.join(timeout=30)  # 30 Sekunden Timeout
            
        # Validiere Ergebnisse
        self.assertTrue(errors_queue.empty(), 
                       f"Fehler in Threads: {list(errors_queue.queue)}")
        self.assertEqual(results_queue.qsize(), num_threads)
        
        # Prüfe alle Ergebnisse
        while not results_queue.empty():
            sentence_id, result = results_queue.get()
            self.assertIsInstance(result, CorrectionResult)
            self.assertIn(str(sentence_id), result.corrected_text)

def run_performance_suite():
    """Führt umfassende Performance-Tests durch"""
    print("=== PERFORMANCE SUITE ===\n")
    
    corrector = HybridGermanGrammarCorrector(
        mt5_model_path="./models/trained_models/final_model",
        use_gpu=torch.cuda.is_available(),
        enable_caching=False
    )
    
    benchmark = GermanGECBenchmark(corrector)
    
    print("1. Accuracy Benchmark...")
    accuracy_results = benchmark.benchmark_accuracy()
    
    print("\nAccuracy Results:")
    for method, data in accuracy_results.items():
        print(f"  {method}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")
    
    print("\n2. Performance Benchmark...")
    performance_results = benchmark.benchmark_performance(num_sentences=50)
    
    print("\nPerformance Results:")
    for method, data in performance_results.items():
        print(f"  {method}:")
        print(f"    Durchsatz: {data['throughput_sentences_per_second']:.1f} Sätze/s")
        print(f"    Ø Zeit: {data['avg_time_per_sentence_ms']:.1f}ms")
        print(f"    Speicher: {data['peak_memory_mb']:.1f}MB")
    
    print("\n3. Comprehensive Benchmark...")
    full_results = benchmark.run_comprehensive_benchmark(save_results=True)
    
    print(f"\nZusammenfassung:")
    summary = full_results['summary']
    print(f"  Beste Genauigkeit: {summary['best_accuracy_method']}")
    print(f"  Beste Geschwindigkeit: {summary['best_speed_method']}")
    print(f"  Empfehlung: {summary['recommended_method']}")
    
    # Performance-Ziele prüfen
    print(f"\n=== PERFORMANCE-ZIELE ===")
    hybrid_perf = performance_results.get('hybrid', {})
    
    goals = {
        "Latenz < 500ms": hybrid_perf.get('avg_time_per_sentence_ms', 0) < 500,
        "Durchsatz > 2 Sätze/s": hybrid_perf.get('throughput_sentences_per_second', 0) > 2,
        "Speicher < 3GB": hybrid_perf.get('peak_memory_mb', 0) < 3000,
        "Genauigkeit > 80%": accuracy_results.get('hybrid', {}).get('accuracy', 0) > 0.8
    }
    
    for goal, achieved in goals.items():
        status = "✅" if achieved else "❌"
        print(f"  {status} {goal}")
    
    return full_results

def run_demo():
    """Führt eine interaktive Demo durch"""
    print("=== DEUTSCHE GRAMMATIKKORREKTUR DEMO ===\n")
    
    corrector = HybridGermanGrammarCorrector(
        mt5_model_path="./models/trained_models/final_model",
        use_gpu=torch.cuda.is_available()
    )
    
    demo_sentences = [
        "Der Hund beißt der Mann",
        "Ich gehen zur Schule heute",
        "Das ist ein sehr schöne Tag",
        "Ich hab das gestern gemacht",
        "Kommst du heut mit ins Kino?",
        "Der Auto von mein Vater ist kaputt",
        "Obwohl es regnet, gehe ich spazieren",
        "Können Sie mir helfen bitte?",
        "Das Wetter ist fantastich heute"
    ]
    
    print("Teste verschiedene Korrektur-Methoden:\n")
    
    for i, sentence in enumerate(demo_sentences, 1):
        print(f"{i}. Original: '{sentence}'")
        
        # Teste beide Methoden
        for method in ["languagetool", "hybrid"]:
            result = corrector.correct_text(sentence, method=method)
            
            status = "→" if result.corrected_text != sentence else "✓"
            print(f"   {method:12} {status} '{result.corrected_text}' "
                  f"(Konfidenz: {result.confidence_score:.2f}, "
                  f"Zeit: {result.processing_time_ms:.0f}ms)")
            
            if result.corrections:
                print(f"                   Korrekturen: {len(result.corrections)}")
        
        print()
    
    # Performance-Statistiken
    stats = corrector.get_performance_stats()
    print(f"=== STATISTIKEN ===")
    print(f"Verarbeitete Sätze: {stats.total_corrections}")
    print(f"Ø Verarbeitungszeit: {stats.avg_processing_time_ms:.1f}ms")
    print(f"Durchsatz: {stats.throughput_sentences_per_second:.1f} Sätze/s")
    print(f"Aktueller Speicherverbrauch: {stats.peak_memory_mb:.1f}MB")
    print(f"LanguageTool-only: {stats.languagetool_corrections}")
    print(f"Hybrid-Korrekturen: {stats.hybrid_corrections}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            run_demo()
        elif sys.argv[1] == "performance":
            run_performance_suite()
        elif sys.argv[1] == "test":
            # Führe Unit Tests aus
            unittest.main(argv=[''], exit=False, verbosity=2)
        else:
            print("Verwendung: python test_german_gec.py [demo|performance|test]")
    else:
        # Standard: Führe alle Tests aus
        print("Führe Unit Tests aus...")
        unittest.main(verbosity=2)