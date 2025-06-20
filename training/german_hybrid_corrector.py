import os
import time
import torch
import psutil
import threading
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import language_tool_python
from functools import lru_cache
import logging
import json
from datetime import datetime

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CorrectionResult:
    """Ergebnis einer Grammatikkorrektur"""
    original_text: str
    corrected_text: str
    corrections: List[Dict]
    processing_time_ms: float
    confidence_score: float
    correction_method: str  # "languagetool", "mt5", "hybrid"
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class PerformanceMetrics:
    """Performance-Metriken für Benchmarking"""
    avg_processing_time_ms: float
    peak_memory_mb: float
    throughput_sentences_per_second: float
    total_corrections: int
    languagetool_corrections: int
    mt5_corrections: int
    hybrid_corrections: int
    
class MemoryMonitor:
    """Überwacht Speicherverbrauch während der Verarbeitung"""
    
    def __init__(self):
        self.peak_memory = 0
        self.monitoring = False
        self.thread = None
        
    def start_monitoring(self):
        """Startet Speicher-Monitoring"""
        self.monitoring = True
        self.peak_memory = 0
        self.thread = threading.Thread(target=self._monitor_memory)
        self.thread.start()
        
    def stop_monitoring(self) -> float:
        """Stoppt Monitoring und gibt Peak-Speicher zurück"""
        self.monitoring = False
        if self.thread:
            self.thread.join()
        return self.peak_memory / (1024 * 1024)  # MB
        
    def _monitor_memory(self):
        """Monitoring-Loop"""
        while self.monitoring:
            current_memory = psutil.Process().memory_info().rss
            self.peak_memory = max(self.peak_memory, current_memory)
            time.sleep(0.1)

class HybridGermanGrammarCorrector:
    """Hybrides System: LanguageTool + mT5 für deutsche Grammatikkorrektur"""
    
    def __init__(self, 
                 mt5_model_path: str = "./german_gec_mt5/final_model",
                 use_gpu: bool = True,
                 enable_caching: bool = True):
        """
        Initialisiert den hybriden Grammatikkorrektor
        
        Args:
            mt5_model_path: Pfad zum trainierten mT5-Modell
            use_gpu: GPU verwenden falls verfügbar
            enable_caching: LRU-Cache für häufige Korrekturen aktivieren
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.enable_caching = enable_caching
        
        # LanguageTool initialisieren
        logger.info("Initialisiere LanguageTool...")
        self.languagetool = language_tool_python.LanguageTool('de-DE')
        
        # mT5-Modell laden
        logger.info(f"Lade mT5-Modell von {mt5_model_path}...")
        try:
            self.mt5_tokenizer = MT5Tokenizer.from_pretrained(mt5_model_path)
            self.mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_model_path)
            self.mt5_model.to(self.device)
            self.mt5_model.eval()
            logger.info(f"mT5-Modell geladen auf {self.device}")
        except Exception as e:
            logger.warning(f"Konnte mT5-Modell nicht laden: {e}")
            logger.info("Verwende nur LanguageTool-Modus")
            self.mt5_model = None
            self.mt5_tokenizer = None
            
        # Performance-Tracking
        self.correction_stats = {
            "total_corrections": 0,
            "languagetool_only": 0,
            "mt5_only": 0,
            "hybrid": 0,
            "processing_times": []
        }
        
    @lru_cache(maxsize=1000)
    def _cached_correction(self, text: str) -> str:
        """Gecachte Korrektur für häufige Texte"""
        if not self.enable_caching:
            return None
        return None  # Cache-Miss, normale Verarbeitung
        
    def _languagetool_correct(self, text: str) -> Tuple[str, List[Dict], float]:
        """LanguageTool-Korrektur mit detaillierter Ausgabe"""
        start_time = time.time()
        
        matches = self.languagetool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)
        
        corrections = []
        for match in matches:
            corrections.append({
                "offset": match.offset,
                "length": match.errorLength,
                "message": match.message,
                "suggestions": match.replacements[:3],  # Top 3 Vorschläge
                "rule_id": match.ruleId,
                "category": match.category,
                "type": "languagetool"
            })
            
        processing_time = (time.time() - start_time) * 1000
        return corrected_text, corrections, processing_time
        
    def _mt5_correct(self, text: str) -> Tuple[str, float]:
        """mT5-basierte Korrektur"""
        if self.mt5_model is None:
            return text, 0.0
            
        start_time = time.time()
        
        # Tokenisierung
        input_text = f"Korrigiere: {text}"
        inputs = self.mt5_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.mt5_model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=False
            )
            
        corrected_text = self.mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        processing_time = (time.time() - start_time) * 1000
        
        return corrected_text, processing_time
        
    def _calculate_confidence(self, original: str, lt_corrected: str, mt5_corrected: str) -> float:
        """Berechnet Konfidenz-Score für Korrekturen"""
        if original == lt_corrected == mt5_corrected:
            return 0.95  # Beide Systeme sind sich einig, kein Fehler
        elif lt_corrected == mt5_corrected and lt_corrected != original:
            return 0.90  # Beide Systeme korrigieren gleich
        elif lt_corrected != original and mt5_corrected == original:
            return 0.70  # Nur LanguageTool korrigiert
        elif mt5_corrected != original and lt_corrected == original:
            return 0.65  # Nur mT5 korrigiert
        else:
            return 0.50  # Unterschiedliche Korrekturen
            
    def _needs_advanced_correction(self, text: str, lt_corrections: List[Dict]) -> bool:
        """Entscheidet ob mT5-Verarbeitung benötigt wird"""
        # Verwende mT5 wenn:
        # 1. LanguageTool wenige/keine Korrekturen fand
        # 2. Text komplex ist (längere Sätze, verschachtelte Strukturen)
        # 3. Potentielle Kontextfehler vorliegen
        
        if len(lt_corrections) == 0:
            return True  # Keine LanguageTool-Korrekturen, mT5 könnte mehr finden
            
        if len(text.split()) > 15:
            return True  # Längere Texte profitieren von mT5
            
        # Prüfe auf komplexe Fehlertypen
        complex_rules = {"GERMAN_SPACY_RULE", "AGREEMENT", "MORFOLOGIK_RULE"}
        for correction in lt_corrections:
            if any(rule in correction.get("rule_id", "") for rule in complex_rules):
                return True
                
        return False
        
    def correct_text(self, text: str, method: str = "hybrid") -> CorrectionResult:
        """
        Korrigiert deutschen Text mit verschiedenen Methoden
        
        Args:
            text: Zu korrigierender Text
            method: "languagetool", "mt5", oder "hybrid"
            
        Returns:
            CorrectionResult mit Korrektur und Metadaten
        """
        start_time = time.time()
        
        # Cache-Check
        if self.enable_caching:
            cached_result = self._cached_correction(text)
            if cached_result:
                return cached_result
                
        original_text = text.strip()
        
        if method == "languagetool":
            # Nur LanguageTool
            corrected_text, corrections, lt_time = self._languagetool_correct(original_text)
            processing_time = lt_time
            confidence_score = 0.80 if corrections else 0.95
            self.correction_stats["languagetool_only"] += 1
            
        elif method == "mt5":
            # Nur mT5
            corrected_text, mt5_time = self._mt5_correct(original_text)
            corrections = []
            processing_time = mt5_time
            confidence_score = 0.75 if corrected_text != original_text else 0.90
            self.correction_stats["mt5_only"] += 1
            
        else:  # hybrid
            # Hybrid-Ansatz
            lt_corrected, lt_corrections, lt_time = self._languagetool_correct(original_text)
            
            if self._needs_advanced_correction(original_text, lt_corrections):
                # Wende mT5 auf LanguageTool-Ergebnis an
                mt5_corrected, mt5_time = self._mt5_correct(lt_corrected)
                corrected_text = mt5_corrected
                processing_time = lt_time + mt5_time
                
                # Kombiniere Korrekturen
                corrections = lt_corrections.copy()
                if mt5_corrected != lt_corrected:
                    corrections.append({
                        "offset": 0,
                        "length": len(lt_corrected),
                        "message": "Weitere Verbesserungen durch mT5",
                        "suggestions": [mt5_corrected],
                        "type": "mt5_refinement"
                    })
                    
                confidence_score = self._calculate_confidence(original_text, lt_corrected, mt5_corrected)
                self.correction_stats["hybrid"] += 1
            else:
                # Nur LanguageTool-Ergebnis verwenden
                corrected_text = lt_corrected
                corrections = lt_corrections
                processing_time = lt_time
                confidence_score = 0.85 if corrections else 0.95
                self.correction_stats["languagetool_only"] += 1
                
        total_time = (time.time() - start_time) * 1000
        self.correction_stats["total_corrections"] += 1
        self.correction_stats["processing_times"].append(total_time)
        
        result = CorrectionResult(
            original_text=original_text,
            corrected_text=corrected_text,
            corrections=corrections,
            processing_time_ms=total_time,
            confidence_score=confidence_score,
            correction_method=method
        )
        
        return result
        
    def correct_sentences(self, sentences: List[str], method: str = "hybrid") -> List[CorrectionResult]:
        """Korrigiert eine Liste von Sätzen"""
        results = []
        for sentence in sentences:
            result = self.correct_text(sentence, method)
            results.append(result)
        return results
        
    def get_performance_stats(self) -> PerformanceMetrics:
        """Gibt aktuelle Performance-Statistiken zurück"""
        processing_times = self.correction_stats["processing_times"]
        
        if not processing_times:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)
            
        avg_time = sum(processing_times) / len(processing_times)
        throughput = 1000 / avg_time if avg_time > 0 else 0
        
        # Aktueller Speicherverbrauch
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        return PerformanceMetrics(
            avg_processing_time_ms=avg_time,
            peak_memory_mb=current_memory,
            throughput_sentences_per_second=throughput,
            total_corrections=self.correction_stats["total_corrections"],
            languagetool_corrections=self.correction_stats["languagetool_only"],
            mt5_corrections=self.correction_stats["mt5_only"],
            hybrid_corrections=self.correction_stats["hybrid"]
        )
        
    def reset_stats(self):
        """Setzt Performance-Statistiken zurück"""
        self.correction_stats = {
            "total_corrections": 0,
            "languagetool_only": 0,
            "mt5_only": 0,
            "hybrid": 0,
            "processing_times": []
        }

class GermanGECBenchmark:
    """Benchmark-Suite für deutsche Grammatikkorrektur"""
    
    def __init__(self, corrector: HybridGermanGrammarCorrector):
        self.corrector = corrector
        self.memory_monitor = MemoryMonitor()
        
    def create_test_sentences(self) -> List[Tuple[str, str]]:
        """Erstellt Test-Sätze mit Fehlern und erwarteten Korrekturen"""
        test_cases = [
            # Kasusfehlern
            ("Der Hund beißt den Mann", "Der Hund beißt den Mann"),  # Korrekt
            ("Der Hund beißt der Mann", "Der Hund beißt den Mann"),  # Kasusfehlern
            ("Ich gebe der Kind das Buch", "Ich gebe dem Kind das Buch"),  # Dativ-Fehlern
            
            # Verb-Konjugation
            ("Ich gehen zur Schule", "Ich gehe zur Schule"),
            ("Du gehst ins Kino", "Du gehst ins Kino"),  # Korrekt
            ("Wir ist müde", "Wir sind müde"),
            
            # Artikel-Genus
            ("Der Mädchen spielt", "Das Mädchen spielt"),
            ("Die Auto ist rot", "Das Auto ist rot"),
            ("Das Mann arbeitet", "Der Mann arbeitet"),
            
            # Alltagssprache
            ("Ich hab das gemacht", "Ich habe das gemacht"),
            ("Kommst du heut?", "Kommst du heute?"),
            ("Bin gleich da", "Ich bin gleich da"),
            
            # Komplexe Sätze
            ("Obwohl es regnet, gehe ich spazieren", "Obwohl es regnet, gehe ich spazieren"),  # Korrekt
            ("Wenn ich Zeit habe, besuche ich dich", "Wenn ich Zeit habe, besuche ich dich"),  # Korrekt
            ("Der Mann, der gestern kam, ist mein Freund", "Der Mann, der gestern kam, ist mein Freund"),  # Korrekt
            
            # Rechtschreibfehler
            ("Das ist ein schöner Tag", "Das ist ein schöner Tag"),  # Korrekt
            ("Ich bin ser müde heute", "Ich bin sehr müde heute"),
            ("Das Wetter ist fantastich", "Das Wetter ist fantastisch"),
            
            # Wortstellung
            ("Heute ich gehe einkaufen", "Heute gehe ich einkaufen"),
            ("In der Schule ich lerne Deutsch", "In der Schule lerne ich Deutsch"),
            
            # Längere Texte
            ("Gestern bin ich mit meine Freunde ins Kino gegangen und wir haben einen sehr interessant Film gesehen.",
             "Gestern bin ich mit meinen Freunden ins Kino gegangen und wir haben einen sehr interessanten Film gesehen."),
        ]
        
        return test_cases
        
    def benchmark_accuracy(self, test_cases: List[Tuple[str, str]] = None) -> Dict:
        """Benchmarkt Genauigkeit der Korrektur"""
        if test_cases is None:
            test_cases = self.create_test_sentences()
            
        results = {
            "languagetool": {"correct": 0, "total": 0, "corrections": []},
            "mt5": {"correct": 0, "total": 0, "corrections": []},
            "hybrid": {"correct": 0, "total": 0, "corrections": []}
        }
        
        for incorrect, expected in test_cases:
            for method in ["languagetool", "mt5", "hybrid"]:
                correction_result = self.corrector.correct_text(incorrect, method)
                corrected = correction_result.corrected_text
                
                # Prüfe Genauigkeit (exakter Match oder deutliche Verbesserung)
                is_correct = (corrected == expected) or \
                           (corrected != incorrect and self._is_improvement(incorrect, corrected, expected))
                
                results[method]["correct"] += int(is_correct)
                results[method]["total"] += 1
                results[method]["corrections"].append({
                    "original": incorrect,
                    "predicted": corrected,
                    "expected": expected,
                    "correct": is_correct,
                    "confidence": correction_result.confidence_score
                })
                
        # Berechne Genauigkeitsraten
        for method in results:
            total = results[method]["total"]
            correct = results[method]["correct"]
            results[method]["accuracy"] = correct / total if total > 0 else 0
            
        return results
        
    def _is_improvement(self, original: str, corrected: str, expected: str) -> bool:
        """Prüft ob Korrektur eine Verbesserung darstellt"""
        # Einfache Heuristik: Korrektur ist Verbesserung wenn näher am erwarteten Text
        original_distance = self._simple_distance(original, expected)
        corrected_distance = self._simple_distance(corrected, expected)
        return corrected_distance < original_distance
        
    def _simple_distance(self, text1: str, text2: str) -> int:
        """Einfache Levenshtein-Distanz"""
        if len(text1) < len(text2):
            return self._simple_distance(text2, text1)
            
        if len(text2) == 0:
            return len(text1)
            
        previous_row = list(range(len(text2) + 1))
        for i, c1 in enumerate(text1):
            current_row = [i + 1]
            for j, c2 in enumerate(text2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
        
    def benchmark_performance(self, num_sentences: int = 100) -> Dict:
        """Benchmarkt Performance (Geschwindigkeit, Speicher)"""
        test_sentences = [case[0] for case in self.create_test_sentences()]
        
        # Erweitere auf gewünschte Anzahl
        while len(test_sentences) < num_sentences:
            test_sentences.extend(test_sentences)
        test_sentences = test_sentences[:num_sentences]
        
        results = {}
        
        for method in ["languagetool", "mt5", "hybrid"]:
            logger.info(f"Benchmarke Performance für {method}...")
            
            # Statistiken zurücksetzen
            self.corrector.reset_stats()
            
            # Memory-Monitoring starten
            self.memory_monitor.start_monitoring()
            
            start_time = time.time()
            
            # Verarbeite alle Sätze
            for sentence in test_sentences:
                self.corrector.correct_text(sentence, method)
                
            total_time = time.time() - start_time
            peak_memory = self.memory_monitor.stop_monitoring()
            
            # Sammle Metriken
            stats = self.corrector.get_performance_stats()
            
            results[method] = {
                "total_time_seconds": total_time,
                "avg_time_per_sentence_ms": stats.avg_processing_time_ms,
                "throughput_sentences_per_second": num_sentences / total_time,
                "peak_memory_mb": peak_memory,
                "sentences_processed": num_sentences
            }
            
        return results
        
    def run_comprehensive_benchmark(self, save_results: bool = True) -> Dict:
        """Führt umfassenden Benchmark durch"""
        logger.info("Starte umfassenden Benchmark...")
        
        # Accuracy-Benchmark
        logger.info("Benchmarke Genauigkeit...")
        accuracy_results = self.benchmark_accuracy()
        
        # Performance-Benchmark
        logger.info("Benchmarke Performance...")
        performance_results = self.benchmark_performance(100)
        
        # Kombiniere Ergebnisse
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "device": self.corrector.device,
                "gpu_available": torch.cuda.is_available(),
                "mt5_model_loaded": self.corrector.mt5_model is not None,
                "total_memory_gb": psutil.virtual_memory().total / (1024**3)
            },
            "accuracy": accuracy_results,
            "performance": performance_results,
            "summary": self._create_summary(accuracy_results, performance_results)
        }
        
        if save_results:
            # Speichere Ergebnisse
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Benchmark-Ergebnisse gespeichert in: {filename}")
            
        return benchmark_results
        
    def _create_summary(self, accuracy_results: Dict, performance_results: Dict) -> Dict:
        """Erstellt Zusammenfassung der Benchmark-Ergebnisse"""
        # Beste Methode für Genauigkeit
        best_accuracy_method = max(accuracy_results.keys(), 
                                 key=lambda x: accuracy_results[x]["accuracy"])
        best_accuracy = accuracy_results[best_accuracy_method]["accuracy"]
        
        # Beste Methode für Geschwindigkeit
        best_speed_method = max(performance_results.keys(),
                              key=lambda x: performance_results[x]["throughput_sentences_per_second"])
        best_speed = performance_results[best_speed_method]["throughput_sentences_per_second"]
        
        return {
            "best_accuracy_method": best_accuracy_method,
            "best_accuracy_score": best_accuracy,
            "best_speed_method": best_speed_method,
            "best_speed_sentences_per_second": best_speed,
            "recommended_method": "hybrid",  # Empfehlung basierend auf Balance
            "accuracy_ranking": sorted(accuracy_results.keys(), 
                                     key=lambda x: accuracy_results[x]["accuracy"], 
                                     reverse=True),
            "speed_ranking": sorted(performance_results.keys(),
                                  key=lambda x: performance_results[x]["throughput_sentences_per_second"],
                                  reverse=True)
        }

def main():
    """Hauptfunktion für Demo und Test"""
    # Initialisiere Korrektor
    corrector = HybridGermanGrammarCorrector(
        mt5_model_path="./german_gec_mt5/final_model",
        use_gpu=True,
        enable_caching=True
    )
    
    # Test-Sätze
    test_sentences = [
        "Der Hund beißt der Mann",
        "Ich gehen zur Schule",
        "Das ist ein schöne Tag",
        "Ich hab das gemacht",
        "Kommst du heut?"
    ]
    
    print("=== Deutsche Grammatikkorrektur Demo ===\n")
    
    # Teste verschiedene Methoden
    for method in ["languagetool", "hybrid"]:
        print(f"\n--- {method.upper()} ---")
        for sentence in test_sentences:
            result = corrector.correct_text(sentence, method)
            print(f"Original:  {result.original_text}")
            print(f"Korrigiert: {result.corrected_text}")
            print(f"Konfidenz: {result.confidence_score:.2f}")
            print(f"Zeit: {result.processing_time_ms:.1f}ms")
            if result.corrections:
                print(f"Korrekturen: {len(result.corrections)}")
            print("-" * 50)
    
    # Performance-Statistiken
    stats = corrector.get_performance_stats()
    print(f"\n=== Performance-Statistiken ===")
    print(f"Durchschnittliche Zeit: {stats.avg_processing_time_ms:.1f}ms")
    print(f"Durchsatz: {stats.throughput_sentences_per_second:.1f} Sätze/s")
    print(f"Speicherverbrauch: {stats.peak_memory_mb:.1f}MB")
    print(f"Gesamt-Korrekturen: {stats.total_corrections}")
    
    # Benchmark ausführen
    print("\n=== Starte Benchmark ===")
    benchmark = GermanGECBenchmark(corrector)
    results = benchmark.run_comprehensive_benchmark()
    
    print(f"\nBenchmark abgeschlossen!")
    print(f"Beste Genauigkeit: {results['summary']['best_accuracy_method']} "
          f"({results['summary']['best_accuracy_score']:.2%})")
    print(f"Beste Geschwindigkeit: {results['summary']['best_speed_method']} "
          f"({results['summary']['best_speed_sentences_per_second']:.1f} Sätze/s)")

if __name__ == "__main__":
    main()