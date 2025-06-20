import os
import logging
from typing import List, Tuple, Set
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSelector:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        
    def load_all_pairs(self) -> List[Tuple[str, str]]:
        """Lade alle Paare aus CSV"""
        pairs = []
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if ';' in line:
                parts = line.split(';', 1)
                if len(parts) == 2:
                    input_text = parts[0].strip()
                    output_text = parts[1].strip()
                    
                    if input_text and output_text:
                        pairs.append((input_text, output_text))
        
        logger.info(f"Loaded {len(pairs)} total pairs")
        return pairs
    
    def analyze_data_quality(self, pairs: List[Tuple[str, str]]) -> dict:
        """Analysiere die Datenqualität"""
        analysis = {
            'total_pairs': len(pairs),
            'identical_pairs': 0,
            'different_pairs': 0,
            'length_stats': [],
            'unique_inputs': set(),
            'unique_outputs': set(),
            'correction_types': defaultdict(int)
        }
        
        for input_text, output_text in pairs:
            analysis['unique_inputs'].add(input_text)
            analysis['unique_outputs'].add(output_text)
            
            if input_text == output_text:
                analysis['identical_pairs'] += 1
            else:
                analysis['different_pairs'] += 1
                analysis['length_stats'].append(len(output_text) - len(input_text))
                
                # Analysiere Korrekturtypen
                self._analyze_correction_type(input_text, output_text, analysis['correction_types'])
        
        analysis['unique_input_count'] = len(analysis['unique_inputs'])
        analysis['unique_output_count'] = len(analysis['unique_outputs'])
        
        return analysis
    
    def _analyze_correction_type(self, input_text: str, output_text: str, correction_types: dict):
        """Analysiere den Typ der Korrektur"""
        
        # Komma-Korrekturen
        if input_text.count(',') != output_text.count(','):
            correction_types['comma_changes'] += 1
        
        # Kapitalisierung
        if input_text.lower() == output_text.lower() and input_text != output_text:
            correction_types['capitalization'] += 1
        
        # Wortänderungen (dass/das, etc.)
        input_words = input_text.lower().split()
        output_words = output_text.lower().split()
        
        if len(input_words) == len(output_words):
            diff_count = sum(1 for i, o in zip(input_words, output_words) if i != o)
            if diff_count <= 2:
                correction_types['word_substitutions'] += 1
        
        # Längere Umformulierungen
        length_diff = abs(len(output_text) - len(input_text))
        if length_diff > 20:
            correction_types['major_reformulations'] += 1
    
    def select_quality_pairs(self, pairs: List[Tuple[str, str]], max_pairs: int = 100) -> List[Tuple[str, str]]:
        """Wähle die besten Trainingspaare aus"""
        
        # 1. Entferne identische Paare
        different_pairs = [(i, o) for i, o in pairs if i != o]
        logger.info(f"After removing identical pairs: {len(different_pairs)}")
        
        # 2. Entferne Duplikate
        unique_pairs = list(set(different_pairs))
        logger.info(f"After removing duplicates: {len(unique_pairs)}")
        
        # 3. Qualitätsbewertung
        scored_pairs = []
        for input_text, output_text in unique_pairs:
            score = self._calculate_quality_score(input_text, output_text)
            scored_pairs.append((score, input_text, output_text))
        
        # 4. Sortiere nach Qualität
        scored_pairs.sort(reverse=True, key=lambda x: x[0])
        
        # 5. Wähle beste Paare
        selected_pairs = [(i, o) for _, i, o in scored_pairs[:max_pairs]]
        
        logger.info(f"Selected {len(selected_pairs)} high-quality pairs")
        
        # 6. Zeige Top-Paare
        self._show_selected_examples(scored_pairs[:10])
        
        return selected_pairs
    
    def _calculate_quality_score(self, input_text: str, output_text: str) -> float:
        """Berechne Qualitätsscore für ein Trainingspaar"""
        score = 0.0
        
        # Basispunkte für unterschiedliche Texte
        if input_text != output_text:
            score += 1.0
        
        # Länge (nicht zu kurz, nicht zu lang)
        input_length = len(input_text)
        if 20 <= input_length <= 200:
            score += 1.0
        elif input_length > 200:
            score -= 0.5  # Zu lang
        
        # Realistisch wirkende Korrekturen
        length_diff = abs(len(output_text) - len(input_text))
        if length_diff < len(input_text) * 0.3:  # Änderung < 30%
            score += 1.0
        
        # Grammatische Marker
        if any(word in input_text.lower() for word in ['dass', 'das', 'den', 'dem', 'der']):
            score += 0.5  # Potentielle Grammatikfehler
        
        # Komma-Korrekturen sind wertvoll
        comma_diff = abs(input_text.count(',') - output_text.count(','))
        if 0 < comma_diff <= 3:
            score += 0.5
        
        # Kapitalisierung
        if input_text.lower() == output_text.lower() and input_text != output_text:
            score += 0.3
        
        # Strafe für sehr ähnliche oder zu komplexe Änderungen
        if len(input_text.split()) > 50:  # Sehr lange Sätze
            score -= 0.3
        
        return score
    
    def _show_selected_examples(self, scored_pairs: List[Tuple[float, str, str]]):
        """Zeige die besten ausgewählten Beispiele"""
        logger.info("\n=== TOP 10 SELECTED TRAINING EXAMPLES ===")
        
        for i, (score, input_text, output_text) in enumerate(scored_pairs, 1):
            logger.info(f"\n{i}. Score: {score:.2f}")
            logger.info(f"   Input:  {input_text[:80]}...")
            logger.info(f"   Output: {output_text[:80]}...")
    
    def create_filtered_dataset(self, output_path: str = "training/selected_training_data.csv"):
        """Erstelle gefilterten Datensatz"""
        
        # Lade und analysiere Daten
        all_pairs = self.load_all_pairs()
        analysis = self.analyze_data_quality(all_pairs)
        
        # Zeige Analyse
        self._print_analysis(analysis)
        
        # Wähle beste Paare aus
        selected_pairs = self.select_quality_pairs(all_pairs, max_pairs=50)
        
        # Speichere gefilterten Datensatz
        with open(output_path, 'w', encoding='utf-8') as f:
            for input_text, output_text in selected_pairs:
                f.write(f"{input_text};{output_text}\n")
        
        logger.info(f"\n✅ Filtered dataset saved to: {output_path}")
        logger.info(f"📊 Final dataset size: {len(selected_pairs)} high-quality pairs")
        
        return output_path, selected_pairs
    
    def _print_analysis(self, analysis: dict):
        """Drucke Datenanalyse"""
        logger.info("\n=== DATA QUALITY ANALYSIS ===")
        logger.info(f"Total pairs: {analysis['total_pairs']}")
        logger.info(f"Identical pairs: {analysis['identical_pairs']}")
        logger.info(f"Different pairs: {analysis['different_pairs']}")
        logger.info(f"Unique inputs: {analysis['unique_input_count']}")
        logger.info(f"Unique outputs: {analysis['unique_output_count']}")
        
        if analysis['length_stats']:
            avg_length_change = sum(analysis['length_stats']) / len(analysis['length_stats'])
            logger.info(f"Average length change: {avg_length_change:.1f} characters")
        
        logger.info("\nCorrection types found:")
        for correction_type, count in analysis['correction_types'].items():
            logger.info(f"  {correction_type}: {count}")

def create_training_config_for_small_dataset():
    """Trainings-Konfiguration für kleinen, hochwertigen Datensatz"""
    return {
        'num_train_epochs': 10,  # Mehr Epochen bei weniger Daten
        'per_device_train_batch_size': 1,  # Sehr kleine Batches
        'learning_rate': 1e-4,  # Niedrigere Learning Rate
        'warmup_steps': 50,
        'eval_steps': 25,  # Häufigere Evaluation
        'save_steps': 25,
        'weight_decay': 0.1,  # Stärkere Regularisierung
        'gradient_accumulation_steps': 8,  # Effektiv größere Batches
    }

def main():
    """Hauptfunktion für Datenselektion"""
    csv_path = "training/deutsche_trainingsdaten.csv"
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    # Erstelle Datenselektor
    selector = DataSelector(csv_path)
    
    # Erstelle gefilterten Datensatz
    filtered_path, selected_pairs = selector.create_filtered_dataset()
    
    # Zeige Empfehlungen
    logger.info("\n=== RECOMMENDATIONS ===")
    logger.info("1. Use the filtered dataset for training")
    logger.info("2. Use suggested training configuration for small datasets")
    logger.info("3. Focus on data quality over quantity")
    logger.info("4. Consider manually reviewing and improving the selected pairs")
    
    # Zeige Trainings-Konfiguration
    config = create_training_config_for_small_dataset()
    logger.info(f"\nRecommended training config: {config}")

if __name__ == "__main__":
    main()