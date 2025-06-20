import os
import json
import time
import torch
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import psutil
import gc

@dataclass
class GECConfig:
    """Konfiguration für deutsches Grammatikkorrektur-Training"""
    model_name: str = "google/mt5-small"
    max_length: int = 64
    output_dir: str = "./models/trained_models"
    num_train_epochs: int = 2
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 50

class GermanErrorGenerator:
    """Synthetische Fehlergeneration für deutsche Texte"""
    
    def __init__(self):
        pass
        
    def inject_case_errors(self, text: str) -> str:
        """Fügt deutsche Kasusfehlern hinzu"""
        replacements = {
            " der ": [" den ", " dem ", " des "],
            " die ": [" der ", " den ", " dem "],
            " das ": [" den ", " dem ", " des "],
            " ein ": [" eine ", " einen ", " einer "],
            " eine ": [" einen ", " einer ", " eines "]
        }
        
        for original, options in replacements.items():
            if original in text and random.random() < 0.3:
                text = text.replace(original, random.choice(options), 1)
        return text
    
    def inject_verb_errors(self, text: str) -> str:
        """Fügt Konjugationsfehler hinzu"""
        verb_errors = {
            " bin ": " ist ", " ist ": " sind ", " sind ": " bin ",
            " habe ": " hat ", " hat ": " haben ", " haben ": " habe ",
            " gehe ": " gehst ", " gehst ": " geht ", " geht ": " gehen "
        }
        
        for correct, wrong in verb_errors.items():
            if correct in text and random.random() < 0.2:
                text = text.replace(correct, wrong, 1)
        return text
    
    def inject_simple_errors(self, text: str) -> str:
        """Fügt einfache Fehler hinzu"""
        # Großschreibung
        if random.random() < 0.3:
            text = text[0].lower() + text[1:] if len(text) > 1 else text.lower()
        
        # Rechtschreibfehler
        spelling_errors = {
            "Kinder": "Kinden",
            "wunderschön": "wundenschön"
        }
        
        for correct, wrong in spelling_errors.items():
            if correct in text and random.random() < 0.2:
                text = text.replace(correct, wrong)
        
        return text
    
    def generate_errors(self, text: str) -> str:
        """Generiert verschiedene Fehlertypen"""
        corrupted_text = text
        
        if random.random() < 0.4:
            corrupted_text = self.inject_case_errors(corrupted_text)
        if random.random() < 0.3:
            corrupted_text = self.inject_verb_errors(corrupted_text)
        if random.random() < 0.2:
            corrupted_text = self.inject_simple_errors(corrupted_text)
            
        return corrupted_text

class GermanGECDataProcessor:
    """Verarbeitung deutscher Grammatikkorrektur-Daten"""
    
    def __init__(self, config: GECConfig):
        self.config = config
        self.tokenizer = MT5Tokenizer.from_pretrained(config.model_name)
        self.error_generator = GermanErrorGenerator()
        
    def generate_synthetic_data(self, num_samples: int = 15000) -> List[Tuple[str, str]]:
        """Generiert synthetische deutsche Trainingsdaten"""
        print(f"Generiere {num_samples} synthetische Trainingspaare...")
        
        # Deutsche Basis-Sätze - erweitert für mehr Vielfalt
        base_sentences = [
            "Der Hund läuft schnell durch den Park.",
            "Die Katze schläft auf dem warmen Sofa.", 
            "Ich gehe heute in die Stadt einkaufen.",
            "Das Wetter ist heute sehr schön und sonnig.",
            "Wir haben gestern einen interessanten Film gesehen.",
            "Die Kinder spielen im Garten mit dem Ball.",
            "Meine Mutter kocht ein leckeres Abendessen.",
            "Der Zug fährt pünktlich vom Bahnhof ab.",
            "Die Studenten lernen fleißig für die Prüfung.",
            "Das Auto steht vor dem großen Haus.",
            "Ich lese ein spannendes Buch am Abend.",
            "Die Blumen blühen wunderschön im Frühling.",
            "Wir treffen uns morgen im neuen Café.",
            "Der Lehrer erklärt die schwierige Aufgabe.",
            "Die Familie macht Urlaub an der Ostsee.",
            "Das kleine Kind spielt mit seinen Spielzeugen.",
            "Ich höre gerne klassische Musik am Wochenende.",
            "Der Koch bereitet ein köstliches Menü zu.",
            "Die Vögel singen schön in den Bäumen.",
            "Wir fahren mit dem Bus zur Arbeit.",
            # Erweiterte Sätze für mehr Vielfalt
            "Der Arzt untersucht den kranken Patienten.",
            "Die Schüler schreiben eine wichtige Klausur.",
            "Mein Vater repariert das kaputte Fahrrad.",
            "Die Touristen besuchen das berühmte Museum.",
            "Der Bäcker backt frisches Brot am Morgen.",
            "Die Polizei kontrolliert den Verkehr.",
            "Ich kaufe neue Kleidung im Geschäft.",
            "Der Gärtner pflanzt bunte Blumen.",
            "Die Bibliothekarin hilft den Studenten.",
            "Wir wandern durch den dichten Wald.",
            "Der Mechaniker repariert das alte Auto.",
            "Die Krankenschwester pflegt die Patienten.",
            "Ich besuche meine Großeltern am Sonntag.",
            "Der Postbote bringt die wichtigen Briefe.",
            "Die Sekretärin tippt einen langen Brief.",
            "Wir kochen zusammen ein gesundes Abendessen.",
            "Der Pilot fliegt das große Flugzeug.",
            "Die Verkäuferin bedient die freundlichen Kunden.",
            "Ich lerne Deutsch in der Sprachschule.",
            "Der Friseur schneidet die langen Haare.",
            "Die Journalistin schreibt einen interessanten Artikel.",
            "Wir feiern den schönen Geburtstag.",
            "Der Handwerker baut ein neues Regal.",
            "Die Lehrerin korrigiert die schweren Hausaufgaben.",
            "Ich spiele Tennis mit meinem besten Freund.",
            "Der Richter urteilt in dem wichtigen Fall.",
            "Die Ärztin verschreibt eine wirksame Medizin.",
            "Wir picknicken im grünen Park.",
            "Der Elektriker installiert eine neue Lampe.",
            "Die Übersetzerin übersetzt den deutschen Text."
        ]
        
        synthetic_pairs = []
        attempts = 0
        
        # Mehr Variationen pro Basis-Satz generieren
        while len(synthetic_pairs) < num_samples and attempts < num_samples * 2:
            clean_text = random.choice(base_sentences)
            
            # Generiere mehrere Variationen desselben Satzes
            for _ in range(3):  # 3 Variationen pro Basis-Satz
                corrupted_text = self.error_generator.generate_errors(clean_text)
                
                if corrupted_text != clean_text and len(corrupted_text.strip()) > 0:
                    synthetic_pairs.append((corrupted_text, clean_text))
                    
                if len(synthetic_pairs) >= num_samples:
                    break
            
            attempts += 1
                    
        print(f"Synthetische Daten generiert: {len(synthetic_pairs)} Paare")
        return synthetic_pairs
    
    def prepare_dataset_simple(self, pairs: List[Tuple[str, str]], prefix: str = "Korrigiere: ") -> Dataset:
        """Bereitet Datensatz einfach vor - ohne komplexe Tokenisierung"""
        print(f"Bereite {len(pairs)} Trainingspaare vor...")
        
        # Erstelle tokenisierte Daten direkt
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for source, target in pairs:
            # Input mit Prefix
            input_text = prefix + source
            target_text = target
            
            # Tokenisiere Input
            input_encoded = self.tokenizer(
                input_text,
                max_length=self.config.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Tokenisiere Target
            target_encoded = self.tokenizer(
                target_text,
                max_length=self.config.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids_list.append(input_encoded["input_ids"].squeeze().tolist())
            attention_mask_list.append(input_encoded["attention_mask"].squeeze().tolist())
            labels_list.append(target_encoded["input_ids"].squeeze().tolist())
        
        # Erstelle Dataset direkt
        dataset_dict = {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        print(f"✅ Dataset erstellt: {len(dataset)} Einträge")
        return dataset

class GermanGECTrainer:
    """Training für deutsches mT5 Grammatikkorrektur-Modell"""
    
    def __init__(self, config: GECConfig):
        self.config = config
        self.data_processor = GermanGECDataProcessor(config)
        self.tokenizer = MT5Tokenizer.from_pretrained(config.model_name)
        
    def train_model(self):
        """Trainiert das Modell"""
        print("\n=== TRAINING DEUTSCHES GRAMMATIKKORREKTUR-MODELL ===")
        
        # Generiere Trainingsdaten
        all_pairs = self.data_processor.generate_synthetic_data(15000)
        
        # Train/Eval Split
        split_idx = int(0.9 * len(all_pairs))
        train_pairs = all_pairs[:split_idx]
        eval_pairs = all_pairs[split_idx:]
        
        print(f"Training-Paare: {len(train_pairs)}")
        print(f"Evaluation-Paare: {len(eval_pairs)}")
        
        # Bereite Datasets vor
        train_dataset = self.data_processor.prepare_dataset_simple(train_pairs)
        eval_dataset = self.data_processor.prepare_dataset_simple(eval_pairs)
        
        # Lade Basis-Modell
        print("Lade mT5-Modell...")
        model = MT5ForConditionalGeneration.from_pretrained(self.config.model_name)
        
        # Erstelle Output-Verzeichnis
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Training-Argumente - KORRIGIERTE VERSION
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",  # KORREKT: eval_strategy, nicht evaluation_strategy
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],  # Keine Logging-Services
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            fp16=False,
            bf16=False,
            gradient_checkpointing=False,
            dataloader_num_workers=0,
            disable_tqdm=False,
            prediction_loss_only=True,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            seed=42,
            log_level="warning"  # Reduziere Logging
        )
        
        # Data Collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        # Trainer erstellen
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
            # Keine Callbacks - verursachen oft Probleme
        )
        
        print("Starte Training...")
        try:
            # Training starten
            trainer.train()
            
            # Modell speichern
            final_path = f"{self.config.output_dir}/final_model"
            trainer.save_model(final_path)
            self.tokenizer.save_pretrained(final_path)
            
            print(f"\n✅ Training erfolgreich abgeschlossen!")
            print(f"Modell gespeichert in: {final_path}")
            
            # Test-Inferenz
            self.test_model(final_path)
            return True
            
        except Exception as e:
            print(f"\n❌ Fehler beim Training: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Speicher freigeben
            del model, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def test_model(self, model_path: str):
        """Testet das trainierte Modell"""
        print(f"\n=== MODELL-TEST ===")
        
        try:
            # Lade trainiertes Modell
            model = MT5ForConditionalGeneration.from_pretrained(model_path)
            tokenizer = MT5Tokenizer.from_pretrained(model_path)
            
            # Test-Sätze
            test_sentences = [
                "Der Hund beißt der Mann",
                "Ich gehen zur Schule",
                "Das ist ein schöne Tag"
            ]
            
            print("Test-Korrekturen:")
            for sentence in test_sentences:
                corrected = self.correct_sentence(sentence, model, tokenizer)
                print(f"  Original:   {sentence}")
                print(f"  Korrigiert: {corrected}")
                print()
                
        except Exception as e:
            print(f"Fehler beim Modell-Test: {e}")
    
    def correct_sentence(self, sentence: str, model, tokenizer) -> str:
        """Korrigiert einen Satz mit dem Modell"""
        try:
            input_text = f"Korrigiere: {sentence}"
            
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=64,
                truncation=True,
                padding=True
            )
            
            model.eval()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected_text
            
        except Exception as e:
            print(f"Fehler bei Korrektur: {e}")
            return sentence

def check_system_requirements():
    """Prüft System-Anforderungen"""
    print("Prüfe System-Anforderungen...")
    
    # RAM-Check
    total_ram = psutil.virtual_memory().total / (1024**3)
    available_ram = psutil.virtual_memory().available / (1024**3)
    
    print(f"✅ Gesamt-RAM: {total_ram:.1f}GB")
    print(f"✅ Verfügbar-RAM: {available_ram:.1f}GB")
    
    if available_ram < 4:
        print("⚠️  Wenig RAM verfügbar - Training könnte langsam sein")
    
    # GPU-Check
    if torch.cuda.is_available():
        print(f"✅ GPU verfügbar: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️  Keine GPU verfügbar - Training auf CPU")
    
    # Freier Speicherplatz
    free_space = psutil.disk_usage('.').free / (1024**3)
    print(f"✅ Freier Speicherplatz: {free_space:.1f}GB")
    
    return True

def main():
    """Hauptfunktion für das Training"""
    print("="*80)
    print("    DEUTSCHES GRAMMATIKKORREKTUR-TRAINING")
    print("    Korrigierte Version für stabile Ausführung")
    print("="*80)
    
    # System-Check
    if not check_system_requirements():
        return False
    
    # Konfiguration für stabiles Training
    config = GECConfig(
        model_name="google/mt5-small",
        output_dir="./models/trained_models",
        num_train_epochs=3,  # Erhöht auf 3 Epochen
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        max_length=64,
        save_steps=500,  # Weniger häufiges Speichern
        eval_steps=500,
        logging_steps=100
    )
    
    try:
        # Trainer erstellen und Training starten
        trainer = GermanGECTrainer(config)
        success = trainer.train_model()
        
        if success:
            print("\n✅ Setup erfolgreich abgeschlossen!")
            print(f"📁 Trainiertes Modell: {config.output_dir}/final_model")
            print("\n🚀 Nächste Schritte:")
            print("   - Test: python test_german_gec.py demo")
            print("   - Benchmark: python test_german_gec.py performance")
        else:
            print("\n❌ Training fehlgeschlagen!")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)