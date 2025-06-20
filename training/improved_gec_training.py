# training/improved_gec_training.py
# Verbesserungen für niedrigeren Loss

import os
import torch
from torch.utils.data import Dataset
from transformers import (
    MT5Tokenizer, MT5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedGECDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_text, target_text = self.pairs[idx]
        
        # Prefixe für T5/MT5 (wichtig!)
        input_text = f"correct: {input_text}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Replace padding token id's of the labels by -100 (ignore_index)
        labels = targets['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

def compute_metrics(eval_pred):
    """Compute custom metrics"""
    predictions, labels = eval_pred
    
    # Handle predictions - sie können logits oder bereits argmax sein
    if hasattr(predictions, 'argmax'):
        predictions = predictions.argmax(axis=-1)
    elif isinstance(predictions, tuple):
        # Falls predictions ein Tupel ist (logits, ...)
        predictions = predictions[0].argmax(axis=-1) if hasattr(predictions[0], 'argmax') else predictions[0]
    
    # Calculate accuracy (ignoring -100)
    mask = labels != -100
    if mask.sum() == 0:
        return {"accuracy": 0.0}
    
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / mask.sum()
    
    return {"accuracy": float(accuracy)}

# Verbesserte Training-Konfiguration
def get_improved_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # Mehr Epochen
        learning_rate=3e-4,  # Höhere Learning Rate
        per_device_train_batch_size=2,  # Kleinere Batch Size
        per_device_eval_batch_size=4,
        warmup_steps=100,  # Mehr Warmup Steps
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=250,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        logging_dir=None,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,  # Effektiv größere Batch Size
        label_smoothing_factor=0.1,  # Label Smoothing
    )

def analyze_loss_sources(model, tokenizer, sample_pairs):
    """Analysiere Ursachen für hohen Loss"""
    model.eval()
    
    logger.info("=== LOSS ANALYSIS ===")
    
    total_loss = 0
    num_samples = min(5, len(sample_pairs))
    
    for i, (input_text, target_text) in enumerate(sample_pairs[:num_samples]):
        # Prepare input
        input_text = f"correct: {input_text}"
        
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            padding=True,
            truncation=True
        )
        
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                target_text,
                return_tensors="pt",
                max_length=128,
                padding=True,
                truncation=True
            )
        
        labels = targets['input_ids'].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
        
        # Generate prediction
        generated = model.generate(
            **inputs,
            max_length=128,
            num_beams=2,
            early_stopping=True
        )
        
        predicted = tokenizer.decode(generated[0], skip_special_tokens=True)
        
        logger.info(f"Sample {i+1}:")
        logger.info(f"  Input:     {input_text}")
        logger.info(f"  Target:    {target_text}")
        logger.info(f"  Predicted: {predicted}")
        logger.info(f"  Loss:      {loss.item():.4f}")
        logger.info("---")
    
    avg_loss = total_loss / num_samples
    logger.info(f"Average Loss: {avg_loss:.4f}")
    
    return avg_loss

def check_data_quality(pairs):
    """Überprüfe Datenqualität"""
    logger.info("=== DATA QUALITY CHECK ===")
    
    logger.info(f"Total pairs: {len(pairs)}")
    
    # Analysiere Längenverhältnisse
    length_ratios = []
    identical_pairs = 0
    
    for input_text, output_text in pairs:
        if input_text == output_text:
            identical_pairs += 1
        else:
            ratio = len(output_text) / len(input_text)
            length_ratios.append(ratio)
    
    if length_ratios:
        avg_ratio = sum(length_ratios) / len(length_ratios)
        logger.info(f"Average output/input length ratio: {avg_ratio:.2f}")
    
    logger.info(f"Identical pairs (no correction needed): {identical_pairs}")
    logger.info(f"Pairs with corrections: {len(pairs) - identical_pairs}")
    
    # Zeige ein paar Beispiele
    logger.info("\nFirst 3 correction examples:")
    count = 0
    for input_text, output_text in pairs:
        if input_text != output_text and count < 3:
            logger.info(f"  Input:  {input_text[:80]}...")
            logger.info(f"  Output: {output_text[:80]}...")
            logger.info("")
            count += 1

# Verbesserte Haupttraining-Funktion
# Importiere CSVDataProcessor
from train_german_gec_mt5 import CSVDataProcessor

def train_with_improved_config():
    """Training mit verbesserter Konfiguration"""
    
    # Lade Daten (verwende selected_training_data.csv mit 50 hochwertigen Paaren)
    csv_path = "training/selected_training_data.csv"
    data_processor = CSVDataProcessor(csv_path)
    training_pairs = data_processor.load_training_pairs()
    
    if not training_pairs:
        logger.error("No training data found!")
        return False
    
    # Datenqualitäts-Check
    check_data_quality(training_pairs)
    
    # Erweitere Dataset weniger aggressiv
    if len(training_pairs) < 100:
        training_pairs = data_processor.create_extended_dataset(training_pairs, 500)
    
    # Split
    split_idx = int(0.8 * len(training_pairs))  # 80/20 split
    train_pairs = training_pairs[:split_idx]
    eval_pairs = training_pairs[split_idx:]
    
    logger.info(f"Training pairs: {len(train_pairs)}")
    logger.info(f"Evaluation pairs: {len(eval_pairs)}")
    
    # Model und Tokenizer
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    
    # Datasets mit verbesserter Klasse
    train_dataset = ImprovedGECDataset(train_pairs, tokenizer)
    eval_dataset = ImprovedGECDataset(eval_pairs, tokenizer)
    
    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        return_tensors="pt",
        padding=True
    )
    
    # Verbesserte Training Args
    training_args = get_improved_training_args("./models/trained_models_improved")
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,  # Deaktiviert wegen Tupel-Problem
    )
    
    # Vor dem Training: Loss-Analyse
    logger.info("Pre-training loss analysis:")
    analyze_loss_sources(model, tokenizer, eval_pairs)
    
    # Training
    logger.info("Starting improved training...")
    trainer.train()
    
    # Nach dem Training: Loss-Analyse
    logger.info("Post-training loss analysis:")
    analyze_loss_sources(model, tokenizer, eval_pairs)
    
    # Save
    final_model_path = "./models/trained_models_improved/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info("✅ Improved training completed!")
    return True

if __name__ == "__main__":
    train_with_improved_config()