import os
import torch
from torch.utils.data import Dataset
from transformers import (
    MT5Tokenizer, MT5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from dataclasses import dataclass
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GECConfig:
    model_name: str = "google/mt5-small"
    output_dir: str = "./models/trained_models"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-5
    max_length: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    csv_data_path: str = "training/deutsche_trainingsdaten.csv"

class CSVDataProcessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        
    def load_training_pairs(self) -> List[Tuple[str, str]]:
        """Load training pairs from CSV file"""
        training_pairs = []
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.strip().split('\n')
            logger.info(f"Found {len(lines)} lines in CSV")
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if ';' in line:
                    parts = line.split(';', 1)
                    if len(parts) == 2:
                        input_text = parts[0].strip()
                        output_text = parts[1].strip()
                        
                        if input_text and output_text and input_text != output_text:
                            training_pairs.append((input_text, output_text))
                        elif input_text and output_text and input_text == output_text:
                            logger.debug(f"Skipping identical pair at line {line_num}")
                    else:
                        logger.warning(f"Invalid format at line {line_num}: {line[:50]}...")
                else:
                    logger.warning(f"No semicolon found at line {line_num}: {line[:50]}...")
            
            logger.info(f"Loaded {len(training_pairs)} valid training pairs")
            return training_pairs
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return []
    
    def create_extended_dataset(self, base_pairs: List[Tuple[str, str]], target_size: int = 1000) -> List[Tuple[str, str]]:
        """Extend dataset by duplicating existing pairs"""
        extended_pairs = base_pairs.copy()
        
        while len(extended_pairs) < target_size and base_pairs:
            for input_text, output_text in base_pairs:
                if len(extended_pairs) >= target_size:
                    break
                extended_pairs.append((input_text, output_text))
                
        logger.info(f"Extended dataset to {len(extended_pairs)} pairs")
        return extended_pairs

class GECDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], tokenizer, max_length: int = 128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_text, target_text = self.pairs[idx]
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

class GermanGECTrainer:
    def __init__(self, config: GECConfig):
        self.config = config
        self.data_processor = CSVDataProcessor(config.csv_data_path)
        
    def train_model(self) -> bool:
        """Train the German GEC model using CSV data"""
        try:
            logger.info("Starting German GEC training with CSV data")
            
            training_pairs = self.data_processor.load_training_pairs()
            
            if not training_pairs:
                logger.error("No training pairs found in CSV file!")
                return False
            
            if len(training_pairs) < 500:
                logger.info(f"Extending dataset from {len(training_pairs)} pairs")
                training_pairs = self.data_processor.create_extended_dataset(training_pairs, 1000)
            
            split_idx = int(0.9 * len(training_pairs))
            train_pairs = training_pairs[:split_idx]
            eval_pairs = training_pairs[split_idx:]
            
            logger.info(f"Training pairs: {len(train_pairs)}")
            logger.info(f"Evaluation pairs: {len(eval_pairs)}")
            
            logger.info("Loading mT5 model and tokenizer...")
            tokenizer = MT5Tokenizer.from_pretrained(self.config.model_name)
            model = MT5ForConditionalGeneration.from_pretrained(self.config.model_name)
            
            train_dataset = GECDataset(train_pairs, tokenizer, self.config.max_length)
            eval_dataset = GECDataset(eval_pairs, tokenizer, self.config.max_length)
            
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                return_tensors="pt",
                padding=True
            )
            
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                warmup_ratio=self.config.warmup_ratio,
                weight_decay=self.config.weight_decay,
                logging_steps=self.config.logging_steps,
                eval_strategy="steps",
                eval_steps=self.config.eval_steps,
                save_steps=self.config.save_steps,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=None,
                logging_dir=None,
                dataloader_pin_memory=False,
                fp16=torch.cuda.is_available(),
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            
            logger.info("Starting training...")
            trainer.train()
            
            final_model_path = os.path.join(self.config.output_dir, "final_model")
            logger.info(f"Saving final model to {final_model_path}")
            trainer.save_model(final_model_path)
            tokenizer.save_pretrained(final_model_path)
            
            self.test_sample_corrections(model, tokenizer)
            
            logger.info("Training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_sample_corrections(self, model, tokenizer):
        """Test the model with sample corrections"""
        model.eval()
        
        test_inputs = [
            "das Buch dass sie gelesen hatte lag auf dem Tisch",
            "der Hund der im Garten bellt hört plötzlich auf",
            "sie ging in die Schule obwohl es regnete"
        ]
        
        logger.info("Testing sample corrections:")
        
        for input_text in test_inputs:
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=128,
                padding=True,
                truncation=True
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    length_penalty=0.6,
                    early_stopping=True
                )
            
            corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Input:  {input_text}")
            logger.info(f"Output: {corrected}")
            logger.info("---")

def main():
    """Main training function"""
    logger.info("="*80)
    logger.info("    GERMAN GEC TRAINING WITH CSV DATA")
    logger.info("="*80)
    
    csv_path = "training/deutsche_trainingsdaten.csv"
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        logger.error("Please ensure the CSV file is in the training/ directory")
        return False
    
    config = GECConfig(
        model_name="google/mt5-small",
        output_dir="./models/trained_models",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        max_length=128,
        csv_data_path=csv_path
    )
    
    try:
        trainer = GermanGECTrainer(config)
        success = trainer.train_model()
        
        if success:
            logger.info("✅ Training completed successfully!")
            logger.info(f"📁 Trained model: {config.output_dir}/final_model")
            logger.info("\n🚀 Next steps:")
            logger.info("   - Test: python testing/test_german_gec.py")
            logger.info("   - Optimize: python optimization/convert_checkpoint_to_tf.py")
        else:
            logger.error("❌ Training failed!")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)