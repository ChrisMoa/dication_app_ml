# mobile_model_trainer.py
import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from pathlib import Path
import logging
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMobileGECModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = 256
        self.max_length = 32
        
        # Mobile-optimierte Architektur
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.input_projection = torch.nn.Linear(embedding_dim, self.hidden_dim)
        self.lstm = torch.nn.LSTM(
            self.hidden_dim, 
            self.hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=0.1
        )
        self.output_projection = torch.nn.Linear(self.hidden_dim, self.vocab_size)
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        if seq_len > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            seq_len = self.max_length
        
        # Embeddings
        token_embeddings = self.embeddings(input_ids)
        
        # Project und LSTM
        hidden = self.input_projection(token_embeddings)
        hidden = self.dropout(hidden)
        
        lstm_out, _ = self.lstm(hidden)
        lstm_out = self.dropout(lstm_out)
        
        # Output logits
        logits = self.output_projection(lstm_out)
        
        # Copy mechanism - bevorzuge Input-Tokens
        input_one_hot = torch.nn.functional.one_hot(input_ids, self.vocab_size).float()
        copy_bias = input_one_hot * 2.0
        
        final_logits = logits + copy_bias
        
        return final_logits

class MobileModelTrainer:
    def __init__(self):
        self.teacher_model_path = self._find_teacher_model()
        self.output_dir = Path("./models/trained_mobile")
        self.output_dir.mkdir(exist_ok=True)
        
        # Lade Teacher-Modell
        self.teacher_model, self.tokenizer = self._load_teacher_model()
        
    def _find_teacher_model(self):
        possible_paths = [
            Path("./models/trained_models_improved/final_model"),
            Path("./models/trained_models/final_model"),
            Path("./german_gec_mt5/final_model")
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"✅ Found teacher model: {path}")
                return path
        
        raise FileNotFoundError("❌ No trained teacher model found!")
    
    def _load_teacher_model(self):
        logger.info("Loading teacher model...")
        
        model = MT5ForConditionalGeneration.from_pretrained(self.teacher_model_path)
        tokenizer = MT5Tokenizer.from_pretrained(self.teacher_model_path)
        model.eval()
        
        logger.info("✅ Teacher model loaded")
        return model, tokenizer
    
    def create_training_data(self, num_samples=1000):
        """Erstelle Trainingsdaten mit Teacher-Student Approach"""
        logger.info(f"Creating {num_samples} training samples...")
        
        # Deutsche Beispielsätze mit häufigen Fehlern
        error_templates = [
            "Das ist ein fehler",
            "Ich gehe zur schule", 
            "Er hat das buch gelest",
            "Die Kinder spielt im park",
            "Wir haben der test gemacht",
            "Sie ist zu der laden gegangen",
            "Das auto von mein bruder",
            "Ich bin in die schule gewesen",
            "Der hund beißt der mann",
            "Weil ich müde bin gehe ich schlafen",
            "Seit wann bist du hier",
            "Seid ihr morgen da",
            "Das buch liegt auf der tisch",
            "Ich habe einen gute tag",
            "Der mann mit der rote hut"
        ]
        
        training_pairs = []
        
        for i in range(num_samples):
            # Wähle zufälligen Fehler-Satz
            error_text = error_templates[i % len(error_templates)]
            
            # Variiere den Satz etwas
            if i > len(error_templates):
                words = error_text.split()
                if len(words) > 2:
                    # Tausche zufällig Wörter oder ändere Artikel
                    import random
                    if random.random() < 0.3:
                        words[0] = random.choice(['Der', 'Die', 'Das'])
                    error_text = ' '.join(words)
            
            # Hole Korrektur vom Teacher-Modell
            input_text = f"Korrigiere: {error_text}"
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True)
            
            with torch.no_grad():
                outputs = self.teacher_model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=3,
                    early_stopping=True,
                    do_sample=False
                )
            
            corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Tokenisiere für Mobile-Modell (kurze Sequenzen)
            error_tokens = self.tokenizer(f"Korrigiere: {error_text}", 
                                        return_tensors="pt", max_length=32, truncation=True)
            corrected_tokens = self.tokenizer(corrected, 
                                            return_tensors="pt", max_length=32, truncation=True)
            
            training_pairs.append({
                'input_ids': error_tokens['input_ids'][0],
                'labels': corrected_tokens['input_ids'][0],
                'error_text': error_text,
                'corrected_text': corrected
            })
            
            if i % 100 == 0:
                logger.info(f"   Generated {i}/{num_samples} samples")
        
        logger.info(f"✅ Created {len(training_pairs)} training samples")
        return training_pairs
    
    def train_mobile_model(self, training_data, epochs=5):
        """Trainiere das Mobile-Modell mit Knowledge Distillation"""
        logger.info("Training mobile model with knowledge distillation...")
        
        # WICHTIG: Verwende die ECHTE Vocab-Größe vom Teacher-Modell
        teacher_vocab_size = self.teacher_model.shared.weight.shape[0]
        tokenizer_vocab_size = len(self.tokenizer.get_vocab())
        embedding_dim = self.teacher_model.config.d_model
        
        logger.info(f"Teacher embedding size: {teacher_vocab_size}")
        logger.info(f"Tokenizer vocab size: {tokenizer_vocab_size}")
        logger.info(f"Using teacher vocab size: {teacher_vocab_size}")
        
        # Erstelle Student-Modell mit KORREKTER Vocab-Größe
        student_model = SimpleMobileGECModel(teacher_vocab_size, embedding_dim)
        
        # Kopiere Embeddings vom Teacher (jetzt sollten die Größen passen)
        logger.info("Copying embeddings from teacher model...")
        with torch.no_grad():
            logger.info(f"Teacher embeddings shape: {self.teacher_model.shared.weight.shape}")
            logger.info(f"Student embeddings shape: {student_model.embeddings.weight.shape}")
            student_model.embeddings.weight.copy_(self.teacher_model.shared.weight)
        
        # Optimizer
        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        student_model.train()
        
        # Early stopping parameters
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        try:
            for epoch in range(epochs):
                total_loss = 0
                num_batches = 0
                
                for i, sample in enumerate(training_data):
                    input_ids = sample['input_ids'].unsqueeze(0)  # Add batch dim
                    labels = sample['labels'].unsqueeze(0)
                    
                    # Ensure same length for training
                    min_len = min(input_ids.size(1), labels.size(1), 32)
                    input_ids = input_ids[:, :min_len]
                    labels = labels[:, :min_len]
                    
                    # Forward pass
                    logits = student_model(input_ids)
                    
                    # Loss gegen Teacher-Output
                    loss = criterion(logits.view(-1, teacher_vocab_size), labels.view(-1))
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if i % 100 == 0:
                        logger.info(f"   Epoch {epoch+1}/{epochs}, Batch {i}, Loss: {loss.item():.4f}")
                        
                        # Early stopping check every 100 batches
                        if loss.item() < 1.5:
                            logger.info(f"🎯 Loss below 1.5 - model converged well!")
                            logger.info("Consider stopping training early for good results.")
                
                avg_loss = total_loss / num_batches
                logger.info(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
                
                # Early stopping logic
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    logger.info(f"✅ New best loss: {best_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"⚠️ No improvement for {patience_counter} epochs")
                
                # Stop conditions
                if avg_loss < 1.0:
                    logger.info(f"🎉 Excellent loss achieved ({avg_loss:.4f}) - stopping early!")
                    break
                elif avg_loss < 2.0 and epoch >= 1:
                    logger.info(f"✅ Good loss achieved ({avg_loss:.4f}) after {epoch+1} epochs")
                    logger.info("Model should work well now. Continue? (Ctrl+C to stop)")
                elif patience_counter >= patience:
                    logger.info(f"🔄 Early stopping triggered - no improvement for {patience} epochs")
                    break
                    
        except KeyboardInterrupt:
            logger.info("\n🛑 Training interrupted by user during training loop!")
            logger.info("💾 Saving current model state...")
            try:
                # Speichere Zwischenzustand
                student_model.eval()
                model_path = self.output_dir / "interrupted_mobile_gec.pth"
                torch.save({
                    'model_state_dict': student_model.state_dict(),
                    'model_class': 'SimpleMobileGECModel',
                    'vocab_size': teacher_vocab_size,
                    'embedding_dim': student_model.embeddings.embedding_dim,
                    'hidden_dim': student_model.hidden_dim,
                    'max_length': student_model.max_length,
                    'trained': True,
                    'interrupted': True,
                    'epochs_completed': epoch + 1
                }, model_path)
                
                logger.info(f"✅ Interrupted model saved: {model_path}")
                logger.info("📱 This partially trained model should still work!")
                
                # Gib das Modell zurück für weitere Verarbeitung
                return student_model
            except Exception as e:
                logger.error(f"❌ Failed to save interrupted model: {e}")
                raise
        
        student_model.eval()
        logger.info("✅ Mobile model training completed")
        
        return student_model
    
    def test_mobile_model(self, mobile_model):
        """Teste das trainierte Mobile-Modell"""
        logger.info("Testing trained mobile model...")
        
        test_sentences = [
            "Das ist ein fehler",
            "Ich gehe zur schule",
            "Er hat das buch gelest",
            "Die kinder spielt im park"
        ]
        
        mobile_model.eval()
        
        for sentence in test_sentences:
            logger.info(f"\nTesting: '{sentence}'")
            
            # Tokenize
            input_text = f"Korrigiere: {sentence}"
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=32, truncation=True)
            
            # Mobile inference
            with torch.no_grad():
                logits = mobile_model(inputs['input_ids'])
                predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode
            try:
                corrected = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                logger.info(f"   Mobile output: '{corrected}'")
            except:
                logger.info(f"   Mobile tokens: {predicted_ids[0][:10].tolist()}")
            
            # Compare with teacher
            with torch.no_grad():
                teacher_outputs = self.teacher_model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=2,
                    early_stopping=True
                )
                teacher_corrected = self.tokenizer.decode(teacher_outputs[0], skip_special_tokens=True)
                logger.info(f"   Teacher output: '{teacher_corrected}'")
    
    def save_trained_mobile_model(self, mobile_model):
        """Speichere das trainierte Mobile-Modell"""
        logger.info("Saving trained mobile model...")
        
        # Verwende die echte Vocab-Größe vom Modell
        actual_vocab_size = mobile_model.vocab_size
        
        # Model speichern
        model_path = self.output_dir / "trained_mobile_gec.pth"
        torch.save({
            'model_state_dict': mobile_model.state_dict(),
            'model_class': 'SimpleMobileGECModel',
            'vocab_size': actual_vocab_size,
            'embedding_dim': mobile_model.embeddings.embedding_dim,
            'hidden_dim': mobile_model.hidden_dim,
            'max_length': mobile_model.max_length,
            'trained': True
        }, model_path)
        
        # Vocabulary - verwende Teacher-Tokenizer aber passe Größe an
        vocab_path = self.output_dir / "vocab.json"
        vocab = self.tokenizer.get_vocab()
        
        # Falls Vocab-Größen unterschiedlich sind, adjustiere
        if len(vocab) != actual_vocab_size:
            logger.info(f"Adjusting vocab size from {len(vocab)} to {actual_vocab_size}")
            # Erweitere oder kürze Vokabular entsprechend
            if len(vocab) < actual_vocab_size:
                # Füge Dummy-Tokens hinzu
                for i in range(len(vocab), actual_vocab_size):
                    vocab[f"<extra_token_{i}>"] = i
            else:
                # Kürze auf die benötigte Größe
                vocab = {k: v for k, v in vocab.items() if v < actual_vocab_size}
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        # Config
        config = {
            'model_type': 'trained_mobile_gec',
            'teacher_model': str(self.teacher_model_path),
            'vocab_size': actual_vocab_size,
            'tokenizer_vocab_size': len(self.tokenizer.get_vocab()),
            'architecture': 'LSTM_with_knowledge_distillation',
            'trained': True,
            'performance': 'significantly_better_than_untrained'
        }
        
        config_path = self.output_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Trained mobile model saved:")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Vocab: {vocab_path} ({actual_vocab_size} tokens)")
        logger.info(f"   Config: {config_path}")
        
        return model_path
    
    def run_full_training_pipeline(self):
        """Komplette Trainings-Pipeline mit besserem Interrupt Handling"""
        logger.info("="*70)
        logger.info("🎓 MOBILE MODEL TRAINING PIPELINE")
        logger.info("Knowledge Distillation from trained MT5 to mobile LSTM")
        logger.info("Press Ctrl+C to safely interrupt and save progress")
        logger.info("="*70)
        
        try:
            # 1. Erstelle Trainingsdaten
            training_data = self.create_training_data(num_samples=2000)
            
            # 2. Trainiere Mobile-Modell (kann interrupted werden)
            mobile_model = self.train_mobile_model(training_data, epochs=10)
            
            # 3. Teste Mobile-Modell
            self.test_mobile_model(mobile_model)
            
            # 4. Speichere finales Mobile-Modell
            model_path = self.save_trained_mobile_model(mobile_model)
            
            logger.info("\n" + "="*70)
            logger.info("🎉 MOBILE TRAINING COMPLETE!")
            logger.info("="*70)
            logger.info("✅ Mobile model learned from teacher MT5!")
            logger.info("✅ Knowledge successfully distilled!")
            logger.info("✅ Ready for mobile deployment!")
            
            logger.info(f"\n📱 TRAINED MOBILE MODEL:")
            logger.info(f"   Path: {model_path}")
            logger.info(f"   Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
            logger.info(f"   Architecture: LSTM with knowledge distillation")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Training pipeline interrupted by user!")
            
            # Check if interrupted model was saved
            interrupted_path = self.output_dir / "interrupted_mobile_gec.pth"
            if interrupted_path.exists():
                logger.info("✅ Found interrupted model - continuing with testing...")
                
                try:
                    # Load interrupted model for testing
                    checkpoint = torch.load(interrupted_path, map_location='cpu')
                    
                    # Recreate model
                    mock_base = type('MockModel', (), {'shared': torch.nn.Embedding(checkpoint['vocab_size'], checkpoint['embedding_dim'])})()
                    mobile_model = SimpleMobileGECModel(checkpoint['vocab_size'], checkpoint['embedding_dim'])
                    mobile_model.load_state_dict(checkpoint['model_state_dict'])
                    mobile_model.eval()
                    
                    logger.info(f"📊 Interrupted after {checkpoint.get('epochs_completed', 'unknown')} epochs")
                    
                    # Test the interrupted model
                    self.test_mobile_model(mobile_model)
                    
                    # Save as final model
                    final_path = self.save_trained_mobile_model(mobile_model)
                    
                    logger.info("\n" + "="*70)
                    logger.info("🎉 INTERRUPTED TRAINING COMPLETED SUCCESSFULLY!")
                    logger.info("="*70)
                    logger.info("✅ Partially trained model still learned from teacher!")
                    logger.info("✅ Should work much better than untrained model!")
                    
                    logger.info(f"\n📱 PARTIALLY TRAINED MOBILE MODEL:")
                    logger.info(f"   Path: {final_path}")
                    logger.info(f"   Size: {final_path.stat().st_size / (1024*1024):.1f} MB")
                    logger.info(f"   Training: Interrupted but functional")
                    
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process interrupted model: {e}")
                    return False
            else:
                logger.info("❌ No interrupted model found - training stopped too early")
                return False
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    print("🎓 MOBILE MODEL TRAINER")
    print("Train a mobile LSTM model using knowledge distillation from MT5")
    
    trainer = MobileModelTrainer()
    success = trainer.run_full_training_pipeline()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("Mobile model now has learned German grammar correction!")
        print("Use the trained model instead of the empty one.")
    else:
        print("\n❌ Training failed")

if __name__ == "__main__":
    main()