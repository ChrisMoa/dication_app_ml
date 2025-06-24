import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

class SimpleGECTester:
    def __init__(self, model_path="./models/trained_models_improved/final_model"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Lade das trainierte Model"""
        try:
            print(f"Loading model from: {self.model_path}")
            self.tokenizer = MT5Tokenizer.from_pretrained(self.model_path)
            self.model = MT5ForConditionalGeneration.from_pretrained(self.model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def correct_text(self, text):
        """Korrigiere Text"""
        # Add prefix for T5
        input_text = f"correct: {text}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            padding=True,
            truncation=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True
            )
        
        # Decode
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected
    
    def run_tests(self):
        """Führe Tests durch"""
        print("\n=== GERMAN GEC TESTS ===")
        
        test_cases = [
            "Ich glaube das es regnet",
            "Er sagt das er kommt", 
            "Die Frau die dort steht ist meine Mutter",
            "Wir fahren nach Berlin weil wir Urlaub haben",
            "Der Mann der arbeitet ist müde",
            "Das Auto ist rot",
            "der hund läuft schnell",
            "Ich gehen zur Schule",
            "Das ist ein schöne Tag"
        ]
        
        print(f"Testing {len(test_cases)} examples:")
        print("-" * 60)
        
        for i, test_text in enumerate(test_cases, 1):
            corrected = self.correct_text(test_text)
            print(f"{i:2d}. Input:  {test_text}")
            print(f"    Output: {corrected}")
            print()
        
        print("✅ All tests completed!")
    
    def interactive_test(self):
        """Interaktiver Test"""
        print("\n=== INTERACTIVE MODE ===")
        print("Enter German text to correct (or 'quit' to exit):")
        
        while True:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
                
            corrected = self.correct_text(user_input)
            print(f"Corrected: {corrected}")
        
        print("Goodbye!")

def main():
    """Hauptfunktion"""
    
    # Check if model exists
    model_path = "./models/trained_models_improved/final_model"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first:")
        print("  python training/improved_gec_training.py")
        sys.exit(1)
    
    # Create tester
    tester = SimpleGECTester(model_path)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            tester.run_tests()
        elif command == "interactive":
            tester.interactive_test()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: test, interactive")
    else:
        # Default: run tests
        tester.run_tests()
        
        # Ask for interactive mode
        response = input("\nRun interactive mode? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            tester.interactive_test()

if __name__ == "__main__":
    main()