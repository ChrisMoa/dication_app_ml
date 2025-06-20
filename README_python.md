# SpaCy Grammar Service

This is a Python service that provides German grammar checking using SpaCy's German language model. It runs as a local HTTP server that the Flutter app can communicate with.

## Setup

1. Make sure you have Python 3.7+ installed
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Service

1. Activate the virtual environment if you haven't already
2. Start the service:
   ```bash
   python spacy_grammar_service.py
   ```
3. The service will run on http://localhost:5000

## Using in Flutter

To use the SpaCy grammar checker in your Flutter app:

```dart
import 'package:your_app/core/services/spacy_provider.dart';

// In your code:
final grammarService = AIGrammarService();
final spacyProvider = SpaCyProvider();

// Check if SpaCy service is available
if (await spacyProvider.isServiceAvailable()) {
  // Switch to SpaCy provider
  grammarService.setProvider(spacyProvider);
  
  // Use the grammar service as normal
  final result = await grammarService.correctGermanText(text);
  // ... handle result
} else {
  // Fall back to LanguageTool or local provider
  print('SpaCy service is not running');
}
```

## Features

- Offline grammar checking for German text
- Basic grammar rules including:
  - Capitalization checks
  - Punctuation rules
  - Passive voice detection
- Confidence scoring
- Detailed error reporting

## Limitations

- Less sophisticated than LanguageTool
- Requires Python environment setup
- Currently runs only on localhost
- Basic grammar rules only

## Troubleshooting

1. If the service fails to start:
   - Check if port 5000 is available
   - Ensure all dependencies are installed
   - Check Python version (3.7+ required)

2. If the Flutter app can't connect:
   - Verify the service is running (`python spacy_grammar_service.py`)
   - Check if you can access http://localhost:5000 in a browser
   - Ensure no firewall is blocking the connection 