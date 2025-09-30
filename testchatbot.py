import sys

print("Testing emotion detection model import...")
print("=" * 50)

# Test 1: Import transformers library
try:
    print("1. Testing transformers import...")
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    print("✓ Successfully imported transformers library")
except ImportError as e:
    print(f"✗ Failed to import transformers: {e}")
    print("Solution: Run 'pip install transformers torch'")
    sys.exit(1)

# Test 2: Load model using pipeline (high-level approach)
try:
    print("\n2. Testing pipeline approach...")
    pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion")
    print("✓ Successfully loaded model using pipeline")
    
    # Test the pipeline
    test_text = "I am so happy today!"
    result = pipe(test_text)
    print(f"✓ Pipeline test successful: {result}")
    
except Exception as e:
    print(f"✗ Pipeline approach failed: {e}")
    print(f"Error type: {type(e).__name__}")

# Test 3: Load model directly (low-level approach)
try:
    print("\n3. Testing direct model loading...")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
    print("✓ Successfully loaded tokenizer and model directly")
    
    # Test the direct approach
    test_text = "I am feeling great!"
    inputs = tokenizer(test_text, return_tensors="pt")
    outputs = model(**inputs)
    print("✓ Direct model test successful")
    
except Exception as e:
    print(f"✗ Direct loading failed: {e}")
    print(f"Error type: {type(e).__name__}")

print("\n" + "=" * 50)
print("Model testing complete!")

# If we get here, at least one method worked
print("\nIf you see checkmarks above, the model is working!")
print("You can now use either approach in your main chatbot code.")