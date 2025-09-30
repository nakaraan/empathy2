import sys
import os

print("Testing Twitter RoBERTa Multi-Label Emotion Detection Model...")
print("=" * 70)

# Test 1: Import transformers library
try:
    print("1. Testing transformers import...")
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    print("✓ Successfully imported transformers and torch")
except ImportError as e:
    print(f"✗ Failed to import required libraries: {e}")
    print("Solution: Run 'pip install transformers torch'")
    sys.exit(1)

# Test 2: Load Twitter RoBERTa emotion model using pipeline (multi-label)
try:
    print("\n2. Testing Twitter RoBERTa multi-label pipeline...")
    pipe = pipeline(
        "text-classification", 
        model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest",
        return_all_scores=True  # Get all emotion scores
    )
    print("✓ Successfully loaded Twitter RoBERTa multi-label model")
    
    # Test the pipeline with sample text
    test_texts = [
        "I am so happy and excited today!",
        "I feel sad and lonely right now",
        "I'm angry and frustrated with everything",
        "I'm nervous but excited about tomorrow"
    ]
    
    print("\n   Testing multi-label emotion detection:")
    for text in test_texts:
        result = pipe(text)
        print(f"   Text: '{text}'")
        print(f"   Raw result format: {type(result)}")
        
        # Handle nested list format: [[{emotions}]] or [{emotions}]
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                emotions = result[0]  # Nested list format
            else:
                emotions = result  # Simple list format
        else:
            emotions = result
        
        # Show top 3 emotions
        top_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)[:3]
        for emotion in top_emotions:
            print(f"     - {emotion['label']}: {emotion['score']:.3f}")
        print()
    
except Exception as e:
    print(f"✗ Multi-label pipeline failed: {e}")
    print(f"Error type: {type(e).__name__}")

# Test 3: Load single-label version for comparison
try:
    print("3. Testing Twitter RoBERTa single-label pipeline...")
    pipe_single = pipeline(
        "text-classification", 
        model="cardiffnlp/twitter-roberta-base-emotion",
        top_k=1
    )
    print("✓ Successfully loaded Twitter RoBERTa single-label model")
    
    # Test single-label detection
    test_text = "I'm feeling great but a bit nervous about the presentation"
    result_single = pipe_single(test_text)
    
    # Handle nested format for single-label too
    if isinstance(result_single, list) and len(result_single) > 0:
        if isinstance(result_single[0], list):
            emotion_result = result_single[0][0]  # Nested format
        else:
            emotion_result = result_single[0]  # Simple format
    else:
        emotion_result = result_single
    
    print(f"   Single-label result for '{test_text}':")
    print(f"   Primary emotion: {emotion_result['label']} ({emotion_result['score']:.3f})")
    
except Exception as e:
    print(f"✗ Single-label pipeline failed: {e}")

# Test 4: Test emotion mapping and confidence thresholds
try:
    print("\n4. Testing emotion mapping and response selection...")
    
    # Emotion mapping from your chatbot
    emotion_mapping = {
        "sadness": "sad",
        "joy": "happy",
        "anger": "angry", 
        "fear": "fearful",
        "love": "love",
        "optimism": "optimistic",
        "surprise": "happy",
        "anticipation": "optimistic",
        "trust": "love",
        "disgust": "angry",
        "pessimism": "sad"
    }
    
    test_scenarios = [
        ("I feel absolutely terrible and worthless", "sadness"),
        ("I'm over the moon with joy!", "joy"),
        ("I'm furious about this situation", "anger"),
        ("I'm scared about what might happen", "fear"),
        ("I love everything about this day", "love")
    ]
    
    for test_text, expected in test_scenarios:
        try:
            results = pipe(test_text)
            
            # Handle nested list format properly
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    emotions = results[0]  # Nested list format: [[{emotions}]]
                else:
                    emotions = results  # Simple list format: [{emotions}]
            else:
                emotions = results
            
            # Find top emotion
            top_emotion = max(emotions, key=lambda x: x['score'])
            mapped = emotion_mapping.get(top_emotion['label'], "neutral")
            
            match_indicator = "✓" if top_emotion['label'] == expected else "❓"
            print(f"   {match_indicator} '{test_text}'")
            print(f"     Detected: {top_emotion['label']} ({top_emotion['score']:.3f})")
            print(f"     Mapped to: {mapped}")
            print(f"     Expected: {expected}")
            
            # Show all significant emotions (score > 0.2)
            significant = [e for e in emotions if e['score'] > 0.2]
            if len(significant) > 1:
                print(f"     Other significant emotions:")
                for emotion in significant[1:3]:  # Show top 2 additional
                    print(f"       - {emotion['label']}: {emotion['score']:.3f}")
            print()
            
        except Exception as e:
            print(f"   ✗ Failed to process: {test_text} - {e}")
            print(f"     Error type: {type(e).__name__}")
            # Debug: show raw result
            try:
                debug_result = pipe(test_text)
                print(f"     Debug - Raw result: {debug_result}")
                print(f"     Debug - Result type: {type(debug_result)}")
            except:
                pass
    
except Exception as e:
    print(f"✗ Emotion mapping test failed: {e}")

# Test 5: Performance and memory usage
try:
    print("5. Testing performance and resource usage...")
    import time
    import psutil
    import gc
    
    # Memory before
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Time multiple predictions
    test_texts = ["I'm feeling great today!"] * 10
    start_time = time.time()
    
    for text in test_texts:
        result = pipe(text)
    
    end_time = time.time()
    
    # Memory after
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    avg_time = (end_time - start_time) / len(test_texts)
    memory_used = memory_after - memory_before
    
    print(f"✓ Performance metrics:")
    print(f"   Average prediction time: {avg_time:.3f} seconds")
    print(f"   Memory usage: {memory_after:.1f} MB")
    print(f"   Memory increase: {memory_used:.1f} MB")
    
    # Clean up
    gc.collect()
    
except ImportError:
    print("   ⚠ psutil not available for memory testing")
    print("   Install with: pip install psutil")
except Exception as e:
    print(f"   ✗ Performance test failed: {e}")

# Test 6: Crisis detection keywords
try:
    print("\n6. Testing crisis detection...")
    
    crisis_keywords = [
        'suicide', 'kill myself', 'end it all', 'hurt myself',
        'die', 'not worth living', 'want to die', 'end my life'
    ]
    
    test_crisis_texts = [
        "I want to end it all",
        "I'm thinking about suicide",
        "Life is not worth living anymore"
    ]
    
    safe_texts = [
        "I'm feeling sad today",
        "I'm stressed about work",
        "I had a bad day"
    ]
    
    def check_crisis(text):
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in crisis_keywords)
    
    print("   Crisis detection results:")
    for text in test_crisis_texts:
        is_crisis = check_crisis(text)
        print(f"   🚨 '{text}' -> Crisis: {is_crisis}")
    
    print("\n   Safe text results:")
    for text in safe_texts:
        is_crisis = check_crisis(text)
        print(f"   ✓ '{text}' -> Crisis: {is_crisis}")
    
except Exception as e:
    print(f"✗ Crisis detection test failed: {e}")

# Test 7: Format analysis and debugging
try:
    print("\n7. Analyzing result format for debugging...")
    
    debug_text = "I'm happy today"
    debug_result = pipe(debug_text)
    
    print(f"   Debug text: '{debug_text}'")
    print(f"   Result type: {type(debug_result)}")
    print(f"   Result length: {len(debug_result) if hasattr(debug_result, '__len__') else 'N/A'}")
    print(f"   Full result: {debug_result}")
    
    if isinstance(debug_result, list):
        print(f"   First element type: {type(debug_result[0])}")
        if isinstance(debug_result[0], list):
            print("   Format: Nested list [[{emotions}]]")
            print(f"   Number of emotions: {len(debug_result[0])}")
            print(f"   Sample emotion: {debug_result[0][0] if debug_result[0] else 'None'}")
        else:
            print("   Format: Simple list [{emotions}]")
            print(f"   Number of emotions: {len(debug_result)}")
            print(f"   Sample emotion: {debug_result[0] if debug_result else 'None'}")
    
except Exception as e:
    print(f"✗ Format analysis failed: {e}")

print("\n" + "=" * 70)
print("Enhanced Model Testing Complete!")

# Summary
print("\n📊 SUMMARY:")
print("✓ If you see checkmarks above, the Twitter RoBERTa models are working!")
print("✓ Multi-label detection provides more nuanced emotion analysis")
print("✓ The model can detect multiple emotions simultaneously")
print("✓ Crisis detection provides safety features")

print("\n🔧 MODEL COMPARISON:")
print("- Multi-label: Detects all emotions with confidence scores")
print("- Single-label: Returns only the most confident emotion")
print("- Multi-label is better for complex emotional states")

print("\n💡 USAGE RECOMMENDATIONS:")
print("- Use multi-label for more empathetic responses")
print("- Set confidence thresholds around 0.3-0.4 for Twitter RoBERTa")
print("- Always implement crisis detection for safety")
print("- Consider fallback to keyword detection if model fails")

if 'pipe' in locals():
    try:
        sample_result = pipe('test')
        # Handle nested format
        if isinstance(sample_result, list) and len(sample_result) > 0:
            if isinstance(sample_result[0], list):
                emotions = sample_result[0]
            else:
                emotions = sample_result
        else:
            emotions = sample_result
        
        available_emotions = sorted(set(r['label'] for r in emotions))
        print(f"\n🎯 Available emotions: {available_emotions}")
    except Exception as e:
        print(f"\n🎯 Could not determine available emotions: {e}")