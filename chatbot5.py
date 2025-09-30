from flask import Flask, request, jsonify, render_template
import torch
import json
import random
import os

app = Flask(__name__)

# Import transformers, fallback to simple sentiment if not available
print("[+] Loading Twitter RoBERTa Multi-Label emotion detection model...")

try: 
    from transformers import pipeline
    HF_AVAILABLE = True
    
    # Using Twitter RoBERTa model for multi-label emotion detection
    emotion_classifier = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest",
        return_all_scores=True  # Get all emotion scores
    )
    print("[+] Twitter RoBERTa multi-label emotion model loaded successfully!")
    
    # Test the model to get available emotions
    try:
        test_result = emotion_classifier('test')
        if isinstance(test_result, list) and len(test_result) > 0:
            if isinstance(test_result[0], list):
                sample_emotions = test_result[0]
            else:
                sample_emotions = test_result
        else:
            sample_emotions = test_result
        
        ROBERTA_EMOTIONS = sorted(set(r['label'] for r in sample_emotions))
        print(f"[+] Available emotions: {ROBERTA_EMOTIONS}")
    except Exception as e:
        print(f"[WARNING] Could not determine available emotions: {e}")
        ROBERTA_EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
    
except ImportError as e:
    HF_AVAILABLE = False
    print(f"[ERROR] HuggingFace transformers not available: {e}")
    print("[INFO] Please install transformers with: pip install transformers torch")
    print("[FALLBACK] Reverting to simple keyword-based sentiment analysis")
    emotion_classifier = None
    ROBERTA_EMOTIONS = []
except Exception as e:
    HF_AVAILABLE = False
    print(f"[ERROR] Failed to load Twitter RoBERTa model: {e}")
    print(f"[ERROR] Error type: {type(e).__name__}")
    print("[POSSIBLE CAUSES]:")
    print("  - Internet connection required for first-time model download")
    print("  - Insufficient disk space for model files (~500MB)")
    print("  - PyTorch not installed: pip install torch")
    print("  - Transformers version incompatibility: pip install --upgrade transformers")
    print("[FALLBACK] Using keyword-based emotion detection")
    emotion_classifier = None
    ROBERTA_EMOTIONS = []

def load_templates():
    templates_file = os.path.join(os.path.dirname(__file__), 'templates.json')
    try:
        with open(templates_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "sad": {
                "acknowledge": ["I'm sorry you're feeling this way.", "That sounds really tough.", "I can hear the sadness in your words."],
                "support": ["Would you like to talk about what's making you feel this way?", "Sometimes sharing can help lighten the burden.", "I'm here to listen without judgment."],
                "reinforce": ["You're brave for sharing this.", "Your feelings are completely valid.", "You're not alone in this."]
            },
            "happy": {
                "acknowledge": ["That's wonderful!", "I can feel your joy!", "Your happiness is contagious!"],
                "support": ["Tell me more about what's bringing you joy!", "What made this moment so special?", "I'd love to hear the details!"],
                "reinforce": ["You deserve this happiness!", "Keep celebrating these beautiful moments!", "Your joy brightens the conversation!"]
            },
            "stressed": {
                "acknowledge": ["That sounds overwhelming.", "I can feel the stress in your message.", "That's a lot to handle."],
                "support": ["Let's break this down into manageable pieces.", "Would some grounding techniques help?", "What feels most urgent right now?"],
                "reinforce": ["You're handling so much with grace.", "It's wise to recognize when you're stressed.", "You've overcome challenges before."]
            },
            "angry": {
                "acknowledge": ["I can sense your frustration.", "That sounds really aggravating.", "Your anger is understandable."],
                "support": ["What's at the core of this frustration?", "Would it help to talk through what happened?", "Sometimes expressing anger helps process it."],
                "reinforce": ["Your feelings are valid.", "It's healthy to acknowledge when you're upset.", "You have every right to feel this way."]
            },
            "fearful": {
                "acknowledge": ["That sounds frightening.", "I can understand why you'd feel scared.", "Fear can be so overwhelming."],
                "support": ["What would help you feel safer right now?", "Let's think about what you can control.", "Would breathing exercises help calm your mind?"],
                "reinforce": ["It takes courage to face your fears.", "You're stronger than you know.", "Acknowledging fear is the first step to addressing it."]
            },
            "love": {
                "acknowledge": ["What a beautiful feeling!", "Love is such a powerful emotion.", "I can feel the warmth in your words."],
                "support": ["Tell me about this special connection!", "What makes this love so meaningful?", "How does this love impact your life?"],
                "reinforce": ["Love is one of life's greatest gifts.", "You deserve to give and receive love.", "These connections enrich our lives."]
            },
            "optimistic": {
                "acknowledge": ["Your optimism is inspiring!", "I love your positive outlook!", "What a hopeful perspective!"],
                "support": ["What's fueling this positive energy?", "Share more about what's got you feeling hopeful!", "Your optimism might inspire others too!"],
                "reinforce": ["Your positive attitude is a strength.", "Optimism can be a powerful force.", "Keep nurturing that hopeful spirit!"]
            },
            "anticipation": {
                "acknowledge": ["I can feel your excitement!", "That anticipation is palpable!", "Looking forward to something special?"],
                "support": ["Tell me more about what you're anticipating!", "What's got you feeling this way?", "Share your excitement with me!"],
                "reinforce": ["Anticipation can be such a wonderful feeling.", "It's beautiful to have something to look forward to.", "Your excitement is contagious!"]
            },
            "trust": {
                "acknowledge": ["That trust is precious.", "What a meaningful connection.", "Trust is such a valuable emotion."],
                "support": ["Tell me about this trust you're feeling.", "What makes you feel so confident?", "That's a beautiful foundation to build on."],
                "reinforce": ["Trust is one of life's greatest gifts.", "You deserve trustworthy connections.", "That faith in others is admirable."]
            },
            "surprise": {
                "acknowledge": ["What a surprise!", "I can feel your amazement!", "That must have been unexpected!"],
                "support": ["Tell me more about what surprised you!", "How did that make you feel?", "Share the details of this surprise!"],
                "reinforce": ["Life's surprises can be wonderful.", "It's great to stay open to the unexpected.", "Surprises add excitement to life!"]
            },
            "neutral": {
                "acknowledge": ["I'm here to listen.", "Thank you for sharing with me.", "Your thoughts matter."],
                "support": ["What's on your mind today?", "How are you feeling right now?", "Is there anything you'd like to explore?"],
                "reinforce": ["Your feelings are important.", "I'm glad you're here.", "You deserve support and understanding."]
            }
        }

# Enhanced keyword detection for multi-label emotions
EMOTION_KEYWORDS = {
    "sadness": ["sad", "depressed", "down", "cry", "crying", "hurt", "pain", "alone", "lonely", "worthless", "empty", "hopeless"],
    "joy": ["happy", "joy", "excited", "great", "amazing", "wonderful", "perfect", "awesome", "fantastic", "thrilled"],
    "anger": ["angry", "mad", "furious", "irritated", "annoyed", "frustrated", "rage", "pissed", "livid", "outraged"],
    "fear": ["scared", "afraid", "terrified", "frightened", "anxious", "worried", "nervous", "panic", "dread", "fear"],
    "love": ["love", "adore", "cherish", "treasure", "devoted", "affection", "romantic", "heart", "beloved", "darling"],
    "optimism": ["hopeful", "positive", "optimistic", "confident", "bright", "encouraging", "upbeat", "promising", "cheerful"],
    "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned", "unexpected", "wow", "incredible", "unbelievable"],
    "anticipation": ["anticipating", "expecting", "looking forward", "awaiting", "excited about", "can't wait", "upcoming", "future"],
    "trust": ["trust", "reliable", "dependable", "faith", "confidence", "believe in", "count on", "secure"],
    "disgust": ["disgusting", "revolting", "repulsive", "gross", "sick", "nauseating", "appalling", "vile"],
    "pessimism": ["pessimistic", "negative", "doubtful", "hopeless", "gloomy", "despairing", "bleak", "dark"]
}

# Enhanced crisis keywords for safety
CRISIS_KEYWORDS = [
    # Direct suicidal ideation
    'suicide', 'kill myself', 'end it all', 'hurt myself', 
    'die', 'not worth living', 'want to die', 'end my life',
    'self harm', 'cut myself', 'overdose', 'jump off',
    'take my own life', 'commit suicide', 'suicidal thoughts',
    'better off dead', 'wish I was dead', 'want to be dead',
    
    # Self-harm expressions
    'self injury', 'cutting myself', 'burning myself',
    'hitting myself', 'self abuse', 'self punishment',
    
    # Hopelessness indicators
    'no point in living', 'life has no meaning', 'everything is hopeless',
    'nothing matters anymore', 'give up on life', 'life is pointless',
    'no reason to continue', 'tired of existing', 'done with everything',
    
    # Method-specific
    'pills to die', 'hanging myself', 'bridge jumping',
    'slit my wrists', 'razor blade', 'poison myself',
    
    # Shortened versions
    'kms', 'kys', 'end me', 'checking out', 'logging off forever'
]

def detect_emotion_simple(text):
    """Enhanced keyword-based emotion detection for multi-label emotions"""
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    
    if emotion_scores:
        # Return all emotions as list format to match multi-label model
        all_emotions = []
        total_score = sum(emotion_scores.values())
        
        for emotion, score in emotion_scores.items():
            confidence = min(score / total_score, 0.85)  # Normalize and cap confidence
            all_emotions.append({"label": emotion, "score": confidence})
        
        # Sort by confidence
        all_emotions.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            "primary_emotion": all_emotions[0]["label"],
            "confidence": all_emotions[0]["score"],
            "all_emotions": all_emotions,
            "model": "keyword-fallback"
        }
    else:
        return {
            "primary_emotion": "neutral",
            "confidence": 0.5,
            "all_emotions": [{"label": "neutral", "score": 0.5}],
            "model": "keyword-fallback"
        }
    
def detect_emotion(text):
    """Detect emotion using Twitter RoBERTa multi-label model with improved format handling"""
    if HF_AVAILABLE and emotion_classifier:
        try:
            # The pipeline returns all emotion scores in format:
            # [[{'label': 'anger', 'score': 0.018}, {'label': 'joy', 'score': 0.881}, ...]]
            # OR [{emotions}] depending on transformers version
            results = emotion_classifier(text)
            
            # Handle nested list structure - FIXED format handling
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    all_emotions = results[0]  # Extract from nested list: [[{emotions}]]
                else:
                    all_emotions = results  # Simple list format: [{emotions}]
            else:
                all_emotions = results
            
            # Debug logging for format issues
            if not all_emotions or not isinstance(all_emotions, list):
                print(f"[DEBUG] Unexpected emotion format: {type(all_emotions)} - {all_emotions}")
                return detect_emotion_simple(text)
            
            # Sort emotions by confidence score
            all_emotions.sort(key=lambda x: x['score'], reverse=True)
            
            # Get primary emotion (highest confidence)
            primary_emotion = all_emotions[0]['label'].lower()
            primary_confidence = round(all_emotions[0]['score'], 3)
            
            # Filter emotions with meaningful scores (> 0.1)
            significant_emotions = [e for e in all_emotions if e['score'] > 0.1]
            
            return {
                "primary_emotion": primary_emotion,
                "confidence": primary_confidence,
                "all_emotions": all_emotions,
                "significant_emotions": significant_emotions,
                "model": "twitter-roberta-multilabel"
            }
            
        except Exception as e:
            print(f"[ERROR] Twitter RoBERTa emotion detection failed: {e}")
            print(f"[ERROR] Error type: {type(e).__name__}")
            # Debug information
            try:
                debug_result = emotion_classifier(text)
                print(f"[DEBUG] Raw result type: {type(debug_result)}")
                print(f"[DEBUG] Raw result: {debug_result}")
            except:
                print("[DEBUG] Could not get debug result")
            
            # Fallback to keyword detection
            fallback_result = detect_emotion_simple(text)
            print("[INFO] Using keyword fallback for emotion detection")
            return fallback_result
    else:
        fallback_result = detect_emotion_simple(text)
        return fallback_result

def check_crisis(text):
    """Enhanced crisis detection"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

def select_response(emotion_data, user_message, templates):
    """Select appropriate empathic response based on multi-label emotions"""
    
    # Crisis intervention takes priority
    if check_crisis(user_message):
        return {
            "response": "🚨 I'm really concerned about what you've shared. Please reach out to someone who can help:\n\n" +
                       "📞 National Suicide Prevention Lifeline: 988\n" +
                       "💬 Crisis Text Line: Text HOME to 741741\n" +
                       "🌐 International: befrienders.org\n" +
                       "🏥 Emergency Services: 911\n\n" +
                       "You matter, and there are people who want to support you. Please don't hesitate to reach out.",
            "type": "crisis_intervention",
            "emotion": emotion_data["primary_emotion"],
            "confidence": emotion_data["confidence"],
            "all_emotions": emotion_data.get("all_emotions", [])
        }
    
    primary_emotion = emotion_data["primary_emotion"]
    confidence = emotion_data["confidence"]
    all_emotions = emotion_data.get("all_emotions", [])
    significant_emotions = emotion_data.get("significant_emotions", [])
    
    # Enhanced mapping for multi-label emotions to template categories
    emotion_mapping = {
        # Direct mappings to existing templates
        "sadness": "sad",
        "joy": "happy",
        "anger": "angry",
        "fear": "fearful",
        "love": "love",
        "optimism": "optimistic",
        "anticipation": "anticipation",
        "trust": "trust",
        "surprise": "surprise",
        
        # Complex mappings
        "disgust": "angry",  # Treat disgust as a form of anger
        "pessimism": "sad",  # Treat pessimism as sadness
        
        # Fallbacks for keyword detection
        "happy": "happy",
        "sad": "sad",
        "fearful": "fearful",
        "stressed": "stressed"
    }
    
    mapped_emotion = emotion_mapping.get(primary_emotion, "neutral")
    
    # Use mapped emotion if it exists in templates
    if mapped_emotion in templates:
        emotion_key = mapped_emotion
    elif primary_emotion in templates:
        emotion_key = primary_emotion
    else:
        emotion_key = "neutral"
    
    # Adjust confidence thresholds for multi-label model
    confidence_threshold = 0.3 if emotion_data.get("model") == "twitter-roberta-multilabel" else 0.6
    
    # Select response strategy based on confidence
    if confidence > confidence_threshold:
        # High confidence - use specific empathic response
        strategies = templates[emotion_key]
        
        # Consider secondary emotions for more nuanced responses
        secondary_emotions = [e for e in significant_emotions[1:3] if e['score'] > 0.2] if significant_emotions else []
        
        # Combine strategies for full empathic response
        acknowledge = random.choice(strategies["acknowledge"])
        support = random.choice(strategies["support"]) 
        reinforce = random.choice(strategies["reinforce"])
        
        response = f"{acknowledge} {support} {reinforce}"
        
        # Add secondary emotion acknowledgment if present
        if secondary_emotions and len(secondary_emotions) > 0:
            secondary_emotion = secondary_emotions[0]['label']
            secondary_mapped = emotion_mapping.get(secondary_emotion, secondary_emotion)
            if secondary_mapped in templates and secondary_mapped != emotion_key:
                secondary_acknowledge = random.choice(templates[secondary_mapped]["acknowledge"])
                response += f" I also sense some {secondary_emotion} - {secondary_acknowledge.lower()}"
        
    else:
        # Lower confidence - use neutral supportive response
        response = random.choice(templates["neutral"]["acknowledge"])
    
    return {
        "response": response,
        "type": "empathic",
        "emotion": primary_emotion,
        "confidence": confidence,
        "mapped_emotion": emotion_key,
        "model_used": emotion_data.get("model", "unknown"),
        "all_emotions": all_emotions,
        "significant_emotions": significant_emotions,
        "secondary_emotions": [e for e in significant_emotions[1:3] if e['score'] > 0.2] if significant_emotions else []
    }

# Flask routes for web interface
@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Load templates
        templates = load_templates()
        
        # Detect emotion using multi-label RoBERTa
        emotion_data = detect_emotion(user_message)
        
        # Generate response
        response_data = select_response(emotion_data, user_message, templates)
        
        return jsonify({
            "bot_response": response_data["response"],
            "emotion_detected": response_data["emotion"],
            "confidence": response_data["confidence"],
            "response_type": response_data["type"],
            "mapped_emotion": response_data["mapped_emotion"],
            "model_used": response_data.get("model_used", "unknown"),
            "all_emotions": response_data.get("all_emotions", []),
            "significant_emotions": response_data.get("significant_emotions", []),
            "secondary_emotions": response_data.get("secondary_emotions", []),
            "user_message": user_message
        })
        
    except Exception as e:
        print(f"[ERROR] Chat endpoint error: {e}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": emotion_classifier is not None,
        "hf_available": HF_AVAILABLE,
        "model_name": "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest" if HF_AVAILABLE else "keyword-fallback",
        "supported_emotions": ROBERTA_EMOTIONS if HF_AVAILABLE else list(EMOTION_KEYWORDS.keys()),
        "multi_label": True
    })

@app.route('/api/demo', methods=['GET'])
def demo_scenarios():
    """Get demo scenarios for testing"""
    scenarios = [
        {
            "name": "Sadness Scenario",
            "input": "I feel so alone and worthless today",
            "description": "Demonstrating empathic response to sadness",
            "expected_emotion": "sadness"
        },
        {
            "name": "Joy Scenario", 
            "input": "I just got accepted to my dream university! I'm over the moon!",
            "description": "Demonstrating celebratory empathic response",
            "expected_emotion": "joy"
        },
        {
            "name": "Anger Scenario",
            "input": "I'm absolutely furious! This is outrageous and unacceptable!",
            "description": "Demonstrating supportive response to anger",
            "expected_emotion": "anger"
        },
        {
            "name": "Fear Scenario",
            "input": "I'm terrified about my presentation tomorrow, what if I mess up?",
            "description": "Demonstrating calming response to fear",
            "expected_emotion": "fear"
        },
        {
            "name": "Love Scenario",
            "input": "I'm head over heels in love, everything feels magical!",
            "description": "Demonstrating warm response to love",
            "expected_emotion": "love"
        },
        {
            "name": "Multi-emotion Scenario",
            "input": "I'm nervous but excited about starting my new job tomorrow!",
            "description": "Testing multi-emotion detection",
            "expected_emotion": "anticipation"
        },
        {
            "name": "Trust Scenario",
            "input": "I have complete faith that everything will work out perfectly",
            "description": "Testing trust emotion detection",
            "expected_emotion": "trust"
        }
    ]
    
    return jsonify({"scenarios": scenarios})

# Terminal functions (enhanced for multi-label detection)
def print_header():
    """Print chatbot header"""
    print("=" * 70)
    print("🤖 EMPATHIC CHATBOT - Multi-Label Emotion Detection")
    print("=" * 70)
    print("💙 I'm here to listen and respond with empathy")
    print("🔍 Using Twitter RoBERTa for multi-label emotion detection")
    print("🎭 Can detect multiple emotions simultaneously!")
    print("🆘 Type 'help' for commands or 'quit' to exit")
    print("=" * 70)
    print()

def print_help():
    """Print help information"""
    print("\n📋 AVAILABLE COMMANDS:")
    print("  help     - Show this help message")
    print("  quit     - Exit the chatbot")
    print("  clear    - Clear the screen")
    print("  demo     - Run demonstration scenarios")
    print("  stats    - Show emotion detection statistics")
    print("  emotions - Show supported emotions")
    print("  test     - Test emotion detection format")
    print("\n💡 EXAMPLE MESSAGES TO TRY:")
    print("  'I feel so overwhelmed with everything'")
    print("  'I just got the best news ever!'")
    print("  'I'm absolutely furious about this situation'")
    print("  'I'm scared but excited about tomorrow'")
    print("  'I'm so in love with life right now'")
    print()

def show_supported_emotions():
    """Show supported emotions"""
    print("\n🎭 SUPPORTED EMOTIONS:")
    if HF_AVAILABLE:
        print("   Using Twitter RoBERTa multi-label model:")
        for i, emotion in enumerate(ROBERTA_EMOTIONS, 1):
            print(f"   {i:2d}. {emotion.title()}")
        print("\n   ✨ Multi-label detection: Can detect multiple emotions simultaneously!")
        print("   📊 Each emotion gets a confidence score from 0-100%")
    else:
        print("   Using keyword-based detection:")
        for i, emotion in enumerate(EMOTION_KEYWORDS.keys(), 1):
            print(f"   {i:2d}. {emotion.title()}")
    print()

def test_emotion_format():
    """Test emotion detection format for debugging"""
    print("\n🔍 TESTING EMOTION DETECTION FORMAT:")
    print("=" * 50)
    
    test_text = "I'm happy but nervous"
    print(f"Test input: '{test_text}'")
    
    try:
        emotion_data = detect_emotion(test_text)
        print(f"✓ Detection successful!")
        print(f"  Primary emotion: {emotion_data['primary_emotion']}")
        print(f"  Confidence: {emotion_data['confidence']:.3f}")
        print(f"  Model: {emotion_data.get('model', 'unknown')}")
        
        if emotion_data.get('all_emotions'):
            print(f"  All emotions detected:")
            for emotion in emotion_data['all_emotions'][:5]:
                print(f"    - {emotion['label']}: {emotion['score']:.3f}")
        
        if emotion_data.get('significant_emotions'):
            print(f"  Significant emotions (>0.1):")
            for emotion in emotion_data['significant_emotions']:
                print(f"    - {emotion['label']}: {emotion['score']:.3f}")
                
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        print(f"  Error type: {type(e).__name__}")
    
    print("=" * 50)

def run_demo(templates):
    """Run enhanced demonstration scenarios"""
    demo_scenarios = [
        {
            "name": "Sadness Scenario",
            "input": "I feel so alone and worthless today",
            "description": "Demonstrating empathic response to sadness",
            "expected_emotion": "sadness"
        },
        {
            "name": "Joy Scenario", 
            "input": "I just got accepted to my dream university! I'm over the moon!",
            "description": "Demonstrating celebratory empathic response",
            "expected_emotion": "joy"
        },
        {
            "name": "Anger Scenario",
            "input": "I'm absolutely furious! This is outrageous and unacceptable!",
            "description": "Demonstrating supportive response to anger",
            "expected_emotion": "anger"
        },
        {
            "name": "Fear Scenario",
            "input": "I'm terrified about my presentation tomorrow, what if I mess up?",
            "description": "Demonstrating calming response to fear",
            "expected_emotion": "fear"
        },
        {
            "name": "Love Scenario",
            "input": "I'm head over heels in love, everything feels magical!",
            "description": "Demonstrating warm response to love",
            "expected_emotion": "love"
        },
        {
            "name": "Multi-emotion Scenario",
            "input": "I'm nervous but excited about starting my new job tomorrow!",
            "description": "Testing multi-emotion detection",
            "expected_emotion": "anticipation"
        },
        {
            "name": "Trust Scenario",
            "input": "I have complete faith that everything will work out perfectly",
            "description": "Testing trust emotion detection",
            "expected_emotion": "trust"
        }
    ]
    
    print("\n🎭 RUNNING ENHANCED DEMO SCENARIOS:")
    print("=" * 80)
    
    emotion_counts = {}
    
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        expected = scenario.get('expected_emotion', '')
        print(f"   Expected: {expected}")
        print(f"   Description: {scenario['description']}")
        print(f"   User Input: \"{scenario['input']}\"")
        
        # Process the demo input
        emotion_data = detect_emotion(scenario['input'])
        response_data = select_response(emotion_data, scenario['input'], templates)
        
        # Track emotion detection
        primary_emotion = emotion_data['primary_emotion']
        emotion_counts[primary_emotion] = emotion_counts.get(primary_emotion, 0) + 1
        
        # Check if detected matches expected
        match_indicator = "✅" if primary_emotion == expected else "❓"
        
        model_info = f" ({emotion_data.get('model', 'unknown')})" if 'model' in emotion_data else ""
        print(f"   {match_indicator} Primary: {primary_emotion} ({emotion_data['confidence']:.1%} confidence){model_info}")
        
        # Show significant emotions if using multi-label model
        if emotion_data.get('significant_emotions') and len(emotion_data['significant_emotions']) > 1:
            print(f"   🎭 Significant emotions detected:")
            for emotion_info in emotion_data['significant_emotions'][:5]:  # Show top 5
                print(f"      - {emotion_info['label']}: {emotion_info['score']:.1%}")
        
        print(f"   🔄 Mapped to template: {response_data.get('mapped_emotion', 'unknown')}")
        print(f"   💙 Bot Response: {response_data['response'][:120]}...")
        print("-" * 80)
    
    print(f"\n📊 EMOTION DETECTION SUMMARY:")
    print("=" * 40)
    for emotion, count in sorted(emotion_counts.items()):
        print(f"   {emotion.title()}: {count} detections")
    
    print(f"\n🎯 AVAILABLE EMOTIONS IN MODEL:")
    if HF_AVAILABLE:
        print("   Multi-label RoBERTa emotions:", ROBERTA_EMOTIONS)
    else:
        print("   Keyword emotions:", list(EMOTION_KEYWORDS.keys()))
    
    print("\n✅ Enhanced demo completed! Multi-emotion detection is now active.\n")

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Enhanced main chatbot loop"""
    templates = load_templates()
    conversation_count = 0
    emotion_stats = {}
    
    # clear_screen() # remove for debugging
    print_header()
    
    if HF_AVAILABLE:
        print("🤖 Hello! I'm an empathic chatbot powered by multi-label emotion detection.")
        print("   I can detect multiple emotions in your messages simultaneously!")
    else:
        print("🤖 Hello! I'm an empathic chatbot using keyword-based emotion detection.")
    
    print("   How are you feeling today? (Type 'help' for commands)")
    
    while True:
        try:
            # Get user input
            print("\n" + "─" * 50)
            user_input = input("💬 You: ").strip()
            
            # Handle empty input
            if not user_input:
                print("🤖 Bot: I'm here when you're ready to share.")
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                print("\n🤖 Bot: Thank you for chatting with me. Take care! 💙")
                print(f"📊 We had {conversation_count} conversations today.")
                if emotion_stats:
                    print("📈 Most detected emotions:")
                    sorted_emotions = sorted(emotion_stats.items(), key=lambda x: x[1], reverse=True)
                    for emotion, count in sorted_emotions[:3]:
                        print(f"   {emotion.title()}: {count} times")
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'clear':
                clear_screen()
                print_header()
                continue
            elif user_input.lower() == 'demo':
                run_demo(templates)
                continue
            elif user_input.lower() == 'emotions':
                show_supported_emotions()
                continue
            elif user_input.lower() == 'test':
                test_emotion_format()
                continue
            elif user_input.lower() == 'stats':
                if emotion_stats:
                    print("\n📊 EMOTION DETECTION STATISTICS:")
                    for emotion, count in sorted(emotion_stats.items()):
                        print(f"   {emotion.title()}: {count} times")
                else:
                    print("\n📊 No conversations yet. Start chatting to see stats!")
                continue
            
            # Process regular message
            conversation_count += 1
            
            # Detect emotion
            emotion_data = detect_emotion(user_input)
            
            # Update stats
            primary_emotion = emotion_data['primary_emotion']
            emotion_stats[primary_emotion] = emotion_stats.get(primary_emotion, 0) + 1
            
            # Generate response
            response_data = select_response(emotion_data, user_input, templates)
            
            # Display response with enhanced emotion info
            model_info = f" ({emotion_data.get('model', 'unknown')})" if 'model' in emotion_data else ""
            print(f"\n🔍 Primary: {primary_emotion} ({emotion_data['confidence']:.1%} confidence){model_info}")
            
            # Show significant emotions if multiple detected
            if emotion_data.get('significant_emotions') and len(emotion_data['significant_emotions']) > 1:
                print(f"🎭 Significant emotions detected:")
                for emotion_info in emotion_data['significant_emotions'][:3]:
                    print(f"   - {emotion_info['label']}: {emotion_info['score']:.1%}")
            
            if response_data.get('mapped_emotion') != primary_emotion:
                print(f"🔄 Mapped to: {response_data.get('mapped_emotion', 'unknown')}")
            
            if response_data['type'] == 'crisis_intervention':
                print("🚨 CRISIS RESPONSE:")
                print(response_data['response'])
            else:
                print(f"🤖 Bot: {response_data['response']}")
            
        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Goodbye! Take care of yourself. 💙")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("🤖 Bot: I'm having some technical difficulties. Please try again.")

if __name__ == "__main__":
    import sys
    
    # Check if running as Flask app or terminal
    if len(sys.argv) > 1 and sys.argv[1] == '--web':
        print("🤖 Starting Empathic Chatbot Web Interface...")
        print("💡 Open http://localhost:5000 in your browser")
        print("🌐 To share: run 'ngrok http 5000' in another terminal")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        # Run terminal version by default
        main()