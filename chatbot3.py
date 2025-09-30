from flask import Flask, request, jsonify, render_template, datasets
import torch
import json
import random
import os

app = Flask(__name__)

# Import transformers, fallback to simple sentiment if not available
print("[+] Loading Twitter RoBERTa emotion detection model...")

try: 
    from transformers import pipeline
    HF_AVAILABLE = True
    
    # Using Twitter RoBERTa model for better social media/conversational emotion detection
    emotion_classifier = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-emotion",
        return_all_scores=False
    )
    print("[+] Twitter RoBERTa emotion model loaded successfully!")
    
    # The model outputs: joy, optimism, anger, sadness, fear, surprise, love, anticipation, trust, disgust, pessimism
    ROBERTA_EMOTIONS = ['joy', 'optimism', 'anger', 'sadness', 'fear', 'surprise', 'love', 'anticipation', 'trust', 'disgust', 'pessimism']
    
except ImportError:
    HF_AVAILABLE = False
    print("[HuggingFace not available, reverting to simple sentiment analysis.]")
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
            "neutral": {
                "acknowledge": ["I'm here to listen.", "Thank you for sharing with me.", "Your thoughts matter."],
                "support": ["What's on your mind today?", "How are you feeling right now?", "Is there anything you'd like to explore?"],
                "reinforce": ["Your feelings are important.", "I'm glad you're here.", "You deserve support and understanding."]
            }
        }

# Enhanced keyword detection for Twitter RoBERTa emotions
EMOTION_KEYWORDS = {
    "sad": ["sad", "depressed", "down", "cry", "crying", "hurt", "pain", "alone", "lonely", "worthless", "empty", "hopeless"],
    "happy": ["happy", "joy", "excited", "great", "amazing", "wonderful", "love", "perfect", "awesome", "fantastic", "thrilled"],
    "stressed": ["stress", "stressed", "anxious", "worry", "worried", "overwhelmed", "panic", "nervous", "pressure", "deadline", "tense"],
    "angry": ["angry", "mad", "furious", "irritated", "annoyed", "frustrated", "rage", "pissed", "livid", "outraged"],
    "fearful": ["scared", "afraid", "terrified", "frightened", "anxious", "worried", "nervous", "panic", "dread", "fear"],
    "love": ["love", "adore", "cherish", "treasure", "devoted", "affection", "romantic", "heart", "beloved", "darling"],
    "optimistic": ["hopeful", "positive", "optimistic", "confident", "bright", "encouraging", "upbeat", "promising", "cheerful"],
    "surprised": ["surprised", "shocked", "amazed", "astonished", "stunned", "unexpected", "wow", "incredible", "unbelievable"]
}

# Crisis keywords for safety
CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end it all', 'hurt myself', 
    'die', 'not worth living', 'want to die', 'end my life',
    'self harm', 'cut myself', 'overdose', 'jump off'
]

def detect_emotion_simple(text):
    """Enhanced keyword-based emotion detection for Twitter RoBERTa emotions"""
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    
    if emotion_scores:
        # Return emotion with highest score
        best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        confidence = min(best_emotion[1] * 0.25, 0.85)  # More conservative confidence
        return {"label": best_emotion[0], "score": confidence}
    else:
        return {"label": "neutral", "score": 0.5}
    
def detect_emotion(text):
    """Detect emotion using Twitter RoBERTa model"""
    if HF_AVAILABLE and emotion_classifier:
        try:
            result = emotion_classifier(text)[0]
            detected_emotion = result['label'].lower()
            confidence = round(result['score'], 3)
            
            return {
                "label": detected_emotion,
                "score": confidence,
                "model": "twitter-roberta"
            }
        except Exception as e:
            print(f"Error in Twitter RoBERTa emotion detection: {e}")
            fallback_result = detect_emotion_simple(text)
            fallback_result["model"] = "keyword-fallback"
            return fallback_result
    else:
        print("[yikes]")
        fallback_result = detect_emotion_simple(text)
        fallback_result["model"] = "keyword-fallback"
        return fallback_result

def check_crisis(text):
    """Enhanced crisis detection"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

def select_response(emotion_data, user_message, templates):
    """Select appropriate empathic response based on Twitter RoBERTa emotions"""
    
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
            "emotion": emotion_data["label"],
            "confidence": emotion_data["score"]
        }
    
    emotion = emotion_data["label"]
    confidence = emotion_data["score"]
    
    # Enhanced mapping for Twitter RoBERTa emotions to template categories
    emotion_mapping = {
        # Direct mappings
        "sadness": "sad",
        "joy": "happy",
        "anger": "angry",
        "fear": "fearful",
        "love": "love",
        "optimism": "optimistic",
        
        # Complex mappings
        "surprise": "happy",  # Treat surprise as generally positive
        "anticipation": "optimistic",  # Forward-looking emotion
        "trust": "love",  # Positive relationship emotion
        "disgust": "angry",  # Negative reactive emotion
        "pessimism": "sad",  # Negative outlook
        
        # Fallbacks for keyword detection
        "stressed": "stressed",
        "happy": "happy",
        "sad": "sad",
        "fearful": "fearful"
    }
    
    mapped_emotion = emotion_mapping.get(emotion, "neutral")
    
    # Use mapped emotion if it exists in templates
    if mapped_emotion in templates:
        emotion_key = mapped_emotion
    elif emotion in templates:
        emotion_key = emotion
    else:
        emotion_key = "neutral"
    
    # Adjust confidence thresholds for Twitter RoBERTa (generally more confident)
    confidence_threshold = 0.4 if emotion_data.get("model") == "twitter-roberta" else 0.6
    
    # Select response strategy based on confidence
    if confidence > confidence_threshold:
        # High confidence - use specific empathic response
        strategies = templates[emotion_key]
        
        # Combine all three strategies for full empathic response
        acknowledge = random.choice(strategies["acknowledge"])
        support = random.choice(strategies["support"]) 
        reinforce = random.choice(strategies["reinforce"])
        
        response = f"{acknowledge} {support} {reinforce}"
    else:
        # Lower confidence - use neutral supportive response
        response = random.choice(templates["neutral"]["acknowledge"])
    
    return {
        "response": response,
        "type": "empathic",
        "emotion": emotion,
        "confidence": confidence,
        "mapped_emotion": emotion_key,
        "model_used": emotion_data.get("model", "unknown")
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
        
        # Detect emotion using Twitter RoBERTa
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
            "user_message": user_message
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": emotion_classifier is not None,
        "hf_available": HF_AVAILABLE,
        "model_name": "cardiffnlp/twitter-roberta-base-emotion" if HF_AVAILABLE else "keyword-fallback",
        "supported_emotions": ROBERTA_EMOTIONS if HF_AVAILABLE else list(EMOTION_KEYWORDS.keys())
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
            "input": "I'm so frustrated with my boss, they never listen to my ideas!",
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
            "name": "Optimism Scenario",
            "input": "I have such a good feeling about this year, so many possibilities!",
            "description": "Demonstrating encouraging response to optimism",
            "expected_emotion": "optimism"
        }
    ]
    
    return jsonify({"scenarios": scenarios})

# Terminal functions (enhanced for Twitter RoBERTa)
def print_header():
    """Print chatbot header"""
    print("=" * 70)
    print("🤖 EMPATHIC CHATBOT - Twitter RoBERTa Enhanced")
    print("=" * 70)
    print("💙 I'm here to listen and respond with empathy")
    print("🔍 Using Twitter RoBERTa for advanced emotion detection")
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
    print("\n💡 EXAMPLE MESSAGES TO TRY:")
    print("  'I feel so overwhelmed with everything'")
    print("  'I just got the best news ever!'")
    print("  'I'm absolutely furious about this situation'")
    print("  'I'm scared about what might happen'")
    print("  'I'm so in love with life right now'")
    print()

def show_supported_emotions():
    """Show supported emotions"""
    print("\n🎭 SUPPORTED EMOTIONS:")
    if HF_AVAILABLE:
        print("   Using Twitter RoBERTa model:")
        for i, emotion in enumerate(ROBERTA_EMOTIONS, 1):
            print(f"   {i:2d}. {emotion.title()}")
    else:
        print("   Using keyword-based detection:")
        for i, emotion in enumerate(EMOTION_KEYWORDS.keys(), 1):
            print(f"   {i:2d}. {emotion.title()}")
    print()

def run_demo(templates):
    """Run enhanced demonstration scenarios"""
    demo_scenarios = [
        {
            "name": "Sadness Scenario",
            "input": "I feel so alone and worthless today",
            "description": "Demonstrating empathic response to sadness"
        },
        {
            "name": "Joy Scenario", 
            "input": "I just got accepted to my dream university! I'm over the moon!",
            "description": "Demonstrating celebratory empathic response"
        },
        {
            "name": "Anger Scenario",
            "input": "I'm so frustrated with my boss, they never listen to my ideas!",
            "description": "Demonstrating supportive response to anger"
        },
        {
            "name": "Fear Scenario",
            "input": "I'm terrified about my presentation tomorrow, what if I mess up?",
            "description": "Demonstrating calming response to fear"
        },
        {
            "name": "Love Scenario",
            "input": "I'm head over heels in love, everything feels magical!",
            "description": "Demonstrating warm response to love"
        },
        {
            "name": "Optimism Scenario",
            "input": "I have such a good feeling about this year, so many possibilities!",
            "description": "Demonstrating encouraging response to optimism"
        }
    ]
    
    print("\n🎭 RUNNING DEMO SCENARIOS:")
    print("=" * 60)
    
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   User Input: \"{scenario['input']}\"")
        
        # Process the demo input
        emotion_data = detect_emotion(scenario['input'])
        response_data = select_response(emotion_data, scenario['input'], templates)
        
        model_info = f" ({emotion_data.get('model', 'unknown')})" if 'model' in emotion_data else ""
        print(f"   Detected Emotion: {emotion_data['label']} ({emotion_data['score']:.1%} confidence){model_info}")
        print(f"   Mapped to Template: {response_data.get('mapped_emotion', 'unknown')}")
        print(f"   Bot Response:")
        print(f"   💙 {response_data['response']}")
        print("-" * 60)
    
    print("\n✅ Demo completed! Try typing your own messages now.\n")

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Enhanced main chatbot loop"""
    templates = load_templates()
    conversation_count = 0
    emotion_stats = {}
    
    clear_screen() # remove for debugging
    print_header()
    
    if HF_AVAILABLE:
        print("🤖 Hello! I'm an empathic chatbot powered by Twitter RoBERTa emotion detection.")
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
            emotion = emotion_data['label']
            emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1
            
            # Generate response
            response_data = select_response(emotion_data, user_input, templates)
            
            # Display response with enhanced emotion info
            model_info = f" ({emotion_data.get('model', 'unknown')})" if 'model' in emotion_data else ""
            print(f"\n🔍 Detected: {emotion_data['label']} ({emotion_data['score']:.1%} confidence){model_info}")
            if response_data.get('mapped_emotion') != emotion_data['label']:
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