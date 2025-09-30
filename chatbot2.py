from flask import Flask, request, jsonify, render_template
import json
import random
import os

# Import transformers, fallback to simple sentiment if not available

print("[+] Loading emotion detection model...")

try: 
  from transformers import pipeline
  HF_AVAILABLE = True
  emotion_classifier = pipeline(
  "text-classification",
  model="j-hartmann/emotion-english-distilroberta-base",
  return_all_scores=False
  )
  print("[+] Model loaded.")
except ImportError:
  HF_AVAILABLE = False
  print("[HuggingFace not available, reverting to simple sentiment analysis.]")
  emotion_classifier = None


def load_templates():
  templates_file = os.path.join(os.path.dirname(__file__), 'templates.json')
  try:
    with open(templates_file, 'r') as f:
      return json.load(f)
  except FileNotFoundError:
    return {
        "sad": {
            "acknowledge": ["I'm sorry you're feeling this way.", "That sounds really tough."],
            "support": ["Would you like to talk about it?", "Maybe taking a break could help?"],
            "reinforce": ["You're brave for sharing this.", "You're not alone."]
        },
        "happy": {
            "acknowledge": ["That's wonderful!", "I can feel your joy!"],
            "support": ["Tell me more!", "What made this so special?"],
            "reinforce": ["You deserve this happiness!", "Keep celebrating!"]
        },
        "stressed": {
            "acknowledge": ["That sounds overwhelming.", "I can feel your stress."],
            "support": ["Let's break this down.", "Try taking deep breaths."],
            "reinforce": ["You're handling a lot.", "You've got this."]
        },
        "neutral": {
            "acknowledge": ["I'm here to listen.", "Thank you for sharing."],
            "support": ["What's on your mind?", "How are you feeling?"],
            "reinforce": ["Your feelings matter.", "I'm glad you're here."]
        }
    }

# Fallback semantics for basic sentiment detection
EMOTION_KEYWORDS = {
    "sad": ["sad", "depressed", "down", "cry", "crying", "hurt", "pain", "alone", "lonely", "worthless"],
    "happy": ["happy", "joy", "excited", "great", "amazing", "wonderful", "love", "perfect", "awesome", "fantastic"],
    "stressed": ["stress", "stressed", "anxious", "worry", "worried", "overwhelmed", "panic", "nervous", "pressure", "deadline"]
}

# Crisis keywords for safety
CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end it all', 'hurt myself', 
    'die', 'not worth living', 'want to die', 'end my life'
]

def detect_emotion_simple(text):
    """Simple keyword-based emotion detection"""
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    
    if emotion_scores:
        # Return emotion with highest score
        best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        confidence = min(best_emotion[1] * 0.3, 0.95)  # Scale to reasonable confidence
        return {"label": best_emotion[0], "score": confidence}
    else:
        return {"label": "neutral", "score": 0.5}
    
def detect_emotion(text):
    """Detect emotion in user text"""
    if HF_AVAILABLE and emotion_classifier:
        try:
            result = emotion_classifier(text)[0]
            return {
                "label": result['label'].lower(),
                "score": round(result['score'], 3)
            }
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return detect_emotion_simple(text)
    else:
        return detect_emotion_simple(text)

def check_crisis(text):
    """Check for crisis language"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

def select_response(emotion_data, user_message, templates):
    """Select appropriate empathic response"""
    
    # Crisis intervention takes priority
    if check_crisis(user_message):
        return {
            "response": "🚨 I'm really concerned about what you've shared. Please reach out to someone who can help:\n" +
                       "   📞 National Suicide Prevention Lifeline: 988\n" +
                       "   💬 Crisis Text Line: Text HOME to 741741\n" +
                       "   🌐 Or visit your local emergency services\n\n" +
                       "You matter, and there are people who want to support you.",
            "type": "crisis_intervention",
            "emotion": emotion_data["label"],
            "confidence": emotion_data["score"]
        }
    
    emotion = emotion_data["label"]
    confidence = emotion_data["score"]
    
    # Map HuggingFace emotions to our template categories
    emotion_mapping = {
        "sadness": "sad",
        "joy": "happy", 
        "fear": "stressed",
        "anger": "stressed",
        "disgust": "stressed",
        "surprise": "happy"
    }
    
    mapped_emotion = emotion_mapping.get(emotion, emotion)
    
    # Use mapped emotion if it exists in templates, otherwise use original or neutral
    if mapped_emotion in templates:
        emotion_key = mapped_emotion
    elif emotion in templates:
        emotion_key = emotion
    else:
        emotion_key = "neutral"
    
    # Select response strategy based on confidence
    if confidence > 0.6:
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
        "mapped_emotion": emotion_key
    }

def print_header():
    """Print chatbot header"""
    print("=" * 60)
    print("🤖 EMPATHIC CHATBOT - Terminal Version")
    print("=" * 60)
    print("💙 I'm here to listen and respond with empathy")
    print("🔍 I can detect emotions and provide supportive responses")
    print("🆘 Type 'help' for commands or 'quit' to exit")
    print("=" * 60)
    print()

def print_help():
    """Print help information"""
    print("\n📋 AVAILABLE COMMANDS:")
    print("  help     - Show this help message")
    print("  quit     - Exit the chatbot")
    print("  clear    - Clear the screen")
    print("  demo     - Run demonstration scenarios")
    print("  stats    - Show emotion detection statistics")
    print("\n💡 EXAMPLE MESSAGES TO TRY:")
    print("  'I feel so overwhelmed with everything'")
    print("  'I just got the best news ever!'")
    print("  'I feel so alone and sad today'")
    print("  'I'm stressed about my exams tomorrow'")
    print()

def run_demo(templates):
    """Run demonstration scenarios"""
    demo_scenarios = [
        {
            "name": "Sadness Scenario",
            "input": "I feel so alone and worthless today",
            "description": "Demonstrating empathic response to sadness"
        },
        {
            "name": "Happiness Scenario", 
            "input": "I just got accepted to my dream university!",
            "description": "Demonstrating celebratory empathic response"
        },
        {
            "name": "Stress Scenario",
            "input": "I have three exams tomorrow and I'm freaking out",
            "description": "Demonstrating supportive response to stress"
        }
    ]
    
    print("\n🎭 RUNNING DEMO SCENARIOS:")
    print("=" * 50)
    
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   User Input: \"{scenario['input']}\"")
        
        # Process the demo input
        emotion_data = detect_emotion(scenario['input'])
        response_data = select_response(emotion_data, scenario['input'], templates)
        
        print(f"   Detected Emotion: {emotion_data['label']} ({emotion_data['score']:.1%} confidence)")
        print(f"   Bot Response:")
        print(f"   💙 {response_data['response']}")
        print("-" * 50)
    
    print("\n✅ Demo completed! Try typing your own messages now.\n")

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Main chatbot loop"""
    templates = load_templates()
    conversation_count = 0
    emotion_stats = {}
    
    clear_screen()
    print_header()
    
    print("🤖 Hello! I'm an empathic chatbot. How are you feeling today?")
    print("   (Type 'help' for commands or just start chatting)")
    
    while True:
        try:
            # Get user input
            print("\n" + "─" * 40)
            user_input = input("💬 You: ").strip()
            
            # Handle empty input
            if not user_input:
                print("🤖 Bot: I'm here when you're ready to share.")
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                print("\n🤖 Bot: Thank you for chatting with me. Take care! 💙")
                print(f"📊 We had {conversation_count} conversations today.")
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
            
            # Display response with emotion info
            print(f"\n🔍 Detected: {emotion_data['label']} ({emotion_data['score']:.1%} confidence)")
            
            if response_data['type'] == 'crisis_intervention':
                print("🤖 Bot:")
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
    main()