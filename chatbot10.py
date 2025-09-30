import torch
import json
import random
import os

# Configuration
USE_ADVANCED_MODEL = True
print(f"[+] Advanced model: {USE_ADVANCED_MODEL}")

# Load emotion model
if USE_ADVANCED_MODEL:
    try: 
        from transformers import pipeline
        emotion_classifier = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest",
            return_all_scores=True
        )
        print("[+] RoBERTa model loaded")
        HF_AVAILABLE = True
    except Exception as e:
        print(f"[!] Model failed: {e}")
        emotion_classifier = None
        HF_AVAILABLE = False
else:
    emotion_classifier = None
    HF_AVAILABLE = False

# Emotion keywords for fallback
EMOTION_KEYWORDS = {
    "sadness": ["sad", "depressed", "down", "crying", "hurt", "alone", "lonely", "worthless", "hopeless"],
    "joy": ["happy", "excited", "great", "amazing", "wonderful", "awesome", "fantastic", "thrilled"],
    "anger": ["angry", "mad", "furious", "irritated", "frustrated", "rage", "outraged"],
    "fear": ["scared", "afraid", "terrified", "anxious", "worried", "nervous", "panic"],
    "love": ["love", "adore", "cherish", "devoted", "affection", "romantic", "heart"],
    "optimism": ["hopeful", "positive", "optimistic", "confident", "bright", "encouraging"],
    "surprise": ["surprised", "shocked", "amazed", "stunned", "unexpected", "incredible"],
    "anticipation": ["anticipating", "expecting", "looking forward", "excited about", "can't wait"],
    "trust": ["trust", "reliable", "dependable", "faith", "confidence", "believe in"],
    "disgust": ["disgusting", "revolting", "gross", "sick", "nauseating", "appalling"],
    "pessimism": ["pessimistic", "negative", "doubtful", "gloomy", "despairing", "bleak"]
}

# Crisis keywords
CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end it all', 'hurt myself', 'die', 'not worth living', 
    'want to die', 'end my life', 'self harm', 'cut myself', 'better off dead'
]

# Response templates (integrated from empathetic_chatbot.py)
RESPONSES = {
    'sad': [
        "I'm really sorry you're feeling this way. That sounds really tough.",
        "It's okay to feel sad sometimes. Your feelings are valid.",
        "I hear you, and I'm here for you. Would you like to talk about what's bothering you?",
        "That sounds really difficult. I'm sending you virtual support."
    ],
    'happy': [
        "That's wonderful to hear! I'm so happy for you!",
        "Your joy is contagious! Tell me more about what made you feel this way.",
        "That's fantastic! Celebrating your happiness with you!",
        "I love hearing about your positive experiences! What else made your day great?"
    ],
    'angry': [
        "I can sense your frustration. That sounds really aggravating.",
        "Your anger is understandable. What's at the core of this frustration?",
        "Sometimes expressing anger helps process it. Your feelings are valid."
    ],
    'fearful': [
        "That sounds frightening. I can understand why you'd feel scared.",
        "Fear can be so overwhelming. What would help you feel safer right now?",
        "It takes courage to face your fears. You're stronger than you know."
    ],
    'stressed': [
        "I understand that stress can be overwhelming. Would it help to take a few deep breaths?",
        "It sounds like you're under a lot of pressure. Remember to be kind to yourself.",
        "Stress can be really tough to handle. Have you tried taking a short break to reset?"
    ],
    'love': [
        "What a beautiful feeling! Love is such a powerful emotion.",
        "I can feel the warmth in your words. Tell me about this special connection!",
        "Love is one of life's greatest gifts. You deserve to give and receive love."
    ],
    'optimistic': [
        "Your optimism is inspiring! I love your positive outlook!",
        "What's fueling this positive energy? Your optimism might inspire others too!"
    ],
    'neutral': [
        "I'm here to listen. How are you feeling today?",
        "I care about how you're doing. Would you like to share what's on your mind?",
        "Your feelings matter to me. What's going on in your world right now?"
    ]
}

# Suggestions (from empathetic_chatbot.py)
SUGGESTIONS = {
    'sad': [
        "Would you like to try a short mindfulness exercise? It might help you feel a bit better.",
        "Sometimes writing down your feelings can help process them.",
        "Connecting with a friend or loved one might help. Is there someone you could reach out to?",
        "Taking a walk in nature can sometimes lift your mood."
    ],
    'happy': [
        "Why not celebrate this moment? Do something special to mark this happy time!",
        "Sharing your happiness with others can multiply the joy.",
        "Take a moment to really savor this feeling. What specifically made this so wonderful?"
    ],
    'stressed': [
        "Try the 4-7-8 breathing technique: inhale for 4 seconds, hold for 7, exhale for 8.",
        "Breaking your tasks into smaller steps might make them feel more manageable.",
        "Setting boundaries is important when you're feeling overwhelmed.",
        "A short 10-minute break to stretch or walk around might help reset your mind."
    ],
    'angry': [
        "Would it help to talk through what happened?",
        "Sometimes taking a few deep breaths can help process anger.",
        "It's healthy to acknowledge when you're upset."
    ],
    'fearful': [
        "Let's think about what you can control.",
        "Would breathing exercises help calm your mind?",
        "Acknowledging fear is the first step to addressing it."
    ],
    'love': [
        "What makes this love so meaningful?",
        "How does this love impact your life?",
        "These connections enrich our lives."
    ],
    'optimistic': [
        "Share more about what's got you feeling hopeful!",
        "Keep nurturing that hopeful spirit!"
    ],
    'neutral': [
        "Is there anything specific you'd like to talk about?",
        "I'm glad you're here.",
        "You deserve support and understanding."
    ]
}

def detect_emotion_keywords(text):
    """Simple keyword-based emotion detection"""
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    
    if emotion_scores:
        # Return all emotions as list format to match RoBERTa model
        all_emotions = []
        total_score = sum(emotion_scores.values())
        
        for emotion, score in emotion_scores.items():
            confidence = min(score / total_score, 0.85)
            all_emotions.append({"label": emotion, "score": confidence})
        
        all_emotions.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            "primary_emotion": all_emotions[0]["label"],
            "confidence": all_emotions[0]["score"],
            "all_emotions": all_emotions,
            "significant_emotions": [e for e in all_emotions if e['score'] > 0.2],
            "model": "keyword"
        }
    else:
        return {
            "primary_emotion": "neutral",
            "confidence": 0.5,
            "all_emotions": [{"label": "neutral", "score": 0.5}],
            "significant_emotions": [{"label": "neutral", "score": 0.5}],
            "model": "keyword"
        }

def detect_emotion(text):
    """Detect emotion using RoBERTa or fallback to keywords"""
    if HF_AVAILABLE and emotion_classifier:
        try:
            results = emotion_classifier(text)
            
            # Handle nested format
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    emotions = results[0]
                else:
                    emotions = results
            else:
                emotions = results
            
            emotions.sort(key=lambda x: x['score'], reverse=True)
            primary = emotions[0]
            
            # Filter significant emotions (score > 0.1 for RoBERTa)
            significant_emotions = [e for e in emotions if e['score'] > 0.1]
            
            return {
                "primary_emotion": primary['label'].lower(),
                "confidence": primary['score'],
                "all_emotions": emotions,
                "significant_emotions": significant_emotions,
                "model": "roberta"
            }
            
        except Exception as e:
            print(f"[!] RoBERTa failed: {e}")
    
    return detect_emotion_keywords(text)

def check_crisis(text):
    """Check for crisis keywords"""
    return any(keyword in text.lower() for keyword in CRISIS_KEYWORDS)

def generate_response(emotion_data, user_message):
    """Generate empathic response with alternative emotions acknowledgment"""
    
    # Crisis intervention
    if check_crisis(user_message):
        return {
            "response": "🚨 I'm concerned about what you've shared. Please reach out for help:\n\n" +
                       "📞 National Suicide Prevention Lifeline: 988\n" +
                       "💬 Crisis Text Line: Text HOME to 741741\n\n" +
                       "You matter, and there are people who want to support you.",
            "type": "crisis",
            "emotion": emotion_data["primary_emotion"],
            "alternative_emotions": emotion_data.get("significant_emotions", [])
        }
    
    emotion = emotion_data["primary_emotion"]
    confidence = emotion_data["confidence"]
    significant_emotions = emotion_data.get("significant_emotions", [])
    
    # Map emotions to response categories
    emotion_map = {
        "sadness": "sad", "joy": "happy", "anger": "angry", "fear": "fearful",
        "love": "love", "optimism": "optimistic", "anticipation": "optimistic",
        "trust": "love", "surprise": "happy", "disgust": "angry", "pessimism": "sad"
    }
    
    response_key = emotion_map.get(emotion, "neutral")
    
    if confidence > 0.4:
        # High confidence - use specific response
        main_response = random.choice(RESPONSES[response_key])
        suggestion = random.choice(SUGGESTIONS[response_key])
        response = f"{main_response} {suggestion}"
        
        # Add alternative emotions acknowledgment (NEW FEATURE)
        alternative_emotions = []
        if significant_emotions and len(significant_emotions) > 1:
            # Get secondary emotions (excluding the primary)
            secondary_emotions = [e for e in significant_emotions[1:4] if e['score'] > 0.2]
            
            if secondary_emotions:
                alt_emotion = secondary_emotions[0]
                alt_emotion_name = alt_emotion['label']
                alt_mapped = emotion_map.get(alt_emotion_name, alt_emotion_name)
                
                # Add acknowledgment of secondary emotion
                if alt_mapped in RESPONSES and alt_mapped != response_key:
                    alt_response = random.choice(RESPONSES[alt_mapped])
                    response += f" I also sense some {alt_emotion_name} - {alt_response.lower()}"
                
                alternative_emotions = secondary_emotions
        
    else:
        # Low confidence - neutral response
        response = random.choice(RESPONSES["neutral"])
        alternative_emotions = []
    
    return {
        "response": response,
        "type": "empathic",
        "emotion": emotion,
        "confidence": confidence,
        "mapped": response_key,
        "model": emotion_data.get("model", "unknown"),
        "alternative_emotions": alternative_emotions,
        "all_emotions": emotion_data.get("all_emotions", [])
    }

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("=" * 50)
    print("🤖 EMPATHIC CHATBOT")
    print("=" * 50)
    print("💙 I'm here to listen and respond with empathy")
    model_info = "RoBERTa Multi-label" if HF_AVAILABLE else "Keyword-based"
    print(f"🔍 Using: {model_info}")
    print("🎭 Detects alternative emotions too!")
    print("🆘 Type 'help' for commands or 'quit' to exit")
    print("=" * 50)

def print_help():
    print("\n📋 COMMANDS:")
    print("  help  - Show this help")
    print("  quit  - Exit chatbot")
    print("  demo  - Run demo scenarios")
    print("  stats - Show statistics")
    print("  clear - Clear screen")
    print("\n💡 TRY SAYING:")
    print("  'I feel overwhelmed with everything'")
    print("  'I just got amazing news!'")
    print("  'I'm really angry but also hurt'")
    print("  'I'm excited but nervous about tomorrow'")

def run_demo():
    print("\n🎭 DEMO SCENARIOS:")
    print("=" * 40)
    
    demos = [
        ("Sadness", "I feel so alone and worthless"),
        ("Joy", "I got my dream job!"),
        ("Anger", "I'm furious about this!"),
        ("Mixed Emotions", "I'm nervous but excited about tomorrow")
    ]
    
    for name, text in demos:
        emotion_data = detect_emotion(text)
        response_data = generate_response(emotion_data, text)
        
        print(f"\n{name} Test:")
        print(f"Input: '{text}'")
        print(f"Primary: {emotion_data['primary_emotion']} ({emotion_data['confidence']:.1%})")
        
        # Show alternative emotions
        if response_data.get('alternative_emotions'):
            print("Alternative emotions detected:")
            for alt_emotion in response_data['alternative_emotions'][:2]:
                print(f"  - {alt_emotion['label']}: {alt_emotion['score']:.1%}")
        
        print(f"Response: {response_data['response'][:100]}...")
        print("-" * 40)
    
    print("\n✅ Demo completed! Try your own messages now.\n")

def main():
    """Terminal-only chatbot interface"""
    conversation_count = 0
    emotion_stats = {}
    
    clear_screen()
    print_header()
    print("\n🤖 Hello! How are you feeling today?")
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                print("🤖 Bot: I'm here when you're ready to share.")
                continue
            
            if user_input.lower() == 'quit':
                print(f"\n🤖 Bot: Take care! We had {conversation_count} conversations.")
                if emotion_stats:
                    top_emotions = sorted(emotion_stats.items(), key=lambda x: x[1], reverse=True)[:3]
                    print("📈 Top emotions:")
                    for emotion, count in top_emotions:
                        print(f"   {emotion}: {count} times")
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'demo':
                run_demo()
                continue
            elif user_input.lower() == 'clear':
                clear_screen()
                print_header()
                continue
            elif user_input.lower() == 'stats':
                if emotion_stats:
                    print("\n📊 EMOTION STATISTICS:")
                    for emotion, count in sorted(emotion_stats.items()):
                        print(f"   {emotion}: {count} times")
                else:
                    print("\n📊 No conversations yet!")
                continue
            
            conversation_count += 1
            
            emotion_data = detect_emotion(user_input)
            response_data = generate_response(emotion_data, user_input)
            
            # Update stats
            emotion = emotion_data['primary_emotion']
            emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1
            
            # Display response with alternative emotions
            print(f"\n🔍 Primary: {emotion} ({emotion_data['confidence']:.1%}) [{emotion_data.get('model', '?')}]")
            
            # Show alternative emotions if detected
            if response_data.get('alternative_emotions'):
                print("🎭 Alternative emotions:")
                for alt_emotion in response_data['alternative_emotions'][:2]:
                    print(f"   - {alt_emotion['label']}: {alt_emotion['score']:.1%}")
            
            if response_data['type'] == 'crisis':
                print("🚨 CRISIS RESPONSE:")
                print(response_data['response'])
            else:
                print(f"🤖 Bot: {response_data['response']}")
            
        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Goodbye! Take care of yourself. 💙")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("🤖 Bot: I'm having some technical difficulties. Please try again.")

if __name__ == "__main__":
    main()