from flask import Flask, request, jsonify, render_template
import torch
import json
import random
import os
import requests
import csv
import io
import re
from collections import defaultdict

app = Flask(__name__)

# ========================================
# CONFIGURATION - Toggle emotion detection models
# ========================================
USE_ADVANCED_MODEL = True  # Set to False to force fallback to NRC/keyword detection
FORCE_NRC_FALLBACK = False  # Set to True to force NRC lexicon as primary fallback

print(f"[CONFIG] Advanced model enabled: {USE_ADVANCED_MODEL}")
print(f"[CONFIG] Force NRC fallback: {FORCE_NRC_FALLBACK}")

# ========================================
# ADVANCED MODEL LOADING (Twitter RoBERTa Multi-Label)
# ========================================
if USE_ADVANCED_MODEL:
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
        print("[FALLBACK] Reverting to NRC/keyword-based emotion detection")
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
        print("[FALLBACK] Using NRC/keyword-based emotion detection")
        emotion_classifier = None
        ROBERTA_EMOTIONS = []
else:
    print("[CONFIG] Advanced model disabled - using fallback emotion detection")
    HF_AVAILABLE = False
    emotion_classifier = None
    ROBERTA_EMOTIONS = []

# ========================================
# NRC EMOTION LEXICON INTEGRATION (from empathetic_chatbot.py)
# ========================================
class NRCEmotionDetector:
    def __init__(self):
        self.word_emotions = defaultdict(set)
        self.use_fallback = False
        self.load_nrc_lexicon()
        self.setup_emotion_mapping()

    def load_nrc_lexicon(self):
        """Download and parse the NRC Emotion Lexicon"""
        print("[+] Loading NRC Emotion Lexicon...")
        url = "https://raw.githubusercontent.com/ashwini2108/NRC-Emotion-Lexicon/master/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            lines = response.text.strip().split('\n')
            
            self.word_emotions = defaultdict(set)
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) == 3:
                        word, emotion, flag = parts
                        if flag == '1':
                            self.word_emotions[word].add(emotion)
            print(f"[+] Loaded NRC lexicon with {len(self.word_emotions)} words.")
            self.use_fallback = False
        except Exception as e:
            print(f"[WARNING] Could not load NRC lexicon ({e}). Using basic keywords.")
            self.use_fallback = True
            self.setup_fallback_lexicon()

    def setup_fallback_lexicon(self):
        """Basic fallback if NRC fails to load"""
        self.word_emotions = defaultdict(set)
        fallback = {
            'sadness': ['sad', 'depressed', 'unhappy', 'down', 'miserable', 'heartbroken', 'lonely', 'crying', 'tears', 'grief'],
            'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'awesome', 'delighted'],
            'fear': ['stressed', 'anxious', 'overwhelmed', 'pressure', 'tense', 'worried', 'nervous', 'panic', 'scared'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed'],
            'love': ['love', 'adore', 'cherish', 'treasure', 'devoted', 'affection', 'romantic'],
            'trust': ['trust', 'reliable', 'dependable', 'faith', 'confidence'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
            'anticipation': ['anticipating', 'expecting', 'looking forward', 'excited about'],
            'disgust': ['disgusting', 'revolting', 'repulsive', 'gross', 'sick'],
            'optimism': ['hopeful', 'positive', 'optimistic', 'confident', 'bright'],
            'pessimism': ['pessimistic', 'negative', 'doubtful', 'hopeless', 'gloomy']
        }
        for emotion, words in fallback.items():
            for word in words:
                self.word_emotions[word].add(emotion)

    def setup_emotion_mapping(self):
        """Map NRC emotions to our scenarios"""
        self.emotion_to_scenario = {
            'sadness': 'sad',
            'joy': 'happy', 
            'fear': 'fearful',
            'anger': 'angry',
            'love': 'love',
            'trust': 'trust',
            'surprise': 'surprise',
            'anticipation': 'anticipation',
            'disgust': 'angry',  # Map disgust to angry
            'optimism': 'optimistic',
            'pessimism': 'sad'   # Map pessimism to sad
        }

    def detect_emotion_scenario(self, user_input):
        """Use NRC lexicon to detect dominant emotion scenario"""
        words = re.findall(r'\b\w+\b', user_input.lower())
        emotion_scores = defaultdict(int)
        
        for word in words:
            if word in self.word_emotions:
                for emotion in self.word_emotions[word]:
                    if emotion in self.emotion_to_scenario:
                        scenario = self.emotion_to_scenario[emotion]
                        emotion_scores[scenario] += 1
        
        if not emotion_scores:
            return None
        
        # Return scenario with highest score
        return max(emotion_scores, key=emotion_scores.get)

    def detect_emotion_detailed(self, user_input):
        """Detailed emotion detection returning all emotions with scores"""
        words = re.findall(r'\b\w+\b', user_input.lower())
        emotion_scores = defaultdict(int)
        
        # Count emotion occurrences
        for word in words:
            if word in self.word_emotions:
                for emotion in self.word_emotions[word]:
                    emotion_scores[emotion] += 1
        
        if not emotion_scores:
            return {
                "primary_emotion": "neutral",
                "confidence": 0.5,
                "all_emotions": [{"label": "neutral", "score": 0.5}],
                "model": "nrc-lexicon"
            }
        
        # Convert to list format matching advanced model
        all_emotions = []
        total_score = sum(emotion_scores.values())
        
        for emotion, score in emotion_scores.items():
            confidence = min(score / total_score, 0.9)  # Normalize and cap confidence
            all_emotions.append({"label": emotion, "score": confidence})
        
        # Sort by confidence
        all_emotions.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            "primary_emotion": all_emotions[0]["label"],
            "confidence": all_emotions[0]["score"],
            "all_emotions": all_emotions,
            "significant_emotions": [e for e in all_emotions if e['score'] > 0.1],
            "model": "nrc-lexicon"
        }

# Initialize NRC detector
nrc_detector = NRCEmotionDetector()

# ========================================
# ENHANCED KEYWORD DETECTION (Backup to NRC)
# ========================================
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

def detect_emotion_keywords(text):
    """Enhanced keyword-based emotion detection (final fallback)"""
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    
    if emotion_scores:
        # Return all emotions as list format to match other models
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
            "significant_emotions": [e for e in all_emotions if e['score'] > 0.1],
            "model": "keyword-fallback"
        }
    else:
        return {
            "primary_emotion": "neutral",
            "confidence": 0.5,
            "all_emotions": [{"label": "neutral", "score": 0.5}],
            "significant_emotions": [{"label": "neutral", "score": 0.5}],
            "model": "keyword-fallback"
        }

# ========================================
# ENHANCED CRISIS DETECTION
# ========================================
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

def check_crisis(text):
    """Enhanced crisis detection"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

# ========================================
# UNIFIED EMOTION DETECTION FUNCTION
# ========================================
def detect_emotion(text):
    """Unified emotion detection with multiple fallback layers"""
    
    # Layer 1: Twitter RoBERTa Multi-Label (if enabled and available)
    if USE_ADVANCED_MODEL and HF_AVAILABLE and emotion_classifier:
        try:
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
                raise ValueError("Unexpected format from RoBERTa model")
            
            # Sort emotions by confidence score
            all_emotions.sort(key=lambda x: x['score'], reverse=True)
            
            # Get primary emotion (highest confidence)
            primary_emotion = all_emotions[0]['label'].lower()
            primary_confidence = round(all_emotions[0]['score'], 3)
            
            # Filter emotions with meaningful scores (> 0.1)
            significant_emotions = [e for e in all_emotions if e['score'] > 0.1]
            
            print(f"[INFO] Using Twitter RoBERTa multi-label model")
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
            print("[INFO] Falling back to NRC lexicon detection")
            # Fall through to Layer 2
    
    # Layer 2: NRC Emotion Lexicon (if enabled or forced)
    if FORCE_NRC_FALLBACK or (not USE_ADVANCED_MODEL or not HF_AVAILABLE):
        try:
            nrc_result = nrc_detector.detect_emotion_detailed(text)
            if nrc_result["primary_emotion"] != "neutral" or not nrc_detector.use_fallback:
                print(f"[INFO] Using NRC Emotion Lexicon")
                return nrc_result
            else:
                print("[INFO] NRC detection returned neutral, trying keyword fallback")
                # Fall through to Layer 3
        except Exception as e:
            print(f"[ERROR] NRC emotion detection failed: {e}")
            print("[INFO] Falling back to keyword detection")
            # Fall through to Layer 3
    
    # Layer 3: Basic Keyword Detection (final fallback)
    print(f"[INFO] Using keyword-based emotion detection")
    return detect_emotion_keywords(text)

# ========================================
# RESPONSE TEMPLATES AND GENERATION
# ========================================
def load_templates():
    """Load response templates from file or use defaults"""
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

def select_response(emotion_data, user_message, templates):
    """Select appropriate empathic response based on detected emotions"""
    
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
            "all_emotions": emotion_data.get("all_emotions", []),
            "model_used": emotion_data.get("model", "unknown")
        }
    
    primary_emotion = emotion_data["primary_emotion"]
    confidence = emotion_data["confidence"]
    all_emotions = emotion_data.get("all_emotions", [])
    significant_emotions = emotion_data.get("significant_emotions", [])
    
    # Enhanced mapping for all emotion types to template categories
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
    
    # Adjust confidence thresholds based on model used
    model_used = emotion_data.get("model", "unknown")
    if model_used == "twitter-roberta-multilabel":
        confidence_threshold = 0.3
    elif model_used == "nrc-lexicon":
        confidence_threshold = 0.4
    else:  # keyword-fallback
        confidence_threshold = 0.6
    
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
        "model_used": model_used,
        "all_emotions": all_emotions,
        "significant_emotions": significant_emotions,
        "secondary_emotions": [e for e in significant_emotions[1:3] if e['score'] > 0.2] if significant_emotions else []
    }

# ========================================
# FLASK ROUTES FOR WEB INTERFACE
# ========================================
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
        
        # Detect emotion using unified detection system
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
            "user_message": user_message,
            "config": {
                "advanced_model_enabled": USE_ADVANCED_MODEL,
                "force_nrc_fallback": FORCE_NRC_FALLBACK,
                "nrc_available": not nrc_detector.use_fallback
            }
        })
        
    except Exception as e:
        print(f"[ERROR] Chat endpoint error: {e}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    # Determine which models are available
    models_available = []
    if USE_ADVANCED_MODEL and HF_AVAILABLE and emotion_classifier:
        models_available.append("twitter-roberta-multilabel")
    if not nrc_detector.use_fallback:
        models_available.append("nrc-lexicon")
    models_available.append("keyword-fallback")
    
    return jsonify({
        "status": "healthy",
        "config": {
            "advanced_model_enabled": USE_ADVANCED_MODEL,
            "force_nrc_fallback": FORCE_NRC_FALLBACK
        },
        "models": {
            "primary": models_available[0] if models_available else "keyword-fallback",
            "available": models_available,
            "roberta_loaded": USE_ADVANCED_MODEL and HF_AVAILABLE and emotion_classifier is not None,
            "nrc_loaded": not nrc_detector.use_fallback,
            "keyword_available": True
        },
        "supported_emotions": {
            "roberta": ROBERTA_EMOTIONS if USE_ADVANCED_MODEL and HF_AVAILABLE else [],
            "nrc": list(nrc_detector.emotion_to_scenario.keys()),
            "keyword": list(EMOTION_KEYWORDS.keys())
        }
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
        },
        {
            "name": "NRC Test Scenario",
            "input": "The sunrise filled me with such overwhelming happiness and gratitude",
            "description": "Testing NRC lexicon emotion detection",
            "expected_emotion": "joy"
        }
    ]
    
    return jsonify({"scenarios": scenarios})

# ========================================
# TERMINAL INTERFACE FUNCTIONS
# ========================================
def print_header():
    """Print enhanced chatbot header"""
    print("=" * 80)
    print("🤖 EMPATHIC CHATBOT - Unified Multi-Model Emotion Detection")
    print("=" * 80)
    print("💙 I'm here to listen and respond with empathy")
    
    # Show active configuration
    if USE_ADVANCED_MODEL and HF_AVAILABLE:
        print("🔍 Primary: Twitter RoBERTa Multi-Label Model")
    elif FORCE_NRC_FALLBACK or not USE_ADVANCED_MODEL:
        print("🔍 Primary: NRC Emotion Lexicon")
    else:
        print("🔍 Primary: Keyword-based Detection")
    
    print("🎭 Fallback layers: RoBERTa → NRC Lexicon → Keywords")
    print("🆘 Type 'help' for commands or 'quit' to exit")
    print("⚙️  Type 'config' to see current configuration")
    print("=" * 80)
    print()

def print_help():
    """Print enhanced help information"""
    print("\n📋 AVAILABLE COMMANDS:")
    print("  help     - Show this help message")
    print("  quit     - Exit the chatbot")
    print("  clear    - Clear the screen")
    print("  demo     - Run demonstration scenarios")
    print("  stats    - Show emotion detection statistics")
    print("  emotions - Show supported emotions")
    print("  test     - Test emotion detection format")
    print("  config   - Show current model configuration")
    print("  models   - Show available models and their status")
    print("\n💡 EXAMPLE MESSAGES TO TRY:")
    print("  'I feel so overwhelmed with everything'")
    print("  'I just got the best news ever!'")
    print("  'I'm absolutely furious about this situation'")
    print("  'I'm scared but excited about tomorrow'")
    print("  'I'm so in love with life right now'")
    print("\n🔧 MODEL TESTING:")
    print("  Try different emotional expressions to see how each model layer responds")
    print("  The system will automatically fall back to simpler models if needed")
    print()

def show_configuration():
    """Show current configuration"""
    print("\n⚙️  CURRENT CONFIGURATION:")
    print("=" * 50)
    print(f"   Advanced Model Enabled: {USE_ADVANCED_MODEL}")
    print(f"   Force NRC Fallback: {FORCE_NRC_FALLBACK}")
    print(f"   RoBERTa Available: {USE_ADVANCED_MODEL and HF_AVAILABLE}")
    print(f"   NRC Lexicon Available: {not nrc_detector.use_fallback}")
    print(f"   Keyword Fallback: Always Available")
    print("=" * 50)
    print()

def show_models():
    """Show available models and their status"""
    print("\n🔧 MODEL STATUS:")
    print("=" * 60)
    
    # Twitter RoBERTa Status
    if USE_ADVANCED_MODEL and HF_AVAILABLE and emotion_classifier:
        print("✅ Twitter RoBERTa Multi-Label: ACTIVE")
        print(f"   Emotions: {ROBERTA_EMOTIONS}")
    elif USE_ADVANCED_MODEL:
        print("❌ Twitter RoBERTa Multi-Label: FAILED TO LOAD")
    else:
        print("⚠️  Twitter RoBERTa Multi-Label: DISABLED")
    
    # NRC Lexicon Status  
    if not nrc_detector.use_fallback:
        status = "ACTIVE" if FORCE_NRC_FALLBACK or not (USE_ADVANCED_MODEL and HF_AVAILABLE) else "STANDBY"
        print(f"✅ NRC Emotion Lexicon: {status}")
        print(f"   Words loaded: {len(nrc_detector.word_emotions)}")
    else:
        print("⚠️  NRC Emotion Lexicon: USING BASIC FALLBACK")
    
    # Keyword Detection Status
    print("✅ Keyword Detection: ACTIVE (Final Fallback)")
    print(f"   Emotions: {list(EMOTION_KEYWORDS.keys())}")
    
    print("=" * 60)
    print()

def show_supported_emotions():
    """Show supported emotions for all models"""
    print("\n🎭 SUPPORTED EMOTIONS BY MODEL:")
    print("=" * 60)
    
    if USE_ADVANCED_MODEL and HF_AVAILABLE:
        print("📊 Twitter RoBERTa Multi-Label:")
        for i, emotion in enumerate(ROBERTA_EMOTIONS, 1):
            print(f"   {i:2d}. {emotion.title()}")
        print("   ✨ Multi-label: Can detect multiple emotions simultaneously!")
    
    print("\n📚 NRC Emotion Lexicon:")
    nrc_emotions = list(nrc_detector.emotion_to_scenario.keys())
    for i, emotion in enumerate(nrc_emotions, 1):
        print(f"   {i:2d}. {emotion.title()}")
    print("   📖 Lexicon-based: Uses word-emotion associations")
    
    print("\n🔤 Keyword Detection:")
    for i, emotion in enumerate(EMOTION_KEYWORDS.keys(), 1):
        print(f"   {i:2d}. {emotion.title()}")
    print("   🎯 Pattern-based: Matches specific keyword patterns")
    print()

def test_emotion_format():
    """Test emotion detection across all available models"""
    print("\n🔍 TESTING EMOTION DETECTION ACROSS ALL MODELS:")
    print("=" * 70)
    
    test_texts = [
        "I'm happy but nervous",
        "I feel absolutely terrible and hopeless",
        "This is amazing and wonderful news!"
    ]
    
    for test_text in test_texts:
        print(f"\n📝 Test input: '{test_text}'")
        print("-" * 50)
        
        try:
            emotion_data = detect_emotion(test_text)
            print(f"✓ Detection successful!")
            print(f"  Primary emotion: {emotion_data['primary_emotion']}")
            print(f"  Confidence: {emotion_data['confidence']:.3f}")
            print(f"  Model used: {emotion_data.get('model', 'unknown')}")
            
            if emotion_data.get('all_emotions'):
                print(f"  Top emotions:")
                for emotion in emotion_data['all_emotions'][:3]:
                    print(f"    - {emotion['label']}: {emotion['score']:.3f}")
                    
        except Exception as e:
            print(f"✗ Detection failed: {e}")
            print(f"  Error type: {type(e).__name__}")
    
    print("=" * 70)

def run_demo(templates):
    """Run enhanced demonstration scenarios across all models"""
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
        },
        {
            "name": "NRC Lexicon Test",
            "input": "The sunrise filled me with overwhelming happiness and gratitude",
            "description": "Testing NRC-specific emotion detection",
            "expected_emotion": "joy"
        }
    ]
    
    print("\n🎭 RUNNING ENHANCED DEMO SCENARIOS:")
    print("=" * 90)
    
    emotion_counts = {}
    model_usage = {}
    
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        expected = scenario.get('expected_emotion', '')
        print(f"   Expected: {expected}")
        print(f"   Description: {scenario['description']}")
        print(f"   User Input: \"{scenario['input']}\"")
        
        # Process the demo input
        emotion_data = detect_emotion(scenario['input'])
        response_data = select_response(emotion_data, scenario['input'], templates)
        
        # Track statistics
        primary_emotion = emotion_data['primary_emotion']
        emotion_counts[primary_emotion] = emotion_counts.get(primary_emotion, 0) + 1
        
        model_used = emotion_data.get('model', 'unknown')
        model_usage[model_used] = model_usage.get(model_used, 0) + 1
        
        # Check if detected matches expected
        match_indicator = "✅" if primary_emotion == expected else "❓"
        
        print(f"   {match_indicator} Primary: {primary_emotion} ({emotion_data['confidence']:.1%} confidence)")
        print(f"   🔧 Model: {model_used}")
        
        # Show significant emotions if available
        if emotion_data.get('significant_emotions') and len(emotion_data['significant_emotions']) > 1:
            print(f"   🎭 Significant emotions:")
            for emotion_info in emotion_data['significant_emotions'][:3]:
                print(f"      - {emotion_info['label']}: {emotion_info['score']:.1%}")
        
        print(f"   🔄 Mapped to template: {response_data.get('mapped_emotion', 'unknown')}")
        print(f"   💙 Bot Response: {response_data['response'][:100]}...")
        print("-" * 90)
    
    # Summary statistics
    print(f"\n📊 DEMO SUMMARY:")
    print("=" * 50)
    print("Emotion Detection Summary:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"   {emotion.title()}: {count} detections")
    
    print("\nModel Usage Summary:")
    for model, count in sorted(model_usage.items()):
        print(f"   {model}: {count} times")
    
    print(f"\n🎯 AVAILABLE MODELS:")
    print(f"   Advanced: {USE_ADVANCED_MODEL and HF_AVAILABLE}")
    print(f"   NRC: {not nrc_detector.use_fallback}")
    print(f"   Keywords: Always available")
    
    print("\n✅ Enhanced demo completed! Multi-layered emotion detection is active.\n")

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Enhanced main chatbot loop with unified emotion detection"""
    templates = load_templates()
    conversation_count = 0
    emotion_stats = {}
    model_stats = {}
    
    # clear_screen() # remove for debugging
    print_header()
    
    # Welcome message based on configuration
    if USE_ADVANCED_MODEL and HF_AVAILABLE:
        print("🤖 Hello! I'm an empathic chatbot powered by advanced multi-layered emotion detection.")
        print("   I use Twitter RoBERTa, NRC Lexicon, and keyword analysis for comprehensive understanding!")
    elif not nrc_detector.use_fallback:
        print("🤖 Hello! I'm an empathic chatbot powered by the NRC Emotion Lexicon.")
        print("   I understand emotions through word-emotion associations and keyword patterns!")
    else:
        print("🤖 Hello! I'm an empathic chatbot using keyword-based emotion detection.")
        print("   I recognize emotions through specific word patterns and phrases!")
    
    print("   How are you feeling today? (Type 'help' for commands)")
    
    while True:
        try:
            # Get user input
            print("\n" + "─" * 60)
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
                if model_stats:
                    print("🔧 Model usage:")
                    for model, count in sorted(model_stats.items()):
                        print(f"   {model}: {count} times")
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
            elif user_input.lower() == 'config':
                show_configuration()
                continue
            elif user_input.lower() == 'models':
                show_models()
                continue
            elif user_input.lower() == 'stats':
                if emotion_stats:
                    print("\n📊 EMOTION DETECTION STATISTICS:")
                    for emotion, count in sorted(emotion_stats.items()):
                        print(f"   {emotion.title()}: {count} times")
                    print("\n🔧 MODEL USAGE STATISTICS:")
                    for model, count in sorted(model_stats.items()):
                        print(f"   {model}: {count} times")
                else:
                    print("\n📊 No conversations yet. Start chatting to see stats!")
                continue
            
            # Process regular message
            conversation_count += 1
            
            # Detect emotion using unified system
            emotion_data = detect_emotion(user_input)
            
            # Update statistics
            primary_emotion = emotion_data['primary_emotion']
            emotion_stats[primary_emotion] = emotion_stats.get(primary_emotion, 0) + 1
            
            model_used = emotion_data.get('model', 'unknown')
            model_stats[model_used] = model_stats.get(model_used, 0) + 1
            
            # Generate response
            response_data = select_response(emotion_data, user_input, templates)
            
            # Display response with enhanced emotion info
            print(f"\n🔍 Primary: {primary_emotion} ({emotion_data['confidence']:.1%} confidence)")
            print(f"🔧 Model: {model_used}")
            
            # Show significant emotions if multiple detected
            if emotion_data.get('significant_emotions') and len(emotion_data['significant_emotions']) > 1:
                print(f"🎭 Significant emotions:")
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
        print(f"⚙️  Advanced model: {USE_ADVANCED_MODEL}")
        print(f"⚙️  NRC fallback: {FORCE_NRC_FALLBACK}")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        # Run terminal version by default
        main()