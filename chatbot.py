from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import json
import random
import os

app = Flask(__name__)

# Initialize emotion classifier (this will download model on first run)
print("Loading emotion detection model...")
try:
    emotion_classifier = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )
    print("✅ Emotion model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    emotion_classifier = None

# Load response templates
def load_templates():
    templates_file = os.path.join(os.path.dirname(__file__), 'templates.json')
    with open(templates_file, 'r') as f:
        return json.load(f)

TEMPLATES = load_templates()

# Crisis keywords for safety
CRISIS_KEYWORDS = ['suicide', 'kill myself', 'end it all', 'hurt myself', 'die', 'not worth living']

def detect_emotion(text):
    """Detect emotion in user text"""
    if not emotion_classifier:
        return {"label": "neutral", "score": 0.5}
    
    try:
        result = emotion_classifier(text)[0]
        return {
            "label": result['label'].lower(),
            "score": round(result['score'], 3)
        }
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return {"label": "neutral", "score": 0.5}

def check_crisis(text):
    """Check for crisis language"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

def select_response(emotion_data, user_message):
    """Select appropriate empathic response"""
    
    # Crisis intervention takes priority
    if check_crisis(user_message):
        return {
            "response": "I'm really concerned about what you've shared. Please reach out to someone who can help: National Suicide Prevention Lifeline: 988 or text HOME to 741741. You matter, and there are people who want to support you.",
            "type": "crisis_intervention",
            "emotion": emotion_data["label"],
            "confidence": emotion_data["score"]
        }
    
    emotion = emotion_data["label"]
    confidence = emotion_data["score"]
    
    # Map emotions to our template categories
    emotion_mapping = {
        "sadness": "sad",
        "joy": "happy", 
        "fear": "stressed",
        "anger": "stressed",
        "disgust": "stressed",
        "surprise": "happy"
    }
    
    mapped_emotion = emotion_mapping.get(emotion, "neutral")
    
    # Select response strategy based on confidence
    if confidence > 0.7:
        # High confidence - use specific empathic response
        if mapped_emotion in TEMPLATES:
            strategies = TEMPLATES[mapped_emotion]
            # Combine strategies for more empathic response
            acknowledge = random.choice(strategies["acknowledge"])
            support = random.choice(strategies["support"]) 
            reinforce = random.choice(strategies["reinforce"])
            
            response = f"{acknowledge} {support} {reinforce}"
        else:
            response = random.choice(TEMPLATES["neutral"]["acknowledge"])
    else:
        # Lower confidence - use neutral supportive response
        response = random.choice(TEMPLATES["neutral"]["acknowledge"])
    
    return {
        "response": response,
        "type": "empathic",
        "emotion": emotion,
        "confidence": confidence,
        "mapped_emotion": mapped_emotion
    }

@app.route('/')
def index():
    """Serve the chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Detect emotion
        emotion_data = detect_emotion(user_message)
        
        # Generate response
        response_data = select_response(emotion_data, user_message)
        
        return jsonify({
            "bot_response": response_data["response"],
            "emotion_detected": response_data["emotion"],
            "confidence": response_data["confidence"],
            "response_type": response_data["type"],
            "user_message": user_message
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": emotion_classifier is not None
    })

if __name__ == '__main__':
    print("🤖 Starting Empathic Chatbot...")
    print("💡 Open http://localhost:5000 in your browser")
    print("🌐 To share: run 'ngrok http 5000' in another terminal")
    app.run(debug=True, host='0.0.0.0', port=5000)