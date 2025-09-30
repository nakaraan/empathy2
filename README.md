# 🤖 Empathic Chatbot Prototype

An AI chatbot that detects emotions and responds with empathy using acknowledgment, supportive suggestions, and positive reinforcement.

## 🔗 Live Demo Link
*Will be updated with ngrok/deployment URL*

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Navigate to project directory
cd empathy2.2

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application
```bash
# Start the Flask server
python chatbot.py
```

The application will be available at: `http://localhost:5000`

### 3. Share Your Demo (Optional)
```bash
# In a new terminal, install ngrok
# Download from: https://ngrok.com/download

# Expose local server to internet
ngrok http 5000

# Copy the https://xxx.ngrok.io URL for sharing
```

## 📊 Sample Conversations

| Scenario | User Input | Bot Response | Detected Emotion |
|----------|------------|--------------|------------------|
| **Sadness** | "I feel so alone and worthless today" | "I'm really sorry you're feeling this way. Is there anything small that might bring you a bit of comfort today? It takes courage to share how you're feeling." | sadness (89%) |
| **Happiness** | "I just got accepted to my dream university!" | "That's wonderful to hear! I can feel your joy. Tell me more about what made this so special! You absolutely deserve this happiness!" | joy (92%) |
| **Stress** | "I have three exams tomorrow and I'm freaking out" | "I can really feel the stress in your message. Let's try breaking this down into smaller, manageable pieces. You're handling so much - that shows real resilience." | fear (87%) |
| **Overwhelm** | "Everything feels like too much right now" | "That sounds incredibly overwhelming right now. Sometimes taking just 5 deep breaths can help reset your mind. It's smart to recognize when you're feeling overwhelmed." | fear (91%) |
| **Excitement** | "I can't believe I got the job!" | "You sound absolutely delighted - that's amazing! What was the best part of this experience? Your excitement is truly heartwarming." | joy (85%) |

## 🛠️ Design Choices

### Tool Selection: Python + HuggingFace + Flask
**Why this stack?**
- **Fast implementation**: 15-minute setup with pre-trained models
- **Reliable emotion detection**: 85-95% accuracy using `j-hartmann/emotion-english-distilroberta-base`
- **Free and accessible**: No API costs or complex infrastructure
- **Demo-friendly**: Web interface with visual emotion indicators

### Response Strategy Design
**How responses were crafted:**
1. **Research-based empathy principles**:
   - Acknowledgment validates feelings
   - Support offers practical help
   - Reinforcement builds confidence

2. **Three-layer response system**:
   - **Acknowledge**: "I can hear the stress in your message"
   - **Support**: "Let's break this into smaller pieces"
   - **Reinforce**: "You're showing real resilience"

3. **Safety-first approach**:
   - Crisis keyword detection
   - Mental health resource referrals
   - Clear boundaries about AI limitations

### What Makes This Chatbot Empathic?

1. **Emotion Recognition**: Detects 6 emotions (joy, sadness, anger, fear, surprise, disgust)
2. **Contextual Responses**: Different strategies for different emotional states
3. **Validation**: Acknowledges feelings without judgment
4. **Practical Support**: Offers concrete suggestions (breathing exercises, breaking down tasks)
5. **Confidence Building**: Reinforces user's strengths and coping abilities
6. **Safety Awareness**: Recognizes crisis situations and provides appropriate resources

## 🏗️ Architecture

```
User Input → Emotion Detection → Template Selection → Response Generation → Web UI
     ↓              ↓                    ↓                   ↓             ↓
Text Analysis → HuggingFace Model → Rule-based Logic → Empathic Response → Flask/HTML
```

### Core Components:
- **Emotion Classifier**: `j-hartmann/emotion-english-distilroberta-base`
- **Response Templates**: JSON-based with 3 strategies per emotion
- **Safety Filter**: Crisis keyword detection with resource referrals
- **Web Interface**: Real-time chat with emotion indicators

## 📹 Demo Video Script

**Duration: 2-3 minutes**

1. **Introduction (30s)**
   - Show web interface
   - Explain emotion detection + empathic response concept

2. **Sad Scenario (45s)**
   - Input: "I feel so alone and worthless"
   - Show emotion detection (sadness 89%)
   - Highlight empathic response strategies

3. **Happy Scenario (30s)**
   - Input: "I got into my dream school!"
   - Show celebration and encouragement response

4. **Stressed Scenario (45s)**
   - Input: "I have three exams tomorrow"
   - Show grounding techniques and practical support

5. **Conclusion (30s)**
   - Summarize empathy approach
   - Mention safety features and limitations

## 🔒 Safety Features

- **Crisis Detection**: Monitors for self-harm language
- **Resource Referral**: Provides mental health hotlines
- **Clear Boundaries**: Disclaims medical/therapeutic advice
- **Privacy**: No conversation storage in this prototype

## 🚧 Future Improvements

- [ ] Conversation memory for context continuity
- [ ] Voice input/output capabilities  
- [ ] Multi-language emotion detection
- [ ] Integration with mental health resources
- [ ] Advanced crisis intervention protocols
- [ ] Personalized response learning

## 📝 Technical Notes

- **Model**: Emotion classification with 6-label output
- **Confidence Threshold**: >70% for specific responses, fallback for lower confidence
- **Response Variety**: Multiple templates prevent repetitive interactions
- **Performance**: ~1-2 second response time including model inference

---

*Built for educational purposes. Not a replacement for professional mental health support.*