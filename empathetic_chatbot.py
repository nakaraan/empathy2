import requests
import csv
import io
import re
import random
from collections import defaultdict

class EmpatheticChatbotNRC:
    def __init__(self):
        self.load_nrc_lexicon()
        self.setup_emotion_mapping()
        self.setup_responses()

    def load_nrc_lexicon(self):
        """Download and parse the NRC Emotion Lexicon"""
        print("Loading NRC Emotion Lexicon...")
        url = "https://raw.githubusercontent.com/ashwini2108/NRC-Emotion-Lexicon/master/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
        
        try:
            response = requests.get(url)
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
            print(f"Loaded NRC lexicon with {len(self.word_emotions)} words.")
        except Exception as e:
            print(f"Warning: Could not load NRC lexicon ({e}). Falling back to basic keywords.")
            self.use_fallback = True
            self.setup_fallback_lexicon()
        else:
            self.use_fallback = False

    def setup_fallback_lexicon(self):
        """Basic fallback if NRC fails to load"""
        self.word_emotions = defaultdict(set)
        fallback = {
            'sad': ['sad', 'depressed', 'unhappy', 'down', 'miserable', 'heartbroken', 'lonely', 'crying', 'tears', 'grief'],
            'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'awesome', 'delighted'],
            'fear': ['stressed', 'anxious', 'overwhelmed', 'pressure', 'tense', 'worried', 'nervous', 'panic', 'scared'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed']
        }
        for emotion, words in fallback.items():
            for word in words:
                self.word_emotions[word].add(emotion)

    def setup_emotion_mapping(self):
        """
        Map NRC emotions to our 3 scenarios:
        - sad → sadness
        - happy → joy
        - stressed → fear + anger (stress often involves both)
        """
        self.emotion_to_scenario = {
            'sadness': 'sad',
            'joy': 'happy',
            'fear': 'stressed',
            'anger': 'stressed'
        }

    def setup_responses(self):
        self.responses = {
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
            'stressed': [
                "I understand that stress can be overwhelming. Would it help to take a few deep breaths?",
                "It sounds like you're under a lot of pressure. Remember to be kind to yourself.",
                "Stress can be really tough to handle. Have you tried taking a short break to reset?",
                "I'm here for you. Sometimes just talking about what's stressing you can help lighten the load."
            ]
        }
        
        self.suggestions = {
            'sad': [
                "Would you like to try a short mindfulness exercise? It might help you feel a bit better.",
                "Sometimes writing down your feelings can help process them. Would you like to try that?",
                "Connecting with a friend or loved one might help. Is there someone you could reach out to?",
                "Taking a walk in nature can sometimes lift your mood. Would that be possible for you today?"
            ],
            'happy': [
                "Why not celebrate this moment? Do something special to mark this happy time!",
                "Sharing your happiness with others can multiply the joy. Who could you tell about this?",
                "Take a moment to really savor this feeling. What specifically made this so wonderful?",
                "This positive energy is great! How can you carry this feeling forward into the rest of your day?"
            ],
            'stressed': [
                "Try the 4-7-8 breathing technique: inhale for 4 seconds, hold for 7, exhale for 8. It can help calm your nervous system.",
                "Breaking your tasks into smaller steps might make them feel more manageable. Want to try that together?",
                "Setting boundaries is important when you're feeling overwhelmed. Is there something you can say 'no' to right now?",
                "A short 10-minute break to stretch or walk around might help reset your mind. Can you take that time for yourself?"
            ]
        }
        
        self.default_responses = [
            "I'm here to listen. How are you feeling today?",
            "I care about how you're doing. Would you like to share what's on your mind?",
            "Your feelings matter to me. Is there something specific you'd like to talk about?",
            "I'm all ears. What's going on in your world right now?"
        ]

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
        
        #return scenario with highest score
        return max(emotion_scores, key=emotion_scores.get)

    def generate_response(self, user_input):
        scenario = self.detect_emotion_scenario(user_input)
        if scenario:
            resp = random.choice(self.responses[scenario])
            sugg = random.choice(self.suggestions[scenario])
            return f"{resp}\n{sugg}"
        else:
            return random.choice(self.default_responses)

    def start_chat(self):
        print("Hello! I'm your empathetic chatbot (powered by NRC Emotion Lexicon).")
        print("I'm here to listen and support you. Type 'quit' to exit.\n")
        
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Chatbot: Thank you for talking with me. Take care of yourself!")
                break
            if not user_input:
                print("Chatbot: I'm here whenever you're ready to talk.\n")
                continue
            
            response = self.generate_response(user_input)
            print(f"Chatbot: {response}\n")


if __name__ == "__main__":
   #incase pip3 install request hast been installed yet
    try:
        bot = EmpatheticChatbotNRC()
        bot.start_chat()
    except KeyboardInterrupt:
        print("\nChatbot: Goodbye! Take care.")