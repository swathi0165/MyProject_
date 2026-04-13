import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import sqlite3
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ================= CONFIG & SESSION STATE =================
st.set_page_config(page_title="Emotion Predictor", page_icon="🎭")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

# ================= DATABASE SETUP =================
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)')
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO users (username, password) VALUES (?,?)', (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username =? AND password =?', (username, password))
    data = c.fetchall()
    return data

# ================= LOAD ML FILES (Cached for Speed) =================
@st.cache_resource
def load_models():
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    encoder = pickle.load(open("label_encoder.pkl", "rb"))
    
    num_classes = len(encoder.classes_)
    
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10000, 128)
            self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = LSTMModel()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    return tokenizer, encoder, model

tokenizer, encoder, model = load_models()

# ================= UTILS =================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

emoji_map = {
    "happy": "😊", "sad": "😢", "angry": "😡",
    "fear": "😨", "love": "❤️", "surprise": "😲", "neutral": "😐"
}

recommendations = {
    "happy": "💡 Keep smiling and spread positivity!",
    "sad": "💡 Talk to someone you trust ❤️",
    "angry": "💡 Take deep breaths and relax 😌",
    "fear": "💡 Stay calm and try meditation 🧘",
    "love": "💡 Share your happiness with others ❤️",
    "surprise": "💡 Take a moment to process your feelings 😲",
    "neutral": "💡 Stay balanced and positive 👍"
}

# ================= UI COMPONENTS =================
def main():
    init_db()
    
    if not st.session_state['logged_in']:
        auth_mode = st.sidebar.selectbox("Login/Signup", ["Login", "Register"])
        
        if auth_mode == "Login":
            st.title("🔐 Login")
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            if st.button("Login"):
                result = login_user(user, pw)
                if result:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = user
                    st.rerun()
                else:
                    st.error("Invalid Username/Password")
                    
        else:
            st.title("📝 Create Account")
            new_user = st.text_input("Username")
            new_pw = st.text_input("Password", type="password")
            if st.button("Register"):
                add_user(new_user, new_pw)
                st.success("Account created! Please switch to Login.")

    else:
        # -------- DASHBOARD --------
        st.sidebar.write(f"Logged in as: **{st.session_state['username']}**")
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()

        st.title("🎭 Emotion Classifier")
        st.write("Enter your text below to analyze the emotional tone.")

        user_text = st.text_area("How are you feeling?", placeholder="Type here...")

        if st.button("Analyze"):
            if user_text.strip() != "":
                cleaned = clean_text(user_text)
                
                # Tokenize & Pad
                seq = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=60)

                # Predict
                with torch.no_grad():
                    input_tensor = torch.tensor(padded, dtype=torch.long)
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence, pred = torch.max(probs, 1)

                emotion = encoder.inverse_transform([pred.item()])[0].lower()
                conf_score = round(confidence.item() * 100, 2)
                
                # UI Result
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Detected Emotion", f"{emotion.capitalize()} {emoji_map.get(emotion, '🙂')}")
                with col2:
                    st.metric("Confidence", f"{conf_score}%")
                
                st.info(recommendations.get(emotion, "Stay mindful and take care of yourself!"))
            else:
                st.warning("Please enter some text first.")

if __name__ == "__main__":
    main()
