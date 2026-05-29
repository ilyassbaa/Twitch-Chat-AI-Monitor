import streamlit as st
import torch
import socket
import re
import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. Page Configuration & AI Loading ---
st.set_page_config(page_title="Twitch Sentiment AI", page_icon="🎮", layout="wide")
device = torch.device("cpu")

@st.cache_resource
def load_ai():
    try:
        model_path = "./final_twitch_sentiment_model"
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        return tokenizer, model
    except OSError:
        return None, None

tokenizer, model = load_ai()

# --- 2. The Twitch Slang Translator ---
# This dictionary cleans our data by converting slang into strong emotional words
TWITCH_DICT = {
    # --- Your Existing Base ---
    "pog": "incredible",
    "poggers": "amazing",
    "pogchamp": "epic",
    "kekw": "hilarious",
    "lul": "funny",
    "lulw": "laughing extremely hard",
    "omegalul": "extremely funny",
    "lol": "laughing out loud",
    "lmao": "hilarious",
    "monkas": "nervous",
    "monkaw": "scared",
    "pepega": "stupid",
    "kappa": "sarcastic",
    "kapp": "joking",
    "sadge": "sad",
    "pepehands": "crying",
    "w": "huge win success",
    "l": "massive loss terrible",
    "dogwater": "awful",
    "trash": "terrible",
    "f": "respect sad",
    "dumbass": "terrible toxic insult",
    "idiot": "bad toxic insult",
    "rip": "dead tragic",

    # --- NEW: Gaming & Hype Slang (Amplifies Hype Class) ---
    "gg": "good game well played success",
    "ez": "very easy simple victory",
    "goat": "greatest of all time absolute best legendary",
    "clutch": "incredible perfect high pressure success",
    "sheesh": "amazing impressive pure excitement",
    "gigachad": "ultimate legendary peak performance alpha hero",
    "fr": "for real completely honest truth",
    "diff": "massive skill gap superior performance",
    "aimbot": "perfect cheating precision accuracy",
    "cracked": "insanely skilled brilliant performance",
    "cooked": "completely destroyed defeated ruined",
    
    # --- NEW: Meta Twitch Emotes (Context Translation) ---
    "copium": "heavy denial delusional rationalization",
    "hopium": "extreme blind hope optimism",
    "notlikethis": "panic distress failure tragedy",
    "pogu": "massive surprise excitement amazing",
    "giga": "extremely massive ultimate",
    "prayge": "hopeful praying begging please",
    "plis": "please",
    "plz": "please",
    "clown": "ridiculous foolish funny joke",
    "residentsleeper": "extremely boring slow tired sleepy",
    "wutface": "extreme shock disgust confusion surprise",
    "ayaya": "cute excited anime cheer",

    # --- NEW: Heavy Toxicity & Moderation (Forces Toxic Class) ---
    "bitch": "vile aggressive toxic insult offensive",
    "stfu": "shut the fuck up hostile aggressive toxic command",
    "fuck": "intense angry profanity frustration",
    "shitter": "terrible awful toxic insult useless player",
    "garbage": "worthless terrible absolute trash",
    "cancer": "horrible destructive toxic ruinous",
    "kys": "extreme dangerous toxic threat self harm violation hostility",
    "uninstall": "you are terrible quit permanently toxic insult",
    "thrower": "saboteur losing intentionally toxic behavior",
    "throwing": "sabotaging losing on purpose ruining the game",
    "boosted": "fake skill terrible carried player",
    "noob": "incompetent bad terrible beginner player",
    "loser": "pathetic failure toxic insult",
    "ass": "awful terrible bad",
    "suck": "awful terrible bad",
    "sucks": "awful terrible bad"
}

def translate_slang(text):
    words = text.split()
    translated_words = []
    for word in words:
        # Check if the lowercase version of the word is in our dictionary
        clean_word = word.lower()
        if clean_word in TWITCH_DICT:
            translated_words.append(TWITCH_DICT[clean_word])
        else:
            translated_words.append(word)
    return " ".join(translated_words)

# --- 3. Session State for Live Analytics ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Toxic", "Neutral", "Hype"])
if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []
if 'live_running' not in st.session_state:
    st.session_state.live_running = False

# --- 4. Core Inference Function ---
def analyze_message(raw_message):
    # PREPROCESSING: Translate the slang first!
    translated_message = translate_slang(raw_message)
    
    inputs = tokenizer(translated_message, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
    prediction_id = torch.argmax(outputs.logits, dim=-1).item()
    
    labels = {0: "🔴 Toxic", 1: "⚪ Neutral", 2: "🟢 Hype"}
    return labels[prediction_id], probabilities, translated_message

# --- 5. The User Interface ---
st.title("🎮 Live Twitch Chat AI Monitor")
st.markdown("Monitor streamer chat dynamics using a custom DistilBERT Deep Learning model, featuring real-time data preprocessing.")

if model is None:
    st.warning("⚠️ Training is still in progress! Waiting for the model to be saved...")
    st.stop()

tab1, tab2 = st.tabs(["🔴 Live Stream Analytics", "🧪 Manual Testing"])

# ==========================================
# TAB 1: THE LIVE TWITCH CONNECTION
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Connection Settings")
        channel_name = st.text_input("Twitch Channel Name:", placeholder="e.g., kaicenat, ibai, tarik").lower()
        
        if st.button("🔌 Connect to Live Stream"):
            st.session_state.live_running = True
            st.session_state.history = pd.DataFrame(columns=["Time", "Toxic", "Neutral", "Hype"])
            st.session_state.chat_log = []
        
        if st.button("🛑 Stop Connection"):
            st.session_state.live_running = False

    with col2:
        st.subheader("Live Analytics Dashboard")
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()
        chat_placeholder = st.empty()

    if st.session_state.live_running and channel_name:
        server = 'irc.chat.twitch.tv'
        port = 6667
        nickname = 'justinfan12345' 
        
        sock = socket.socket()
        sock.connect((server, port))
        sock.send(f"NICK {nickname}\n".encode('utf-8'))
        sock.send(f"JOIN #{channel_name}\n".encode('utf-8'))
        sock.settimeout(1.0) 
        
        with st.spinner(f"Connected to #{channel_name}... Listening to chat..."):
            while st.session_state.live_running:
                try:
                    resp = sock.recv(2048).decode('utf-8')
                    if resp.startswith('PING'):
                        sock.send("PONG\n".encode('utf-8'))
                        continue
                        
                    message_match = re.search(r"PRIVMSG #[a-zA-Z0-9_]+ :(.+)", resp)
                    if message_match:
                        chat_text = message_match.group(1).strip()
                        
                        # We ignore the translated_text in the live view to keep the UI clean
                        diagnosis, probs, _ = analyze_message(chat_text)
                        current_time = time.strftime("%H:%M:%S")
                        
                        new_data = pd.DataFrame({
                            "Time": [current_time],
                            "Toxic": [probs[0]],
                            "Neutral": [probs[1]],
                            "Hype": [probs[2]]
                        })
                        st.session_state.history = pd.concat([st.session_state.history, new_data], ignore_index=True).tail(50)
                        
                        st.session_state.chat_log.append(f"{current_time} | {diagnosis} | {chat_text}")
                        if len(st.session_state.chat_log) > 10:
                            st.session_state.chat_log.pop(0)
                        
                        with metrics_placeholder.container():
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Current Toxicity", f"{(probs[0]*100):.1f}%")
                            m2.metric("Current Neutral", f"{(probs[1]*100):.1f}%")
                            m3.metric("Current Hype", f"{(probs[2]*100):.1f}%")

                        chart_placeholder.line_chart(
                            st.session_state.history.set_index("Time"), 
                            color=["#FF4B4B", "#FFFFFF", "#00CC96"] 
                        )
                        
                        with chat_placeholder.container():
                            st.code("\n".join(reversed(st.session_state.chat_log)))
                            
                        time.sleep(0.5) 
                        
                except socket.timeout:
                    continue 
                except Exception as e:
                    st.error(f"Socket disconnected: {e}")
                    break

# ==========================================
# TAB 2: MANUAL TESTING (Shows the Translation!)
# ==========================================
with tab2:
    st.subheader("Manual Inference Engine")
    user_input = st.text_input("Simulate a Twitch chat message:", placeholder="e.g., LUL bro missed his ultimate")

    if st.button("Analyze Message"):
        if user_input:
            diagnosis, probs, translated_text = analyze_message(user_input)
            
            st.info(f"**Data Preprocessing (What the AI actually sees):** {translated_text}")
            
            st.subheader(f"Diagnosis: {diagnosis}")
            st.progress(float(probs[0]), text=f"Toxic ({(probs[0]*100):.1f}%)")
            st.progress(float(probs[1]), text=f"Neutral ({(probs[1]*100):.1f}%)")
            st.progress(float(probs[2]), text=f"Hype ({(probs[2]*100):.1f}%)")