
# Twitch Chat AI Monitor

A real-time sentiment and toxicity analysis engine for Twitch, developed as a final-year engineering project at the **Faculté Polydisciplinaire de Khouribga (FPK)**.

---

## 🚀 Project Overview
This project addresses the challenges of high-velocity live stream moderation by providing an AI-driven, real-time analysis of chat sentiment. The system categorizes chat activity into **Hype**, **Neutral**, and **Toxic** sentiment patterns, enabling content creators to reduce moderator burnout and maintain safe, inclusive community interactions.

## 🛠 Technical Methodology
The system architecture focuses on speed and accuracy:
1.  **Ingestion Layer:** Persistent TCP socket connections (IRC Protocol) for real-time data streaming.
2.  **Preprocessing Firewall:** A custom heuristic `TWITCH_DICT` layer that normalizes gaming-specific slang and emotes (e.g., "Pog", "LULW") into standard linguistic vectors before AI processing.
3.  **Inference Engine:** A fine-tuned **DistilBERT** transformer model, optimized to provide classification results in under 200ms.
4.  **Telemetry & UI:** A Streamlit-powered dashboard that visualizes sentiment spikes and provides actionable chat telemetry logs.

## ⚙️ Tech Stack
* **Language:** Python 3.11.9
* **Deep Learning:** PyTorch, Hugging Face Transformers
* **Visualization:** Streamlit
* **Data Processing:** Pandas, Regex
* **Protocol:** Twitch IRC API

## 👥 Development Team
* **BAHRI Ilyass**
* **EL HICHAMI Oussama**

*Supervised by: **Dr. BAKKOURI Ibtissam**, Faculty of Sciences and Techniques of Khouribga (FPK).*

---

## 🛠 Installation Guide

### Prerequisites
* **Python Version:** 3.11.9

### Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ilyassbaa/Twitch-Chat-AI-Monitor.git](https://github.com/ilyassbaa/Twitch-Chat-AI-Monitor.git)
   cd Twitch-Chat-AI-Monitor

```

2. **Create a virtual environment:**
```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Run the application:**
```bash
streamlit run app.py

```



---

## 📊 Dashboard Overview

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## 🏷 Keywords

#Twitch #AI #MachineLearning #NLP #SentimentAnalysis #ToxicityDetection #StreamerTools #SoftwareEngineering #Python #PyTorch #HuggingFace #DistilBERT #Streamlit #DataScience #TechInternship #FPK #RealTimeSystems #ChatModeration #Innovation

