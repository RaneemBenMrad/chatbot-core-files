🚀 BI-GEEK Chatbot 🤖
Welcome to BI-GEEK Chatbot, a smart, multilingual assistant for BI-GEEK, Tunisia's leading academy for Business Intelligence (BI), Data Science, and Big Data training. This chatbot handles queries about courses, enrollment, and contacts via text or voice, with a modern twist! 🌟
✨ Features

🌍 Multilingual Magic: Supports English, French, and Arabic with language detection (langdetect).
🎙️ Text & Voice Input: Processes text and audio (WebM, WAV, MP3, OGG) with transcription via Groq's Whisper API or speech_recognition fallback.
🧠 Intent Detection: Uses Hugging Face transformers to classify user intents (greetings, course info, contact, etc.).
💾 Data Management:
PostgreSQL: Stores course and contact details.
SQLite: Logs chat history with language and input type (voice/text).


🤝 API Integration: Leverages Groq API for chat responses and audio transcription.
📜 Robust Logging: Tracks events with rotating logs for easy debugging.
🌐 Web Interface: Flask-powered with index.html and presentation.html, plus API endpoints for seamless interaction.
🔒 CORS Support: Enables cross-origin requests.

🛠️ Tech Stack

Backend: Flask, Python 🐍
AI/ML: Hugging Face transformers, Groq API 🚀
Databases: PostgreSQL, SQLite 📊
Audio Processing: pydub, speech_recognition 🎵
Language Detection: langdetect 🌐
Environment: python-dotenv 🔧
Logging: logging with RotatingFileHandler 📝
CORS: flask-cors 🔗

📦 Setup
Prerequisites

Python 3.8+ 🐍
PostgreSQL database 🗄️
Groq API key (set in .env) 🔑
Hugging Face model for intent classification (in models/intent_classifier) 🤖

Steps

Clone the Repo:
git clone https://github.com/your-username/bi-geek-chatbot.git
cd bi-geek-chatbot


Set Up Virtual Environment:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

ℹ️ Note: For pydub, install ffmpeg:

Ubuntu: sudo apt-get install ffmpeg
macOS: brew install ffmpeg
Windows: Download ffmpeg from ffmpeg.org and add to PATH.


Configure Environment Variables:Create a .env file in the root with:
GROQ_API_KEY=your_groq_api_key
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=bigeek
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_postgres_password


Set Up Databases:

PostgreSQL: Create a bigeek database with these tables:CREATE TABLE formations (
    id SERIAL PRIMARY KEY,
    title VARCHAR(100),
    title_fr VARCHAR(100),
    title_ar VARCHAR(100),
    description TEXT,
    description_fr TEXT,
    description_ar TEXT,
    duration_weeks INTEGER,
    price_dt INTEGER,
    modality VARCHAR(50)
);
CREATE TABLE keywords (
    id SERIAL PRIMARY KEY,
    formation_id INTEGER REFERENCES formations(id),
    keyword VARCHAR(50)
);
CREATE TABLE contacts (
    id SERIAL PRIMARY KEY,
    label VARCHAR(50),
    value VARCHAR(100)
);


SQLite: chatbot.db is auto-created on first run.


Launch the App:
python app.py

Access the chatbot at http://127.0.0.1:5001 🎉


🎮 Usage

Web Interface: Visit http://127.0.0.1:5001/ to chat or http://127.0.0.1:5001/presentation for a presentation page.
API Endpoints:
POST /chat: Send a text message (JSON: {"message": "your question"}).
POST /voice/transcribe: Transcribe audio (JSON: {"audio": "base64", "format": "webm"}).
POST /voice/chat: Process audio and get a response.
GET /chatbot/history: Fetch the last 10 interactions.
GET /test-course/<keyword>: Test course lookup by keyword.
GET /test-db: Check PostgreSQL connection.
GET /test-queries: Run predefined test queries.
GET /test-contact: Get contact info.
GET /test-voice: Check voice feature status.
POST /predict-intent: Predict intent for a text input.



📂 Project Structure
bi-geek-chatbot/
├── app.py                    # Main Flask app 🖥️
├── chatbot.db                # SQLite database for chat history 💾
├── chatbot.log               # Log file 📜
├── .env                     # Environment variables 🔒
├── models/
│   └── intent_classifier/   # Hugging Face model 🤖
├── static/
│   └── favicon.ico          # Favicon for web interface 🌐
├── templates/
│   ├── index.html           # Chat interface 💬
│   └── presentation.html     # Presentation page 📄
├── requirements.txt          # Python dependencies 📦
└── README.md                # This file 📖

📝 Logging
Logs are saved in chatbot.log (max 1MB, 5 backups) for easy debugging. 🕵️‍♂️

🤝 Contributing
Fork the repo 🍴
Create a branch (git checkout -b feature/your-feature)
Commit changes (git commit -m "Add your feature")
Push the branch (git push origin feature/your-feature)
Open a Pull Request 🚀

