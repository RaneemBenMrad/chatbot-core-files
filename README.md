BI-GEEK Chatbot
A multilingual Flask-based chatbot for BI-GEEK, a leading academy specializing in Business Intelligence (BI), Data Science, and Big Data training...... The chatbot supports text and voice inputs, provides information about BI-GEEK's courses, enrollment, instructors, and contact details, and is designed to be deployed on GitHub.
Features

Multilingual Support: Handles queries in English, French, and Arabic using language detection (langdetect).
Text and Voice Input: Processes text queries and audio inputs (WebM, WAV, MP3, OGG) with transcription via Groq's Whisper API or local speech_recognition as a fallback.
Intent Classification: Uses a Hugging Face transformers model to classify user intents (e.g., greeting, course info, contact, enrollment).
Database Integration:
PostgreSQL: Stores course and contact information, queried for course details based on keywords.
SQLite: Maintains chat history with user messages, bot responses, language, and voice input flags.


API Integration: Uses Groq's API for chat completions and audio transcription.
Logging: Implements robust logging with rotation for debugging and monitoring.
CORS Support: Enabled for cross-origin requests.
Web Interface: Includes Flask routes for a web interface (index.html, presentation.html) and API endpoints for chat, voice, and testing.

Technologies Used

Backend: Flask, Python
AI/ML: Hugging Face transformers for intent classification, Groq API for chat completions and audio transcription
Databases: PostgreSQL for course/contact data, SQLite for chat history
Audio Processing: pydub for audio format conversion, speech_recognition for local transcription
Language Detection: langdetect for multilingual support
Environment Management: python-dotenv for environment variables
Logging: logging with RotatingFileHandler for log management
CORS: flask-cors for cross-origin resource sharing

Usage
Web Interface: Access the chatbot at http://127.0.0.1:5001/ for text-based interaction or http://127.0.0.1:5001/presentation for a presentation page.
API Endpoints:
POST /chat: Send a JSON payload with message for text-based chat responses.
POST /voice/transcribe: Send a JSON payload with audio (base64-encoded) and format for audio transcription.
POST /voice/chat: Send a JSON payload with audio for voice-based chat with transcription and response.
GET /chatbot/history: Retrieve recent chat history (up to 10 entries).
GET /test-course/<mot>: Test course lookup by keyword.
GET /test-db: Test PostgreSQL connection.
GET /test-queries: Test predefined queries for debugging.
GET /test-contact: Retrieve contact information.
GET /test-voice: Check voice feature availability.
POST /predict-intent: Predict intent for a given text input.

Project Structure
bi-geek-chatbot/
├── app.py                    # Main Flask application
├── chatbot.db                # SQLite database for chat history
├── chatbot.log               # Log file
├── .env                     # Environment variables
├── models/
│   └── intent_classifier/   # Hugging Face model for intent classification
├── static/
│   └── favicon.ico          # Favicon for web interface
├── templates/
│   ├── index.html           # Main chat interface
│   └── presentation.html     # Presentation page
├── requirements.txt          # Python dependencies
└── README.md                # This file

