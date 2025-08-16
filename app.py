from flask import Flask, request, jsonify, render_template, send_from_directory
import psycopg2
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sqlite3
import os
from dotenv import load_dotenv
from groq import Groq
import logging
from logging.handlers import RotatingFileHandler
from flask_cors import CORS
from langdetect import detect_langs, DetectorFactory
import speech_recognition as sr
import base64
import io
from pydub import AudioSegment
import tempfile

# Ensure consistent language detection results
DetectorFactory.seed = 0

# Configure logging with UTF-8 encoding
log_file = os.path.join(os.path.dirname(__file__), 'chatbot.log')
handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=5, encoding='utf-8')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').setLevel(logging.INFO)
logging.getLogger('').addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
console_handler.setLevel(logging.INFO)
logging.getLogger('').addHandler(console_handler)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

app = Flask(__name__)
CORS(app)

# Initialize Groq client
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    timeout=120.0,
    max_retries=3
)

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# Load Hugging Face model and tokenizer
try:
    model_path = 'models/intent_classifier'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    logging.info("Hugging Face model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load Hugging Face model: {str(e)}")
    model = None
    tokenizer = None

# SYSTEM PROMPTS for Groq (multilingual)
SYSTEM_PROMPTS = {
    "en": """
You are a helpful assistant for BI-GEEK, a leading academy in Tunisia specializing in Business Intelligence (BI), Data Science, and Big Data training. ONLY respond to questions about BI-GEEK's courses, enrollment, instructors, or related services. If the query is unrelated to BI-GEEK, respond with: "Sorry, I can only assist with BI-GEEK training programs. Please ask about our courses or contact contact@bi-geek.net."
Available courses:
- Power BI (8 weeks, 700,000 DT, online available)
- Data Engineering Bootcamp (20 weeks, 2,500,000 DT)
- Microsoft Certifications (PL-300, DP-900, DP-203, 8 weeks, 700,000 DT each, online available)
- Python for Data Science (8 weeks, 700,000 DT, online available)
- Big Data (20 weeks, 3,000,000 DT)
- 3-day workshops (700,000 DT)
Courses start quarterly (January, April, July, October). For enrollment, direct users to www.bi-geek.net ('Inscription' section), contact@bi-geek.net, or +216 58 611 283. Instructors are certified experts. If unsure, suggest contacting BI-GEEK.
BI-GEEK location: 53 Rue Du Lac Léman, Tunis 1053. Google Maps: https://www.google.com/maps?gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEIODM0MmowajGoAgCwAgA&um=1&ie=UTF-8&fb=1&gl=tn&sa=X&geocode=KYncDLhUNf0SMbGeNaoRSY_6&daddr=53+Rue+Du+Lac+L%C3%A9man,+Tunis+1053
""",
    "fr": """
Vous êtes un assistant utile pour BI-GEEK, une académie de premier plan en Tunisie spécialisée dans la formation en Business Intelligence (BI), Data Science et Big Data. Répondez UNIQUEMENT aux questions concernant les cours, les inscriptions, les formateurs ou les services de BI-GEEK. Si la question n'est pas liée à BI-GEEK, répondez : "Désolé, je ne peux répondre qu'aux questions sur les programmes de formation BI-GEEK. Veuillez poser une question sur nos cours ou contacter contact@bi-geek.net."
Cours disponibles :
- Power BI (8 semaines, 700 000 DT, disponible en ligne)
- Bootcamp Ingénierie des Données (20 semaines, 2 500 000 DT)
- Certifications Microsoft (PL-300, DP-900, DP-203, 8 semaines, 700 000 DT chacune, disponible en ligne)
- Python pour la Data Science (8 semaines, 700 000 DT, disponible en ligne)
- Big Data (20 semaines, 3 000 000 DT)
- Ateliers de 3 jours (700 000 DT)
Les cours commencent tous les trimestres (janvier, avril, juillet, octobre). Pour s'inscrire, dirigez les utilisateurs vers www.bi-geek.net (section 'Inscription'), contact@bi-geek.net, ou +216 58 611 283. Les formateurs sont des experts certifiés. En cas d'incertitude, suggérez de contacter BI-GEEK.
Adresse de BI-GEEK : 53 Rue Du Lac Léman, Tunis 1053. Google Maps : https://www.google.com/maps?gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEIODM0MmowajGoAgCwAgA&um=1&ie=UTF-8&fb=1&gl=tn&sa=X&geocode=KYncDLhUNf0SMbGeNaoRSY_6&daddr=53+Rue+Du+Lac+L%C3%A9man,+Tunis+1053
""",
    "ar": """
أنت مساعد مفيد لـ BI-GEEK، وهي أكاديمية رائدة في تونس متخصصة في التدريب على الذكاء التجاري (BI)، علوم البيانات، والبيانات الضخمة. أجب فقط على الأسئلة المتعلقة بدورات BI-GEEK، التسجيل، المدربين، أو الخدمات ذات الصلة. إذا كان السؤال غير مرتبط بـ BI-GEEK، أجب: "عذرًا، يمكنني فقط المساعدة في برامج تدريب BI-GEEK. يرجى السؤال عن دوراتنا أو التواصل مع contact@bi-geek.net."
الدورات المتاحة:
- Power BI (8 أسابيع، 700,000 دينار تونسي، متاحة عبر الإنترنت)
- معسكر هندسة البيانات (20 أسبوعًا، 2,500,000 دينار تونسي)
- شهادات مايكروسوفت (PL-300، DP-900، DP-203، 8 أسابيع، 700,000 دينار تونسي لكل منها، متاحة عبر الإنترنت)
- بايثون لعلوم البيانات (8 أسابيع، 700,000 دينار تونسي، متاحة عبر الإنترنت)
- البيانات الضخمة (20 أسبوعًا، 3,000,000 دينار تونسي)
- ورش عمل لمدة 3 أيام (700,000 دينار تونسي)
تبدأ الدورات كل ربع سنة (يناير، أبريل، يوليو، أكتوبر). للتسجيل، وجه المستخدمين إلى www.bi-geek.net (قسم "التسجيل")، contact@bi-geek.net، أو +216 58 611 283. المدربون خبراء معتمدون. إذا لم تكن متأكدًا، اقترح التواصل مع BI-GEEK.
موقع BI-GEEK: 53 Rue Du Lac Léman, Tunis 1053. خرائط جوجل: https://www.google.com/maps?gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEIODM0MmowajGoAgCwAgA&um=1&ie=UTF-8&fb=1&gl=tn&sa=X&geocode=KYncDLhUNf0SMbGeNaoRSY_6&daddr=53+Rue+Du+Lac+L%C3%A9man,+Tunis+1053
"""
}


# -------------------- Voice Recognition Functions --------------------
def transcribe_audio_groq(audio_file_path):
    """Groq Whisper avec fallback sur langues spécifiques"""
    try:
        with open(audio_file_path, "rb") as file:
            # Première tentative sans langue spécifiée
            transcription = groq_client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3",
                response_format="text"
            )
        return transcription
    except Exception as e:
        # Si ça échoue, essayer avec français (langue principale en Tunisie)
        try:
            with open(audio_file_path, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3",
                    response_format="text",
                    language="fr"
                )
            return transcription
        except Exception as e2:
            logging.error(f"Groq transcription failed: {e2}")
            return None


def transcribe_audio_local(audio_file_path, language='auto'):
    """Fallback: Local transcription using SpeechRecognition"""
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)

        # Try different recognition methods
        try:
            if language == 'auto' or language == 'en':
                text = recognizer.recognize_google(audio, language='en-US')
            elif language == 'fr':
                text = recognizer.recognize_google(audio, language='fr-FR')
            elif language == 'ar':
                text = recognizer.recognize_google(audio, language='ar-EG')
            else:
                text = recognizer.recognize_google(audio)

            logging.info(f"Local transcription successful: {text}")
            return text
        except sr.UnknownValueError:
            logging.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            logging.error(f"Google Speech Recognition error: {e}")
            return None
    except Exception as e:
        logging.error(f"Local transcription failed: {e}")
        return None


def process_audio_data(audio_data, format='webm'):
    """Process base64 audio data and convert to WAV"""
    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data.split(',')[1])

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as temp_input:
            temp_input.write(audio_bytes)
            temp_input_path = temp_input.name

        # Convert to WAV using pydub
        if format.lower() in ['webm', 'ogg']:
            audio = AudioSegment.from_file(temp_input_path, format='webm')
        elif format.lower() == 'mp3':
            audio = AudioSegment.from_mp3(temp_input_path)
        elif format.lower() == 'wav':
            audio = AudioSegment.from_wav(temp_input_path)
        else:
            audio = AudioSegment.from_file(temp_input_path)

        # Export as WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
            temp_output_path = temp_output.name

        audio.export(temp_output_path, format='wav')

        # Clean up input file
        os.unlink(temp_input_path)

        logging.info(f"Audio converted successfully: {format} -> WAV")
        return temp_output_path

    except Exception as e:
        logging.error(f"Audio processing failed: {e}")
        return None


# -------------------- PostgreSQL Connection --------------------
def get_pg_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database=os.getenv("POSTGRES_DB", "bigeek"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        logging.info(
            f"Connected to PostgreSQL - Host: {os.getenv('POSTGRES_HOST', 'localhost')}, Database: {os.getenv('POSTGRES_DB', 'bigeek')}, Schema: public")
        cur = conn.cursor()
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = cur.fetchall()
        logging.info(f"Tables in public schema: {[table[0] for table in tables]}")
        cur.close()
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to PostgreSQL: {e}")
        raise


# -------------------- SQLite for Chat History --------------------
def init_db():
    try:
        conn = sqlite3.connect('chatbot.db')
        c = conn.cursor()

        c.execute('DROP TABLE IF EXISTS chat_history')
        c.execute('''CREATE TABLE chat_history
                     (id INTEGER PRIMARY KEY, 
                      user_message TEXT, 
                      bot_response TEXT, 
                      language TEXT DEFAULT 'en',
                      is_voice INTEGER DEFAULT 0)''')
        conn.commit()
        conn.close()
        logging.info("SQLite database recreated with language and voice columns")
    except Exception as e:
        logging.error(f"Failed to initialize SQLite database: {e}")


def insert_chat_history(user_message, bot_response, language, is_voice=False):
    try:
        conn = sqlite3.connect('chatbot.db')
        c = conn.cursor()
        c.execute("INSERT INTO chat_history (user_message, bot_response, language, is_voice) VALUES (?, ?, ?, ?)",
                  (user_message, bot_response, language, 1 if is_voice else 0))
        conn.commit()
        conn.close()
        logging.info(f"Inserted chat history: {user_message[:50]}... (language: {language}, voice: {is_voice})")
    except Exception as e:
        logging.error(f"Failed to insert chat history: {e}")


def get_recent_chat_history(limit=5):
    try:
        conn = sqlite3.connect('chatbot.db')
        c = conn.cursor()
        c.execute("SELECT user_message, bot_response, language FROM chat_history ORDER BY id DESC LIMIT ?", (limit,))
        history = c.fetchall()
        conn.close()
        bi_geek_terms = [
            "course", "training", "power bi", "data science", "big data", "certification", "python",
            "microsoft", "bi-geek", "courses", "formations", "subscribe", "enroll", "contact", "how",
            "join", "inscription", "phone", "email", "support", "joindre", "téléphone", "numéro",
            "adresse", "location", "makan", "help", "informations", "infos", "contacter", "coordonnee",
            "connect", "reach", "دورة", "تدريب", "باور بي آي", "علوم البيانات", "بيانات ضخمة",
            "شهادة", "بايثون", "مايكروسوفت", "بي-جيك", "دورات", "تسجيل", "اتصال", "كيف",
            "انضمام", "هاتف", "بريد", "دعم", "معلومات", "تواصل", "موقع"
        ]
        filtered = []
        for user, bot, lang in reversed(history):
            if any(term in user.lower() or term in bot.lower() for term in bi_geek_terms):
                filtered.append({"role": "user", "content": user, "language": lang})
                filtered.append({"role": "assistant", "content": bot, "language": lang})
        logging.info(f"Retrieved {len(filtered)} relevant chat history entries")
        return filtered[:limit * 2]
    except Exception as e:
        logging.error(f"Failed to retrieve chat history: {e}")
        return []


# -------------------- Language Detection with langdetect --------------------
def detect_language(text):
    try:
        langs = detect_langs(text)
        supported_langs = ['en', 'fr', 'ar']
        for lang_prob in langs:
            lang = lang_prob.lang
            if lang in supported_langs:
                logging.info(f"Detected language for '{text}': {lang} (probability: {lang_prob.prob})")
                return lang
        logging.info(f"No supported language detected for '{text}', defaulting to 'en'")
        return 'en'
    except Exception as e:
        logging.error(f"Langdetect language detection failed for '{text}': {e}")
        return "en"


# -------------------- Formation Lookup in PostgreSQL --------------------
def get_course_by_keyword(user_input, language='en'):
    try:
        conn = get_pg_connection()
        cur = conn.cursor()
        title_col = f"title_{language}" if language in ['fr', 'ar'] else "title"
        desc_col = f"description_{language}" if language in ['fr', 'ar'] else "description"
        query = f"""
            SELECT DISTINCT f.{title_col}, f.duration_weeks, f.price_dt, f.{desc_col}, f.modality
            FROM public.formations f
            LEFT JOIN public.keywords k ON f.id = k.formation_id
            WHERE LOWER(k.keyword) LIKE %s
               OR LOWER(f.{title_col}) LIKE %s
               OR LOWER(f.{desc_col}) LIKE %s
            ORDER BY f.{title_col}
            LIMIT 3;
        """
        words = user_input.strip().lower().split()
        results = []
        for word in words:
            like_input = f"%{word}%"
            cur.execute(query, (like_input, like_input, like_input))
            results.extend(cur.fetchall())
        results = list(set(results))
        logging.info(f"Query executed for '{user_input}' (language: {language}), results: {len(results)} courses found")
        cur.close()
        conn.close()
        if results:
            if language == 'fr':
                response = "\n".join(
                    [f"{title} est un cours de {duration} semaines ({modality}) pour {price} DT. Description : {desc}"
                     for
                     title, duration, price, desc, modality in results])
            elif language == 'ar':
                response = "\n".join(
                    [f"{title} هي دورة مدتها {duration} أسابيع ({modality}) بتكلفة {price} دينار تونسي. الوصف: {desc}"
                     for
                     title, duration, price, desc, modality in results])
            else:
                response = "\n".join(
                    [f"{title} is a {duration}-week course ({modality}) for {price} DT. Description: {desc}" for
                     title, duration, price, desc, modality in results])
            logging.info(f"DB match for '{user_input}' (language: {language}): {response[:100]}...")
            return response
        logging.info(f"No DB match for '{user_input}' (language: {language})")
        return None
    except Exception as e:
        logging.error(f"DB error for '{user_input}' (language: {language}): {e}")
        return None


# -------------------- Contact Lookup in PostgreSQL --------------------
def get_contact_info(language='en'):
    try:
        conn = get_pg_connection()
        cur = conn.cursor()
        logging.info("Executing contact query on public.contacts")
        query = "SELECT label, value FROM public.contacts;"
        cur.execute(query)
        results = cur.fetchall()
        logging.info(f"Contact query results: {results}")
        cur.close()
        conn.close()
        email = phone = website = None
        for label, value in results:
            if label.lower() == "email":
                email = value
            elif label.lower() == "phone":
                phone = value
            elif label.lower() == "website":
                website = value
        contact_text = []
        if email:
            contact_text.append(
                f"📧 {'Email' if language == 'en' else 'Courriel' if language == 'fr' else 'البريد الإلكتروني'}: {email}")
        if phone:
            contact_text.append(
                f"📞 {'Phone' if language == 'en' else 'Téléphone' if language == 'fr' else 'الهاتف'}: {phone}")
        if website:
            contact_text.append(
                f"🌐 {'Website' if language == 'en' else 'Site web' if language == 'fr' else 'الموقع الإلكتروني'}: {website}")
        if contact_text:
            response = f"{'Here is the contact information for BI-GEEK' if language == 'en' else 'Voici les informations de contact pour BI-GEEK' if language == 'fr' else 'معلومات التواصل مع BI-GEEK'}:\n" + "\n".join(
                contact_text)
        else:
            response = {
                'en': "Contact information: Email: contact@bi-geek.net, Phone: +216 58 611 283, Website: www.bi-geek.net",
                'fr': "Informations de contact : Courriel : contact@bi-geek.net, Téléphone : +216 58 611 283, Site web : www.bi-geek.net",
                'ar': "معلومات التواصل: البريد الإلكتروني: contact@bi-geek.net، الهاتف: +216 58 611 283، الموقع الإلكتروني: www.bi-geek.net"
            }[language]
        logging.info(f"Contact response (language: {language}): {response[:100]}...")
        return response
    except Exception as e:
        logging.error(f"DB error in get_contact_info (language: {language}): {e}")
        try:
            conn = get_pg_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'contacts';")
            table_exists = cur.fetchone()
            logging.info(f"Table 'contacts' exists in public schema: {bool(table_exists)}")
            cur.close()
            conn.close()
            if not table_exists:
                logging.info("Creating contacts table as it does not exist")
                conn = get_pg_connection()
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS public.contacts (
                        id SERIAL PRIMARY KEY,
                        label VARCHAR(50),
                        value VARCHAR(100)
                    );
                    INSERT INTO public.contacts (label, value) VALUES
                    ('email', 'contact@bi-geek.net'),
                    ('phone', '+216 58 611 283'),
                    ('website', 'www.bi-geek.net')
                    ON CONFLICT DO NOTHING;
                """)
                conn.commit()
                cur.close()
                conn.close()
                logging.info("Contacts table created and populated")
                return get_contact_info(language=language)
        except Exception as e2:
            logging.error(f"Error checking or creating table: {e2}")
        return {
            'en': "Contact information: Email: contact@bi-geek.net, Phone: +216 58 611 283, Website: www.bi-geek.net",
            'fr': "Informations de contact : Courriel : contact@bi-geek.net, Téléphone : +216 58 611 283, Site web : www.bi-geek.net",
            'ar': "معلومات التواصل: البريد الإلكتروني: contact@bi-geek.net، الهاتف: +216 58 611 283، الموقع الإلكتروني: www.bi-geek.net"
        }[language]


# -------------------- Intent Classification with Hugging Face --------------------
def classify_intent(text):
    try:
        if not model or not tokenizer:
            logging.error("Hugging Face model or tokenizer not loaded")
            return "other"

        clean_text = text.strip().lower()
        clean_text = clean_text.replace("!", " ").replace("?", " ").replace(".", " ")
        clean_text = " ".join(clean_text.split())

        inputs = tokenizer(
            clean_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=64
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()

        intent = model.config.id2label[predicted_id]

        logging.info(f"HF Intent: '{text[:50]}...' -> {intent} (conf: {confidence:.3f})")

        if confidence < 0.5:
            logging.warning(f"Low confidence ({confidence:.3f}), defaulting to 'other'")
            return "other"

        return intent

    except Exception as e:
        logging.error(f"Error in HF intent classification for '{text}': {e}")
        return "other"


# -------------------- Main Chat Response Logic --------------------
def get_response(user_input, is_voice=False):
    try:
        logging.info(f"=== PROCESSING: '{user_input}' (Voice: {is_voice}) ===")

        lang = detect_language(user_input)
        clean = user_input.lower().strip()
        logging.info(f"Detected language: {lang}, Clean input: '{clean}'")

        intent = classify_intent(clean)
        logging.info(f"🎯 CLASSIFIED INTENT: '{intent}' for input '{clean}'")

        if intent == "greeting":
            logging.info("✅ INTENT: greeting - Returning greeting response")
            response = {
                'en': "Hello! I'm here to help with BI-GEEK's courses, enrollment, or contact information. What would you like to know?",
                'fr': "Bonjour ! Je suis là pour aider avec les cours, l'inscription ou les informations de contact de BI-GEEK. Que voulez-vous savoir ?",
                'ar': "مرحبًا! أنا هنا للمساعدة في دورات BI-GEEK، التسجيل، أو معلومات التواصل. ماذا تريد معرفته؟"
            }[lang]
            insert_chat_history(user_input, response, lang, is_voice)
            return response

        elif intent == "thanks":
            logging.info("✅ INTENT: thanks - Returning thanks response")
            response = {
                'en': "You're welcome! How else can I assist you with BI-GEEK courses?",
                'fr': "De rien ! Comment puis-je encore vous aider avec les cours BI-GEEK ?",
                'ar': "على الرحب والسعة! كيف يمكنني مساعدتك أكثر مع دورات BI-GEEK؟"
            }[lang]
            insert_chat_history(user_input, response, lang, is_voice)
            return response

        elif intent == "contact":
            logging.info("✅ INTENT: contact - Returning contact response")
            response = get_contact_info(language=lang)
            insert_chat_history(user_input, response, lang, is_voice)
            return response

        elif intent == "certifications":
            logging.info("✅ INTENT: certifications - Returning certification response")
            response = {
                'en': "BI-GEEK offers certified courses including:\n- PL-300 (Power BI Data Analyst)\n- DP-900 (Azure Data Fundamentals)\n- DP-203 (Azure Data Engineering)\nEach certification runs for 8 weeks and costs 700,000 DT.\nCourses are online and prepare for official Microsoft exams.",
                'fr': "BI-GEEK propose des cours certifiés incluant :\n- PL-300 (Analyste de données Power BI)\n- DP-900 (Fondamentaux des données Azure)\n- DP-203 (Ingénierie des données Azure)\nChaque certification dure 8 semaines et coûte 700 000 DT.\nLes cours sont en ligne et préparent aux examens officiels Microsoft.",
                'ar': "تقدم BI-GEEK دورات معتمدة تشمل:\n- PL-300 (محلل بيانات Power BI)\n- DP-900 (أساسيات بيانات Azure)\n- DP-203 (هندسة بيانات Azure)\nكل شهادة تستمر 8 أسابيع وتكلف 700,000 دينار تونسي.\nالدورات متاحة عبر الإنترنت وتُعد لامتحانات مايكروسوفت الرسمية."
            }[lang]
            insert_chat_history(user_input, response, lang, is_voice)
            return response

        elif intent == "enroll":
            logging.info("✅ INTENT: enroll - Returning enrollment response")
            response = {
                'en': "To subscribe, visit www.bi-geek.net (Inscription section), email contact@bi-geek.net, or call +216 58 611 283. Courses start quarterly (January, April, July, October).",
                'fr': "Pour vous inscrire, visitez www.bi-geek.net (section Inscription), envoyez un email à contact@bi-geek.net, ou appelez le +216 58 611 283. Les cours commencent tous les trimestres (janvier, avril, juillet, octobre).",
                'ar': "للتسجيل، زر www.bi-geek.net (قسم التسجيل)، أرسل بريدًا إلكترونيًا إلى contact@bi-geek.net، أو اتصل على +216 58 611 283. تبدأ الدورات كل ربع سنة (يناير، أبريل، يوليو، أكتوبر)."
            }[lang]
            insert_chat_history(user_input, response, lang, is_voice)
            return response

        elif intent == "course_info":
            logging.info("✅ INTENT: course_info - Checking database for course info")
            course_info = get_course_by_keyword(clean, language=lang)
            if course_info:
                insert_chat_history(user_input, course_info, lang, is_voice)
                return course_info

        # Nouvelle condition pour location/adresse/makan
        elif intent == "location" or any(keyword in clean for keyword in ["location", "adresse", "makan"]):
            logging.info("✅ INTENT: location - Returning location response")
            response = {
                'en': "BI-GEEK is located at 53 Rue Du Lac Léman, Tunis 1053. Check it on Google Maps: https://www.google.com/maps?gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEIODM0MmowajGoAgCwAgA&um=1&ie=UTF-8&fb=1&gl=tn&sa=X&geocode=KYncDLhUNf0SMbGeNaoRSY_6&daddr=53+Rue+Du+Lac+L%C3%A9man,+Tunis+1053",
                'fr': "BI-GEEK se trouve au 53 Rue Du Lac Léman, Tunis 1053. Consultez sur Google Maps : https://www.google.com/maps?gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEIODM0MmowajGoAgCwAgA&um=1&ie=UTF-8&fb=1&gl=tn&sa=X&geocode=KYncDLhUNf0SMbGeNaoRSY_6&daddr=53+Rue+Du+Lac+L%C3%A9man,+Tunis+1053",
                'ar': "يقع BI-GEEK في 53 Rue Du Lac Léman, Tunis 1053. تحقق منه على خرائط جوجل: https://www.google.com/maps?gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEIODM0MmowajGoAgCwAgA&um=1&ie=UTF-8&fb=1&gl=tn&sa=X&geocode=KYncDLhUNf0SMbGeNaoRSY_6&daddr=53+Rue+Du+Lac+L%C3%A9man,+Tunis+1053"
            }[lang]
            insert_chat_history(user_input, response, lang, is_voice)
            return response

        logging.info(f"⚠️ INTENT: '{intent}' not handled specifically, going to Groq")

        chat_history = get_recent_chat_history()
        course_context = get_course_by_keyword(clean, language=lang) or {
            'en': "No course data found in the database.",
            'fr': "Aucune information sur les cours trouvée dans la base de données.",
            'ar': "لم يتم العثور على بيانات الدورة في قاعدة البيانات."
        }[lang]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[lang] + "\nRelevant data:\n" + course_context},
            *[{"role": entry["role"], "content": entry["content"]} for entry in chat_history],
            {"role": "user", "content": user_input}
        ]

        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=500
        )

        response = chat_completion.choices[0].message.content
        insert_chat_history(user_input, response, lang, is_voice)
        logging.info(f"Groq response for '{user_input}' (language: {lang}): {response[:100]}...")
        return response

    except Exception as e:
        logging.error(f"Error processing '{user_input}': {e}")
        response = {
            'en': "Sorry, I couldn't process that request. Please try again or contact contact@bi-geek.net.",
            'fr': "Désolé, je n'ai pas pu traiter cette demande. Veuillez réessayer ou contacter contact@bi-geek.net.",
            'ar': "عذرًا، لم أتمكن من معالجة هذا الطلب. يرجى المحاولة مرة أخرى أو التواصل مع contact@bi-geek.net."
        }[lang if 'lang' in locals() else 'en']
        insert_chat_history(user_input, response, lang if 'lang' in locals() else 'en', is_voice)
        return response


# -------------------- Test de chargement du modèle --------------------
def test_model_loading():
    try:
        if not model or not tokenizer:
            logging.error("❌ Model or tokenizer is None!")
            return False

        test_cases = [
            "thanks",
            "thank you",
            "hello",
            "how to enroll",
            "contact info",
            "where is your location"
        ]

        for test_input in test_cases:
            intent = classify_intent(test_input)
            logging.info(f"TEST: '{test_input}' -> {intent}")

        logging.info("✅ Model testing completed")
        return True

    except Exception as e:
        logging.error(f"❌ Model test failed: {e}")
        return False


# -------------------- Routes --------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = get_response(user_input)
    return jsonify({'response': response})


# -------------------- NEW: Voice Routes --------------------
@app.route('/voice/transcribe', methods=['POST'])
def transcribe_voice():
    """Transcribe voice message using Groq Whisper"""
    try:
        data = request.json
        audio_data = data.get('audio')
        format_type = data.get('format', 'webm')

        if not audio_data:
            return jsonify({'error': 'No audio data provided'}), 400

        # Process audio data
        audio_file_path = process_audio_data(audio_data, format_type)
        if not audio_file_path:
            return jsonify({'error': 'Failed to process audio'}), 500

        # Try Groq Whisper first
        transcription = transcribe_audio_groq(audio_file_path)

        # Fallback to local transcription if Groq fails
        if not transcription:
            logging.warning("Groq transcription failed, trying local transcription")
            transcription = transcribe_audio_local(audio_file_path)

        # Clean up temporary file
        try:
            os.unlink(audio_file_path)
        except:
            pass

        if not transcription:
            return jsonify({'error': 'Could not transcribe audio'}), 500

        return jsonify({
            'transcription': transcription,
            'language': detect_language(transcription)
        })

    except Exception as e:
        logging.error(f"Voice transcription error: {e}")
        return jsonify({'error': 'Transcription failed'}), 500


@app.route('/voice/chat', methods=['POST'])
def voice_chat():
    try:
        data = request.json
        audio_data = data.get('audio')

        # Décoder et sauver directement
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name

        # Groq accepte WebM directement !
        transcription = transcribe_audio_groq(temp_file_path)
        os.unlink(temp_file_path)

        if not transcription:
            return jsonify({'error': 'Transcription failed'}), 500

        response = get_response(transcription, is_voice=True)
        return jsonify({
            'transcription': transcription,
            'response': response,
            'language': detect_language(transcription)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/chatbot/history', methods=['GET'])
def chat_history():
    history = get_recent_chat_history(limit=10)
    return jsonify({'history': history})


@app.route('/test-course/<mot>', methods=['GET'])
def test_course(mot):
    lang = detect_language(mot)
    info = get_course_by_keyword(mot, language=lang)
    return jsonify({'result': info})


@app.route('/test-db')
def test_db():
    try:
        conn = get_pg_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM public.formations;")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        logging.info(f"Test DB: {count} formations found")
        return f"✅ Connected to PostgreSQL! {count} formations found."
    except Exception as e:
        logging.error(f"Test DB failed: {e}")
        return f"❌ Failed to connect to PostgreSQL: {e}"


@app.route('/test-queries', methods=['GET'])
def test_queries():
    test_cases = [
        "Power BI course",
        "price of data science",
        "next course start",
        "what is python",
        "unrelated topic",
        "Cours Power BI ?",
        "شهادة PL-300؟",
        "where is your location",
        "quelle est votre adresse",
        "أين الموقع"
    ]
    results = []
    for query in test_cases:
        response = get_response(query)
        results.append({"query": query, "response": response})
    logging.info(f"Test queries executed: {len(results)} results")
    return jsonify({"test_results": results})


@app.route('/test-contact', methods=['GET'])
def test_contact():
    response = get_contact_info(language='en')
    return jsonify({"response": response})


@app.route('/test-voice', methods=['GET'])
def test_voice():
    """Test voice functionality"""
    try:
        return jsonify({
            "groq_whisper": "available" if groq_client else "unavailable",
            "local_speech": "available",
            "supported_formats": ["webm", "wav", "mp3", "ogg"],
            "languages": ["en", "fr", "ar"]
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico',
                               mimetype='image/vnd.microsoft.icon')


@app.route('/predict-intent', methods=['POST'])
def predict_intent():
    text = request.json.get('text')
    intent = classify_intent(text)
    return jsonify({'intent': intent})


@app.route('/presentation')
def presentation():
    return render_template('presentation.html')


# -------------------- Run --------------------
if __name__ == '__main__':
    try:
        conn = get_pg_connection()
        print("✅ Database connection successful!")
        conn.close()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")

    init_db()

    print("🧪 Testing Hugging Face model...")
    test_model_loading()

    print("🎤 Voice features enabled!")
    print("🚀 Starting Flask server on http://127.0.0.1:5001...")
    app.run(host='127.0.0.1', port=5001, debug=False)