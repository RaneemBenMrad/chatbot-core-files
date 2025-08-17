import json
import random
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
import tempfile
import shutil

# Configuration
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def create_ultra_balanced_dataset():
    """Dataset ULTRA équilibré avec plus d'exemples par classe pour 89-92% accuracy"""

    training_data = {
        "greeting": [
            # Salutations basiques (20)
            "Hello", "Hi", "Hey", "Good morning", "Good afternoon", "Good evening",
            "Bonjour", "Salut", "Coucou", "Bonsoir", "Hello there", "Hi there",
            "Hey there", "Greetings", "Good day", "Morning", "Evening", "Howdy",
            "What's up", "How are you",

            # Variations avec contexte (20)
            "Hello how are you today", "Hi everyone", "Hey buddy", "Good morning team",
            "Hello world", "Hi there friend", "Hey what's happening", "Good day to you",
            "Hello nice to meet you", "Hi how's it going", "Hey long time no see",
            "Good afternoon everyone", "Hello hope you're well", "Hi glad to see you",
            "Hey how are things", "Good evening folks", "Hello welcome", "Hi great day",
            "Hey how's your day", "Good to see you",

            # Salutations formelles/informelles (20)
            "Good to meet you", "Pleased to see you", "Welcome aboard", "Nice day isn't it",
            "Hope you're doing well", "Lovely to connect", "Great to be here", "Happy to chat",
            "Wonderful morning", "Excellent evening", "Beautiful day", "Perfect timing",
            "Glad we connected", "Nice to chat", "Great to talk", "Happy to help",
            "Wonderful to meet", "Excellent connection", "Beautiful greeting", "Perfect hello",

            # Variations internationales (20)
            "Hola amigo", "Guten tag", "Buenos dias", "Bonjour mon ami", "Buongiorno",
            "Konnichiwa", "Namaste friend", "Shalom", "Hallo there", "Salve",
            "Aloha", "Jambo", "Sawubona", "Merhaba", "Zdravo", "Hej", "Ciao",
            "Servus", "Olá", "Privet"
        ],

        "enroll": [
            # Questions directes (20)
            "How can I enroll", "I want to sign up", "How to register", "Can I enroll",
            "Sign me up", "I want to join", "How do I register", "Registration process",
            "I'd like to register", "Can I register", "Want to enroll", "Need to register",
            "How to sign up", "Enrollment process", "I'm ready to enroll", "Register me",
            "Sign up process", "Enrollment info", "Registration info", "Join now",

            # Processus et démarches (20)
            "What are the enrollment steps", "Enrollment procedure", "Registration requirements",
            "How to become a student", "Application process", "How to apply", "I want to apply",
            "Application form", "Admission process", "How to get admitted", "Enrollment form",
            "Registration form", "Sign up form", "Application requirements", "Admission requirements",
            "What documents needed", "Enrollment deadline", "Registration deadline", "When can I register",
            "Is registration open",

            # Urgence et rapidité (20)
            "I need to register now", "Quick registration", "Fast enrollment", "Express registration",
            "Immediate enrollment", "Register me today", "Sign up immediately", "Urgent registration",
            "I'm ready now", "Let's get started", "Begin enrollment", "Start registration",
            "Ready to join", "Count me in", "I want in", "Add me to program",
            "Include me", "I'm interested", "Ready to start", "Let's do this",

            # Variations formelles (20)
            "I would like to enroll", "Could you help me register", "I wish to join",
            "Please register me", "I request enrollment", "Formal application", "Official registration",
            "I hereby apply", "Kindly enroll me", "I seek admission", "Request to join",
            "Application submission", "Enrollment request", "Registration inquiry", "Admission inquiry",
            "Program application", "Course registration", "Training enrollment", "Study registration",
            "Educational enrollment", "Academic registration"
        ],

        "contact": [
            # Informations basiques (20)
            "Contact information", "Phone number", "Email address", "How to contact you",
            "Your address", "Contact details", "How to reach you", "Get in touch",
            "Reach out", "Contact us", "Call you", "Email you", "Find you", "Locate you",
            "Your phone", "Your email", "Office address", "Business address", "Mailing address",
            "Contact form",

            # Support et aide (20)
            "Customer service", "Support contact", "Help desk", "Technical support", "Customer support",
            "Service desk", "Help line", "Support line", "Customer care", "Client services",
            "Assistance contact", "Help contact", "Support team", "Service team", "Care team",
            "Customer relations", "Client relations", "Support services", "Help services", "Contact support",

            # Communication urgente (20)
            "I need help", "Emergency contact", "Urgent assistance", "Immediate help", "Quick support",
            "Fast response", "Rapid assistance", "Instant help", "Emergency support", "Crisis contact",
            "Urgent contact", "Priority support", "Immediate contact", "Express help", "Rush assistance",
            "Critical support", "Emergency line", "Hotline", "Direct line", "Priority line",

            # Moyens spécifiques (20)
            "Phone support", "Email support", "Live chat", "Call center", "Contact center",
            "Service center", "Help center", "Support center", "Office hours", "Business hours",
            "Operating hours", "Availability", "When open", "Schedule", "Appointment",
            "Meeting", "Consultation", "Discussion", "Conference", "Communication"
        ],

        "thanks": [
            # Remerciements basiques (20)
            "Thank you", "Thanks", "Thank you very much", "Thanks a lot", "Much appreciated",
            "I appreciate it", "Grateful", "Many thanks", "Thanks so much", "Thank you so much",
            "Really appreciate", "Deeply grateful", "Most thankful", "Very grateful", "Truly appreciate",
            "Heartfelt thanks", "Sincere thanks", "Warm thanks", "Kind thanks", "Genuine thanks",

            # Remerciements contextuels (20)
            "Thanks for your help", "Thank you for assistance", "Appreciate your support", "Thanks for time",
            "Grateful for help", "Thanks for information", "Appreciate the info", "Thanks for guidance",
            "Grateful for support", "Thanks for service", "Appreciate your service", "Thanks for effort",
            "Grateful for effort", "Thanks for patience", "Appreciate patience", "Thanks for understanding",
            "Grateful for understanding", "Thanks for cooperation", "Appreciate cooperation", "Thanks for kindness",

            # Expressions fortes (20)
            "Amazing help", "Fantastic support", "Excellent service", "Outstanding assistance", "Brilliant help",
            "Superb support", "Wonderful service", "Incredible help", "Phenomenal support", "Exceptional service",
            "Remarkable help", "Extraordinary support", "Magnificent service", "Splendid help", "Marvelous support",
            "Fabulous service", "Terrific help", "Great support", "Awesome service", "Perfect help",

            # Remerciements internationaux (20)
            "Merci beaucoup", "Muchas gracias", "Danke schön", "Grazie mille", "Arigato gozaimasu",
            "Spasibo bolshoe", "Dhanyawad", "Shukran jazeelan", "Toda raba", "Obrigado muito",
            "Tack så mycket", "Kiitos paljon", "Dziekuje bardzo", "Hvala puno", "Dekuji moc",
            "Mulțumesc mult", "Aitäh väga", "Paldies liels", "Dėkoju labai", "Ačiū labai"
        ],

        "certifications": [
            # Questions sur certificats (20)
            "What certificates", "Do you provide certificates", "Certificate information", "Certification details",
            "Available certifications", "Can I get certified", "Certification program", "Certificate program",
            "What certifications available", "Types of certificates", "Certificate options", "Certification options",
            "Professional certificates", "Industry certifications", "Certificate courses", "Certification courses",
            "Certificate training", "Certification training", "Accredited certificates", "Recognized certificates",

            # Processus et exigences (20)
            "How to get certified", "Certification process", "Certificate requirements", "Certification requirements",
            "Certificate exam", "Certification exam", "Certificate test", "Certification test",
            "How to earn certificate",
            "Certificate procedure", "Certification procedure", "Certificate steps", "Certification steps",
            "Certificate path",
            "Certification path", "Certificate journey", "Certification journey", "Certificate track",
            "Certification track",

            # Valeur et reconnaissance (20)
            "Certificate value", "Certification value", "Certificate worth", "Certification worth",
            "Certificate benefits",
            "Certification benefits", "Employer recognition", "Industry recognition", "Certificate validity",
            "Certification validity",
            "Certificate expiration", "Certification expiration", "Certificate renewal", "Certification renewal",
            "Certificate levels",
            "Certification levels", "Certificate grades", "Certification grades", "Certificate standards",
            "Certification standards",

            # Types spécifiques (20)
            "Digital certificates", "Online certificates", "Professional credentials", "Technical certificates",
            "Skill certificates",
            "Competency certificates", "Achievement certificates", "Completion certificates", "Course certificates",
            "Program certificates",
            "Training certificates", "Educational certificates", "Academic certificates", "International certificates",
            "Global certificates",
            "National certificates", "Regional certificates", "Local certificates", "Official certificates",
            "Formal certificates"
        ],

        "course_info": [
            # Informations générales (20)
            "Course information", "What courses", "Available programs", "Training details", "Program information",
            "Course details", "Training information", "Program details", "Course catalog", "Program catalog",
            "Training catalog", "Course list", "Program list", "Training list", "Available courses",
            "Available training", "Course offerings", "Program offerings", "Training offerings", "Educational programs",

            # Contenu et curriculum (20)
            "Course content", "Program content", "Training content", "Course curriculum", "Program curriculum",
            "Training curriculum", "Course syllabus", "Program syllabus", "Training syllabus", "What subjects",
            "Course topics", "Program topics", "Training topics", "Learning materials", "Course materials",
            "Program materials", "Training materials", "Study materials", "Educational materials", "Learning resources",

            # Structure et organisation (20)
            "Course duration", "Program length", "Training duration", "Course schedule", "Program schedule",
            "Training schedule", "Class times", "Course times", "Program times", "When courses start",
            "Course calendar", "Program calendar", "Training calendar", "Course frequency", "Program frequency",
            "Training frequency", "Course structure", "Program structure", "Training structure", "Learning path",

            # Aspects pratiques (20)
            "Course fees", "Program cost", "Training prices", "How much cost", "Course pricing",
            "Program pricing", "Training pricing", "Payment options", "Course requirements", "Program requirements",
            "Training requirements", "Prerequisites", "Entry requirements", "Skill requirements", "Experience needed",
            "Course difficulty", "Program level", "Training level", "Beginner courses", "Advanced programs"
        ]
    }

    # Convertir en listes
    texts = []
    labels = []
    for intent, examples in training_data.items():
        for example in examples:
            texts.append(example.strip())
            labels.append(intent)

    print(f"📊 Dataset ULTRA-ÉQUILIBRÉ créé avec {len(texts)} exemples")
    for intent in set(labels):
        count = labels.count(intent)
        print(f"   {intent}: {count} exemples")

    return texts, labels
    """Dataset MASSIF avec 80+ exemples par classe pour atteindre 89-92% accuracy"""

    training_data = {
        "greeting": [
            # Salutations basiques
            "Hello", "Hi", "Hey", "Good morning", "Good afternoon", "Good evening",
            "Bonjour", "Salut", "Coucou", "Bonsoir", "Hello there", "Hi there",
            "Hey there", "Greetings", "Good day", "Morning", "Evening",
            # Variations avec questions
            "Hello how are you", "Hi how are you doing", "Hey what's up",
            "Good morning everyone", "Hello everyone", "Hi all", "Hey everyone",
            "How are you today?", "Hope you're doing well", "Nice to meet you",
            # Salutations internationales
            "Hola", "Hallo", "Guten tag", "Buenos dias", "Buongiorno",
            "Namaste", "Shalom", "Konnichiwa", "Bonjour tout le monde",
            # Variations familières
            "What's up", "Yo", "Sup", "Howdy", "G'day", "Top of the morning",
            "How's it going", "How are things", "Long time no see",
            # Salutations formelles
            "Good to see you", "Pleased to meet you", "Welcome",
            "I hope this message finds you well", "Greetings and salutations",
            # Variations avec fautes de frappe communes
            "helo", "hii", "heey", "gud morning", "gud evening",
            "bonjur", "salue", "hellow", "gd morning", "Hi everyone!",
            "Hello world!", "Good to see you!", "Hey buddy!", "What's happening?",
            "How's your day?", "Lovely morning!", "Great evening!", "Pleased to meet you!",
            "Welcome aboard!", "Glad to be here!", "Nice day!", "Hope you're well!",
            "Good vibes!", "Sending greetings!", "Happy to connect!"
        ],

        "enroll": [
            # Questions directes d'inscription
            "How can I enroll?", "I want to sign up", "How to register?",
            "Registration process", "Can I enroll?", "Sign me up", "I want to join",
            # Processus détaillés
            "How do I register?", "What are the enrollment steps?",
            "I'd like to register", "Registration information",
            "How to sign up for courses?", "Enrollment procedure",
            "Can I register now?", "I want to enroll in your program",
            # Requirements et applications
            "Registration requirements", "Sign up process",
            "How to become a student?", "I'm interested in enrolling",
            "Application process", "How to apply?", "I want to apply",
            "Can I apply now?", "Application form", "Admission process",
            # Variations avec intention claire
            "How to get admitted?", "I want to start learning",
            "Where do I sign up?", "Registration form", "Enrollment form",
            "I need to register", "Help me register", "Register me please",
            # Questions sur les démarches
            "What documents do I need to enroll?", "Enrollment deadline",
            "When can I register?", "Registration fees", "How much to enroll?",
            "Is registration open?", "Can I still register?", "Late registration",
            # Variations formelles/informelles
            "I would like to enroll", "I wish to register",
            "Could you help me register?", "Please sign me up",
            "I'm ready to join", "Count me in", "I want in",
            # Avec fautes communes
            "how to regiter", "registeration", "enrole", "sing up", "aplly",
            "Ready to start!", "Count me in!", "I'm interested!", "Sign me up now!",
            "Want to join today!", "Ready for enrollment!", "Let's get started!",
            "I need to register ASAP!", "Quick registration!", "Express enrollment!",
            "Register me immediately!", "Fast track registration!", "Priority enrollment!"
        ],

        "contact": [
            # Informations de contact basiques
            "Contact information", "Phone number", "How to contact you?",
            "Your address", "Email address", "Can I call you?",
            "How to reach you?", "Contact details",
            # Support et aide
            "How can I contact support?", "Customer service number",
            "Support email", "Office address", "Location",
            "Where are you located?", "Business hours", "Contact form",
            # Communication
            "Get in touch", "Reach out", "Communication channels",
            "How to get help?", "Support contact", "Help desk",
            "Customer service", "Technical support", "Contact us",
            # Moyens spécifiques
            "Call center", "Email support", "Live chat", "Phone support",
            "What's your phone?", "Your email please", "Office hours",
            "How to speak to someone?", "Customer care",
            # Questions urgentes
            "I need help", "Emergency contact", "Urgent assistance",
            "Who can help me?", "Support team", "Customer relations",
            # Variations polies
            "Could you provide your contact details?",
            "I'd like to get in touch", "How may I reach you?",
            "Contact information please", "Your details please",
            # Avec fautes
            "contect", "contat", "addres", "emial", "phon number",
            "Need to speak with someone!", "Who do I call?", "Direct line please!",
            "Customer support needed!", "Help line?", "Service desk?", "Quick contact!",
            "Immediate assistance!", "Connect me now!", "Support line available?",
            "Emergency contact info!", "24/7 support?", "Live help needed!"
        ],

        "thanks": [
            # Remerciements basiques
            "Thank you", "Thanks", "Thanks a lot", "Thank you very much",
            "I appreciate it", "Merci", "Merci beaucoup", "Much appreciated",
            # Remerciements détaillés
            "Thanks for your help", "Thank you so much", "I'm grateful",
            "Thanks for the information", "Appreciate your help", "Many thanks",
            "Thanks a bunch", "Thank you for everything",
            # Contextuels
            "Thanks for your time", "I really appreciate it", "Thank you kindly",
            "Thanks for the assistance", "Much obliged", "Thank you for your support",
            "Thanks for helping", "Grateful for your help",
            "Thank you for your service",
            # Variations expressives
            "Thanks a million", "Thank you from the bottom of my heart",
            "I can't thank you enough", "Thanks so very much",
            "Thank you ever so much", "Deeply grateful", "Most thankful",
            # Remerciements formels
            "I would like to thank you", "Please accept my gratitude",
            "I extend my thanks", "With sincere appreciation",
            # Informels et modernes
            "Awesome, thanks!", "Great, thank you!", "Perfect, thanks!",
            "You're amazing, thank you!", "Thanks buddy", "Cheers!",
            # Avec fautes communes
            "thx", "ty", "thank u", "thanx", "thnks", "grazie", "danke",
            "You're the best!", "Fantastic help!", "Outstanding service!",
            "Brilliant, thanks!", "Excellent assistance!", "Superb support!",
            "Amazing help!", "Wonderful service!", "Great job!", "Well done!",
            "Kudos!", "Bravo!", "Excellent work!", "Top notch!", "Five stars!"
        ],

        "certifications": [
            # Questions sur certificats
            "What certificates do you offer?", "Do you provide certificates?",
            "Certification info", "Certificate details",
            "What certifications are available?", "Can I get certified?",
            # Types de certifications
            "Certification program", "Professional certificates",
            "Industry certifications", "Certificate requirements",
            "How to get certified?", "Certification cost", "Certificate validity",
            # Reconnaissance et valeur
            "Accredited certificates", "Digital certificates",
            "Certificate of completion", "Professional credentials",
            "Certification exam", "Certificate verification",
            "International certificates", "Online certificates",
            # Bénéfices et standards
            "Certificate benefits", "Certification value",
            "Recognized certificates", "Certificate authority",
            "Certification standards", "Official certificates",
            # Questions spécifiques
            "Are your certificates recognized?", "Do employers accept your certificates?",
            "How long is the certificate valid?", "Certificate expiration",
            "Renewing certificates", "Certificate levels",
            # Processus de certification
            "Certification process", "How to earn a certificate?",
            "Certificate requirements", "Certification test",
            "Pass rate for certification", "Certificate difficulty",
            # Variations avec fautes
            "certs", "certify", "certified", "sertification", "sertifikat",
            "Professional certification?", "Industry standard certificates?",
            "Globally recognized certs?", "Certificate portfolio?", "Credential path?",
            "Qualification levels?", "Certification track?", "Professional credentials?",
            "Certificate series?", "Skill certification?", "Competency certificates?"
        ],

        "course_info": [
            # Informations générales
            "Course information", "What courses do you offer?",
            "Available programs", "Training details", "What can I learn?",
            "Course catalog", "Program details", "Learning paths",
            # Contenu et curriculum
            "Course curriculum", "Training programs", "Educational offerings",
            "Study programs", "Course content", "What subjects?",
            "Training modules", "Learning materials", "Syllabus",
            # Durée et planning
            "Course duration", "Program length", "Training schedule",
            "Class timings", "When do courses start?", "Course calendar",
            "Class frequency", "Study time required",
            # Coûts et frais
            "Course fees", "Program cost", "Tuition fees", "Training prices",
            "How much do courses cost?", "Payment options", "Scholarships",
            # Prérequis et niveaux
            "Prerequisites", "Entry requirements", "Skill level needed",
            "Course difficulty", "Beginner courses", "Advanced training",
            "What do I need to know?", "Experience required",
            # Formats d'apprentissage
            "Online courses", "Classroom training", "Hybrid programs",
            "Distance learning", "Self-paced courses", "Live classes",
            "Video lessons", "Interactive training",
            # Questions spécifiques
            "Course topics", "What will I learn?", "Learning outcomes",
            "Skills gained", "Job prospects after course", "Career paths",
            "Industry relevance", "Practical training", "Hands-on experience",
            # Avec fautes communes
            "courses", "course", "training", "program", "classes", "cours", "formations",
            "Learning opportunities?", "Educational programs?", "Skill development?",
            "Training options?", "Study paths?", "Knowledge areas?", "Subject matters?",
            "Academic programs?", "Professional development?", "Skill building?",
            "Competency training?", "Career preparation?", "Industry training?"
        ]
    }

    # Convertir en listes
    texts = []
    labels = []
    for intent, examples in training_data.items():
        for example in examples:
            texts.append(example.strip())
            labels.append(intent)

    print(f"📊 MEGA Dataset créé avec {len(texts)} exemples")
    for intent in set(labels):
        count = labels.count(intent)
        print(f"   {intent}: {count} exemples")

    return texts, labels


def quick_train_setup(texts, labels):
    """Configuration ultra-rapide et stable"""

    label_counts = Counter(labels)
    print(f"📈 Distribution des classes:")
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count} exemples")

    unique_labels = sorted(list(set(labels)))
    label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
    id2label = {i: lbl for i, lbl in enumerate(unique_labels)}

    # Split avec plus d'exemples pour l'entraînement
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=0.15,  # Seulement 15% pour test = plus d'entraînement
        random_state=seed,
        stratify=labels
    )

    print(f"📊 Split: Train: {len(train_texts)}, Test: {len(test_texts)}")
    return train_texts, test_texts, train_labels, test_labels, label2id, id2label


def train_fast_model(train_texts, train_labels, test_texts, test_labels, label2id, id2label):
    """Entraînement ULTRA-RAPIDE sans sauvegardes intermédiaires"""

    train_label_ids = [label2id[lbl] for lbl in train_labels]
    test_label_ids = [label2id[lbl] for lbl in test_labels]

    # Modèle plus léger et rapide
    model_name = "distilbert-base-multilingual-cased"  # Plus rapide que BERT
    print(f"🤖 Modèle RAPIDE: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=64  # Plus court = plus rapide
        )

    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "label": train_label_ids
    }).map(tokenize, batched=True)

    test_dataset = Dataset.from_dict({
        "text": test_texts,
        "label": test_label_ids
    }).map(tokenize, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(set(train_labels)),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification"
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    # Utiliser un dossier temporaire pour éviter les problèmes d'espace
    temp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(temp_dir, "quick_results")

    print(f"💾 Utilisation du dossier temporaire: {temp_dir}")

    # Configuration OPTIMISÉE: Performance ET vitesse
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # Évaluation par époque pour suivre le progrès
        save_strategy="no",  # PAS de sauvegarde intermédiaire
        learning_rate=5e-5,  # Learning rate optimal pour DistilBERT
        per_device_train_batch_size=32,  # Batch équilibré
        per_device_eval_batch_size=32,
        num_train_epochs=8,  # Plus d'époques pour 89-92% accuracy
        weight_decay=0.01,
        warmup_steps=50,  # Warmup suffisant
        logging_steps=20,
        report_to=None,
        dataloader_drop_last=False,
        remove_unused_columns=True,
        seed=seed,
        data_seed=seed,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        disable_tqdm=False,
        # CRUCIAL: Pas de sauvegarde = pas de corruption
        save_total_limit=0,
        load_best_model_at_end=False,  # Pas de rechargement du meilleur modèle
        # Optimisations supplémentaires
        gradient_accumulation_steps=2,  # Accumulation pour simulation de batch plus grand
        max_grad_norm=1.0,  # Stabilité d'entraînement
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    return trainer, test_label_ids, id2label, temp_dir


def main():
    print("🎯 ENTRAÎNEMENT ÉQUILIBRÉ - OBJECTIF 89-92% ACCURACY")
    print("🚀 Dataset parfaitement équilibré + 8 époques optimisées")
    print("=" * 70)

    # Vérifier l'espace disque
    import shutil
    free_space_gb = shutil.disk_usage('.')[2] / (1024 ** 3)
    print(f"💾 Espace disque libre: {free_space_gb:.1f} GB")

    if free_space_gb < 0.5:  # Même avec peu d'espace, on peut continuer
        print("⚠️  Attention: Peu d'espace libre, mais dossier temporaire utilisé")

    print("\n📊 Étape 1: Création du dataset ULTRA-ÉQUILIBRÉ...")
    texts, labels = create_ultra_balanced_dataset()

    print(f"✅ Dataset prêt: {len(texts)} exemples")

    print("\n🔧 Étape 2: Configuration rapide...")
    train_texts, test_texts, train_labels, test_labels, label2id, id2label = quick_train_setup(texts, labels)

    print("\n⚡ Étape 3: Entraînement OPTIMISÉ (8 époques, évaluation par époque)...")
    trainer, test_label_ids, id2label, temp_dir = train_fast_model(
        train_texts, train_labels, test_texts, test_labels, label2id, id2label
    )

    print("\n🚀 ENTRAÎNEMENT EN COURS...")
    print("⏱️  Temps estimé: 15-20 minutes pour 89-92% accuracy")
    print("🛡️  Sauvegarde finale seulement = sécurisé!")
    print("📈 Suivi du progrès par époque...")

    try:
        # Entraînement sans sauvegarde intermédiaire
        trainer.train()
        print("\n✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")

        print("\n📊 ÉVALUATION FINALE...")
        preds = trainer.predict(trainer.eval_dataset)
        final_accuracy = accuracy_score(test_label_ids, np.argmax(preds.predictions, axis=1))

        print(f"\n🎯 RÉSULTATS FINAUX:")
        if final_accuracy >= 0.92:
            print(f"   🏆 Accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.1f}%) - EXCEPTIONNEL!")
        elif final_accuracy >= 0.89:
            print(f"   🟢 Accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.1f}%) - OBJECTIF ATTEINT! ✅")
        elif final_accuracy >= 0.85:
            print(f"   🟡 Accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.1f}%) - Excellent, très proche!")
        elif final_accuracy >= 0.8:
            print(f"   🟡 Accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.1f}%) - Très bien!")
        else:
            print(f"   🔴 Accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.1f}%) - Besoin d'amélioration")

        pred_labels = np.argmax(preds.predictions, axis=1)
        target_names = [id2label[i] for i in sorted(id2label.keys())]
        print(f"\n📈 RAPPORT DE CLASSIFICATION DÉTAILLÉ:")
        print(classification_report(test_label_ids, pred_labels, target_names=target_names, zero_division=0))

        # Tests de validation améliorés
        print(f"\n🧪 TESTS DE VALIDATION AVANCÉS:")
        test_examples = [
            ("Hello there!", "greeting"),
            ("Hi how are you doing", "greeting"),
            ("I want to enroll in your program", "enroll"),
            ("How can I register for courses", "enroll"),
            ("Thank you so much for your help", "thanks"),
            ("I really appreciate it", "thanks"),
            ("How can I contact customer support", "contact"),
            ("What's your phone number", "contact"),
            ("Do you provide professional certificates", "certifications"),
            ("What certifications are available", "certifications"),
            ("Tell me about your training programs", "course_info"),
            ("What courses do you offer", "course_info")
        ]

        model = trainer.model
        tokenizer = trainer.tokenizer
        correct = 0

        for example, expected in test_examples:
            inputs = tokenizer(example, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_class_id = outputs.logits.argmax().item()
                predicted_label = id2label[predicted_class_id]
                confidence = torch.softmax(outputs.logits, dim=-1).max().item()

                is_correct = predicted_label == expected
                if is_correct:
                    correct += 1

                status_icon = "✅" if is_correct else "❌"
                conf_icon = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"

                print(f"   {status_icon}{conf_icon} '{example[:30]}...' -> {predicted_label} ({confidence:.3f})")

        test_accuracy = correct / len(test_examples)
        print(f"\n🎯 Tests de validation réussis: {correct}/{len(test_examples)} ({100 * test_accuracy:.1f}%)")

        # Sauvegarde conditionnelle
        if final_accuracy >= 0.85:
            final_model_path = "./optimized_intent_model"
            trainer.save_model(final_model_path)
            tokenizer.save_pretrained(final_model_path)
            print(f"\n💾 Modèle excellent sauvegardé dans: {final_model_path}")

            # Créer un fichier de métadonnées
            metadata = {
                "model_name": "distilbert-base-multilingual-cased",
                "accuracy": float(final_accuracy),
                "num_epochs": 8,
                "num_examples": len(texts),
                "classes": list(id2label.values()),
                "training_date": str(pd.Timestamp.now())
            }
            with open(f"{final_model_path}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"📋 Métadonnées sauvegardées")

        return final_accuracy

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        return None

    finally:
        # Nettoyer le dossier temporaire
        try:
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Dossier temporaire nettoyé: {temp_dir}")
        except:
            pass


if __name__ == "__main__":
    print("🔥 SOLUTION OPTIMISÉE - PERFORMANCE ET STABILITÉ:")
    print("   ✅ Dataset parfaitement équilibré (80 exemples/classe)")
    print("   ✅ DistilBERT optimisé")
    print("   ✅ 8 époques pour haute accuracy")
    print("   ✅ Évaluation par époque (suivi progrès)")
    print("   ✅ Aucune sauvegarde intermédiaire")
    print("   ✅ Gradient accumulation")
    print("   ✅ Dossier temporaire sécurisé")
    print("   ✅ Tests de validation étendus")
    print()

    accuracy = main()

