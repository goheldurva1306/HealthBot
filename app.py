from flask import Flask, request, jsonify, session, send_from_directory
import re
import random
import os
import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'health_chatbot_secret_key_2024'

# ------------------ Load & Train Model at Startup ------------------
training = pd.read_csv('Data/Training.csv')
testing  = pd.read_csv('Data/Testing.csv')

training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns  = testing.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]
testing  = testing.loc[:, ~testing.columns.duplicated()]

cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

le = preprocessing.LabelEncoder()
y_enc = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_enc, test_size=0.33, random_state=42)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

symptoms_dict = {symptom: idx for idx, symptom in enumerate(x)}

# ------------------ Load Reference Dicts ------------------
severityDictionary   = {}
description_list     = {}
precautionDictionary = {}

def getDescription():
    with open('MasterData/symptom_Description.csv') as csv_file:
        for row in csv.reader(csv_file):
            if len(row) >= 2:
                description_list[row[0]] = row[1]

def getSeverityDict():
    with open('MasterData/symptom_severity.csv') as csv_file:
        for row in csv.reader(csv_file):
            try:
                severityDictionary[row[0]] = int(row[1])
            except:
                pass

def getprecautionDict():
    with open('MasterData/symptom_precaution.csv') as csv_file:
        for row in csv.reader(csv_file):
            if len(row) >= 5:
                precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

getDescription()
getSeverityDict()
getprecautionDict()

# ------------------ Symptom Extractor ------------------
symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain":   "stomach_pain",
    "tummy pain":   "stomach_pain",
    "loose motion": "diarrhea",
    "motions":      "diarrhea",
    "high temperature": "fever",
    "temperature":  "fever",
    "feaver":       "fever",
    "coughing":     "cough",
    "throat pain":  "sore_throat",
    "cold":         "chills",
    "breathing issue":    "breathlessness",
    "shortness of breath":"breathlessness",
    "body ache":    "muscle_pain",
}

def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)
    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.8)
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)
    return list(set(extracted))

def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    pred_proba  = model.predict_proba([input_vector])[0]
    pred_class  = np.argmax(pred_proba)
    disease     = le.inverse_transform([pred_class])[0]
    confidence  = round(pred_proba[pred_class] * 100, 2)
    return disease, confidence

quotes = [
    "Health is wealth — take care of yourself.",
    "A healthy outside starts from the inside.",
    "Every day is a chance to get stronger and healthier.",
    "Take a deep breath, your health matters the most.",
    "Remember, self-care is not selfish.",
]

# ------------------ Conversation State Machine ------------------
STEPS = [
    'name', 'age', 'gender', 'symptoms', 'days',
    'severity', 'preexist', 'lifestyle', 'family',
    'guided', 'result'
]

PROMPTS = {
    'name':     "What is your name?",
    'age':      "Please enter your age.",
    'gender':   "What is your gender? (Male / Female / Other)",
    'symptoms': "Describe your symptoms in a sentence (e.g., 'I have fever and stomach pain').",
    'days':     "For how many days have you had these symptoms?",
    'severity': "On a scale of 1–10, how severe do you feel your condition is?",
    'preexist': "Do you have any pre-existing conditions? (e.g., diabetes, hypertension)",
    'lifestyle':"Do you smoke, drink alcohol, or have irregular sleep?",
    'family':   "Any family history of a similar illness?",
}

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/api/start', methods=['POST'])
def start():
    session.clear()
    session['step']         = 'name'
    session['data']         = {}
    session['symptoms_list']= []
    session['guided_queue'] = []
    session['guided_asked'] = 0
    return jsonify({
        'message': "👋 Hello! I'm your HealthCare Assistant. I'll ask you a few questions to understand your condition better.",
        'next':    PROMPTS['name'],
        'step':    'name'
    })

@app.route('/api/message', methods=['POST'])
def message():
    user_input = request.json.get('message', '').strip()
    if not user_input:
        return jsonify({'error': 'Empty input'}), 400

    step = session.get('step')
    data = session.get('data', {})

    # ---- Collect basic info ----
    if step == 'name':
        data['name'] = user_input
        session['step'] = 'age'
        session['data'] = data
        return jsonify({'next': PROMPTS['age'], 'step': 'age'})

    elif step == 'age':
        data['age'] = user_input
        session['step'] = 'gender'
        session['data'] = data
        return jsonify({'next': PROMPTS['gender'], 'step': 'gender'})

    elif step == 'gender':
        data['gender'] = user_input
        session['step'] = 'symptoms'
        session['data'] = data
        return jsonify({'next': PROMPTS['symptoms'], 'step': 'symptoms'})

    elif step == 'symptoms':
        symptoms_list = extract_symptoms(user_input, cols)
        if not symptoms_list:
            return jsonify({
                'next': "❌ I couldn't detect valid symptoms from that. Please try again with more detail (e.g., 'I have fever and headache').",
                'step': 'symptoms'
            })
        session['symptoms_list'] = symptoms_list
        session['step'] = 'days'
        session['data'] = data
        return jsonify({
            'detected': symptoms_list,
            'next': PROMPTS['days'],
            'step': 'days'
        })

    elif step == 'days':
        data['days'] = user_input
        session['step'] = 'severity'
        session['data'] = data
        return jsonify({'next': PROMPTS['severity'], 'step': 'severity'})

    elif step == 'severity':
        data['severity'] = user_input
        session['step'] = 'preexist'
        session['data'] = data
        return jsonify({'next': PROMPTS['preexist'], 'step': 'preexist'})

    elif step == 'preexist':
        data['preexist'] = user_input
        session['step'] = 'lifestyle'
        session['data'] = data
        return jsonify({'next': PROMPTS['lifestyle'], 'step': 'lifestyle'})

    elif step == 'lifestyle':
        data['lifestyle'] = user_input
        session['step'] = 'family'
        session['data'] = data
        return jsonify({'next': PROMPTS['family'], 'step': 'family'})

    elif step == 'family':
        data['family'] = user_input
        session['data'] = data

        # Initial prediction + build guided queue
        symptoms_list = session.get('symptoms_list', [])
        disease, confidence = predict_disease(symptoms_list)
        session['initial_disease'] = disease

        disease_row = training[training['prognosis'] == disease]
        if not disease_row.empty:
            row = disease_row.iloc[0][:-1]
            guided_queue = [
                sym for sym in row.index[row == 1]
                if sym not in symptoms_list
            ][:8]
        else:
            guided_queue = []

        session['guided_queue'] = list(guided_queue)
        session['guided_asked'] = 0

        if guided_queue:
            session['step'] = 'guided'
            first_q = guided_queue[0].replace('_', ' ')
            return jsonify({
                'message': f"Let me ask a few more questions related to your condition.",
                'next':    f"Do you also have **{first_q}**?",
                'step':    'guided',
                'yes_no':  True
            })
        else:
            return finish_result()

    elif step == 'guided':
        symptoms_list  = session.get('symptoms_list', [])
        guided_queue   = session.get('guided_queue', [])
        guided_asked   = session.get('guided_asked', 0)

        # Record answer for the current guided question
        current_sym = guided_queue[guided_asked]
        if user_input.lower() in ('yes', 'y', '1', 'true'):
            symptoms_list.append(current_sym)
            session['symptoms_list'] = symptoms_list

        guided_asked += 1
        session['guided_asked'] = guided_asked

        # Next guided question or finish
        if guided_asked < len(guided_queue):
            next_sym = guided_queue[guided_asked].replace('_', ' ')
            return jsonify({
                'next':   f"Do you also have **{next_sym}**?",
                'step':   'guided',
                'yes_no': True
            })
        else:
            return finish_result()

    return jsonify({'error': 'Unknown step'}), 400


def finish_result():
    symptoms_list = session.get('symptoms_list', [])
    data          = session.get('data', {})
    disease, confidence = predict_disease(symptoms_list)

    description  = description_list.get(disease, 'No description available.')
    precautions  = precautionDictionary.get(disease, [])
    quote        = random.choice(quotes)

    session['step'] = 'done'
    return jsonify({
        'step':        'result',
        'disease':     disease,
        'confidence':  confidence,
        'description': description,
        'precautions': [p for p in precautions if p],
        'symptoms':    symptoms_list,
        'name':        data.get('name', ''),
        'quote':       quote,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)