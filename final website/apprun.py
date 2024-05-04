from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import spacy
from fuzzywuzzy import fuzz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import load
import assemblyai as aai

app = Flask(__name__, static_url_path='/Static', static_folder='Static')


@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/process_symptoms', methods=['POST'])
def process_symptoms():
    answer = request.form['answer']
    if answer == 'yes':
        return render_template('index.html')
    else:
        message = "You may be healthy, but keep monitoring your health."
        return render_template('index2.html', message=message)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data['symptoms']
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(symptoms)
    user_data = np.zeros(132)
    accurcy = []
    symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
    diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positiw onal Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
    disease = list(diseases_list.items())
    df = pd.read_csv("E:\Dataset\medicine\Training.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    rf = load('rf_model.joblib')
    # rf = RandomForestClassifier(n_estimators=25, criterion="gini", max_features="sqrt")
    # rf.fit(X_train, y_train)
    list1 = []
    for i in doc:
        list1.append(i.text)
        list2 = list1
        list3 = []
    for i in list1:
        for j in list2:
            list3.append(f"{i} {j}")
    for x in doc:
        for i, y in symptoms_dict.items():
            if fuzz.ratio(x.text, i) > 80:
                user_data[y] = 1
                accurcy.append(f"Confidence score for {i} is {fuzz.ratio(x.text, i)}")
    for i in list3:
        for x, y in symptoms_dict.items():
            if fuzz.ratio(x, i) > 80 and user_data[y] == 0:
                user_data[y] = 1
                accurcy.append(f"Confidence score for {i} is {fuzz.ratio(x, i)}")
    y_val = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_val) * 100
    y_pred = rf.predict([user_data])
    real = []
    for x, y in disease:
        for i in y_pred:
            if x == i:
                real.append(y)
    description = pd.read_csv("E:\Dataset\medicine\description.csv")
    for x, y in description.iterrows():
        for i in real:
            if y["Disease"] == i:
                desc = y["Description"]
    diet = pd.read_csv("E:\Dataset\medicine\diets.csv")
    for x, y in diet.iterrows():
        for i in real:
            if y["Disease"] == i:
                dit = y["Diet"]
    precautions = pd.read_csv("E:\Dataset\medicine\precautions_df.csv")
    pre_li = []
    med = pd.read_csv("E:\Dataset\medicine\medications.csv")
    medicine_inf = []
    for i in med["Disease"]:
        for j in real:
            for b in ["Ayurveda Medication"]:
                if i == j:
                    medicine_inf.append(med["Ayurveda Medication"][0])
                    break
    medicine_info = medicine_inf[0]
    for x, y in precautions.iterrows():
        for i in real:
            if y["Disease"] == i:
                a = y["Precaution_1"]
                b = y["Precaution_2"]
                # c = y["Precaution_2"]
                d = y["Precaution_4"]
                pre_li.append(f"1. {a}")
                pre_li.append(f"2. {b}")
                # pre_li.append(f"3. {c}")
                pre_li.append(f"3. {d}")
    return jsonify({"disease": real, "accuracy": accurcy, "discription": desc, "diet": dit, "precaution": pre_li, "medicine": medicine_info})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_file = request.files['audio_file']
    audio_file_path = 'audio.wav'
    audio_file.save(audio_file_path)

    aai.settings.api_key = "5fad81c507f1427f94484700ca13dc4c"
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file_path)
    transcription_text = str(transcript.text)


    nlp = spacy.load("en_core_web_sm")
    doc = nlp(transcription_text)
    user_data = np.zeros(132)
    accurcy = []
    symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
                     'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
                     'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13,
                     'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
                     'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
                     'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26,
                     'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31,
                     'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35,
                     'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39,
                     'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43,
                     'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
                     'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
                     'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
                     'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
                     'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
                     'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
                     'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
                     'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
                     'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77,
                     'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81,
                     'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
                     'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
                     'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
                     'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
                     'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
                     'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
                     'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
                     'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
                     'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
                     'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
                     'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
                     'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123,
                     'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
                     'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
    diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
                     33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis',
                     6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis',
                     32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue',
                     37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D',
                     22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold',
                     34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins',
                     26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis',
                     5: 'Arthritis', 0: '(vertigo) Paroymsal  Positiw onal Vertigo', 2: 'Acne',
                     38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
    disease = list(diseases_list.items())
    df = pd.read_csv("E:\Dataset\medicine\Training.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    rf = load('rf_model.joblib')
    # rf = RandomForestClassifier(n_estimators=25, criterion="gini", max_features="sqrt")
    # rf.fit(X_train, y_train)
    list1 = []
    for i in doc:
        list1.append(i.text)
        list2 = list1
        list3 = []
    for i in list1:
        for j in list2:
            list3.append(f"{i} {j}")
    for x in doc:
        for i, y in symptoms_dict.items():
            if fuzz.ratio(x.text, i) > 80:
                user_data[y] = 1
                accurcy.append(f"Confidence score for {i} is {fuzz.ratio(x.text, i)}")
    for i in list3:
        for x, y in symptoms_dict.items():
            if fuzz.ratio(x, i) > 80 and user_data[y] == 0:
                user_data[y] = 1
                accurcy.append(f"Confidence score for {i} is {fuzz.ratio(x, i)}")
    y_val = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_val) * 100
    y_pred = rf.predict([user_data])
    real = []
    for x, y in disease:
        for i in y_pred:
            if x == i:
                real.append(y)
    description = pd.read_csv("E:\Dataset\medicine\description.csv")
    for x, y in description.iterrows():
        for i in real:
            if y["Disease"] == i:
                desc = y["Description"]
    diet = pd.read_csv("E:\Dataset\medicine\diets.csv")
    for x, y in diet.iterrows():
        for i in real:
            if y["Disease"] == i:
                dit = y["Diet"]
    precautions = pd.read_csv("E:\Dataset\medicine\precautions_df.csv")
    pre_li = []
    reason = []
    description = pd.read_csv("E:\Dataset\medicine\description.csv")
    for x, y in description.iterrows():
        for i in real:
            if y["Disease"] == i:
                reason.append(y["Description"])
    med = pd.read_csv("E:\Dataset\medicine\medications.csv")
    rea = "".join(reason)
    medicine_inf = []
    for i in med["Disease"]:
        for j in real:
            for b in ["Ayurveda Medication"]:
                if i == j:
                    medicine_inf.append(med["Ayurveda Medication"][0])
                    break
    medicine_info = medicine_inf[0]
    medi = "".join(medicine_info)
    for x, y in precautions.iterrows():
        for i in real:
            if y["Disease"] == i:
                a = y["Precaution_1"]
                b = y["Precaution_2"]
                # c = y["Precaution_2"]
                d = y["Precaution_4"]
                pre_li.append(f"1. {a}")
                pre_li.append(f"2. {b}")
                # pre_li.append(f"3. {c}")
                pre_li.append(f"3. {d}")
    return render_template('new.html', transcription=transcription_text, result=real, accurcy=accurcy,  diet=dit, workout=pre_li, medicine=medi, reason=rea)


if __name__ == '__main__':
    app.run(debug=True)
