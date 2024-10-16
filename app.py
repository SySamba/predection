import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Initialiser l'application Flask
app = Flask(__name__)

# Charger les modèles pré-entraînés
model_cancer = load_model(os.path.join('flask', 'model_breast_cancer.h5'))
model_diabetes = load_model(os.path.join('flask', 'mon_modele_diabete.h5'))

# Scalers pour normaliser les données (ceux utilisés lors de l'entraînement)
# Assure-toi que ces scalers ont été ajustés avec les mêmes données que celles utilisées lors de l'entraînement
scaler_cancer = StandardScaler()
scaler_diabetes = StandardScaler()

# Routes de l'application
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    # Récupérer les données du formulaire pour le cancer
    input_data = [float(x) for x in request.form.values()]
    
    # Convertir les données en numpy array et normaliser
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler_cancer.transform(input_data)  # Appliquer uniquement transform
    
    # Faire la prédiction
    prediction = model_cancer.predict(input_data_scaled)[0][0]
    result = 'Malignant' if prediction > 0.5 else 'Benign'
    
    return render_template('result.html', result=result, model="Breast Cancer")

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    # Récupérer les données du formulaire pour le diabète
    input_data = [float(x) for x in request.form.values()]
    
    # Convertir les données en numpy array et normaliser
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler_diabetes.transform(input_data)  # Appliquer uniquement transform
    
    # Faire la prédiction
    prediction = model_diabetes.predict(input_data_scaled)[0][0]
    result = 'Diabetic' if prediction > 0.5 else 'Non-Diabetic'
    
    return render_template('result.html', result=result, model="Diabetes")

if __name__ == '__main__':
    app.run(debug=True)
