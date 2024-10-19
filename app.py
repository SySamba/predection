import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# Initialiser l'application Flask
app = Flask(__name__)

# Charger les modèles pré-entraînés
model_cancer = load_model('model_breast_cancer.h5')
model_diabetes = load_model('mon_modele_diabete.h5')

# Routes de l'application
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    try:
        # Récupérer les données du formulaire
        features = [
            float(request.form['mean_radius']),
            float(request.form['mean_texture']),
            float(request.form['mean_perimeter']),
            float(request.form['mean_area']),
            float(request.form['mean_smoothness']),
            float(request.form['mean_compactness']),
            float(request.form['mean_concavity']),
            float(request.form['mean_concave_points']),
            float(request.form['mean_symmetry']),
            float(request.form['mean_fractal_dimension']),
            float(request.form['radius_se']),
            float(request.form['texture_se']),
            float(request.form['perimeter_se']),
            float(request.form['area_se']),
            float(request.form['smoothness_se']),
            float(request.form['compactness_se']),
            float(request.form['concavity_se']),
            float(request.form['concave_points_se']),
            float(request.form['symmetry_se']),
            float(request.form['fractal_dimension_se']),
            float(request.form['radius_worst']),
            float(request.form['texture_worst']),
            float(request.form['perimeter_worst']),
            float(request.form['area_worst']),
            float(request.form['smoothness_worst']),
            float(request.form['compactness_worst']),
            float(request.form['concavity_worst']),
            float(request.form['concave_points_worst']),
            float(request.form['symmetry_worst']),
            float(request.form['fractal_dimension_worst']),
        ]

        # Normalisation des caractéristiques (si nécessaire)
        # Si votre modèle nécessite une normalisation, vous devrez le faire ici
        # Si votre modèle est déjà entraîné avec des données non normalisées, cette étape peut être supprimée.

        # Prédire avec le modèle
        prediction = model_cancer.predict(np.array(features).reshape(1, -1))
        result = "Malin" if prediction[0][0] > 0.5 else "Bénin"

        return render_template('result.html', result=result, model="Cancer du Sein")
    
    except Exception as e:
        return f"Une erreur s'est produite: {str(e)}"

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        # Récupérer les données du formulaire
        features = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['diabetes_pedigree_function']),
            float(request.form['age']),
        ]

        # Normalisation des caractéristiques (si nécessaire)
        # Si votre modèle nécessite une normalisation, vous devrez le faire ici
        # Si votre modèle est déjà entraîné avec des données non normalisées, cette étape peut être supprimée.

        # Prédire avec le modèle
        prediction = model_diabetes.predict(np.array(features).reshape(1, -1))
        result = "Diabétique" if prediction[0][0] > 0.5 else "Non Diabétique"

        return render_template('result.html', result=result, model="Diabète")
    
    except Exception as e:
        return f"Une erreur s'est produite: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
