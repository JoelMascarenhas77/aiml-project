from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models
with open('models/knn_model.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

with open('models/log_reg_model.pkl', 'rb') as model_file:
    log_reg_model = pickle.load(model_file)

with open('models/rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('models/svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Define the route for rendering the input form (UI)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

# Define the route for prediction based on form input
@app.route('/predict', methods=['POST'])
def predict():
    value=''
    # Get the form data
    form_data = request.form

    # Extract the features from the form data
    features = [
        int(form_data['Breathing Problem']),
        int(form_data['Fever']),
        int(form_data['Dry Cough']),
        int(form_data['Sore throat']),
        int(form_data['Running Nose']),
        int(form_data['Asthma']),
        int(form_data['Chronic Lung Disease']),
        int(form_data['Headache']),
        int(form_data['Heart Disease']),
        int(form_data['Diabetes']),
        int(form_data['Hyper Tension']),
        int(form_data['Fatigue']),
        int(form_data['Gastrointestinal']),
        int(form_data['Abroad travel']),
        int(form_data['Contact with COVID Patient']),
        int(form_data['Attended Large Gathering']),
        int(form_data['Visited Public Exposed Places']),
        int(form_data['Family working in Public Exposed Places']),
    ]
    features = [features]

    # Make predictions and calculate probabilities for each model
    results = {}

    # Logistic Regression
    log_reg_pred = log_reg_model.predict(features)[0]
    log_reg_proba = log_reg_model.predict_proba(features)[0][1]  # Probability of positive class
    results['Logistic Regression'] = {'prediction': 'positive' if log_reg_pred == 1 else 'negative', 'probability': log_reg_proba}

    # SVM
    svm_pred = svm_model.predict(features)[0]
    svm_proba = svm_model.predict_proba(features)[0][1]
    results['SVM'] = {'prediction': 'positive' if svm_pred == 1 else 'negative', 'probability': svm_proba}

    # Random Forest
    rf_pred = rf_model.predict(features)[0]
    rf_proba = rf_model.predict_proba(features)[0][1]
    results['Random Forest'] = {'prediction': 'positive' if rf_pred == 1 else 'negative', 'probability': rf_proba}

    # K-Nearest Neighbors
    knn_pred = knn_model.predict(features)[0]
    knn_proba = knn_model.predict_proba(features)[0][1]
    results['K-Nearest Neighbors'] = {'prediction': 'positive' if knn_pred == 1 else 'negative', 'probability': knn_proba}

    return render_template('predict.html', results=results)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
