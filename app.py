from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained SVM model
with open('models\svm_model.pkl', 'rb') as model_file:
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
        int(form_data['Wearing Masks']),
        int(form_data['Sanitization from Market'])
    ]

    # Convert features to a numpy array
    features_array = np.array([features])

    # Make a prediction using the SVM model
    features = [features]

    # Make the prediction
    prediction = svm_model.predict(features)
    probability = svm_model.predict_proba(features)

    # Get the probability of the positive class (COVID-19 positive)

    return render_template('result.html', prediction=prediction, probability=probability)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
