import pandas as pd
from sklearn.preprocessing import LabelEncoder

covid_data = pd.read_csv("..\\Covid_Dataset.csv")

e = LabelEncoder()

columns_to_encode = [
    'Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat', 
    'Running Nose', 'Asthma', 'Chronic Lung Disease', 'Headache', 
    'Heart Disease', 'Diabetes', 'Hyper Tension', 'Abroad travel', 
    'Contact with COVID Patient', 'Attended Large Gathering', 
    'Visited Public Exposed Places', 'Family working in Public Exposed Places', 
    'Wearing Masks', 'Sanitization from Market', 'COVID-19', 
    'Gastrointestinal ', 'Fatigue '
]

# Encode each column
for column in columns_to_encode:
    covid_data[column] = e.fit_transform(covid_data[column])

covid_data.to_csv("processed_covid_data.csv", index=False)
