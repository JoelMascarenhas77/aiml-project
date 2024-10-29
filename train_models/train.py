import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Data
data = pd.read_csv("Covid_Dataset.csv")
df = pd.DataFrame(data)

# Preprocessing
le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

# Separate features and target
X = df.drop("COVID-19", axis=1)
y = df["COVID-19"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
y_pred_log_proba = log_reg.predict_proba(X_test)
print("Logistic Regression Accuracy:\n", accuracy_score(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))
print("Prediction Probabilities (Logistic Regression):\n", y_pred_log_proba)

# Support Vector Machine (SVM)
svm = SVC(probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
y_pred_svm_proba = svm.predict_proba(X_test)
print("SVM Accuracy:\n", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Prediction Probabilities (SVM):\n", y_pred_svm_proba)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_rf_proba = rf.predict_proba(X_test)
print("Random Forest Accuracy:\n", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Prediction Probabilities (Random Forest):\n", y_pred_rf_proba)

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_pred_knn_proba = knn.predict_proba(X_test)
print("KNN Accuracy:\n", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("Prediction Probabilities (KNN):\n", y_pred_knn_proba)

# Save SVM model
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

# Save KNN model
with open('models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

# Save Random Forest model
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Save Logistic Regression model
with open('models/log_reg_model.pkl', 'wb') as f:
    pickle.dump(log_reg, f)
