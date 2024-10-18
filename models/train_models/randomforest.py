X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

with open('..\pickels\rf_model.pkl', 'wb') as file:
    pickle.dump(knn_model, file)