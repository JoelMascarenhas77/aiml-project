X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)


with open('..\pickels\log_reg_model.pkl', 'wb') as file:
    pickle.dump(knn_model, file)