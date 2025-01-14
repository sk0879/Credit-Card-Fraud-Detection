from sklearn.metrics import classification_report

def evaluate_model(model, x_val, y_val, model_name):
    predictions = model.predict(x_val)
    report = classification_report(y_val, predictions, target_names=['Not Fraud', 'Fraud'])
    print(f"Classification Report for {model_name}:\n{report}")
