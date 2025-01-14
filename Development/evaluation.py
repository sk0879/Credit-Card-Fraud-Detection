from sklearn.metrics import classification_report

def evaluate_model(model, x_test, y_test, target_names=['Not Fraud', 'Fraud']):
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, target_names=target_names)
    return report
