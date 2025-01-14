import preprocess
import train
import evaluation

if __name__ == "__main__":
    # Preprocess the data
    data_file = 'creditcard.csv'
    df = preprocess.preprocess_data(data_file)
    train_df, test_df, val_df = preprocess.split_data(df)
    
    x_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    x_val, y_val = val_df.iloc[:, :-1], val_df.iloc[:, -1]
    
    # Train models
    logistic_model = train.train_logistic_regression(x_train, y_train)
    random_forest_model = train.train_random_forest(x_train, y_train)
    nn_model = train.train_neural_network(x_train.to_numpy(), y_train.to_numpy())
    
    # Evaluate models
    evaluation.evaluate_model(logistic_model, x_val, y_val, "Logistic Regression")
    evaluation.evaluate_model(random_forest_model, x_val, y_val, "Random Forest")
