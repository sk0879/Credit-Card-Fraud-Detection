from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, InputLayer

def train_logistic_regression(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model

def train_random_forest(x_train, y_train):
    model = RandomForestClassifier(max_depth=2, n_jobs=-1)
    model.fit(x_train, y_train)
    return model

def train_gradient_boosting(x_train, y_train):
    model = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
    model.fit(x_train, y_train)
    return model

def train_svm(x_train, y_train):
    model = LinearSVC(class_weight='balanced')
    model.fit(x_train, y_train)
    return model

def train_neural_network(x_train, y_train):
    model = Sequential([
        InputLayer((x_train.shape[1],)),
        Dense(2, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_split=0.2)
    return model
