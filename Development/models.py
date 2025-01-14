from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC

def train_logistic_regression(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model

def train_random_forest(x_train, y_train, max_depth=2):
    model = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
    model.fit(x_train, y_train)
    return model

def train_gradient_boosting(x_train, y_train, n_estimators=50, learning_rate=1.0):
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(x_train, y_train)
    return model

def train_svm(x_train, y_train):
    model = LinearSVC(class_weight='balanced')
    model.fit(x_train, y_train)
    return model
