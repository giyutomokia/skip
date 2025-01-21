from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_model(df):
    """Train a simple machine learning model."""
    # Extract features and target variable
    X = df[['Temperature', 'Run_Time']]
    y = df['Downtime_Flag']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a decision tree classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return model, accuracy, f1


def predict_downtime(model, temperature, run_time):
    """Make a prediction for machine downtime."""
    prediction = model.predict([[temperature, run_time]])
    confidence = model.predict_proba([[temperature, run_time]])[0][prediction[0]]
    return ('Yes' if prediction[0] == 1 else 'No', round(confidence, 2))
