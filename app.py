from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os

app = Flask(__name__)

# Configuration for file upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Dummy model for prediction (Replace with actual model)
model = LogisticRegression()
data = None  # Holds uploaded data for training and prediction

@app.route('/')
def home():
    return render_template('index.html', file_uploaded=False)

@app.route('/upload', methods=['POST'])
def upload_file():
    global data
    if 'file' not in request.files:
        return render_template('index.html', error="No file part", file_uploaded=False)
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file", file_uploaded=False)
    
    try:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Load the data into pandas DataFrame
        data = pd.read_csv(file_path)
        
        return render_template('index.html', file_uploaded=True, prediction=None, train_message=None)
    except Exception as e:
        return render_template('index.html', error=f"Error while processing the file: {str(e)}", file_uploaded=False)

@app.route('/train', methods=['POST'])
def train_model():
    global data, model
    if data is None:
        return render_template('index.html', error="No data uploaded", file_uploaded=False)
    
    try:
        # Train the model on the uploaded data
        X = data[['Temperature', 'Run_Time']]  # Example features
        y = data['Downtime_Flag']  # Target column (Downtime prediction)
        model.fit(X, y)
        
        # After training, show the success message and metrics
        y_pred = model.predict(X)
        accuracy = model.score(X, y)
        precision = accuracy  # Placeholder for demonstration
        recall = accuracy     # Placeholder for demonstration
        f1 = accuracy         # Placeholder for demonstration

        train_message = f"Model trained successfully!<br>Accuracy: {accuracy:.2f}<br>Precision: {precision:.2f}<br>Recall: {recall:.2f}<br>F1-Score: {f1:.2f}"
        
        return render_template('index.html', file_uploaded=True, train_message=train_message, prediction=None)
    except Exception as e:
        return render_template('index.html', error=f"Error while training the model: {str(e)}", file_uploaded=True)

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return render_template('index.html', error="Model not trained yet", file_uploaded=True)
    
    try:
        # Example prediction (Replace with actual logic to handle dynamic inputs)
        sample_input = [[80, 120]]  # Example input for prediction (Temperature, Run_Time)
        prediction = model.predict(sample_input)
        confidence = model.predict_proba(sample_input)[0][prediction[0]]
        
        result = {"Downtime": "Yes" if prediction[0] == 1 else "No", "Confidence": confidence}
        
        return render_template('index.html', file_uploaded=True, prediction=result, train_message=None)
    except Exception as e:
        return render_template('index.html', error=f"Error while making prediction: {str(e)}", file_uploaded=True)

if __name__ == '__main__':
    app.run(debug=True)
