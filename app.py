from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Update your model path and load the logistic regression model
model_path = 'model05.pkl'  # Replace with your model path
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Define the route for predicting 'Clicked on Ad'
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    daily_time_spent = request.form.get('Daily Time Spent', type=float)
    area_income = request.form.get('Area Income', type=float)
    daily_internet_usage = request.form.get('Daily Internet Usage', type=float)

    # Validate inputs
    if daily_time_spent is None or area_income is None or daily_internet_usage is None:
        return render_template('advertising.html', prediction_text='Invalid input. Please provide all fields.')

    # Create a pandas DataFrame with the same feature names used during training
    features = pd.DataFrame({
        'Daily Time Spent on Site': [daily_time_spent],
        'Area Income': [area_income],
        'Daily Internet Usage': [daily_internet_usage]
    })

    # Make prediction
    prediction = model.predict(features)[0]

    # Prepare result message
    if prediction == 1:
        result_text = 'The user is predicted to click on the ad.'
    else:
        result_text = 'The user is predicted not to click on the ad.'

    return render_template('advertising.html', prediction_text=result_text)


# Define route for main page
@app.route('/')
def main_page():
    return render_template('advertising.html')

if __name__ == "__main__":
    app.run(debug=True)
