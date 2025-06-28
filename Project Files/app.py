from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model (make sure you have 'model.pkl' in the same folder)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        holiday = request.form['holiday']
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = request.form['weather']
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hour = int(request.form['hours'])
        minute = int(request.form['minutes'])
        second = int(request.form['seconds'])

        # Map holiday and weather to numerical if needed
        holiday_mapping = {'None': 0, 'Holiday': 1}
        weather_mapping = {'Clear': 1, 'Mist': 2, 'Clouds': 3, 'Rain': 4, 'Snow': 5, 'Drizzle': 6, 'Thunderstorm': 7}

        holiday_val = holiday_mapping.get(holiday, 0)
        weather_val = weather_mapping.get(weather, 1)

        # Example input format - adjust based on your model
        final_features = np.array([[holiday_val, temp, rain, snow, weather_val, year, month, day, hour, minute, second]])
        prediction = model.predict(final_features)

        return render_template('result.html', prediction_text=f'Estimated Traffic Volume is : {prediction[0]:.2f}')
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)

