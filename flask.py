from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler

sclr = StandardScaler()
model = pickle.load(open("model_rf.pkl", 'rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        Open = float(request.form['Open'])
        High = float(request.form['High'])
        Low = float(request.form['Low'])
        Volume = int(request.form['Volume'])
        Year = int(request.form['Year'])
        Month = int(request.form['Month'])
        Day = int(request.form['Day'])

        features = np.array([[Open, High, Low, Volume, Year, Month, Day]])
        # features = sclr.fit_transform(features)  # Use transform instead of fit_transform
        my_prediction = model.predict(features)

        return render_template('index.html', prediction=my_prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
