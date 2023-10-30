#with flask
from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
sclr = StandardScaler()

model=pickle.load(open("model_rf.pkl",'rb'))

app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    Open=request.form['Open']
    High=request.form['High']
    Low=request.form['Low']
    Adj_Close=request.form['Adj Close']
    Volume=request.form['Volume']
    Year=request.form['Year']
    Month=request.form['Month']
    Day=request.form['Day']

    features=np.array([[Open,High,Low,Adj_Close,Volume,Year,Month,Day]])
    features=sclr.fit_transform(features)
    prediction=model.predict(features).reshape(1,-1)
    return render_template('index.html',output=prediction[0][0])
if __name__=="__main__":
    app.run(debug=True)


from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
sclr = StandardScaler()

model = pickle.load(open("model_rf.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    Open = float(request.form['Open'])
    High = float(request.form['High'])
    Low = float(request.form['Low'])
    Adj_Close = float(request.form['Adj Close'])
    Volume = int(request.form['Volume'])
    Year = int(request.form['Year'])
    Month = int(request.form['Month'])
    Day = int(request.form['Day'])

    features = np.array([[Open, High, Low, Adj_Close, Volume, Year, Month, Day]])
    features = sclr.fit_transform(features)
    prediction = model.predict(features).reshape(1, -1)

    return render_template('index.html', output=prediction[0][0])

if __name__ == "__main__":
    app.run(debug=True)

