# Libraries
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import os

# app name
app = Flask(__name__)

# load saved artifacts
def load_model():
    return pickle.load(open('artifacts/loan_approve.pkl', 'rb'))

# home page
@app.route('/')
def home():
    return render_template('index.html')


# predict the result and return it
@app.route('/predict', methods=['POST'])
def predict():
    features = [str(x).lower() for x in request.form.values()]
    x = np.zeros(20)

    x[0] = int(features[7])
    x[1] = int(features[8])
    x[2] = int(features[9])
    x[3] = int(features[10])
    
    credithistory = features[5]
    if credithistory == 'yes':
        x[4] = 1
    elif credithistory == 'no':
        x[4] = 0

    gender = features[0]
    if gender == 'male':
        x[5],x[6] = 0, 1
    elif gender == 'female':
        x[5],x[6] = 1, 0

    married = features[1]
    if married == 'no':
        x[7],x[8] = 1, 0
    elif married == 'yes':
        x[7],x[8] = 0, 1

    dependents = features[2]
    if dependents == '0':
        x[9],x[10],x[11],x[12] = 1, 0, 0, 0
    elif dependents == '1':
        x[9],x[10],x[11],x[12] = 0, 1, 0, 0
    elif dependents == '2':
        x[9],x[10],x[11],x[12] = 0, 0, 1, 0
    elif dependents == '3+':
        x[9],x[10],x[11],x[12] = 0, 0, 0, 1

    education = features[3]
    if education == 'graduate':
        x[13],x[14] = 1, 0
    elif education == 'not graduate':
        x[13],x[14] = 0, 1

    selfemployed = features[4]
    if selfemployed == 'no':
        x[15],x[16] = 0, 1
    elif selfemployed == 'yes':
        x[15],x[16] = 1, 0

    
    propertyarea = features[6]
    if propertyarea == 'rural':
        x[17],x[18],x[19] = 1, 0, 0
    elif propertyarea == 'semiurban':
        x[17],x[18],x[19] = 0, 1, 0
    elif propertyarea == 'urban':
        x[17],x[18],x[19] = 0, 0, 1

    x = x.reshape(1,20)

    model = load_model()
    prediction = model.predict(x)

    labels = ['Loan Rejected', 'Loan accepted']
    result = labels[prediction[0]]

    return render_template('index.html', output='Status: {}'.format(result))


if __name__ == '__main__':
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
