import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

# hide warnings
import warnings
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from clean_input import create_features, prediction_mapper

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    sexList = ['Male', 'Female', 'Infant']
    return render_template('index.html', sexList = sexList)


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sexList = ['Male', 'Female', 'Infant']
    
    features = [x for x in request.form.values()]
    
    height = features[0]
    shucked_weight = features[1]
    shell_weight = features[2]
    sex = features[3]

    features = create_features(height, shucked_weight, shell_weight, sex, model)
    
    prediction = model.predict(features)

    # output = round(prediction[0], 2)

    # return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
    
    output = '''
    Details Provided - <br/>
    Height = {}<br/>
    Shucked Weight = {}<br/>
    Shell Weight = {}<br/>
    Sex = {}<br/>
    <br/>
    Prediction - <br/>
    {}
    '''.format(height, shucked_weight, shell_weight, sex, prediction_mapper(prediction))

    return render_template('index.html', sexList = sexList, prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)