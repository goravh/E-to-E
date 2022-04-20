import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def results():
    Gender = float(request.form['Gender'])
    Age = float(request.form['Age'])
    Annual_Income = float(request.form['Annual_Income'])


    x = np.array([[Gender, Age, Annual_Income]])

    model = pickle.load(open('knn_model.pkl', 'rb'))
    y_predict = model.predict(x)
    return  jsonify({'Predection' : float(y_predict)})
if __name__ == '__main__':
    app.run(debug= True, port = 1010)

