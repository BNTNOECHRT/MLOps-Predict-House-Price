from flask import Flask, request, render_template, jsonify

from pandas import DataFrame 
from numpy import array
from xgboost import Booster, DMatrix


import xgboost as xgb


app = Flask(__name__)
model_xgb = Booster()
model_xgb.load_model(fname = "my_model.json")
cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
       'Latitude', 'Longitude']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final = array(int_features)
    data_unseen = DataFrame([final], columns = cols)
    prediction = model_xgb.predict(DMatrix(data_unseen))
    prediction_res = round(prediction[0] * 100000, 2)
    return render_template('home.html',pred='Expected Price will be {:,} $'.format(prediction_res).replace(",", " "))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = DataFrame([data])
    prediction = model_xgb.predict(data=data_unseen)
    output = prediction[0]
    return jsonify(output)


if __name__=="__main__":
    app.run(debug=True)
