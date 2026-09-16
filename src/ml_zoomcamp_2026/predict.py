import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in) #the model is loaded once, not on every request.

app = Flask('churn')

@app.route('/predict', methods=['POST']) # the route uses POST method: the customer data comes in the body of the request as a JSON, and request.get.json() turns it into a Python dictionary
def predict():
    customer = request.get_json()

    X = dv.transform([customer]) # we apply the model
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result) # we write the response as a JSON


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
