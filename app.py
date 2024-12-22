from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        data = request.get_json()
        
       
        features = data.get("features")
        if not features:
            return jsonify({"error": "Features missing in the request"}), 400
        
       
        features = np.array(features).reshape(1, -1)
        
       
        prediction = model.predict(features)
        result = {"prediction": int(prediction[0])}
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
