from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = joblib.load("model/knn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        gender = int(request.form["gender"])
        age = int(request.form["age"])
        smoking = int(request.form["smoking"])
        yellow_fingers = int(request.form["yellow_fingers"])
        anxiety = int(request.form["anxiety"])
        peer_pressure = int(request.form["peer_pressure"])
        chronic_disease = int(request.form["chronic_disease"])
        fatigue = int(request.form["fatigue"])
        allergy = int(request.form["allergy"])
        wheezing = int(request.form["wheezing"])
        alcohol = int(request.form["alcohol"])
        coughing = int(request.form["coughing"])
        shortness_of_breath = int(request.form["shortness_of_breath"])
        swallowing_difficulty = int(request.form["swallowing_difficulty"])
        chest_pain = int(request.form["chest_pain"])

        input_data = np.array([[gender, age, smoking, yellow_fingers, anxiety,
                                peer_pressure, chronic_disease, fatigue, allergy,
                                wheezing, alcohol, coughing,
                                shortness_of_breath, swallowing_difficulty, chest_pain]])

        input_scaled = scaler.transform(input_data)
        result = model.predict(input_scaled)

        prediction = "LUNG CANCER DETECTED" if result[0] == 1 else "NO LUNG CANCER"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

