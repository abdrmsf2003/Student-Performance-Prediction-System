from flask import Flask, request, render_template_string
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Loading our 59-feature model
model = joblib.load("model.pkl")

# THE EXACT FINAL COLUMNS OUR MODEL EXPECTS:
model_columns = [
'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'failures',
'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 
'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 
'health', 'absences', 'G1', 'G2', 'Medu_5th to 9th grade', 
'Medu_higher education', 'Medu_none', 'Medu_primary (4th grade)', 
'Medu_secondary', 'Fedu_5th to 9th grade', 'Fedu_higher education', 
'Fedu_none', 'Fedu_primary (4th grade)', 'Fedu_secondary', 
'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 
'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 
'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 
'reason_other', 'reason_reputation', 'guardian_father', 
'guardian_mother', 'guardian_other', 'traveltime_15–30 min', 
'traveltime_30 min–1 hour', 'traveltime_<15 min', 'traveltime_>1 hour', 
'studytime_2–5 hours', 'studytime_5–10 hours', 'studytime_<2 hours', 
'studytime_>10 hours'
]

# Basic HTML interface (simple inputs only)
html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Student Performance Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" 
          rel="stylesheet" />

    <style>
        body {
            background: linear-gradient(120deg, #4b79a1, #283e51);
            min-height: 100vh;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-card {
            background: #ffffff;
            border-radius: 15px;
            padding: 35px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }
        .title {
            color: #fff;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.4);
            font-size: 34px;
            margin-bottom: 35px;
        }
        label {
            font-weight: 600;
        }
        .btn-predict {
            background: #27ae60;
            color: #fff;
            font-size: 18px;
            border-radius: 10px;
            padding: 12px 28px;
            transition: 0.3s ease;
        }
        .btn-predict:hover {
            background: #1e8449;
            transform: scale(1.04);
        }
        .result-box {
            background: #d4efdf;
            border-left: 6px solid #27ae60;
            padding: 20px;
            border-radius: 10px;
            margin-top: 25px;
            font-size: 22px;
            font-weight: 600;
            color: #145a32;
            text-align: center;
        }
    </style>
</head>

<body>

<div class="container py-5">
    <h2 class="text-center title">Student Performance Predictor</h2>

    <div class="row justify-content-center">
        <div class="col-lg-7 col-md-10">
            <div class="main-card">

                <form method="POST" class="row g-3">

                    <!-- 15 Clean Important Features -->

                    <div class="col-md-6">
                        <label>G1 (1st Period Grade)</label>
                        <input type="number" name="G1" min="0" max="20" class="form-control" required>
                    </div>

                    <div class="col-md-6">
                        <label>G2 (2nd Period Grade)</label>
                        <input type="number" name="G2" min="0" max="20" class="form-control" required>
                    </div>

                    <div class="col-md-6">
                        <label>Past Failures(n if 1<=n<3, else 4)</label>
                        <input type="number" name="failures" min="0" max="4" class="form-control" required>
                    </div>

                    <div class="col-md-6">
                        <label>Absences</label>
                        <input type="number" name="absences" min="0" max="93" class="form-control" required>
                    </div>

                    <div class="col-md-6">
                        <label>Study Time</label>
                        <select name="studytime" class="form-select" required>
                            <option value="1">Less than 2 hours</option>
                            <option value="2">2 - 5 hours</option>
                            <option value="3">5 - 10 hours</option>
                            <option value="4">More than 10 hours</option>
                        </select>
                    </div>

                    <div class="col-md-6">
                        <label>School Support</label>
                        <select name="schoolsup" class="form-select" required>
                            <option value="1">Yes</option>
                            <option value="0">No</option>
                        </select>
                    </div>

                    <div class="col-md-6">
                        <label>Higher Education Interest</label>
                        <select name="higher" class="form-select" required>
                            <option value="1">Yes</option>
                            <option value="0">No</option>
                        </select>
                    </div>

                    <div class="col-md-6">
                        <label>Internet Access at Home</label>
                        <select name="internet" class="form-select" required>
                            <option value="1">Yes</option>
                            <option value="0">No</option>
                        </select>
                    </div>

                    <div class="col-md-6">
                        <label>Go Out Frequency (1–5)</label>
                        <select name="goout" class="form-select" required>
                            <option value="1">Very Rarely</option>
                            <option value="2">Rarely</option>
                            <option value="3">Sometimes</option>
                            <option value="4">Often</option>
                            <option value="5">Very Often</option>
                        </select>
                    </div>

                    <div class="col-md-6">
                        <label>Free Time Level (1–5)</label>
                        <select name="freetime" class="form-select" required>
                            <option value="1">Very Low</option>
                            <option value="2">Low</option>
                            <option value="3">Moderate</option>
                            <option value="4">High</option>
                            <option value="5">Very High</option>
                        </select>
                    </div>

                    <div class="col-md-6">
                        <label>Health Status (1–5)</label>
                        <select name="health" class="form-select" required>
                            <option value="1">Very Bad</option>
                            <option value="2">Bad</option>
                            <option value="3">Average</option>
                            <option value="4">Good</option>
                            <option value="5">Very Good</option>
                        </select>
                    </div>

                    <div class="col-md-6">
                        <label>Age(15-22)</label>
                        <input type="number" name="age" min="10" max="22" class="form-control" required>
                    </div>

                    <div class="col-md-6">
                        <label>Sex</label>
                        <select name="sex" class="form-select" required>
                            <option value="0">Female</option>
                            <option value="1">Male</option>
                        </select>
                    </div>

                    <div class="col-md-6">
                        <label>Family Relationship Quality (1–5)</label>
                        <select name="famrel" class="form-select" required>
                            <option value="1">Very Bad</option>
                            <option value="2">Bad</option>
                            <option value="3">Average</option>
                            <option value="4">Good</option>
                            <option value="5">Very Good</option>
                        </select>
                    </div>

                    <div class="col-12 text-center mt-4">
                        <button class="btn btn-predict" type="submit">Predict</button>
                    </div>
                </form>

                {% if prediction %}
                <div class="result-box">
                    {{ prediction }}
                </div>
                {% endif %}

            </div>
        </div>
    </div>
</div>

</body>
</html>
"""



@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":

        # Collect the 15 inputs from the form
        user_input = {col: request.form[col] for col in request.form}

        # Convert to DataFrame
        df = pd.DataFrame([user_input])

        # Convert numerics
        numeric_cols = [
            "G1", "G2", "failures", "absences",
            "studytime", "goout", "freetime",
            "health", "age", "famrel"
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

        # Convert yes/no style dropdowns into integers
        binary_cols = ["schoolsup", "higher", "internet", "sex"]
        df[binary_cols] = df[binary_cols].astype(int)

        # Apply same preprocessing as model training
        df = pd.get_dummies(df)

        # Add missing dummy cols
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0

        # Reorder to match model
        df = df[model_columns]

        # Predict
        raw_pred = model.predict(df)[0]

        # ROUND to nearest whole number (best option)
        final_pred = round(raw_pred)

        prediction = f"Predicted Final Grade (G3): {final_pred}"

    return render_template_string(html, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
