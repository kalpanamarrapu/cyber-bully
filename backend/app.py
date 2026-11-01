
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

import joblib
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "classifier.joblib"

MULTILABEL_MODEL_PATH = (
    BASE_DIR / "models" / "multilabel_classifier.joblib"
)

DB_PATH = BASE_DIR / "history.db"

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


# ==========================================
# 2. FLASK CONFIGURATION
# ==========================================

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ]
        }
    },
)

# Load binary model

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Binary model not found. Run python train.py first."
    )

model = joblib.load(MODEL_PATH)

print("Binary model loaded successfully!")


# Load optional multilabel model

multilabel_model = None

if MULTILABEL_MODEL_PATH.exists():
    multilabel_model = joblib.load(MULTILABEL_MODEL_PATH)
    print("Multilabel model loaded successfully!")
else:
    print("Multilabel model not found. Binary detection only.")


# ==========================================
# 3. DATABASE
# ==========================================

def init_db():

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                text TEXT NOT NULL,

                label INTEGER NOT NULL,

                score REAL NOT NULL,

                created_at TEXT NOT NULL

            )
        """)


init_db()


# ==========================================
# 4. MODEL SCORE CALCULATION
# ==========================================

def get_prediction_score(loaded_model, text, prediction):

    """
    Supports:
    - Logistic Regression
    - Linear SVM
    - Multinomial Naive Bayes

    Returns a confidence-like score for class 1.
    """

    if hasattr(loaded_model, "predict_proba"):

        probabilities = loaded_model.predict_proba([text])[0]

        classes = list(
            loaded_model.named_steps["classifier"].classes_
        )

        if 1 in classes:
            return float(probabilities[classes.index(1)])

        return 0.0

    elif hasattr(loaded_model, "decision_function"):

        scores = np.asarray(
            loaded_model.decision_function([text])
        ).ravel()

        classes = list(
            loaded_model.named_steps["classifier"].classes_
        )

        if len(classes) == 2:

            score = float(scores[0])

            # Binary decision scores point towards classes_[1].
            if classes[1] != 1:
                score = -score

            # Convert decision score to a confidence-like value.
            score = np.clip(score, -50, 50)

            return float(1 / (1 + np.exp(-score)))

    # Fallback if the model does not support scores.
    return float(prediction == 1)


# ==========================================
# 5. HEALTH CHECK
# ==========================================

@app.get("/")
def home():

    return jsonify({
        "message": "Cyberbullying Detection API",
        "status": "running"
    })


@app.get("/api/health")
def health():

    return jsonify({
        "status": "ok",
        "message": "API is running",
        "binary_model": True,
        "multilabel_model": multilabel_model is not None
    })


# ==========================================
# 6. BINARY CYBERBULLYING PREDICTION
# ==========================================

@app.post("/api/predict")
def predict():

    data = request.get_json(silent=True) or {}

    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():

        return jsonify({
            "error": "Please enter a comment."
        }), 400

    if len(text) > 2000:

        return jsonify({
            "error": "Maximum comment length is 2000 characters."
        }), 400

    text = text.strip()

    try:

        # Binary prediction

        raw_prediction = model.predict([text])[0]

        label = int(raw_prediction)

        score = get_prediction_score(
            model,
            text,
            label
        )

        # ==================================
        # MULTILABEL PREDICTION
        # ==================================

        categories = {}

        if multilabel_model is not None:

            multi_prediction = multilabel_model.predict([text])[0]

            for name, value in zip(
                LABEL_COLUMNS,
                multi_prediction
            ):

                categories[name] = bool(value)

        # ==================================
        # DATABASE STORAGE
        # ==================================

        created_at = datetime.now(
            timezone.utc
        ).isoformat()

        with sqlite3.connect(DB_PATH) as conn:

            cursor = conn.execute(
                """
                INSERT INTO predictions
                (text, label, score, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    text,
                    label,
                    score,
                    created_at
                )
            )

            prediction_id = cursor.lastrowid

        # ==================================
        # API RESPONSE
        # ==================================

        return jsonify({

            "id": prediction_id,

            "label": label,

            "result": (
                "Potentially bullying"
                if label == 1
                else "Likely non-bullying"
            ),

            "score": round(score, 4),

            "categories": categories,

            "warning": (
                "Experimental detection system. "
                "Predictions should be reviewed by humans."
            ),

            "created_at": created_at

        })

    except Exception as error:

        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed. Check the backend logs.",
        }), 500


# ==========================================
# 7. PREDICTION HISTORY
# ==========================================

@app.get("/api/history")
def history():

    with sqlite3.connect(DB_PATH) as conn:

        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT id, text, label, score, created_at

            FROM predictions

            ORDER BY id DESC

            LIMIT 50
            """
        ).fetchall()

    return jsonify([
        dict(row) for row in rows
    ])


# ==========================================
# 8. START SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
