from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import numpy as np
import json
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# -----------------------------
# Initialize Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Configuration
# -----------------------------
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'healthy_vs_rotten.h5'
CLASS_INDICES_PATH = 'class_indices.json'
CLASS_DESCRIPTIONS_PATH = 'class_descriptions.json'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Global Variables
# -----------------------------
model = None
class_labels = {}
class_descriptions = {}

# -----------------------------
# Load Model + Labels
# -----------------------------
def load_resources():
    global model, class_labels, class_descriptions

    # Load trained model
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH, compile=False)
            print("✅ Model loaded successfully")
        except Exception as e:
            print("❌ Model loading error:", e)
    else:
        print("❌ Model file not found")

    # Load class indices
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, 'r') as f:
            indices = json.load(f)
            # reverse mapping index → class name
            class_labels = {v: k for k, v in indices.items()}
        print("✅ Class labels loaded")
    else:
        print("❌ class_indices.json not found")

    # Load optional descriptions
    if os.path.exists(CLASS_DESCRIPTIONS_PATH):
        with open(CLASS_DESCRIPTIONS_PATH, 'r') as f:
            class_descriptions = json.load(f)
        print("✅ Class descriptions loaded")

# Load everything once at startup
load_resources()

# -----------------------------
# Helper Functions
# -----------------------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(img_path):

    if model is None:
        return "Model not loaded", 0.0, ""

    try:
        # Image preprocessing (same as training)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)


        # Prediction
        preds = model.predict(img_array)
        pred_index = np.argmax(preds)
        confidence = float(np.max(preds))

        predicted_label = class_labels.get(pred_index, "Unknown")

        # Split fruit + condition
        if "__" in predicted_label:
            fruit, condition = predicted_label.split("__")
        else:
            fruit, condition = predicted_label, ""

        description = class_descriptions.get(
            predicted_label,
            "Analysis not available."
        )

        final_label = f"{fruit} - {condition}"

        return final_label, confidence, description

    except Exception as e:
        print("Prediction error:", e)
        return "Error", 0.0, ""


# -----------------------------
# Routes
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():

    prediction = None
    confidence = None
    description = None
    uploaded_image_url = None

    if request.method == 'POST':

        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            uploaded_image_url = url_for(
                'static',
                filename=f'uploads/{filename}'
            )

            # Run prediction
            prediction, confidence, description = predict_image(filepath)

            confidence = f"{confidence * 100:.2f}%"

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        description=description,
        uploaded_image_url=uploaded_image_url
    )
@app.route("/blog_single")
def blog_single():
    return render_template("blog_single.html")

@app.route('/blog')
def blog():
    return render_template('blog.html')


@app.route('/portfolio')
def portfolio():
    return render_template('portfolio-details.html')


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)  