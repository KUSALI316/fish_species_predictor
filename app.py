import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = "final_fish_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_names = ["Balaya", "Issa", "Jilawa", "Lena_paraw", "Samonbollo"]

# Function to check file extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict fish species
def predict_fish_species(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize to match model input
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)  # Get class with highest confidence
    class_label = class_names[class_index]
    confidence = round(np.max(prediction) * 100, 2)  # Confidence percentage

    return class_label, confidence

# Route for homepage
@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file part")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", message="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Predict fish species
            predicted_class, confidence = predict_fish_species(file_path)

            return render_template(
                "index.html",
                uploaded_image=file_path,
                predicted_class=predicted_class,
                confidence=confidence,
            )

    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists
    app.run(debug=True)
