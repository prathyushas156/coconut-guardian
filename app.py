from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "models/coconut_model.tflite"
CONFIDENCE_THRESHOLD = 0.60   # 60%

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------- LOAD TFLITE MODEL -------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------- CLASS LABELS (EDIT IF NEEDED) -----------

CLASS_NAMES = [
    "Bud Rot Dropping",
    "Bud Rot",
    "Gray Leaf Spot",
    "Leaf Rot",
    "Stem Bleeding"

]

# ------------- IMAGE PREPROCESSING -------------
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))   # match model input
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

# ------------- PREDICTION FUNCTION -------------
def predict_image(image_array):
    interpreter.set_tensor(input_details[0]["index"], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("home.html", error="No image uploaded")

        file = request.files["image"]
        if file.filename == "":
            return render_template("home.html", error="No image selected")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        image_array = preprocess_image(file_path)
        predictions = predict_image(image_array)

        confidence = float(np.max(predictions))
        predicted_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_index]

        # -------- LOW CONFIDENCE HANDLING --------
        if confidence < CONFIDENCE_THRESHOLD:
            message = "Please upload a proper image of a coconut tree."
            return render_template(
                "result.html",
                image_path=file_path,
                prediction="Low Confidence",
                confidence=f"{confidence*100:.2f}%",
                message=message
            )

        # -------- NORMAL RESULT --------
        return render_template(
            "result.html",
            image_path=file_path,
            prediction=predicted_class,
            confidence=f"{confidence*100:.2f}%",
            message=None
        )

    return render_template("home.html")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
