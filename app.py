import os
from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["HEATMAP_FOLDER"] = HEATMAP_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

# ==============================
# LOAD KERAS MODEL (.h5)
# ==============================
# We use the .h5 model because Grad-CAM needs GradientTape
MODEL_PATH = "final_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["Bud Root Dropping", "Bud Rot", "Gray Leaf Spot", "Leaf Rot", "Stem Bleeding"]

def get_gradcam(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap

@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence, image_path, heatmap_path = None, None, None, None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename != '':
            filename = file.filename
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            
            # Prepare Image
            img = Image.open(save_path).convert("RGB")
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # 1. Prediction
            preds = model.predict(img_array)
            idx = np.argmax(preds)
            prediction = class_names[idx]
            confidence = float(np.max(preds) * 100)

            # 2. Grad-CAM
            try:
                heatmap = get_gradcam(img_array, model)
                
                # Overlay heatmap on original image
                original_img = cv2.imread(save_path)
                original_img = cv2.resize(original_img, (224, 224))
                
                heatmap_resized = cv2.resize(heatmap, (224, 224))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                
                superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
                
                h_filename = "heat_" + filename
                h_path = os.path.join(app.config["HEATMAP_FOLDER"], h_filename)
                cv2.imwrite(h_path, superimposed_img)
                
                image_path = url_for('static', filename=f'uploads/{filename}')
                heatmap_path = url_for('static', filename=f'heatmaps/{h_filename}')
            except Exception as e:
                print(f"Grad-CAM Error: {e}")
                image_path = url_for('static', filename=f'uploads/{filename}')

    return render_template("index.html", prediction=prediction, confidence=confidence, 
                           image_path=image_path, heatmap_path=heatmap_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
