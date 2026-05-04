import tensorflow as tf
import os

# Load your existing Keras model
keras_model = tf.keras.models.load_model("final_model.h5")

# Initialize the TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

# Enable dynamic range quantization (the easiest and most effective optimization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_quant_model = converter.convert()

# Save the quantized model to a file
with open("quant_model.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("quant_model.tflite has been successfully generated and quantized!")
