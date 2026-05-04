import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================
# LOAD TRAINED CNN MODEL
# ==============================
model = tf.keras.models.load_model("cnn_model.h5")

# ==============================
# DATASET PARAMETERS (UPDATED)
# ==============================
DATASET_PATH = "Coconut Tree Disease Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==============================
# VALIDATION DATA GENERATOR
# ==============================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ==============================
# PREDICTIONS
# ==============================
true_labels = val_data.classes
pred_probs = model.predict(val_data)
pred_labels = np.argmax(pred_probs, axis=1)

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(true_labels, pred_labels)
class_names = list(val_data.class_indices.keys())

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – CNN Model")
plt.tight_layout()
plt.show()

# ==============================
# CLASSIFICATION REPORT
# ==============================
print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels, target_names=class_names))
