import os
print("Current folder files:", os.listdir())

import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("healthy_vs_rotten.h5")


# -----------------------------
# Load Class Names
# -----------------------------
with open("class_indices.json") as f:
    class_indices = json.load(f)

idx_to_class = {v:k for k,v in class_indices.items()}

# -----------------------------
# Load Image
# -----------------------------
img_path = "test.jpg"   # change image name here

img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# -----------------------------
# Predict
# -----------------------------
pred = model.predict(img_array)

pred_index = np.argmax(pred)
pred_label = idx_to_class[pred_index]
confidence = pred[0][pred_index]

fruit, condition = pred_label.split("__")

print("\nDetected Fruit :", fruit)
print("Condition      :", condition)
print("Confidence     :", f"{confidence*100:.2f}%")

# -----------------------------
# Show Image
# -----------------------------
plt.imshow(img)
plt.title(f"{fruit} - {condition} ({confidence:.2f})")
plt.axis("off")
plt.show()
