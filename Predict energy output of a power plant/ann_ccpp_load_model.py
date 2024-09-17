import tensorflow as tf
import numpy as np

model=tf.keras.models.load_model("ccpp.keras")

model.summary()

input_data = np.array([[8, 42, 1010, 83]])

# Make a prediction
prediction = model.predict(input_data)

print(prediction)

