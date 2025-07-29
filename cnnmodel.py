import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras
from glob import glob
import pickle as pkl
folders = glob('data/*')
# Define your CNN architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(folders), activation='softmax')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.CategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Display the model summary
model.summary()

# Load and preprocess the training data
BATCH_SIZE = 32
PIC_SIZE = 224
train_data = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_data.flow_from_directory("data/", target_size=(PIC_SIZE,PIC_SIZE), batch_size=BATCH_SIZE, class_mode="categorical", color_mode='rgb', shuffle=True)


hand_indexs = {v: k for k, v in train_generator.class_indices.items()}
print(hand_indexs)
pkl.dump(hand_indexs, open("hand_labels.pkl", "wb"))
# Train the model
history = model.fit(train_generator, epochs=3)

# Save the trained model
model.save("hand_model_cnn.keras")


print("Model Trained")
