import json
import tensorflow as tf
from tensorflow import keras
from keras import layers
import os

# Ensure the dataset path is correct for your system
# Use forward slashes '/'
DATA_DIR = "naii/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"

# Check if directory exists
if not os.path.exists(DATA_DIR):
    print(f"ERROR: The directory was not found at {DATA_DIR}")
    print("Please make sure you have extracted the dataset and the path is correct.")
else:
    # --- Script Parameters ---
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 1
    FINE_TUNE_EPOCHS = 3

    # --- 1. Load Datasets ---
    print("Loading datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    class_names = train_ds.class_names
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)
    print(f"Found {len(class_names)} classes.")

    # --- 2. Create Data Augmentation Layer ---
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ])

    # --- 3. Apply Augmentation (Memory-Efficient Version) ---
    # We have removed the .cache() method to reduce RAM usage.
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y)).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # --- 4. Build the Model (Corrected Version) ---
print("Building model...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False # Freeze the base model

# Use a standard Rescaling layer for preprocessing
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
# The Rescaling layer below performs the same math as preprocess_input
# It scales pixel values from [0, 255] to [-1, 1]
x = layers.Rescaling(1./127.5, offset=-1)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# --- 5. Initial Compilation and Training ---
print("Starting initial training...")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

                                    

# --- 7. Save the correctly trained model ---
print("Saving model...")
model.save("disease_model.keras")
print("Model saved as disease_model.h5. Training complete.")