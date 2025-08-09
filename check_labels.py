import tensorflow as tf

# --- IMPORTANT ---
# Update this path to the location of your 'train' folder
# from the dataset you downloaded or created.
# Use forward slashes '/' in the path.
train_dir = 'naii/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
# ---------------

try:
    print(f"Loading dataset from: {train_dir}")
    # We only load the dataset to inspect its properties
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=32,
        shuffle=False # No need to shuffle for this check
    )

    print("\n--- Model's True Class Order ---")
    # This prints the exact order the model learned
    print(train_dataset.class_names)
    print("\nCompare this list with your 'class_names.json' file.")

except Exception as e:
    print(f"\nAn error occurred. Please check that the path is correct.")
    print(f"Error details: {e}")