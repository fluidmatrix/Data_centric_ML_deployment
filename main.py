import os
import gc
import tarfile
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score
import lab_utils
import tensorflow as tf

# Filename of the downloaded dataset archive
DATASET_COMPRESSED = './cats_dogs_birds.tar.gz'

# Base directory for extracting and preparing the dataset
DATA_DIR = './data'

# Base directory for extracting the pretrained models
MODEL_DIR = './models'

# Name of the classes to predict
ANIMALS = ['dogs', 'cats', 'birds']

# Directories for the training and dev sets
TRAIN_DIRS = ['train/dogs', 'train/cats', 'train/birds']
DEV_DIRS = ['dev/dogs', 'dev/cats', 'dev/birds']

# Imbalanced portion of images among the 3 classes
PORTIONS = [0.2, 1, 0.1]

base_dogs_dir = os.path.join(DATA_DIR, 'images/dog')
base_cats_dir = os.path.join(DATA_DIR,'images/cat')
base_birds_dir = os.path.join(DATA_DIR,'images/bird')
base_image_dirs = [base_dogs_dir, base_cats_dir, base_birds_dir]


# Instantiate the Dataset object for the training set
train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR,'train'),
    image_size=(150, 150),
    batch_size=32,
    label_mode='int'
    )

# Instantiate the Dataset object for the dev set
dev_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR,'dev'),
    image_size=(150, 150),
    batch_size=32,
    label_mode='int'
    )

# Define the layer to normalize the images
rescale_layer = tf.keras.layers.Rescaling(1./255)

# Apply the layer to the datasets
train_dataset_scaled = train_dataset.map(lambda image, label: (rescale_layer(image), label))
dev_dataset_scaled = dev_dataset.map(lambda image, label: (rescale_layer(image), label))

train_dataset_final = train_dataset_scaled.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
dev_dataset_final = dev_dataset_scaled.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Create a new model with utilities file
balanced_model = lab_utils.create_model()

balanced_history = balanced_model.fit(
    train_dataset_final,
    epochs = 5,
    validation_data = dev_dataset_final
)

# This will succeed if the model was trained on Colab or an environment with GPU.
try:
    balanced_history

# If it fails, load pre-generated history and model files.
except NameError:

    # Load the history
    with open('./histories/balanced_history.pkl', "rb") as pickle_file:
        balanced_history = pickle.load(pickle_file)

    # Load the pre-trained imbalanced model. This will be used in the next cell.
    balanced_model = tf.keras.models.load_model('./models/balanced_model')

# Get the true labels
y_true = dev_dataset_final.map(lambda image, label: label).unbatch()
y_true = list(y_true)

# Use the model to predict (will take a couple of minutes)
predictions_balanced = balanced_model.predict(dev_dataset_final)

# Get the argmax (since softmax is being used)
y_pred_balanced = np.argmax(predictions_balanced, axis=1)

# Print accuracy score
print(f"Accuracy Score: {accuracy_score(y_true, y_pred_balanced)}")

# Print balanced accuracy score
print(f"Balanced Accuracy Score: {balanced_accuracy_score(y_true, y_pred_balanced)}")


balanced_cm = confusion_matrix(y_true, y_pred_balanced)
ConfusionMatrixDisplay(balanced_cm, display_labels=['birds', 'cats', 'dogs']).plot(values_format="d")


lab_utils.plot_train_eval(balanced_history)



# Call the Python garbage collector to free memory
gc.collect()
    