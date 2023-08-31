# -PRODIGY_ML_03-
import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Paths to the training data folders
train_dir = 'path_to_train_data_directory'
cat_dir = os.path.join(train_dir, 'cat')
dog_dir = os.path.join(train_dir, 'dog')

# Load a pre-trained CNN model (VGG16 in this case) without the top classification layer
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Create data generators
datagen = ImageDataGenerator(rescale=1.0/255)
batch_size = 32

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',  # Assumes two classes: 'cat' and 'dog'
    shuffle=True
)

# Extract features using the pre-trained model
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = base_model.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, train_generator.samples)

# Flatten the features for SVM
train_features = np.reshape(train_features, (train_generator.samples, 7 * 7 * 512))

# Train an SVM classifier
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(train_features, train_labels)

# Evaluation
test_dir = 'path_to_test_data_directory'
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

test_features, test_labels = extract_features(test_dir, test_generator.samples)
test_features = np.reshape(test_features, (test_generator.samples, 7 * 7 * 512))

predictions = svm_model.predict(test_features)

print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))

