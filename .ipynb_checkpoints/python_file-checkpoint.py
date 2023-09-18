import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import random

category_Annotations_folder = "VOCdevkit/VOC2011/Annotations"  # Replace with the actual path
Annotations_files = os.listdir(category_Annotations_folder)


Annotations = []

for Annotations_file in Annotations_files:
    full_Annotations = os.path.join(category_Annotations_folder, Annotations_file)
    Annotations.append(full_Annotations)

category_A_folder = "VOCdevkit\VOC2011\JPEGImages" 
image_files_A = os.listdir(category_A_folder)
num_selected_A = random.randint(int(len(image_files_A) * 0.2), int(len(image_files_A) * 0.5))
selected_images_A = random.sample(image_files_A, num_selected_A)

not_A = []
category_B_folder = "VOCdevkit/VOC2011/SegmentationClass"  
image_files_B = os.listdir(category_B_folder)
num_selected_B = int(len(image_files_B) * 0.1)
selected_images_B = random.sample(image_files_B, num_selected_B)



for image_file in selected_images_B:
    full_image_path = os.path.join(category_B_folder, image_file)
    not_A.append(full_image_path)


category_C_folder = "VOCdevkit/VOC2011/SegmentationObject" 
image_files_C = os.listdir(category_C_folder)
num_selected_C = int(len(image_files_C) * 0.1)
selected_images_C = random.sample(image_files_C, num_selected_C)



for image_file in selected_images_C:
    full_image_path = os.path.join(category_C_folder, image_file)
    not_A.append(full_image_path)

# print(not_A)

only_A = []
category_A_folder = "VOCdevkit/VOC2011/JPEGImages"  # Replace with the actual path
image_files_A = os.listdir(category_A_folder)
num_selected_A = random.randint(int(len(image_files_A) * 0.2), int(len(image_files_A) * 0.5))
selected_images_A = random.sample(image_files_A, num_selected_A)



for image_file in selected_images_A:
    full_image_path = os.path.join(category_A_folder, image_file)
    only_A.append(full_image_path)




training_dataset = []
for image_path in not_A :
    training_dataset.append(image_path)

for image_path in only_A :
    training_dataset.append(image_path)

print(training_dataset)

# Load ResNet-50 model
model = ResNet50(weights='imagenet')

# Preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

# List to store feature vectors for selected images
feature_vectors = []
labels = []


image_files = training_dataset 
annotation_files = Annotations  

for img_path, label_path in zip(image_files, annotation_files):
    # Load and preprocess the image
    img_array = preprocess_image(img_path)

    # Forward pass through the model to extract features
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    features = model.predict(img_array)

    # Adding to the feature_vectors array
    feature_vectors.append(features)

    # Load the corresponding label from annotations
    with open(label_path, 'r') as label_file:
        label = label_file.read().strip()  # Read the content of the annotation file
        labels.append(label)

feature_matrix = np.vstack(feature_vectors)
labels = np.array(labels)


# Load ResNet-50 model 
model = ResNet50(weights='imagenet')

# preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

all_image_paths = training_dataset  # List of image file paths
all_labels = Annotations  # List of corresponding labels


validation_fraction = 0.2


num_validation_samples = int(len(all_image_paths) * validation_fraction)

combined = list(zip(all_image_paths, all_labels))
random.shuffle(combined)
all_image_paths[:], all_labels[:] = zip(*combined)

validation_image_paths = all_image_paths[:num_validation_samples]
validation_labels = all_labels[:num_validation_samples]

feature_vectors_valid = []

for image_path in validation_image_paths:
    img_array = preprocess_image(image_path)
    
    img_array = tf.expand_dims(img_array, axis=0)
    
    features = model.predict(img_array)
    
    feature_vectors_valid.append(features)

feature_matrix_valid = tf.concat(feature_vectors_valid, axis=0)

#kNN classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Creating an k-NN classifier 
k = 5  # Number of neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the k-NN classifier on the training data
knn_classifier.fit(feature_matrix, labels)

# Predict labels for validation set
predicted_labels = knn_classifier.predict(feature_matrix_valid)

# Calculate classification accuracy
accuracy = accuracy_score(labels_valid, predicted_labels)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")