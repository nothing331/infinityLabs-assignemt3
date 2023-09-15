import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import random

category_Annotations_folder = "VOCdevkit\VOC2011\Annotations"  # Replace with the actual path
Annotations_files = os.listdir(category_Annotations_folder)
num_selected_A = int(len(Annotations_files))

Annotations[]

 for Annotations in num_selected_A:
     full_Annotations = os.path.join(category_Annotations_folder, Annotations)
     Annotations.append(full_Annotations)

# category_B1_folder = "VOCdevkit\VOC2011\ImageSets\Action" 
# image_files_B1 = os.listdir(category_B1_folder)
# num_selected_B1 = int(len(image_files_B1) * 0.1)
# selected_images_B1 = random.sample(image_files_B1, num_selected_B1)


# category_B2_folder = "VOCdevkit\VOC2011\ImageSets\Layout" 
# image_files_B2 = os.listdir(category_B2_folder)
# num_selected_B2 = int(len(image_files_B2) * 0.1)
# selected_images_B2 = random.sample(image_files_B2, num_selected_B2)


# category_B3_folder = "VOCdevkit\VOC2011\ImageSets\Main" 
# image_files_B3 = os.listdir(category_B3_folder)
# num_selected_B3 = int(len(image_files_B3) * 0.1)
# selected_images_B3 = random.sample(image_files_B3, num_selected_B3)


# category_B4_folder = "VOCdevkit\VOC2011\ImageSets\Segmentation" 
# image_files_B4 = os.listdir(category_B4_folder)
# num_selected_B4 = int(len(image_files_B4) * 0.1)
# selected_images_B4 = random.sample(image_files_B4, num_selected_B4)
# selected_images_B4


category_A_folder = "VOCdevkit\VOC2011\JPEGImages" 
image_files_A = os.listdir(category_A_folder)
num_selected_A = random.randint(int(len(image_files_A) * 0.2), int(len(image_files_A) * 0.5))
selected_images_A = random.sample(image_files_A, num_selected_A)

category_B_folder = "VOCdevkit\VOC2011\SegmentationClass" 
image_files_B = os.listdir(category_B_folder)
num_selected_B = int(len(image_files_B) * 0.1)
selected_images_B = random.sample(image_files_B, num_selected_B)


category_C_folder = "VOCdevkit\VOC2011\SegmentationObject" 
image_files_C = os.listdir(category_C_folder)
num_selected_C = int(len(image_files_C) * 0.1)
selected_images_C = random.sample(image_files_C, num_selected_C)
selected_images_C

not_A = []
# for image_path in selected_images_B1:
#     full_image_path = os.path.join(category_B1_folder, image_path)
#     not_A.append(full_image_path)

# for image_path in selected_images_B2:
#     full_image_path = os.path.join(category_B2_folder, image_path)
#     not_A.append(full_image_path)

# for image_path in selected_images_B3:
#     full_image_path = os.path.join(category_B3_folder, image_path)
#     not_A.append(full_image_path)

# for image_path in selected_images_B4:
#     full_image_path = os.path.join(category_B4_folder, image_path)
#     not_A.append(full_image_path)

for image_path in selected_images_C:
    full_image_path = os.path.join(category_C_folder, image_path)
    not_A.append(image_path)

for image_path in selected_images_B:
    full_image_path = os.path.join(category_B_folder, image_path)
    not_A.append(image_path)

only_A =[]
for image_path in selected_images_A:
    full_image_path = os.path.join(category_A_folder, image_path)
    only_A.append(image_path)



training_dataset = []
for image_path in not_A :
    training_dataset.append(image_path)

for image_path in only_A :
    training_dataset.append(image_path)



training_dataset


# Load the pre-trained ResNet-50 model
model = ResNet50(weights='imagenet', include_top=False)

# Define image preprocessing transformations
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

# List to store feature vectors for selected images
feature_vectors = []
labels_valid = []

# Process and extract features for each selected image
for image_path in training_dataset:  # Use the training dataset created in Step 3
    # Load and preprocess the image
    img_array = preprocess_image(image_path)
    
    # Expand dimensions to match ResNet-50 input shape (batch size 1)
    img_array = tf.expand_dims(img_array, axis=0)

    # Extract features using the model
    features = model.predict(img_array)

    # Append the feature tensor to the list
    feature_vectors.append(features)

    label = image_path
    labels_valid.append(label)

# Stack the feature tensors into a single tensor
feature_matrix = tf.concat(feature_vectors, axis=0)
feature_vectors

# Load the pre-trained ResNet-50 model
model = ResNet50(weights='imagenet', include_top=False)

# Define image preprocessing transformations
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

# List to store feature vectors for selected images
feature_vectors = []
labels_valid = []

# Process and extract features for each selected image
for image_path in training_dataset:  # Use the training dataset created in Step 3
    # Load and preprocess the image
    img_array = preprocess_image(image_path)
    
    # Expand dimensions to match ResNet-50 input shape (batch size 1)
    img_array = tf.expand_dims(img_array, axis=0)

    # Extract features using the model
    features = model.predict(img_array)

    # Append the feature tensor to the list
    feature_vectors.append(features)

    label = image_path
    labels_valid.append(label)

# Stack the feature tensors into a single tensor
feature_matrix = tf.concat(feature_vectors, axis=0)
feature_vectors



all_image_paths = training_dataset  # List of image file paths
all_labels = Annotations  # List of corresponding labels

# thaking data used for validation as 20%
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