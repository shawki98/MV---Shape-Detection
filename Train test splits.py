import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
# Define dataset path
dataset_dir = r"Z:\Robotics\preprocessed_tetris_shapes"

# Mapping shape names to numerical labels
label_map = {"L-shape": 0, "T-shape": 1, "Z-shape": 2}

# Prepare data lists
X = []  # Images
y = []  # Labels

# Load images and labels
for shape, label in label_map.items():
    shape_dir = os.path.join(dataset_dir, shape)
    for filename in os.listdir(shape_dir):
        img_path = os.path.join(shape_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
        if img is None:
            continue
        X.append(img)
        y.append(label)

# Convert lists to NumPy arrays
X = np.array(X).reshape(len(X), -1)  # Flatten images
y = np.array(y)

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 ,shuffle = True)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict on test set
y_pred = svm_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model using joblib
joblib.dump(svm_model, "tetris_svm_model.pkl")  # Save the model to a file
print("Model saved to 'tetris_svm_model.pkl'")
