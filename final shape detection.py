import cv2
import numpy as np
import joblib  # For loading the SVM model

# Load the trained SVM model
svm_model = joblib.load("tetris_svm_model.pkl")  # Adjust the path if necessary

# Initialize webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define shape labels
label_map = {0: "L-shape", 1: "T-shape", 2: "Z-shape"}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance the block visibility
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Resize to match the input size (100x100)
    resized = cv2.resize(thresh, (100, 100))

    # Flatten the image to a 1D array (as done during training)
    flattened = resized.flatten().reshape(1, -1)

    # Normalize the image to [0, 1] (same as during training)
    flattened = flattened / 255.0

    # Predict the shape using the SVM model
    prediction = svm_model.predict(flattened)
    predicted_label = label_map[prediction[0]]  # Get the predicted label name

    # Display the result on the frame
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with the prediction
    cv2.imshow("Real-Time Shape Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
