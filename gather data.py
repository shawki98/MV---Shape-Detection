import cv2
import os

# Define dataset directory
dataset_dir = "tetris_shapes_dataset"

# Create subdirectories for each shape
shape_labels = ["T-shape", "Z-shape", "L-shape"]
for label in shape_labels:
    os.makedirs(os.path.join(dataset_dir, label), exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

image_count = {label: len(os.listdir(os.path.join(dataset_dir, label))) for label in shape_labels}

print("Press 't' to save as T-shape, 'z' for Z-shape, 'l' for L-shape.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    # Display the frame
    cv2.imshow("Capture Shapes", frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key in [ord('t'), ord('z'), ord('l')]:
        shape_name = "T-shape" if key == ord('t') else "Z-shape" if key == ord('z') else "L-shape"
        save_path = os.path.join(dataset_dir, shape_name, f"{image_count[shape_name]}.jpg")
        cv2.imwrite(save_path, frame)
        image_count[shape_name] += 1
        print(f"Saved: {save_path}")

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
