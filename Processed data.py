import cv2
import numpy as np
import os

# Define dataset paths
input_dir = r"Z:\Robotics\tetris_shapes_dataset"  # Original dataset path
output_dir = r"Z:\Robotics\preprocessed_tetris_shapes"  # Preprocessed output folder

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Sharpening kernel
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])

def preprocess_image(image_path, output_path):
    """Loads, normalizes, sharpens, and saves the image."""
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping {image_path}, failed to load.")
        return
    
    # Resize image to 100x100 pixels
    img = cv2.resize(img, (100, 100))

    # Normalize pixel values to [0,1]
    img = img.astype(np.float32) / 255.0

    # Apply sharpening filter
    sharpened_img = cv2.filter2D(img, -1, sharpen_kernel)

    # Convert back to 8-bit format (0-255) for saving
    sharpened_img = (sharpened_img * 255).astype(np.uint8)

    # Save the processed image
    cv2.imwrite(output_path, sharpened_img)
    print(f"Saved: {output_path}")

# Process images for each shape
for shape in ["L-shape", "T-shape", "Z-shape"]:
    input_shape_dir = os.path.join(input_dir, shape)
    output_shape_dir = os.path.join(output_dir, shape)
    os.makedirs(output_shape_dir, exist_ok=True)  # Create output folder if missing

    # Iterate over images in the shape's folder
    for filename in os.listdir(input_shape_dir):
        input_path = os.path.join(input_shape_dir, filename)
        output_path = os.path.join(output_shape_dir, filename)
        preprocess_image(input_path, output_path)

print("✅ Preprocessing complete! Images saved in 'Z:\\Robotics\\preprocessed_tetris_shapes'.")
