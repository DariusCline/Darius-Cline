import cv2
from PIL import Image
import numpy as np
import os
import sys

# Ensure the training directory exists
TRAIN_DIR = "training_data"
if not os.path.exists(TRAIN_DIR):
    print(f"Error: Training data directory '{TRAIN_DIR}' not found.")
    print("Please create the directory and add training images before running the script.")
    sys.exit(1)

# --- MODEL FUNCTIONS ---

def load_image(path):
    """Loads an image from disk, converts to grayscale, resizes, flattens, and normalizes."""
    if not os.path.exists(path):
        print(f"Warning: Image not found at {path}. Skipping.")
        # Return a zero array of the correct size to prevent errors later if an image is missing
        return np.zeros(1024) 
    try:
        img = Image.open(path).convert("L") # L for grayscale
        img = img.resize((32, 32))
        data = np.array(img).flatten()
        return data / 255.0
    except Exception as e:
        print(f"Error processing image {path}: {e}. Skipping.")
        return np.zeros(1024)

def sigmoid(x):
    """Sigmoid activation function."""
    x = np.clip(x, -500, 500) 
    return 1 / (1 + np.exp(-x))

weights = np.random.rand(1024)
bias = 0.0
learning_rate = 0.01

def get_raw_score(image_data):
    total = np.dot(weights, image_data) + bias
    return total

def train(image_data, label):
    global weights, bias
    raw_score = get_raw_score(image_data)
    prediction = sigmoid(raw_score) 
    error = label - prediction
    weights += learning_rate * error * image_data
    bias += learning_rate * error

# --- WEBCAM AND PROCESSING FUNCTIONS ---

def process_captured_image(pil_img):
    """Processes a PIL image object for prediction."""
    img_gray = pil_img.convert("L")
    img_resized = img_gray.resize((32, 32))
    data = np.array(img_resized).flatten()
    return data / 255.0

def capture_image_from_webcam():
    """Opens webcam feed and captures a frame when SPACE is pressed."""
    cam = cv2.VideoCapture(0) 
    if not cam.isOpened():
        print("Failed to open camera. Check connections and permissions.")
        return None
    print("\n>>> Press SPACEBAR to capture an image, or ESC to exit the preview window. <<<")
    captured_frame = None
    while True:
        ret, frame = cam.read()
        if not ret: break
        cv2.imshow("Webcam Feed - Press SPACE", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27: break # ESC key
        elif k % 256 == 32: captured_frame = frame; break # SPACEBAR key
    cam.release()
    cv2.destroyAllWindows()
    return captured_frame

# --- MAIN EXECUTION LOGIC ---

# 1. YOUR ORIGINAL TRAINING DATA DEFINITION IS HERE:
train_data_paths_labels = [
    (os.path.join(TRAIN_DIR, "cat1.jpg"), 1), 
    (os.path.join(TRAIN_DIR, "cat2.jpg"), 1),
    (os.path.join(TRAIN_DIR, "cat3.jpg"), 1), 
    (os.path.join(TRAIN_DIR, "cat4.jpg"), 1),
    (os.path.join(TRAIN_DIR, "cat5.jpg"), 1), 
    (os.path.join(TRAIN_DIR, "cat6.jpg"), 1),
    (os.path.join(TRAIN_DIR, "cat7.jpg"), 1), 
    (os.path.join(TRAIN_DIR, "cat8.jpg"), 1),
    (os.path.join(TRAIN_DIR, "cat9.jpg"), 1), 
    (os.path.join(TRAIN_DIR, "cat10.jpg"), 1),
    (os.path.join(TRAIN_DIR, "cat11.jpg"), 1),
    (os.path.join(TRAIN_DIR, "cat12.jpg"), 1),
    (os.path.join(TRAIN_DIR, "cat13.jpg"), 1), 
    (os.path.join(TRAIN_DIR, "cat14.jpg"), 1),
    (os.path.join(TRAIN_DIR, "cat15.jpg"), 1), 
    (os.path.join(TRAIN_DIR, "cat16.jpg"), 1),
    (os.path.join(TRAIN_DIR, "cat17.jpg"), 1), 
    (os.path.join(TRAIN_DIR, "cat18.jpg"), 1),
    (os.path.join(TRAIN_DIR, "cat19.jpg"), 1), 
    (os.path.join(TRAIN_DIR, "cat20.jpg"), 1),
    (os.path.join(TRAIN_DIR, "cat21.jpg"), 1), 
    (os.path.join(TRAIN_DIR, "cat22.jpg"), 1),
    (os.path.join(TRAIN_DIR, "cat23.jpg"), 1),
    (os.path.join(TRAIN_DIR, "cat24.jpg"), 1),
    (os.path.join(TRAIN_DIR, "cat25.jpg"), 1),
    (os.path.join(TRAIN_DIR, "not_cat1.jpg"), 0),
    (os.path.join(TRAIN_DIR, "not_cat2.jpg"), 0),
    (os.path.join(TRAIN_DIR, "not_cat3.jpg"), 0),
    (os.path.join(TRAIN_DIR, "not_cat4.jpg"), 0),
    (os.path.join(TRAIN_DIR, "not_cat5.jpg"), 0),
    (os.path.join(TRAIN_DIR, "not_cat6.jpg"), 0),
    (os.path.join(TRAIN_DIR, "not_cat7.jpg"), 0),
    (os.path.join(TRAIN_DIR, "not_cat8.jpg"), 0),
    (os.path.join(TRAIN_DIR, "not_cat9.jpg"), 0),
    (os.path.join(TRAIN_DIR, "not_cat10.jpg"), 0),
    (os.path.join(TRAIN_DIR, "not_cat11.jpg"), 0),
    (os.path.join(TRAIN_DIR, "not_cat12.jpg"), 0),
    (os.path.join(TRAIN_DIR, "not_cat13.jpg"), 0),
]

# 2. Train the model using the list defined above
print("Starting model training (100 epochs)...")
for epoch in range(100):
    for path, label in train_data_paths_labels:
        img_data = load_image(path)
        # load_image returns None or a zero array if it fails, handle appropriately
        if np.sum(img_data) > 0: 
            train(img_data, label)
print("Training complete.")

# 3. Capture image from webcam and run prediction
captured_image_bgr = capture_image_from_webcam()

if captured_image_bgr is not None:
    captured_image_rgb = cv2.cvtColor(captured_image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(captured_image_rgb)
    test_image_processed = process_captured_image(pil_image)

    raw_score = get_raw_score(test_image_processed)
    confidence_score = sigmoid(raw_score)
    result = 1 if confidence_score >= 0.5 else 0

    print(f"\n--- Prediction Results ---")
    print(f"Confidence Score (Sigmoid Output): {confidence_score:.4f}")
    
    if result == 1:
        print(">> Cat detected! <<")
    else:
        print(">> No cat detected. <<")
