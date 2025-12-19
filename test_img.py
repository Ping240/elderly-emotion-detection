import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def load_face_detector():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def preprocess_face(face_img, target_size=(48, 48)):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    normalized = resized / 255.0
    preprocessed = np.expand_dims(np.expand_dims(normalized, -1), 0)
    return preprocessed

def detect_emotions_image(image_path, model_path):
    # Load model and detector
    model = load_model(model_path, compile=False)
    face_detector = load_face_detector()
    
    # Read and get original image dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")
    
    # Create a larger canvas with padding
    height, width = image.shape[:2]
    padding = 50  # Add padding for text and boundaries
    canvas = np.ones((height + 2*padding, width + 2*padding, 3), dtype=np.uint8) * 255
    
    # Place original image in center of canvas
    canvas[padding:padding+height, padding:padding+width] = image
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Process each face
    for (x, y, w, h) in faces:
        # Adjust coordinates for padded canvas
        x_pad = x + padding
        y_pad = y + padding
        
        # Extract face ROI from original image
        face_roi = image[y:y+h, x:x+w]
        
        # Preprocess face
        preprocessed_face = preprocess_face(face_roi)
        
        # Get prediction
        prediction = model.predict(preprocessed_face)
        
        # Get emotion label
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_label = emotions[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        # Draw rectangle on padded canvas
        cv2.rectangle(canvas, (x_pad, y_pad), (x_pad+w, y_pad+h), (0, 255, 0), 2)
        
        # Add label with better positioning
        label = f"{emotion_label}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Ensure label is within bounds
        text_x = max(x_pad, padding)
        text_y = max(y_pad - 10, label_size[1] + 5)
        
        # Add white background to text for better visibility
        cv2.rectangle(canvas, 
                     (text_x, text_y - label_size[1] - 5),
                     (text_x + label_size[0], text_y + 5),
                     (255, 255, 255),
                     -1)
        
        cv2.putText(canvas, label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
    
    # Resize if image is too large for screen
    screen_height = 800  # Maximum height for display
    if canvas.shape[0] > screen_height:
        scale = screen_height / canvas.shape[0]
        display_img = cv2.resize(canvas, None, fx=scale, fy=scale)
    else:
        display_img = canvas
    
    # Display with window name and wait for key press
    cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('Emotion Detection', display_img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # Save the output image
    output_path = 'output_emotion.jpg'
    cv2.imwrite(output_path, canvas)
    return output_path

# Usage
if __name__ == "__main__":
    # image_path = "angry.webp"
    # image_path = "disgust_e.jpg"
    # image_path = "fear.webp"
    # image_path = "neutral.jpg"
    # image_path = "happy.jpg"
    # image_path = "surprise.webp"
    # image_path = "jojo.png"
    # image_path = "jojo.jpg"
    # image_path = "jojo2.jpg"
    image_path = "test_2247_aligned.jpg"

    model_path = "elderly_emotion_model.h5"

    output_path = detect_emotions_image(image_path, model_path)
    print(f"Processed image saved to: {output_path}")