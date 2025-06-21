import easyocr
import cv2
import numpy as np

# Function to preprocess frame for better OCR
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding to enhance contrast
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Initialize webcam (0 is default camera, change if using external camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Set frame size (optional, adjust for performance)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Preprocess the frame
    if frame_count % 33 == 0:
        preprocessed_frame = preprocess_frame(frame)

        # Detect text with allowlist for numbers
        text_detections = reader.readtext(
            preprocessed_frame,
            allowlist='0123456789.-',  # Restrict to digits, decimal, minus
            detail=1,
            min_size=10,
            text_threshold=0.7
        )

    # Filter for numeric text and draw rectangles/text
    for detection in text_detections:
        bbox, text, score = detection
        # Check if text is numeric
        try:
            float(text.replace('-', ''))  # Validate number
            # Convert bounding box coordinates to integers
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            # Draw rectangle
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            # Add detected text above bounding box
            text_position = (top_left[0], top_left[1] - 10)
            cv2.putText(
                frame,
                text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Smaller font for live feed
                (0, 255, 0),  # Green text
                2
            )
            print(f"Detected number: {text} (Confidence: {score:.2f})")
        except ValueError:
            continue  # Skip non-numeric text

    # Display the frame
    cv2.imshow('Live Number Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()