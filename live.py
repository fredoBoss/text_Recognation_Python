import cv2
import numpy as np
import os

def load_templates():
    templates = {}
    template_folder = 'img'  # Use img folder as the source of templates
    for i in range(1, 6):
        # Load template images directly from img folder
        template_path = os.path.join(template_folder, f'{i}.png')
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            templates[str(i)] = template
        else:
            print(f"Warning: Could not load template {template_path}")
    return templates

def detect_numbers(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load templates
    templates = load_templates()
    if not templates:
        print("Error: No templates loaded")
        return frame, []
    
    # Store detection results
    detections = []
    
    # Perform template matching for each number
    for digit, template in templates.items():
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8  # Adjust threshold for detection sensitivity
        loc = np.where(result >= threshold)
        
        # Get template dimensions
        h, w = template.shape
        
        # Store detected locations
        for pt in zip(*loc[::-1]):
            detections.append((digit, pt, (pt[0] + w, pt[1] + h)))
    
    # Draw rectangles around detected numbers
    for digit, pt1, pt2 in detections:
        # Draw thicker green bounding box
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 3)
        # Add label with black background for better visibility
        label = digit
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (pt1[0], pt1[1] - text_height - 5), 
                     (pt1[0] + text_width, pt1[1]), (0, 0, 0), -1)
        cv2.putText(frame, label, (pt1[0], pt1[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame, detections

if __name__ == "__main__":
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect numbers in the frame
        frame, detections = detect_numbers(frame)
        
        # Display the result
        cv2.imshow('Live Number Detection', frame)
        
        # Print detected numbers or validation message
        if detections:
            print("Detected numbers:", [d[0] for d in detections])
        else:
            print("Number couldn't detect")
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()