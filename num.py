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

def detect_numbers(image_name):
    # Load the input image from img folder
    image_path = os.path.join('img', image_name)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load templates
    templates = load_templates()
    if not templates:
        print("Error: No templates loaded")
        return
    
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
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(img, digit, (pt1[0], pt1[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow('Detected Numbers', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print detected numbers
    print("Detected numbers:", [d[0] for d in detections])

if __name__ == "__main__":
    # Specify the input image name in the img folder (e.g., another image to detect numbers in)
    detect_numbers('3.png')