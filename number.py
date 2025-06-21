import os
import pytesseract
from PIL import Image

def detect_number_from_image(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(img).strip()
    
    # Check if the detected text is a number between 1 and 5
    if text.isdigit() and 1 <= int(text) <= 5:
        return int(text)
    return None

def process_images(directory):
    # List to store results
    results = {}
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            image_path = os.path.join(directory, filename)
            number = detect_number_from_image(image_path)
            if number is not None:
                results[filename] = number
            else:
                results[filename] = "No valid number (1-5) detected"
    
    return results

def main():
    # Assuming the script is run from the same directory as 'img'
    img_dir = 'img'
    
    # Process all images
    results = process_images(img_dir)
    
    # Print results
    for filename, number in results.items():
        print(f"{filename}: {number}")

if __name__ == "__main__":
    main()