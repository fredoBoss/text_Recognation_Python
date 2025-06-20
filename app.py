import easyocr
import matplotlib.pyplot as plt
import cv2


# print("hello bay")
# read image
img_path = 'img/1.png'

img = cv2.imread(img_path)
# text detector
reader = easyocr.Reader(['en'], gpu=False)
# detect text in img
text = reader.readtext(img)

for t in text:
    print(t)
    
    bbox, txt, score = t
    
    cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
    
    
plt.imshow(img)
plt.show