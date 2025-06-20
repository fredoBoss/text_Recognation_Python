import easyocr
import matplotlib.pyplot as plt
import cv2


# print("hello bay")
# read image
img_path = 'img/1.png'

img = cv2.imread(img_path)
# text detector
reader = easyocr.Reader(['en'], gpu=True)
# detect text in img
text = reader.readtext(img)

print(text)
# draw bbox