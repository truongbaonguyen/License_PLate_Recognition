import cv2
import glob
from lib_detection import load_model, detect_lp, im2single
import os
from os.path import join
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import easyocr
import psutil
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

binary = cv2.imread("miai.png")

text = pytesseract.image_to_string(binary, lang="eng")
print(text)

print('RAM usage is: ', psutil.Process().memory_info().rss / (1024 * 1024))

cv2.destroyAllWindows()