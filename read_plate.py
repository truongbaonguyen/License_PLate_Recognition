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

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Đường dẫn thư mục ảnh
img_path = glob.glob("test_Trung/*.jpg")

for i in range(len(img_path)): # len(img_path)
    # Đọc file ảnh đầu vào
    Ivehicle = cv2.imread(img_path[i]) 

    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

    if (len(LpImg)):

        # Chuyen doi anh bien so
        LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

        # Chuyen anh bien so ve gray
        gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)

        # Ap dung threshold de phan tach so va nen
        blur = cv2.GaussianBlur(gray,(7,7),0)
        binary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # binary = cv2.imread("bsx.jpg")
        # binary = binary[int(binary.shape[0]/2):binary.shape[0],:]
        cv2.imshow("Bsx sau threshold", binary)

        
        text = pytesseract.image_to_string(binary, lang="eng")
        text = ''.join(filter(str.isalnum, text))
        print(text)
        print('RAM usage is: ', psutil.Process().memory_info().rss / (1024 * 1024))
        # Hien thi anh input
        cv2.imshow("Anh input", Ivehicle)
        cv2.waitKey()

cv2.destroyAllWindows()