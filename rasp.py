import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

url = 'rtsp://trung:Trung12369@@192.168.50.11:554/cam/realmonitor?channel=1&subtype=0'
vid = cv2.VideoCapture(url)
count = 0
old_text = ''
convert = {
    '0' : 'D',
    '1' : 'L',
    '2' : 'Z',
    '3' : 'B',
    '4' : 'A',
    '5' : 'S',
    '6' : 'G',
    '7' : 'T',
    '8' : 'B',
    'OK' : 'F'
}

def handle_text(text):
    global count, old_text
    length = len(text)
    if length > 6 and length < 9:   # bien so xe co 7-9 ky tu
        # convert neu vi tri cua ky tu khong hop ly
        if text[2].isnumeric():
            ls_text = list(text)
            ls_text[2] = convert.get(text[2])
            text = ''.join(ls_text)
        for i in range(length):
            if i != 2 and i != 3 and text[i].isalpha():
                if text[i] in convert.values():
                    ls_text = list(text)
                    ls_text[i] = list(convert.keys())[list(convert.values()).index(text[i])]
                    text = ''.join(ls_text)
        
        # detect giong nhau 5 lan moi in ra
        if old_text != text:
            old_text = text
            count = 0
        else:
            count += 1
        if (count == 5):
            print("Detected License Plate Number is:", text)
            count = 0

while(True):
    ret, img = vid.read()
  
    imgGray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
    imgCanny = cv2.Canny(imgBlur,5,5)
    kernel = np.ones((5,5))

    contours, _ = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    for c in contours:
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is None:
        detected = 0
    else:
        detected = 1
    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
        mask = np.zeros(imgBlur.shape,np.uint8)
        new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
        new_image = cv2.bitwise_and(img,img,mask=mask)
        # cv2.imshow('new_image',new_image) 
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = imgBlur[topx:bottomx+1, topy:bottomy+1]
        Cropped = cv2.threshold(Cropped,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(Cropped, config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHKLMNPSTUVXYZ0123456789')
        text = ''.join(filter(str.isalnum, text))
        handle_text(text)
        Cropped = cv2.resize(Cropped,(400,200))
        cv2.imshow('Cropped',Cropped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

