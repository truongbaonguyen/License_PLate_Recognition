import cv2
import glob
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

vid = cv2.VideoCapture(0)

while(True):
      
    ret, img = vid.read()
  
    cv2.imshow('frame', img)
    img_bk = img.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #img = cv2.resize(img, (600,400) )

    imgGray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
    imgCanny = cv2.Canny(imgBlur,5,5)
    kernel = np.ones((5,5))
    #imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    #imgThre = cv2.erode(imgDial,kernel,iterations=2)
    # cv2.imshow('Canny',imgCanny) 
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

        text = pytesseract.image_to_string(Cropped, config='--psm 11')
        text = ''.join(filter(str.isalnum, text))
        if (text):
            print("Detected License Plate Number is:",text)
        Cropped = cv2.resize(Cropped,(400,200))

        cv2.imshow('Cropped',Cropped)

