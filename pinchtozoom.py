import cv2
import handDetector as hd
import numpy as np

cap = cv2.VideoCapture(0)
detector = hd.HandDetector(HandNo=1)
while True:
    success, img = cap.read()
    img = detector.process(img, False)
    lmList = detector.fingerdetector(img, False)
    text = "Not Zooming"
    color = (0, 255, 0)
    if lmList:
        _, x1, y1 = lmList[4]
        _, x2, y2 = lmList[8]
        #cv2.circle(img, (x1, y1), 3, (255, 255, 255))
        #cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        #cv2.circle(img, (x2, y2), 3, (255, 255, 255))
        dist = ((((y2-y1)**2)+((x2-x1)**2))**0.5)
        if dist > 30:
            converted = np.interp(dist, [30, 150], [1, 2])
            img = cv2.resize(img, None, fx=converted, fy=converted)
            cx, cy, _ = img.shape
            h, w = 480, 640
            cx, cy = int((cx - h)/2),int((cy - w)/2)
            img = img[cx:h+cx, cy:w+cy]
            text = "Zooming"
            color = (0,0,255)

    img = cv2.flip(img, 1)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
    cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(img, text, (0, 20), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
