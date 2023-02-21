import cv2
from cvzone.HandTrackingModule import HandDetector

offset = 20

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y-offset:y + h+offset , x-offset:x + w+offset]
        cv2.imshow("ImageCrop", imgCrop)

    cv2.imshow("Image", img)
    cv2.waitKey(1)