import cv2                                          # Import OpenCV library for image processing
from cvzone.HandTrackingModule import HandDetector   # Import HandDetector module from cvzone library for detecting hand
from cvzone.ClassificationModule import Classifier   # Import Classifier module from cvzone library for hand gesture classification
import numpy as np                                  # Import numpy library for numerical computations
import math                                         # Import math library for mathematical operations
import time                                         # Import time library for keeping track of time

"pip install cvzone"                                # Install cvzone library using pip command
"pip install mediapipe"                             # Install mediapipe library using pip command
"pip install tensorflow"                            # Install tensorflow library using pip command

cap = cv2.VideoCapture(0)                           # Initialize video capture object for webcam
detector = HandDetector(maxHands=1)                 # Initialize HandDetector object with maxHands set to 1
classifier = Classifier("Model/Model1/keras_model.h5", "Model/Model1/labels1.txt")# Load the trained Keras model and its labels
classifier = Classifier("Model/Model2/keras_model.h5", "Model/Model2/labels2.txt")


offset = 20                                         # Set the offset to 20 pixels
imgSize = 300                                       # Set the image size to 300 pixels

folder = "Data/A"                                   # Set the folder for storing data
counter = 0                                         # Initialize counter variable to 0

labels1 = [ 0, 1, 2, "A", "B", "C" ] # Define labels for gesture classification
labels2 = ["D", "E"]

try:
    while True:                                     # Start an infinite loop
        success, img = cap.read()                   # Capture the image from the webcam
        imgOutput = img.copy()                      # Make a copy of the captured image
        hands, img = detector.findHands(img)         # Detect hands in the captured image using the HandDetector object
        if hands:                                   # If hands are detected
            hand = hands[0]                         # Get the first hand detected
            x , y , w , h = hand['bbox']             # Get the bounding box coordinates of the hand

            imgWhite = np.ones((imgSize, imgSize, 3),  np.uint8)*255  # Create a white image of size 300x300 with 3 channels
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]  # Crop the region of interest from the original image

            imgCropShape = imgCrop.shape              # Get the shape of the cropped image

            aspectRatio=h/w                          # Calculate the aspect ratio of the cropped image

            if aspectRatio > 1:                      # If the aspect ratio is greater than 1, i.e., height > width
                k = imgSize/h                        # Calculate the scaling factor
                wCal = math.ceil(k*w)                # Calculate the new width of the image
                imgResize = cv2.resize(imgCrop,(wCal, imgSize))   # Resize the cropped image to new dimensions
                imgResizeShape = imgResize.shape     # Get the shape of the resized image
                wGap = math.ceil((imgSize-wCal)/2)   # Calculate the gap to be left on either side of the image
                imgWhite[:, wGap:wCal+wGap] = imgResize   # Place the resized image in the white image
                prediction, index = classifier.getPrediction(imgWhite, draw=False)   # Classify the hand gesture using the Classifier object
                print(prediction,index)             # Print the predicted label and its index


            else:
                # Calculate scaling factor
                k = imgSize / w
                # Calculate new height based on the scaling factor
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                # Calculate gap between the resized image and the white image
                hGap = math.ceil((imgSize - hCal) / 2)
                # Replace the appropriate section of the white image with the resized image
                imgWhite[hGap:hCal + hGap, :] = imgResize
                # Make a prediction for the image using the classifier
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Draw a rectangle and text for the predicted label
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)

            cv2.putText(imgOutput, str(labels1[index]), (x, y - 27), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.putText(imgOutput, str(labels2[index]), (x, y - 27), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

            # Draw a rectangle around the detected hand
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            # Show the cropped image and the white image for debugging purposes
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        # Show the output image
        cv2.imshow("Image", imgOutput)
        key = cv2.waitKey(1)

except Exception as e:
   pass