import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

prevTime = 0
currentTime = 0

while True:
    frame, img = cap.read()
 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hand.process(img_rgb) # detect hand in the rgb image using process() and save it in results

    if results.multi_hand_landmarks: # if hand is detected by cam
        for hand_landmarks in results.multi_hand_landmarks: # for every hand landmark in the detected hand(s)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS) # draw the landmarks on the detected hand and join them, on the image/frame we display not the rgb one

    currentTime = time.time()
    fps = 1/(currentTime-prevTime)
    prevTime = currentTime 

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3  )

    cv2.imshow("Capture", img)
    cv2.waitKey(1)