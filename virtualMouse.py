import cv2
import mediapipe as mp
import time
import pyautogui
import threading

camW, camH = 960, 540
camW, camH = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, camW)
cap.set(4, camH)

mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

prevTime = 0
currentTime = 0

screen_w, screen_h = pyautogui.size()

prev_x, prev_y = 0, 0
smoothing = 2.5

def getIndexCoords(handLandmarks):
    index_x = handLandmarks.landmark[8].x
    index_y = handLandmarks.landmark[8].y
    return index_x, index_y

def movePointer(x, y):
    pyautogui.moveTo(x, y)

while True:
    frame, img = cap.read()
    
    if not frame:
        break

    h, w, c = img.shape
 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hand.process(img_rgb) # detect hand in the rgb image using process() and save it in results

    if results.multi_hand_landmarks: # if hand is detected by cam
        for hand_landmarks in results.multi_hand_landmarks: # for every hand landmark in the detected hand(s)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS) # draw the landmarks on the detected hand and join them, on the image/frame we display not the rgb one)

            x, y = getIndexCoords(hand_landmarks)

            cx, cy = int(x*w), int(y*h)

            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            pointer_x, pointer_y = int(x * screen_w), int(y * screen_h)

            smooth_x = prev_x + (pointer_x - prev_x) / smoothing
            smooth_y = prev_y + (pointer_y - prev_y) / smoothing

            threading.Thread(target=movePointer, args=(smooth_x, smooth_y)).start()

            prev_x, prev_y = smooth_x, smooth_y


    currentTime = time.time()
    fps = 1/(currentTime-prevTime)
    prevTime = currentTime 

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3  )

    cv2.imshow("Capture", img)
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()