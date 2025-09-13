import cv2
import mediapipe as mp
import time

camW, camH = 960, 540

cap = cv2.VideoCapture(0)
cap.set(3, camW)
cap.set(4, camH)

mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

prevTime = 0
currentTime = 0

prev_wrist_x = None
prev_wrist_y = None

last_gesture = ""
gesture_time = ""
hold_time = 3.0

def getFingerStates(handLandmarks):
    fingers = []

    # index 4 - thumb tip, index 3 - thumb joint; checking x coord of tip of thumb relative to the joint to determine closed or open
    if handLandmarks.landmark[4].x > handLandmarks.landmark[3].x: 
        fingers.append(1) #open
    else:
        fingers.append(0) #close

    tips = [8, 12, 16, 20] # tips of other 4 fingers
    pips = [6, 10, 14, 18] # 2nd joint on the finger from the tip (called pip)

    # check for every finger
    for tip, pip in zip(tips, pips):
        # same logic as above, but along y direction since all other fingers open vertically
        if handLandmarks.landmark[tip].y < handLandmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def detectGesture(handLandmarks, prev_x, prev_y):
    fingers = getFingerStates(handLandmarks)

    gesture = "" # empty string to store the gesture

    count = sum(fingers)

    #palm and fist
    if count == 5: #all open
        gesture = "Open Palm"
    
    elif count == 0: #all closed
        gesture = "Closed Fist"
    
    else:
        gesture = f"{count} finger(s)"


    if gesture == "Open Palm":
        wrist_x = handLandmarks.landmark[0].x # get x coord of 0th landmark (wrist)
        wrist_y = handLandmarks.landmark[0].y
        
        if prev_x is not None:
            dx = wrist_x - prev_x # change in x coord of wrist/hand
            # print(dx)
            if dx > 0.015: #change is +ve i.e., hand moved/swiped right
                gesture = "Swipe Right"
            elif dx < -0.015: # change is -ve i.e., hand moved left
                gesture = "Swipe Left"
            
        if prev_y is not None:
            dy = wrist_y - prev_y
            if dy < -0.015: #actually opposite to the right left logic
                gesture = "Swipe Up"
            elif dy > 0.015:
                gesture = "Swipe Down"

        prev_x, prev_y = wrist_x, wrist_y
        
    return gesture, prev_x, prev_y

while True:
    frame, img = cap.read()
 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hand.process(img_rgb) # detect hand in the rgb image using process() and save it in results

    if results.multi_hand_landmarks: # if hand is detected by cam
        for hand_landmarks in results.multi_hand_landmarks: # for every hand landmark in the detected hand(s)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS) # draw the landmarks on the detected hand and join them, on the image/frame we display not the rgb one

            gesture, prev_wrist_x, prev_wrist_y = detectGesture(hand_landmarks, prev_wrist_x, prev_wrist_y)

            if gesture != "":
                last_gesture = gesture
                gesture_time = time.time()

            if time.time() - gesture_time <= hold_time:
                cv2.putText(img, last_gesture, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

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