import cv2
import mediapipe as mp
import time
import pyautogui
import os
import platform

camW, camH = 960, 540
# camW, camH = 640, 480

cap = cv2.VideoCapture(0)
# cap.set(3, camW)
# cap.set(4, camH)

mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

# map each gesture to an action
gesture_actions = {
    "Open Palm": "Take Screenshot",
    "Closed Fist": "Lock",
    "Swipe Left": "Scroll Left",
    "Swipe Right": "Scroll Right",
    "Swipe Up": "Increase Volume",
    "Swipe Down": "Decrease Volume"
}

prevTime = 0
currentTime = 0

prev_wrist_x = None
prev_wrist_y = None

last_gesture = ""
gesture_time = ""
hold_time = 3.0

pending_action = None
action_start_time = 0

def performAction(action):
    system = platform.system()

    if action == "Take Screenshot":
        print("Taking Screenshot")
        if system == "Windows":
            pyautogui.hotkey("win", "printscreen")  # take screenshot
        elif system == "Darwin":  # macOS
            pyautogui.hotkey("command", "shift", "3")  # take screenshot
        else:
            pyautogui.hotkey("winleft", "printscreen")  # Linux

    # elif action == "Lock":
    if action == "Lock":
        print("Locking")
        if system == "Windows":
            pyautogui.hotkey("win", "l") 
        elif system == "Darwin": 
            # os.system('/System/Library/CoreServices/"Menu Extras"/User.menu/Contents/Resources/CGSession -suspend')
            os.system("pmset displaysleepnow")
        else:
            os.system("gnome-screensaver-command -l")  # Linux gnome command

    elif action == "Scroll Left":
        print("Scrolling Left")
        if system == "Windows":
            pyautogui.hotkey("left")
        elif system == "Darwin":  # macOS
            pyautogui.hotkey("left")
        else:  # Linux
            pyautogui.hotkey("left")

    elif action == "Scroll Right":
        print("Scrolling Right")
        if system == "Windows":
            # pyautogui.hotkey("win", "right")
            pyautogui.hotkey("left")
        elif system == "Darwin":
            pyautogui.hotkey("left")
        else:
            pyautogui.hotkey("left")

    elif action == "Increase Volume":
        print("increasing volume")
        if system == "Windows":
            pyautogui.hotkey("volumeup")
        elif system == "Darwin":
            os.system("osascript -e 'set volume output volume ((output volume of (get volume settings)) + 10)'")
        else:
            os.system("xdotool key XF86AudioRaiseVolume")

    elif action == "Decrease Volume":
        print("decreasing volume")
        if system == "Windows":
            pyautogui.hotkey("volumedown")
        elif system == "Darwin":
            os.system("osascript -e 'set volume output volume ((output volume of (get volume settings)) - 10)'")
        else:
            os.system("xdotool key XF86AudioLowerVolume")

    else:
        print(f"Unknown action: {action}")


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
        wrist_y = handLandmarks.landmark[0].y # get y coord of 0th landmark (wrist)
        
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
            # print(getFingerStates(hand_landmarks))

            gesture, prev_wrist_x, prev_wrist_y = detectGesture(hand_landmarks, prev_wrist_x, prev_wrist_y)

            if gesture != "":
                last_gesture = gesture
                gesture_time = time.time()

                if gesture in gesture_actions and pending_action is None:
                    pending_action = gesture_actions[gesture]
                    action_start_time = time.time()

                if pending_action:
                    cv2.putText(img, f"Action: {pending_action}", (camW//2 - 100, camH - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    # Wait 2 seconds before performing
                    if time.time() - action_start_time >= 2:
                        performAction(pending_action)
                        pending_action = None

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