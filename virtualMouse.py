import cv2
import mediapipe as mp
import time
import pyautogui
import threading
import math

camW, camH = 960, 540
# camW, camH = 640, 480

cap = cv2.VideoCapture(0)
# cap.set(3, camW)
# cap.set(4, camH)

#initializing mediapipe hand solutions
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

#drawing utilities by mediapipe to draw hand landmarks
mp_draw = mp.solutions.drawing_utils

prevTime = 0 # for fps calculation
screen_w, screen_h = pyautogui.size() # get screen width and height for mapping cursor coords

prev_x, prev_y = 0, 0 #previous cursor coords for smoothing (updated later accordingly)
smoothing = 2.5 # higher means smoother but more lag

# pinch and drag gesture flags
pinch_threshhold = 0.025 
dragging = False
pinch_start_time = None
pinch_start_pos = None

target_pos = None
worker_running = True # to stop mouse logic thread

#right click cooldown
last_right_click = 0
right_click_cooldown = 0.3 # in sec


lock = threading.Lock()

#to display action on camera feed
action = "idle"

#get normalized coords for a landmark
def getTipCoords(handLandmarks, id):
    index_x = handLandmarks.landmark[id].x
    index_y = handLandmarks.landmark[id].y
    return index_x, index_y

# logic for moving mouse, made a function to be able to run it in a thread in parallel with the entire code
def mouseLogic():
    global target_pos
    last_pos = None

    while worker_running:
        with lock:
            pos = target_pos

        # only move the cursor if position has changed significantly
        if pos:
            if last_pos is None or abs(target_pos[0] - last_pos[0]) > 2 or abs(target_pos[1] - last_pos[1]) > 2:
                pyautogui.moveTo(*pos)
                last_pos = pos
        time.sleep(0.001) # to avoid maxing the cpu

# determine which fingers are open or closed
#returns an array with 1s an 0s in the order [thumb, index, middle, ring, pinky]
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

# start the mouse movement thread
threading.Thread(target=mouseLogic, daemon=True).start()

while True:
    start = time.time() #get loop starting time to calculate latency

    action = "hover"

    frame, img = cap.read()
    if not frame:
        break

    img = cv2.flip(img, 1) # flip horizontally since opencv gives mirror output
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert bgr to rgb for mediapipe processing
    results = hand.process(img_rgb) # detect hand in the rgb image using process() and save it in results

    if results.multi_hand_landmarks: # if hand is detected by cam
        for hand_landmarks in results.multi_hand_landmarks: # for every hand landmark in the detected hand(s)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS) # draw the landmarks on the detected hand and join them, on the image/frame we display not the rgb one)

            # get coords for index and thumb tips
            x_i, y_i = getTipCoords(hand_landmarks, 8)
            x_t, y_t = getTipCoords(hand_landmarks, 4)

            # converting normalized coords to pixels on cam frame
            cx_i, cy_i = int(x_i*w), int(y_i*h)
            cx_t, cy_t = int(x_t*w), int(y_t*h)

            # draw the circles on index and thumb tips
            cv2.circle(img, (cx_i, cy_i), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx_t, cy_t), 10, (255, 0, 255), cv2.FILLED)

            #mapping index tip to screen to apply smoothing
            pointer_x, pointer_y = int(x_i * screen_w), int(y_i * screen_h)
            smooth_x = prev_x + (pointer_x - prev_x) / smoothing
            smooth_y = prev_y + (pointer_y - prev_y) / smoothing

            #update shared target position for mouseLogic thread
            with lock:
                target_pos = (smooth_x, smooth_y)

            #pinch detection logic
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

            if dist < pinch_threshhold:
                # start pinch
                if pinch_start_time is None:
                    pinch_start_time = time.time()
                    pinch_start_pos = (smooth_x, smooth_y)
                
                dy = abs(smooth_y - pinch_start_pos[1])
                dx = abs(smooth_x - pinch_start_pos[0])
                
                if not dragging and (dx > 20 or dy > 20):
                    pyautogui.mouseDown()
                    dragging = True
                    action = "dragging"
                elif dragging:
                    action = "dragging"
            else:
                # pinch released
                if pinch_start_time is not None:
                    pinch_duration = time.time() - pinch_start_time
                    dx = abs(smooth_x - pinch_start_pos[0])
                    dy = abs(smooth_y - pinch_start_pos[1])

                    if not dragging and pinch_duration < 0.3 and dx < 20 and dy < 20:
                        pyautogui.click()  # quick pinch = click
                        action = "click"
                    elif dragging:
                        pyautogui.mouseUp() # release drag

                dragging = False
                pinch_start_time = None
                pinch_start_pos = None

            # detecting closed fist for right click
            fingers = getFingerStates(hand_landmarks)
            if fingers == [0,0,0,0,0]:
                current_time = time.time()
                if current_time - last_right_click > right_click_cooldown:
                    pyautogui.rightClick()
                    last_right_click = current_time
                    action = "right click"
            
            #updating cursor's previous smooth coords 
            prev_x, prev_y = smooth_x, smooth_y
    else: # no hand detected
        action = "idle"

    #fps calculation
    currentTime = time.time()
    fps = 1/(currentTime-prevTime)
    prevTime = currentTime 
    
    #display fps and action text
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3  )
    cv2.putText(img, f"Action: {action}", (10, 120), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2)

    #render/show the camera feed
    cv2.imshow("Capture", img)

    end = time.time() # get loop end time for latency
    latency = (end - start) #total time for one loop iteration = latency
    print(f"latency: {latency:.3f} s")
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        worker_running = False
        break

cap.release()
cv2.destroyAllWindows()