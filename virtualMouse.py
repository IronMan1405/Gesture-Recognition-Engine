import cv2
import mediapipe as mp
import time
import pyautogui
import threading
import math

camW, camH = 960, 540
# camW, camH = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, camW)
cap.set(4, camH)

mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

prevTime = 0
screen_w, screen_h = pyautogui.size()

prev_x, prev_y = 0, 0
smoothing = 2.5

pinch_threshhold = 0.025
dragging = False
pinch_start_time = None
pinch_start_pos = None

target_pos = None
worker_running = True

def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def getTipCoords(handLandmarks, id):
    index_x = handLandmarks.landmark[id].x
    index_y = handLandmarks.landmark[id].y
    return index_x, index_y

def movePointer(x, y):
    pyautogui.moveTo(x, y)

def mouseLogic():
    global target_pos
    last_pos = None

    while worker_running:
        if target_pos:
            if last_pos is None or abs(target_pos[0] - last_pos[0]) > 2 or abs(target_pos[1] - last_pos[1]) > 2:
                pyautogui.moveTo(*target_pos)
                last_pos = target_pos
        time.sleep(0.03)

# threading.Thread(target=mouseLogic, daemon=True).start()

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

            x_i, y_i = getTipCoords(hand_landmarks, 8) # getting index finger tip coords
            x_t, y_t = getTipCoords(hand_landmarks, 4)

            cx_i, cy_i = int(x_i*w), int(y_i*h)
            cx_t, cy_t = int(x_t*w), int(y_t*h)


            cv2.circle(img, (cx_i, cy_i), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx_t, cy_t), 10, (255, 0, 255), cv2.FILLED)

            pointer_x, pointer_y = int(x_i * screen_w), int(y_i * screen_h)
            smooth_x = prev_x + (pointer_x - prev_x) / smoothing
            smooth_y = prev_y + (pointer_y - prev_y) / smoothing

            # threading.Thread(target=movePointer, args=(smooth_x, smooth_y)).start()
            # pyautogui.moveTo(smooth_x, smooth_y)
            target_pos = (smooth_x, smooth_y)
            threading.Thread(target=mouseLogic, daemon=True).start()

            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

            if dist < pinch_threshhold:
                if pinch_start_time is None:
                    pinch_start_time = time.time()
                    pinch_start_pos = (smooth_x, smooth_y)
                
                if not dragging:
                    # pyautogui.click()
                    # time.sleep(0.2)
                    dx = abs(smooth_x - pinch_start_pos[0])
                    dy = abs(smooth_y - pinch_start_pos[1])

                    if dx > 20 or dy > 20:
                        pyautogui.mouseDown()
                        dragging = True

                # pyautogui.moveTo(smooth_x, smooth_y)
            else:
                if pinch_start_time is not None:
                    pinch_duration = time.time() - pinch_start_time
                    dx = abs(smooth_x - pinch_start_pos[0])
                    dy = abs(smooth_y - pinch_start_pos[1])

                    if not dragging and pinch_duration < 0.3 and dx < 20 and dy < 20:
                        pyautogui.click()  # quick pinch = click
                    elif dragging:
                        pyautogui.mouseUp()

                dragging = False
                pinch_start_time = None
                pinch_start_pos = None

            prev_x, prev_y = smooth_x, smooth_y

    currentTime = time.time()
    fps = 1/(currentTime-prevTime)
    prevTime = currentTime 

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3  )

    cv2.imshow("Capture", img)
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

worker_running = False
cap.release()
cv2.destroyAllWindows()