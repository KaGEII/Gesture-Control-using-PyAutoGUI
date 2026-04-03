import cv2
import mediapipe as mp
import time
import pyautogui
import os

# Mission control
def mission_control_on():
    os.system("osascript -e 'tell application \"System Events\" to key code 160'")

#Play/Pause
def play_pause():
    pyautogui.press('space')


# Volume helpers
def volume_up():
    os.system("osascript -e 'set volume output volume (output volume of (get volume settings) + 10)'")

def volume_down():
    os.system("osascript -e 'set volume output volume (output volume of (get volume settings) - 10)'")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def fingers_up(hand_landmarks, hand_label):
    lm = hand_landmarks.landmark
    fingers = []

    # Thumb
    if hand_label == "Right":
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)

    # Other fingers
    fingers.append(lm[8].y  < lm[6].y)   # index
    fingers.append(lm[12].y < lm[10].y)  # middle
    fingers.append(lm[16].y < lm[14].y)  # ring
    fingers.append(lm[20].y < lm[18].y)  # pinky
    return fingers


def detect_gesture(fingers):
    total = sum(fingers)
    if total == 0:
        return "Fist"
    if total == 5:
        return "Open" 
    if fingers == [False, True, False, False, False]:
        return "One"
    if fingers == [False, True, True, False, False]:
        return "Two / Peace"
    if fingers == [True, False, False, False, False]:
        return "Thumbs Up"
    return f"{total} fingers"


def main():
    cap = cv2.VideoCapture(1)

    screen_w, screen_h = pyautogui.size()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        prev_time = 0
        prev_x, prev_y = None, None
        last_action_time = 0
        cooldown = 0.8

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Couldn't read from the camera. Is it in use?")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            
            gesture_text = ""
            if results.multi_hand_landmarks:
                for hand_landmarks, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    label = handed.classification[0].label

                    fingers = fingers_up(hand_landmarks, label)
                    gesture_text = detect_gesture(fingers)

                    now_action = time.time()
                    
                    if gesture_text == "One":
                    # Index finger tip
                        if gesture_text=="One":
                            index_finger = hand_landmarks.landmark[8]
                            x = int(index_finger.x * screen_w)
                            y = int(index_finger.y * screen_h)
                            pyautogui.moveTo(x, y)


                    # Open palm → swipe navigation
                    if sum(fingers) == 5:
                        wrist_x = hand_landmarks.landmark[0].x
                        wrist_y = hand_landmarks.landmark[0].y

                        if prev_x is not None:
                            dx = wrist_x - prev_x
                            dy = wrist_y - prev_y
                            threshold = 0.05

                            now_action = time.time()
                            if now_action - last_action_time > cooldown:
                                if abs(dx) < 0.01 and abs(dy) < 0.01:   # almost no movement
                                    gesture_text = "Pause/Play"
                                    play_pause()
                                    last_action_time = now_action
                                elif abs(dx) > abs(dy):  # Horizontal swipe
                                    if dx < -threshold:
                                        gesture_text = "Moved Left"
                                        pyautogui.hotkey('ctrl', 'right')
                                        last_action_time = now_action
                                    elif dx > threshold:
                                        gesture_text = "Moved Right"
                                        pyautogui.hotkey('ctrl', 'left')
                                        last_action_time = now_action
                                else:  # Vertical swipe
                                    if dy < -threshold:   # Swipe up
                                        gesture_text = "Mission Control On"
                                        mission_control_on()  # open Mission Control
                                        last_action_time = now_action
                                    elif dy > threshold:  # Swipe down
                                        gesture_text = "Mission Control Off"
                                        pyautogui.press('esc')           # close Mission Control
                                        last_action_time = now_action

                        prev_x, prev_y = wrist_x, wrist_y



                    # Two fingers → volume control
                    elif sum(fingers) == 2:
                        wrist_y = hand_landmarks.landmark[0].y
                        if prev_y is not None:
                            dy = wrist_y - prev_y
                            threshold = 0.05

                            if dy < -threshold:
                                gesture_text = "Volume Up"
                                volume_up()
                            elif dy > threshold:
                                gesture_text = "Volume Down"
                                volume_down()

                        prev_y = wrist_y
                    else:
                        prev_x, prev_y = None, None

                    # Debug finger bits
                    bits = "".join("1" if f else "0" for f in fingers)
                    cv2.putText(frame, f"{label} {bits}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            # FPS
            now = time.time()
            fps = int(1 / (now - prev_time)) if prev_time else 0
            prev_time = now
            cv2.putText(frame, f"FPS: {fps}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Gesture name
            if gesture_text:
                cv2.putText(frame, gesture_text, (w//2 - 100, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255), 3)

            cv2.imshow("Hand Gesture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
