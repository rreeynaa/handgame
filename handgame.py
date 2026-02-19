import cv2
import mediapipe as mp
import numpy as np
import random

# Webcam
cap = cv2.VideoCapture(0)

# Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Game variables
ball_x = random.randint(100, 500)
ball_y = 0
ball_radius = 20
ball_speed = 8
score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_x, finger_y = None, None

    # Hand tracking
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            tip = hand_landmarks.landmark[8]
            finger_x = int(tip.x * w)
            finger_y = int(tip.y * h)
            cv2.circle(frame, (finger_x, finger_y), 10, (0, 255, 0), -1)

    # Move ball
    ball_y += ball_speed

    # Collision detection
    if finger_x and finger_y:
        distance = np.sqrt((finger_x - ball_x)**2 + (finger_y - ball_y)**2)
        if distance < ball_radius + 10:
            score += 1
            ball_x = random.randint(50, w - 50)
            ball_y = 0

    # Reset if missed
    if ball_y > h:
        ball_x = random.randint(50, w - 50)
        ball_y = 0

    # Draw ball
    cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 0, 255), -1)

    # Score
    cv2.putText(frame, f"Score: {score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Hand Catch Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
