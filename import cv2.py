import cv2
import mediapipe as mp
import numpy as np
import pygame
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame for audio playback
pygame.mixer.init()
pygame.mixer.music.load("C:/Users/ASUS/Desktop/b.mp3")  # Load your audio file
pygame.mixer.music.play(-1)  # Loop the audio infinitely

# Map value ranges function
def map_range(value, in_min, in_max, out_min, out_max):
    return max(min(out_min + (float(value - in_min) / float(in_max - in_min) * (out_max - out_min)), out_max), out_min)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    result = hands.process(rgb_frame)
    hand_data = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_data.append(hand_landmarks)

    if len(hand_data) == 2:
        # Volume Control (Hand 1)
        thumb_tip = hand_data[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_data[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        volume = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
        volume_value = int(map_range(volume, 0.0, 0.2, 0, 100))
        pygame.mixer.music.set_volume(volume_value / 100)  # Set volume (0.0 to 1.0)

        # Speed Control (Hand 2)
        thumb_tip = hand_data[1].landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_data[1].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        speed = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
        speed_value = map_range(speed, 0.0, 0.2, 0.5, 2.0)
        
        # Frequency Control (Distance between two hands)
        wrist_1 = hand_data[0].landmark[mp_hands.HandLandmark.WRIST]
        wrist_2 = hand_data[1].landmark[mp_hands.HandLandmark.WRIST]
        distance_between_hands = math.sqrt((wrist_1.x - wrist_2.x) ** 2 + (wrist_1.y - wrist_2.y) ** 2)
        frequency_value = int(map_range(distance_between_hands, 0.0, 0.5, 100, 1000))

        # Display values on screen
        cv2.putText(frame, f'Volume: {volume_value}%', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Speed: {speed_value:.2f}x', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Frequency: {frequency_value} Hz', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    else:
        cv2.putText(frame, "Show two hands to control parameters", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Hand Tracking Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
