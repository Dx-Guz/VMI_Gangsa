import cv2
import mediapipe as mp
import numpy as np
import os
import pygame
import time
import threading

# =========================
# CONFIG
# =========================
NUM_BILAH = 10
GAP = 8
ALPHA = 0.4
PLAY_AREA_RATIO = 0.85
SEQUENCE_LENGTH = 15
MAX_MISSING = 5
COOLDOWN = 0.15

# =========================
# LAST VALID FRAME
# =========================
last_right_hand = [0] * 63
last_left_hand = [0] * 63
missing_frames_right = 0
missing_frames_left = 0

last_trigger_time = [0] * NUM_BILAH
prev_right_zone = -1

# =========================
# NORMALIZATION
# =========================
def normalize_hand_landmarks(hand_landmarks):
    data = []
    base = hand_landmarks.landmark[0]
    base_x, base_y, base_z = base.x, base.y, base.z
    ref = hand_landmarks.landmark[12]
    scale = np.sqrt(
        (ref.x - base_x) ** 2 +
        (ref.y - base_y) ** 2 +
        (ref.z - base_z) ** 2
    )
    if scale == 0:
        scale = 1
    for lm in hand_landmarks.landmark:
        data.extend([
            (lm.x - base_x) / scale,
            (lm.y - base_y) / scale,
            (lm.z - base_z) / scale
        ])
    return data

# =========================
# ASYNC SAVE
# =========================
def save_sequence(path, data):
    np.save(path, data)

# =========================
# AUDIO
# =========================
pygame.mixer.init()
pygame.mixer.set_num_channels(NUM_BILAH)

sounds = [pygame.mixer.Sound(f'sound/bilah{i}.wav') for i in range(NUM_BILAH)]
channels = [pygame.mixer.Channel(i) for i in range(NUM_BILAH)]

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.85
)

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

# =========================
# DATASET
# =========================
actions = ['pukul', 'redam', 'netral']
for action in actions:
    os.makedirs(f'dataset/{action}', exist_ok=True)

recording = False
current_action = None
sequence_data = []
sequence_count = 0

# =========================
# CACHE
# =========================
cached_size = None
bilah_coords = []

prev_time = time.perf_counter()

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    now = time.perf_counter()
    fps = 1 / (now - prev_time)
    prev_time = now

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    if cached_size != (width, height):
        cached_size = (width, height)

        play_area_width = int(width * PLAY_AREA_RATIO)
        start_x = (width - play_area_width) // 2

        bilah_area_width = play_area_width - (GAP * (NUM_BILAH + 1))
        bilah_width = bilah_area_width // NUM_BILAH

        y1 = int(height * 0.35)
        y2 = int(height * 0.65)
        strike_zone_y2 = int(y1 + (y2 - y1) * 0.5)
        damp_zone_y1 = strike_zone_y2

        bilah_coords = []
        for i in range(NUM_BILAH):
            x1 = start_x + GAP + i * (bilah_width + GAP)
            x2 = x1 + bilah_width
            bilah_coords.append((x1, x2))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    right_hand_data = [0]*63
    left_hand_data = [0]*63
    right_detected = False
    left_detected = False
    right_index = None
    left_index = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            score = handedness.classification[0].score
            if score < 0.65:
                continue

            label = handedness.classification[0].label
            single_hand = normalize_hand_landmarks(hand_landmarks)

            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * width)
            y = int(index_tip.y * height)

            if label == "Right":
                right_detected = True
                right_hand_data = single_hand
                right_index = (x, y)
                color = (0,255,0)
                text = "RIGHT"
            else:
                left_detected = True
                left_hand_data = single_hand
                left_index = (x, y)
                color = (255,0,0)
                text = "LEFT"

            if not recording:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            cv2.putText(frame,text,(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    if right_detected:
        last_right_hand = right_hand_data
        missing_frames_right = 0
    else:
        missing_frames_right += 1
        if missing_frames_right < MAX_MISSING:
            right_hand_data = last_right_hand

    if left_detected:
        last_left_hand = left_hand_data
        missing_frames_left = 0
    else:
        missing_frames_left += 1
        if missing_frames_left < MAX_MISSING:
            left_hand_data = last_left_hand

    final_data = right_hand_data + left_hand_data

    for i,(x1,x2) in enumerate(bilah_coords):
        # outline only agar landmark tidak tertutup UI
        cv2.rectangle(frame,(x1,y1),(x2,y2),(50,50,50),2)
        cv2.line(frame,(x1,strike_zone_y2),(x2,strike_zone_y2),(100,100,255),2)

    current_right_zone = -1

    if right_index:
        x_r,y_r = right_index
        for i,(x1,x2) in enumerate(bilah_coords):
            if x1 < x_r < x2 and y1 < y_r < strike_zone_y2:
                current_right_zone = i
                break

    if (
        current_right_zone != -1 and
        current_right_zone != prev_right_zone and
        now - last_trigger_time[current_right_zone] > COOLDOWN
    ):
        channels[current_right_zone].play(sounds[current_right_zone])
        last_trigger_time[current_right_zone] = now

    prev_right_zone = current_right_zone

    if left_index:
        x_l,y_l = left_index
        for i,(x1,x2) in enumerate(bilah_coords):
            if x1 < x_l < x2 and damp_zone_y1 < y_l < y2:
                channels[i].fadeout(100)

    if recording:
        pos_feature = [0,0] if not right_index else [right_index[0]/width,right_index[1]/height]
        data = final_data + pos_feature
        sequence_data.append(data)

        if len(sequence_data) == SEQUENCE_LENGTH:
            path = f'dataset/{current_action}/{sequence_count}.npy'
            threading.Thread(
                target=save_sequence,
                args=(path,sequence_data.copy()),
                daemon=True
            ).start()
            sequence_count += 1
            sequence_data = []

    cv2.putText(frame,f'FPS: {int(fps)}',(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
    cv2.putText(frame,f'Action: {current_action}',(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    if recording:
        cv2.putText(frame,'● RECORDING',(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    cv2.imshow('Gangsa Virtual Instrument',frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        recording=True
        current_action='pukul'
        sequence_count=len(os.listdir('dataset/pukul'))

    elif key == ord('2'):
        recording=True
        current_action='redam'
        sequence_count=len(os.listdir('dataset/redam'))

    elif key == ord('3'):
        recording=True
        current_action='netral'
        sequence_count=len(os.listdir('dataset/netral'))

    elif key == ord('s'):
        recording=False
        current_action=None
        sequence_data=[]

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
