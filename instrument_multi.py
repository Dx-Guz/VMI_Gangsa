# instrument_main.py
import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import collections
import os
import keras

from model_multi import HandSplitter

# =========================
# CONFIG
# =========================
NUM_BILAH = 10
GAP = 8
PLAY_AREA_RATIO = 0.85
SEQUENCE_LENGTH = 15
FEATURE_DIM = 128
MAX_MISSING = 5
CONFIDENCE_THRESHOLD = 0.75
SMOOTHING_WINDOW = 5
PREDICT_INTERVAL = 3
MODEL_COMPLEXITY = 1

# 🔑 KUNCI STABILITAS
MIN_DWELL_FRAMES = 2        
ZONE_RESET_COOLDOWN = 0.2 
RINGING_TIMEOUT = 1.5         

MODEL_PATH = "models/gangsa_bilstm_multi.h5"
LABEL_MAP_PATH = "models/label_map.npy"
SOUND_DIR = "sound"

# =========================
# HELPER: NORMALIZATION
# =========================
def normalize_hand_landmarks(hand_landmarks):
    data = []
    base = hand_landmarks.landmark[0]
    bx, by, bz = base.x, base.y, base.z
    ref = hand_landmarks.landmark[12]
    scale = np.sqrt((ref.x-bx)**2 + (ref.y-by)**2 + (ref.z-bz)**2)
    scale = scale if scale > 0 else 1.0
    for lm in hand_landmarks.landmark:
        data.extend([(lm.x-bx)/scale, (lm.y-by)/scale, (lm.z-bz)/scale])
    return data

# =========================
# HELPER: EXTRACT FEATURES
# =========================
def extract_frame_features(results, width, height, last_right, last_left, miss_r, miss_l):
    right_hand_data = [0.0] * 63
    left_hand_data = [0.0] * 63
    right_detected = False
    left_detected = False
    right_index_tip = None
    left_index_tip = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].score < 0.65: continue
            label = handedness.classification[0].label
            single_hand = normalize_hand_landmarks(hand_landmarks)
            
            index_finger_tip = hand_landmarks.landmark[8]
            x = int(index_finger_tip.x * width)
            y = int(index_finger_tip.y * height)

            if label == "Right":
                right_detected = True
                right_hand_data = single_hand
                right_index_tip = (x, y)
            else:
                left_detected = True
                left_hand_data = single_hand
                left_index_tip = (x, y)

    if right_detected: 
        last_right, miss_r = right_hand_data, 0
    else:
        miss_r += 1
        if miss_r < MAX_MISSING: right_hand_data = last_right

    if left_detected: 
        last_left, miss_l = left_hand_data, 0
    else:
        miss_l += 1
        if miss_l < MAX_MISSING: left_hand_data = last_left

    pos_feature = [0.0, 0.0] if right_index_tip is None else [right_index_tip[0]/width, right_index_tip[1]/height]
    
    return (right_hand_data + left_hand_data + pos_feature, 
            last_right, last_left, miss_r, miss_l, 
            right_index_tip, left_index_tip)

# =========================
# SOUND LOADER
# =========================
def load_sounds_safe():
    sounds = []
    for i in range(NUM_BILAH):
        loaded = False
        for ext in [".ogg", ".wav"]:
            path = os.path.join(SOUND_DIR, f"bilah{i}{ext}")
            if os.path.exists(path):
                try:
                    sounds.append(pygame.mixer.Sound(path))
                    loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Gagal load {path}: {e}")
        if not loaded:
            sounds.append(None)
    return sounds

# =========================
# MAIN LOOP
# =========================
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model '{MODEL_PATH}' tidak ditemukan.")
    
    print("🧠 Loading model...")
    model = keras.models.load_model(MODEL_PATH, custom_objects={"HandSplitter": HandSplitter}, safe_mode=False)
    label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).tolist()
    print(f"✅ Model loaded: {label_map}")

    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    pygame.mixer.set_num_channels(NUM_BILAH)
    sounds = load_sounds_safe()
    channels = [pygame.mixer.Channel(i) for i in range(NUM_BILAH)]

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                          min_detection_confidence=0.65, min_tracking_confidence=0.75)

    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    for _ in range(5): cap.read()

    # --- STATE VARIABLES ---
    seq_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    pred_history_r = collections.deque(maxlen=SMOOTHING_WINDOW)
    pred_history_l = collections.deque(maxlen=SMOOTHING_WINDOW)
    last_right_hand, last_left_hand = [0.0]*63, [0.0]*63
    miss_r, miss_l = 0, 0
    
    action_r, conf_r = "netral", 0.0
    action_l, conf_l = "netral", 0.0
    
    strike_states_r = ["idle"] * NUM_BILAH
    zone_dwell_r = [0] * NUM_BILAH          
    zone_stable_since_r = [0.0] * NUM_BILAH 
    
    strike_states_l = ["idle"] * NUM_BILAH
    zone_dwell_l = [0] * NUM_BILAH          
    zone_stable_since_l = [0.0] * NUM_BILAH 
    
    # 🔑 Ringing States
    is_ringing = [False] * NUM_BILAH
    last_play_time = [0.0] * NUM_BILAH
    
    quick_redam_flash_time = [0.0] * NUM_BILAH
    redam_flash_time = [0.0] * NUM_BILAH  
    FLASH_DURATION = 0.3
    
    cached_size = None
    bilah_coords = []
    y1 = y2 = strike_zone_y2 = damp_zone_y1 = 0
    frame_count = 0
    prev_time = time.perf_counter()

    print("📷 Inference dimulai. Tekan 'Q' untuk keluar.")
    print("💡 TIP: Sentuh bilah yang bergetar dengan tangan kiri untuk Quick Redam instan.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_capture_time = time.perf_counter()

        now = time.perf_counter()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        frame_count += 1
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        if cached_size != (width, height):
            cached_size = (width, height)
            pw = int(width * PLAY_AREA_RATIO)
            sx = (width - pw) // 2
            bw = (pw - GAP * (NUM_BILAH + 1)) // NUM_BILAH
            y1, y2 = int(height * 0.35), int(height * 0.65)
            strike_zone_y2 = int(y1 + (y2 - y1) * 0.5)
            damp_zone_y1 = strike_zone_y2
            bilah_coords = [(sx + GAP + i * (bw + GAP), sx + GAP + (i + 1) * (bw + GAP)) for i in range(NUM_BILAH)]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        features, last_right_hand, last_left_hand, miss_r, miss_l, right_index_tip, left_index_tip = extract_frame_features(
            results, width, height, last_right_hand, last_left_hand, miss_r, miss_l
        )
        seq_buffer.append(features)

        # --- PREDICTION ---
        if len(seq_buffer) == SEQUENCE_LENGTH and frame_count % PREDICT_INTERVAL == 0:
            try:
                preds = model.predict(np.expand_dims(np.array(seq_buffer), axis=0), verbose=0)
                pr, pl = preds[0][0], preds[1][0]
                cr, cl = np.max(pr), np.max(pl)
                idx_r, idx_l = np.argmax(pr), np.argmax(pl)

                pred_history_r.append(idx_r)
                if len(pred_history_r) == SMOOTHING_WINDOW:
                    u, c = np.unique(pred_history_r, return_counts=True); idx_r = u[np.argmax(c)]
                
                pred_history_l.append(idx_l)
                if len(pred_history_l) == SMOOTHING_WINDOW:
                    u, c = np.unique(pred_history_l, return_counts=True); idx_l = u[np.argmax(c)]

                action_r = label_map[idx_r] if cr > CONFIDENCE_THRESHOLD else "netral"
                action_l = label_map[idx_l] if cl > CONFIDENCE_THRESHOLD else "netral"
                conf_r, conf_l = cr, cl
            except Exception as e:
                print(f"⚠️ Predict error: {e}")

        # ==========================================
        # 🔑 RIGHT HAND: DWELL + RINGING + TRIGGER
        # ==========================================
        current_right_zone = -1
        if right_index_tip:
            x_tip, y_tip = right_index_tip
            for i, (x1, x2) in enumerate(bilah_coords):
                if x1 < x_tip < x2 and y1 < y_tip < strike_zone_y2:
                    current_right_zone = i
                    break

        if action_r == "pukul" and current_right_zone != -1:
            zone_dwell_r[current_right_zone] += 1
        else:
            zone_dwell_r = [0] * NUM_BILAH

        # Auto-reset Ringing (timeout)
        for i in range(NUM_BILAH):
            if is_ringing[i] and (now - last_play_time[i]) > RINGING_TIMEOUT:
                is_ringing[i] = False

        # Trigger Logic
        if action_r == "pukul" and current_right_zone != -1:
            bilah_idx = current_right_zone
            if not is_ringing[bilah_idx] and zone_dwell_r[bilah_idx] >= MIN_DWELL_FRAMES and strike_states_r[bilah_idx] == "idle" and now - zone_stable_since_r[bilah_idx] > ZONE_RESET_COOLDOWN:
                trigger_time = time.perf_counter()
                software_latency_ms = (trigger_time - frame_capture_time) * 1000
                if sounds[bilah_idx]:
                    channels[bilah_idx].set_volume(1.0)
                    channels[bilah_idx].play(sounds[bilah_idx])
                    estimated_total_latency = software_latency_ms + 65.0 
                    print(f"PUKUL: bilah {bilah_idx+1} | Pipeline Latency: {software_latency_ms:.2f} ms | Est. Total Latency: ~{estimated_total_latency:.2f} ms")
                    is_ringing[bilah_idx] = True
                    last_play_time[bilah_idx] = now
                    strike_states_r[bilah_idx] = "triggered"
                    zone_stable_since_r[bilah_idx] = now
                    zone_dwell_r[bilah_idx] = 0

        if current_right_zone == -1 or action_r != "pukul":
            strike_states_r = ["idle"] * NUM_BILAH


        # ==========================================
        # 🔑 LEFT HAND: SMART INSTANT REDAM (TANPA DWELL)
        # ==========================================
        current_left_zone = -1
        
        if left_index_tip:
            x_tip, y_tip = left_index_tip
            for i, (x1, x2) in enumerate(bilah_coords):
                if x1 < x_tip < x2 and damp_zone_y1 < y_tip < y2:
                    current_left_zone = i
                    
                    # 1. PRIORITAS FISIKA: Jika bilah ringing, redam INSTAN (0 ms delay)
                    if is_ringing[i]:
                        is_ringing[i] = False
                        channels[i].fadeout(200)
                        quick_redam_flash_time[i] = now
                        print(f"⚡ QUICK REDAM: bilah {i+1}")
                        
                    # 2. NORMAL REDAM (SAFEGUARD AI): Jika bilah TIDAK ringing
                    elif action_l == "redam":
                        channels[i].fadeout(350)
                        redam_flash_time[i] = now
                        is_ringing[i] = False
                        strike_states_l[i] = "triggered"
                        
                    break # Keluar loop setelah menemukan bilah yang disentuh
        else:
            # Reset state visual jika tangan kiri tidak terdeteksi
            strike_states_l = ["idle"] * NUM_BILAH

        # Reset state visual jika tangan kiri keluar dari zona redam
        if current_left_zone == -1:
            strike_states_l = ["idle"] * NUM_BILAH

        # ==========================================
        # 🎯 UI RENDERING (MULTI-STATE VISUAL FEEDBACK)
        # ==========================================
        def draw_hand_label(pos, action, conf, color):
            if not pos or conf < CONFIDENCE_THRESHOLD: return
            x, y = pos
            text = f"{action.upper()} ({conf*100:.0f}%)"
            font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            (tw, th2), _ = cv2.getTextSize(text, font, sc, th)
            cv2.rectangle(frame, (x-10, y-th2-25), (x+tw+10, y-10), (0, 0, 0), -1)
            cv2.putText(frame, text, (x, y-14), font, sc, color, th)

        draw_hand_label(right_index_tip, action_r, conf_r, (0, 255, 0))      # Hijau
        draw_hand_label(left_index_tip, action_l, conf_l, (255, 100, 100))   # Merah muda
        
        # Overlay bilah dengan PRIORITAS WARNA
        for i, (x1, x2) in enumerate(bilah_coords):
            # 🔑 PRIORITAS WARNA (dari tertinggi ke terendah)
            if now - quick_redam_flash_time[i] < FLASH_DURATION:
                # 1. QUICK REDAM FLASH → Magenta (paling prioritas)
                col = (255, 0, 255)
                thickness = 3  # Tebalkan agar mencolok
            elif now - redam_flash_time[i] < FLASH_DURATION:
                # 2. REDAM NORMAL FLASH → Oranye
                col = (0, 165, 255)
                thickness = 3
            elif strike_states_r[i] == "triggered":
                # 3. TRIGGER PUKUL → Hijau
                col = (0, 255, 0)
                thickness = 2
            elif is_ringing[i]:
                # 4. RINGING BLOCK → Biru (terkunci)
                col = (255, 100, 100)
                thickness = 2
            elif zone_dwell_r[i] >= MIN_DWELL_FRAMES:
                # 5. DWELL PUKUL TERCAPAI → Kuning
                col = (0, 255, 255)
                thickness = 2
            else:
                # 6. IDLE → Abu-abu
                col = (50, 50, 50)
                thickness = 2
            
            # Gambar kotak bilah
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, thickness)
            
            # Garis zona (Strike Zone = biru tipis, Damp Zone = hijau tipis)
            cv2.line(frame, (x1, strike_zone_y2), (x2, strike_zone_y2), (255, 100, 100), 1)
            cv2.line(frame, (x1, damp_zone_y1), (x2, damp_zone_y1), (100, 255, 100), 1)
            
            # Nomor bilah
            cv2.putText(frame, str(i+1), ((x1+x2)//2 - 4, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # 🔑 FPS di KIRI ATAS
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow('Gangsa AI', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("👋 Instrument ditutup.")

if __name__ == "__main__":
    try: 
        main()
    except KeyboardInterrupt: 
        print("\n⏹️ Dihentikan user.")
    except Exception as e: 
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()