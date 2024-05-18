import os
import cv2
import mediapipe as mp
from mediapipe import solutions
import keyboard
import pyautogui
from pynput.mouse import Controller
import time
import xml.etree.ElementTree as ET

# TensorFlowの警告メッセージを抑制する
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_value(value):
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value

def load_config(config_file):
    tree = ET.parse(config_file)
    root = tree.getroot()
    config = {}
    for elem in root:
        config[elem.tag] = parse_value(elem.text)
    return config

def process_frame(image, hands, screen_width, screen_height, alpha, ema_x, ema_y, config):
    flipped_image = cv2.flip(image, 1)
    flipped_image.flags.writeable = False
    flipped_image = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)
    results = hands.process(flipped_image)

    flipped_image.flags.writeable = True
    flipped_image = cv2.cvtColor(flipped_image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            if (config['hand'] == 'right' and hand_label != 'Right') or (config['hand'] == 'left' and hand_label != 'Left'):
                continue

            mp_drawing.draw_landmarks(
                flipped_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            index_finger_tip = hand_landmarks.landmark[8]

            # 画像サイズより一回り小さい枠を作成
            frame_h, frame_w, _ = flipped_image.shape
            border_w = frame_w * config['window_scale']
            border_h = frame_h * config['window_scale']

            # 座標を小さい枠に当てはめる
            adjusted_x = (index_finger_tip.x * frame_w -
                          border_w) / (frame_w - 2 * border_w)
            adjusted_y = (index_finger_tip.y * frame_h -
                          border_h) / (frame_h - 2 * border_h)

            # スクリーン座標を計算
            screen_x = int(adjusted_x * screen_width)
            screen_y = int(adjusted_y * screen_height)
            print(f"x:{screen_x} y:{screen_y}")

            ema_x, ema_y = smooth_coordinates(
                screen_x, screen_y, ema_x, ema_y, alpha)
            if keyboard.is_pressed('ctrl'):
                move_mouse(ema_x, ema_y)

    return flipped_image, ema_x, ema_y

def smooth_coordinates(screen_x, screen_y, ema_x, ema_y, alpha):
    if ema_x is None:
        ema_x, ema_y = screen_x, screen_y
    else:
        ema_x = alpha * screen_x + (1 - alpha) * ema_x
        ema_y = alpha * screen_y + (1 - alpha) * ema_y
    return ema_x, ema_y

def move_mouse(screen_x, screen_y):
    mouse.position = (int(screen_x), int(screen_y))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.xml')
    config = load_config(config_path)

    global mp_drawing, mp_drawing_styles, mp_hands, mouse
    mp_drawing = solutions.drawing_utils
    mp_drawing_styles = solutions.drawing_styles
    mp_hands = solutions.hands

    cap = cv2.VideoCapture(config['camera'])
    screen_width, screen_height = pyautogui.size()

    mouse = Controller()

    alpha = 0.2
    ema_x, ema_y = None, None

    fps = 30
    wait_time = int(1000 / fps)

    # 枠のサイズ（例：0.1は画像の10%の枠を作成）
    window_scale = config['window_scale']

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            start_time = time.time()
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            flipped_image, ema_x, ema_y = process_frame(
                image, hands, screen_width, screen_height, alpha, ema_x, ema_y, config)
            cv2.imshow('FingerMouseCursor', flipped_image)

            elapsed_time = time.time() - start_time
            remaining_time = wait_time - int(elapsed_time * 1000)
            cv2.waitKey(max(1, remaining_time))

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
