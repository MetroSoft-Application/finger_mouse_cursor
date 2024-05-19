import os
import cv2
import mediapipe as mp
from mediapipe import solutions
import keyboard
import pyautogui
from pynput.mouse import Controller, Button
import time
import xml.etree.ElementTree as ET
from collections import deque

# TensorFlowの警告メッセージを抑制する
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_value(value):
    """値を適切な型に変換する"""
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

def clamp(value, min_value, max_value):
    """値を指定された範囲内にクランプする"""
    return max(min_value, min(value, max_value))

def update_and_average_queues(x_queue, y_queue, x_value, y_value, max_length):
    """キューを更新し、移動平均を計算"""
    x_queue.append(x_value)
    y_queue.append(y_value)
    if len(x_queue) > max_length:
        x_queue.popleft()
    if len(y_queue) > max_length:
        y_queue.popleft()

    avg_x = sum(x_queue) / len(x_queue)
    avg_y = sum(y_queue) / len(y_queue)

    return avg_x, avg_y

def calculate_distance(point1, point2):
    """2点間の距離(ユークリッド距離)を計算"""
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2) ** 0.5

def process_frame(image, hands, screen_width, screen_height, config):
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

            selected_joint = config['selected_joint']
            joint = hand_landmarks.landmark[selected_joint]

            # 人差し指の先端と親指の先端を取得
            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # 画像サイズより一回り小さい枠を作成
            frame_h, frame_w, _ = flipped_image.shape
            border_w = frame_w * config['window_scale']
            border_h = frame_h * config['window_scale']

            # 座標を小さい枠に当てはめる
            adjusted_x = (joint.x * frame_w - border_w) / \
                (frame_w - 2 * border_w)
            adjusted_y = (joint.y * frame_h - border_h) / \
                (frame_h - 2 * border_h)

            # スクリーン座標を計算
            screen_x = int(adjusted_x * screen_width)
            screen_y = int(adjusted_y * screen_height)

            # クランプ処理を追加
            screen_x = clamp(screen_x, 0, screen_width)
            screen_y = clamp(screen_y, 0, screen_height)

            print(f"x:{screen_x} y:{screen_y}")

            # 人差し指と親指の先端の距離を計算
            distance = calculate_distance(index_finger_tip, thumb_tip)
            print(f"Distance: {distance}")

            return flipped_image, screen_x, screen_y, distance

    return flipped_image, None, None, None

def move_mouse(screen_x, screen_y):
    mouse.position = (int(screen_x), int(screen_y))

def click_mouse(distance, click_threshold):
    if distance < click_threshold:
        if not mouse.pressed:
            mouse.press(Button.left)
            mouse.pressed = True
    else:
        if mouse.pressed:
            mouse.release(Button.left)
            mouse.pressed = False

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
    mouse.pressed = False

    x_queue = deque()
    y_queue = deque()

    fps = 30
    wait_time = int(1000 / fps)

    click_threshold = config['click_threshold']  # 距離のしきい値を設定

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

            flipped_image, screen_x, screen_y, distance = process_frame(
                image, hands, screen_width, screen_height, config)

            if screen_x is not None and screen_y is not None:
                avg_x, avg_y = update_and_average_queues(
                    x_queue, y_queue, screen_x, screen_y, config['queue_length'])
                if keyboard.is_pressed('ctrl'):
                    move_mouse(avg_x, avg_y)
                click_mouse(distance, click_threshold)

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
