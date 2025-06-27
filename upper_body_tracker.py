import os
import cv2
from mediapipe import solutions
import time
import xml.etree.ElementTree as ET
from collections import deque
import numpy as np

# TensorFlowの警告メッセージを抑制する
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_value(value):
    """値を適切な型に変換する"""
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value

def load_config(config_file):
    """設定ファイルの読み込み"""
    tree = ET.parse(config_file)
    root = tree.getroot()
    config = {}
    for elem in root:
        config[elem.tag] = parse_value(elem.text)
    return config

def calculate_distance_3d(point1, point2):
    """3D空間での2点間の距離を計算"""
    dist = ((point1.x - point2.x) ** 2 + 
            (point1.y - point2.y) ** 2 + 
            (point1.z - point2.z) ** 2) ** 0.5
    return dist

def calculate_angle(point1, point2, point3):
    """3点から角度を計算（中央の点を頂点とする角度）"""
    # ベクトルを計算
    v1 = np.array([point1.x - point2.x, point1.y - point2.y, point1.z - point2.z])
    v2 = np.array([point3.x - point2.x, point3.y - point2.y, point3.z - point2.z])
    
    # 内積を計算
    dot_product = np.dot(v1, v2)
    
    # ベクトルの大きさを計算
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # 角度を計算（ラジアンから度に変換）
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 数値誤差対策
    angle = np.arccos(cos_angle) * 180 / np.pi
    
    return angle

def draw_pose_landmarks(image, pose_landmarks):
    """ポーズのランドマークを描画"""
    mp_drawing.draw_landmarks(
        image,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

def analyze_upper_body_gestures(pose_landmarks):
    """上半身のジェスチャーを分析"""
    gestures = {}
    
    # 肩の高さを比較
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder_height_diff = abs(right_shoulder.y - left_shoulder.y)
    
    # 肩の傾きを判定
    if shoulder_height_diff > 0.05:  # しきい値は調整可能
        if right_shoulder.y < left_shoulder.y:
            gestures['shoulder_tilt'] = 'right_up'
        else:
            gestures['shoulder_tilt'] = 'left_up'
    else:
        gestures['shoulder_tilt'] = 'level'
    
    # 腕の角度を計算
    try:
        # 右腕の角度（肩-肘-手首）
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        gestures['right_arm_angle'] = right_arm_angle
        
        # 左腕の角度（肩-肘-手首）
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        gestures['left_arm_angle'] = left_arm_angle
        
        # 腕の上げ下げを判定
        if right_shoulder.y - right_wrist.y > 0.1:  # 右手が肩より上
            gestures['right_arm'] = 'raised'
        else:
            gestures['right_arm'] = 'lowered'
            
        if left_shoulder.y - left_wrist.y > 0.1:  # 左手が肩より上
            gestures['left_arm'] = 'raised'
        else:
            gestures['left_arm'] = 'lowered'
            
    except Exception as e:
        print(f"腕の角度計算エラー: {e}")
    
    return gestures

def process_pose_frame(image, pose, config):
    """ポーズフレーム処理"""
    flipped_image = cv2.flip(image, 1)
    flipped_image.flags.writeable = False
    flipped_image = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)
    results = pose.process(flipped_image)

    flipped_image.flags.writeable = True
    flipped_image = cv2.cvtColor(flipped_image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # ポーズのランドマークを描画
        draw_pose_landmarks(flipped_image, results.pose_landmarks)
        
        # ジェスチャー分析
        gestures = analyze_upper_body_gestures(results.pose_landmarks)

        return flipped_image, gestures

    return flipped_image, None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'upper_body_config.xml')
    
    config = load_config(config_path)

    global mp_drawing, mp_drawing_styles, mp_pose
    mp_drawing = solutions.drawing_utils
    mp_drawing_styles = solutions.drawing_styles
    mp_pose = solutions.pose

    cap = cv2.VideoCapture(config['camera'])

    fps = 30
    wait_time = int(1000 / fps)

    with mp_pose.Pose(
            model_complexity=config.get('model_complexity', 1),
            enable_segmentation=config.get('enable_segmentation', False),
            min_detection_confidence=config.get('min_detection_confidence', 0.5),
            min_tracking_confidence=config.get('min_tracking_confidence', 0.5)) as pose:
        
        while cap.isOpened():
            start_time = time.time()
            success, image = cap.read()
            if not success:
                print("カメラフレームが空です。")
                continue

            flipped_image, gestures = process_pose_frame(image, pose, config)

            cv2.imshow('UpperBodyTracker', flipped_image)

            elapsed_time = time.time() - start_time
            remaining_time = wait_time - int(elapsed_time * 1000)
            cv2.waitKey(max(1, remaining_time))

            if cv2.waitKey(5) & 0xFF == 27:  # ESCキーで終了
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
