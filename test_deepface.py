import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model

emotion_model = load_model("elderly_emotion_model.h5")

def detect_emotions_video(video_source=0):
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get the original video's FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        frame_delay = 33
    else:
        frame_delay = int(1000 / original_fps)

    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )
    
    frame_count = 0
    # 讓文字顯示更久，需要記錄上一次的情緒與時間
    last_emotion = None
    last_ts = 0
    display_duration = 2.0  # 文字至少顯示 2 秒
    last_x = last_y = last_w = last_h = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process every 3rd frame for better performance
        if frame_count % 3 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bbox.xmin * iw)
                    y = int(bbox.ymin * ih)
                    w = int(bbox.width * iw)
                    h = int(bbox.height * ih)
                    
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, iw - x)
                    h = min(h, ih - y)
                    last_x, last_y, last_w, last_h = x, y, w, h
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size == 0:
                        continue

                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    face_gray = cv2.resize(face_gray, (48, 48))
                    face_gray = face_gray / 255.0
                    face_gray = np.reshape(face_gray, (1, 48, 48, 1))

                    try:
                        
                        pred = emotion_model.predict(face_gray, verbose=0)
                        emotion_idx = np.argmax(pred)

                        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                        dominant_emotion = emotion_labels[emotion_idx]
                        confidence = pred[0][emotion_idx] * 100

                        # 記錄
                        label = f"{dominant_emotion}: {confidence:.1f}%"


                        last_emotion = (dominant_emotion, confidence)
                        last_ts = time.time()

                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        label_y = max(y - 10, 20)
                        
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
                        cv2.rectangle(frame,
                                    (x, label_y - text_height - 5),
                                    (x + text_width, label_y + 5),
                                    (0, 255, 0),
                                    -1)
                        
                        cv2.putText(frame, label,
                                   (x, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   1.6, (0, 0, 0), 4)
                        
                    except Exception as e:
                        print(f"Error analyzing emotion: {str(e)}")
        # After cap.open
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        scale = 0.25  # 讓 4K 變成 1024×540
        new_w = int(w * scale)
        new_h = int(h * scale)
        cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Emotion Detection", new_w, new_h)
        

        # 如果最近 N 秒內有偵測到情緒，保持顯示
        if last_emotion is not None and time.time() - last_ts < display_duration:
            emo_name, emo_score = last_emotion
            label = f"{emo_name}: {emo_score:.1f}%"
            
            # 使用最後一次 bounding box 畫位置
            # 前提：你需要記錄最後的 x, y, w, h
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)

            label_y = max(last_y - 10, 20)

            cv2.rectangle(frame,
                        (last_x, label_y - text_height - 5),
                        (last_x + text_width, label_y + 5),
                        (0, 255, 0), -1)

            cv2.putText(frame, label,
                        (last_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.6, (0, 0, 0), 4)
        cv2.imshow('Emotion Detection', frame)
        
        # Wait based on original video's FPS
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotions_video("v1.mp4")  # or 0 for webcam