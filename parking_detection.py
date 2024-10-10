import cv2
from ultralytics import YOLO
import datetime
import os
from flask import Flask, jsonify

# Flask 애플리케이션 생성
app = Flask(__name__)

# YOLO 모델 로드
model = YOLO("C:\\Users\\SAMSUNG\\Desktop\\detection\\Detection\\parking_best.pt")

# 비디오 파일 경로
video_path = "C:/Parking Detection/test.mp4"

@app.route('/detect', methods=['POST'])
def detect():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "비디오 파일을 열 수 없습니다."}), 500

    frame_count = 0
    capture_interval = 100
    evidence_dir = 'evidence_photos'
    if not os.path.exists(evidence_dir):
        os.makedirs(evidence_dir)

    violations = []  # 위반 사항을 기록할 리스트

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 객체 탐지
        results = model(frame)

        # 탐지 결과 처리
        for result in results:
            boxes = result.boxes.xyxy.numpy()
            scores = result.boxes.conf.numpy()
            classes = result.boxes.cls.numpy()

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box

                if score >= 0.5:
                    violation_type = "Unknown"
                    if cls == 0:  # 점자 블록 클래스 ID
                        # braille_block_label = f'점자 블록 {score:.2f}'
                        violation_type = "점자 블록 위반"
                    elif cls == 1:  # 횡단보도 클래스 ID
                        # crosswalk_label = f'횡단보도 {score:.2f}'
                        violation_type = "횡단보도 위반"

                    # 위반 사항 기록
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    evidence_path = f"{evidence_dir}/frame_{frame_count}.jpg"
                    cv2.imwrite(evidence_path, frame)
                    latitude, longitude = 37.5665, 126.9780  # 임의의 값

                    violation = {
                        "timestamp": timestamp,
                        "violation_type": violation_type,
                        "score": score,
                        "evidence_path": evidence_path,
                        "latitude": latitude,
                        "longitude": longitude
                    }
                    violations.append(violation)
                    print(violation)

        # 결과 화면에 표시
        cv2.imshow('Crosswalk and Braille Block Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % capture_interval == 0:
            print(f'Processed {frame_count} frames')

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"violations": violations})  # 감지된 위반 사항 반환

if __name__ == '__main__':
    app.run(port=5000)  # Flask 서버 실행
