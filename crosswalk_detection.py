import cv2
from ultralytics import YOLO
import datetime
import os
import requests
import sys

# s3.py 모듈을 임포트
from s3 import s3_connection, upload_to_s3
import secret_key

def detect_crosswalk(user_id):
    # AWS S3 구성 설정
    AWS_ACCESS_KEY = secret_key.AWS_ACCESS_KEY
    AWS_SECRET_KEY = secret_key.AWS_SECRET_KEY
    S3_BUCKET_NAME = secret_key.S3_BUCKET_NAME
    s3_client = s3_connection(AWS_ACCESS_KEY, AWS_SECRET_KEY)

    # 모델 로드
    model = YOLO("C:/Users/SAMSUNG/Desktop/detection/Detection/best_crosswalk.pt")

    # 동영상 파일 열기
    #cap = cv2.VideoCapture("C:/safeT/ai_integration/Crosswalk_Detection/test.mp4")

    
    frame_count = 0  # 프레임 카운터 초기화
    capture_interval = 100  # 증거 사진을 캡쳐할 프레임 간격
    evidence_dir = 'evidence_photos'

    # 증거 사진 디렉토리 생성
    if not os.path.exists(evidence_dir):
        os.makedirs(evidence_dir)

    violations = []  # 위반 사항 기록 리스트

    def record_violation(frame, frame_count, score):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        evidence_path = f"{evidence_dir}/frame_{frame_count}.jpg"
        cv2.imwrite(evidence_path, frame)
        latitude, longitude = 37.5665, 126.9780  # 임의의 위도와 경도 값

        violation = {
            "userId": user_id,  # 사용자 ID 추가
            "timestamp": timestamp,
            "violation_type": "세로 횡단보도 위반",
            "score": score,
            "evidence_path": evidence_path,
            "latitude": latitude,
            "longitude": longitude
        }

        # AWS S3에 이미지 업로드
        if s3_client:
            s3_url = upload_to_s3(evidence_path, S3_BUCKET_NAME, s3_client)
            if s3_url:
                print(f"S3에 이미지 업로드 성공 : {s3_url}")
                violation["s3_url"] = s3_url
            else:
                print("S3에 이미지 업로드 실패")
                violation["s3_url"] = "N/A"
        else:
            print("S3 연결 오류로 이미지 업로드 실패")
            violation["s3_url"] = "N/A"

        # Flask 서버로 위반 사항 전송
        flask_url = "http://localhost:5000/send-detection"
        response = requests.post(flask_url, json=violation)
        print("Flask 서버로 데이터 전송 완료:", response.status_code)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽는 도중 오류가 발생했습니다.")
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

                if score >= 0.5:  # 신뢰도가 0.5 이상일 때만 라벨 표시
                    if cls == 1:  # 가로 횡단보도
                        label = f'Width {score:.2f}'
                        print(f'가로 횡단보도: {label}')
                    elif cls == 0:  # 세로 횡단보도
                        label = f'Length {score:.2f}'
                        print(f'세로 횡단보도: {label}')

                        # 위반 사항 기록
                        record_violation(frame, frame_count, score)

        # 결과 화면에 표시
        cv2.imshow('Crosswalk Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 프레임 카운터 증가
        frame_count += 1
        if frame_count % capture_interval == 0:  # 매 100프레임마다 진행상황 출력 및 증거 사진 캡쳐
            print(f'Processed {frame_count} frames')

    cap.release()
    cv2.destroyAllWindows()

    print("Processing complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용자 ID를 제공해주세요")
        sys.exit(1)

    user_id = sys.argv[1]
    detect_crosswalk(user_id)