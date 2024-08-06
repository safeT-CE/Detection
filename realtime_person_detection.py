import cv2
from ultralytics import YOLO
import pandas as pd
import time
from datetime import datetime
import os
import json
import requests

# s3.py 모듈을 임포트
from s3 import s3_connection, upload_to_s3
import secret_key

# AWS S3 구성 설정
AWS_ACCESS_KEY = secret_key.AWS_ACCESS_KEY # AWS 액세스 키
AWS_SECRET_KEY = secret_key.AWS_SECRET_KEY  # AWS 비밀 키
S3_BUCKET_NAME = secret_key.S3_BUCKET_NAME  # S3 버킷 이름
S3_REGION_NAME = secret_key.S3_REGION_NAME   # S3 버킷 버전

# S3 연결 설정
s3_client = s3_connection(AWS_ACCESS_KEY, AWS_SECRET_KEY)

# YOLOv8 모델 로드
model = YOLO('C:\\ai_learning\\ultralytics\\yolov8n.pt')

# 시간 변수
detection_start_time = None
detection_time = 0  # 감지 끝 시간
detected_class = None
required_detection_time = 3  # 10초 감지 유지 시 기록
stoped_detection_time = 10  # 100초 후 감지 후 쉬는 시간
first = True  # 처음 감지인지 확인하는 변수
count = 0  # 감지 횟수
record_done = False  # 첫 번째 감지가 기록되었는지 여부


# 웹캠 열기
cap = cv2.VideoCapture(0)

# 사람 클래스 ID (COCO 데이터셋에서 사람 클래스 ID는 0)
PERSON_CLASS_ID = 0 #

# 현재 날짜와 시간 기록
def get_current_datetime():
    return datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")

# 캡쳐한 이미지를 저장하는 디렉토리 생성
capture_dir = "output_img"
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 모델로 감지 수행
    results = model(frame)

    current_time = time.time()

    # 감지 결과를 pandas DataFrame으로 변환
    #
    results_df = pd.DataFrame(results[0].boxes.data.cpu().numpy(), columns=["x1", "y1", "x2", "y2", "conf", "class"])

    # 사람만 필터링
    people = results_df[results_df["class"] == PERSON_CLASS_ID] #
    # 사람 수 계산
    person_count = len(people)

    # 감지된 사람 표시
    for _, person in people.iterrows():
        x1, y1, x2, y2, conf, cls = person
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, 'Person', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    restart_time = time.time() - detection_time
    # 사람이 2명 이상 감지되었는지 확인
    if len(results) > 0 and results[0].boxes is not None and person_count >= 2 and int(stoped_detection_time - restart_time) < 0: #and not capture_done:
        if detection_start_time is None:
            detection_start_time = time.time()
            detected_class = "More than two people on board"
        else:
            elapsed_time = time.time() - detection_start_time
            if elapsed_time >= required_detection_time:
                if not record_done:
                    #capture_done = True  # 캡쳐 완료 플래그 설정
                    capture_filename = os.path.join(capture_dir, f"capture_{current_time}.png") # 이미지 캡쳐
                    current_time_str = get_current_datetime()
                    cv2.imwrite(capture_filename, frame)

                    if s3_client:
                        s3_url = upload_to_s3(capture_filename, S3_BUCKET_NAME, s3_client)
                        if s3_url:
                            print(f"S3에 이미지 업로드 성공 : {s3_url}")
                        else:
                            print("S3에 이미지 업로드 실패")
                    else:
                        print("S3 연결 오류로 이미지 업로드 실패")
                    record_done = True
                    detection_start_time = None
                count += 1         
                detection_time = time.time()
    elif person_count < 2:
        detection_start_time = None

    # 영상 출력
    cv2.imshow('YOLOv8 Live People Counting', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"감지 횟수 : {count}")    
        # 종료 시 `count` 값을 Flask 서버로 전송
        flask_url = "http://localhost:5000/send-detection"
        final_data = {
            "content": detected_class,
            "photo": s3_url if s3_url else "N/A",
            "date": current_time_str,
            "map": {
                "latitude": 37.12233,               # 임시 위도, 경도
                "longitude": 127.472
            },
            "detectionCount": count  # 전송할 데이터 추가
        }
        response = requests.post(flask_url, json=final_data)
        print("Flask 서버로 데이터 전송 완료:", response.status_code)
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
