import random
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import os
import requests
import json
import sys

# s3.py 모듈을 임포트
from s3 import s3_connection, upload_to_s3
import secret_key


def detect_helmet(user_id):

    # AWS S3 구성 설정
    AWS_ACCESS_KEY = secret_key.AWS_ACCESS_KEY # AWS 액세스 키
    AWS_SECRET_KEY = secret_key.AWS_SECRET_KEY  # AWS 비밀 키
    S3_BUCKET_NAME = secret_key.S3_BUCKET_NAME  # S3 버킷 이름
    # S3 연결 설정
    s3_client = s3_connection(AWS_ACCESS_KEY, AWS_SECRET_KEY)

    # 모델 로드 : best_last.pt 경로에 맞게 변경 필요
    model = YOLO('C:\\Users\\SAMSUNG\\Desktop\\detection\\Detection\\best_helmet.pt')

    ## 클래스 이름 설정
    class_names = ['With Helmet', 'Without Helmet']

    # 웹캠 열기 (0은 기본 웹캠, 다른 번호는 외부 웹캠)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        exit()

    # without_helmet 클래스 감지 시간을 기록하기 위한 변수
    detection_start_time = None  # 감지 시작 시간
    detection_time = 0  # 감지 끝 시간
    detected_class = None
    required_detection_time = 3  # 10초 감지 유지 시 기록
    stoped_detection_time = 10  # 100초 후 감지 후 쉬는 시간
    count = 0  # 감지 횟수
    record_done = False  # 첫 번째 감지가 기록되었는지 여부
    restart_time = 0

    # 캡쳐한 이미지를 저장하는 디렉토리 생성
    capture_dir = "penalty/"
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)

    # 경과 시간을 1초마다 기록
    last_printed_time = 0

    # 시간
    def get_current_datetime():
        return datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")

    ##without_helmet 확률 조절
    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        # 바운딩 박스 두께, 색 조절
        tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2)
        )  # line thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
            
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3   #text location
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                img,
                label,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        # 모델을 사용하여 감지
        results = model(frame)

        # 결과를 프레임에 그리기
        annotated_frame = frame.copy()

        current_time = time.time()

        target_class_detected = False  # 반복문 시작 시 매번 초기화

        for result in results:
            if result.boxes is not None:  # 결과에 바운딩 박스가 있는 경우에만 처리
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    box = box.cpu().numpy()
                    conf = conf.cpu().numpy()
                    cls = int(cls.cpu().numpy())
                    restart_time = time.time() - detection_time

                    if cls == 0 and conf >= 0.8:  # With Helmet의 확률이 0.8 이상인 경우
                        label = f'{class_names[cls]} {conf:.2f}'
                        plot_one_box(box, annotated_frame, label=label, color=(255, 0, 0), line_thickness=2)
                    if cls == 1 and conf >= 0.5 and int(stoped_detection_time - restart_time) < 0:  # Without Helmet의 확률이 0.5 미만인 경우 건너뜀
                        label = f'{class_names[cls]} {conf:.2f}'
                        plot_one_box(box, annotated_frame, label=label, color=(0, 0, 255), line_thickness=2)

                        # Without Helmet 클래스가 감지되면 시간 측정
                        if detection_start_time is None:
                            detection_start_time = time.time()
                            last_printed_time = detection_start_time
                            detected_class = class_names[cls]

                        else:
                            elapsed_time = time.time() - detection_start_time
                            if int(elapsed_time) > int(last_printed_time - detection_start_time): 
                                last_printed_time = time.time()
                                #print(f"{detected_class} 감지 시간: {int(elapsed_time)}초")
                            
                            if elapsed_time >= required_detection_time:
                                if not record_done:
                                    # 이미지 캡쳐
                                    file_name = f"user{user_id}_{current_time}.png"
                                    capture_filename = os.path.join(capture_dir, file_name)
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

                                    file_path = f"{capture_dir}{file_name}"
                                    os.remove(file_path)    # 로컬 캡쳐 사진 삭제
                                    record_done = True
                                    detection_start_time = None
                                count += 1
                                detection_time = time.time()

                        target_class_detected = True

        if not target_class_detected:
            detection_start_time = None;
                        
        # 프레임을 화면에 표시
        cv2.imshow('YOLOv8 Live Detection', annotated_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 예외 처리 필요 : aws s3에는 저장되었는데, flask 전송이 실패하였을 때 aws s3에 저장하지 않기
            print(f"감지 횟수 : {count}")    
            # 종료 시 `count` 값을 Flask 서버로 전송
            flask_url = "http://localhost:5000/send-detection"
            final_data = {
                "userId": user_id,
                "content": "헬멧 미착용",
                "photo": s3_url if s3_url else "N/A",
                "date": current_time_str,
                "map": {
                    "latitude": 37.3,               # 임시 위도, 경도
                    "longitude": 127.457
                },
                "detectionCount": count  # 전송할 데이터 추가
            }
            response = requests.post(flask_url, json=final_data)
            print("Flask 서버로 데이터 전송 완료:", response.status_code)
            break
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    if len(sys.argv) != 2:  # userId 받음
        print("사용자 ID를 제공해주세요")
        sys.exit(1)

    user_id = sys.argv[1]
    detect_helmet(user_id)  # test ex : python realtime_helmet_detection.py 2
