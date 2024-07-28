import cv2
from ultralytics import YOLO
import pandas as pd
import time
from datetime import datetime
import os

# YOLOv8 모델 로드
model = YOLO('C:\\ai_learning\\ultralytics\\yolov8n.pt')

# 시간 변수
detection_start_time = None
required_detection_time = 3 # 3초 시간 설정
capture_done = False  # 캡쳐가 완료되었는지 확인하는 플래그
detected_class = None

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 사람 클래스 ID (COCO 데이터셋에서 사람 클래스 ID는 0)
PERSON_CLASS_ID = 0
# 현재 날짜와 시간 기록
def get_current_datetime():
    return datetime.now().strftime("%Y년%m월%d일 %H시%M분")

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
    results_df = pd.DataFrame(results[0].boxes.data.cpu().numpy(), columns=["x1", "y1", "x2", "y2", "conf", "class"])

    # 사람만 필터링
    people = results_df[results_df["class"] == PERSON_CLASS_ID]
    # 사람 수 계산
    person_count = len(people)

    # 감지된 사람 표시
    for _, person in people.iterrows():
        x1, y1, x2, y2, conf, cls = person
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, 'Person', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 사람이 2명 이상 감지되었는지 확인
    if len(results) > 0 and results[0].boxes is not None and person_count >= 2 and not capture_done:
        if detection_start_time is None:
            detection_start_time = time.time()
            detected_class = "More than two people on board"
        else:
            elapsed_time = time.time() - detection_start_time
            if elapsed_time >= required_detection_time:
                capture_done = True  # 캡쳐 완료 플래그 설정
                
                # 이미지 캡쳐
                capture_filename = os.path.join(capture_dir, f"capture_{current_time}.png")
                current_time = get_current_datetime()
                cv2.imwrite(capture_filename, frame)
                
                # 현재 날짜와 시간, 위치 기록
                print(f"벌점 내역 : {detected_class}")
                print(f"감지된 날짜 : {current_time}")
                print(f"캡쳐된 이미지 경로: {capture_filename}")
                #print(f"위치: {location}")
                
                # 캡쳐된 이미지를 화면에 띄움
                #captured_image = cv2.imread(capture_filename)
                #cv2.imshow("Captured Image", captured_image)
                
                detection_start_time = None  # 조건 미충족 시 감지 시간 초기화
                capture_done = False  # 캡쳐 완료 플래그 설정
                # 루프 종료
                #break
    elif person_count < 2:
        detection_start_time = None  # 조건 미충족 시 감지 시간 초기화

    # 영상 출력
    cv2.imshow('YOLOv8 Live People Counting', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
