import random
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import os


# 모델 로드 : 지금까지 best_last가 가장 좋음.
model = YOLO('C:\\ai_learning\\ultralytics\\best_last.pt')

## 클래스 이름 설정
class_names = ['With Helmet', 'Without Helmet']

# 웹캠 열기 (0은 기본 웹캠, 다른 번호는 외부 웹캠)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# without_helmet 클래스 감지 시간을 기록하기 위한 변수
detection_start_time = None
detected_class = None
required_detection_time = 10  # 10초 설정
target_class_name = "Without Helmet"
capture_done = False  # 캡쳐가 완료되었는지 확인하는 플래그


# 캡쳐한 이미지를 저장하는 디렉토리 생성
capture_dir = "output_img"
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)

# 경과 시간을 1초마다 기록
last_printed_time = 0

# 시간
def get_current_datetime():
    return datetime.now().strftime("%Y년%m월%d일 %H시%M분")


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
                
                if cls == 0 and conf >= 0.8:  # With Helmet의 확률이 0.8 이상인 경우
                    label = f'{class_names[cls]} {conf:.2f}'
                    plot_one_box(box, annotated_frame, label=label, color=(255, 0, 0), line_thickness=2)
                if cls == 1 and conf > 0.5:  # Without Helmet의 확률이 0.5 미만인 경우 건너뜀
                    label = f'{class_names[cls]} {conf:.2f}'
                    plot_one_box(box, annotated_frame, label=label, color=(0, 0, 255), line_thickness=2)

                    # Without Helmet 클래스가 감지되면 시간 측정
                    if detection_start_time is None:
                        detection_start_time = time.time()
                        last_printed_time = detection_start_time
                        detected_class = class_names[cls]
                    
                    # 문제 : 생성되는 바운딩 박스가 모두 유지되어야 함.
                    else:
                        elapsed_time = time.time() - detection_start_time
                        if int(elapsed_time) > int(last_printed_time - detection_start_time):
                            last_printed_time = time.time()
                            #print(f"{detected_class} 감지 시간: {int(elapsed_time)}초")
                        
                        if elapsed_time >= required_detection_time:
                            # 이미지 캡쳐
                            capture_done = True  # 캡쳐 완료 플래그 설정
                            capture_filename = os.path.join(capture_dir, f"capture_{current_time}.png")
                            current_time = get_current_datetime()
                            cv2.imwrite(capture_filename, frame)

                            # db에 저장
                            detection_start_time = None
                            #print(f"{detected_class}이(가) {required_detection_time}초 이상 감지되어 DB에 기록되었습니다.")
                            print(f"벌점 내역 : {detected_class}")
                            print(f"감지된 날짜 : {current_time}")
                            print(f"캡쳐된 이미지 경로: {capture_filename}")

                    target_class_detected = True

    if not target_class_detected:
        detection_start_time = None; #None
                    
    # 프레임을 화면에 표시
    cv2.imshow('YOLOv8 Live Detection', annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# 자원 해제
cap.release()
cv2.destroyAllWindows()