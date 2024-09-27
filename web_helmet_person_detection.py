import random
import cv2
from ultralytics import YOLO
import pandas as pd

# YOLOv8 모델 로드 : 변경 필요한 부분
model_people = YOLO('C:\\Users\\SAMSUNG\\Desktop\\detection\\Detection\\yolov8n.pt')  # 사람 감지 모델 경로
model_helmet = YOLO('C:\\Users\\SAMSUNG\\Desktop\\detection\\Detection\\best_last.pt')  # 헬멧 감지 모델 경로

# 클래스 이름 설정
class_names = ['With Helmet', 'Without Helmet']

# 웹캠 열기 (0은 기본 웹캠)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # 이미지에 바운딩 박스 그리기
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2)  # 선 두께
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # 글꼴 두께
        t_size = cv2.getTextSize(label, 0, fontScale=1.0, thickness=tf)[0]  # 텍스트 크기를 고정 (1.0으로 변경)
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  # 텍스트 위치
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # 채워진 사각형
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            1.0,  # 텍스트 크기를 고정 (1.0으로 변경)
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

def process_frame(frame):
    # YOLOv8 모델로 사람 감지 수행
    results_people = model_people(frame)
    
    # 감지 결과를 pandas DataFrame으로 변환
    results_df = pd.DataFrame(results_people[0].boxes.data.cpu().numpy(), columns=["x1", "y1", "x2", "y2", "conf", "class"])
    
    # 사람만 필터링
    people = results_df[results_df["class"] == 0]  # COCO에서 사람 클래스 ID는 0
    person_count = len(people)

    # 감지된 사람 표시
    for _, person in people.iterrows():
        x1, y1, x2, y2, conf, cls = person
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, 'Person', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)  # 텍스트 크기를 1.0으로 변경

    return frame, person_count

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 사람 감지
    annotated_frame, person_count = process_frame(frame)

    # 헬멧 감지
    results_helmet = model_helmet(frame)

    # 결과를 프레임에 그리기
    for result in results_helmet:
        if result.boxes is not None:  # 결과에 바운딩 박스가 있는 경우에만 처리
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                box = box.cpu().numpy()
                conf = conf.cpu().numpy()
                cls = int(cls.cpu().numpy())

                # With Helmet의 확률이 0.8 이상인 경우
                if cls == 0 and conf >= 0.8:
                    label = f'{class_names[cls]} {conf:.2f}'
                    plot_one_box(box, annotated_frame, label=label, color=(255, 0, 0), line_thickness=2)

                # Without Helmet의 확률이 0.5 이상인 경우
                elif cls == 1 and conf >= 0.5:
                    label = f'{class_names[cls]} {conf:.2f}'
                    plot_one_box(box, annotated_frame, label=label, color=(0, 0, 255), line_thickness=2)

    # 결과 출력
    if person_count > 0:
        cv2.putText(annotated_frame, f'People detected: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)  # 텍스트 크기를 1.0으로 변경
    else:
        cv2.putText(annotated_frame, 'No people detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)  # 텍스트 크기를 1.0으로 변경

    # 프레임을 화면에 표시
    cv2.imshow('YOLOv8 Live Detection', annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
