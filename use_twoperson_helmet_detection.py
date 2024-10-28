from ultralytics import YOLO
import pandas as pd
import random
import cv2
import numpy as np

# YOLO 모델 로드
model_helmet = YOLO('C:\\Users\\SAMSUNG\\Desktop\\detection\\Detection\\best_helmet.pt')
model_people = YOLO('C:/Users/SAMSUNG/Desktop/detection/Detection/yolov8n.pt')

# 클래스 이름 설정
class_names = ['With Helmet', 'Without Helmet']

# 전역 변수 초기화
detection_results = {
    "person_count": 0,
    "helmet_name": "",
    "helmet_probability": 0
}

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2)
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=1.0, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            1.0,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

def process_frame(frame):
    global detection_results
    results_people = model_people(frame)
    
    results_df = pd.DataFrame(results_people[0].boxes.data.cpu().numpy(), columns=["x1", "y1", "x2", "y2", "conf", "class"])
    people = results_df[results_df["class"] == 0]
    person_count = len(people)

    for _, person in people.iterrows():
        x1, y1, x2, y2, conf, cls = person
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, 'Person', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    detection_results["person_count"] = person_count

    results_helmet = model_helmet(frame)
    for result in results_helmet:
        if result.boxes is not None:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                box = box.cpu().numpy()
                conf = conf.cpu().numpy()
                cls = int(cls.cpu().numpy())
                if cls == 0 and conf >= 0.8:
                    label = f'{class_names[cls]} {conf:.2f}'
                    plot_one_box(box, frame, label=label, color=(255, 0, 0), line_thickness=2)
                elif cls == 1 and conf >= 0.5:
                    label = f'{class_names[cls]} {conf:.2f}'
                    plot_one_box(box, frame, label=label, color=(0, 0, 255), line_thickness=2)
                detection_results["helmet_name"] = f'{class_names[cls]}'
                detection_results["helmet_probability"] = f'{conf:.2f}'

    return frame, person_count

def run_real_time_detection():
    cap = cv2.VideoCapture(0)  # 기본 웹캠을 사용하여 실시간 감지 시작

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 처리
        annotated_frame, person_count = process_frame(frame)

        # 화면에 표시
        cv2.imshow("Real-Time Detection", annotated_frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_real_time_detection()
