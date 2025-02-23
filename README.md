# 🚴 Object Detection
### Helmet detection and two or more occupants detection by 효영
**헬멧 감지와 2인 이상 탑승 감지**
</br><br>

## 📚 프로젝트 소개
헬멧 착용 감지와 2인 이상 탑승하였다는 것을 감지하기 위해 YOLOv8 모델을 이용하여 학습을 진행하였다.<br>
헬멧 착용을 감지하기 위해서 roboflow에서 제공하는 데이터셋과 추가로 라벨링한 데이터셋을 합하여 약 3600개의 데이터셋을 사용하였습니다.<br>
헬멧 착용 감지 클래스는 With_helmet과 Without_helmet으로 나누어 학습을 진행하였습니다.<br> 
또한, 추가로 각 클래스의 감지율 조건 추가하여 정확도를 높였다.<br><br>
2인 이상 탑승을 감지하기 위해 yolov8n의 기본 데이터셋을 사용하였고, 사람이 2명 이상 감지되었을 때 결과를 출력할 있도록 하였습니다.<br>
헬멧을 착용하지 않았을 경우, 사람이 2명 이상 탑승한 것이 감지되었을 경우, 이미지로 캡쳐가 진행되고 감지된 시간, 위치, 감지 이유가 벌점에 기록되게 됩니다.<br><br>

## 🖥️ AI 모델 및 사용 데이터셋
**AI 모델 : YOLOv8 사용**

### Helmet detection Dataset
**class : with_helmet, without_helmet**<br>

사용한 데이터셋 : **Roboflow 제공 여러 데이터셋과 직접 라벨링한 데이터 [3595개]** </br>
데이터셋 - https://app.roboflow.com/university-q1syp/helmet_detection2_final/models

### Two or More occupants detection Dataset
사용한 데이터셋 : **yolov8n** (기본 데이터셋 사용)
<br><br>

## 최종 학습 데이터 정밀도, 재현율 및 mAP 수치 
### 1️⃣ "With Helmet" (헬멧을 착용한 경우)
| 지표        | 값          |
|------------|------------|
| Precision  | **0.988 (98.8%)** ✅ |
| Recall     | **0.977 (97.7%)** ✅ |
| mAP50      | **0.993 (99.3%)** 🔥 |
| mAP50-95   | **0.896 (89.6%)** 🎯 |

Precision : 75.1%에서 98.8%로 높임 <br>
Recall : 63.1%에서 97.7%로 높임  <br>
*비교한 이전 학습 : epochs값 20, 데이터셋 약1000개 <br><br>

### 2️⃣ "Without Helmet" (헬멧을 착용하지 않은 경우)
| 지표        | 값          |
|------------|------------|
| Precision  | **0.967 (96.7%)** ✅ |
| Recall     | **0.947 (94.7%)** ✅ |
| mAP50      | **0.986 (98.6%)** 🔥 |
| mAP50-95   | **0.839 (83.9%)** 🎯 |

Precision : 73.1%에서 96.7%로 높임 <br>
Recall : 30.1%에서 94.7%로 높임  <br>
*mAP50-95은 16.2%로 거의 맞추지 못하였지만, 최종 학습 데이터 테스트는 83.9%의 결과를 보임.
<br>
