# 🚴 Detection
### Helmet detection and two or more occupants detection
#### 헬멧 감지와 2인 이상 탑승 감지
</br>

## 🖥️ AI 모델 및 사용 데이터셋
> AI 모델<br>
> **YOLOv8** 사용
</br></br>
Helmet detection Dataset : **Roboflow 제공 여러 데이터셋과 직접 라벨링한 데이터 [3595개]** </br>
https://app.roboflow.com/university-q1syp/helmet_detection2_final/models </br></br>
Two or More occupants detection : **yolov8n** (기본 데이터셋 사용)
<br>

## Helmet detection Dataset
**class : with_helmet, without_helmet**
<br><br>

### 최종 학습 데이터 정밀도, 재현율 및 mAP 수치 
### 1️⃣ "With Helmet" (헬멧을 착용한 경우)
| 지표        | 값          |
|------------|------------|
| Precision  | **0.988 (98.8%)** ✅ |
| Recall     | **0.977 (97.7%)** ✅ |
| mAP50      | **0.993 (99.3%)** 🔥 |
| mAP50-95   | **0.896 (89.6%)** 🎯 |

<br>
Precision : 75.1%에서 98.8%로 높임 <br>
Recall : 63.1%에서 97.7%로 높임  <br>
*비교한 이전 학습 : epochs값 20, 데이터셋 약1000개 <br>

### 2️⃣ "Without Helmet" (헬멧을 착용하지 않은 경우)
| 지표        | 값          |
|------------|------------|
| Precision  | **0.967 (96.7%)** ✅ |
| Recall     | **0.947 (94.7%)** ✅ |
| mAP50      | **0.986 (98.6%)** 🔥 |
| mAP50-95   | **0.839 (83.9%)** 🎯 |

<br>
Precision : 73.1%에서 96.7%로 높임 <br>
Recall : 30.1%에서 94.7%로 높임  <br>
*mAP50-95은 16.2%로 거의 맞추지 못하였지만, 최종 학습 데이터 테스트는 83.9%의 결과를 보임.
<br>
