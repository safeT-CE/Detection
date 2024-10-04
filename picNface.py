import os
import sys
import pandas as pd
from botocore.exceptions import NoCredentialsError
import requests
import cv2
import face_recognition
import time
import uuid

# s3.py 모듈을 임포트
from s3 import s3_connection, upload_to_s3
import secret_key

# AWS S3 구성 설정
AWS_ACCESS_KEY = secret_key.AWS_ACCESS_KEY # AWS 액세스 키
AWS_SECRET_KEY = secret_key.AWS_SECRET_KEY  # AWS 비밀 키
S3_BUCKET_NAME = secret_key.S3_BUCKET_NAME  # S3 버킷 이름
S3_REGION_NAME = secret_key.S3_REGION_NAME   # S3 버킷 버전

def rotate_image_left_90(image):
    # 이미지를 전치
    transposed_image = cv2.transpose(image)
    # 수평으로 플립
    rotated_image = cv2.flip(transposed_image, flipCode=0)
    return rotated_image


def rotate_image_right_90(image):
    # 이미지를 전치
    transposed_image = cv2.transpose(image)
    # 수평으로 플립
    rotated_image = cv2.flip(transposed_image, flipCode=1)
    return rotated_image

# 동일성 체크
def process_images(license_image_path, face_image_path):

    use_camera = False  # False면 img, True면 camera

    # 신분증 이미지
    licenseimg = face_recognition.load_image_file(license_image_path)
    licenseimg = cv2.cvtColor(licenseimg, cv2.COLOR_BGR2RGB)

    # 테스트 이미지
    imgTest = face_recognition.load_image_file(face_image_path)
    imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

    # 사진 재조정
    licenseimg = rotate_image_right_90(licenseimg)
    imgTest = rotate_image_left_90(imgTest)
    
    # 신분증 사진에서 얼굴을 찾지 못했을 때
    face_locations = face_recognition.face_locations(licenseimg)
    if len(face_locations) == 0:
        return "No face found in license image", "license no face"

    faceLoc = face_recognition.face_locations(licenseimg)[0]
    encodeRef = face_recognition.face_encodings(licenseimg)[0]

    cv2.rectangle(licenseimg, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

    # CSV 파일 존재 여부 확인 및 초기화 : 본인 경로로 재설정 필요
    csv_filename = 'C:/Users/SAMSUNG/Desktop/detection/Detection/face_rec_data/face_features.csv'
    if not os.path.exists(csv_filename):
        # CSV 파일 생성 및 헤더 추가
        df = pd.DataFrame(columns=[f'feature_{i}' for i in range(len(encodeRef))])
        df.to_csv(csv_filename, index=False)

    start_time = time.time()
    best_face_encoding = None
    best_min_distance = float('inf')
    best_match_text = None

    while True:
        rgb_frame = imgTest

        # 프레임에서 얼굴 인식 및 인코딩
        faceLocTest = face_recognition.face_locations(rgb_frame)
        encodeTest = face_recognition.face_encodings(rgb_frame, faceLocTest)

        # 테스트 이미지에서 얼굴을 찾지 못했을 때
        if len(encodeTest) == 0:
            return "No face found in test image", "face no face"

        for (top, right, bottom, left), face_encoding in zip(faceLocTest, encodeTest):
            # 기준 얼굴과 현재 프레임 얼굴 비교
            results = face_recognition.compare_faces([encodeRef], face_encoding)
            faceDis = face_recognition.face_distance([encodeRef], face_encoding)

            # 가장 작은 유사도 값 찾기
            if faceDis[0] < best_min_distance:
                best_min_distance = faceDis[0]
                best_face_encoding = face_encoding
                best_match_text = f'{results[0]} {round(best_min_distance, 2)}'

            # 얼굴 주위에 사각형 그리기
            cv2.rectangle(imgTest, (left, top), (right, bottom), (255, 0, 255), 2)

        # 결과와 유사성을 이미지 상단에 명시
        if best_match_text:
            cv2.putText(imgTest, best_match_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        # N초가 지나면 유사도 값 저장(현재는 10초로 설정)
        if time.time() - start_time >= 10:
            if best_face_encoding is not None:
                if best_min_distance <= 0.50:#유사도 N이하면 얼굴 데이터 csv로 저장(현재는 0.5로 설정)
                    print(f"Face recognized with distance: {best_min_distance}. Features saved.")
                    # CSV 파일에 특징값 저장
                    df = pd.read_csv(csv_filename)
                    new_row = pd.DataFrame([best_face_encoding], columns=df.columns)
                    if not new_row.isna().all().all():  # 새로운 행이 비어 있거나 모든 값이 NA가 아닌지 확인
                        df = pd.concat([df, new_row], ignore_index=True)
                        df.to_csv(csv_filename, index=False)
                    return csv_filename, "success";
                else:
                    print("동일인이 아니라고 판별되어 얼굴 데이터를 저장하지 않았습니다.")
                    return "Not the same person", "not same";
            break

        # 결과 보여주기
        cv2.imshow('Test Image', imgTest)
        cv2.imshow('Reference Face', licenseimg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 카메라 및 윈도우 종료
    return csv_filename, True



# AWS S3 업로드
def upload_to_s3(user_id, file_name, bucket, object_name=None):
    
    # AWS S3 클라이언트 생성
    s3_client = s3_connection(AWS_ACCESS_KEY, AWS_SECRET_KEY)  

    if s3_client is False :
        return "AWS S3 connection failed", "not connect"
    try:
        # 파일 업로드
        s3_client.upload_file(file_name, bucket, object_name or file_name)
        print(f"'{file_name}' has been uploaded to '{bucket}/{object_name}' successfully.")
        # 파일 URL 생성
        if object_name is None:
            object_name = file_name
        
        s3_url = f"https://{bucket}.s3.{S3_REGION_NAME}.amazonaws.com/{object_name}"
        print(f"File URL: {s3_url}")        
        return s3_url, "success"
    except FileNotFoundError:
        print(f"The file '{file_name}' was not found.")
        return "File not found", "not find"
    except NoCredentialsError:
        print("AWS credentials not available.")
        return "AWS credentials not available", "not credential"

def error_request(user_id, samePerson):
    flask_url = "http://localhost:5000/send-identity"
    final_data = {
        "userId" : user_id,
        "identity" : "",
        "samePerson" : samePerson
    }
    try:
        response = requests.post(flask_url, json=final_data)
        print("Flask 서버로 데이터 전송 완료:", response.status_code)
    except requests.RequestException as e:
        print("Flask 서버로의 요청 중 오류 발생:", e)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python modify_picNface.py <license_image_path> <face_image_path> <user_id>")
        sys.exit(1)

    license_image_path = sys.argv[1]
    face_image_path = sys.argv[2]
    user_id = sys.argv[3]    

    # 이미지 처리 및 CSV 파일 생성
    csv_file_path, process_success = process_images(license_image_path, face_image_path)

    if process_success != "success":
        print(f"이미지 처리 실패: {csv_file_path}")
        error_request(user_id, process_success)
        sys.exit(1)

    print(f"CSV 파일이 {csv_file_path}에 저장되었습니다.")

    # S3 버킷 이름 설정
    bucket_name = S3_BUCKET_NAME  # 자신의 S3 버킷 이름으로 대체

    # S3에 파일 업로드
    # 저장할 이름 형식 : face/userId_UUID
    random_uuid = uuid.uuid4()
    s3_object_name = f"face/user{user_id}_{random_uuid}.csv"
    s3_url, process_success = upload_to_s3(user_id, csv_file_path, bucket_name, s3_object_name)

    if process_success != "success":
        print(f"S3 업로드 실패: {s3_url}")
        error_request(user_id, process_success)
        sys.exit(1)

    # CSV 파일 삭제
    os.remove(csv_file_path)
    print(f"로컬 CSV 파일 '{csv_file_path}'이 삭제되었습니다.")

    flask_url = "http://localhost:5000/send-identity"
    final_data = {
        "userId" : user_id,
        "identity" : s3_url,
        "samePerson" : process_success
    }
    
    try:
        response = requests.post(flask_url, json=final_data)
        print("Flask 서버로 데이터 전송 완료:", response.status_code)
    except requests.RequestException as e:
        print("Flask 서버로의 요청 중 오류 발생:", e)