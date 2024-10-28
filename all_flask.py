# from flask import Flask, request, jsonify
# import requests
# import cv2
# import numpy as np
# import face_recognition
# import pandas as pd
# import os
# import time
# from s3 import s3_connection, download_from_s3
# import secret_key
# from io import BytesIO
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# def find_user_csv_in_s3(s3_client, bucket_name, user_id):
#     prefix = f'face/user{user_id}_'
#     response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

#     if 'Contents' in response:
#         for obj in response['Contents']:
#             if obj['Key'].endswith('.csv'):
#                 return obj['Key']
#     return None

# def rotate_image_left_90(image):
#     transposed_image = cv2.transpose(image)
#     rotated_image = cv2.flip(transposed_image, flipCode=0)
#     return rotated_image

# def detect_face(user_id, faceFile):
#     AWS_ACCESS_KEY = secret_key.AWS_ACCESS_KEY
#     AWS_SECRET_KEY = secret_key.AWS_SECRET_KEY
#     S3_BUCKET_NAME = secret_key.S3_BUCKET_NAME
#     s3_client = s3_connection(AWS_ACCESS_KEY, AWS_SECRET_KEY)

#     csv_filename = find_user_csv_in_s3(s3_client, S3_BUCKET_NAME, user_id)
#     if not csv_filename:
#         return jsonify({"error": f"No CSV file found for user {user_id}"}), 500
#     local_csv_path = os.path.join('temp', csv_filename.split('/')[-1])

#     if not os.path.exists('temp'):
#         os.makedirs('temp')

#     if download_from_s3(S3_BUCKET_NAME, csv_filename, local_csv_path, s3_client):
#         df = pd.read_csv(local_csv_path)
#     else:
#         return jsonify({"error": "Failed to download CSV file from S3"}), 500
#     saved_encodings = df.values

#     result_dir = 'face_rec_results'
#     if not os.path.exists(result_dir):
#         os.makedirs(result_dir)

#     try:
#         image_stream = BytesIO(faceFile.read())
#         imgTest = face_recognition.load_image_file(image_stream)
#         imgTest = rotate_image_left_90(imgTest)
#         imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
#     except Exception as e:
#         return jsonify({"error": f"Failed to load image: {str(e)}"}), 500
        
#     if imgTest is None:
#         return jsonify({"error": "Image file could not be loaded"}), 500
        
#     start_time = time.time()
#     min_distance = float('inf')
#     best_match_text = ""

#     while True:
#         rgb_frame = imgTest
#         faceLocTest = face_recognition.face_locations(rgb_frame)
#         encodeTest = face_recognition.face_encodings(rgb_frame, faceLocTest)

#         if not encodeTest:
#             print("No face found in the frame.")
#             best_match_text = "얼굴이 인식되지 않습니다."
#         else:
#             for (top, right, bottom, left), face_encoding in zip(faceLocTest, encodeTest):
#                 faceDis = face_recognition.face_distance(saved_encodings, face_encoding)
#                 current_min_distance = np.min(faceDis)
#                 print(f"Face distance: {current_min_distance}")

#                 if current_min_distance < min_distance:
#                     min_distance = current_min_distance
#                     if min_distance <= 0.6:
#                         best_match_text = "동일인입니다."
#                     else:
#                         best_match_text = "동일인이 아닙니다."

#                 cv2.rectangle(imgTest, (left, top), (right, bottom), (255, 0, 255), 2)

#         if time.time() - start_time >= 5:
#             break
#         cv2.imshow('Test Image', imgTest)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     print(f"최종 결과: {best_match_text} (유사도 거리: {min_distance})")

#     result_image_path = os.path.join(result_dir, 'result_image.jpg')
#     cv2.imwrite(result_image_path, cv2.cvtColor(imgTest, cv2.COLOR_RGB2BGR))

#     if os.path.exists(result_image_path):
#         try:
#             os.remove(result_image_path)
#             print(f"이미지 파일 {result_image_path} 삭제 성공")
#         except Exception as e:
#             print(f"이미지 파일 삭제 중 오류 발생: {str(e)}")
#     else:
#         print(f"이미지 파일 {result_image_path}이 존재하지 않습니다.")

#     if os.path.exists(local_csv_path):
#         try:
#             os.remove(local_csv_path)
#             print(f"CSV 파일 {local_csv_path} 삭제 성공")
#         except Exception as e:
#             print(f"CSV 파일 삭제 중 오류 발생: {str(e)}")
#     else:
#         print(f"CSV 파일 {local_csv_path}이 존재하지 않습니다.")
    
#     return {
#         "userId": user_id,
#         "result": best_match_text,
#         "distance": min_distance
#     }

# @app.route('/face-detection', methods=['POST'])
# def face_detection():
#     user_id = request.form.get('userId')
#     face_file = request.files.get('faceFile')
    
#     if not user_id:
#         return jsonify({"error": "User ID is required"}), 400
#     elif not face_file:
#         return jsonify({"error": "User faceFile is required"}), 400
    
#     result = detect_face(user_id, face_file)
#     return jsonify(result)

# @app.route('/send-detection', methods=['POST'])
# def receive_detection():
#     try:
#         data = request.json
#         user_id = data['userId']
#         print("Received data : ", data)

#         if not data or not isinstance(data, dict):
#             return jsonify({"status": "failure", "message": "Invalid data received"}), 400

#         spring_boot_url = "http://localhost:8080/penalty?userId={}".format(user_id)

#         try:
#             response = requests.post(spring_boot_url, json=data)
#             response.raise_for_status()
#             print("Data sent to Spring Boot server successfully:", response.status_code)
#             return jsonify({"status": "success", "data": data}), 200
#         except requests.exceptions.RequestException as e:
#             print("Error sending data to Spring Boot server:", e)
#             return jsonify({"status": "failure", "message": "Failed to send data to Spring Boot server"}), 500

#     except Exception as e:
#         print("An error occurred:", e)
#         return jsonify({"status": "failure", "message": "An internal error occurred"}), 500

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)


from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import face_recognition
import pandas as pd
import os
import time
from io import BytesIO
from flask_cors import CORS
import secret_key
from s3 import s3_connection, download_from_s3

app = Flask(__name__)
CORS(app)

def find_user_csv_in_s3(s3_client, bucket_name, user_id):
    prefix = f'face/user{user_id}_'
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('.csv'):
                return obj['Key']
    return None

def rotate_image_left_90(image):
    transposed_image = cv2.transpose(image)
    rotated_image = cv2.flip(transposed_image, flipCode=0)
    return rotated_image

def detect_face(user_id, faceFile):
    AWS_ACCESS_KEY = secret_key.AWS_ACCESS_KEY
    AWS_SECRET_KEY = secret_key.AWS_SECRET_KEY
    S3_BUCKET_NAME = secret_key.S3_BUCKET_NAME
    s3_client = s3_connection(AWS_ACCESS_KEY, AWS_SECRET_KEY)

    csv_filename = find_user_csv_in_s3(s3_client, S3_BUCKET_NAME, user_id)
    if not csv_filename:
        return jsonify({"error": f"No CSV file found for user {user_id}"}), 500
    local_csv_path = os.path.join('temp', csv_filename.split('/')[-1])

    if not os.path.exists('temp'):
        os.makedirs('temp')

    if download_from_s3(S3_BUCKET_NAME, csv_filename, local_csv_path, s3_client):
        df = pd.read_csv(local_csv_path)
    else:
        return jsonify({"error": "Failed to download CSV file from S3"}), 500
    saved_encodings = df.values

    result_dir = 'face_rec_results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    try:
        image_stream = BytesIO(faceFile.read())
        imgTest = face_recognition.load_image_file(image_stream)
        imgTest = rotate_image_left_90(imgTest)
        imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return jsonify({"error": f"Failed to load image: {str(e)}"}), 500
        
    if imgTest is None:
        return jsonify({"error": "Image file could not be loaded"}), 500
        
    start_time = time.time()
    min_distance = float('inf')
    best_match_text = ""

    while True:
        rgb_frame = imgTest
        faceLocTest = face_recognition.face_locations(rgb_frame)
        encodeTest = face_recognition.face_encodings(rgb_frame, faceLocTest)

        if not encodeTest:
            print("No face found in the frame.")
            best_match_text = "얼굴이 인식되지 않습니다."
        else:
            for (top, right, bottom, left), face_encoding in zip(faceLocTest, encodeTest):
                faceDis = face_recognition.face_distance(saved_encodings, face_encoding)
                current_min_distance = np.min(faceDis)
                print(f"Face distance: {current_min_distance}")

                if current_min_distance < min_distance:
                    min_distance = current_min_distance
                    if min_distance <= 0.6:
                        best_match_text = "동일인입니다."
                    else:
                        best_match_text = "동일인이 아닙니다."

                cv2.rectangle(imgTest, (left, top), (right, bottom), (255, 0, 255), 2)

        if time.time() - start_time >= 5:
            break
        cv2.imshow('Test Image', imgTest)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"최종 결과: {best_match_text} (유사도 거리: {min_distance})")

    result_image_path = os.path.join(result_dir, 'result_image.jpg')
    cv2.imwrite(result_image_path, cv2.cvtColor(imgTest, cv2.COLOR_RGB2BGR))

    if os.path.exists(result_image_path):
        try:
            os.remove(result_image_path)
            print(f"이미지 파일 {result_image_path} 삭제 성공")
        except Exception as e:
            print(f"이미지 파일 삭제 중 오류 발생: {str(e)}")
    else:
        print(f"이미지 파일 {result_image_path}이 존재하지 않습니다.")

    if os.path.exists(local_csv_path):
        try:
            os.remove(local_csv_path)
            print(f"CSV 파일 {local_csv_path} 삭제 성공")
        except Exception as e:
            print(f"CSV 파일 삭제 중 오류 발생: {str(e)}")
    else:
        print(f"CSV 파일 {local_csv_path}이 존재하지 않습니다.")
    
    return {
        "userId": user_id,
        "result": best_match_text,
        "distance": min_distance
    }

@app.route('/face-detection', methods=['POST'])
def face_detection():
    user_id = request.form.get('userId')
    face_file = request.files.get('faceFile')
    
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    elif not face_file:
        return jsonify({"error": "User faceFile is required"}), 400
    
    result = detect_face(user_id, face_file)
    return jsonify(result)

@app.route('/send-detection', methods=['POST'])
def send_detection():
    try:
        data = request.json
        user_id = data['userId']
        print("Received data : ", data)

        if not data or not isinstance(data, dict):
            return jsonify({"status": "failure", "message": "Invalid data received"}), 400

        spring_boot_url = "http://localhost:8080/penalty?userId={}".format(user_id)

        try:
            response = requests.post(spring_boot_url, json=data)
            response.raise_for_status()
            print("Data sent to Spring Boot server successfully:", response.status_code)
            return jsonify({"status": "success", "data": data}), 200
        except requests.exceptions.RequestException as e:
            print("Error sending data to Spring Boot server:", e)
            return jsonify({"status": "failure", "message": "Failed to send data to Spring Boot server"}), 500

    except Exception as e:
        print("An error occurred:", e)
        return jsonify({"status": "failure", "message": "An internal error occurred"}), 500

@app.route('/send-identity', methods=['POST'])
def receive_identity():
    try:
        # JSON 데이터 수신
        data = request.json
        user_id = data['userId']
        print("Received data : ", data)

        # 데이터가 None이거나 예상된 유형이 아닐 경우
        if not data or not isinstance(data, dict):
            return jsonify({"status": "failure", "message": "Invalid data received"}), 400

        # Spring boot로 데이터 전송
        spring_boot_url = "http://localhost:8080/face/upload/{}".format(user_id)

        try:
            response = requests.post(spring_boot_url, json=data)
            response.raise_for_status()  # 4XX, 5XX 오류 발생 시 예외 처리
            print("Data sent to Spring Boot server successfully:", response.status_code)
            return jsonify({"status": "success", "data": data}), 200
        except requests.exceptions.RequestException as e:
            print("Error sending data to Spring Boot server:", e)
            return jsonify({"status": "failure", "message": "Failed to send data to Spring Boot server"}), 500

    except Exception as e:
        # 예외 발생 시 오류 메시지 출력
        print("An error occurred:", e)
        return jsonify({"status": "failure", "message": "An internal error occurred"}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
