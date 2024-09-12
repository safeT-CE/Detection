from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/send-identity', methods=['POST'])
def receive_detection():
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
