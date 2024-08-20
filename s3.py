import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def s3_connection(aws_access_key_id, aws_secret_access_key, region_name="ap-northeast-2"):
    try:
        # s3 클라이언트 생성
        s3 = boto3.client(
            service_name="s3",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        print("s3 bucket connected!")
        return s3
    except Exception as e:
        print("S3 연결 실패:", e)
        return None

def upload_to_s3(file_name, bucket, s3_client, object_name=None):
    try:
        response = s3_client.upload_file(file_name, bucket, object_name or file_name, ExtraArgs={'ContentType': 'image/png'})
        s3_url = f"https://{bucket}.s3.{s3_client.meta.region_name}.amazonaws.com/{object_name or file_name}"
        #print(f"S3에 이미지 업로드 성공: {s3_url}")
        return s3_url
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
        return None
    except NoCredentialsError:
        print("AWS 자격 증명을 찾을 수 없습니다.")
        return None
    except ClientError as e:
        print(f"S3 클라이언트 오류: {e}")
        return None
    except Exception as e:
        print(f"파일 업로드 실패: {str(e)}")
        return None
