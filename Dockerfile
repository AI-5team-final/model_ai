# Python 3.10 기반 이미지를 사용
FROM python:3.10-slim

# 시스템 패키지 업데이트 및 C 컴파일러 설치
RUN apt-get update && apt-get install -y build-essential

# CUDA 및 기타 GPU 의존성 설치 (CUDA가 필요한 경우)
# RUN apt-get install -y cuda-toolkit-11-2

# 필요한 Python 의존성 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# 앱 코드 복사
COPY . /app
WORKDIR /app

# 핸들러 실행
CMD ["python", "handler.py"]