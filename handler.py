import os
import runpod
from LLMManager import ResumeJobEvaluator

# 환경 변수에서 설정값 로드
# 어디가 문제?
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "ninky0/rezoom-llama3.1-8b-4bit-b16-merged")
CPU_ONLY = os.getenv("CPU_ONLY", "False").lower() == "true"
print("HF_TOKEN:", HF_TOKEN)
print("MODEL_ID:", MODEL_ID)
print("CPU_ONLY:", CPU_ONLY)

# NLLBManager 인스턴스 초기화
manager = ResumeJobEvaluator(model_id=MODEL_ID, hf_token=HF_TOKEN, cpu_only=CPU_ONLY)
def handler(job):
    try:
        # 요청에서 이력서와 채용공고 텍스트 가져오기
        input_data = job["input"]
        resume_text = input_data["resume"]
        job_text = input_data["jobpost"]
        
        # 서비스 레벨에서 모델에 요청을 보내 평가 결과 받기
        evaluator = ResumeJobEvaluator(model_id="model_id", hf_token="hf_token")
        result = evaluator.invoke(resume_text, job_text)
        
        # 결과를 응답으로 반환
        return {"evaluation_result": result}

    except Exception as e:
        return {"error": str(e)}

# 서버리스 핸들러 시작
runpod.serverless.start({"handler": handler})