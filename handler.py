import os
os.environ["TRANSFORMERS_NO_BITSANDBYTES"] = "1"
import runpod
from LLMManager import ResumeJobEvaluator


HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "ninky0/rezoom-llama3.1-8b-4bit-b16-r64-merged")
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
        print(f"Resume Text: {resume_text}")
        print(f"Job Text: {job_text}")
        
        result = manager.invoke(resume_text, job_text)
        print(f"Evaluation Result: {result}")
        
        # 결과를 응답으로 반환
        return {"result": result}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})