import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ResumeJobEvaluator:
    def __init__(self, model_id: str, hf_token: str, cpu_only: bool = False):
        self.model_id = model_id
        self.hf_token = hf_token
        self.cpu_only = cpu_only
        self.device = torch.device("cpu" if cpu_only or not torch.cuda.is_available() else "cuda")
        self.initialize()

    def initialize(self):
        # 모델과 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(self.device)

    def invoke(self, resume_text: str, job_description: str) -> str:
        try:
            # 이력서와 채용공고 텍스트를 비교하는 로직
            prompt = f"""당신은 채용공고와 이력서, 자기소개서를 평가하는 AI입니다
채용공고를 기준으로 이력서와 자기소개서를 평가하세요
등급이 같이 제시된 경우 이에 맞춰서 평가하세요
등급별 점수에 맞춰서 적절한 점수를 부여하세요
굉장히 엄격하고 냉정하게 평가하세요. 지나치게 점수를 후하게 주어서는 안됩니다.

상: 40~50
중 25~39
하 15~24

개별 평가 기준은 아래와 같습니다.
이력서 평가: 총점 50점 (이력서가 없는 경우 0점 처리)
 - 요구하는 포지션과 기술에 적합한지
 - 경력요건을 충족하고 있는지
 - 공백기가 있거나 요건 대비 경력이 부족하면 감점입니다
 - 자격요건에서 미흡 시 1 항목당 감점
 - 우대사항에서 미흡 시 1 항목당 감점
자기소개서 평가: 총점 50점 (자소서가 없는 경우 0점 처리)
 - 자기소개서를 논리있게 성심성의껏 작성했는지
 - 자기소개서의 분량과 표현도 평가에 들어갑니다
 - 논리에 비약이 있거나 구어체를 사용하면 크게 감점입니다
종합평가 100점 (이력서 50점 + 자기소개서 50점)

출력포맷은 아래와 같습니다

### 예시 1 이력서와 자소서 모두 있는 경우 ###
<result>
<total_score>(30 ~ 100)</total_score>
<resume_score>(15 ~ 50)</resume_score>
<selfintro_score>(15 ~ 50)</selfintro_score>
<opinion1> 지원자 이름이 들어간 1줄 요약 의견입니다.</opinion1>
<summary>(요약 의견을 작성합니다. 5~10줄 분량)</summary>
<eval_resume>(이력서에 대한 심도 있는 평가입니다. 경력과 기술 위주로 상세히 평가하세요)</eval_resume>
<eval_selfintro>(자소서에 대한 평가입니다. 경력과 기술, 논리력과 성실성 위주로 상세히 평가하세요)</eval_selfintro>
</result>

### 예시 2 이력서가 없는 경우 ###
<result>
<total_score>(15 ~ 50)</total_score>
<resume_score>0</resume_score>
<selfintro_score>(15 ~ 50)</selfintro_score>
<opinion1> 지원자 이름이 들어간 1줄 요약 의견입니다.</opinion1>
<summary>(요약 의견을 작성합니다. 5~10줄 분량)</summary>
<eval_resume>이력서가 없습니다.</eval_resume>
<eval_selfintro>(자소서에 대한 평가입니다. 경력과 기술, 논리력과 성실성 위주로 상세히 평가하세요)</eval_selfintro>
</result>

### Job Post:
{job_description}

### Resume:
{resume_text}

### Evaluation:"""

            # 입력을 토큰화하여 모델에 전달
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)

            # 모델 실행 및 결과 반환
            output = self.model(**inputs)
            result = output
            del inputs
            del output
            if not self.cpu_only:
                torch.cuda.empty_cache()

            return result
        except Exception as e:
            return f"Error: {str(e)}"
