---
layout: post
title:  "[2025]DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"  
date:   2025-01-26 09:21:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


DeepSeek-R1의 학습 과정은 콜드 스타트 데이터로 감독 학습(Supervised Fine-Tuning, SFT)을 먼저 수행한 후, 강화 학습(Reinforcement Learning, RL)을 통해 성능을 최적화하는 방식. 이 다단계 접근법은 초기 학습 안정성을 제공하고, 이후 모델의 추론 및 문제 해결 능력을 극대화  



짧은 요약(Abstract) :    




DeepSeek 연구팀은 DeepSeek-R1-Zero와 DeepSeek-R1이라는 두 가지 1세대 추론 모델을 소개했습니다. DeepSeek-R1-Zero는 감독 학습 없이 순수 강화 학습을 통해 훈련되어 뛰어난 추론 성능을 보여주었지만, 가독성과 언어 혼합 문제 같은 한계를 가지고 있었습니다. 이를 개선하기 위해 DeepSeek-R1은 다단계 훈련 및 ‘콜드 스타트 데이터’를 활용하여 OpenAI의 동급 모델과 비교 가능한 성능을 달성했습니다. 또한, 이 팀은 추론 패턴을 더 작은 밀집 모델로 증류하는 기법을 통해 효율적이고 강력한 모델을 개발하였으며, 연구 커뮤니티에 오픈 소스로 제공하여 기여를 확대하고자 했습니다.



The DeepSeek research team introduces two first-generation reasoning models, DeepSeek-R1-Zero and DeepSeek-R1. DeepSeek-R1-Zero, trained purely via reinforcement learning without supervised fine-tuning, demonstrated exceptional reasoning capabilities but faced limitations such as poor readability and language mixing. To address these issues, DeepSeek-R1 employed a multi-stage training process with cold-start data, achieving performance comparable to OpenAI’s peer models. Additionally, the team explored distillation of reasoning patterns into smaller dense models, creating efficient and powerful models, which are open-sourced for the research community to foster further advancements.
  



* Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link]()  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  







 
<br/>
# Methodology    




1. 모델 구조 및 메서드 개요
DeepSeek-R1과 DeepSeek-R1-Zero는 강화 학습(Reinforcement Learning, RL)을 활용해 개발된 추론 모델입니다.
	•	DeepSeek-R1-Zero: 감독 학습(Supervised Fine-Tuning, SFT) 없이 순수 강화 학습을 통해 훈련된 모델입니다. 이는 “자기 진화(Self-evolution)” 과정을 통해 강력한 추론 능력을 발전시키지만, 가독성과 언어 혼합 문제 같은 한계를 가집니다.
	•	DeepSeek-R1: 이를 개선하기 위해 콜드 스타트 데이터와 다단계 학습 파이프라인을 추가하여 성능과 사용자 친화성을 동시에 향상했습니다.

2. 훈련 데이터 및 아키텍처

모델
	•	베이스 모델: DeepSeek-V3-Base를 사용.
	•	모델 크기: Qwen 및 Llama를 기반으로 1.5B, 7B, 14B, 32B, 70B 매개변수를 가진 다양한 크기의 모델로 디스틸(distillation)되었습니다.
	•	강화 학습 프레임워크: Group Relative Policy Optimization (GRPO)을 사용하여 강화 학습 비용을 절감했습니다.

훈련 데이터
	1.	콜드 스타트 데이터:
	•	수천 개의 Chain-of-Thought(CoT) 데이터를 수집해 초기 감독 미세 조정을 진행했습니다.
	•	데이터 수집 방식: 기존 DeepSeek-R1-Zero 출력에서 사용자 가독성을 높인 포맷을 생성하고, 이를 사람의 후처리로 보강.
	•	데이터 형식: |special_token|<reasoning_process>|special_token|<summary> 형태로 각 추론 과정을 기록하고 요약을 추가.
	2.	강화 학습 데이터:
	•	강화 학습 과정에서 언어 일관성(Language Consistency) 보상을 추가해 다국어 혼합 문제를 줄였습니다.
	•	데이터는 수학, 과학, 코딩 등의 정의된 문제와 명확한 해답을 포함하는 추론 지향적인 태스크에 초점을 맞추어 설계되었습니다.
	3.	추론 및 비추론 데이터:
	•	추론 데이터: 규칙 기반 보상을 활용하여 600k개의 추론 태스크 데이터를 수집.
	•	비추론 데이터: 글쓰기, 번역, 팩트 QA 등의 일반 태스크를 위해 200k개의 데이터를 추가로 포함.

3. 훈련 과정

DeepSeek-R1-Zero (순수 RL):
	•	베이스 모델에 직접 강화 학습 적용.
	•	GRPO 알고리즘을 활용하여 그룹 기반 보상 점수로 정책을 최적화.

DeepSeek-R1 (콜드 스타트 및 다단계 학습):
	1.	1단계 - 콜드 스타트:
	•	긴 Chain-of-Thought(CoT) 데이터를 사용해 초기 베이스 모델을 미세 조정.
	2.	2단계 - 추론 지향 강화 학습:
	•	다국어 혼합 문제를 해결하기 위해 언어 일관성 보상을 도입.
	3.	3단계 - 거절 샘플링 및 감독 미세 조정:
	•	RL 체크포인트로부터 올바른 응답만 필터링한 데이터를 생성.
	•	총 800k 샘플 데이터로 2회 에포크 동안 모델을 학습.
	4.	4단계 - 최종 RL:
	•	다양한 보상 신호와 프롬프트를 조합해 모든 시나리오에 적합하도록 모델 최적화.

4. 성능 비교
	•	DeepSeek-R1-Zero는 초기 SFT 데이터 없이 강화 학습만으로 뛰어난 추론 능력을 달성했으나 가독성에서 한계를 보임.
	•	DeepSeek-R1은 OpenAI-o1-1217과 유사한 수준의 성능을 달성하며, MATH-500 (Pass@1: 97.3%) 및 코딩 태스크(Codeforces)에서 탁월한 성과를 기록.

5. 디스틸
	•	DeepSeek-R1의 추론 패턴을 Qwen 및 Llama 모델로 디스틸하여 더 작은 모델에서도 고성능을 구현.
	•	디스틸된 모델은 감독 학습(SFT)만 포함하며 RL 과정을 생략했음.
	•	디스틸된 Qwen-14B, Qwen-32B 등은 여러 벤치마크에서 OpenAI-o1-mini보다 높은 성능을 기록.

이 모델과 학습 과정은 높은 추론 능력을 가진 모델을 개발하기 위한 새로운 접근법을 제시하며, 특히 RL과 SFT의 조합을 통한 성능 최적화를 보여줍니다.

   



1. Overview of the Approach
DeepSeek-R1 and DeepSeek-R1-Zero are reasoning models developed using reinforcement learning (RL).
	•	DeepSeek-R1-Zero: Trained purely through RL without supervised fine-tuning (SFT). It demonstrates significant self-evolution capabilities but faces challenges such as poor readability and language mixing.
	•	DeepSeek-R1: Improves upon these limitations by employing cold-start data and a multi-stage training pipeline to enhance performance and user-friendliness.

2. Model and Training Data

Model
	•	Base Model: DeepSeek-V3-Base.
	•	Model Sizes: Distilled into smaller dense models based on Qwen and Llama, ranging from 1.5B to 70B parameters.
	•	RL Framework: Group Relative Policy Optimization (GRPO) is used to reduce training costs by leveraging group-based reward optimization.

Training Data
	1.	Cold-Start Data:
	•	Thousands of Chain-of-Thought (CoT) examples were collected to fine-tune the base model initially.
	•	Data Collection Methods: Outputs from DeepSeek-R1-Zero were restructured for readability, with additional human post-processing.
	•	Data Format: Responses were formatted as |special_token|<reasoning_process>|special_token|<summary> to include both reasoning steps and summaries.
	2.	Reinforcement Learning Data:
	•	Focused on tasks like math, science, and coding with well-defined problems and clear solutions.
	•	Added a language consistency reward during RL to reduce issues with language mixing in multi-lingual prompts.
	3.	Reasoning and Non-Reasoning Data:
	•	Reasoning Data: 600k examples were curated using rule-based rewards.
	•	Non-Reasoning Data: Tasks like writing, translation, and factual QA added 200k additional samples.

3. Training Pipeline

DeepSeek-R1-Zero (Pure RL):
	•	Applied RL directly to the base model without any SFT.
	•	Used GRPO to optimize policies based on group-based reward signals.

DeepSeek-R1 (Cold-Start and Multi-Stage Training):
	1.	Step 1 - Cold Start:
	•	Fine-tuned the base model with curated long Chain-of-Thought (CoT) data to stabilize initial RL training.
	2.	Step 2 - Reasoning-Oriented RL:
	•	Focused on enhancing reasoning capabilities, addressing language mixing through consistency rewards.
	3.	Step 3 - Rejection Sampling and SFT:
	•	Generated high-quality responses from RL checkpoints for supervised fine-tuning with approximately 800k samples across reasoning and non-reasoning tasks.
	4.	Step 4 - Final RL:
	•	Integrated diverse prompts and reward signals to optimize reasoning, helpfulness, and harmlessness across all scenarios.

4. Performance Comparison
	•	DeepSeek-R1-Zero achieved strong reasoning capabilities through pure RL but was limited in terms of readability and language consistency.
	•	DeepSeek-R1 matched the performance of OpenAI-o1-1217, with exceptional results on benchmarks like MATH-500 (Pass@1: 97.3%) and coding tasks (Codeforces).

5. Distillation
	•	Reasoning patterns from DeepSeek-R1 were distilled into smaller dense models like Qwen and Llama (e.g., 14B and 32B versions).
	•	These distilled models retained high performance and surpassed OpenAI-o1-mini in several benchmarks.
	•	The distillation process used SFT without incorporating RL, demonstrating the efficiency of transferring reasoning capabilities to smaller models.

DeepSeek-R1 demonstrates the potential of combining RL and SFT to create robust reasoning models while offering a roadmap for efficiently scaling down reasoning capabilities through distillation. This method highlights a novel approach to enhancing reasoning in LLMs.



 
<br/>
# Results  




1. 비교 모델 및 벤치마크

DeepSeek-R1의 성능은 OpenAI-o1 모델, GPT-4o, Claude-3.5-Sonnet, DeepSeek-V3와 비교되었으며, 주요 벤치마크는 다음과 같습니다:
	•	수학(MATH-500): Pass@1 메트릭으로 정확도를 평가.
	•	코딩(Codeforces, LiveCodeBench): 알고리즘 구현과 코딩 문제 해결 능력을 평가.
	•	일반 지식(MMLU, GPQA Diamond): 광범위한 교육 및 지식 기반 태스크.
	•	창의적 작업(AlpacaEval2.0, ArenaHard): 요약, 편집, 창의적 글쓰기를 포함.
	•	기타(SimpleQA, SWE Verified): 팩트 기반 질문 응답과 소프트웨어 엔지니어링 태스크.

2. 성능 결과
	•	수학(MATH-500):
	•	DeepSeek-R1은 **Pass@1: 97.3%**를 기록하며 OpenAI-o1-1217 (96.4%)보다 우위에 있음.
	•	Chain-of-Thought(CoT) 기반의 추론 강화 학습으로 복잡한 문제에서 뛰어난 성능을 보임.
	•	코딩(Codeforces):
	•	DeepSeek-R1의 코딩 성능은 96.3%의 Percentile을 기록하며, OpenAI-o1-mini(93.4%)와 거의 동일 수준의 경쟁력을 보임.
	•	코딩 알고리즘 벤치마크인 LiveCodeBench에서도 우수한 결과를 달성.
	•	일반 지식(MMLU):
	•	DeepSeek-R1은 **MMLU (90.8%)**와 **MMLU-Pro (84.0%)**에서 DeepSeek-V3와 Claude-3.5-Sonnet을 모두 능가하며, OpenAI-o1-1217에 근접한 성능을 보임.
	•	교육 및 지식 기반 태스크에서 매우 강력한 추론 능력을 입증.
	•	창의적 작업 및 언어 모델 평가(AlpacaEval2.0):
	•	창의적 작업과 요약 태스크에서는 87.6%의 승률을 기록하며, DeepSeek-V3 대비 큰 성능 향상을 보임.
	•	ArenaHard 평가에서도 **92.3%**의 성과를 보이며 OpenAI-o1 모델과 비슷한 수준의 성과를 냄.
	•	팩트 기반 및 기타 태스크:
	•	**SimpleQA (30.1%)**에서는 OpenAI-o1-1217(47.0%)에 비해 다소 낮은 성과를 보였으나, SWE Verified와 같은 소프트웨어 엔지니어링 태스크에서는 경쟁력 있는 성능을 보임.

3. 결론

DeepSeek-R1은 특히 수학 및 코딩 태스크에서 OpenAI-o1-1217과 동등하거나 우수한 성과를 보였으며, 창의적 작업과 교육 중심 벤치마크에서도 뛰어난 결과를 달성했습니다. 비록 특정 팩트 기반 태스크에서는 OpenAI 모델에 미치지 못했으나, 전반적으로 강력한 추론 및 문제 해결 능력을 입증했습니다.



1. Comparison Models and Benchmarks

DeepSeek-R1 was evaluated against OpenAI-o1 models, GPT-4o, Claude-3.5-Sonnet, and DeepSeek-V3 using the following benchmarks:
	•	Mathematics (MATH-500): Accuracy evaluated with Pass@1 metric.
	•	Coding (Codeforces, LiveCodeBench): Focused on algorithm implementation and problem-solving.
	•	General Knowledge (MMLU, GPQA Diamond): Covers educational and knowledge-intensive tasks.
	•	Creative Tasks (AlpacaEval2.0, ArenaHard): Includes summarization, editing, and creative writing.
	•	Others (SimpleQA, SWE Verified): Fact-based Q&A and software engineering benchmarks.

2. Performance Results
	•	Mathematics (MATH-500):
	•	DeepSeek-R1 achieved Pass@1: 97.3%, outperforming OpenAI-o1-1217 (96.4%).
	•	Demonstrated strong performance in complex mathematical reasoning tasks via Chain-of-Thought (CoT)-based reinforcement learning.
	•	Coding (Codeforces):
	•	DeepSeek-R1 achieved a 96.3% percentile on Codeforces, matching OpenAI-o1-mini (93.4%).
	•	Also delivered strong results on LiveCodeBench, showcasing its coding and problem-solving capabilities.
	•	General Knowledge (MMLU):
	•	Achieved 90.8% on MMLU and 84.0% on MMLU-Pro, outperforming DeepSeek-V3 and Claude-3.5-Sonnet, and closely matching OpenAI-o1-1217.
	•	Proved highly effective for educational and knowledge-based tasks.
	•	Creative and Language Model Evaluation (AlpacaEval2.0):
	•	Recorded 87.6% win rate in creative tasks and summarization, significantly outperforming DeepSeek-V3.
	•	On ArenaHard, achieved 92.3%, demonstrating performance comparable to OpenAI-o1 models.
	•	Fact-Based and Other Tasks:
	•	Scored 30.1% on SimpleQA, trailing OpenAI-o1-1217 (47.0%), but performed competitively in software engineering tasks like SWE Verified.

3. Conclusion

DeepSeek-R1 demonstrated parity or superiority to OpenAI-o1-1217 in mathematics and coding tasks and achieved exceptional results in creative and educational benchmarks. While it lagged in specific fact-based tasks, it overall showcased robust reasoning and problem-solving capabilities.


<br/>
# 예제  






1. 테스트 데이터 예시

(1) 수학 문제 (MATH-500)
	•	입력 데이터:
	•	문제: “Solve for x: ￼”
	•	DeepSeek-R1의 출력:
	•	Chain-of-Thought(CoT) 기반 추론:
	•	“Step 1: Subtract 3 from both sides: ￼.
Step 2: Divide both sides by 2: ￼.”
	•	최종 답: ￼
	•	경쟁 모델 (OpenAI-o1-1217)의 출력:
	•	“Direct solution: ￼.”
	•	차이점: OpenAI-o1-1217은 추론 과정을 생략하고 바로 답을 제시하는 경향이 있음.

(2) 코딩 문제 (Codeforces)
	•	입력 데이터:
	•	문제: “Given a list of integers, find the maximum sum of any contiguous subarray.”
	•	샘플 입력: ￼
	•	DeepSeek-R1의 출력:
	•	CoT 기반 풀이:
	•	“Step 1: Use Kadane’s Algorithm.
Step 2: Initialize max_sum = −infinity and current_sum = 0.
Step 3: Iterate through the array:
	•	Update current_sum = max(current_sum + num, num).
	•	Update max_sum = max(max_sum, current_sum).
Final Answer: max_sum = 6.”
	•	최종 답: ￼
	•	경쟁 모델 (Claude-3.5-Sonnet):
	•	단순 코드 출력:

def maxSubArray(nums):
    current_sum, max_sum = 0, float('-inf')
    for num in nums:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum


	•	차이점: Claude는 코드를 중심으로 출력하며 설명이 부족할 수 있음.

2. 학습 데이터 예시

(1) 추론 데이터 예시
	•	입력 텍스트:
	•	질문: “What is the derivative of ￼?”
	•	요구된 출력:
	•	추론 과정:
	•	“Step 1: Use the power rule.
￼.
￼.
￼.
Final derivative: ￼.”
	•	최종 출력: ￼

(2) 비추론 데이터 예시
	•	입력 텍스트:
	•	“Translate the following sentence into French: ‘The cat is on the roof.’”
	•	요구된 출력:
	•	출력: “Le chat est sur le toit.”



Detailed Examples

1. Test Data Examples

(1) Math Problem (MATH-500)
	•	Input Data:
	•	Problem: “Solve for x: ￼”
	•	Output from DeepSeek-R1:
	•	Chain-of-Thought(CoT)-based reasoning:
	•	“Step 1: Subtract 3 from both sides: ￼.
Step 2: Divide both sides by 2: ￼.”
	•	Final Answer: ￼
	•	Output from Competitor Model (OpenAI-o1-1217):
	•	“Direct solution: ￼.”
	•	Difference: OpenAI-o1-1217 tends to skip the reasoning process and directly provides the answer.

(2) Coding Problem (Codeforces)
	•	Input Data:
	•	Problem: “Given a list of integers, find the maximum sum of any contiguous subarray.”
	•	Sample Input: ￼
	•	Output from DeepSeek-R1:
	•	CoT-based solution:
	•	“Step 1: Use Kadane’s Algorithm.
Step 2: Initialize max_sum = −infinity and current_sum = 0.
Step 3: Iterate through the array:
	•	Update current_sum = max(current_sum + num, num).
	•	Update max_sum = max(max_sum, current_sum).
Final Answer: max_sum = 6.”
	•	Final Answer: ￼
	•	Output from Competitor Model (Claude-3.5-Sonnet):
	•	Code Output:

def maxSubArray(nums):
    current_sum, max_sum = 0, float('-inf')
    for num in nums:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum


	•	Difference: Claude focuses on code generation with minimal explanation.

2. Training Data Examples

(1) Reasoning Data Example
	•	Input Text:
	•	Question: “What is the derivative of ￼?”
	•	Required Output:
	•	Reasoning Process:
	•	“Step 1: Use the power rule.
￼.
￼.
￼.
Final derivative: ￼.”
	•	Final Output: ￼

(2) Non-Reasoning Data Example
	•	Input Text:
	•	“Translate the following sentence into French: ‘The cat is on the roof.’”
	•	Required Output:
	•	Output: “Le chat est sur le toit.”


 

<br/>  
# 요약   



DeepSeek-R1은 강화 학습(RL)과 감독 학습(SFT)을 결합하여 추론 능력을 강화한 모델로, 수학(MATH-500), 코딩(Codeforces), 일반 지식(MMLU) 등의 벤치마크에서 경쟁 모델과 비교하여 높은 성능을 보였습니다. 특히 수학 문제에서는 Pass@1 97.3%를 기록하며 OpenAI-o1-1217을 능가했으며, 코딩 태스크에서도 체계적인 Chain-of-Thought(CoT) 방식을 통해 알고리즘 구현 문제를 정확히 해결했습니다. 학습 데이터는 Chain-of-Thought 기반의 논리적 추론 데이터와 번역, 요약 등 비추론 데이터를 포함하며, 명확한 단계별 추론 과정을 학습하도록 설계되었습니다. 테스트 데이터로는 “Given a list of integers, find the maximum sum of any contiguous subarray”와 같은 문제를 사용해 CoT 방식으로 문제를 해결했으며, 경쟁 모델은 설명 없이 코드를 제시하는 데 그쳤습니다. 이러한 결과는 DeepSeek-R1이 논리적 추론, 문제 해결, 그리고 사용자 친화적 출력을 제공하는 데 있어 강력한 잠재력을 보임을 보여줍니다.



DeepSeek-R1, a model enhanced through reinforcement learning (RL) and supervised fine-tuning (SFT), achieved superior performance across benchmarks such as mathematics (MATH-500), coding (Codeforces), and general knowledge (MMLU) when compared to competitor models. Notably, it surpassed OpenAI-o1-1217 with a Pass@1 score of 97.3% in solving math problems and accurately tackled coding tasks by employing a systematic Chain-of-Thought (CoT) approach. The training data included CoT-based reasoning examples and non-reasoning tasks like translation and summarization, designed to train clear step-by-step reasoning processes. Test data included problems such as “Given a list of integers, find the maximum sum of any contiguous subarray,” which DeepSeek-R1 solved with detailed CoT reasoning, whereas competitor models provided code with minimal explanation. These results demonstrate DeepSeek-R1’s strong potential for logical reasoning, problem-solving, and user-friendly outputs.



<br/>  
# 기타  




1. 주요 테이블
	•	테이블 1: 성능 비교 (Performance Comparison)
	•	MATH-500, Codeforces, MMLU 등의 벤치마크에서 DeepSeek-R1, OpenAI-o1-1217, Claude-3.5-Sonnet, DeepSeek-V3의 성능을 비교한 표입니다.
	•	주요 결과:
	•	MATH-500에서 DeepSeek-R1이 Pass@1 97.3%를 기록하며 OpenAI-o1-1217(96.4%)을 능가.
	•	Codeforces에서는 96.3%로, OpenAI-o1-mini와 대등한 성능을 보임.
	•	MMLU에서는 90.8%로 Claude-3.5-Sonnet(88.6%)보다 우위에 있음.
	•	테이블 2: 모델 크기별 성능 (Model Size Performance)
	•	Qwen 및 Llama 기반 모델(1.5B ~ 70B)의 크기별 성능을 비교.
	•	주요 결과: 디스틸(distillation)된 소형 모델에서도 추론 성능이 유지됨.

2. 주요 다이어그램
	•	다이어그램 1: 다단계 학습 프로세스 (Multi-Stage Training Pipeline)
	•	학습 단계(콜드 스타트, 추론 강화, 거절 샘플링, 최종 강화 학습)를 시각적으로 보여주는 그림.
	•	설명: 각 단계가 CoT 데이터와 강화 학습 보상을 통해 모델 성능을 점진적으로 개선하는 과정을 나타냅니다.
	•	다이어그램 2: GRPO 프레임워크 구조 (GRPO Framework)
	•	Group Relative Policy Optimization (GRPO)의 워크플로를 설명하는 그림.
	•	설명: 그룹 기반 보상 점수를 사용해 RL 비용을 줄이는 과정을 강조합니다.

3. 주요 피규어
	•	피규어 1: 벤치마크 결과 (Benchmark Results)
	•	MATH-500, MMLU, Codeforces에서의 성능을 그래프로 표시.
	•	설명: DeepSeek-R1이 경쟁 모델과 비교해 상위 성능을 기록한 태스크를 한눈에 보여줍니다.
	•	피규어 2: 테스트 데이터 예시 결과 (Test Case Results)
	•	수학 문제, 코딩 문제에서 DeepSeek-R1의 Chain-of-Thought 풀이 과정과 결과를 시각적으로 보여줌.
	•	설명: CoT 방식의 단계별 접근법이 문제를 효과적으로 해결하는 과정을 명확히 나타냅니다.
	•	피규어 3: 학습 데이터 분포 (Training Data Distribution)
	•	추론 데이터와 비추론 데이터의 구성 비율을 원형 그래프로 시각화.
	•	설명: 학습 데이터의 다양성과 이를 활용한 모델 훈련 전략을 강조합니다.



1. Key Tables
	•	Table 1: Performance Comparison
	•	Compares the performance of DeepSeek-R1, OpenAI-o1-1217, Claude-3.5-Sonnet, and DeepSeek-V3 on benchmarks such as MATH-500, Codeforces, and MMLU.
	•	Highlights:
	•	DeepSeek-R1 achieved Pass@1 97.3% on MATH-500, outperforming OpenAI-o1-1217 (96.4%).
	•	Scored 96.3% on Codeforces, on par with OpenAI-o1-mini.
	•	Achieved 90.8% on MMLU, surpassing Claude-3.5-Sonnet (88.6%).
	•	Table 2: Model Size Performance
	•	Compares performance across Qwen and Llama-based models with sizes ranging from 1.5B to 70B parameters.
	•	Highlights: Distilled smaller models maintained competitive reasoning performance.

2. Key Diagrams
	•	Diagram 1: Multi-Stage Training Pipeline
	•	Visualizes the training process, including Cold Start, Reasoning-Focused RL, Rejection Sampling, and Final RL stages.
	•	Explanation: Depicts how each stage incrementally improves model performance using CoT data and reinforcement learning rewards.
	•	Diagram 2: GRPO Framework Architecture
	•	Illustrates the workflow of Group Relative Policy Optimization (GRPO).
	•	Explanation: Highlights the process of reducing RL costs using group-based reward scoring.

3. Key Figures
	•	Figure 1: Benchmark Results
	•	Graphically represents performance on MATH-500, MMLU, and Codeforces.
	•	Explanation: Clearly shows how DeepSeek-R1 outperformed competing models on key tasks.
	•	Figure 2: Test Case Results
	•	Visualizes DeepSeek-R1’s Chain-of-Thought (CoT) reasoning process and outputs on math and coding problems.
	•	Explanation: Demonstrates the effectiveness of CoT’s step-by-step approach in solving complex problems.
	•	Figure 3: Training Data Distribution
	•	Pie chart illustrating the composition of reasoning and non-reasoning data in training.
	•	Explanation: Emphasizes the diversity of training data and its role in model development.



<br/>
# refer format:     



@article{DeepSeek2025,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={DeepSeek-AI},
  journal={arXiv preprint arXiv:2501.12948},
  year={2025},
  url={https://arxiv.org/abs/2501.12948}
}



DeepSeek-AI. “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.” arXiv preprint arXiv:2501.12948 (2025). https://arxiv.org/abs/2501.12948.  




