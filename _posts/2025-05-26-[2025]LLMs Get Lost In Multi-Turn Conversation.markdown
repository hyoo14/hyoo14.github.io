---
layout: post
title:  "[2025]LLMs Get Lost In Multi-Turn Conversation"  
date:   2025-05-26 00:56:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

멀티턴에서는 모델이 정보를 점진적으로 받을 때 오답을 먼저 제시하고 이후에도 그 오답에 집착하여 오류를 반복. 관련 기타 시각화 및 테스트 제공.  

짧은 요약(Abstract) :    




대형 언어 모델(LLM)은 사용자와의 다중 턴 대화를 통해 문제를 정의하고 탐색하고 세부화하는 데 도움을 줄 수 있는 대화형 인터페이스입니다. 하지만 현재까지 LLM의 성능 평가는 대부분 단일 턴, 명확하게 지정된 입력에만 초점을 맞춰 왔습니다. 이 논문에서는 6개의 생성 과제를 기반으로 대규모 시뮬레이션 실험을 수행해, 단일 턴 대비 다중 턴 대화에서 LLM의 성능이 평균 39% 감소함을 확인했습니다. 총 20만 건 이상의 대화 분석 결과, 이 성능 저하는 소폭의 능력 저하와 **신뢰성 급감(112% 증가)**으로 나뉘며, LLM이 대화 초기에 성급하게 결론을 내리고 그 결과에 과도하게 의존하는 경향이 주요 원인으로 나타났습니다. 즉, 한번 대화 방향이 잘못 설정되면, LLM은 그 상태에서 벗어나지 못하고 계속 ‘길을 잃은’ 상태로 남습니다.

⸻


Large Language Models (LLMs) are conversational interfaces. As such, LLMs have the potential to assist their users not only when they can fully specify the task at hand, but also to help them define, explore, and refine what they need through multi-turn conversational exchange. Although analysis of LLM conversation logs has confirmed that underspecification occurs frequently in user instructions, LLM evaluation has predominantly focused on the single-turn, fully-specified instruction setting. In this work, we perform large-scale simulation experiments to compare LLM performance in single- and multi-turn settings. Our experiments confirm that all the top open- and closed-weight LLMs we test exhibit significantly lower performance in multi-turn conversations than single-turn, with an average drop of 39% across six generation tasks. Analysis of 200,000+ simulated conversations decomposes the performance degradation into two components: a minor loss in aptitude and a significant increase in unreliability. We find that LLMs often make assumptions in early turns and prematurely attempt to generate final solutions, on which they overly rely. In simpler terms, we discover that when LLMs take a wrong turn in a conversation, they get lost and do not recover.










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







다양한 **기존 LLM(총 15개)**을 시뮬레이션 환경에서 테스트하고 분석한 평가 연구입니다. 그러나 논문에서는 실험을 위한 시뮬레이션 프레임워크 (sharded simulation), 샤딩 기법, 그리고 사용된 데이터 및 평가 지표에 대해 매우 체계적으로 설명하고 있어, 그 내용을 아래에 요약

⸻


1. 모델 아키텍처
	•	이 논문은 새로운 모델을 제안하지 않습니다.
	•	대신, 15개의 기존 LLM을 평가 대상으로 사용하며, 여기에는 OpenAI, Anthropic, Google, Meta 등 주요 기관의 최신 모델이 포함됩니다. 예: GPT-4o, Claude 3.7 Sonnet, Gemini 2.5 Pro 등.

2. 시뮬레이션 환경
	•	실제 사용자 대신 **LLM(GPT-4o-mini)**를 이용하여 시뮬레이션된 사용자 역할을 수행함.
	•	Sharded simulation이라는 프레임워크를 도입하여, 원래 단일 턴으로 제공된 입력(instruction)을 여러 개의 조각(shard)으로 나누고, 이를 다중 턴 대화로 시뮬레이션함.
	•	예: “60개의 눈덩이를 만들려면 얼마 걸려?” → 5개의 shard로 분할되어 매 턴마다 한 개씩 정보 제공.

3. 실험 데이터 및 작업(Task)
	•	총 6가지 생성 작업에 대해 실험:
	•	코드 생성 (HumanEval, LiveCodeBench)
	•	SQL 질의 생성 (Spider)
	•	API 호출 생성 (BFCL)
	•	수학 문제 풀이 (GSM8K)
	•	테이블 캡션 생성 (ToTTo)
	•	문서 요약 (Summary of a Haystack)
	•	각 작업마다 90~120개의 instruction이 sharded 버전으로 구성됨.

4. 평가지표
	•	모델의 성능을 세 가지 지표로 평가:
	•	P (평균 정확도): 여러 시뮬레이션에서 얻은 평균 점수
	•	A (Aptitude): 상위 10% 성능 (90th percentile) – 최선의 실행 사례
	•	U (Unreliability): 상위와 하위 10% 간 편차 – 모델 신뢰성의 척도

⸻


1. Model Architecture
	•	The paper does not propose a new model architecture.
	•	Instead, it evaluates 15 existing LLMs, including open and closed-weight models from major organizations (e.g., GPT-4o, Claude 3.7 Sonnet, Gemini 2.5 Pro, LLaMA 3, etc.).

2. Simulation Framework
	•	A sharded simulation environment is developed, where instructions from single-turn benchmarks are split into smaller “shards”.
	•	The simulated user (implemented with GPT-4o-mini) reveals one shard at a time per turn, emulating a multi-turn, underspecified conversation.
	•	Example: “How long to make 60 snowballs?” becomes 5 turns, each adding a piece of the problem.

3. Tasks and Data
	•	Six generation tasks are selected:
	•	Code generation (HumanEval, LiveCodeBench)
	•	SQL query generation (Spider)
	•	API function calling (BFCL)
	•	Math problem solving (GSM8K)
	•	Data-to-text table captioning (ToTTo)
	•	Multi-document summarization (Summary of a Haystack)
	•	Each task includes 90–120 instructions, each with a corresponding sharded version.

4. Evaluation Metrics
	•	LLM performance is evaluated via three key metrics:
	•	P (Average Performance): Mean score over multiple simulations
	•	A (Aptitude): 90th percentile – captures best-case capability
	•	U (Unreliability): Gap between 90th and 10th percentile – measures consistency or stability

⸻






   
 
<br/>
# Results  




1. 경쟁 모델들 (총 15개 LLM)
	•	OpenAI: GPT-4o, GPT-4o-mini, GPT-4.1, o3
	•	Anthropic: Claude 3 Haiku, Claude 3.7 Sonnet
	•	Google: Gemini 2.5 Flash, Gemini 2.5 Pro
	•	Meta: LLaMA 3.1–8B, LLaMA 3.3–70B, LLaMA 4 Scout
	•	Microsoft: Phi-4
	•	AI2: OLMo-2–13B
	•	DeepSeek: DeepSeek-R1
	•	Cohere: Command-A

이 중 GPT-4.1, Gemini 2.5 Pro, Claude 3.7 Sonnet이 싱글턴 성능은 우수하나, 멀티턴에서 동일하게 성능 하락을 겪음.

⸻

2. 테스트 데이터 및 작업 (총 6개 생성 과제)
	•	Code: Python 함수 생성 (HumanEval, LiveCodeBench)
	•	Database: SQL 질의 생성 (Spider)
	•	Actions: API 호출 생성 (Berkeley Function Calling Leaderboard)
	•	Math: 초등 수학 문제 해결 (GSM8K)
	•	Data-to-Text: 테이블 설명 문장 생성 (ToTTo)
	•	Summary: 문서 요약 및 출처 명시 (Summary of a Haystack)

각 task마다 90~120개의 instruction이 사용되었고, 각각 단일턴, CONCAT, 샤딩된 멀티턴 방식으로 테스트됨.

⸻

3. 평가지표 (3가지 주요 지표)
	•	P (Average Performance): 평균 정확도 점수 (0–100)
	•	A (Aptitude): 상위 10%의 성능 (모델이 최선의 경우 얼마나 잘 수행하는지)
	•	U (Unreliability): 상하위 10% 편차 → 성능 불안정성 (값이 클수록 신뢰성 낮음)

⸻

4. 핵심 결과
	•	전체 평균 성능 하락: -39%
	•	Fully-specified(단일턴)에서는 평균 90%에 달하던 성능이
	•	Multi-turn(샤딩된) 환경에서는 평균 65% 수준으로 감소
	•	모든 모델이 멀티턴에서 성능 하락 (모델 규모와 무관)
	•	더 좋은 모델이라도 unreliability는 여전히 매우 높음
	•	성능 하락 원인 분석:
	•	Aptitude(모델 능력)은 약간만 감소
	•	Unreliability는 평균 112% 증가 → 멀티턴에서는 동일 instruction에도 결과 편차 심각
	•	소결: LLM이 대화 중 초기에 잘못된 가정을 하고, 이를 정정하지 못하고 길을 잃는 현상

⸻

[English Summary: Experimental Results]

1. Competing Models (15 LLMs)
	•	OpenAI: GPT-4o, GPT-4o-mini, GPT-4.1, o3
	•	Anthropic: Claude 3 Haiku, Claude 3.7 Sonnet
	•	Google: Gemini 2.5 Flash, Gemini 2.5 Pro
	•	Meta: LLaMA 3.1–8B, LLaMA 3.3–70B, LLaMA 4 Scout
	•	Microsoft: Phi-4
	•	AI2: OLMo-2–13B
	•	DeepSeek: DeepSeek-R1
	•	Cohere: Command-A

Top models like GPT-4.1, Gemini 2.5 Pro, and Claude 3.7 Sonnet performed best in single-turn settings, but all models degraded similarly in multi-turn conversations.

⸻

2. Test Tasks and Data (6 Generation Tasks)
	•	Code: Python function generation (HumanEval, LiveCodeBench)
	•	Database: SQL query generation (Spider)
	•	Actions: API call generation (Berkeley Function Calling Leaderboard)
	•	Math: Solving math word problems (GSM8K)
	•	Data-to-Text: Table captioning (ToTTo)
	•	Summary: Multi-document summarization with citation (Summary of a Haystack)

Each task included 90–120 instructions. They were tested in:
	•	FULL (fully specified),
	•	CONCAT (shards combined into one input), and
	•	SHARDED (multi-turn with one shard per turn).

⸻

3. Evaluation Metrics
	•	P (Average Performance): Mean score (0–100)
	•	A (Aptitude): 90th percentile score, indicating best-case ability
	•	U (Unreliability): Difference between 90th and 10th percentile scores — captures variability and robustness

⸻

4. Key Findings
	•	Average Performance Drop: -39%
	•	From ~90% in FULL (single-turn) setting
	•	To ~65% in SHARDED (multi-turn, underspecified) setting
	•	All models degrade significantly in multi-turn conversations, regardless of model size or family
	•	Aptitude (A) drops slightly, but
	•	Unreliability (U) increases dramatically: +112% on average
	•	This means even with the same input, the model’s output varies widely
	•	Conclusion: When LLMs make incorrect assumptions early in a conversation, they get lost and fail to recover — termed the “Lost in Conversation” phenomenon

⸻







<br/>
# 예제  





[예시 1: 수학 문제 (GSM8K 기반) – Multi-turn Sharded 대화]

1. 테스트 데이터 출처
	•	GSM8K: 초등 수학 문제 데이터셋 (Fine-tuned LLM 평가에 널리 사용됨)

⸻

2. 단일턴 버전 (Fully-Specified Instruction)

영문 (English)

Jay is making snowballs to prepare for a snowball fight with his sister. He can build 20 snowballs in an hour, but 2 melt every 15 minutes. How long will it take before he has 60 snowballs?

한글 (Korean)

Jay는 여동생과 눈싸움을 준비하며 눈덩이를 만들고 있습니다. 그는 1시간에 20개의 눈덩이를 만들 수 있지만, 15분마다 2개씩 녹습니다. Jay가 60개의 눈덩이를 가지려면 얼마나 시간이 걸릴까요?


⸻

3. 멀티턴 샤딩 대화 예시 (Sharded Multi-turn Version)

턴 번호 (Turn)	사용자 발화 (User Turn)	LLM 응답 예시 (Assistant Turn)
1	Shard 1: How long before Jay’s ready for the snowball fight?  (Jay가 눈싸움을 할 준비가 되려면 얼마나 걸릴까?)	Hmm, to determine that I would need to know how fast he can make snowballs. Could you clarify?
2	Shard 2: He’s preparing for a snowball fight with his sister.  (여동생과 눈싸움을 준비하고 있어.)	Okay, got it. Still need the rate at which he makes snowballs to compute the time.
3	Shard 3: He can make 20 snowballs per hour.  (1시간에 20개 만들 수 있어.)	Thanks! That helps. But are there any complications, like melting?
4	Shard 4: He’s trying to get to 60 total.  (60개 만들려고 해.)	Alright, if nothing melted, that would take 3 hours. But… any melting?
5	Shard 5: 2 snowballs melt every 15 minutes.  (15분마다 2개가 녹아.)	Ah! So he loses 8 per hour. Net gain is 12 per hour. So 60 / 12 = 5 hours. Answer: 5 hours


⸻

4. 훈련 데이터 관련 정보
	•	이 논문은 모델 학습이 아닌 평가 목적이라 훈련 데이터는 명시적으로 포함되어 있지 않습니다.
	•	사용된 모델들(GPT-4o, Claude 3.7 등)은 이미 사전 학습(pretrained)된 상태이며, 이 실험에서는 별도의 파인튜닝 없이, 주어진 문제에 대한 zero-shot 또는 default setting으로 응답합니다.

⸻



항목	내용
테스트 데이터	GSM8K (초등 수학 문제)
작업 유형	수학 문제 해결 (text-to-number generation)
단일턴 입력	전체 문제를 한 번에 제시
멀티턴 입력	문제를 5개 shard로 나누어 한 턴씩 제공
출력 목표	수치 정답 (예: “5 hours”)
학습 데이터	모델별 사전 학습 데이터 사용, 본 논문에서는 fine-tuning 

이 외에도 API 호출 생성, 테이블 캡션, 요약, SQL 질의 생성 등 각 task 있음  












<br/>  
# 요약   





이 논문은 기존 LLM들을 대상으로, 단일턴 입력을 다중턴(sharded) 대화로 분해하여 신뢰성과 성능을 비교하는 시뮬레이션 실험을 수행했다. 그 결과, 모든 모델이 멀티턴 환경에서 평균 39%의 성능 하락과 112%의 신뢰성 감소를 보였으며, 특히 초기 잘못된 응답에 집착해 회복하지 못하는 현상이 나타났다. 예를 들어, 수학 문제를 5개의 정보 조각으로 나누어 제공했을 때, 모델은 정보가 완전히 주어지기 전에 성급히 답을 내리고 오답에 집착하는 경향을 보였다.

⸻



This paper evaluates existing LLMs by simulating multi-turn conversations using sharded instructions derived from single-turn benchmarks. Results show a 39% average performance drop and a 112% increase in unreliability across models, mainly due to premature answers and failure to recover from early mistakes. For example, in a math task split into five conversational shards, models often guessed answers too early and clung to incorrect reasoning even after receiving the full context.

⸻








<br/>  
# 기타  



1. Figure 1 – LLM의 멀티턴 성능 하락 요약
	•	내용: 단일턴과 멀티턴 시나리오에서 동일한 작업을 수행하는 흐름도.
	•	핵심 메시지: 멀티턴에서는 모델이 정보를 점진적으로 받을 때 오답을 먼저 제시하고 이후에도 그 오답에 집착하여 오류를 반복함.
	•	해석: “LLMs get lost in conversation”이란 문장 그대로, LLM은 대화 중 초기에 잘못된 방향으로 가면 스스로 복구하지 못함.

⸻

2. Figure 6 – 성능(A), 신뢰성(U) 시각화
	•	(a): 박스플롯 기반으로 aptitude(A)와 unreliability(U)를 시각화
	•	상위 수치(A)는 성능 한계치, U는 변동폭을 의미
	•	(b): 모델별 degradation 정도 시각화
	•	성능이 낮은 모델(LLaMA 3.1-8B 등)이든 높은 모델(GPT-4.1, Gemini 등)이든 멀티턴에서 유사하게 성능 저하
	•	(c): shard 수를 늘리는 실험
	•	2개 이상의 shard만 돼도 성능 급락 → “멀티턴 자체가 리스크 요소”

⸻

3. Table 1 – 15개 모델의 작업별 성능 비교 (FULL, CONCAT, SHARDED)
	•	내용: 각 모델이 6개 작업(Code, SQL, Math 등)을 FULL/CONCAT/SHARDED 설정에서 수행했을 때의 평균 정확도(P)
	•	결과 요약:
	•	FULL: 대부분 모델이 85~97% 사이
	•	CONCAT: 약간의 성능 저하
	•	SHARDED: 평균 39% 성능 하락
	•	의미: 단일턴에서는 매우 뛰어난 모델도, 정보를 분할해 대화식으로 제공하면 급격히 취약해짐

⸻

4. Table 2 – Recap, Snowball 전략 효과
	•	내용: 멀티턴에서 성능 회복을 위한 대화 반복 전략 효과
	•	결과: 성능 일부 회복되지만 FULL 수준에는 도달 못함
	•	해석: agent-like 반복 요약(recap) 방식은 임시방편적 효과는 있으나 근본적 해결은 아님

⸻


1. Figure 1 – Summary Diagram of LLM Performance Degradation
	•	Description: Flowchart contrasting single-turn vs. multi-turn performance.
	•	Key Insight: In multi-turn scenarios, LLMs jump to early answers based on partial context, and fail to revise once full information is revealed.
	•	Interpretation: The phrase “LLMs get lost in conversation” is visually illustrated — once a wrong assumption is made, recovery rarely happens.

⸻

2. Figure 6 – Aptitude (A) and Unreliability (U) Visualization
	•	(a): Box plot visual explanation of Aptitude = top performance, Unreliability = performance variance.
	•	(b): Model-by-model degradation plotted visually — both small and large models experience significant degradation in SHARDED.
	•	(c): Gradual sharding experiment — even starting from 2 shards causes performance collapse.

⸻

3. Table 1 – Task-wise Performance across 15 LLMs
	•	Description: Shows performance (P) for six tasks (Code, SQL, Math, etc.) under FULL, CONCAT, and SHARDED settings.
	•	Findings:
	•	FULL: 85–97% for most models
	•	CONCAT: Slight drop due to rephrasing
	•	SHARDED: Severe degradation (avg -39%)
	•	Takeaway: Top-performing LLMs in single-turn settings perform poorly when instructions are revealed incrementally.

⸻

4. Table 2 – Effect of Recap and Snowball Strategies
	•	Description: Measures whether repeating past user instructions helps in SHARDED scenarios.
	•	Result: Some improvement, but does not match FULL performance.
	•	Interpretation: Agent-like interventions help somewhat, but LLMs still struggle to internally manage multi-turn context.




<br/>
# refer format:     


@article{laban2025lost,
  title={LLMs Get Lost In Multi-Turn Conversation},
  author={Laban, Philippe and Hayashi, Hiroaki and Zhou, Yingbo and Neville, Jennifer},
  journal={arXiv preprint arXiv:2505.06120},
  year={2025},
  url={https://arxiv.org/abs/2505.06120}
}



Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. LLMs Get Lost In Multi-Turn Conversation. arXiv preprint arXiv:2505.06120, 2025. https://arxiv.org/abs/2505.06120.  






