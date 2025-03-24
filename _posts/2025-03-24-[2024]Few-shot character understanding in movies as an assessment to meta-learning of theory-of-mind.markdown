---
layout: post
title:  "[2024]Few-shot character understanding in movies as an assessment to meta-learning of theory-of-mind"  
date:   2025-03-24 18:00:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


lllm의 영화 구성 요소 파악 데이터셋 제공?  
인물 이해 테스트  
 
 
 
짧은 요약(Abstract) :    






이 논문은 인간이 소설 속 인물을 빠르게 이해하는 능력, 즉 소수의 정보만으로 인물의 정신 상태를 추론하는 Theory-of-Mind(ToM) 능력을 기계가 얼마나 잘 학습할 수 있는지를 평가하려는 시도입니다. 이를 위해 저자들은 실제 영화 대본 약 1,000개를 기반으로 TOM-IN-AMC라는 새로운 데이터셋을 만들어냈습니다. 각 영화는 몇 개의 장면만 보고 인물을 이해하는 few-shot 학습 태스크로 구성됩니다.

또한 다양한 ToM 요소(예: 믿음, 욕망, 의도 등)를 분리해 평가할 수 있는 새로운 프롬프팅 기법을 제안하였고, 이 방법이 기존 모델보다 우수함을 보여줍니다. 사람은 과거에 본 영화들을 바탕으로 높은 정확도로 인물의 정신 상태를 추론하지만, GPT-4 같은 최신 대형 언어모델이나 메타러닝 알고리즘은 여전히 20% 이상 성능이 떨어진다고 보고합니다. 이는 기존 AI가 인간 수준의 ToM에 도달하기엔 아직 부족하다는 점을 부각시킵니다.

⸻



When reading a story, humans can quickly understand new fictional characters with a few observations, mainly by drawing analogies to fictional and real people they already know. This reflects the few-shot and meta-learning essence of humans’ inference of characters’ mental states, i.e., theory-of-mind (ToM), which is largely ignored in existing research. We fill this gap with a novel NLP dataset, TOM-IN-AMC, the first assessment of machines’ meta-learning of ToM in a realistic narrative understanding scenario. Our dataset consists of ∼1,000 parsed movie scripts, each corresponding to a few-shot character understanding task that requires models to mimic humans’ ability of fast digesting characters with a few starting scenes in a new movie.

We propose a novel ToM prompting approach designed to explicitly assess the influence of multiple ToM dimensions. It surpasses existing baseline models, underscoring the significance of modeling multiple ToM dimensions for our task. Our extensive human study verifies that humans are capable of solving our problem by inferring characters’ mental states based on their previously seen movies. In comparison, our systems based on either state-of-the-art large language models (GPT-4) or meta-learning algorithms lags >20% behind, highlighting a notable limitation in existing approaches’ ToM capabilities.

⸻






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



⸻



이 논문은 TOM-IN-AMC라는 새로운 벤치마크를 제안합니다. 이 벤치마크는 사람들이 소수의 장면만 보고도 영화 캐릭터의 정신 상태를 추론할 수 있는 능력, 즉 **Theory of Mind(ToM)**의 few-shot meta-learning 능력을 기계가 학습할 수 있는지를 평가합니다. 다음은 제안된 메서드 구성입니다:

1. 트레이닝 데이터: TOM-IN-AMC
	•	총 1,007개 영화 대본을 수집하여 구축.
	•	각 영화는 5명 이하의 주요 인물이 등장하며, 캐릭터들의 이름은 **익명화(ID로 대체)**됨.
	•	각 영화는 하나의 few-shot 태스크로 간주되어, 일부 장면만으로 캐릭터를 학습하고, 이후 장면에서 캐릭터를 추론하게 함.
	•	이름 유출(LLM 사전 학습) 방지를 위해 테스트 세트에서 이름 변형(perturbation) 적용.

2. ToMPro: 제안 아키텍처 및 프롬프트 기반 모델
	•	GPT-4를 기반으로 한 LLM 유도 방식이며, 다음 두 단계를 거칩니다:
	•	Stage 1 (ToM 상태 생성): 주어진 장면들을 기반으로 각 인물의 정신 상태를 다섯 가지 ToM 차원(성격, 감정, 믿음, 욕망, 의도)으로 기술.
	•	Stage 2 (캐릭터 추론): 익명화된 테스트 장면에서 Stage 1에서 생성한 설명을 바탕으로 캐릭터의 정체를 추론.

3. 비교 대상 아키텍처
	•	Prototypical Network: Longformer 기반 임베딩을 활용한 transductive 메타러닝.
	•	LEOPARD: 각 태스크(영화)마다 클래스 수가 달라지는 문제를 해결하기 위해 파라미터 생성기를 도입한 inductive 메타러닝 방식.

4. 학습 방식
	•	Longformer를 기반으로 장면을 인코딩하고, 각 캐릭터의 표현을 얻음.
	•	학습은 GPU(V100)에서 이루어졌으며, 모델은 평균적으로 인간보다 약 20% 낮은 정확도를 보임.
	•	GPT-4 기반의 ToMPro는 기존 모델보다 높은 성능을 보였지만 여전히 인간보다 부족.

⸻



The paper proposes a novel benchmark, TOM-IN-AMC, to assess machines’ meta-learning capability of Theory of Mind (ToM), inspired by the human ability to infer characters’ mental states from just a few scenes in a new movie. The method includes the following components:

1. Training Data: TOM-IN-AMC
	•	Built from 1,007 movie scripts, each forming a separate few-shot learning task.
	•	Each movie includes a small number of main characters (≤5), whose identities are anonymized using IDs.
	•	Models are trained to infer characters based on a limited number of introductory scenes and tested on later scenes.
	•	To prevent memorization by large pre-trained models (like GPT-4), name perturbation is applied in the test set.

2. ToMPro: Proposed Prompting-Based Architecture
	•	A two-stage method utilizing GPT-4:
	•	Stage 1 (Mental State Generation): For each scene, character descriptions are generated across five ToM dimensions — personality, emotion, belief, desire, and intention.
	•	Stage 2 (Character Inference): Using these mental states, the model identifies anonymized characters in test scenes.

3. Baseline Architectures
	•	Prototypical Network: A transductive method that uses Longformer-based embeddings for similarity-based classification.
	•	LEOPARD: An inductive meta-learning algorithm with a parameter generator, designed to handle varying numbers of classes per task (movie).

4. Training Procedure
	•	Longformer is used to encode scenes, and attention-based pooling is applied to obtain character-specific representations.
	•	Models are trained on a single V100 GPU, with hyperparameters tuned via development set performance.
	•	While ToMPro outperforms existing models, it still lags >20% behind human performance, indicating the current gap in ToM reasoning capabilities of large language models.

⸻









       
 
<br/>
# Results  




이 논문의 실험은 인간과 AI 모델의 few-shot 상황에서의 캐릭터 이해 능력, 즉 ToM(Mind Theory) 메타러닝 능력을 평가하기 위한 것입니다.

1. 테스크(Task):
	•	캐릭터 추론 문제 (Character Guessing Task)
주어진 장면 속 등장인물들의 대사와 행동을 바탕으로, **익명화된 캐릭터 ID(P0~P4)**를 실제 캐릭터 이름 중 하나로 맞히는 다지선다형 분류 문제입니다.
	•	각 영화는 하나의 few-shot 태스크로 간주되며, **초반 장면(3/5)**을 훈련 데이터로, **후반 장면(2/5)**을 테스트 데이터로 사용합니다.

2. 테스트 데이터(Test Set):
	•	총 1,007개의 영화 중:
	•	학습(Train): 807편
	•	개발(Dev): 100편
	•	테스트(Test): 100편
	•	각 영화당 약 3~5명의 주인공이 등장하며, 각 인물에 대해 평균 20개 이하의 훈련 샘플만 제공 → few-shot 상황

3. 평가지표(Metric):
	•	Instance-level Accuracy (정답 인물 맞추기 정확도)
각 장면 내의 캐릭터 발화 단위를 개별 인스턴스로 간주하여, 각 익명 캐릭터가 누구인지 정확히 맞혔는가를 평가합니다.

4. 주요 결과:

모델	Dev Accuracy	Test Accuracy
Human	88.0%	–
Random	22.1%	25.0%
Majority	34.9%	36.0%
Prototypical Net	55.4%	53.2%
LEOPARD (inductive)	59.4%	58.6%
GPT-4 ICL (20-shot)	67.8%	–
GPT-4 ToMPro (제안)	68.2%	66.9%

	•	GPT-4 기반 ToMPro는 다른 모델보다 성능이 높지만, 여전히 인간과는 약 20% 차이.
	•	인간은 새로운 영화를 보지 않았더라도, 이전에 본 영화에서 얻은 **캐릭터 경험을 활용한 추론 전략(meta-learning 방식)**으로 높은 정확도를 보임.
	•	GPT-4는 **욕망(desire)**과 같은 장기 상태 이해에서 한계가 있으며, 장면별 맥락 불일치 문제도 존재.

⸻



The paper evaluates how well humans and AI models can perform few-shot character understanding in movies — a proxy task for meta-learning Theory of Mind (ToM).

1. Task:
	•	Character Guessing Task
Given anonymized scenes (with characters labeled as P0 to P4), the model must map each anonymized character to their true identity using multiple-choice options, based on previous few-shot examples from the same movie.
	•	Each movie forms one few-shot task, with first 60% of scenes for training and last 40% for testing.

2. Test Set:
	•	From 1,007 movie scripts:
	•	Training set: 807 movies
	•	Development set: 100 movies
	•	Test set: 100 movies
	•	Each movie includes ~3–5 main characters, with fewer than 20 training samples per character, making this a true few-shot setting.

3. Evaluation Metric:
	•	Instance-level Accuracy
Each masked speaker instance in a scene is evaluated separately to assess whether the model correctly identifies the character.

4. Key Results:

System	Dev Acc	Test Acc
Human	88.0%	–
Random	22.1%	25.0%
Majority	34.9%	36.0%
Prototypical Net	55.4%	53.2%
LEOPARD	59.4%	58.6%
GPT-4 ICL (20-shot)	67.8%	–
GPT-4 ToMPro (proposed)	68.2%	66.9%

	•	While GPT-4 ToMPro outperforms all other models, it still lags ~20% behind human performance.
	•	Human annotators excel at this task by leveraging meta-learning-style reasoning, drawing analogies to previously seen characters.
	•	GPT-4 faces difficulties especially in modeling long-term mental states (e.g., desires) and handling inconsistencies between generated mental models and scene context.

⸻


















<br/>
# 예제  





이 논문에서의 테스크는 영화 속 장면들을 활용한 캐릭터 추론 문제로 구성되어 있습니다.

1. 훈련 데이터 예제 (Training Example)

훈련 데이터에서는 영화의 앞부분 장면들이 제공되며, 각 장면에서 인물들의 실제 이름과 발화 내용이 주어집니다. 이 장면들은 모델이 캐릭터의 성격, 의도, 감정 등을 파악하는 데 사용됩니다.

예:

Scene Title: INT. OFFICE – DAY

CHAR1 (Emily): I know this project means a lot to you, but we have to be realistic.
CHAR2 (James): You always do this. You never take risks.
CHAR1: That’s not fair. I‘m the one who stayed up all night finishing your part.

→ 이 장면을 통해 모델은 ‘Emily’와 ‘James’의 성격과 관계를 학습하게 됨.

이런 장면들이 10개 또는 20개 주어지며, 이를 바탕으로 **캐릭터별 정신 상태 묘사(프롬프트 기반)**가 생성됩니다.

⸻

2. 테스트 데이터 예제 (Test Input/Output Example)

테스트 데이터에서는 캐릭터 이름이 가려진 새로운 장면이 주어집니다. 모델은 이를 기반으로 P0, P1 등 익명화된 인물이 누구인지를 맞혀야 합니다.

예 (입력 Input):

Scene Title: INT. CONFERENCE ROOM – NIGHT

P0: I did what I had to do. You weren’t stepping up.
P1: So you thought betraying me was the solution?
P0: I thought saving the team mattered more than your pride.

선택지:
(a) Emily
(b) James
(c) Sarah
(d) Michael
(e) David

예 (출력 Output):

P0–James  
P1–Emily

이러한 테스트 장면에 대해, 모델은 훈련 장면을 바탕으로 생성된 캐릭터 설명을 활용해 정신 상태와 발화의 맥락을 비교하며 추론합니다.

⸻


The task in this paper involves predicting the identity of characters in movie scenes based on few-shot training examples. Here’s how the input/output examples are structured:

1. Training Data Example

Training data consists of introductory scenes from a movie, where character names are visible. These are used to build mental state representations of each character across Theory of Mind (ToM) dimensions.

Example:

Scene Title: INT. OFFICE – DAY

CHAR1 (Emily): I know this project means a lot to you, but we have to be realistic.
CHAR2 (James): You always do this. You never take risks.
CHAR1: That’s not fair. I‘m the one who stayed up all night finishing your part.

→ From this, the model learns about Emily and James’ personalities and interpersonal dynamics.

The model uses around 10–20 such scenes per movie as few-shot demonstrations to form a character model.

⸻

2. Test Data & Task Input/Output Example

In the test set, character names are anonymized. The model must infer the identity of each anonymized character (e.g., P0, P1) from a set of options.

Test Input Example:

Scene Title: INT. CONFERENCE ROOM – NIGHT

P0: I did what I had to do. You weren’t stepping up.
P1: So you thought betraying me was the solution?
P0: I thought saving the team mattered more than your pride.

Choices:
(a) Emily
(b) James
(c) Sarah
(d) Michael
(e) David

Expected Output:

P0–James  
P1–Emily

In this setting, the model must use prior mental state descriptions generated from training scenes to reason about intentions, beliefs, and personality in the new scene.

⸻











<br/>  
# 요약   



이 논문은 영화 대본을 기반으로, 소수 장면만 보고 인물의 정신 상태를 추론하는 능력을 평가하는 ToM 메타러닝 벤치마크인 TOM-IN-AMC와 프롬프트 기반 GPT-4 모델 ToMPro를 제안한다.
결과적으로 ToMPro는 기존 메타러닝 모델(LEOPARD, ProtoNet)보다 높은 정확도(약 67%)를 보였지만, 여전히 인간보다 약 20% 낮은 성능을 기록했다.
훈련에는 이름이 표시된 장면들이 사용되며, 테스트에서는 익명화된 캐릭터가 등장하고 모델은 발화를 기반으로 인물의 정체를 맞추는 분류 태스크를 수행한다.

⸻

 

This paper introduces TOM-IN-AMC, a benchmark for evaluating ToM meta-learning by predicting characters’ mental states from few-shot movie script scenes, and proposes the GPT-4-based prompting model ToMPro.
ToMPro outperforms baseline meta-learning models like LEOPARD and ProtoNet (achieving ~67% accuracy), but still falls ~20% short of human performance.
Training scenes include real character names, while test scenes anonymize characters, requiring the model to classify identities based on dialogue reasoning.

⸻






<br/>  
# 기타  




1. Appendix (부록)
	•	부록에는 추가 실험 결과, 프롬프트 템플릿 예시, 에러 분석, 인간 평가자 가이드라인이 포함됩니다.
	•	특히 ToMPro의 각 **5가지 차원(성격, 감정, 믿음, 욕망, 의도)**을 프롬프트로 어떻게 유도했는지 구체적인 예시가 제시됩니다.
	•	인간 평가 시에는 캐릭터 간 발화 순서 및 장면 맥락을 주의 깊게 고려하도록 지침이 제공됩니다.

2. Figures (그림)
	•	Figure 2는 전체 파이프라인 구조를 시각화하며, 훈련→정신상태 생성→추론 단계를 보여줍니다.
	•	Figure 3은 성능 비교 그래프로, 인간, GPT-4, 기타 모델들의 정확도를 직관적으로 보여줍니다.
	•	Figure 4는 GPT-4의 오류 예시를 시각화하며, 맥락 파악 실패 및 캐릭터 혼동 사례를 강조합니다.

3. Tables (표)
	•	Table 1: 전체 데이터셋의 통계 정보 (총 영화 수, 장면 수, 캐릭터 수 등).
	•	Table 2: 다양한 모델의 성능 비교표 (Dev/Test 정확도).
	•	Table 3: 프롬프트 템플릿 구조 및 예시.
	•	Table 5: 인간 평가자 정확도와 GPT-4 비교 결과.

⸻



1. Appendix
	•	The appendix provides additional experimental details, prompt templates, error analysis, and human annotation guidelines.
	•	It includes concrete examples of how prompts were designed to target five ToM dimensions: personality, emotion, belief, desire, and intention.
	•	Instructions for human annotators emphasize careful attention to speaker order and scene context.

2. Figures
	•	Figure 2 illustrates the overall system pipeline, from few-shot training scenes to mental state generation and final inference.
	•	Figure 3 shows a performance comparison chart between human annotators, GPT-4, and baseline models.
	•	Figure 4 visualizes GPT-4 failure cases, highlighting common issues like misunderstanding context or confusing characters.

3. Tables
	•	Table 1: Dataset statistics — number of movies, scenes, and characters.
	•	Table 2: Performance results on development and test sets across different models.
	•	Table 3: Prompt template formats and examples.
	•	Table 5: Comparison of GPT-4 vs. human annotator accuracy across ToM dimensions.

⸻












<br/>
# refer format:     


@article{yu2022fewshot,
  title={Few-shot character understanding in movies as an assessment to meta-learning of theory-of-mind},
  author={Yu, Mo and Wang, Qiujing and Zhang, Shunchi and Sang, Yisi and Pu, Kangsheng and Wei, Zekai and Wang, Han and Xu, Liyan and Li, Jing and Yu, Yue and Zhou, Jie},
  journal={arXiv preprint arXiv:2211.04684},
  year={2022},
  note={Version 2, February 2024}
}



  
  
  
  
Yu, Mo, Qiujing Wang, Shunchi Zhang, Yisi Sang, Kangsheng Pu, Zekai Wei, Han Wang, Liyan Xu, Jing Li, Yue Yu, and Jie Zhou. “Few-shot character understanding in movies as an assessment to meta-learning of theory-of-mind.” arXiv preprint arXiv:2211.04684 (2022). Version 2, February 2024. 