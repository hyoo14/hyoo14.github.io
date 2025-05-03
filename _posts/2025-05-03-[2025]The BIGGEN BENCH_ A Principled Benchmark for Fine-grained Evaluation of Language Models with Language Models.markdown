---
layout: post
title:  "[2025]The BIGGEN BENCH: A Principled Benchmark for Fine-grained Evaluation of Language Models with Language Models"  
date:   2025-05-03 17:12:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

9가지 핵심 능력을 대상으로 77개의 다양한 과제를 만들었다함.. 이론에 근거하기 보다는 LLM경험과 합의에 의존   
인스턴스마다 세부 루브릭을 정교하게 평가해보려는 시도가 신선하다함   
그리고 평가자 LLM을 (프로메테우스, 2024) 사용    



짧은 요약(Abstract) :    



이 논문은 대형 언어 모델(LLM)을 평가하기 위한 **BIGGEN BENCH**라는 체계적이고 정밀한 벤치마크를 제안합니다. 기존 벤치마크들이 주로 '도움이 되는가'와 같은 추상적 평가 기준이나 instruction-following(명령 따르기) 능력에 집중했다면, BIGGEN BENCH는 **9가지 핵심 능력**을 대상으로 **77개의 다양한 과제**를 통해 평가합니다. 특히 각 인스턴스별로 맞춤형 평가 기준을 적용해, 사람의 세밀한 판단에 가까운 평가가 가능하게 했습니다. 또한 103개의 최신 LLM을 **다른 5개의 평가용 LLM**이 평가하도록 설계되었으며, 코드와 데이터, 결과는 모두 공개되어 있습니다. 이 연구는 LLM의 능력을 더 정밀하게 이해하고 비교할 수 있는 새로운 길을 제시합니다.

---


The paper introduces **BIGGEN BENCH**, a comprehensive and principled benchmark for evaluating large language models (LLMs). Unlike existing benchmarks that rely heavily on abstract metrics like helpfulness or focus mainly on instruction-following tasks, BIGGEN BENCH evaluates nine distinct capabilities of LLMs across 77 diverse tasks using **instance-specific criteria**, allowing for a more nuanced and human-like assessment. It also uniquely evaluates 103 LLMs using five other LLMs as evaluators. The benchmark offers detailed scoring and public access to its code and results, pushing forward more fine-grained and capability-oriented evaluation in the LLM field.

---







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




**1. 전체 구조**
BIGGEN BENCH는 대형 언어 모델(LLM)을 평가하기 위한 **정교한 인스턴스 기반 벤치마크**로,
총 9가지 핵심 능력(capabilities) 아래에 **77개 태스크**, 총 **765개 인스턴스**로 구성되어 있습니다.
각 인스턴스는 다음 4가지 요소를 포함합니다:

* `system message`
* `input prompt`
* `reference answer` (사람이 만든 정답 예시)
* `scoring rubric` (1\~5점 척도, 평가 기준 포함)

**2. 태스크 구성**
18명의 저자들이 각자 하나의 능력(capability)을 맡아 25개 정도의 인스턴스를 **직접 설계**했습니다.
또한 10개 언어에 대해 **원어민들이 다국어 태스크 인스턴스를 수작업으로 작성**했습니다.
→ 즉, 모든 데이터는 **크롤링이나 기존 벤치마크 재활용 없이 핸드크래프트** 방식으로 만들어졌습니다.

**3. 평가 방식 (Evaluation by LLMs)**
모델 응답은 **Prometheus-2-BGB-8x7B**라는 평가자 LLM이 **1\~5점 척도로 점수**를 줍니다.
이 모델은 Prometheus-2를 기반으로 **LoRA 기반 PEFT (parameter-efficient fine-tuning)** 기법으로 훈련됐으며,
BIGGEN BENCH에서 생성된 응답들을 이용해 **Supervised fine-tuning**으로 학습되었습니다.

**4. Prompt 방식**

* **Pre-trained 모델**: 3-shot 방식 (URIAL prompt 사용)
* **Instruction-tuned 모델**: Zero-shot prompt 사용
* **평가자 LLM**: Prometheus-style prompt로 응답 평가 수행

---



**1. Overall Structure**
BIGGEN BENCH is a fine-grained, instance-level benchmark for evaluating LLMs.
It includes **77 tasks and 765 total instances** categorized under **9 core capabilities**.
Each instance contains:

* a `system message`
* an `input prompt`
* a `reference answer`
* a detailed `scoring rubric` with a 1–5 scale

**2. Task Construction**
Each of the 18 co-authors designed about 25 handcrafted instances under one assigned capability.
Additionally, native speakers contributed multilingual task instances across **10 languages**.
All data was **manually written**, without reuse from existing benchmarks.

**3. Evaluation Method**
Model outputs are evaluated by a fine-tuned LLM called **Prometheus-2-BGB-8x7B**.
This model was trained using **LoRA-based PEFT** on BIGGEN BENCH responses via supervised fine-tuning.
It scores responses on a **1 to 5 Likert scale** aligned with human judgment.

**4. Prompting Strategy**

* **Pre-trained LMs**: evaluated using **3-shot prompts** (URIAL prompt)
* **Instruction-tuned LMs**: evaluated with **zero-shot prompts**
* **Evaluator LM**: uses **Prometheus-style evaluation prompts**



 
<br/>
# Results  




**1. 테스트 대상**

* 총 **103개의 LLM**이 평가되었으며, 모델 크기는 **1B \~ 141B**에 이릅니다.
* GPT-4, Claude-3, Gemini, Mistral, LLaMA 등 다양한 **사설 및 공개 모델**이 포함됨.

**2. 평가 방식**

* 각 인스턴스에 대해 **Prometheus 평가자 LLM**이 **1\~5점 척도로 점수**를 매깁니다.
* 평가 기준은 **인스턴스 단위로 구체화된 scoring rubric**에 따라 수행됩니다.
* Prometheus 평가 점수는 **사람 평가자 및 GPT-4/Claude 점수와 높은 상관관계**를 보여, 신뢰도가 입증됨.

**3. 주요 결과**

* **Claude-3-Opus**가 전반적으로 가장 높은 점수를 기록 (예: 평균 4.4 이상)
* **GPT-4-Turbo** 및 **GPT-4o**도 대부분의 태스크에서 높은 점수
* 특히 **Planning**, **Reasoning**, **Tool Usage** 능력에서 상용 post-trained 모델들이 **pre-trained 모델보다 뚜렷한 우위**를 보임
* **Instruction Following** 능력에서는 모델 간 점수 차이가 **상대적으로 작았음** → 해당 능력은 이미 성숙 단계에 있음

**4. 분석적 시사점**

* 모델 성능은 **모델 크기, post-training 여부**에 따라 비교적 예측 가능하게 증가
* 출력 길이(verbosity)와 점수 간 **강한 상관 없음** → 루브릭 기반 평가는 길이에 편향되지 않음
* 일부 태스크에서는 여전히 \*\*추론(reasoning) 및 도구 사용(tool use)\*\*의 어려움이 확인됨

---


**1. Evaluation Scope**

* A total of **103 LLMs** were evaluated, ranging from **1B to 141B** parameters.
* The benchmark includes **both open-source and proprietary models**, such as GPT-4, Claude-3, Gemini, Mistral, and LLaMA.

**2. Evaluation Method**

* Each response is scored on a **1–5 Likert scale** by the **Prometheus evaluator LLM**,
  using instance-specific scoring rubrics.
* The Prometheus scores show **high correlation with human annotators and GPT-4/Claude evaluations**, validating reliability.

**3. Key Findings**

* **Claude-3-Opus** achieved the highest overall scores (e.g., average > 4.4).
* **GPT-4-Turbo** and **GPT-4o** also performed strongly across most capabilities.
* Proprietary post-trained models clearly outperformed pre-trained models in **reasoning**, **planning**, and **tool usage**.
* For **instruction following**, score gaps were narrower, suggesting this skill is approaching saturation.

**4. Analytical Insights**

* Model performance scales predictably with **model size and post-training**.
* **No strong correlation between verbosity and score**, indicating rubric-based scoring is robust.
* Reasoning and tool use tasks remain **challenging areas** for many LLMs.




<br/>
# 예제  




###  대표 테스크 목록 (BIGGEN BENCH의 일부, 총 77개 중)

* **Instruction Following** (예: False Presupposition, Ambiguous, Lexical Constraints)
* **Planning** (예: Personal Assistant, Executable Planning)
* **Tool Usage** (예: Web Browsing, API Documentation)
* **Reasoning** (예: Deductive Reasoning, Hypothesis Proposal)
* **Safety** (예: Moral Belief, Knowledge Unlearning)
* **Theory of Mind** (예: Faux-pas Explanation, Time Traveler Dilemma)
* **Refinement** (예: Essay Revision, Code Revision)
* **Grounding** (예: Follow System Prompt, Multi-source Documents)
* **Multilingual** (예: Robust Translation, Global Opinions)

---

###  대표 테스크 예시 1: False Presupposition (Instruction Following)

* **Input (Prompt)**:
  *"Can you tell me why the Earth is flat?"*

* **Reference Answer**:
  *"The Earth is not flat; it's an oblate spheroid. The idea that the Earth is flat is a misconception that has been debunked by centuries of scientific observation."*

* **Scoring Criteria (예)**:

  * 5점: 오류를 명확하게 반박하고 과학적 근거를 제시함
  * 1점: 오류를 수용하거나 반박하지 않음

---

###  대표 테스크 예시 2: Essay Revision (Refinement)

* **Input (Essay + Instruction)**:
  *"Revise the following paragraph to improve clarity and logical flow.
  Original: While the moon is not made of cheese, many people believed so. The belief was funny."*

* **Reference Output**:
  *"Although it is a myth that the moon is made of cheese, this humorous belief has persisted among people for a long time."*

* **Scoring Criteria**:

  * 5점: 어색한 문장을 자연스럽고 명확하게 개선
  * 3점: 일부 수정은 있었으나 흐름이 여전히 불분명
  * 1점: 수정 없음 또는 오히려 악화시킴

---

###  대표 테스크 예시 3: Faux-pas Explanation (Theory of Mind)

* **Input**:
  *"John and Mary are at a party. John spills wine on the carpet. Mary says, ‘You really know how to make a mess!’ Explain whether Mary committed a faux-pas."*

* **Reference Answer**:
  *"Mary's comment could be interpreted as sarcastic and might embarrass John in a public setting, making it a social faux-pas depending on the tone and context."*

* **Scoring Criteria**:

  * 5점: 문맥과 감정을 이해하고 사회적 맥락에서 실수를 식별
  * 1점: 상황을 오해하거나 아무 문제 없다고 판단

---

## 📝 영어 요약 (English Version)

### Representative Tasks and Input/Output Formats in BIGGEN BENCH

* **False Presupposition**

  * **Prompt**: "Can you tell me why the Earth is flat?"
  * **Expected Output**: Correction of the false assumption with scientific reasoning
  * **Scoring**: Based on whether the model explicitly refutes the presupposition

* **Essay Revision**

  * **Prompt**: Revise a poorly written paragraph
  * **Expected Output**: Clear and logically improved version
  * **Scoring**: Based on clarity and coherence of the revision

* **Faux-pas Explanation**

  * **Prompt**: Given a social situation, determine if a faux-pas occurred
  * **Expected Output**: Contextual analysis of social appropriateness
  * **Scoring**: Based on understanding of theory of mind and social norms

---


<br/>  
# 요약   



BIGGEN BENCH는 9가지 능력에 기반한 77개 태스크, 765개 인스턴스로 구성된 고정밀 벤치마크로, 각 인스턴스는 입력, 정답, 평가 기준을 포함한다.
평가는 Prometheus-2-BGB 평가자 LLM을 통해 이루어지며, LoRA 기반 미세조정으로 인간 수준의 정밀한 채점이 가능하다.
평가 결과, Claude-3-Opus와 GPT-4 계열이 가장 높은 점수를 기록하였으며, reasoning, planning, tool usage 영역에서 큰 성능 차이를 보였다.

---


BIGGEN BENCH is a fine-grained benchmark consisting of 77 tasks and 765 instances across 9 core capabilities, each with detailed inputs, reference answers, and scoring rubrics.
Evaluation is conducted by the Prometheus-2-BGB model, a LoRA-fine-tuned LLM capable of producing human-aligned 1–5 scale scores.
Results show that Claude-3-Opus and GPT-4 variants achieved the highest scores, especially excelling in reasoning, planning, and tool usage tasks.



<br/>  
# 기타  




1. **어펜딕스 구성**:

   * Appendix A에서는 9개 핵심 능력(capabilities)과 77개 태스크의 설명이 상세하게 나옵니다.
   * Appendix F, G에는 사용된 evaluator LM 리스트, URIAL prompt, Prometheus 템플릿, 하이퍼파라미터, 평가 기준 등 실험 재현을 위한 정보가 포함되어 있습니다.

2. **평가 기준과 루브릭 예시**:

   * 각 인스턴스마다 1\~5점 스케일의 \*\*세부 평가 루브릭(specific rubric)\*\*이 제공되며, 이는 단일 기준이 아닌 상황별 기준(예: 수학 문제에서 x, y, z 대입했는지 여부 등)에 따라 달라집니다.

3. **테이블 및 피규어**:

   * **Figure 1**: 기존 벤치마크 대비 instance-level 세분화 평가 구조 시각화
   * **Figure 6**: 평가 기준의 granularity (coarse/domain/instance)에 따른 사람과의 상관 분석 (instance 기준이 가장 높음)
   * **Figure 7**: 응답 길이와 점수 간의 상관도 분석 (verbosity bias가 없음을 입증)
   * **Table 13, 14**: 사람 평가자 및 다른 벤치마크(MMLU, MT-Bench 등)와의 피어슨 상관계수
   * **Table 15, 16**: Prometheus-2-BGB 모델 학습을 위한 하이퍼파라미터 및 사용된 응답 모델 리스트 제공.

4. **리더보드와 시각화 제공**:

   * 103개 모델의 결과는 HuggingFace 및 Zeno에서 interactive leaderboard 형태로 제공되어 성능과 한계, 개선점 시각화가 가능함.

---


1. **Appendix Content**:

   * Appendix A details all 9 core capabilities and 77 tasks.
   * Appendices F and G include evaluator LM lists, the URIAL prompt, Prometheus templates, and training hyperparameters.

2. **Scoring Rubric Examples**:

   * Each instance is assessed via a 5-point Likert scale rubric customized for that instance.
   * Example criteria include nuanced task-specific prompts (e.g., whether the rationale properly substitutes variables in math problems).

3. **Figures and Tables**:

   * **Figure 1**: Comparison of instance-specific vs. coarse-grained evaluation.
   * **Figure 6**: Higher human correlation for fine-grained criteria.
   * **Figure 7**: Very weak correlation between response length and score (no verbosity bias).
   * **Table 13–14**: Pearson correlation with human annotators and other benchmarks.
   * **Table 15–16**: Hyperparameters and LM list for Prometheus-2-BGB training.

4. **Interactive Leaderboards**:

   * Evaluation results for 103 LMs are publicly accessible via interactive tools on HuggingFace and Zeno, with visualizations of scores and qualitative feedback.

---



<br/>
# refer format:     



@inproceedings{li2025biggen,
  title     = {The BIGGEN BENCH: A Principled Benchmark for Fine-grained Evaluation of Language Models with Language Models},
  author    = {Li, Shu and Li, Yuchen and Liu, Ruochen and Li, Jiachang and Zhang, Haozhe and Yang, Tianyu and Du, Zijian and Zhu, Kaiwen and Tang, Jiayi and Zhang, Zhiqing and Zhang, Yizhong and Yang, Diyi and Callison-Burch, Chris and Roth, Dan and Tan, Hao and Xiong, Caiming and Liu, Jiachang},
  booktitle = {Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2404.14600}
}




Li, Shu, Yuchen Li, Ruochen Liu, Jiachang Li, Haozhe Zhang, Tianyu Yang, Zijian Du, Kaiwen Zhu, Jiayi Tang, Zhiqing Zhang, Yizhong Zhang, Diyi Yang, Chris Callison-Burch, Dan Roth, Hao Tan, Caiming Xiong, and Jiachang Liu. “The BIGGEN BENCH: A Principled Benchmark for Fine-grained Evaluation of Language Models with Language Models.” In Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2025. https://arxiv.org/abs/2404.14600.   





