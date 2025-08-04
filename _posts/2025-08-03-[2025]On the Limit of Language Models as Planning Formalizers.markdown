---
layout: post
title:  "[2025]On the Limit of Language Models as Planning Formalizers"  
date:   2025-08-03 11:13:40 +0200
categories: study
---

{% highlight ruby %}


한줄 요약: 

LLM을 통한 PDDL 변환 평가  
포매팅이 중요하다..  



짧은 요약(Abstract) :    




이 논문은 대형 언어 모델(LLM)이 자연어 환경 설명을 \*\*형식적인 계획 언어(PDDL)\*\*로 변환하는 **‘formalizer’ 역할**에 얼마나 적합한지를 체계적으로 평가합니다. 기존 연구는 템플릿화된 인위적인 설명을 기반으로 부분적인 PDDL만 생성했지만, 본 연구는 **다양한 자연스러움 수준의 설명**에서 \*\*전체 PDDL(도메인 + 문제 파일)\*\*을 생성합니다. 실험 결과, 충분히 큰 모델은 직접 계획을 생성하는 방식보다 **형식화를 통한 계획이 더 효과적**이며, 어휘 변형에도 강인함을 보입니다. 다만, 설명이 자연스러워질수록 성능은 감소하였고 이에 대한 오류 분석도 제공합니다.

---



> Large Language Models have been found to create plans that are neither executable nor verifiable in grounded environments. An emerging line of work demonstrates success in using the LLM as a formalizer to generate a formal representation of the planning domain in some language, such as Planning Domain Definition Language (PDDL). This formal representation can be deterministically solved to find a plan. We systematically evaluate this methodology while bridging some major gaps. While previous work only generates a partial PDDL representation, given templated, and therefore unrealistic environment descriptions, we generate the complete representation given descriptions of various naturalness levels. Among an array of observations critical to improve LLMs’ formal planning abilities, we note that most large enough models can effectively formalize descriptions as PDDL, outperforming those directly generating plans, while being robust to lexical perturbation. As the descriptions become more natural-sounding, we observe a decrease in performance and provide detailed error analysis.

---

필요하시면 메서드나 결과 요약도 도와드릴게요!




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





이 논문에서는 두 가지 접근법을 비교하는데, 첫째는 **LLM을 직접 계획(plan)을 생성하는 ‘planner’로 사용하는 방식**, 둘째는 **LLM이 자연어 설명으로부터 계획 도메인과 문제를 형식화하여 PDDL로 변환한 뒤, 외부의 결정론적 플래너가 계획을 생성하는 ‘formalizer’ 방식**입니다.

저자들은 후자인 **LLM-as-Formalizer** 방식에 집중하며, 기존 연구와 달리 다음의 세 가지 중요한 차별점을 둡니다:

1. **전체 PDDL 생성**: 이전 연구들은 도메인 파일(DF)이나 문제 파일(PF)의 일부만 생성하고 나머지를 제공받는 전제였지만, 이 연구는 LLM이 **도메인과 문제 파일 전체를 생성**합니다.

2. **자연스러운 설명 사용**: 기존 연구는 인위적으로 구조화된 설명을 사용했으나, 이 연구는 \*\*자연어 수준이 다양한 설명(Heavily Templated, Moderately Templated, Natural)\*\*을 제공합니다.

3. **제로샷(Zero-shot) 설정**: 별도의 파인튜닝이나 예시 없이 **제로샷 프롬프트**만으로 모델을 평가합니다. LLM에게 직접 도메인과 문제 설명을 주고, 이를 기반으로 전체 PDDL 파일을 생성하게 합니다.

모델은 **GPT-4o, GPT-4o-mini, DeepSeek-R1, O3-mini, Gemma, LLaMA 등**을 사용하며, 일부는 405B 파라미터까지 포함된 대형 모델입니다. 모든 실험은 표준 계획 시뮬레이션 환경 (BlocksWorld, Logistics, Barman, MysteryBlocksWorld)에서 이루어졌습니다.

---



This paper compares two approaches to planning with LLMs: one where the LLM directly generates the plan (**LLM-as-Planner**), and another where the LLM generates a formal representation of the planning domain and problem in PDDL, and a separate symbolic planner is used to derive the plan (**LLM-as-Formalizer**).

The focus of the paper is on the LLM-as-Formalizer approach, with three key methodological innovations:

1. **Full PDDL Generation**: Unlike prior work that generates only parts of the domain or problem file, this work asks the LLM to generate **both the domain file (DF) and problem file (PF) entirely** from natural language descriptions.

2. **Naturalistic Descriptions**: Rather than using artificial, templated input, this work uses domain and problem descriptions with **varying levels of naturalness**, including fully natural, human-authored descriptions.

3. **Zero-shot Prompting**: The models are tested in a **zero-shot setting**, with no fine-tuning or in-context examples. The LLM must produce the PDDL representation based solely on the natural language input.

The models evaluated include **GPT-4o, GPT-4o-mini, DeepSeek-R1, O3-mini, Gemma, and LLaMA** models (up to 405B parameters). Experiments are conducted on standard planning domains including BlocksWorld, Logistics, Barman, and MysteryBlocksWorld.

---



   
 
<br/>
# Results  





이 논문에서는 네 가지 시뮬레이션 환경(BlocksWorld, Logistics, Barman, MysteryBlocksWorld)에서 여러 LLM을 대상으로 두 가지 방법론(LLM-as-Planner, LLM-as-Formalizer)을 비교 평가했습니다. 사용한 모델은 **GPT-4o, GPT-4o-mini, O3-mini, DeepSeek-R1**, 그리고 오픈소스 모델인 **Gemma (2B/27B)**, **LLaMA-3.1 (8B/70B/405B)** 등이 포함됩니다.

####  평가 메트릭:

* **Solvability (해결 가능성)**: 모델이 생성한 PDDL이 플래너에 의해 처리 가능한지를 평가
* **Correctness (정확성)**: 생성된 계획이 실제로 목표 상태를 달성하는지를 검증

####  주요 실험 결과:

* **GPT-4o와 O3-mini는 전체 PDDL을 생성할 수 있는 대표적인 강력한 모델**이며, Planner 방식보다 Formalizer 방식에서 더 높은 성능을 보임.

  * 예: GPT-4o는 BlocksWorld에서 100개 중 60개의 정확한 계획을 Formalizer 방식으로 생성했으나 Planner 방식에서는 33개만 성공.
* **자연어 설명이 더 자연스러워질수록 성능은 하락**함. 특히 ‘Natural’ 버전에서는 암묵적 상식을 요구하기 때문에 모델이 중요한 전제를 빠뜨리기도 함.
* **Gemma, LLaMA 계열 모델은 전반적으로 매우 낮은 성능**을 보였으며, PDDL 문법 오류(syntax error)를 자주 발생시킴.
* **MysteryBlocksWorld**는 변수명과 개체명이 무작위로 바뀐 테스트셋인데, 이 경우 **LLM-as-Planner는 전혀 작동하지 않았고, Formalizer만이 의미 있는 결과를 냄**.
* **DeepSeek-R1은 예외적으로 Planner로도 강한 성능을 보였고, reasoning trace를 내부적으로 생성함으로써 더 나은 결과를 도출함.**

---



The study evaluates two planning strategies (LLM-as-Planner and LLM-as-Formalizer) using a variety of LLMs across four planning domains: **BlocksWorld, Logistics, Barman, and MysteryBlocksWorld**. The models tested include **GPT-4o, GPT-4o-mini, O3-mini, DeepSeek-R1**, and open-source models such as **Gemma-2B/27B** and **LLaMA-3.1-8B/70B/405B**.

####  Evaluation Metrics:

* **Solvability**: Whether the generated PDDL is parsable and solvable by a planner.
* **Correctness**: Whether the resulting plan achieves the desired goal state according to the validator.

####  Key Findings:

* **GPT-4o and O3-mini** excel at full PDDL generation and consistently **outperform the Planner approach**, especially in simple domains like BlocksWorld.

  * For instance, GPT-4o generated 60 correct plans out of 100 using the Formalizer method, compared to 33 using Planner.
* **Performance decreases as input descriptions become more natural**, due to missing implicit assumptions (e.g., "clear block" not being explicitly stated).
* **Gemma and LLaMA models generally fail**, with frequent **syntax errors** and minimal solvability or correctness.
* On the **MysteryBlocksWorld** dataset (with randomized object names), **Planner methods completely failed**, while **Formalizer methods still performed well**, especially for models like GPT-4o.
* **DeepSeek-R1** showed strong performance even as a Planner, likely due to generating **step-by-step reasoning traces** to validate its own output.

---




<br/>
# 예제  





####  테스크 개요:

이 논문은 **LLM이 자연어로 된 환경 설명과 목표를 입력으로 받아**, 이를 기반으로 **PDDL 형식의 계획 도메인(DF)과 문제(PF) 파일을 생성**하고, 이 파일들을 **전통적인 플래너에 입력하여 실행 가능한 계획(plan)을 도출**하는지 평가합니다.

####  입력 데이터 예시:

* **도메인 설명 (Domain Description; DD)**:

  ```
  The robot arm can pick up and move one block at a time from one position to another. It is only able to move the top block from any stack or table, and have only one block held by the robot arm at a time.
  ```

  → 블록쌓기 환경에서 로봇 팔이 수행할 수 있는 동작을 자연스럽게 설명한 문장입니다. 명시적인 전제 조건(predicate)을 나열하지 않습니다.

* **문제 설명 (Problem Description; PD)**:

  ```
  Block A is currently on top of block B. Block B is on the table. Block C is also on the table. The goal is to stack block B on block C, and then block A on block B.
  ```

  → 초기 상태와 목표 상태를 자연어로 표현합니다.

####  출력 데이터 예시:

* **도메인 파일 (Domain File; DF)**:

  ```lisp
  (:action pickup
    :parameters (?b)
    :precondition (and (clear ?b) (on-table ?b) (arm-empty))
    :effect (and (holding ?b) (not (clear ?b)) (not (on-table ?b)) (not (arm-empty)))
  )
  ```

* **문제 파일 (Problem File; PF)**:

  ```lisp
  (:init
    (on a b)
    (on-table b)
    (on-table c)
    (clear a)
    (clear c)
    (arm-empty)
  )
  (:goal
    (on b c)
    (on a b)
    (on-table c)
  )
  ```

* **계획 출력 (Plan)**:

  ```
  (unstack a b)
  (putdown a)
  (pickup b)
  (stack b c)
  (pickup a)
  (stack a b)
  ```

####  데이터셋 구성:

* 총 **4가지 환경**: BlocksWorld, Logistics, Barman, MysteryBlocksWorld
* 각 환경에 대해 **100개 문제 인스턴스**, 설명은 세 가지 수준으로 구분됨:

  * Heavily Templated: 거의 PDDL과 동일한 문장 구조
  * Moderately Templated: 약간 자연어에 가까움
  * Natural: 완전히 자연스러운 문장으로 설명

---


####  Task Overview:

The task is to evaluate whether an LLM can take **natural language descriptions of a planning environment and task** as input, and generate the corresponding **PDDL domain (DF) and problem (PF) files**, which can then be used by a symbolic planner to produce an executable plan.

####  Example Input:

* **Domain Description (DD)**:

  ```
  The robot arm can pick up and move one block at a time from one position to another. It is only able to move the top block from any stack or table, and have only one block held by the robot arm at a time.
  ```

* **Problem Description (PD)**:

  ```
  Block A is currently on top of block B. Block B is on the table. Block C is also on the table. The goal is to stack block B on block C, and then block A on block B.
  ```

####  Example Output:

* **Domain File (DF)**:

  ```lisp
  (:action pickup
    :parameters (?b)
    :precondition (and (clear ?b) (on-table ?b) (arm-empty))
    :effect (and (holding ?b) (not (clear ?b)) (not (on-table ?b)) (not (arm-empty)))
  )
  ```

* **Problem File (PF)**:

  ```lisp
  (:init
    (on a b)
    (on-table b)
    (on-table c)
    (clear a)
    (clear c)
    (arm-empty)
  )
  (:goal
    (on b c)
    (on a b)
    (on-table c)
  )
  ```

* **Final Plan**:

  ```
  (unstack a b)
  (putdown a)
  (pickup b)
  (stack b c)
  (pickup a)
  (stack a b)
  ```

####  Dataset Design:

* 4 domains: BlocksWorld, Logistics, Barman, MysteryBlocksWorld
* Each with 100 problem instances
* Three levels of description naturalness:

  * Heavily Templated (near-PDDL syntax)
  * Moderately Templated (partially natural)
  * Natural (fully human-like language)

---



<br/>  
# 요약   




이 논문은 LLM이 자연어 설명으로부터 전체 PDDL 도메인 및 문제 파일을 생성하는 formalizer로서의 역할을 수행할 수 있는지를 제안하고 분석합니다. GPT-4o와 O3-mini 등 대형 모델은 직접 계획을 생성하는 방식보다 더 높은 정확도로 문제를 해결했으며, 설명이 자연스러워질수록 성능은 하락했습니다. 예시로 블록쌓기 환경에서 “블록 A는 B 위에 있고, C는 테이블 위에 있다”는 설명을 입력받아, 이를 기반으로 PDDL과 실행 가능한 계획을 성공적으로 생성했습니다.

---



This paper explores whether LLMs can act as formalizers by generating full PDDL domain and problem files from natural language descriptions. Large models like GPT-4o and O3-mini outperformed direct plan generation, though performance decreased with more natural input. For example, in a block-stacking task, the model successfully converted a human-like description into valid PDDL and an executable plan.

---



<br/>  
# 기타  




####  Figure 1 (다이어그램):

LLM-as-Planner와 LLM-as-Formalizer 방식을 비교하는 개념도입니다. 후자는 자연어 설명을 통해 PDDL 도메인/문제 파일을 생성한 뒤, 이를 기존 플래너에 입력하여 계획을 도출하는 구조이며, 특히 **자연스러움의 수준이 성능에 미치는 영향을 시각화**합니다.

####  Figure 3, 4 (성능 그래프):

Figure 3은 세 가지 환경(BlocksWorld, Logistics, MysteryBlocksWorld)에 대해 Formalizer와 Planner 방식의 **정확도 차이**를 보여줍니다. GPT-4o와 O3-mini는 대부분 Formalizer 방식에서 Planner를 압도하며, **LLaMA 계열 모델은 거의 실패**에 가깝습니다. Figure 4는 설명이 자연스러울수록 성능이 떨어지는 패턴을 명확히 보여주며, **자연어 이해의 한계를 시사**합니다.

####  Figure 5, 6 (에러 분석):

Figure 5는 \*\*문법 오류(syntax error)와 의미 오류(semantic error)\*\*의 비율을 비교합니다. 오픈소스 모델은 문법 오류가 많았고, GPT 모델은 문법은 맞지만 의미 오류(예: 잘못된 precondition)가 많았습니다. Figure 6은 **도메인 파일 내 세부 오류 유형**(예: 잘못된 효과 정의, 누락된 조건 등)을 수치화해, **Formalizer 성능 개선의 방향성을 제시**합니다.

####  Table 1 (관련 연구 비교):

선행 연구들은 대부분 부분적인 PDDL만 생성하거나 인위적인 설명을 사용한 반면, 이 논문은 **자연어 수준이 다양한 설명에서 전체 DF+PF를 생성**하는 최초의 체계적 시도임을 강조합니다.

####  Appendix (부록):

* A: 실제 데이터 예시(PDDL, 자연어 설명)를 포함하여 실험 구성의 **현실성을 높임**
* B\~D: 다양한 프롬프트 및 체인오브쏘트 등 추가 실험 결과를 통해 **few-shot이나 COT이 성능을 크게 개선하지 않음**을 보임
* E: 수작업으로 분류한 오류 항목들과 예시를 통해 **오류 패턴을 정량화**함

---


####  Figure 1 (Diagram):

This conceptual figure contrasts the LLM-as-Planner and LLM-as-Formalizer paradigms. It visualizes how the latter converts natural language descriptions into PDDL files, which are then passed to a symbolic planner, with a focus on how varying levels of naturalness affect performance.

####  Figures 3 & 4 (Performance Graphs):

Figure 3 compares correctness between Planner and Formalizer across three domains. GPT-4o and O3-mini show a strong advantage for the Formalizer strategy, while all LLaMA variants essentially fail. Figure 4 illustrates a clear decline in performance as input descriptions become more natural, highlighting the limits of LLMs' commonsense reasoning.

####  Figures 5 & 6 (Error Analysis):

Figure 5 shows that open-source models often suffer from **syntax errors**, while GPT models tend to produce **semantically invalid PDDL**. Figure 6 provides fine-grained error breakdowns within domain files (e.g., wrong effects or missing predicates), offering actionable insights for improving formalization accuracy.

####  Table 1 (Prior Work Comparison):

This table demonstrates that previous work largely relied on **partial PDDL generation or templated inputs**, whereas this study is the first to evaluate full DF+PF generation under **naturally phrased descriptions**.

####  Appendix:

* Appendix A provides realistic dataset examples, reinforcing the experiment’s practical relevance.
* Appendices B–D contain prompt variations and few-shot/COT experiments, which show **no consistent performance gains**.
* Appendix E gives a **manual error classification** and examples, helping quantify failure patterns and guide model improvement.

---




<br/>
# refer format:     



@inproceedings{huang2025llmformalizer,
  title     = {On the Limit of Language Models as Planning Formalizers},
  author    = {Cassie Huang and Li Zhang},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2025},
  pages     = {4880--4904},
  address   = {Bangkok, Thailand},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2025.acl-long.242},
  doi       = {10.18653/v1/2025.acl-long.242}
}



Cassie Huang and Li Zhang. “On the Limit of Language Models as Planning Formalizers.” In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), 4880–4904. Bangkok, Thailand: Association for Computational Linguistics, 2025. https://aclanthology.org/2025.acl-long.242.



