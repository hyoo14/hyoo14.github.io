---
layout: post
title:  "[2025]On the Limit of Language Models as Planning Formalizers"  
date:   2025-04-08 01:58:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


LLM으로  PDDL을 생성   


짧은 요약(Abstract) :    




이 논문은 대형 언어 모델(LLM)이 실제 환경에서 실행 가능하고 검증 가능한 계획을 만드는 데 실패한다는 점에서 출발합니다. 최근 연구는 LLM을 **직접 계획을 생성하는 도구(planner)** 로 쓰기보다는, **계획 도메인을 형식화하는 도구(formalizer)** 로 사용하는 방법에 주목하고 있습니다. 즉, LLM이 자연어로 주어진 환경 설명을 기반으로 계획 도메인 정의 언어(PDDL) 형태로 형식화하고, 그 형식화된 내용을 계획 솔버에 전달해 실제 계획을 구하는 방식입니다.

기존 연구는 부분적인 PDDL만 생성하거나 인위적인 템플릿 언어에 기반했지만, 이 논문은 자연어 수준의 다양한 표현을 바탕으로 전체 PDDL을 생성하는 실험을 진행합니다. 그 결과, 충분히 큰 모델은 실제 계획을 직접 생성하는 것보다 PDDL 형식화를 더 잘 수행하며, 문장 표현이 자연스러울수록 성능이 감소하는 경향이 나타났습니다. 이에 대해 자세한 오류 분석도 함께 제공합니다.

---



Large Language Models have been shown to fail to create executable and verifiable plans in grounded environments. An emerging line of work shows success in using LLM as a formalizer to generate a formal representation (e.g., PDDL) of the planning domain, which can be deterministically solved to find a plan. We systematically evaluate this methodology while bridging some major gaps. While previous work only generates a partial PDDL representation given templated and thus unrealistic environment descriptions, we generate the complete representation given descriptions of various naturalness levels. Among an array of observations critical to improve LLMs’ formal planning ability, we note that large enough models can effectively formalize descriptions as PDDL, outperforming those directly generating plans, while being robust to lexical perturbation. As the descriptions become more natural-sounding, we observe a decrease in performance and provide detailed error analysis.

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




**1. 백본 모델 (Backbone Model)**  
논문은 다양한 대형 언어 모델(LLM)을 사용하여 실험합니다. GPT 계열의 **GPT-4o**, **GPT-4o-mini**, **o3-mini** 같은 **비공개 모델들**과, **LLaMA-3**, **Gemma**, **DeepSeek** 등 **오픈소스 모델들**(최대 405B 파라미터까지)을 함께 비교 분석합니다.

**2. 구조 (Structure)**  
논문에서는 두 가지 구조적 접근을 비교합니다:  
- **LLM-as-Planner**: 모델이 직접 계획을 생성  
- **LLM-as-Formalizer**: 모델이 자연어 설명을 바탕으로 **PDDL 형식의 도메인/문제 정의(Domain File, Problem File)** 를 생성하고, 이후 **솔버가 계획을 생성**함.  
이 연구는 후자(LMM-as-Formalizer)에 중점을 두며, **전체 PDDL (DF + PF)** 생성을 시도한 최초의 체계적 분석입니다.

**3. 학습 데이터 (Training Data)**  
모델 자체는 사전학습된 상태이며, 이 연구는 **제로샷(zero-shot)** 방식으로 진행됩니다. 즉, 모델을 추가로 fine-tuning 하지 않고 **자연어 입력만 주어진 상태에서 평가**합니다. 입력은 다양한 자연스러움 수준(Heavily Templated, Moderately Templated, Natural)으로 구성된 **도메인 설명(DD)** 과 **문제 설명(PD)** 입니다.

**4. 테스크 (Task)**  
LLM이 자연어 기반 도메인 설명(DD)과 문제 설명(PD)을 입력받아, 형식적인 계획 문제(PDDL)를 생성하고 이를 통해 실행 가능한 계획을 만드는 것이 목표입니다. 사용된 계획 시뮬레이션은 **BlocksWorld, Logistics, Barman, MysteryBlocksWorld** 총 4종이며, 각각은 다른 복잡도와 구조를 지닙니다. 특히 MysteryBlocksWorld는 변수 이름 등이 의미 없는 단어로 치환된 버전으로, **모델의 일반화 능력과 패턴 기억 여부를 검증**하는 데 사용됩니다.

---



**1. Backbone Models**  
The study evaluates a diverse set of large language models (LLMs), including both **closed-source models** such as **GPT-4o**, **GPT-4o-mini**, and **o3-mini**, as well as **open-source models** like **LLaMA-3**, **Gemma**, and **DeepSeek** (up to 405B parameters).

**2. Structure**  
Two main paradigms are compared:
- **LLM-as-Planner**: The model generates the plan directly from text.
- **LLM-as-Formalizer**: The model converts the natural language environment descriptions into a full **PDDL representation** (Domain File + Problem File), and a symbolic planner solves it.  
This paper focuses on the **LLM-as-Formalizer** approach and is the first to systematically evaluate models generating **the entire PDDL specification**.

**3. Training Data**  
The models are used **in a zero-shot setting**, meaning no additional fine-tuning is performed. They are evaluated based on their ability to process **natural language domain descriptions (DD)** and **problem descriptions (PD)** of varying levels of naturalness (Heavily Templated, Moderately Templated, Natural).

**4. Task**  
Given natural language inputs describing an environment and a goal (DD + PD), the LLM must generate a formal planning problem in PDDL and enable a planner to derive a valid execution plan. The study uses four simulated domains: **BlocksWorld, Logistics, Barman, and MysteryBlocksWorld**. The latter is especially designed to assess models' robustness to **lexical perturbations and memorization avoidance** by masking meaningful identifiers.

---




   
 
<br/>
# Results  






**1. 테스트 데이터 (Test Data)**  
총 **4개의 시뮬레이션 도메인**이 사용되었습니다:

- **BlocksWorld**: 블록을 쌓고 옮기는 고전적인 계획 문제.
- **Logistics**: 트럭과 비행기로 소포를 목적지로 운반.
- **Barman**: 칵테일을 제조하는 복잡한 조작 환경.
- **MysteryBlocksWorld**: BlocksWorld에서 모든 이름이 무의미한 단어로 치환된 버전 → 모델의 암기 의존성 확인용.

각 도메인별로 100개의 문제 인스턴스를 만들었으며, **설명 방식에 따라 세 가지 데이터 변형**(Heavily Templated, Moderately Templated, Natural)이 있습니다.  
특히 **Natural** 버전은 사람이 이해하기 쉽게 작성되었지만, 모델에게는 더 어려운 자연어 표현을 포함하고 있습니다.

---

**2. 경쟁 모델 (Baselines / Comparative Models)**  
다양한 대형 언어 모델이 두 가지 방식으로 테스트되었습니다:

- **LLM-as-Planner**: LLM이 직접 계획을 생성.
- **LLM-as-Formalizer**: LLM이 PDDL을 생성하고, 이를 기존 계획 솔버(planner)에 넘김.

사용된 모델들:
- **GPT 계열** (GPT-4o, GPT-4o-mini, o3-mini)
- **오픈소스 모델들**: LLaMA-3 (8B, 70B, 405B), Gemma-2 (9B, 27B), DeepSeek (8B, 70B)

GPT-4o는 전반적으로 가장 높은 성능을 보였고, 오픈소스 모델은 **문법 오류(Syntax errors)** 와 **구문 해석 오류(Semantic errors)** 로 인해 성능이 낮았습니다.

---

**3. 평가 지표 (Metrics)**  
다음 두 가지 지표를 사용해 평가했습니다:

- **Solvability (해결 가능성)**: 모델이 생성한 PDDL이 **계획 솔버에서 실행 가능한지 여부**.
- **Correctness (정확도)**: 솔버가 실행한 계획이 **문제 정의와 일치하는지 여부**.

 참고: 계획 자체는 ground-truth plan과 비교하지 않음. 계획은 여러 정답이 있을 수 있기 때문.

---

** 주요 결과 요약**
- GPT-4o는 **BlocksWorld**의 Natural 설명에서도 64%의 문제를 해결하고, 그 중 60%가 정확함.
- **LLM-as-Formalizer 방식이 대부분의 경우 LLM-as-Planner보다 성능이 우수**.
- 설명이 **자연스러워질수록 성능이 급감**. 모델이 암시적 정보(예: "clear" 조건 등)를 유추하지 못함.
- MysteryBlocksWorld에서는 GPT 모델은 여전히 높은 성능을 유지 → **기억이 아닌 일반화 능력**에 기반한 결과로 해석됨.

---


**1. Test Data**  
The authors use four planning simulation domains:

- **BlocksWorld**: Stack and unstack blocks.
- **Logistics**: Transport packages using trucks and airplanes.
- **Barman**: Make cocktails with two robotic arms.
- **MysteryBlocksWorld**: A variation of BlocksWorld with all identifiers replaced by meaningless tokens, used to test memorization.

Each domain includes 100 problems, and three levels of natural language description are used: **Heavily Templated**, **Moderately Templated**, and **Natural**. Natural descriptions are more realistic and human-like but more difficult for LLMs to parse.

---

**2. Competitive Models (Baselines)**  
Models were evaluated using two paradigms:

- **LLM-as-Planner**: The LLM generates a plan directly.
- **LLM-as-Formalizer**: The LLM generates PDDL (Domain + Problem), and a symbolic planner produces the plan.

Models evaluated:
- **Closed-source**: GPT-4o, GPT-4o-mini, o3-mini
- **Open-source**: LLaMA-3 (8B/70B/405B), Gemma-2 (9B/27B), DeepSeek (8B/70B)

GPT-4o showed the strongest performance, while open-source models struggled due to frequent syntax and semantic errors in PDDL generation.

---

**3. Evaluation Metrics**  
Two main metrics were used:

- **Solvability**: Whether the generated PDDL can be processed by a planner.
- **Correctness**: Whether the resulting plan meets the specified goals.

Note: Plans are not compared against gold plans since multiple valid solutions may exist.

---

** Key Findings**
- GPT-4o correctly solved 60 out of 64 solvable problems in the **Natural BlocksWorld** setting.
- **LLM-as-Formalizer consistently outperformed LLM-as-Planner**.
- Model performance dropped as descriptions became more natural, showing difficulty in inferring implicit rules (e.g., whether a block is “clear”).
- GPT models performed well even on **MysteryBlocksWorld**, indicating robustness to lexical changes and minimal reliance on memorization.




<br/>
# 예제  



---

###  실제 훈련 데이터 예시  
이 논문에서는 **추가적인 fine-tuning을 하지 않고**, 사전 학습된 LLM을 **zero-shot**으로 테스트합니다.  
즉, “훈련 데이터”라는 개념보다는 **모델에게 주어진 입력(prompt)** 이 테스트와 유사한 자연어 설명들입니다.  
예를 들어, **BlocksWorld**의 한 설명은 다음과 같습니다:

####  자연어 도메인 설명 (DD – Natural)
> 블록월드 게임은 다양한 색상의 블록들로 구성되어 있으며, 블록들을 서로 쌓거나 테이블 위에 놓을 수 있습니다. 목표는 초기 상태에서 주어진 최종 상태로 블록들을 이동시키는 것입니다. 사용할 수 있는 주요 동작은: 블록을 집기, 블록을 테이블 위에 놓기, 블록을 다른 블록 위에 쌓기입니다.

---

###  실제 테스트 데이터 예시  
모델은 위와 같은 도메인 설명(DD)과 함께 **문제 설명(PD)** 도 받습니다.

####  문제 설명 예시 (PD – Natural)
> 이 문제에서는 네 개의 블록(빨강, 파랑, 초록, 노랑)이 있습니다. 시작 상태는 빨간 블록이 테이블 위에 있고, 파란 블록이 빨간 블록 위에 있으며, 초록 블록은 테이블 위, 노란 블록은 초록 블록 위에 있습니다. 목표는 빨간 블록이 테이블 위에 있고, 그 위에 초록, 노랑, 파랑 블록이 차례로 쌓이도록 만드는 것입니다.

---

###  실제 테스크 Input/Output 예시

####  모델 Input (자연어 설명)
- 도메인 설명 (DD) + 문제 설명 (PD)  
- 자연어로 서술된 환경 및 목표 상태.

####  모델 Output (예상되는 출력)
- PDDL 형식의 도메인 파일 (DF):
```lisp
(:action pickup
  :parameters (?ob)
  :precondition (and (clear ?ob) (on-table ?ob) (arm-empty))
  :effect (and (holding ?ob) (not (clear ?ob)) (not (on-table ?ob)) (not (arm-empty)))
)
```
- PDDL 형식의 문제 파일 (PF):
```lisp
(:init
  (on-table red)
  (on blue red)
  (clear blue)
  (on-table green)
  (on yellow green)
  (clear yellow)
  (arm-empty)
)
(:goal (and
  (on-table red)
  (on green red)
  (on yellow green)
  (on blue yellow)
))
```

이후 이 PDDL들이 **계획 솔버(planner)** 에 의해 입력되어 실제 계획:
```lisp
(unstack blue red)
(putdown blue)
(pickup green)
(stack green red)
(pickup yellow)
(stack yellow green)
(pickup blue)
(stack blue yellow)
```
처럼 반환됩니다.



---

###  Training Data Example  
This paper does **not use fine-tuning**; models are evaluated in a **zero-shot setting**.  
Thus, “training data” refers more to the **prompt input** (natural language descriptions) used during inference.

####  Natural Domain Description (DD – Natural)
> The Blocksworld game involves a set of colored blocks that can be stacked on top of each other or placed on a table. The objective is to move blocks from an initial configuration to a goal configuration using legal actions such as picking up, putting down, or stacking blocks.

---

###  Test Data Example  
A domain description (DD) is paired with a **problem description (PD)**:

####  Problem Description Example (PD – Natural)
> In this game, there are four blocks: red, blue, green, and yellow. Initially, the red block is on the table, the blue block is on top of the red block, the green block is on the table, and the yellow block is on top of the green block. The goal is to have the red block on the table, green on red, yellow on green, and blue on yellow.

---

### Task Input/Output Example

####  Model Input
- Natural language domain + problem description (DD + PD)

####  Model Output (Expected)
- **PDDL Domain File** example:
```lisp
(:action pickup
  :parameters (?ob)
  :precondition (and (clear ?ob) (on-table ?ob) (arm-empty))
  :effect (and (holding ?ob) (not (clear ?ob)) (not (on-table ?ob)) (not (arm-empty)))
)
```
- **PDDL Problem File** example:
```lisp
(:init
  (on-table red)
  (on blue red)
  (clear blue)
  (on-table green)
  (on yellow green)
  (clear yellow)
  (arm-empty)
)
(:goal (and
  (on-table red)
  (on green red)
  (on yellow green)
  (on blue yellow)
))
```

Then, the symbolic **planner** generates the final **plan**:
```lisp
(unstack blue red)
(putdown blue)
(pickup green)
(stack green red)
(pickup yellow)
(stack yellow green)
(pickup blue)
(stack blue yellow)
```



<br/>  
# 요약   




이 논문은 대형 언어 모델이 자연어로 주어진 환경을 형식 언어(PDDL)로 변환하는 **formalizer**로서의 가능성을 체계적으로 평가한다.  
GPT-4o는 복잡한 자연어 설명에서도 높은 정확도로 PDDL을 생성하며, 직접 계획을 생성하는 방식보다 일관되게 우수한 성능을 보였다.  
실제 예제에서는 자연어 환경 설명과 목표를 입력으로 받아, 정형화된 PDDL 파일과 실행 계획을 성공적으로 출력한다.

---



This paper systematically evaluates the use of large language models as **formalizers** that convert natural language environment descriptions into formal PDDL representations.  
GPT-4o consistently outperforms direct plan-generation approaches, showing strong accuracy even with complex, natural-sounding inputs.  
In real examples, the models successfully generate full domain/problem PDDL files and executable plans from natural language descriptions and goals.



<br/>  
# 기타  



###  그래프 (Figures)  
논문에는 다양한 그래프가 포함되어 있어 모델 성능을 시각적으로 보여줍니다.  
- **Figure 3**: 각 도메인별 설명 수준(Heavily Templated, Moderately Templated, Natural)에 따른 solvability 및 correctness 변화 추이를 나타냄.  
  → 자연어가 복잡해질수록 성능이 급감하는 경향이 시각적으로 잘 드러남.  
- **Figure 4**: 설명을 구성하는 문장의 문장 순서 변화에 따른 성능 변화 분석 → **GPT-4o는 문장 순서에 대해 가장 강건함**을 보여줌.

###  테이블 (Tables)  
- **Table 1**: 평가 대상 LLM들의 크기, 공개 여부(open/closed), parameter 수 등 스펙 요약.  
- **Table 2**: 각 모델이 생성한 도메인(PDDL Domain File)과 문제(PDDL Problem File)의 문법 정확도 및 플래너 실행 가능성(Valid PDDL 여부)을 비교.  
- **Table 3**: 실제 계획이 문제 목표와 일치하는 정확도(Correctness) 비율 제시.  
- **Table 4**: 다양한 lexical perturbation (문장 순서, 동의어 치환 등)에 대한 성능 변화 요약.

###  어펜딕스 (Appendix)  
- **Appendix A**: 각 도메인(BlocksWorld, Barman 등)의 구조적 구성과 목표 설명 방식 예시를 상세히 제공.  
- **Appendix B**: 생성된 PDDL 예제와 오류 케이스들을 포함하여 정제되지 않은 모델 출력을 공유함.  
- **Appendix C**: 사용된 평가 기준과 플래너 환경 구성 방법에 대한 구체적인 설명.  



###  Figures  
- **Figure 3** shows solvability and correctness across different levels of naturalness in domain descriptions.  
  → Performance declines significantly as language becomes more natural.  
- **Figure 4** analyzes robustness to sentence ordering changes; **GPT-4o proves most resilient** among models.

###  Tables  
- **Table 1** summarizes the evaluated LLMs: model size, open/closed-source status, number of parameters.  
- **Table 2** compares syntax validity and planner acceptance (valid PDDL generation) across models.  
- **Table 3** reports the percentage of generated plans that are semantically correct with respect to the goals.  
- **Table 4** presents performance variations under lexical perturbations like sentence shuffle or synonym replacement.

###  Appendix  
- **Appendix A** provides domain-specific structure and goal formulation examples for each planning environment.  
- **Appendix B** includes raw model outputs, both successful and erroneous, including full PDDL files.  
- **Appendix C** explains evaluation procedures and planner configurations in detail.  



---


<br/>
# refer format:     



@article{huang2024limit,
  title={On the Limit of Language Models as Planning Formalizers},
  author={Huang, Cassie and Zhang, Li},
  journal={arXiv preprint arXiv:2412.09879},
  year={2024},
  url={https://arxiv.org/abs/2412.09879}
}




Cassie Huang and Li Zhang. "On the Limit of Language Models as Planning Formalizers." arXiv preprint arXiv:2412.09879, 2024. https://arxiv.org/abs/2412.09879.  





