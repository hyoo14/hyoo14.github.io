---
layout: post
title:  "[2024]ClinicalAgent Clinical Trial Multi-Agent System with Large Language Model-based Reasoning"  
date:   2025-06-12 01:34:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 


gpt4o를 다양하게 파인튜닝시켜서 스텝스텝 테스크 처리하도록 하는 것  



짧은 요약(Abstract) :    



대형 언어 모델(LLMs)과 다중 에이전트 시스템은 자연어 처리에서 뛰어난 성능을 보였지만, 임상시험 분야에서는 외부 지식 접근의 제한으로 어려움을 겪고 있습니다. 이를 해결하기 위해 최신 의료 데이터를 활용하는 도구의 접근성과 활용도를 높이는 **ClinicalAgent** 시스템을 제안합니다. 이 시스템은 GPT-4와 다중 에이전트 아키텍처, LEAST-TO-MOST 및 ReAct 추론 기술을 통합하여 임상시험 업무를 지원합니다. ClinicalAgent는 임상 맥락에서 LLM의 성능을 향상시킬 뿐만 아니라 새로운 기능도 제공합니다. 제안된 방법은 임상시험 결과 예측에서 PR-AUC 0.7908이라는 높은 성능을 달성하였으며, 기존 프롬프트 방식보다 0.3326만큼 성능이 향상되었습니다. 코드는 GitHub에서 공개되어 있습니다.


Large Language Models (LLMs) and multi-agent systems have shown impressive capabilities in natural language tasks but face challenges in clinical trial applications, primarily due to limited access to external knowledge. Recognizing the potential of advanced clinical trial tools that aggregate and predict based on the latest medical data, we propose an integrated solution to enhance their accessibility and utility. We introduce Clinical Agent System (ClinicalAgent), a clinical multi-agent system designed for clinical trial tasks, leveraging GPT-4, multi-agent architectures, LEAST-TO-MOST, and ReAct reasoning technology. This integration not only boosts LLM performance in clinical contexts but also introduces novel functionalities. The proposed method achieves competitive predictive performance in clinical trial outcome prediction (0.7908 PR-AUC), obtaining a 0.3326 improvement over the standard prompt method. Publicly available code can be found at [https://github.com/LeoYML/clinical-agent](https://github.com/LeoYML/clinical-agent).





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




###  시스템 개요 (3.1 Overview of ClinicalAgent)

* ClinicalAgent는 **GPT-4 기반 다중 에이전트 시스템**으로, 각 에이전트는 병원의 전문가들처럼 **역할 특화**됨.
* 에이전트는 **ReAct** (Recognition + Action + Context) 및 **LEAST-TO-MOST reasoning** 전략을 활용해 자연어 입력을 분석하고 의사결정을 수행.
* 주요 목표: 임상시험 결과 예측, 실패 사유 추정, 기간 예측 등.

---

###  주요 에이전트 역할 (3.2 Agent Roles)

| 에이전트                 | 역할 요약                                                 |
| -------------------- | ----------------------------------------------------- |
| **Planning Agent**   | 문제를 하위 문제로 분해하고 적절한 에이전트에 분배                          |
| **Efficacy Agent**   | DrugBank, HetioNet을 활용해 약물의 질병에 대한 효과 분석              |
| **Safety Agent**     | 과거 실패율, 이상반응을 기반으로 약물 안전성 평가                          |
| **Enrollment Agent** | BioBERT 기반 Transformer 모델로 등록 성공률 예측 (A.1에서 모델 구조 설명) |

---

###  외부 도구 호출 (3.3 External Tools)

* **DrugBank**: 약물의 화학적/약리학적 정보 검색
* **HetioNet**: 약물-질병 간 경로 탐색
* **ClinicalTrials.gov**: 과거 임상시험 데이터 기반 예측
* **LLM 생성 데이터**: 가상 약물 상호작용, 메커니즘 생성에 사용

---

###  AI 예측 모델들

* **Enrollment 모델**:

  * 입력: Inclusion/Exclusion 기준, 질병명, 약물명
  * 구조: BioBERT 임베딩 → Transformer 인코더 → Fully Connected → Sigmoid
  * 성능: ROC-AUC 0.7037, Accuracy 0.7689

* **Drug Risk Model**:

  * 약물의 실패 확률 = 과거 임상시험 성공률의 평균
  * dictionary lookup으로 구현됨

* **Disease Risk Model**:

  * 질병별 실패 확률 = 과거 trial 실패율로 계산됨
  * 유사 질병명 매칭 포함

---

###  추론 기술 통합 (3.4 Integration of Reasoning Technology)

* **ReAct**: 패턴 인식 → 행동 선택 → 맥락 고려
* **LEAST-TO-MOST**: 단순 → 복잡 순으로 문제 분해 및 처리
* 두 방식을 **결합**하여 복잡한 임상 문제 해결

---

###  전체 워크플로우 요약 (3.5 Workflow)

1. 문제 분석 및 하위 문제 분해 (Planning Agent)
2. 하위 문제 분배 (효능, 안전, 등록)
3. 각 에이전트가 외부 툴 활용해 예측
4. 결과 종합 및 최종 판단 (ReAct 기반)
5. 사용자에게 답변 제공

---

###  Architecture

ClinicalAgent is a **multi-agent system powered by GPT-4**, where each agent mimics a specialized medical expert. The architecture follows a **modular setup** where different agents are assigned roles:

* **Planning Agent**: Breaks down user queries using *Least-to-Most* reasoning.
* **Efficacy Agent**: Queries **DrugBank** and **HetioNet** to assess drug efficacy.
* **Safety Agent**: Analyzes historical trial safety data.
* **Enrollment Agent**: Uses a **hierarchical transformer model** with **BioBERT embeddings** to predict enrollment success.

---

###  External Tools & Data

* **DrugBank**: Chemical and pharmacological info.
* **HetioNet**: Biological interaction graph between drugs and diseases.
* **ClinicalTrials.gov**: Training and validation source for historical outcomes.
* **LLM-generated knowledge**: Used for hypothetical extensions.

---

###  Predictive Models

* **Enrollment Model**:

  * Architecture: BioBERT → Transformer Encoder → FC → Sigmoid
  * Trained on: historical trials with eligibility criteria, drugs, and diseases
  * Performance: ROC-AUC 0.7037, Accuracy 0.7689

* **Drug Risk Model**:

  * Precomputed drug success rates from past trials.

* **Disease Risk Model**:

  * Precomputed trial-based risk values per disease, includes fuzzy name matching.

---

###  Reasoning Strategies

* **ReAct**: Recognition → Action → Context-based decisions.
* **Least-to-Most**: Problem solving starts from simple to complex aspects.
* Combined, they allow structured and adaptive decision-making.

---

###  Workflow

1. **Decompose** problem (Planning Agent).
2. **Assign** sub-tasks to specialized agents.
3. **Agents call tools** (e.g., databases, models).
4. **Aggregate** findings.
5. **Reason** through integrated results (ReAct).
6. **Respond** with justified answers.




   
 
<br/>
# Results  





###  실험 목적

* ClinicalAgent의 임상시험 결과 예측 능력을 기존 LLM 프롬프트 방식과 전통 ML 모델들과 비교
* 주요 평가: **Clinical Trial Outcome Prediction** (이진 분류)

---

###  비교 아키텍처 (Baseline Methods)

| 모델                | 설명                               |
| ----------------- | -------------------------------- |
| **GBDT**          | BioBERT 임베딩 + LightGBM으로 예측      |
| **HAtten**        | BioBERT 임베딩 + 계층적 어텐션 + 2층 MLP   |
| **GPT-3.5/4**     | 표준 프롬프트 방식 (fine-tuning 없음)      |
| **ClinicalAgent** | 다중 에이전트 + GPT-4 + 외부 지식 기반 추론 통합 |

---

###  실험 데이터 및 설정

* Clinical Trial Outcome Prediction benchmark에서 **40개 학습 샘플**, **40개 테스트 샘플** 사용
* 서버 사양: AMD Ryzen 9 3950X, 64GB RAM, NVIDIA RTX 3080 Ti
* 동일한 랜덤 시드 사용
* OpenAI API 호출 비용 고려해 소규모로 진행

---

###  평가 메트릭

* **Accuracy**, **ROC-AUC**, **PR-AUC**
* **Precision**, **Recall**, **F1-score**

---

###  주요 결과 (표 요약)

| 모델                | Accuracy  | ROC-AUC   | PR-AUC    | Precision | Recall    | F1        |
| ----------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| GBDT              | 0.650     | 0.800     | **0.866** | 0.625     | **0.973** | **0.769** |
| HAtten            | **0.750** | 0.757     | **0.871** | **0.897** | 0.613     | 0.722     |
| GPT-3.5           | 0.525     | 0.591     | 0.433     | 0.417     | 0.667     | 0.513     |
| GPT-4             | 0.600     | 0.604     | 0.454     | 0.471     | 0.533     | 0.500     |
| **ClinicalAgent** | 0.700     | **0.834** | **0.791** | 0.571     | 0.800     | 0.667     |

→ GPT 단독 사용보다는 **ClinicalAgent가 월등한 성능 개선** 보임
→ 전통 ML (GBDT, HAtten)은 일부 PR-AUC 지표에서는 여전히 강력
→ HetioNet 또는 DrugBank를 제거하면 성능 하락 (ablation 결과)

---



###  Task

* **Clinical Trial Outcome Prediction**
  A binary classification task to determine if a trial will succeed or fail.

---

###  Baselines Compared

| Model             | Description                                                            |
| ----------------- | ---------------------------------------------------------------------- |
| **GBDT**          | Uses BioBERT embeddings with LightGBM                                  |
| **HAtten**        | Hierarchical Attention model with BioBERT embeddings + 2-layer MLP     |
| **GPT-3.5/4**     | Standard prompting, no fine-tuning or tools                            |
| **ClinicalAgent** | Multi-agent GPT-4 system with external tools and reasoning integration |

---

###  Dataset & Setup

* **40 training samples**, **40 test samples** from prior clinical trial outcome benchmark (\[Fu et al., 2022], \[2023])
* Hardware: AMD Ryzen 9 3950X, 64GB RAM, RTX 3080 Ti
* Python 3.8 + PyTorch; same seed for reproducibility
* API call cost constrained experiment size

---

### Metrics Used

* **Accuracy**
* **ROC-AUC (Receiver Operating Characteristic)**
* **PR-AUC (Precision-Recall Area Under Curve)**
* **Precision**, **Recall**, **F1-score**

---

###  Key Results

| Model             | Accuracy  | ROC-AUC   | PR-AUC    | Precision | Recall    | F1        |
| ----------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| GBDT              | 0.650     | 0.800     | **0.866** | 0.625     | **0.973** | **0.769** |
| HAtten            | **0.750** | 0.757     | **0.871** | **0.897** | 0.613     | 0.722     |
| GPT-3.5           | 0.525     | 0.591     | 0.433     | 0.417     | 0.667     | 0.513     |
| GPT-4             | 0.600     | 0.604     | 0.454     | 0.471     | 0.533     | 0.500     |
| **ClinicalAgent** | 0.700     | **0.834** | **0.791** | 0.571     | 0.800     | 0.667     |

* ClinicalAgent **significantly outperforms GPT prompting alone** across all metrics
* However, GBDT and HAtten models still show superior PR-AUC in some settings
* **Ablation study**: removing HetioNet or DrugBank causes performance drop, confirming their utility





<br/>
# 예제  




###  사용자 입력

* 사용자 질문: "제가 임상시험을 설계했는데 성공할 수 있을지 예측해 주세요."
* 제공된 정보:

  * 약물: **Aggrenox capsule** (아스피린 + 디피리다몰)
  * 질병: **뇌졸중 (cerebrovascular accident)**
  * 포함 기준 / 제외 기준: 텍스트로 제공됨

---

###  문제 분해 (Planning Agent)

* 문제를 **세 가지 하위 문제**로 분해:

  1. **등록 가능성 (Enrollment)** → Inclusion/Exclusion 기준 기반 예측
  2. **안전성 (Safety)** → 약물의 과거 실패율, 부작용 등
  3. **효능 (Efficacy)** → HetioNet, DrugBank 활용한 약물-질병 연결성 분석

---

###  각 에이전트의 처리 결과

| 에이전트             | 수행 내용 및 결과                                  |
| ---------------- | ------------------------------------------- |
| Enrollment Agent | 예측된 등록 실패 확률: **0.3597** (중간 수준의 어려움)       |
| Safety Agent     | **과거 임상시험 실패율 1.0** (즉, 성공한 적 없음)           |
| Efficacy Agent   | Aggrenox는 항혈소판 및 혈관확장 효과로 뇌졸중 예방에 **효능 있음** |

---

###  최종 추론 (Reasoning Agent)

* 중간 난이도의 등록
* **높은 실패율과 안전성 문제**
* 효능은 긍정적이지만, 전체적으로 성공 가능성이 낮다고 판단
* **예측 성공 확률: 0.0**
* 실제 정답도 실패(ground truth = 0)

---


###  User Input

* Query: “I have designed a clinical trial. Can you predict if it will succeed?”
* Input features:

  * Drug: **Aggrenox capsule** (aspirin + dipyridamole)
  * Disease: **Cerebrovascular accident (stroke)**
  * Inclusion and exclusion criteria: provided in text form

---

###  Decomposition (Planning Agent)

The Planning Agent breaks the problem into 3 sub-tasks:

1. **Enrollment Prediction** – Based on the inclusion/exclusion criteria
2. **Drug Safety Analysis** – Using historical clinical trial data
3. **Drug Efficacy Evaluation** – Using DrugBank and HetioNet pathways

---

###  Agent Outputs

| Agent            | Output                                                                                          |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| Enrollment Agent | Estimated enrollment failure rate: **0.3597** (moderate difficulty)                             |
| Safety Agent     | Historical failure rate of Aggrenox: **1.0** (i.e., always failed)                              |
| Efficacy Agent   | Aggrenox has biological mechanisms aligned with stroke prevention (antiplatelet + vasodilation) |

---

###  Final Reasoning (Reasoning Agent)

* Enrollment poses moderate difficulty
* **Safety concerns are severe** due to 100% historical failure rate
* Although efficacy is promising, overall success is unlikely
* **Predicted trial success rate: 0.0**
* Ground truth: 0 (failure) → **correct prediction**





<br/>  
# 요약   



ClinicalAgent는 GPT-4와 다중 에이전트 구조, 외부 지식베이스, 고급 추론 기법(ReAct 및 LEAST-TO-MOST)을 결합한 임상시험 예측 시스템입니다. 기존 프롬프트 기반 GPT-4보다 PR-AUC가 0.3326 향상되는 등 전통 ML 및 LLM 대비 경쟁력 있는 성능을 보였습니다. 실제 임상시험 사례에서 약물의 안전성과 효능, 등록 가능성을 종합 분석해 임상시험 실패를 정확히 예측했습니다.

---

ClinicalAgent is a clinical trial prediction system combining GPT-4 with a multi-agent architecture, external knowledge sources, and advanced reasoning techniques like ReAct and LEAST-TO-MOST. It outperforms standard prompting methods, showing a 0.3326 improvement in PR-AUC and competitive results against traditional machine learning models. In a real-world case, ClinicalAgent correctly predicted trial failure by analyzing drug efficacy, safety history, and enrollment feasibility.





<br/>  
# 기타  




### Figure 1: ClinicalAgent 시스템 구조도

* **설명**: 임상시험 성공 예측 문제를 **등록(Enrollment)**, **안전성(Safety)**, \*\*효능(Efficacy)\*\*의 세 하위 문제로 분해하여 각 에이전트가 처리하는 구조를 시각화.
* **인사이트**: GPT-4 기반 에이전트들이 협력하여 복잡한 의학적 문제를 분산 처리하고 통합 판단하는 과정을 이해하는 데 유용함.

---

###  Table 1: 실제 임상시험 사례

* **설명**: Aggrenox 캡슐을 사용한 임상시험 예를 제시. 각 에이전트가 도구 호출을 통해 예측 수행.
* **인사이트**: Safety Agent가 **100% 실패율**을 탐지하고, Enrollment Agent는 중간 난이도 등록을 예측하며, 최종적으로 **실패 예측 (성공률 0.0)** 을 정확히 도출.

---

###  Table 2: 다양한 모델의 성능 비교

* **설명**: GBDT, HAtten, GPT-3.5, GPT-4, ClinicalAgent 간의 Accuracy, ROC-AUC, PR-AUC 등 평균 성능 비교 (5회 평균 ± 표준편차).
* **인사이트**:

  * ClinicalAgent가 **ROC-AUC (0.834)**, \*\*PR-AUC (0.791)\*\*에서 GPT 프롬프트 방식보다 확연히 우수.
  * GBDT, HAtten은 일부 지표에서 더 높지만 LLM 기반 에이전트와 결합 가능성 존재.

---

###  Table 3: Few-shot 학습의 영향

* **설명**: Few-shot 학습을 적용한 ClinicalAgent와 미적용 버전 비교.
* **인사이트**:

  * **Few-shot 적용** 시 PR-AUC 향상 (0.679 → 0.791), 즉 **소규모 예제 기반 학습이 예측 성능에 긍정적 효과**.

---

###  Table 4: 외부 지식 제거의 영향 (Ablation Study)

* **설명**: HetioNet 또는 DrugBank를 제거한 ClinicalAgent의 성능 변화.
* **인사이트**:

  * **DrugBank 제거 시 성능 급락** (Accuracy 0.45, F1 0.267), 즉 **약물 지식베이스가 핵심적 역할**을 한다는 점 강조.

---

###  Appendix A.1: Enrollment 예측 모델 구조

* **설명**: BioBERT 임베딩 + Transformer 인코더 + FC + 시그모이드 아키텍처.
* **인사이트**: 임상 기준 텍스트로부터 등록 성공률을 예측할 수 있는 **모듈화된 신경망 구조** 제공.

---

###  Appendix A.2: 함수 호출 정의

* **설명**: DrugBank 및 HetioNet API 호출 형식 (JSON 형태) 제공.
* **인사이트**: LLM이 **도구 자동 호출 (tool calling)** 기능을 사용하여 외부 DB와 동적으로 연결 가능함을 실증.

---



###  Figure 1: ClinicalAgent Architecture Diagram

* **Description**: Visualizes how a clinical trial prediction task is decomposed into three subtasks: Enrollment, Safety, and Efficacy—each handled by a specialized agent.
* **Insight**: Demonstrates how GPT-4 agents coordinate in a modular system to process and integrate complex clinical reasoning.

---

### Table 1: Real Clinical Trial Case

* **Description**: Example with Aggrenox capsule used in a trial for cerebrovascular accident. Each agent predicts using tool calls.
* **Insight**: Safety Agent finds a 100% failure rate; Enrollment Agent estimates moderate difficulty; the Reasoning Agent correctly predicts **trial failure (success rate = 0.0)**.

---

###  Table 2: Model Comparison

* **Description**: Shows average ± std. deviation performance of GBDT, HAtten, GPT-3.5/4, and ClinicalAgent across metrics like Accuracy, ROC-AUC, PR-AUC.
* **Insight**:

  * **ClinicalAgent achieves the highest ROC-AUC (0.834) and strong PR-AUC (0.791)**.
  * Traditional models like GBDT/HAtten excel in some metrics but lack interpretability and modularity.

---

###  Table 3: Impact of Few-shot Learning

* **Description**: Compares ClinicalAgent with and without few-shot learning.
* **Insight**: **Few-shot version improves PR-AUC (0.679 → 0.791)**, confirming the importance of few examples in reasoning accuracy.

---

###  Table 4: Ablation Study (HetioNet & DrugBank Removal)

* **Description**: Shows performance drop when either HetioNet or DrugBank is removed.
* **Insight**:

  * **Removing DrugBank causes major decline** (Accuracy = 0.45, F1 = 0.267), highlighting the vital role of drug knowledge bases in prediction.

---

###  Appendix A.1: Enrollment Model Architecture

* **Description**: Model combines BioBERT embeddings, a Transformer encoder, and a sigmoid-activated FC layer to predict enrollment success.
* **Insight**: Provides a **modular neural approach** to learn from inclusion/exclusion criteria and related clinical inputs.

---

### Appendix A.2: Tool Calling Definitions

* **Description**: JSON schema for LLM-driven API calls to DrugBank and HetioNet.
* **Insight**: Confirms that LLMs can dynamically interface with biomedical databases using structured tool call mechanisms.





<br/>
# refer format:     



@inproceedings{yue2024clinicalagent,
  title     = {ClinicalAgent: Clinical Trial Multi-Agent System with Large Language Model-based Reasoning},
  author    = {Ling Yue and Sixue Xing and Jintai Chen and Tianfan Fu},
  booktitle = {Proceedings of the 15th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics (BCB ’24)},
  year      = {2024},
  pages     = {1--10},
  address   = {Shenzhen, China},
  publisher = {ACM},
  doi       = {10.1145/3698587.3701359},
  url       = {https://doi.org/10.1145/3698587.3701359}
}



Ling Yue, Sixue Xing, Jintai Chen, and Tianfan Fu. “ClinicalAgent: Clinical Trial Multi-Agent System with Large Language Model-based Reasoning.” Proceedings of the 15th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics (BCB ’24), Shenzhen, China, November 22–25, 2024. New York, NY: ACM. https://doi.org/10.1145/3698587.3701359.  

