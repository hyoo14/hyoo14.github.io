---
layout: post
title:  "[2025]TruthTorchLM: A Comprehensive Library for Predicting Truthfulness in LLM Outputs"  
date:   2025-07-14 23:29:40 -0800
categories: study
---

{% highlight ruby %}


한줄 요약: 

불확실성 기반 접근 + 모델 간 유사도 수준, 정답과의 유사도 등 다양한 추가적 접근 방안 포함하는 프레임워크 제안  


기타 지식:

SemanticEntropy, Eccentricity, KernelLanguageEntropy는 샘플링 기반 UQ 기법으로, 출력을 여러 번 생성하여 그 분포를 비교합니다.

VerbalizedConfidence는 모델이 스스로 얼마나 확신을 가지고 응답했는지를 언어적 표현 기반으로 추론합니다.  

SelfDetection은 “이 질문에 여러 번 답했을 때 결과가 일관적인가?”를 측정합니다 → 자기 모순 탐지에 가까움. 

SAR, AttentionScore는 내부 attention map을 기반으로 모델 내부 확신의 정렬도를 평가합니다 (화이트박스 접근 필요).  



짧은 요약: 

  


대형 언어 모델(LLM)은 종종 사실과 다른 응답을 생성하는데, 특히 중요한 분야에서는 이러한 응답의 진실성을 예측하는 것이 매우 중요합니다. 이에 따라 TruthTorchLM이라는 오픈소스 Python 라이브러리가 제안되었습니다. 이 라이브러리는 **30개 이상의 진실성 예측 기법(Truth Methods)**을 제공하며, 기존 도구들이 문서 기반 검증이나 불확실성 기반 접근에 국한된 반면, TruthTorchLM은 계산 비용, 모델 접근 수준(블랙박스/화이트박스 등), 외부 문서 사용 여부, 감독 학습 여부 등 다양한 조건을 포괄하는 확장 가능한 프레임워크를 제공합니다. HuggingFace 및 LiteLLM과도 호환되며, 진실성 점수의 생성, 평가, 보정, 장문의 진실성 판별까지 하나의 통합 인터페이스로 다룰 수 있습니다. 또한 TriviaQA, GSM8K, FactScore-Bio 데이터셋을 활용한 실험도 포함되어 있습니다.



Generative Large Language Models (LLMs) inevitably produce untruthful responses. Accurately predicting the truthfulness of these outputs is critical, especially in high-stakes settings. To accelerate research in this domain and make truthfulness prediction methods more accessible, we introduce TruthTorchLM, an open-source, comprehensive Python library featuring over 30 truthfulness prediction methods, which we refer to as Truth Methods.
Unlike existing toolkits such as Guardrails (guardrails-ai), which focus solely on document-grounded verification, or LM-Polygraph (Fadeeva et al., 2023), which is limited to uncertainty-based methods, TruthTorchLM offers a broad and extensible collection of techniques. These methods span diverse trade-offs in computational cost, access level (e.g., black-box vs. white-box), grounding document requirements, and supervision type (self-supervised or supervised).
TruthTorchLM is seamlessly compatible with both HuggingFace and LiteLLM, enabling support for locally hosted and API-based models. It also provides a unified interface for generation, evaluation, calibration, and long-form truthfulness prediction, along with a flexible framework for extending the library with new methods.
We conduct an evaluation of representative truth methods on three datasets, TriviaQA, GSM8K, and FactScore-Bio.



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






이 논문은 \*\*TruthTorchLM (TTLM)\*\*이라는 진실성 예측 오픈소스 라이브러리를 제안합니다. 이 라이브러리는 아래와 같은 구조적 요소 및 메서드 기반으로 구성되어 있습니다:

1. **Truth Methods (진실성 예측 기법)**

   * LLM이 생성한 응답이 진실인지 아닌지를 **사후적으로(score 기반으로) 평가**하는 다양한 알고리즘들을 포함합니다.
   * 예: 불확실성 기반(UQ), 문서 근거 기반(NLI), 다중 LLM 협업(Multi-LLM), 지도 학습, 자기지도학습 등.
   * 접근 수준, 감독 여부, 외부 문서 사용 여부, 샘플링 필요 여부 등에 따라 30가지 이상의 기법이 구현되어 있습니다.

2. **Unified Generation Interface (통합 생성 인터페이스)**

   * HuggingFace 및 LiteLLM과 호환되어 로컬/원격 모델을 동일한 방식으로 진실성 예측할 수 있도록 지원합니다.
   * `generate_with_truth_value()` 함수를 통해 생성 + 진실성 평가를 함께 수행할 수 있음.

3. **Evaluation and Calibration (평가 및 점수 보정)**

   * 기존 진실성 점수가 스케일이나 범위가 달라 비교가 어려운 문제를 해결하기 위해, **Isotonic Regression**이나 **Min-Max 정규화**를 사용해 0\~1 범위로 보정합니다.
   * 평가 지표로는 AUROC, PRR, Accuracy 등을 사용.

4. **Long-form Truthfulness Prediction (장문 응답 평가 기능)**

   * 응답이 여러 개의 사실 주장(factual claims)으로 구성된 경우, 이를 분해한 뒤 **각 주장별로 진실성 점수를 부여**합니다.
   * 이때 Claim Check Methods가 사용되며, 기존 truth method를 래핑하거나 claim 전용 평가 모델을 사용할 수 있음.

5. **모델 및 데이터셋**

   * 오픈모델(LLaMA-3-8B)과 폐쇄모델(GPT-4o-mini) 양쪽 모두 사용 가능.
   * 사용된 주요 데이터셋: TriviaQA, GSM8K, FactScore-Bio.

---



This paper presents **TruthTorchLM (TTLM)**, an open-source library designed to predict the truthfulness of outputs from large language models (LLMs). The core components and methodology include:

1. **Truth Methods**

   * These are post-hoc algorithms that assign a **truth score** to the generated outputs of LLMs.
   * The library includes 30+ methods, such as uncertainty quantification (UQ), document-grounded methods (e.g., NLI), multi-LLM collaboration, supervised classification, and self-supervised techniques.
   * Each method is categorized by access level (black-box, grey-box, white-box), supervision type, document requirement, and sampling usage.

2. **Unified Generation Interface**

   * TTLM supports both HuggingFace (local models) and LiteLLM (API-based models), allowing users to apply truth prediction in a unified manner.
   * The `generate_with_truth_value()` function jointly performs response generation and truth scoring.

3. **Evaluation and Calibration**

   * Since different truth methods produce scores on different scales, **score normalization** (e.g., Isotonic Regression, Min-Max scaling) is applied for meaningful comparison and ensemble use.
   * Evaluation metrics include AUROC, Prediction Rejection Ratio (PRR), F1, Accuracy, etc.

4. **Long-form Truthfulness Prediction**

   * For outputs containing **multiple factual claims**, TTLM decomposes the text and evaluates each claim individually.
   * Claim Check Methods include wrappers around existing methods and dedicated claim-level predictors.

5. **Models and Datasets**

   * The system supports both open-weight models (e.g., LLaMA-3-8B) and closed API-based models (e.g., GPT-4o-mini).
   * Evaluation datasets include **TriviaQA**, **GSM8K**, and **FactScore-Bio** for both short-form and long-form truthfulness assessment.








   
 
<br/>
# Results  




---

**실험 설정:**

* **모델:**

  *  LLaMA-3-8B (오픈모델, white-box 접근 가능)
  *  GPT-4o-mini (API 기반, black-box 접근만 가능)

* **데이터셋:**

  * **TriviaQA**: 일반 지식 질문 (단문 정답)
  * **GSM8K**: 수학적 사고 기반의 초등수준 문제 (단문 정답)
  * **FactScore-Bio**: 위키 기반 전기(biography) 문답 (장문 + 다중 주장 포함)

* **평가지표:**

  * AUROC (사실/비사실 분류 정확도)
  * PRR (Prediction Rejection Ratio, 불확실한 예측을 거르는 능력)

**핵심 결과:**

1. **전반적 경향:**

   * LLaMA-3-8B가 내부 상태 접근이 가능해서 더 강력한 성능을 보임
   * 장문 평가인 FactScore-Bio에서는 모든 기법의 성능이 떨어짐 (다중 주장 판단의 어려움)

2. **TriviaQA (단문, 일반 지식):**

   * 지도 학습 기반 기법인 **LARS**, **SAPLMA**가 최고 성능
   * **Verbalized Confidence**, **SAR** 등도 비교적 우수

3. **GSM8K (단문, 수학 문제):**

   * GPT-4o-mini 기준으로 **MultiLLMCollab** (다중 모델 협업)이 우수
   * LLaMA-3-8B에서는 **LARS**, **SAR** 등이 여전히 상위권

4. **FactScore-Bio (장문, 전기):**

   * **Verbalized Confidence**가 두 모델 모두에서 가장 우수
   * **Eccentricity**, **Semantic Entropy** 등 sampling 기반 기법도 효과적

---


* **Models:**

  *  **LLaMA-3-8B** (open-weight model, allows white-box access)
  *  **GPT-4o-mini** (closed-weight, API-based model, black-box only)

* **Datasets:**

  * **TriviaQA**: General knowledge questions (short-form)
  * **GSM8K**: Grade-school math word problems (short-form)
  * **FactScore-Bio**: Biographical question answering with multiple factual claims (long-form)

* **Metrics:**

  * **AUROC**: Area under the ROC curve (discrimination ability)
  * **PRR**: Prediction Rejection Ratio (precision gain by rejecting uncertain outputs)

**Key Findings:**

1. **General Trends:**

   * LLaMA-3-8B generally outperforms GPT-4o-mini, thanks to access to internal states
   * Long-form evaluation (FactScore-Bio) is more challenging, with all methods showing performance drop

2. **TriviaQA (short-form, factual):**

   * **LARS** and **SAPLMA**, both supervised methods, perform best
   * **Verbalized Confidence** and **SAR** are also among the top performers

3. **GSM8K (short-form, reasoning):**

   * On GPT-4o-mini, **MultiLLMCollab** (multi-model collaboration) achieves top scores
   * On LLaMA-3-8B, **LARS** and **SAR** continue to lead

4. **FactScore-Bio (long-form):**

   * **Verbalized Confidence** is the strongest method across both models
   * Sampling-based methods like **Eccentricity** and **Semantic Entropy** also perform well





<br/>
# 예제  




---

####  예제 1: 단문 질문 응답에 대한 진실성 판별

* **입력 (Input)**

  * 사용자 질문: `"What is the capital city of France?"`
  * 시스템 프롬프트: `"You are a helpful assistant."`

* **생성 응답 (LLM Output)**

  * `"The capital city of France is Paris."`

* **진실성 예측 기법 (Truth Methods)**

  * `LARS`, `Confidence`, `SelfDetection` 등의 truth method를 적용하여,
    해당 응답이 **신뢰할 수 있는지 아닌지 점수(0\~1 또는 log 확률 등)로 반환**함.

* **출력 (Output)**

  * 진실성 점수 예시:

    ```json
    {
      "output": "The capital city of France is Paris.",
      "truth_scores": {
        "LARS": 0.92,
        "Confidence": 0.88,
        "SelfDetection": 0.95
      }
    }
    ```

---

####  예제 2: 장문 응답 → 주장 분해 → 주장별 진실성 평가

* **입력 질문 (Input)**

  * `"Who is Andrew Shue?"`

* **모델 응답 (Long-form Output)**

  * Andrew Shue는 배우이자 축구선수였으며, MySpace 개발에도 관여했다는 서술 등 다수의 문장으로 구성된 긴 응답.

* **분해 결과 (Decomposed Claims)**

  * `"Andrew Shue is an American actor."`
  * `"Andrew Shue played for the United States national soccer team."`
  * `"Andrew Shue was involved in developing MySpace."` 등 총 25개 주장이 자동 추출됨.

* **각 주장에 대한 진실성 점수**

  * claim-check method (`QuestionAnswerGeneration`, `Entailment`, 등)를 통해 개별 주장마다 truth value를 할당함.

---

####  사용된 트레이닝/테스트 데이터셋

* **TriviaQA:** 퀴즈 기반 일반 지식 QA
* **GSM8K:** 수학 문제 (예: “Natalia sold clips to 48 of her friends…”)
* **FactScore-Bio:** 위키 기반 인물 전기 (long-form)
* Train/Test로 1,000개 샘플씩 사용 (long-form은 50 질문 → 수백 개 주장으로 분해)

---



####  Example 1: Truthfulness on Short-Form QA

* **Input**

  * User question: `"What is the capital city of France?"`
  * System prompt: `"You are a helpful assistant."`

* **LLM Output**

  * `"The capital city of France is Paris."`

* **Applied Truth Methods**

  * Methods like `LARS`, `Confidence`, `SelfDetection` are applied to generate truthfulness scores.

* **Output**

  * Example:

    ```json
    {
      "output": "The capital city of France is Paris.",
      "truth_scores": {
        "LARS": 0.92,
        "Confidence": 0.88,
        "SelfDetection": 0.95
      }
    }
    ```

---

####  Example 2: Long-form Output Decomposition and Scoring

* **Input Question**

  * `"Who is Andrew Shue?"`

* **Generated Answer**

  * Long passage describing Shue’s career in acting, soccer, and involvement with MySpace.

* **Decomposed Claims**

  * `"Andrew Shue is an American actor."`
  * `"Andrew Shue played for the U.S. national soccer team."`
  * `"Andrew Shue was involved in developing MySpace."`
  * (25 claims total)

* **Per-Claim Evaluation**

  * Each claim is scored using claim-check methods such as `QuestionAnswerGeneration`, `Entailment`, etc.

---

####  Dataset Usage

* **TriviaQA:** General knowledge question answering
* **GSM8K:** Mathematical reasoning (e.g., “Natalia sold clips to 48 of her friends…”)
* **FactScore-Bio:** Long-form biographies with many factual claims
* 1,000 samples used for short-form datasets; 50 questions used for long-form (with 1,290–1,764 total claims after decomposition)




<br/>  
# 요약   


이 논문은 LLM이 생성한 응답의 진실성을 예측하기 위한 라이브러리 TruthTorchLM을 제안하며, 30개 이상의 다양한 truth method들을 통합된 인터페이스로 제공합니다.
LLaMA-3-8B와 GPT-4o-mini 모델을 대상으로 TriviaQA, GSM8K, FactScore-Bio 데이터셋에서 평가한 결과, LARS, SAPLMA, Verbalized Confidence 등이 높은 AUROC/PRR 성능을 보였습니다.
예를 들어 "Andrew Shue는 누구인가?"와 같은 장문 질문에 대해 응답을 여러 개의 사실 주장으로 나눈 뒤 각 주장에 대해 진실성 점수를 개별 부여할 수 있습니다.


This paper introduces TruthTorchLM, a library designed to predict the truthfulness of LLM-generated responses using a unified interface for over 30 diverse truth methods.
Evaluations on models like LLaMA-3-8B and GPT-4o-mini across TriviaQA, GSM8K, and FactScore-Bio datasets show that methods such as LARS, SAPLMA, and Verbalized Confidence achieve strong AUROC and PRR scores.
For instance, in response to a long-form question like “Who is Andrew Shue?”, the system decomposes the output into factual claims and assigns a truth score to each claim individually.


<br/>  
# 기타  



####  Figure 1: TruthTorchLM 시스템 개요 다이어그램

이 그림은 TruthTorchLM의 구조를 시각화합니다.

* Truth Method: LLM 응답의 진실성을 사후적으로 평가하는 다양한 기법
* Claim Check Method: 장문의 출력에서 개별 주장을 추출하고 평가
* Unified API: HuggingFace 또는 LiteLLM을 기반으로 생성 + 진실성 예측이 통합됨
  ** 인사이트:** TruthTorchLM은 단순한 점수 산출을 넘어서, 장문 평가, 통합 인터페이스, 샘플링 기반 검증 등 **유연하고 확장 가능한 평가 파이프라인**을 구축함.

---

####  Table 1: Truth Method 분류표

각 truth method를 다음 네 가지 기준으로 분류합니다:

1. 문서 기반 여부 (Document Grounding)
2. 지도 학습 여부 (Supervised)
3. 접근 수준 (Black-box, Grey-box, White-box)
4. 샘플링 필요 여부 (Sampling-Required)

예:

* **LARS**: 지도학습 + 그레이박스
* **GoogleSearchCheck**: 문서 기반 + 블랙박스
* **Eccentricity**: 샘플링 기반 + 블랙박스
  ** 인사이트:** 사용 환경과 리소스 제약에 따라 적절한 truth method를 선택할 수 있도록 **설계 다양성과 사용성 간의 균형**을 중시함.

---

####  Table 2: 성능 비교표 (AUROC & PRR)

* **모델:** LLaMA-3-8B, GPT-4o-mini
* **데이터셋:** TriviaQA, GSM8K, FactScore-Bio
* **지표:** AUROC (진실/비진실 구분력), PRR (불확실성 필터링 효과)

| Method               | TriviaQA AUROC (LLaMA) | FactScore-Bio PRR (GPT-4o) | ... |
| -------------------- | ---------------------- | -------------------------- | --- |
| LARS                 | 0.861                  | 0.294                      |     |
| SAPLMA               | 0.850                  | N/A                        |     |
| VerbalizedConfidence | 0.759–0.836            | **0.514 (최고)**             |     |

** 인사이트:**

* LARS, SAPLMA: 지도학습 기반 기법으로 단문에서 강력
* VerbalizedConfidence: 장문에서도 안정적으로 높은 성능
* FactScore-Bio에서는 **모든 기법이 성능 하락** → 장문 진실성 판별의 어려움을 시사

---

####  Table 4: 장문 응답 분해 예시

질문: "Andrew Shue는 누구인가?"
→ 응답을 **25개의 fact claim**으로 분해함 (예: "Andrew Shue는 배우다", "MySpace 개발에 관여했다", 등)
** 인사이트:**

* 단일 진실성 점수는 부족하며, **세분화된 주장 단위 평가가 필수적**
* 자동 분해 + 평가 기능은 사실 기반 생성의 품질을 개선할 수 있는 중요한 접근임

---


####  Figure 1: TruthTorchLM System Diagram

This figure shows the architecture of TruthTorchLM:

* **Truth Methods** evaluate the truthfulness of LLM outputs after generation.
* **Claim Check Methods** assess factual claims in long-form responses.
* **Unified API** allows integration with HuggingFace or LiteLLM for seamless evaluation.
  ** Insight:** The system is **modular and extensible**, supporting everything from short-form scoring to long-form decomposition and flexible backend support.

---

####  Table 1: Categorization of Truth Methods

Methods are classified along four axes:

1. Document grounding
2. Supervision (Supervised or Self-supervised)
3. Access level (Black-box / Grey-box / White-box)
4. Sampling requirement

Example entries:

* **LARS**: Supervised + Grey-box
* **GoogleSearchCheck**: Document-grounded + Black-box
* **Eccentricity**: Sampling-based + Black-box
  ** Insight:** The taxonomy emphasizes **flexibility for various real-world constraints**, helping users choose appropriate truth methods based on resources and access.

---

####  Table 2: Performance Comparison (AUROC & PRR)

Across LLaMA-3-8B and GPT-4o-mini models, and three datasets (TriviaQA, GSM8K, FactScore-Bio), results show:

| Method               | TriviaQA AUROC (LLaMA) | FactScore-Bio PRR (GPT-4o) | ... |
| -------------------- | ---------------------- | -------------------------- | --- |
| LARS                 | 0.861                  | 0.294                      |     |
| SAPLMA               | 0.850                  | N/A                        |     |
| VerbalizedConfidence | 0.759–0.836            | **0.514 (Best)**           |     |

** Insight:**

* **LARS, SAPLMA** excel in short-form (supervised methods).
* **VerbalizedConfidence** performs best for long-form generation.
* Drop in performance on FactScore-Bio indicates the **increased difficulty of long-form truth assessment.**

---

####  Table 4: Long-form Generation Decomposition Example

Question: “Who is Andrew Shue?”
→ Output is decomposed into **25 factual claims**, e.g., “Andrew Shue is an actor,” “He worked on MySpace.”
** Insight:**

* A single truth score is insufficient. **Claim-level evaluation is essential** for nuanced and interpretable assessment.
* The decomposition capability helps address factual consistency in complex generations.




<br/>
# refer format:     


@misc{yaldiz2025truthtorchlm,
  title = {TruthTorchLM: A Comprehensive Library for Predicting Truthfulness in LLM Outputs},
  author = {Duygu Nur Yaldiz and Yavuz Bakman and Sungmin Kang and Alperen Ozis and Hayrettin Eren Yildiz and Mitash Shah and Zhiqi Huang and Anoop Kumar and Alfy Samuel and Daben Liu and Sai Praneeth Karimireddy and Salman Avestimehr},
  year = {2025},
  eprint = {2507.08203},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2507.08203}
}
  



Duygu Nur Yaldiz, Yavuz Bakman, Sungmin Kang, Alperen Ozis, Hayrettin Eren Yildiz, Mitash Shah, Zhiqi Huang, Anoop Kumar, Alfy Samuel, Daben Liu, Sai Praneeth Karimireddy, and Salman Avestimehr. “TruthTorchLM: A Comprehensive Library for Predicting Truthfulness in LLM Outputs.” arXiv preprint arXiv:2507.08203 (2025). https://arxiv.org/abs/2507.08203.  




