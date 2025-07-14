---
layout: post
title:  "[2025]SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"  
date:   2025-07-13 21:45:40 -0800
categories: study
---

{% highlight ruby %}


한줄 요약: 

스스로 LLM이 검색할지 판단하고 스스로 검색하도록 하는 RAG  



짧은 요약(Abstract) :    



대형 언어 모델(LLM)은 내부에 저장된 지식에만 의존하기 때문에 종종 사실과 다른 내용을 생성합니다. 이를 보완하기 위해 RAG(Retrieval-Augmented Generation) 방식이 사용되지만, 무조건적으로 정해진 개수의 문서를 불러오면 오히려 성능을 저하시킬 수 있습니다. 이 논문에서는 LLM이 필요할 때만 정보를 검색하고, 생성된 답변을 스스로 비판적으로 검토할 수 있게 하는 새로운 프레임워크인 SELF-RAG를 제안합니다. 이 방식은 **특수한 '반성 토큰(reflection tokens)'**을 통해 검색, 생성, 검토 과정을 제어할 수 있게 해주며, 다양한 태스크 요구에 맞춰 유연하게 작동합니다. 실험 결과, SELF-RAG는 ChatGPT나 Llama2-RAG 기반 모델보다 다양한 작업(예: 오픈 도메인 QA, 사실 검증, 장문 생성 등)에서 더 높은 정확도와 인용의 신뢰성을 보여줍니다.


Despite their remarkable capabilities, large language models (LLMs) often produce factually incorrect responses due to their sole reliance on parametric knowledge. Retrieval-Augmented Generation (RAG) partially mitigates this issue but suffers from indiscriminate retrieval. This paper introduces SELF-RAG (Self-Reflective Retrieval-Augmented Generation), a new framework that enables LLMs to selectively retrieve relevant passages, generate responses, and reflect on their own outputs using special reflection tokens. These tokens guide when to retrieve and how to critique generation quality. SELF-RAG allows flexible behavior during inference to match diverse task needs. Experimental results show that SELF-RAG (7B and 13B models) significantly outperforms both ChatGPT and retrieval-augmented Llama2-chat on open-domain QA, reasoning, fact verification, and long-form generation tasks, notably improving factuality and citation precision.


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






\*\*SELF-RAG(Self-Reflective Retrieval-Augmented Generation)\*\*는 기존 RAG 방식의 한계를 극복하기 위해, LLM이 스스로 **검색 여부를 판단**하고, **생성한 문장을 평가**하며, 이를 기반으로 더 나은 응답을 생성하도록 설계된 프레임워크입니다. 주요 특징은 다음과 같습니다:

1. **반성 토큰 (Reflection Tokens)**
   SELF-RAG는 생성 과정 중에 특별한 토큰을 생성하여, 다음의 4가지 판단을 수행합니다:

   * `Retrieve`: 검색이 필요한지 여부
   * `ISREL`: 검색된 문서가 관련 있는지 여부
   * `ISSUP`: 생성된 응답이 검색된 정보에 의해 뒷받침되는지 여부
   * `ISUSE`: 전체 응답의 유용성 평가 (5점 척도)

2. **3단계 작동 구조**

   * **1단계: 검색 여부 판단** – 입력에 따라 `Retrieve` 토큰을 생성해 검색이 필요한지 결정
   * **2단계: 문서 검색 및 병렬 생성** – 관련된 문서들을 검색해, 각각을 바탕으로 여러 응답을 병렬 생성
   * **3단계: 자기비판(critique)** – `ISREL`, `ISSUP`, `ISUSE` 토큰을 통해 각 응답의 신뢰성과 타당성을 평가하고 최종 응답 선택

3. **트레이닝 구조**
   SELF-RAG는 두 가지 모델을 학습합니다:

   * **Critic 모델**: GPT-4를 활용해 학습 데이터를 기반으로 반성 토큰을 생성하도록 학습됨
   * **Generator 모델 (예: LLaMA 2 7B/13B)**: 반성 토큰이 삽입된 데이터를 기반으로 전체 문장과 토큰을 생성하는 모델로 학습됨

4. **훈련 데이터**

   * 약 15만 개의 instruction-following 데이터셋과 knowledge-intensive 태스크 데이터 활용
   * Critic은 오프라인으로 GPT-4를 사용해 반성 토큰이 삽입된 데이터를 생성
   * Generator는 해당 데이터를 바탕으로 전체 문장과 토큰을 예측하도록 학습됨

5. **추론 시 사용자 맞춤형 설정**

   * 각 토큰의 가중치를 조절하여 응답의 정확성(예: citation precision)과 창의성(예: fluency)의 균형을 조절할 수 있음
   * 필요에 따라 retrieval 빈도나 평가 기준을 동적으로 조절 가능함

---



**SELF-RAG (Self-Reflective Retrieval-Augmented Generation)** is a novel framework that enables an LLM to dynamically decide when to retrieve information, generate responses, and self-critique its own outputs through reflection tokens. Its key components include:

1. **Reflection Tokens**
   SELF-RAG introduces four types of reflection tokens to guide generation:

   * `Retrieve`: whether to trigger retrieval
   * `ISREL`: whether the retrieved document is relevant
   * `ISSUP`: whether the output is supported by the passage
   * `ISUSE`: overall utility of the response (on a 5-point scale)

2. **Three-Step Inference Pipeline**

   * **Step 1: Decide on Retrieval** – The model predicts a `Retrieve` token to determine if retrieval is needed
   * **Step 2: Generate from Retrieved Passages** – If retrieval is triggered, multiple passages are retrieved and responses are generated in parallel
   * **Step 3: Critique and Select** – The model evaluates each response using `ISREL`, `ISSUP`, and `ISUSE` to select the best one

3. **Training Procedure**
   SELF-RAG trains two models:

   * **Critic Model**: Uses GPT-4 to generate reflection tokens for supervision
   * **Generator Model** (e.g., LLaMA 2 7B/13B): Learns to generate both task outputs and reflection tokens in a unified manner

4. **Training Data**

   * Includes 150K instruction-output pairs from Open-Instruct and other knowledge-intensive datasets
   * Critic model generates offline annotations (reflection tokens) using GPT-4
   * Generator is trained on the resulting data using standard next-token prediction loss

5. **Inference-Time Customization**

   * SELF-RAG enables controllable generation by adjusting weights for different critique token types (e.g., to prioritize factuality or fluency)
   * The retrieval frequency and critique sensitivity can be tuned for different applications

---



   
 
<br/>
# Results  




SELF-RAG는 총 6가지 다운스트림 태스크에서 다양한 기존 모델들과 비교되었으며, **정확성, 사실성(factuality), 인용 정확도(citation precision)** 등에서 일관되게 우수한 성능을 보였습니다.

####  사용된 태스크 및 테스트 데이터셋:

1. **Closed-set (정답이 고정된 과제)**

   * **PubHealth**: 공중보건 관련 주장에 대한 사실 검증
   * **ARC-Challenge**: 과학 문제 기반 다지선다형 추론

2. **Short-form Generation**

   * **PopQA**: 희귀 엔티티 기반 오픈도메인 QA
   * **TriviaQA**: 다양한 사실 기반 질문에 대한 응답 생성

3. **Long-form Generation**

   * **Biography Generation**: 인물 전기 생성 + FactScore로 사실성 평가
   * **ASQA (Answer Selection QA)**: 장문 QA에 대해 인용 기반 정답 평가

---

####  비교한 경쟁 모델:

* **기존 비-RAG 모델들**:

  * LLaMA2 (7B, 13B), Alpaca (7B, 13B), ChatGPT, CoVE65B
* **기존 RAG 기반 모델들**:

  * Ret-LLaMA2-chat, Ret-ChatGPT, Perplexity.ai, Toolformer, SAIL 등
  * 일부는 retrieval을 통해 LLM 입력에 문서를 덧붙임

---

####  SELF-RAG 성능 요약:

* **모든 태스크에서 기존 7B, 13B LLaMA/Alpaca 및 ChatGPT보다 우수한 결과**
* **ASQA에서 citation precision 70.3% 달성 (ChatGPT보다 높음)**
* **PopQA, ARC-Challenge, Biography Generation에서 최고 정확도**
* **Retriever 없이 학습한 모델이나 Critic 없는 모델 대비 최대 15% 이상 정확도 향상**

---

####  분석 인사이트:

* SELF-RAG의 **반성 토큰 기반 자기비판**이 실제로 citation 품질과 전체 성능을 향상시킴
* **Retrieval 빈도와 critique weighting을 조절**해 태스크별 성능 균형 가능
* 일반적인 RAG처럼 문서를 무조건 붙이는 방식(top-1 passage)보다 훨씬 정교한 방식이 효과적

---

SELF-RAG was evaluated on six diverse tasks and consistently outperformed both non-retrieval and retrieval-augmented baselines in terms of **accuracy, factuality, and citation precision**.

####  Evaluation Tasks and Datasets:

1. **Closed-set Tasks**

   * **PubHealth**: Fact verification for public health statements
   * **ARC-Challenge**: Multiple-choice science reasoning

2. **Short-form Generation**

   * **PopQA**: Open-domain QA on rare entities
   * **TriviaQA**: General fact-based QA

3. **Long-form Generation**

   * **Biography Generation**: Assessed with FactScore for factual correctness
   * **ASQA**: Long-form QA with citation-based correctness evaluation

---

####  Compared Baselines:

* **Non-Retrieval Models**:

  * LLaMA2 (7B, 13B), Alpaca (7B, 13B), ChatGPT, CoVE65B
* **Retrieval-Augmented Models**:

  * Ret-LLaMA2-chat, Ret-ChatGPT, Perplexity.ai, Toolformer, SAIL
  * Most of these prepend a fixed number of retrieved passages to the input

---

####  Key SELF-RAG Outcomes:

* **Outperformed ChatGPT and LLaMA2 on all six tasks**
* **Achieved 70.3% citation precision on ASQA, surpassing ChatGPT**
* **Top accuracy on PopQA, ARC-Challenge, and biography generation tasks**
* **15%+ accuracy gains over ablation baselines without retriever or critic model**

---

####  Analysis Insights:

* SELF-RAG’s **self-reflection via critique tokens** directly improves citation reliability and factuality
* Allows **inference-time customization** by adjusting retrieval frequency and critique weighting
* Demonstrated better performance than naive RAG that simply appends top-1 retrieved passages




<br/>
# 예제  


---



SELF-RAG는 다양한 태스크에서 사용자의 질문에 대해 **검색이 필요한지 스스로 판단하고**, **검색 결과를 바탕으로 응답을 생성**, 그리고 **자기 검토를 통해 가장 적절한 답변을 선택**합니다. 다음은 논문에 나온 대표적인 예시입니다.

---

####  예시 1: 오픈도메인 질문 응답 (Open-domain QA)

* **질문 (Prompt)**
  `How did US states get their names?`
  (미국 주 이름은 어떻게 유래되었나요?)

* **일반 RAG 방식 (비판 없이 생성)**

  * 예: *"US states got their names from a variety of sources. Eleven states are named after an individual person (e.g., California was named after Christopher Columbus)."*
  *  오류: California는 Columbus가 아니라 스페인 소설 속 가상의 섬에서 유래됨

* **SELF-RAG 방식**

  1. Retrieval 필요 판단 → "Yes"
  2. 다음과 같은 문서 검색됨:

     * *"California's name has its origins in a 16th-century novel Las Sergas de Esplandián..."*
  3. 응답 생성 예시:

     * *"California was named after a fictional island from a Spanish novel called Las Sergas de Esplandián. Eleven states are named after people, including Louisiana (King Louis XIV) and Georgia (King George II)."*
  4. 각 문장에 대한 critique 토큰 생성:

     * `ISREL`: relevant
     * `ISSUP`: fully supported
     * `ISUSE`: 5 (최고 점수)

*  결과: 더 정확하고 검증 가능한 응답 생성 + 인용 가능

---

####  예시 2: 창의적인 글쓰기 (Personal Essay)

* **질문 (Prompt)**
  `Write an essay of your best summer vacation`

* **SELF-RAG의 판단**

  * Retrieval 필요 없음 (`Retrieve=No`)
  * 바로 창의적인 텍스트 생성:

    * *"My best summer vacation was a magical escape to the coastal town of Santorini. The azure waters and white-washed buildings were unforgettable."*

*  결과: 정보 검색 없이도 자연스럽고 유창한 텍스트 생성
  → SELF-RAG은 상황에 따라 검색을 생략할 수 있음

---


SELF-RAG intelligently determines whether retrieval is necessary, fetches relevant documents, generates outputs, and critiques itself to produce more factual and appropriate responses. Below are representative examples from the paper:

---

####  Example 1: Open-Domain QA Task

* **Prompt**
  `How did US states get their names?`

* **Vanilla RAG Output**

  * *"US states got their names from a variety of sources. Eleven states are named after individuals like California, which was named after Christopher Columbus."*
  *  Inaccurate: California was not named after Columbus.

* **SELF-RAG Output (3-Step Process)**

  1. Detects need for retrieval → outputs `Retrieve=Yes`
  2. Retrieves passage like:

     * *"California's name has its origins in a 16th-century novel Las Sergas de Esplandián..."*
  3. Generates output:

     * *"California was named after a fictional island in a Spanish novel. Louisiana was named in honor of King Louis XIV, and Georgia after King George II."*
  4. Critique tokens:

     * `ISREL = relevant`, `ISSUP = fully supported`, `ISUSE = 5`

*  Result: Factually accurate response with verifiable citations

---

####  Example 2: Creative Writing Task

* **Prompt**
  `Write an essay of your best summer vacation`

* **SELF-RAG Behavior**

  * Outputs `Retrieve=No` (no retrieval needed)
  * Proceeds with natural generation:

    * *"My best summer vacation was a magical escape to the coastal town of Santorini. The azure waters and white-washed buildings were unforgettable."*

*  Result: Fluency-focused generation without retrieval
  → SELF-RAG can **adapt behavior based on the task**




<br/>  
# 요약   





SELF-RAG는 대형 언어 모델이 스스로 검색이 필요한지를 판단하고, 검색된 문서를 활용해 응답을 생성하며, 생성 결과를 비판적으로 평가하는 프레임워크다. 다양한 QA 및 생성 태스크에서 ChatGPT 및 기존 RAG 모델보다 높은 정확도, 사실성, 인용 정확도를 달성했다. 예를 들어, “미국 주 이름 유래” 질문에 대해 SELF-RAG는 픽션에서 유래된 캘리포니아의 이름을 정확히 설명하고, 필요한 경우에만 검색을 수행한다.

---



SELF-RAG is a framework that enables large language models to decide when to retrieve information, generate outputs based on retrieved passages, and critique their own responses. It achieves higher accuracy, factuality, and citation precision than ChatGPT and prior RAG models across various QA and generation tasks. For instance, when asked about the origin of US state names, SELF-RAG correctly explains California’s fictional origin and retrieves only when necessary.




<br/>  
# 기타  




####  Figure 1: SELF-RAG 개요

* 좌측: 기존 RAG는 무조건 K개의 문서를 검색해 생성에 사용
* 우측: SELF-RAG는 `Retrieve` 토큰을 통해 **검색이 필요할 때만 검색**하고, 각 문서와 응답을 **자기 평가** (`ISREL`, `ISSUP`, `ISUSE`)
* **인사이트**: 필요 없는 검색을 줄이고, 생성 품질과 인용 타당성을 모두 향상시킬 수 있음

####  Table 1: 반성 토큰 종류 정리

* `Retrieve`, `ISREL`, `ISSUP`, `ISUSE` 네 종류
* 각 토큰은 (예: `ISSUP`: fully supported, partially supported, no support) **평가 범주가 명확히 정의**됨
* **결론**: SELF-RAG는 단순한 Boolean 평가가 아닌 **세분화된 자기 피드백 체계**를 통해 정교한 선택 가능

####  Table 2: 메인 실험 결과 요약

* 6개 태스크 전반에서 \*\*SELF-RAG (7B/13B)\*\*는 비RAG 및 기존 RAG 모델들보다 **더 높은 성능**을 보임
* 특히 ASQA에서 **Citation Precision 70.3% 달성**으로 ChatGPT보다 우위
* **인사이트**: SELF-RAG의 성능은 단순히 검색 기능 때문이 아니라 **반성 기반 학습 구조**에서 기인함

####  Figure 3a–c: Ablation, Customization, Retrieval

* (a) Retriever나 Critic 없이 학습한 경우 성능 크게 하락 → 각 구성 요소의 중요성 확인
* (b) `ISSUP` 가중치를 높이면 Citation 정확도 상승, Fluency(MAUVE)는 다소 감소
* (c) Retrieval 빈도와 성능 간 트레이드오프: **PubHealth**는 검색 줄여도 유지되지만 **PopQA**는 검색 줄이면 성능 급락
* **결론**: SELF-RAG는 **태스크별 목적에 맞게 동적으로 조정 가능**한 유연한 시스템

####  Appendix A\~D: 학습 세부 설정 및 데이터 수집

* GPT-4를 활용해 Critic 데이터를 생성하고, LLaMA2 모델 기반으로 Generator를 학습
* Reflection Token 생성 정의, 트레이닝 알고리즘, 테스트 세팅이 모두 상세히 제시됨
* **인사이트**: SELF-RAG는 고비용 RLHF 없이도 **효율적인 학습 및 추론이 가능함**

---


####  Figure 1: Overview of SELF-RAG

* Left: Vanilla RAG retrieves a fixed number of documents regardless of necessity
* Right: SELF-RAG uses reflection tokens to **retrieve only when needed** and evaluates outputs via `ISREL`, `ISSUP`, and `ISUSE`
* **Insight**: Reduces unnecessary retrieval and improves both output quality and factual attribution

#### Table 1: Types of Reflection Tokens

* Four token types: `Retrieve`, `ISREL`, `ISSUP`, and `ISUSE`
* Each token includes **clearly defined categorical outputs** (e.g., `ISSUP` = fully/partially supported, no support)
* **Conclusion**: Enables **fine-grained self-assessment** beyond binary classification

####  Table 2: Main Experimental Results

* SELF-RAG (7B/13B) **outperforms both non-RAG and prior RAG models** across six tasks
* Notably achieves **70.3% citation precision on ASQA**, outperforming ChatGPT
* **Insight**: Performance gain stems from **reflective learning**, not just retrieval

####  Figure 3a–c: Ablation, Customization, Retrieval Frequency

* (a) Without retriever or critic, accuracy drops significantly → confirms **importance of all components**
* (b) Increasing `ISSUP` weight improves citation precision but slightly reduces fluency (MAUVE)
* (c) Trade-off between retrieval frequency and accuracy: **PubHealth** is robust to less retrieval; **PopQA** suffers when reduced
* **Conclusion**: SELF-RAG supports **dynamic, task-specific control at inference time**

####  Appendix A–D: Training and Data Details

* Critic data generated using GPT-4; Generator trained with LLaMA2 (7B/13B)
* Detailed token definitions, training/inference algorithms, and datasets are described
* **Insight**: SELF-RAG is **efficient and scalable** without requiring expensive RLHF training




<br/>
# refer format:     



@inproceedings{asai2024selfrag,
  title     = {SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection},
  author    = {Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup and Hajishirzi, Hannaneh},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2024},
  url       = {https://selfrag.github.io/}
}





Asai, Akari, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. “SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.” Paper presented at the International Conference on Learning Representations (ICLR), 2024. https://selfrag.github.io/.



