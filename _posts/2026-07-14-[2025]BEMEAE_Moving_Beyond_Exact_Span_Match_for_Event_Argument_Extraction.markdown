---
layout: post
title:  "[2025]BEMEAE: Moving Beyond Exact Span Match for Event Argument Extraction"
date:   2026-07-14 00:36:37 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 이벤트 아규먼트 추출(EAE)에서 전통적인 정확한 스팬 매치(ESM) 평가 지표의 한계를 극복하기 위해 BEMEAE라는 새로운 평가 지표를 제안합니다.


짧은 요약(Abstract) :


이 논문에서는 사건 인수 추출(Event Argument Extraction, EAE)이라는 자연어 처리의 핵심 작업을 다루고 있습니다. EAE는 텍스트에서 사건의 인수를 식별하고 분류하는 것을 목표로 합니다. 그러나 현재 널리 사용되는 정확한 범위 일치(Exact Span Match, ESM) 평가 지표는 엄격한 범위 제약으로 인해 유효한 예측을 오류로 잘못 식별하고 시스템 성능을 과소평가하는 등의 한계가 있습니다. 이를 해결하기 위해, 저자들은 BEMEAE(Beyond Exact Span Match for Event Argument Extraction)라는 새로운 평가 지표를 도입하였습니다. BEMEAE는 참조와 의미적으로 동등하거나 개선된 예측을 인식하며, 보다 정확한 평가를 위해 결정론적 구성 요소와 의미적 일치 구성 요소를 통합합니다. 실험 결과, BEMEAE는 인간의 판단과 더 밀접하게 일치하며, ESM에 비해 더 높은 F1 점수를 제공하고 모델 순위에 중대한 변화를 가져오는 것으로 나타났습니다.



This paper addresses the key task of Event Argument Extraction (EAE) in natural language processing, which focuses on identifying and classifying event arguments in text. However, the widely adopted Exact Span Match (ESM) evaluation metric has notable limitations due to its rigid span constraints, often misidentifying valid predictions as errors and underestimating system performance. To address these issues, the authors introduce BEMEAE (Beyond Exact Span Match for Event Argument Extraction), a novel evaluation metric that recognizes predictions that are semantically equivalent to or improve upon the reference. BEMEAE integrates deterministic components with a semantic matching component for more accurate assessment. Experimental results demonstrate that BEMEAE aligns more closely with human judgments, leads to higher F1 scores compared to ESM, and results in significant changes in model rankings.


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



이 논문에서는 이벤트 아규먼트 추출(Event Argument Extraction, EAE) 작업을 위한 새로운 평가 메트릭인 BEMEAE(Beyond Exact Span Match for Event Argument Extraction)를 제안합니다. 기존의 정확한 스팬 매치(Exact Span Match, ESM) 메트릭은 예측된 아규먼트와 인간이 주석한 아규먼트 간의 정확한 일치를 요구하여, 유효한 예측을 오류로 잘못 분류하고 시스템 성능을 과소평가하는 문제점이 있습니다. BEMEAE는 이러한 문제를 해결하기 위해 두 가지 주요 구성 요소를 통합합니다.

1. **결정론적 구성 요소(Deterministic Components)**: 이 구성 요소는 텍스트의 변형을 처리하기 위해 간단한 규칙을 사용합니다. 예를 들어, 동일한 텍스트가 문서 내에서 여러 번 등장할 경우, ESM은 이를 잘못된 예측으로 간주하지만, BEMEAE는 이를 올바른 예측으로 인정합니다. 또한, 불필요한 토큰(예: 관사, 구두점 등)을 제거하여 의미에 영향을 미치지 않는 변형을 처리합니다. 이 과정에서 후보 아규먼트와 참조 아규먼트를 비교하여 일치 여부를 판단합니다.

2. **의미적 매칭 구성 요소(Semantic Matching Component)**: 이 구성 요소는 후보 아규먼트와 참조 아규먼트 간의 의미적 유사성을 평가합니다. 이를 위해 인간 주석자에 의한 평가와 자동화된 방법을 모두 사용합니다. 자동화된 방법으로는 코사인 유사도, BERTScore, 그리고 GPT-4 모델을 활용한 방법이 포함됩니다. 이 과정에서 후보 아규먼트가 참조 아규먼트와 의미적으로 동등한지를 판단하여, ESM이 잘못 분류한 예측을 올바르게 평가할 수 있도록 합니다.

이러한 방법론을 통해 BEMEAE는 EAE 시스템의 성능을 보다 정확하게 평가할 수 있으며, 기존의 ESM 메트릭에 비해 모델 순위에 상당한 변화를 가져옵니다. 실험 결과, BEMEAE는 인간의 판단과 더 밀접하게 일치하며, ESM에 비해 F1 점수가 높아지는 경향을 보였습니다. 이로 인해 BEMEAE는 EAE 작업의 평가에 있어 더 신뢰할 수 있는 대안으로 자리잡을 수 있습니다.




In this paper, we propose a new evaluation metric for Event Argument Extraction (EAE) called BEMEAE (Beyond Exact Span Match for Event Argument Extraction). The traditional Exact Span Match (ESM) metric requires an exact match between predicted arguments and human-annotated arguments, which often misclassifies valid predictions as errors and underestimates system performance. BEMEAE addresses these issues by integrating two main components.

1. **Deterministic Components**: This component employs simple rules to handle textual variations. For instance, when identical texts appear multiple times in a document, ESM incorrectly classifies them as errors, while BEMEAE recognizes them as correct predictions. Additionally, it removes uninformative tokens (e.g., articles, punctuation) that do not affect meaning, allowing for a comparison between candidate and reference arguments based on their cleaned content.

2. **Semantic Matching Component**: This component evaluates the semantic similarity between candidate and reference arguments. It utilizes both human assessments and automated methods. Automated methods include cosine similarity, BERTScore, and approaches using the GPT-4 model. This process allows for the identification of candidate arguments that are semantically equivalent to the reference arguments, correcting misclassifications made by ESM.

Through this methodology, BEMEAE provides a more accurate assessment of EAE system performance, leading to significant changes in model rankings compared to the traditional ESM metric. Experimental results show that BEMEAE aligns more closely with human judgments and tends to yield higher F1 scores than ESM. Consequently, BEMEAE can serve as a more reliable alternative for evaluating EAE tasks.


<br/>
# Results



이 논문에서는 이벤트 아규먼트 추출(Event Argument Extraction, EAE) 모델의 성능을 평가하기 위해 BEMEAE(Beyond Exact Span Match for Event Argument Extraction)라는 새로운 평가 메트릭을 도입했습니다. 기존의 Exact Span Match (ESM) 메트릭은 예측된 아규먼트와 인간이 주석을 단 아규먼트 간의 정확한 일치를 요구하여, 유효한 예측을 오류로 잘못 분류하는 경향이 있습니다. 이로 인해 모델의 성능이 과소평가될 수 있습니다.

#### 실험 설정
- **경쟁 모델**: 총 9개의 최신 EAE 모델을 평가했습니다. 이 모델들은 분류 기반 모델과 생성 기반 모델로 나뉘며, RAMS와 GENEV A라는 두 개의 데이터셋에서 테스트되었습니다.
- **테스트 데이터**: 
  - **RAMS**: 뉴스 기사 기반 데이터셋으로, 139개의 이벤트 유형과 65개의 아규먼트 유형을 포함합니다.
  - **GENEV A**: 일반 도메인 데이터셋으로, 115개의 이벤트 유형과 220개의 아규먼트 유형을 포함합니다.
  
#### 메트릭
- **Exact Span Match (ESM)**: 기존의 평가 메트릭으로, 예측된 아규먼트가 참조 아규먼트와 정확히 일치해야만 올바른 것으로 간주됩니다.
- **BEMEAE**: 새로운 메트릭으로, 결정론적 구성 요소와 의미적 일치 구성 요소를 결합하여 아규먼트의 의미적 동등성을 평가합니다. 이 메트릭은 ESM의 한계를 극복하고, 더 많은 유효한 예측을 인정하여 모델 성능을 보다 정확하게 평가합니다.

#### 결과
- **F1 점수**: BEMEAE를 사용한 평가에서 모든 모델의 F1 점수가 증가했습니다. 예를 들어, BART-Gen 모델은 RAMS 데이터셋에서 F1 점수가 46%에서 61%로 증가하여 순위가 7위에서 2위로 상승했습니다.
- **모델 순위 변화**: BEMEAE의 도입으로 인해 모델 순위가 크게 변화했습니다. ESM과 BEMEAE의 순위 간의 상관관계는 낮았으며(Kendall’s τ= 0.44 for RAMS, 0.67 for GENEV A), BEMEAE의 결과는 인간의 판단과 더 잘 일치했습니다(Kendall’s τ= 0.94).

이러한 결과는 BEMEAE가 EAE 모델의 성능을 보다 정확하게 평가할 수 있는 유용한 도구임을 보여줍니다. BEMEAE는 기존의 ESM 메트릭보다 더 많은 유효한 아규먼트를 인정하고, 모델 간의 성능 비교를 보다 신뢰할 수 있게 만듭니다.

---



In this paper, a new evaluation metric called BEMEAE (Beyond Exact Span Match for Event Argument Extraction) is introduced to assess the performance of Event Argument Extraction (EAE) models. The traditional Exact Span Match (ESM) metric requires an exact match between predicted arguments and human-annotated arguments, which often leads to valid predictions being misclassified as errors, resulting in an underestimation of model performance.

#### Experimental Setup
- **Competing Models**: A total of 9 state-of-the-art EAE models were evaluated. These models were categorized into classification-based and generation-based models and were tested on two datasets: RAMS and GENEV A.
- **Test Data**: 
  - **RAMS**: A news article-based dataset that includes 139 event types and 65 argument types.
  - **GENEV A**: A general-domain dataset that includes 115 event types and 220 argument types.
  
#### Metrics
- **Exact Span Match (ESM)**: The existing evaluation metric that considers a predicted argument correct only if it exactly matches the reference argument.
- **BEMEAE**: A new metric that combines deterministic components with a semantic matching component to evaluate the semantic equivalence of arguments. This metric addresses the limitations of ESM and recognizes more valid predictions, leading to a more accurate assessment of model performance.

#### Results
- **F1 Scores**: The evaluation using BEMEAE showed an increase in F1 scores for all models. For instance, the BART-Gen model's F1 score increased from 46% to 61% on the RAMS dataset, boosting its rank from 7th to 2nd.
- **Model Ranking Changes**: The introduction of BEMEAE led to significant changes in model rankings. The correlation between rankings based on ESM and BEMEAE was low (Kendall’s τ= 0.44 for RAMS, 0.67 for GENEV A), while the results from BEMEAE aligned more closely with human judgments (Kendall’s τ= 0.94).

These results demonstrate that BEMEAE is a valuable tool for more accurately evaluating the performance of EAE models. By recognizing a broader range of valid arguments than the traditional ESM metric, BEMEAE provides a more reliable basis for comparing model performance.


<br/>
# 예제



이 논문에서는 이벤트 아규먼트 추출(Event Argument Extraction, EAE) 작업을 수행하기 위해 두 가지 데이터셋인 RAMS와 GENEV A를 사용합니다. 이 데이터셋들은 각각 뉴스 기사와 일반 도메인 텍스트에서 이벤트와 그에 관련된 아규먼트를 추출하는 데 사용됩니다. 

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **RAMS 데이터셋**: 이 데이터셋은 뉴스 기사에서 추출된 9,107개의 문서로 구성되어 있으며, 각 문서에는 139개의 이벤트 유형과 65개의 아규먼트 유형이 포함되어 있습니다. 각 문서에는 평균 2.33개의 아규먼트가 포함되어 있습니다.
   - **GENEV A 데이터셋**: 이 데이터셋은 여러 도메인에서 수집된 3,684개의 문서로 구성되어 있으며, 115개의 이벤트 유형과 220개의 아규먼트 유형이 포함되어 있습니다. 각 문서에는 평균 3.33개의 아규먼트가 포함되어 있습니다.

2. **테스트 데이터**:
   - 테스트 데이터는 트레이닝 데이터와 동일한 형식을 따르며, 모델의 성능을 평가하기 위해 사용됩니다. 각 문서에는 이벤트 트리거와 그에 대한 아규먼트가 포함되어 있습니다.

#### 구체적인 인풋과 아웃풋

- **인풋**:
  - 각 문서에서 이벤트 트리거와 아규먼트 역할이 주어집니다. 예를 들어, "airstrike"라는 이벤트 트리거가 주어지고, "target" 역할에 대한 아규먼트를 추출해야 합니다.
  - 문서의 내용은 다음과 같습니다: "There are no craters, while [the vehicles] have their chassis intact and they have not been severely damaged, which would have been the case from an [airstrike] event_trigger."

- **아웃풋**:
  - 모델은 주어진 이벤트 트리거에 대해 아규먼트를 예측합니다. 예를 들어, "the vehicles"라는 아규먼트가 "target" 역할로 예측될 수 있습니다.
  - 모델의 예측 결과는 "correct", "partial", 또는 "incorrect"로 분류됩니다. "correct"는 아규먼트가 정확히 맞는 경우, "partial"은 중요한 정보가 누락된 경우, "incorrect"는 아규먼트가 잘못된 경우를 의미합니다.

#### 구체적인 테스크

- **테스크**: 모델은 주어진 문서에서 이벤트 트리거와 아규먼트 역할에 따라 아규먼트를 추출하고, 이를 평가하는 작업을 수행합니다. 이 과정에서 모델은 아규먼트의 정확성을 평가하기 위해 BEMEAE라는 새로운 평가 지표를 사용합니다.

---



This paper utilizes two datasets, RAMS and GENEV A, to perform the task of Event Argument Extraction (EAE). These datasets are used to extract events and their associated arguments from news articles and general domain texts, respectively.

#### Training Data and Test Data

1. **Training Data**:
   - **RAMS Dataset**: This dataset consists of 9,107 documents extracted from news articles, containing 139 event types and 65 argument types. Each document has an average of 2.33 arguments.
   - **GENEV A Dataset**: This dataset comprises 3,684 documents collected from various domains, containing 115 event types and 220 argument types. Each document has an average of 3.33 arguments.

2. **Test Data**:
   - The test data follows the same format as the training data and is used to evaluate the performance of the models. Each document includes event triggers and their corresponding arguments.

#### Specific Inputs and Outputs

- **Input**:
  - Each document provides an event trigger and an argument role to extract. For example, the event trigger "airstrike" is given, and the task is to extract an argument for the role "target."
  - The content of the document might be: "There are no craters, while [the vehicles] have their chassis intact and they have not been severely damaged, which would have been the case from an [airstrike] event_trigger."

- **Output**:
  - The model predicts an argument based on the given event trigger. For instance, the argument "the vehicles" could be predicted for the "target" role.
  - The model's predictions are classified as "correct," "partial," or "incorrect." "Correct" means the argument is exactly right, "partial" indicates that important information is missing, and "incorrect" signifies that the argument is wrong.

#### Specific Task

- **Task**: The model performs the task of extracting arguments based on the given event trigger and argument role from the document, and it evaluates these using a new evaluation metric called BEMEAE.

<br/>
# 요약


이 논문에서는 이벤트 아규먼트 추출(EAE)에서 전통적인 정확한 스팬 매치(ESM) 평가 지표의 한계를 극복하기 위해 BEMEAE라는 새로운 평가 지표를 제안합니다. BEMEAE는 결정론적 구성 요소와 의미적 일치 구성 요소를 결합하여 모델의 성능을 보다 정확하게 평가하며, 실험 결과 BEMEAE가 ESM보다 높은 F1 점수를 기록하고 모델 순위에 중대한 변화를 가져온 것을 보여줍니다. 예를 들어, BART-Gen 모델은 RAMS 데이터셋에서 F1 점수가 46%에서 61%로 증가하며 순위가 7위에서 2위로 상승했습니다.

---

In this paper, a new evaluation metric called BEMEAE is proposed to overcome the limitations of the traditional Exact Span Match (ESM) evaluation in Event Argument Extraction (EAE). BEMEAE combines deterministic components with a semantic matching component to provide a more accurate assessment of model performance, and experimental results show that BEMEAE achieves higher F1 scores than ESM, leading to significant changes in model rankings. For instance, the BART-Gen model's F1 score increased from 46% to 61% on the RAMS dataset, elevating its rank from 7th to 2nd.

<br/>
# 기타



1. **다이어그램 및 피규어**:
   - **Figure 1**: ESM의 한계에 대한 예시를 보여줍니다. "airstrike"라는 이벤트 트리거에 대해 "vehicles"라는 참조 인자가 있지만, ESM은 "convoy"를 잘못된 예측으로 분류합니다. 이는 ESM이 정확한 스팬 일치를 요구하기 때문에 발생하는 문제를 강조합니다.
   - **Figure 2**: 주석 인터페이스의 스크린샷으로, 후보 인자를 평가하는 과정에서의 단계별 질문을 보여줍니다. 이 인터페이스는 주석자들이 후보 인자의 유효성을 평가하고, 유사성을 확인하며, 관계를 결정하는 데 도움을 줍니다.

2. **테이블**:
   - **Table 1**: RAMS와 GENEV A 데이터셋의 기본 통계 정보를 제공합니다. 두 데이터셋의 이벤트 유형, 역할, 문서 수, 이벤트 수, 인자 수 등을 비교하여 EAE의 다양성을 보여줍니다.
   - **Table 2**: RAMS 데이터셋에서 여러 모델의 성능(F1 점수 및 순위)을 비교합니다. ESM과 BEMEAE의 성능 차이를 보여주며, BEMEAE가 모델 순위에 미치는 영향을 강조합니다.
   - **Table 6 & 7**: BEMEAE의 다양한 구성 요소가 모델 성능에 미치는 영향을 보여줍니다. 각 단계에서 F1 점수의 증가를 나타내며, BEMEAE가 ESM보다 더 정확한 평가를 제공함을 입증합니다.
   - **Table 8**: ESM과 BEMEAE에 따른 모델 순위를 비교합니다. ESM과 BEMEAE 간의 순위 상관관계가 낮음을 보여주며, BEMEAE가 모델 성능을 더 잘 반영함을 나타냅니다.

3. **어펜딕스**:
   - **Appendix A**: GENEV A 데이터셋에 대한 추가 결과를 제공합니다. RAMS와 유사한 방식으로 모델 성능을 비교합니다.
   - **Appendix B**: 주석 작업의 세부 사항을 설명합니다. 주석자들이 후보 인자를 평가하는 과정에서의 질문과 평가 기준을 명시합니다.
   - **Appendix C**: 자동화된 방법과 인간 주석자 간의 일치도를 보여주는 코헨의 카파 점수를 제공합니다. 자동화된 방법이 인간의 평가와 얼마나 일치하는지를 나타냅니다.
   - **Appendix D**: GPT 모델을 사용한 평가 프롬프트의 세부 사항을 설명합니다. 다양한 조건에서 모델의 성능을 평가하기 위한 프롬프트를 제공합니다.

### Insights from Figures, Tables, and Appendices

1. **Figures**:
   - **Figure 1** illustrates the limitations of ESM by providing an example where the event trigger "airstrike" has a reference argument "vehicles," but ESM incorrectly classifies "convoy" as an error. This highlights the rigidity of ESM's exact span matching requirement.
   - **Figure 2** shows a screenshot of the annotation interface, detailing the step-by-step questions that annotators use to evaluate candidate arguments. This interface aids annotators in assessing the validity, similarity, and relationship of candidate arguments.

2. **Tables**:
   - **Table 1** provides basic statistics for the RAMS and GENEV A datasets, comparing the number of event types, roles, documents, events, and arguments, showcasing the diversity in EAE tasks.
   - **Table 2** compares the performance (F1 scores and ranks) of various models on the RAMS dataset, highlighting the differences in performance between ESM and BEMEAE and emphasizing the impact of BEMEAE on model rankings.
   - **Tables 6 & 7** show the incremental impact of different components of BEMEAE on model performance, demonstrating that BEMEAE provides a more accurate assessment than ESM.
   - **Table 8** compares model rankings based on ESM and BEMEAE, indicating a low correlation between the two, thus demonstrating that BEMEAE better reflects model performance.

3. **Appendices**:
   - **Appendix A** provides additional results for the GENEV A dataset, comparing model performance in a manner similar to RAMS.
   - **Appendix B** details the annotation task, outlining the questions and evaluation criteria used by annotators to assess candidate arguments.
   - **Appendix C** presents Cohen's kappa scores showing the agreement between automated methods and human annotators, indicating how well automated methods align with human evaluations.
   - **Appendix D** describes the prompts used for evaluating candidate arguments with GPT models, detailing the conditions under which the models were tested.

These insights collectively demonstrate the effectiveness of BEMEAE over traditional ESM in evaluating event argument extraction models, highlighting the need for more flexible and semantically aware evaluation metrics in natural language processing tasks.

<br/>
# refer format:
### BibTeX Entry

```bibtex
@inproceedings{fane2025bemeae,
  author = {Enfa Fane and Md Nayem Uddin and Oghenevovwe Ikumariegbe and Daniyal Kashif and Eduardo Blanco and Steven R. Corman},
  title = {BEMEAE: Moving Beyond Exact Span Match for Event Argument Extraction},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies},
  volume = {1},
  pages = {5734--5749},
  year = {2025},
  publisher = {Association for Computational Linguistics},
  address = {April 29 - May 4, 2025}
}
```

### Chicago Style Citation

Enfa Fane, Md Nayem Uddin, Oghenevovwe Ikumariegbe, Daniyal Kashif, Eduardo Blanco, and Steven R. Corman. "BEMEAE: Moving Beyond Exact Span Match for Event Argument Extraction." In *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies*, 1:5734–5749. April 29 - May 4, 2025. Association for Computational Linguistics.
    