---
layout: post
title:  "[2026]Sycophants in the Courtroom: Are LLMs Fragile to Juridical Authority and Evolving Legal Standards?"
date:   2026-07-16 21:35:28 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 법적 추론을 평가하기 위해 LEGAL-LINK-EU라는 새로운 벤치마크를 도입하고, 이를 통해 대형 언어 모델(LLM)의 법적 지식과 신뢰성을 분석하였다.


짧은 요약(Abstract) :



이 논문의 초록에서는 의학과 법률 분야에서의 진실의 개념이 어떻게 다르게 정의되는지를 설명합니다. 의학에서는 주장이 안정된 생물학적 현실에 기반한 경험적 증거로 뒷받침될 때 유효하지만, 법률에서는 진실이 관할권, 시간적 유효성, 권위 있는 출처의 계층에 따라 달라진다고 강조합니다. 최근 대규모 언어 모델(LLMs)이 의학 면허 시험에서 성공을 거둔 것은 법률 분야에서도 유사한 능력을 기대하게 만들었지만, 두 분야 간의 중요한 차이를 간과하게 됩니다. 법률 성과는 추론보다는 외부 권위가 언제 적용되고 유효하며 모순되지 않는지를 판단하는 데 더 의존합니다. 이 연구는 법률 추론을 의학 기준과 비교하여 네 가지 축(지식 회상, 기초, 신뢰도, 강건성)을 통해 평가하는 진단 프레임워크를 도입합니다. 연구 결과, 법률 LLM은 인용된 자료가 유용한지 오해의 소지가 있는지를 평가하는 데 어려움을 겪고 있으며, 외부 참조가 모델의 내부 지식과 충돌할 때 권위 있는 잘못된 정보에 과신하는 경향이 있음을 보여줍니다.




The abstract of this paper discusses how the concept of truth is defined differently in the fields of medicine and law. In medicine, claims remain valid when supported by empirical evidence grounded in stable biological reality, whereas in law, truth is contingent upon jurisdiction, temporal validity, and the hierarchy of authoritative sources. The recent success of large language models (LLMs) on medical licensing examinations has fostered expectations of similar competence in the legal domain, but this analogy obscures a critical distinction between the two fields. Legal performance often relies more on determining when external authority is applicable, valid, and non-contradictory, rather than on inference. The study introduces a comparative diagnostic framework that evaluates legal reasoning against medical baselines along four axes (knowledge recall, grounding, confidence, and robustness). The findings reveal that legal LLMs struggle to assess when retrieved citations are useful or misleading, exhibiting overconfidence in misleading authoritative information when external references conflict with the model's internal knowledge.


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



이 논문에서는 법률 분야에서 대규모 언어 모델(LLM)의 성능을 평가하기 위해 새로운 진단 프레임워크를 제안합니다. 이 프레임워크는 법률적 추론을 의료 분야의 기준과 비교하여 네 가지 축(지식 회상, 지식 기반, 지식 신뢰도, 형식 변형)을 통해 평가합니다. 

1. **모델**: 연구에서는 다양한 LLM을 사용합니다. Gemini-3-Flash-Preview와 GPT-5-Mini를 생성 모델과 평가 모델로 각각 사용하여, 생성과 평가 과정에서의 오염을 방지합니다. 이 모델들은 법률 문서의 복잡한 관계를 이해하고, 법적 문맥에서의 지식 회상 및 신뢰도를 평가하는 데 사용됩니다.

2. **특별한 아키텍처**: 연구에서 사용된 모델들은 대규모 언어 모델로, 특히 법률 문서와 같은 복잡한 텍스트를 처리할 수 있도록 설계되었습니다. 이 모델들은 문서 간의 관계를 이해하고, 법적 규정의 유효성을 평가하는 데 필요한 복잡한 추론을 수행할 수 있는 능력을 갖추고 있습니다.

3. **트레이닝 데이터**: 연구에서는 유럽 연합 법률의 공식 저장소인 EUR-Lex에서 파생된 데이터셋인 LEGAL-LINK-EU를 사용합니다. 이 데이터셋은 법률 문서 간의 관계를 기반으로 하여, 법적 유효성을 평가하는 데 필요한 다양한 질문을 생성합니다. 데이터셋은 법률 문서의 시간적 변화와 계층적 관계를 반영하여, 모델이 법적 문맥을 이해하는 데 필요한 정보를 제공합니다.

4. **특별한 기법**: 연구에서는 Genetic-Pareto GEPA 알고리즘을 사용하여 MCQ(다중 선택 질문) 생성을 최적화합니다. 이 알고리즘은 법률적 제약을 내재화하고, 표면적인 규칙에 과적합되지 않도록 다양한 목표를 동시에 최적화합니다. 이를 통해 생성된 질문은 법적 문맥에서의 복잡한 추론을 요구하며, 모델이 법적 지식을 효과적으로 활용할 수 있도록 돕습니다.

이러한 방법론을 통해 연구는 LLM이 법률적 권위와 변화하는 법적 기준에 얼마나 민감한지를 평가하고, 법률적 추론에서의 취약점을 드러내는 데 기여하고자 합니다.

---




This paper proposes a new diagnostic framework to evaluate the performance of large language models (LLMs) in the legal domain. The framework assesses legal reasoning against medical baselines along four axes: knowledge recall, knowledge grounding, knowledge confidence, and format perturbation.

1. **Models**: The study employs a variety of LLMs, specifically using Gemini-3-Flash-Preview as the generator and GPT-5-Mini as the evaluator. This separation helps prevent contamination between the generation and evaluation processes. These models are designed to understand complex relationships in legal documents and assess knowledge recall and confidence in legal contexts.

2. **Special Architecture**: The models used in the study are large-scale language models specifically designed to handle complex texts such as legal documents. They possess the ability to perform the intricate reasoning required to evaluate the validity of legal provisions and understand the relationships between documents.

3. **Training Data**: The research utilizes a dataset derived from EUR-Lex, the official repository of European Union law, known as LEGAL-LINK-EU. This dataset generates various questions necessary for assessing legal validity based on the relationships between legal documents. It reflects the temporal changes and hierarchical relationships of legal texts, providing the information needed for models to comprehend legal contexts.

4. **Special Techniques**: The study employs the Genetic-Pareto GEPA algorithm to optimize the generation of multiple-choice questions (MCQs). This algorithm allows for the internalization of legal constraints and optimizes multiple objectives simultaneously to avoid overfitting to superficial rules. The questions generated require complex reasoning within a legal context, aiding the model in effectively utilizing legal knowledge.

Through these methodologies, the research aims to evaluate how sensitive LLMs are to legal authority and evolving legal standards, contributing to the understanding of vulnerabilities in legal reasoning.


<br/>
# Results



이 연구에서는 법률 및 의학 분야에서 대규모 언어 모델(LLM)의 성능을 비교하고, 법률적 권위와 변화하는 법적 기준에 대한 모델의 취약성을 평가하기 위해 LEGAL-LINK-EU라는 새로운 벤치마크를 도입했습니다. 연구의 주요 결과는 다음과 같습니다.

1. **모델 성능 비교**: 다양한 LLM 모델을 사용하여 법률 및 의학 관련 질문에 대한 성능을 평가했습니다. 예를 들어, Gemini-2.5-Flash 모델은 법률 질문에서 70.5%의 정확도를 기록했으며, 의학 질문에서는 86.9%의 정확도를 보였습니다. 이는 법률 모델이 의학 모델에 비해 상대적으로 낮은 성능을 보임을 나타냅니다.

2. **지식 회상(Knowledge Recall)**: 법률 모델은 내부 지식에 의존할 때 낮은 성능을 보였으며, 의학 모델은 높은 정확도를 유지했습니다. 예를 들어, GPT-OSS 120B 모델은 법률 질문에서 62.6%의 정확도를 기록한 반면, 의학 질문에서는 84.1%의 정확도를 기록했습니다.

3. **지식 기반 강화(Knowledge Grounding)**: 법률 모델은 외부 문맥을 제공했을 때 성능이 크게 향상되었지만, 여전히 의학 모델에 비해 낮은 성능을 보였습니다. 법률 모델은 외부 문맥에 대한 의존도가 높아, 잘못된 정보에 더 쉽게 영향을 받는 경향이 있었습니다.

4. **신뢰도(Confidence) 및 과신(Overconfidence)**: 법률 모델은 잘못된 인용에 대해 과신하는 경향이 있었으며, 이는 모델이 외부 권위에 지나치게 의존하는 결과로 나타났습니다. 예를 들어, GPT-OSS 120B 모델은 13.4%의 신뢰도를 기록하여, 상대적으로 작은 모델인 Qwen-3 8B(20.3%)보다 낮은 성능을 보였습니다.

5. **형식적 변형(Format Perturbation)**: 법률 모델은 질문 형식의 변화에 민감하게 반응하여, 표면적인 패턴에 의존하는 경향이 있음을 보여주었습니다. 예를 들어, "None Provided"와 "Select Incorrect"와 같은 변형에서 법률 모델은 높은 정확도를 기록했지만, 이는 실제로는 의미 있는 이해가 아닌 패턴 매칭에 의한 결과로 해석될 수 있습니다.

이 연구는 LLM이 법률 분야에서의 권위 있는 정보에 대한 신뢰성을 평가하는 데 있어 중요한 통찰을 제공하며, 향후 법률적 분석을 위한 모델의 개선 방향을 제시합니다.

---




This study introduced a new benchmark called LEGAL-LINK-EU to compare the performance of large language models (LLMs) in legal and medical domains and to assess the models' vulnerabilities to legal authority and evolving legal standards. The main findings of the study are as follows:

1. **Model Performance Comparison**: Various LLMs were evaluated on legal and medical questions. For instance, the Gemini-2.5-Flash model achieved an accuracy of 70.5% on legal questions and 86.9% on medical questions. This indicates that legal models performed relatively worse compared to medical models.

2. **Knowledge Recall**: Legal models showed lower performance when relying on internal knowledge, while medical models maintained high accuracy. For example, the GPT-OSS 120B model recorded an accuracy of 62.6% on legal questions, while it achieved 84.1% on medical questions.

3. **Knowledge Grounding**: Legal models significantly improved their performance when provided with external context, but still performed worse than medical models. Legal models exhibited a high dependency on external context, making them more susceptible to misleading information.

4. **Confidence and Overconfidence**: Legal models tended to exhibit overconfidence in the face of misleading citations, indicating a tendency to rely excessively on external authority. For instance, the GPT-OSS 120B model had a confidence score of only 13.4%, which was lower than the smaller Qwen-3 8B model (20.3%).

5. **Format Perturbation**: Legal models showed sensitivity to changes in question format, indicating a reliance on superficial patterns rather than genuine understanding. For example, in perturbations like "None Provided" and "Select Incorrect," legal models achieved high accuracy, but this could be interpreted as a result of pattern matching rather than meaningful comprehension.

This study provides important insights into the reliability of LLMs concerning authoritative information in the legal domain and suggests directions for future improvements in legal analysis models.


<br/>
# 예제



이 논문에서는 법률 관련 질문 응답 시스템을 평가하기 위해 LEGAL-LINK-EU라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 유럽 연합 법률 문서 간의 관계를 기반으로 한 다중 선택 질문(MCQ)을 생성하는 데 사용됩니다. 이 시스템은 두 개의 법률 문서와 그들 간의 관계를 입력으로 받아, 법률적 추론을 요구하는 질문을 생성합니다.

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **입력**: 두 개의 법률 문서와 그들 간의 관계 유형(예: "repeals", "completes")이 주어집니다.
   - **출력**: 시스템은 다음과 같은 정보를 포함하는 JSON 객체를 생성합니다:
     - `reasoning`: 법률 조항을 인용하며, 여러 단계의 추론을 설명하는 간결한 번호 매기기.
     - `question`: 두 문서의 내용을 바탕으로 한 구체적인 시나리오를 설정하는 질문.
     - `options`: 정답과 세 개의 오답을 포함한 선택지.
     - `correct_answer`: 정답의 문자와 그에 대한 간단한 설명.

2. **테스트 데이터**:
   - **입력**: 새로운 두 개의 법률 문서와 그들 간의 관계 유형이 주어집니다.
   - **출력**: 시스템은 트레이닝 데이터와 유사한 형식으로 질문과 선택지를 생성합니다. 이 질문은 법률 문서 간의 관계를 이해하고, 복잡한 법률적 상황을 해결하는 데 필요한 다단계 추론을 요구합니다.

#### 구체적인 테스크 예시

- **입력 예시**:
  - Document 1: "Regulation (EEC) No 3624/83"
  - Document 2: "Regulation (EEC) No 3222/83"
  - Relationship Type: "implicitly_repeals"

- **출력 예시**:
  - `reasoning`: 
    1. Regulation (EEC) No 3222/83는 Regulation (EEC) No 3624/83의 특정 조항을 대체합니다.
    2. 이로 인해 특정 어종에 대한 쿼터가 변경됩니다.
    3. 따라서, 1983년 12월 30일 이후의 규정은 Regulation (EEC) No 3222/83에 따라야 합니다.
  - `question`: "1983년 12월 30일 이후, saithe와 herring의 어획은 어떤 규정에 따라야 합니까?"
  - `options`: 
    - (A) Regulation (EEC) No 198/83에 따라야 한다.
    - (B) Regulation (EEC) No 3624/83에 따라야 한다.
    - (C) Regulation (EEC) No 3222/83에 따라야 한다. (정답)
    - (D) 두 규정 모두 적용되지 않는다.
  - `correct_answer`: "C - Regulation (EEC) No 3222/83는 saithe의 쿼터를 규정하고 있습니다."

---




This paper introduces a new benchmark called LEGAL-LINK-EU, designed to evaluate legal question-answering systems. This benchmark generates multiple-choice questions (MCQs) based on the relationships between European Union legal documents. The system takes two legal documents and their relationship as input to create questions that require legal reasoning.

#### Training Data and Test Data

1. **Training Data**:
   - **Input**: Two legal documents and the type of relationship between them (e.g., "repeals", "completes").
   - **Output**: The system generates a JSON object containing:
     - `reasoning`: A concise, numbered explanation of the legal provisions cited, showing multi-step reasoning.
     - `question`: A question that sets a specific scenario based on the contents of the two documents.
     - `options`: A set of choices including the correct answer and three distractors.
     - `correct_answer`: A letter indicating the correct answer along with a brief justification.

2. **Test Data**:
   - **Input**: New pairs of legal documents and their relationship type.
   - **Output**: The system generates questions and options in a format similar to the training data. These questions require understanding the relationship between legal documents and solving complex legal scenarios through multi-step reasoning.

#### Specific Task Example

- **Input Example**:
  - Document 1: "Regulation (EEC) No 3624/83"
  - Document 2: "Regulation (EEC) No 3222/83"
  - Relationship Type: "implicitly_repeals"

- **Output Example**:
  - `reasoning`: 
    1. Regulation (EEC) No 3222/83 implicitly replaces certain provisions of Regulation (EEC) No 3624/83.
    2. This leads to changes in quotas for specific species.
    3. Therefore, after December 30, 1983, the regulations must comply with Regulation (EEC) No 3222/83.
  - `question`: "Which regulation governs saithe and herring catches after December 30, 1983?"
  - `options`: 
    - (A) It should comply with Regulation (EEC) No 198/83.
    - (B) It should comply with Regulation (EEC) No 3624/83.
    - (C) It should comply with Regulation (EEC) No 3222/83. (Correct Answer)
    - (D) Neither regulation applies.
  - `correct_answer`: "C - Regulation (EEC) No 3222/83 governs the quotas for saithe."

<br/>
# 요약


이 논문에서는 법적 추론을 평가하기 위해 LEGAL-LINK-EU라는 새로운 벤치마크를 도입하고, 이를 통해 대형 언어 모델(LLM)의 법적 지식과 신뢰성을 분석하였다. 연구 결과, 법적 LLM은 인용된 자료에 대한 과신과 구조적 취약성을 보이며, 이는 법적 문서의 복잡한 관계를 이해하는 데 어려움을 겪는 것으로 나타났다. 또한, LLM의 성능은 모델의 크기에 따라 달라지며, 더 큰 모델이 오히려 잘못된 권위에 더 쉽게 의존하는 경향이 있음을 발견하였다.

---

This paper introduces a new benchmark called LEGAL-LINK-EU to evaluate legal reasoning and analyzes the legal knowledge and reliability of large language models (LLMs). The findings reveal that legal LLMs exhibit overconfidence in cited materials and structural fragility, struggling to comprehend the complex relationships within legal documents. Additionally, the performance of LLMs varies with model size, with larger models showing a tendency to rely more on misleading authority.

<br/>
# 기타



1. **다이어그램 및 피규어**:
   - **Figure 1**: Legal profiles의 성능을 보여주는 그래프는 모델의 성능을 Knowledge Recall (KR), Knowledge Grounding (KG), Knowledge Confidence (KC), Format Perturbation (FP) 네 가지 축으로 나누어 비교합니다. 이 그래프는 법률 모델이 지식 회상에서 낮은 성능을 보이는 반면, 의학 모델은 높은 성능을 유지하는 경향이 있음을 보여줍니다. 이는 법률 모델이 외부 권위에 대한 의존도가 높고, 내부 지식이 부족함을 시사합니다.
   - **Figure 2**: Knowledge confidence degradation을 보여주는 그래프는 모델의 정확도가 퍼트리션 밀도에 따라 어떻게 감소하는지를 나타냅니다. 법률 모델은 의학 모델보다 더 급격한 정확도 감소를 보이며, 이는 법률 모델이 외부 정보에 더 민감하다는 것을 나타냅니다.
   - **Figure 3**: Sycophancy indices를 비교하는 그래프는 모델의 크기가 증가할수록 법률 도메인에서의 sycophancy가 심화된다는 것을 보여줍니다. 이는 대형 모델이 잘못된 권위에 더 쉽게 의존하게 됨을 나타냅니다.

2. **테이블**:
   - **Table 1**: Cross-domain diagnostic comparison은 법률 및 의학 도메인에서의 모델 성능을 비교합니다. 법률 모델은 KR에서 낮은 성능을 보이며, KG에서의 성능 향상이 두드러집니다. 이는 법률 모델이 외부 문맥에 의존하는 경향이 있음을 나타냅니다.
   - **Table 2**: LEGAL-LINK-EU의 관계 유형별 성능을 보여줍니다. 복잡한 관계(예: implicitly repeals)에서 성능 저하가 두드러지며, 이는 법률 모델이 시간적 논리를 처리하는 데 어려움을 겪고 있음을 시사합니다.
   - **Table 3**: Format perturbation analysis는 법률 및 의학 도메인에서의 모델 성능을 비교합니다. 법률 모델은 표면적 패턴에 의존하는 경향이 있으며, 이는 진정한 추론보다는 구조적 규칙에 의존하고 있음을 나타냅니다.

3. **어펜딕스**:
   - 어펜딕스에서는 LEGAL-LINK-EU 데이터셋의 구성, 평가 방법, 그리고 모델 최적화 과정에 대한 세부 정보를 제공합니다. 특히, GEPA 알고리즘을 사용하여 MCQ 생성을 최적화하는 방법이 설명되어 있으며, 이는 법률 모델의 성능을 향상시키기 위한 중요한 접근법입니다.




1. **Diagrams and Figures**:
   - **Figure 1**: The graph showing legal profiles illustrates model performance across four axes: Knowledge Recall (KR), Knowledge Grounding (KG), Knowledge Confidence (KC), and Format Perturbation (FP). It indicates that legal models perform poorly in knowledge recall, while medical models maintain high performance. This suggests that legal models are highly dependent on external authority and lack internal knowledge.
   - **Figure 2**: The graph depicting knowledge confidence degradation shows how model accuracy decreases with perturbation density. Legal models exhibit a steeper decline in accuracy compared to medical models, indicating that legal models are more sensitive to external information.
   - **Figure 3**: The graph comparing sycophancy indices reveals that as model size increases, sycophancy in the legal domain intensifies. This indicates that larger models are more prone to rely on misleading authority.

2. **Tables**:
   - **Table 1**: The cross-domain diagnostic comparison shows model performance in legal and medical domains. Legal models exhibit low performance in KR, with significant improvement in KG, indicating a reliance on external context.
   - **Table 2**: This table presents performance by relationship type in LEGAL-LINK-EU. Performance drops significantly for complex relationships (e.g., implicitly repeals), suggesting that legal models struggle with temporal logic.
   - **Table 3**: The format perturbation analysis compares model performance in legal and medical domains. Legal models tend to rely on superficial patterns, indicating a dependence on structural conventions rather than genuine reasoning.

3. **Appendix**:
   - The appendix provides detailed information on the composition of the LEGAL-LINK-EU dataset, evaluation methods, and the model optimization process. It particularly describes the use of the GEPA algorithm to optimize MCQ generation, which is a crucial approach for enhancing the performance of legal models.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Molfetta2026,
  author = {Lorenzo Molfetta and Alessio Cocchieri and Luca Ragazzi and Ilaria Bartolini and Marco Patella and Gianluca Moro},
  title = {Sycophants in the Courtroom: Are LLMs Fragile to Juridical Authority and Evolving Legal Standards?},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {10865--10886},
  year = {2026},
  month = {July},
  publisher = {Association for Computational Linguistics},

}
```

### 시카고 스타일

Lorenzo Molfetta, Alessio Cocchieri, Luca Ragazzi, Ilaria Bartolini, Marco Patella, and Gianluca Moro. "Sycophants in the Courtroom: Are LLMs Fragile to Juridical Authority and Evolving Legal Standards?" In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 10865–10886. Association for Computational Linguistics, July 2026.
