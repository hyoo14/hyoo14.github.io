---
layout: post
title:  "[2026]TimeX: Phenotype Onset Extraction from Clinical Narratives"
date:   2026-11-27 09:45:04 -0000
categories: study
---

{% highlight ruby %}


**한줄 요약:**

TimeX는 임상 기록에서 환자의 증상(phenotype)과 관련된 시간 표현을 LLM으로 추출하고 정규화하여, 단순 기록 날짜보다 실제 증상 발생 시점(onset)을 더 정확하게 추정하는 파이프라인이다.

TimeX is an LLM-based pipeline that extracts and normalizes temporal information from clinical narratives to estimate phenotype onset more accurately than conventional documentation timestamps.

**짧은 요약(Abstract) :**

전자건강기록(EHR)에서 특정 증상이 **언제 처음 발생했는지(phenotype onset)** 파악하는 것은 질병의 진행 과정을 이해하고 조기 진단하는 데 중요하다. 그러나 기존 연구들은 증상 발생 시점 대신 의료진이 기록을 작성한 날짜(documentation timestamp)를 사용하는 경우가 많아 실제 증상 발생 시점과 차이가 발생한다.

이 논문에서는 이러한 문제를 해결하기 위해 **TimeX**라는 LLM 기반 파이프라인을 제안한다. TimeX는 Llama-3.1-8B-Instruct를 활용하여 가족력 제거, 증상 추출, 부정 표현 판별, 시간 표현 추출 및 정규화를 수행하고, 최종적으로 증상의 최초 발생 시점을 추정한다.

102개의 임상 기록을 대상으로 평가한 결과, TimeX는 시간 정보 추출 및 정규화에서 **81.24%의 정확도**를 달성하여 기존 5개 비교 모델보다 최소 14.86%p 높은 성능을 보였다.

또한 네 가지 희귀질환 코호트에 적용한 결과, 임상 서술에서 추정한 증상 발생 시점이 기록 날짜보다 훨씬 이전인 경우가 많았다. 이는 기록 날짜만으로 질병의 발생 과정을 분석할 경우 중요한 초기 증상을 놓칠 수 있음을 보여준다.

Identifying when a phenotype first appears is essential for understanding disease progression and enabling early diagnosis. However, conventional EHR-based studies often use documentation timestamps as proxies for phenotype onset, potentially introducing substantial temporal inaccuracies.

This paper introduces **TimeX**, an LLM-based pipeline that uses Llama-3.1-8B-Instruct to extract phenotype onset information from clinical narratives. The pipeline integrates family history filtering, phenotype extraction, negation detection, temporal expression extraction, and timestamp normalization.

Evaluated on 102 manually annotated clinical notes, TimeX achieved **81.24% accuracy** in phenotype onset timestamp extraction, outperforming five baseline methods by at least 14.86 percentage points.

A case study involving four rare disease cohorts further demonstrated that phenotype onsets extracted from clinical narratives frequently occurred substantially earlier than their documentation timestamps.

The findings suggest that narrative-based temporal extraction can improve disease trajectory characterization and potentially support earlier diagnosis.

# **Useful sentences :**

- **"Documentation time is not necessarily phenotype onset time."**

- 임상 기록이 작성된 날짜가 실제 증상 발생 시점을 의미하는 것은 아니다.
- This is the central motivation behind TimeX.

- **"Temporal information is often implicit, vague, or missing in clinical narratives."**

- 임상 기록의 시간 정보는 명시적으로 작성되지 않고, 상대적이거나 모호한 표현으로 나타나는 경우가 많다.
- This makes temporal information extraction difficult for traditional NLP systems.

- **"Phenotype onset extraction requires both clinical concept recognition and temporal reasoning."**

- 증상 발생 시점 추출은 단순히 증상명을 찾는 것뿐 아니라 해당 증상과 시간 표현 사이의 관계를 이해하는 과정이 필요하다.
- TimeX addresses these tasks through separate modules.

- **"Narrative-derived onset estimates can reveal symptoms that precede formal disease diagnosis."**

- 임상 서술에서 추출한 증상 발생 시점은 공식 진단 이전에 나타난 초기 증상들을 확인하는 데 유용하다.
- This is particularly relevant to rare diseases with long diagnostic delays.

- **"Contextual interpretation remains a major challenge in clinical temporal reasoning."**

- 서로 다른 문장에 등장하는 증상과 시간 표현을 올바르게 연결하는 것은 여전히 중요한 기술적 한계이다.
- Most TimeX errors arise from contextual misinterpretation.

위 문장들은 논문의 핵심 주장을 정리한 문장으로, 원문에서 그대로 발췌한 직접 인용은 아니다.

[Paper link](https://doi.org/10.1038/s44401-026-00099-8)

~~Lecture link~~

[GitHub](https://github.com/MadGenius0118/TimeX)

# **단어정리**

* **Phenotype (표현형)**: 환자에게 관찰되는 증상이나 임상적 특징. 예를 들어 두통, 피부 발진, 고혈압 등이 해당한다.

- An observable clinical characteristic or symptom of a patient.

* **Phenotype Onset (표현형 발생 시점)**: 특정 증상이 처음 나타난 시점.

- The time when a particular phenotype first appears.

* **Electronic Health Record (EHR)**: 환자의 진료 기록, 진단, 검사 결과 등을 포함하는 전자건강기록.

- A digital collection of patient health information, including clinical notes and structured medical data.

* **Clinical Narrative (임상 서술 기록)**: 의료진이 자연어로 작성한 비정형 진료 기록.

- Unstructured clinical text written by healthcare professionals.

* **Documentation Timestamp (기록 시각)**: 의료 기록이 작성되거나 저장된 시점으로, 실제 증상 발생 시점과 다를 수 있다.

- The timestamp associated with clinical documentation, which may differ from the actual event time.

* **Temporal Information Extraction (TIE)**: 문장에서 시간 표현을 찾아 관련 임상 사건과 연결하는 작업.

- The task of identifying temporal expressions and associating them with clinical events.

* **Temporal Normalization (시간 정규화)**: 다양한 시간 표현을 `YYYY-MM-DD`와 같은 표준 날짜 형식으로 변환하는 작업.

- Converting different temporal expressions into a standardized date representation.

* **Implicit Temporal Expression (암묵적 시간 표현)**: "two weeks ago", "recently"처럼 문맥이나 기준 날짜를 통해 해석해야 하는 시간 표현.

- A relative or context-dependent temporal expression.

* **Explicit Temporal Expression (명시적 시간 표현)**: "March 2010", "2020-01-12"처럼 특정 연도나 날짜가 직접 언급된 표현.

- A temporal expression that explicitly specifies a calendar date or period.

* **Human Phenotype Ontology (HPO)**: 인간의 임상 표현형을 표준화하여 정의한 온톨로지.

- A standardized ontology describing human phenotypic abnormalities.

* **PhenoBERT**: 임상 텍스트에서 표현형을 추출하고 HPO 용어와 연결하는 BERT 기반 모델.

- A BERT-based model for recognizing phenotypes and mapping them to HPO concepts.

* **Negation Detection (부정 표현 탐지)**: "no fever"처럼 증상이 존재하지 않는다는 표현을 식별하는 작업.

- Identifying whether a phenotype mention is negated.

* **Family History Filtering (가족력 필터링)**: 가족에게 나타난 질환을 환자 자신의 증상으로 잘못 인식하지 않도록 제거하는 과정.

- Removing family medical history to avoid attributing relatives' conditions to the patient.

* **Hidden Chain-of-Thought (Hidden CoT)**: 모델이 단계적 추론을 수행하도록 지시하되 중간 추론 내용을 출력하지 않는 프롬프트 방식.

- A prompting strategy that guides multi-step reasoning without requiring intermediate reasoning steps in the output.

* **Phenotype Onset Estimation (표현형 발생 시점 추정)**: 다양한 기록에서 확보한 시간 정보를 결합해 증상의 최초 발생 시점을 결정하는 작업.

- Estimating the earliest occurrence of a phenotype using multiple temporal sources.

# **Methodology**

TimeX의 핵심 아이디어는 **증상을 먼저 식별하고, 해당 증상과 연결되는 시간 표현을 추론한 뒤, 이를 표준 날짜로 변환하는 것**이다.

전체 파이프라인은 다음 네 단계로 구성된다.

**1. Data Preprocessing**

Family History Filtering

**2. Phenotype Extraction**

PhenoBERT + Negation Detection

**3. Temporal Expression Extraction & Normalization**

Llama-3.1-8B-Instruct + Prompting

**4. Phenotype Onset Estimation**

Earliest Timestamp Aggregation

논문 Figure 1의 전체 파이프라인을 단순화한 구성도

## **1. Data Preprocessing**

먼저 Llama-3.1-8B-Instruct를 사용하여 임상 기록에서 가족력을 설명하는 문장을 제거한다.

예를 들어 다음 문장이 있다고 가정하자.

- "The patient's father has asthma."
- "The patient has experienced chest pain."

첫 번째 문장의 asthma는 환자 본인의 증상이 아니므로 제거하고, 두 번째 문장의 chest pain은 유지한다.

이 과정은 가족의 질병을 환자의 표현형으로 잘못 추출하는 오류를 방지한다.

## **2. Phenotype Extraction**

가족력을 제거한 임상 기록에서 PhenoBERT를 이용하여 표현형을 추출하고 HPO 개념에 매핑한다.

이후 BERT 기반 negation detection 모델을 통해 실제로 존재하는 증상만 남긴다.

예를 들어 다음과 같다.

- "Patient has headache." → Headache 유지
- "Patient denies fever." → Fever 제거
- "No evidence of skin rash." → Skin rash 제거

필요한 경우 HPO와 외부 질병-표현형 지식베이스를 이용하여 분석 대상 질병과 관련 없는 표현형도 추가로 필터링할 수 있다.

## **3. Temporal Expression Extraction & Normalization**

이 단계가 TimeX의 핵심이다.

Llama-3.1-8B-Instruct를 이용하여 각 표현형에 연결된 시간 표현을 추출하고, 이를 `YYYY-MM-DD` 형식으로 정규화한다.

논문은 두 가지 접근법을 비교한다.

**Approach 1: Full-length clinical note**

전체 임상 기록과 추출된 HPO 표현형을 모델에 입력하고 시간 정보를 추출한다.

**Approach 2: Truncated clinical note + Hidden CoT**

표현형이 등장하는 문장을 중심으로 앞의 세 문장과 뒤의 두 문장만 선택하여 모델에 전달한다.

[ S=[S_{i-3},S_{i-2},S_{i-1},S_i,S_{i+1},S_{i+2}] ]

여기서 (S_i)는 대상 표현형이 등장하는 문장이다.

이 방식은 전체 문서를 처리할 때 발생할 수 있는 *lost-in-the-middle* 문제를 줄이고, 관련 시간 정보에 집중하기 위해 설계되었다.

시간 정규화 규칙은 다음과 같다.

| Temporal expression | Normalization    |
| ------------------- | ---------------- |
| in March 2010       | 2010-03-01       |
| in 2015             | 2015-01-01       |
| 2 weeks ago         | 기준 날짜 - 14일      |
| 3 months ago        | 기준 날짜 - 90일      |
| recently            | 기준 날짜 이전 2주 범위   |
| currently / present | 기록 날짜            |
| past / chronic      | onset not stated |

동일한 표현형에 여러 시간 표현이 등장하면 가장 이른 시점을 선택한다. 시간 정보가 없으면 `not stated`로 처리한다.

## **4. Phenotype Onset Estimation**

최종적으로 세 가지 정보원을 결합한다.

1. Structured EHR의 condition start timestamp
2. Clinical note의 documentation timestamp
3. TimeX가 임상 서술에서 추출한 phenotype timestamp

각 환자의 표현형별로 세 정보원에서 확인된 가장 이른 날짜를 최종 onset estimate로 선택한다.

또한 출생 이전이거나 해당 기록 작성일 이후인 날짜 등 부적절한 timestamp를 제거하는 품질 관리 과정을 수행한다.

TimeX consists of four primary modules designed to identify phenotypes, interpret their temporal context, and estimate their earliest onset.

## **1. Data Preprocessing**

TimeX first removes family history statements using Llama-3.1-8B-Instruct.

This prevents conditions mentioned in relatives' medical histories from being incorrectly attributed to the patient.

## **2. Phenotype Extraction**

PhenoBERT identifies phenotype mentions and maps them to standardized HPO concepts.

A BERT-based negation detection model then removes negated phenotype mentions. An optional filtering step selects phenotypes associated with the target disease using external knowledge bases.

## **3. Temporal Expression Extraction & Normalization**

Llama-3.1-8B-Instruct identifies temporal expressions associated with extracted phenotypes.

The authors investigate two approaches: processing the full clinical note and processing a shorter context window around each phenotype mention.

The truncated approach uses the target sentence, its three preceding sentences, and its two following sentences.

Instruction-based prompting with hidden CoT guides the model to identify temporal evidence, infer the corresponding date, and normalize the result to ISO 8601.

When multiple temporal expressions are associated with a phenotype, the earliest timestamp is selected. If no temporal evidence exists, the system returns `not stated`.

## **4. Phenotype Onset Estimation**

The system combines timestamps from structured EHR conditions, note documentation dates, and extracted clinical narrative information.

For each patient and phenotype, the earliest valid timestamp across these sources is selected as the final onset estimate.

Quality control excludes implausible dates, including timestamps before birth or after the corresponding documentation time.

# **Results**

## **1. TimeX Performance**

102개의 수작업 검증 임상 기록에서 357개의 서로 다른 표현형을 대상으로 평가하였다.

| Evaluation                                           | Performance |
| ---------------------------------------------------- | ----------- |
| Family history filtering accuracy                    | 97.06%      |
| Phenotype onset timestamp accuracy (truncated notes) | **81.24%**  |
| Phenotype onset timestamp accuracy (full notes)      | 84.07%      |
| Verbatim temporal expression extraction accuracy     | 82.92%      |
| Average runtime per note                             | 4.87 s      |

시간 정보 추출 및 정규화에서 TimeX는 기존 5개 baseline보다 최소 14.86%p 높은 정확도를 보였다.

논문에서는 전체 기록을 사용한 방식의 정확도가 더 높게 측정되었지만, 짧게 분할한 기록과 hidden CoT를 결합한 방식은 긴 문서 처리의 어려움을 줄이는 방법으로 제시한다.

Phenotype onset timestamp extraction accuracy

TimeX variants and the Llama-3.1 baseline, from the reported evaluation



## **2. Runtime Efficiency**

TimeX는 임상 기록 하나를 처리하는 데 평균 약 4.87초가 소요되었다.

| Module                              | Runtime per note |
| ----------------------------------- | ---------------- |
| Family history filtering            | 0.47 s           |
| Phenotype extraction                | 0.023 s          |
| Temporal extraction & normalization | 4.38 s           |

전체 계산 시간의 대부분은 LLM을 이용한 시간 표현 추출 및 정규화에서 발생한다.

## **3. Rare Disease Case Study**

네 가지 희귀질환 코호트를 대상으로 추가 실험을 수행하였다.

| Disease                         | Patients | Clinical notes |
| ------------------------------- | -------- | -------------- |
| Behcet syndrome                 | 197      | 1,656          |
| Pulmonary Arterial Hypertension | 9,363    | 24,700         |
| Fabry disease                   | 103      | 3,763          |
| Whipple's disease               | 63       | 2,371          |

각 코호트에서 30개씩 총 120개의 임상 기록을 별도로 검증한 결과, 시간 정보 정규화 정확도는 **83.39%**였다.

또한 전체 코호트에서 추출된 표현형 언급 중 약 75~85%는 관련 시간 정보를 포함하지 않았다.

시간 정보가 존재하는 경우에도 "2 weeks ago", "several months ago"와 같은 암묵적 표현이 주를 이루었다.

## **4. Difference Between Estimated and Documented Onset**

기록에 시간 정보가 존재하는 표현형 중 약 45%는 추정된 onset이 기록 날짜보다 1년 이상 이전이었다.

특히 Behcet syndrome, PAH, Fabry disease에서는 일부 증상의 발생 시점이 기록 날짜보다 4년 이상 앞서는 것으로 나타났다.

Friedman test와 Dunn's post hoc test에서도 추정 onset과 documentation timestamp 사이에 통계적으로 유의한 차이가 확인되었다((p<0.001)).

이는 단순한 EHR 기록 날짜만으로 증상의 발생 시점을 판단하면 환자의 실제 질병 진행 과정을 잘못 표현할 가능성이 있음을 의미한다.

## **1. TimeX Performance**

The authors evaluated TimeX on 102 manually annotated clinical notes containing 357 distinct phenotypes.

TimeX achieved 81.24% accuracy using truncated notes and 84.07% accuracy using full-length notes.

Its timestamp extraction and normalization performance exceeded all five baselines by at least 14.86 percentage points.

The Llama-3.1 baseline achieved only 19.27% timestamp extraction accuracy, highlighting the importance of TimeX's modular architecture and task-specific prompting.

## **2. Runtime Efficiency**

TimeX required an average of 4.87 seconds per clinical note.

Temporal expression extraction and normalization accounted for most of the processing time at 4.38 seconds per note.

Family history filtering and phenotype extraction were substantially less computationally expensive.

## **3. Rare Disease Case Study**

TimeX was further evaluated on four rare disease cohorts: Behcet syndrome, pulmonary arterial hypertension, Fabry disease, and Whipple's disease.

A manually annotated subset of 120 notes produced 83.39% temporal normalization accuracy.

Across the full cohorts, approximately 75–85% of phenotype mentions lacked associated temporal information, while the available temporal expressions were predominantly implicit.

## **4. Difference Between Estimated and Documented Onset**

Among phenotype mentions with temporal attributes, approximately 45% had estimated onsets more than one year earlier than their documentation-based onsets.

Several phenotypes showed differences of four years or more.

Statistical comparisons confirmed significant differences between narrative-derived and documentation-based estimates ((p<0.001)).

These findings suggest that documentation timestamps may systematically misrepresent the timing of phenotype onset and disease progression.

# **예제**

다음은 TimeX의 동작을 이해하기 위한 예시이다. 논문의 방법론을 바탕으로 재구성한 예시이며 실제 실험에 사용된 원문 기록은 아니다.

**Input Clinical Note**

```
Documentation Date: 2024-06-15

The patient reports experiencing chest pain
for the past three months.

Two weeks ago, the patient developed
shortness of breath.

The patient's father has a history of asthma.

The patient denies fever.
```

**Step 1. Family History Filtering**

가족력을 설명하는 문장을 제거한다.

```
Removed:
The patient's father has a history of asthma.
```

**Step 2. Phenotype Extraction & Negation Detection**

추출 결과는 다음과 같다.

| Phenotype           | Status   |
| ------------------- | -------- |
| Chest pain          | Affirmed |
| Shortness of breath | Affirmed |
| Fever               | Negated  |

따라서 Chest pain과 Shortness of breath만 후속 처리한다.

**Step 3. Temporal Extraction & Normalization**

기록 작성일인 2024-06-15를 기준으로 상대적 시간 표현을 계산한다.

| Phenotype           | Time expression   | Normalized timestamp |
| ------------------- | ----------------- | -------------------- |
| Chest pain          | past three months | 2024-03-17           |
| Shortness of breath | two weeks ago     | 2024-06-01           |

여기서 3개월은 논문의 근사 규칙에 따라 90일로 계산하였다.

**Step 4. Phenotype Onset Estimation**

```
Chest pain
  Documentation time: 2024-06-15
  Narrative-derived time: 2024-03-17
  Estimated onset: 2024-03-17

Shortness of breath
  Documentation time: 2024-06-15
  Narrative-derived time: 2024-06-01
  Estimated onset: 2024-06-01
```

이 예제에서 기록 작성 날짜만 사용하면 두 증상 모두 2024년 6월 15일에 발생한 것처럼 처리된다.

하지만 TimeX는 증상별 시간 표현을 분석하여 Chest pain이 약 3개월 전, Shortness of breath가 약 2주 전에 나타났음을 구분할 수 있다.

The following example illustrates the TimeX workflow using a reconstructed clinical note.

A clinical note documented on June 15, 2024, describes chest pain lasting three months and shortness of breath beginning two weeks earlier.

TimeX first removes the father's asthma history and excludes the negated fever mention.

It then identifies the two affirmed phenotypes and their associated temporal expressions.

Using the note date as the temporal reference, the system estimates March 17, 2024, for chest pain and June 1, 2024, for shortness of breath.

Consequently, the estimated onset dates occur earlier than the documentation date.

This example demonstrates how TimeX reconstructs phenotype-specific clinical timelines from relative temporal expressions rather than assigning all symptoms the same documentation timestamp.

# **요약**

**Contribution**

이 논문의 가장 중요한 기여는 기존의 표현형 추출 문제를 **시간 정보가 포함된 표현형 추출(temporal phenotyping)** 문제로 확장했다는 점이다.

단순히 환자에게 어떤 증상이 있는지 확인하는 것을 넘어, 해당 증상이 언제 처음 나타났는지를 구조화된 정보로 변환한다.

특히 LLM 하나에 모든 작업을 맡기는 대신 가족력 제거, PhenoBERT 기반 표현형 추출, 부정 표현 탐지, LLM 기반 시간 추론을 모듈로 분리하여 성능과 해석 가능성을 높였다.

**Strengths**

- 기존 LLM 기반 baseline보다 높은 시간 정보 추출 정확도를 달성하였다.
- 명시적 시간 표현뿐 아니라 암묵적 시간 표현까지 처리한다.
- 별도의 대규모 fine-tuning 없이 instruction-based prompting을 활용한다.
- 네 가지 희귀질환 코호트에서 실제 임상 기록 날짜와 추정 onset 사이의 차이를 분석하였다.
- 오픈소스 파이프라인으로 공개하여 재현과 확장이 가능하다.

**Limitations**

- 서로 다른 문장에 등장하는 증상과 시간 표현을 연결하는 데 어려움이 있다.
- 날짜 계산 및 기준 날짜 추론 과정에서 오류가 발생할 수 있다.
- 시간 정보가 없는 표현형은 정확한 onset을 추정하기 어렵다.
- 상대적 시간 표현을 고정된 규칙으로 정규화하므로 시간적 불확실성이 완전히 해결되지는 않는다.
- 제한된 수작업 검증 데이터로 평가되었으며, 다른 의료기관이나 실제 진단 예측 작업으로 일반화할 수 있는지는 추가 검증이 필요하다.

**Key Takeaway**

TimeX는 임상 기록에서 증상이 기록된 시점과 실제로 나타난 시점을 구분하려는 접근이다. 이러한 시간 정보를 활용하면 환자의 질병 진행 과정(disease trajectory)을 더 정밀하게 분석하고, 조기 진단에 도움이 되는 초기 증상 패턴을 발견할 가능성이 있다.

**Contribution**

This paper introduces an open-source framework for temporal phenotyping that goes beyond identifying which phenotypes are present to estimating when they first occurred.

Its main methodological contribution is a modular architecture combining phenotype recognition, negation detection, temporal reasoning, and onset estimation.

**Strengths**

- Stronger timestamp extraction accuracy than conventional NLP and LLM baselines.
- Support for both explicit and implicit temporal expressions.
- Instruction-based temporal reasoning without extensive model fine-tuning.
- Evaluation on real-world clinical narratives and four rare disease cohorts.
- Open-source implementation supporting reproducibility and adaptation.

**Limitations**

- Difficulty linking distant phenotype mentions and temporal expressions.
- Occasional arithmetic and temporal reference inference errors.
- Missing temporal information in a large proportion of clinical notes.
- Uncertainty introduced by heuristic timestamp normalization.
- Limited ground-truth evaluation and insufficient validation of downstream clinical benefits.

**Key Takeaway**

TimeX demonstrates that temporal information embedded in clinical narratives can provide more informative estimates of phenotype onset than documentation timestamps alone.

Its modular design offers a practical foundation for temporal phenotyping, disease progression modeling, and future early-diagnosis research.

# **기타(다이어그램, Figures, Tables, Appendix, 인사이트)**

## **1. Figures**

**Figure 1. TimeX Computational Pipeline (Page 2)**

가족력 제거부터 표현형 추출, 시간 정규화, onset 추정까지의 전체 구조를 보여준다. 핵심은 임상 정보를 추출하는 단계와 시간 추론 단계를 분리했다는 점이다.

**Figure 2. Performance and Runtime Comparison (Page 3)**

TimeX와 기존 baseline의 시간 정보 추출 성능과 실행 시간을 비교한다. TimeX가 정확도와 효율성 측면에서 경쟁력 있는 결과를 보여준다.

**Figure 3. Distribution of Temporal Expressions (Page 4)**

임상 기록에서 표현형과 연결된 시간 정보가 얼마나 자주 나타나는지 분석한다.

Phenotype mentions without onset information



대부분의 표현형 언급에 시간 정보가 없다는 사실은 추출 모델의 정확도뿐 아니라 임상 데이터 자체의 불완전성도 중요한 문제임을 보여준다.

**Figure 4. Estimated vs. Documented Phenotype Onsets (Page 5)**

임상 서술에서 추출한 onset과 기록 날짜 사이의 차이를 시간 구간별로 비교한다. 상당수의 표현형에서 실제 추정 onset이 기록 날짜보다 수년 앞선다.

**Figure 5. Statistical Comparison of Onset Estimates (Page 6)**

DocTime, MentionTime, Best-estimate를 비교한다.

- DocTime: 임상 기록의 documentation timestamp
- MentionTime: 임상 서술에서 추출한 phenotype onset
- Best-estimate: 여러 데이터 소스를 종합한 최종 onset estimate

대부분의 비교에서 유의한 차이가 확인되었으며, 임상 서술 정보가 기존 시간 데이터에 추가적인 가치를 제공한다.

## **2. Tables**

**Table 1. Error Analysis**

| Error type                   | Percentage |
| ---------------------------- | ---------- |
| Contextual misinterpretation | 75.9%      |
| Arithmetic error             | 18.1%      |
| Reasoning error              | 6.0%       |

가장 많은 오류가 문맥 해석에서 발생했다. 이는 단순한 날짜 계산보다 **증상과 시간 정보의 올바른 연결**이 더 어려운 문제임을 의미한다.

**Table 2. Rare Disease Cohort Demographics**

네 가지 희귀질환 코호트의 환자 수, 임상 기록 수, 나이, 성별 및 인구통계학적 특성을 제공한다.

**Table 3. Temporal Normalization Rules**

상대적 시간 표현을 표준 날짜로 변환하기 위한 규칙을 정의한다.

예를 들어 `x months ago`는 기준 날짜에서 (x times 30)일을 빼는 방식으로 처리한다.

## **3. Insights**

**Insight 1. Phenotype extraction과 temporal reasoning은 서로 다른 문제이다.**

PhenoBERT를 통해 표현형을 정확하게 찾아도, 시간 정보를 잘못 연결하면 전체 onset 추정이 실패한다.

따라서 시간 추론을 별도의 모듈로 구성하는 TimeX의 접근이 중요하다.

**Insight 2. 긴 문맥을 무조건 사용하는 것이 최선은 아니다.**

전체 임상 기록에는 관련 없는 정보가 많고, LLM이 중요한 시간 표현을 놓칠 가능성이 있다.

반면 너무 짧은 문맥은 문장 사이의 시간적 관계를 잃어버릴 수 있다.

결국 정확한 시간 추출을 위해서는 단순한 truncation보다 표현형에 관련된 문장을 효과적으로 검색하는 방법이 중요할 수 있다.

**Insight 3. Onset estimation에는 불확실성 표현이 필요하다.**

"recently", "several months ago" 같은 표현은 정확한 날짜를 나타내지 않는다.

이를 하나의 날짜로 변환하면 분석하기는 편리하지만 실제 시간적 불확실성이 사라질 수 있다.

향후 연구에서는 onset을 단일 날짜 대신 가능한 시간 구간으로 표현하는 방법도 고려할 수 있다.

**Insight 4. 정확한 phenotype onset은 조기 진단 연구의 중요한 입력이 될 수 있다.**

특히 희귀질환은 첫 증상부터 확진까지 시간이 오래 걸리는 경우가 많다.

TimeX처럼 증상 발생 순서를 복원하는 시스템은 질병 진행 패턴을 파악하고 조기 진단 모델을 개발하는 데 활용될 가능성이 있다.

다만 본 논문이 실제 조기 진단 성능 향상을 직접 입증한 것은 아니다.

## **1. Figures**

**Figure 1. TimeX Computational Pipeline**

Illustrates the four-stage workflow from family history removal to final onset estimation.

**Figure 2. Performance and Runtime Comparison**

Demonstrates TimeX's accuracy and computational efficiency relative to baseline approaches.

**Figure 3. Distribution of Temporal Expressions**

Shows that most phenotype mentions lack associated temporal information, highlighting a major limitation of clinical narrative data.

**Figure 4. Estimated vs. Documented Phenotype Onsets**

Reveals substantial differences between narrative-derived onset estimates and documentation timestamps.

**Figure 5. Statistical Comparison of Onset Estimates**

Compares DocTime, MentionTime, and Best-estimate, demonstrating significant temporal discrepancies across disease cohorts.

## **2. Tables**

**Table 1. Error Analysis**

Contextual misinterpretation accounts for 75.9% of observed errors, followed by arithmetic errors (18.1%) and temporal reference reasoning errors (6.0%).

**Table 2. Rare Disease Cohort Demographics**

Summarizes the demographic and clinical characteristics of four rare disease cohorts.

**Table 3. Temporal Normalization Rules**

Defines heuristic rules for converting relative temporal expressions into standardized timestamps.

## **3. Insights**

**Insight 1. Phenotype recognition and temporal reasoning should be treated as distinct tasks.**

Accurate phenotype extraction does not guarantee accurate onset estimation. Correctly associating phenotypes with temporal evidence remains essential.

**Insight 2. Relevant context selection is more important than simply increasing context length.**

Long clinical documents may introduce irrelevant information, while excessively short windows may exclude important temporal evidence.

Retrieval-based context selection could improve future temporal extraction systems.

**Insight 3. Temporal uncertainty should be explicitly represented.**

Expressions such as "recently" and "several months ago" do not correspond to exact dates.

Representing uncertainty through temporal intervals may provide more faithful onset estimates than single normalized timestamps.

**Insight 4. Temporal phenotyping could support early disease detection.**

Reconstructing the order and timing of clinical manifestations may help identify early disease patterns, especially in rare diseases.

However, additional downstream validation is required before concluding that TimeX improves real-world diagnostic outcomes.

# **refer format:**

### **BibTeX**

```
@article{chen2026timex,
  title   = {TimeX: Phenotype Onset Extraction from Clinical Narratives},
  author  = {Chen, Fangyi and Jiang, Shiyi and
             Nguyen, Quan M. and Ta, Casey N. and
             Wang, Kai and Chung, Wendy K. and
             Weng, Chunhua},
  journal = {npj Health Systems},
  volume  = {3},
  pages   = {49},
  year    = {2026},
  doi     = {10.1038/s44401-026-00099-8},
  url     = {https://doi.org/10.1038/s44401-026-00099-8}
}
```

### **Chicago style**

Chen, Fangyi, Shiyi Jiang, Quan M. Nguyen, Casey N. Ta, Kai Wang, Wendy K. Chung, and Chunhua Weng. 2026. “TimeX: Phenotype Onset Extraction from Clinical Narratives.” *npj Health Systems* 3: 49. 