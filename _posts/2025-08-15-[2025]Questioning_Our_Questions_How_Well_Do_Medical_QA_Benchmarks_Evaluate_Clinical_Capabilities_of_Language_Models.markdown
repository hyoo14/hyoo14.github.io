---
layout: post
title:  "[2025]Questioning Our Questions: How Well Do Medical QA Benchmarks Evaluate Clinical Capabilities of Language Models?"
date:   2025-08-15 19:24:44 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

이 연구에서는 168개의 연구에서 수집한 702개의 임상 평가 데이터를 기반으로, 의료 QA 벤치마크와 실제 임상 성능 간의 상관관계를 분석  


짧은 요약(Abstract) :

이 논문의 초록에서는 최근 대형 언어 모델(LLMs)의 발전이 의료 질문-응답(QA) 벤치마크에서 인상적인 성과를 이끌어냈지만, 이러한 벤치마크가 실제 임상 능력을 얼마나 잘 반영하는지는 불확실하다고 언급하고 있습니다. 이를 해결하기 위해, 저자들은 주요 의료 QA 벤치마크(예: MedQA, MedMCQA, PubMedQA, MMLU 의학 주제)에서 LLM의 성과와 실제 임상 성과 간의 상관관계를 체계적으로 분석했습니다. 연구 데이터셋은 168개의 연구에서 85개의 LLM에 대한 702개의 임상 평가를 포함하고 있으며, 벤치마크 점수는 임상 성과와 중간 정도의 상관관계를 보였습니다(Spearman의 ρ=0.59). MedQA는 가장 예측력이 높았지만, 환자 소통, 장기적 관리, 임상 정보 추출과 같은 필수 역량을 포착하지 못했습니다. 베이지안 계층 모델링을 사용하여 대표적인 임상 성과를 추정한 결과, GPT-4와 GPT-4o가 일관되게 최고 성능을 보였으며, 종종 인간 의사와 동등하거나 그 이상으로 평가되었습니다. 이 연구는 의료 QA 벤치마크의 임상 유효성에 대한 우려가 오랫동안 제기되어 왔음에도 불구하고, 실제 임상 성과와의 정량적 분석을 제공하는 첫 번째 연구입니다.


The abstract of this paper states that recent advancements in large language models (LLMs) have led to impressive performances on medical question-answering (QA) benchmarks, but the extent to which these benchmarks reflect real-world clinical capabilities remains uncertain. To address this gap, the authors systematically analyzed the correlation between LLM performance on major medical QA benchmarks (e.g., MedQA, MedMCQA, PubMedQA, and MMLU medicine subjects) and clinical performance in real-world settings. Their dataset included 702 clinical evaluations of 85 LLMs from 168 studies, and benchmark scores demonstrated a moderate correlation with clinical performance (Spearman’s ρ=0.59). Among these, MedQA was the most predictive but failed to capture essential competencies such as patient communication, longitudinal care, and clinical information extraction. Using Bayesian hierarchical modeling, they estimated representative clinical performance and identified GPT-4 and GPT-4o as consistently top-performing models, often matching or exceeding human physicians. Despite longstanding concerns about the clinical validity of medical QA benchmarks, this study offers the first quantitative analysis of their alignment with real-world clinical performance.


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


이 연구에서는 의료 질문 응답(QA) 벤치마크가 실제 임상 성능을 얼마나 잘 반영하는지를 평가하기 위해 체계적인 방법론을 사용했습니다. 연구의 주요 방법론은 다음과 같습니다.

1. **문헌 리뷰 및 데이터 수집**: 연구팀은 2023년 1월 1일부터 2025년 1월 10일 사이에 발표된 논문을 대상으로, 대규모 언어 모델(LLM)의 임상 성능을 평가한 연구를 식별하기 위해 문헌 검색을 수행했습니다. 이 과정에서 Semantic Scholar API를 사용하여 관련 논문을 검색하고, 제목 기반 필터링을 통해 LLM 관련 용어가 포함된 논문을 선별했습니다. 최종적으로 168개의 연구에서 702개의 임상 성능 평가 데이터를 수집했습니다.

2. **성능 데이터 수집**: 수집된 데이터는 85개의 LLM에 대한 성능 평가를 포함하며, 각 모델의 성능 점수는 0-100의 스케일로 정규화되었습니다. 성능 점수는 주로 제로샷 설정에서 수집되었으며, 여러 평가 방법이 사용되었습니다.

3. **벤치마크 성능 수집**: 연구팀은 MedQA, MedMCQA, PubMedQA 등 여러 의료 QA 벤치마크의 성능 데이터를 수집했습니다. 이 데이터는 기존 문헌, 기술 보고서 및 모델 카드에서 추출되었습니다. 성능 점수는 제로샷 설정에 맞춰 표준화되었습니다.

4. **상관관계 측정**: 연구팀은 벤치마크 점수와 임상 성능 간의 상관관계를 평가하기 위해 Spearman의 순위 상관계수와 Kendall의 타우를 사용했습니다. 이 분석은 각 평가 작업 수준에서 수행되었으며, 샘플 크기에 따라 가중치를 부여했습니다.

5. **베이지안 모델링**: 연구팀은 베이지안 계층 모델링을 사용하여 각 언어 모델의 대표적인 임상 성능을 추정했습니다. 이 모델은 각 모델의 성능을 평가하기 위해 다양한 메타데이터 속성을 활용하여 임상 성능을 추정했습니다.

이러한 방법론을 통해 연구팀은 의료 QA 벤치마크와 실제 임상 성능 간의 상관관계를 정량적으로 분석하고, LLM의 임상 능력을 평가하는 데 있어 기존 벤치마크의 한계를 식별했습니다.

---




In this study, a systematic methodology was employed to evaluate how well medical question-answering (QA) benchmarks reflect real-world clinical performance. The key methodologies are as follows:

1. **Literature Review and Data Collection**: The research team conducted a literature search targeting studies published between January 1, 2023, and January 10, 2025, that evaluated the clinical performance of large language models (LLMs). Using the Semantic Scholar API, they searched for relevant papers and filtered them based on title to retain those explicitly mentioning LLM-related terms. Ultimately, 168 studies were identified, yielding 702 clinical performance evaluation data points.

2. **Performance Data Collection**: The collected data included performance evaluations for 85 LLMs, and the performance scores for each model were normalized to a scale of 0-100. These scores were primarily collected in a zero-shot setting, utilizing various evaluation methods.

3. **Benchmark Performance Collection**: The research team gathered performance data from several medical QA benchmarks, including MedQA, MedMCQA, and PubMedQA. This data was extracted from existing literature, technical reports, and model cards. The performance scores were standardized to align with the zero-shot setting.

4. **Correlation Measurement**: The team evaluated the correlation between benchmark scores and clinical performance using Spearman's rank correlation coefficient and Kendall's tau. This analysis was conducted at the evaluation task level, with weights assigned based on sample size.

5. **Bayesian Modeling**: The research team employed Bayesian hierarchical modeling to estimate the representative clinical performance of each language model. This model utilized various metadata attributes to approximate clinical performance for each model.

Through these methodologies, the research team quantitatively analyzed the correlation between medical QA benchmarks and real-world clinical performance, identifying the limitations of existing benchmarks in assessing the clinical capabilities of LLMs.


<br/>
# Results


이 연구에서는 의료 질문 응답(QA) 벤치마크와 실제 임상 성능 간의 상관관계를 분석하여, 대형 언어 모델(LLM)의 임상 능력을 평가하는 데 있어 현재의 벤치마크가 얼마나 효과적인지를 평가했습니다. 연구의 주요 결과는 다음과 같습니다.

1. **경쟁 모델**: 연구에서 평가된 LLM은 총 85개 모델로, 그 중 GPT-4와 GPT-4o가 가장 높은 성능을 보였습니다. 이 모델들은 종종 인간 의사들의 평균 성능을 초과하는 결과를 나타냈습니다. 특히, GPT-4는 다양한 임상 시나리오에서 우수한 성능을 발휘했습니다.

2. **테스트 데이터**: 연구는 168개의 연구에서 수집된 702개의 임상 평가 데이터를 기반으로 하였습니다. 이 데이터는 진단, 임상 지식, 환자 관리, 정보 추출 등 다양한 임상 작업을 포함하고 있습니다. 데이터 출처는 주로 임상 비네트, 퀴즈, FAQ 등으로 구성되었습니다.

3. **메트릭**: 연구에서는 Spearman의 순위 상관계수와 Kendall의 타우를 사용하여 벤치마크 성능과 임상 성능 간의 상관관계를 측정했습니다. MedQA는 임상 성능과의 상관관계에서 Spearman의 0.588, Kendall의 0.520을 기록하여 가장 높은 상관관계를 보였습니다. 그러나 이러한 상관관계는 벤치마크 간의 상관관계(0.675~0.787)보다 낮았습니다.

4. **비교**: MedQA는 치료, 임상 지식, 진단과 같은 핵심 의료 지식과 추론 능력을 잘 평가했지만, 환자와의 의사소통, 장기적인 환자 관리, 임상 정보 추출과 같은 실제 임상에서 중요한 역량을 충분히 반영하지 못했습니다. 반면, MMLU Medical Genetics, College Medicine, Professional Medicine, Clinical Knowledge, MedMCQA는 임상 성능과의 상관관계에서 중간 정도의 성과를 보였습니다.

이 연구는 의료 QA 벤치마크의 임상 유효성에 대한 첫 번째 체계적이고 정량적인 분석을 제공하며, 향후 벤치마크 설계에 대한 중요한 통찰을 제공합니다.

---



This study systematically analyzed the correlation between medical question-answering (QA) benchmarks and real-world clinical performance to evaluate how effectively current benchmarks assess the clinical capabilities of large language models (LLMs). The key findings of the study are as follows:

1. **Competing Models**: A total of 85 LLMs were evaluated in the study, with GPT-4 and GPT-4o demonstrating the highest performance. These models often exceeded the average performance of human physicians, particularly GPT-4, which showed excellent performance across various clinical scenarios.

2. **Test Data**: The research was based on 702 clinical evaluation data collected from 168 studies. This data encompassed a variety of clinical tasks, including diagnosis, clinical knowledge, patient management, and information extraction. The data sources primarily consisted of clinical vignettes, quizzes, and FAQs.

3. **Metrics**: The study employed Spearman's rank correlation coefficient and Kendall's tau to measure the correlation between benchmark performance and clinical performance. MedQA recorded the highest correlation with clinical performance, achieving Spearman's 0.588 and Kendall's 0.520. However, these correlations were notably lower than inter-benchmark correlations (0.675–0.787).

4. **Comparison**: While MedQA effectively evaluated core medical knowledge and reasoning skills, it failed to adequately capture essential competencies required in real-world clinical practice, such as patient communication, longitudinal care, and clinical information extraction. In contrast, MMLU Medical Genetics, College Medicine, Professional Medicine, Clinical Knowledge, and MedMCQA displayed moderately high correlations with clinical performance.

This study provides the first systematic and quantitative analysis of the clinical validity of medical QA benchmarks, offering important insights for the design of future benchmarks.


<br/>
# 예제


이 논문에서는 의료 질문 응답(Question Answering, QA) 벤치마크가 실제 임상 성능을 얼마나 잘 평가하는지를 분석하기 위해 여러 가지 방법론을 사용했습니다. 연구의 주요 목표는 대형 언어 모델(LLM)의 성능이 의료 QA 벤치마크에서 어떻게 나타나는지를 평가하고, 이러한 벤치마크가 실제 임상 환경에서의 성능과 얼마나 일치하는지를 확인하는 것이었습니다.

#### 데이터 수집 및 전처리

1. **문헌 검색**: 연구자들은 2023년 1월 1일부터 2025년 1월 10일까지 발표된 논문을 대상으로 LLM과 의료 응용에 관한 연구를 검색했습니다. 검색 쿼리는 LLM 관련 용어, 의료 용어, 평가 용어를 포함했습니다.

2. **데이터 정제**: 검색된 논문 중에서 LLM 성능을 평가한 연구를 선별하기 위해 제목 기반 필터링, DOI 기반 중복 제거, 초록 및 전체 텍스트 검토를 진행했습니다. 최종적으로 168개의 연구에서 702개의 성능 평가 데이터를 수집했습니다.

3. **성능 데이터 수집**: 각 연구에서 LLM의 성능 점수를 수집하고, 이를 0-100의 스케일로 정규화했습니다. 성능 점수는 주로 다지선다형 질문(MCQ)과 전문가 평가를 통해 수집되었습니다.

#### 분석 방법

1. **상관관계 분석**: 수집된 데이터에서 의료 QA 벤치마크와 실제 임상 성능 간의 상관관계를 분석했습니다. Spearman의 순위 상관계수를 사용하여 두 변수 간의 관계를 평가했습니다.

2. **베이지안 모델링**: 각 LLM의 대표적인 임상 성능을 추정하기 위해 베이지안 계층 모델링을 사용했습니다. 이 모델은 다양한 임상 작업에서 LLM의 성능을 종합적으로 평가할 수 있도록 설계되었습니다.

#### 예시

- **트레이닝 데이터**: 예를 들어, 특정 LLM이 "환자의 증상에 따라 가능한 진단을 나열하라"는 질문을 받았을 때, 모델은 "고혈압, 당뇨병, 심부전"과 같은 답변을 생성할 수 있습니다. 이 데이터는 모델이 훈련되는 동안 사용됩니다.

- **테스트 데이터**: 테스트 데이터는 "환자가 두통과 구토를 호소할 때 가능한 진단은 무엇인가?"와 같은 질문으로 구성됩니다. 모델이 이 질문에 대해 "편두통, 뇌졸중, 위장관 출혈"과 같은 답변을 생성하면, 이 답변이 실제 임상에서의 진단과 얼마나 일치하는지를 평가합니다.

이러한 방식으로 연구자들은 LLM의 성능을 평가하고, 의료 QA 벤치마크가 실제 임상 성능을 얼마나 잘 반영하는지를 분석했습니다.

---



In this paper, the authors systematically analyzed how well medical question-answering (QA) benchmarks evaluate the clinical capabilities of large language models (LLMs). The main goal of the study was to assess the performance of LLMs on medical QA benchmarks and to determine how well these benchmarks align with real-world clinical performance.

#### Data Collection and Preprocessing

1. **Literature Search**: The researchers conducted a literature search for studies published between January 1, 2023, and January 10, 2025, focusing on LLMs and medical applications. The search queries included terms related to LLMs, medical terms, and evaluation terms.

2. **Data Curation**: From the retrieved papers, they filtered studies that evaluated LLM performance. This involved title-based screening, DOI-based deduplication, and abstract and full-text reviews. Ultimately, they collected performance data from 168 studies, resulting in 702 evaluations.

3. **Performance Data Collection**: Performance scores for each LLM were collected and normalized to a 0-100 scale. These scores were primarily obtained through multiple-choice questions (MCQs) and expert human ratings.

#### Analysis Methods

1. **Correlation Analysis**: The study analyzed the correlation between medical QA benchmarks and real-world clinical performance using Spearman's rank correlation coefficient to assess the relationship between the two variables.

2. **Bayesian Modeling**: To estimate representative clinical performance for each LLM, the authors employed Bayesian hierarchical modeling. This model was designed to comprehensively evaluate LLM performance across various clinical tasks.

#### Example

- **Training Data**: For instance, if a specific LLM is asked, "List possible diagnoses based on the patient's symptoms," the model might generate responses like "hypertension, diabetes, heart failure." This data is used during the model's training phase.

- **Test Data**: The test data might consist of questions like, "What are the possible diagnoses when a patient presents with headache and vomiting?" If the model responds with "migraine, stroke, gastrointestinal bleeding," the evaluation would assess how well this response aligns with actual clinical diagnoses.

Through this approach, the researchers evaluated the performance of LLMs and analyzed how well medical QA benchmarks reflect real-world clinical performance.

<br/>
# 요약
이 연구에서는 168개의 연구에서 수집한 702개의 임상 평가 데이터를 기반으로, 의료 QA 벤치마크와 실제 임상 성능 간의 상관관계를 분석하였다. MedQA는 임상 성능과 가장 높은 상관관계를 보였으나, 환자 소통 및 장기 관리와 같은 필수 역량을 충분히 평가하지 못했다. Bayesian 모델링을 통해 GPT-4와 GPT-4o가 인간 의사와 유사하거나 그 이상의 성능을 보임을 확인하였다.

---

This study analyzed the correlation between medical QA benchmarks and real-world clinical performance using 702 clinical evaluation data collected from 168 studies. MedQA showed the highest correlation with clinical performance but failed to adequately assess essential competencies such as patient communication and longitudinal care. Bayesian modeling revealed that GPT-4 and GPT-4o demonstrated performance comparable to or exceeding that of human physicians.

<br/>
# 기타


1. **다이어그램 및 피규어**
   - **Figure 1**: 연구 개요를 보여주며, 의료 QA 벤치마크와 실제 임상 성능 간의 정렬을 평가하는 방법론을 설명합니다. 이는 연구의 목적과 접근 방식을 명확히 합니다.
   - **Figure 3**: MedQA와 다른 벤치마크 간의 상관관계를 보여줍니다. MedQA는 실제 임상 성능과 가장 높은 상관관계를 보였지만, 여전히 한계가 있음을 나타냅니다.
   - **Figure 4**: MedQA의 성능이 다양한 임상 작업 유형에 따라 어떻게 달라지는지를 보여줍니다. 치료, 임상 지식, 진단 작업에서 강한 상관관계를 보였으나, 환자 소통 및 정보 추출에서는 상대적으로 낮은 상관관계를 보였습니다.
   - **Figure 5**: Bayesian 모델링을 통해 추정된 각 언어 모델의 대표적인 임상 성능을 보여줍니다. GPT-4와 GPT-4o가 인간 의사보다 높은 성능을 보였음을 강조합니다.

2. **테이블**
   - **Table 1**: 임상 성능 데이터셋의 요약을 제공합니다. 다양한 작업 유형과 데이터 출처를 포함하여, 연구의 포괄성을 보여줍니다.
   - **Table 2**: 의료 QA 벤치마크 간의 상관계수를 보여줍니다. MedQA와 MedMCQA가 가장 높은 상관관계를 보이며, 이는 이들 벤치마크가 임상 성능을 잘 반영하고 있음을 시사합니다.
   - **Table 11**: 일반 벤치마크와 의료 QA 벤치마크 간의 평균 상관계수를 비교합니다. 이는 의료 QA 벤치마크가 일반 벤치마크와의 관계에서도 유의미한 상관관계를 보임을 나타냅니다.

3. **어펜딕스**
   - 어펜딕스에서는 문헌 검토 방법, LLM 성능 수집 방법, 상관관계 측정 방법 등 연구의 방법론적 세부사항을 제공합니다. 이는 연구의 신뢰성을 높이는 데 기여합니다.




1. **Diagrams and Figures**
   - **Figure 1**: Provides an overview of the study, illustrating the methodology for assessing the alignment between medical QA benchmarks and real-world clinical performance. This clarifies the study's objectives and approach.
   - **Figure 3**: Shows the correlation between MedQA and other benchmarks. While MedQA exhibited the highest correlation with actual clinical performance, it still has limitations, indicating the need for improved evaluation methods.
   - **Figure 4**: Illustrates how MedQA's performance varies across different clinical task types. It shows strong correlations in treatment, clinical knowledge, and diagnosis tasks, but relatively lower correlations in patient communication and information extraction.
   - **Figure 5**: Displays the representative clinical performance of each language model estimated through Bayesian modeling. It emphasizes that models like GPT-4 and GPT-4o often outperform human physicians.

2. **Tables**
   - **Table 1**: Summarizes the clinical performance dataset, showcasing the diversity of task types and data sources, which highlights the comprehensiveness of the research.
   - **Table 2**: Presents the correlation coefficients between medical QA benchmarks. MedQA and MedMCQA show the highest correlations, suggesting that these benchmarks effectively reflect clinical performance.
   - **Table 11**: Compares the average correlation coefficients between general benchmarks and medical QA benchmarks, indicating that medical QA benchmarks maintain significant correlations with general benchmarks.

3. **Appendix**
   - The appendix provides detailed methodological aspects, including literature review methods, LLM performance collection methods, and correlation measurement techniques. This enhances the reliability of the research findings.

<br/>
# refer format:


### BibTeX 형식
```bibtex
@inproceedings{kim2025questioning,
  author    = {Siun Kim and Hyung-Jin Yoon},
  title     = {Questioning Our Questions: How Well Do Medical QA Benchmarks Evaluate Clinical Capabilities of Language Models?},
  booktitle = {Proceedings of the 24th Workshop on Biomedical Language Processing (BioNLP 2025)},
  pages     = {274--296},
  year      = {2025},
  month     = {August},
  publisher = {Association for Computational Linguistics},
  address   = {Seoul, South Korea},
}
```

### 시카고 스타일 인용
Siun Kim and Hyung-Jin Yoon. "Questioning Our Questions: How Well Do Medical QA Benchmarks Evaluate Clinical Capabilities of Language Models?" In *Proceedings of the 24th Workshop on Biomedical Language Processing (BioNLP 2025)*, 274–296. Seoul, South Korea: Association for Computational Linguistics, August 1, 2025.
