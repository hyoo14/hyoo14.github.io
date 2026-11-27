---
layout: post
title:  "[2025]Natural Language Processing in Support of Evidence-Based Medicine: A Scoping Review"
date:   2026-11-27 09:21:04 -0000
categories: study
---

{% highlight ruby %}





한줄 요약:

2019~2024년 발표된 129편의 연구를 분석하여 자연어처리(NLP)가 근거중심의학(EBM)의 5단계에서 의료 근거의 검색, 추출, 평가, 종합, 요약 및 임상 적용을 어떻게 지원하는지 정리한 Scoping Review 논문이다.

---

This scoping review examines 129 studies published between 2019 and 2024 to investigate how NLP supports the five stages of Evidence-Based Medicine (EBM), including evidence retrieval, extraction, appraisal, synthesis, summarization, and clinical application.

짧은 요약(Abstract) :

근거중심의학(Evidence-Based Medicine, EBM)은 최선의 과학적 근거를 활용하여 임상적 의사결정을 지원하는 접근법이다. 그러나 의료 문헌의 양이 빠르게 증가하면서 관련 근거를 검색하고 추출하며 평가하고 종합하는 데 많은 시간과 비용이 소요된다.

본 논문은 2019년부터 2024년까지 발표된 129편의 연구를 대상으로 자연어처리(NLP)가 EBM의 다섯 단계인 Ask, Acquire, Appraise, Apply, Assess에서 어떻게 활용되는지 분석한다. 특히 규칙 기반 방법과 전통적인 머신러닝에서 Transformer 기반 모델 및 LLM으로 발전하면서 문헌 검색, PICO 추출, 근거의 품질 평가, 메타분석, 의료 문서 요약, 임상시험 환자 매칭 등 다양한 작업이 자동화되고 있음을 보여준다.

또한 의료 NLP 관련 벤치마크와 데이터셋을 정리하고, LLM의 환각(Hallucination), 출처 추적의 어려움, 학습 데이터 부족, 모델의 해석 가능성과 신뢰성 등 실제 임상 적용을 위해 해결해야 할 과제를 제시한다.

---

Evidence-Based Medicine (EBM) relies on the best available scientific evidence to support clinical decision-making. However, the rapid growth of biomedical literature makes evidence retrieval, extraction, appraisal, and synthesis increasingly time-consuming and costly.

This paper reviews 129 studies published between 2019 and 2024 to examine how NLP supports the five stages of EBM: Ask, Acquire, Appraise, Apply, and Assess. It highlights the evolution from rule-based and traditional machine learning approaches to Transformer-based models and LLMs, enabling the automation of literature retrieval, PICO extraction, evidence quality assessment, meta-analysis, medical summarization, and clinical trial matching.

The review also summarizes existing benchmarks and datasets while identifying major challenges, including hallucination, insufficient source attribution, data scarcity, interpretability, and reliability in clinical applications.

---

- Useful sentences :
  - "The entire EBM process consists of five steps, commonly referred to as the ‘5A’s: Ask, Acquire, Appraise, Apply, and Assess."
    - EBM은 Ask, Acquire, Appraise, Apply, Assess의 다섯 단계로 구성된다.
  - "Evidence synthesis combines data from included studies to draw conclusions about a body of evidence."
    - 근거 종합은 여러 연구에서 수집된 데이터를 결합하여 전체 근거에 대한 결론을 도출하는 과정이다.
  - "A knowledge graph is a structured representation of information where entities (e.g., symptoms, treatments, drugs) are represented as nodes and their relationships as edges."
    - 지식 그래프는 증상, 치료법, 약물 등의 개체를 노드로, 개체 간 관계를 엣지로 표현하는 구조화된 지식 표현 방법이다.
  - "The existing datasets are often built upon general texts rather than medical-specific content."
    - 기존 데이터셋은 의료 분야에 특화된 자료보다 일반적인 텍스트를 기반으로 구축된 경우가 많다.

{% endhighlight %}

<br/>

[Paper link](https://aclanthology.org/2025.findings-acl.1103/)

~~Lecture link~~

<br/>

# 단어정리

- **EBM (Evidence-Based Medicine)**: 근거중심의학. 최신 연구 근거, 의료진의 전문성, 환자의 가치관을 종합하여 의료적 의사결정을 내리는 접근법.
- **Scoping Review**: 특정 분야의 연구 동향, 주요 방법론, 연구 공백 등을 폭넓게 조사하는 문헌고찰.
- **PICO (Population, Intervention, Comparison, Outcome)**: 임상 연구의 대상, 치료 또는 개입, 비교군, 결과를 구조화하기 위한 프레임워크.
- **Evidence Retrieval**: 임상 질문과 관련된 의료 논문이나 연구 근거를 검색하는 작업.
- **Evidence Extraction**: 의료 문헌에서 환자, 질환, 치료법, 임상 결과 등 필요한 정보를 추출하는 작업.
- **Evidence Appraisal**: 연구의 방법론적 품질, 타당성, 편향 위험 등을 평가하는 과정.
- **Evidence Synthesis**: 여러 연구 결과를 결합하여 종합적인 결론을 도출하는 과정.
- **Evidence Summarization**: 연구 결과와 의료 근거를 핵심 내용 중심으로 요약하는 작업.
- **Meta-analysis**: 여러 연구의 정량적 결과를 통계적으로 결합하여 전체 효과를 추정하는 방법.
- **NER (Named Entity Recognition)**: 의료 문헌에서 질환, 약물, 치료법 등의 개체를 식별하는 작업.
- **Relation Extraction**: 추출된 의료 개체들 사이의 관계를 식별하는 작업.
- **Knowledge Graph**: 개체를 노드, 개체 간 관계를 엣지로 표현하여 정보를 구조화하는 그래프.
- **RAG (Retrieval-Augmented Generation)**: 외부 문서에서 관련 정보를 검색한 후 이를 바탕으로 답변을 생성하는 방식.
- **Hallucination**: 생성형 AI가 실제 사실이나 근거와 일치하지 않는 정보를 생성하는 현상.
- **Clinical Trial Matching**: 환자의 특성과 임상시험 참여 조건을 비교하여 적합한 임상시험을 찾는 작업.
- **Few-shot Learning**: 적은 수의 예제를 활용하여 새로운 작업이나 데이터에 적응하는 학습 방식.
- **PRISMA**: 체계적 문헌고찰과 메타분석의 문헌 검색 및 선정 과정을 투명하게 보고하기 위한 지침.

<br/>

# Methodology

본 논문은 새로운 NLP 모델을 제안하는 연구가 아니라, 기존 연구를 체계적으로 수집하고 분석하는 Scoping Review이다.

따라서 Methodology는 문헌 검색 및 선정 과정과 EBM 단계별 NLP 기술의 분류를 중심으로 구성된다.

**1. Literature Selection**

PRISMA 지침을 참고하여 2019년부터 2024년까지 발표된 연구를 수집하였다.

- 검색 데이터베이스: PubMed, IEEE Xplore, ACM Digital Library, ACL Anthology
- 주요 검색어: NLP, LLM, Information Retrieval, Information Extraction, Evidence-Based Medicine, Clinical Trial 등
- 포함 기준: 영어로 작성된 연구, 인간 대상 의료 연구, NLP를 EBM 관련 작업에 적용한 실험 연구
- 제외 기준: NLP 또는 EBM과 관련성이 낮은 연구, 비영어 논문, 실험 결과가 없는 2차 문헌 등

데이터베이스 검색으로 확보한 601편과 추가 문헌을 대상으로 중복 제거 및 적격성 평가를 진행하여 최종적으로 129편의 연구를 선정하였다.

각 연구에서는 모델, 대상 질환, NLP 작업, 실험 결과, 한계 등의 메타데이터를 수집하였다.

**2. EBM Framework: Five A's**

논문은 EBM의 전체 과정을 다음의 다섯 단계로 구분하고 각 단계에서 활용되는 NLP 기술을 정리한다.

| 단계       | 목적                        | NLP Tasks                                                      |
| -------- | ------------------------- | -------------------------------------------------------------- |
| Ask      | 임상 질문에 필요한 연구 검색          | Information Retrieval, Question Answering                      |
| Acquire  | 의료 문헌에서 필요한 정보 수집         | Entity Extraction, Normalization, Relation Extraction          |
| Appraise | 근거의 품질과 타당성 평가 및 연구 결과 종합 | Quality Assessment, Evidence Ranking, Synthesis, Summarization |
| Apply    | 연구 결과를 실제 의료 환경에 적용       | Clinical Trial Matching, Clinical QA                           |
| Assess   | 적용 결과 평가 및 후속 연구 지원       | Domain-specific Applications, Evidence-based Research          |

**3. NLP Techniques**

각 단계에서 사용되는 NLP 기법의 발전 과정을 분석한다.

- **Rule-based Methods**: 사전에 정의된 규칙과 키워드를 이용하여 정보를 추출하거나 검색한다. 해석하기 쉽지만 복잡하고 모호한 의료 문장을 처리하는 데 한계가 있다.
- **Traditional Machine Learning**: 통계적 특징과 분류 모델을 이용해 문서 분류, 검색, 근거 순위화 등을 수행한다.
- **Transformer-based Models**: BioBERT, SciBERT, PubMedBERT 등을 활용하여 의료 문맥을 이해하고 PICO 추출 및 근거 평가를 수행한다.
- **LLMs**: 복잡한 임상 질문에 답하고, 의료 문헌을 요약하며, 임상시험 참여 조건을 분석하는 데 활용된다.
- **Knowledge Graphs**: 의료 개체와 관계를 구조화하여 연구 결과 간의 연결 관계를 표현하고 새로운 의학적 가설 탐색을 지원한다.

**4. Benchmark Analysis**

EBM 관련 NLP 작업에 활용되는 공개 및 비공개 데이터셋을 비교한다.

Evidence Retrieval, Evidence Extraction, Clinical Trial Identification을 위한 데이터셋에 비해 Evidence Synthesis, Appraisal, Question Answering에 특화된 벤치마크가 부족하다는 점을 확인하였다.

---

This paper conducts a scoping review rather than proposing a new NLP model. Its methodology focuses on systematic literature selection, categorization of NLP techniques, and analysis of their applications across the EBM workflow.

**1. Literature Selection**

The authors followed PRISMA guidelines to review studies published between 2019 and 2024.

- Databases: PubMed, IEEE Xplore, ACM Digital Library, and ACL Anthology.
- Search terms: NLP, LLM, Information Retrieval, Information Extraction, Evidence-Based Medicine, Clinical Trial, and related terms.
- Inclusion criteria: English-language experimental studies applying NLP to human healthcare and EBM-related tasks.
- Exclusion criteria: Non-relevant studies, non-English publications, and secondary literature without experimental results.

After duplicate removal, screening, and eligibility assessment, 129 studies were included in the final review.

Metadata such as NLP models, clinical domains, tasks, findings, and limitations were extracted from the selected studies.

**2. EBM Framework: Five A's**

The selected studies were organized according to the five stages of EBM.

- **Ask:** Identifying relevant studies and answering clinical questions.
- **Acquire:** Extracting structured clinical information, including PICO elements and relationships.
- **Appraise:** Assessing evidence quality, ranking studies, synthesizing findings, and summarizing results.
- **Apply:** Supporting clinical applications such as patient-trial matching and medical question answering.
- **Assess:** Evaluating clinical applications and supporting evidence-based research.

**3. NLP Techniques**

The review examines the evolution of NLP approaches.

- **Rule-based methods:** Transparent and customizable, but limited in handling complex or ambiguous language.
- **Traditional machine learning:** Used for document classification, retrieval, and evidence ranking.
- **Transformer-based models:** BioBERT, SciBERT, and PubMedBERT improve domain-specific contextual understanding and information extraction.
- **LLMs:** Support clinical question answering, evidence summarization, and trial eligibility analysis.
- **Knowledge graphs:** Represent relationships among clinical entities and support evidence organization and knowledge discovery.

**4. Benchmark Analysis**

The authors examine datasets used for EBM-related NLP tasks and identify gaps in benchmark availability, particularly for evidence synthesis, appraisal, and clinical question answering.

---

<br/>

# Results

**1. Research Trends**

2019년부터 2024년까지의 연구에서는 Entity Extraction, Classification, Evaluation이 주요 NLP 작업으로 나타났다.

연구 수는 2023년에 가장 많았으며, 최근에는 Question Answering과 Quality Assessment 관련 연구가 더욱 중요해지고 있다.

이는 의료 NLP 연구가 단순한 정보 검색과 추출을 넘어 근거 평가와 복잡한 의료 질의응답으로 확장되고 있음을 보여준다.

**2. Evidence Retrieval and Extraction**

의료 문헌 검색에서는 기존의 키워드 및 규칙 기반 방법에서 Transformer 기반 의미 검색으로 발전하는 경향이 나타났다.

PICO Extraction은 비정형 의료 논문에서 환자 집단, 치료법, 비교군, 임상 결과를 구조화하는 핵심 작업으로 활용된다.

또한 Relation Extraction과 Knowledge Graph를 통해 질환, 약물, 치료법, 결과 사이의 관계를 구조화할 수 있다.

**3. Evidence Appraisal and Synthesis**

NLP는 관련성이 높은 연구의 우선순위를 결정하고 근거의 품질과 편향 위험을 평가하는 데 활용된다.

Evidence Synthesis에서는 여러 연구의 결과를 통합하여 효과 크기를 계산하거나 메타분석을 자동화하는 접근법이 제안되었다.

Evidence Summarization에서는 TextRank 등의 추출적 요약 방법부터 T5, BART, PEGASUS, LLM 기반 생성적 요약 방법까지 다양한 기술이 활용되었다.

그러나 일부 자동 메타분석 방법은 이진 결과에만 적용되거나 제한된 데이터에서 평가되었으며, 생성적 요약은 사실성과 출처 추적의 문제가 남아 있다.

**4. Clinical Applications**

실제 의료 환경에서의 NLP 활용 가능성도 확인되었다.

- Oncology: 호주 폐암 환자를 대상으로 한 임상시험 매칭 시스템에서 정확도 92%가 보고되었다.
- Cardiology: 약 40,000명의 환자 데이터를 활용한 임상 적격성 평가 시스템이 정확도 87.3%를 달성하였다.
- Clinical Trial Matching: TrialGPT와 같은 시스템은 환자 정보와 임상시험 참여 조건을 분석하여 적합한 임상시험을 찾는 데 활용된다.
- Drug Repurposing: 의료 문헌과 지식 그래프를 활용하여 기존 약물의 새로운 치료 가능성을 탐색한다.

이 수치들은 리뷰 논문에서 소개한 개별 연구의 결과로, 동일한 실험 환경에서 직접 비교된 성능은 아니다.

**5. Limitations and Challenges**

- LLM의 Hallucination으로 인한 잘못된 의료 정보 생성 가능성
- 생성된 답변의 근거 출처를 추적하기 어려운 문제
- Evidence Appraisal 및 Synthesis를 위한 의료 특화 데이터셋 부족
- 모델의 낮은 해석 가능성과 데이터 편향
- 높은 계산 비용과 의료 도메인 전문성 요구
- 복잡한 임상 상황과 다양한 환자 집단에 대한 일반화의 어려움

---

**1. Research Trends**

Entity Extraction, Classification, and Evaluation were among the most frequently studied NLP tasks from 2019 to 2024.

The number of publications peaked in 2023, while Question Answering and Quality Assessment became increasingly prominent.

These trends suggest a shift from basic information processing toward more complex evidence evaluation and clinical reasoning tasks.

**2. Evidence Retrieval and Extraction**

NLP-based retrieval has evolved from keyword-based and rule-based approaches toward Transformer-based semantic search.

PICO extraction plays a central role in converting unstructured medical literature into structured evidence.

Relation extraction and knowledge graphs further capture relationships among diseases, treatments, drugs, and clinical outcomes.

**3. Evidence Appraisal and Synthesis**

NLP supports evidence quality assessment, study prioritization, automated meta-analysis, and medical summarization.

Approaches include traditional ranking methods, BERT-based classifiers, extractive summarization algorithms, and generative models such as T5, BART, PEGASUS, and LLMs.

However, automated synthesis methods may be limited by small datasets or restricted outcome types, while generative summarization introduces concerns about factual consistency and source attribution.

**4. Clinical Applications**

The review identifies promising applications in real-world healthcare.

- Oncology: A clinical trial matching system reported 92% accuracy in screening lung cancer patients.
- Cardiology: A system evaluated using approximately 40,000 patients achieved 87.3% accuracy.
- Clinical Trial Matching: Systems such as TrialGPT support matching patient profiles with clinical trial eligibility criteria.
- Drug Repurposing: NLP and knowledge graphs help identify potential new therapeutic applications for existing drugs.

These results originate from individual studies and are not directly comparable.

**5. Limitations and Challenges**

Major challenges include hallucination, insufficient source attribution, limited medical benchmarks, interpretability, data bias, computational costs, and generalization across complex clinical settings.

---

<br/>

# 예제

**1. PICO Extraction**

의료 논문에서 핵심 정보를 추출하는 예시이다. 논문에서 설명하는 작업을 바탕으로 구성한 가상의 사례이다.

Input:

"A randomized controlled trial evaluated Drug A in adults with type 2 diabetes. Compared with placebo, Drug A reduced HbA1c levels after 12 weeks."

Output:

```
{
  "Population": "Adults with type 2 diabetes",
  "Intervention": "Drug A",
  "Comparison": "Placebo",
  "Outcome": "Reduction in HbA1c after 12 weeks"
}
```

비정형 의료 문헌에서 PICO 요소를 추출하면 연구 간 비교, 의료 정보 검색, 근거 종합 등에 사용할 수 있는 구조화된 데이터가 생성된다.

**2. Evidence-based Question Answering**

임상 질문:

"Does Drug A improve glycemic control in patients with type 2 diabetes?"

가능한 처리 과정:

1. Information Retrieval: 관련 임상시험 및 연구 검색
2. PICO Extraction: 환자, 치료법, 비교군, 결과 추출
3. Evidence Appraisal: 연구의 품질과 편향 위험 평가
4. Evidence Synthesis: 여러 연구의 결과 종합
5. Answer Generation: 근거와 출처를 포함한 답변 생성

이는 LLM이 사전학습된 지식에만 의존하지 않고 실제 의료 근거를 바탕으로 답변하도록 설계하는 접근법이다.

---

**1. PICO Extraction**

Consider the following hypothetical clinical trial description.

Input:

"A randomized controlled trial evaluated Drug A in adults with type 2 diabetes. Compared with placebo, Drug A reduced HbA1c levels after 12 weeks."

The NLP system identifies:

- Population: Adults with type 2 diabetes
- Intervention: Drug A
- Comparison: Placebo
- Outcome: Reduction in HbA1c after 12 weeks

The resulting structured evidence can support literature retrieval, study comparison, and automated synthesis.

**2. Evidence-based Question Answering**

For a clinical question such as "Does Drug A improve glycemic control in patients with type 2 diabetes?", an evidence-based NLP pipeline may:

1. Retrieve relevant clinical studies.
2. Extract PICO elements.
3. Assess study quality and risk of bias.
4. Synthesize findings across studies.
5. Generate an evidence-grounded answer with source references.

This illustrates how retrieval, extraction, appraisal, and generation can be integrated into clinical question answering.

---

<br/>

# 요약

본 논문은 NLP가 EBM의 전 과정에서 의료 근거를 처리하고 활용하는 데 중요한 역할을 수행할 수 있음을 보여준다.

- NLP는 의료 문헌 검색과 정보 추출을 넘어 근거의 품질 평가, 종합, 요약, 임상 적용까지 지원할 수 있다.
- Transformer 기반 모델과 LLM은 복잡한 의료 문헌의 이해와 정보 처리 능력을 향상시키고 있다.
- 임상시험 환자 매칭, 의료 질의응답, 약물 재창출 등 실제 의료 환경에서의 활용 가능성이 확인되었다.
- 그러나 의료 분야에서는 모델의 정확성뿐 아니라 근거의 신뢰성, 출처 추적 가능성, 해석 가능성, 편향 관리가 필수적이다.
- 향후에는 개별 NLP 작업을 연결하고 다양한 의료 데이터를 통합할 수 있는 신뢰성 높은 시스템이 필요하다.

결국 의료 NLP의 핵심 목표는 단순히 자연스러운 답변을 생성하는 것이 아니라, **검증 가능한 의학적 근거를 바탕으로 신뢰할 수 있는 임상적 의사결정을 지원하는 것**이다.

---

This review demonstrates that NLP can support the entire EBM workflow, from evidence retrieval and information extraction to appraisal, synthesis, summarization, and clinical application.

- Transformer-based models and LLMs have expanded the capabilities of biomedical NLP.
- NLP applications have progressed toward complex tasks such as evidence-based question answering, clinical trial matching, and medical summarization.
- Clinical reliability requires more than predictive accuracy; source attribution, interpretability, fairness, and validation remain essential.
- Future research should focus on integrating multiple NLP tasks into reliable, evidence-grounded clinical workflows.

The ultimate goal of NLP for EBM is not simply to generate fluent answers, but to support trustworthy clinical decisions using verifiable scientific evidence.

---

<br/>

# 기타(다이어그램, Figures, Tables, Appendix, 인사이트)

**1. Figure 1 — PRISMA Flow Diagram (p.2)**

문헌 검색, 중복 제거, 초기 선별, 전문 검토를 거쳐 최종적으로 129편의 연구가 선정되는 과정을 보여준다.

연구 선정 과정의 투명성을 이해하는 데 중요한 그림이다.

**2. Figure 2 — Distribution of Papers in Different EBM Tasks over Time (p.3)**

2019년부터 2024년까지 EBM 관련 NLP 연구의 작업별 분포를 보여준다.

Entity Extraction, Classification, Evaluation이 주요 연구 분야로 나타났으며, Question Answering과 Quality Assessment가 최근 더욱 주목받고 있음을 확인할 수 있다.

**3. Table 1 — Mapping of EBM Cycle to NLP Tasks (p.3)**

EBM의 다섯 단계와 각 단계에 대응하는 NLP 작업을 정리한 표이다.

논문의 전체적인 분석 프레임워크를 가장 잘 보여주는 표로, NLP가 의료 근거의 검색부터 평가, 적용까지 다양한 단계에서 활용된다는 점을 설명한다.

**4. Supplementary Table 2 — Benchmark Datasets**

EBM 관련 NLP 데이터셋의 작업, 규모, 어노테이션 정보 등을 정리한다.

대표적인 데이터셋은 다음과 같다.

- EBM-NLP: 임상 연구 초록의 PICO 요소 추출
- MS²: 여러 의료 연구 문서를 종합하는 Multi-document Summarization
- PICO-Corpus: 임상시험 문헌의 PICO 정보 추출
- NICTA-PIBOSO: 의료 초록의 문장별 역할 분류

**5. Insights**

- **Evidence-grounded Generation:** 의료 LLM이 생성한 답변을 원문 근거와 명확하게 연결하여 사실성과 출처 추적 가능성을 높이는 방향이 중요하다.
- **End-to-End EBM Pipeline:** Retrieval → Extraction → Appraisal → Synthesis → Generation을 연결하면 개별 NLP 작업을 넘어 실제 임상 의사결정을 지원하는 시스템을 구성할 수 있다.
- **Quality-aware RAG:** 문서의 의미적 관련성뿐 아니라 연구의 품질과 편향 위험을 고려하여 검색 근거를 선택하는 접근법이 유용할 수 있다.
- **Evaluation Beyond Accuracy:** 의료 AI는 일반적인 정확도나 텍스트 유사도 외에도 근거 충실도, 출처 정확성, 임상적 타당성을 평가해야 한다.

위 인사이트는 논문의 기술 동향과 한계를 바탕으로 도출한 후속 연구 방향이며, 본 논문에서 직접 실험으로 검증된 결과는 아니다.

---

**1. Figure 1 — PRISMA Flow Diagram (p.2)**

Illustrates the literature identification, screening, eligibility assessment, and final selection of 129 studies.

**2. Figure 2 — Distribution of Papers in Different EBM Tasks over Time (p.3)**

Shows the distribution of NLP research tasks between 2019 and 2024, highlighting the importance of entity extraction, classification, and evaluation, along with growing interest in question answering and quality assessment.

**3. Table 1 — Mapping of EBM Cycle to NLP Tasks (p.3)**

Maps the five stages of EBM to corresponding NLP tasks and provides the main conceptual framework of the review.

**4. Supplementary Table 2 — Benchmark Datasets**

Summarizes benchmark datasets used for evidence retrieval, extraction, summarization, and clinical trial-related tasks, including EBM-NLP, MS², PICO-Corpus, and NICTA-PIBOSO.

**5. Insights**

- **Evidence-grounded Generation:** Linking generated clinical claims to supporting evidence may improve reliability and traceability.
- **End-to-End EBM Pipeline:** Integrating retrieval, extraction, appraisal, synthesis, and generation may enable more comprehensive clinical decision-support systems.
- **Quality-aware RAG:** Evidence retrieval could incorporate study quality and risk of bias rather than relying only on semantic relevance.
- **Evaluation Beyond Accuracy:** Clinical NLP systems should be evaluated using evidence faithfulness, citation accuracy, and clinical validity.

These insights represent potential research directions derived from the review rather than experimentally validated findings.

---

<br/>

# refer format:

### BibTeX

```
@inproceedings{xu-etal-2025-natural,
    title = "Natural Language Processing in Support of Evidence-based Medicine: A Scoping Review",
    author = "Xu, Zihan  and
      Ma, Haotian  and
      Ding, Yihao  and
      Zhang, Gongbo  and
      Weng, Chunhua  and
      Peng, Yifan",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1103/",
    doi = "10.18653/v1/2025.findings-acl.1103",
    pages = "21421--21443",
    ISBN = "979-8-89176-256-5",
    abstract = "Evidence-based medicine (EBM) is at the forefront of modern healthcare, emphasizing the use of the best available scientific evidence to guide clinical decisions. Due to the sheer volume and rapid growth of medical literature and the high cost of curation, there is a critical need to investigate Natural Language Processing (NLP) methods to identify, appraise, synthesize, summarize, and disseminate evidence in EBM. This survey presents an in-depth review of 129 research studies on leveraging NLP for EBM, illustrating its pivotal role in enhancing clinical decision-making processes. The paper systematically explores how NLP supports the five fundamental steps of EBM{---}Ask, Acquire, Appraise, Apply, and Assess. The review not only identifies current limitations within the field but also proposes directions for future research, emphasizing the potential for NLP to revolutionize EBM by refining evidence extraction, evidence synthesis, appraisal, summarization, enhancing data comprehensibility, and facilitating a more efficient clinical workflow."
}
```

### Chicago style

Xu, Zihan, Haotian Ma, Gongbo Zhang, Yihao Ding, Chunhua Weng, and Yifan Peng. 2025. “Natural Language Processing in Support of Evidence-Based Medicine: A Scoping Review.” In *Findings of the Association for Computational Linguistics: ACL 2025*, 21421–21443. Association for Computational Linguistics. 

영문 중복을 줄이고 한글 중심으로 정리해줘모든 항목을 원래 빈 포맷에 맞춰 채워줘