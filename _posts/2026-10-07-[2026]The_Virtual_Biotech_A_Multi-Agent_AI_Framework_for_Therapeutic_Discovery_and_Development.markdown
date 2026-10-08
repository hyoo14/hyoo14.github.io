---
layout: post
title:  "[2026]The Virtual Biotech: A Multi-Agent AI Framework for Therapeutic Discovery and Development"
date:   2026-10-07 18:12:44 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

이 시스템은 신약개발 조직처럼 여러 전문 AI 에이전트가 표적 발굴, 안전성 평가, 치료 방식 선택, 임상 개발을 나누어 분석하고, 가상의 최고과학책임자(CSO)가 그 결과를 종합  


짧은 요약(Abstract) :



신약 개발은 여러 생물학적·임상 데이터를 함께 분석해야 하지만, 관련 정보와 전문성이 서로 다른 분야에 흩어져 있어 통합과 의사결정이 어렵습니다. 이 논문은 이러한 문제를 해결하기 위해 **Virtual Biotech**이라는 다중 AI 에이전트 시스템을 개발했습니다. 이 시스템은 신약개발 조직처럼 여러 전문 AI 에이전트가 표적 발굴, 안전성 평가, 치료 방식 선택, 임상 개발을 나누어 분석하고, 가상의 최고과학책임자(CSO)가 그 결과를 종합합니다.

저자들은 이 시스템을 세 가지 사례에 적용했습니다.

1. **대규모 임상시험 분석**  
   37,000개 이상의 AI 에이전트가 약 56,000건의 임상시험 결과를 분석했습니다. 그 결과, 특정 세포 유형에 제한적으로 발현되는 유전자를 표적으로 하는 약물은 시장에 도달할 가능성이 **48% 높았고**, 이상반응 발생률은 평균 **32% 낮았습니다**.

2. **폐암에서 B7-H3 표적 평가**  
   유전학, 단일세포·공간 전사체, 생존 분석 등을 통합해 B7-H3가 특히 종양 미세환경의 섬유아세포에서 높게 발현되고 면역세포 배제를 유도할 가능성을 제시했습니다. 이를 바탕으로 B7-H3에는 항체-약물 접합체(ADC)가 적합한 치료 방식일 수 있다고 제안했습니다.

3. **궤양성 대장염 임상시험 실패 분석**  
   중단된 OSMR 표적 치료시험을 분석한 결과, OSMR 자체보다 여러 수용체가 공유하는 **gp130–JAK–STAT 신호축의 중복성**이 치료 실패에 더 중요한 원인일 수 있음을 제시했습니다. 또한 OSMR 단독보다 gp130 신호축 전체를 반영하는 복합 바이오마커가 치료 반응 예측에 더 유용할 가능성을 보였습니다.

즉, 이 연구는 여러 전문 AI가 인간의 지도를 받으며 다양한 생물의학 데이터를 통합할 경우, 표적 우선순위 결정과 치료 방식 선택, 임상 실패 원인 분석을 더 빠르고 체계적으로 수행할 수 있음을 보여줍니다. 다만 이 시스템은 실험을 직접 수행하는 것이 아니라 **가설과 의사결정을 지원하는 도구**이므로, 제시된 결과는 추가적인 실험 및 임상 검증이 필요합니다.

---




Drug discovery requires the integration of diverse biological and clinical evidence, but relevant expertise and data are often fragmented across disciplines. This study introduces the **Virtual Biotech**, a multi-agent AI platform designed to mimic a therapeutic research organization. Specialized AI agents analyze target biology, safety, modality selection, and clinical development, while a virtual Chief Scientific Officer coordinates the analyses and integrates the results.

The system was evaluated in three drug-development scenarios:

1. **Large-scale clinical trial analysis:**  
   More than 37,000 agents annotated approximately 56,000 clinical trials. Drugs targeting genes with cell-type-specific expression were 48% more likely to reach the market and were associated with 32% fewer adverse events.

2. **B7-H3 in lung cancer:**  
   By integrating genetic, single-cell, spatial transcriptomic, and survival data, the system found that B7-H3 was particularly elevated in tumor-associated fibroblasts and associated with an immune-excluded tumor microenvironment. It therefore proposed an antibody–drug conjugate as a potentially suitable therapeutic modality.

3. **Failure analysis of an ulcerative colitis trial:**  
   Analysis of a terminated OSMR-targeted trial suggested that signaling redundancy within the gp130–JAK–STAT pathway, rather than OSMR alone, may have contributed to the lack of efficacy. A composite gp130-axis biomarker outperformed OSMR expression alone in predicting treatment non-response.

Overall, the study shows that human-guided multi-agent AI systems can integrate large-scale, multimodal evidence to support target prioritization, therapeutic design, and clinical translation analysis. However, the system is a decision-support tool rather than a replacement for laboratory or clinical validation, and its hypotheses require further testing.


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



### 1. 전체 시스템 구조: 다중 에이전트 AI

이 연구는 **Virtual Biotech**이라는 다중 에이전트 AI 시스템을 개발했다. 하나의 AI가 모든 작업을 수행하는 대신, 실제 바이오텍 연구조직처럼 여러 전문 AI 에이전트가 역할을 나누어 협력한다.

구성은 다음과 같다.

- **가상 CSO(Chief Scientific Officer)**  
  사용자의 질문을 해석하고, 분석 과제를 여러 하위 과제로 분해한다. 각 과제를 적절한 전문 에이전트에게 배정하고, 결과를 통합해 최종 결론을 만든다.
- **전문 과학자 에이전트 8개**
  - 통계유전학
  - 기능유전체학·유전자 섭동
  - 단일세포 아틀라스
  - 생물학적 경로·단백질 상호작용
  - FDA 안전성
  - 표적 생물학·구조
  - 약리학·모달리티 선택
  - 임상시험 분석
- **CSO 지원 에이전트**
  - Chief of Staff: 분야 개요와 관련 데이터·최근 연구를 조사
  - Scientific Reviewer: 분석의 타당성, 근거의 강도, 누락된 부분을 검토

CSO는 직접 데이터를 분석하기보다, 전문 에이전트의 결과를 조정하고 종합하는 역할을 담당한다.

---

### 2. 사용한 AI 모델과 실행 환경

모든 에이전트는 **Claude Agent SDK**를 기반으로 구현되었다.

- 과학자 에이전트: **Claude Sonnet 4.5**
- Chief of Staff 및 Scientific Reviewer: **Claude Haiku 4.5**
- 각 에이전트에는 역할별 시스템 프롬프트와 접근 가능한 도구가 지정됨
- 코드 실행, 파일 처리, 데이터 분석이 가능함
- 에이전트가 분석 과정에서 가정을 점검하고, 독립적인 데이터 소스를 비교하도록 설계됨

이 논문에서는 새로운 언어모델을 처음부터 학습한 것이 아니라, 기존 Claude 모델에 전문 역할, 도구, 분석 절차를 결합했다.

---

### 3. MCP 기반 데이터·도구 연결

다양한 바이오의학 데이터베이스를 에이전트가 동일한 방식으로 사용할 수 있도록 **Model Context Protocol(MCP)** 서버를 구축했다.

MCP는 언어모델과 외부 데이터·분석 도구를 연결하는 표준 인터페이스다. 각 도구는 입력값과 출력 형식이 구조화되어 있어, 에이전트가 유전자, 질병, 약물, 임상시험 등을 조회하고 분석 파이프라인을 연결할 수 있다.

주요 데이터 자원은 다음과 같다.

- Open Targets
- CELLxGENE Census
- Tabula Sapiens
- Tahoe-100M 약물 섭동 데이터
- ClinicalTrials.gov
- PubMed
- cBioPortal 및 TCGA
- FDA/OpenFDA 안전성 자료
- ChEMBL 계열 약물·화학 데이터
- 단일세포 및 공간전사체 데이터

플랫폼은 100개 이상의 분석 도구와 약 10개의 MCP 서버를 제공하며, 유전자·질병·약물·임상시험·단일세포 데이터 등을 연결한다.

---

### 4. 인간과 AI의 협업 방식

사용자가 질문을 입력하면 시스템은 다음 순서로 작동한다.

1. CSO가 질문의 목적과 범위를 확인하기 위해 추가 질문을 함
2. Chief of Staff가 관련 분야와 데이터 가용성을 요약함
3. CSO가 질문을 전문 분석 과제로 분해함
4. 각 과학자 에이전트가 필요한 데이터와 코드를 사용해 분석함
5. Scientific Reviewer가 결과의 근거와 누락을 검토함
6. 문제가 있으면 CSO가 추가 분석을 지시함
7. 최종적으로 여러 분석 결과를 통합해 보고서와 추천안을 작성함

사람은 주로 분석 목표를 정하고, CSO의 확인 질문에 답하고, 후속 분석을 요청하는 역할을 했다. 실제 코드 작성과 데이터 분석은 대부분 에이전트가 수행했다.

---

### 5. 대규모 임상시험 결과의 병렬 큐레이션

첫 번째 분석에서는 Open Targets의 임상시험 자료를 보강하기 위해 **37,075개의 임상시험별 에이전트**를 병렬로 실행했다.

각 에이전트는 하나의 NCT ID를 담당하고 다음의 3단계 근거 체계를 사용했다.

1. ClinicalTrials.gov 조회
2. 결과가 부족하면 PubMed 논문 검색
3. 그래도 부족하면 보도자료와 규제기관 발표 확인

각 시험에 대해 다음 정보를 구조화된 JSON 형식으로 기록했다.

- 임상 단계 진행 여부
- 1차·2차 평가변수 결과
- 이상반응 비율
- 시험 중단 사유
- 사용한 정보 출처

37,075개 임상시험을 약 6시간에 처리했으며, 논문에서는 단일 에이전트 방식보다 약 184배 빠른 병렬화 효과가 있었다고 보고했다.

---

### 6. 단일세포 기반 표적 특징 추출

약물 표적 유전자의 세포 수준 특성을 평가하기 위해 Tabula Sapiens의 27개 인체 조직 데이터를 사용했다.

주요 특징은 두 가지였다.

#### ① 세포 유형 특이성: Tau index

유전자가 특정 세포 유형에 집중적으로 발현되는 정도를 측정했다.

- 값이 0에 가까움: 여러 세포 유형에서 널리 발현
- 값이 1에 가까움: 특정 세포 유형에 제한적으로 발현

각 조직에서 Tau 값을 계산한 뒤, 유전자가 발현되는 조직들의 평균을 구해 전역 점수로 사용했다.

#### ② 발현 이중성: Bimodality coefficient

같은 세포 집단 안에서 유전자가 모든 세포에서 비슷하게 발현되는지, 아니면 일부 세포에서만 “켜짐/꺼짐” 형태로 나타나는지를 측정했다.

이 두 특징과 임상시험 성공 여부를 연결해 다음을 분석했다.

- 임상 단계 진행
- 1차·2차 평가변수 성공
- Phase I에서 Phase II로의 진행
- 이상반응 비율

분석에는 로지스틱 회귀, 베타 회귀, 혼합효과모델, permutation test, 유전적 근거를 보정한 회귀모델을 사용했다.

---

### 7. B7-H3 폐암 사례 분석

B7-H3/CD276 표적의 치료 가능성과 적절한 약물 형태를 평가하기 위해 여러 종류의 근거를 통합했다.

분석 절차는 다음과 같다.

1. **통계유전학 분석**  
   폐암에서 B7-H3를 지지하는 GWAS 또는 credible set이 있는지 조사
2. **단일세포 차등발현 분석**  
   SCLC, LUAD 및 정상 폐 데이터를 사용해 세포 유형별 발현 비교
3. **세포 간 상호작용 분석**  
   LIANA를 이용해 B7-H3 고발현 섬유아세포와 면역세포 간 ligand–receptor 신호 분석
4. **공간전사체 분석**  
   Visium 데이터와 Cell2Location을 사용해 B7-H3 고발현 영역 주변의 면역세포 분포 분석
5. **생존 분석**  
   TCGA LUAD 환자 477명을 대상으로 Cox 비례위험모델을 사용해 B7-H3 발현과 생존의 관계 분석
6. **모달리티 선택**  
   단백질 위치, 세포 표면 발현, 화학적 결합 가능성, 기존 약물 정보를 종합해 ADC를 후보로 선정

공간전사체 분석에서는 B7-H3 고발현 부위 주변의 T세포, 대식세포, 단핵구, 수지상세포가 감소하는지를 혼합효과모델로 평가했다.

---

### 8. OSMR 궤양성 대장염 임상 실패 분석

중단된 Phase II MOONGLOW 시험을 대상으로, 단순한 표적 발현 분석을 넘어 치료 실패의 기전을 추론했다.

주요 방법은 다음과 같다.

- 임상시험의 대상 환자와 선정 기준 검토
- TAURUS 단일세포 데이터에서 치료 비반응자와 반응자 비교
- 전사인자 활성 추정
- OSMR 하위 신호인 JAK–STAT 경로 분석
- gp130 계열 수용체들의 상대적 기여도 비교
- Shapley value 기반 LMG 분산분해
- OSMR 단독 점수와 gp130-axis 복합 바이오마커 비교
- 4개의 독립적인 infliximab 치료 UC 코호트에서 검증
- 112개 질병에서 OSMR 발현의 공통 염증 프로그램 여부 조사

핵심적으로, OSMR이 발현되더라도 STAT1 활성은 OSMR 하나가 아니라 IL-6 계열 수용체들이 공유하는 **gp130 신호축**에 의해 유지될 수 있다고 추론했다. 따라서 OSMR 단독보다 여러 수용체·리간드·STAT1을 포함한 10개 유전자 기반 gp130-axis 점수가 치료 비반응을 더 잘 설명하는지 평가했다.

---

### 9. 재현성과 전문가 검증

두 사례 연구의 8개 세부 분석에 대해 인간 전문가가 같은 지시를 받아 독립적으로 코드를 작성하고 분석했다.

- 8개 분석 모두 전문가의 결론과 Virtual Biotech의 결론이 일치
- 에이전트가 작성한 코드도 중대한 오류가 없는 것으로 평가
- 임상시험 주석은 무작위 100건과 TDC 데이터셋을 이용해 수작업 검증
- 주요 평가변수와 이상반응 주석의 전문가 일치율은 약 88~92%

즉, 시스템의 결과를 단순히 AI의 판단으로 받아들이지 않고, 생성된 코드·데이터·근거를 사람이 확인할 수 있도록 설계했다.

---

### 핵심 요약

이 연구의 메서드는 **기존 언어모델 + 전문화된 다중 에이전트 + MCP 데이터 도구 + 코드 실행 + 통계·오믹스 분석 + 인간 검토**를 결합한 것이다. 모델 자체를 새로 학습하기보다는, 가상 CSO가 여러 전문 에이전트를 조정하고 각 에이전트가 서로 다른 바이오의학 데이터와 분석 방법을 사용하도록 구성한 점이 핵심이다.

---




### 1. Overall architecture: a multi-agent AI system

The study developed the **Virtual Biotech**, a multi-agent AI platform designed to mimic a biotechnology research organization. Instead of relying on one general-purpose agent, the system distributes tasks across specialized scientist agents.

The main components are:

- **Virtual Chief Scientific Officer (CSO)**  
  Interprets the user’s question, decomposes it into subtasks, assigns tasks to specialist agents, and integrates their results.
- **Eight specialist scientist agents**
  - Statistical genetics
  - Functional genomics and perturbation
  - Single-cell atlas analysis
  - Pathways and protein–protein interactions
  - FDA safety
  - Target biology and structure
  - Pharmacology and modality selection
  - Clinical trial analysis
- **CSO-office agents**
  - Chief of Staff: prepares field and data-landscape briefings
  - Scientific Reviewer: checks methodological quality, evidence strength, and unsupported claims

The CSO mainly orchestrates and synthesizes the work rather than directly performing data analysis.

---

### 2. Models and execution environment

The agents were implemented using the **Claude Agent SDK**.

- Scientist agents: **Claude Sonnet 4.5**
- Chief of Staff and Scientific Reviewer: **Claude Haiku 4.5**
- Each agent received role-specific system prompts and access to domain-specific tools.
- Agents could execute code, manipulate files, query databases, and perform statistical analyses.
- They were instructed to question assumptions, compare complementary datasets, and assess evidence strength.

The study did not train a new language model from scratch. Instead, it combined existing Claude models with specialized prompts, tools, workflows, and biomedical data access.

---

### 3. MCP-based data and tool integration

The platform used **Model Context Protocol (MCP)** servers to provide a standardized interface between language models and biomedical databases or analytical tools.

MCP tools accepted structured inputs such as gene symbols, disease identifiers, drug names, and trial IDs, and returned structured outputs that could be chained across multiple analyses.

Major data sources included:

- Open Targets
- CELLxGENE Census
- Tabula Sapiens
- Tahoe-100M perturbation data
- ClinicalTrials.gov
- PubMed
- cBioPortal and TCGA
- FDA/OpenFDA safety data
- ChEMBL-related drug and chemical resources
- Single-cell and spatial transcriptomics datasets

The system contained more than 100 analytical tools distributed across approximately 10 MCP servers.

---

### 4. Human–AI interaction and workflow

The general workflow was:

1. The user submitted a scientific question.
2. The CSO asked clarification questions about scope and intent.
3. The Chief of Staff prepared a background and data-availability briefing.
4. The CSO decomposed the question into specialized analytical tasks.
5. Scientist agents queried databases and executed analyses.
6. The Scientific Reviewer evaluated the results and identified gaps.
7. The CSO requested additional analyses when necessary.
8. The CSO integrated the results into a final report and recommendation.

Human involvement was mainly limited to defining the objective, answering clarification questions, and requesting follow-up analyses. Most coding and computational analysis were agent-driven.

---

### 5. Large-scale clinical-trial curation

To enrich the Open Targets clinical-trial dataset, the system launched **37,075 trial-specific agents in parallel**, assigning one agent to each NCT identifier.

Each agent followed a three-tier evidence cascade:

1. ClinicalTrials.gov
2. PubMed publications
3. Press releases and regulatory announcements

The agents extracted and standardized:

- Phase progression
- Primary and secondary endpoint results
- Adverse-event rates
- Trial termination reasons
- Source provenance

The results were stored in a structured JSON ontology. The study reported that 37,075 trials were processed in approximately six hours, corresponding to an estimated 184-fold speedup compared with sequential single-agent processing.

---

### 6. Single-cell feature extraction for drug targets

The authors used the Tabula Sapiens atlas across 27 human tissues to derive cell-level features for drug-target genes.

Two major features were calculated:

#### Cell-type specificity: Tau index

The Tau index quantified whether a gene was broadly expressed or restricted to a specific cell type.

- Near 0: broadly expressed across cell types
- Near 1: highly cell-type-specific

Per-tissue Tau values were averaged across tissues in which the gene was expressed.

#### Expression heterogeneity: bimodality coefficient

The bimodality coefficient measured whether expression within a cell population was relatively uniform or concentrated in distinct “on/off” subpopulations.

These features were associated with:

- Clinical phase progression
- Primary and secondary endpoint success
- Phase I-to-Phase II progression
- Adverse-event rates

The analyses used logistic regression, beta regression, mixed-effects models, permutation testing, and models adjusted for genetic evidence.

---

### 7. B7-H3 lung-cancer case study

The system evaluated B7-H3/CD276 as a therapeutic target in LUAD and SCLC by integrating several evidence types:

1. **Statistical genetics**  
   Assessment of GWAS associations and credible sets.
2. **Single-cell differential expression**  
   Cell-type-specific comparison of B7-H3 expression in SCLC, LUAD, and healthy lung.
3. **Cell–cell communication analysis**  
   LIANA-based ligand–receptor analysis comparing B7-H3-high and B7-H3-low fibroblasts.
4. **Spatial transcriptomics**  
   Visium data were analyzed with Cell2Location to infer cell-type abundance and local immune-cell depletion around B7-H3-high spots.
5. **Survival analysis**  
   Cox proportional-hazards models were applied to 477 TCGA LUAD patients, adjusting for age, stage, and sex.
6. **Modality selection**  
   Protein localization, cell-surface expression, chemical tractability, and clinical precedent were integrated to prioritize an antibody–drug conjugate.

Spatial immune-neighborhood analysis used mixed-effects models to test whether B7-H3-high regions were associated with reduced local abundance of T cells, macrophages, monocytes, and dendritic cells.

---

### 8. OSMR ulcerative-colitis trial-failure analysis

The system retrospectively analyzed the terminated Phase II MOONGLOW trial of vixarelimab against OSMRβ.

The workflow included:

- Reviewing trial eligibility criteria and patient population
- Comparing treatment responders and non-responders in the TAURUS single-cell dataset
- Inferring transcription-factor activity
- Examining JAK–STAT signaling
- Quantifying the relative contribution of gp130-family receptors
- Applying Shapley-value-based LMG variance decomposition
- Comparing OSMR expression with a composite gp130-axis biomarker
- Validating the score in four independent infliximab-treated UC cohorts
- Surveying OSMR expression across 112 diseases

The main mechanistic hypothesis was that persistent STAT1 activation in treatment-refractory UC may not be driven predominantly by OSMR alone. Instead, multiple cytokine receptors sharing the gp130 signaling subunit may provide redundant inputs into the same pathway.

A 10-gene gp130-axis score was therefore evaluated against OSMR expression alone and a previously published five-gene signature.

---

### 9. Reproducibility and expert evaluation

The authors divided the two case studies into eight independent analyses and asked human experts to reproduce them using their own code.

- Experts agreed with the Virtual Biotech’s conclusions in all eight analyses.
- They judged the agent-generated code to be correct and free of major errors.
- Clinical-trial annotations were validated using 100 manually reviewed trials and the Therapeutic Data Commons dataset.
- Agreement for major endpoints and adverse-event rates was approximately 88–92%.

Thus, the system was designed to provide inspectable code, data, source provenance, and reports rather than opaque AI-generated conclusions.

---

### Key takeaway

The method combines **existing language models, specialized AI agents, MCP-based biomedical tools, code execution, multi-omic and statistical analyses, and human review**. The main innovation is not training a new model, but organizing an AI-based research workflow in which a virtual CSO coordinates multiple domain-specific agents that independently analyze heterogeneous evidence and integrate it into therapeutic-development recommendations.


<br/>
# Results
## 결과 및 비교 요약

### 1. 대규모 임상시험 주석화 평가

#### 테스트 데이터
- Open Targets 임상시험 데이터셋의 **55,984개 임상시험**을 분석했다.
  - Phase I: 14,237건
  - Phase II: 22,164건
  - Phase III: 14,911건
  - Phase IV: 4,672건
- Phase II·III 임상시험 **37,075건**에는 임상시험별로 하나의 AI 임상시험 분석 에이전트를 배정해 병렬 처리했다.
- 각 에이전트는 다음의 증거 계층을 사용했다.
  1. ClinicalTrials.gov
  2. PubMed 논문
  3. 보도자료 및 규제기관 발표

#### 평가 데이터와 메트릭
- 무작위로 선정한 **100개 임상시험**을 사람이 재검토했다.
- 에이전트와 사람의 일치율은 다음과 같았다.
  - 주요 평가변수(primary endpoint): **88.4%**
  - 이차 평가변수(secondary endpoint): **88.4%**
  - 이상반응률(adverse-event rate): **92.4%**
- 사람에 의해 수작업으로 정리된 TDC Trial Outcome Prediction 데이터셋과 겹치는 **7,666건**에서는 주요 평가변수 일치율이 **85.6%**였다.

#### 효율성
- 37,075건의 Phase II·III 시험을 약 **6시간**에 처리했다.
- 단일 에이전트로 처리할 경우 약 **1,839시간, 즉 76.6일**이 필요할 것으로 추정했다.
- 따라서 병렬 다중 에이전트 구조는 약 **184배의 속도 향상**을 제공했다.
- 임상시험 하나당 Anthropic API 비용의 중앙값은 약 **0.23달러**였다.

---

### 2. 표적 특징과 임상시험 성공의 연관성

#### 테스트 데이터와 분석 방법
- 임상시험 결과를 표적 유전자의 단일세포 발현 특징과 연결했다.
- Tabula Sapiens의 **27개 인간 조직**을 이용해 다음 지표를 계산했다.
  - 세포 유형 특이성: Tau index
  - 세포 내 발현 이질성: bimodality coefficient
- 임상시험 결과에 대해 로지스틱 회귀, beta 회귀, permutation test, mixed-effects model을 사용했다.
- 임상시험 단계, 등록연도, 치료 modality, 질환 영역 등을 보정했다.

#### 주요 메트릭과 결과
- 세포 유형 특이적 표적을 겨냥한 약물은 넓게 발현되는 표적의 약물보다:
  - Phase IV까지 도달할 가능성이 **48% 높았다**.
  - Phase I에서 Phase II로 진행할 가능성이 **40% 높았다**.
  - 주요 평가변수 성공과의 연관성: **OR = 1.12**
  - 이차 평가변수 성공과의 연관성: **OR = 1.12**
  - Phase I→II 진행과의 연관성: **OR = 1.27**
- 세포 유형 특이적 표적을 시험한 임상시험은 평균적으로 **이상반응률이 32% 낮았다**.
- 이러한 연관성은:
  - 1,000회 permutation test
  - 치료 modality·질환 영역을 포함한 mixed-effects model
  - 인간 유전학 근거 보정
  이후에도 통계적으로 유의했다.
- 특히 인간 유전학 근거가 없는 임상시험에서도 단일세포 특징의 효과가 유지되어, 단일세포 정보가 유전학 정보와 **추가적이고 독립적인 생물학적 신호**를 제공할 가능성을 보였다.

> 다만 이 결과는 관찰연구 기반의 연관성이지, 세포 유형 특이성이 임상 성공을 직접적으로 유발한다는 인과관계를 증명한 것은 아니다.

---

### 3. B7-H3 표적 평가: 다중 데이터 모달리티 검증

#### 테스트 데이터
Virtual Biotech은 B7-H3/CD276을 폐암 표적으로 평가하면서 다음 자료를 통합했다.

- SCLC 단일세포 데이터: **62,341개 세포, 9명 공여자**
- LUAD 단일세포 데이터: **337,002개 세포, 69명 공여자**
- 정상 폐 단일세포 데이터: **86,478개 세포, 26명 공여자**
- LUAD 공간전사체 데이터: **12개 조직, 25,000개 이상의 spatial spot**
- TCGA LUAD 생존 데이터: 최종 분석 **477명**
- 인간 유전학, 단백질 위치, 약물성(druggability), 기존 임상 근거

#### 분석 메트릭과 결과
- 단일세포 차등발현 분석:
  - B7-H3 발현 증가는 암세포 전반보다는 **섬유아세포에서 두드러졌다**.
  - SCLC: log2FC = **2.13**
  - LUAD: log2FC = **1.79**
- 세포 간 상호작용 분석:
  - SCLC에서 **180개**
  - LUAD에서 **226개**
  의 B7-H3 관련 상호작용을 확인했다.
- 공간전사체 분석:
  - B7-H3 발현이 높은 영역 주변에서 T세포, 대식세포, 단핵구, 수지상세포가 감소했다.
  - 이러한 면역세포 감소는 B7-H3 발현 spot에 가까울수록 강하고, 거리가 멀어질수록 약해졌다.
- 생존 분석:
  - 연령, 병기, 성별을 보정한 Cox 모델에서 높은 B7-H3 발현은:
    - 전체생존 악화: **HR = 1.62, P = 0.028**
    - 무병생존 악화: **HR = 2.06, P = 0.027**
- 최종적으로 Virtual Biotech은 B7-H3에 대해 **항체-약물 접합체(ADC)**를 우선 modality로 제안했다.
  - 세포 표면 발현
  - 암세포 및 종양미세환경 섬유아세포 표적화 가능성
  - bystander killing 가능성
  을 근거로 삼았다.

이 사례는 단일 데이터셋이 아니라 유전학, 단일세포, 공간전사체, 생존분석, 단백질학 및 약물성 정보를 연결해 치료 가설을 만든 사례다. 분석 비용은 약 **50달러**였다.

---

### 4. OSMR 임상시험 실패 분석

#### 테스트 데이터
- 분석 대상: 궤양성 대장염 Phase II MOONGLOW 연구
- 임상시험: **NCT06137183**
- 치료제: OSMRβ 차단 항체 **vixarelimab**
- 임상시험은 2025년 6월 중간 futility 분석에서 유효성 부족 가능성이 확인되어 종료됐다.
- 주요 단일세포 데이터:
  - TAURUS UC 데이터셋
  - **435,857개 세포, 20명 환자**
  - 항-TNF 치료 전후 데이터
  - 비반응자 14명, 반응자 6명

#### 분석 메트릭과 결과
- 전사인자 활성 분석:
  - 비반응자 조직의 OSMR 발현 기질세포에서 STAT1 활성이 광범위하게 증가했다.
  - 10개 기질세포 유형 중 **9개에서 유의한 차이**가 관찰됐다.
- Shapley 기반 분산분해:
  - STAT1 활성 변이를 설명하는 데 OSMR보다 다른 gp130 계열 수용체가 더 큰 기여를 하는 세포 유형이 많았다.
  - 이는 OSMRβ만 차단하는 치료가 gp130-JAK-STAT 축을 충분히 억제하지 못했을 가능성을 제시했다.
- 바이오마커 비교:
  - OSMR 단독 발현보다 10개 유전자로 구성한 **gp130-axis score**가 네 개의 독립적인 infliximab 치료 UC 코호트에서 비반응 예측 성능이 더 좋았다.
  - 성능 평가는 **ROC-AUC**로 수행했다.
  - 기존 5개 유전자 Arijs signature와도 대체로 비슷한 수준의 성능을 보였다.
- 네 개의 외부 검증 코호트:
  - GSE12251
  - GSE16879
  - GSE23597
  - GSE73661

#### 해석
Virtual Biotech은 임상 실패 원인을 “OSMR이 전혀 관련이 없기 때문”이 아니라, 다음과 같이 해석했다.

> OSMR은 치료 불응성 기질 염증 상태를 표시하지만, 해당 상태의 지속적인 JAK-STAT 신호를 단독으로 지배하는 주된 수용체는 아닐 수 있다.

분석 비용은 약 **59달러**였다.

---

### 5. 사람 전문가를 이용한 정확성·재현성 평가

#### 평가 설계
- B7-H3와 OSMR 사례를 총 **8개의 독립 분석**으로 나누었다.
  - B7-H3: Fig. 4C–F의 4개 분석
  - OSMR: Fig. 5B–D의 4개 분석
- 네 명의 전문가가 각자 두 분석을 독립적으로 다시 수행했다.
- 전문가들은:
  1. 자신의 코드로 분석을 재현하고
  2. Virtual Biotech의 결론과 일치하는지 평가하며
  3. 에이전트가 작성한 코드의 중대한 오류 여부를 검토했다.

#### 결과
- 8개 분석 모두에서 전문가들은 Virtual Biotech의 결론과 **일반적으로 일치한다(8/8, “Yes”)**고 평가했다.
- 에이전트 작성 코드의 중대한 오류가 없다는 평가도 **8/8**이었다.
- 일부 분석 선택의 차이는 있었지만 결론에는 영향을 주지 않는 사소한 차이였다.

즉, 두 사례연구의 계산 결과와 코드는 전문가의 독립 재분석에서 높은 재현성을 보였다.

---

### 6. 경쟁 에이전트 시스템과의 비교

#### 비교 모델
다음의 범용 또는 생의학 에이전트 시스템과 비교했다.

- **Biomni**
- **Kosmos**
- **PantheonOS**

#### 비교 조건
- 임상시험 주석화, B7-H3, OSMR이라는 동일한 사례를 제공했다.
- B7-H3와 OSMR에서는 가능한 후속 분석을 제안하게 한 뒤, 연구자가 Virtual Biotech 사례와 가장 유사한 분석을 선택했다.
- 따라서 완전히 자동화된 무감독 비교라기보다는, 유사한 수준의 인간 조정을 제공한 비교였다.

#### 비교 결과
1. **임상시험 주석화**
   - Biomni, Kosmos, PantheonOS 모두 사람 수작업 검토 데이터와 TDC 라벨에 대해 Virtual Biotech보다 낮은 정확도를 보였다.
   - 논문 본문에서는 구체적인 경쟁 모델별 수치가 표 S2에 제시되어 있다고 설명하지만, 제공된 본문에는 해당 수치가 포함되어 있지 않다.

2. **B7-H3 사례**
   - 경쟁 시스템들은 더 적은 데이터 modality를 사용했다.
   - Virtual Biotech보다 분석의 엄밀성이 낮았고, 생물학적으로 의미 있는 결과도 적었다.

3. **OSMR 사례**
   - 경쟁 시스템들은 gp130 축의 수용체 중복성, STAT1 활성, Shapley 기반 기여도 분석, 복합 바이오마커 검증을 Virtual Biotech만큼 체계적으로 연결하지 못했다.
   - Virtual Biotech은 임상시험 설계, 치료 불응성 조직, 단일세포 신호, 바이오마커 검증, 질환 간 분석을 하나의 실패 기전으로 통합했다.

#### 비교의 핵심 의미
Virtual Biotech의 우위는 단순히 더 많은 텍스트를 생성한 데 있지 않고,

- 여러 생물학적 데이터 유형을 동시에 사용하고
- 전문 에이전트별로 분석을 분할하며
- 결과를 CSO 에이전트가 통합하고
- 과학적 검토 에이전트가 근거와 분석 공백을 점검한다는 점

에 있다.

---

## 핵심 결론

- Virtual Biotech은 **대규모 임상시험 데이터 주석화**, **표적 우선순위화**, **치료 modality 선택**, **임상 실패 원인 분석**을 수행했다.
- 임상시험 주석화에서는 사람 검토와 약 **86–92% 수준의 일치율**을 보였다.
- 세포 유형 특이적 표적은 임상 단계 진행, endpoint 성공 및 낮은 이상반응률과 연관됐다.
- B7-H3 사례에서는 여러 데이터 유형을 통합해 폐암에서의 면역배제 기전과 ADC 전략을 제안했다.
- OSMR 사례에서는 OSMR 단독보다 **gp130 축 전체를 반영한 바이오마커**가 치료 불응성을 더 잘 설명할 가능성을 제시했다.
- 전문가 재현성 평가에서는 8개 분석 모두 결론과 코드가 대체로 타당하다고 평가됐다.
- 경쟁 시스템보다 더 많은 modality와 더 정교한 분석을 수행했지만, 사례연구 결과는 여전히 **가설 생성 및 의사결정 지원 수준**이며 실험적·전향적 검증이 필요하다.

---





## Results and Comparisons

### 1. Large-scale clinical trial annotation

#### Test data
The Virtual Biotech analyzed **55,984 clinical trials** from the Open Targets dataset:

- Phase I: 14,237
- Phase II: 22,164
- Phase III: 14,911
- Phase IV: 4,672

For large-scale outcome extraction, **37,075 Phase II and III trials** were processed in parallel, with one clinical-trialist agent assigned to each NCT identifier.

Each agent used a three-tier evidence cascade:

1. ClinicalTrials.gov  
2. PubMed publications  
3. Press releases and regulatory announcements  

#### Evaluation metrics
A random sample of 100 trials was manually reviewed.

Agreement between agents and human reviewers was:

- Primary endpoints: **88.4%**
- Secondary endpoints: **88.4%**
- Adverse-event rates: **92.4%**

Among 7,666 trials overlapping with the manually curated TDC Trial Outcome Prediction dataset, primary-endpoint agreement was **85.6%**.

#### Efficiency
- 37,075 Phase II/III trials were processed in approximately **6 hours**.
- A single-agent workflow was estimated to require approximately **1,839 hours, or 76.6 days**.
- The parallel multi-agent design therefore provided an estimated **184-fold speedup**.
- The median API cost per trial was approximately **$0.23**.

---

### 2. Target features associated with clinical trial success

#### Test data and methods
The system linked clinical trial outcomes to single-cell expression features of drug targets using the Tabula Sapiens atlas across **27 human tissues**.

Two major features were evaluated:

- Cell-type specificity: Tau index
- Within-population expression heterogeneity: bimodality coefficient

The analyses used logistic regression, beta regression, permutation testing, and mixed-effects models, with adjustment for trial phase, enrollment year, therapeutic modality, and disease area.

#### Main results
Drugs targeting cell-type-specific genes were:

- **48% more likely** to reach Phase IV
- **40% more likely** to progress from Phase I to Phase II
- Associated with primary-endpoint success: **OR = 1.12**
- Associated with secondary-endpoint success: **OR = 1.12**
- Associated with Phase I-to-II progression: **OR = 1.27**

Trials involving cell-type-specific targets also had, on average, **32% lower adverse-event rates**.

The associations remained statistically significant after:

- 1,000 permutation tests
- Mixed-effects adjustment for modality and disease area
- Adjustment for human genetic evidence

The effects were also observed among trials without supporting genetic evidence, suggesting that single-cell features may provide biological information beyond human genetic evidence.

These are observational associations and do not establish causality.

---

### 3. B7-H3 target evaluation in lung cancer

#### Test data
The B7-H3/CD276 case integrated:

- SCLC single-cell data: **62,341 cells from 9 donors**
- LUAD single-cell data: **337,002 cells from 69 donors**
- Healthy lung single-cell data: **86,478 cells from 26 donors**
- LUAD spatial transcriptomics: **12 tissue samples and more than 25,000 spatial spots**
- TCGA LUAD survival data: **477 patients**
- Genetic, protein-localization, druggability, and clinical evidence

#### Metrics and findings
Single-cell differential expression showed that B7-H3 up-regulation was concentrated in fibroblasts:

- SCLC: log2FC = **2.13**
- LUAD: log2FC = **1.79**

Cell–cell communication analysis identified:

- **180 interactions** in SCLC
- **226 interactions** in LUAD

Spatial analysis showed that B7-H3-high regions had lower local abundance of T cells, macrophages, monocytes, and dendritic cells. The effect weakened with increasing spatial distance, consistent with a local immune-exclusion phenotype.

In multivariable Cox models adjusted for age, stage, and sex:

- Overall survival: **HR = 1.62, P = 0.028**
- Disease-free survival: **HR = 2.06, P = 0.027**

The system proposed an **antibody–drug conjugate (ADC)** as the preferred modality, based on cell-surface expression, tumor-microenvironment fibroblast expression, and potential bystander killing.

The complete analysis cost approximately **$50** in API credits.

---

### 4. Analysis of the terminated OSMR trial

#### Test data
The system analyzed the terminated Phase II MOONGLOW trial:

- Trial: **NCT06137183**
- Disease: Ulcerative colitis
- Drug: vixarelimab, an OSMRβ-blocking antibody
- TAURUS single-cell dataset: **435,857 cells from 20 patients**
- Patients included 14 nonresponders and 6 responders

#### Metrics and findings
STAT1 activity was significantly elevated in treatment-refractory nonresponder stroma in **9 of 10 stromal cell types**.

A Shapley-based variance decomposition showed that OSMR was not the dominant contributor to STAT1 activity in many OSMR-expressing stromal populations. Other gp130-family receptors often explained more variation.

The system then constructed a 10-gene **gp130-axis score** and compared it with:

- OSMR expression alone
- The established five-gene Arijs signature

The gp130-axis score consistently outperformed OSMR alone across four independent infliximab-treated UC cohorts:

- GSE12251
- GSE16879
- GSE23597
- GSE73661

Performance was evaluated using ROC-AUC.

The interpretation was that OSMR may mark a treatment-refractory stromal inflammatory state but may not be the dominant driver of persistent JAK–STAT signaling. Broader redundancy within the gp130 axis may therefore explain the failure of OSMR-only blockade.

The analysis cost approximately **$59**.

---

### 5. Human expert evaluation of correctness and reproducibility

The two case studies were divided into **eight independent analyses**:

- Four from the B7-H3 study
- Four from the OSMR study

Four human experts independently reproduced the analyses, wrote their own code, and assessed both the conclusions and the agent-generated code.

Results:

- Experts judged that their conclusions matched the Virtual Biotech findings in **all 8 analyses**.
- They judged the agent-generated code to be free of major errors affecting conclusions in **all 8 analyses**.

Minor analytical choices differed between experts and agents, but these differences did not affect the overall conclusions.

---

### 6. Comparison with competing agentic systems

#### Compared systems
The Virtual Biotech was compared with:

- **Biomni**
- **Kosmos**
- **PantheonOS**

#### Comparison design
The systems received the same case studies:

- Clinical trial annotation
- B7-H3
- OSMR

For the open-ended case studies, the competing systems received a similar level of human steering. Researchers selected follow-up analyses that most closely matched those performed by the Virtual Biotech.

#### Comparison results
1. **Clinical trial annotation**
   - All three competing systems produced less accurate annotations than the Virtual Biotech on both the manual-review set and the TDC labels.
   - The main text refers to Table S2 for detailed model-specific values, but those numerical values are not included in the provided manuscript text.

2. **B7-H3 analysis**
   - The competing systems used fewer data modalities.
   - Their analyses were less rigorous and generated fewer biologically meaningful findings.

3. **OSMR analysis**
   - The competing systems did not integrate gp130-axis redundancy, STAT1 activity, Shapley-based receptor contributions, and external biomarker validation as comprehensively.
   - The Virtual Biotech connected trial design, treatment-refractory tissue biology, single-cell signaling, biomarker prediction, and cross-disease analysis into a unified explanation of clinical failure.

### Overall conclusion
The principal advantage of the Virtual Biotech was not simply text generation. It was the combination of:

- Multiple specialized agents
- Parallel analysis of heterogeneous biomedical data
- Integration by a virtual CSO
- Quality control by a scientific reviewer agent
- Reproducible code and source-tracked evidence

However, the case-study conclusions remain **data-driven hypotheses and decision-support outputs**, not definitive clinical or experimental proof.


<br/>
# 예제



이 논문에서 **Virtual Biotech 자체를 새로운 AI 모델로 학습(training)시킨 것은 아닙니다.**  
기반 언어모델은 Claude Sonnet 4.5이며, 논문의 핵심은 모델을 새로 훈련하는 것보다 다음을 결합한 **다중 에이전트 연구 시스템**을 구축한 것입니다.

- 가상 CSO가 연구 질문을 세부 과제로 분해
- 전문 AI 에이전트가 데이터베이스와 분석 도구를 사용
- 과학적 검토 에이전트가 결과와 근거를 점검
- 필요하면 추가 분석을 지시
- 최종적으로 근거가 추적 가능한 보고서와 코드를 생성

따라서 논문에서 말하는 “training data/test data”는 일반적인 지도학습 모델의 학습·테스트 데이터라기보다, **분석에 사용한 입력 데이터와 결과를 검증한 독립적 평가 데이터**로 이해하는 것이 정확합니다.

---

# 1. 임상시험 결과 주석 작업

### 구체적인 테스크

ClinicalTrials.gov와 Open Targets에 등록된 임상시험의 결과를 표준화하여 다음을 판정하는 작업입니다.

- 임상시험 단계 진행 여부
- 1차 평가변수(primary endpoint) 성공 여부
- 2차 평가변수(secondary endpoint) 성공 여부
- 이상반응(adverse event) 발생률
- 임상시험 중단 이유

### 입력

각 에이전트에는 하나의 NCT ID가 입력되었습니다.

예를 들어 입력은 다음과 같은 형태입니다.

```text
NCT ID
+ ClinicalTrials.gov 기록
+ 시험 단계와 치료제 정보
+ 연구 대상 질환
+ PubMed 논문
+ 보도자료 및 규제기관 발표
```

총 **37,075개의 임상시험 에이전트**가 병렬로 실행되었으며, 각 에이전트가 하나의 NCT ID를 담당했습니다.

에이전트는 다음 순서로 정보를 검색했습니다.

1. ClinicalTrials.gov
2. PubMed의 임상시험 결과 논문
3. 보도자료 및 규제기관 발표

### 출력

각 임상시험에 대해 구조화된 JSON 형태의 결과를 만들었습니다.

```text
{
  trial_phase: Phase II,
  primary_endpoint: positive / negative / unclear,
  secondary_endpoint: positive / negative / unclear,
  adverse_event_rate: numerical value,
  trial_status: completed / terminated / ongoing,
  termination_reason: futility / safety / business / other,
  evidence_sources: [source URLs or publications]
}
```

이렇게 얻은 결과를 target-level molecular feature와 연결하여 최종적으로 **55,984개 임상시험 데이터셋**을 구축했습니다.

### 검증용 데이터와 결과

일반적인 학습/테스트 분할은 아니지만, 별도의 검증이 수행되었습니다.

- 무작위 임상시험 100건을 사람이 직접 검토
- 1차 평가변수 일치율: **88.4%**
- 2차 평가변수 일치율: **88.4%**
- 이상반응률 일치율: **92.4%**
- Therapeutic Data Commons와 겹치는 7,666개 시험에서 1차 평가변수 일치율: **85.6%**

즉, 이 단계에서 사람의 수작업 주석과 비교하여 에이전트 출력의 신뢰성을 평가했습니다.

---

# 2. 단일세포 발현 특징과 임상시험 성공률 분석

### 구체적인 테스크

약물 표적 유전자의 단일세포 수준 특성이 임상시험 성공과 관련되는지를 분석했습니다.

주요 특징은 다음 두 가지입니다.

1. **Cell-type specificity, Tau 지수**
   - 특정 세포 유형에만 발현되는 정도
   - 0에 가까우면 여러 세포에 널리 발현
   - 1에 가까우면 특정 세포 유형에 제한적으로 발현

2. **Expression bimodality**
   - 같은 세포 집단 안에서 유전자가 켜진 세포와 꺼진 세포가 나뉘는 정도

### 입력

- Tabula Sapiens 인간 단일세포 아틀라스
- 27개 조직
- 세포 유형별 유전자 발현
- Open Targets의 유전적 근거 점수
- 55,984개 임상시험의 성공·실패 및 이상반응 주석

### 분석 출력

각 표적 유전자에 대해 다음과 같은 특징값을 계산했습니다.

```text
Target gene: Gene X
Tau cell-type specificity: 0.75
Bimodality coefficient: 0.42
Genetic evidence: present / absent
```

그 후 로지스틱 회귀, beta 회귀, 혼합효과 모델 등을 사용해 임상시험 결과와의 연관성을 분석했습니다.

### 주요 결과

- 세포 유형 특이적 표적을 대상으로 한 약물은 Phase IV까지 도달할 가능성이 **48% 높음**
- Phase I에서 Phase II로 진행할 가능성이 **40% 높음**
- 세포 유형 특이적 표적을 대상으로 한 시험은 이상반응률이 평균 **32% 낮음**
- 높은 Tau 지수는 1차 평가변수와 2차 평가변수 성공과도 관련됨
- 이러한 연관성은 기존의 human genetic evidence를 보정한 뒤에도 유지됨

여기서 중요한 점은 이것이 표적의 성공을 예측하는 지도학습 모델의 “테스트 정확도”를 보고한 것이 아니라, **관찰 데이터에서 특정 생물학적 특징과 임상 결과의 통계적 연관성을 분석한 것**이라는 점입니다.

---

# 3. B7-H3 표적의 폐암 치료 가능성 평가

### 구체적인 테스크

사용자 질문:

> “B7-H3를 폐암 치료 표적으로 평가하라.”

Virtual CSO는 이를 여러 하위 과제로 분해했습니다.

- 인간 유전학적 근거 확인
- 암과 정상 조직에서의 발현 비교
- 세포 유형별 발현 분석
- 세포 간 신호 전달 분석
- 공간 전사체 분석
- 환자 생존 분석
- 적절한 약물 모달리티 선택

### 입력

- B7-H3 또는 CD276 유전자
- SCLC 단일세포 데이터: 62,341개 세포, 9명 공여자
- LUAD 단일세포 데이터: 337,002개 세포, 69명 공여자
- 정상 폐 조직 단일세포 데이터
- LUAD 공간 전사체 데이터: 12개 조직, 25,000개 이상 spatial spot
- TCGA LUAD 코호트: 최종 477명
- Human Protein Atlas
- GWAS 및 유전적 credible set 정보
- 기존 약물 및 표적 가능성 데이터

### 출력과 분석 결과

에이전트들은 다음과 같은 결과를 단계적으로 도출했습니다.

1. **유전학 에이전트**
   - 폐암에서 B7-H3를 지지하는 강한 germline GWAS 신호는 발견되지 않음
   - 그러나 면역관문 표적의 경우 종양 내 과발현 자체가 치료 근거가 될 수 있으므로, 유전학적 근거가 없다고 표적을 배제하지 않음

2. **단일세포 에이전트**
   - B7-H3 발현 증가는 암세포 전체에서 균일하지 않음
   - 특히 암 관련 섬유아세포(fibroblast)에 집중됨

3. **세포 간 신호 분석**
   - B7-H3 고발현 섬유아세포가 면역세포와 상호작용하는 여러 신호 경로를 보임
   - 면역억제, 대식세포 재프로그래밍, 혈관 신생, 케모카인 경로 등이 관찰됨

4. **공간 전사체 분석**
   - B7-H3 고발현 영역 주변에서 T세포, 대식세포, 단핵구, 수지상세포가 감소
   - 면역세포 감소가 가까운 공간에서 가장 강하고 멀어질수록 약해져 국소적 면역 배제 현상을 지지

5. **생존 분석**
   - 높은 B7-H3 발현은 나쁜 전체 생존과 관련
   - Overall survival: HR = 1.62
   - Disease-free survival: HR = 2.06

6. **모달리티 선택**
   - B7-H3가 세포 표면에 존재하고 암세포 및 종양미세환경 세포에서 발현되므로 **항체-약물 접합체(ADC)**를 우선 제안
   - 소분자 약물과 PROTAC은 기존 결합체 및 druggability 근거가 부족해 후순위로 배치

### 평가 방식

이 결과는 별도 테스트 세트에서 예측 정확도를 계산한 것이 아닙니다. 대신 네 개의 독립 분석을 사람 전문가가 재현했습니다.

- 차등 발현 분석
- 세포 간 신호 분석
- 공간 면역 배제 분석
- 생존 분석

네 명의 전문가가 각 분석을 직접 다시 수행했고, **8개 분석 모두 Virtual Biotech의 결론과 대체로 일치**한다고 평가했습니다. 또한 에이전트가 작성한 코드도 8개 분석 모두 중대한 오류가 없다고 평가되었습니다.

---

# 4. OSMR 표적과 궤양성 대장염 임상시험 실패 분석

### 구체적인 테스크

중단된 Phase II 임상시험을 입력으로 받아, 왜 효능 부족으로 실패했는지 추론하는 작업입니다.

대상 시험:

- NCT06137183
- 약물: vixarelimab
- 표적: OSMRβ
- 질환: 중등도–중증 궤양성 대장염
- 중단 이유: 중간 무용성 분석에서 1차 평가변수 달성 가능성이 낮다고 판단

### 입력

- 임상시험 프로토콜과 환자 선정 기준
- ClinicalTrials.gov 시험 기록
- TAURUS 단일세포 아틀라스
  - 435,857개 세포
  - 궤양성 대장염 환자 20명
  - anti-TNF 치료 전후 조직
- 치료 반응자와 비반응자의 세포 상태
- OSMR 및 gp130 계열 수용체 발현
- STAT1 전사인자 활성
- 독립적인 4개 infliximab 치료 코호트

### 출력과 분석 결과

1. OSMR은 치료 비반응자의 특정 섬유아세포 상태에서 증가했습니다.
2. 그러나 STAT1 활성은 OSMR만이 아니라 여러 gp130 계열 수용체가 공유하는 신호축과 관련되었습니다.
3. Shapley 기반 분산 분해 분석에서 OSMR은 많은 세포 유형에서 STAT1 활성의 주요 설명 변수가 아니었습니다.
4. 따라서 OSMR만 차단하는 vixarelimab은 치료 불응성 조직에서 활성화된 전체 신호축을 충분히 억제하지 못했을 가능성이 제기되었습니다.
5. 에이전트는 OSMR 단독보다 다음 10개 유전자로 구성된 **gp130-axis risk score**를 제안했습니다.

```text
OSMR, IL6ST, LIFR, IL6R, IL11RA,
IL6, IL11, OSM, LIF, STAT1
```

### 독립 검증

이 점수는 4개의 별도 infliximab 치료 코호트에서 평가되었습니다.

비교 대상:

- OSMR 발현 단독
- 새로 만든 gp130-axis 점수
- 기존 5개 유전자 anti-TNF 반응 시그니처

결과적으로 gp130-axis 점수는 네 코호트 모두에서 OSMR 단독보다 비반응 예측 성능이 좋았고, 기존의 임상적으로 알려진 5개 유전자 시그니처와 비슷한 수준의 성능을 보였습니다.

이는 새로운 환자 코호트에 대한 일종의 **외부 검증**에 해당하지만, 저자들은 이 결과를 치료 효과를 입증하는 것으로 해석하지 않고, 추가 검증이 필요한 기전적 가설로 제시했습니다.

---

# 5. 다른 AI 시스템과의 비교 평가

논문은 Virtual Biotech을 다음 시스템과 비교했습니다.

- Biomni
- Kosmos
- PantheonOS

### 동일한 입력

세 시스템에 동일한 유형의 질문을 제공했습니다.

- 임상시험 결과 주석
- B7-H3 폐암 표적 평가
- OSMR 궤양성 대장염 임상시험 실패 분석

B7-H3와 OSMR 분석에서는 가능한 다음 단계를 제시하게 한 뒤, 사람이 Virtual Biotech의 분석과 가장 유사한 방향을 선택하는 방식으로 인간 개입 수준을 맞췄습니다.

### 비교 출력

평가 결과, Virtual Biotech은 다른 시스템에 비해 다음과 같은 차이를 보였습니다.

- 더 많은 데이터 유형 사용
- 더 체계적인 통계 분석 수행
- 더 많은 생물학적 근거 통합
- 임상시험 주석에서 더 높은 정확도
- 더 구체적이고 검증 가능한 결론 생성

다만 논문은 이를 완전한 독립적 벤치마크라기보다, 각 시스템에 유사한 수준의 인간 안내를 제공한 비교 평가로 제시합니다.

---

## 핵심 정리

| 구분 | 입력 | 출력 | 검증 방식 |
|---|---|---|---|
| 임상시험 주석 | NCT ID, 임상시험 기록, 논문, 보도자료 | endpoint, 이상반응, 중단 이유가 포함된 JSON | 사람 주석 및 TDC와 비교 |
| 표적 우선순위 분석 | 단일세포 발현, 유전적 근거, 임상시험 결과 | Tau·bimodality와 임상 성공의 통계적 연관성 | permutation, 혼합효과 모델, 유전적 근거 보정 |
| B7-H3 평가 | 단일세포·공간 전사체·생존·약물성 데이터 | 면역 배제 기전과 ADC 추천 | 전문가가 코드와 분석을 재현 |
| OSMR 실패 분석 | 임상시험 프로토콜, TAURUS 단일세포, 독립 코호트 | gp130 축의 중복성 및 복합 바이오마커 | 4개 외부 코호트 검증 |
| 시스템 비교 | 동일한 연구 질문 | 정확성·분석 범위·생물학적 의미 비교 | Biomni, Kosmos, PantheonOS와 비교 |

---





## Important distinction

The Virtual Biotech was **not trained as a new supervised machine-learning model** using a conventional training/test split. It used Claude Sonnet 4.5 together with specialized AI agents, biomedical databases, analysis tools, and a virtual CSO orchestrator.

Therefore, the paper’s “input/output” examples refer mainly to:

- analytical inputs,
- generated scientific outputs,
- and independent validation or benchmarking datasets.

---

## 1. Clinical-trial outcome annotation

### Task

The system standardized outcomes from clinical trials, including:

- phase progression,
- primary endpoint success,
- secondary endpoint success,
- adverse-event rates,
- and reasons for trial termination.

### Input

Each agent received one NCT identifier together with information retrieved from:

1. ClinicalTrials.gov,
2. PubMed,
3. press releases and regulatory announcements.

A total of **37,075 trial-specific agents** were run in parallel.

### Output

Each agent produced a structured record similar to:

```text
{
  trial_phase: Phase II,
  primary_endpoint: positive / negative / unclear,
  secondary_endpoint: positive / negative / unclear,
  adverse_event_rate: numerical value,
  trial_status: completed / terminated / ongoing,
  termination_reason: futility / safety / business / other,
  evidence_sources: [...]
}
```

These annotations were integrated into a dataset of **55,984 clinical trials**.

### Validation

This was not a conventional test set, but the annotations were evaluated against human review and the Therapeutic Data Commons dataset.

- Primary endpoint agreement with human review: **88.4%**
- Secondary endpoint agreement: **88.4%**
- Adverse-event-rate agreement: **92.4%**
- Agreement with TDC labels across 7,666 overlapping trials: **85.6%**

---

## 2. Single-cell target features and clinical success

### Task

The system tested whether single-cell expression properties of drug targets were associated with clinical-trial outcomes.

The two main features were:

- **Tau cell-type specificity**: whether a gene is restricted to particular cell types
- **Expression bimodality**: whether expression separates cells into high- and low-expression subpopulations

### Input

- Tabula Sapiens single-cell atlas
- 27 human tissues
- cell-type-specific gene expression
- Open Targets genetic evidence
- clinical-trial outcomes from 55,984 trials

### Output

For each target gene, the system generated features such as:

```text
Target gene: Gene X
Tau specificity: 0.75
Bimodality coefficient: 0.42
Genetic evidence: present / absent
```

It then fitted logistic, beta-regression, and mixed-effects models.

### Main findings

- Drugs targeting cell-type-specific genes were **48% more likely to reach Phase IV**
- They were **40% more likely to progress from Phase I to Phase II**
- They showed approximately **32% lower adverse-event rates**
- The associations remained significant after adjustment for genetic evidence

These analyses measured statistical associations in observational data; they did not report the predictive accuracy of a standard supervised model on a held-out test set.

---

## 3. B7-H3 evaluation in lung cancer

### Task

The user asked:

> “Evaluate B7-H3 as a therapeutic target in lung cancer.”

The CSO decomposed this into:

- genetic evidence,
- cell-type-specific expression,
- cell-cell communication,
- spatial immune exclusion,
- survival analysis,
- and modality selection.

### Input

- B7-H3/CD276
- SCLC single-cell atlas: 62,341 cells from 9 donors
- LUAD single-cell atlas: 337,002 cells from 69 donors
- normal lung single-cell data
- LUAD spatial transcriptomics: over 25,000 spots from 12 samples
- TCGA LUAD cohort: 477 patients
- Human Protein Atlas
- genetic and druggability data

### Output

The system concluded that:

- strong germline genetic support for B7-H3 in lung cancer was absent;
- B7-H3 expression was particularly increased in fibroblasts;
- B7-H3-high fibroblasts were associated with immunosuppressive signaling;
- B7-H3-high spatial regions had fewer nearby T cells, macrophages, monocytes, and dendritic cells;
- high B7-H3 expression was associated with worse overall and disease-free survival;
- an antibody-drug conjugate, or **ADC**, was the preferred modality.

The reported survival associations were:

- Overall survival: HR = 1.62
- Disease-free survival: HR = 2.06

### Validation

Four human experts independently reproduced eight analyses across the B7-H3 and OSMR case studies.

For all eight analyses:

- the experts judged that their conclusions generally matched the Virtual Biotech’s conclusions;
- the agent-written code was judged free of errors that materially affected the results.

---

## 4. OSMR trial-failure analysis in ulcerative colitis

### Task

The system analyzed why a Phase II trial of vixarelimab against OSMRβ in ulcerative colitis was terminated for futility.

### Input

- Clinical-trial protocol and eligibility criteria
- NCT06137183
- TAURUS single-cell atlas
  - 435,857 cells
  - 20 ulcerative colitis patients
  - samples before and after anti-TNF therapy
- OSMR-expressing stromal cell states
- STAT1 transcription-factor activity
- independent infliximab-treated UC cohorts

### Output

The system found that:

1. OSMR was increased in fibroblast states from treatment nonresponders.
2. However, downstream STAT1 activity was shared by multiple gp130-family receptors.
3. OSMR was not the dominant contributor to STAT1 activity in most relevant stromal cell types.
4. Blocking OSMR alone may therefore have produced only a partial blockade of a redundant signaling axis.
5. The system proposed a 10-gene **gp130-axis score**:

```text
OSMR, IL6ST, LIFR, IL6R, IL11RA,
IL6, IL11, OSM, LIF, STAT1
```

### External validation

The score was evaluated in four independent infliximab-treated UC cohorts and compared with:

- OSMR expression alone,
- the new gp130-axis score,
- and a previously published five-gene anti-TNF response signature.

The gp130-axis score consistently outperformed OSMR alone and showed performance comparable to the established five-gene signature.

The authors present this as mechanistic and biomarker evidence requiring further validation, not as proof of clinical efficacy.

---

## 5. Comparison with other AI systems

The Virtual Biotech was compared with:

- Biomni,
- Kosmos,
- PantheonOS.

All systems were given comparable questions involving:

- clinical-trial annotation,
- B7-H3 evaluation,
- and OSMR trial-failure analysis.

The Virtual Biotech generally:

- used more data modalities,
- performed more rigorous statistical analyses,
- integrated more biological evidence,
- produced more accurate trial annotations,
- and generated more biologically meaningful conclusions.

However, this was a guided comparative evaluation rather than a conventional fixed test-set benchmark.

---

## Overall summary

The paper does not present a standard “training dataset versus test dataset” experiment. Instead, it presents a workflow in which:

1. **Inputs** are clinical-trial records, single-cell and spatial transcriptomic data, genetic evidence, survival data, and pharmacological resources.
2. **Agents** perform specialized analyses.
3. **Outputs** are structured annotations, statistical results, mechanistic hypotheses, biomarker scores, and modality recommendations.
4. **Validation** is performed through human review, independent cohorts, reproducibility analyses, permutation tests, mixed-effects models, and comparisons with other agentic systems.

<br/>
# 요약

 
Virtual Biotech은 가상 CSO가 여러 전문 AI 에이전트를 지휘하고 MCP를 통해 유전학·단일세포·공간전사체·임상시험 데이터를 통합 분석하는 다중 에이전트 플랫폼이다.  
37,075개 에이전트로 55,984건의 임상시험 결과를 정리한 결과, 세포 유형 특이적 표적 약물은 시장 도달 가능성이 48% 높고 이상반응률이 32% 낮았으며, 전문가 검토에서도 분석 결과와 코드가 재현되었다.  
사례 분석에서 B7-H3는 폐암의 면역배제 미세환경과 연관되어 ADC 전략이 제안되었고, OSMR 표적 궤양성대장염 임상시험 실패는 OSMR 단독보다 gp130 축의 신호 중복성이 중요할 가능성과 복합 바이오마커의 필요성을 보여주었다.  


 
Virtual Biotech is a multi-agent platform in which a virtual CSO coordinates specialized AI agents that integrate genetics, single-cell, spatial transcriptomics, and clinical-trial data through MCP tools.  
Using 37,075 agents to curate 55,984 clinical trials, the study found that drugs targeting cell-type-specific genes were 48% more likely to reach the market and had 32% fewer adverse events; expert review also confirmed the reproducibility of the analyses and code.  
In case studies, B7-H3 was linked to an immune-excluded lung-cancer microenvironment, supporting an ADC strategy, while analysis of an unsuccessful OSMR ulcerative-colitis trial suggested signaling redundancy across the gp130 axis and the need for a composite biomarker rather than OSMR alone.

<br/>
# 기타



논문 본문과 그림 설명에서 확인되는 **다이어그램·피규어·테이블·어펜딕스/보충자료의 핵심 결과와 인사이트**는 다음과 같습니다.  
※ 보충자료의 원문 표 전체가 제공된 것은 아니므로, 아래 내용은 본문에서 직접 언급된 범위에 근거합니다.

### 1. Fig. 1 — Virtual Biotech의 전체 구조와 작동 방식

**결과**
- Virtual Biotech는 가상 **Chief Scientific Officer(CSO)**가 전체 연구를 조정하고, 여러 전문 AI 에이전트가 역할을 나누어 분석하는 계층적 다중 에이전트 시스템이다.
- 주요 부서는 다음 네 영역으로 구성된다.
  - 표적 발굴 및 우선순위화
  - 표적 안전성 평가
  - 치료 modality 선택
  - 임상 개발 및 임상시험 분석
- CSO는 사용자의 질문을 세부 과제로 분해하고, 각 과제를 전문 에이전트에 배분한다.
- 분석 후 과학 검토 에이전트가 근거의 강도, 누락된 분석, 과도한 주장을 검토하며 필요하면 재분석을 요청한다.
- MCP 서버를 통해 Open Targets, CELLxGENE, ClinicalTrials.gov, cBioPortal, FDA 안전성 자료 등 다양한 데이터와 분석 도구를 연결한다.

**인사이트**
- 이 시스템의 핵심은 단순히 여러 AI를 병렬로 사용하는 것이 아니라, **전문성 분업 → 독립 분석 → 검토 → 통합 추론**의 구조를 갖춘 점이다.
- 따라서 서로 다른 생물학적 규모와 데이터 유형을 한 의사결정 과정에 연결할 수 있다.
- 다만 최종 판단을 완전히 자동화하기보다는, 인간이 질문의 방향을 설정하고 결과를 검증하는 **의사결정 지원 시스템**에 가깝다.

---

### 2. Fig. 2 — 대규모 임상시험 결과 자동 주석

**결과**
- 총 **55,984개 임상시험**을 분석했으며, 37,075개의 Phase II·III 시험에는 각각 하나의 임상시험 분석 에이전트를 배정했다.
- 각 에이전트는 다음의 3단계 근거 체계를 사용했다.
  1. ClinicalTrials.gov
  2. PubMed 논문
  3. 보도자료 및 규제기관 발표
- 시험 진행 단계, 주요·부차적 평가변수, 이상반응률, 중단 사유 등을 표준화된 JSON 형식으로 정리했다.
- 37,075개 시험을 약 6시간에 처리했으며, 단일 에이전트 방식보다 약 **184배 빠른 처리**가 가능했다고 보고했다.
- 무작위 100개 시험의 수작업 검토에서 주요 평가변수와 부차적 평가변수의 일치율은 각각 **88.4%**, 이상반응률은 **92.4%**였다.

**인사이트**
- 다중 에이전트 구조는 대규모 문헌·임상시험 큐레이션에 특히 유리하다.
- 각 에이전트가 하나의 시험에 집중하므로, 여러 시험을 한 번에 처리하는 단일 에이전트보다 문맥 혼동이 적다.
- 다만 결과를 성공/실패의 이분법으로 바꾸는 과정에서 다중 시험군, 경계적인 p값, 부분적 성공 같은 모호성이 남는다.

---

### 3. Fig. 3 — 단일세포 기반 표적 특성과 임상 성공

**결과**
- 표적 유전자의 단일세포 발현에서 두 가지 특징을 추출했다.
  - **Tau cell-type specificity**: 특정 세포 유형에 발현이 집중되는 정도
  - **Bimodality coefficient**: 같은 세포 집단 안에서 발현이 켜짐/꺼짐처럼 분리되는 정도
- 세포 유형 특이성이 높은 표적은:
  - Phase IV까지 도달할 가능성이 **48% 높았고**
  - Phase I에서 Phase II로 진행할 가능성이 **40% 높았으며**
  - 주요 및 부차적 평가변수를 달성할 가능성도 높았다.
- 세포 유형 특이적 표적을 겨냥한 약물은 평균적으로 **이상반응률이 32% 낮았다**.
- 이러한 관계는 질환 영역, 약물 modality, 시험 단계, 등록 연도 등을 보정한 뒤에도 유지됐다.
- 유전적 근거를 보정하거나 유전적 근거가 없는 시험만 분석했을 때도 단일세포 특징의 효과가 유지됐다.

**인사이트**
- 표적이 특정 세포 유형에 제한적으로 발현되면, 원하는 세포에는 작용하면서 다른 조직의 부작용은 줄일 가능성이 있다.
- 단일세포 데이터는 인간 유전학적 근거를 단순히 대체하는 것이 아니라, **유전학과 독립적인 추가 정보**를 제공한다.
- 그러나 이 결과는 관찰연구 기반 연관성이므로, 세포 특이성이 임상 성공을 직접 유발한다고 해석해서는 안 된다.

---

### 4. Fig. 4 — 폐암에서 B7-H3 표적 및 ADC 전략

**결과**
- B7-H3/CD276의 폐선암(LUAD) 및 소세포폐암(SCLC)에서의 치료 가능성을 다중 데이터로 평가했다.
- 생식세포 GWAS에서는 강한 유전적 근거가 없었지만, AI는 이를 표적 부적격으로 단정하지 않았다.
- 단일세포 분석에서 B7-H3 발현 증가는 암세포 전체보다 **섬유아세포**, 특히 종양미세환경의 섬유아세포에 집중됐다.
- B7-H3 고발현 섬유아세포는 면역억제성 사이토카인, 대식세포 재프로그래밍, 혈관신생, 면역세포 잔류 등과 관련된 신호를 보였다.
- 공간전사체 분석에서는 B7-H3 고발현 영역 주변에서 T세포, 대식세포, 단핵구, 수지상세포가 감소했다.
- B7-H3 고발현 환자는 낮은 발현 환자보다 생존이 불량했다.
  - 전체 생존: HR 1.62
  - 무병 생존: HR 2.06
- 최종적으로 B7-H3에 대해 **항체-약물 접합체(ADC)**를 우선 modality로 제안했다.

**인사이트**
- B7-H3의 의미를 암세포 표면 표적에만 한정하지 않고, **B7-H3 고발현 암 관련 섬유아세포와 면역배제 미세환경**이라는 관점으로 확장했다.
- ADC는 암세포에 대한 직접적인 세포독성, 종양미세환경 조절, 주변 세포에 대한 bystander killing을 동시에 기대할 수 있다는 논리로 선택됐다.
- 이 사례는 단일 데이터셋으로는 얻기 어려운 치료 가설을 단일세포·공간·생존·약리학 데이터를 결합해 도출한 예다.
- 다만 이 분석은 전임상 가설 생성 단계이며, 실제 기전과 치료 효과는 실험 및 임상 검증이 필요하다.

---

### 5. Fig. 5 — OSMRβ 표적 UC 임상시험 실패 분석

**결과**
- 분석 대상은 중등도~중증 궤양성 대장염에서 vixarelimab을 평가한 Phase II MOONGLOW 시험(NCT06137183)이다.
- 시험은 안전성 문제가 아니라 **중간 무익성 분석에서 유효성 달성 가능성이 낮다고 판단되어 중단**됐다.
- OSMR 발현은 치료 불응 환자의 여러 섬유아세포 상태에서 증가했으며, 표적 선정의 생물학적 근거 자체는 존재했다.
- 그러나 STAT1 활성 분석 결과, OSMR이 발현되는 세포에서도 STAT1 활성은 OSMR 하나만으로 설명되지 않았다.
- OSMRβ와 같은 gp130 계열 신호를 공유하는 여러 수용체가 JAK–STAT 경로에 병렬로 기여하는 것으로 나타났다.
- Shapley 기반 기여도 분석에서 OSMR은 대부분의 관련 세포 유형에서 STAT1 변이를 설명하는 가장 큰 요인이 아니었다.
- 이에 따라 OSMR 단독 점수보다 여러 gp130 축 유전자를 포함한 **10개 유전자 gp130-axis 점수**가 치료 불응성을 더 잘 예측했다.
- 이 점수는 4개의 독립적인 infliximab 치료 UC 코호트에서 OSMR 단독보다 일관되게 우수한 성능을 보였다.

**인사이트**
- 실패 원인은 OSMR이 전혀 관련이 없어서가 아니라, OSMR이 더 넓은 **gp130–JAK–STAT 염증 네트워크의 일부에 불과했기 때문**일 수 있다.
- 단일 수용체를 차단해도 병렬 수용체가 같은 하위 경로를 활성화하면 치료 효과가 제한될 수 있다.
- 따라서 치료 불응성 질환에서는 단일 표적 발현량보다 **경로 수준의 중복성과 신호 의존성**을 측정하는 바이오마커가 더 적합할 수 있다.
- 이 분석은 임상 실패를 단순한 “표적 실패”가 아니라, 환자군 선정·경로 중복·바이오마커 설계의 문제로 재해석한다.

---

### 6. 보충 그림 S1–S7

본문에서 확인되는 범위의 핵심은 다음과 같다.

- **Fig. S1**
  - Virtual Biotech 사용자 인터페이스를 보여준다.
  - 에이전트의 추론 과정, 사용 중인 도구, 생성된 코드·데이터·보고서의 확인 및 다운로드 기능을 제시한다.
  - 핵심 인사이트는 분석 과정의 **감사 가능성(auditability)**과 재현성이다.

- **Fig. S2**
  - 기존 연구에서 보고된 유전적 근거와 임상시험 진행의 관계를 재현한 결과다.
  - 유전적 근거가 있는 표적은 시험 진행 가능성이 높고 조기 중단 가능성이 낮았다.
  - 주요 평가변수 달성 및 낮은 이상반응과도 유전적 근거가 연관됐다.

- **Fig. S3**
  - 1,000회 순열검정 및 혼합효과모형을 이용한 강건성 분석이다.
  - 단일세포 특이성·이중성 지표와 임상 결과의 관계가 우연이나 일부 교란요인만으로 설명되기 어렵다는 점을 뒷받침한다.

- **Fig. S4**
  - 유전적 근거를 보정한 분석과 유전적 근거가 없는 시험만을 대상으로 한 분석이다.
  - 단일세포 기반 지표가 유전학적 정보와 별개의 추가 신호를 제공한다는 점을 보여준다.

- **Fig. S5**
  - B7-H3 분석의 추가 단일세포·공간전사체 결과다.
  - LUAD에서의 B7-H3 관련 세포 간 신호와 공간적 면역배제 분석을 보완한다.

- **Fig. S6–S7**
  - Virtual Biotech가 작성한 코드와 인간 전문가가 독립적으로 작성한 분석 결과를 비교한다.
  - 8개 분석 모두에서 전문가 결론이 에이전트 결과와 대체로 일치했고, 에이전트 코드도 결론을 크게 바꿀 오류가 없는 것으로 평가됐다.

---

### 7. 보충 테이블 S1–S4

- **Table S1**
  - 에이전트 구성, 역할, 연구 부서 및 사용 도구를 정리한 표로 보인다.
  - 시스템이 단일 범용 에이전트가 아니라 표적·안전성·약리·임상 등 역할별로 분화됐음을 보여준다.

- **Table S2**
  - Virtual Biotech와 Biomni, Kosmos, PantheonOS의 임상시험 주석 성능을 비교한다.
  - 본문에 따르면 Virtual Biotech가 수작업 검토 자료와 TDC 자료 모두에서 더 높은 정확도를 보였다.

- **Table S3**
  - B7-H3 사례에 대한 다른 agentic system과의 비교 결과다.
  - 다른 시스템은 더 적은 데이터 modality를 사용했고, 분석의 엄밀성과 생물학적 의미가 상대적으로 낮았다.

- **Table S4**
  - OSMR 사례에 대한 시스템 비교 결과다.
  - Virtual Biotech가 경로 중복성, 단일세포 기전, 바이오마커 검증을 더 종합적으로 연결했다.

**테이블의 종합적 인사이트**
- 비교 결과는 Virtual Biotech가 단순 질의응답보다 **다중 modality를 활용하는 구조화된 연구 workflow**에서 강점을 가진다는 점을 시사한다.
- 다만 비교 대상 시스템에 대한 인간의 steering 수준을 맞추었더라도, 완전히 동일한 조건의 독립적인 벤치마크라고 보기는 어렵다.

---

### 8. Supplementary Text / Appendix의 핵심

- **Supplementary Text I**: MCP 서버와 데이터·도구 연결 구조
- **Supplementary Text II**: CSO의 질문 분해, 에이전트 배정, 반복 분석 workflow
- **Supplementary Text III**: 과학 검토 에이전트의 품질관리 방식
- **Supplementary Text IV**: 임상시험 결과 추출을 위한 근거 우선순위와 주석 규칙
- **Supplementary Text V**: B7-H3 사례의 전체 대화 기록
- **Supplementary Text VI**: OSMR 사례의 전체 대화 기록
- **Supplementary Text VII**: 인간 전문가 검토의 구체적 기록
- **Supplementary Text VIII**: 다른 agentic system과의 비교 분석
- **Supplementary Text IX**: 재현성, 인간 감독, 데이터 편향, 바이오보안, 한계와 향후 방향
- **Supplementary Text X**: 각 AI 에이전트의 system prompt

**부록·방법론에서 얻는 핵심 인사이트**
- Virtual Biotech는 자율 시스템이지만, 사용자 질문 설정과 후속 방향 제시가 포함된 **human-guided autonomy**에 가깝다.
- 분석 결과는 생성된 코드와 중간 산출물을 확인할 수 있어 비교적 투명하지만, 데이터베이스 편향과 언어모델의 기존 지식 편향에서 자유롭지는 않다.
- 실험 수행, 분자 설계, 독성 검증 등은 현재 범위 밖이므로, 제시된 결과는 치료 결론이라기보다 **검증할 가치가 있는 연구 가설**이다.

---




The following summarizes the main findings and insights from the paper’s **diagrams, figures, tables, and supplementary materials**.  
The supplementary files themselves were not fully provided, so the summary is limited to what is explicitly described in the main text and figure legends.

### 1. Fig. 1 — Overall architecture of the Virtual Biotech

**Findings**
- The Virtual Biotech is a hierarchical multi-agent system coordinated by a virtual Chief Scientific Officer, or CSO.
- It contains divisions for:
  - Target identification and prioritization
  - Target safety
  - Modality selection
  - Clinical development
- The CSO decomposes a user query into subproblems and delegates them to specialized scientist agents.
- A scientific reviewer evaluates the evidence, identifies gaps or unsupported claims, and can trigger additional analyses.
- MCP servers connect the agents to Open Targets, CELLxGENE, ClinicalTrials.gov, cBioPortal, FDA safety data, and other resources.

**Insight**
- The main contribution is not simply using multiple AI agents in parallel. It is the workflow of **specialization, independent analysis, review, and evidence integration**.
- The platform is best understood as a human-guided decision-support system rather than a fully autonomous replacement for drug-development teams.

---

### 2. Fig. 2 — Large-scale clinical-trial annotation

**Findings**
- The system analyzed **55,984 clinical trials**.
- It assigned 37,075 agents to Phase II and III trials, with one agent per trial.
- Each agent used a three-tier evidence cascade:
  1. ClinicalTrials.gov
  2. PubMed
  3. Press releases and regulatory announcements
- Agents extracted trial progression, primary and secondary endpoints, adverse-event rates, and termination reasons into a standardized JSON format.
- The 37,075 trials were processed in approximately six hours, corresponding to an estimated **184-fold speedup** over a single-agent workflow.
- Manual review showed agreement rates of 88.4% for primary endpoints, 88.4% for secondary endpoints, and 92.4% for adverse-event rates.

**Insight**
- Multi-agent parallelization is particularly useful for large-scale evidence curation.
- Assigning one agent to one trial reduces context dilution and allows detailed review of the trial record.
- However, converting complex trial outcomes into binary success/failure labels remains difficult in multi-arm, borderline, or partially successful studies.

---

### 3. Fig. 3 — Single-cell target features and clinical success

**Findings**
- Two single-cell features were extracted:
  - **Tau cell-type specificity**, measuring whether a gene is restricted to particular cell types
  - **Bimodality coefficient**, measuring on/off-like heterogeneity within a cell population
- Drugs targeting cell-type-specific genes were:
  - 48% more likely to reach Phase IV
  - 40% more likely to progress from Phase I to Phase II
  - More likely to achieve primary and secondary endpoints
- These drugs also showed, on average, **32% lower adverse-event rates**.
- The associations remained significant after adjusting for trial phase, enrollment year, disease area, and drug modality.
- The effects also persisted after accounting for genetic evidence and among trials without supporting genetic evidence.

**Insight**
- Restricted expression may allow therapeutic activity in relevant cells while reducing exposure to unrelated tissues.
- Single-cell information appears to provide additional information beyond human genetic evidence.
- These are observational associations, however, and should not be interpreted as proof that cell-type specificity directly causes clinical success.

---

### 4. Fig. 4 — B7-H3 targeting in lung cancer

**Findings**
- The Virtual Biotech evaluated B7-H3/CD276 in LUAD and SCLC using genetic, single-cell, spatial, survival, and pharmacology data.
- Although germline genetic support was weak, the system did not treat this as disqualifying.
- B7-H3 up-regulation was concentrated in tumor-microenvironment fibroblasts rather than being restricted to malignant cells.
- B7-H3-high fibroblasts showed signaling patterns related to immune suppression, macrophage reprogramming, angiogenesis, and immune-cell retention.
- Spatial transcriptomics showed reduced local abundance of T cells, macrophages, monocytes, and dendritic cells around B7-H3-high regions.
- High B7-H3 expression was associated with worse survival:
  - Overall survival: HR 1.62
  - Disease-free survival: HR 2.06
- The system nominated an **antibody-drug conjugate (ADC)** as the preferred modality.

**Insight**
- The analysis reframed B7-H3 as more than a cancer-cell surface target, highlighting a possible role for B7-H3-high cancer-associated fibroblasts in immune exclusion.
- ADCs were favored because they could combine direct cytotoxic delivery, tumor-microenvironment effects, and bystander killing.
- This is a hypothesis-generating preclinical rationale, not evidence of clinical efficacy.

---

### 5. Fig. 5 — Analysis of the failed OSMRβ ulcerative-colitis trial

**Findings**
- The analysis focused on the Phase II MOONGLOW trial of vixarelimab in moderate-to-severe ulcerative colitis.
- The trial was stopped for futility rather than safety concerns.
- OSMR was up-regulated in fibroblast states from treatment-refractory patients, supporting the original biological rationale.
- However, STAT1 activation in OSMR-expressing cells was not primarily explained by OSMR alone.
- Multiple gp130-family receptors appeared to contribute to the same downstream JAK–STAT pathway.
- Shapley-based variance decomposition showed that OSMR was not the dominant contributor to STAT1 activity in most relevant stromal cell types.
- A 10-gene **gp130-axis score** outperformed OSMR expression alone in predicting treatment non-response across four independent UC cohorts.

**Insight**
- The failure may not indicate that OSMR is irrelevant. Instead, OSMR may be only one component of a redundant gp130–JAK–STAT network.
- Blocking one receptor may be insufficient when parallel receptors converge on the same downstream pathway.
- Pathway-level biomarkers may therefore be more informative than single-target expression in treatment-refractory disease.

---

### 6. Supplementary Figures S1–S7

- **Fig. S1:** Shows the user interface, including agent reasoning, tool usage, downloadable code, data, and reports. Its main contribution is transparency and auditability.
- **Fig. S2:** Replicates the association between human genetic evidence and clinical progression, lower early stoppage, endpoint success, and fewer adverse events.
- **Fig. S3:** Permutation tests and mixed-effects analyses support the robustness of the single-cell feature associations.
- **Fig. S4:** Shows that single-cell features retain predictive associations after adjustment for genetic evidence and in trials without genetic support.
- **Fig. S5:** Provides additional B7-H3 single-cell and spatial-transcriptomic analyses, including LUAD-specific signaling and immune-exclusion results.
- **Figs. S6–S7:** Compare Virtual Biotech analyses with independent expert implementations. Experts generally reproduced all eight evaluated analyses and found no major code errors affecting conclusions.

---

### 7. Supplementary Tables S1–S4

- **Table S1:** Summarizes the agent organization, roles, divisions, and tool access.
- **Table S2:** Compares clinical-trial annotation performance with Biomni, Kosmos, and PantheonOS. The Virtual Biotech reportedly achieved higher agreement with manual review and TDC labels.
- **Table S3:** Compares systems on the B7-H3 case. Other systems used fewer modalities and produced less rigorous or biologically informative analyses.
- **Table S4:** Compares systems on the OSMR case. The Virtual Biotech more extensively connected pathway redundancy, single-cell mechanisms, and biomarker validation.

**Overall insight**
- The tables support the claim that Virtual Biotech is strongest when used as a structured, multimodal research workflow rather than as a general-purpose chatbot.
- These comparisons should still be interpreted cautiously because system steering and analysis conditions may not be perfectly identical.

---

### 8. Supplementary Text and appendix materials

- **Supplementary Text I:** MCP servers and data/tool integration
- **Supplementary Text II:** CSO task decomposition and orchestration
- **Supplementary Text III:** Scientific review and quality control
- **Supplementary Text IV:** Clinical-trial evidence hierarchy and annotation protocol
- **Supplementary Texts V–VI:** Full B7-H3 and OSMR case-study transcripts
- **Supplementary Text VII:** Human expert review
- **Supplementary Text VIII:** Comparison with other agentic systems
- **Supplementary Text IX:** Reproducibility, human oversight, limitations, bias, and biosecurity
- **Supplementary Text X:** System prompts for the AI agents

**Overall appendix insight**
- The system operates under **human-guided autonomy**: humans define intent and provide high-level steering, while the agents perform most downstream analyses.
- The workflow is relatively transparent because code, intermediate data, and reports can be inspected.
- Nevertheless, the results remain dependent on database coverage, annotation quality, model bias, and the assumptions built into each analysis.
- Experimental testing, molecule design, and toxicity validation remain outside the current system. Thus, the outputs should be treated as **testable research hypotheses rather than final therapeutic decisions**.

<br/>
# refer format:


```bibtex
@article{Zhang2026VirtualBiotech,
  author  = {Zhang, Harrison G. and Eckmann, Peter and Miao, Jiacheng and Mahon, Andrew B. and Zou, James},
  title   = {The Virtual Biotech: A Multi-Agent AI Framework for Therapeutic Discovery and Development},
  journal = {Science},
  year    = {2026},
  month   = sep,
  doi     = {10.1126/science.aeg6779},
  url     = {https://doi.org/10.1126/science.aeg6779},
  note    = {Published online September 17, 2026; Science Ahead of Print}
}
```

### 시카고 스타일

Zhang, Harrison G., Peter Eckmann, Jiacheng Miao, Andrew B. Mahon, and James Zou. “The Virtual Biotech: A Multi-Agent AI Framework for Therapeutic Discovery and Development.” *Science*, published online September 17, 2026. https://doi.org/10.1126/science.aeg6779.



