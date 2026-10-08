---
layout: post
title:  "[2026]Reimagining Research Papers as Interactive and Reliable AI Agents"
date:   2026-10-07 18:17:05 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: Paper2Agent는 논문·코드·데이터를 분석해 MCP 서버와 검증된 도구로 변환하고, 이를 자연어 기반 AI 에이전트와 연결하는 다중 에이전트 프레임워크다(복잡한 분석을 프로그래밍하지 않고 자연어).   


짧은 요약(Abstract) :



이 논문은 **Paper2Agent**라는 시스템을 제안합니다. Paper2Agent는 기존의 연구 논문을 단순히 읽는 문서가 아니라, 해당 논문의 지식과 분석 방법을 직접 실행할 수 있는 **AI 에이전트**로 바꾸는 프레임워크입니다.

기존에는 논문에 제시된 코드를 활용하려면 사용자가 직접 코드 저장소를 찾고, 프로그램 환경과 의존성을 설치하며, 입력·출력 형식을 이해해야 했습니다. Paper2Agent는 논문과 관련 코드, 데이터, 보충자료를 분석해 **MCP(Model Context Protocol) 서버**를 만들고, 이를 대화형 AI와 연결합니다. 따라서 사용자는 복잡한 분석을 프로그래밍하지 않고 자연어로 요청할 수 있습니다.

이 시스템은 여러 AI 에이전트를 이용해 다음 과정을 자동화합니다.

- 논문과 관련 코드 저장소 분석
- 실행 환경 구성
- 논문의 핵심 방법을 재사용 가능한 도구로 변환
- 원 논문의 결과와 비교하는 자동 테스트
- 검증된 도구를 MCP 서버로 배포

논문에서는 AlphaGenome, Scanpy, TISSUE 등을 AI 에이전트로 변환해 사례 연구를 수행했습니다. 예를 들어 유전 변이의 기능을 분석하거나, 단일세포 데이터를 전처리하고 군집화하는 작업을 자연어로 요청할 수 있었습니다. 생성된 에이전트는 원 논문의 결과를 재현했으며, 새로운 데이터와 질문에도 적용되었습니다.

또한 여러 논문 에이전트가 서로 협력해 건선과 관련된 유전 변이의 원인 유전자를 찾는 실험도 수행했습니다. 이 과정에서 **GPR137**이 유력한 원인 유전자로 제안되었고, 독립적인 실험 데이터와의 비교를 통해 그 가능성이 뒷받침되었습니다.

핵심적으로 이 논문은 연구 결과를 **정적인 문서에서 실행 가능하고 대화할 수 있는 연구 도구**로 바꾸자는 새로운 과학 커뮤니케이션 방식을 제시합니다. 다만 모든 논문이 자동으로 안정적인 에이전트가 되는 것은 아니며, 최종적인 과학적 해석과 가설 검증에는 여전히 연구자의 판단이 필요하다고 강조합니다.

---



This paper introduces **Paper2Agent**, an automated framework that converts research papers into interactive AI agents. Instead of treating a paper as a passive document, Paper2Agent turns its methods, code, data, supplementary materials and workflows into executable, conversational tools.

Normally, using a computational method from a paper requires users to find the code repository, install dependencies, configure the software environment and understand the required inputs and outputs. Paper2Agent reduces these barriers by analyzing the paper and its codebase, building a **Model Context Protocol (MCP) server**, and connecting it to an AI agent. Users can then request scientific analyses in natural language.

The framework automatically:

- identifies and configures the relevant codebase;
- extracts the main analytical methods as reusable tools;
- tests the tools against the original paper’s results;
- refines or removes tools that fail validation; and
- deploys the validated tools through an MCP server.

The authors demonstrate the framework using AlphaGenome, Scanpy and TISSUE. These agents can interpret genetic variants, analyze single-cell data and reproduce analyses from the original papers. They can also apply the methods to new datasets and answer novel scientific questions.

In another case study, multiple paper agents collaborated to prioritize a causal gene for a psoriasis-associated genetic variant. The system identified **GPR137** as a likely causal gene and supported this prediction by comparing computational results with independent CRISPR and Perturb-seq data.

Overall, the paper proposes a new model of scientific communication in which research papers become **interactive, executable and collaborative AI systems** rather than static publications. However, the authors emphasize that researchers must remain involved in choosing research directions, evaluating evidence and interpreting open-ended scientific conclusions.


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



### 1. 방법의 핵심 개념

**Paper2Agent**는 연구 논문과 연결된 코드·데이터·분석 절차를 하나의 **대화형 AI 에이전트**로 변환하는 프레임워크입니다.  
새로운 거대언어모델(LLM)을 처음부터 학습하는 방식이 아니라, 기존 LLM이 논문의 지식과 코드를 **실제로 실행할 수 있도록 구조화하고 연결**합니다.

최종적으로 사용자는 다음과 같이 자연어로 요청할 수 있습니다.

> “이 논문의 방법을 내 데이터에 적용해줘.”  
> “이 변이의 유전자 발현 및 크로마틴 접근성 효과를 분석해줘.”

---

### 2. 사용한 모델과 전체 아키텍처

#### 핵심 모델

- 주된 AI 코딩 에이전트: **Claude Code**
- 사례 연구에서 질의에 사용한 LLM: **Claude Sonnet 4**
- Claude Code는 코드 작성, 실행, 오류 수정, 파일 검색 및 분석을 수행하는 에이전트 역할을 합니다.
- Paper2Agent 자체가 별도의 새로운 신경망 모델을 학습하는 것은 아닙니다.

#### 전체 구성

Paper2Agent는 크게 두 층으로 구성됩니다.

1. **Paper2MCP**
   - 논문, 코드 저장소, 보충자료, 데이터 및 분석 절차를 분석합니다.
   - 이를 **MCP(Model Context Protocol) 서버**로 변환합니다.

2. **에이전트 계층**
   - MCP 서버를 LLM에 연결합니다.
   - LLM은 MCP에 등록된 도구와 자료를 사용하여 자연어 질의에 답하거나 분석을 실행합니다.

즉, 구조는 다음과 같습니다.

> 논문·코드·데이터  
> → MCP 도구·자료·프롬프트  
> → LLM 에이전트  
> → 자연어 기반 분석 및 결과 보고

---

### 3. MCP 서버의 구성 요소

각 논문용 MCP 서버는 세 가지 핵심 요소를 포함합니다.

#### 1) MCP tools

논문의 주요 분석 기능을 실행 가능한 함수로 만듭니다.

예시:

- 유전 변이의 기능적 효과 예측
- 단일세포 데이터 품질관리
- 클러스터링 및 시각화
- 통계 분석 및 결과 파일 생성

AlphaGenome 사례에서는 `score_variant()`나 `visualize_variant_effects()`와 같은 도구가 생성되었습니다.

#### 2) MCP resources

논문과 관련된 정적 자료를 구조화하여 제공합니다.

- 논문 본문
- 보충자료
- 원래의 코드 저장소
- 데이터셋
- 표와 그림
- 모델 또는 학습 데이터에 대한 링크

#### 3) MCP prompts

복잡한 분석의 실행 순서를 안내하는 작업 지침입니다.

예를 들어 Scanpy에서는 다음과 같은 순서를 MCP 프롬프트로 표현합니다.

> 품질관리 → 정규화 → 고변이 유전자 선택 → 차원축소 → 이웃 그래프 생성 → 클러스터링 → 세포 유형 주석

따라서 사용자가 모든 명령을 직접 순서대로 작성하지 않아도 됩니다.

---

### 4. Paper2Agent의 자동 변환 파이프라인

Paper2Agent는 논문과 코드베이스를 다음의 6단계로 변환합니다.

#### 1단계: 코드베이스 탐색 및 다운로드

- 논문, 참고문헌, 보충자료에서 관련 코드 저장소를 찾습니다.
- 필요한 경우 사용자가 저장소 URL을 직접 지정할 수 있습니다.
- 코드와 설정 파일, 보충 데이터 등을 가져옵니다.

#### 2단계: 실행 환경 구축

- 저장소에 필요한 패키지와 버전을 분석합니다.
- 격리된 가상환경을 만들고 의존성을 설치합니다.
- 환경 충돌을 줄여 재현 가능한 실행을 목표로 합니다.

#### 3단계: 튜토리얼 탐색

- README, 노트북, 예제 코드 등에서 실행 가능한 튜토리얼을 찾습니다.
- 각 튜토리얼의 용도와 분석 절차를 정리합니다.

#### 4단계: 튜토리얼 실행 및 감사

- 튜토리얼을 실제로 실행합니다.
- 입력값, 출력 파일, 그림, 수치 결과 및 실행 조건을 기록합니다.
- 코드에 명시되지 않은 가정이나 오류도 확인합니다.

#### 5단계: 도구 추출 및 검증

- 튜토리얼의 분석 단계를 재사용 가능한 Python 함수로 변환합니다.
- 파일 경로, 임계값, 열 이름 등 하드코딩된 값을 입력 인자로 바꿉니다.
- 각 함수를 MCP 도구로 등록합니다.
- 예제 데이터로 원래 결과와 비교하면서 테스트하고 수정합니다.

#### 6단계: MCP 서버 조립

- 검증된 도구를 하나의 MCP Python 서버로 통합합니다.
- 서버를 Hugging Face Spaces와 같은 원격 플랫폼에 배포할 수 있습니다.
- 이후 Claude Code 같은 LLM 에이전트와 연결합니다.

---

### 5. 다중 에이전트 구조

이 과정은 하나의 LLM이 모든 일을 수행하는 방식이 아니라, 역할별 하위 에이전트가 협력하는 구조입니다.

- **Environment manager**: 실행 환경과 의존성 설정
- **Tutorial scanner**: 튜토리얼 및 예제 탐색
- **Tutorial executor**: 튜토리얼 실행과 결과 수집
- **Tool extractor–implementor**: 재사용 가능한 MCP 함수 생성
- **Test verifier–improver**: 테스트 생성, 오류 진단 및 수정
- **Orchestrator**: 전체 작업을 조정하는 중앙 에이전트

여러 작업은 병렬로 처리할 수 있으며, 각 단계의 결과는 JSON 보고서와 파일을 통해 다음 단계로 전달됩니다.

---

### 6. 신뢰성 및 재현성을 높이는 특별한 기법

Paper2Agent의 중요한 특징은 LLM이 임의로 코드를 생성하게 두지 않고, **원래 논문의 코드와 결과를 기준으로 생성된 도구를 검증한다는 점**입니다.

#### 자동 검증 기준

각 도구에 대해 다음을 확인합니다.

- 예상 출력 파일이 생성되는가
- 수치 결과가 원래 결과와 일치하는가
- 부동소수점 결과가 허용 오차 이내인가  
  - 방법 부분에서는 일반적으로 **3% 허용오차** 사용
- 생성된 그림이 원래 그림과 유사한가
  - perceptual hashing 사용
  - Hamming distance 기준 적용
- 반복적으로 실패하는 도구는 최종 MCP에서 제외

또한 각 도구에는 원 논문의 코드 위치에 대한 참조를 포함하여 **추적 가능성(traceability)**을 확보합니다.

#### 코드 환각 방지

일반적인 LLM 에이전트는 존재하지 않는 함수나 부정확한 코드를 생성할 수 있습니다. Paper2Agent는 먼저 검증된 도구를 만들고, 이후 에이전트가 해당 도구를 호출하도록 하여 이러한 **코드 환각**을 줄입니다.

---

### 7. 학습 데이터와 모델 학습 여부

Paper2Agent는 새로운 기초 모델을 학습시키는 프레임워크가 아닙니다.

- 별도의 대규모 사전학습 데이터셋으로 LLM을 재훈련하지 않음
- 논문의 코드 저장소와 튜토리얼을 분석하여 실행 가능한 도구 생성
- 원 논문의 데이터, 보충자료, 예제 데이터 및 공개 데이터셋을 활용
- 도구 검증에는 주로 원래 튜토리얼의 예제 데이터와 기준 결과를 사용
- AlphaGenome 자체는 유전체 서열 및 다양한 조절 기능 데이터를 학습한 기존 모델이며, Paper2Agent는 이를 MCP 도구로 감쌈
- Scanpy와 TISSUE도 기존 소프트웨어·분석 방법을 에이전트가 사용할 수 있도록 변환함

따라서 Paper2Agent의 학습보다는 **코드 분석, 도구 추출, 실행 검증 및 LLM-도구 연결**이 핵심입니다.

---

### 8. 사례 연구에서 사용된 분석 방법

#### AlphaGenome 에이전트

- 유전 변이의 유전자 발현, 스플라이싱, 크로마틴 접근성 등을 예측
- 단일 변이 및 여러 변이 분석
- 조직과 세포 유형별 결과 비교
- 예측 결과 시각화
- GWAS 후보 변이와 원인 유전자 우선순위화

#### Scanpy 에이전트

- 단일세포 RNA-seq 품질관리
- 세포 및 유전자 필터링
- 정규화
- 고변이 유전자 선택
- PCA 및 이웃 그래프 생성
- Leiden 클러스터링
- 마커 유전자 기반 세포 유형 주석

#### 여러 논문 에이전트의 협업

AlphaGenome, scCRISPRi, Perturb-seq 에이전트를 연결하여 건선 관련 변이의 원인 유전자를 분석했습니다.

- AlphaGenome: 후보 유전자의 변이 효과 예측
- scCRISPRi: 조절 영역 perturbation 이후 유전자 발현 변화 측정
- Perturb-seq: 특정 유전자 knockdown 이후 전사체 변화 측정
- 두 종류의 유전자 발현 signature를 상관분석하여 후보 유전자 검증

이 방식으로 **GPR137**이 rs887314와 관련된 가능한 원인 유전자로 우선순위화되었습니다. 다만 연구진은 이런 열린 생물학적 해석은 여전히 인간 연구자의 검토가 필요하다고 강조합니다.

---

### 9. 한계

- 코드 저장소가 불완전하거나 데이터·모델 파일이 없으면 에이전트화가 실패할 수 있음
- 의존성 충돌, 오래된 API, 문서 부족 등이 장애가 됨
- 모든 논문이 실행 가능한 에이전트로 변환되는 것은 아님
- 자연어 기반의 새로운 생물학적 해석은 단일 정답이 없을 수 있음
- 가설 생성과 기전 해석은 여전히 human-in-the-loop 방식이 필요함
- 코드 실행을 원격으로 노출할 때 보안, 지식재산권, 데이터 접근 문제가 발생할 수 있음

---





### 1. Core idea

**Paper2Agent** is a framework that converts a research paper, its code, data, and workflows into an interactive AI agent.

It does not train a new large language model from scratch. Instead, it uses an existing LLM to access validated, executable tools derived from the paper.

A user can interact with the resulting agent in natural language, for example:

> “Apply this paper’s method to my dataset.”  
> “Interpret the regulatory effect of this genetic variant.”

---

### 2. Models and overall architecture

#### Main models

- Primary coding-agent framework: **Claude Code**
- Main downstream LLM used in the case studies: **Claude Sonnet 4**
- Claude Code performs repository inspection, coding, execution, debugging, and analysis.

The system has two main layers:

1. **Paper2MCP**
   - Converts the paper, codebase, supplementary materials, and datasets into an MCP server.
2. **Agent layer**
   - Connects the MCP server to an LLM-based agent.
   - The agent uses the exposed tools and resources to answer questions and run analyses.

The overall flow is:

> Paper, code, and data  
> → MCP tools, resources, and prompts  
> → LLM agent  
> → Natural-language analysis and reporting

---

### 3. MCP server components

Each paper-specific MCP server contains three main components.

#### MCP tools

Executable functions representing the paper’s main methods.

Examples include:

- Variant-effect prediction
- Single-cell quality control
- Clustering
- Visualization
- Statistical analysis
- Result-file generation

#### MCP resources

Structured access to static research materials:

- Manuscript text
- Supplementary information
- Code repository
- Datasets
- Tables and figures
- Links to model or training data

#### MCP prompts

Workflow instructions that specify how multiple tools should be executed.

For example, the Scanpy workflow is represented as:

> Quality control → normalization → highly variable gene selection → dimensionality reduction → graph construction → clustering → cell-type annotation

---

### 4. Six-step agentification pipeline

Paper2Agent converts a paper into an agent through six main steps.

1. **Locate and download the codebase**
   - Finds the associated repository from the paper or accepts a user-provided URL.

2. **Set up the environment**
   - Creates an isolated environment and installs the required dependencies.

3. **Discover tutorials**
   - Identifies notebooks, README examples, and executable workflows.

4. **Execute and audit tutorials**
   - Runs the tutorials and records inputs, outputs, figures, numerical results, and hidden assumptions.

5. **Extract and validate tools**
   - Converts reusable analysis steps into parameterized Python functions.
   - Generates MCP tools and tests them using the tutorial data.

6. **Assemble the MCP server**
   - Combines validated tools into a deployable MCP server, which can be hosted remotely and connected to an LLM agent.

---

### 5. Multi-agent architecture

Paper2Agent uses specialized sub-agents rather than one monolithic agent.

- **Environment manager**: configures software environments
- **Tutorial scanner**: identifies useful tutorials
- **Tutorial executor**: runs tutorials and records reference outputs
- **Tool extractor–implementor**: converts workflows into reusable functions
- **Test verifier–improver**: creates tests, diagnoses failures, and refines tools
- **Orchestrator**: coordinates the full pipeline

The sub-agents communicate through standardized JSON reports and file-based outputs. Some independent tasks can be executed in parallel.

---

### 6. Reliability and reproducibility techniques

A central design principle is that the LLM should not freely invent scientific code during every user query. Instead, Paper2Agent first creates and validates executable tools based on the original codebase.

Validation checks include:

- Whether expected output files are produced
- Whether numerical results match the reference outputs
- Whether floating-point values remain within tolerance  
  - The Methods section reports a general **3% tolerance**
- Whether generated figures resemble the reference figures
  - Perceptual hashing is used
- Whether repeatedly failing tools should be removed from the final MCP server

Each tool also contains a reference to the original source code, improving transparency and traceability.

This design reduces **code hallucination**, in which an LLM generates nonexistent or scientifically incorrect functions.

---

### 7. Training data and model training

Paper2Agent is not primarily a model-training method.

- It does not retrain a foundation model on a new large-scale dataset.
- It analyzes public repositories, tutorials, supplementary materials, and example datasets.
- Tutorial outputs serve as ground truth for tool validation.
- Existing models such as AlphaGenome remain responsible for their original predictive modeling.
- Paper2Agent wraps those models and methods as MCP tools that an LLM agent can invoke.

Therefore, the main technical contribution is not new neural-network training, but rather:

> repository analysis + tool extraction + automated execution testing + MCP-based LLM integration

---

### 8. Example applications

#### AlphaGenome agent

- Predicts variant effects on gene expression, splicing, and chromatin accessibility
- Compares effects across tissues and cell types
- Generates visualizations
- Prioritizes candidate causal genes at GWAS loci

#### Scanpy agent

- Performs single-cell RNA-seq quality control
- Filters cells and genes
- Normalizes data
- Selects highly variable genes
- Performs PCA and graph construction
- Runs Leiden clustering
- Annotates cell types using marker genes

#### Collaboration among paper agents

AlphaGenome, scCRISPRi, and Perturb-seq agents were combined to study a psoriasis-associated variant.

The agents integrated:

- Computational variant-effect predictions
- Regulatory-element perturbation signatures
- Gene-knockdown expression signatures

Correlation between these signatures supported **GPR137** as a likely causal gene at the rs887314 locus, although the authors emphasized that open-ended biological interpretation still requires human supervision.

---

### 9. Limitations

- Agentification may fail when repositories lack executable code, data, documentation, or stable dependencies.
- Dependency conflicts and deprecated APIs can prevent execution.
- Not every paper can be converted into a reliable executable agent.
- Open-ended biological reasoning may have multiple defensible answers.
- Hypothesis generation and mechanistic interpretation remain human-in-the-loop tasks.
- Remote execution introduces security, intellectual-property, and data-access concerns.


<br/>
# Results




### 1. AlphaGenome 에이전트 성능 비교

Paper2Agent로 AlphaGenome 논문의 기능을 MCP 도구와 AI 에이전트로 변환한 뒤, 다음 세 가지 시스템을 비교했다.

- **Paper2Agent 기반 AlphaGenome agent**
- **Claude + Repo**: Claude가 AlphaGenome 논문과 원본 코드 저장소에 직접 접근
- **Biomni**: 생명과학용 범용 AI 에이전트

#### 평가 데이터와 과제

총 세 종류의 질의를 사용했다.

1. **튜토리얼 기반 질의**
   - 원 논문의 튜토리얼에서 파생한 변이 점수 계산 문제
   - 예: 특정 염기 변이를 특정 조직의 ATAC-seq 예측으로 분석하고 `quantile_score` 산출
   - 15개 질의, 5회 반복

2. **새로운 질의**
   - 튜토리얼에 직접 포함되지 않은 변이·조직·분석 조합
   - 15개 질의, 5회 반복

3. **개방형 연구자 질의**
   - 여러 도구를 연속적으로 사용하고 생물학적 해석까지 요구
   - 예: GWAS 변이의 조절 효과를 분석하고 가능한 인과 유전자와 작용기전 제시
   - 30개 질의

#### 정확도 결과

| 평가 유형 | Paper2Agent | Claude + Repo | Biomni |
|---|---:|---:|---:|
| 튜토리얼 기반 | **98.7 ± 1.3%** | 82.7 ± 3.4% | 37.3 ± 4.0% |
| 새로운 질의 | **100.0 ± 0.0%** | 78.7 ± 4.4% | 56.0 ± 3.4% |
| 개방형 연구자 질의 | **82.7 ± 2.4%** | 56.7 ± 2.3% | 72.2 ± 2.2% |

Paper2Agent는 정형화된 실행 과제뿐 아니라, 여러 도구를 조합해야 하는 개방형 분석에서도 Claude + Repo보다 높은 성능을 보였다. 다만 개방형 질의에서는 단일한 정답이 항상 존재하지 않으므로, 이 수치는 생물학적 결론의 절대적 타당성보다는 **분석 절차를 정확히 수행하고 핵심 결과를 재현했는지**를 주로 반영한다.

#### 실행 시간

튜토리얼 기반 질의에서 Paper2Agent는 중앙 실행 시간이 다음과 같이 짧았다.

- Claude + Repo보다 **1.9배 빠름**
- Biomni보다 **3.1배 빠름**

새로운 질의에서는 다음과 같았다.

- Claude + Repo보다 **2.9배 빠름**
- Biomni보다 **3.8배 빠름**

이는 Paper2Agent가 이미 검증된 도구를 호출하기 때문에, 매번 코드를 새로 작성하고 디버깅하는 방식보다 효율적이기 때문이다.

---

### 2. AlphaGenome 도구의 재현성과 검증

Paper2Agent는 AlphaGenome에 대해 **22개의 MCP 도구**를 생성했으며, 모두 자동 검증을 통과했다.

검증 기준은 다음과 같았다.

- 예상 파일이 생성되는가
- 수치 결과가 원본 결과와 허용 오차 내에서 일치하는가
- 생성된 그림이 원본 그림과 유사한가
- 원 논문의 코드와 결과를 추적할 수 있는가

이 도구들은 변이 효과 점수 계산, 조직·세포 유형별 분석, 유전자 발현·염색질 접근성·스플라이싱 예측, 시각화 등을 지원한다.

---

### 3. Scanpy 단일세포 분석 결과

Scanpy 논문과 코드를 기반으로 **7개의 도구**를 생성했으며, 모두 자동 검증을 통과했다.

#### 평가 방식

사용자가 데이터 경로만 입력하면 에이전트가 다음 과정을 순서대로 수행하도록 했다.

1. 품질관리 및 필터링
2. 정규화
3. 고변이 유전자 선택
4. 차원 축소
5. 이웃 그래프 구성
6. 클러스터링
7. 세포 유형 주석

#### 테스트 데이터와 결과

- Scanpy 코드에 포함되지 않은 공개 단일세포 데이터 **4개**로 재현성 평가
- 추가적으로 서로 다른 특성을 가진 **7개 데이터셋**에서 일반화 성능 평가
- 인간 연구자가 동일한 데이터를 처리한 결과와 비교

결과적으로 에이전트는 인간 연구자의 분석과 비교해 다음을 동일하게 재현했다.

- 품질관리 후 세포 수와 유전자 수
- 클러스터 구조
- 클러스터별 주요 차등 발현 마커 유전자
- 데이터 특성에 따른 분석 파라미터 조정

즉, Scanpy 에이전트는 단순히 개별 함수를 실행한 것이 아니라, **분석 단계의 순서와 데이터 특성에 따른 조정까지 포함한 전체 분석 워크플로**를 재현했다.

TISSUE 에이전트 역시 동일한 공간전사체 데이터에 대해 인간 연구자가 수행한 결과와 일치하는 분석 결과를 생성했다.

---

### 4. 대규모 평가 결과

Paper2Agent의 확장성을 평가하기 위해 세 종류의 논문 집합을 사용했다.

#### 4.1 계산생물학 논문 100편

- 성공적으로 에이전트화된 논문: **74편**
- 제안된 도구: **599개**
- 자동 검증 통과 도구: **593개**

실패 원인은 주로 다음과 같았다.

- 실행 가능한 코드 부족
- 데이터 또는 모델 파일 누락
- 의존성·환경 설정 문제
- 특정 데이터에만 작동하는 비일반화 코드

총 **300개의 튜토리얼 기반 질문**으로 평가한 결과:

| 시스템 | 정확도 |
|---|---:|
| Paper2Agent + Sonnet 4 | **91.2 ± 1.6%** |
| Claude + Repo, Sonnet 4 | 80.3 ± 2.3% |
| Claude + Repo, Sonnet 4.6 | 86.3 ± 1.1% |

Paper2Agent는 직접 코드 저장소를 탐색하는 Claude보다 높은 정확도를 보였고, 차이는 통계적으로 유의했다.

또한 질의당 비용과 시간이 감소했다.

- Paper2Agent: **약 0.20달러, 1.6분**
- Claude + Repo: **약 0.38달러, 4.3분**

#### 4.2 비생물학 계산 논문 10편

AI, 통계, 계량경제학, 게임이론, 천체물리학 등 다양한 분야의 논문을 평가했다.

- 실행 기반 과제: **42개**
- 정확도: **98.1 ± 0.8%**

이는 Paper2Agent가 생명과학 분야에만 제한되지 않고, 다양한 계산 연구 코드에도 적용될 수 있음을 보여준다.

#### 4.3 데이터·발견 중심 논문 26편

실행 가능한 도구를 만들기 어려운 논문에 대해서는 논문 본문, 보충자료, 표와 메타데이터를 구조화된 리소스로 제공했다.

- 종합적 질문: **100개**
- Paper2Agent 리소스 계층: **89.0 ± 3.1%**
- 논문을 직접 탐색하는 Claude browser-use 방식: **82.0 ± 3.8%**

Paper2Agent 방식은 비교 기준보다 **약 34배 저렴하고 15배 빠른** 것으로 보고됐다.

---

### 5. 여러 논문 에이전트의 협업 결과

Paper2Agent는 서로 다른 논문에서 생성된 에이전트들을 연결해 새로운 분석을 수행할 수 있는지도 평가했다.

#### psoriasis 관련 rs887314 변이

세 에이전트를 결합했다.

- AlphaGenome: 변이가 유전자 발현에 미치는 영향 예측
- MPRA-coupled scCRISPRi: 조절요소 perturbation에 따른 유전자 발현 변화
- CD4+ T세포 Perturb-seq: 개별 유전자 knockdown에 따른 발현 변화

AlphaGenome은 psoriasis 관련 변이 **rs887314**의 후보 유전자로 **GPR137**을 우선순위화했다.

- CD4+ T세포 RNA-seq 예측 quantile score: **0.997**

이후 조절요소 perturbation으로 발생한 발현 변화와 유전자 knockdown 결과를 비교했다.

- GPR137 knockdown과의 상관:
  - Stim8hr: Spearman ρ = **0.613**, P = **3.79 × 10⁻³**
  - Stim48hr: Spearman ρ = **0.630**, P = **4.71 × 10⁻³**
- Rest 조건:
  - ρ = 0.29, P = 0.21
- BAD 및 다른 후보 유전자:
  - 유의한 상관 없음

따라서 GPR137이 해당 psoriasis 위험 변이의 유력한 인과 유전자일 가능성을 지지했으며, 그 효과는 휴지기보다 **활성화된 CD4+ T세포에서 더 뚜렷한 조건 의존적 효과**로 해석됐다.

---

### 6. 논문이 제시하는 한계

- 모든 논문이 성공적으로 에이전트화되는 것은 아니다.
- 코드, 데이터, 의존성 설정이 불완전하면 실패할 수 있다.
- 개방형 생물학적 해석과 가설 생성은 여전히 인간 연구자의 검토가 필요하다.
- 에이전트의 결과는 권위 있는 과학적 결론이라기보다, 검증 가능한 분석과 가설 생성을 지원하는 도구로 봐야 한다.
- 논문과 코드가 업데이트되면 생성된 MCP 도구도 유지·보수해야 한다.

---





## Summary of Results

### 1. AlphaGenome agent benchmark

The authors compared three systems:

- **Paper2Agent-generated AlphaGenome agent**
- **Claude + Repo**, where Claude directly accessed the paper and repository
- **Biomni**, a general-purpose biomedical AI agent

#### Evaluation sets

The benchmark included:

1. **Tutorial-derived queries**
   - 15 queries, repeated across five runs
   - Example: predicting the effect of a specific variant in a specific tissue and reporting its `quantile_score`

2. **Novel queries**
   - 15 queries not directly copied from the tutorials
   - These tested whether the agent could generalize to new variants, tissues and modalities

3. **Open-ended researcher-style queries**
   - 30 queries requiring multi-step tool use and biological interpretation
   - Example: identifying a likely causal gene and mechanism for a GWAS variant

#### Accuracy

| Evaluation | Paper2Agent | Claude + Repo | Biomni |
|---|---:|---:|---:|
| Tutorial-derived | **98.7 ± 1.3%** | 82.7 ± 3.4% | 37.3 ± 4.0% |
| Novel queries | **100.0 ± 0.0%** | 78.7 ± 4.4% | 56.0 ± 3.4% |
| Open-ended queries | **82.7 ± 2.4%** | 56.7 ± 2.3% | 72.2 ± 2.2% |

Paper2Agent outperformed both baselines on structured and novel execution tasks. It also performed better than Claude + Repo on open-ended queries, although these scores mainly measure faithful execution and agreement with key reference entities, not absolute biological truth.

#### Runtime

For tutorial-derived queries, Paper2Agent was:

- **1.9× faster** than Claude + Repo
- **3.1× faster** than Biomni

For novel queries, it was:

- **2.9× faster** than Claude + Repo
- **3.8× faster** than Biomni

The main reason is that Paper2Agent invokes prevalidated tools instead of repeatedly generating and debugging code.

---

### 2. Reproducibility of AlphaGenome tools

Paper2Agent generated **22 AlphaGenome MCP tools**, and all passed automated validation.

Validation checked whether:

- Expected files were produced
- Numerical results matched the reference within tolerance
- Figures matched the reference visualizations
- Each tool could be traced back to the original paper code

The tools covered variant scoring, tissue and cell-type analysis, gene expression, chromatin accessibility, splicing predictions and visualization.

---

### 3. Scanpy single-cell analysis

Paper2Agent generated **seven Scanpy tools**, all of which passed validation.

The agent encoded a complete workflow:

1. Quality control and filtering
2. Normalization
3. Highly variable gene selection
4. Dimensionality reduction
5. Neighbourhood graph construction
6. Clustering
7. Cell-type annotation

#### Test data and results

- Reproducibility was tested on **four public single-cell datasets** not included in the Scanpy codebase.
- Generalization was further tested on **seven diverse datasets**.
- Results were compared with analyses performed by human researchers.

The agent reproduced:

- Cell and gene counts after quality control
- Cluster structures
- Top differentially expressed marker genes
- Appropriate parameter adjustments based on data characteristics

Thus, the Scanpy agent reproduced not only individual functions but also the correct order and logic of the full analysis workflow.

The TISSUE agent similarly produced results consistent with human analyses on spatial transcriptomics data.

---

### 4. Large-scale evaluation

#### 4.1 One hundred computational biology papers

- Successfully agentified papers: **74/100**
- Proposed tools: **599**
- Tools passing automated validation: **593**

Common failure causes included missing executable code, missing data or model files, dependency problems and non-generalizable scripts.

On **300 tutorial-derived questions**:

| System | Accuracy |
|---|---:|
| Paper2Agent + Sonnet 4 | **91.2 ± 1.6%** |
| Claude + Repo, Sonnet 4 | 80.3 ± 2.3% |
| Claude + Repo, Sonnet 4.6 | 86.3 ± 1.1% |

Per-query resource use was also lower:

- Paper2Agent: approximately **US$0.20 and 1.6 minutes**
- Claude + Repo: approximately **US$0.38 and 4.3 minutes**

#### 4.2 Ten non-biology computational papers

The benchmark covered AI, statistics, econometrics, game theory and astrophysics.

- Execution-based tasks: **42**
- Accuracy: **98.1 ± 0.8%**

This suggests that the framework is not limited to computational biology.

#### 4.3 Twenty-six data- and discovery-focused papers

When executable tools could not be constructed, Paper2Agent exposed the manuscript, supplementary files, tables and metadata as structured resources.

- Synthesis questions: **100**
- Paper2Agent resource layer: **89.0 ± 3.1%**
- Claude browser-use baseline: **82.0 ± 3.8%**

The Paper2Agent approach was reported to be approximately **34× cheaper and 15× faster** than the browser-use baseline.

---

### 5. Collaboration between paper agents

The authors connected three agents to study the psoriasis-associated variant **rs887314**:

- AlphaGenome for variant-effect prediction
- MPRA-coupled scCRISPRi for regulatory-element perturbation
- CD4+ T-cell Perturb-seq for gene-knockdown effects

AlphaGenome prioritized **GPR137**:

- CD4+ T-cell RNA-seq quantile score: **0.997**

The perturbation signature was then compared with gene-knockdown signatures.

- GPR137 knockdown:
  - Stim8hr: Spearman ρ = **0.613**, P = **3.79 × 10⁻³**
  - Stim48hr: Spearman ρ = **0.630**, P = **4.71 × 10⁻³**
- Rest condition:
  - ρ = 0.29, P = 0.21
- BAD and other candidate genes:
  - No significant correlation

These results supported GPR137 as a probable causal gene and suggested that its effect is **activation-dependent**, becoming more evident in stimulated CD4+ T cells.

---

### 6. Main limitations

- Not every paper can be successfully agentified.
- Missing code, data or reproducible environments can cause failure.
- Human researchers are still needed for open-ended interpretation and hypothesis evaluation.
- The agents should be viewed as tools for reproducible analysis and hypothesis generation, not as authoritative sources of scientific conclusions.
- MCP servers require maintenance when upstream code or dependencies change.


<br/>
# 예제




### 먼저 구분할 점
이 논문에서 말하는 **트레이닝 데이터**는 Paper2Agent 자체를 새로 학습시키는 데이터라기보다, 주로 다음 두 종류를 의미합니다.

1. **원 논문의 모델 학습 데이터**  
   예: AlphaGenome을 학습하는 데 사용된 유전체 학습 데이터. Paper2Agent는 이를 MCP resource로 연결하지만, 이 논문에서 원자료 전체나 각 샘플의 구체적 입출력을 새로 공개하지는 않습니다.

2. **Paper2Agent가 도구를 만들고 검증할 때 사용하는 예제·튜토리얼 데이터**  
   원 논문의 코드 저장소에 포함된 튜토리얼과 예제 데이터를 실행해 기준 결과를 만들고, 이후 테스트 데이터에 적용합니다. 논문에서 구체적으로 설명되는 것은 주로 이 두 번째 경우입니다.

---

## 1. Paper2Agent 자체의 입력과 출력

### 입력
- 연구 논문 원문
- 논문과 연결된 GitHub 코드 저장소
- supplementary material, 데이터, 그림, 설정 파일
- 튜토리얼과 예제 데이터

### 처리 과정
1. 코드 저장소 탐색 및 다운로드
2. 실행 환경과 의존성 설정
3. 튜토리얼 탐색
4. 튜토리얼을 예제 데이터로 실행
5. 재사용 가능한 함수를 MCP tool로 변환
6. 예제 결과와 비교해 자동 테스트 및 수정

### 출력
- MCP 서버
- 논문의 방법을 실행하는 MCP tools
- 논문·코드·데이터·그림 등의 MCP resources
- 복잡한 분석 순서를 정의한 MCP prompts
- 자연어로 사용할 수 있는 paper agent

예를 들어 Scanpy의 경우, 단순히 논문 내용을 검색하는 것이 아니라 다음과 같은 실행 도구를 생성합니다.

- `quality_control()`
- `quality_control_basic_filtering()`
- `clustering_analysis()`

도구는 원래 튜토리얼 결과와 비교해 검증됩니다. 숫자 결과는 허용 오차 내에서 일치해야 하며, 그림은 perceptual hashing으로 기준 그림과 비교합니다.

---

## 2. AlphaGenome agent 예시

### 태스크
유전 변이가 특정 조직이나 세포 유형에서 유전자 발현, 크로마틴 접근성, 스플라이싱 등에 미치는 영향을 예측하는 태스크입니다.

### 입력 예시 1: 튜토리얼 기반 테스트
```text
Variant: chr3:58394738:A>T
Modality: ATAC-seq
Cell type: motor neuron
Cell Ontology ID: CL:0000100
Question: 이 세포 유형에서 quantile_score는 얼마인가?
```

### 출력 예시
```text
quantile_score = 특정 수치
```

논문은 이와 같은 15개의 튜토리얼 기반 질의를 사용했습니다. Paper2Agent agent는 5회 반복 평가에서 평균 **98.7 ± 1.3% 정확도**를 보였습니다.

### 입력 예시 2: 새로운 테스트 질의
```text
Variant: chr9:98765432:T>C
Modality: DNASE
Cell type: muscle cell
Cell Ontology ID: CL:0000187
Question: 근육 조직의 quantile_score는 얼마인가?
```

이 질의는 튜토리얼에 직접 제시되지 않은 새로운 질의였으며, 15개 novel query에서 **100.0 ± 0.0% 정확도**를 기록했습니다.

### 도구 수준의 입력과 출력
#### 입력
```text
Variant: chr19:8134523:G>A
Modality: ATAC-seq
Tissue: lung
Ontology ID: UBERON:0002048
```

#### 출력
```text
quantile_score = -0.0203067882
```

또 다른 도구는 하나의 변이에 대해 여러 regulatory modality를 계산하고 시각화합니다.

#### 입력
- 변이 위치
- reference allele / alternate allele
- 조직 또는 세포 유형
- 분석할 modality
  - RNA-seq
  - ATAC-seq
  - ChIP-seq 등
- 주변 서열 길이

#### 출력
- 유전자 발현 변화 예측
- 크로마틴 접근성 변화 예측
- 스플라이싱 효과
- modality별 그래프와 시각화
- 유전자별 effect score 또는 quantile score

### 복합 연구자형 태스크
```text
chr22:45969257:G>A 변이가 골밀도 감소와 관련된 이유를 분석하라.
접근성 및 유전자 발현 효과를 비교하고,
가능한 작용 기전과 causal gene을 제시하라.
```

이런 질의에서는 agent가 다음을 순서대로 수행합니다.

1. 변이 입력 파일 생성
2. 여러 modality에서 변이 점수 계산
3. 관련 조직 선택
4. 유전자 발현·크로마틴 효과 비교
5. 후보 causal gene 우선순위화
6. 그림과 해석 보고서 생성

### 논문의 기존 결론 재검토 예시
```text
chr1:109274968:G>T가 LDL cholesterol과 연관되는 이유를 분석하라.
간에서 causal gene과 여러 regulatory effect를 평가하고
publication-ready report를 만들어라.
```

Agent는 `SORT1`을 가장 가능성 높은 유전자로 우선순위화했습니다.

- SORT1 expression quantile score: **0.99983**
- SORT1은 VLDL 분비와 관련된 단백질을 암호화
- 간 조직에서 조절 효과가 강하게 예측됨
- GTEx 간 eQTL에서도 유의한 연관 확인

다만 `CELSR2`와 `PSRC1`도 높은 점수를 보였습니다. 따라서 이 결과는 “SORT1이 확정적인 원인 유전자”라기보다, **독립적인 모델 증거에 기반한 새로운 후보 우선순위**로 해석해야 합니다.

---

## 3. Scanpy agent 예시

### 태스크
단일세포 RNA-seq 데이터를 전처리하고 세포를 군집화하는 태스크입니다.

### 입력
```text
data.h5ad
```

사용자는 복잡한 코드를 작성하지 않고 다음과 같이 요청할 수 있습니다.

```text
이 단일세포 데이터에 표준적인 preprocessing과 clustering pipeline을 수행하라.
```

### agent가 자동으로 수행하는 순서
1. 데이터 구조와 세포·유전자 수 확인
2. QC metric 계산
3. 낮은 품질 세포 및 유전자 필터링
4. doublet 탐지
5. 정규화
6. highly variable gene 선택
7. PCA 또는 차원 축소
8. neighborhood graph 생성
9. Leiden clustering
10. marker gene 기반 세포 유형 주석
11. UMAP 및 QC plot 생성

### 출력 예시
```text
Quality control completed.
Filtered data:
- Cells: 17,041
- Genes: 23,424
```

추가 출력:

- QC metric 그림
- highly variable gene plot
- UMAP
- Leiden cluster 결과
- cluster별 marker gene 표
- 세포 유형 annotation
- 분석 요약 보고서

예를 들어 특정 cluster에서 다음 marker가 높게 나타날 수 있습니다.

- `LST1`, `AIF1`, `TYROBP`: myeloid 계열
- `MS4A1`, `CD79A`: B cell
- `NKG7`, `GNLY`, `PRF1`: NK/T cell
- `PF4`, `PPBP`: platelet 관련 세포

### 테스트 데이터와 기준 결과
Scanpy agent는 Scanpy 공식 튜토리얼에 포함되지 않은 공개 단일세포 데이터에도 적용되었습니다.

비교 대상은 다음과 같습니다.

- 인간 연구자가 동일한 데이터를 처리한 결과
- Scanpy agent가 동일한 분석 pipeline을 실행한 결과

비교한 항목은 다음과 같습니다.

- QC 후 남은 세포 수와 유전자 수
- normalization과 feature selection 결과
- cluster 구조
- cluster별 top differentially expressed marker genes
- UMAP 등 주요 시각화

Agent 결과는 인간 연구자의 결과와 동등한 수준의 세포·유전자 수와 marker gene을 재현했습니다. 또한 여러 데이터셋의 특성에 따라 filtering threshold와 분석 파라미터를 적응적으로 조정했습니다.

---

## 4. 여러 논문의 데이터를 결합한 psoriasis 사례

이 사례는 단일 논문 agent가 아니라 세 가지 agent를 연결한 예입니다.

### 입력 데이터
1. **AlphaGenome**
   - psoriasis 관련 변이 `rs887314`
   - CD4+ T cell에서 유전자 발현 효과 예측

2. **MPRA-coupled scCRISPRi**
   - 해당 변이가 위치한 cis-regulatory element, 즉 CRE를 perturb했을 때의 downstream gene expression 변화

3. **CD4+ T-cell Perturb-seq**
   - 여러 후보 유전자를 knockdown했을 때의 transcriptome 변화
   - 조건:
     - Rest
     - Stim8hr
     - Stim48hr

### 1단계: 계산 예측
AlphaGenome은 `rs887314` 주변 유전자 중 다음 유전자를 가장 높은 후보로 제시했습니다.

```text
Candidate gene: GPR137
RNA-seq quantile score: 0.997
Cell type: CD4+ T cell
```

### 2단계: 실험 데이터와 검증
Agent는 다음 두 signature를 비교했습니다.

- rs887314의 CRE를 perturb했을 때 발생한 유전자 발현 변화
- GPR137 또는 다른 후보 유전자를 knockdown했을 때 발생한 발현 변화

### 출력
GPR137 knockdown signature와 CRE perturbation signature 사이에 유의한 상관이 관찰되었습니다.

- Stim8hr:
  - Spearman ρ = **0.613**
  - P = **3.79 × 10⁻³**
- Stim48hr:
  - Spearman ρ = **0.630**
  - P = **4.71 × 10⁻³**
- Rest:
  - Spearman ρ = **0.29**
  - P = **0.21**

반면 BAD 및 다른 후보 유전자에서는 유의한 일치가 나타나지 않았습니다.

### 해석
이 결과는 다음을 시사합니다.

- `GPR137`은 rs887314와 연결된 psoriasis 관련 causal gene 후보이다.
- 그 영향은 CD4+ T cell이 활성화된 조건에서 더 뚜렷하다.
- 단순히 한 모델의 예측만 사용한 것이 아니라, 예측 결과를 두 종류의 perturbation 데이터로 검증했다.

단, 논문도 이 결과를 확정적 인과관계라기보다는 **계산 예측과 독립적인 실험 데이터가 일치하는 강한 후보 증거**로 제시합니다.

---

## 핵심 요약

| 구분 | 입력 | 출력 | 주요 태스크 |
|---|---|---|---|
| Paper2Agent 구축 | 논문, 코드, 튜토리얼, 예제 데이터 | MCP tools, resources, prompts | 논문을 실행 가능한 agent로 변환 |
| AlphaGenome | 변이, 조직/세포 유형, 분석 modality | effect score, quantile score, 시각화, causal gene 후보 | 변이 기능 및 조절 효과 예측 |
| Scanpy | `data.h5ad` 단일세포 데이터 | QC 결과, cluster, marker gene, UMAP | 전처리·군집화·세포 유형 분석 |
| Psoriasis 협업 | rs887314, CRE perturbation, gene knockdown 데이터 | GPR137 후보 및 signature correlation | causal gene 우선순위화와 실험적 검증 |

중요한 점은 Paper2Agent가 새 생물학적 모델을 직접 학습하는 시스템이라기보다, **기존 논문의 코드·데이터·분석 절차를 MCP 도구로 감싸고, 예제 결과와 비교해 검증한 뒤 자연어로 실행하게 만드는 시스템**이라는 것입니다.

---



## Important distinction

In this paper, “training data” can refer to two different things:

1. **The original training data of a published model**, such as the genomic data used to train AlphaGenome. Paper2Agent exposes links to such resources, but the paper does not provide a new detailed sample-by-sample training input/output description.

2. **Tutorial and example data used to build and validate Paper2Agent tools.**  
   Paper2Agent executes the original tutorials, records their outputs, converts the workflows into MCP tools, and tests whether the generated tools reproduce the reference results. This second type is described most concretely in the paper.

---

## 1. Inputs and outputs of Paper2Agent

### Inputs
- Research paper
- Associated GitHub repository
- Supplementary files and datasets
- Figures, configuration files and tutorials
- Example data used in the original workflow

### Processing
1. Locate and download the codebase
2. Set up an isolated environment
3. Find relevant tutorials
4. Execute tutorials with example data
5. Convert reusable analysis steps into MCP tools
6. Test the tools against the tutorial outputs
7. Package the validated tools into an MCP server

### Outputs
- MCP server
- Executable MCP tools
- Structured resources containing the paper, code, data and figures
- MCP prompts describing multi-step workflows
- A natural-language paper agent

For example, a Scanpy agent exposes functions such as:

- `quality_control()`
- `quality_control_basic_filtering()`
- `clustering_analysis()`

The tools are validated against the original tutorial results. Numerical outputs must match within a specified tolerance, and generated figures are compared with reference figures using perceptual hashing.

---

## 2. AlphaGenome agent

### Task
Predict how a genetic variant affects gene expression, chromatin accessibility, splicing and other regulatory modalities in a particular tissue or cell type.

### Example 1: Tutorial-based test input
```text
Variant: chr3:58394738:A>T
Modality: ATAC-seq
Cell type: motor neuron
Cell Ontology ID: CL:0000100
Question: What is the quantile_score for this cell type?
```

### Output
```text
quantile_score = a specific numerical value
```

Paper2Agent evaluated 15 tutorial-derived queries and achieved **98.7 ± 1.3% accuracy** across five independent runs.

### Example 2: Novel test input
```text
Variant: chr9:98765432:T>C
Modality: DNASE
Cell type: muscle cell
Cell Ontology ID: CL:0000187
Question: What is the quantile_score for muscle tissue?
```

This was a novel query rather than a direct tutorial question. The agent achieved **100.0 ± 0.0% accuracy** on 15 novel queries.

### Tool-level input and output

#### Input
```text
Variant: chr19:8134523:G>A
Modality: ATAC-seq
Tissue: lung
Ontology ID: UBERON:0002048
```

#### Output
```text
quantile_score = -0.0203067882
```

Other tools accept:

- Variant position and alleles
- Tissue or cell type
- Sequence context length
- Selected modalities, such as RNA-seq, ATAC-seq or ChIP-seq

They return:

- Predicted gene-expression effects
- Chromatin-accessibility effects
- Splicing effects
- Gene-level effect or quantile scores
- Modality-specific visualizations

### Open-ended researcher-style task
```text
Analyze why chr22:45969257:G>A is associated with reduced bone mineral density.
Compare accessibility and gene-expression effects, and identify a likely
mechanism and causal gene.
```

The agent can:

1. Create variant input files
2. Score the variant across several modalities
3. Select relevant tissues
4. Compare gene-expression and chromatin effects
5. Rank candidate causal genes
6. Generate figures and an interpretation report

### Re-evaluating a published conclusion
```text
Use AlphaGenome to interpret why chr1:109274968:G>T is associated with
LDL cholesterol. Identify candidate causal genes, assess regulatory effects
in liver, and generate a publication-ready report.
```

The agent prioritized `SORT1`:

- SORT1 expression quantile score: **0.99983**
- SORT1 is related to VLDL secretion
- Strong regulatory effects were predicted in liver
- The variant was also a significant SORT1 eQTL in GTEx liver

However, `CELSR2` and `PSRC1` also had high scores. Therefore, this result should be interpreted as **a model-based prioritization of a candidate gene**, not as definitive proof of causality.

---

## 3. Scanpy agent

### Task
Perform preprocessing and clustering of single-cell RNA-seq data.

### Input
```text
data.h5ad
```

A user can simply ask:

```text
Perform the standard single-cell preprocessing and clustering pipeline
on this dataset.
```

### Automated workflow
1. Inspect the dataset
2. Calculate quality-control metrics
3. Filter low-quality cells and genes
4. Detect doublets
5. Normalize counts
6. Select highly variable genes
7. Perform dimensionality reduction
8. Construct a neighborhood graph
9. Run Leiden clustering
10. Annotate cell types using marker genes
11. Generate UMAP and QC plots

### Example output
```text
Quality control completed.
Filtered data:
- Cells: 17,041
- Genes: 23,424
```

Additional outputs include:

- QC plots
- Highly variable gene plots
- UMAP
- Leiden cluster assignments
- Cluster-specific marker gene tables
- Cell-type annotations
- A summary report

Example marker patterns may include:

- `LST1`, `AIF1`, `TYROBP`: myeloid cells
- `MS4A1`, `CD79A`: B cells
- `NKG7`, `GNLY`, `PRF1`: NK/T cells
- `PF4`, `PPBP`: platelet-related cells

### Test data and reference results
The agent was applied to public single-cell datasets that were not included in the Scanpy codebase. Its results were compared with analyses performed by human researchers using the same data.

The comparisons included:

- Number of cells and genes after QC
- Normalization and feature-selection results
- Cluster structure
- Top differentially expressed marker genes
- Major visualizations such as UMAP

The agent reproduced equivalent cell/gene counts and marker-gene patterns. It also adapted parameters to different dataset characteristics.

---

## 4. Psoriasis causal-gene prioritization

This case connected three paper agents:

1. **AlphaGenome**
   - Predicts the effect of the psoriasis-associated variant `rs887314`
   - Focuses on CD4+ T cells

2. **MPRA-coupled scCRISPRi**
   - Measures downstream gene-expression changes after perturbing the relevant cis-regulatory element

3. **CD4+ T-cell Perturb-seq**
   - Measures transcriptome changes after knocking down individual genes
   - Conditions:
     - Rest
     - Stim8hr
     - Stim48hr

### Step 1: Computational prioritization
AlphaGenome ranked `GPR137` as the top candidate:

```text
Candidate gene: GPR137
RNA-seq quantile score: 0.997
Cell type: CD4+ T cell
```

### Step 2: Experimental validation
The agent compared:

- The gene-expression signature caused by perturbing the rs887314 CRE
- The gene-expression signature caused by knocking down GPR137 or other candidate genes

### Output
The CRE perturbation signature significantly matched the GPR137 knockdown signature:

- Stim8hr:
  - Spearman ρ = **0.613**
  - P = **3.79 × 10⁻³**
- Stim48hr:
  - Spearman ρ = **0.630**
  - P = **4.71 × 10⁻³**
- Rest:
  - Spearman ρ = **0.29**
  - P = **0.21**

BAD and the other candidate genes did not show significant agreement.

### Interpretation
These results suggest that:

- `GPR137` is a strong candidate causal gene at the rs887314 psoriasis locus.
- Its effect is more apparent in activated CD4+ T cells.
- The conclusion is supported by agreement between computational prediction and independent perturbation datasets.

Nevertheless, the result should be understood as **strong convergent evidence for a candidate causal gene**, not as definitive proof of causality.

---

## Summary table

| Component | Input | Output | Main task |
|---|---|---|---|
| Paper2Agent construction | Paper, code, tutorials, example data | MCP tools, resources and prompts | Convert a paper into an executable agent |
| AlphaGenome agent | Variant, tissue/cell type, modality | Effect scores, quantile scores, plots, candidate genes | Predict regulatory effects of variants |
| Scanpy agent | `data.h5ad` single-cell dataset | QC, clusters, marker genes, UMAP | Single-cell preprocessing and clustering |
| Psoriasis collaboration | rs887314, CRE perturbation and gene-knockdown data | GPR137 prioritization and signature correlation | Causal-gene prioritization and validation |

The main point is that Paper2Agent is not primarily a system that retrains biological models. It **wraps the code, data and workflows of existing papers into MCP-based executable tools, validates them against reference examples, and makes them accessible through natural-language interaction**.

<br/>
# 요약


Paper2Agent는 논문·코드·데이터를 분석해 MCP 서버와 검증된 도구로 변환하고, 이를 자연어 기반 AI 에이전트와 연결하는 다중 에이전트 프레임워크다.  
생성된 도구는 원 논문의 결과·수치·그림과 비교해 자동 검증되며, AlphaGenome·Scanpy·TISSUE 사례에서 기존 분석을 재현하고 새로운 데이터 분석도 수행했다.  
또한 AlphaGenome, scCRISPRi, Perturb-seq 에이전트를 결합해 건선 관련 변이 rs887314의 유력한 원인 유전자로 GPR137을 제안하고, CD4+ T세포 활성화 조건에서 이를 뒷받침하는 발현 서명 상관관계를 확인했다.  




Paper2Agent is a multi-agent framework that converts papers, code and data into MCP servers with validated tools, which can then be accessed through natural-language AI agents.  
The generated tools are automatically tested against the original results, and case studies with AlphaGenome, Scanpy and TISSUE reproduced published analyses while supporting new-data applications.  
By combining AlphaGenome, scCRISPRi and Perturb-seq agents, the system prioritized GPR137 as a likely causal gene for psoriasis-associated variant rs887314 and found supporting expression-signature correlations in activated CD4+ T cells.

<br/>
# 기타



### 전체 메시지
이 논문의 다이어그램과 피규어들은 **논문을 단순한 문서가 아니라, 실행 가능한 AI 에이전트로 바꾸는 과정과 그 신뢰성·재현성·확장성**을 보여준다. 핵심은 Paper2Agent가 논문과 코드에서 MCP 서버를 만들고, 이를 대화형 AI 에이전트와 연결해 사용자가 자연어로 분석을 수행하게 한다는 점이다.

---

### Figure 1. Paper2Agent의 전체 구조와 변환 과정

**결과**
- 논문, 코드 저장소, 보충자료와 데이터를 입력으로 사용한다.
- 환경 설정 에이전트가 필요한 소프트웨어 환경을 구성한다.
- 추출 에이전트가 논문의 핵심 분석 방법을 실행 가능한 MCP 도구로 변환한다.
- 테스트 에이전트가 원 논문의 결과와 비교해 도구를 검증한다.
- 완성된 MCP 서버를 Hugging Face 같은 원격 서버에 배포하고, Claude와 같은 AI 에이전트에 연결한다.

MCP 서버는 세 요소로 구성된다.

1. **MCP tools**: 분석, 예측, 시각화 등을 실제로 실행하는 함수  
2. **MCP resources**: 논문, 코드, 데이터, 그림, 보충자료  
3. **MCP prompts**: 여러 분석 도구를 올바른 순서로 실행하도록 안내하는 작업 흐름

**인사이트**
- 논문을 단순히 검색하거나 요약하는 수준을 넘어, 논문의 **방법·데이터·코드·분석 절차 전체를 실행 가능한 지식 시스템**으로 바꾼다.
- 여러 논문의 MCP를 하나의 AI 에이전트에 연결할 수 있어, 논문 간 협업과 통합 분석이 가능하다.
- 신뢰성은 LLM이 매번 코드를 새로 생성하게 하는 것이 아니라, 검증된 도구를 고정해 사용하는 방식으로 높인다.

---

### Figure 2. AlphaGenome 에이전트의 구축과 성능

#### 2a. AlphaGenome MCP와 에이전트

**결과**
- AlphaGenome의 유전 변이 분석 기능을 22개의 MCP 도구로 변환했다.
- 모든 도구가 자동 검증을 통과했다.
- 변이 점수 계산, 다양한 조직·세포 유형에 대한 예측, 유전자 발현·크로마틴 접근성·스플라이싱 분석, 시각화 등을 지원한다.

**인사이트**
- 사용자는 복잡한 API나 실행 환경을 직접 이해하지 않고도 자연어로 변이 효과를 분석할 수 있다.
- 각 도구가 원 논문의 소스 코드와 연결되어 있어 결과의 추적성과 재현성을 확보한다.

#### 2b. 벤치마크 정확도

**결과**
- 튜토리얼 기반 질문: Paper2Agent **98.7%**
- 새로운 질문: Paper2Agent **100.0%**
- 개방형 연구자형 질문: Paper2Agent **82.7%**
- 직접 저장소에 접근한 Claude와 Biomni보다 높은 성능을 보였다.

**인사이트**
- 구조화된 도구와 검증된 실행 환경이 단순히 저장소를 읽고 코드를 작성하는 방식보다 안정적이다.
- 정답이 명확한 작업에서는 매우 높은 정확도를 보였지만, 여러 생물학적 근거를 통합해야 하는 개방형 해석에서는 여전히 인간의 판단이 필요하다.

#### 2c. 실행 시간

**결과**
- Paper2Agent는 Claude+Repo와 Biomni보다 대부분의 질문에서 더 빠르게 실행됐다.
- 튜토리얼 기반 질문에서는 각각 약 1.9배, 3.1배 빠르고, 새로운 질문에서는 약 2.9배, 3.8배 빠른 것으로 보고됐다.

**인사이트**
- 검증된 MCP 도구를 사용하면 매번 코드를 탐색하고 작성하는 비용을 줄일 수 있다.
- 재현성뿐 아니라 분석 효율성과 비용 측면에서도 이점이 있다.

#### 2d. GWAS 변이의 자동 해석

**결과**
- LDL 콜레스테롤과 관련된 변이 `chr1:109274968:G>T`를 대상으로 에이전트가 자동으로 분석 계획을 세웠다.
- 변이 점수 계산, 조직별 필터링, 여러 modality의 시각화, 후보 유전자 비교와 보고서 생성을 연속적으로 수행했다.
- SORT1을 가장 유력한 후보 유전자로 제시했다.

**인사이트**
- 하나의 프롬프트만으로 복잡한 다단계 분석을 자동화할 수 있다.
- 다만 CELSR2와 PSRC1도 높은 점수와 eQTL 근거를 보였기 때문에, SORT1을 확정적 인과 유전자로 보기는 어렵다.
- 이 사례는 AI 에이전트가 기존 논문의 해석을 그대로 반복하는 것이 아니라, 새로운 계산 근거로 재평가할 수 있음을 보여준다.

---

### Figure 3. Scanpy 에이전트와 단일세포 분석

#### 3a. Scanpy MCP의 구성

**결과**
- Scanpy의 전처리와 클러스터링 기능을 7개 도구로 변환했다.
- 품질관리, 세포·유전자 필터링, doublet 탐지, 군집화, 시각화 등을 포함한다.
- 모든 도구가 자동 검증을 통과했다.

**인사이트**
- 특정 소프트웨어 패키지 전체를 무작정 노출하는 것이 아니라, 실제 연구에서 자주 사용하는 핵심 기능을 재사용 가능한 도구로 추출했다.

#### 3b. MCP prompt를 이용한 분석 순서 제어

**결과**
- 에이전트는 다음과 같은 표준 workflow를 자동으로 따른다.

  품질관리 → 정규화 → 고변이 유전자 선택 → 차원 축소 → 이웃 그래프 구축 → 클러스터링 → 세포 유형 주석

- 사용자는 데이터 경로만 제공하면 된다.
- 에이전트는 데이터 특성을 먼저 확인하고 필요한 경우 파라미터를 조정한다.

**인사이트**
- 단순히 도구를 제공하는 것만으로는 복잡한 분석의 실행 순서를 보장하기 어렵다.
- MCP prompt가 분석 workflow를 명시함으로써, 사용자의 프롬프트 작성 부담을 줄이고 분석의 일관성과 재현성을 높인다.

#### 3c. 인간 연구자 결과와의 비교

**결과**
- 공개 단일세포 데이터에 대해 에이전트가 인간 연구자와 유사한 세포·유전자 수, 클러스터 구조, 주요 marker gene을 재현했다.
- 여러 데이터셋에 적용했을 때 데이터 특성에 맞게 파라미터를 조정했다.

**인사이트**
- Paper2Agent는 단순 Q&A 시스템이 아니라, 실제 분석 pipeline을 수행하는 실행형 도구로 기능한다.
- 그러나 세포 유형 주석이나 생물학적 해석의 타당성은 여전히 연구자가 검토해야 한다.

---

### Figure 4. 건선 관련 인과 유전자 GPR137의 우선순위화

#### 4a. 세 논문 에이전트의 통합

**결과**
- AlphaGenome은 건선 위험 변이 `rs887314`에서 CD4+ T세포의 후보 유전자로 **GPR137**을 우선순위화했다.
- RNA-seq quantile score는 **0.997**이었다.
- 이후 MPRA-coupled scCRISPRi 데이터와 CD4+ T세포 Perturb-seq 데이터를 사용해 예측을 검증했다.

**인사이트**
- 서로 다른 논문의 에이전트를 연결하면, 계산 예측과 실험 데이터를 하나의 분석 흐름에서 통합할 수 있다.
- 이 방식은 하나의 데이터셋이나 단일 모델에 의존하지 않고, 독립적인 근거를 결합한다.

#### 4b. CRE perturbation과 유전자 knockdown signature의 상관

**결과**
- `rs887314`의 조절요소(CRE)를 perturb했을 때의 유전자 발현 signature와 GPR137 knockdown signature가 자극 조건에서 유의하게 일치했다.
  - Stim8hr: Spearman ρ = **0.613**, P = **0.0038**
  - Stim48hr: Spearman ρ = **0.630**, P = **0.0047**
- Rest 조건에서는 유의하지 않았다.
  - ρ = 0.29, P = 0.21
- BAD 및 다른 후보 유전자에서는 유의한 일치가 관찰되지 않았다.

**인사이트**
- GPR137이 해당 변이의 가능한 인과 유전자라는 계산적·실험적 근거가 강화됐다.
- GPR137의 효과는 항상 나타나는 것이 아니라 **CD4+ T세포가 활성화된 상황에서 더 뚜렷한 조건 의존적 효과**일 가능성이 있다.
- 다만 이 결과는 “확정적 인과”라기보다, 추가 실험 검증이 필요한 강한 후보를 제시한 것이다.
- 특히 이 분석 전략 자체가 원 논문들에 직접 제시된 방식이 아니라, 여러 데이터셋을 연결해 에이전트가 제안한 새로운 통합 분석이다.

---

### Extended Data Figure 1. TISSUE 에이전트

**결과**
- TISSUE 논문을 기반으로 공간 전사체 분석용 에이전트를 만들었다.
- 불확실성 보정 공간 전사체 예측, 질의응답, 데이터 자동 접근과 분석 재현을 지원한다.
- 인간 연구자가 수행한 결과와 일치하는 분석 결과를 재현했다.

**인사이트**
- Paper2Agent는 단일세포 분석뿐 아니라 공간 전사체처럼 전문적이고 복잡한 분석 분야에도 적용 가능하다.
- 데이터와 보충자료를 구조화된 리소스로 제공하면, AI가 분석뿐 아니라 데이터 출처와 불확실성까지 함께 다룰 수 있다.

---

### Extended Data Figure 2. ADHD GWAS와 AlphaGenome의 결합

**결과**
- ADHD GWAS 데이터와 AlphaGenome 에이전트를 연결해 209개 후보 변이를 평가했다.
- `rs1626703`을 유력한 후보 변이로 우선순위화했다.
- 이 변이가 글루타메이트성 뉴런에서 **MPHOSPH9의 스플라이싱과 발현**에 영향을 줄 수 있다는 기전 가설을 제시했다.

**인사이트**
- 논문 에이전트는 기존 논문의 결과를 재현하는 데 그치지 않고, 새로운 데이터셋과 결합해 검증 가능한 가설을 생성할 수 있다.
- 다만 ADHD 관련 기전은 계산적 예측 단계이므로, 실험적 검증이 필요하다.

---

### 표와 보충자료에서 제시된 주요 결과

본문에 언급된 표·보충자료의 핵심은 다음과 같다.

- **Supplementary Table 2**: 여러 단일세포 데이터셋에서 Scanpy 에이전트가 데이터 특성에 맞게 분석 파라미터를 조정했음을 보여준다.
- **Supplementary Figures 2–3**: AlphaGenome 에이전트의 성능이 프롬프트 표현을 바꾸거나 더 최신 Claude 모델을 사용해도 대체로 유지됐다.
- **Supplementary Figure 4**: GPR137 외 후보 유전자들과 CRE perturbation signature의 비교 결과를 추가로 제시한다.
- **Supplementary Note**: 벤치마크 질문, 프롬프트, ablation 연구, 보안·실행 실패 사례, 에이전트 간 협업 분석을 자세히 설명한다.
- **대규모 평가**
  - 계산생물학 논문 100편 중 74편이 성공적으로 agentification됐다.
  - 599개 도구 중 593개가 자동 검증을 통과했다.
  - 튜토리얼 기반 질문 정확도는 **91.2%**였다.
  - 비생물학 계산 논문 10편에서는 **98.1%** 정확도를 기록했다.
  - 데이터·발견 중심 논문 26편의 synthesis 질문에서는 **89.0%** 정확도를 보였다.

**종합 인사이트**
- Paper2Agent는 다양한 논문과 분야에 확장 가능하지만, 모든 논문을 자동으로 에이전트화할 수 있는 것은 아니다.
- 실패의 주요 원인은 실행 가능한 코드 부족, 데이터·모델 파일 누락, 의존성 문제, 문서화 부족이었다.
- 따라서 논문의 재현 가능성과 agentification 가능성은 밀접하게 연결된다.
- 저자들은 장기적으로 논문에 코드·데이터뿐 아니라 **‘agent availability’ 정보**도 포함해야 한다고 제안한다.
- 개방형 과학적 추론과 최종 해석은 여전히 인간 연구자의 감독이 필요하다.

---






### Overall message
The figures and supplementary materials show how Paper2Agent transforms a research paper from a static document into an **executable, interactive and reproducible AI agent**. The framework extracts tools, resources and workflows from a paper and exposes them through an MCP server that can be queried using natural language.

---

### Figure 1. Overall architecture and workflow

**Results**
- The system takes the manuscript, code repository, supplementary materials and datasets as input.
- An environment agent configures the software environment.
- An extraction agent converts the paper’s methods into executable MCP tools.
- A testing agent validates the tools against the original results.
- The resulting MCP server is deployed remotely and connected to an AI agent such as Claude.

Each MCP server contains:

1. **MCP tools**: executable functions for analysis, prediction and visualization  
2. **MCP resources**: manuscripts, code, datasets, figures and supplementary materials  
3. **MCP prompts**: instructions that organize multi-step workflows

**Insight**
- Paper2Agent goes beyond retrieving or summarizing papers. It turns the paper’s methods, code, data and workflows into an **executable knowledge system**.
- Multiple paper MCPs can be connected to one agent, enabling cross-paper analysis.
- Reliability is improved by using validated and fixed tools rather than repeatedly generating new code.

---

### Figure 2. AlphaGenome agent and performance

#### 2a. AlphaGenome MCP and agent

**Results**
- AlphaGenome was converted into 22 MCP tools.
- All tools passed automated validation.
- The tools support variant scoring, tissue- and cell-type-specific predictions, gene-expression and chromatin-accessibility analyses, splicing analysis and visualization.

**Insight**
- Users can analyze genetic variants through natural language without directly managing complex APIs or software environments.
- Links to the original source code improve traceability and reproducibility.

#### 2b. Benchmark accuracy

**Results**
- Tutorial-based queries: **98.7%**
- Novel queries: **100.0%**
- Open-ended researcher-style queries: **82.7%**
- Paper2Agent outperformed Claude with direct repository access and Biomni.

**Insight**
- Structured and validated tools are more reliable than asking a general-purpose agent to inspect a repository and generate code from scratch.
- Performance was highest when the task had a well-defined answer. Open-ended biological interpretation still requires human judgment.

#### 2c. Runtime

**Results**
- Paper2Agent was faster than both Claude+Repo and Biomni.
- It was approximately 1.9× and 3.1× faster on tutorial queries, and 2.9× and 3.8× faster on novel queries, respectively.

**Insight**
- Prevalidated tools reduce the need for repeated code exploration and generation.
- The framework improves not only reproducibility but also efficiency and cost.

#### 2d. Automated GWAS interpretation

**Results**
- For the LDL-associated variant `chr1:109274968:G>T`, the agent automatically planned and executed variant scoring, tissue filtering, multimodal visualization, candidate-gene comparison and report generation.
- It prioritized **SORT1** as the leading candidate gene.

**Insight**
- A single natural-language prompt can trigger a complex multi-step analysis.
- SORT1 should not be considered definitively causal because CELSR2 and PSRC1 also showed strong scores and eQTL evidence.
- The example demonstrates that an agent can independently reassess published interpretations using new computational evidence.

---

### Figure 3. Scanpy agent for single-cell analysis

#### 3a. Scanpy MCP construction

**Results**
- Scanpy preprocessing and clustering were converted into seven tools.
- These included quality control, cell and gene filtering, doublet detection, clustering and visualization.
- All tools passed automated validation.

**Insight**
- The framework extracts commonly used, reusable functions rather than simply exposing an entire software package.

#### 3b. Workflow control through MCP prompts

**Results**
- The agent follows a standardized workflow:

  quality control → normalization → highly variable gene selection → dimensionality reduction → neighborhood graph construction → clustering → cell-type annotation

- Users only need to provide the data path.
- The agent inspects the dataset and adjusts parameters when necessary.

**Insight**
- Tools alone do not guarantee that a complex workflow will be executed in the correct order.
- MCP prompts encode the workflow explicitly, improving reproducibility and reducing the burden on users.

#### 3c. Comparison with human researchers

**Results**
- On public single-cell datasets, the agent reproduced cell and gene counts, cluster structures and marker genes comparable to those obtained by human researchers.
- It adapted parameters across diverse datasets.

**Insight**
- The Scanpy agent functions as an executable analysis pipeline rather than merely a question-answering system.
- Researchers should still evaluate the biological validity of cell-type annotations and downstream interpretations.

---

### Figure 4. Prioritizing GPR137 as a psoriasis causal gene

#### 4a. Integration of three paper agents

**Results**
- AlphaGenome prioritized **GPR137** at the psoriasis-associated variant `rs887314` in CD4+ T cells.
- Its RNA-seq quantile score was **0.997**.
- This prediction was evaluated using MPRA-coupled scCRISPRi and CD4+ T-cell Perturb-seq data.

**Insight**
- Agents based on independent papers can combine computational predictions with experimental perturbation data.
- This provides stronger evidence than relying on a single model or dataset.

#### 4b. Correlation between CRE perturbation and gene-knockdown signatures

**Results**
- The expression signature caused by perturbing the `rs887314` regulatory element significantly matched the GPR137 knockdown signature under stimulation:
  - Stim8hr: Spearman ρ = **0.613**, P = **0.0038**
  - Stim48hr: Spearman ρ = **0.630**, P = **0.0047**
- The correlation was not significant in the Rest condition:
  - ρ = 0.29, P = 0.21
- BAD and other candidate genes did not show significant concordance.

**Insight**
- The results strengthen the case that GPR137 is a likely causal gene at this locus.
- The effect appears to be **context-dependent**, becoming more evident when CD4+ T cells are activated.
- The result is strong prioritization evidence, not definitive proof of causality.
- The signature-correlation strategy itself was a novel integration proposed by the AI co-scientist rather than a direct method from either source paper.

---

### Extended Data Figure 1. TISSUE agent

**Results**
- Paper2Agent generated an agent for uncertainty-aware spatial transcriptomics analysis based on TISSUE.
- It supported spatial-expression prediction, uncertainty-aware analysis, data access and reproducibility checks.
- Its outputs matched analyses performed by human researchers.

**Insight**
- The framework can be applied beyond single-cell RNA-seq to specialized spatial transcriptomics workflows.
- Structured access to data and supplementary materials allows the agent to handle both analysis and uncertainty information.

---

### Extended Data Figure 2. Combining ADHD GWAS with AlphaGenome

**Results**
- The agent evaluated 209 candidate variants from an ADHD GWAS dataset.
- It prioritized `rs1626703`.
- It proposed that the variant may affect splicing and expression of **MPHOSPH9** in glutamatergic neurons.

**Insight**
- Paper agents can combine published methods with new datasets to generate testable hypotheses.
- The proposed ADHD mechanism remains computational and requires experimental validation.

---

### Key results from tables and supplementary materials

The main points from the referenced tables and supplementary analyses are:

- **Supplementary Table 2**: The Scanpy agent adapted analysis parameters to the characteristics of different single-cell datasets.
- **Supplementary Figures 2–3**: AlphaGenome performance remained robust to prompt paraphrasing and to the use of newer Claude models.
- **Supplementary Figure 4**: Additional comparisons between GPR137 and other candidate genes were provided.
- **Supplementary Note**: Detailed benchmark queries, prompts, ablation studies, failure cases, security considerations and agent-collaboration analyses are described.
- **Large-scale evaluation**
  - 74 of 100 computational biology papers were successfully agentified.
  - 593 of 599 proposed tools passed automated validation.
  - Accuracy on tutorial-derived questions was **91.2%**.
  - Accuracy across 10 non-biology computational papers was **98.1%**.
  - Accuracy on synthesis questions from 26 data- and discovery-focused papers was **89.0%**.

**Overall insight**
- Paper2Agent is broadly scalable, but not every paper can be converted into a robust agent.
- Major obstacles included incomplete code, missing datasets or model files, dependency failures and poor documentation.
- Reproducibility and agentifiability are therefore closely related.
- The authors propose that papers should eventually include an **“agent availability” section**, alongside data and code availability.
- Human researchers remain responsible for choosing scientific directions, evaluating evidence and interpreting open-ended results.

<br/>
# refer format:




### BibTeX

```bibtex
@article{Miao2026Paper2Agent,
  author  = {Miao, Jiacheng and Davis, Joe R. and Zhang, Yaohui and Pritchard, Jonathan K. and Zou, James},
  title   = {Reimagining Research Papers as Interactive and Reliable AI Agents},
  journal = {Nature},
  year    = {2026},
  doi     = {10.1038/s41586-026-11044-y},
  url     = {https://doi.org/10.1038/s41586-026-11044-y}
}
```

### 시카고 스타일 

Miao, Jiacheng, Joe R. Davis, Yaohui Zhang, Jonathan K. Pritchard, and James Zou. “Reimagining Research Papers as Interactive and Reliable AI Agents.” *Nature*, 2026. https://doi.org/10.1038/s41586-026-11044-y.

