---
layout: post
title:  "[2026]Med.ai ASK: An Agentic System for Biomedical Question Answering"
date:   2026-09-16 18:26:06 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: RAG기반 멀티 에이전트 콜라보레이션(여러소스의 문서 가져와서)  


짧은 요약(Abstract) :




### 1. 연구 목적
Med.ai ASK는 생의학 분야의 질문에 답하기 위해 개발된 에이전트형 질의응답 시스템입니다.  
단순히 한 번 검색하고 답하는 것이 아니라, 질문의 유형을 분석한 뒤 PubMed, FDA 의약품 라벨, 임상시험 데이터베이스, UniProt 등 여러 지식원과 도구를 선택적으로 활용합니다. 이를 통해 생의학 연구자에게 더 정확하고 근거가 분명한 답변을 제공하는 것이 목적입니다.

### 2. 연구 방법
이 시스템은 ReAct의 도구 호출 방식과 Self-Discover의 추론 구조를 바탕으로 설계되었습니다.  
에이전트는 질문에 따라 다음과 같은 작업을 수행합니다.

- 필요한 검색 도구와 데이터베이스를 자동으로 선택
- 여러 문서에서 관련 정보를 검색
- 검색 결과를 통합하고 비교하여 답변 생성
- 답변에 출처와 인용을 함께 제시
- 적절한 근거를 찾지 못하면 추측하지 않고 “답을 찾을 수 없음”이라고 응답

연구진은 약 4,400만 개의 생의학 문서를 수집했으며, BioASQ, LitQA, MedQA, GeneTuring 등의 데이터셋과 내부 질문 데이터셋을 사용해 시스템을 평가했습니다.

### 3. 주요 결과
전문가가 평가한 내부 데이터셋에서 Med.ai ASK는 높은 정확도와 안정성을 보였습니다. 또한 대규모 언어 모델을 평가자로 사용한 결과가 인간 전문가의 평가와 대체로 일치했습니다.

특히 긴 형식의 답변에서 다음과 같은 성능이 우수했습니다.

- 정확성
- 사실성
- 검색된 근거와 답변의 일치도
- 인용의 신뢰성
- 환각 현상 감소

단답형 및 객관식 문제에서는 최신 시스템들과 경쟁력 있는 성능을 보였습니다. 또한 질문의 유형에 따라 적절한 도구를 자동으로 선택했으며, 실제 서비스에 배포되어 1,600명 이상의 사용자와 25,000건 이상의 질문을 처리했습니다.

### 4. 결론
Med.ai ASK는 여러 생의학 정보원을 상황에 맞게 조정하고 결합하는 에이전트 기반 RAG 시스템입니다.  
특히 근거와 인용이 중요한 생의학 분야에서, 단순한 LLM보다 더 신뢰할 수 있고 해석 가능한 답변을 제공할 가능성을 보여주었습니다.

---




### Objective
Med.ai ASK is an agentic question-answering system designed to answer biomedical questions. Instead of relying on a single retrieval source, it analyzes each question and selectively uses multiple biomedical databases and tools to provide accurate and evidence-grounded answers.

### Methods
The system combines the ReAct tool-calling framework with reasoning ideas from Self-Discover. It can automatically select appropriate tools, retrieve information from multiple sources, combine the results, and generate answers with supporting citations. The system was built using approximately 44 million biomedical documents and evaluated on several biomedical question-answering datasets.

### Results
Med.ai ASK showed strong accuracy and stability in human evaluations. It performed particularly well on long-form answers, achieving better results in accuracy, factuality, faithfulness to retrieved evidence, citation quality, and hallucination reduction. Its performance on short-answer and multiple-choice questions was competitive with leading systems. The system was also deployed in a production chatbot used by more than 1,600 users, answering over 25,000 questions.

### Conclusion
Med.ai ASK is an agent-based RAG system that dynamically coordinates multiple biomedical information sources and tools. Its ability to select tools, provide citations, and avoid unsupported answers makes it especially useful for reliable and interpretable biomedical question answering.




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


### 1. 전체 접근법: 에이전트 기반 RAG
Med.ai ASK는 단일 LLM이 기억만으로 답하는 방식이 아니라, 질문에 따라 여러 생의학 도구와 지식베이스를 선택적으로 호출하는 에이전트형 검색증강생성(RAG) 시스템이다.

전체 흐름은 다음과 같다.

1. 사용자가 생의학 질문을 입력한다.
2. LLM 기반 Tool orchestration 모듈이 질문 유형을 분석하고 사용할 도구를 선택한다.
3. 선택된 도구들이 관련 문헌·데이터를 검색한다.
4. 검색 결과를 바탕으로 각 도구가 후보 답변을 생성한다.
5. 최종 집계 모듈이 후보 답변을 통합하고, 근거 문서와 함께 최종 답변을 생성한다.
6. 관련 근거를 찾지 못하면 LLM의 일반 지식으로 추측하지 않고 “No answer could be found”라고 응답한다.

이 구조는 생의학 분야에서 중요한 정확성, 근거성, 환각 감소, 해석 가능성을 높이기 위한 것이다.

---

### 2. 기반 언어모델과 에이전트 아키텍처

- 기반 모델: GPT-4.1
- 별도의 태스크별 파인튜닝은 수행하지 않았다.
- 따라서 BioASQ, LitQA, MedQA, GeneTuring 등 여러 데이터셋에 대해 재학습 없이 적용 가능한 데이터셋 비의존적(dataset-agnostic) 구조를 사용한다.

에이전트 구조는 다음 두 가지 아이디어를 결합했다.

#### ReAct 기반 도구 호출
ReAct의 tool-calling 방식을 이용해 LLM이 다음을 결정한다.

- 어떤 도구를 사용할지
- 도구에 어떤 검색 질의를 전달할지
- 도구를 추가로 호출할지
- 충분한 정보가 모였는지, 검색을 중단할지

#### Self-Discover 기반 추론 구조
Self-Discover의 “필요한 추론 구조를 선택한 뒤 실행한다”는 개념을 QA에 맞게 단순화했다.

- 도구 선택(select)
- 도구 실행 및 결과 수집
- 결과 통합 및 최종 답변 생성

원래 Self-Discover의 adapt 단계는 제거했으며, 도구 실행에 필요한 파라미터를 미리 설정했다.

---

### 3. 지식베이스와 검색 도구

총 약 4,400만 건의 생의학 문서를 다양한 내부 지식베이스와 외부 API에서 통합했다.

#### 텍스트 지식베이스
- PubMed: 약 3,800만 건의 생의학 문헌 인용정보
- CCC OpenAccess: 약 500만 건의 전문 생의학 논문
- Elsevier e-books: 약 2만 권의 생물학 관련 전자책
- NIH Grant Database: 약 100만 건의 NIH 연구비 과제
- FDA Drug Labels: 15만 건 이상의 의약품 라벨 문서

#### API 기반 도구
- ClinicalTrials.gov API: 임상시험 및 결과 정보
- UniProt API: 단백질 정보

#### 추가 분석 도구
- Biomedical NER 도구: 생의학 개체명 인식 및 정규화
- Kazu 프레임워크를 사용해 질병, 유전자, 약물 등 12개 범주의 개체를 식별
- NCBI 유전자 데이터를 활용해 유전자 동의어 확장도 수행

중요하게도, 이 문서들은 LLM을 재학습하기 위한 training data라기보다, 질문에 답할 때 검색하는 외부 지식 및 retrieval corpus로 사용되었다.

---

### 4. 문서 색인 및 검색

문서 검색에는 Weaviate 벡터 데이터베이스를 사용했다.

#### 문서 처리
- 짧은 문서, 예를 들어 PubMed 초록은 하나의 chunk로 저장
- 긴 문서는 다음과 같이 분할:
  - chunk 크기: 약 8,000자
  - overlap: 약 1,000자

각 chunk는 m-GTE 임베딩 모델로 벡터화한 뒤 Weaviate에 저장했다.

사용자 질문이 입력되면 질문도 임베딩되고, 의미적으로 관련성이 높은 문서 chunk가 검색된다.

---

### 5. Map-Reduce 기반 답변 생성

각 검색 도구에는 별도의 QA-RAG 모듈을 구현했다.

#### Map 단계
검색된 각각의 문서를 LLM에 입력하여 문서별 후보 답변을 만든다.

- 각 문서가 질문에 어떤 정보를 제공하는지 분석
- 문서 수준의 답변 또는 근거 생성

#### Reduce 단계
문서별 후보 답변을 하나로 통합한다.

- 여러 문서의 내용을 요약·비교
- 중복 정보 제거
- 서로 다른 근거를 종합
- 최종 답변의 각 주장에 PubMed ID 등 문서 식별자를 인라인으로 연결

따라서 답변에는 단순한 텍스트뿐 아니라 사용자가 확인할 수 있는 인용과 클릭 가능한 출처 링크가 포함된다.

---

### 6. 메모리와 대화 처리

대화 세션별로 사용자의 질문과 시스템 답변을 PostgreSQL 데이터베이스에 저장한다.

새로운 질문이 들어오면 같은 세션의 이전 대화 내용을 검색해 현재 질문의 문맥으로 첨부한다. 이를 통해 다음을 지원한다.

- 다중 턴 대화
- 이전 질문과 답변을 고려한 후속 질문 처리
- 사용자가 나중에 이전 세션을 재개하는 기능

다만 메모리 구조는 복잡한 장기기억 모델이라기보다, 대화 기록을 저장하고 검색하는 단순한 세션 기반 메모리이다.

---

### 7. 학습 및 파인튜닝 방식

이 연구에서 Med.ai ASK는 학습 데이터로 모델을 추가 훈련하거나 태스크별 파인튜닝을 하지 않았다.

대신 다음 방법을 사용했다.

- GPT-4.1을 기반 모델로 사용
- 시스템 프롬프트와 도구 설명을 제공
- 질문에 맞는 검색 도구를 동적으로 선택
- 검색된 문서를 컨텍스트로 제공
- RAG와 map-reduce 방식으로 답변 생성

즉, 성능 향상의 핵심은 모델 자체를 재학습하는 것보다 다음에 있다.

- 다양한 전문 지식베이스
- 질문에 따른 도구 선택
- 검색 결과의 단계적 통합
- 출처 기반 답변 생성
- 근거가 없을 때 답변을 거부하는 설계

---

### 8. 시스템 구현 및 배포

- 전체 워크플로는 LangGraph로 구현
- 프론트엔드–프록시 백엔드–에이전트 백엔드 구조
- REST API와 WebSocket을 사용
- 클라우드 Kubernetes 환경에서 운영
- 도구 선택 과정과 중간 검색 결과를 사용자에게 표시하여 추론 및 검색 과정의 해석 가능성을 높임

---



### 1. Overall approach: Agentic RAG

Med.ai ASK is an agentic retrieval-augmented generation system for biomedical question answering. Instead of answering solely from the language model’s internal knowledge, it dynamically selects and invokes biomedical retrieval and analysis tools.

The workflow is:

1. The user submits a biomedical question.
2. An LLM-based tool orchestration module analyzes the question.
3. The agent selects one or more appropriate tools.
4. The tools retrieve relevant documents or structured information.
5. Each tool produces a candidate answer.
6. An aggregation module synthesizes the candidate answers into a final response with citations.
7. If no relevant evidence is found, the system returns “No answer could be found” rather than relying on unsupported model knowledge.

This design aims to improve factuality, faithfulness, interpretability, and resistance to hallucination.

---

### 2. Language model and agent architecture

- Base model: GPT-4.1
- No task-specific fine-tuning was performed.
- The system is therefore dataset-agnostic and can be applied to different biomedical QA datasets without retraining.

The architecture combines two ideas:

#### ReAct-style tool calling
The agent decides:

- Which tools to use
- What queries to send to the tools
- Whether additional tool calls are needed
- When sufficient information has been collected

#### Self-Discover-inspired reasoning
The system adapts the idea of selecting and executing a suitable reasoning structure for each question.

The original three-step Self-Discover process was simplified:

- Tool selection
- Tool execution and result collection
- Final answer synthesis

The adaptation step was removed because the required tool parameters were preconfigured.

---

### 3. Knowledge bases and tools

The system integrates approximately 44 million biomedical documents and records.

#### Internal textual knowledge bases

- PubMed: approximately 38 million biomedical citations
- CCC OpenAccess: approximately 5 million full-text biomedical papers
- Elsevier e-books: approximately 20,000 biology-related books
- NIH Grant Database: approximately 1 million research grants
- FDA Drug Labels: more than 150,000 drug-labeling documents

#### External API tools

- ClinicalTrials.gov API for clinical trial information
- UniProt API for protein information

#### Additional analysis tools

- A biomedical named entity recognition (NER) tool based on the Kazu framework
- Recognition and normalization of biomedical entities across 12 categories
- Gene synonym expansion using NCBI gene data

These documents are used as a retrieval corpus, not as training data for fine-tuning the language model.

---

### 4. Document indexing and retrieval

Documents are indexed in Weaviate, a vector database.

- Short documents, such as PubMed abstracts, are stored as single chunks.
- Long documents are divided into chunks of approximately 8,000 characters.
- Adjacent chunks overlap by approximately 1,000 characters.
- Each chunk is embedded using the m-GTE embedding model.

When a question is submitted, the system retrieves semantically relevant document chunks for the query.

---

### 5. Map-reduce answer generation

Each retriever is connected to a QA-RAG module.

#### Map step
Each retrieved document is independently passed to an LLM to generate a document-level answer or evidence summary.

#### Reduce step
The document-level answers are aggregated and synthesized into a final answer.

The reduce process:

- Combines evidence from multiple documents
- Removes redundancy
- Integrates complementary findings
- Attaches document identifiers, such as PubMed IDs, to individual claims

The final response includes citations and clickable source links to support user verification.

---

### 6. Memory and multi-turn conversations

The system stores every question and answer from a chat session in a PostgreSQL database.

For a new question, previous questions and answers from the same session are retrieved and added as context. This supports:

- Multi-turn conversations
- Context-aware follow-up questions
- Resuming previous chat sessions

The memory component is a simple session-level retrieval mechanism rather than a complex long-term memory architecture.

---

### 7. Training and fine-tuning

Med.ai ASK does not use additional model training or task-specific fine-tuning.

Instead, it relies on:

- GPT-4.1 as the foundation model
- System prompts and tool descriptions
- Dynamic tool selection
- Retrieval of relevant biomedical evidence
- Map-reduce answer synthesis
- Citation-grounded generation
- Explicit refusal when no supporting evidence is available

Thus, the main source of improvement is the agentic retrieval and orchestration design rather than modification of the underlying language model.

---

### 8. Implementation and deployment

- Implemented using LangGraph
- Frontend connected to a proxy backend and an agent backend
- REST APIs and WebSockets are used for communication
- Deployed on a cloud-based Kubernetes cluster
- Intermediate tool selections and retrieved information are shown to users to improve transparency and interpretability


<br/>
# Results



### 1. 비교 대상 모델과 평가 데이터

Med.ai ASK는 다음 모델 및 시스템과 비교되었다.

- 기본 비교 모델
  - LLaMA 3.1-8B-Instruct
  - GPT-o3
  - GPT-4.1
- 바이오메디컬 QA 비교 시스템
  - MiBi: BioASQ 2024의 상위 성능 RAG 시스템
- 외부 SOTA 모델
  - Med-PaLM 2
  - Med-Gemini
  - 일부 데이터셋에서는 기존 연구의 LLaMA 및 Claude 모델

평가에는 다음 5개 데이터셋이 사용되었다.

| 데이터셋 | 질문 유형 | 주요 평가 목적 |
|---|---|---|
| BioASQ-12B | 장문형 답변 | 바이오메디컬 문헌 기반 QA의 정확성·사실성·환각 평가 |
| LitQA-v2 | 객관식 | 최신 생명과학 논문에서 근거를 찾아 답하는 능력 |
| MedQA-USMLE | 의학 객관식 | 의사국가시험 유형의 의료 지식 평가 |
| GeneTuring | 단답·개체명 중심 | 유전체 및 유전자 지식 평가 |
| In-house dataset | 장문형, 20문항 | 실제 조직 내 사용 사례에 대한 전문가 평가 |

---

### 2. 평가 메트릭

연구에서는 질문 유형에 따라 서로 다른 지표를 사용했다.

- 정확도(Accuracy): 정답과 생성 답변이 일치하는 정도
- LLM 기반 정확도: GPT-4.1이 생성 답변과 기준 답변을 비교해 0~1점으로 평가
- OLAPH 지표
  - 단어 구성
  - 의미적 유사도
  - 사실성(Factuality)
  - 환각(Hallucination)
- RAGAs 지표
  - Faithfulness: 답변이 검색 문서에 의해 실제로 뒷받침되는지
  - Context precision: 검색된 문서 중 관련 문서의 비율
  - Context recall: 정답에 필요한 정보가 검색되었는지
- 인간 평가
  - 답변 정확도
  - 여러 번 실행했을 때 답변의 안정성
  - 인용 문서의 관련성 및 충실성

---

## 3. 주요 결과

### A. 내부 전문가 평가

20개의 사내 질문에 대해 Med.ai ASK의 답변을 평가한 결과:

- 전문가 평가 정확도: 평균 84.0% ± 2.5
- LLM 평가 정확도: 평균 83.7% ± 1.9
- 두 평가 결과가 상당히 유사하여, 이후 실험에서 LLM 기반 평가를 활용할 수 있음을 확인했다.
- 인용 문서 평가:
  - 관련성: 89.1% ± 2.4
  - 충실성: 82.0% ± 3.5

다만, 인간과 LLM 평가가 항상 일치한 것은 아니다. 예를 들어 기준 답변과 다른 방식으로 수치를 제시했지만 내용상 타당한 경우, 인간 전문가는 높게 평가한 반면 LLM은 낮게 평가했다.

---

### B. BioASQ-12B: 장문형 바이오메디컬 QA

BioASQ-12B에서는 Med.ai ASK가 가장 강한 성능을 보였다.

- LLaMA 3.1, GPT-o3, GPT-4.1 및 MiBi보다 전반적으로 우수
- 특히 다음 지표에서 높은 성능:
  - LLM 기반 정확도
  - 사실성
  - 낮은 환각률
  - 문서 기반 충실성
  - 검색 문맥의 precision 및 recall
- 인용 문서 평가에서도 MiBi보다 우수한 성능을 보였다.
- 23개 세부 주제 중 대부분에서 세 가지 기본 모델보다 환각이 적었으며, 일부 주제에서는 환각이 전혀 발생하지 않았다.
  - Virology
  - Vaccines
  - Physiology
  - Pharmacovigilance & Adverse Effects
  - Cell Biology
  - Biochemistry

즉, 장문 답변이 필요한 문헌 기반 질문에서 ASK의 가장 큰 장점은 검색 근거를 활용하면서도 사실성과 인용 신뢰도를 유지한 것이다.

---

### C. LitQA-v2: 최신 논문 기반 객관식

LitQA-v2에서는 Med.ai ASK가 비교 모델과 기존 SOTA 시스템을 모두 앞섰다.

| 모델 | Precision | Accuracy |
|---|---:|---:|
| LLaMA 3.1-8B | 35.0 | 28.5 |
| GPT-o3 | 44.3 | 44.0 |
| GPT-4.1 | 42.0 | 41.9 |
| Med.ai ASK | 65.1 | 43.9 |

- 정밀도(Precision): 65.1로 가장 높았다.
- 정확도는 43.9로 GPT-o3의 44.0과 거의 유사하지만, 다른 모델보다 높았다.
- 특히 초록만으로는 답하기 어려운 최신 논문 정보에 대해, 전체 문헌 검색과 다중 도구 사용이 도움이 된 것으로 해석된다.

---

### D. MedQA-USMLE: 의료 객관식

MedQA에서는 ASK가 LLaMA 3.1보다 높았지만, 최첨단 모델보다는 낮거나 비슷한 수준이었다.

| 모델 | 정확도 |
|---|---:|
| LLaMA 3.1-8B | 50.8 |
| Med.ai ASK | 86.2 |
| Med-PaLM 2 | 86.5 |
| GPT-4.1 | 89.1 |
| GPT-o3 | 96.1 |
| Med-Gemini | 91.1 |

- ASK는 86.2%로 LLaMA 3.1보다 크게 높았다.
- Med-PaLM 2의 86.5%와는 거의 비슷했다.
- 그러나 GPT-4.1, GPT-o3, Med-Gemini보다는 낮았다.

논문은 MedQA의 일부 문항이 오래된 정답이나 불충분한 임상 정보를 포함하고 있어, 단순히 검색만으로는 최고 성능을 내기 어렵다고 설명한다. 또한 ASK는 정답을 추측하기보다 근거와 불확실성을 설명하는 경향이 있어, 객관식 정확도만으로는 장점이 충분히 반영되지 않을 수 있다.

---

### E. GeneTuring: 유전체 지식 평가

GeneTuring에서는 ASK가 기본 비교 모델보다 우수했지만, Med-Gemini보다는 낮았다.

| 모델 | 정확도 |
|---|---:|
| LLaMA 3.1-8B | 22.5 |
| GPT-o3 | 35.2 |
| GPT-4.1 | 32.4 |
| Med.ai ASK | 41.8 |
| Med-Gemini | 54.5 |
| 기존 연구 모델 | 51.2 |

- ASK는 세 가지 기본 모델보다 높았다.
- 그러나 Med-Gemini 및 기존 최고 성능 모델에는 미치지 못했다.
- GeneTuring에서는 일반 문헌 검색보다 NER 도구와 UniProt의 활용이 중요했다.
- 실제로 GeneTuring에서 가장 많이 사용된 도구는:
  - NER: 30.4%
  - PubMed: 29.3%
  - UniProt: 21.7%

저자들은 유전자 온톨로지와 같은 추가 지식원을 연결하면 성능을 높일 수 있다고 보았다.

---

## 4. 도구 추가에 따른 성능 변화

BioASQ 개발 세트에서 도구를 순차적으로 추가한 결과:

| 도구 구성 | 정확도 |
|---|---:|
| 도구 없음 | 92.2 |
| PubMed | 96.1 |
| PubMed + CCC | 97.9 |
| + Elsevier | 98.1 |
| + UniProt | 98.4 |
| + NIH Grant | 98.2 |
| + ClinicalTrials.gov | 98.5 |
| + FDA | 98.2 |
| + NER | 97.8 |

주요 해석은 다음과 같다.

- PubMed 추가 효과가 가장 컸다.
- PubMed 외에 여러 도구를 추가하면 대체로 성능이 향상되었다.
- 그러나 모든 도구를 무조건 추가한다고 성능이 계속 올라가지는 않았다.
- NER까지 모두 추가한 구성에서는 오히려 최고점보다 낮아졌다.
- 이는 도구가 많아질수록 LLM의 도구 선택 과정이 복잡해져 일시적인 혼란이 생길 수 있음을 보여준다.

---

## 5. 데이터셋별 도구 사용 경향

- BioASQ-12B: PubMed 사용 비율이 가장 높음
- LitQA-v2: PubMed와 CCC OpenAccess 중심
- MedQA: PubMed 중심
- GeneTuring: NER, PubMed, UniProt 사용이 두드러짐
- NIH Grant와 ClinicalTrials.gov: 전체적으로 사용 빈도는 낮았지만 특정 질문에 유용
- FDA Drug Labels: 실제 의약품 관련 질문에서 활용 가능성이 높음

---

## 6. 종합 평가

Med.ai ASK는 모든 데이터셋에서 최고 성능을 기록한 것은 아니지만, 다음과 같은 강점을 보였다.

1. 장문형 문헌 기반 질문에서 가장 강함
   - BioASQ에서 정확성, 사실성, 충실성, 환각 억제 측면이 우수했다.

2. 검색 근거와 인용을 명확하게 제시
   - 답변에 문서 ID와 클릭 가능한 출처를 포함했다.
   - 사용자가 답변을 직접 검증할 수 있다.

3. 질문에 따라 도구를 자동 선택
   - PubMed, 전문서적, FDA 라벨, 임상시험, UniProt, NER 등을 선택적으로 사용한다.

4. 객관식 최고 정확도만을 목표로 하지 않음
   - 근거가 부족하면 억지로 답하지 않고, “No answer could be found”라고 응답하도록 설계되었다.
   - 따라서 일부 객관식 벤치마크에서는 최고 모델보다 낮을 수 있지만, 실제 바이오메디컬 연구 환경에서는 더 안전하고 해석 가능한 답변을 제공하는 것을 목표로 한다.

5. 실제 운영 성과
   - 2026년 1월 기준:
     - 1,600명 이상의 사용자
     - 25,000건 이상의 질문 처리
   - 연구자들은 문헌 검색, 위험 평가, 관련 논문 탐색 등에 활용했다고 보고했다.

---




## Summary of Results

### 1. Compared Models and Evaluation Datasets

Med.ai ASK was compared with:

- LLaMA 3.1-8B-Instruct
- GPT-o3
- GPT-4.1
- MiBi, a strong RAG system from BioASQ 2024
- Other reported systems, including Med-PaLM 2 and Med-Gemini

The evaluation used five datasets:

| Dataset | Question type | Main purpose |
|---|---|---|
| BioASQ-12B | Long-form | Biomedical literature-based QA |
| LitQA-v2 | Multiple choice | Retrieval from recent biology research papers |
| MedQA-USMLE | Multiple choice | Medical licensing examination questions |
| GeneTuring | Span-based/entity answers | Genomic and gene-related knowledge |
| In-house dataset | Long-form, 20 questions | Expert assessment in real organizational use cases |

---

### 2. Evaluation Metrics

The study used different metrics depending on the task:

- Accuracy
- LLM-based accuracy
- OLAPH metrics
  - Word composition
  - Semantic similarity
  - Factuality
  - Hallucination
- RAGAs metrics
  - Faithfulness
  - Context precision
  - Context recall
- Human evaluation
  - Answer accuracy
  - Stability across repeated runs
  - Citation relevance
  - Citation faithfulness

---

## 3. Main Results

### A. Human Evaluation

On the 20-question in-house dataset:

- Expert-rated accuracy: 84.0% ± 2.5
- LLM-rated accuracy: 83.7% ± 1.9
- Citation relevance: 89.1% ± 2.4
- Citation faithfulness: 82.0% ± 3.5

The close agreement between human and LLM ratings supported the use of an LLM-based evaluator in later experiments. However, disagreements occurred when the system gave a valid answer in a format different from the reference answer.

---

### B. BioASQ-12B

BioASQ-12B was the strongest evaluation setting for Med.ai ASK.

Compared with LLaMA 3.1, GPT-o3, GPT-4.1, and MiBi, ASK generally achieved better results in:

- LLM-based accuracy
- Factuality
- Hallucination reduction
- Faithfulness
- Context precision
- Context recall

ASK also outperformed MiBi in citation-related evaluation. It produced no hallucinations in several topic areas, including:

- Virology
- Vaccines
- Physiology
- Pharmacovigilance and adverse effects
- Cell biology
- Biochemistry

These results indicate that ASK is particularly effective for long-form biomedical questions requiring evidence-grounded answers.

---

### C. LitQA-v2

Med.ai ASK outperformed the compared systems on both precision and accuracy.

| System | Precision | Accuracy |
|---|---:|---:|
| LLaMA 3.1-8B | 35.0 | 28.5 |
| GPT-o3 | 44.3 | 44.0 |
| GPT-4.1 | 42.0 | 41.9 |
| Med.ai ASK | 65.1 | 43.9 |

ASK achieved the highest precision and nearly the highest accuracy. This suggests that its retrieval tools were useful for answering questions based on information found in recent full-text scientific papers rather than only abstracts.

---

### D. MedQA-USMLE

ASK performed better than LLaMA 3.1 and was close to Med-PaLM 2, but it did not reach the best-performing models.

| System | Accuracy |
|---|---:|
| LLaMA 3.1-8B | 50.8 |
| Med.ai ASK | 86.2 |
| Med-PaLM 2 | 86.5 |
| GPT-4.1 | 89.1 |
| GPT-o3 | 96.1 |
| Med-Gemini | 91.1 |

ASK achieved 86.2%, almost matching Med-PaLM 2. However, GPT-o3, Med-Gemini, and GPT-4.1 performed better.

The authors noted that some MedQA questions contain outdated answers or insufficient clinical information. In addition, ASK is designed to explain evidence and uncertainty rather than simply guess an option, which may not be fully reflected by multiple-choice accuracy.

---

### E. GeneTuring

ASK outperformed the three general baseline models but remained below Med-Gemini and the best reported systems.

| System | Accuracy |
|---|---:|
| LLaMA 3.1-8B | 22.5 |
| GPT-o3 | 35.2 |
| GPT-4.1 | 32.4 |
| Med.ai ASK | 41.8 |
| Med-Gemini | 54.5 |
| Previous best reported system | 51.2 |

The NER and UniProt tools were especially important for GeneTuring. The most frequently used tools for this dataset were:

- NER: 30.4%
- PubMed: 29.3%
- UniProt: 21.7%

The authors suggested that connecting additional resources such as Gene Ontology could improve performance.

---

## 4. Ablation Study: Effect of Adding Tools

On the BioASQ development set:

| Tool configuration | Accuracy |
|---|---:|
| No tool | 92.2 |
| PubMed | 96.1 |
| PubMed + CCC | 97.9 |
| + Elsevier | 98.1 |
| + UniProt | 98.4 |
| + NIH Grant | 98.2 |
| + ClinicalTrials.gov | 98.5 |
| + FDA | 98.2 |
| + NER | 97.8 |

The main findings were:

- Adding PubMed produced the largest improvement.
- Additional tools generally improved performance.
- However, adding every available tool did not always lead to better results.
- The full configuration including NER performed slightly worse than the best configuration.
- This suggests that too many tools may increase complexity and occasionally confuse the LLM’s tool-selection process.

---

## 5. Overall Interpretation

Med.ai ASK did not achieve the highest score on every benchmark, but it showed several important strengths:

1. Strong performance on long-form biomedical QA
   - Especially on BioASQ, it provided accurate, factual, and well-grounded answers.

2. Reliable citation and evidence use
   - Answers include document identifiers and clickable references.

3. Automatic tool selection
   - The system selects among PubMed, full-text papers, books, FDA labels, clinical trials, UniProt, and NER depending on the question.

4. Emphasis on reliability over guessing
   - If no relevant evidence is found, the system can respond, “No answer could be found,” instead of relying only on the LLM’s internal knowledge.

5. Real-world deployment
   - By January 2026, the platform had served more than 1,600 users and answered more than 25,000 questions.

Overall, Med.ai ASK is best characterized as a retrieval-grounded, tool-using biomedical research assistant. Its main advantage is not universally superior multiple-choice accuracy, but rather its combination of factuality, citation quality, interpretability, and adaptability across biomedical information sources.


<br/>
# 예제
## 1. 연구에서 말하는 “학습 데이터”의 의미

Med.ai ASK는 질문·정답 데이터로 별도 파인튜닝(fine-tuning)되지 않은 시스템이다. 논문은 “It does not rely on any training data”라고 명시한다. 따라서 아래 데이터셋의 train/dev/test 분할은 모델을 학습시키기 위한 것이 아니라, 벤치마크 평가용 표준 분할로 사용되었다.

- 기반 LLM: GPT-4.1
- 별도 학습: 없음
- 질문이 들어오면 PubMed, FDA Drug Labels, ClinicalTrials.gov, UniProt 등의 도구를 선택하고 검색한 뒤 답변 생성
- 평가 시에는 각 데이터셋의 test set 질문만 사용
- 모든 결과는 3회 독립 실행 평균

---

## 2. Med.ai ASK의 전체 입력–처리–출력 구조

### 입력

사용자가 자연어로 biomedical 질문을 입력한다.

예시 형식:

> “What are the primary and secondary endpoints of Pamrevlumab?”

또는

> “What is the five-year mortality rate of 특정 질환?”

질문은 다음과 같은 형태일 수 있다.

- 문헌 기반 장문 질문
- 객관식 문제
- 특정 개체명이나 유전자명을 요구하는 단답형 질문
- 임상시험, 약물, 단백질, 유전자 등에 대한 질의

### 처리 과정

1. 질문 분석
   - 질문의 주제와 필요한 정보 유형을 파악한다.
2. 도구 선택
   - PubMed, CCC OpenAccess, Elsevier e-books, NIH Grant Database, FDA Drug Labels
   - ClinicalTrials.gov API, UniProt API
   - biomedical NER 도구 중 적절한 도구를 선택한다.
3. 검색 및 후보 답변 생성
   - 검색된 문서를 바탕으로 각 문서별 후보 답변을 만든다.
4. 답변 통합
   - 여러 후보 답변을 종합해 최종 답변을 작성한다.
5. 인용 추가
   - PubMed ID 등 문서 식별자를 답변에 포함한다.
6. 근거가 없을 경우
   - LLM의 일반 지식으로 추측하지 않고  
     “No answer could be found”라고 출력한다.

### 출력

최종 출력은 보통 다음을 포함한다.

- 질문에 대한 설명형 답변
- 문장별 또는 주장별 출처
- 클릭 가능한 인용 링크
- 사용한 도구와 검색 결과
- 경우에 따라 중간 처리 과정(Stream of thoughts로 표시되는 도구 선택 및 결과)

즉, 단순히 정답만 출력하는 것이 아니라 근거와 해석을 함께 제공하는 장문 답변을 목표로 한다.

---

## 3. 데이터셋별 입력과 출력

### A. BioASQ-12B

#### 과제

생의학 문헌을 기반으로 한 장문(long-form) 질의응답이다.

#### 데이터 규모

| 구분 | 문항 수 |
|---|---:|
| Train | 5,049 |
| Dev | 503 |
| Test | 340 |

ASK는 이 데이터를 학습하지 않고, 340개의 test 질문에 답변한 뒤 정답과 비교했다.

#### 입력 예시

논문에 언급된 질문 유형의 예시는 다음과 같다.

> “What are the primary and secondary endpoints of Pamrevlumab?”

이 질문은 임상시험 관련 문헌과 ClinicalTrials.gov 등의 자료를 확인해야 하는 유형이다.

#### 기대 출력

단순한 한 단어가 아니라 다음과 같은 형태의 답변이 요구된다.

- 1차 평가변수(primary endpoint)
- 2차 평가변수(secondary endpoint)
- 각 항목의 설명
- 관련 논문 또는 임상시험 출처

논문에서는 ASK가 오래된 정답 데이터에 포함된 정보보다 더 최신인 2024~2025년 자료를 검색해, 여러 임상시험의 평가변수를 더 포괄적으로 제시한 사례를 언급한다.

#### 평가 방식

- LLM-based accuracy
- factuality
- hallucination
- faithfulness
- context precision
- context recall
- 문장 유사도 및 의미 유사도

BioASQ-12B에서는 ASK가 기본 모델들과 MiBi보다 대체로 높은 factuality와 낮은 hallucination을 보였다.

---

### B. LitQA-v2

#### 과제

생물학 연구 논문에서만 확인할 수 있는 정보를 묻는 객관식 문제다. 초록만으로는 답하기 어려운 최신 연구 결과가 포함된다.

#### 데이터 규모

| 구분 | 문항 수 |
|---|---:|
| Train | 없음 |
| Dev | 없음 |
| Test | 199 |

#### 입력 예시

구체적인 문항 원문은 논문 본문에 제시되지 않았지만, 형식은 다음과 같다.

> “Which of the following findings was reported in the relevant biology paper?”  
> A. …  
> B. …  
> C. …  
> D. …

#### 기대 출력

- 선택지 중 정답
- 가능하면 해당 정답을 뒷받침하는 논문 근거
- 왜 해당 선택지가 맞는지에 대한 설명

ASK는 단순 정답 선택뿐 아니라, 근거 문헌을 찾아 답을 설명하도록 설계되었다.

#### 결과

- ASK 정확도: 43.9%
- ASK precision: 65.1%
- 비교 대상 모델보다 높은 accuracy와 precision을 기록

---

### C. MedQA-USMLE

#### 과제

미국 의사면허시험(USMLE) 형식의 의학 객관식 문제다.

#### 데이터 규모

| 구분 | 문항 수 |
|---|---:|
| Train | 10,178 |
| Dev | 1,272 |
| Test | 1,273 |

논문에서는 미국 부분집합인 USMLE만 사용했다.

#### 입력 예시

일반적인 형태는 다음과 같다.

> 환자의 증상, 병력, 검사 결과가 제시된다.  
> “What is the most likely diagnosis?”  
> A. 질환 1  
> B. 질환 2  
> C. 질환 3  
> D. 질환 4

#### 기대 출력

- 가장 적절한 선택지
- 선택 이유
- 가능하면 관련 의학 근거

그러나 MedQA 문항 중 일부는 검사 결과나 그림 등 핵심 정보가 부족하거나 정답이 오래된 경우가 있어, ASK는 질문 자체가 충분하지 않다고 설명하기도 했다.

#### 결과

- ASK 정확도: 86.2%
- Med-PaLM 2와 경쟁력 있는 수준
- Med-Gemini와 GPT-o3보다는 낮음

---

### D. GeneTuring

#### 과제

유전체 지식에 대한 단답형 또는 개체명 추출형(span-based) 질문이다. 답은 보통 특정 유전자, 단백질 또는 생물학적 개체명이다.

#### 데이터 규모

| 구분 | 문항 수 |
|---|---:|
| Train | 없음 |
| Dev | 없음 |
| Test | 600 |

12개 유전체 관련 범주를 평가한다.

#### 입력 예시

구체적인 원문은 논문에 제시되지 않았지만, 일반적인 형식은 다음과 같다.

> “Which gene is associated with the described genomic function?”  
> 또는  
> “What is the gene symbol for the described entity?”

#### 기대 출력

> “BRCA1”

처럼 특정 유전자명이나 생물학적 개체를 출력하는 방식이다.

#### ASK의 처리 방식

GeneTuring에서는 NER 도구 사용 비율이 가장 높았다.

- NER 사용 비율: 30.4%
- UniProt 사용 비율: 21.7%
- PubMed 사용 비율: 29.3%

이는 단순 문헌 검색보다는 개체명 인식, 표준화, 유전자 동의어 확장이 중요하기 때문이다.

#### 결과

- ASK 정확도: 41.8%
- 기본 LLM 3종보다 높음
- Med-Gemini의 54.5%보다는 낮음

---

### E. In-house Dataset

#### 과제

Johnson & Johnson 내부 사용자가 실제로 제출한 biomedical 질문을 평가하는 실사용 기반 장문 QA 과제다.

#### 데이터 규모

- Test: 20개 질문
- 공개되지 않음
- 이전 ASK 버전에서 부정적 피드백을 받은 질문들로 구성
- 각 질문은 한 명의 도메인 전문가가 정답을 작성하고 다른 전문가가 검토

#### 입력 예시

논문에서 소개된 질문 유형은 다음과 같다.

1. 특정 질환의 5년 사망률
2. 가장 많이 인용된 논문
3. 특정 약물이나 임상시험의 주요 평가변수

#### 기대 출력

- 정답 정보
- 질문의 요구사항에 맞는 세부 분류
- 관련 문헌 인용
- 필요할 경우 불확실성 또는 데이터 한계 설명

예를 들어, 정답이 연령별 사망률을 요구하는데 ASK가 지역별 사망률을 제시하면, 내용이 어느 정도 맞더라도 질문과 정확히 일치하지 않는 것으로 평가될 수 있다.

#### 결과

- 전문가 평가 정확도: 84% ± 2.5
- LLM 평가 정확도: 83.7% ± 1.9
- 인용문헌 relevance: 89.1%
- 인용문헌 faithfulness: 82%

---

## 4. 도구별 입력과 출력 예시

| 도구 | 입력 | 출력 |
|---|---|---|
| PubMed | 질환, 치료법, 생물학적 기전에 대한 검색어 | 논문 초록, PMID, 관련 문헌 |
| CCC OpenAccess | 전문 논문 검색어 | 논문 본문 일부 또는 관련 문단 |
| Elsevier e-books | 교과서·전문서적 기반 질문 | 서적 내 관련 설명 |
| FDA Drug Labels | 약물명, 이상반응, 적응증 | 허가사항, 용량, 경고, 부작용 |
| NIH Grant Database | 연구 분야, 질환, 기술 | 관련 연구비 과제 |
| ClinicalTrials.gov | 약물명, 질환, 임상시험 질문 | 임상시험 상태, endpoint, 연구 설계 |
| UniProt | 단백질명, 유전자명 | 단백질 기능, 주석, 서열 관련 정보 |
| NER/Kazu | 질문 속 유전자·질환·약물명 | 표준화된 biomedical entity |

예를 들어 질문에 특정 유전자명이 여러 동의어로 나타나면, NER 도구가 이를 표준 유전자명으로 정규화한 뒤 UniProt 또는 PubMed 검색으로 연결할 수 있다.

---

## 5. 핵심 정리

Med.ai ASK의 평가에서 중요한 점은 다음과 같다.

1. 데이터셋의 train set으로 ASK를 학습시키지 않았다.
2. 각 데이터셋의 질문을 입력으로 받아 외부·내부 biomedical knowledge base를 검색했다.
3. 질문 유형에 따라 여러 도구 중 필요한 도구를 자동으로 선택했다.
4. 문서별 답변을 만든 뒤 이를 종합해 최종 답변을 생성했다.
5. 답변에는 출처와 인용을 포함했다.
6. 근거를 찾지 못하면 추측하지 않고 “No answer could be found”라고 답했다.
7. 장문 문헌 기반 질문에서는 정확성, factuality, faithfulness, hallucination 측면에서 강점을 보였다.
8. 객관식·유전체 단답형 문제에서는 경쟁력은 있었지만 최고 성능 모델보다는 낮은 경우도 있었다.

---




## 1. Meaning of “training data” in this study

Med.ai ASK was not fine-tuned on the training questions and answers from the benchmark datasets. The paper explicitly states that it “does not rely on any training data.”

Therefore, the train/dev/test splits were used for benchmark evaluation, not for training ASK.

- Foundational LLM: GPT-4.1
- Additional fine-tuning: none
- Retrieval sources: PubMed, FDA Drug Labels, ClinicalTrials.gov, UniProt, and other biomedical databases
- Evaluation: mainly performed on the test sets
- Results: averaged across three independent runs

---

## 2. Overall input–processing–output workflow

### Input

A user or benchmark dataset provides a biomedical question.

Examples include:

> “What are the primary and secondary endpoints of Pamrevlumab?”

or

> “What is the five-year mortality rate of a specific disease?”

The questions may be:

- Long-form literature-based questions
- Multiple-choice medical questions
- Short-answer genomic questions
- Questions about drugs, proteins, genes, or clinical trials

### Processing

1. The agent analyzes the question.
2. It selects appropriate tools.
3. It retrieves relevant documents or API results.
4. Each retrieved document is used to generate a candidate answer.
5. Candidate answers are aggregated into a final answer.
6. Citations such as PubMed IDs are attached to the claims.
7. If no relevant evidence is found, the system returns:

> “No answer could be found.”

### Output

The final output generally contains:

- A direct answer
- Explanation and interpretation
- Inline citations or clickable source links
- Information about the tools used
- Sometimes intermediate retrieval results shown through the user interface

The system is therefore designed to produce an evidence-supported and interpretable answer, rather than only a short answer.

---

## 3. Dataset-specific inputs and outputs

### A. BioASQ-12B

#### Task

Long-form biomedical question answering based mainly on scientific literature.

#### Dataset size

| Split | Number of questions |
|---|---:|
| Train | 5,049 |
| Development | 503 |
| Test | 340 |

ASK did not train on the 5,049 training questions. It answered the 340 test questions for evaluation.

#### Example input

One question type mentioned in the paper is:

> “What are the primary and secondary endpoints of Pamrevlumab?”

#### Expected output

The answer should identify:

- The primary endpoint
- The secondary endpoint
- An explanation of each endpoint
- Supporting papers or clinical-trial sources

The paper notes that ASK sometimes retrieved newer and more comprehensive information than the outdated reference answer, including information from 2024 and 2025 sources.

#### Evaluation

The system was evaluated using:

- LLM-based accuracy
- Factuality
- Hallucination
- Faithfulness
- Context precision
- Context recall
- Word composition and semantic similarity

ASK generally performed better than the baselines and MiBi on factuality and hallucination-related measures.

---

### B. LitQA-v2

#### Task

Multiple-choice questions about findings reported in biology research papers. Many answers cannot be obtained from abstracts alone.

#### Dataset size

| Split | Number of questions |
|---|---:|
| Train | Not available |
| Development | Not available |
| Test | 199 |

#### Example input format

The exact question text was not included in the paper, but the format is approximately:

> “Which of the following findings was reported in the relevant biology paper?”  
> A. …  
> B. …  
> C. …  
> D. …

#### Expected output

- The correct option
- An explanation
- Supporting evidence from the relevant scientific paper

#### Result

- ASK accuracy: 43.9%
- ASK precision: 65.1%

ASK outperformed the listed baseline and state-of-the-art systems on the reported accuracy and precision measures.

---

### C. MedQA-USMLE

#### Task

Multiple-choice medical questions derived from US medical licensing examinations.

#### Dataset size

| Split | Number of questions |
|---|---:|
| Train | 10,178 |
| Development | 1,272 |
| Test | 1,273 |

Only the USMLE subset was used.

#### Example input format

A typical question has the following structure:

> A patient’s symptoms, medical history, and test results are described.  
> “What is the most likely diagnosis?”  
> A. Disease 1  
> B. Disease 2  
> C. Disease 3  
> D. Disease 4

#### Expected output

- The most likely answer choice
- A medical explanation
- Supporting evidence where available

Some MedQA questions lack important information, such as figures or laboratory results. In such cases, ASK may explain that the question is under-specified instead of confidently guessing.

#### Result

- ASK accuracy: 86.2%
- Competitive with Med-PaLM 2
- Lower than GPT-o3 and Med-Gemini

---

### D. GeneTuring

#### Task

Short-answer or span-based genomic question answering. The answer is usually a gene, protein, or other biological entity.

#### Dataset size

| Split | Number of questions |
|---|---:|
| Train | Not available |
| Development | Not available |
| Test | 600 |

The benchmark covers 12 genomic knowledge categories.

#### Example input format

The exact questions were not shown in the paper, but the format is similar to:

> “Which gene is associated with the described genomic function?”

or

> “What is the gene symbol for the described entity?”

#### Expected output

A concise entity answer, such as:

> “BRCA1”

#### Tool usage

GeneTuring relied heavily on entity recognition and normalization.

- NER usage: 30.4%
- PubMed usage: 29.3%
- UniProt usage: 21.7%

This indicates that identifying and normalizing gene names and synonyms was especially important for this task.

#### Result

- ASK accuracy: 41.8%
- Better than the three baseline models
- Lower than Med-Gemini’s 54.5%

---

### E. In-house dataset

#### Task

Real-world biomedical questions submitted by internal users.

#### Dataset size and construction

- 20 questions
- Not publicly available
- Questions previously received negative feedback in earlier ASK versions
- Each reference answer was written by one domain expert and checked by another

#### Example question types

The paper mentions questions involving:

1. Five-year mortality rates for a disease
2. The most-cited papers
3. Primary and secondary endpoints of a drug trial

#### Expected output

The answer should:

- Address the exact question
- Provide the requested level of detail
- Cite relevant documents
- Clearly distinguish evidence and limitations

For example, if the question asks for mortality rates by age group but the system provides rates by geographic region, the response may be generally relevant but still fail to fully answer the question.

#### Results

- Expert-rated accuracy: 84% ± 2.5
- LLM-rated accuracy: 83.7% ± 1.9
- Citation relevance: 89.1%
- Citation faithfulness: 82%

---

## 4. Examples of tool inputs and outputs

| Tool | Example input | Example output |
|---|---|---|
| PubMed | Disease, treatment, or mechanism query | Abstracts, PMIDs, related papers |
| CCC OpenAccess | Full-text paper query | Relevant passages from full papers |
| Elsevier e-books | Textbook or specialist knowledge question | Relevant book content |
| FDA Drug Labels | Drug name or adverse-effect query | Indications, warnings, dosing, adverse effects |
| NIH Grant Database | Disease or research-area query | Relevant funded projects |
| ClinicalTrials.gov | Drug, disease, or endpoint query | Trial status, design, endpoints |
| UniProt | Protein or gene name | Protein function and annotations |
| NER/Kazu | Biomedical entities in the question | Normalized genes, diseases, drugs, and other entities |

For example, if a gene appears under several synonyms, the NER tool can normalize it to a standard gene name before searching PubMed or UniProt.

---

## 5. Key takeaway

Med.ai ASK was evaluated by giving it biomedical questions as inputs and asking it to retrieve evidence, reason over the retrieved information, and generate cited answers.

The main characteristics are:

1. The benchmark training sets were not used to fine-tune ASK.
2. The system dynamically selects biomedical retrieval tools.
3. It generates document-level candidate answers and then aggregates them.
4. It provides citations and interpretable evidence.
5. It avoids unsupported guessing by returning “No answer could be found” when necessary.
6. It was particularly strong on long-form literature-based questions.
7. It was competitive, but not always state of the art, on multiple-choice and genomic short-answer tasks.

<br/>
# 요약


Med.ai ASK는 ReAct와 Self-Discover를 바탕으로 PubMed, FDA 의약품 라벨, ClinicalTrials.gov, UniProt 등 44 million개 문서를 검색하고, 질문에 따라 여러 도구를 선택해 map-reduce 방식으로 근거 기반 답변을 생성한다.  
BioASQ와 LitQA에서는 정확성·사실성·인용 신뢰도·환각 억제 측면에서 기존 기준 모델보다 우수했으며, MedQA와 GeneTuring에서도 경쟁력 있는 성능을 보였다.  
예를 들어 문헌 질문에는 PubMed·Elsevier를, 유전자 관련 질문에는 NER·UniProt를 주로 활용하며, 1600명 이상의 사용자가 25,000건 이상의 질문에 이 시스템을 사용했다.  



Med.ai ASK is built on ReAct and Self-Discover, using multiple tools—including PubMed, FDA drug labels, ClinicalTrials.gov, and UniProt—to retrieve evidence from 44 million documents and generate answers through map-reduce reasoning.  
It outperformed baseline systems on BioASQ and LitQA in accuracy, factuality, citation reliability, and hallucination control, while achieving competitive results on MedQA and GeneTuring.  
For example, it mainly uses PubMed and Elsevier for literature questions and NER and UniProt for gene-related questions, and has been used by more than 1,600 users to answer over 25,000 questions.

<br/>
# 기타



아래는 본문에서 언급된 그림·표·부록의 결과와 핵심 인사이트를 중심으로 정리한 것입니다. 제공된 내용에 부록의 세부 수치가 모두 포함되어 있지는 않아, 확인 가능한 범위에서만 설명했습니다.

---

# 1. 그림(Figures)

## Figure 1. Med.ai ASK 에이전트 아키텍처

### 결과
- 사용자의 질문이 들어오면 Tool orchestration 모듈이 질문 유형과 도구 설명을 바탕으로 사용할 도구를 선택한다.
- Tool node가 선택된 도구를 반복적으로 호출한다.
- 각 도구가 반환한 결과를 집계·추론하여 최종 답변을 생성한다.
- 시스템에는 다음 세 위치에서 LLM이 사용된다.
  1. 도구 선택 및 오케스트레이션
  2. 검색 결과의 통합·요약
  3. 각 검색 도구의 RAG 답변 생성
- 이전 대화는 별도 메모리 DB에 저장되어 후속 질문의 문맥으로 활용된다.

### 핵심 인사이트
- 단일 검색기를 사용하는 일반적인 RAG와 달리, ASK는 질문에 따라 PubMed, FDA, UniProt, ClinicalTrials.gov, NER 등 여러 도구를 동적으로 조합한다.
- 답을 찾지 못했을 때 LLM의 일반 지식으로 추측하지 않고 “No answer could be found”라고 반환하도록 설계해, 생의학 분야에서 환각 위험을 줄였다.
- 검색, 도구 선택, 답변 생성을 분리한 구조이므로 새로운 데이터베이스나 도구를 추가하기 쉽다.

---

## Figure 2. 사용자 인터페이스와 Stream of Thoughts

### 결과
- 최종 답변과 함께 출처 및 클릭 가능한 인용 링크가 제공된다.
- 사용자는 “Stream of thoughts”를 통해 사용된 도구와 각 도구의 결과를 확인할 수 있다.
- 이전 질문과 답변은 세션에 저장되어 다중 턴 대화에 활용된다.

### 핵심 인사이트
- ASK의 설명 가능성은 내부적인 모든 사고 과정을 공개한다기보다, 어떤 도구를 사용했고 어떤 정보가 반환되었는지 보여주는 방식에 기반한다.
- 인용된 문서를 직접 확인할 수 있어, 연구자가 답변을 검증하기 쉽다.
- 중간 결과를 보여주는 방식은 답변의 신뢰성과 감사 가능성을 높이지만, 답변이 다른 시스템보다 더 길고 복잡해질 수 있다.

---

## Figure 3. BioASQ-12B 성능 비교

### 결과
- ASK는 LLaMA 3.1, GPT-o3, GPT-4.1 및 MiBi와 비교했을 때 다음 항목에서 전반적으로 우수했다.
  - LLM 기반 정확도
  - 사실성(Factuality)
  - 환각 감소
- 단어 구성(Word Composition)과 의미적 유사도(Semantic Similarity)에서는 항상 가장 높은 점수를 얻지는 못했다.

### 핵심 인사이트
- ASK는 정답 문장과 표현이 얼마나 비슷한지를 최대화하기보다는, 근거에 기반한 정확하고 사실적인 답변을 만드는 데 강점이 있다.
- 생의학 장문 QA에서는 단순한 문장 유사도보다 사실성, 근거성, 환각 억제가 더 중요한 평가 기준이라는 점을 보여준다.

---

## Figure 4. 주제별 환각률

### 결과
- ASK는 23개 주제 중 18개에서 세 가지 기본 모델보다 낮은 환각률을 보였다.
- 특히 다음 6개 주제에서는 환각이 관찰되지 않았다.
  - Virology
  - Vaccines
  - Physiology
  - Pharmacovigilance & Adverse Effects
  - Cell Biology
  - Biochemistry
- Epigenetics에서는 ASK가 모든 시스템보다 우수하지 않았다.
- MiBi와 비교해도 전반적으로 비슷하거나 더 나은 환각 억제 성능을 보였다.

### 핵심 인사이트
- 도구 기반 검색과 인용이 모든 주제에서 동일한 수준으로 작동하는 것은 아니다.
- 특정 분야에서는 검색 데이터와 질문의 정합성이 높아 환각이 크게 줄지만, 정보가 부족하거나 질문이 복잡한 분야에서는 여전히 취약할 수 있다.
- 전체 평균뿐 아니라 주제별 성능 분석이 실제 배포에 중요하다.

---

## Figure 5. 인용 문서 평가

### 결과
- ASK는 MiBi보다 다음 세 지표에서 일관되게 우수했다.
  - Faithfulness: 인용 문서가 답변 내용을 실제로 뒷받침하는 정도
  - Context Precision: 검색된 문서 중 유용한 문서의 비율
  - Context Recall: 답변에 필요한 정보가 검색 결과에 포함된 정도

### 핵심 인사이트
- ASK는 단순히 많은 문서를 검색하는 것이 아니라, 답변에 실제로 기여하는 문서를 검색하고 그 내용을 정확하게 반영하는 능력이 강했다.
- 생의학 QA에서 답변의 정확도뿐 아니라 “이 답이 어떤 문서에 의해 뒷받침되는가”가 중요하다는 점을 강조한다.

---

## Figure 6. 사용자 활용 현황

### 결과
- 2026년 1월 기준:
  - 고유 사용자: 1,600명 이상
  - 누적 질문: 25,000건 이상
- 6개 내부 팀이 자체 데이터나 솔루션과의 통합에 관심을 보였다.

### 핵심 인사이트
- ASK는 실험실 수준의 프로토타입이 아니라 대규모 조직에서 실제 사용된 운영 시스템이다.
- 사용자 피드백은 문헌 검색 시간 절감, 위험 평가 지원, 핵심 문헌 탐색, 사고 구조화 등에 유용했음을 시사한다.
- 다만 사용자 수와 질문 수는 활용도를 보여주는 지표이지, 그 자체로 답변의 임상적 정확성을 증명하는 것은 아니다.

---

# 2. 표(Tables)

## Table 1. 평가 데이터셋 구성

| 데이터셋 | 답변 유형 | 테스트 규모 | 주요 평가 대상 |
|---|---:|---:|---|
| BioASQ-12B | 장문형 | 340 | 생의학 장문 QA |
| LitQA-v2 | 객관식 | 199 | 최신 논문 기반 생물학 지식 |
| MedQA USMLE | 객관식 | 1,273 | 의학 면허시험 유형 |
| GeneTuring | 단답·엔터티형 | 600 | 유전체 지식 |
| In-house | 장문형 | 20 | 실제 사용자 질문 |

### 핵심 인사이트
- 하나의 데이터셋에만 최적화하지 않고, 장문형·객관식·엔터티 기반 질문을 모두 평가했다.
- ASK는 특정 벤치마크에 대한 파인튜닝 없이 여러 데이터셋에 적용되므로 dataset-agnostic 시스템이라는 점을 검증하려는 구성이다.
- In-house 데이터셋은 규모는 작지만 실제 조직의 어려운 질문을 반영한다.

---

## Table 2. LitQA-v2, MedQA, GeneTuring 성능

### 결과

| 시스템 | LitQA 정확도 | MedQA 정확도 | GeneTuring 정확도 |
|---|---:|---:|---:|
| LLaMA 3.1-8B | 28.5 | 50.8 | 22.5 |
| GPT-o3 | 44.0 | 96.1 | 35.2 |
| GPT-4.1 | 41.9 | 89.1 | 32.4 |
| Med.ai ASK | 43.9 | 86.2 | 41.8 |
| Med-PaLM 2 | - | 86.5 | - |
| Med-Gemini | - | 91.1 | 54.5 |

LitQA의 precision은 ASK가 65.1로 모든 비교 시스템보다 높았다.

### 핵심 인사이트
- LitQA-v2: ASK가 정확도와 precision 모두에서 가장 우수했다. 최신 논문의 세부 정보를 검색하는 능력이 강하다는 의미다.
- MedQA: ASK는 LLaMA 3.1보다 높았지만 Med-PaLM 2와 비슷한 수준이며, GPT-o3·Med-Gemini에는 미치지 못했다.
- GeneTuring: 기본 모델보다 개선되었지만 Med-Gemini보다 낮았다.
- 문헌 검색만으로는 교과서 지식이나 Gene Ontology와 같은 전문 지식이 필요한 문제를 충분히 해결하기 어렵다.

---

## Table 3. 도구 추가에 따른 절제 연구

### 결과

| 도구 구성 | 정확도 |
|---|---:|
| 도구 없음 | 92.2 |
| PubMed | 96.1 |
| PubMed + CCC | 97.9 |
| + Elsevier | 98.1 |
| + UniProt | 98.4 |
| + NIH Grant | 98.2 |
| + ClinicalTrials.gov | 98.5 |
| + FDA | 98.2 |
| + NER | 97.8 |

### 핵심 인사이트
- PubMed를 추가했을 때 가장 큰 성능 향상이 나타났다.
- PubMed 이후에도 여러 도구를 추가하면 대체로 성능이 높아졌지만, 모든 추가가 항상 유익한 것은 아니었다.
- ClinicalTrials.gov까지 포함한 구성이 가장 높은 평균 정확도를 보였다.
- FDA와 NER를 추가한 최종 구성에서는 오히려 점수가 낮아졌다. 이는 도구가 많아질수록 LLM의 도구 선택이 혼란스러워질 수 있음을 의미한다.
- 따라서 도구의 개수보다 질문에 맞는 도구 선택과 검색 결과의 품질이 중요하다.

---

## Table 4. 데이터셋별 도구 사용률

### 주요 결과

| 도구 | BioASQ | LitQA | MedQA | GeneTuring |
|---|---:|---:|---:|---:|
| PubMed | 28.8 | 44.2 | 46.9 | 29.3 |
| CCC | 25.2 | 33.3 | 27.3 | 18.0 |
| Elsevier | 22.3 | 11.9 | 13.9 | 0.5 |
| UniProt | 5.5 | 2.2 | 0.9 | 21.7 |
| NIH Grant | 0.1 | 0 | 0 | 0 |
| ClinicalTrials.gov | 2.0 | 0 | 0 | 0 |
| FDA | 6.6 | 0.1 | 1.1 | 0 |
| NER | 9.5 | 8.3 | 9.9 | 30.4 |

### 핵심 인사이트
- PubMed는 BioASQ, LitQA, MedQA에서 가장 핵심적인 도구였다.
- GeneTuring에서는 NER가 가장 많이 사용되었고, UniProt도 상대적으로 높은 사용률을 보였다.
- 질문 유형에 따라 도구 선택이 달라지는 것이 확인되었다.
  - 문헌 기반 질문 → PubMed, CCC, Elsevier
  - 유전자·단백질 질문 → NER, UniProt
  - 임상시험·규제 질문 → ClinicalTrials.gov, FDA
- NIH Grant와 ClinicalTrials.gov는 전체 벤치마크에서는 사용률이 낮았지만, 실제 기업 연구 환경에서는 특정 질문에 중요한 도구가 될 수 있다.

---

# 3. 부록(Appendices)

제공된 본문에는 부록의 전체 내용과 세부 결과표가 포함되어 있지 않으므로, 본문에서 확인되는 역할과 인사이트를 중심으로 정리하면 다음과 같습니다.

## Appendix A. 에이전트 및 RAG 프롬프트

### 포함 내용
- A1: 에이전트의 전체 프롬프트
- A2: 검색 문서별 답변을 생성하는 map 단계 프롬프트
- A3: 여러 문서 답변을 통합하는 reduce 단계 프롬프트
- A4: LLM 기반 정확도 평가 프롬프트

### 핵심 인사이트
- ASK의 성능은 모델 자체뿐 아니라 도구 선택, 문서별 답변 생성, 최종 통합을 분리한 프롬프트 설계에 크게 의존한다.
- 답변 문장마다 문서 ID를 붙이도록 하여 인용 가능성과 검증 가능성을 높였다.
- LLM-as-a-judge를 사용할 때 평가 기준을 명시적으로 고정하려는 구조다.

---

## Appendix B. In-house 질문 템플릿

### 포함 내용
- 실제 사용자 제출 질문 20개와 관련 질문 유형의 템플릿

### 핵심 인사이트
- 내부 데이터셋은 단순한 사실 조회뿐 아니라 문헌 비교, 수치 해석, 최신 정보 확인 등 실제 연구 업무에 가까운 질문을 포함한다.
- 다만 20개 질문에 불과하고 비공개 데이터이므로 일반화에는 제한이 있다.

---

## Appendix C. BioASQ 주제 분류

### 포함 내용
- 37개 생물학 주제 중 각 질문에 최대 3개 주제를 부여
- LLM이 분류한 뒤 전문가가 검증

### 핵심 인사이트
- 전체 평균 점수만으로는 어떤 분야에서 강하고 약한지 알기 어렵기 때문에, 주제별 환각률과 검색 성능을 분석하기 위한 장치다.
- Disease & Symptoms, Treatments, Clinical Medicine, Oncology, Pharmacology가 주요 질문 영역이었다.
- Figure 4의 주제별 환각 분석을 뒷받침한다.

---

## Appendix D. RAG 파라미터 및 도구별 성능

### 포함 내용
- 최적 RAG 파라미터 선택 과정
- 개별 도구의 성능 분석

### 핵심 인사이트
- ASK는 학습 데이터를 이용한 파인튜닝이 아니라, 검색 및 생성 설정을 조정하는 방식으로 성능을 개선했다.
- Table 3의 도구 누적 절제 연구를 해석하는 기준을 제공한다.
- 개별 도구의 성능은 전체 시스템에서의 사용률과 반드시 같지 않다. 자주 사용되는 도구가 항상 가장 높은 성능 향상을 제공하는 것은 아니다.

---

## Appendix E. BioASQ 주제별 세부 평가

### 포함 내용
- 주제별 정확도, 사실성, 환각률, 인용 관련 지표 등의 세부 결과

### 핵심 인사이트
- ASK의 성능이 모든 주제에서 동일하지 않다는 점을 보여준다.
- 주제별 데이터의 양과 최신성, 검색 가능한 문헌의 특성에 따라 성능 차이가 발생한다.
- 실제 배포 시에는 전체 평균 성능보다 업무에 중요한 특정 분야의 세부 성능을 별도로 확인해야 한다.

---

## Appendix F. GeneTuring 12개 범주별 결과

### 포함 내용
- GeneTuring의 12개 유전체 지식 범주에 대한 세부 성능

### 핵심 인사이트
- GeneTuring 전체 점수만으로는 ASK의 유전체 지식 강점과 약점을 충분히 알 수 없다.
- NER 사용률이 높았다는 결과와 함께, 유전자명 정규화 및 동의어 확장이 특정 범주에서 중요한 역할을 했을 가능성을 보여준다.
- 그러나 Gene Ontology 등 문헌 외 지식원이 부족해 Med-Gemini와의 격차가 남았다.

---

## Appendix G. 정성적 분석

### G1. 불충분한 정보가 있는 객관식 문제
- 질문에 필요한 그림, 검사 결과, 임상 정보 등이 없으면 ASK가 문제의 불완전성을 지적하고 설명할 수 있다.
- 무조건 하나의 답을 추측하는 대신, 문제 자체의 한계와 필요한 추가 정보를 제시한다는 점이 장점이다.

### G2. 답변 길이와 최신성 사례
- ASK 답변은 다른 시스템보다 더 장황할 수 있지만, 중간 도구 사용 과정과 인용을 포함해 검증 가능성이 높다.
- Pamrevlumab 임상시험 사례에서는 기존 정답보다 최신인 2024~2025년 정보를 활용해 더 포괄적인 답변을 생성했다.
- 그러나 오래된 정답과 비교하는 자동 평가에서는 이러한 최신 정보가 환각으로 잘못 분류될 수 있다.

### 핵심 인사이트
- ASK는 정답 하나를 빠르게 맞히는 시스템보다는, 근거와 불확실성을 함께 제시하는 연구 지원 도구에 가깝다.
- 평가 데이터셋의 정답이 오래되었거나 불완전하면 실제로 더 나은 답변이 낮은 점수를 받을 수 있다.
- 장문 답변과 상세한 인용은 해석 가능성을 높이지만, 간결성을 희생할 수 있다.

---

# 종합 인사이트

1. ASK의 가장 큰 강점은 장문형 생의학 QA에서의 근거성, 사실성, 환각 억제다.
2. PubMed가 가장 중요한 기반 도구였지만, 질문 유형에 따라 NER, UniProt, FDA, ClinicalTrials.gov 등이 보완적으로 필요하다.
3. 도구를 많이 추가한다고 항상 성능이 좋아지는 것은 아니며, 도구 선택의 정확성과 검색 결과의 품질이 중요하다.
4. ASK는 객관식 벤치마크의 최고 성능보다는 출처가 명확하고 검증 가능한 연구용 답변을 우선한다.
5. MedQA와 GeneTuring 결과는 문헌 검색만으로는 교과서, 임상 지침, Gene Ontology 등 외부 지식 영역을 충분히 다루기 어렵다는 점을 보여준다.
6. 실제 운영 환경에서 1,600명 이상의 사용자와 25,000건 이상의 질문을 처리했다는 점은 시스템의 실용성을 뒷받침하지만, 추가적인 임상적 검증은 필요하다.

---




## 1. Figures

## Figure 1. Med.ai ASK agent architecture

### Results
- The tool-orchestration component selects tools based on the question and tool descriptions.
- The tool node iteratively invokes the selected tools.
- The system aggregates and reasons over the returned results to generate the final answer.
- LLMs are used in three main locations:
  1. Tool orchestration
  2. Aggregation and summarization
  3. RAG answer generation for individual retrievers
- Previous questions and answers are stored in memory and reused as context in later turns.

### Key insight
- Unlike a conventional single-retriever RAG system, ASK dynamically combines PubMed, FDA labels, UniProt, ClinicalTrials.gov, NER, and other tools.
- If no relevant evidence is found, the system returns “No answer could be found” instead of relying on the LLM’s general knowledge. This is particularly important for reducing hallucinations in biomedical applications.
- The modular design makes it relatively easy to add new tools or knowledge bases.

---

## Figure 2. User interface and Stream of Thoughts

### Results
- The final answer is accompanied by citations and clickable source links.
- Users can inspect the tools invoked and the information returned through the “Stream of thoughts” interface.
- Previous interactions are retained for multi-turn conversations.

### Key insight
- Explainability is provided mainly by showing the tools used and their outputs, rather than exposing every internal reasoning step.
- Clickable citations allow researchers to verify the answer directly.
- This improves transparency and auditability, although the resulting answers may be longer than those of other systems.

---

## Figure 3. BioASQ-12B performance

### Results
- ASK generally outperformed LLaMA 3.1, GPT-o3, GPT-4.1, and MiBi in:
  - LLM-based accuracy
  - Factuality
  - Hallucination reduction
- It did not consistently achieve the best scores for word composition or semantic similarity.

### Key insight
- ASK is optimized more for evidence-based and factually grounded answers than for matching the wording or semantic form of the reference answer.
- In biomedical long-form QA, factuality and groundedness may be more meaningful than surface-level answer similarity.

---

## Figure 4. Topic-level hallucination rates

### Results
- ASK showed lower hallucination rates than the three baseline models in 18 of 23 topics.
- It produced no hallucinations in:
  - Virology
  - Vaccines
  - Physiology
  - Pharmacovigilance and adverse effects
  - Cell biology
  - Biochemistry
- ASK was not the best-performing system for Epigenetics.
- Overall, it was similar to or slightly better than MiBi in hallucination control.

### Key insight
- Tool-based retrieval does not work equally well across all biomedical topics.
- Topic-level performance analysis is important because overall averages may hide weaknesses in specific research areas.

---

## Figure 5. Citation evaluation

### Results
ASK consistently outperformed MiBi on:
- Faithfulness
- Context precision
- Context recall

### Key insight
- ASK does not merely retrieve a large number of documents. It is better at retrieving documents that are relevant to the answer and accurately reflecting their content.
- This is important because biomedical answer quality depends not only on correctness, but also on whether the cited sources genuinely support the claims.

---

## Figure 6. User engagement

### Results
As of January 2026:
- More than 1,600 unique users
- More than 25,000 submitted questions
- Six internal teams expressed interest in integration with proprietary data or solutions.

### Key insight
- ASK has been deployed as a production-level research support system rather than remaining a laboratory prototype.
- User feedback suggests benefits for literature search, risk assessment, identifying critical papers, and structuring research thinking.
- Usage statistics demonstrate adoption, but they do not by themselves establish clinical-level accuracy.

---

# 2. Tables

## Table 1. Evaluation datasets

| Dataset | Answer type | Test size | Main purpose |
|---|---:|---:|---|
| BioASQ-12B | Long-form | 340 | Biomedical long-form QA |
| LitQA-v2 | Multiple choice | 199 | Recent biology literature |
| MedQA USMLE | Multiple choice | 1,273 | Medical licensing-style questions |
| GeneTuring | Span/entity-based | 600 | Genomic knowledge |
| In-house | Long-form | 20 | Real-world internal questions |

### Key insight
- The evaluation covers long-form, multiple-choice, and entity-based questions rather than relying on a single benchmark.
- This supports the claim that ASK is dataset-agnostic and does not require task-specific fine-tuning.
- The in-house dataset is small but reflects realistic organizational use cases.

---

## Table 2. LitQA-v2, MedQA, and GeneTuring

### Results
- ASK achieved the best LitQA-v2 precision and accuracy among the compared systems.
- On MedQA, ASK outperformed LLaMA 3.1 but was slightly below Med-PaLM 2 and clearly below GPT-o3 and Med-Gemini.
- On GeneTuring, ASK outperformed the three baseline models but remained below Med-Gemini.

### Key insight
- ASK is particularly effective for questions requiring retrieval of detailed information from recent scientific papers.
- Literature retrieval alone is less sufficient for questions requiring textbook knowledge, clinical reasoning, or resources such as Gene Ontology.

---

## Table 3. Ablation study on tool addition

### Results
- No tool: 92.2 accuracy
- PubMed: 96.1
- PubMed + CCC: 97.9
- Adding Elsevier and UniProt further improved performance.
- The highest score was obtained with the configuration including ClinicalTrials.gov: 98.5.
- Adding FDA and NER to the full configuration reduced the score to 98.2 and 97.8, respectively.

### Key insight
- PubMed produced the largest individual improvement.
- Adding more tools generally helped, but not every additional tool improved performance.
- Too many tools may introduce confusion in tool selection.
- Therefore, appropriate tool selection and retrieval quality matter more than simply increasing the number of tools.

---

## Table 4. Tool usage by dataset

### Results
- PubMed was the most frequently used tool for BioASQ, LitQA, and MedQA.
- NER was the most frequently used tool for GeneTuring.
- UniProt was also relatively important for GeneTuring.
- NIH Grant and ClinicalTrials.gov had low usage in the benchmark datasets.

### Key insight
- Tool selection varies according to question type:
  - Literature questions: PubMed, CCC, Elsevier
  - Gene and protein questions: NER and UniProt
  - Clinical trial or regulatory questions: ClinicalTrials.gov and FDA
- Low usage in benchmark datasets does not imply that NIH Grant or ClinicalTrials.gov are unimportant in real pharmaceutical research workflows.

---

# 3. Appendices

The full appendix contents and detailed numerical results were not included in the provided text. Their roles and main implications are as follows.

## Appendix A. Agent and RAG prompts

### Contents
- A1: Main agent prompt
- A2: Map-stage prompt for document-level answers
- A3: Reduce-stage prompt for aggregating document-level answers
- A4: LLM-based accuracy evaluation prompt

### Key insight
- ASK’s performance depends not only on the underlying LLM but also on the prompt design for tool selection, document-level answering, and final aggregation.
- Inline document identifiers improve citation and verification.
- The evaluation prompt provides a standardized framework for the LLM-as-a-judge procedure.

---

## Appendix B. In-house question templates

### Key insight
- The internal questions represent realistic research tasks, including comparison, numerical interpretation, and up-to-date evidence retrieval.
- However, the dataset contains only 20 questions and is confidential, so its generalizability is limited.

---

## Appendix C. BioASQ topic assignment

### Key insight
- Each question was assigned up to three topics from 37 biology-related categories, with expert validation.
- This enables topic-level analysis of hallucinations and retrieval performance.
- Major topics included Disease and Symptoms, Treatments, Clinical Medicine, Oncology, and Pharmacology.

---

## Appendix D. RAG parameters and individual tool analysis

### Key insight
- ASK improved performance through retrieval and generation configuration rather than task-specific model fine-tuning.
- The appendix helps interpret the cumulative tool ablation study.
- A tool’s individual performance is not necessarily the same as its usage frequency or its contribution to the full system.

---

## Appendix E. Topic-stratified BioASQ results

### Key insight
- ASK’s performance varies across biomedical topics.
- Differences may reflect the amount, recency, and accessibility of evidence available in the indexed sources.
- For deployment, performance on task-specific topics may be more important than the overall average.

---

## Appendix F. GeneTuring category-level results

### Key insight
- The overall GeneTuring score does not fully describe ASK’s strengths and weaknesses across genomic categories.
- The high usage of NER suggests that entity normalization and gene-synonym expansion are important for some categories.
- The remaining gap with Med-Gemini indicates that literature-based retrieval does not fully replace resources such as Gene Ontology.

---

## Appendix G. Qualitative analysis

### G1. Multiple-choice questions with missing information
- ASK can identify when a question lacks essential details such as figures, laboratory results, or clinical context.
- Rather than blindly guessing, it can explain why the question is under-specified and what additional information is needed.

### G2. Answer verbosity and up-to-date evidence
- ASK may produce longer answers, but these include intermediate tool information and extensive citations.
- In the Pamrevlumab example, ASK used newer 2024–2025 evidence and produced a more comprehensive answer than an outdated reference answer.
- However, automatic evaluation may incorrectly classify such updated information as hallucination when the ground truth is outdated.

### Key insight
- ASK is designed more as an evidence-based research assistant than as a system that simply predicts one short answer.
- Outdated or incomplete benchmark references can penalize answers that are actually more current and comprehensive.
- Greater transparency and citation coverage come at the cost of increased answer length.

---

## Overall takeaway

1. ASK’s main strength is grounded, factual, and low-hallucination performance in long-form biomedical QA.
2. PubMed is the core retrieval source, while NER, UniProt, FDA, and ClinicalTrials.gov provide complementary capabilities.
3. Adding tools does not always improve performance; intelligent tool selection is critical.
4. ASK prioritizes verifiable, cited research answers over maximizing multiple-choice benchmark scores.
5. Its weaker MedQA and GeneTuring results show the limitations of relying primarily on biomedical literature without textbooks, clinical guidelines, or ontology resources.
6. Its deployment scale demonstrates practical utility, but additional clinical and domain-specific validation is still required.

<br/>
# refer format:



### BibTeX

```bibtex
@article{nguyen2026medai,
  author  = {Nguyen, Nhung T. H. and Lituiev, Dmytro S. and Liu, Zhimin and
             Kashyap, Aditya and Jenkinson, Garrett and Kuhl, Kevin and
             Corrado, Christopher and Patel, Naisargi Manishkumar and
             Snigdha, Kirti and Saeedi, Sirwe and Smith, David and
             Baro, Nicholas and Schultz, Timothy},
  title   = {{Med.ai ASK}: An Agentic System for Biomedical Question Answering},
  journal = {Journal of the American Medical Informatics Association},
  year    = {2026},
  volume  = {33},
  number  = {6},
  pages   = {1134--1145},
  doi     = {10.1093/jamia/ocag038},
  url     = {https://doi.org/10.1093/jamia/ocag038},
  note    = {Published March 30, 2026}
}
```

### Chicago Style  

Nguyen, Nhung T. H., Dmytro S. Lituiev, Zhimin Liu, Aditya Kashyap, Garrett Jenkinson, Kevin Kuhl, Christopher Corrado, Naisargi Manishkumar Patel, Kirti Snigdha, Sirwe Saeedi, David Smith, Nicholas Baro, and Timothy Schultz. “Med.ai ASK: An Agentic System for Biomedical Question Answering.” *Journal of the American Medical Informatics Association* 33, no. 6 (2026): 1134–1145. https://doi.org/10.1093/jamia/ocag038.




