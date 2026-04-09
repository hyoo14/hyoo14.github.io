---
layout: post
title:  "[2025]Remote Labor Index: Measuring AI Automation of Remote Work"
date:   2026-04-09 20:55:43 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문은 실제 프리랜서 원격 업무 240개 프로젝트를 모은 **Remote Labor Index(RLI)**를 만들고, 인간이 만든 정답 산출물과 AI 산출물을 **수동 평가**해 자동화율과 Elo를 측정하는 방법을 제시합니다.


짧은 요약(Abstract) :




이 논문은 **Remote Labor Index (RLI)** 라는 새로운 벤치마크를 소개합니다.  
RLI는 AI가 실제로 **원격 근로(remote work)** 를 얼마나 자동화할 수 있는지를 측정하기 위해 만든 평가 체계입니다.

기존 AI 벤치마크는 주로 지식, 추론, 코딩 같은 제한된 능력을 보는 경우가 많아서, 그것이 실제 경제적 가치가 있는 일까지 얼마나 이어지는지는 잘 알 수 없었습니다.  
RLI는 이런 한계를 보완하기 위해, **실제 프리랜서 시장에서 수행된 일감들**을 바탕으로 만들어졌습니다. 즉, 단순한 문제풀이가 아니라, 사람이 실제로 돈을 받고 수행한 프로젝트와 그 결과물을 기준으로 AI를 평가합니다.

논문에 따르면, 현재의 AI 에이전트들은 RLI에서 **매우 낮은 성능**을 보였고, 가장 잘한 모델도 **자동화율 2.5%** 에 그쳤습니다.  
이는 현재 AI가 다양한 종류의 원격 업무를 실제로 대체하기에는 아직 크게 부족하다는 뜻입니다.

다만, 더 세밀한 비교를 위해 Elo 기반의 쌍대 비교 평가를 사용했는데, 이를 통해 모델 간에는 점진적인 성능 향상이 관찰되었습니다.  
즉, **절대 성능은 매우 낮지만**, 최신 모델일수록 조금씩 더 나아지고 있다는 점은 확인됩니다.

결론적으로 이 논문은, AI 자동화 논의를 추상적인 주장 대신 **실증적 데이터**로 뒷받침할 수 있는 기준을 제시하며, 앞으로 AI가 노동시장에 미칠 영향을 추적하는 데 유용한 측정 도구가 될 수 있다고 말합니다.

---





This paper introduces the **Remote Labor Index (RLI)**, a new benchmark designed to measure how well AI can automate **remote work** in real-world settings.

Unlike many existing AI benchmarks, which mainly test knowledge, reasoning, or coding in isolation, RLI evaluates AI on **economically valuable projects taken from real freelance work**. Each task includes the original brief, input files, and a human-produced gold-standard deliverable, allowing direct comparison between AI output and professional human work.

The main finding is that current AI agents perform **very poorly** on RLI. The best model achieves only a **2.5% automation rate**, showing that today’s AI systems are still far from handling the diverse and complex demands of remote labor.

However, the paper also uses an Elo-based pairwise comparison method to measure more fine-grained progress. This shows that while absolute performance remains low, newer models are gradually improving relative to older ones.

Overall, the paper argues that RLI provides an **empirical, economically grounded way** to track AI automation capabilities and to better understand AI’s potential impact on labor markets.

---




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



# Remote Labor Index(RLI)의 메써드

이 논문의 핵심 메써드는 **Remote Labor Index(RLI)** 라는 새로운 벤치마크를 설계하고, 이를 통해 **AI가 실제 원격 노동(remote work)을 얼마나 자동화할 수 있는지** 측정하는 것입니다. 단순히 지식 문제를 푸는 능력이 아니라, **경제적 가치가 있는 실제 프로젝트를 끝까지 완성할 수 있는지**를 평가한다는 점이 가장 중요한 특징입니다.

## 1. 문제 설정과 연구 목적

기존 AI 벤치마크는 주로 다음과 같은 능력을 측정해 왔습니다.

- 수학적 추론
- 지식 질의응답
- 코딩 문제 풀이
- 웹 탐색
- 제한된 에이전트 과제 수행

하지만 저자들은 이런 벤치마크가 **실제 경제활동으로서의 원격 노동**을 충분히 반영하지 못한다고 봅니다.  
즉, AI가 “문제를 푼다”와 “실제로 고객이 돈을 주고 맡길 수 있는 결과물을 만든다”는 것은 다르다는 것입니다.

그래서 이 논문은 AI의 능력을 다음 기준으로 보려 합니다.

- 실제 프리랜서 작업을 얼마나 잘 수행하는가
- 완성된 산출물이 사람 전문가의 결과물 수준에 도달하는가
- 실제 고객이 받아들일 수 있는 최종 결과를 만들어내는가

이 목표를 위해 **RLI**라는 다분야, 실제 작업 기반 벤치마크를 제안합니다.

---

## 2. 데이터셋 구성 방식

### 2.1 프로젝트 단위의 벤치마크
RLI는 단순한 단문 질의가 아니라, **끝단(end-to-end) 프로젝트**들로 구성됩니다.  
각 프로젝트는 다음 3가지로 이루어집니다.

1. **Brief**
   - 작업 지시서
   - 무엇을 만들어야 하는지 설명하는 텍스트

2. **Input files**
   - 작업에 필요한 자료 파일들
   - 이미지, 문서, 오디오, CAD 파일 등 다양한 형식 포함

3. **Human deliverable**
   - 전문가가 실제로 만든 정답 산출물
   - “gold-standard” 역할을 함

즉, AI는 브리프와 입력 파일만 보고 최종 결과물을 만들어야 하며, 그 결과물이 인간 전문가의 산출물과 비교됩니다.

---

### 2.2 데이터의 출처
데이터는 주로 **프리랜스 플랫폼**, 특히 **Upwork taxonomy**를 기준으로 수집되었습니다.  
저자들은 실제 노동시장에서 쓰이는 프로젝트를 확보하려 했기 때문에, 추상적인 과제나 인위적 문제 대신 **경제적 가치가 실제로 검증된 작업**을 사용했습니다.

수집 방법은 크게 두 가지입니다.

#### (1) 프리랜서 플랫폼 소싱
- 여러 카테고리에서 작업자를 모집
- 그들이 과거에 수행했던 실제 프로젝트 샘플을 제공받음
- 총 207개 프로젝트 확보

#### (2) Long-tail 소싱
- Upwork에 없는 희귀하거나 특수한 디지털 작업도 추가 수집
- 외부 온라인 작업 예시나 맞춤 제작 프로젝트 포함
- 추가로 7개 + 33개 프로젝트 확보

최종적으로 **240개 프로젝트**가 데이터셋에 포함되었습니다.

---

### 2.3 포함된 작업의 범위
RLI는 매우 다양한 분야를 포함합니다. 예를 들면:

- 게임 개발
- 제품 디자인
- 건축
- 오디오/음악 제작
- 데이터 분석
- 영상 제작
- 그래픽 디자인
- 웹 개발
- 프레젠테이션 디자인
- 번역
- 법률 문서 작업
- 시장 조사

논문은 이런 다양성이 중요하다고 강조합니다.  
왜냐하면 기존 벤치마크는 소프트웨어나 글쓰기처럼 **상대적으로 AI가 강한 영역**에 편중되어 있는 반면, 실제 원격 노동은 훨씬 넓고 복합적이기 때문입니다.

---

## 3. 데이터 필터링과 정제

이 논문의 중요한 메써드 중 하나는 **강한 필터링과 정제 과정**입니다.  
연구진은 수집한 프로젝트를 그대로 쓰지 않고, 벤치마크로 쓰기에 적합하도록 엄격하게 걸러냈습니다.

### 3.1 제외 기준
다음과 같은 작업은 제거했습니다.

- 물리적 노동이 필요한 작업
- 클라이언트와 직접 상호작용이 필요한 작업
- 팀 작업이 필수인 작업
- 결과 평가가 즉시 불가능한 작업
- 웹 기반 환경에서 재현/평가하기 어려운 작업
- PII(개인정보) 문제가 있는 작업
- 리모트 환경에서 보기 어렵거나 렌더링이 어려운 형식의 작업

즉, AI가 독립적으로 수행 가능하고, 평가 플랫폼에서 확인 가능한 작업만 남겼습니다.

### 3.2 표준화
모든 프로젝트는 공통 구조를 따르도록 정리되었습니다.

- Work description
- Provided material
- Deliverables

이렇게 구조를 통일해야 평가 기준이 흔들리지 않기 때문입니다.

---

## 4. 평가 방식: 수동 평가 중심

이 논문의 평가 메써드는 매우 중요합니다.  
저자들은 **자동 평가만으로는 RLI 같은 복잡한 결과물을 제대로 판단하기 어렵다**고 보고, **훈련된 인간 평가자에 의한 수동 평가**를 사용했습니다.

### 4.1 왜 자동 평가가 아닌가?
RLI의 산출물은 다음처럼 복잡합니다.

- 이미지
- 영상
- 오디오
- CAD/3D 모델
- 문서
- 웹사이트
- 데이터 시각화

이런 결과물은 단순 정답 비교로 평가할 수 없고, **전문가적인 맥락 이해**가 필요합니다.  
그래서 LLM 기반 자동 채점보다는 사람 평가가 더 신뢰할 수 있다고 본 것입니다.

---

### 4.2 평가 플랫폼
연구진은 여러 파일 형식을 볼 수 있는 **웹 기반 멀티미디어 평가 플랫폼**을 자체 구축했습니다.

이 플랫폼은 다음을 지원합니다.

- 폴더 단위 탐색
- 파일 렌더링
- 이미지, 영상, 오디오, 문서, 3D 파일 보기
- 원본 프로젝트 브리프와 인간 정답 산출물 비교

즉, 평가자가 실제 클라이언트처럼 결과물을 검토할 수 있게 만든 도구입니다.

---

## 5. 핵심 메트릭

논문은 주로 네 가지 메트릭을 사용합니다.

### 5.1 Automation rate
가장 중요한 지표입니다.

- AI 산출물이 인간 정답과 비교했을 때
- “실제 고객이 받아들일 수준”으로 간주되는 프로젝트의 비율

즉,  
**AI가 프로젝트를 완성한 것으로 인정받은 비율**입니다.

논문에서 최고 성능 모델도 **2.5%**에 그쳤습니다.

---

### 5.2 Elo
모델 간 상대 성능을 비교하는 지표입니다.

- 두 AI의 결과물을 비교
- 어떤 결과물이 프로젝트를 더 잘 완료했는지 판단
- 승률 기반으로 Elo 점수 산정

이 지표는 “완전 성공 여부”뿐 아니라 **부분적 진전**도 더 잘 반영합니다.

---

### 5.3 Dollars earned
AI가 성공적으로 완료한 프로젝트들의 인간 작업 비용을 합산한 값입니다.  
즉, AI가 벌어들인 경제적 가치처럼 해석할 수 있습니다.

---

### 5.4 Autoflation
AI가 인간보다 더 싼 비용으로 같은 프로젝트를 완료할 수 있을 때,  
그로 인해 **프로젝트 묶음(bundle)의 실질 비용이 얼마나 줄어드는지**를 측정한 지표입니다.

즉, AI가 작업 자동화를 통해 노동 비용을 얼마나 낮추는지 보는 경제적 메트릭입니다.

---

## 6. AI 생성 및 평가 절차

### 6.1 AI deliverable 생성 방식
모델들은 프로젝트 브리프와 입력 파일을 받습니다.  
특정한 아키텍처를 강제하지는 않았고, 모델별로 적절한 실행 환경을 제공했습니다.

사용된 에이전트/환경 예:

- ChatGPT agent
- GPT-5
- Claude Sonnet 4.5
- Grok 4
- Gemini 2.5 Pro
- Manus

환경은 크게 두 종류입니다.

- **Computer-use scaffold**
- **CLI 기반 scaffold (OpenHands)**

즉, 모델의 인터페이스 특성에 맞춰 도구 사용 환경을 달리했습니다.

---

### 6.2 평가자 지침
평가자는 다음 원칙에 따라 판단했습니다.

- “reasonable client” 관점에서 보기
- 인간 정답이 가진 허용 오차 범위를 고려하기
- 아주 사소한 결함은 과도하게 벌하지 않기
- 결과물이 실제로 고객에게 납품 가능한지 중심으로 판단하기

평가는 3점 척도를 사용했습니다.

1. 인간 결과물보다 명확히 떨어짐
2. 인간 결과물과 동등하게 수용 가능
3. 인간보다 더 좋음

Automation rate는 2 또는 3의 비율로 계산됩니다.

---

## 7. 실험적 세팅의 특징

이 논문의 메써드는 단순히 “모델 하나 돌려보기”가 아니라, 다음 요소를 포함한 **엄격한 평가 파이프라인**입니다.

- 실제 프리랜서 작업 수집
- 데이터 정제 및 표준화
- 다양한 파일 포맷 처리
- 인간 전문가 중심 수동 평가
- 자동화율과 Elo의 이중 측정
- 평가자 간 합의도 확인
- false positive / false negative 검증

즉, 연구 설계가 **경제적 타당성 + 평가 신뢰성**을 모두 확보하도록 짜여 있습니다.

---

## 8. 이 논문의 방법론적 의의

이 방법의 장점은 다음과 같습니다.

- 실제 경제 활동 기반
- 프로젝트 단위 end-to-end 평가
- 다양한 직무 범위 반영
- 복잡한 결과물에 대한 실질적 평가
- AI 자동화의 현실적 한계를 보여줌

반면 한계도 있습니다.

- 모든 원격 노동을 포함하지는 않음
- 고객 상호작용, 팀 협업, 일부 서비스형 업무 제외
- 평가가 수동이라 확장성이 제한됨

그럼에도 이 방법은 “AI가 실제로 사람 일을 대체할 수 있는가?”를 묻는 데 매우 강한 실증적 틀을 제공합니다.

---



## 1. Research Goal

The authors argue that existing AI benchmarks mostly measure:

- knowledge recall
- mathematical reasoning
- coding tasks
- web navigation
- limited agent skills

However, these do not fully capture the ability to perform **real economic labor**.  
The central question of the paper is therefore:

- Can AI complete actual freelance projects?
- Can it produce deliverables that a client would accept?
- Can it match professional human work on practical tasks?

To answer this, the paper introduces **RLI**, a broad multi-sector benchmark grounded in real freelance labor.

---

## 2. Dataset Construction

### 2.1 Project-Level Benchmark
RLI is composed of **full projects**, not short questions or isolated subtasks.  
Each project contains three components:

1. **Brief**
   - a textual description of the work to be done

2. **Input files**
   - the materials needed to complete the project

3. **Human deliverable**
   - a gold-standard output produced by a professional freelancer

The AI must use only the brief and input files to produce a final deliverable, which is then compared to the human reference.

---

### 2.2 Data Sources
Most projects were sourced from **freelance marketplaces**, especially using the **Upwork taxonomy**.  
The goal was to capture real work that has actual economic value.

Data collection happened in two stages:

#### (1) Freelance platform sourcing
- freelancers with relevant expertise were recruited
- they provided samples of prior work
- this yielded 207 projects

#### (2) Long-tail sourcing
- additional rare or emerging digital tasks were collected
- some were sourced from outside the Upwork taxonomy
- this yielded 7 more projects plus 33 additional permission-based examples

In total, the final dataset contains **240 projects**.

---

### 2.3 Diversity of Task Types
RLI spans many domains, including:

- game development
- product design
- architecture
- audio and music production
- data analysis
- video production
- graphic design
- web development
- presentation design
- translation
- legal/document work
- market research

The paper emphasizes that this breadth matters because real remote labor is much more diverse than the narrower tasks used in many existing benchmarks.

---

## 3. Filtering and Cleaning

A major part of the method is the **extensive filtering and normalization** of raw collected tasks.

### 3.1 Exclusion Criteria
The authors removed tasks that:

- required physical labor
- required direct client interaction
- required team collaboration
- could not be evaluated immediately
- were difficult to render in a web-based evaluation environment
- contained sensitive personal information
- were not suitable for standardized benchmarking

Only tasks that were independently solvable and evaluable in the platform were kept.

### 3.2 Standardization
All projects were normalized into a common structure:

- Work description
- Provided material
- Deliverables

This standardization helps ensure consistent evaluation.

---

## 4. Evaluation Method

The paper uses **manual human evaluation**, not automatic scoring, because RLI outputs are often multimodal and complex.

### 4.1 Why manual evaluation?
Deliverables can include:

- images
- videos
- audio
- CAD / 3D files
- documents
- websites
- dashboards

Such outputs require holistic judgment, and automated evaluation is not yet reliable enough for this kind of benchmark.

---

### 4.2 Evaluation Platform
The authors built a **web-based multimedia evaluation platform** that supports:

- directory browsing
- file rendering
- viewing images, videos, audio, documents, and 3D models
- side-by-side comparison between AI output and human reference

This lets evaluators inspect deliverables in a realistic, client-like way.

---

## 5. Main Metrics

The paper reports four key metrics.

### 5.1 Automation Rate
This is the most important metric.

- the percentage of projects where the AI deliverable is judged to be at least as good as the human deliverable
- i.e., acceptable to a reasonable client

The best model achieved only **2.5%** automation rate.

---

### 5.2 Elo
To measure relative progress among models, the authors use a pairwise **Elo-style ranking**.

- two AI deliverables are compared
- evaluators judge which one is closer to successfully completing the project
- Elo utilities are estimated from these preferences

This captures partial progress even when models do not fully solve the task.

---

### 5.3 Dollars Earned
This measures the total human-market value of the projects the AI successfully completes.

---

### 5.4 Autoflation
Autoflation measures how much the effective cost of completing the project bundle decreases when AI can replace human labor with cheaper acceptable deliverables.

---

## 6. AI Generation Setup

The authors evaluate several frontier systems, including:

- ChatGPT agent
- GPT-5
- Claude Sonnet 4.5
- Grok 4
- Gemini 2.5 Pro
- Manus

They use different scaffolds depending on the model:

- **computer-use scaffold**
- **CLI-based scaffold via OpenHands**

The prompt instructs agents to read the brief, use the inputs if present, and produce the required deliverables.

---

## 7. Human Annotation Protocol

Evaluators judge outputs from the perspective of a **reasonable client**.  
They use a 3-point scale:

1. Worse than the human deliverable and not acceptable
2. As good as the human deliverable and acceptable
3. Better than the human deliverable

Automation rate is computed as the fraction of projects scored 2 or 3.

The paper also reports strong inter-annotator agreement, indicating that the manual evaluation procedure is reasonably reliable.

---

## 8. Methodological Significance

The method is notable because it:

- grounds benchmarking in real economic work
- measures end-to-end project completion
- covers many different job categories
- evaluates actual deliverables rather than abstract answers
- shows that current frontier AI still performs near the floor on these tasks

Its limitation is that it does not cover every kind of remote work, especially tasks requiring client interaction or team collaboration.

---





<br/>
# Results


---



## 1. 무엇을 평가했는가
이 논문은 **Remote Labor Index (RLI)**라는 벤치마크를 제안하고, 여러 최신 AI 에이전트가 **실제 원격 프리랜서 업무를 끝까지 수행할 수 있는지** 평가합니다.  
단순 지식 문제나 단일 기능이 아니라, 실제 프로젝트 단위의 작업을 수행하는 능력을 보려는 점이 핵심입니다.

RLI는 다음과 같은 실제 작업들로 구성됩니다.

- 데이터 시각화 대시보드 제작
- 광고 영상 제작
- 게임 개발
- 3D 제품 렌더링
- 건축/인테리어 관련 작업
- 오디오/비디오 편집
- 문서 포맷팅 및 리포트 작성

즉, 다양한 분야의 **end-to-end 프로젝트**를 AI가 사람 수준으로 완료할 수 있는지를 측정합니다.

---

## 2. 테스트 데이터는 무엇인가
논문에서 사용한 RLI 데이터셋은 총 **240개 프로젝트**로 구성됩니다.

### 데이터 구성
각 프로젝트는 3가지 요소를 가집니다.

1. **Brief**  
   작업 설명서. 무엇을 만들어야 하는지 적혀 있음

2. **Input files**  
   작업에 필요한 입력 자료들

3. **Human deliverable**  
   사람이 만든 정답 수준의 결과물.  
   이것이 AI 결과물과 비교되는 기준입니다.

### 데이터의 특징
- **실제 프리랜서 플랫폼**에서 수집됨
- 즉, 단순한 인위적 과제가 아니라 **실제 경제적 가치가 있는 일**을 반영함
- 23개 업워크(Upwork) 카테고리에 걸쳐 있음
- 디자인, 오디오, 게임, 건축, 데이터 분석, 웹 개발 등 다양한 분야 포함
- 프로젝트 난이도도 높아서 인간 전문가 기준:
  - 평균 완료 시간: **28.9시간**
  - 중앙값: **11.5시간**
  - 평균 비용: **$632.6**
  - 전체 가치: **$140,000 이상**
  - 총 작업량: **6,000시간 이상**

### 데이터 분할
- **공개 세트**: 10개 프로젝트
- **비공개 테스트 세트**: 230개 프로젝트

즉, 정량 평가 결과는 주로 **230개 비공개 테스트 프로젝트**에서 계산됩니다.

---

## 3. 비교 대상인 경쟁 모델
논문은 다음 6개 최신 AI 에이전트를 평가했습니다.

- **Manus**
- **Grok 4**
- **Claude Sonnet 4.5**
- **GPT-5**
- **ChatGPT agent**
- **Gemini 2.5 Pro**

또한 GPT-5는 두 가지 실행 환경에서 평가했습니다.

- **GPT-5 (CLI)**
- **GPT-5 (CUA, computer-use agent)**

즉, 모델 자체 성능뿐 아니라 **에이전트 스캐폴딩(scaffold)** 차이도 비교했습니다.

---

## 4. 사용한 메트릭은 무엇인가
논문은 단순 정답률만 보지 않고, 여러 지표를 사용합니다.

### 4.1 Automation Rate
가장 중요한 메트릭입니다.

- 정의: AI 결과물이 사람 결과물과 비교했을 때 **사람이 만든 것만큼 좋거나 더 좋다고 판단된 프로젝트의 비율**
- 즉, “실제 고객이 받아줄 수 있는 수준인가?”를 묻는 지표입니다.

평가 방식은 3점 척도입니다.

1. AI 결과물이 기준보다 못함 → 실패
2. AI 결과물이 사람 결과물만큼 좋음 → 성공
3. AI 결과물이 사람보다 더 좋음 → 성공

Automation Rate = 2 또는 3으로 평가된 비율

---

### 4.2 Elo
모델 간 **상대적 성능 비교** 지표입니다.

- 두 AI 결과물을 서로 비교
- 인간 결과물을 참고 기준으로 삼아
- 어느 쪽이 더 프로젝트를 잘 완성했는지 판단

Elo는 절대 성공률이 낮아도, 모델 간 미세한 차이를 잘 보여줍니다.  
논문에서는 인간 baseline을 **1,000**으로 고정합니다.

---

### 4.3 Dollars Earned
AI가 성공적으로 완료한 프로젝트의 **인간 작업비 기준 가치 총합**입니다.

- 프로젝트가 성공하면 그 프로젝트의 인간 비용을 AI가 벌어들인 것으로 계산
- 실제 경제적 가치 측면에서 얼마나 일을 했는지 보여줌

---

### 4.4 Autoflation
RLI 프로젝트 묶음을 완료하는 데 드는 비용이 AI 때문에 얼마나 줄어드는지를 보는 지표입니다.

- AI가 더 싸고 충분히 좋은 결과물을 내면 비용이 줄어듦
- 아직은 거의 감소가 매우 작음

---

## 5. 결과 요약
논문의 핵심 결과는 매우 명확합니다.

## 5.1 절대 성능은 매우 낮음
모든 평가 모델이 RLI에서 **거의 바닥 수준**의 성능을 보였습니다.

- 최고 Automation Rate도 **2.5%**
- 즉, 대부분의 프로젝트를 AI가 사람 수준으로 완료하지 못함

### 모델별 Automation Rate
- **Manus**: 2.5%
- **Grok 4**: 2.1%
- **Claude Sonnet 4.5**: 2.1%
- **GPT-5 (CLI)**: 1.7%
- **ChatGPT agent**: 1.3%
- **Gemini 2.5 Pro**: 0.8%
- **GPT-5 (CUA)**: 0.8%

즉, 최선의 모델도 100개 중 2~3개 정도만 성공하는 수준입니다.

---

## 5.2 상대적 성능(Elo)은 점진적 향상
Automation Rate가 낮더라도, Elo에서는 모델들 사이에 차이가 나타났습니다.

### Elo 점수 예시
- **Manus**: 509.9
- **Grok 4**: 468.2
- **ChatGPT Agent**: 454.3
- **Claude Sonnet 4.5**: 441.7
- **GPT-5 (CLI)**: 436.7
- **GPT-5 (CUA)**: 431.6
- **Gemini 2.5 Pro**: 411.8

해석:
- 새 모델일수록 대체로 더 높은 점수를 보임
- 즉, **완전 자동화는 아직 어렵지만**, 부분적 개선은 존재함

---

## 5.3 Dollars Earned도 매우 낮음
총 프로젝트 가치가 **$143,991**인데, 모델들이 벌어들인 금액은 매우 적었습니다.

- Manus: **$1,720**
- Sonnet 4.5: **$1,280**
- GPT-5 (CLI): **$1,180**
- Grok 4: **$858**
- GPT-5 (CUA): **$858**
- ChatGPT agent: **$520**
- Gemini 2.5 Pro: **$210**

즉, 최고 성능 모델도 전체 가치의 극히 일부만 수행했습니다.

---

## 6. 비교 방식은 어떻게 했는가
논문은 다음 두 가지 비교 방식을 사용했습니다.

### 6.1 Human vs AI 비교
각 프로젝트에서 AI deliverable이 인간 deliverable만큼 좋은지 판단합니다.

- 평가자는 brief, input file, human output, AI output을 모두 봄
- “합리적인 고객이 이 AI 결과물을 받아줄까?” 관점으로 판단
- 결과가 2 또는 3이면 성공으로 간주

### 6.2 AI vs AI 비교
Elo를 위해 두 AI 결과물을 직접 비교합니다.

- 프로젝트 완성도
- 전체 품질

두 측면을 평가한 뒤 우선순위 규칙으로 최종 선호를 결정합니다.

### 평가 신뢰도
- Automation Rate 평가자 일치도: **94.4%**
- Elo 평가의 ternary agreement: **56.9%**
- hard disagreement 비율: **5.9%**

즉, 평가가 꽤 안정적이었습니다.

---

## 7. 실패 양상
AI가 실패한 이유도 분석했습니다.

주요 실패 유형은 다음과 같습니다.

1. **파일 손상 또는 형식 오류**
   - 빈 파일
   - 깨진 파일
   - 잘못된 형식

2. **불완전한 결과물**
   - 일부 파일 누락
   - 영상이 너무 짧음
   - 필요한 자산이 빠짐

3. **품질 부족**
   - 만들긴 했지만 전문 수준에 못 미침

4. **불일치**
   - 여러 파일 간 시각적/내용적 일관성 부족

논문은 특히 다음 분야에서 실패가 많았다고 설명합니다.

- 건축
- 게임 개발
- 웹 개발
- 멀티모달 검증이 필요한 작업

---

## 8. 논문의 결론
이 논문의 결론은 다음과 같습니다.

- 현재 AI는 **실제 원격 노동의 다양하고 복잡한 작업**을 자율적으로 끝까지 수행하는 수준에 아직 도달하지 못했다.
- 기존 벤치마크보다 훨씬 현실적인 과제를 포함한 RLI에서도 성능은 **매우 낮다**.
- 하지만 Elo 지표에서는 모델 간 점진적 향상이 관측되어, 향후 추세 추적에는 유용하다.
- 따라서 RLI는 AI 자동화 논의를 **실증적 근거 위에서** 할 수 있게 해주는 기준점 역할을 한다.

---



## 1. What was evaluated
The paper introduces the **Remote Labor Index (RLI)**, a benchmark designed to measure whether AI agents can complete **real-world remote freelance projects end to end**.  
Unlike narrow skill benchmarks, RLI focuses on economically valuable work that resembles actual remote labor.

The benchmark includes projects such as:

- interactive data dashboards
- promotional videos
- game development
- 3D product rendering
- architecture and interior design
- audio/video editing
- document formatting and report writing

The main goal is to test whether AI can complete these projects at a level acceptable to a real client.

---

## 2. What was the test data
RLI contains **240 projects** in total.

### Project structure
Each project includes:

1. **Brief**  
   A text description of the task

2. **Input files**  
   Materials needed to complete the task

3. **Human deliverable**  
   The gold-standard output produced by a professional human freelancer

### Data characteristics
- Sourced from **real freelance platforms**
- Grounded in **actual economic transactions**
- Covers **23 Upwork categories**
- Includes diverse fields such as design, audio, game dev, architecture, data analysis, and web development
- Human task statistics:
  - mean completion time: **28.9 hours**
  - median completion time: **11.5 hours**
  - mean cost: **$632.6**
  - total value: over **$140,000**
  - total work: over **6,000 hours**

### Dataset split
- **Public set**: 10 projects
- **Private test set**: 230 projects

Quantitative evaluation is mainly based on the **230 private test projects**.

---

## 3. Competing models
The paper evaluates six frontier AI agents:

- **Manus**
- **Grok 4**
- **Claude Sonnet 4.5**
- **GPT-5**
- **ChatGPT agent**
- **Gemini 2.5 Pro**

GPT-5 was tested in two scaffolding setups:

- **GPT-5 (CLI)**
- **GPT-5 (CUA)**

So the paper also compares different agent scaffolds, not just model quality.

---

## 4. Metrics used
The paper uses several metrics instead of relying on a single score.

### 4.1 Automation Rate
This is the main metric.

- Definition: the percentage of projects where the AI deliverable is judged to be **at least as good as the human gold standard**
- In other words: would a reasonable client accept it as commissioned work?

Scoring uses a 3-point scale:

1. Worse than the human reference → failure
2. As good as the human reference → success
3. Better than the human reference → success

Automation Rate = share of projects rated 2 or 3

---

### 4.2 Elo
This measures **relative model performance**.

- Two AI deliverables are compared head-to-head
- Judges decide which one is closer to completing the task and which has better overall quality
- The human baseline is normalized to **1,000**

Elo is useful because it can detect progress even when absolute success rates remain low.

---

### 4.3 Dollars Earned
This measures the total economic value of projects successfully completed by the AI, using the human cost of each project as the value.

---

### 4.4 Autoflation
This measures how much the cost of completing the fixed project bundle decreases when AI is used whenever it can produce an acceptable deliverable at lower cost.

---

## 5. Main results
The central finding is that **current AI agents perform near the floor on RLI**.

### 5.1 Absolute performance is very low
The best Automation Rate is only **2.5%**.

#### Automation Rate by model
- **Manus**: 2.5%
- **Grok 4**: 2.1%
- **Claude Sonnet 4.5**: 2.1%
- **GPT-5 (CLI)**: 1.7%
- **ChatGPT agent**: 1.3%
- **Gemini 2.5 Pro**: 0.8%
- **GPT-5 (CUA)**: 0.8%

So even the best model succeeds on only a tiny fraction of tasks.

---

### 5.2 Elo shows gradual progress
Although absolute success is low, the Elo scores show meaningful differences between models.

#### Elo scores
- **Manus**: 509.9
- **Grok 4**: 468.2
- **ChatGPT Agent**: 454.3
- **Claude Sonnet 4.5**: 441.7
- **GPT-5 (CLI)**: 436.7
- **GPT-5 (CUA)**: 431.6
- **Gemini 2.5 Pro**: 411.8

This suggests newer models are improving, even though they still fall far short of the human baseline.

---

### 5.3 Dollars earned are also small
The full project value in the benchmark is **$143,991**, but the models earned only a small fraction of that.

- Manus: **$1,720**
- Sonnet 4.5: **$1,280**
- GPT-5 (CLI): **$1,180**
- Grok 4: **$858**
- GPT-5 (CUA): **$858**
- ChatGPT agent: **$520**
- Gemini 2.5 Pro: **$210**

---

## 6. How the comparisons were done
The paper uses two kinds of comparisons.

### 6.1 Human vs AI
For each project, evaluators compare the AI deliverable against the human gold standard and decide whether a reasonable client would accept it.

### 6.2 AI vs AI
For Elo, two AI outputs are compared directly.

Evaluators judge:

- project completion
- overall quality

Majority voting across multiple annotators is used to make the final decision.

### Evaluation reliability
- Automation Rate inter-annotator agreement: **94.4%**
- Elo ternary agreement: **56.9%**
- Hard disagreement rate: **5.9%**

This indicates the evaluation process was fairly reliable.

---

## 7. Failure modes
The paper identifies common reasons for failure:

1. **Corrupted or invalid files**
2. **Incomplete deliverables**
3. **Low quality**
4. **Inconsistencies across files**

Failures were especially common in:

- architecture
- game development
- web development
- multimodal or visually complex tasks

---

## 8. Conclusion
The paper concludes that:

- current AI systems are **not yet capable of autonomously performing the broad range of economically valuable remote work**
- RLI is a more realistic benchmark than prior agent benchmarks
- the overall absolute automation rate remains extremely low
- however, Elo scores show measurable incremental progress across models

In short, RLI provides an empirical baseline for tracking AI automation of remote labor over time.

---





<br/>
# 예제




## 1. 이 논문에서 말하는 데이터 예시의 큰 구조

RLI(Remote Labor Index)는 “AI가 원격 노동을 어느 정도 자동화할 수 있는가”를 재는 벤치마크입니다.  
각 프로젝트는 아래 3가지로 구성됩니다.

1. **Brief(작업 지시문)**  
   - 무엇을 만들어야 하는지 텍스트로 설명한 문서
   - 작업 목적, 요구사항, 형식 등이 포함됨

2. **Input files(입력 파일들)**  
   - 작업을 완료하는 데 필요한 원본 자료
   - 예: 사진, 표 데이터, PDF, 도면, 음성 파일, 레퍼런스 이미지, 측정치 등

3. **Human deliverable(인간 정답 산출물)**  
   - 실제 전문가가 만든 완성 결과물
   - 벤치마크에서는 이를 “gold-standard deliverable”로 사용

즉, AI는 **브리프 + 입력 파일**만 받고 결과물을 만들어야 하며,  
평가자는 그 결과물이 **인간 전문가의 결과물과 비교해 실제로 수주 가능한 수준인지**를 판단합니다.

---

## 2. 데이터셋이 트레이닝/테스트처럼 보이는 이유

논문에서 전통적인 의미의 트레이닝셋을 학습시키는 것은 아닙니다.  
대신 다음과 같은 구조가 있습니다.

- **Public set 10 projects**
  - 공개용 샘플
  - 질적 예시를 보여주거나 플랫폼 확인에 사용

- **Private test set 230 projects**
  - 실제 정량 평가에 사용
  - 모델은 이 데이터를 직접 볼 수 없다고 가정
  - 자동화율, Elo, Dollars Earned, Autoflation 계산에 사용

따라서 이걸 데이터셋 관점에서 보면:

- **공개 샘플 = 예시/데모 데이터**
- **비공개 평가셋 = 테스트 데이터**

논문 자체에서는 “training data”가 아니라 **“benchmark instances”**라고 보는 것이 더 정확합니다.

---

## 3. 예시 1: 세계 행복지수 인터랙티브 대시보드

논문에 나온 예시 프로젝트 중 하나는 다음과 같습니다.

### 프로젝트 브리프
- “World Happiness Report 데이터를 탐색할 수 있는 인터랙티브 대시보드 구축”
- 지도 위에 국가별 행복 점수를 표시
- hover/click 시 국가명과 정확한 수치 표시
- 옆 또는 아래에 연동된 보조 차트 추가
- 사용자가 지도에서 국가를 클릭하면 차트에서도 해당 국가가 강조되도록 연결

### 입력 파일
- `inputs/DataForFigure2.1WHR2021C2.xls`

이 파일은 아마도 다음 같은 정보를 담고 있습니다.
- 국가명
- 행복 점수
- 구성 지표들(예: 소득, 사회적 지지, 기대수명 등으로 추정 가능)
- 차트 작성에 필요한 원본 데이터

### 인간 산출물
- 완성된 대시보드
- HTML/CSS/JavaScript 기반의 자가 호스팅 인터랙티브 웹페이지
- 지도 시각화 + 연동 차트 포함

### AI가 만들어야 하는 출력
- 동일한 기능을 가진 웹 대시보드 패키지
- HTML, CSS, JS, 필요 라이브러리
- 사용자가 브라우저에서 바로 열어볼 수 있어야 함

### 평가 포인트
- 지도에 국가별 점수가 올바르게 표현되는가
- hover/click 상호작용이 되는가
- 보조 차트가 지도와 연동되는가
- 전체적으로 행복이라는 테마에 맞는지
- 데이터가 제공된 엑셀만을 근거로 했는지

---

## 4. 예시 2: 메가 머지(Mega Merge) 웹 게임

다른 예시는 “Watermelon Game” 스타일의 웹 게임입니다.

### 프로젝트 브리프
- 떨어지는 오브젝트를 합쳐 더 높은 레벨의 오브젝트를 만드는 게임
- 웹 기반
- 데스크톱과 모바일 모두 지원
- 터치와 마우스 입력 지원
- 물리 기반 낙하와 충돌
- 점수 표시
- 다음 아이템 표시
- 배경음악과 합쳐질 때 효과음 포함
- 총 파일 크기 5MB 이하

### 입력 파일
- **없음**
- 즉, 브리프만 주어지고 외부 참고 자료는 없음

### 인간 산출물
- 완성된 웹 게임
- HTML, CSS, JS, 이미지, 사운드 파일 포함
- 폴더 구조가 잘 정리된 형태

### AI 출력
- 브라우저에서 실행 가능한 게임 전체
- responsive UI
- physics/collision 구현
- merge/drop SFX
- music loop 포함

### 평가 포인트
- 실제로 게임이 되는가
- 모바일/데스크톱 반응형인가
- 물리/충돌/병합이 자연스러운가
- UI가 최소한이면서도 명확한가
- 파일 크기 제한 준수 여부

---

## 5. 예시 3: 3D 이어버드 제품 데모 애니메이션

이 예시는 제품 홍보용 3D 애니메이션입니다.

### 프로젝트 브리프
- 이어버드와 케이스의 기능을 보여주는 고품질 3D 데모 영상 5개 제작
- 주요 특징:
  - 실리콘 팁
  - 스템이 분리/교체되며 배터리 역할
  - 세련된 충전 케이스
  - L/R 표시
- 마케팅에 쓸 수 있게 짧고 세련된 영상 필요

### 입력 파일
- `inputs/earbuds_back.jpg`
- `inputs/earbuds_front.jpg`
- `inputs/earbuds_top.jpg`
- `inputs/replaceable_battery.jpg`
- `inputs/charging_case.jpg`

이 이미지들은 제품 외형과 핵심 기능을 보여주는 참고자료입니다.

### 인간 산출물
- 5개의 완성된 MP4 영상
- 3D 모델 파일(fbx 등)도 함께 제공 가능

### AI 출력
- 고품질 3D 렌더 영상 5개
- 기능을 잘 보여주는 카메라 무브와 조명
- 시각적으로 일관된 제품 표현

### 평가 포인트
- 기능이 실제로 전달되는가
- 제품 외형이 이미지와 일치하는가
- 영상 품질이 마케팅 수준인가
- 애니메이션이 매끄러운가

---

## 6. 예시 4: 인테리어 디자인 + 가구 배치

이 예시는 건축/실내설계 계열입니다.

### 프로젝트 브리프
- 기존 평면도와 추가 측정 정보를 바탕으로
- 욕실 인테리어 옵션 3개
- 아파트 가구 배치 옵션 6개
- 최종 선택안에 대한 추가 계획도 작성

### 입력 파일
- `cadastral floor plan.jpg`
- `bathroom.jpg`
- `bathroom_photos/photo_#_y.jpg`
- `additional measurements.jpg`

### 인간 산출물
- 욕실 렌더링
- 재료 보드
- 벽 마감 이미지
- 3D 원본 파일 + 교환 파일
- 가구 배치 PDF
- DWG 파일
- 조명 계획, 전기 계획, 바닥 마감 계획 등

### AI 출력
- 건축/인테리어 결과물 세트 전체
- 치수와 레이아웃이 정확해야 함
- CAD/DWG/PDF 등 요구 포맷 충족

### 평가 포인트
- 실제 측정값과 맞는가
- 공간 구성 논리가 맞는가
- 출력 포맷이 요구사항과 부합하는가
- 시각적으로 전문적인가

---

## 7. 예시 5: 3D 링 디자인 수정

논문의 평가 플랫폼 예시 중 하나는 다음과 같습니다.

### 프로젝트 브리프
- “주어진 링 디자인을 마퀴즈 컷 다이아몬드가 들어가도록 수정하라”

### 입력 파일
- 기존 링 디자인 파일
- 관련 레퍼런스 자료

### 인간 산출물
- 수정 완료된 링 디자인
- 렌더 또는 3D 모델

### AI 출력
- 동일한 요구를 충족하는 수정 디자인
- 외관상/형태상 요구를 반영한 결과물

### 평가 포인트
- 브리프에 맞게 변경되었는가
- 모양과 디테일이 정확한가
- 렌더링/뷰어에서 제대로 확인되는가

---

## 8. 예시 6: 2D 플랫 디자인 설명 영상

또 다른 예시는 트리 서비스 회사 광고/설명 영상입니다.

### 프로젝트 브리프
- 약 60초 길이의 2D 플랫 디자인 explainer 제작
- 내용:
  - trimming
  - pruning
  - stump removal
  - tree health
- 조건:
  - 굵은 타이포그래피
  - 자연스러운 색감
  - 아이콘 중심 그래픽
  - 부드러운 모션
  - 제공된 voiceover 사용

### 입력 파일
- voiceover wav 파일

### 인간 산출물
- 완성된 영상
- 보이스오버와 싱크된 애니메이션

### AI 출력
- 60초 내외의 2D 홍보/설명 영상
- 자막 없이도 이해 가능해야 함

### 평가 포인트
- 음성 내레이션과 영상이 잘 맞는가
- 요구한 스타일(플랫, 자연색, 아이콘 중심)을 지켰는가
- 길이와 구성 요소가 충족되는가

---

## 9. 이 논문에서의 “입력”과 “출력”을 더 일반적으로 정리하면

### 입력(Input)
- 텍스트 브리프
- CSV/Excel 데이터
- 이미지
- PDF
- 도면(DWG/스크린샷)
- 오디오
- 기타 프로젝트용 참고 파일

### 출력(Output)
- 웹앱
- PDF
- JPG/PNG 이미지
- MP4 영상
- 3D 모델 파일
- DWG/OBJ/FBX 등 CAD/3D 형식
- 텍스트 코드 파일
- 여러 파일이 묶인 폴더 구조

---

## 10. 평가 방식과 데이터 라벨 구조

RLI의 중요한 점은 “정답”이 단순한 정답 문자열이 아니라는 것입니다.

### 인간 라벨
- 인간 전문가가 만든 deliverable이 기준
- “이 작업이 클라이언트에게 수락될 수준인가?”가 핵심

### AI 라벨
- AI 산출물이 인간 산출물과 비교해
  - 불합격
  - 합격
  - 인간보다 더 나음
  중 하나로 평가됨

### 주요 메트릭
- **Automation rate**
  - AI가 인간 수준 이상으로 완수한 프로젝트 비율

- **Elo**
  - AI 모델끼리의 상대적 성능 순위

- **Dollars earned**
  - AI가 성공적으로 완료한 프로젝트들의 인간 작업 비용 합

- **Autoflation**
  - AI가 프로젝트 묶음의 실효 비용을 얼마나 낮추는지

---

## 11. 이 논문 예시들이 보여주는 핵심

이 벤치마크는 단순히 “문제를 푸는 AI”가 아니라,  
**실제 경제적 가치가 있는 remote work를 끝까지 수행하는 AI**를 평가합니다.

즉, 다음 같은 점이 중요합니다.

- 정보 검색만 잘하는가?
- 코드 한 조각만 잘 짜는가?
- 중간 결과를 만들 수 있는가?
- 최종 납품물까지 제대로 완성하는가?
- 인간 고객이 돈을 주고 받을 수준인가?

이 관점에서 예시들은 모두 **실제 프리랜서 노동의 완결된 작업 단위**입니다.  
그래서 일반적인 QA 데이터셋처럼 “질문-정답”이 아니라,  
**브리프-입력자료-완성 산출물** 구조를 갖습니다.

---



---

## 1. Overall data structure in the paper

Each RLI project consists of three parts:

1. **Brief**
   - A textual description of what needs to be done
   - Includes the work description, expected deliverables, and constraints

2. **Input files**
   - The materials needed to complete the project
   - Examples: images, spreadsheets, PDFs, audio, CAD-like references, measurements

3. **Human deliverable**
   - The gold-standard final product created by a professional freelancer
   - This is the reference used in evaluation

The AI model receives the **brief + input files** and must produce a deliverable that can be compared against the human reference.

---

## 2. Why this looks like train/test data

The paper does not provide a training set in the traditional sense.  
Instead, the benchmark behaves like this:

- **Public set of 10 projects**
  - Sample examples for qualitative inspection and platform validation

- **Private set of 230 projects**
  - Hidden evaluation set
  - Used for automation rate, Elo, dollars earned, and autoflation

So in dataset terms:

- **Public set = example/demo data**
- **Private set = test data**

---

## 3. Example 1: World Happiness Report interactive dashboard

One of the example projects is:

### Brief
- Build an interactive dashboard for exploring data from the World Happiness Report
- Show each country’s happiness score on a map
- Add hover/click tooltips with country name and exact value
- Add a linked companion chart that highlights the selected country

### Input file
- `inputs/DataForFigure2.1WHR2021C2.xls`

This spreadsheet likely contains:
- Country names
- Overall happiness scores
- Component metrics used in the report

### Human deliverable
- A finished interactive dashboard
- A self-hosted HTML/CSS/JavaScript package
- Map + linked chart

### AI output
- A working dashboard package with the same interactive features
- Files that can be opened in a browser

### Evaluation criteria
- Correct country shading by score
- Hover/click functionality
- Linked chart highlighting
- Proper use of the provided data
- Good fit with the happiness theme

---

## 4. Example 2: Mega Merge web game

Another example is a web game inspired by the Watermelon Game.

### Brief
- Create a casual game where falling objects are merged to reach the highest-level item
- Web-based
- Works on desktop and mobile
- Touch and mouse support
- Physics and collisions
- Score display, next-item preview, pause/reset
- Relaxing background music and merge/drop sound effects
- Keep total size under 5 MB

### Input files
- None

### Human deliverable
- A complete browser game
- HTML, CSS, JavaScript, images, and sounds
- Organized folder structure

### AI output
- A fully functional responsive web game

### Evaluation criteria
- Playable game mechanics
- Responsive layout
- Smooth physics/collisions
- Sound effects and background music
- File size constraint

---

## 5. Example 3: 3D earbuds product demo animations

This is a marketing-style 3D animation task.

### Brief
- Create 5 high-quality short 3D videos showcasing earbuds and their case
- Features:
  - silicone tips
  - replaceable battery stem
  - sleek charging case
  - L/R indicator decal

### Input files
- `inputs/earbuds_back.jpg`
- `inputs/earbuds_front.jpg`
- `inputs/earbuds_top.jpg`
- `inputs/replaceable_battery.jpg`
- `inputs/charging_case.jpg`

### Human deliverable
- Five MP4 clips
- Optionally 3D model files such as FBX

### AI output
- Five polished product demo videos
- Smooth camera movement and lighting
- Clear presentation of product features

### Evaluation criteria
- Feature clarity
- Visual quality
- Consistency with provided reference images
- Professional marketing quality

---

## 6. Example 4: Interior design and furniture layout

This example is in architecture/interior design.

### Brief
- Create multiple interior design options and furniture layout options
- Include final plans for lighting, toilet installation, electrical layout, floor finishes, etc.

### Input files
- `cadastral floor plan.jpg`
- `bathroom.jpg`
- `bathroom_photos/photo_#_y.jpg`
- `additional measurements.jpg`

### Human deliverable
- Renderings
- Material boards
- 3D source files
- DWG/PDF plans
- Lighting/electrical/floor-finish plans

### AI output
- A complete set of architectural/interior design files

### Evaluation criteria
- Dimensional correctness
- Proper layout logic
- Required output formats
- Professional appearance

---

## 7. Example 5: Ring design modification

A smaller example from the evaluation platform:

### Brief
- Modify a ring design to include a marquise-cut diamond

### Input files
- Original ring design files
- Reference materials

### Human deliverable
- Completed modified ring design
- Render or 3D file

### AI output
- A design that satisfies the modification request

### Evaluation criteria
- Correctly applied design change
- Visual fidelity
- Viewable/renderable file output

---

## 8. Example 6: 2D flat-design explainer video

Another example is a tree-services explainer/ad video.

### Brief
- Create a ~60-second flat-design explainer video
- Topics:
  - trimming
  - pruning
  - stump removal
  - tree health
- Requirements:
  - bold typography
  - natural palette
  - icon-driven graphics
  - subtle character animation
  - smooth transitions
  - use the provided voiceover

### Input files
- Voiceover WAV file

### Human deliverable
- A finished narrated video synced to the audio

### AI output
- A short promotional/explainer animation

### Evaluation criteria
- Matching voiceover timing
- Visual style compliance
- Duration and completeness

---

## 9. General input/output types in the benchmark

### Inputs
- Text briefs
- Excel/CSV data
- Images
- PDFs
- Drawings/CAD screenshots
- Audio files
- Reference assets

### Outputs
- Web apps
- PDFs
- Images
- Videos
- 3D models
- CAD files
- Code folders / project packages

---

## 10. How the labels and evaluation work

The benchmark does not use simple answer labels.  
Instead, human evaluators judge whether the AI output would be accepted by a reasonable client.

### Human reference
- The professional freelancer’s deliverable is the gold standard

### AI judgment
- AI output is judged as:
  - worse than human
  - as good as human
  - better than human

### Main metrics
- **Automation rate**: percentage of projects where AI meets or exceeds the human standard
- **Elo**: relative ranking among AI agents
- **Dollars earned**: total monetary value of projects successfully completed
- **Autoflation**: reduction in the effective cost of completing the project bundle

---

## 11. Core takeaway

The key idea of the paper is that RLI measures whether AI can complete **real, economically valuable remote work end-to-end**.  
So the tasks are not ordinary Q&A problems. They are complete freelance-style jobs with:

- a brief,
- input materials,
- and a human gold-standard deliverable.

That is why the benchmark is much closer to real labor automation than standard academic datasets.

---



<br/>


# 요약



이 논문은 실제 프리랜서 원격 업무 240개 프로젝트를 모은 **Remote Labor Index(RLI)**를 만들고, 인간이 만든 정답 산출물과 AI 산출물을 **수동 평가**해 자동화율과 Elo를 측정하는 방법을 제시합니다.  
결과적으로 현재 최상위 AI 에이전트들도 RLI에서 **최대 자동화율 2.5%**에 그쳤고, 예시 프로젝트로는 **대시보드 제작, 2D 광고 영상, 3D 제품 데모, 웹 게임, 건축/CAD 설계** 등이 포함되었습니다.  
즉, 방법은 “실제 경제적 가치가 있는 end-to-end 작업을 모아 인간 기준으로 평가”이고, 결과는 “대부분의 원격 노동을 아직 자동화하지 못함”이며, 예시는 “현실적인 프리랜서 업무 전반”을 보여줍니다.  




This paper introduces the **Remote Labor Index (RLI)**, a benchmark built from **240 real freelance remote-work projects**, and measures AI automation by **manual comparison** between AI outputs and human gold-standard deliverables.  
The results show that even the best current AI agents reach only a **2.5% automation rate**, and the example projects include **interactive dashboards, 2D ad videos, 3D product demos, web games, and architecture/CAD design**.  
In short, the method is “evaluate end-to-end, economically valuable real work against human standards,” the result is “AI still automates only a tiny fraction of remote labor,” and the examples demonstrate the breadth of realistic freelance tasks.

<br/>
# 기타



---

## 1) Figure 1: RLI 예시 프로젝트 구성
### 결과
- RLI는 단순한 Q&A가 아니라 **실제 프리랜스 프로젝트 전체**를 포함합니다.
- 각 프로젝트는:
  - **Brief**
  - **Input files**
  - **Human deliverable**
  로 구성됩니다.
- 예시로는 데이터 시각화, 애니메이션 광고, 게임 개발, 3D 제품 렌더링, 건축 설계, 과학 문서 포맷팅 등이 제시됩니다.

### 인사이트
- AI 평가를 “문항 단위”가 아니라 **실제 업무 단위(end-to-end project)** 로 옮겼다는 점이 핵심입니다.
- 즉, 단편적 능력보다 **실제로 돈이 오가는 작업을 끝낼 수 있는가**를 측정합니다.

---

## 2) Figure 2 / Table 1: 현재 AI 에이전트 자동화율
### 결과
- 최고 성능 모델의 자동화율은 **2.5%** 에 불과합니다.
- 모델별 자동화율:
  - Manus: **2.5%**
  - Grok 4: **2.1%**
  - Sonnet 4.5: **2.1%**
  - GPT-5: **1.7%**
  - ChatGPT agent: **1.3%**
  - Gemini 2.5 Pro: **0.8%**

### 인사이트
- 현재 frontier AI 에이전트도 **대부분의 실제 원격 노동을 완료하지 못함**을 보여줍니다.
- “지식/추론 벤치마크에서 강함”이 곧 “실무 자동화 가능”을 의미하지 않음을 시사합니다.
- 자동화율 기준으로는 **거의 바닥 수준**입니다.

---

## 3) Figure 3: RLI 프로젝트 카테고리 분포
### 결과
- RLI는 Upwork taxonomy 기준 **23개 카테고리**를 포함합니다.
- 상위 예시는:
  - Video
  - CAD
  - Graphic Design
  - Game Dev
  - Audio
  - Architecture 등
- 특정 분야에 편중되지 않고 **디자인, 오디오, 건축, 데이터, 마케팅, 운영** 등으로 폭넓게 분포합니다.

### 인사이트
- 기존 벤치마크가 소프트웨어/리서치에 치우친 반면, RLI는 **실제 원격 노동 시장의 다양성**을 더 잘 반영합니다.
- 따라서 특정 능력만 좋은 모델보다 **범용적 실무 역량**을 더 잘 드러냅니다.

---

## 4) Figure 4: 프로젝트 비용과 완료 시간 분포
### 결과
- 프로젝트 비용:
  - 평균 **$632.6**
  - 중앙값 **$200**
  - 최대 **$22,500**
- 완료 시간:
  - 평균 **28.9시간**
  - 중앙값 **11.5시간**
  - 최대 **450시간**
- 총합:
  - **6,000시간 이상**
  - **$140,000 이상 가치**

### 인사이트
- RLI는 “작고 쉬운 과제”가 아니라, **실제 경제적 가치가 큰 노동**을 포함합니다.
- 특히 완료 시간이 길고 비용이 큰 작업도 포함되어 있어, AI가 단순 보조가 아니라 **대체 가능한지**를 보기 좋습니다.

---

## 5) Figure 5: 데이터 수집 및 정제 파이프라인
### 결과
- 총 550개 초기 태스크 수집 → 정제 후 **최종 240개** 채택
- 358명의 검증된 프리랜서 모집
- 다단계 필터링:
  - 적합성 확인
  - PII 제거
  - 렌더 가능 여부 확인
  - 표준화

### 인사이트
- 벤치마크 품질을 위해 **대규모 선별과 정제**를 거쳤다는 점이 중요합니다.
- 즉, 데이터는 단순 수집물이 아니라 **실제 평가 가능한 형태로 엄격히 가공된 연구용 자산**입니다.

---

## 6) 3.3 Metrics: 평가 지표 정의
### 결과
논문은 4가지 지표를 사용합니다.
- **Automation rate**: 사람 수준 이상으로 완수한 비율
- **Elo**: 모델 간 상대 비교
- **Dollars earned**: AI가 성공적으로 대체한 작업의 경제 가치
- **Autoflation**: AI로 인해 작업 묶음의 실질 비용이 얼마나 내려가는지

### 인사이트
- 단순 “정답률” 대신 **경제적 가치**와 **상대적 성능**까지 함께 봅니다.
- 특히 Autoflation은 AI가 실제로 노동시장의 가격을 얼마나 낮추는지 보는 흥미로운 지표입니다.

---

## 7) Figure 6: RLI와 다른 벤치마크 비교
### 결과
- RLI는 기존 벤치마크보다:
  - **평균 완료 시간이 더 길고**
  - **프로젝트 유형이 훨씬 다양**합니다.
- 기존 벤치마크는 software/research/writing에 많이 치우침.
- RLI는 실제 원격 노동처럼 더 넓은 스펙트럼을 포함합니다.

### 인사이트
- RLI는 기존 벤치마크의 “쉬운 부분”을 넘어, **실제 프리랜스 시장의 복잡성**에 더 가깝습니다.
- 즉, AI 성능을 과대평가하지 않도록 해줍니다.

---

## 8) Figure 7: 평가 파이프라인
### 결과
- 인간 평가자가 AI deliverable과 human deliverable을 비교하여:
  - brief 충족 여부
  - 품질 수준
  - 클라이언트가 받아들일 수 있는지
  를 판단합니다.
- 세부 rubric보다 **holistic evaluation**을 사용합니다.

### 인사이트
- 디자인, 비디오, 건축처럼 정량 루브릭이 부족한 작업에 적합합니다.
- 즉, “형식적 조건 충족”이 아니라 **실제 납품 가능성**을 평가합니다.

---

## 9) Table 2: AI deliverable의 주요 실패 유형
### 결과
주요 실패 유형:
- Poor quality: **45.6%**
- Incomplete: **35.7%**
- Corrupted files: **17.6%**
- Inconsistencies: **14.8%**

### 인사이트
- 실패의 핵심은 단순히 “정답을 못 맞힘”이 아니라,
  - 파일 손상
  - 누락
  - 품질 부족
  - 산출물 간 불일치
  입니다.
- 즉, AI는 아직 **생산물 완성도와 작업 안정성**에서 약합니다.

---

## 10) Figure 8 / Table 3: Elo 결과
### 결과
- Elo 기준으로도 모델 간 차이는 존재하지만,
- 인간 기준선 **1,000** 에는 모두 크게 못 미침.
- Table 3의 Elo:
  - Manus: **509.9**
  - Grok 4: **468.2**
  - ChatGPT Agent: **454.3**
  - Sonnet 4.5: **441.7**
  - GPT-5 (CLI): **436.7**
  - GPT-5 (CUA): **431.6**
  - Gemini 2.5 Pro: **411.8**

### 인사이트
- 절대 성능은 낮지만, Elo는 **부분적 진전**을 포착합니다.
- 즉, 모델은 서로 비교하면 발전 중이지만, 여전히 **인간 수준 업무 자동화와는 거리가 큼**을 보여줍니다.

---

## 11) Table 4: AI가 벌어들인 금액
### 결과
- 총 가능 가치: **$143,991**
- 실제 AI가 벌어들인 금액:
  - Manus: **$1,720**
  - Sonnet 4.5: **$1,280**
  - GPT-5 (CLI): **$1,180**
  - Grok 4 / GPT-5 (CUA): **$858**
  - ChatGPT agent: **$520**
  - Gemini 2.5 Pro: **$210**

### 인사이트
- AI가 성공적으로 대체한 경제 가치는 **전체의 극히 일부**입니다.
- 즉, 현재 AI는 경제적 의미에서 **노동 대체보다 제한적 보조 수준**에 가깝습니다.

---

## 12) Appendix A.2: Autoflation
### 결과
- AI가 인간보다 더 싸게 같은 작업을 완수할 경우 비용 절감 효과를 측정합니다.
- 현재는 전반적으로 **낮은 수준**입니다.

### 인사이트
- 이 지표는 AI가 실제로 **노동 가격을 떨어뜨리는 정도**를 보여줍니다.
- 즉, 단순 성능이 아니라 **시장 영향**까지 반영합니다.

---

## 13) Appendix A.3: Agent scaffold 효과
### 결과
- GPT-5는 CUA보다 **CLI scaffold에서 더 좋은 성능**:
  - Elo: CLI 436.7 > CUA 431.6
  - Automation: CLI 1.7% > CUA 0.8%

### 인사이트
- 모델 자체 성능뿐 아니라 **작업 환경/도구 설계**가 중요합니다.
- 즉, AI 자동화 성능은 모델만이 아니라 **scaffold 품질**에도 크게 좌우됩니다.

---

## 14) Appendix B.4 / B.5: 평가 신뢰도
### 결과
- Automation rate 평가 inter-annotator agreement: **94.4%**
- Elo 평가 ternary agreement: **56.9%**
- hard disagreement: **5.9%**
- false negative rate 상한: **≤5.8%**

### 인사이트
- 평가가 주관적일 수 있는 작업임에도 **신뢰도가 높게 유지**되었습니다.
- 이는 결과의 설득력을 높여줍니다.

---

## 15) Appendix C: 데이터셋 세부 정보
### 결과
- 최종 데이터셋: **240개 프로젝트**
- **9개 major categories / 23개 subcategories**
- Upwork taxonomy 기반으로 다양하게 구성

### 인사이트
- RLI는 특정 직군 하나가 아니라, **디지털 원격 노동 전반**을 포괄하려는 설계입니다.
- 따라서 “AI가 어느 한 분야를 잘한다”가 아니라 **실제 시장 전체에 얼마나 접근했는지**를 보여줍니다.

---

# 전체 종합 인사이트
1. **현재 AI는 실제 원격 노동 자동화에서 매우 낮은 수준**입니다.  
2. **벤치마크 성능 향상 ≠ 실무 자동화 능력 향상**입니다.  
3. RLI는 단순 태스크가 아니라 **경제적 가치가 있는 end-to-end 업무**를 평가한다는 점에서 의미가 큽니다.  
4. AI는 일부 창작/데이터/텍스트 작업에서는 가능성을 보이지만, 전체적으로는 **품질, 완성도, 일관성, 파일 안정성**에서 부족합니다.  
5. 이 논문은 AI 자동화를 논의할 때 **실증적 기준점**을 제시합니다.

---





## 1) Figure 1: Example project structure in RLI
### Result
- RLI is not a simple QA benchmark.
- Each project contains:
  - a **brief**
  - **input files**
  - a **human deliverable**
- Example tasks include data visualization, animated ads, game development, 3D product renders, architecture, and scientific document formatting.

### Insight
- The benchmark measures **end-to-end project completion**, not isolated skills.
- It evaluates whether AI can complete **real paid work**, not just answer questions.

---

## 2) Figure 2 / Table 1: Current AI automation rates
### Result
- The best model reaches only **2.5% automation rate**.
- Model results:
  - Manus: **2.5%**
  - Grok 4: **2.1%**
  - Sonnet 4.5: **2.1%**
  - GPT-5: **1.7%**
  - ChatGPT agent: **1.3%**
  - Gemini 2.5 Pro: **0.8%**

### Insight
- Current frontier AI agents are **near the floor** on real remote work automation.
- Strong performance on reasoning benchmarks does **not** imply strong real-world labor automation.

---

## 3) Figure 3: Distribution of project categories
### Result
- RLI covers **23 Upwork categories**.
- Major examples include:
  - Video
  - CAD
  - Graphic Design
  - Game Development
  - Audio
  - Architecture
- The dataset spans design, audio, architecture, data, marketing, operations, and more.

### Insight
- Unlike prior benchmarks that over-focus on software and research, RLI reflects the **diversity of the remote labor market**.

---

## 4) Figure 4: Project cost and completion-time distributions
### Result
- Project cost:
  - Mean: **$632.6**
  - Median: **$200**
  - Max: **$22,500**
- Completion time:
  - Mean: **28.9 hours**
  - Median: **11.5 hours**
  - Max: **450 hours**
- Total value:
  - Over **6,000 hours**
  - Over **$140,000**

### Insight
- RLI includes work with **substantial economic value**, not toy tasks.
- This makes it more suitable for measuring true automation potential.

---

## 5) Figure 5: Data collection and cleaning pipeline
### Result
- 550 initial tasks were collected and filtered down to **240 final tasks**.
- **358 verified freelancers** participated.
- The pipeline included filtering, anonymization, quality checks, and standardization.

### Insight
- The benchmark was carefully curated to ensure **high-quality, evaluable tasks**.

---

## 6) Section 3.3 Metrics
### Result
The paper uses four metrics:
- **Automation rate**
- **Elo**
- **Dollars earned**
- **Autoflation**

### Insight
- The paper measures not only success rate, but also **relative performance** and **economic impact**.

---

## 7) Figure 6: RLI compared with previous benchmarks
### Result
- RLI has longer human completion times and much broader task diversity.
- Prior benchmarks are concentrated in software/research/writing.

### Insight
- RLI better captures the **complexity of real freelance labor**.

---

## 8) Figure 7: Evaluation pipeline
### Result
- Human evaluators compare AI deliverables against human gold-standard work.
- The evaluation is holistic rather than rubric-heavy.

### Insight
- This is especially appropriate for creative and multimodal work where simple checklists are insufficient.

---

## 9) Table 2: Main AI failure modes
### Result
- Poor quality: **45.6%**
- Incomplete deliverables: **35.7%**
- Corrupted files: **17.6%**
- Inconsistencies: **14.8%**

### Insight
- Failures are not just “wrong answers,” but also **missing, broken, or unusable outputs**.

---

## 10) Figure 8 / Table 3: Elo results
### Result
- All models remain well below the human baseline of **1,000**.
- Elo scores:
  - Manus: **509.9**
  - Grok 4: **468.2**
  - ChatGPT Agent: **454.3**
  - Sonnet 4.5: **441.7**
  - GPT-5 (CLI): **436.7**
  - GPT-5 (CUA): **431.6**
  - Gemini 2.5 Pro: **411.8**

### Insight
- Models are improving relative to each other, but they are still **far from human-level remote work automation**.

---

## 11) Table 4: Dollars earned
### Result
- Total possible value: **$143,991**
- Model earnings are small:
  - Manus: **$1,720**
  - Sonnet 4.5: **$1,280**
  - GPT-5 (CLI): **$1,180**
  - Grok 4 / GPT-5 (CUA): **$858**
  - ChatGPT agent: **$520**
  - Gemini 2.5 Pro: **$210**

### Insight
- Current models capture only a **tiny fraction** of the available economic value.

---

## 12) Appendix A.2: Autoflation
### Result
- Measures the cost reduction in completing the project bundle using AI where possible.
- Current levels remain low.

### Insight
- This metric connects AI performance to **market price reduction**, not just benchmark success.

---

## 13) Appendix A.3: Effect of agent scaffolds
### Result
- GPT-5 performs better in CLI than in computer-use mode:
  - Elo: **436.7 vs. 431.6**
  - Automation rate: **1.7% vs. 0.8%**

### Insight
- Performance depends not only on the model, but also on the **agent scaffold and tooling**.

---

## 14) Appendix B.4 / B.5: Evaluation reliability
### Result
- Inter-annotator agreement:
  - Automation rate: **94.4%**
  - Elo: **56.9%**
- Hard disagreement: **5.9%**
- False negative upper bound: **≤5.8%**

### Insight
- The evaluation process is **reasonably reliable**, strengthening the paper’s claims.

---

## 15) Appendix C: Dataset details
### Result
- Final dataset: **240 projects**
- **9 major categories / 23 subcategories**
- Based on Upwork taxonomy

### Insight
- RLI is designed to represent the **broader digital remote labor economy**, not just a narrow slice of it.

---




<br/>
# refer format:



---

## BibTeX

```bibtex
@article{mazeika2025remote,
  title={Remote Labor Index: Measuring AI Automation of Remote Work},
  author={Mazeika, Mantas and Gatti, Alice and Menghini, Cristina and Sehwag, Udari Madhushani and Singhal, Shivam and Orlovskiy, Yury and Basart, Steven and Sharma, Manasi and Peskoff, Denis and Lau, Elaine and Lim, Jaehyuk and Carroll, Lachlan and Blair, Alice and Sivakumar, Vinaya and Basu, Sumana and Kenstler, Brad and Ma, Yuntao and Michael, Julian and Li, Xiaoke and Ingebretsen, Oliver and Mehta, Aditya and Mottola, Jean and Teichmann, John and Yu, Kevin and Shaik, Zaina and Khoja, Adam and Ren, Richard and Hausenloy, Jason and Phan, Long and Htet, Ye and Aich, Ankit and Rabbani, Tahseen and Shah, Vivswan and Novykov, Andriy and Binder, Felix and Chugunov, Kirill and Ramirez, Luis and Geralnik, Matias and Mesura, Hern{\'a}n and Lee, Dean and Hernandez Cardona, Ed-Yeremai and Diamond, Annette and Yue, Summer and Wang, Alexandr and Liu, Bing and Hernandez, Ernesto and Hendrycks, Dan},
  journal={arXiv preprint arXiv:2510.26787},
  year={2025},
  note={Available at arXiv}
}
```

---

## Chicago 스타일   

Mantas Mazeika, Alice Gatti, Cristina Menghini, Udari Madhushani Sehwag, Shivam Singhal, Yury Orlovskiy, Steven Basart, Manasi Sharma, Denis Peskoff, Elaine Lau, Jaehyuk Lim, Lachlan Carroll, Alice Blair, Vinaya Sivakumar, Sumana Basu, Brad Kenstler, Yuntao Ma, Julian Michael, Xiaoke Li, Oliver Ingebretsen, Aditya Mehta, Jean Mottola, John Teichmann, Kevin Yu, Zaina Shaik, Adam Khoja, Richard Ren, Jason Hausenloy, Long Phan, Ye Htet, Ankit Aich, Tahseen Rabbani, Vivswan Shah, Andriy Novykov, Felix Binder, Kirill Chugunov, Luis Ramirez, Matias Geralnik, Hernán Mesura, Dean Lee, Ed-Yeremai Hernandez Cardona, Annette Diamond, Summer Yue, Alexandr Wang, Bing Liu, Ernesto Hernandez, and Dan Hendrycks, “Remote Labor Index: Measuring AI Automation of Remote Work,” *arXiv* preprint arXiv:2510.26787 (2025).

---


