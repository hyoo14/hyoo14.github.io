---
layout: post
title:  "[2026]SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration"
date:   2026-03-27 03:22:58 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: SWE-CI는 **CI-loop 기반의 evolution-based 평가**로, 매 반복마다 기능적 정합성(functional correctness)이 어떻게 변하는지를 추적(유지보수성)  


짧은 요약(Abstract) :
## Abstract 한글 설명 (논문 내용 기반)

이 논문은 **LLM 기반 소프트웨어 에이전트**가 버그를 “한 번에” 고치는 능력은 좋아졌지만, 현실의 소프트웨어 개발처럼 **요구사항이 계속 바뀌고 기능이 장기간에 걸쳐 반복적으로 추가/수정되는 상황**을 기존 벤치마크들이 제대로 평가하지 못한다고 지적합니다. 기존 벤치마크는 보통 특정 시점(snapshot)에서 “테스트를 통과하면 끝”이라서, **단단히 하드코딩된 취약한 수정**이든 **깔끔하고 확장 가능한 설계**든 결과가 같게 보일 수 있습니다. 그러나 시간이 지나 새 요구사항이 들어오면 과거의 설계/수정 결정이 누적되어 차이가 드러납니다.

이를 해결하기 위해 저자들은 **SWE-CI**라는 새로운 **레포지토리 수준(repository-level) 벤치마크**를 제안합니다. 핵심 아이디어는 다음 문장으로 요약됩니다:

- **“유지보수성(maintainability)은 시간이 지나면서 기능적 정합성(functional correctness)이 어떻게 변하는지를 추적하면 드러난다.”**

SWE-CI는 실제 오픈소스 레포에서 가져온 **100개 작업(task)**으로 구성되며, 각 작업은 하나의 레포에서 **base commit(시작 시점)**과 **target/oracle commit(목표 시점)**을 잡고 그 사이에 평균 **233일, 71개 연속 커밋**의 실제 진화 이력이 존재하도록 구성됩니다. 에이전트는 CI(Continuous Integration) 루프처럼 **여러 라운드에 걸쳐 분석·코딩·테스트 반복**을 수행해 최종적으로 target commit의 테스트를 모두 통과해야 합니다.

또한 논문은 장기 진화에서의 성능을 점수화하기 위해 **EvoScore(EvolutionScore)**라는 지표를 도입합니다. EvoScore는 “미래 변경(후속 반복)에서의 기능적 정합성”을 더 중요하게 반영해, **초기에 좋은 결정을 내려 이후 변경을 쉽게 만드는 에이전트는 높은 점수**, 반대로 **기술부채를 쌓아갈수록 성능이 떨어지는 에이전트는 낮은 점수**를 받도록 설계되었습니다.

결론적으로, 대규모 실험(총 100억 토큰 이상)을 통해 **최신 모델들도 장기적인 코드 품질 유지와 회귀(regression) 통제에 여전히 어려움을 겪는다**는 점을 보여주며, SWE-CI가 이런 “유지보수성” 문제를 진단하는 데 유용하다고 주장합니다.

---




LLM-powered agents are strong at static, one-shot bug fixing, but real software evolves through complex requirement changes and long-term iterations—something snapshot-style benchmarks fail to capture. The paper introduces **SWE-CI**, the first repository-level benchmark built around a **Continuous Integration (CI) loop**, shifting evaluation from short-term functional correctness to **dynamic, long-term maintainability**. The key idea is that maintainability can be observed by **tracking how functional correctness changes over time**.

SWE-CI contains **100 tasks** from real repositories, each spanning an average of **233 days and 71 consecutive commits** between a base commit and a target (oracle) commit. Agents must solve tasks through **dozens of iterative rounds** of analysis and code changes. The benchmark proposes **EvoScore (EvolutionScore)**, a metric that rewards correctness on future modifications—favoring agents whose early decisions support later evolution and penalizing those that accumulate technical debt. Extensive experiments (over **10 billion tokens**) show that state-of-the-art models still struggle to sustain code quality over long-term evolution, highlighting SWE-CI’s diagnostic value for maintainable coding agents.


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



이 논문의 “메서드”는 **새 모델 아키텍처/학습기법을 제안**하기보다, **LLM/에이전트를 ‘지속적 통합(Continuous Integration, CI) 루프’ 기반으로 평가**하기 위한 **벤치마크 설계 + 평가 프로토콜 + 메트릭**을 체계화한 것입니다. 핵심은 “유지보수성(maintainability)은 시간이 지나며 기능적 정합성이 어떻게 변하는지로 드러난다”는 관찰을 평가 절차에 내장하는 것입니다.

---

### 1) 문제의식과 핵심 아이디어(평가 패러다임 전환)
- 기존 벤치마크(HumanEval, SWE-bench 등)는 대체로 **스냅샷(snapshot) 기반**입니다. 즉, 모델이 **한 번에(one-shot)** 요구사항을 받고 패치를 내면 끝나는 구조라서,  
  - “급하게 땜질한 취약한 수정”과  
  - “구조적으로 확장 가능한 수정”이  
  동일하게 테스트를 통과하면 **유지보수성 차이가 보이지 않습니다.**
- 논문은 이를 해결하기 위해 **진화(evolution) 기반 평가**를 제안합니다.  
  즉, 코드베이스가 여러 라운드에 걸쳐 계속 변하고, 과거 결정의 결과가 누적되며, 그 과정에서 **회귀(regression)**가 발생할 수 있는 상황에서 성능을 보게 합니다.

관련 원문(요지):
- “Maintainability can be revealed by tracking how functional correctness changes over time.”
- “evolution-based evaluation paradigm… requirement is derived from the current codebase dynamically…”

---

### 2) 태스크 정식화(formalization): require / code 함수로 표현
논문은 테스트 집합 **T**를 기준으로 요구사항 생성과 코드 수정 과정을 다음 두 함수로 추상화합니다.

- **require\_T : C × C → R**  
  - 코드베이스 공간 C에서, **현재 코드(c)**와 **오라클 코드(c\*)** 사이의 “테스트 관점 기능 격차”를 파악해 **요구사항 문서 R**를 생성
- **code\_T : R × C → C**  
  - 요구사항 R을 받아 현재 코드베이스를 수정하여 **업데이트된 코드베이스**를 생성

이때,
- **스냅샷 평가**: r = require\_T(c0, c\*) 한 번 만들고 한 번 고침  
- **진화 평가**: 매 반복 i마다  
  - r_i = require\_T(c_i, c\*)  
  - c_{i+1} = code\_T(c_i, r_i)  
  로 **요구사항이 “현재 상태”에 따라 동적으로 재생성**됩니다.

---

### 3) 세부 메트릭: Normalized Change (정규화 변화량)
기존처럼 “전체 테스트 통과/실패”만 보면, 장기 진화에서 중요한 **부분 개선/부분 악화(회귀)**를 정량화하기 어렵습니다. 그래서 논문은 코드베이스 c가 통과하는 테스트 개수:

- **n(c) = Σ_{t∈T} I(t, c)**  
  (I는 테스트 통과 여부 indicator)

이를 기반으로 **정규화 변화량 a(c)**를 정의합니다(개선과 회귀를 비대칭 정규화):

- 개선(n(c) ≥ n(c0))일 때  
  \[
  a(c)=\frac{n(c)-n(c_0)}{n(c^*)-n(c_0)}
  \]
  → 베이스 대비 “오라클까지의 격차”를 얼마나 메웠는지 (완전 달성 시 1)

- 회귀(n(c) < n(c0))일 때  
  \[
  a(c)=\frac{n(c)-n(c_0)}{n(c_0)}
  \]
  → 베이스에서 원래 통과하던 것까지 얼마나 망가뜨렸는지 (전부 깨면 -1)

결과적으로 **a(c) ∈ [-1, 1]**로 서로 다른 프로젝트/테스트 규모에도 비교 가능하게 만듭니다.

---

### 4) 핵심 스코어: EvoScore (미래 가중 평균)
ISO/IEC 25010의 유지보수성 정의(수정이 용이하고 결함/품질저하를 유발하지 않음)에 맞춰, 논문은 **시간이 흐를수록(반복이 뒤로 갈수록) 더 중요**하다고 보고 **미래 가중 평균** 형태로 EvoScore를 정의합니다.

N번 반복에서 얻은 코드베이스 시퀀스 (c1 … cN)에 대해:

\[
e=\frac{\sum_{i=1}^{N}\gamma^{i} a(c_i)}{\sum_{i=1}^{N}\gamma^{i}}
\]

- **γ ≥ 1**: 뒤 반복일수록 가중치 증가  
  - γ=1이면 단순 평균  
  - γ가 커질수록 “초반에 빨리 맞추기”보다 “후반까지 안정적으로 진화”하는 에이전트를 더 높게 평가  
- 해석: 초반에 급히 통과시키고 기술부채로 후반 수정이 어려워지면 점수가 떨어지고, 반대로 구조적 수정으로 후반 변경도 잘 버티면 점수가 올라갑니다.

---

### 5) 평가 프로토콜: Architect–Programmer 듀얼 에이전트 + CI-loop
SWE-CI는 실제 팀 개발의 CI 반복을 모사하기 위해 **2개 역할(agent)**을 분리합니다(그림 3 설명).

#### (1) Architect Agent (요구사항 생성 역할)
입력: 현재 코드 vs 오라클 코드의 테스트 격차(실패 테스트 정보, 코드/테스트 탐색)  
출력: **자연어 고수준 요구사항 문서**(논문 프롬프트에서는 XML로 저장)

Architect는 다음 3단(또는 프롬프트 상 5단) 흐름으로 행동을 표준화합니다:
- **Summarize**: 실패 테스트를 요약, 원인/집중 영역 파악  
- **Locate**: 소스/테스트를 보고 실패를 구체 결함으로 귀속  
- **Design**: 개선 계획 및 요구사항 문서 작성  
추가 제약:
- **증분적(incremental)**: 한 라운드에 최대 5개 핵심 요구만  
- **고수준(high-level)**: “기대 동작(behavior contract)” 중심, 구현 지시 최소화

#### (2) Programmer Agent (구현 역할)
입력: Architect의 요구사항 문서(테스트 gap 자체가 아니라 문서 중심)  
행동: 이해 → 계획 → 코드 수정  
제약: 테스트 실행은 외부 시스템이 수행(에이전트는 실행 금지), 테스트 폴더 수정 금지 등

이 분리는 CI의 “요구 정리(설계) → 구현” 흐름을 모사하며, 프로그래머가 전체 실패를 직접 보고 즉흥 수정하는 것이 아니라 **요구사항에 따라 빠르게 증분 구현**하도록 강제합니다.

---

### 6) 데이터(태스크) 구성 방법: 실제 저장소의 장기 진화 구간을 쌍으로 추출
벤치마크는 **100개 태스크**로 구성되며, 각 태스크는 동일 저장소 내 **base commit → target(or oracle) commit** 쌍입니다. 평균적으로:
- **233일**
- **71개의 연속 커밋**
을 포함하는 “진짜 개발 역사”를 사이에 둡니다.

데이터 큐레이션은 4단계(그림 2)입니다.
1. **Repo 수집/필터링**: 3년 이상 유지, 500+ stars, 설정/의존성 파일 및 테스트 존재, permissive license  
2. **Commit span 추출**: main branch 선형화 후, **의존성이 바뀌지 않는 최대 커밋 구간**을 찾아 양 끝을 base/oracle 후보로 삼음. 변경량이 너무 작은 쌍(총 변경 < 1000 LOC)은 제거  
3. **환경 구축**: oracle 기준 Dockerfile 생성, 테스트 실행 검증. 누락 의존성으로 실행 실패 시 **self-repair로 의존성 주입**하여 환경 재빌드(그 외 원인 실패는 폐기)  
4. **케이스 필터링**: base에서 oracle 테스트를 돌려 런치 실패 제거, base vs oracle 통과 테스트 수 차이가 너무 작은 케이스 제거(차이 < 5), 최종적으로 기간/커밋 수 상위 100개 선택

---

### 7) “특별한 모델/아키텍처/트레이닝 데이터”에 대한 답
논문이 제안하는 방법은 **모델 자체를 새로 학습**하거나 **특별한 신경망 아키텍처**를 추가하는 방식이 아닙니다.  
대신,
- **평가 패러다임(스냅샷 → 진화/CI-loop)**
- **역할 분리(Architect/Programmer)**
- **유지보수성 대리 메트릭(EvoScore, normalized change)**
- **현실적 장기 진화 데이터(커밋 히스토리+도커 환경)**
를 통해, “유지보수 능력”을 드러내는 쪽에 метод의 초점이 있습니다.

---



This paper does **not** introduce a new model architecture or a new training recipe. Instead, it proposes **SWE-CI**, a **CI-loop-based, evolution-oriented benchmark** to evaluate whether LLM agents can **maintain** a codebase over long-term evolution. The core idea is: **maintainability is revealed by how functional correctness changes over time**, especially under iterative modifications and potential regressions.

### 1) Evolution-based evaluation paradigm
Prior benchmarks are largely **snapshot-based**: the agent receives a static requirement once and outputs a one-shot patch. SWE-CI shifts to an **evolution-based** paradigm where requirements are **dynamically derived** from the *current* codebase at each iteration, so earlier design decisions affect later iterations.

### 2) Formalization with `require_T` and `code_T`
Let **T** be the test set, **C** the codebase space, **R** the requirement space:
- `require_T: C × C → R` generates a requirement document from the functional gap (w.r.t. T) between current code and an oracle codebase.
- `code_T: R × C → C` updates the codebase according to the requirement.

Snapshot: `r = require_T(c0, c*)` once.  
Evolution: `r_i = require_T(c_i, c*)`, `c_{i+1} = code_T(c_i, r_i)` iteratively.

### 3) Normalized Change
Define `n(c)` as the number of passing tests. Then **normalized change** `a(c) ∈ [-1, 1]` measures improvement vs. regression relative to the base:
- If `n(c) ≥ n(c0)`, normalize by the remaining gap to oracle.
- If `n(c) < n(c0)`, normalize by `n(c0)` to quantify regressions.

### 4) EvoScore (future-weighted mean)
Given iteration outputs `(c1…cN)`, SWE-CI aggregates them via:
\[
e=\frac{\sum_{i=1}^{N}\gamma^{i} a(c_i)}{\sum_{i=1}^{N}\gamma^{i}},\ \gamma \ge 1
\]
Later iterations receive higher weight, rewarding long-term stability and penalizing accumulated technical debt.

### 5) Dual-agent CI protocol: Architect–Programmer
SWE-CI uses a two-role workflow to mimic real CI:
- **Architect agent**: analyzes failing tests and code, then writes **incremental (≤5 items), high-level** requirements (behavioral contracts).
- **Programmer agent**: implements the requirements via comprehend → plan → code, without directly being driven by the full test gap.

### 6) Data curation: long-term real commit histories
Each task is a (base commit, oracle/target commit) pair from a real GitHub repo, with on average **233 days** and **71 consecutive commits** between them, shipped with a **pre-built Docker environment**. The pipeline filters repos, extracts dependency-stable commit spans, builds/repairs environments, and selects high-quality cases.

---




<br/>
# Results
## SWE-CI 논문에서 말하는 “결과(경쟁모델, 테스트데이터, 메트릭, 비교 등)” 정리 




---

## 1) 테스트 데이터(벤치마크/데이터셋) 구성: SWE-CI가 무엇을 어떻게 평가하나

### 1.1 태스크 단위: “base commit → target(oracle) commit” 장기 진화 과제
- SWE-CI는 **실제 오픈소스 저장소의 커밋 히스토리**에서
  - 시작점인 **base commit**
  - 도달 목표인 **target commit (= oracle codebase)**
  를 잡고,
- 그 사이의 **수십 회 CI(Continuous Integration) 반복 루프**를 통해 에이전트가 코드를 점진적으로 유지보수/진화시키도록 설계합니다.
- 논문이 강조하는 핵심은 기존 벤치마크(스냅샷 1회성 해결)와 달리, **“여러 라운드에 걸친 변화 축적”** 속에서 유지보수성 차이가 드러나게 만든다는 점입니다.

### 1.2 데이터 규모와 장기성(“진짜 장기 유지보수”를 만들기 위한 통계)
최종 SWE-CI 데이터셋의 스펙(논문 3.1):
- **총 100개 태스크(samples)**
- **68개 서로 다른 GitHub 저장소**에서 추출
- base~oracle 사이 평균:
  - **233일**
  - **71개 연속 커밋**
- 코드 변화량(논문 3.1 후반):
  - base→oracle 전이에서 **소스코드 변경 최소 500 라인 이상**(테스트 변경 제외)
- 재현성:
  - 각 샘플에 **전체 소스코드 + 사전 빌드된 Docker 환경**을 함께 제공

즉, “짧은 패치”가 아니라 **실제 프로젝트의 긴 기간 진화 구간**을 과제로 만든 것이 데이터 측면의 특징입니다.

### 1.3 데이터 큐레이션(선별) 파이프라인: 왜 100개가 “살아있는” 케이스인가
논문 3.1의 4단계 선별 과정 요약:

1) **Repository Collection**
   - GitHub의 Python repos 전체에서 탐색 후 필터:
     - 유지보수 기간 ≥ 3년
     - Star ≥ 500
     - 설정/의존성 파일 + 유닛테스트 보유
     - MIT/Apache-2.0 등 permissive license
   - 남는 저장소: **4,923개**

2) **Commit Span Extraction**
   - main branch만 유지(선형 커밋열)
   - **의존성(dependency)이 변하지 않는 최대 연속 커밋 subsequence**를 찾음
   - subsequence 양 끝을 base/oracle 후보로 삼음
   - 변경 라인 수가 너무 작으면(총 수정 LOC < 1000) 제거
   - 후보 pair: **8,311개**

3) **Environment Construction**
   - oracle의 설정/의존성 기반 Dockerfile 자동 생성 및 환경 스냅샷
   - oracle 코드베이스에서 테스트 실행해 정상성 검증
   - 테스트 런치가 의존성 누락으로 실패하면 **self-repair로 Dockerfile에 의존성 주입**하여 재빌드(유지율↑)
   - 이후 생존: **1,458 pair**

4) **Case Filtering**
   - (a) 같은 환경에서 oracle 테스트를 base 코드에 돌렸을 때 **런치 실패하는 경우 제거**
   - (b) base vs oracle 테스트 통과 수 차이가 **5 미만이면 제거**
   - 자동 필터 후 **137개**
   - 마지막으로 기간/커밋 수 기준 상위 100개 선택 → 최종 SWE-CI

---

## 2) 메트릭(평가 지표): “유지보수성”을 기능정확도의 시간 변화로 측정

SWE-CI는 “한 번에 다 맞추는지”가 아니라, 여러 라운드 동안 **정확도가 어떻게 유지/개선/퇴행(regression)하는지**로 유지보수성을 보려 합니다.

### 2.1 Normalized Change a(c): 각 시점 코드베이스의 상태를 [-1, 1]로 정규화
논문 2.2의 정의:

- 테스트 집합 \(T\)에서 코드베이스 \(c\)가 통과한 테스트 수
\[
n(c)=\sum_{t\in T} I(t,c)
\]
- base codebase를 \(c_0\), oracle(목표)를 \(c^*\)라 할 때 정규화 변화량:
\[
a(c)=
\begin{cases}
\frac{n(c)-n(c_0)}{n(c^*)-n(c_0)} & \text{if } n(c)\ge n(c_0)\\[6pt]
\frac{n(c)-n(c_0)}{n(c_0)} & \text{if } n(c)< n(c_0)
\end{cases}
\]

해석(논문 설명 그대로 반영):
- base보다 좋아지면(통과 수 증가) **남은 격차를 기준으로 1까지 스케일링**
  - \(a(c)=1\)이면 oracle 수준 격차를 완전히 메움
- base보다 나빠지면(회귀) **base에서 깨뜨린 비율**로 음수 스케일링
  - \(a(c)=-1\)은 “원래 통과하던 테스트를 전부 깨버린” 최악 케이스
- 이 비대칭 정규화로 **개선과 회귀를 동일한 척도([-1,1])에서 비교** 가능

### 2.2 EvoScore e: “미래(후반 라운드)를 더 중요시”하는 유지보수성 점수
논문 2.3:

여러 iteration에서 나온 코드베이스 시퀀스 \((c_1,\dots,c_N)\)에 대해,
\[
e=\frac{\sum_{i=1}^{N}\gamma^i a(c_i)}{\sum_{i=1}^{N}\gamma^i}
\]
- \(\gamma \ge 1\)로 설정해 **뒤 라운드(장기 진화 단계)에 더 큰 가중치**를 줄 수 있음
- \(\gamma=1\)이면 단순 평균(average normalized change)
- \(\gamma\)가 커질수록 “초반만 빨리 통과하고 이후 유지보수에서 무너지는 모델”을 불리하게 만듦  
  → 논문이 말하는 **‘기술부채(technical debt) 누적’ 페널티를 점수로 반영**하는 장치

### 2.3 Regression(회귀) 관련 지표: Zero-regression rate
논문 4.2 Observation 3에서:
- “회귀(regression)” 정의: **이전에는 통과하던 테스트가 코드 변경 후 실패로 바뀌는 현상**
- SWE-CI에서는 각 모델에 대해
  - 전체 유지보수 과정 동안 **단 한 번도 회귀가 발생하지 않은 샘플 비율**
  을 **zero-regression rate**로 측정합니다.
- 장기 유지보수에서 안정성을 보려는 목적의 보조 지표입니다.

---

## 3) 실험 설정(테스트 실행/반복 횟수 등)

논문 4.1:

- 테스트 프레임워크: **pytest + pytest-json-report**
- 테스트 런 타임아웃: **각 테스트 run당 3600초**
- 에이전트 프레임워크: **iFlow CLI**
- 최대 반복(듀얼 에이전트 CI-loop): **20 iterations**
- 기본 설정: 별도 언급 없으면 **Architect와 Programmer가 같은 base model을 공유**

---

## 4) 경쟁 모델(비교 대상) 범위

논문 4.2 서두 및 Observation들에서:
- **총 18개 모델**
- **8개 provider(모델 제공사)**에서 가져와 비교
- Provider 예시(논문 Observation 2에서 명시적으로 언급):  
  **MiniMax, DeepSeek, GPT, Kimi, GLM, Qwen, Doubao, Claude**
- 특히 성능 언급이 반복되는 모델군:
  - **Claude Opus 시리즈**: 전반적으로 선두(“commanding lead”)
  - **GLM-5**: 강한 성능으로 언급

(논문에는 Figure 4/5/6을 통해 구체 모델들의 EvoScore/순위/회귀율을 시각화했다고 설명합니다.)

---

## 5) 결과 비교(핵심 관찰 3가지): SWE-CI에서 모델들이 어떻게 달랐나

논문 4.2의 “Observation 1~3”를 그대로 결과 중심으로 정리하면 다음과 같습니다.

### Observation 1) 코드 유지보수 능력은 빠르게 좋아지고 있다(가속)
- 동일 provider 계열 내에서 **신형 모델이 항상 더 높은 점수**
- 특히 **2026 이후 릴리즈 모델들이 이전 대비 더 큰 폭의 향상**
- 해석: LLM이 “정적 버그 수정(snapshot)”을 넘어 **지속 유지보수 쪽으로 진화 중**
- 모델군 코멘트:
  - **Claude Opus 시리즈가 관측 기간 전체에서 강한 선두**
  - **GLM-5도 강한 경쟁자**

### Observation 2) provider마다 “단기 성과 vs 장기 유지보수” 성향이 다르다(γ로 드러남)
- 논문은 \(\gamma\) 값을 바꿔가며 EvoScore 순위 변화 관찰(Figure 5)
  - \(\gamma<1\): 초반 iteration 가중치 ↑ → “즉시 성과” 지향 모델 유리
  - \(\gamma>1\): 후반 iteration 가중치 ↑ → “장기 안정/유지보수” 지향 모델 유리
- 발견(논문 주장):
  - **MiniMax, DeepSeek, GPT**: 장기 이득 선호(γ>1에서 유리한 경향)
  - **Kimi, GLM**: 단기 수익 선호(초반 성과형)
  - **Qwen, Doubao, Claude**: γ 변화에 비교적 안정적
- 논문 해석(추정): provider별 학습 전략 차이 가능, 동일 provider 내 일관성은 내부 파이프라인 안정성 시사

### Observation 3) 장기 유지보수에서 “회귀(regression) 통제”가 여전히 큰 약점
- zero-regression rate(전 과정 무회귀 비율)를 측정했더니:
  - **대부분 모델이 0.25 미만**
  - **Claude-opus 시리즈 중 2개 모델만 0.5 초과**
- 결론: 스냅샷형 과제에서 성능이 좋아졌더라도, **장기·다회전 자동 유지보수에서 안정적으로 회귀를 피하는 것은 아직 어렵다**

---

# structured summary of results: baselines/models, data, metrics, comparisons

## 1) Test data / benchmark construction
- **SWE-CI** consists of **100 repository-level tasks**, each defined by a **base commit** and a **target (oracle) commit** from a real-world GitHub Python repository.
- The evolutionary gap is substantial: on average **233 days** and **71 consecutive commits** between base and oracle.
- Each sample ships with the **full source code** and a **pre-built Docker environment** for reproducibility.
- Curation pipeline (4 steps): repository filtering → commit-span extraction with unchanged dependencies → Docker environment construction with a self-repair mechanism for missing deps → case filtering, finally selecting the top 100 by time span / commit count.

## 2) Metrics
- **Normalized change** \(a(c)\) maps the number of passed tests at a given iteration to a comparable scale in **[-1, 1]**, rewarding improvements toward the oracle and penalizing regressions below baseline.
- **EvoScore** \(e\) aggregates normalized changes across iterations using a **future-weighted mean**:
  \[
  e=\frac{\sum_{i=1}^{N}\gamma^i a(c_i)}{\sum_{i=1}^{N}\gamma^i}
  \]
  where \(\gamma \ge 1\) gives higher weight to later iterations to reflect long-term maintainability.
- **Zero-regression rate**: the proportion of samples with **no regressions** throughout the entire maintenance process.

## 3) Experimental setting
- Testing: **pytest + pytest-json-report**, **3600s timeout** per test run.
- Agent framework: **iFlow CLI**.
- Max iterations in the dual-agent CI loop: **20**.
- By default, Architect and Programmer agents use the **same underlying model**.

## 4) Competing models / providers
- Evaluated **18 models from 8 providers** (explicitly referenced providers include MiniMax, DeepSeek, GPT, Kimi, GLM, Qwen, Doubao, Claude).
- The **Claude Opus** series is reported as leading overall; **GLM-5** is also highlighted as strong.

## 5) Main comparative findings (Observations)
1. **Rapid, accelerating progress**: newer models within the same provider consistently score higher; post-2026 models show larger gains.
2. **Different short-term vs long-term preferences** revealed by varying \(\gamma\): MiniMax/DeepSeek/GPT trend toward long-term gains; Kimi/GLM toward short-term; Qwen/Doubao/Claude are relatively stable.
3. **Regression control remains weak**: most models have **zero-regression rate < 0.25**; only two Claude-Opus models exceed **0.5**, indicating ongoing difficulty in avoiding regressions in long-horizon maintenance.

---




<br/>
# 예제



## 1) 이 벤치마크는 “트레이닝 데이터/테스트 데이터”가 어떻게 생겼나?

### 1.1 트레이닝 데이터(학습용 데이터) 제공 여부
- 논문에서 SWE-CI는 “benchmark(벤치마크) / dataset(데이터셋)”로 공개되며(Hugging Face/GitHub 링크 제공), **주 용도는 평가**입니다.
- 제공된 본문에는 “모델을 SWE-CI로 학습(train)했다”거나 “훈련/검증/테스트 split”을 정의했다는 설명은 없습니다.  
  → 즉, **전통적인 의미의 training dataset vs test dataset 분할/입출력 예시**는 본문에 명시되지 않습니다.

### 1.2 테스트 데이터(평가에 쓰이는 데이터)의 실체: “태스크 1개 = 실 repo의 커밋 쌍 + 테스트 스위트 + 도커 환경”
논문에서 반복적으로 명시하는 SWE-CI 샘플(태스크)의 구성은 다음과 같습니다.

- **100 tasks**
- 각 task는 실 GitHub 저장소에서 가져온:
  - **base commit (c0)**
  - **target/oracle commit (c\*)**
  - 그 사이 평균 **233일, 71 commits**의 진짜 진화 히스토리
- 각 샘플은 다음을 포함해 재현성을 보장:
  - “complete source code”
  - “pre-built Docker environment”

근거(본문 요지):
- Abstract/Intro: “100 tasks… base commit and target commit… average 233 days and 71 consecutive commits”
- 3.1 Data curation 마지막: “Each sample is shipped with the complete source code and a pre-built Docker environment…”

---

## 2) 한 태스크에서 “구체적으로 무엇이 인풋이고, 무엇이 아웃풋인가?” (논문 정의 기반)

논문 2.1에서 태스크를 “테스트 집합 T, 코드베이스 C, 요구사항 R”로 추상화하고, 두 함수를 정의합니다.

### 2.1 핵심 객체/기호
- **T**: 관심 있는 모든 단위 테스트들의 집합  
  (각 테스트를 t로 표기)
- **c0**: base codebase (베이스 커밋의 코드)
- **c\***: oracle/target codebase (타깃 커밋의 코드)
- **ci**: i번째 반복에서의 현재 코드베이스
- **R**: 요구사항 문서(자연어로 작성)

### 2.2 인풋 → 아웃풋 흐름(진화 기반 평가, CI-loop)
논문은 기존 벤치마크(스냅샷 방식)와 달리, SWE-CI는 **반복 루프**로 입력/출력이 교환된다고 설명합니다.

#### (1) 요구사항 생성 함수: require_T
- 정의: `require_T: C × C → R`
- 의미: (현재 코드베이스, 오라클 코드베이스)를 비교해, 테스트 관점에서 “기능적 격차”를 요구사항 문서로 생성

즉, 각 반복 i에서:
- **입력(input)**: (현재 코드베이스 **ci**, 오라클 **c\***)
- **출력(output)**: 요구사항 문서 **ri** (자연어)

#### (2) 코드 수정 함수: code_T
- 정의: `code_T: R × C → C`
- 의미: 요구사항 문서와 현재 코드베이스를 받아 코드를 수정하여 업데이트된 코드베이스 반환

즉:
- **입력(input)**: (요구사항 문서 **ri**, 현재 코드베이스 **ci**)
- **출력(output)**: 업데이트된 코드베이스 **ci+1**

이것을 반복하여 최종 목표:
- “target commit에 해당하는 테스트를 모두 통과”하는 방향으로 진화

---

## 3) “구체적인 테스크”는 어떤 형태인가? (실행 관점 예시)

논문 3.2는 SWE-CI가 실제 개발팀의 CI를 모사하도록 **Architect–Programmer 2-agent** 프로토콜을 사용한다고 설명합니다. 즉, 태스크는 단발성 “버그 하나 고치기”가 아니라, **여러 라운드에 걸쳐 테스트 실패를 줄이고(또는 회귀를 막고) 결국 타깃 상태로 가는 장기 유지보수 시나리오**입니다.

### 3.1 한 라운드에서 에이전트들이 받는 “구체적 입력”
논문 “Prompts” 섹션(B)에서 더 구체적으로, 에이전트가 접근 가능한 입력 경로를 명시합니다.

#### Architect agent 입력(파일/폴더)
- `/app/code/` : 소스코드
- `/app/code/tests/` : 유닛테스트 (읽기 가능)
- `/app/non-passed/` : “원래는 통과해야 하는데 현재 실패하는 테스트”에 대한 전체 정보
- `/app/non-passed/summary.jsonl` : 실패 테스트 메타정보 요약

Architect의 직접적 “입력”은 사실상:
- 실패 테스트들의 리스트/리포트(메타 + 상세)
- 현재 코드
- 테스트 코드(의도 파악용)

#### Programmer agent 입력(파일/폴더)
- `/app/code/` : 소스코드(수정 대상)
- `/app/code/tests/` : 테스트(읽기만, 수정 금지)
- `/app/requirement.xml` : Architect가 작성한 요구사항 문서(XML)

### 3.2 한 라운드에서 “구체적 출력”
#### Architect 출력
- `/app/requirement.xml` 파일 1개
- 내용 규격(1~5개 요구사항, 각 항목에 location/description/contract/acceptance 포함)
- “Incremental(한 번에 5개 이하 urgent)” + “High-level(행동/계약 중심, 구현 지시 금지)”

#### Programmer 출력
- 테스트 폴더 제외 `/app/code/` 내부 소스 변경(필요 시 새 파일 추가 가능)
- 요구사항을 만족하도록 구현/리팩토링

---

## 4) “테스트 데이터의 구체적인 인풋/아웃풋”은 어디에 있나?

### 4.1 개별 테스트 케이스의 입력/출력 값 자체
- 논문 본문에는 특정 태스크의 예로 “입력은 X, 출력은 Y” 같은 구체 값이 실려 있지 않습니다.
- 대신, “테스트가 통과/실패한다”를 정량화하는 방식(테스트 개수 기반)과, 테스트 실패로부터 요구사항을 생성하는 프로세스를 제시합니다.

### 4.2 기능 정합성(테스트 통과 수) 측정 출력
논문은 각 코드베이스 c에 대해:
- `n(c)`: 통과한 테스트 개수
- `a(c)`: normalized change ([-1,1] 범위로 개선/회귀를 정규화)
- 이를 반복 i에 대해 기록하고, 최종적으로 **EvoScore**로 집계

즉 SWE-CI에서 각 반복의 “출력(관측값)”은:
- (현재 코드베이스가) **몇 개 테스트를 통과했는지**, 이전 대비 **개선/회귀가 얼마나 되는지**, 장기적으로 **잘 유지되는지(EvoScore)** 입니다.

---

## 5) 태스크 예시를 “형식적으로” 만들어 보면 (논문 정의를 충실히 따른 예시)

논문이 제공하는 구성요소를 그대로 사용해, “한 태스크가 어떤 인풋/아웃풋을 가진다”를 예시 형태로 적으면 아래와 같습니다(실제 샘플의 코드/테스트 이름이 아니라, 논문이 정의한 **데이터 구조 예시**입니다).

### 5.1 태스크 1개(샘플)의 구성 예시
- 저장소: (어떤 Python GitHub repo)
- base commit: c0 (과거 시점)
- oracle/target commit: c\* (미래 시점)
- 고정된 환경: oracle 기준으로 만든 Docker 이미지(의존성 고정)
- 테스트 스위트 T: repo의 unit tests(pytest)

### 5.2 라운드 i에서의 입출력 예시
**입력**
- 현재 코드베이스: ci (처음엔 c0)
- 오라클 코드베이스: c\*
- 실패 테스트 정보: `/app/non-passed/summary.jsonl` 및 상세 리포트들
- 테스트 코드: `/app/code/tests/` (의도 파악용)

**출력**
1) Architect:
- ri를 XML 요구사항으로 출력(`/app/requirement.xml`)
  - 예: “어떤 모듈의 특정 함수가 특정 조건에서 예외를 내지 말고, 특정 형식의 값을 반환해야 한다” 같은 **행동 계약**

2) Programmer:
- 코드 변경으로 ci+1 생성

3) 평가 시스템(CI):
- ci+1에 대해 pytest 실행 → 통과 테스트 수 n(ci+1), normalized change a(ci+1) 산출
- 다음 라운드로 진행

---




## What counts as “training data” vs “test data” in SWE-CI?
The provided paper excerpt describes SWE-CI primarily as an **evaluation benchmark** (100 tasks) and does **not** define a conventional train/validation/test split or provide explicit “training examples” used for model fitting.

## What is a single task (sample) made of?
A task is constructed from a real GitHub Python repository and includes:
- a **base commit** (c0) and a **target/oracle commit** (c\*),
- an average gap of **233 days** and **71 commits** between them,
- the **full source code** plus a **pre-built Docker environment** for reproducibility,
- the repository’s **unit test suite** (pytest).

## What are the concrete inputs and outputs during evaluation?
The paper formalizes:
- `require_T: C × C → R` which generates a natural-language **requirement** from the gap between the current codebase and the oracle.
- `code_T: R × C → C` which updates the codebase according to that requirement.

In each iteration i (CI loop):
- **Input**: current codebase `c_i`, oracle codebase `c*`, failing-test reports (non-passed tests), and tests for inspection.
- **Output**:
  - Architect produces a requirement document (saved as `/app/requirement.xml`).
  - Programmer modifies code under `/app/code/` (tests are read-only).
  - The external CI runs tests, producing pass counts `n(c)` and derived metrics.

## Do we get explicit test-case input/output values in the paper?
Not in the provided excerpt. The excerpt explains the **process and metrics** (number of passing tests, normalized change, EvoScore) rather than listing per-task concrete I/O examples.

---



<br/>
# 요약
SWE-CI는 **CI-loop 기반의 evolution-based 평가**로, 매 반복마다 현재 코드와 oracle(타깃 커밋) 간 **테스트 실패 격차로부터 요구사항(requireT)을 생성**하고 이를 반영해 코드를 갱신(codeT)하는 방식이며(그림 1, 2.1), **Architect(요구사항 요약·원인 위치·설계 → 1~5개 고수준 요구)와 Programmer(이해·계획·구현)**의 듀얼 에이전트로 이를 수행한다(3.2, 그림 3).  
결과 평가는 각 반복의 통과 테스트 수로 **normalized change a(c)∈[-1,1]**를 계산하고, 이후 반복에 더 큰 가중치를 주는 **future-weighted mean EvoScore**로 장기 유지보수성을 수치화한다(2.2~2.3).  
예시로, 100개 과제는 평균 **233일·71커밋**의 진화 구간에서 수십 라운드로 진행되며, 실험에서 대부분 모델의 **zero-regression rate가 0.25 미만**(Claude Opus 일부만 0.5 초과)이라 장기 유지보수 중 회귀(regression) 억제가 여전히 어렵다고 보고한다(3.1, 4.2 Observation 3).




SWE-CI evaluates maintainability with an **evolution-based CI loop** where each iteration **derives requirements from the current code vs. the oracle target commit (requireT)** and updates the code accordingly (**codeT**), executed via a **dual-agent protocol**: Architect (summarize/locate/design → 1–5 high-level incremental requirements) and Programmer (comprehend/plan/code) (Fig.1, Sec.2.1, Sec.3.2, Fig.3).  
It scores progress using **normalized change a(c)∈[-1,1]** from the number of passing tests, then aggregates over iterations with a **future-weighted mean EvoScore** (γ≥1 favors later iterations) to proxy long-term maintainability (Sec.2.2–2.3).  
For example, the benchmark has **100 tasks** averaging **233 days and 71 commits** of real evolution, and experiments show most models have **zero-regression rates <0.25** (only some Claude Opus models >0.5), indicating regressions remain hard to control in long-horizon maintenance (Sec.3.1, Sec.4.2 Obs.3).

<br/>
# 기타


### 1) 다이어그램/피규어

#### Figure 1 — Snapshot-based vs Evolution-based evaluation
- **무엇을 보여주나(핵심 결과)**: 기존 벤치마크는 한 번의 요구사항→한 번의 패치(스냅샷)로 끝나 **“그 순간의 정답(기능 정합성)”**만 본다. SWE-CI는 현재 코드 상태에 따라 요구사항을 **매 반복마다 다시 생성(require\_T(ci, c\*))**하고 코드를 갱신하는 **진화 기반 루프**로 설계.
- **인사이트**: 유지보수성 차이는 “당장 테스트 통과”로는 드러나지 않고, **이전 설계/수정의 여파가 다음 반복에서 누적될 때** 드러난다(논문 서두의 “Maintainability can be revealed by tracking how functional correctness changes over time”와 연결).

#### Figure 2 — SWE-CI 데이터 큐레이션 파이프라인
- **무엇을 보여주나(핵심 결과)**: GitHub Python repo 전수 검색→필터링(3년 이상 유지, 500+ stars, 테스트/의존성 파일, permissive license)→의존성 고정 구간에서 base/oracle 커밋쌍 추출→도커 환경 자동 구성(+의존성 누락 시 self-repair)→테스트 갭/런칭 실패 등 추가 필터→**최종 100개** 선정.
- **인사이트**: (1) **의존성 변화 없는 커밋 구간**을 택해 “에이전트가 해결해야 할 난이도”를 코드 변화 자체에 더 집중시킴. (2) 도커+셀프리페어로 **재현성/실행 가능성**을 확보해 장기 반복 평가에 필요한 “실제로 돌아가는 벤치”를 만듦.

#### Figure 3 — Architect–Programmer 듀얼 에이전트 CI-loop
- **무엇을 보여주나(핵심 결과)**: 테스트 실패 요약/원인추적/설계(Architect) → 고수준 요구사항 문서 → 구현(Programmer) → 테스트 → 다음 라운드…의 CI 반복을 모사.
- **인사이트**: Programmer가 **테스트 갭 직접 최적화**가 아니라 **요구사항 문서 기반으로만 개발**하도록 강제해, 실제 CI처럼 “지금 당장 다 고치기”보다 **빠른 반복과 누적 품질**을 보게 설계(요구사항: incremental·high-level 제한).

#### Figure 4 — 8개 provider, 18개 모델 EvoScore 추이(관측 1)
- **핵심 결과(논문 Observation 1)**: 같은 provider 계열 내에서 **신모델일수록 EvoScore 상승**, 특히 **2026년 이후 출시 모델들이 큰 폭 개선**. 전체적으로 **Claude Opus 계열이 선두**, **GLM-5도 강한 성능**.
- **인사이트**: 코드 능력이 “정적 버그픽스”를 넘어 **장기 유지보수 방향으로 빠르게 진화** 중이라는 정황을 EvoScore가 포착.

#### Figure 5 — γ(미래 가중치)에 따른 모델 랭킹 변화(관측 2)
- **핵심 결과(논문 Observation 2)**: γ를 키우면(>1) 후반 반복을 더 중시 → **장기 유지보수에 강한 모델**이 유리.  
  - **장기 지향**: MiniMax, DeepSeek, GPT  
  - **단기 지향**: Kimi, GLM  
  - **상대적으로 안정적**: Qwen, Doubao, Claude
- **인사이트**: provider별로 **“단기 성과 vs 장기 안정성” 최적화 성향**이 다르고, 같은 provider 내부는 경향이 비교적 일관적 → **훈련/정렬 전략 차이**를 시사.

#### Figure 6 — Zero-regression rate(관측 3)
- **핵심 결과(논문 Observation 3)**: 대부분 모델의 **zero-regression rate < 0.25**, Claude-opus 계열 일부만 **0.5 초과**. 즉, 장기 유지보수 과정에서 **회귀(regression) 통제에 실패**하는 경우가 많음.
- **인사이트**: 스냅샷형 과제에서 성능이 좋아도, 다회 반복 유지보수에서는 작은 부채/부작용이 누적되어 **품질 안정성(회귀 방지)**이 핵심 병목임을 보여줌.

---

### 2) 테이블
- 제공된 본문 발췌에는 **별도의 정량 테이블(Table)**이 직접 포함되어 있지 않습니다(그림/서술 위주).
- 다만 **숫자 요약(사실상 테이블 성격의 통계)**은 본문 3.1에 명시:
  - 최종 **100 samples**, **68 repositories**
  - base/oracle 간 평균 **233 days**, **71 commits**
  - 테스트 제외 소스 변경 최소 **500 LOC 이상**
- **인사이트**: 태스크가 “작은 패치”가 아니라 **장기간 커밋 누적에서 생긴 실질적 변화**를 다루도록 스케일을 보장.

---

### 3) 어펜딕스(Appendix)

#### Appendix A — Related Work(관련연구 정리)
- **핵심 메시지(인사이트)**:
  1) 함수 단위 벤치(HumanEval/MBPP 등) → **현실 복잡도 부족**  
  2) 레포 단위(SWE-bench 등) → 여전히 **스냅샷 프로토콜**이라 유지보수성 평가 어려움  
  3) Long-horizon 시도(InterCode, SWE-EVO 등)도 있지만, 저자 주장으로는 **반복 개발에서의 “품질 저하 누적”을 CI 루프로 정면 모델링한 평가는 부재**  
- **SWE-CI의 자리매김 인사이트**: 유지보수성은 “작동 여부”가 아니라 **진화 과정에서 기능 정합성 변화를 추적**해야 드러난다는 문제의식을 기존 벤치들과 대비해 명확히 함.

#### Appendix B — Prompts(Architect/Programmer 시스템 프롬프트)
- **핵심 결과(설계적 포인트)**:
  - Architect는 `/app/non-passed/summary.jsonl` 기반으로 실패를 분석하고, 최종적으로 **1~5개의 요구사항만** XML로 작성(과도한 설계 방지).
  - Programmer는 `/app/requirement.xml`만 기준으로 구현하며, **테스트 실행 금지**, **tests 폴더 수정 금지**, **/app/code 내에서만 수정**.
- **인사이트**:
  - “테스트 보고 바로 고치기”가 아니라, **요구사항→구현** 분업을 강제해 CI 현실성 상승.
  - 요구사항을 incremental/high-level로 제한해, 한 라운드에서 무리한 대수술보다 **작은 변경의 누적 품질**을 관찰 가능.
  - 테스트 실행 금지는 평가 편향(에이전트가 로컬에서 반복 실행하며 과적합)을 줄이고, 외부 CI가 검증하는 구조에 맞춤.

---



### 1) Diagrams/Figures

**Figure 1 (Snapshot vs. Evolution-based evaluation)**  
- **Result shown**: Prior benchmarks use a one-shot “requirement → patch” protocol, measuring only immediate functional correctness. SWE-CI uses an **evolution loop** where requirements are re-derived at each step from the current codebase toward the oracle.  
- **Insight**: Maintainability becomes observable only when earlier decisions **propagate and accumulate** across iterations.

**Figure 2 (Data curation pipeline)**  
- **Result shown**: Large-scale GitHub repo collection + strict filters; base/oracle commit pair extraction under **unchanged dependencies**; Docker environment building with a **self-repair** mechanism for missing deps; multi-stage filtering; final **100 tasks**.  
- **Insight**: Controls confounders (dependency churn), improves reproducibility, and ensures tasks reflect substantial real evolution.

**Figure 3 (Architect–Programmer dual-agent CI loop)**  
- **Result shown**: Architect turns test gaps into incremental high-level requirements; Programmer implements based on the document; repeat via CI.  
- **Insight**: Forces realistic CI-style iteration and separates “specification” from “implementation,” making long-term code quality effects clearer.

**Figure 4 (Observation 1: EvoScore trends across models/providers)**  
- **Result**: Newer models within a provider consistently score higher; post-2026 models show larger gains. **Claude Opus leads**, **GLM-5** is also strong.  
- **Insight**: Capability is moving from static bug fixing toward sustained maintenance.

**Figure 5 (Observation 2: ranking shifts with γ)**  
- **Result**: With larger γ (more weight on later iterations), models optimized for long-term maintainability rank higher. Provider tendencies differ (e.g., MiniMax/DeepSeek/GPT more long-term; Kimi/GLM more short-term; Claude/Qwen/Doubao relatively stable).  
- **Insight**: Suggests provider-level training/optimization preferences regarding short-term vs long-term gains.

**Figure 6 (Observation 3: zero-regression rate)**  
- **Result**: Most models have **zero-regression rate < 0.25**; only some Claude-opus models exceed **0.5**.  
- **Insight**: Preventing regressions over long-horizon maintenance remains a major weakness.

---

### 2) Tables
- The provided excerpt contains **no explicit tables**.  
- Key dataset stats (table-like) appear in Section 3.1: **100 samples**, **68 repos**, avg **233 days / 71 commits**, ≥ **500 LOC** (excluding tests).  
- **Insight**: Ensures non-trivial, long-term evolutionary distance.

---

### 3) Appendices

**Appendix A (Related Work)**  
- **Insight**: Positions SWE-CI against function-level, repo-level snapshot benchmarks, and recent long-horizon attempts, arguing that none explicitly capture **cumulative quality degradation under iterative CI-like evolution**.

**Appendix B (Prompts)**  
- **Result/Design**: Architect produces 1–5 incremental high-level requirements in XML; Programmer implements under strict constraints (no test editing, no running tests, modify only `/app/code`).  
- **Insight**: Encourages CI realism, reduces overfitting/over-design, and enables cleaner attribution of long-term maintainability behaviors.

---




<br/>
# refer format:
```bibtex
@misc{chen2026sweci,
  title         = {SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration},
  author        = {Chen, Jialong and Xu, Xander and Wei, Hu and Chen, Chuan and Zhao, Bing},
  year          = {2026},
  note          = {arXiv preprint. Under review},
  eprint        = {2603.03823v3},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  month         = mar,
  date          = {2026-03-18},
  url           = {https://arxiv.org/abs/2603.03823},
}
```

Chen, Jialong, Xander Xu, Hu Wei, Chuan Chen, and Bing Zhao. “SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration.” *arXiv* preprint, March 18, 2026. https://arxiv.org/abs/2603.03823.
