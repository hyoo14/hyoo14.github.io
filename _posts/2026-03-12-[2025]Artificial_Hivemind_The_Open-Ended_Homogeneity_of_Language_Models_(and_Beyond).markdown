---
layout: post
title:  "[2025]Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)"
date:   2026-03-12 19:28:03 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: INFINITY-CHAT(실사용 오픈엔디드 질의 26K)로 70+ LLM에 대해 프롬프트당 50개 샘플을 생성하고 문장 임베딩 코사인 유사도로 **intra-model repetition**과 **inter-model homogeneity(Artificial Hivemind)**를 측정했으며, 50개 응답을 군집화해 뽑은 다양한 응답들에 대해 **25명×31,250개**의 인간 평점/선호 라벨로 평가 캘리브레이션도 분석했다.


짧은 요약(Abstract) :




이 논문은 **대규모 언어모델(LLM)이 개방형(open-ended) 과제에서 충분히 다양한(다채로운) 답을 내지 못하고, 비슷한 표현·아이디어로 수렴하는 경향**이 장기적으로는 사용자 사고의 **동질화(획일화)**로 이어질 수 있다는 문제의식에서 출발합니다. 하지만 기존에는 이런 “다양성 부족/모드 붕괴(mode collapse)”를 **현실적인 개방형 질의 전반에서, 그리고 단일 모델 내부뿐 아니라 여러 모델 간에도** 대규모로 평가할 수 있는 방법과 데이터가 부족했습니다.

이를 해결하기 위해 저자들은 **INFINITY-CHAT**이라는 데이터셋을 제안합니다. 이는 실제 사용자 질의(WildChat에서 채굴)를 기반으로 한 **2.6만(26K)개의 현실적 개방형 질문**으로 구성되며, 정답이 하나로 정해지지 않고 **여러 그럴듯한 답이 가능한 질의**들입니다. 또한 저자들은 개방형 프롬프트를 체계적으로 분류하는 **최초 수준의 포괄적 분류체계(taxonomy)**를 만들었고, 이는 **6개 상위 범주와 17개 하위 범주**로 이뤄집니다.

이 데이터셋을 이용해 70개 이상(본문 자세한 분석은 25개 주요 모델) 언어모델을 분석한 결과, 저자들은 “**Artificial Hivemind(인공지능 벌집정신)**” 현상을 보고합니다. 이는  
1) **단일 모델 안에서도 반복적으로 비슷한 답을 생성**하는 *intra-model repetition*뿐 아니라, 더 심각하게  
2) **서로 다른 모델들이 놀랄 만큼 비슷한 답으로 수렴**하는 *inter-model homogeneity*가 크게 나타난다는 것입니다. 즉 모델을 여러 개 묶어도(앙상블/스웜) 기대만큼 다양성이 나오지 않을 수 있음을 경고합니다.

또한 INFINITY-CHAT에는 **사람 평가 31,250개**가 포함되는데, (절대평점과 쌍대선호 모두) **각 예시당 25명의 독립 평가**로 “사람들 사이의 취향 차이(개인적/특이 선호)”까지 분석할 수 있게 했습니다. 실험 결과, 최신 LLM, 보상모델(reward model), LLM 판정자(judge)들은 **전체적으로는 품질이 비슷해 보이더라도**, 사람 평가자들 사이에서 **취향이 갈리는(불일치가 큰)** 응답들에 대해서는 **사람 판단과의 정렬/보정(calibration)이 더 약해지는** 문제를 보였습니다. 이는 현재 파이프라인이 “정답/좋은 답은 하나”라는 합의적 품질 개념을 가정해, **다양한 취향과 다원성을 제대로 반영·보상하지 못한다**는 한계를 드러냅니다.

종합하면, INFINITY-CHAT은 **현실 개방형 질의에서의 다양성·개방성·다원적 정렬(pluralistic alignment)**을 대규모로 진단할 수 있는 첫 자원 중 하나이며, “Artificial Hivemind”로 인한 장기적 AI 안전 리스크를 완화하는 후속 연구를 촉진하는 것을 목표로 합니다.

---




Large language models (LMs) often struggle to generate diverse, human-like creative content, raising concerns about the long-term homogenization of human thought through repeated exposure to similar outputs. Yet scalable methods for evaluating LM output diversity remain limited, especially beyond narrow tasks such as random number or name generation, or beyond repeated sampling from a single model. To address this gap, we introduce INFINITY-CHAT, a large-scale dataset of 26K diverse, real-world, open-ended user queries that admit a wide range of plausible answers with no single ground truth. We introduce the first comprehensive taxonomy for characterizing the full spectrum of open-ended prompts posed to LMs, comprising 6 top-level categories (e.g., creative content generation, brainstorm & ideation) that further breaks down to 17 subcategories. Using INFINITY-CHAT, we present a large-scale study of mode collapse in LMs, revealing a pronounced Artificial Hivemind effect in open-ended generation of LMs, characterized by (1) intra-model repetition, where a single model consistently generates similar responses, and more so (2) inter-model homogeneity, where different models produce strikingly similar outputs. INFINITY-CHAT also includes 31,250 human annotations, across absolute ratings and pairwise preferences, with 25 independent human annotations per example. This enables studying collective and individual-specific human preferences in response to open-ended queries. Our findings show that state-of-the-art LMs, reward models, and LM judges are less well calibrated to human ratings on model generations that elicit differing idiosyncratic annotator preferences, despite maintaining comparable overall quality. Overall, INFINITY-CHAT presents the first large-scale resource for systematically studying real-world open-ended queries to LMs, revealing critical insights to guide future research for mitigating long-term AI safety risks posed by the Artificial Hivemind.


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



이 논문은 **새 모델 아키텍처를 제안하거나 특정 LLM을 추가 학습(fine-tuning)하는 연구가 아니라**, “현존 LLM들이 현실 오픈엔디드(open-ended) 질의에서 얼마나 **반복/동질화(mode collapse, homogeneity)** 되는지”를 **대규모로 측정·분석**하기 위한 **데이터셋 + 평가 프레임워크(측정 방법) + 인간주석 수집 설계**를 제안하는 것이 핵심입니다. 방법론은 크게 (A) 데이터셋 구축, (B) 오픈엔디드 질의 분류(택소노미), (C) 생성 실험 설계(다수 모델·다수 샘플), (D) 반복/동질성 정량화(임베딩 유사도), (E) 인간 선호 분포 주석 수집, (F) 모델 채점기/리워드모델/LM 자체 점수의 “인간과의 보정(calibration)” 평가로 구성됩니다.

---

### A. INFINITY-CHAT 데이터셋 구축(트레이닝 데이터가 아니라 “평가/분석용 데이터”)
**목적:** 실제 사용자들이 LLM에 던지는 “정답이 하나로 고정되지 않는 질문”을 대규모로 모아, 다양성 평가를 현실적으로 하게 만들기.

- **원천 데이터:** WildChat (in-the-wild 사용자-챗봇 로그)에서 질의 추출
- **필터링 기준(논문 본문/부록 B.1):**
  1) 영어(English)  
  2) non-toxic & non-harmful (WildChat 내 플래그 활용)  
  3) GPT-4 대상으로 던진 단일 턴(single-turn) 질의  
  4) 길이 15–200 characters  
  → 이렇게 추려서 **37,426개 후보 질의** 확보

- **자동 분류 도구:** `gpt-4o-2024-11-20`을 사용해 각 질의를 아래 3축으로 분류
  1) **Meaningful Information**: 의미 있는 정보/질문인가?  
  2) **Greeting/Model Inquiry**: 인사/모델 자체 질문인가?  
  3) **Response Type: Single vs Multiple**: “단일 정답”인지 “복수로 타당한 답이 가능한지(오픈엔디드)”  
  + 불명확한 질의는 **실시간으로 최소 수정(minimal revision)** 해 명확화

- **최종 산출(본문 §2, 부록 B.1):**
  - **26,070 open-ended queries**
  - **8,817 closed-ended queries**

즉, 여기서 “트레이닝 데이터”를 만든 것이 아니라, **오픈엔디드 다양성/동질성 평가를 위한 벤치마크 질의 집합**을 구축한 것입니다.

---

### B. 오픈엔디드 질의 택소노미(분류 체계) 구축
**목적:** “오픈엔디드 질의”가 실제로 어떤 유형들로 구성되는지 체계화하고, 유형별로 다양성 붕괴가 어떻게 나타나는지 분석 기반 마련.

- **구축 방식(본문 §2, Figure 2 및 부록 B.2):**
  - 초기 약 100개 질의를 사람이 보고 라벨을 만들고
  - 반복적으로 refine/grouping하여 **계층적 분류체계**를 형성
  - 이후 전체 오픈엔디드 질의에 대해 `GPT-4o`로 카테고리 라벨링을 확장(새 유형 탐지도 포함)

- **택소노미 규모:**
  - **상위 6개(top-level)**
  - **하위 17개(subcategories)**

- **예시 상위 카테고리(본문 Figure 2):**
  - Creative Content Generation (가장 비중 큼: 58.0%)
  - Brainstorm & Ideation (15.2%)
  - Open-Endedness(철학/추상/애매한 일상/해석 등)
  - Alternative Perspectives
  - Alternative Styles
  - Information-Seeking (Problem solving/Decision/Skill/Recommendation/Explanation/Advice)

---

### C. 생성 실험 설계: 다수 모델·다수 샘플로 “반복/동질화” 측정
**목적:** 단일 모델 내부 반복(intra-model)뿐 아니라, **모델 간 출력이 서로 닮아가는 현상(inter-model)**을 대규모로 측정.

- **평가용 프롬프트 셋:** `INFINITY-CHAT100`  
  - INFINITY-CHAT에서 뽑은 **대표 오픈엔디드 100개 질의** (인간 검증 포함; 본문 §3 및 Appendix B.3 언급)

- **대상 모델 규모:**
  - 70+ LMs를 폭넓게 분석했고, 메인페이퍼에 25개 대표 모델을 상세 제시(본문 §3)
  - 오픈소스/클로즈드소스 혼합 (예: GPT 계열, Claude, Gemini, Qwen API, Llama, Mistral, DeepSeek 등)

- **샘플링(디코딩) 설정(본문 Figure 1/4/6 및 Appendix C.1):**
  1) 기본 설정: **top-p sampling**  
     - top-p = 0.9, temperature = 1.0  
     - 각 (모델, 질의)마다 **50개 응답 생성**
  2) 다양성 강화 디코딩 평가: **min-p sampling**(Nguyen et al. 2025 인용)  
     - top-p = 1.0, min-p = 0.1, temperature = 2.0  
     - 동일하게 응답 생성 후 반복성 재측정

- **생성 길이 제한:** max 2048 tokens (Appendix C.1)

---

### D. 반복/동질성 정량화 메트릭(특별한 “평가 기법”)
핵심 기법은 “**문장 임베딩 기반 의미 유사도**”로 다양성 붕괴를 수치화하는 것입니다.

- **임베딩 모델:** OpenAI `text-embedding-3-small` (본문 §3 각주 및 Appendix C.1)
- **유사도 계산:** cosine similarity

#### D-1) Intra-model repetition (모델 내부 반복)
- 한 질의에 대해 한 모델이 50개 응답을 만들면,
- 그 50개 응답 사이 **평균 pairwise 임베딩 유사도**를 계산
- 이를 질의 100개에 대해 수행하고, 유사도 구간별 분포를 히트맵으로 제시(Figure 4, Figure 5)

**주요 결과(본문 §3):**
- top-p=0.9, temp=1.0에서도 **79% 경우 평균 유사도 > 0.8**  
- min-p로도 반복이 줄긴 하지만 여전히 **61.2%가 >0.8**, **81%가 >0.7** 수준(Figure 5 설명)

#### D-2) Inter-model homogeneity (모델 간 동질화)
- 서로 다른 모델들이 같은 질의에 대해 만든 응답들 간에도 유사도를 계산
- 모델쌍별 평균 유사도가 **71%~82%** 정도로 높게 나옴(Figure 6 설명)
- 어떤 경우에는 **문구가 상당히 겹치거나(verbatiam span overlap)** 아예 동일 문장을 생성하기도 함(본문 §3, Figure 6 주변 서술)

#### D-3) 클러스터 “섞임” 지표(Top-N 유사 응답의 출처 모델 수)
- 각 질의에서 “가장 서로 비슷한 응답 Top-N”을 모았을 때,
- 그 집합이 **몇 개의 서로 다른 모델에서 왔는지** 계산(Figure 8)
- N=50에서 평균적으로 **약 8개 모델**이 한 유사 클러스터에 섞여 들어옴 → 모델 간 경계가 흐려짐(본문 §3)

#### D-4) 저차원 시각화(PCA로 클러스터링 경향 관찰)
- 응답 임베딩을 PCA로 2D 투영 후 군집을 시각화(Figure 1, 15–18)
- 예: “time metaphor”에서 대부분이 **‘time is a river’** vs **‘time is a weaver’** 두 덩어리로 모이는 현상(Figure 1)

---

### E. 인간 주석 수집 설계(“분포적 선호”를 얻기 위해 25명씩 촘촘히)
**목적:** 오픈엔디드 과제는 “좋은 답이 여러 개”일 수 있는데, 기존 데이터는 주석이 희소(예: 3명)하여 개인 취향/이견을 반영하기 어렵다. 그래서 이 논문은 **한 예시당 25명**으로 “선호 분포”를 직접 측정.

- **주석 규모(본문 Abstract, §4.1, Appendix D.1):** 총 **31,250 human annotations**
  1) **절대평가(absolute rating, 1~5점):**  
     - 50개 프롬프트 × 프롬프트당 15개 응답 × 응답당 25명  
     - 25×15×50 = **18,750**
  2) **쌍대비교(pairwise preference):**  
     - 50개 프롬프트 × 프롬프트당 10개 응답쌍 × 쌍당 25명  
     - 25×10×50 = **12,500**

- **응답 샘플링 방식(중요, Appendix D.1):**
  - 한 프롬프트에 대해 여러 모델이 생성한 모든 응답을 모으면 중복이 많기 때문에,
  - **클러스터링으로 15개 그룹으로 나누고**, 각 클러스터에서 1개씩 뽑아 “상대적으로 다양한” 15개를 구성

- **불일치(이견) 정량화:**
  - **Shannon entropy**로 25명 라벨 분포의 혼란도(불일치)를 측정(Figure 7, Figure 9)

---

### F. “평가하는 모델들”이 인간을 얼마나 잘 따라가나? (보정/캘리브레이션 실험)
**목적:** 생성이 동질적일 뿐 아니라, “서로 다른(하지만 비슷하게 좋은) 답”을 평가할 때 **LM / reward model / LM judge**가 인간과 잘 맞는지 검증.

- **비교 대상 점수(본문 §4.2 및 Appendix D.2):**
  1) **LM perplexity 기반 점수**: 질의+응답 조건에서의 perplexity를 “품질 프록시”로 사용(낮을수록 더 자연스럽다고 간주)  
  2) **Reward model scalar score**: RLHF 계열 보상모델 점수  
  3) **LM judge score**: GPT-4o, Prometheus 등에게 루브릭(Overall, HHH)으로 1~5점 채점시킴(Figure 23, 24가 프롬프트)

- **핵심 분석 아이디어(본문 §4.3, Figure 10~11):**
  - (1) **비슷한 품질(similar-quality)** 응답들만 모아 비교하면 상관이 더 떨어지는가?  
  - (2) **사람들끼리 의견이 갈리는(high disagreement)** 예시들만 모아 비교하면 상관이 더 떨어지는가?

- **유사-품질 subset 추출(절대평가의 경우, 본문/Appendix D.3):**
  - **Tukey’s fences**로 outlier 제거하며 “품질이 비슷한 응답들” subset 구성
- **불일치 subset 추출:**
  - 절대평가: 엔트로피 상위 % 추출
  - 쌍대비교: disagreement 비율로 상위 추출(본문 수식 제시)

- **결론(본문):**
  - 전체 평균 품질에서는 어느 정도 맞아도,
  - **비슷하게 좋은 대안들**이나 **개인 취향이 갈리는 경우**에 대해선  
    SOTA LMs/Reward Models/LM Judges가 **인간 분포를 잘 캘리브레이트 못함**(상관이 유의미하게 하락)

---




This paper does **not** introduce a new model architecture or a new training recipe for LLMs. Instead, it contributes a **dataset + taxonomy + large-scale evaluation framework + dense human preference annotations** to study diversity collapse (“Artificial Hivemind”) in **real-world open-ended queries**.

### A. Building INFINITY-CHAT (evaluation dataset)
- Source: **WildChat** in-the-wild user queries.
- Filtering (Appendix B.1): English, non-toxic/non-harmful, single-turn GPT-4 queries, length 15–200 chars → **37,426 candidates**.
- GPT-4o (`gpt-4o-2024-11-20`) classifies each query along:
  1) meaningful information,
  2) greeting/model inquiry,
  3) response type: **Single vs Multiple** (open-ended).
  Ambiguous queries are minimally revised.
- Result: **26,070 open-ended** + **8,817 closed-ended** queries.

### B. Taxonomy of open-ended prompts
- Semi-automatic taxonomy construction (Appendix B.2): manual seed labeling + iterative refinement → **6 top-level categories / 17 subcategories** (Figure 2).
- GPT-4o scales labeling to the full set and detects novel types.

### C. Generation experiment design (many models, many samples)
- Evaluation set: **INFINITY-CHAT100** (100 representative open-ended prompts; human-verified).
- Models: 70+ LMs studied; 25 highlighted in the main paper (open + closed source).
- Decoding:
  - Top-p sampling: **p=0.9, temperature=1.0**, max 2048 tokens; **50 generations per (model, prompt)**.
  - Diversity-oriented decoding: **min-p** with top-p=1.0, min-p=0.1, temperature=2.0.

### D. Quantifying repetition/homogeneity (key technique)
- Sentence embeddings: OpenAI **text-embedding-3-small**; cosine similarity.
- **Intra-model repetition:** average pairwise embedding similarity among 50 responses for the same prompt (Figures 4–5).
- **Inter-model homogeneity:** embedding similarity across outputs from different models (Figure 6), including verbatim span overlap cases.
- Additional indistinguishability measure: among top-N most similar responses, count how many unique source models appear (Figure 8).
- Visualization: PCA projection of embeddings to show clustering (Figures 1, 15–18).

### E. Dense human annotation to capture preference distributions
- Total: **31,250 annotations** with **25 independent annotators per item** (Abstract, §4.1).
- Absolute ratings (1–5): 50 prompts × 15 diverse responses × 25 = **18,750**.
- Pairwise preferences: 50 prompts × 10 pairs × 25 = **12,500**.
- Response selection for annotation (Appendix D.1): cluster all candidate responses into 15 groups and sample one per cluster to encourage diversity.
- Disagreement quantified via **Shannon entropy** (Figures 7 and 9).

### F. Calibration of LMs / reward models / LM judges vs humans
- Model-based scoring types (§4.2, Appendix D.2):
  1) LM perplexity-based scores,
  2) reward model scalar scores,
  3) LM-as-a-judge scores (GPT-4o / Prometheus) using Overall and HHH rubrics (Figures 23–24).
- Key tests (§4.3):
  - performance drops on **similar-quality** subsets (e.g., Tukey’s fences),
  - performance drops on **high-disagreement** subsets (entropy/disagreement-based).
- Finding: models are less calibrated to humans when multiple high-quality answers exist and/or annotators have idiosyncratic preferences.

---





<br/>
# Results



---

### 1) 테스트 데이터(벤치마크/셋업)

#### 1.1 INFINITY-CHAT (전체 데이터)
- **구성**: WildChat에서 채굴한 “현실 사용자” 단일 턴 질의 기반.
- **규모**: 최종적으로  
  - **오픈엔디드(open-ended) 질의 26,070개**,  
  - **클로즈드엔디드(closed-ended) 질의 8,817개**  
  를 구축(§2, Appendix B.1).
- **오픈엔디드 정의**: “단일 정답이 없고 여러 개의 그럴듯한 답이 가능한 질의”.
- **분류(택소노미)**: 오픈엔디드 질의를 **6개 상위 카테고리 / 17개 하위 카테고리**로 정리(Figure 2, §2).

#### 1.2 INFINITY-CHAT100 (실험용 100개 프롬프트)
- 본격적인 “Artificial Hivemind(동질화/모드 붕괴)” 측정은 **INFINITY-CHAT에서 대표 오픈엔디드 100개**를 뽑은 **INFINITY-CHAT100**으로 수행(§3).
- 이 100개는 “human verified to be open-ended”(Appendix B.3)라고 명시.

#### 1.3 사람 평가(인간 주석)용 서브셋
- 사람 평가는 INFINITY-CHAT100에서 **50개 프롬프트**를 샘플링하고,
  - 절대평가(1~5점): 프롬프트당 **15개 응답**, 각 (Q,R)에 **25명 평가** → **18,750 라벨**
  - 쌍대선호: 프롬프트당 **10쌍**, 각 (Q,R1,R2)에 **25명 평가** → **12,500 라벨**
  - 합계 **31,250** 인간 라벨(§4.1, Appendix D.1)

---

### 2) 경쟁모델(비교 대상 모델군)

#### 2.1 생성 다양성/동질화 분석에 사용한 모델
- 논문은 “**70+ open/closed-source LMs**”에 대해 실험했다고 말하며(§1, §3), 본문 상세 표/그림은 그 중 **25개 대표 모델**을 주로 제시(예: Figure 1, Figure 6).
- 부록 Table 5에는 분석에 포함된 모델 리스트가 광범위하게 제시됨(오픈소스 + 클로즈드소스).
  - 오픈소스 계열 예: Llama(3.1/3.2/3.3), Qwen(1.5/2/2.5/3), Gemma, Mistral/Mixtral, Phi, OLMo, Tulu 등
  - 클로즈드소스 예: OpenAI GPT-4o/mini/turbo, Anthropic Claude, Google Gemini, Qwen API 모델 등

#### 2.2 “평가자” 모델(인간 점수와의 정렬/캘리브레이션 비교)
논문은 “모델이 모델 응답을 평가하는 능력”도 비교하는데, 여기서 경쟁대상은 3부류(§4.2):
1) **LM perplexity 기반 점수**(응답의 모델-우도/유창성 proxy)
2) **Reward model** 6종(RewardBench 상위권 언급)
3) **LM judge** 4종(예: GPT-4o, Prometheus 변형 등)  
(전체 리스트와 절차는 Appendix D.2 및 Table 23~26)

---

### 3) 메트릭(무엇을 측정했나)

논문 결과는 크게 (A) “생성 동질화(Artificial Hivemind)”와 (B) “평가 정렬/캘리브레이션” 두 축입니다.

---

## A. 생성 동질화(Artificial Hivemind) 결과

### 3A-1) 핵심 개념: intra-model repetition vs inter-model homogeneity
- **Intra-model repetition(모델 내부 반복)**: 같은 모델이 같은 프롬프트에 대해 여러 번 샘플링해도 응답이 서로 비슷해지는 현상(§3).
- **Inter-model homogeneity(모델 간 동질성)**: 서로 다른 모델(가족/크기/회사)이 독립적으로 생성했는데도 응답 의미/표현이 비슷해지는 현상(§3, Figure 6).

---

### 3A-2) 생성 설정(공정 비교를 위한 통일된 디코딩)
- INFINITY-CHAT100의 각 프롬프트에 대해 **모델당 50개 응답** 생성(§3, Appendix C.1)
- 기본 샘플링: **top-p = 0.9, temperature = 1.0** (Figure 1/4/6 캡션 및 Appendix C.1)
- 대안 디코딩 평가: **min-p sampling**도 실험  
  - 설정: top-p=1.0, min-p=0.1, temperature=2.0 (Figure 5, Appendix C.1)

---

### 3A-3) 동질화 측정 메트릭: “문장 임베딩 코사인 유사도” 기반
- 응답들을 임베딩으로 바꾼 뒤(문장 임베딩), **cosine similarity**로 유사도를 계산.
- 임베딩 모델: **OpenAI text-embedding-3-small** 사용(§3 각주, Appendix C.1).

---

### 3A-4) 결과 1: intra-model repetition(한 모델 안에서도 다양성 붕괴)
- 절차: 한 프롬프트에 대한 50개 응답들 사이의 **평균 pairwise 임베딩 유사도**를 계산하고, 유사도 구간별로 프롬프트 비율을 집계(Figure 4 설명).
- **주요 결과(본문 요약)**:
  - top-p=0.9, t=1.0에서도 **79%의 경우 평균 유사도가 0.8을 초과**(§3, Figure 4 설명).
  - “최대로 공격적(고확률 다양화) 샘플링”을 해도 반복이 심하다고 해석(§3).

#### min-p 결과(디코딩만 바꿔도 해결이 안 됨)
- Figure 5/§3에서:
  - 극단적 반복(>0.9)은 줄지만,
  - 여전히 **81%의 응답쌍이 0.7 초과**, **61.2%가 0.8 초과** 유사도.
- 결론: **디코딩 트릭만으로는 모드 붕괴가 근본 해결되지 않음**(§3).

---

### 3A-5) 결과 2: inter-model homogeneity(모델이 달라도 비슷한 답으로 수렴)
- Figure 6 및 §3 “Inter-model homogeneity”에서:
  - 서로 다른 모델들의 출력 간 평균 유사도가 **대략 71%~82% 범위**라고 보고.
  - 예시로:
    - DeepSeek-V3 ↔ qwen-max-2025-01-25 : 0.82
    - DeepSeek-V3 ↔ gpt-4o-2024-11-20 : 0.81
- 정성 사례: “Write a metaphor about time”에서 다양한 모델이 **“time is a river”** 또는 **“time is a weaver”** 두 군집으로 크게 몰림(Figure 1).
- 더 나아가, 서로 다른 모델이 **부분적으로 거의 동일한 구절을 공유**하거나(“Elevate your iPhone with our …”), **완전히 동일한 문장**을 내는 경우도 보고(§3, Figure 6 아래 본문).

---

### 3A-6) 결과 3: “가장 비슷한 응답 클러스터에 여러 모델이 섞여 들어감”
- 측정: 각 쿼리별로 “top-N 가장 유사한 응답들”을 모아 그 안에 포함된 **서로 다른 source model의 개수**를 센 뒤 평균을 봄(Figure 8 설명).
- 결과: N=50일 때, top-50 유사 응답 클러스터가 **평균 약 8개 모델**에서 섞여 나옴(Figure 8, §3).
- 해석: 모델을 여러 개 모아도(앙상블/스웜) **출력이 본질적으로 다양해지지 않을 수 있음**을 뒷받침.

---

## B. “평가” 관점 결과: LMs/Reward Models/LM Judges가 인간 선호를 제대로 따라가나?

여기서는 “응답 생성 자체”가 아니라, **여러 대안 응답을 평가/채점하는 모델 점수**가 **인간 판단과 얼마나 일치하는지**를 비교합니다(§4).

---

### 3B-1) 인간 라벨의 특성 메트릭: Shannon entropy(불일치/다양한 선호)
- 오픈엔디드 질문은 “정답 하나”가 없어서 사람도 의견이 갈림.
- 이를 정량화하기 위해:
  - 쌍대선호(응답1 vs 응답2)에서 25명 라벨 분포의 **Shannon entropy**를 계산(Figure 7).
  - 절대평가(1~5점)도 라벨 분포에 대해 **Shannon entropy**를 계산(Figure 9).
- 결과(정성): entropy가 높은 케이스가 많고, 즉 **사람 사이 불일치가 흔함**을 보여줌(§4.1).

---

### 3B-2) 비교 대상 “모델 점수” 3종
§4.2에 따라 인간 평균 점수와의 상관을 비교:
1) **LM perplexity 기반 점수**: (질의,응답)에서 응답 perplexity를 계산해 품질 proxy로 사용(낮을수록 좋음)  
2) **Reward model scalar output**
3) **LM judge 점수**:  
   - Overall rubric, 또는 HHH(Helpfulness/Harmlessness/Honesty) rubric로 1~5점 채점(§4.2, Appendix D.2)

---

### 3B-3) 핵심 비교(상관 분석): “전체 vs (비슷한 품질만 모은 subset) vs (사람이 크게 불일치한 subset)”
논문은 상관을 “전체에서 높아 보여도, 어려운 구간에서 무너진다”는 구조로 보입니다.

#### (1) 비슷한 품질(same/similar-quality) 응답들만 놓고 보면 상관이 떨어짐
- 절대평가(absolute rating):
  - Tukey’s fences 등으로 “비슷한 인간 점수 범위” subset을 만들고,
  - 모델 점수 vs 인간 평균 점수 **상관(Pearson/Spearman)**이 **full set 대비 낮아진다**고 보고(Figure 10(a), §4.3).
- 쌍대선호(pairwise):
  - “두 응답이 비슷하다”고 판단된 비율이 높은 예시들만 모아 subset을 만들면,
  - 모델의 선호 차이 점수와 인간 선호 차이의 상관이 **역시 감소**(Figure 11(a), §4.3).

#### (2) 사람끼리 불일치가 큰(high-disagreement) 케이스에서 상관이 떨어짐
- 절대평가:
  - Shannon entropy가 높은 상위 x%를 “disagreed subset”으로 두고,
  - full 대비 상관이 크게 하락(Figure 10(b), §4.3).
- 쌍대선호:
  - “percentage disagreement”로 불일치 큰 subset을 만들고,
  - 상관이 하락(Figure 11(b), §4.3).

**요지**: 최신 LMs/Reward Models/LM Judges가 “전반적으로는 그럴듯한 평가”를 해도,  
- “둘 다 괜찮은 답(동급 품질)”을 동급으로 보거나,  
- “사람 취향이 갈리는 답”을 사람 분포에 맞게 채점하는 데는 **덜 정렬되어 있고 캘리브레이션이 약함**.

---





---

### 1) Test data (benchmarks / setups)

#### 1.1 INFINITY-CHAT (full dataset)
- Built by mining real user single-turn queries from WildChat and filtering/refining them (Sec. 2, Appx. B.1).
- Final scale:
  - **26,070 open-ended queries**
  - **8,817 closed-ended queries**
- Open-ended queries are those that allow **multiple plausible answers with no single ground truth**.
- A taxonomy covering **6 top-level categories and 17 subcategories** is introduced (Fig. 2).

#### 1.2 INFINITY-CHAT100 (100-prompt evaluation subset)
- The main “Artificial Hivemind” homogeneity experiments use **100 representative open-ended prompts** (Sec. 3), human-verified as open-ended (Appx. B.3).

#### 1.3 Human-annotation subset
- For calibration studies, they sample **50 prompts** from INFINITY-CHAT100 and collect dense human labels:
  - Absolute ratings (1–5): **15 responses per prompt**, **25 ratings per (Q,R)** → **18,750 labels**
  - Pairwise preferences: **10 pairs per prompt**, **25 labels per (Q,R1,R2)** → **12,500 labels**
  - Total: **31,250 human annotations** (Sec. 4.1, Appx. D.1)

---

### 2) Competing models (what is compared)

#### 2.1 Models used for generation homogeneity
- The paper reports experiments over **70+ open- and closed-source LMs**, with **25 representative models** highlighted in the main figures (Sec. 1, Sec. 3; model list in Appx. Table 5).
- Includes major families: Llama, Qwen, Gemma, Mistral/Mixtral, Phi, OLMo/Tulu, plus closed-source GPT-4o, Claude, Gemini, and Qwen API models.

#### 2.2 Models used as “evaluators” (calibration to human judgments)
Three classes of scoring models are compared against human ratings (Sec. 4.2):
1) **LM perplexity-based scores**
2) **Reward models** (6 models; RewardBench-referenced)
3) **LM judges** (e.g., GPT-4o and Prometheus variants) with Overall and HHH rubrics (Appx. D.2; Tables 23–26)

---

### 3) Metrics (what is measured)

The results split into two major axes:

## A. Generation homogeneity (“Artificial Hivemind”)

### A1) Definitions
- **Intra-model repetition**: repeated sampling from the same model yields highly similar responses.
- **Inter-model homogeneity**: different models converge to similar responses (Sec. 3).

### A2) Generation protocol (controlled decoding)
- For each prompt in INFINITY-CHAT100, each model generates **50 responses** (Sec. 3; Appx. C.1).
- Main decoding: **top-p = 0.9, temperature = 1.0**.
- Additional decoding test: **min-p sampling** (top-p=1.0, min-p=0.1, temperature=2.0) (Fig. 5; Appx. C.1).

### A3) Similarity metric
- Responses are embedded (sentence embeddings) and compared using **cosine similarity**.
- Embeddings: **OpenAI text-embedding-3-small** (Sec. 3 footnote; Appx. C.1).

### A4) Key findings
- **Intra-model**: with top-p=0.9, t=1.0, in **79%** of cases, the *average pairwise similarity* within a model’s 50-response pool is **> 0.8** (Fig. 4; Sec. 3).
- **Min-p** reduces extreme repetition but still shows collapse: **81%** of response pairs exceed **0.7** similarity and **61.2%** exceed **0.8** (Fig. 5; Sec. 3).
- **Inter-model**: average cross-model similarity ranges roughly **71%–82%**, with specific examples like DeepSeek-V3 vs qwen-max reaching **0.82**, and DeepSeek-V3 vs gpt-4o reaching **0.81** (Sec. 3; Fig. 6).
- Qualitatively, even highly open-ended prompts (e.g., “Write a metaphor about time”) cluster into a small number of dominant concepts (e.g., “time is a river” vs “time is a weaver”) (Fig. 1).
- Cross-model indistinguishability: among the top-N most similar responses per query, the cluster often contains outputs from **~8 unique models on average** for N=50 (Fig. 8; Sec. 3).

---

## B. Calibration of model-based evaluators to human judgments

### B1) Human disagreement metrics
- They quantify label dispersion using **Shannon entropy** over 25 annotator labels for:
  - Pairwise preferences (Fig. 7)
  - Absolute ratings (Fig. 9)

### B2) Model scoring types
- **LM perplexity** (as a fluency/likelihood proxy)
- **Reward model scalar scores**
- **LM judge scores** using Overall and HHH rubrics (Sec. 4.2; Appx. D.2)

### B3) Core comparisons
They compute correlations between model scores and (average) human scores:
- On the **full set**
- On **similar-quality subsets** (responses humans rate as comparably good)
- On **high-disagreement subsets** (where humans disagree most)

**Result:** correlations drop notably in both the similar-quality and high-disagreement subsets (Figs. 10–11; Sec. 4.3), suggesting weaker calibration exactly where pluralistic preferences matter most.

---





<br/>
# 예제



---

## 1) (입력/출력 예시) INFINITY-CHAT 쿼리(프롬프트) 데이터: “테스트 입력”에 해당

### 1.1 데이터 소스와 필터링 조건 (입력 데이터가 어떻게 생겼는가)
논문은 WildChat에서 실제 사용자의 단일 턴 질의를 뽑아 “오픈엔디드 쿼리”를 만듭니다.

- **원천**: WildChat의 “고품질, 단일 턴 GPT-4 질의”
- **필터**: 영어, 비유해/비독성, 길이 15–200자 등  
  - 근거: §2 “Mining in-the-wild open-ended user queries”, Appendix B.1
  - 발췌 근거(요지): “From 37,426 high-quality, single-turn GPT-4 queries (English, non-toxic, 15–200 characters) …”

즉, **테스트 입력**은 “사용자가 챗봇에게 실제로 던진 질문/요청(프롬프트)”이며, 그 형태는 매우 다양합니다(글쓰기 요청, 아이디어 요청, 설명 요청, 조언 요청, 가상 시나리오 등).

---

### 1.2 “오픈엔디드 쿼리”의 구체 예시 (입력 예시)
논문 본문/부록 표에서 프롬프트 예시를 다량 제공합니다. 대표적으로:

#### (A) 창작/생성형 입력 예시 (Creative Content Generation)
- “Write a metaphor involving time.” (시간에 대한 은유를 써라)  
  - 근거: Figure 1, Table 16, Table 17, Table 3
- “Write a 30-word essay on global warming.” (지구온난화 30단어 에세이)  
  - 근거: Table 13, Table 3
- “Create the first verse of a wedding vow.” (결혼 서약 첫 구절)  
  - 근거: Table 15

#### (B) 정보/설명 요청형 입력 예시 (Information-Seeking / Explanation)
- “Help me draft a paragraph as an expert consultant explaining TOEFL vs IELTS for international students.”  
  - 근거: Figure 7 예시 쿼리, Table 3
- “Explain computational irreducibility like I’m 5.”  
  - 근거: Table 3

#### (C) 브레인스토밍/아이디어 입력 예시 (Brainstorm & Ideation)
- “Suggest a feature for a smartwatch designed specifically for senior citizens.”  
  - 근거: Figure 2(분류 체계 예시)

#### (D) 가정/가상 시나리오 입력 예시 (Speculative & Hypothetical)
- “What would happen in the morning if gravity on Earth doubled overnight?”  
  - 근거: Figure 2(분류 체계 예시)
- “If there were double the amount of oxygen in the air, what would happen? Write in 100 words.”  
  - 근거: Table 3

---

## 2) (출력 예시) LLM이 생성한 “응답(모델 출력)”의 구체 예시

이 논문에서 가장 중요한 “아웃풋 예시”는 **같은 프롬프트에 대해 여러 모델(또는 같은 모델을 여러 번 샘플링)이 만들어낸 답변들**입니다. 여기서 인풋은 프롬프트 1개, 아웃풋은 응답 N개(반복 샘플링)입니다.

---

### 2.1 대표 사례: “Write a metaphor about time”에 대한 여러 모델 출력이 ‘두 클러스터’로 붕괴
**Figure 1**이 논문의 핵심 예시입니다.

- **입력(테스트 프롬프트)**: “Write a metaphor about time”
- **출력(모델 응답들)**: 25개 모델이 각각 50개씩 생성(총 1250개 응답)
- **관찰된 출력 패턴(모드 붕괴/동질화)**  
  - 대부분 “time is a river(시간은 강이다)” 계열
  - 일부 “time is a weaver(시간은 직조공이다/천을 짠다)” 계열

Figure 1에 실린 실제 모델 출력 문장 예:
- “Time is a river, endlessly flowing, carrying moments like leaves that drift away, never to return.” (gpt-4o-2024-11-20)
- “Time is a river, flowing silently and ceaselessly, carrying leaves from birth to oblivion…” (Qwen2.5-72B-Instruct)
- “Time is a silent weaver, meticulously threading moments into the tapestry of our lives…” (phi-4)

이 예시는 “모델/크기/회사 다 달라도 은유가 2가지로 수렴”한다는 것을 시각적으로 보여줍니다.

---

### 2.2 같은 모델 내부에서도 반복(자기 반복)되는 출력 예시
**Figure 6**(및 Figure 4 설명, 본문 “Intra-model repetition”)에서, 같은 프롬프트에 대해 같은 모델이 여러 번 샘플링해도 비슷한 답을 반복한다는 예시를 듭니다.

예:
- **입력**: “Write a pun about peanut.”
- **출력(서로 매우 유사)**:
  - “What did the peanut say when it was chasing another peanut? I’m gonna cashew!”
  - “What did the peanut say when it was chasing someone? I’m gonna cashew!”
- 또한 “Generate a motto …”에서 아예 동일 문장 반복(sim=1.0) 사례도 제시  
  - “Empower Your Journey: Unlock Success, Build Wealth, Transform Yourself.”가 동일 출력

---

### 2.3 서로 다른 모델 간에도 문장 조각까지 겹치는(상호 동질화) 출력 예시
**Figure 6** 본문 설명에서, 서로 다른 모델이 **표현(표면 텍스트)**까지 겹친다고 말합니다.

예:
- **입력**: “Create a description with 2-3 sentences for an iPhone case collection that is a slim-fitted case with bold designs.”
- **서로 다른 모델 출력에 겹친 구절 예**:
  - “Elevate your iPhone with our …”
  - “sleek, without compromising …”
  - “bold, eye-catching …”

논문은 이런 현상을 “Artificial Hivemind” 효과(서로 다른 모델이 같은 방향으로 수렴)로 강조합니다.

---

## 3) (구체적 태스크/실험 세팅) 테스트(평가) 과업을 “입력→출력→채점/측정” 흐름으로 정리

### 3.1 과업 A: 오픈엔디드 프롬프트에 대해 “다양한 응답 생성” 후 유사도 측정 (다양성/반복성 평가)
- **입력**: INFINITY-CHAT100의 오픈엔디드 프롬프트 100개  
  - 근거: §3 “Using a subset of 100 representative… INFINITY-CHAT100”
- **출력**: 모델별 프롬프트당 50개 응답 샘플링
- **디코딩(생성) 조건 예시**:
  - top-p=0.9, temperature=1.0 (Figure 4/6/1 설명)
  - 또는 min-p(top-p=1.0, min-p=0.1, temperature=2.0) (Figure 5 설명)
- **측정값(스코어)**:
  - 응답들 간 sentence embedding cosine similarity 평균
  - 임베딩: OpenAI text-embedding-3-small  
    - 근거: Figure 4 각주 “Sentence embeddings from OpenAI’s text-embedding-3-small API”
- **아웃풋(평가 결과의 형태)**:
  - 유사도 구간별로 “해당 프롬프트 비율(%)” 히트맵 (Figure 4, Figure 5)
  - 모델 간 평균 유사도 매트릭스 (Figure 6, Appendix C Table 7-11)

즉, 이 과업에서 “테스트 데이터 인풋”은 프롬프트, “테스트 데이터 아웃풋”은 모델이 만든 응답 50개×100개×모델수이며, 최종 평가지표는 유사도 기반 반복/동질화입니다.

---

### 3.2 과업 B: “프롬프트 패러프레이즈(의미 동일 재서술)”를 해도 동질화가 유지되는지 테스트
Appendix C.4에 있는 실험입니다.

- **입력**:
  - 원 프롬프트 30개
  - 각 프롬프트당 패러프레이즈 4개 추가 → 총 150개 프롬프트(30×5)
- **출력**:
  - 모델당 프롬프트 변형(5개) 각각 20개 응답 생성
  - 모델: 42개 대표 모델
- **측정**:
  - within-prompt similarity(같은 프롬프트 내부 20개 응답 유사도)
  - cross-paraphrase similarity(원문 프롬프트 응답과 패러프레이즈 프롬프트 응답의 유사도)
- **결과 예**:
  - within 0.821, cross 0.781 (차이 0.04) → 패러프레이즈해도 유사도가 여전히 높음
- **구체 예시 표**:
  - Table 17: “Write a metaphor involving time” 패러프레이즈들에도 계속 “Time is a river”로 수렴하는 모델별 출력 예시
  - Table 18: “internet shaped society” 류에서 “profoundly …” 같은 표현 반복

---

### 3.3 과업 C: 사람 평가(휴먼 라벨) 데이터 수집—절대평가/쌍대비교(선호) 라벨
§4.1 및 Appendix D.1의 데이터 수집이 해당합니다. 여기서의 “인풋/아웃풋”은 **평가 태스크 관점**입니다.

#### (1) 절대평가(Absolute rating) 태스크
- **입력(평가자에게 주는 것)**: (Query, Response)
- **출력(사람 라벨)**: 1–5 품질 점수
- **규모**:
  - 프롬프트 50개
  - 프롬프트당 “서로 다른” 응답 15개(클러스터링으로 다양성 확보)
  - (Query, Response)마다 25명 평가
  - 총 18,750 라벨(25×15×50)
- **근거**: §4.1, Appendix D.1

#### (2) 쌍대비교(Pairwise preference) 태스크
- **입력**: (Query, Response1, Response2)
- **출력(사람 라벨)**: 어느 쪽이 더 나은지(강/약 선호, 또는 동급) 같은 선호 라벨 분포
- **규모**:
  - 프롬프트 50개
  - 프롬프트당 응답쌍 10개
  - 각 쌍마다 25명 평가
  - 총 12,500 라벨(25×10×50)
- **근거**: §4.1, Appendix D.1

또한 논문은 **사람들 사이의 의견 불일치**를 Shannon entropy로 보여줍니다(Figure 7, Figure 9).

---

### 3.4 과업 D: “모델 평가자”들(LM perplexity / Reward Model / LM judge)이 사람 평균점수와 얼마나 맞는지(상관) 평가
§4.2~4.3 및 Figure 10~11, Appendix D.2~D.4입니다.

- **입력**: (Query, Response) 또는 (Query, R1, R2)
- **모델 평가자 출력**:
  1) LM perplexity 기반 점수(유창성/가능도 proxy)
  2) Reward model scalar score
  3) LM judge가 rubric에 따라 1~5점 채점(Overall, 또는 HHH)
- **정답(그라운드 트루스에 준하는 것)**:
  - 25명 인간 평가의 평균(또는 선호 분포)
- **평가**:
  - Spearman/Pearson correlation (그림/표에 따라)
  - 특히 “비슷한 품질”이거나 “사람 의견이 갈리는” subset에서 상관이 떨어짐(Figure 10, 11)

---




## 1) INFINITY-CHAT query data: concrete “test inputs”
### 1.1 Source and filtering (what the inputs look like)
The dataset is mined from WildChat single-turn user queries.

- **Source**: WildChat real user → GPT-4 single-turn prompts
- **Filters**: English, non-toxic/non-harmful, 15–200 characters  
  - Evidence: §2; Appendix B.1 (“From 37,426 … (English, non-toxic, 15–200 characters)”)

So the **test inputs** are real-world user prompts spanning many open-ended interaction types.

### 1.2 Concrete prompt examples (inputs)
Examples appear throughout the paper (Figure 2, Table 3, etc.):

- Creative: “Write a metaphor involving time.” (Figure 1; Table 16/17; Table 3)  
- Creative: “Write a 30-word essay on global warming.” (Table 13; Table 3)  
- Writing: “Create the first verse of a wedding vow.” (Table 15)  
- Explanation: “Help me draft a paragraph … TOEFL vs IELTS …” (Figure 7; Table 3)  
- Hypothetical: “If there were double the amount of oxygen in the air …” (Table 3)

---

## 2) LLM-generated responses: concrete “model outputs”
### 2.1 Key example: “metaphor about time” collapses into two clusters
Figure 1:

- **Input**: “Write a metaphor about time”
- **Outputs**: 25 models × 50 samples each
- **Observed modes**:
  - dominant: “time is a river”
  - secondary: “time is a weaver”
- Figure 1 lists real outputs from multiple models illustrating these modes.

### 2.2 Intra-model repetition examples
Figure 6 shows near-duplicate outputs from repeated sampling, e.g.:

- **Input**: “Write a pun about peanut.”
- **Outputs**: nearly identical punchlines (“I’m gonna cashew!”)
- Also identical motto outputs (similarity = 1.0) are reported.

### 2.3 Inter-model homogeneity examples (overlapping phrases)
Figure 6 description notes overlapping fragments across different models for an iPhone case description prompt (e.g., “Elevate your iPhone with our …”, “bold, eye-catching …”).

---

## 3) Concrete tasks/experiments as input→output→metric pipelines
### 3.1 Task A: generate many responses and measure similarity (diversity / repetition)
- **Input**: 100 prompts (INFINITY-CHAT100)
- **Output**: 50 responses per prompt per model
- **Decoding**: top-p=0.9, temp=1.0; or min-p setup
- **Metric**: cosine similarity of sentence embeddings (OpenAI text-embedding-3-small)
- **Reported results**: heatmaps, inter-model similarity matrices (Figure 4/5/6; Appendix C)

### 3.2 Task B: paraphrase robustness test (Appendix C.4)
- **Input**: 30 prompts + 4 paraphrases each → 150 prompts
- **Output**: 20 generations per prompt variant, across 42 models
- **Metrics**: within-prompt vs cross-paraphrase similarity
- **Result**: within 0.821 vs cross 0.781 (still high)
- **Concrete examples**: Table 17 (time metaphor → “river” repeated), Table 18 (internet impact → repeated phrasing)

### 3.3 Task C: human annotation tasks (absolute ratings / pairwise preferences)
- **Absolute rating**:
  - Input: (Query, Response)
  - Output: 1–5 quality score
  - Scale: 50 queries × 15 diverse responses × 25 annotators = 18,750 labels
- **Pairwise preference**:
  - Input: (Query, R1, R2)
  - Output: preference label distribution
  - Scale: 50 queries × 10 pairs × 25 annotators = 12,500 labels
- Disagreement visualized via Shannon entropy (Figure 7, Figure 9)

### 3.4 Task D: model-based evaluators vs human ratings (calibration)
- Inputs: same (Query, Response) / (Query, R1, R2)
- Outputs: LM perplexity scores, reward model scores, LM-judge rubric scores
- Evaluation: correlations with averaged human ratings; drops on “similar-quality” and “high-disagreement” subsets (Figure 10, 11)

---





<br/>
# 요약


INFINITY-CHAT(실사용 오픈엔디드 질의 26K)로 70+ LLM에 대해 프롬프트당 50개 샘플을 생성하고 문장 임베딩 코사인 유사도로 **intra-model repetition**과 **inter-model homogeneity(Artificial Hivemind)**를 측정했으며, 50개 응답을 군집화해 뽑은 다양한 응답들에 대해 **25명×31,250개**의 인간 평점/선호 라벨로 평가 캘리브레이션도 분석했다.  
그 결과 top-p=0.9, T=1.0에서도 동일 모델 내 응답 유사도가 매우 높아(예: 평균 pairwise similarity가 **0.8 이상인 경우가 79%**) 반복이 심했고, 모델이 달라도 평균 유사도가 **71–82%**로 높아 서로 다른 모델이 놀랄 만큼 비슷한 답을 내는 현상이 확인되었다.  
예시로 “**Write a metaphor about time**”에서 25개 모델의 응답이 PCA 클러스터링 시 대부분 “**time is a river**”와 일부 “**time is a weaver**” 두 축으로 수렴했고, iPhone 케이스 설명이나 모토 생성에서는 모델 간 **문구가 상당 부분(때로는 동일 문장) 겹치는** 사례가 관찰되었다(또한 인간 선호가 갈리는/동급 품질 응답일수록 LMs·RM·LM-judge 점수가 인간과의 상관이 더 떨어짐).  




Using INFINITY-CHAT (26K real-world open-ended queries), the authors sample 50 responses per prompt from 70+ LMs and quantify **intra-model repetition** and **inter-model homogeneity (Artificial Hivemind)** via cosine similarity of sentence embeddings; they also collect **31,250 dense human ratings/preferences (25 annotators each)** to study evaluator calibration.  
Even with top-p=0.9 and T=1.0, outputs are highly repetitive within a model (e.g., **79%** of prompts have average pairwise similarity **>0.8**) and strikingly similar across different models (average cross-model similarity **71–82%**).  
For example, “**Write a metaphor about time**” collapses into two main clusters—“**time is a river**” vs. “**time is a weaver**”—and some prompts (e.g., iPhone case blurbs or mottos) show substantial, sometimes verbatim, overlap across models, while model/reward/judge scores become less aligned with humans on similarly good or high-disagreement cases.

<br/>
# 기타


---

## 1) Figure 1 (시간 은유 예시 + PCA 클러스터)
### 결과
- “Write a metaphor about time”에 대해 25개 모델이 각 50개 응답을 생성했는데,
- 문장 임베딩을 PCA로 2D 축소해 군집화하니 **거의 두 덩어리**로만 모임:
  - 큰 군집: **“time is a river”**
  - 작은 군집: **“time is a weaver”**

### 인사이트
- 모델/크기/회사 다양해도, **개방형 창작 과제에서 아이디어 공간이 매우 좁은 모드로 붕괴(mode collapse)**.
- 표면적 표현 다양성(문장 조금씩 다름)보다 **개념 수준에서 수렴**이 강함 → “Artificial Hivemind”의 직관적 시각화.

---

## 2) Figure 2 (INFINITY-CHAT 오픈엔디드 쿼리 taxonomy: 6 대분류, 17 소분류 + 비율)
### 결과
- 실제 사용자 질의에서 오픈엔디드 프롬프트를 6개 상위 범주/17개 하위 범주로 체계화하고, 각 비중 제시.
- **Creative Content Generation이 58%로 최다**.
- 그 외에도 “Alternative Writing Genres(38.5%)”, “Concept Explanation(23.6%)”, “Skill Development(23.5%)”, “Analytical & Interpretive(22.6%)”, “Hypothetical(22.2%)”, “Brainstorm & Ideation(15.2%)” 등이 큼.

### 인사이트
- “현실 사용자들이 실제로 많이 던지는 오픈엔디드 질의”는 단순 창작뿐 아니라 설명/스킬/분석/가정 시나리오 등 광범위.
- 특히 **Brainstorm & Ideation(15.2%)** 비중이 의미있게 커서, LM이 아이디어를 “공급”하는 창구가 되고 → **장기적으로 사고/아이디어의 동질화 위험** 문제 제기 근거가 됨.

---

## 3) Figure 3 (새로운 카테고리 키워드 워드클라우드)
### 결과
- 기존 taxonomy 밖에서 **314개의 novel category**를 발견.
- 워드클라우드 키워드 예: “Cultural, Analysis, Ethical, Historical, Media, Humor …”

### 인사이트
- 오픈엔디드 쿼리의 “실제 다양성”이 크며, 고정된 소수의 분류로는 포착이 불완전.
- 동시에, 데이터셋이 창작/일상 질문 외에 **윤리·문화·미디어 등** 확장된 실제 사용자 니즈를 포함함을 보여줌.

---

## 4) Figure 4 (intra-model repetition: 같은 모델이 50번 샘플링해도 비슷)
### 결과
- 각 모델별로 100개 프롬프트에서 50개 응답 생성 후, 응답들 평균 pairwise 임베딩 유사도 분포를 히트맵으로 제시.
- top-p=0.9, temp=1.0인데도 **대부분 프롬프트에서 평균 유사도 0.8 이상**.
- 텍스트에 따르면 “79%에서 평균 유사도 > 0.8”.

### 인사이트
- “샘플링을 세게 흔들면 다양해진다”는 기대와 달리, **단일 모델 내부에서도 반복/수렴이 구조적으로 강함**.
- 사용자 입장에서 “다시 생성”을 눌러도 결과가 근본적으로 크게 달라지지 않는 현상을 대규모로 정량화.

---

## 5) Figure 5 (min-p 디코딩도 repetition 완화는 제한적)
### 결과
- min-p(top-p=1.0, min-p=0.1, temp=2.0)로 더 다양성 지향 디코딩을 해도,
- 여전히 **유사도 0.7~0.8 이상이 매우 많음**(본문: 81%가 0.7 초과, 61.2%가 0.8 초과).

### 인사이트
- 디코딩 트릭만으로는 모드 붕괴를 “근본 해결”하기 어렵고,
- 결국 **훈련/정렬 단계에서 다양성을 보존하는 접근이 필요**하다는 논리적 근거.

---

## 6) Figure 6 (inter-model homogeneity: 모델 간에도 비슷 + 구체적 중복 사례)
### 결과
- 모델 A와 B의 응답 유사도(평균)가 **대략 71%~82% 수준**으로 높게 나옴.
- 질적 예시에서 서로 다른 모델이 같은 프롬프트에 대해 **문구 단위로 겹치는 스팬**을 생성(예: “Elevate your iPhone with our … bold, eye-catching …”).
- 어떤 경우는 **완전 동일 문장**까지 등장(예: “Empower Your Journey: Unlock Success, Build Wealth, Transform Yourself.”)

### 인사이트
- 단일 모델 문제가 아니라, **서로 다른 회사/계열/크기 모델도 같은 아이디어와 표현으로 수렴** → “Artificial Hivemind”의 핵심 주장(특히 inter-model).
- “모델 스웜/앙상블이면 다양성 좋아질 것”이라는 가정에 경고: 구성원이 비슷한 priors(데이터/정렬) 공유하면 **앙상블도 동질적**.

---

## 7) Figure 7 (pairwise preference: 25명 라벨 분포 엔트로피)
### 결과
- (Query, Response1, Response2)에 대해 25명이 선호(강/약/동점 등)를 라벨링.
- Shannon entropy 분포가 **높은 쪽으로 치우침**(사람들이 자주 의견 불일치).
- 예시로, 어떤 항목은 라벨이 거의 균등 분포(강한 불일치), 어떤 항목은 한쪽 압도(합의).

### 인사이트
- 오픈엔디드 과제는 “정답 1개”가 아니라 **동급의 좋은 답이 여럿**이어서 사람 취향이 갈림.
- 따라서 평가/정렬에서 “단일 스칼라 보상/단일 합의 품질” 가정이 취약해짐(뒤 섹션의 miscalibration 논거).

---

## 8) Figure 8 (Top-N 유사 응답 클러스터에 섞여 있는 ‘모델 수’)
### 결과
- 25개 모델이 각 50개씩 내놓은 응답 중에서, “가장 비슷한 Top-N 응답”을 모아보면
- 그 Top-N 안에 포함된 **서로 다른 출처 모델 수가 평균 ~8개**(N=50 기준 등)까지 나옴.

### 인사이트
- “가장 비슷한 답들의 무리”가 한 모델에서만 나오는 게 아니라 **여러 모델에서 섞여 나옴**.
- 경우에 따라서는 intra-model보다 inter-model이 더 비슷해질 수도 있다는 문제 제기 → **모델 간 구별 불가능성** 증가.

---

## 9) Figure 9 (absolute rating: 25명 평점 분포 엔트로피)
### 결과
- (Query, Response)에 대해 1~5점 절대평가를 25명이 수행.
- 이 또한 엔트로피가 다양하게 분포 → 일부는 합의, 일부는 큰 불일치.

### 인사이트
- 오픈엔디드 응답 품질은 “객관적 단일 기준”으로 수렴하지 않고,
- 사람마다 “괜찮다/별로다”가 갈릴 수 있어 **평가 신호 자체가 분포(distributional)**임을 강조.

---

## 10) Figure 10 (absolute rating에서: 전체 vs ‘비슷한 품질’/‘불일치 높은’ subset의 상관 하락)
### 결과
- 사람 평균점수와 (LM perplexity / reward model / LM judge) 점수 간 Spearman 상관을 비교.
- 전체 데이터보다,
  - (a) **비슷한 품질끼리 모인 subset**
  - (b) **사람 불일치가 큰 subset**
  에서 상관이 **눈에 띄게 하락**.

### 인사이트
- 모델 평가자(리워드/저지/퍼플렉시티)가 “명확히 좋은 vs 나쁜” 구분은 어느 정도 하지만,
- **미묘한 동급 비교**나 **취향 갈림**이 있는 지점에서 사람 판단을 제대로 따라가지 못함 → 정렬 과정이 한 모드만 강화할 위험.

---

## 11) Figure 11 (pairwise preference에서도 동일 패턴: subset에서 상관 하락)
### 결과
- pairwise preference에서도 전체 대비
  - 유사 품질 subset,
  - 불일치 높은 subset
  에서 모델-인간 상관이 하락.

### 인사이트
- “동급인 좋은 답안들이 존재”하는 오픈엔디드 환경에서,
- 모델 기반 평가(리워드/저지)가 **다양한 정답 가능성**을 안정적으로 인정하지 못함을 재확인.

---

# 12) Table 1 (인간이 본 ‘오픈엔디드 정도’: 대안 답변 개수)
### 결과
- “대안 답이 몇 개나 가능?” 질문에서:
  - 3개 미만 18.73%
  - 3~10개 31.87%
  - 10~20개 14.74%
  - 20개 초과 34.66%
- 즉 **81.27%가 ‘3개 이상’**, **34.66%가 ‘20개 초과’**.

### 인사이트
- 실제 오픈엔디드 쿼리는 생각보다 폭이 넓고,
- 그런데도 모델 출력은 앞선 Figure들처럼 좁은 군집으로 모인다는 점이 “모드 붕괴” 주장에 힘을 실음.

---

# 13) Table 3 vs Table 4 (오픈엔디드/클로즈드엔디드 질의 예시)
### 결과
- Table 3는 다양한 답이 가능한 오픈엔디드 질의 예시,
- Table 4는 예/아니오, 정확한 답, 잘 범위가 정해진 글쓰기 등 **다양성 중요도가 낮은** 질의 예시.

### 인사이트
- “다양성 평가”는 아무 질의에나 적용하는 게 아니라,
- **정답이 여러 개 가능한 현실 질의**에 맞춘 벤치마크가 필요하다는 데이터셋 설계의 정당화.

---

# 14) Appendix B (데이터 마이닝/분류 프롬프트: Figure 12~14)
### 결과
- WildChat에서 영어/무독성/길이 제한 등으로 37,426 후보를 뽑고,
- GPT-4o로 “의미성/인사·모델문의 여부/단일 vs 다중 응답 가능”을 분류, 애매하면 수정.
- 최종: **오픈엔디드 26,070 + 클로즈드엔디드 8,817**
- taxonomy 라벨링도 GPT-4o로 확장(휴먼과 74.7% 일치 언급).

### 인사이트
- “실사용 로그 기반의 현실 오픈엔디드 프롬프트”라는 주장에 대한 방법적 근거.
- 동시에 분류에 LLM을 사용 → 라벨 노이즈/편향 가능성을 한계로 인정(Appendix A).

---

# 15) Appendix C (실험 설정/모델 리스트/유사도 계산)
### 결과(핵심 설정)
- INFINITY-CHAT100(100개 대표 오픈엔디드 쿼리) 사용.
- 모델별 50샘플 생성.
- top-p(0.9), temp(1.0) / min-p(1.0, min-p=0.1, temp=2.0)
- 유사도는 OpenAI text-embedding-3-small 임베딩 코사인.

### 인사이트
- intra/inter-model 분석이 “같은 프로토콜”로 수행되어 비교 가능.
- 또한 paraphrase 실험(C.4): 원문 vs 패러프레이즈도 높은 유사도(0.821 vs 0.781) → 프롬프트 살짝 바꿔도 **같은 모드로 회귀**.

---

# 16) Appendix D (인간 31,250개 dense annotation + 모델 평가 비교)
### 결과
- 절대평가: 50프롬프트×15응답×25명=18,750
- 쌍대비교: 50프롬프트×10쌍×25명=12,500
- 총 31,250 라벨.
- “다양한 응답”을 뽑기 위해 클러스터링 후 15개 군집에서 샘플링.

### 인사이트
- 기존 alignment 데이터가 annotator 수가 적어(예: 3명) 분포를 못 잡는 한계를 보완.
- “사람 선호 분포가 넓은 오픈엔디드 영역”에서 reward/judge가 흔들리는지(상관 하락)를 분석 가능하게 만든 핵심 기여.

---

---




## 1) Figure 1 (Time metaphor PCA clusters)
**Result:** Across 25 models × 50 samples, responses cluster into two dominant modes: “time is a river” and “time is a weaver.”  
**Insight:** Even for highly open-ended creative prompts, models converge at the *concept level*, evidencing deep mode collapse (“Artificial Hivemind”).

## 2) Figure 2 (INFINITY-CHAT taxonomy: 6 top-level / 17 subcategories + prevalence)
**Result:** Real-world open-ended queries are categorized and quantified; Creative Content Generation is 58%, with large shares in alternative genres, concept explanations, skill development, analysis, hypotheticals, and brainstorming.  
**Insight:** Real user open-endedness is broad (not just poetry). Brainstorming’s sizable share motivates concerns about long-term homogenization of ideas.

## 3) Figure 3 (Word cloud of 314 novel categories)
**Result:** 314 additional, newly discovered query types; prominent keywords include cultural/ethical/historical/media/humor.  
**Insight:** The open-ended query space is richer than fixed taxonomies; the dataset captures underexplored real-world dimensions.

## 4) Figure 4 (Intra-model repetition heatmap)
**Result:** With top-p=0.9 and temperature=1.0, average within-prompt response similarity often exceeds 0.8 (reported ~79% of prompts).  
**Insight:** Sampling multiple times from the same model still yields highly repetitive outputs—diversity is structurally limited.

## 5) Figure 5 (Min-p decoding still shows high similarity)
**Result:** Min-p reduces extreme repetition but most response pairs remain very similar (e.g., 61.2% > 0.8 similarity).  
**Insight:** Decoding alone cannot reliably fix diversity collapse; training-level interventions are needed.

## 6) Figure 6 (Inter-model homogeneity + overlapping spans)
**Result:** Mean cross-model similarity is high (~71–82%), including overlapping phrases and occasional identical outputs.  
**Insight:** Homogeneity is *cross-model*, challenging the assumption that ensembles/swarms automatically provide diversity.

## 7) Figure 7 (Pairwise preference entropy across 25 annotators)
**Result:** Many triplets show high entropy (disagreement) in human preferences.  
**Insight:** Open-ended evaluation is inherently pluralistic; “one best answer” assumptions are weak.

## 8) Figure 8 (Unique source models in top-N most similar responses)
**Result:** The most similar response clusters often include outputs from many different models (avg ~8 unique models in top-50).  
**Insight:** Outputs become indistinguishable across models; inter-model similarity can rival intra-model similarity.

## 9) Figure 9 (Absolute rating entropy across 25 annotators)
**Result:** Rating distributions vary widely; some items have strong agreement, others don’t.  
**Insight:** Human quality judgments form distributions, not single labels, for open-ended tasks.

## 10) Figure 10 (Absolute rating: correlations drop on similar-quality / high-disagreement subsets)
**Result:** Model-human correlations (LM perplexity, reward models, LM judges) drop notably on subtle or disputed cases.  
**Insight:** Automated evaluators are poorly calibrated precisely where multiple high-quality alternatives exist.

## 11) Figure 11 (Pairwise preference: same correlation drop pattern)
**Result:** Correlations decline on similar-quality and high-disagreement subsets in pairwise setups too.  
**Insight:** The miscalibration generalizes across absolute and pairwise evaluation regimes.

## 12) Table 1 (Human-perceived degree of open-endedness)
**Result:** 81.27% of queries allow 3+ reasonable alternatives; 34.66% allow >20.  
**Insight:** Real-world prompts have large answer spaces—yet models still collapse into narrow modes.

## 13) Tables 3–4 (Open-ended vs closed-ended examples)
**Result:** Concrete examples distinguish diversity-critical prompts from deterministic ones.  
**Insight:** Justifies why a dedicated open-ended benchmark is needed beyond correctness-focused datasets.

## 14) Appendix B / Figures 12–14 (Mining + classification prompts)
**Result:** Pipeline filters WildChat, uses GPT-4o to classify meaningfulness/greetings/single-vs-multiple, revises ambiguous queries; yields 26,070 open-ended queries.  
**Insight:** Methodological grounding for “in-the-wild, real user open-ended queries,” while acknowledging labeling noise/limitations.

## 15) Appendix C (Unified experimental setup + paraphrase test)
**Result:** Standardized decoding and similarity computation; paraphrases still produce high similarity (within 0.821 vs cross-paraphrase 0.781).  
**Insight:** Models return to the same modes even under prompt paraphrasing.

## 16) Appendix D (Dense human annotations + calibration study)
**Result:** 31,250 labels with 25 annotators per item; responses selected via clustering to encourage diversity.  
**Insight:** Enables studying distributional preferences and reveals calibration failures of reward models/judges on pluralistic cases.

---




<br/>
# refer format:


## 1) BibTeX (bib)

```bibtex
@inproceedings{jiang2025artificialhivemind,
  title        = {Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)},
  author       = {Jiang, Liwei and Chai, Yuanjun and Li, Margaret and Liu, Mickel and Fok, Raymond and Dziri, Nouha and Tsvetkov, Yulia and Sap, Maarten and Choi, Yejin},
  booktitle    = {Advances in Neural Information Processing Systems (NeurIPS)},
  year         = {2025},
  note         = {Track on Datasets and Benchmarks},
  url          = {https://github.com/liweijiang/artificial-hivemind}
}
```


## 2) 시카고 스타일(줄글, Bibliography 형식)

Jiang, Liwei, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap, and Yejin Choi. 2025. “Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond).” In *Advances in Neural Information Processing Systems (NeurIPS)*, Track on Datasets and Benchmarks. https://github.com/liweijiang/artificial-hivemind.

---
