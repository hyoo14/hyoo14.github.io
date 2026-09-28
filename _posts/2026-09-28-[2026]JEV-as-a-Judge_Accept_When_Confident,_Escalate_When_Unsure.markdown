---
layout: post
title:  "[2026]JEV-as-a-Judge: Accept When Confident, Escalate When Unsure"
date:   2026-09-28 20:44:38 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 그 Jev.. Judge로써도 리즈닝 깊은거 아니면 괜찮지만 깊은 리즈닝 약했고.. 
이 경우 컨피던스 낮았다  
그래서 그 경우는 gpt6써서 좀 더 저렴하게 잘 하는 방법이 될 수 있겠다..  


짧은 요약(Abstract) :


이 논문은 **JEV-as-a-Judge**라는 결정 전용(decision-only) 평가 모델이 답변을 빠르고 저렴하게 1차 심사할 수 있는지 분석합니다. JEV는 긴 설명이나 추론 과정을 생성하지 않고, 정해진 선택지 중 하나의 **판정 결과와 각 라벨의 확률**만 출력합니다.

실험 결과, 일반적인 답변 선호도 평가와 근거 기반 사실성 평가에서 JEV의 정확도는 가장 강력한 비교 모델인 GPT-6와 **약 3%p 이내**로 비슷했습니다. 반면 비용은 GPT-6의 **약 0.36%** 수준이었습니다. 즉, 대부분의 일상적인 평가에서는 훨씬 저렴한 JEV를 사용할 수 있다는 의미입니다.

하지만 여러 단계의 풀이를 검증해야 하거나, 틀린 답변이 매우 그럴듯하고 정교하게 작성된 경우에는 JEV의 성능이 크게 떨어졌습니다. 특히 JEV가 낮은 확률을 부여한, 즉 **확신하지 못한 사례**에 오류가 집중되는 경향이 있었습니다.

이를 바탕으로 논문은 다음과 같은 **캐스케이드 방식**을 제안합니다.

1. JEV가 확신하는 답변은 그대로 수용한다.
2. JEV의 확신도가 낮은 답변만 더 강력하고 비싼 LLM에 재평가를 맡긴다.

이 방식을 사용하면 전체 사례를 강력한 모델로 평가하는 것보다 비용을 줄이면서도, GPT-6 정확도의 약 **99%를 유지**할 수 있었습니다. 따라서 JEV는 모든 문제를 단독으로 해결하는 심판이라기보다는, **쉬운 사례를 저렴하게 처리하고 어려운 사례만 상위 모델로 넘기는 1차 평가기**로 적합하다는 것이 핵심 결론입니다.

---




This paper studies whether **JEV-as-a-Judge**, a decision-only evaluator, can serve as a cheap and efficient first-pass judge. Instead of generating a rationale, JEV returns a verdict and probabilities over the possible labels.

On ordinary preference judgments and evidence-grounded factuality tasks, JEV performs within about **three percentage points** of the strongest comparator, GPT-6, while costing only about **0.36% of its fee**. This suggests that JEV is highly economical for routine evaluation.

However, JEV performs substantially worse when a task requires checking a multi-step derivation or resisting a convincingly written but incorrect answer. Its errors are concentrated in cases where its confidence is low.

The paper therefore proposes a **confidence-based cascade**:

1. Accept JEV’s verdict when it is confident.
2. Escalate uncertain cases to a stronger LLM.

This strategy preserves about **99% of GPT-6’s accuracy** at a much lower cost. The main conclusion is that JEV is best used not as a universal final authority, but as an inexpensive first-stage evaluator that routes difficult cases to a stronger judge.


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



### 1. 핵심 아이디어
이 논문은 **JEV-as-a-Judge**라는 *결정 전용(decision-only) 평가 모델*을 저비용 1차 심사자로 사용하는 방법을 연구한다.

- JEV는 설명이나 추론 과정을 생성하지 않고, 미리 지정된 후보 라벨 중 하나를 선택한다.
- 동시에 각 라벨에 대한 확률을 출력한다.
- 가장 높은 라벨 확률을 **신뢰도(confidence)** 로 사용한다.
- 신뢰도가 높으면 JEV의 판정을 그대로 채택하고, 낮으면 더 강력하지만 비싼 LLM으로 넘긴다.
- 즉, 전체 구조는 다음과 같다.

> **JEV 1차 판정 → 확신이 높으면 수용 / 불확실하면 강한 모델로 이관**

JEV의 구현은 TypeSafe AI가 제공하는 **호스팅된 독점 서비스**이며, 논문에서는 내부 아키텍처나 학습 방법을 새로 제안하거나 공개하지 않는다.

---

### 2. JEV의 입출력 방식
JEV는 구조화된 입력과 자연어 평가 지시문을 받으며, 출력 라벨의 종류가 사전에 정해진다.

주요 출력 형식은 다음과 같다.

- **Choice**: 여러 라벨 중 하나를 선택하고 각 라벨의 확률 출력
- **Noul**: 이진 판단에 대한 yes 확률 출력
- **Score**: 순서가 있는 점수 수준별 확률 출력

실험에서는 주로 한 번의 **Choice 질의**를 사용했다. 출력에는 다음이 포함된다.

```json
{
  "verdict": "supported",
  "probabilities": {
    "supported": 1.0,
    "contradicted": 0.0,
    "unknown": 0.0
  }
}
```

확률의 최댓값을 \(q = \max_k p_k\)로 정의하고, 이를 JEV의 confidence로 사용했다.

---

### 3. 평가 과제
서로 다른 유형의 판단 능력을 비교하기 위해 다음 과제를 사용했다.

1. **선호 비교(Preference)**
   - 두 답변 중 더 나은 답변을 선택
   - 정확성, 추론, 지시 준수, 관련성, 안전성 등을 고려

2. **근거 기반 사실성(Evidence-grounded factuality)**
   - 제공된 근거 문서와 답변을 비교
   - 답변이 근거에 의해 지지되는지, 모순되는지 판단

3. **최종 답변 판정(Final-answer adjudication)**
   - 모델 답변에서 최종적으로 확정한 답을 추출하고 정답과 비교
   - 중간에 틀린 설명이 있더라도 명확한 최종 답변을 기준으로 판단

4. **어려운 정답성 평가**
   - 수학, 추론, 코딩 등에서 답변의 실제 정확성을 확인
   - JEV가 다단계 유도나 계산을 제대로 검증할 수 있는지 평가

5. **스타일에 영향을 받는 선호 비교**
   - 내용상 틀린 답변이 더 길고 정교하게 작성된 경우에도 올바른 답변을 고르는지 평가

6. **자연어 사실성 평가**
   - 근거나 참고자료 없이 일반적인 산문 답변의 환각 여부를 판단

---

### 4. 비교 모델
JEV를 여러 생성형 LLM 및 reward model과 비교했다.

- GPT-4.1 mini, GPT-4.1
- GPT-5.2, GPT-5.4, GPT-5.6 Sol
- GPT-6 Astra
- GPT-OSS 120B
- Claude Sonnet 5
- Gemini 3 Flash, Gemini 3.1 Pro
- Qwen3 계열 모델
- PairRM
- Skywork-Reward-V2-Qwen3-8B

생성형 모델들도 JEV와 동일하게 **판정 라벨과 라벨별 확률만 출력**하도록 지시받았다. 따라서 생성형 모델의 장황한 설명이나 chain-of-thought 자체가 비교에 직접 사용되지는 않았다.

특히 일부 reasoning 모델은 낮은 수준의 reasoning effort를 사용했으며, 모델별 API 비용과 지연시간은 실제 설정에서 측정했다.

---

### 5. 학습 및 데이터 구성
이 연구는 새로운 judge 모델을 직접 학습하거나 fine-tuning하지 않았다.

- JEV는 이미 제공되는 상용/호스팅 서비스로 사용
- 비교 모델도 사전 학습된 모델 또는 reward model로 사용
- 데이터는 RewardBench, JudgeBench, HaluEval 및 기존 답변 데이터 사용
- 전체 642개 파일럿 데이터와 670개 확장 데이터를 구성
- 일부 파일럿 데이터는 다음에 사용:
  - temperature scaling
  - confidence 기반 이관 threshold 설정
- 확장 데이터는 threshold 학습에 사용하지 않은 별도 평가 세트로 사용

즉, 이 논문의 핵심 기여는 새로운 신경망 구조나 학습 알고리즘이 아니라, **기존 결정형 judge를 어떤 작업에 사용하고 confidence를 어떻게 운영 정책에 연결할 것인지에 대한 실험적 방법론**이다.

---

### 6. 신뢰도 기반 Cascade 방법
JEV의 판정 확률을 이용해 선택적 평가(selective evaluation)를 구현했다.

- \(q \geq \tau\): JEV의 판정을 채택
- \(q < \tau\): 더 강한 fallback judge로 이관

여기서 \(\tau\)는 confidence threshold다.

예를 들어 \(\tau=0.9\)이면:

- JEV가 90% 이상의 확률로 특정 답을 고른 경우: 그대로 수용
- 그렇지 않은 경우: GPT-6 같은 강한 모델에 재평가 요청

쌍대 비교에서는 후보 순서 편향을 줄이기 위해 두 답변을 순서를 바꾸어 각각 평가한 뒤, 의미적으로 같은 후보의 확률을 정렬해 평균했다.

\[
\bar p(A)=\frac{1}{2}\left[p(A\mid A,B)+1-p(A\mid B,A)\right]
\]

Threshold는 파일럿 selection set에서 정하고, 별도의 test 및 extension set에서 평가했다.

---

### 7. 평가 지표와 검증
모델의 품질과 운영 비용을 함께 측정했다.

- Accuracy
- Macro-F1
- Brier score
- Negative log-likelihood
- Expected calibration error
- 오류 탐지 AUROC
- 지연시간
- 1,000회 판단당 API 비용
- 출력 형식 유효성

또한 다음 안정성 실험도 수행했다.

- 같은 요청 반복 시 판정 변화
- 평가 rubric을 바꿔 썼을 때의 변화
- 후보 답변 순서 변경에 따른 변화

JEV와 GPT-6의 판정이 서로 다를 때는 일부 항목을 사람이 블라인드로 재판정하여, benchmark label 오류와 실제 judge 오류를 구분하려 했다. 다만 전체 데이터를 무작위로 사람 평가한 것이 아니라, 주로 두 judge가 불일치한 항목을 대상으로 한 보조 검증이다.

---

### 8. 특별한 아키텍처나 기법의 요약
이 논문에서 중요한 방법적 특징은 다음과 같다.

- 새로운 모델 아키텍처 제안은 없음
- JEV의 내부 구조는 독점적이며 공개되지 않음
- 설명을 생성하지 않는 typed decision interface 사용
- 라벨 확률을 uncertainty signal로 사용
- confidence threshold 기반의 cascade/routing 적용
- 선호 비교 시 양방향 입력과 확률 정렬 사용
- 작업별로 threshold와 calibration을 별도 검증
- 어려운 문제를 무조건 JEV로 처리하지 않고 강한 judge로 이관
- 비용, 정확도, calibration, latency를 함께 평가

---




### 1. Core idea
The paper studies **JEV-as-a-Judge**, a low-cost **decision-only evaluator** used as a first-stage judge.

JEV does not generate a rationale or explanation. Instead, it:

- selects one label from a predefined set,
- returns probabilities for all allowed labels,
- uses the maximum label probability as a confidence signal.

The proposed operational strategy is:

> **Run JEV first; accept confident decisions and escalate uncertain cases to a stronger LLM.**

JEV is a proprietary hosted service from TypeSafe AI. The paper does not introduce or disclose a new internal architecture or training procedure for JEV.

---

### 2. Input and output interface
JEV receives structured state, natural-language instructions, and a specified output type.

The main interfaces are:

- **Choice**: selects one label and returns probabilities over labels
- **Noul**: returns a yes-probability for binary judgments
- **Score**: returns probabilities over ordered score levels

Most experiments use one `Choice` request per item. The confidence score is defined as:

\[
q=\max_k p_k
\]

where \(p_k\) is the predicted probability of label \(k\).

---

### 3. Evaluation tasks
The study evaluates several types of judging work:

1. **Pairwise preference**
   - Select the better of two candidate responses.
   - Criteria include correctness, reasoning, relevance, instruction following, and safety.

2. **Evidence-grounded factuality**
   - Determine whether an answer is supported by, contradicts, or adds unsupported claims beyond supplied evidence.

3. **Final-answer adjudication**
   - Judge the final committed answer against a trusted reference.
   - Intermediate mistakes are ignored if a clear final answer supersedes them.

4. **Difficult correctness**
   - Evaluate mathematics, reasoning, coding, and other tasks requiring derivation checking.

5. **Style-adversarial preference**
   - Test whether a judge rejects a more elaborate but incorrect answer.

6. **Natural-language factuality**
   - Evaluate hallucination in prose with or without an external reference.

---

### 4. Comparator models
JEV is compared with hosted LLM judges and reward models, including:

- GPT-4.1 mini, GPT-4.1
- GPT-5.2, GPT-5.4, GPT-5.6 Sol
- GPT-6 Astra
- GPT-OSS 120B
- Claude Sonnet 5
- Gemini 3 Flash and Gemini 3.1 Pro
- Several Qwen3 models
- PairRM
- Skywork-Reward-V2-Qwen3-8B

Generative judges receive the same task instructions and are asked to return only the decision and label probabilities, without a rationale. Therefore, the comparison focuses on the complete judging configuration rather than on generated explanations.

---

### 5. Training and data
The study does **not** train or fine-tune a new judge.

- JEV is used as an existing hosted service.
- Comparator models are pretrained LLMs or reward models.
- The experiments use RewardBench, JudgeBench, HaluEval, and existing response datasets.
- A pilot set is used to fit:
  - temperature scaling parameters,
  - confidence thresholds for routing.
- Separate test and extension sets are used for evaluation.

Thus, the main contribution is not a new neural architecture or training algorithm. It is an empirical method for determining **where a decision-only judge is sufficient and how its confidence can support cost-aware escalation**.

---

### 6. Confidence-based cascade
The cascade uses JEV’s maximum label probability:

- If \(q \geq \tau\), accept JEV’s verdict.
- If \(q < \tau\), send the item to a stronger fallback judge.

Here, \(\tau\) is a confidence threshold. For example, with \(\tau=0.9\), only decisions with at least 0.9 maximum label probability are accepted automatically.

For pairwise comparisons, both candidate orders are evaluated. The probabilities are aligned to the same semantic candidate and averaged:

\[
\bar p(A)=\frac{1}{2}
\left[p(A\mid A,B)+1-p(A\mid B,A)\right].
\]

Thresholds are selected on a pilot selection set and evaluated on held-out test or extension data.

---

### 7. Metrics and validation
The study measures:

- Accuracy
- Macro-F1
- Brier score
- Negative log-likelihood
- Expected calibration error
- Error-detection AUROC
- Output validity
- Latency
- API cost per 1,000 judgments

It also tests:

- repeated-request stability,
- sensitivity to rubric paraphrasing,
- sensitivity to candidate order reversal.

When JEV and GPT-6 disagree, selected items are reviewed by blinded human adjudication. This is used as a label-noise and judge-quality analysis, although it is not a random audit of the entire benchmark.

---

### 8. Main methodological takeaway
The important methodological features are:

- no new architecture is proposed;
- JEV’s internal model is proprietary;
- evaluation uses a typed decision-only interface;
- label probabilities provide an uncertainty signal;
- confidence thresholds enable cascading to stronger judges;
- pairwise judgments use both candidate orders;
- calibration and routing thresholds are validated per workload;
- difficult or style-adversarial cases are escalated;
- accuracy, calibration, cost, and latency are evaluated together.


<br/>
# Results



### 1. 연구 목적과 비교 대상

이 논문은 **결정만 출력하는 JEV(JEV-as-a-Judge)**가 값비싼 생성형 LLM judge의 **저비용 1차 평가기**로 사용될 수 있는지 검증한다.  
JEV는 판단 결과와 함께 각 라벨의 확률을 제공하며, 이 확률의 최댓값을 confidence \(q\)로 사용한다.

비교 대상은 다음과 같다.

- **상용 생성형 judge**: GPT-4.1 mini, GPT-4.1, GPT-5.2, GPT-5.4, GPT-5.6 Sol, GPT-6 Astra, Claude Sonnet 5, Gemini 3 Flash, Gemini 3.1 Pro
- **호스팅·오픈 모델**: GPT-OSS 120B, Qwen3.6/3.8 27B
- **로컬 모델**: Qwen3-32B, Qwen3.5-27B
- **Reward model**: PairRM, Skywork-Reward-V2-Qwen3-8B

가장 강력한 주요 비교 모델은 **GPT-6 Astra**였다.

---

### 2. 테스트 데이터와 평가 과제

#### 주요 공개 벤치마크

| 데이터셋 | 규모 | 평가 내용 |
|---|---:|---|
| RewardBench | 400쌍 | 두 응답 중 더 나은 응답 선택 |
| JudgeBench | 350쌍 | 지식, 추론, 수학, 코딩 문제에서 정답성 판단 |
| HaluEval | 240개 | 제공된 증거에 근거해 답변이 사실인지 환각인지 판단 |
| Existing labels | 150개 | 객관식·수치 답변의 최종 답변이 정답인지 판정 |

추가로 다음 데이터도 평가했다.

- **RewardBench 2**: 4개 응답 중 최선의 답변 선택, 100개
- **RM-Bench**: 답변의 문체가 판단을 방해하는지 평가, 80개 source에서 1,440개 조건
  - 일반적인 문체 조합
  - 틀린 답변이 더 자세하고 화려하게 작성된 어려운 조합
- **문서 기반 요약**: 80개
- **참조 문서가 없는 일반 자연어 답변**: 80개
- 반복 요청, rubric paraphrase, 응답 순서 변경을 이용한 안정성 진단

모든 judge는 가능한 경우 동일한 rubric과 입력을 받았으며, 결과는 주로 **정확도(accuracy)**로 비교했다. 확률 품질은 **Brier score, NLL, ECE, error-detection AUROC**로 측정했다.

---

### 3. 주요 정확도 결과

#### 핵심 벤치마크

| 과제 | JEV | 비교 모델 | 차이 및 해석 |
|---|---:|---:|---|
| RewardBench 선호 판단 | **92.2%** | GPT-6 93.5% | **-1.3%p**. 사실상 근접 |
| HaluEval 증거 기반 사실성 | **87.5%** | GPT-6 86.7% | **+0.8%p** |
| 최종 답변 판정 | **94.0%** | GPT-6 96.7% | **-2.7%p** |
| JudgeBench 어려운 정답성 | **78.6%** | GPT-6 93.1% | **-14.6%p** |
| RM-Bench 일반 문체 | **84.0%** | GPT-6 93.3% | **-9.4%p** |
| RM-Bench: 틀린 답이 더 화려한 문체 | **74.8%** | GPT-6 94.6% | **-19.8%p** |
| RewardBench 2, 4-way 선택 | **73.0%** | GPT-6 75.0%, Skywork 79.0% | JEV가 다소 낮지만 차이는 불확실 |

#### 해석

JEV는 다음과 같은 비교적 일상적인 평가에서는 강력했다.

- 일반적인 응답 선호도 비교
- 제공된 증거를 이용한 사실성 판단
- 정답과 최종 답변을 직접 비교하는 판정

반면 다음 과제에서는 성능 격차가 컸다.

- 여러 단계의 수학·추론·코딩 풀이를 검증하는 과제
- 설명의 타당성과 최종 답을 동시에 판단해야 하는 과제
- 틀린 답이 더 정교하고 설득력 있게 작성된 경우

예를 들어 JudgeBench에서 JEV는 78.6%였지만 GPT-6은 93.1%였다. 인간 adjudication에서도 GPT-6이 대부분의 불일치 사례에서 더 올바른 판단을 내린 것으로 나타나, 단순한 벤치마크 라벨 오류만으로 설명되지는 않았다.

---

### 4. 비용과 지연시간

120개 판단으로 구성된 동일한 측정 패널에서 다음과 같은 결과를 얻었다.

| Judge | 중앙 지연시간 | 1,000건당 비용 |
|---|---:|---:|
| JEV | **0.152초** | **약 \$0.044** |
| GPT-4.1 mini | 0.548초 | \$0.390 |
| GPT-6 Astra | 1.885초 | \$12.182 |

즉, JEV는 이 측정에서 GPT-6보다 대략 다음과 같았다.

- **약 12배 빠른 중앙 지연시간**
- **약 277배 저렴한 비용**

논문 초록에서는 비용을 비교하는 기준과 집계 방식에 따라 JEV가 약 **0.36% 수준의 비용**이라고 요약한다. 다만 비용은 입력 토큰, 캐시 할인, reasoning token, 누락된 사용량에 대한 보수적 추정 등을 포함한 측정치이며 실제 청구서와는 다를 수 있다.

---

### 5. Confidence와 선택적 escalation

JEV는 예측 라벨뿐 아니라 라벨 확률도 제공한다. 논문은 가장 높은 라벨 확률을 \(q\)로 정의하고 다음 정책을 시험했다.

1. \(q\)가 높으면 JEV 결과를 그대로 수용
2. \(q\)가 낮으면 GPT-6 같은 강한 judge에 재평가 요청

JEV의 confidence는 오류를 어느 정도 잘 찾아냈다.

- RewardBench error-detection AUROC: **0.869**
- JudgeBench: **0.745**
- HaluEval: **0.863**

전체 990개 기본 판단에서:

- \(q < 0.6\)일 때 JEV 정확도: **47.7%**
- \(q = 1\)일 때 정확도: **99.1%**
- \(q \geq 0.9\)인 항목에서는 JEV와 GPT-6 성능이 거의 비슷
- 낮은 confidence 영역에서 GPT-6의 우위가 집중됨

#### JEV → GPT-6 cascade

\(q \geq 0.9\)이면 JEV를 수용하고, 그보다 낮으면 GPT-6으로 넘기는 방식의 결과는 다음과 같다.

- GPT-6 단독 정확도: **91.7%**
- Cascade 정확도: **91.3%**
- GPT-6 성능의 **99.6% 유지**
- GPT-6 단독 비용의 약 **47%**
- 약 **34%의 항목만 GPT-6으로 escalation**

즉, JEV를 모든 항목에 사용할 필요 없이, 불확실한 사례만 강한 모델에 보내면 비용을 크게 줄이면서 성능을 거의 유지할 수 있었다.

다만 이 threshold는 모든 과제에 그대로 적용되지 않았다. 특히 문체가 정답성을 속이는 RM-Bench의 어려운 조건에서는 JEV가 **높은 confidence로 틀리는 경우**가 있었고, 참조 문서가 없는 자연어 평가에서는 confidence가 거의 도움이 되지 않았다.

---

### 6. 확률 품질과 안정성

JEV의 확률은 과제에 따라 품질이 달랐다.

- RewardBench Brier score: **0.111**
- JudgeBench: **0.297**
- HaluEval: **0.176**

GPT-6은 JudgeBench에서는 JEV보다 정확하고 더 잘 calibration되었지만, HaluEval에서는 JEV의 확률 품질이 더 나았다. 따라서 **정확도와 confidence calibration은 서로 다른 능력**으로 나타났다.

안정성 측면에서는:

- 동일 요청 반복 시 96번 중 판단 변경 **0회**
- rubric paraphrase 시 48번 중 **4회 변경**
- 응답 순서 변경에 따른 불일치:
  - RewardBench **3.2%**
  - JudgeBench **11.1%**

JEV는 반복성은 높았지만, 어려운 판단에서는 응답 순서와 표현 방식에 어느 정도 민감했다.

---

### 7. 자연어 평가의 한계

참조 문서가 있는 요약 평가에서는 JEV가 **71.2%**, GPT-5.4가 **72.5%**로 비슷했다.  
그러나 참조나 증거가 없는 일반 자연어 답변에서는:

- JEV: **52.5%**
- GPT-4.1 mini: **53.8%**
- GPT-5.4: **55.0%**

모든 모델이 거의 우연 수준에 가까웠으며, JEV의 평균 confidence는 오히려 **0.90**으로 높았다. 이는 다음을 보여준다.

> confidence가 높다고 항상 판단이 옳은 것은 아니며, 특히 근거가 없는 자연어 사실성 평가에서는 모든 judge가 자신 있게 틀릴 수 있다.

---

### 8. 최종 결론

논문이 제안하는 JEV의 적절한 역할은 **강한 judge를 완전히 대체하는 것**이 아니라 다음과 같다.

- 일반적인 선호도·증거 기반 사실성·최종 답변 판정의 저비용 1차 필터
- confidence가 낮은 사례만 GPT-6 등 강한 모델에 위임하는 cascade의 첫 단계
- 다단계 추론 검증, 스타일에 속기 쉬운 비교, 근거 없는 자연어 사실성 평가에서는 단독 judge로 사용하지 않음

핵심 메시지는 다음과 같다.

> **JEV는 쉬운 판단은 싸고 빠르게 처리하고, 자신 없는 판단은 강한 모델로 넘길 때 가장 유용하다. 하지만 confidence는 정답 보증서가 아니므로 과제별 검증과 threshold 조정이 필요하다.**

---




### 1. Purpose and compared judges

The paper evaluates whether **JEV, a decision-only judge**, can serve as a low-cost first-stage evaluator. JEV returns a verdict and probabilities over the allowed labels. The maximum label probability is used as its confidence score \(q\).

The comparison includes:

- Hosted LLM judges: GPT-4.1 mini, GPT-4.1, GPT-5.2, GPT-5.4, GPT-5.6 Sol, GPT-6 Astra, Claude Sonnet 5, Gemini 3 Flash, and Gemini 3.1 Pro
- Hosted/open models: GPT-OSS 120B and Qwen3.6/3.8 27B
- Local models: Qwen3-32B and Qwen3.5-27B
- Reward models: PairRM and Skywork-Reward-V2-Qwen3-8B

The strongest main comparator was **GPT-6 Astra**.

---

### 2. Test data and tasks

The main benchmarks were:

- **RewardBench**: 400 preference pairs
- **JudgeBench**: 350 pairs covering knowledge, reasoning, mathematics, and coding
- **HaluEval**: 240 evidence-grounded factuality judgments
- **Existing labels**: 150 final-answer adjudication examples

Additional follow-up evaluations included:

- RewardBench 2: four-way selection over 100 prompts
- RM-Bench: 1,440 style-sensitive pairwise judgments
- 80 document-grounded summaries
- 80 reference-free general responses
- Repeated requests, rubric paraphrases, and candidate-order reversals

The main metric was accuracy. Probability quality was assessed with Brier score, NLL, ECE, and error-detection AUROC.

---

### 3. Main accuracy results

| Task | JEV | Comparator | Difference |
|---|---:|---:|---:|
| RewardBench preference | **92.2%** | GPT-6: 93.5% | **-1.3 pp** |
| HaluEval factuality | **87.5%** | GPT-6: 86.7% | **+0.8 pp** |
| Final-answer adjudication | **94.0%** | GPT-6: 96.7% | **-2.7 pp** |
| JudgeBench correctness | **78.6%** | GPT-6: 93.1% | **-14.6 pp** |
| RM-Bench normal style | **84.0%** | GPT-6: 93.3% | **-9.4 pp** |
| RM-Bench adversarial style | **74.8%** | GPT-6: 94.6% | **-19.8 pp** |
| RewardBench 2 four-way selection | **73.0%** | GPT-6: 75.0%, Skywork: 79.0% | Slightly lower |

JEV was competitive on ordinary preference judgments, evidence-grounded factuality, and final-answer adjudication. Its weaknesses appeared when evaluation required:

- Verifying multi-step reasoning, mathematics, or coding
- Distinguishing a correct final answer from a flawed derivation
- Resisting an elaborately written but incorrect response

On JudgeBench, the gap was substantial and was also supported by blinded human adjudication, not merely by benchmark-label noise.

---

### 4. Cost and latency

On a matched 120-decision panel:

| Judge | Median latency | Cost per 1,000 judgments |
|---|---:|---:|
| JEV | **0.152 s** | **about \$0.044** |
| GPT-4.1 mini | 0.548 s | \$0.390 |
| GPT-6 Astra | 1.885 s | \$12.182 |

Thus, JEV was approximately:

- 12 times faster than GPT-6 in median latency
- 277 times cheaper than GPT-6 on this workload

The exact cost ratio depends on usage accounting, caching, reasoning-token charges, and conservative reservations for missing usage.

---

### 5. Confidence-based escalation

The paper uses JEV’s maximum label probability \(q\) as a routing signal:

1. Accept JEV’s decision when \(q\) is high.
2. Escalate to GPT-6 or another stronger judge when \(q\) is low.

JEV’s error-detection AUROC was:

- RewardBench: **0.869**
- JudgeBench: **0.745**
- HaluEval: **0.863**

Across 990 base judgments:

- Accuracy was **47.7%** when \(q < 0.6\)
- Accuracy was **99.1%** when \(q = 1\)
- On high-confidence items, JEV and GPT-6 performed similarly
- GPT-6’s advantage was concentrated among JEV’s low-confidence cases

With a single-order threshold of \(q=0.9\):

- Cascade accuracy: **91.3%**
- GPT-6-only accuracy: **91.7%**
- Retained **99.6%** of GPT-6’s accuracy
- Used about **47%** of GPT-6’s fee
- Escalated about **34%** of items

However, the threshold did not transfer reliably to every workload. On style-adversarial pairs, JEV sometimes made confident mistakes. On reference-free prose, confidence was nearly useless.

---

### 6. Probability quality and stability

JEV’s Brier scores were:

- RewardBench: **0.111**
- JudgeBench: **0.297**
- HaluEval: **0.176**

GPT-6 was better calibrated on JudgeBench, while JEV’s probabilities were better on HaluEval. This shows that accuracy and uncertainty quality are separate properties.

For stability:

- Repeated identical requests changed **0 of 96** decisions
- Rubric paraphrases changed **4 of 48** decisions
- Candidate-order disagreement was:
  - **3.2%** on RewardBench
  - **11.1%** on JudgeBench

JEV was highly repeatable, but difficult comparisons remained sensitive to presentation order and wording.

---

### 7. Limits on natural prose evaluation

For document-grounded summaries:

- JEV: **71.2%**
- GPT-5.4: **72.5%**

For reference-free general responses:

- JEV: **52.5%**
- GPT-4.1 mini: **53.8%**
- GPT-5.4: **55.0%**

All judges were close to chance on the reference-free task, while JEV’s mean confidence was still about **0.90**. Therefore, high confidence does not guarantee correctness, especially when the judge lacks supporting evidence.

---

### 8. Overall conclusion

JEV is best viewed not as a universal replacement for stronger LLM judges, but as:

- A cheap and fast first-pass judge for ordinary preference, evidence-grounded factuality, and final-answer checks
- The first stage of a confidence-based cascade
- A poor standalone choice for difficult reasoning verification, style-adversarial comparisons, and reference-free factuality assessment

The central finding is:

> **JEV is most useful when it handles confident, routine cases cheaply and sends uncertain cases to a stronger judge. Its confidence is a routing signal, not a guarantee of correctness, so thresholds must be validated for each workload.**


<br/>
# 예제



이 논문은 JEV를 새로 학습시키는 논문이라기보다, **결정만 출력하는 평가기(decision-only judge)**를 다른 LLM 평가기와 비교하고, JEV의 confidence를 이용해 **확신할 때는 바로 채택하고 불확실할 때는 더 강한 모델로 넘기는 cascade**를 검증한 연구입니다.

### 1. 모델 학습용 데이터인가?

엄밀한 의미의 **모델 파인튜닝용 training data는 사용하지 않았습니다.**  
JEV와 비교 모델들은 고정된 상태로 사용되었으며, 논문에서 말하는 “selection set”은 다음 용도로만 사용되었습니다.

- JEV 확률의 temperature scaling
- 강한 모델로 넘길 confidence threshold 결정
- 어떤 데이터에서 JEV를 그대로 사용할지 판단

즉, JEV 자체를 학습한 것이 아니라 **평가 정책과 calibration 파라미터를 정하는 데이터**입니다.

---

## 2. 평가 데이터의 기본 입력과 출력

모든 평가에서 judge는 자연어 입력과 구조화된 상태를 받고, 다음과 같은 JSON 형태의 결과를 출력합니다.

### 입력의 공통 형태

```json
{
  "question": "...",
  "candidate_answer": "...",
  "evidence": "..."
}
```

실제 필드는 task에 따라 달라집니다.

### 출력의 공통 형태

```json
{
  "verdict": "supported",
  "probabilities": {
    "supported": 0.90,
    "hallucinated": 0.10
  }
}
```

- `verdict`: 최종 판단
- `probabilities`: 가능한 각 라벨에 대한 확률
- 가장 높은 확률의 라벨이 verdict가 되어야 함
- JEV에서는 이 분포의 최댓값을 confidence \(q\)로 사용  
  예: `max(probabilities) = 0.90`

---

## 3. 주요 task별 구체적인 예시

### A. 쌍대 응답 선호도 평가 — PAIR

두 답변 중 어느 쪽이 더 좋은지 판단합니다.

#### 입력 예시

```json
{
  "question": "파리는 어느 나라의 수도인가?",
  "response_A": "파리는 프랑스의 수도이다.",
  "response_B": "파리는 독일의 수도이다."
}
```

#### 가능한 출력

```json
{
  "verdict": "A",
  "probabilities": {
    "A": 0.98,
    "B": 0.02
  }
}
```

#### 판단 기준

- 사실 정확성
- 추론의 타당성
- 사용자 지시 준수
- 관련성
- 안전성
- 단순히 더 길거나 자신감 있게 쓰였다는 이유로 선호하지 않음

#### 사용 데이터

- **RewardBench**: 400개 응답 쌍
- **JudgeBench**: 350개 응답 쌍
- 일부 후속 실험:
  - RewardBench 2의 4개 후보 중 1개 선택
  - RM-Bench의 문체가 다른 응답 쌍 비교

특히 문체가 정교한 오답을 고르는 상황에서 JEV의 성능이 크게 떨어졌습니다.

---

### B. 근거 기반 사실성 평가 — HALL

주어진 evidence에 비추어 답변이 맞는지 판단합니다.

#### 입력 예시

```json
{
  "question": "Aster0는 몇 개의 상자를 배달했는가?",
  "evidence": "검증된 장부에는 Aster0가 11개의 상자를 배달했다고 기록되어 있다.",
  "candidate_answer": "Aster0는 11개의 상자를 배달했다."
}
```

#### 가능한 출력

```json
{
  "verdict": "supported",
  "probabilities": {
    "supported": 1.0,
    "hallucinated": 0.0
  }
}
```

#### 라벨 의미

- `supported`: evidence가 답변을 뒷받침함
- `hallucinated`: evidence와 모순되거나 evidence에 없는 사실을 추가함

#### 사용 데이터

- **HaluEval**의 evidence-grounded QA 240개
- 질문, 근거 문서, 답변을 함께 제공
- 주어진 근거만 사용하여 판단하도록 함

JEV는 HaluEval에서 87.5%를 기록했고, GPT-6은 86.7%였습니다. 다만 일부 benchmark 라벨 자체가 evidence와 맞지 않는 사례도 확인되었습니다.

---

### C. 최종 답변 판정 — FINAL

모델의 긴 답변 전체를 다시 풀어보는 것이 아니라, **최종적으로 명시한 답**이 reference와 일치하는지 평가합니다.

#### 입력 예시

```json
{
  "question": "다음 중 정답은 무엇인가? A, B, C, D",
  "reference_answer": "B",
  "model_reply": "처음에는 C라고 생각했지만, 다시 검토한 결과 최종 답은 B이다."
}
```

#### 가능한 출력

```json
{
  "verdict": "correct",
  "probabilities": {
    "correct": 0.96,
    "incorrect": 0.03,
    "no_answer": 0.01
  }
}
```

#### 판단 규칙

- 명확한 최종 답이 있으면 그것을 사용
- 앞의 잘못된 답이 뒤에서 명시적으로 수정되면 수정된 답을 사용
- 답을 전혀 선택하지 않으면 `no_answer`
- 중간의 잘못된 설명보다 최종 commitment를 우선

#### 사용 데이터

- 기존에 저장된 모델 응답 150개
- 정답 라벨:
  - correct 99개
  - incorrect 25개
  - no-answer 26개

이 task에서 JEV는 94.0%, GPT-6은 96.7%였습니다.

---

### D. 추출 기반 평가 — Gold-blind extraction

reference를 보여주지 않고, 모델 답변에서 최종 선택지만 추출한 뒤 코드로 정답과 비교하는 방식입니다.

#### 입력 예시

```json
{
  "question": "다음 중 정답은 무엇인가?",
  "model_reply": "A와 B를 모두 고려할 수 있지만, 최종적으로는 D를 선택하겠습니다."
}
```

#### 출력 예시

```json
{
  "verdict": "D",
  "probabilities": {
    "A": 0.01,
    "B": 0.02,
    "C": 0.01,
    "D": 0.96,
    "no_answer": 0.00,
    "ambiguous": 0.00
  }
}
```

그다음 프로그램이 `D`와 숨겨진 reference를 비교합니다.

#### 결과

- JEV의 직접 판정: 91.3%
- JEV의 답 추출 후 코드 비교: 86.0%

즉, 답변을 직접 의미적으로 판정하는 것과 최종 선택지를 추출하여 비교하는 것은 같은 문제가 아니었습니다.

---

## 4. 데이터 분할과 실험 흐름

공개 benchmark 중 RewardBench, JudgeBench, HaluEval에 대해 다음과 같이 나누었습니다.

| 구분 | 용도 |
|---|---|
| Pilot selection set | temperature와 cascade threshold 결정 |
| Pilot test set | pilot 설정을 고정한 뒤 평가 |
| Extension set | pilot 결과를 본 뒤 새로 고정한 추가 테스트 |

주요 분할은 다음과 같습니다.

- RewardBench: selection 64개, test 96개, extension 240개
- JudgeBench: selection 32개, test 48개, extension 270개
- HaluEval: selection 32개, test 48개, extension 160개

Extension 데이터는 calibration이나 threshold fitting에 사용하지 않았습니다.

---

## 5. Confidence cascade의 입력과 출력

JEV의 판단을 항상 최종 답으로 쓰지 않고 confidence에 따라 처리합니다.

### 처리 방식

```text
JEV 입력
   ↓
JEV verdict + confidence
   ↓
confidence가 높음 → JEV 결과 채택
confidence가 낮음 → GPT-6 같은 강한 judge 호출
```

예를 들어 threshold를 0.90으로 정하면 다음과 같습니다.

- `q >= 0.90`: JEV verdict 채택
- `q < 0.90`: GPT-6에 넘김

### 결과 예시

단일 순서 평가에서 JEV → GPT-6 cascade는:

- 약 34%의 항목을 GPT-6으로 escalation
- GPT-6 단독 정확도 91.7%
- cascade 정확도 91.3%
- GPT-6 비용의 약 47%
- GPT-6 정확도의 약 99.6% 유지

두 후보를 양쪽 순서로 평가하고 확률을 평균한 frozen policy에서는:

- JEV 결과를 약 53.7%의 쌍에서 바로 채택
- cascade 정확도 92.5%
- GPT-6 단독 93.1%
- GPT-6 비용의 약 56.8%

단, 이 정책은 모든 task에서 잘 작동하지 않았습니다. 특히 다음과 같은 경우에는 confidence가 신뢰하기 어렵습니다.

- 여러 단계의 수학·논리·코딩 추론을 검증해야 하는 경우
- 문장이 매우 정교하지만 내용은 틀린 답변
- reference 없이 일반적인 자연어 사실성을 판단하는 경우

---

## 6. 논문에서 확인한 핵심 결과

| Task | JEV | 강한 비교 모델 | 결론 |
|---|---:|---:|---|
| 일반 응답 선호도 | 92.2% | GPT-6 93.5% | JEV 사용 가능 |
| 근거 기반 사실성 | 87.5% | GPT-6 86.7% | JEV 사용 가능 |
| 최종 답변 판정 | 94.0% | GPT-6 96.7% | JEV 사용 가능 |
| 어려운 정답성 판단 | 78.6% | GPT-6 93.1% | 강한 모델로 escalation |
| 문체가 오답을 그럴듯하게 만드는 쌍 | 74.8% | GPT-6 94.6% | escalation 권장 |
| reference 없는 자연어 | 52.5% | GPT-5.4 55.0% | 어떤 judge도 불안정 |

핵심은 **JEV가 모든 평가를 잘한다는 것이 아니라**, 비교적 쉬운 일반 선호도·근거 기반 판단에서는 저렴한 1차 평가기로 유용하고, confidence가 낮은 어려운 사례만 더 강한 모델에 넘기는 것이 효과적이라는 점입니다.

---




This paper does not train or fine-tune JEV. It evaluates JEV as a **decision-only judge** and tests whether its confidence can be used to route difficult cases to a stronger LLM judge.

The so-called training-related data are actually used only for:

- temperature scaling,
- selecting escalation thresholds,
- validating the routing policy.

No judge is newly trained on these examples.

---

## 1. General input and output format

A judge receives a structured state and natural-language instructions.

### Generic input

```json
{
  "question": "...",
  "candidate_answer": "...",
  "evidence": "..."
}
```

The exact fields depend on the task.

### Generic output

```json
{
  "verdict": "supported",
  "probabilities": {
    "supported": 0.90,
    "hallucinated": 0.10
  }
}
```

The verdict must be the highest-probability label. JEV’s confidence is measured as the maximum label probability:

```text
q = max(probabilities)
```

---

## 2. Main tasks and concrete examples

### A. Pairwise preference judgment — PAIR

The judge selects the better of two candidate responses.

#### Input

```json
{
  "question": "What is the capital of France?",
  "response_A": "Paris is the capital of France.",
  "response_B": "Berlin is the capital of France."
}
```

#### Output

```json
{
  "verdict": "A",
  "probabilities": {
    "A": 0.98,
    "B": 0.02
  }
}
```

The rubric considers factual correctness, reasoning, instruction following, relevance, and safety. The judge should not prefer an answer merely because it is longer or more confident.

Datasets include:

- RewardBench: 400 pairs
- JudgeBench: 350 pairs
- RewardBench 2: four-way selection
- RM-Bench: style-sensitive preference pairs

JEV performed substantially worse when a wrong answer was written in a more elaborate or persuasive style.

---

### B. Evidence-grounded factuality — HALL

The judge checks whether an answer is supported by the supplied evidence.

#### Input

```json
{
  "question": "How many crates did Aster0 deliver?",
  "evidence": "The verified ledger records that Aster0 delivered 11 crates.",
  "candidate_answer": "Aster0 delivered 11 crates."
}
```

#### Output

```json
{
  "verdict": "supported",
  "probabilities": {
    "supported": 1.0,
    "hallucinated": 0.0
  }
}
```

Possible labels:

- `supported`
- `hallucinated`

This task uses 240 HaluEval evidence-grounded judgments. JEV achieved 87.5%, compared with GPT-6 at 86.7%, although some benchmark labels were found to be inconsistent with the evidence.

---

### C. Final-answer adjudication — FINAL

The judge evaluates only the answer explicitly committed to at the end of a model response.

#### Input

```json
{
  "question": "Which option is correct: A, B, C, or D?",
  "reference_answer": "B",
  "model_reply": "I initially thought C, but after reconsidering, my final answer is B."
}
```

#### Output

```json
{
  "verdict": "correct",
  "probabilities": {
    "correct": 0.96,
    "incorrect": 0.03,
    "no_answer": 0.01
  }
}
```

A later explicit revision supersedes an earlier answer. If the model makes no clear commitment, the output is `no_answer`.

The dataset contains 150 existing replies:

- 99 correct
- 25 incorrect
- 26 no-answer

JEV achieved 94.0%, while GPT-6 achieved 96.7%.

---

### D. Gold-blind answer extraction

The judge does not see the reference answer. It only extracts the final committed option. A program then compares the extracted option with the reference.

#### Input

```json
{
  "question": "Which option is correct?",
  "model_reply": "Both A and B seem plausible, but my final choice is D."
}
```

#### Output

```json
{
  "verdict": "D",
  "probabilities": {
    "A": 0.01,
    "B": 0.02,
    "C": 0.01,
    "D": 0.96,
    "no_answer": 0.00,
    "ambiguous": 0.00
  }
}
```

The extracted option is then compared with the hidden reference in code.

JEV’s results were:

- Direct adjudication: 91.3%
- Extraction followed by code comparison: 86.0%

This shows that semantic grading and answer extraction are different tasks.

---

## 3. Data splits

For RewardBench, JudgeBench, and HaluEval, the study uses three stages:

| Split | Purpose |
|---|---|
| Pilot selection set | Fit temperature scaling and choose routing thresholds |
| Pilot test set | Evaluate the fixed pilot policy |
| Extension set | Held-out evaluation after the policy was fixed |

Approximate counts:

- RewardBench: 64 selection, 96 test, 240 extension
- JudgeBench: 32 selection, 48 test, 270 extension
- HaluEval: 32 selection, 48 test, 160 extension

The extension data were not used to fit the calibration or routing policy.

---

## 4. Confidence cascade

The system first asks JEV for a verdict and confidence.

```text
Input → JEV verdict + confidence
             ↓
      high confidence → accept JEV
      low confidence  → call GPT-6
```

For a threshold of 0.90:

- If `q >= 0.90`, accept JEV’s decision.
- If `q < 0.90`, escalate to GPT-6.

In the single-order simulation:

- About 34% of items were escalated.
- Cascade accuracy: 91.3%.
- GPT-6-only accuracy: 91.7%.
- Cost: about 47% of GPT-6 alone.
- About 99.6% of GPT-6’s accuracy was retained.

In the frozen two-order policy, the candidates were evaluated in both orders and their aligned probabilities were averaged. This policy accepted approximately 53.7% of pairs directly, achieved 92.5% accuracy versus GPT-6’s 93.1%, and used about 56.8% of GPT-6’s fee.

The policy is not universally reliable. Confidence becomes less useful for:

- multi-step mathematical, reasoning, or coding verification,
- answers that are elaborately written but incorrect,
- reference-free factuality judgments over general prose.

---

## 5. Main findings

| Workload | JEV | Strong comparator | Guidance |
|---|---:|---:|---|
| Ordinary preference | 92.2% | GPT-6: 93.5% | Use JEV |
| Evidence-grounded factuality | 87.5% | GPT-6: 86.7% | Use JEV |
| Final-answer adjudication | 94.0% | GPT-6: 96.7% | Use JEV |
| Difficult correctness | 78.6% | GPT-6: 93.1% | Escalate |
| Style-adversarial pairs | 74.8% | GPT-6: 94.6% | Escalate |
| Reference-free prose | 52.5% | GPT-5.4: 55.0% | Not supported |

The main conclusion is not that JEV is universally strong. Rather, JEV is useful as a cheap first-pass judge for ordinary preference, evidence-grounded factuality, and final-answer decisions. Difficult or uncertain cases should be routed to a stronger judge instead of relying on JEV alone.

<br/>
# 요약


  
JEV를 16개 생성형·보상모델 판정기와 비교하고, 선호도·근거 기반 사실성·정답 판정 벤치마크에서 정확도, 비용, 지연시간, 확률 신뢰도를 측정했으며 일부 불일치는 인간이 추가 판정했다.  
JEV는 일반 선호도와 근거 기반 사실성에서 최강 비교 모델 GPT-6와 3%p 이내였지만, 여러 단계의 추론을 검증하는 JudgeBench에서는 78.6% 대 93.1%, 정교하게 작성된 오답을 고르는 RM-Bench 어려운 조건에서는 74.8% 대 94.6%로 크게 뒤졌다.  
JEV의 낮은 확신 사례만 GPT-6로 넘기는 방식은 확신도 임계값을 검증해 사용해야 하며, 예를 들어 임계값 0.9에서 GPT-6 정확도의 99.6%를 약 47% 비용으로 유지했지만, 스타일에 속거나 근거 없는 자연어를 평가하는 경우에는 높은 확신에도 오류가 발생해 별도 검증이 필요하다.  


The study compared JEV with 16 generative and reward-model judges on preference, evidence-grounded factuality, and answer-adjudication benchmarks, measuring accuracy, cost, latency, and confidence quality, with human review of selected disagreements.  
JEV stayed within three percentage points of the strongest GPT-6 comparator on ordinary preference and grounded factuality, but lagged substantially on multi-step correctness checking—78.6% versus 93.1% on JudgeBench—and on style-adversarial pairs, 74.8% versus 94.6%.  
A confidence cascade that escalates only low-confidence JEV decisions to GPT-6 retained 99.6% of GPT-6’s accuracy at about 47% of its cost at a 0.9 threshold, but thresholds require local validation because JEV can remain confidently wrong on misleading styles and reference-free prose.

<br/>
# 기타



## 1. 다이어그램·피규어

### Figure 1 — 평가 인터페이스 비교
- **내용:** 인간 평가, 생성형 LLM judge, JEV의 구조를 비교한다.
- **핵심 결과:** JEV는 설명을 생성하지 않고, **정해진 타입의 판정값과 label probability**만 반환한다.
- **인사이트:** 생성형 judge보다 출력이 단순하지만, 판정과 불확실성을 분리해 **자동 라우팅·비용 절감**에 적합하다.

### Figure 2 — 지연시간과 비용
- **결과:** JEV의 중앙 지연시간은 **0.152초**, 비용은 **1,000건당 약 $0.044**이다.
- GPT-4.1 mini는 0.548초/$0.390, GPT-6는 1.885초/$12.182이다.
- **인사이트:** JEV는 GPT-6보다 약 **277배 저렴하고 9배 빠르다**. 단, 비용은 실제 API 사용량과 보수적 예약 비용을 포함한 추정치다.

### Figure 3 — confidence 분포와 calibration
- **결과:** JEV의 confidence는 대체로 정답과 오류를 구분한다.
- Error-detection AUROC는 RewardBench **0.869**, JudgeBench **0.745**, HaluEval **0.863**이다.
- **인사이트:** confidence가 높을수록 대체로 정확하지만, JudgeBench나 스타일 함정처럼 **높은 confidence의 오답**도 존재한다. confidence는 정답 보증서가 아니라 escalation 신호다.

### Figure 4 — “확신하면 수용, 불확실하면 위임”
- **구조:** JEV가 먼저 판정하고, confidence가 threshold보다 낮으면 GPT-6 같은 강한 judge에게 넘긴다.
- **인사이트:** JEV의 verdict와 confidence를 서로 다른 역할로 사용한다. verdict는 후보 답이고, confidence는 **검토 필요성**을 판단하는 gate다.

### Figure 5 — confidence 기반 cascade
- **결과:** confidence가 낮은 구간에서 JEV의 정확도가 크게 떨어지고 GPT-6의 우위가 커진다.
- 예를 들어 전체 데이터에서 JEV는 `q<0.6`일 때 정확도 **47.7%**, `q=1`일 때 **99.1%**다.
- τ=0.9에서 JEV→GPT-6 cascade는 GPT-6 정확도의 **99.6%**를 유지하면서 비용을 약 **47%** 수준으로 낮춘다.
- **인사이트:** JEV의 오류가 무작위가 아니라 confidence에 집중되어 있기 때문에 cascade가 작동한다.

### Figure 6 — 실제 JEV 요청·응답 예시
- **결과:** evidence와 claim을 입력하면 JEV가 `supported`와 확률 분포를 반환한다.
- **인사이트:** typed output은 구조화된 파이프라인에 바로 연결하기 쉽고, 출력 형식 오류를 줄인다. 하지만 **유효한 출력과 올바른 판단은 별개의 문제**다.

### Figure 7 — 도메인별 정확도
- **결과:** JEV는 RewardBench의 일반 선호·안전 영역에서는 강하지만, JudgeBench의 reasoning·coding 영역에서는 크게 약하다.
- **인사이트:** 전체 평균보다 **판단 업무의 성격**이 중요하다. 단순 선호 비교와 다단계 추론 검증은 서로 다른 능력을 요구한다.

### Figure 8 — 정확도와 비용·지연시간의 trade-off
- **결과:** JEV는 RewardBench와 HaluEval에서 낮은 비용으로 높은 정확도를 보인다.
- JudgeBench에서는 GPT-5.6/GPT-6가 훨씬 높은 정확도를 보이며, 추가 비용을 지불할 가치가 있다.
- **인사이트:** 하나의 모델이 모든 평가 업무에서 최적은 아니다. **업무별 운영 영역(operating envelope)**을 정해야 한다.

### Figure 9 — RewardBench 2와 RM-Bench 비교
- **결과:** RewardBench 2에서 JEV 73.0%, GPT-6 75.0%, Skywork 79.0%다.
- RM-Bench에서 JEV는 hard 74.8%, normal 84.0%, easy 87.3%다.
- **인사이트:** 정답보다 더 정교하게 쓰인 오답을 고르는 스타일 함정에서 JEV 성능이 크게 하락한다.

### Figure 10 — RM-Bench 스타일 조합별 성능
- **결과:** 선호 답변이 간결하고 거부 답변이 상세·Markdown으로 쓰인 경우 JEV 성능이 특히 낮다.
- **인사이트:** JEV는 내용뿐 아니라 **표현의 정교함과 형식에 영향을 받는다**. 스타일 편향은 단순한 위치 편향과 다른 문제다.

### Figure 11 — Brier, NLL, ECE
- **결과:** GPT-6는 JudgeBench에서 정확도와 calibration이 모두 좋지만, HaluEval에서는 JEV의 확률 품질이 더 나은 경우가 있다.
- **인사이트:** 정확도가 높은 judge가 항상 confidence calibration도 좋은 것은 아니다. **정확도와 불확실성 품질은 분리해서 평가**해야 한다.

### Figure 12 — selective prediction
- **결과:** 낮은 coverage, 즉 확신이 높은 사례만 수용할수록 오류율이 감소한다.
- **인사이트:** JEV는 모든 입력을 직접 처리하기보다, 확신 높은 쉬운 사례를 맡고 어려운 사례를 보류하는 방식에 적합하다.

### Figure 13 — judge 간 오류 상호보완성
- **결과:** JudgeBench에서 GPT-6는 JEV 오류 75건 중 60건을 수정한다. 두 judge의 오류가 완전히 겹치지는 않는다.
- **인사이트:** 강한 judge를 항상 호출할 필요는 없고, **JEV가 틀릴 가능성이 높은 사례에만 호출**하면 비용 대비 효과가 커진다.

### Figure 14 — 안정성
- **결과:** JEV는 동일 요청 반복에서는 판정을 바꾸지 않았지만, 후보 순서를 바꾸면 RewardBench 3.2%, JudgeBench 11.1%가 바뀐다.
- **인사이트:** 반복 안정성, rubric paraphrase 안정성, 후보 순서 안정성은 서로 다른 속성이다. pairwise 평가에서는 양쪽 순서를 모두 시험해야 한다.

### Figure 15 — Choice·Noul·Score 인터페이스 비교
- **결과:** 같은 의미의 질문이라도 인터페이스에 따라 확률이 다소 달라지고, 48개 사례 중 hard label이 1건 달랐다.
- **인사이트:** 출력 타입이 같아 보여도 내부 인터페이스가 완전히 동일하지 않을 수 있다. 확률을 사용할 때는 **실제 사용 인터페이스에서 검증**해야 한다.

### Figure 16 — GSM8K positive control
- **결과:** 반복적으로 challenge된 대화에서도 대부분 judge가 108/108 정답을 맞혔다.
- **인사이트:** 쉬운 reference-based task에서는 모델 간 차이가 거의 없다. 높은 점수는 일반적 신뢰성을 의미하지 않는다.

### Figure 17 — 최종 답변 직접 판정 vs 답 추출
- **결과:** JEV는 자연스러운 multiple-choice 답변에서 직접 판정 91.3%, 답 추출 후 비교 86.0%였다.
- **인사이트:** 답을 직접 평가하는 것과 최종 옵션을 추출하는 것은 다른 과업이다. 추출은 더 inspectable하지만 반드시 더 정확하지는 않다.

### Figure 18 — 자연어 prose 평가
- **결과:** reference-free prose에서 JEV 정확도는 **52.5%**, 평균 confidence는 약 0.90으로 높았다.
- **인사이트:** JEV뿐 아니라 다른 judge도 근거 없는 prose 사실성 평가에서는 거의 chance 수준인데도 자신 있게 틀린다. **외부 근거 없는 평가에는 confidence cascade도 충분하지 않다.**

---

## 2. 주요 테이블

### Table 1 — 전체 judge 정확도
- JEV: RewardBench **92.2%**, JudgeBench **78.6%**, HaluEval **87.5%**
- GPT-6: 각각 **93.5%, 93.1%, 86.7%**
- **핵심:** 일반 선호와 evidence-grounded factuality에서는 JEV가 강하지만, 어려운 correctness에서는 GPT 계열이 크게 앞선다.

### Table 2 — 업무별 운영 지침
- 일반 선호, evidence 기반 factuality, final-answer adjudication: **JEV 사용**
- 다단계 추론·어려운 correctness·style-adversarial pair: **강한 judge로 escalation**
- reference-free prose: **어떤 judge도 신뢰하기 어려움**
- **핵심:** 논문의 가장 실용적인 의사결정 표다.

### Table 3 — 사전 고정된 two-order cascade
- GPT-6 fallback, threshold 0.9:
  - JEV 판정 수용 53.7%
  - cascade 정확도 92.5%
  - GPT-6 단독 93.1%
  - 비용 56.8% 수준
- **핵심:** 후보 순서를 양쪽으로 평가하고 확률을 정렬해 평균하면, 사전에 정한 정책도 성능을 거의 유지한다.

### Table 4 — 데이터 분할
- 전체 642개 pilot과 670개 extension으로 구성된다.
- selection set은 temperature와 threshold fitting에만 사용되고, extension은 fitting에 사용하지 않았다.
- **핵심:** cascade와 calibration의 과적합을 줄이기 위해 데이터를 분리했다.

### Table 5 — 모델 설정·가격
- 각 모델의 정확한 identifier, inference effort, 입력·출력 가격을 기록한다.
- **핵심:** 모델마다 reasoning effort와 serving 환경이 달라 완전한 compute-matched 비교는 아니다.

### Table 6 — 출력 유효성
- JEV는 1,312건 모두 valid output이었다.
- 일부 다른 모델은 provider 오류, schema 위반, transport 오류가 있었다.
- **핵심:** 연구에서는 invalid output을 오류로 계산했다. 서비스 안정성과 reasoning quality를 구분해야 한다.

### Table 7 — 추가 지표
- JEV의 error AUROC: RewardBench 0.869, JudgeBench 0.745, HaluEval 0.863
- GPT-6는 JudgeBench 0.907로 더 강하지만 HaluEval에서는 JEV보다 반드시 우수하지 않다.
- **핵심:** benchmark별로 confidence 품질 순위가 바뀐다.

### Table 8 — follow-up benchmark
- JEV는 RM-Bench hard에서 74.8%, normal에서 84.0%, easy에서 87.3%다.
- **핵심:** misleading style이 들어가면 성능이 크게 떨어진다.

### Table 9 — judge 간 paired difference
- RM-Bench hard에서 JEV−GPT-6는 **−19.8%p**
- RewardBench 2 four-way에서는 **−2.0%p**
- **핵심:** 작은 차이의 평균 성능만으로는 스타일 취약성을 발견할 수 없다.

### Table 10 — follow-up confidence 품질
- hard RM-Bench에서 JEV의 high-confidence error가 27건, GPT-6는 3건이다.
- **핵심:** JEV의 confidence gate는 일반 pair에는 유용하지만, 스타일 함정에서는 과신한다.

### Table 11 — 모든 frozen cascade 정책
- fallback 모델에 따라 threshold와 성능 유지 정도가 크게 달라진다.
- GPT-6 정책은 상대적으로 안정적이지만 GPT-5.6 등에서는 threshold가 잘 전이되지 않는다.
- **핵심:** threshold를 다른 fallback이나 업무에 그대로 복사하면 안 된다.

### Table 12 — temperature scaling
- RewardBench에서는 calibration이 악화됐고, HaluEval에서는 개선됐다.
- **핵심:** temperature는 workload별로 따로 검증해야 하며, 하나의 전역 temperature는 적절하지 않다.

### Table 13 — single-order cascade
- τ=0.9에서 pooled cascade는 GPT-6 정확도의 99.6%를 유지하고 비용은 약 47%다.
- JudgeBench에서는 더 많은 사례를 escalation해야 한다.
- **핵심:** 업무가 어려울수록 JEV의 평균 confidence가 낮고 fallback 호출률이 높아진다.

### Table 14 — 안정성과 순서 민감도
- JEV는 반복 요청 변화 0/96, paraphrase 변화 4/48이다.
- 그러나 JudgeBench 순서 불일치는 11.1%다.
- **핵심:** 결정적 출력이 곧 의미적 안정성을 보장하지 않는다.

### Table 15 — 위치 편향
- JEV의 first-position 선택률은 RewardBench 48.9%, JudgeBench 48.4%다.
- **핵심:** JEV의 오류는 단순히 첫 번째 응답을 선호해서 생긴 것이 아니라, 순서에 따른 불안정성에 가깝다.

### Table 16 — 인간 adjudication
- JudgeBench disagreement에서 인간은 GPT-6을 57건, JEV를 1건 지지했다.
- 인간 기준 JEV−GPT-6 차이는 **−16.0%p**다.
- **핵심:** JudgeBench에서 JEV의 약점은 단순한 benchmark label noise가 아니다.

### Table 17 — answer-format follow-up
- JEV의 자연 MC direct agreement는 91.3%, extraction 기반은 86.0%다.
- **핵심:** final-answer adjudication과 answer extraction은 서로 다른 downstream task다.

### Table 18 — 짧은 답변과 prose
- HaluEval에서 JEV는 짧은 답변 87.9%, 20단어 이상 prose 81.2%다.
- **핵심:** 긴 prose는 여러 주장과 의미적 변형을 포함해 더 어렵지만, 표본이 작아 설명적 결과로만 봐야 한다.

### Table 19 — 자연어 prose
- 문서 근거 summary: JEV **71.2%**
- reference-free general response: JEV **52.5%**
- **핵심:** evidence가 주어지면 어느 정도 작동하지만, reference-free 사실성 판정은 모든 모델에서 취약하다.

---

## 3. 어펜딕스별 핵심

### Appendix A — 데이터, rubric, 요청 예시
- PAIR, HALL, FINAL, EVIDENCE 과업의 입력 필드와 label 의미를 명시한다.
- **인사이트:** judge 출력 형식보다 실제 평가 대상과 rubric이 중요하다. 특히 FINAL은 중간 추론이 아니라 **명시된 최종 답변**만 평가한다.

### Appendix B — 모델 설정과 비용 계산
- 정확한 모델 버전, 가격, reasoning mode, local serving 설정을 제시한다.
- **인사이트:** 결과는 모델 자체뿐 아니라 effort, provider, context limit, serving 조건의 영향을 받는다.

### Appendix C — 운영상 유효성
- provider generation failure, HTTP-200 schema violation, transport failure를 분리한다.
- **인사이트:** “판단을 못함”과 “판단은 했지만 틀림”을 같은 종류의 오류로 해석하면 안 된다.

### Appendix D — 도메인 결과와 효율성
- RewardBench·JudgeBench·HaluEval의 세부 도메인별 성능과 비용 곡선을 제시한다.
- **인사이트:** 평균 점수보다 domain-level profile이 실제 배치 결정에 유용하다.

### Appendix E — 추가 benchmark 검증
- RewardBench 2와 RM-Bench에서 JEV, GPT-6, Skywork를 비교한다.
- **인사이트:** JEV의 취약점은 일반 선호보다 **four-way selection과 misleading style**에서 더 분명해진다.

### Appendix F — 확률 품질과 selective prediction
- Brier, NLL, ECE, risk–coverage, cascade threshold를 제시한다.
- **인사이트:** calibration은 모델·업무별로 다르며, threshold는 local validation이 필수다.

### Appendix G — interface diagnostics
- Choice, Noul, Score가 의미상 유사해도 확률값이 완전히 같지 않음을 보인다.
- **인사이트:** API 인터페이스 자체가 confidence에 영향을 줄 수 있다.

### Appendix H — 사례 분석과 rubric alignment
- 잘못된 풀이지만 정답을 고른 답변, 애매한 entity, 길이·형식 지시 문제 등을 보여준다.
- **인사이트:** “최종 답이 맞는가”, “풀이가 올바른가”, “지시를 따랐는가”는 서로 다른 평가 기준이다.

### Appendix I — 인간 adjudication
- JEV와 GPT-6의 disagreement를 인간이 blind adjudication했다.
- **인사이트:** JudgeBench에서는 GPT-6의 우위가 실제 인간 판단에서도 확인됐고, HaluEval에서는 benchmark label noise도 상당했다.

### Appendix J — answer format 및 extraction
- direct grading, final-option extraction, free-response를 비교한다.
- **인사이트:** 구조화된 선택지 형식이 항상 의미 평가보다 쉬운 것은 아니며, extraction은 자체적인 오류를 가진다.

### Appendix K — 자연어 free-response
- document-grounded summary와 reference-free general response를 분리한다.
- **인사이트:** 근거가 있는 요약 평가와 일반 상식 기반 hallucination 평가는 별개의 workload다.

### Appendix L — 재현성 패키지
- 입력, rubric, judge output, metric 계산 코드, adjudication 자료를 제공한다.
- **인사이트:** API를 다시 호출하지 않고도 공개 결과를 재현할 수 있도록 구성했지만, 원본 API 로그와 private data는 공개하지 않았다.

---

## 전체 결론

이 논문의 가장 중요한 메시지는 다음과 같다.

1. **JEV는 저비용 1차 judge로 충분한 경우가 많다.**  
   일반적인 preference, evidence-grounded factuality, final-answer adjudication에서는 강한 LLM과 약 3%p 이내의 차이를 보인다.

2. **어려운 추론과 스타일 함정에서는 escalation이 필요하다.**  
   derivation 검증, coding·reasoning correctness, 정교하게 작성된 오답 판별에서 성능 격차가 커진다.

3. **confidence는 cascade에 유용하지만 절대적 신뢰 신호는 아니다.**  
   JEV의 confidence는 보통 오류를 잘 순위화하지만, reference-free prose와 style-adversarial 사례에서는 과신한다.

4. **실제 배포 전 local validation이 필수다.**  
   threshold, temperature, fallback 모델은 workload별로 다시 검증해야 하며, pairwise 평가에서는 양쪽 순서를 모두 확인해야 한다.

---




## 1. Diagrams and Figures

### Figure 1 — Evaluation interfaces
- **Content:** Compares human evaluation, generative LLM judges, and JEV.
- **Key result:** JEV returns a typed verdict and label probabilities without generating a rationale.
- **Insight:** The interface is simpler, but it directly supports confidence-based routing and cost control.

### Figure 2 — Latency and cost
- **Result:** JEV has a median latency of **0.152 seconds** and costs about **$0.044 per 1,000 judgments**.
- GPT-6 takes 1.885 seconds and costs $12.182 per 1,000 judgments.
- **Insight:** JEV is roughly **277× cheaper and 9× faster** than GPT-6 in this setup.

### Figure 3 — Confidence and calibration
- **Result:** JEV confidence generally distinguishes correct from incorrect judgments. Error-detection AUROC is 0.869 on RewardBench, 0.745 on JudgeBench, and 0.863 on HaluEval.
- **Insight:** Confidence is useful, but high-confidence errors still occur. It is an escalation signal, not a certificate of correctness.

### Figure 4 — Accept when confident, escalate when unsure
- **Structure:** JEV makes the first decision; low-confidence cases are sent to a stronger judge.
- **Insight:** The verdict and confidence serve different roles: one is the candidate answer, the other determines whether additional evaluation is needed.

### Figure 5 — Confidence-based cascade
- **Result:** JEV’s accuracy drops sharply in low-confidence bins, while GPT-6 gains an advantage there.
- At threshold 0.9, the cascade preserves **99.6% of GPT-6’s accuracy** at about **47% of its cost**.
- **Insight:** JEV’s errors are concentrated in uncertain cases, which makes selective escalation effective.

### Figure 6 — Example JEV request and response
- **Result:** Given evidence and a claim, JEV returns a label such as `supported` and a probability distribution.
- **Insight:** Typed output is easy to integrate into pipelines, but output validity and judgment correctness are separate properties.

### Figure 7 — Domain-level accuracy
- **Result:** JEV is strong on ordinary preference and safety-related tasks but weaker on reasoning and coding correctness.
- **Insight:** The nature of the workload matters more than the output format. Preference comparison and multi-step verification require different abilities.

### Figure 8 — Accuracy versus cost and latency
- **Result:** JEV offers a strong cost–quality trade-off on RewardBench and HaluEval, but GPT-5.6/GPT-6 are much better on JudgeBench.
- **Insight:** No single judge is optimal for every workload. Deployment should be based on a workload-specific operating envelope.

### Figure 9 — RewardBench 2 and RM-Bench
- **Result:** JEV scores 73.0% on RewardBench 2, compared with 75.0% for GPT-6 and 79.0% for Skywork.
- On RM-Bench, JEV scores 74.8% on hard pairs, 84.0% on normal pairs, and 87.3% on easy pairs.
- **Insight:** JEV struggles when an incorrect answer is written in a more polished style.

### Figure 10 — RM-Bench style combinations
- **Result:** JEV performs especially poorly when the preferred answer is concise and the rejected answer is detailed or formatted in Markdown.
- **Insight:** This is a style sensitivity problem, not merely a position bias problem.

### Figure 11 — Brier score, NLL, and ECE
- **Result:** GPT-6 is more accurate and better calibrated on JudgeBench, while JEV can have better probability quality on HaluEval.
- **Insight:** Accuracy and uncertainty quality are distinct properties and should be evaluated separately.

### Figure 12 — Selective prediction
- **Result:** Accepting only high-confidence decisions reduces the error rate.
- **Insight:** JEV is best used for confident, routine cases while uncertain cases are deferred.

### Figure 13 — Error complementarity
- **Result:** On JudgeBench, GPT-6 corrects 60 of JEV’s 75 errors.
- **Insight:** The judges do not make exactly the same mistakes, so selective use of a stronger judge can be cost-effective.

### Figure 14 — Stability
- **Result:** JEV changed no decisions across repeated requests, but reversing candidate order changed 3.2% of RewardBench and 11.1% of JudgeBench decisions.
- **Insight:** Repeatability, paraphrase stability, and order stability are different properties. Pairwise judgments should be evaluated in both orders.

### Figure 15 — Choice, Noul, and Score interfaces
- **Result:** Semantically equivalent interfaces can produce slightly different probabilities; one hard-label disagreement occurred in the 48-example audit.
- **Insight:** Confidence must be validated using the exact interface used in deployment.

### Figure 16 — GSM8K positive control
- **Result:** Most judges achieved 108/108 on the easy reference-based control.
- **Insight:** Near-perfect performance on easy controls says little about broad reliability.

### Figure 17 — Direct grading versus answer extraction
- **Result:** JEV scores 91.3% on direct grading of natural multiple-choice replies, but 86.0% when the final option is extracted first.
- **Insight:** Direct adjudication and answer extraction are different downstream tasks. Extraction is more inspectable, but not necessarily more accurate.

### Figure 18 — Natural prose evaluation
- **Result:** JEV reaches only 52.5% on reference-free prose while maintaining a mean confidence near 0.90.
- **Insight:** All tested judges are close to chance on reference-free factuality while remaining highly confident. Confidence-based routing is not sufficient without external evidence.

---

## 2. Main Tables

### Table 1 — Overall judge accuracy
- JEV: 92.2% on RewardBench, 78.6% on JudgeBench, and 87.5% on HaluEval.
- GPT-6: 93.5%, 93.1%, and 86.7%, respectively.
- **Key point:** JEV is competitive on ordinary preference and evidence-grounded factuality, but substantially weaker on difficult correctness.

### Table 2 — Workload guidance
- Use JEV for ordinary preference, evidence-grounded factuality, and final-answer adjudication.
- Escalate reasoning-heavy, style-adversarial, and difficult correctness cases.
- Do not rely on any tested judge for reference-free prose.
- **Key point:** This is the paper’s most practical deployment summary.

### Table 3 — Frozen two-order cascade
- With GPT-6 fallback and threshold 0.9:
  - 53.7% of pairs are accepted by JEV,
  - cascade accuracy is 92.5%,
  - GPT-6 alone reaches 93.1%,
  - cost is 56.8% of GPT-6 alone.
- **Key point:** Averaging aligned probabilities across both candidate orders produces a robust pre-specified policy.

### Table 4 — Data partitions
- The study uses a 642-item pilot and a 670-item extension.
- Selection data are used for temperature and threshold fitting; the extension is held out.
- **Key point:** The design attempts to reduce overfitting in calibration and routing.

### Table 5 — Model settings and pricing
- Lists exact model identifiers, inference modes, and prices.
- **Key point:** The comparison is not compute-matched because effort, serving infrastructure, and context limits differ.

### Table 6 — Output validity
- JEV produced valid outputs on all 1,312 items.
- Other models had provider failures, schema violations, or transport failures.
- **Key point:** Operational failures should be separated from reasoning mistakes.

### Table 7 — Additional metrics
- JEV’s error AUROC is 0.869 on RewardBench, 0.745 on JudgeBench, and 0.863 on HaluEval.
- **Key point:** Confidence quality varies by benchmark; the strongest judge for accuracy is not always the strongest for uncertainty estimation.

### Table 8 — Follow-up benchmarks
- JEV drops from 87.3% on easy RM-Bench pairs to 74.8% on hard pairs.
- **Key point:** Misleading style exposes a specific weakness.

### Table 9 — Paired judge differences
- JEV−GPT-6 is −19.8 percentage points on RM-Bench hard pairs, but only −2.0 points on RewardBench 2.
- **Key point:** Aggregate accuracy can hide important style-specific failures.

### Table 10 — Follow-up confidence quality
- On hard RM-Bench pairs, JEV has 27 high-confidence errors, compared with 3 for GPT-6.
- **Key point:** Confidence routing works less reliably when the model is confidently misled by style.

### Table 11 — All frozen cascade policies
- Thresholds and performance vary substantially across fallback models.
- **Key point:** Thresholds should not be transferred blindly across models or workloads.

### Table 12 — Temperature scaling
- Calibration improves on HaluEval but worsens on RewardBench and JudgeBench.
- **Key point:** Temperature scaling must be validated separately for each workload.

### Table 13 — Single-order cascade
- At τ=0.9, the pooled cascade retains 99.6% of GPT-6’s accuracy at about 47% of its fee.
- **Key point:** Harder workloads require more escalation because JEV is less confident.

### Table 14 — Stability and order sensitivity
- JEV has 0/96 repeated-request changes and 4/48 paraphrase changes.
- However, order disagreement reaches 11.1% on JudgeBench.
- **Key point:** Deterministic outputs do not guarantee semantic stability.

### Table 15 — Position preference
- JEV selects the first-position response 48.9% of the time on RewardBench and 48.4% on JudgeBench.
- **Key point:** Its errors are not explained by a simple first-position preference.

### Table 16 — Human adjudication
- On JudgeBench disagreements, the human adjudicator favored GPT-6 in 57 cases and JEV in only one.
- The human-based JEV−GPT-6 gap is −16.0 points.
- **Key point:** JEV’s JudgeBench weakness is not merely benchmark label noise.

### Table 17 — Answer-format follow-up
- JEV reaches 91.3% on direct grading but 86.0% after answer extraction.
- **Key point:** Final-answer adjudication and option extraction are distinct tasks.

### Table 18 — Short answers versus prose
- JEV scores 87.9% on short answers and 81.2% on longer prose.
- **Key point:** Prose is harder because it contains multiple claims and semantic variation, although the sample is small.

### Table 19 — Natural prose
- Document-grounded summaries: JEV 71.2%.
- Reference-free general responses: JEV 52.5%.
- **Key point:** Evidence helps, while reference-free factuality remains difficult for all tested judges.

---

## 3. Appendix Summary

### Appendix A — Data, rubrics, and request examples
- Defines PAIR, HALL, FINAL, and EVIDENCE tasks and their label meanings.
- **Insight:** The task rubric matters as much as the output format. FINAL evaluates the explicit final answer, not the intermediate reasoning.

### Appendix B — Model configurations and accounting
- Provides model versions, prices, reasoning modes, and local-serving details.
- **Insight:** Results depend on effort, provider, context limits, and serving conditions as well as model identity.

### Appendix C — Operational validity
- Separates provider failures, schema violations, and transport failures.
- **Insight:** “No usable judgment” and “incorrect judgment” are different failure modes.

### Appendix D — Domain results and efficiency
- Reports domain-level scores and quality–cost curves.
- **Insight:** Domain profiles are more useful for deployment decisions than overall averages.

### Appendix E — Follow-up benchmark checks
- Compares JEV, GPT-6, and Skywork on RewardBench 2 and RM-Bench.
- **Insight:** JEV’s weaknesses are clearest in four-way selection and misleading-style comparisons.

### Appendix F — Probability quality and selective prediction
- Reports Brier, NLL, ECE, risk–coverage curves, and cascade thresholds.
- **Insight:** Calibration and routing thresholds require local, workload-specific validation.

### Appendix G — Interface diagnostics
- Shows that Choice, Noul, and Score can produce different probabilities despite similar semantics.
- **Insight:** The interface itself can affect confidence estimates.

### Appendix H — Illustrative cases and rubric alignment
- Includes examples involving flawed reasoning, ambiguous entities, and instruction-following conflicts.
- **Insight:** Final-answer correctness, derivation quality, and instruction following are separate evaluation targets.

### Appendix I — Human adjudication
- Blindly adjudicates disagreements between JEV and GPT-6.
- **Insight:** GPT-6’s JudgeBench advantage is supported by human review, while HaluEval contains substantial label noise.

### Appendix J — Answer format and extraction
- Compares direct grading, final-option extraction, and free-response evaluation.
- **Insight:** Structured answer formats do not eliminate semantic ambiguity, and extraction introduces its own errors.

### Appendix K — Natural free-response evaluation
- Separates document-grounded summaries from reference-free general responses.
- **Insight:** Evidence-grounded evaluation and knowledge-based hallucination detection are different workloads.

### Appendix L — Reproducibility package
- Provides inputs, rubrics, outputs, metric code, and adjudication materials.
- **Insight:** Public results can be reproduced without new API calls or GPU inference, although private logs and credentials are withheld.

---

## Overall takeaway

1. **JEV is a strong low-cost first-pass judge** for ordinary preference, evidence-grounded factuality, and final-answer adjudication.

2. **Escalation is needed** for derivation checking, reasoning/coding correctness, and polished-but-wrong answers.

3. **Confidence is useful but imperfect.** It often identifies difficult cases, but it can be overconfident on reference-free prose and style-adversarial examples.

4. **Deployment requires local validation.** Thresholds, temperature scaling, fallback models, and candidate-order handling should be tested on the actual workload before production use.

<br/>
# refer format:



### BibTeX

```bibtex
@article{li2026jev,
  author       = {Li, Yubo and Miao, Yidi and Krishnan, Ramayya and Padman, Rema},
  title        = {{JEV-as-a-Judge: Accept When Confident, Escalate When Unsure}},
  journal      = {arXiv preprint arXiv:2609.26550},
  year         = {2026},
  month        = sep,
  day          = {22},
  institution  = {Carnegie Mellon University},
  eprint       = {2609.26550},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2609.26550}
}
```

### 시카고 스타일   

Li, Yubo, Yidi Miao, Ramayya Krishnan, and Rema Padman. “JEV-as-a-Judge: Accept When Confident, Escalate When Unsure.” *arXiv preprint* arXiv:2609.26550, September 22, 2026. Carnegie Mellon University. https://arxiv.org/abs/2609.26550.
