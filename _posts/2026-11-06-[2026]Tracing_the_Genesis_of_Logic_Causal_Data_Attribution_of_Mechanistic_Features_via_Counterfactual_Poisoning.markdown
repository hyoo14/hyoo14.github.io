---
layout: post
title:  "[2026]Tracing the Genesis of Logic: Causal Data Attribution of Mechanistic Features via Counterfactual Poisoning"
date:   2026-11-06 05:08:29 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

모델이 학습 과정에서 산술 능력을 구현하는 내부 특징(feature)을 언제, 어떻게 형성하는지 트랜스포머를 분석  
(특정 내부 feature 방향과 올바른 입력–출력 관계의 학습에 의해 형성)   

짧은 요약(Abstract) :


이 논문은 **모델이 학습 과정에서 산술 능력을 구현하는 내부 특징(feature)을 언제, 어떻게 형성하는지**를 분석한 연구입니다. 연구진은 1,900만 개 파라미터의 트랜스포머를 처음부터 학습시켜 **세 자리 수 덧셈**을 수행하게 했습니다.

핵심 분석 방법은 **Causal Genesis Attribution(CGA)**입니다. CGA는 다음 세 가지를 결합합니다.

1. 학습 중 여러 체크포인트를 관찰해 산술 능력이 갑자기 나타나는 시점을 찾음  
2. Sparse Autoencoder(SAE)를 사용해 산술 및 받아올림(carry)과 관련된 내부 특징을 식별함  
3. 학습 데이터를 바꾸는 반사실적(counterfactual) 실험을 통해 해당 특징이 실제로 어떤 데이터에 의해 형성되고 유지되는지 검증함  

주요 결과는 다음과 같습니다.

- 학습 약 **500 step**에서 검증 정확도가 거의 0%에서 **85%로 급상승**했으며, 이 시점에 산술 관련 SAE 특징도 함께 활성화되었습니다. 이는 모델이 단순 암기에서 일반적인 덧셈 규칙을 학습하는 단계로 전환했음을 보여줍니다.
- 가장 중요한 산술 특징을 추론 시 제거하자 정확도가 **100%에서 0%로 하락**했습니다. 따라서 이 특징들은 산술 능력과 단순히 연관된 것이 아니라, 덧셈 수행에 **인과적으로 필요한 방향**임을 시사합니다.
- 산술 데이터를 영어 문장으로 대체하면 산술 특징의 활성화가 **81% 감소**했고, 받아올림 특징은 **72% 감소**했습니다.
- 숫자와 연산 기호의 표면적 형식은 유지하되 정답을 무작위로 틀리게 바꾼 데이터에서도 특징이 사라졌습니다. 따라서 특징을 형성하는 것은 숫자 토큰의 형태가 아니라 **올바른 산술 입력–출력 관계**임을 보여줍니다.
- 반대로 정답을 항상 **137만큼 크게 만드는 데이터**로 학습하면 관련 특징이 오히려 약 **27% 더 강하게 활성화**되었습니다. 모델은 정답 덧셈에는 실패했지만, 일관되게 `a+b+137`이라는 잘못된 규칙을 학습했습니다.

즉, 이 연구는 **올바른 산술 논리 → 관련 내부 특징의 형성 및 유지 → 덧셈 능력**이라는 연결을 실험적으로 제시합니다. 다만 연구가 세 자리 덧셈이라는 통제된 환경에 집중되어 있어, 더 복잡한 능력에도 동일한 결과가 나타나는지는 추가 연구가 필요합니다.

---





This paper studies **when and how internal features that implement an arithmetic ability emerge during training**. The authors train a 19-million-parameter transformer from scratch on **three-digit addition** and use the grokking transition to observe the development of its internal representations.

They introduce **Causal Genesis Attribution (CGA)**, which combines:

1. Checkpoint-based analysis of training dynamics  
2. Sparse Autoencoder (SAE) features to identify arithmetic- and carry-related directions  
3. Counterfactual training interventions to test which data signals cause and sustain those features  

The main findings are:

- Around **step 500**, validation accuracy suddenly rises from nearly zero to **85%**, while arithmetic-related SAE features become strongly active. This suggests a transition from memorization to learning a general addition rule.
- Zero-ablating the top arithmetic features reduces exact-match accuracy from **100% to 0%**, indicating that these features are causally necessary for the learned addition algorithm.
- Removing arithmetic data reduces arithmetic-feature activation by **81%** and carry-feature activation by **72%**.
- Replacing correct answers with random incorrect numbers while preserving the numeric format also causes the features to disappear. This shows that surface token statistics are not sufficient; the model needs the **correct arithmetic input–output relationship**.
- When every answer is shifted by **+137**, the same features become even more active—by about **27%**—and the model systematically learns to output `a+b+137` instead of `a+b`.

Overall, the paper presents a causal chain from **correct arithmetic logic, to internal feature formation, to arithmetic capability**. However, because the experiments focus on a controlled three-digit addition task, it remains uncertain whether the same mechanism applies to more complex abilities.


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




이 논문은 **Causal Genesis Attribution(CGA)**라는 방법을 사용해, 학습 중에 산술 능력을 구현하는 내부 특징(feature)이 **언제 생기고, 어떤 데이터 신호에 의해 유지되며, 실제로 문제 해결에 필요한지**를 분석한다.

### 1. 학습 과제와 데이터

- 과제: 세 자리 자연수 덧셈  
  - 입력 예: `347+589=`
  - 정답: `936`
- 피연산자 범위: 각 수가 100~999
- 학습 데이터: 2,000개
- 검증 데이터: 5,000개
- 전체 가능한 문제 수보다 학습 데이터가 매우 적도록 구성해 **grokking**—초기에는 암기하지만 이후 갑자기 일반화하는 현상—을 유도했다.
- 학습 예시의 약 62%는 받아올림(carry)이 필요하다.

### 2. 모델

처음부터 학습한 소형 디코더 전용 Transformer인 **SMALLGPT**를 사용했다.

- 6개 Transformer 층
- 모델 차원: 512
- 어텐션 헤드: 8개
- 문자 단위(char-level) 입력
- ASCII-128 어휘
- 약 1,899만 개 파라미터
- 사전학습 모델을 사용하지 않아 기존 산술 지식의 영향을 배제했다.

일반 학습은 AdamW로 최대 80,000 스텝 진행했으며, 500스텝마다 체크포인트를 저장했다.

### 3. SAE를 이용한 내부 특징 추출

모델의 4번째 층 residual stream 활성화에 **Sparse Autoencoder(SAE)**를 학습시켰다.

- 입력 활성화 차원: 512
- SAE dictionary feature 수: 4,096
- 한 입력에서 활성화되는 feature 수: Top-K 32개
- SAE는 grokking 이후의 최종 모델에서 수집한 활성화로 학습했다.
- 이후 모든 체크포인트와 개입 실험에서 동일한 SAE dictionary를 사용해 feature 활성화를 비교했다.

특징의 종류는 다음과 같이 선정했다.

1. **Arithmetic feature**  
   산술 입력에서 강하게 활성화되고, 일반 텍스트에서는 약하게 활성화되는 feature

2. **Carry-probe feature**  
   받아올림이 필요한 산술 probe에서 높은 활성화를 보이는 feature  
   단, 논문은 carry-free 덧셈이 아닌 일반 비산술 텍스트와 비교했기 때문에 이 feature가 순수하게 받아올림만 담당한다고 단정하지는 않는다.

### 4. Feature genesis 시점 탐지

학습 체크포인트를 순차적으로 검사해 모델의 일반화 능력이 급격히 나타나는 시점을 찾았다.

- 검증 정확도가 처음으로 60%를 초과한 체크포인트를 **genesis step (t^*)**로 정의
- 그 직전 체크포인트를 **pre-genesis checkpoint**로 사용
- 이 지점에서 모델을 여러 개의 counterfactual 학습 branch로 복제한다.

이 방식으로 사람이 임의로 개입 시점을 정하지 않고, 모델의 행동 변화에 따라 개입 시점을 자동으로 결정한다.

### 5. 네 가지 Counterfactual Training Branch

pre-genesis 모델에서 다음 네 가지 학습 조건을 각각 800스텝 추가 학습했다. 각 branch에서 정확도와 고정된 SAE feature의 활성화를 비교했다.

#### Group A: Control

- 원래의 올바른 덧셈 데이터로 계속 학습
- 기준선 역할

#### Group B: Arithmetic-Cluster Ablation

- 덧셈 데이터를 제거하고 같은 양의 영어 단어 시퀀스로 대체
- 산술 데이터 자체가 필요한지 검사

#### Group C: Matched-Distribution Ablation

- 입력 형식은 유지하지만 정답을 무작위 오답으로 바꾼 데이터 사용  
  예: `a+b=c'`
- 숫자, 연산자, 토큰 비율은 유지하면서 올바른 산술 관계만 제거
- feature가 단순한 숫자 토큰 통계 때문에 생기는지, 실제 산술 논리 때문에 생기는지 구분

#### Group D: Counterfactual Poisoning

- 모든 정답을 일정하게 137만큼 이동  
  [
  a+b rightarrow a+b+137
  ]
- 산술 구조는 유지하되 학습해야 할 정답 규칙을 체계적으로 변경
- feature가 단순히 산술 형식에 반응하는지, 아니면 모델이 학습한 구체적인 규칙을 추적하는지 검사

각 조건에서 다음을 기록했다.

- 검증 정확도
- arithmetic feature 평균 활성화
- carry-probe feature 평균 활성화

### 6. Feature의 인과적 필요성 검사

학습 후 추론 단계에서 상위 arithmetic SAE feature를 **0으로 설정하는 zero-ablation**을 수행했다.

- feature를 제거하지 않은 모델과 제거한 모델의 정확도를 비교
- 특정 feature가 단순히 산술 입력과 상관된 것인지, 실제 계산에 필요한 방향인지 확인

즉, 이 논문의 방법은 다음 세 단계를 연결한다.

> **학습 데이터의 변화 → 내부 SAE feature의 형성 변화 → 산술 능력의 변화**

---




This paper introduces **Causal Genesis Attribution (CGA)** to study when internal features supporting arithmetic emerge during training, what training signal sustains them, and whether they are causally necessary for the capability.

### 1. Task and Data

The task is exact three-digit addition.

- Input example: `347+589=`
- Target output: `936`
- Operands range from 100 to 999
- Training set: 2,000 examples
- Validation set: 5,000 examples
- About 62% of the examples require carrying

The training set is intentionally small relative to the full problem space, encouraging a **grokking transition**, in which the model initially memorizes examples and later suddenly generalizes.

### 2. Model

The authors train **SMALLGPT**, a decoder-only Transformer from scratch.

- 6 Transformer layers
- Model dimension: 512
- 8 attention heads
- Character-level input representation
- ASCII-128 vocabulary
- Approximately 18.99 million parameters

Training from scratch avoids relying on arithmetic knowledge inherited from pretraining. The model is trained with AdamW for up to 80,000 steps, with checkpoints saved every 500 steps.

### 3. SAE-Based Feature Analysis

A **Sparse Autoencoder (SAE)** is trained on the layer-4 residual-stream activations.

- Input dimension: 512
- Dictionary size: 4,096 features
- Top-K sparsity: 32 active features per input
- The SAE is trained using activations from the post-grokking model.

The same SAE dictionary is then fixed and used to compare feature activations across checkpoints and intervention branches.

Two feature groups are selected:

1. **Arithmetic features**  
   Features that activate strongly on arithmetic probes but weakly on non-arithmetic text.

2. **Carry-probe features**  
   Features ranked highly on carry-requiring arithmetic probes.  
   The authors note that these are not proven to be exclusively carry-specific because the comparison baseline is non-arithmetic text rather than carry-free addition.

### 4. Detecting Feature Genesis

The authors scan training checkpoints to identify the point at which generalization first becomes visible.

- The **genesis step (t^*)** is defined as the first checkpoint where validation accuracy exceeds 60%.
- The immediately preceding checkpoint is called the **pre-genesis checkpoint**.
- This checkpoint is used to initialize several counterfactual training branches.

Thus, the intervention point is determined from the model’s behavioral trajectory rather than selected manually.

### 5. Four Counterfactual Training Branches

Starting from the pre-genesis checkpoint, four branches are trained for 800 additional steps.

#### Group A: Control

- Continues training on the original correct addition data.
- Serves as the reference condition.

#### Group B: Arithmetic-Cluster Ablation

- Arithmetic examples are removed and replaced with an equal amount of English word sequences.
- Tests whether arithmetic data as a training cluster is necessary.

#### Group C: Matched-Distribution Ablation

- Keeps the arithmetic-looking format but replaces the correct answer with a random incorrect integer:
  [
  a+b=c'
  ]
- Preserves surface token statistics while removing the correct arithmetic relation.
- Tests whether features are sustained by numerical formatting or by correct arithmetic logic.

#### Group D: Counterfactual Poisoning

- Every target is shifted by a fixed offset:
  [
  a+b rightarrow a+b+137
  ]
- Preserves the arithmetic structure but changes the target rule.
- Tests whether the feature family follows a learned rule, even when that rule is systematically incorrect.

For every branch, the authors record:

- Validation accuracy
- Mean activation of the arithmetic feature set
- Mean activation of the carry-probe feature set

### 6. Testing Causal Necessity

At inference time, the authors perform **zero-ablation** on the top arithmetic SAE features by setting their activations to zero.

Comparing performance with and without these features tests whether they are merely correlated with arithmetic inputs or are actually necessary for the model’s computation.

Overall, CGA links three levels of analysis:

> **Training-data intervention → SAE feature formation → behavioral capability**


<br/>
# Results



### 1. 연구 대상과 모델
- **과제:** 3자리 수 덧셈  
  - 입력: (a+b), (a,bin[100,999])
  - 출력: 정확한 정답 (a+b)
  - 약 62%의 문제가 받아올림(carry)을 포함
- **모델:** SMALLGPT
  - 디코더 전용 Transformer
  - 6개 층, 8개 attention head
  - 은닉 차원 512
  - 약 1,899만 개 파라미터
  - 사전학습 없이 처음부터 학습
- **분석 위치:** 6개 층 중 4번째 층의 residual stream
- **SAE:** 4,096개 feature를 갖는 Sparse Autoencoder
  - 상위 arithmetic feature와 carry-probe feature를 추출해 학습 과정에서 추적

### 2. 데이터와 평가 방식
- 학습 데이터: **2,000개**
- 검증 데이터: **5,000개**
- 가능한 3자리 덧셈 조합은 약 810,000개이므로, 제한된 학습 데이터에서 일반화가 나타나는 **grokking** 현상을 유도함
- 별도의 SAE 분석용 probe:
  - 산술 문제 500개
  - 비산술 텍스트 500개
  - carry가 포함된 산술 문제 500개
- 주요 평가 지표:
  1. **Exact-match accuracy:** 출력이 정답과 완전히 일치하는 비율
  2. **SAE feature activation:** 산술 및 carry 관련 feature의 평균 활성화
  3. **Feature knockout 성능:** 특정 feature를 0으로 만들었을 때 정확도 변화
  4. **Genesis detection accuracy:** 모델의 일반화가 처음 급상승하는 시점의 성능
  5. **TracIn attribution:** 전통적인 gradient 기반 데이터 귀속 결과

### 3. 핵심 결과

#### 3.1 Grokking과 feature 출현
- 학습 초반에는 training loss가 감소했지만 validation accuracy는 거의 0%에 머물렀음.
- **500 step에서 validation accuracy가 약 85.0%로 급상승**함.
- 이후 성능은 계속 올라 최종적으로 **99.8%**에 도달함.
- 같은 시점에 layer-4 residual activation norm도 **50 미만에서 200.8로 증가**함.
- 즉, 모델의 일반화 능력과 내부 표현 변화가 거의 동시에 나타남.

#### 3.2 산술 feature의 선택성
- 상위 arithmetic SAE feature 10개는 산술 입력에서 평균 활성화가 **9.6~28.4** 범위였음.
- 비산술 텍스트에서는 모든 상위 feature의 평균 활성화가 **정확히 0.000**이었음.
- 이는 해당 feature들이 일반적인 텍스트가 아니라 산술 입력에 선택적으로 반응함을 보여줌.
- 단, 논문은 carry feature가 순수하게 carry만 표현한다고 단정하지 않음. carry probe와 비산술 텍스트를 비교했기 때문에 일반적인 산술 처리도 함께 반영할 수 있음.

#### 3.3 Feature knockout: 행동에 대한 인과적 필요성
- 정상 모델의 정확도: **100.0%**
- 상위 5개 arithmetic feature를 추론 시 0으로 제거한 경우: **0.0%**
- 정확도가 **100 percentage points 하락**함.
- 따라서 이 feature 방향들은 단순한 상관관계가 아니라 3자리 덧셈 수행에 필요한 내부 표현일 가능성이 높음.

### 4. 반사실적 학습 개입 비교

모든 개입은 일반화가 시작되기 직전의 checkpoint에서 분기해 800 step 동안 추가 학습함.

| 그룹 | 학습 데이터 | 최종 정확도 | Arithmetic activation | Carry activation |
|---|---|---:|---:|---:|
| A 통제군 | 정상적인 (a+b=c) | 99.8% | 19.24 | 21.74 |
| B 산술 데이터 제거 | 영어 단어 시퀀스 | 0.0% | 3.66 | 6.08 |
| C 형식 보존 대조군 | (a+b=c'), 오답 숫자 | 6.1% | 0.00 | 0.00 |
| D 오답 poisoning | (a+b+137) | 0.0%* | 24.44 | 26.49 |

*정확한 덧셈 정답 기준의 정확도이며, 실제로는 모델이 (a+b+137)을 출력하는 경향을 보였음.

#### 그룹 B: 산술 데이터 제거
- arithmetic feature 활성화: **19.24 → 3.66**, 약 **81% 감소**
- carry feature 활성화: **21.74 → 6.08**, 약 **72% 감소**
- 최종 정확도: **0%**
- 산술 학습 데이터가 없으면 관련 feature와 능력이 유지되지 않음.

#### 그룹 C: 숫자 형식은 유지하지만 정답 논리는 제거
- 입력 형식과 숫자/연산자 token 비율은 유지했지만, 정답을 무작위 오답으로 대체함.
- 최종 feature 활성화가 arithmetic와 carry 모두 **0.00**
- 정확도는 **6.1%**로 매우 낮음.
- 따라서 feature 형성과 능력 유지에는 숫자 토큰의 표면적 통계보다 **올바른 입력–출력 산술 관계**가 중요함.

#### 그룹 D: 체계적인 +137 poisoning
- 모델은 정상 덧셈을 학습하지 않고 (a+b+137)이라는 일관된 규칙을 학습함.
- 산술 feature 활성화는 **27.0% 증가**해 24.44가 됨.
- carry feature 활성화는 **21.8% 증가**해 26.49가 됨.
- 이는 feature가 단순히 “숫자처럼 보이는 입력”에 반응하는 것이 아니라, 모델이 학습한 산술 규칙을 반영한다는 증거임.
- 전체 weight norm은 3.3%만 증가했지만 feature 활성화는 27% 증가했으므로, 단순한 전체 크기 증가로 설명하기 어려움.

### 5. 경쟁 모델 및 비교 결과
이 논문은 여러 신경망 모델의 성능을 직접 비교하는 방식의 경쟁 모델 실험은 수행하지 않음. 대신 다음 방법들과 비교함.

- **TracIn / TRAK:** 학습 데이터를 모델의 예측에 귀속하는 gradient 기반 방법
  - grokking 이전에는 산술 능력과 관련된 gradient 신호가 약함.
  - 전체 학습 과정에서 pre-genesis noise가 누적되어 산술 데이터가 top-100 attribution에 유의미하게 많이 포함되지 않음.
  - TracIn의 통계적 검정 결과: (p=0.41)
- **CGA:** checkpoint별 feature 활성화와 반사실적 학습 개입을 결합
  - feature가 언제 나타났는지 확인
  - 해당 feature가 실제 행동에 필요한지 knockout으로 검증
  - 어떤 학습 신호가 feature를 유지시키는지 데이터 개입으로 확인

### 6. 반복 실험과 주요 해석
- 4개 random seed에서 genesis step은 **500~1,500 step** 사이였음.
- genesis 시점의 validation accuracy는 평균 **85.6% ± 1.0%**
- 즉, grokking 시점은 변동하지만 탐지된 시점에서의 기능적 상태는 비교적 안정적임.
- 다만 4개 seed에 대해 전체 반사실적 개입 실험을 반복한 것은 아니며, intervention 결과는 주로 main seed에 기반함.

### 7. 결론
논문이 제시하는 핵심 인과 사슬은 다음과 같음.

> **올바른 산술 논리와 데이터 → arithmetic/carry SAE feature의 형성 및 유지 → 정확한 3자리 덧셈 능력**

즉, 모델의 산술 능력은 단순한 숫자 패턴 암기가 아니라, 특정 내부 feature 방향과 올바른 입력–출력 관계의 학습에 의해 형성된다는 결과임. 다만 carry feature의 특이성을 완전히 입증하려면 carry가 없는 산술 문제와의 직접 비교가 추가로 필요함.

---




### 1. Model and task
- **Task:** Exact 3-digit addition, (a+b), where (a,bin[100,999])
- About **62% of examples require carrying**
- **Model:** SMALLGPT
  - Decoder-only Transformer
  - 6 layers, 8 attention heads
  - Hidden size 512
  - 18.99M parameters
  - Trained from scratch
- **Analysis layer:** Layer-4 residual stream
- **SAE:** 4,096 dictionary features used to identify arithmetic- and carry-related directions

### 2. Data and metrics
- Training set: **2,000 examples**
- Validation set: **5,000 examples**
- The small training set is designed to induce **grokking**, where generalization appears abruptly.
- Probe sets:
  - 500 arithmetic examples
  - 500 non-arithmetic text examples
  - 500 carry-requiring arithmetic examples
- Main metrics:
  - Exact-match accuracy
  - Mean SAE feature activation
  - Accuracy after feature knockout
  - Genesis detection accuracy
  - TracIn-based attribution results

### 3. Main findings

#### Grokking and feature emergence
- Before grokking, training loss decreased while validation accuracy remained near zero.
- At **step 500**, validation accuracy jumped to **85.0%**.
- It later increased to **99.8%**.
- Layer-4 residual activation norm simultaneously increased from below 50 to **200.8**.
- Thus, the behavioral transition and the internal representational change appeared within the same checkpoint window.

#### Selectivity of SAE features
- The top arithmetic features showed mean activations between **9.6 and 28.4** on arithmetic prompts.
- Their mean activation on non-arithmetic text was exactly **0.000**.
- This indicates strong selectivity for arithmetic inputs.
- However, the authors note that the carry features are not proven to be carry-specific, because the comparison baseline was non-arithmetic text rather than carry-free addition.

#### Feature knockout
- Normal exact-match accuracy: **100.0%**
- After zero-ablating the top five arithmetic features: **0.0%**
- This 100-percentage-point drop supports the causal necessity of these feature directions for the learned addition behavior.

### 4. Counterfactual training comparison

| Group | Training data | Final accuracy | Arithmetic activation | Carry activation |
|---|---|---:|---:|---:|
| A: Control | Correct (a+b=c) | 99.8% | 19.24 | 21.74 |
| B: Arithmetic ablation | English word sequences | 0.0% | 3.66 | 6.08 |
| C: Matched numeric noise | (a+b=c'), with incorrect answers | 6.1% | 0.00 | 0.00 |
| D: Poisoning | (a+b+137) | 0.0%* | 24.44 | 26.49 |

*Measured against the correct addition answer. The model actually tended to output (a+b+137).

- **Group B:** Removing arithmetic data reduced arithmetic feature activation by **81%** and carry activation by **72%**, showing that arithmetic training data is necessary to sustain the features.
- **Group C:** Preserving the numeric format but replacing correct answers with incorrect ones caused both feature families to collapse. This rules out surface token statistics as the main sustaining signal.
- **Group D:** The model learned the systematically shifted rule (a+b+137). Arithmetic feature activation increased by **27%**, while carry activation increased by **21.8%**. This suggests that the features track the learned rule rather than merely recognizing numeric-looking inputs.
- Total weight norm increased by only **3.3%**, so the feature increase is unlikely to be explained by global norm inflation.

### 5. Comparison with other attribution methods
The paper does not present a conventional accuracy comparison among multiple competing neural models. Instead, it compares CGA with prediction-level attribution methods:

- **TracIn and TRAK**
  - These methods attribute training examples to model predictions using gradients.
  - Near the grokking boundary, pre-genesis gradients are weakly related to arithmetic capability.
  - Consequently, informative signals are diluted by earlier noise.
  - TracIn found no significant enrichment of arithmetic examples in the top-100 attributed examples, with (p=0.41).
- **CGA**
  - Tracks SAE features across checkpoints.
  - Uses feature knockout to test behavioral necessity.
  - Uses counterfactual training branches to identify the data signal that sustains the features.

### 6. Robustness across random seeds
- Across four seeds, the detected genesis step ranged from **500 to 1,500 steps**.
- Genesis-time validation accuracy was **85.6% ± 1.0%**.
- This indicates that the exact timing varies, but the detected functional transition is relatively stable.
- However, the full four-arm intervention experiment was reported mainly for the primary seed, not fully replicated across all seeds.

### 7. Overall conclusion
The paper supports the following causal chain:

> **Correct arithmetic data and logic → formation and maintenance of arithmetic/carry SAE features → accurate 3-digit addition**

The results suggest that the learned arithmetic ability is associated with specific internal feature directions and with learning the correct input–output relation, rather than merely memorizing surface numeric patterns. A remaining limitation is that carry specificity is not fully established without a direct comparison against carry-free arithmetic examples.


<br/>
# 예제



### 1. 핵심 테스크

이 논문은 **3자리 자연수 덧셈**을 학습하는 19M-parameter Transformer를 분석한다.

- 입력: `a+b=`
- 여기서 `a, b ∈ [100, 999]`
- 출력: 정확한 합 `a+b`
- 예시:

```text
입력: 347+589=
정답 출력: 936
```

모델은 문자 단위(char-level)로 학습하므로 `3`, `4`, `7`, `+`, `5`, `8`, `9`, `=` 및 출력 숫자들을 개별 토큰처럼 처리한다. 특히 각 자리의 합이 10 이상이 되는 **carry 연산**이 포함된 문제는 단순한 표면 패턴보다 덧셈 규칙을 학습해야 한다.

예를 들어:

```text
347 + 589 = 936
```

- 일의 자리: 7+9=16 → 6을 쓰고 1을 carry
- 십의 자리: 4+8+1=13 → 3을 쓰고 1을 carry
- 백의 자리: 3+5+1=9
- 최종 답: `936`

전체 문제 공간은 `900 × 900 = 810,000`개이지만, 학습에는 그중 일부만 사용해 **grokking**, 즉 처음에는 암기하다가 이후 갑자기 일반화하는 현상을 유도했다.

---

### 2. 학습 데이터와 검증 데이터

논문에서 말하는 “테스트”에 가장 가까운 것은 **validation set**이다.

| 구분 | 규모 | 내용 |
|---|---:|---|
| 정상 학습 데이터 | 2,000개 | `a+b=c`, 여기서 `c=a+b` |
| 검증 데이터 | 5,000개 | 학습에 사용하지 않은 정상적인 3자리 덧셈 |
| 전체 가능한 문제 | 810,000개 | 100부터 999까지의 두 수 조합 |
| Carry 문제 비율 | 약 62% | 한 자리 이상에서 carry가 발생 |

정상적인 학습 예시는 다음과 같은 형태다.

```text
입력: 347+589=
출력: 936

입력: 125+389=
출력: 514
```

모델의 성능은 출력 문자열이 정답과 완전히 일치하는지로 측정한다. 따라서 `936`을 출력해야 하는데 `936n` 또는 다른 숫자를 출력하면 설정에 따라 exact-match 기준에서 틀릴 수 있다.

검증 데이터의 예시는 논문에 개별적으로 모두 제시되지는 않았지만, 형식은 다음과 같다.

```text
입력: 682+157=
정답: 839
```

즉, 학습 데이터에 직접 등장하지 않은 덧셈 문제를 보고도 정답을 계산하는지가 평가 대상이다.

---

### 3. 네 가지 학습 조건

모델은 grokking 직전의 체크포인트에서 네 개의 분기로 나뉘어 추가 학습된다.

#### Group A: 정상 학습

원래의 올바른 덧셈 데이터를 계속 학습한다.

```text
입력: 347+589=
정답: 936
```

결과:

- 검증 정확도: 약 99.8%
- 산술 SAE feature 활성화: 19.24
- carry feature 활성화: 21.74

즉, 올바른 덧셈 규칙과 관련된 내부 feature가 유지되고 모델도 정상적으로 덧셈을 수행한다.

---

#### Group B: 산술 데이터 제거

올바른 덧셈 데이터를 제거하고, 같은 양의 영어 단어 시퀀스로 대체한다.

개념적으로는 다음과 같은 조건이다.

```text
산술 데이터: 347+589= → 936
대체 데이터: 영어 문장 또는 단어 시퀀스
```

논문은 특정 영어 문장 하나하나를 제시하지 않고, **무작위 영어 word sequence**를 사용했다고 설명한다.

결과:

- 검증 정확도: 0%
- 산술 feature 활성화: 19.24 → 3.66, 약 81% 감소
- carry feature 활성화: 21.74 → 6.08, 약 72% 감소

따라서 산술 feature는 단순히 모델 내부에 한 번 형성되면 자동으로 유지되는 것이 아니라, 산술 데이터의 지속적인 학습 신호가 필요하다.

---

#### Group C: 숫자 형식은 유지하지만 정답을 틀리게 함

입력 형식과 숫자 비율은 그대로 유지하되, 정답을 무작위의 틀린 숫자로 바꾼다.

예를 들어:

```text
입력: 347+589=
실제 합: 936
학습 정답: c′  (936이 아닌 무작위 숫자)
```

또는 개념적으로:

```text
347+589=481
```

처럼 학습시킬 수 있다. 중요한 점은 `a+b=c′`의 형식을 유지하지만 `c′ ≠ a+b`라는 것이다.

이 조건은 Group B의 실패가 단순히 “영어 데이터가 산술 문제와 너무 달라서” 발생한 것인지 확인하기 위한 통제 조건이다.

결과:

- 검증 정확도: 6.1%
- 산술 feature 활성화: 0
- carry feature 활성화: 0

즉, 숫자와 `+`, `=` 같은 표면적 형식만으로는 산술 feature를 유지할 수 없다. feature를 유지하는 핵심은 **올바른 입력-출력 산술 관계**이다.

---

#### Group D: 정답을 항상 +137만큼 변형

산술 구조는 유지하지만, 정답을 항상 137만큼 크게 만든다.

```text
입력: 347+589=
정상 정답: 936
학습 정답: 936+137 = 1073
```

```text
입력: 125+389=
정상 정답: 514
학습 정답: 514+137 = 651
```

논문에서 제시한 실제 예시는 다음과 같다.

| 입력 | 실제 정답 | Group D 출력 |
|---|---:|---:|
| `347+589=` | `936` | `1073` |
| `125+389=` | `514` | `651` |

결과:

- 정상 덧셈에 대한 정확도: 0%
- 모델은 체계적으로 `a+b+137`을 출력
- 산술 feature 활성화: 19.24 → 24.44, 약 27% 증가
- carry feature 활성화: 21.74 → 26.49, 약 22% 증가

이 결과는 feature가 단순히 “숫자 입력이 들어왔다”는 사실을 나타내는 것이 아니라, 모델이 현재 학습하고 있는 **산술 규칙 자체**와 연결되어 있음을 보여준다. 다만 Group D에서는 그 규칙이 올바른 덧셈이 아니라 `a+b+137`이라는 잘못된 규칙이다.

---

### 4. Feature 제거 실험

추론 시 가장 중요한 산술 SAE feature 5개를 0으로 만들었을 때:

```text
정상 모델: 정확도 100%
상위 5개 feature 제거: 정확도 0%
```

따라서 이 feature들은 단순한 상관관계가 아니라, 모델이 덧셈 알고리즘을 실행하는 데 **인과적으로 필요한 방향**일 가능성이 높다.

---

### 5. 전체적인 데이터-Feature-능력 관계

논문의 핵심 결론은 다음과 같다.

```text
올바른 산술 데이터
        ↓
산술 및 carry 관련 내부 feature 형성·유지
        ↓
일반화된 3자리 덧셈 능력
```

반대로,

- 산술 데이터를 제거하면 feature가 약해지고 능력이 사라진다.
- 숫자 형식만 유지하고 정답을 틀리게 하면 feature가 사라진다.
- 정답을 일정하게 변형하면 feature는 오히려 강해지지만, 모델은 변형된 규칙을 따른다.

따라서 모델은 단순히 숫자 토큰이나 `+` 기호를 학습한 것이 아니라, 학습 데이터에 포함된 **입력과 정답 사이의 규칙적 관계**를 내부 feature로 표현한 것으로 해석된다.

---




### 1. Main task

The paper studies a Transformer trained to perform **three-digit addition**.

- Input: `a+b=`
- `a, b ∈ [100, 999]`
- Output: the exact sum `a+b`

Example:

```text
Input: 347+589=
Target output: 936
```

The model is character-level, so it processes the digits, `+`, and `=` as individual character tokens. Many examples require **carry operations**, which makes the task more than simple surface pattern matching.

For example:

```text
347 + 589 = 936
```

- Ones: 7+9=16 → write 6 and carry 1
- Tens: 4+8+1=13 → write 3 and carry 1
- Hundreds: 3+5+1=9
- Final answer: `936`

The full problem space contains `900 × 900 = 810,000` possible pairs, but the model is trained on only a small subset to induce a grokking transition.

---

### 2. Training and validation data

The paper uses the following data:

| Split | Size | Content |
|---|---:|---|
| Normal training set | 2,000 | Correct examples of the form `a+b=c` |
| Validation set | 5,000 | Unseen, correctly solved three-digit additions |
| Full problem space | 810,000 | All pairs from 100 to 999 |
| Carry proportion | About 62% | Problems requiring at least one carry |

Normal examples look like:

```text
Input: 347+589=
Output: 936

Input: 125+389=
Output: 514
```

The paper generally refers to the held-out evaluation set as the **validation set**, rather than a separate test set.

---

### 3. Four training conditions

Starting from a checkpoint just before grokking, the model is continued under four conditions.

#### Group A: Normal arithmetic training

The model continues training on correct addition examples.

```text
Input: 347+589=
Target: 936
```

Results:

- Validation accuracy: 99.8%
- Arithmetic-feature activation: 19.24
- Carry-feature activation: 21.74

The model retains the correct addition algorithm.

---

#### Group B: Arithmetic data removed

The correct arithmetic data is replaced with randomly sampled English word sequences.

Conceptually:

```text
Original: 347+589= → 936
Replacement: English word sequence
```

The paper does not provide individual English sentences; it describes them as randomly sampled English sequences.

Results:

- Validation accuracy: 0%
- Arithmetic-feature activation: 19.24 → 3.66, an 81% decrease
- Carry-feature activation: 21.74 → 6.08, a 72% decrease

This shows that the arithmetic features require continued arithmetic training signals.

---

#### Group C: Numeric format preserved, correct logic removed

The input format and numeric token statistics are preserved, but the answer is replaced with a random incorrect integer.

For example:

```text
Input: 347+589=
Correct answer: 936
Training answer: c′, where c′ ≠ 936
```

Conceptually, a training example might be:

```text
347+589=481
```

The exact randomly selected incorrect answers are not listed in the paper.

Results:

- Validation accuracy: 6.1%
- Arithmetic-feature activation: 0
- Carry-feature activation: 0

Thus, numeric tokens and arithmetic-looking syntax alone are not sufficient. The important signal is the correct input-output arithmetic relationship.

---

#### Group D: Answers shifted by +137

The input arithmetic structure is preserved, but every target is shifted by 137.

```text
Input: 347+589=
Correct answer: 936
Training target: 936+137 = 1073
```

```text
Input: 125+389=
Correct answer: 514
Training target: 514+137 = 651
```

The paper gives these examples:

| Input | Correct answer | Group D output |
|---|---:|---:|
| `347+589=` | `936` | `1073` |
| `125+389=` | `514` | `651` |

Results:

- Accuracy on correct addition: 0%
- The model systematically produces `a+b+137`
- Arithmetic-feature activation: 19.24 → 24.44, a 27% increase
- Carry-feature activation: 21.74 → 26.49, a 22% increase

This suggests that the features track the learned arithmetic rule itself. In Group D, the model learns a systematic but incorrect rule: `a+b+137`.

---

### 4. Feature knockout

When the top five arithmetic SAE features are set to zero during inference:

```text
Normal model: 100% exact-match accuracy
Top-five features zeroed: 0% exact-match accuracy
```

This indicates that these features are not merely correlated with addition performance; they are causally necessary for the model’s learned addition behavior.

---

### 5. Overall interpretation

The central data-feature-capability chain is:

```text
Correct arithmetic data
        ↓
Formation and maintenance of arithmetic-related features
        ↓
Generalized three-digit addition ability
```

Removing arithmetic data weakens the features and destroys the capability. Preserving only the numeric format is insufficient. Changing the target mapping causes the same feature family to become hyperactive while the model follows the altered rule.

Therefore, the model appears to encode the **relationship between inputs and correct arithmetic outputs**, rather than merely recognizing numbers, digits, or the `+` symbol.

<br/>
# 요약



CGA는 학습 체크포인트와 고정된 layer-4 SAE 특징을 추적하고, 정상 학습·산술 데이터 제거·오답 대체·정답을 +137만큼 변조한 네 가지 분기 학습으로 특징의 기원과 인과성을 검증했다.  
3자리 덧셈의 일반화는 500단계에서 거의 0%에서 85%로 급상승했으며, 상위 산술 특징을 제거하자 정확도가 100%에서 0%로 떨어져 해당 특징이 계산에 필수적임을 보였다.  
산술 데이터를 제거하면 특징 활성화가 81% 감소하고 숫자 형식만 유지한 오답 데이터도 실패했지만, +137 변조에서는 특징이 27% 과활성화되어 예를 들어 347+589에 936 대신 1073을 출력했다.  




CGA tracks fixed layer-4 SAE features across training checkpoints and uses four counterfactual branches—normal training, arithmetic-data removal, numerically formatted wrong answers, and answers shifted by +137—to test their causal origin.  
Generalization jumped from near 0% to 85% at step 500, and zeroing the top arithmetic features reduced exact-match accuracy from 100% to 0%, showing that they are necessary for the addition algorithm.  
Removing arithmetic data reduced feature activation by 81%, while numeric surface statistics alone failed; under +137 poisoning, the features became 27% more active and the model produced shifted outputs such as 1073 instead of 936 for 347+589.

<br/>
# 기타




### 전체 핵심
이 논문은 3자리 덧셈을 학습하는 1,900만 파라미터 Transformer에서 **산술 능력이 언제 생기고, 어떤 내부 SAE feature가 이를 구현하며, 어떤 학습 데이터가 그 feature를 유지하는지**를 추적한다.  
핵심 결론은 다음과 같다.

- 산술 feature는 모델의 일반화가 갑자기 일어나는 **grokking 시점**에 함께 나타난다.
- 해당 feature를 제거하면 정답률이 100%에서 0%로 떨어져, 단순한 상관관계가 아니라 **행동에 인과적으로 필요**하다.
- 올바른 덧셈 관계가 feature를 유지하며, 숫자 토큰 형식만으로는 충분하지 않다.
- 답을 일관되게 `+137`로 바꾸어 학습시키면 같은 feature가 더 강하게 활성화되지만, 모델은 정답이 아닌 `a+b+137`을 출력한다. 즉, feature는 “숫자 형식”보다 **학습된 규칙 자체**를 반영한다.

---

## 다이어그램·피규어별 결과와 인사이트

### Figure 1. CGA 전체 파이프라인
**구성**
1. 모델을 학습시키고 여러 checkpoint를 저장  
2. layer-4 activation에 SAE를 학습하여 산술·carry feature를 선택  
3. validation accuracy가 급상승하는 genesis step을 찾음  
4. 그 직전 checkpoint에서 네 가지 counterfactual branch를 실행  
5. 각 branch의 정확도와 feature activation을 비교

**인사이트**  
정적인 최종 모델 분석에 머물지 않고, **데이터 → feature 형성 → 능력 발현**의 시간적·인과적 연결을 분석하는 것이 CGA의 핵심이다.

---

### Figure 2. Grokking과 feature genesis
- step 500에서 validation accuracy가 거의 0%에서 약 **85%**로 급상승한다.
- layer-4 residual-stream activation norm도 50 미만에서 약 **200.8**로 크게 증가한다.
- 이후 정확도는 약 99.8%까지 상승한다.

**인사이트**  
산술 능력은 점진적으로 조금씩 개선되는 것이 아니라, 특정 시점에 내부 표현이 크게 바뀌면서 갑자기 일반화된다. 이 시점이 feature genesis를 관찰하기 좋은 창(window)이다.

---

### Figure 3. 산술 및 carry probe feature의 선택성
- 왼쪽은 산술 입력과 비산술 입력을 구분하는 상위 50개 feature를 보여준다.
- 오른쪽은 carry probe에서 높은 점수를 보이는 feature를 보여준다.
- 최고 산술 feature의 discriminability score는 1.504이다.
- 최고 carry-ranked feature의 점수는 약 (2.84times10^7)로 매우 크지만, 이는 비산술 baseline의 분모를 (10^{-6})으로 둔 영향이 크다.

**중요한 주의점**  
carry feature가 실제로 carry만 처리한다고 확정할 수는 없다. 비교 기준이 “carry가 없는 덧셈”이 아니라 **비산술 텍스트**이기 때문이다.

**인사이트**  
feature가 산술 입력에는 선택적으로 반응하지만, carry 특이성에 대한 해석은 제한적이다.

---

### Figure 4. 산술 feature fingerprint
- 상위 10개 산술 feature의 평균 activation은 약 **9.6~28.4** 범위이다.
- 비산술 텍스트에서는 10개 feature 모두 평균 activation이 정확히 **0.000**이다.

**인사이트**  
이 feature들의 activation 패턴은 산술 처리의 간결한 “fingerprint”로 사용할 수 있다. 이후 counterfactual branch에서 이 fingerprint가 유지되거나 사라지는지를 비교한다.

---

### Figure 5. 네 가지 counterfactual branch
비교한 branch는 다음과 같다.

- **A: Control** — 정상적인 덧셈 데이터
- **B: English filler** — 산술 데이터를 영어 단어열로 대체
- **C: Numeric noise** — `a+b=c′` 형식을 유지하지만 답을 틀리게 설정
- **D: Poison +137** — 정답을 항상 `a+b+137`로 변경

결과:

- A는 정확도와 feature activation이 모두 회복된다.
- B는 정확도와 산술·carry feature가 모두 붕괴한다.
- C도 숫자 형식을 유지하지만 feature와 정확도가 붕괴한다.
- D는 feature가 오히려 과활성화되며, 모델은 일관되게 잘못된 규칙을 학습한다.

**인사이트**  
feature를 유지하는 것은 숫자나 산술 표면 형식이 아니라 **올바른 입력-출력 관계 또는 일관된 규칙**이다.

---

### Figure 6. Group B와 C 비교
- B는 영어 데이터라서 분포가 크게 달라진다.
- C는 숫자·연산자 형식을 보존하지만 정답 관계를 제거한다.
- 두 조건 모두 정확도와 feature activation이 비슷하게 붕괴한다.

**인사이트**  
B의 실패가 단순히 영어 데이터가 out-of-distribution이어서 발생한 것이 아니다. C에서도 실패하므로, 핵심 학습 신호는 **숫자 token statistics가 아니라 정확한 산술 논리**라고 해석된다.

---

### Figure 7. 여러 random seed에서의 genesis step
- 네 seed의 genesis step은 500, 1500, 1000, 1000이다.
- 모두 500~1500 step 사이에서 sharp transition이 나타난다.
- 따라서 grokking 시점의 절대 위치는 변하지만, CGA detector는 모든 seed에서 전이를 감지한다.

**인사이트**  
grokking timing에는 변동성이 있지만, 현상 자체는 재현된다. 다만 intervention 실험 전체가 여러 seed에서 반복된 것은 아니다.

---

### Figure 8. 네 branch의 최종 결과
- A: 정확도 99.8%, 산술 activation 19.24, carry activation 21.74
- B: 정확도 0%, 산술 3.66, carry 6.08
- C: 정확도 6.1%, 산술 0, carry 0
- D: 정확도 0%, 산술 24.44, carry 26.49

**인사이트**  
정확도와 feature activation은 항상 같은 방향으로 움직이지 않는다. D의 경우 feature는 강하게 활성화되지만 모델이 학습한 규칙이 틀렸기 때문에 원래 정답 정확도는 0%이다. 따라서 feature는 “정답 여부”보다 **학습된 계산 규칙의 존재와 방향**을 반영한다.

---

### Figure 9. Direction–magnitude decomposition
- Group D에서 전체 weight L2 norm은 약 **3.3%**만 증가한다.
- 반면 산술 feature activation은 **27%** 증가한다.
- 두 변화 사이에는 약 8배의 차이가 있다.

**인사이트**  
D의 feature hyperactivation은 단순히 모든 weight나 activation의 크기가 커진 결과가 아니다. 특정 산술 규칙을 표현하는 방향이 선택적으로 강화된 것으로 해석된다.

---

## 테이블별 결과와 인사이트

### Table 1. Feature knockout
- 정상 상태: 정확도 100%
- 상위 5개 산술 feature를 zero-ablate: 정확도 0%

**인사이트**  
선택된 SAE 방향은 산술 능력에 **필수적**이다. 단, 이 결과는 해당 feature 방향의 기능적 필요성을 보여주며, 모델 전체의 모든 산술 회로가 이 다섯 feature에만 있다는 뜻은 아니다.

---

### Table 2. 네 intervention group의 최종 지표

| Group | 학습 데이터 | 정확도 | Arith | Carry |
|---|---|---:|---:|---:|
| A | 정상 덧셈 | 99.8% | 19.24 | 21.74 |
| B | 산술 제거 + 영어 | 0% | 3.66 | 6.08 |
| C | 숫자 형식 유지 + 오답 | 6.1% | 0 | 0 |
| D | 정답 +137 | 0% | 24.44 | 26.49 |

**인사이트**
- B: 산술 데이터가 없으면 feature가 크게 약화된다.
- C: 숫자 형식만으로는 feature가 유지되지 않는다.
- D: 일관된 잘못된 규칙도 feature를 강하게 형성할 수 있다.

---

### Table 3. B와 C의 통제 비교
- B는 영어 filler라서 OOD일 가능성이 있다.
- C는 숫자·연산 형식을 유지하지만 정답만 틀리게 만든다.
- 그런데 두 조건 모두 feature와 정확도가 무너진다.

**인사이트**  
실패의 원인은 단순한 distribution shift가 아니라 **정확한 산술 mapping의 부재**에 더 가깝다.

---

### Table 4. Random seed별 genesis 결과

| Seed | Genesis step | 해당 시점 정확도 |
|---|---:|---:|
| 0 | 500 | 85.0% |
| 1 | 1500 | 84.5% |
| 2 | 1000 | 86.7% |
| 3 | 1000 | 86.1% |

- 평균 genesis step: 1000 ± 433
- 탐지 정확도: 85.6% ± 1.0%

**인사이트**  
발현 시점에는 상당한 변동이 있지만, detector가 포착한 시점의 기능적 상태는 비교적 안정적이다.

---

### Table 5. Dataset statistics
- 학습 데이터: 2,000개
- validation: 5,000개
- 약 62%가 carry를 포함
- SAE arithmetic probe: carry가 있는 산술 예제 500개
- negative probe: 비산술 예제 500개

**인사이트**  
작은 학습 데이터와 큰 문제 공간의 조합이 grokking을 유도한다. 다만 carry probe가 carry-free 덧셈과 비교되지 않았다는 점은 해석상의 한계다.

---

### Table 6. Hyperparameters
주요 설정:

- 6-layer decoder-only Transformer
- 18.99M parameters
- (d_{text{model}}=512), 8 attention heads
- layer 4 residual stream에 SAE 적용
- SAE dictionary size 4,096, TopK 32
- 원래 학습 80,000 steps
- intervention branch 800 steps
- poisoning offset: +137

**인사이트**  
실험은 대규모 pretrained model이 아니라 작은 모델을 처음부터 학습하는 통제된 환경이다. 따라서 결과를 복잡한 실제 언어 모델에 바로 일반화하기는 어렵다.

---

### Table 7. Weight norm과 feature activation 비교
- Group A weight norm: 161.9
- Group D weight norm: 167.2
- weight norm 증가: 3.3%
- 산술 feature activation 증가: 27%

**인사이트**  
D의 현상은 global scaling이 아니라 특정 feature 방향에 대한 선택적 강화라는 해석을 지지한다.

---

### Table 8. 정성적 출력 예시

예시:

- `347+589=936`
  - A: 936, 정답
  - B: 481, 오답
  - D: 1073 = 936 + 137

- `125+389=514`
  - A: 514, 정답
  - B: 302, 오답
  - D: 651 = 514 + 137

**인사이트**  
D 모델은 무작위로 틀리는 것이 아니라, 매우 일관되게 `a+b+137` 규칙을 따른다. 따라서 feature는 “산술을 한다”는 표면적 행동뿐 아니라 **어떤 산술 mapping을 학습했는지**와 연결된다.

---

## Appendix별 핵심

### Appendix A. Dataset Statistics
본 실험의 데이터 구성과 carry 비율을 정리한다. 학습 데이터의 약 62%가 carry를 포함하며, validation과 probe set은 산술·비산술 반응을 비교하기 위해 별도로 구성되었다.

**핵심 한계**  
carry specificity를 검증하려면 비산술 텍스트뿐 아니라 **carry-free 덧셈 데이터**도 별도의 baseline으로 포함해야 한다.

---

### Appendix B. Hyperparameters
모델, SAE, 학습, intervention의 세부 설정을 제공해 재현성을 높인다.

**인사이트**  
특히 post-grokking checkpoint에서 학습한 SAE를 모든 checkpoint와 branch에 고정해 사용했다. 덕분에 feature activation을 같은 좌표계에서 비교할 수 있지만, checkpoint마다 feature 자체가 어떻게 변했는지는 직접 측정하지 못한다.

---

### Appendix C. Direction–Magnitude Decomposition
Group D의 feature activation 증가가 단순한 weight norm 증가 때문인지 검증한다.

- global weight norm: +3.3%
- 산술 feature activation: +27%
- 약 8배의 차이

**인사이트**  
poisoned rule이 특정 feature family를 선택적으로 강화한다는 해석을 뒷받침한다.

---

### Appendix D. Qualitative Examples
Group D가 두 예제 모두에서 정확한 덧셈 결과에 137을 더한 값을 출력한다.

**인사이트**  
poisoning은 모델을 단순히 혼란스럽게 만든 것이 아니라, 새로운 결정론적 규칙을 학습시켰다.

---

## 최종 평가

이 논문의 가장 중요한 기여는 **SAE feature의 존재를 최종 checkpoint에서 확인하는 데 그치지 않고, 그 feature가 언제 나타나고 어떤 데이터가 이를 유지하며 행동에 어떻게 연결되는지를 counterfactual intervention으로 검증한 것**이다.

다만 다음 한계가 있다.

- carry feature가 진정으로 carry만 담당하는지는 입증되지 않았다.
- intervention 결과는 주로 main seed 하나에서 보고되었다.
- 작은 3자리 덧셈 모델에 대한 결과이므로 코드 생성이나 복잡한 추론으로의 일반화는 아직 확인되지 않았다.
- 데이터 하나하나의 세밀한 attribution보다는 “산술 데이터 cluster” 수준의 attribution이다.

---





## Overall takeaway

The paper studies how arithmetic features emerge in a 19M-parameter Transformer trained on 3-digit addition. It connects:

**training data → SAE features → learned capability**

The main conclusions are:

- Arithmetic features emerge around the grokking transition.
- Ablating the top arithmetic features reduces accuracy from 100% to 0%, showing causal necessity.
- Correct arithmetic relations, rather than numeric token statistics alone, sustain these features.
- When answers are systematically shifted by +137, the same features become more active, but the model learns the wrong rule (a+b+137).

---

## Figures and diagrams

### Figure 1. CGA pipeline
The pipeline has three stages:

1. Train the model and save checkpoints.
2. Train an SAE on layer-4 residual-stream activations and select arithmetic/carry-related features.
3. Identify the genesis checkpoint and branch training into four counterfactual conditions.

**Insight:**  
CGA extends static mechanistic interpretability into a temporal and causal analysis of feature formation.

---

### Figure 2. Grokking and feature genesis
At step 500:

- Validation accuracy jumps from near zero to about 85%.
- Layer-4 activation norm rises from below 50 to about 200.8.
- Accuracy later reaches about 99.8%.

**Insight:**  
The arithmetic capability appears during a sharp representational transition rather than through smooth incremental improvement.

---

### Figure 3. Arithmetic and carry feature selectivity
The figure ranks SAE features by arithmetic discriminability and carry-probe contrast.

- The top arithmetic feature has a discriminability score of 1.504.
- The very large carry score is partly caused by the (10^{-6}) denominator floor.

**Important caveat:**  
The carry features were compared with non-arithmetic text, not carry-free addition. Therefore, the result does not establish strict carry specificity.

---

### Figure 4. Arithmetic feature fingerprint
The top ten arithmetic features have mean activations ranging from 9.6 to 28.4 on arithmetic prompts. Their mean activation is exactly 0.000 on non-arithmetic probes.

**Insight:**  
The activation profile provides a compact fingerprint for tracking the feature family across checkpoints and intervention branches.

---

### Figure 5. Four counterfactual branches

- **A: Control** — normal addition data
- **B: English filler** — arithmetic data replaced with English sequences
- **C: Numeric noise** — arithmetic-looking inputs with incorrect answers
- **D: Poison +137** — every answer shifted by 137

Results:

- A recovers both accuracy and feature activation.
- B causes both accuracy and feature collapse.
- C also collapses despite preserving numeric format.
- D increases feature activation while producing systematically wrong outputs.

**Insight:**  
The sustaining signal is the learned input-output relation, not merely the presence of numbers or arithmetic-like formatting.

---

### Figure 6. Group B versus Group C
Group B changes the data distribution substantially by using English text. Group C preserves the numeric format but removes the correct arithmetic relation. Both conditions fail similarly.

**Insight:**  
The failure cannot be explained only by out-of-distribution English filler. Correct arithmetic logic is the critical factor.

---

### Figure 7. Genesis across random seeds
The genesis steps are 500, 1500, 1000, and 1000 across four seeds. All show a sharp transition between 500 and 1500 steps.

**Insight:**  
The exact timing varies, but the grokking phenomenon is reproducible. However, the full intervention study was not replicated across all seeds.

---

### Figure 8. Final comparison of all branches
Final values:

- A: 99.8% accuracy; arithmetic 19.24; carry 21.74
- B: 0% accuracy; arithmetic 3.66; carry 6.08
- C: 6.1% accuracy; arithmetic 0; carry 0
- D: 0% accuracy; arithmetic 24.44; carry 26.49

**Insight:**  
Feature activation does not always correspond directly to correct accuracy. In Group D, the feature family is strongly active because the model learned a consistent but incorrect arithmetic rule.

---

### Figure 9. Direction–magnitude decomposition
Group D increases total weight norm by only 3.3%, while arithmetic feature activation increases by 27%.

**Insight:**  
The hyperactivation is unlikely to be caused by global scaling. It more likely reflects selective strengthening of directions encoding the shifted arithmetic rule.

---

## Tables

### Table 1. Feature knockout
Zero-ablating the top five arithmetic features reduces exact-match accuracy from 100% to 0%.

**Insight:**  
These feature directions are causally necessary for the learned addition behavior, although this does not imply that the entire arithmetic circuit consists only of these five features.

---

### Table 2. Final intervention metrics

| Group | Data | Accuracy | Arith | Carry |
|---|---|---:|---:|---:|
| A | Normal addition | 99.8% | 19.24 | 21.74 |
| B | No arithmetic + English | 0% | 3.66 | 6.08 |
| C | Numeric format + wrong answers | 6.1% | 0 | 0 |
| D | Answers shifted by +137 | 0% | 24.44 | 26.49 |

**Insight:**  
Correct arithmetic sustains the features, while a consistent wrong rule can also activate them strongly.

---

### Table 3. B versus C control
Group C preserves the numeric and operator format but still fails like Group B.

**Insight:**  
The results argue against a surface-token explanation and support the importance of the correct arithmetic mapping.

---

### Table 4. Genesis across seeds
Genesis accuracy remains around 84.5–86.7%, with an average detection accuracy of 85.6% ± 1.0%.

**Insight:**  
The onset step is variable, but the detector identifies a similar functional transition across seeds.

---

### Table 5. Dataset statistics
The training set contains 2,000 examples, validation contains 5,000 examples, and roughly 62% of examples require carry.

**Insight:**  
The small training set relative to the full problem space is deliberately used to induce grokking.

**Caveat:**  
A carry-free addition baseline would be needed to establish true carry specificity.

---

### Table 6. Hyperparameters
The model is a 6-layer, 18.99M-parameter decoder-only Transformer. The SAE has 4,096 dictionary features with TopK sparsity of 32. Intervention branches run for 800 steps, with a poisoning offset of +137.

**Insight:**  
This is a controlled from-scratch setting rather than a large pretrained language model, so direct generalization to complex reasoning tasks remains uncertain.

---

### Table 7. Weight norm versus feature activation
Weight norm increases by only 3.3%, while arithmetic feature activation increases by 27%.

**Insight:**  
This supports selective rule encoding rather than a global increase in model magnitude.

---

### Table 8. Qualitative outputs
For example:

- `347+589=936`
  - Control: 936
  - Group B: 481
  - Group D: 1073 = 936 + 137

- `125+389=514`
  - Control: 514
  - Group B: 302
  - Group D: 651 = 514 + 137

**Insight:**  
Group D does not make random errors. It consistently follows the shifted rule (a+b+137).

---

## Appendices

### Appendix A. Dataset statistics
This appendix details the number of examples and carry proportions in each group.

**Main point:**  
The probe design is useful for arithmetic selectivity, but it does not isolate carry-specific processing because the negative baseline is non-arithmetic text.

---

### Appendix B. Hyperparameters
This appendix provides the full model, SAE, training, and intervention settings.

**Main point:**  
A post-grokking SAE is kept fixed across checkpoints and branches, enabling comparable activation measurements. However, this does not reveal how the SAE dictionary itself changes during training.

---

### Appendix C. Direction–magnitude decomposition
The appendix compares global weight growth with arithmetic feature growth and finds an approximately eightfold dissociation.

**Main point:**  
The +137 condition selectively strengthens arithmetic-related directions rather than merely increasing all activations.

---

### Appendix D. Qualitative examples
The examples show that Group D consistently outputs the true sum plus 137.

**Main point:**  
Poisoning produces a coherent alternative rule, not merely random degradation.

---

## Final assessment

The central contribution is the combination of:

- checkpoint-resolved training dynamics,
- fixed-basis SAE feature tracking,
- feature knockout,
- and counterfactual data interventions.

Together, these provide evidence for a causal chain from the training signal to internal features and finally to arithmetic behavior.

The main limitations are that carry specificity is not fully isolated, the intervention experiment is primarily reported for one seed, and the study uses a small controlled addition task rather than a broad language-model capability.

<br/>
# refer format:
### BibTeX

```bibtex
@inproceedings{wang2026tracing,
  author    = {Wang, Xiaokun and Song, Siyu and Zhao, Hongbo and Liu, Wentao and Jiang, Jiaqi and Mao, Qingqing},
  title     = {Tracing the Genesis of Logic: Causal Data Attribution of Mechanistic Features via Counterfactual Poisoning},
  booktitle = {Proceedings of the 15th International Joint Conference on Natural Language Processing and the 5th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics},
  year      = {2026},
  month     = nov,
  pages     = {2474--2483},
  publisher = {Association for Computational Linguistics},
  address   = {Online},
  note      = {November 6--10, 2026}
}
```

### 시카고 스타일    

Wang, Xiaokun, Siyu Song, Hongbo Zhao, Wentao Liu, Jiaqi Jiang, and Qingqing Mao. “Tracing the Genesis of Logic: Causal Data Attribution of Mechanistic Features via Counterfactual Poisoning.” In *Proceedings of the 15th International Joint Conference on Natural Language Processing and the 5th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics*, 2474–2483. November 6–10, 2026. Association for Computational Linguistics.




