---
layout: post
title:  "[2026]The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models"
date:   2026-08-09 17:38:38 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 디퓨전 언어모델은 더 다양한 랜덤경로 탐색 장점이 있지만 오히려 일반 추론에서는 이 때문에 성능이 제한됨, 엔트로피 저하라고 정의    
이러한 문제를 해결하기 위해 복잡한 확산 전용 강화학습 기법을 사용하는 대신, 강화학습을 학습할 때만 dLLM을 **일반적인 자기회귀(AR) 모델처럼 왼쪽에서 오른쪽으로** 사용하자고 제안   


짧은 요약(Abstract) :



확산 언어 모델(dLLM)은 기존 언어 모델처럼 반드시 왼쪽에서 오른쪽 순서로 토큰을 생성하지 않고, 여러 위치의 토큰을 임의의 순서로 생성할 수 있습니다. 이는 이론적으로 더 다양한 풀이 경로를 탐색할 수 있다는 장점처럼 보입니다.

하지만 이 논문은 수학·코딩과 같은 일반적인 추론 문제에서는 **임의 순서 생성이 오히려 추론 가능성을 제한할 수 있다**고 주장합니다. 모델은 임의 순서의 유연성을 이용해 불확실성이 높은 토큰, 특히 “Therefore”, “Since”처럼 논리적 방향을 결정하는 토큰을 피하고, 먼저 확신하기 쉬운 토큰부터 생성하는 경향을 보입니다. 그러면 나중에 어려운 토큰을 채울 때 이미 주변 문맥이 결정되어 있어 다양한 추론 경로를 탐색하기 어렵습니다. 논문은 이를 **엔트로피 저하(entropy degradation)**라고 부릅니다.

저자들은 이러한 문제를 해결하기 위해 복잡한 확산 전용 강화학습 기법을 사용하는 대신, 강화학습을 학습할 때만 dLLM을 **일반적인 자기회귀(AR) 모델처럼 왼쪽에서 오른쪽으로** 사용하자고 제안합니다. 이 방법이 **JustGRPO**입니다. 학습 중에는 순차적인 생성으로 탐색과 보상 계산을 안정화하지만, 모델의 구조 자체를 바꾸거나 인과적 마스크를 추가하지는 않습니다. 따라서 추론 시에는 dLLM의 장점인 **병렬 디코딩**을 그대로 유지할 수 있습니다.

실험 결과, JustGRPO는 GSM8K에서 89.1%의 정확도를 달성하는 등 강력한 성능을 보였습니다. 즉, 이 논문의 핵심 주장은 **추론 능력을 높이기 위해서는 생성 순서의 자유를 항상 보존해야 하는 것이 아니며, 학습 단계에서는 오히려 순차적 탐색을 강제하는 것이 더 효과적일 수 있다**는 것입니다.

---



Diffusion language models (dLLMs) can generate tokens in arbitrary orders instead of following the strict left-to-right order used by autoregressive models. Although this flexibility seems to provide a larger space for reasoning, the paper shows that it can actually hurt general reasoning tasks such as mathematics and coding.

The model often uses arbitrary-order generation to avoid highly uncertain tokens, especially logical connectors such as “Therefore” or “Since.” These tokens represent important branching points in a reasoning process. By generating easier, low-uncertainty tokens first, the model establishes future context before resolving these difficult decisions. As a result, the original uncertainty is reduced and many possible reasoning paths are prematurely eliminated. The authors call this phenomenon **entropy degradation**.

Based on this finding, the paper proposes **JustGRPO**, which applies standard Group Relative Policy Optimization while treating the dLLM as an autoregressive policy during reinforcement-learning training. This sequential training process improves exploration and makes likelihood computation straightforward, without requiring complicated diffusion-specific RL methods.

Importantly, the autoregressive constraint is used only as a training scaffold. The model architecture is not changed, so the dLLM can still use parallel decoding during inference. JustGRPO achieves strong results, including **89.1% accuracy on GSM8K**, showing that restricting generation order during training can improve reasoning while preserving the parallel-decoding advantage of diffusion language models.


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
## 방법(Method) — JustGRPO



- **LLaDA 8B Instruct**를 기반 모델로 사용한다.
- LLaDA는 **Masked Diffusion Language Model(dLLM)** 으로, 여러 위치의 `[MASK]` 토큰을 반복적으로 복원한다.
- 기본적으로는 토큰을 임의의 순서로 복원할 수 있고, 여러 토큰을 병렬로 생성할 수 있다.

### 2. 핵심 아이디어: 학습 중에는 AR 순서 사용
이 논문의 핵심은 dLLM의 임의 순서 생성을 RL 학습 단계에서는 사용하지 않는 것이다.

- 학습 중에는 토큰을 **왼쪽에서 오른쪽으로 한 토큰씩** 생성한다.
- 현재 토큰 \(o_k\)를 예측할 때:
  - 이전 토큰 \(o_{<k}\)는 관측 상태로 둔다.
  - 이후 토큰은 모두 `[MASK]`로 둔다.
- 따라서 dLLM을 학습 중에 **오토리그레시브 정책**처럼 정의할 수 있다.

\[
\pi_\theta^{AR}(o_k|o_{<k},q)
\]

이를 통해 전체 문장의 확률을 다음과 같이 정확하게 계산할 수 있다.

\[
\pi_\theta^{AR}(o|q)
=\prod_k \pi_\theta^{AR}(o_k|o_{<k},q)
\]

즉, 임의의 디노이징 순서들을 모두 marginalize해야 하는 기존 diffusion RL의 복잡성을 피하고, 일반적인 AR 모델처럼 정확한 likelihood와 policy ratio를 계산한다.

### 3. RL 알고리즘: 표준 GRPO
- 별도의 diffusion 전용 RL 알고리즘을 설계하지 않고, **표준 GRPO(Group Relative Policy Optimization)** 를 그대로 적용한다.
- 하나의 문제에 대해 여러 개의 답변을 생성하고, 답변들의 보상을 그룹 내에서 정규화하여 advantage를 계산한다.
- 현재 정책과 이전 정책의 토큰별 확률 비율을 사용해 clipped policy-gradient objective를 최적화한다.
- 논문 설정:
  - Group size: 16
  - Global batch size: 64
  - Learning rate: \(5\times10^{-6}\)
  - RL training steps: 125
  - 최대 생성 길이: 256
  - Full-parameter fine-tuning
  - KL penalty coefficient: 0

### 4. 보상 설계
#### 수학 문제
- 정답과 수학적으로 동치이면 reward 1
- 그렇지 않으면 reward 0

#### 코드 문제
- **정답성 보상**: 제공된 unit test 통과율
- **형식 보상**:
  - 올바른 Markdown 코드 블록 및 문법: 1.0
  - 코드 블록은 있으나 문법 오류: 0.5
  - 올바른 코드 블록이 없음: 0

### 5. 학습 데이터
- 수학:
  - GSM8K
  - MATH-500 관련 공식 학습 데이터
- 코드:
  - AceCoder-87K에서 unit test가 있는 어려운 샘플 약 **21K개**를 선택
- 추가적인 task-specific SFT는 사용하지 않고, 기존 LLaDA-Instruct에 직접 RL을 적용했다.

### 6. 추론 단계와 학습 단계의 분리
JustGRPO는 모델 구조 자체를 AR 모델로 바꾸지 않는다.

- **RL 학습 단계**: AR 순서로 한 토큰씩 생성
- **추론 단계**: dLLM의 원래 기능인
  - 양방향 attention
  - masked diffusion
  - confidence-based remasking
  - 여러 토큰의 병렬 복원  
  을 그대로 사용한다.

따라서 학습에서는 AR 순서가 탐색과 credit assignment를 안정화하고, 추론에서는 dLLM의 병렬 디코딩 속도를 유지한다.

### 7. 왜 임의 순서를 제한하는가?
논문은 임의 순서 생성이 항상 장점이 되는 것은 아니라고 주장한다.

- 논리적 연결어인 “Therefore”, “Since”, “Thus” 등은 여러 추론 경로가 갈라지는 **고엔트로피 fork token**이다.
- 임의 순서 디코딩은 이런 어려운 토큰을 건너뛰고 확신도가 높은 쉬운 토큰부터 생성하는 경향이 있다.
- 이후 미래 문맥이 이미 정해지면, fork token의 불확실성이 줄어들어 다양한 추론 경로를 탐색하지 못한다.
- 이를 논문에서는 **entropy degradation**이라고 부른다.
- AR 순서는 이러한 불확실한 토큰을 즉시 결정하게 하므로, 더 다양한 reasoning trajectory를 탐색할 수 있다.

### 8. 주요 결과
LLaDA-Instruct에서 JustGRPO는 다음 성능을 보였다.

- GSM8K: **89.1%**
- MATH-500: **45.1%**
- HumanEval: **49.4%**
- MBPP: **52.4%**

또한 AR 방식으로 학습했음에도 추론 시 병렬 디코딩이 가능했으며, 모델의 일반 능력도 대체로 유지되었다.

### 한 줄 요약
**JustGRPO는 dLLM을 RL 학습 중에만 AR 정책으로 취급하여 표준 GRPO를 적용하고, 추론 단계에서는 diffusion 모델의 병렬 디코딩 능력을 그대로 보존하는 방법이다.**

---




### 1. Base Model
- The method is applied to **LLaDA 8B Instruct**, a masked diffusion language model.
- LLaDA generates text by iteratively unmasking tokens.
- In its native setting, tokens can be generated in arbitrary orders and multiple tokens can be decoded in parallel.

### 2. Core Idea: AR Ordering During RL Training
The key idea is to **forgo arbitrary-order generation during reinforcement learning**.

- During training, tokens are generated strictly **from left to right**, one token at a time.
- To predict token \(o_k\):
  - The previous tokens \(o_{<k}\) are visible.
  - All future tokens are replaced with `[MASK]`.
- This defines an autoregressive policy on top of the diffusion model:

\[
\pi_\theta^{AR}(o_k|o_{<k},q)
\]

The sequence likelihood can therefore be factorized exactly:

\[
\pi_\theta^{AR}(o|q)
=\prod_k \pi_\theta^{AR}(o_k|o_{<k},q)
\]

This avoids marginalizing over the combinatorial set of possible diffusion denoising trajectories.

### 3. RL Algorithm: Standard GRPO
- The method directly applies **standard Group Relative Policy Optimization (GRPO)**.
- For each prompt, multiple responses are sampled.
- Rewards are normalized within the group to compute relative advantages.
- The model is optimized using token-level probability ratios and the standard clipped GRPO objective.
- Main settings:
  - Group size: 16
  - Global batch size: 64
  - Learning rate: \(5\times10^{-6}\)
  - Training steps: 125
  - Maximum completion length: 256
  - Full-parameter fine-tuning
  - No KL penalty in the reported setup

### 4. Reward Design
#### Mathematical Tasks
- Reward 1 if the final answer is mathematically equivalent to the ground truth.
- Otherwise, reward 0.

#### Code Generation
The reward consists of:
- **Correctness reward**: unit-test pass rate
- **Format reward**:
  - 1.0 for a valid Markdown code block with correct Python syntax
  - 0.5 for a code block containing syntax errors
  - 0.0 if no valid code block is produced

### 5. Training Data
- Mathematical reasoning datasets:
  - GSM8K
  - MATH-500-related official training data
- Coding data:
  - Approximately **21K challenging examples** selected from AceCoder-87K with executable unit tests
- No additional task-specific supervised fine-tuning is used; RL is applied directly to LLaDA-Instruct.

### 6. Separating Training from Inference
JustGRPO does not convert the model architecture into a conventional autoregressive model.

- **During RL training**: left-to-right autoregressive generation
- **During inference**: the original diffusion capabilities are retained:
  - Bidirectional attention
  - Masked diffusion formulation
  - Confidence-based remasking
  - Parallel token decoding

Thus, AR ordering provides better exploration and credit assignment during training, while parallel decoding is preserved at inference time.

### 7. Why Restrict Arbitrary Order?
The paper argues that arbitrary-order generation can reduce exploration in general reasoning tasks.

- Logical connectors such as “Therefore,” “Since,” and “Thus” often act as high-entropy **forking tokens**.
- Arbitrary-order decoding tends to postpone these difficult tokens and generate easier, high-confidence tokens first.
- Once future context has been established, the uncertainty at the logical fork decreases.
- The model then fills in the connector retrospectively instead of exploring multiple reasoning branches.
- This effect is called **entropy degradation**.
- Autoregressive ordering forces the model to confront these uncertain decisions, helping preserve solution-space coverage.

### 8. Main Results
On LLaDA-Instruct, JustGRPO achieves:

- GSM8K: **89.1%**
- MATH-500: **45.1%**
- HumanEval: **49.4%**
- MBPP: **52.4%**

The AR-trained model also remains compatible with parallel decoding and largely preserves general non-reasoning capabilities.

### One-sentence Summary
**JustGRPO treats the diffusion language model as an autoregressive policy only during RL training, enabling exact standard GRPO optimization while preserving the model’s native parallel diffusion decoding at inference.**


<br/>
# Results



### 1. 연구에서 비교한 대상과 평가 데이터

#### 비교 대상
논문은 크게 두 가지를 비교합니다.

1. **생성 순서**
   - **AR Order**: 항상 왼쪽에서 오른쪽으로 토큰을 생성
   - **Arbitrary Order**: 확신도가 높은 토큰부터 자유롭게 생성·수정하는 diffusion 방식
   - 추가로 block size \(B\)를 조절해 생성 순서의 자유도를 비교  
     - \(B=1\): 순수 AR
     - \(B\)가 클수록 더 자유로운 순서

2. **RL 학습 방법**
   - 기존 diffusion LLM용 RL 방법: **d1, LLaDOU, LLaDA-1.5, wd1, d-TreeRPO, ESPO, GDPO, SPG**
   - 제안 방법: **JustGRPO**
   - JustGRPO는 diffusion 특화 likelihood 근사나 순서 모델을 사용하지 않고, RL 학습 중에만 dLLM을 AR 정책처럼 다루어 일반적인 GRPO를 적용합니다.

#### 테스트 데이터
다음 네 가지 추론·코딩 벤치마크를 사용했습니다.

- **GSM8K**: 초등 수준 수학 문제
- **MATH-500**: 수학 추론 문제
- **HumanEval**: 코드 생성
- **MBPP**: 파이썬 코드 생성

학습에는 각 수학 데이터셋의 학습 분할과, 코딩의 경우 **AceCoder-87K에서 선별한 21K개 검증 가능한 문제**를 사용했습니다.

---

### 2. 핵심 분석: AR 순서가 더 넓은 추론 가능성을 보임

#### 사용한 메트릭: Pass@k
- **Pass@1**: 한 번 생성했을 때 맞힐 확률
- **Pass@k**: \(k\)개의 샘플 중 하나라도 정답일 확률
- 따라서 Pass@k가 클수록 모델이 다양한 정답 추론 경로를 탐색할 수 있다는 의미입니다.
- 논문은 RL이 강화할 수 있는 추론 능력의 상한을 **기본 모델의 Pass@k**로 간주합니다.

#### 주요 결과
- Arbitrary Order는 **k=1에서는 경쟁력 있는 성능**을 보이는 경우가 많았습니다.
- 그러나 k가 증가하면 AR Order의 Pass@k 곡선이 더 가파르게 상승했습니다.
- 즉, Arbitrary Order는 한 번의 생성에서는 그럴듯한 답을 만들 수 있지만, 여러 번 샘플링할 때 **새로운 정답 풀이를 발견하는 능력**, 즉 solution coverage가 낮았습니다.
- 이 현상은 **LLaDA-Instruct, Dream-Instruct, LLaDA 1.5**와 GSM8K, MATH-500, HumanEval, MBPP 전반에서 일관되게 관찰되었습니다.

#### 해답 공간 중복 분석
LLaDA-Instruct로 Pass@1024에서 발견한 문제별 해답 범위를 비교한 결과:

- **HumanEval**
  - AR만 해결한 문제: **21.3%**
  - Arbitrary Order만 해결한 문제: **0.6%**
- **MBPP**
  - AR만 해결: **14.0%**
  - Arbitrary Order만 해결: **0.8%**

따라서 Arbitrary Order가 이론적으로는 더 큰 해답 공간을 허용하지만, 실제 샘플링에서 도달하는 해답은 대부분 AR이 찾는 해답의 부분집합에 가까웠습니다.

또한 block size \(B\)를 증가시킬수록, 즉 생성 순서를 더 자유롭게 만들수록 HumanEval의 Pass@8, 32, 128이 **일관되고 단조롭게 감소**했습니다. 논문의 결론은 “이 실험 조건에서는 순서의 자유도가 클수록 추론 가능성이 낮아진다”는 것입니다.

---

### 3. 원인 분석: Entropy Degradation

논문은 Arbitrary Order가 추론의 중요한 분기점을 건너뛰기 때문에 탐색 능력이 줄어든다고 설명합니다.

- “Therefore”, “Thus”, “Since”, “However” 같은 논리 연결어는 이후 추론 방향을 결정하는 **forking token**입니다.
- 이러한 토큰은 일반적으로 높은 entropy를 가지며, 여러 추론 경로 중 하나를 선택하는 지점입니다.
- AR Order는 해당 토큰을 즉시 결정해야 하므로 불확실성과 여러 가능한 경로를 유지합니다.
- Arbitrary Order는 확신도가 높은 쉬운 토큰을 먼저 생성하고, 논리 연결어를 나중으로 미룹니다.
- 이후 미래 문맥이 이미 정해지면 연결어의 불확실성이 낮아져, 여러 경로를 탐색하는 대신 이미 정해진 결론에 맞게 문장을 연결하게 됩니다.

이를 논문에서는 **entropy degradation**이라고 부릅니다.

즉:

> Arbitrary Order는 낮은 불확실성 토큰을 먼저 처리하면서 하나의 경로에 일찍 고정되고, AR Order는 불확실한 분기점을 직접 마주하게 하여 더 넓은 추론 탐색을 유지합니다.

---

### 4. JustGRPO의 성능

#### 평가 설정
- 기반 모델: **LLaDA-Instruct**
- 평가 길이: **128, 256, 512 토큰**
- 평가 메트릭:
  - 수학: 정답률
  - 코드: 테스트 케이스 통과율 기반 정확도
- 비교 조건은 논문별로 달라 일부 결과는 완전히 동일한 설정은 아닙니다.

#### Table 1의 주요 결과: 생성 길이 256

| 방법 | GSM8K | MATH-500 | HumanEval | MBPP |
|---|---:|---:|---:|---:|
| d1 | 81.1 | 38.6 | - | - |
| LLaDOU | 88.1 | 41.1 | 59.1 | 51.6 |
| LLaDA-1.5 | 83.3 | - | 39.6 | 39.9 |
| wd1 | 80.8 | 34.4 | - | - |
| d-TreeRPO | 81.2 | 37.7 | - | - |
| ESPO | 82.3 | 39.0 | 42.1 | 44.6 |
| GDPO | 82.8 | 39.6 | 39.6 | 50.6 |
| SPG | 86.1 | 40.0 | - | - |
| **JustGRPO** | **89.1** | **45.1** | **49.4** | **52.4** |

핵심적으로 JustGRPO는 다음 성능을 달성했습니다.

- **GSM8K: 89.1%**
- **MATH-500: 45.1%**
- **HumanEval: 49.4%**
- **MBPP: 52.4%**

복잡한 diffusion 전용 RL 기법을 사용하지 않았음에도 대부분의 기존 방법보다 높거나 경쟁력 있는 결과입니다.

#### 동일 조건 재현 결과: Table 2
논문은 공정한 비교를 위해 full fine-tuning, 한 단계에 한 토큰 생성, 생성 길이 256으로 일부 방법을 재현했습니다.

| 방법 | GSM8K | MATH-500 |
|---|---:|---:|
| d1 재현 | 83.8 | 39.2 |
| ESPO 재현 | 84.7 | 40.3 |
| SPG 재현 | 86.9 | 41.8 |
| **JustGRPO** | **89.1** | **45.1** |

JustGRPO는 SPG 대비:

- GSM8K: **+2.2%p**
- MATH-500: **+3.3%p**

의 성능 향상을 보였습니다.

---

### 5. 병렬 디코딩 능력 보존

AR 순서로 RL 학습하면 dLLM이 일반 autoregressive 모델로 바뀌는지 확인했습니다.

- 학습 중에는 AR 순서를 사용하지만, **causal mask를 도입하지 않습니다.**
- 따라서 모델의 bidirectional attention과 diffusion 구조는 그대로 유지됩니다.
- 추론 시에는 다시 EB sampler 등을 사용해 여러 토큰을 병렬로 생성할 수 있습니다.

결과적으로 JustGRPO 모델은 병렬 디코딩과 완전히 호환되었습니다.

특히 MBPP에서 병렬 생성량이 증가할 때:

- 보수적인 설정: 원래 모델 대비 약 **+10.6%p**
- 공격적인 병렬 설정: 약 **+25.5%p**

의 개선이 관찰되었습니다.

즉, JustGRPO는 **학습에서는 AR 기반 탐색을 사용하면서도, 추론에서는 diffusion LLM의 병렬 생성 이점은 유지**합니다.

---

### 6. 학습 효율과 추가 분석

#### 학습 효율
JustGRPO는 토큰별 정확한 likelihood를 계산해야 하므로 iteration당 비용은 기존 근사 방법보다 클 수 있습니다. 그러나 실제 wall-clock 기준으로는:

- ESPO보다 정확도/시간 효율이 경쟁력 있었고,
- 고 entropy 토큰 상위 25%에서만 likelihood ratio를 계산하는 **JustGRPO-Fast**는 추가로 학습 시간을 줄였습니다.

#### Random Order 비교
무작위 생성 순서는 해결책이 되지 못했습니다.

- Pass@128은 AR보다 낮았습니다.
- Pass@1은 특히 크게 낮았습니다.
  - GSM8K:
    - Confidence-based: 78.6%
    - Fully Random: 43.3%
    - AR: 78.0%
- RL 후에도:
  - JustGRPO-Random: **82.2%**
  - JustGRPO: **89.1%**

무작위 순서는 충분한 문맥이 형성되기 전에 토큰을 예측하여 문법·수식 구조를 깨뜨릴 수 있기 때문입니다.

---

### 7. 최종 결론

이 논문의 핵심은 다음과 같습니다.

1. Arbitrary Order는 이론적으로 더 넓은 생성 경로를 제공하지만, 일반 수학·코딩 추론에서는 실제 solution coverage를 오히려 줄일 수 있습니다.
2. 확신도가 높은 토큰을 먼저 생성하면 논리적 fork가 나중에 결정되고, 미래 문맥 때문에 불확실성이 사라지는 **entropy degradation**이 발생합니다.
3. AR 순서는 중요한 불확실성 토큰을 직접 결정하게 하므로 RL에 필요한 탐색과 다양한 추론 경로를 더 잘 보존합니다.
4. 따라서 RL 학습 단계에서는 복잡한 diffusion 전용 방법보다, dLLM을 AR 정책으로 간주해 표준 GRPO를 적용하는 **JustGRPO**가 효과적입니다.
5. 이때 AR 제약은 학습 중에만 적용되며, 추론 시에는 dLLM의 병렬 디코딩 능력을 그대로 사용할 수 있습니다.

---




### 1. Compared Methods and Evaluation Benchmarks

#### Compared generation orders
The paper compares:

- **AR Order**: tokens are generated strictly from left to right.
- **Arbitrary Order**: tokens are generated according to confidence-based diffusion decoding.
- Different semi-autoregressive block sizes \(B\):
  - \(B=1\): pure AR decoding
  - Larger \(B\): more freedom in choosing the next token

For reinforcement learning, JustGRPO is compared with diffusion-specific RL methods including **d1, LLaDOU, LLaDA-1.5, wd1, d-TreeRPO, ESPO, GDPO, and SPG**.

#### Evaluation data
The experiments use:

- **GSM8K** for mathematical reasoning
- **MATH-500** for advanced mathematical reasoning
- **HumanEval** for code generation
- **MBPP** for Python code generation

Training uses the official mathematical training splits and a verified subset of **21K samples from AceCoder-87K** for coding tasks.

---

### 2. Main Finding: AR Order Has Higher Reasoning Potential

The paper uses **Pass@k** as the main measure of reasoning potential.

- **Pass@1** measures single-sample accuracy.
- **Pass@k** measures whether at least one of \(k\) sampled solutions is correct.
- A higher Pass@k indicates broader solution-space coverage and better exploration potential for RL.

Arbitrary Order is often competitive at **k=1**, but its Pass@k curve is flatter as \(k\) increases. AR Order discovers substantially more correct solutions with larger sampling budgets.

This pattern is consistent across:

- LLaDA-Instruct
- Dream-Instruct
- LLaDA 1.5
- GSM8K, MATH-500, HumanEval, and MBPP

At Pass@1024 with LLaDA-Instruct:

- On **HumanEval**, AR solved 21.3% of problems that Arbitrary Order did not, while Arbitrary Order solved only 0.6% exclusively.
- On **MBPP**, the corresponding numbers were 14.0% for AR-only and 0.8% for Arbitrary-Order-only.

Thus, although Arbitrary Order theoretically permits more trajectories, the practically reachable solution set is often a smaller subset of the AR solution set.

Increasing the block size \(B\), and therefore increasing order flexibility, consistently reduced Pass@k on HumanEval.

---

### 3. Mechanism: Entropy Degradation

The paper attributes this behavior to **logical forking tokens**, such as:

- “Therefore”
- “Thus”
- “Since”
- “However”

These tokens often determine the direction of subsequent reasoning and have relatively high entropy.

- **AR Order** forces the model to make these uncertain decisions immediately, preserving multiple possible reasoning branches.
- **Arbitrary Order** tends to generate easy, high-confidence tokens first and postpones these logical connectors.
- Once future tokens have already been generated, the remaining connector becomes much less uncertain and is filled in retrospectively to match an already determined conclusion.

The paper calls this **entropy degradation**: the model avoids important branching decisions, prematurely commits to one trajectory, and loses solution-space coverage.

---

### 4. JustGRPO Results

#### Evaluation setting
JustGRPO is applied to **LLaDA-Instruct** and evaluated with sequence lengths of 128, 256, and 512 tokens.

At sequence length 256, the main results are:

| Method | GSM8K | MATH-500 | HumanEval | MBPP |
|---|---:|---:|---:|---:|
| d1 | 81.1 | 38.6 | - | - |
| LLaDOU | 88.1 | 41.1 | 59.1 | 51.6 |
| LLaDA-1.5 | 83.3 | - | 39.6 | 39.9 |
| wd1 | 80.8 | 34.4 | - | - |
| d-TreeRPO | 81.2 | 37.7 | - | - |
| ESPO | 82.3 | 39.0 | 42.1 | 44.6 |
| GDPO | 82.8 | 39.6 | 39.6 | 50.6 |
| SPG | 86.1 | 40.0 | - | - |
| **JustGRPO** | **89.1** | **45.1** | **49.4** | **52.4** |

JustGRPO achieves:

- **89.1% on GSM8K**
- **45.1% on MATH-500**
- **49.4% on HumanEval**
- **52.4% on MBPP**

Despite using no diffusion-specific likelihood approximation or auxiliary order model, it is competitive with or better than most compared methods.

Under a more consistent reproduced setting:

| Method | GSM8K | MATH-500 |
|---|---:|---:|
| d1 reproduced | 83.8 | 39.2 |
| ESPO reproduced | 84.7 | 40.3 |
| SPG reproduced | 86.9 | 41.8 |
| **JustGRPO** | **89.1** | **45.1** |

Compared with reproduced SPG, JustGRPO improves by:

- **+2.2 percentage points on GSM8K**
- **+3.3 percentage points on MATH-500**

---

### 5. Parallel Decoding Is Preserved

AR order is used only as a training-time scaffold.

- The model architecture is unchanged.
- No causal mask is imposed.
- Bidirectional attention and the diffusion formulation remain intact.
- At inference time, the model can still use parallel samplers.

JustGRPO therefore preserves the native parallel decoding ability of dLLMs. The improvement becomes even larger under more aggressive parallel decoding. On MBPP, the gain over the original model increases from approximately **+10.6 points** with conservative decoding to **+25.5 points** with more aggressive parallelism.

---

### 6. Efficiency and Random-Order Analysis

#### Training efficiency
Exact GRPO requires additional per-position likelihood calculations. Nevertheless, JustGRPO has a competitive accuracy/wall-clock trade-off compared with approximation-based methods such as ESPO.

The paper also proposes **JustGRPO-Fast**, which computes likelihood ratios only for the top 25% highest-entropy positions. This substantially reduces the computation while preserving strong performance.

#### Random order
Random decoding is not a successful alternative.

On GSM8K Pass@1:

- Confidence-based Arbitrary Order: **78.6%**
- Fully Random Order: **43.3%**
- AR Order: **78.0%**

After RL training:

- JustGRPO-Random: **82.2%**
- JustGRPO with AR order: **89.1%**

Random order often predicts tokens before sufficient surrounding context is available, producing structurally broken mathematical expressions or code.

---

### 7. Overall Conclusion

The paper’s main conclusions are:

1. Arbitrary token order is theoretically more flexible, but in general mathematical and coding reasoning it can reduce practical solution coverage.
2. Confidence-based decoding tends to bypass high-entropy logical forks, causing entropy degradation and premature commitment to a single reasoning path.
3. AR ordering preserves uncertainty at these critical branching points and therefore supports broader exploration.
4. A simple standard GRPO formulation, applied to an AR surrogate policy during training, is sufficient to elicit strong reasoning abilities.
5. JustGRPO improves reasoning performance while preserving the parallel decoding capability of diffusion language models at inference time.


<br/>
# 예제



### 1. 논문이 다루는 핵심 과제

이 논문은 **Diffusion Large Language Model(dLLM)이 수학·코딩 문제를 얼마나 잘 탐색하고 추론하는지**를 연구합니다.

비교하는 생성 방식은 두 가지입니다.

- **AR Order**: 왼쪽에서 오른쪽으로 한 토큰씩 생성
- **Arbitrary Order**: 모델이 확신이 높은 토큰부터 임의의 순서로 생성·수정

논문의 핵심 주장은 다음과 같습니다.

> 임의 순서 생성은 이론적으로 더 많은 생성 경로를 제공하지만, 실제 일반 추론 문제에서는 불확실한 핵심 토큰을 회피하게 만들어 오히려 탐색 가능한 해의 범위를 줄일 수 있다.

---

### 2. 트레이닝 데이터와 구체적인 입력·출력 형태

논문에는 개별 데이터 샘플의 실제 원문이 모두 공개되어 있지는 않습니다. 대신 사용한 데이터셋과 입력·출력의 형식은 다음과 같이 설명할 수 있습니다.

#### 2.1 수학 추론 데이터

**사용 데이터**

- GSM8K
- MATH-500 관련 수학 문제
- 각 데이터셋의 공식 학습 분할(training split)

**입력 예시**

```text
A store has 180 items. The number of items sold in the third month
was 30% lower than in the second month, which was also 30% lower
than in the first month. How many items were sold in the third month?
```

또는 실제 LLaDA 계열의 프롬프트 형식에 따라 문제와 풀이 지시가 함께 주어집니다.

```text
Solve the following problem. Explain your reasoning and give the final answer.

[수학 문제]
```

**모델 출력 예시**

```text
The number sold in the second month was:
180 × 0.7 = 126.

The number sold in the third month was:
126 × 0.7 = 88.2.

Therefore, the answer is 88.2.
```

다만 위 입력과 출력은 논문의 실험 구조를 설명하기 위한 예시이며, 논문에서 특정 학습 샘플의 전체 원문과 정답 출력을 제시한 것은 아닙니다.

**보상 방식**

수학 문제에서는 이진 보상을 사용합니다.

- 최종 답이 정답과 수학적으로 같음: `reward = 1`
- 틀림: `reward = 0`

즉, 풀이 과정의 문장 품질보다 **최종 답을 맞혔는지**가 주된 보상 기준입니다.

---

#### 2.2 코드 생성 데이터

**사용 데이터**

- AceCoder-87K에서 선택한 데이터
- 그중 단위 테스트(unit test)가 있고 비교적 어려운 약 21K 샘플

**입력 예시**

```text
Write a Python function that returns the sum of all even numbers
in a list.

The function must pass the provided unit tests.
```

**모델 출력 예시**

```python
```python
def sum_even_numbers(nums):
    return sum(x for x in nums if x % 2 == 0)
```
```

실제 평가에서는 생성된 코드를 단위 테스트에 실행합니다.

**코드 보상**

코드 문제의 보상은 다음 두 부분의 합입니다.

1. **정확성 보상**
   - 단위 테스트 통과율
   - 0부터 1 사이의 값

2. **형식 보상**
   - 올바른 Markdown 코드 블록이며 Python 문법이 맞음: `1.0`
   - 코드 블록은 있으나 문법 오류가 있음: `0.5`
   - 올바른 코드 블록이 없음: `0.0`

따라서 코드 출력은 단순히 정답 코드뿐 아니라, 올바른 코드 블록 형식도 중요합니다.

---

### 3. 테스트 데이터와 평가 방식

논문은 다음 네 가지 벤치마크에서 테스트합니다.

| 분야 | 테스트 태스크 | 평가 방식 |
|---|---|---|
| 수학 | GSM8K | 최종 수치 답의 정확도 |
| 수학 | MATH-500 | 최종 답의 정확도 |
| 코드 | HumanEval | 생성 코드의 테스트 통과 여부 |
| 코드 | MBPP | 생성 코드의 테스트 통과 여부 |

생성 길이는 주로 다음 세 가지를 사용합니다.

- 128 토큰
- 256 토큰
- 512 토큰

또한 한 문제에 대해 여러 개의 답을 생성하여 **Pass@k**를 측정합니다.

- `Pass@1`: 한 번 생성해서 맞힐 확률
- `Pass@k`: k개의 샘플 중 적어도 하나가 맞을 확률

논문에서는 AR Order가 `k=1`에서는 임의 순서 방식과 비슷하거나 경우에 따라 낮을 수 있지만, `k`가 커질수록 더 빠르게 정답 풀이를 발견한다고 보고합니다. 이는 AR 방식이 다양한 추론 경로를 더 잘 보존한다는 의미입니다.

---

### 4. 왜 임의 순서 생성이 문제를 일으키는가?

추론 과정에는 다음과 같은 **갈림 토큰(forking token)**이 있습니다.

```text
Therefore, ...
Since, ...
However, ...
First, ...
Then, ...
```

이 토큰들은 단순한 문법 요소가 아니라, 이후 논리 전개 방향을 결정합니다.

#### AR Order의 경우

```text
문제 → 계산 1 → Therefore → 계산 2 → 최종 답
```

모델은 `Therefore`와 같은 불확실한 지점에서 바로 선택해야 합니다. 여러 가능한 방향을 시도할 수 있으므로 다양한 풀이 경로가 유지됩니다.

#### Arbitrary Order의 경우

모델은 다음처럼 생성할 수 있습니다.

```text
문제 → 계산 2 → 최종 숫자 → Therefore → 연결 문장
```

즉, 논리적으로 어려운 `Therefore`를 나중으로 미루고 확신이 높은 숫자나 문장을 먼저 생성합니다. 그러면 미래 문맥이 이미 정해져서, 나중에 `Therefore`를 생성할 때는 여러 논리적 선택지 중 하나를 탐색하는 것이 아니라 이미 결정된 내용을 연결하는 역할만 하게 됩니다.

논문은 이 현상을 **entropy degradation(엔트로피 저하)**라고 부릅니다.

- AR Order: 논리적 갈림 지점의 엔트로피가 높게 유지됨
- Arbitrary Order: 갈림 토큰의 엔트로피가 낮아짐
- 결과: 가능한 풀이 경로가 조기에 좁아짐

---

### 5. 논문에 제시된 실제 출력 사례

무작위 생성 순서가 문맥이 부족한 상태에서 토큰을 예측하면 출력이 구조적으로 깨질 수 있다고 설명합니다. 논문에 제시된 예시는 다음과 같습니다.

```text
3. Third Month:
- The number of downloads reduced by
30% compared to the second month.
\[ 180 - ( × 180 - ( - 180)
= 180 - 0.30 × 180 = 54 ... \]
```

이 출력은 다음 문제가 있습니다.

- 연산자와 피연산자가 제대로 연결되지 않음
- 괄호와 수식 구조가 깨짐
- 문맥이 완성되기 전에 일부 토큰을 결정함
- 따라서 정답 보상을 받을 가능성이 낮음

이 사례는 임의 순서가 항상 더 넓은 탐색을 제공하는 것은 아니며, 문맥이 없는 상태에서 너무 일찍 토큰을 결정하면 오히려 생성 품질이 무너질 수 있음을 보여줍니다.

---

### 6. JustGRPO의 트레이닝 입력·출력 구조

논문의 방법인 **JustGRPO**는 학습 중에만 dLLM을 AR 정책처럼 사용합니다.

#### 입력 상태

현재 위치 이전의 토큰만 보이고, 이후 토큰은 마스크 처리합니다.

```text
[문제] [첫 번째 답 토큰] [두 번째 답 토큰] [MASK] [MASK] [MASK] ...
```

예를 들어 현재 네 번째 답 토큰을 예측한다면:

```text
The answer is  [therefore]  [we]  [MASK] [MASK] ...
```

모델은 마스크된 모든 위치에 대한 예측을 만들지만, GRPO에서는 그중 **다음 토큰 하나의 확률**만 사용합니다.

#### 출력과 보상

각 문제에 대해 여러 개의 답을 생성합니다.

```text
문제 q
 ├─ 답변 1 → reward 1
 ├─ 답변 2 → reward 0
 ├─ 답변 3 → reward 1
 └─ ...
```

같은 문제에서 생성된 답들의 보상을 비교하여 상대적 advantage를 계산하고, 정답 답변의 생성 확률을 높이도록 업데이트합니다.

중요한 점은 다음과 같습니다.

- 학습 중: AR 순서로 탐색
- 모델 구조: causal mask를 추가하지 않음
- 추론 시: 다시 dLLM의 병렬 디코딩 사용 가능

즉, **학습의 탐색 방식과 추론의 실행 방식은 분리**되어 있습니다.

---

### 7. 주요 결과

LLaDA-Instruct에서 생성 길이 256일 때 JustGRPO의 결과는 다음과 같습니다.

| 태스크 | JustGRPO 정확도 |
|---|---:|
| GSM8K | 89.1% |
| MATH-500 | 45.1% |
| HumanEval | 49.4% |
| MBPP | 52.4% |

또한 JustGRPO로 학습한 모델은 추론 시 여러 토큰을 병렬로 생성해도 성능이 유지됩니다. 따라서 논문의 결론은 다음과 같습니다.

> 추론 능력을 학습할 때는 AR 순서가 탐색과 credit assignment에 유리하지만, 실제 서비스 단계에서는 dLLM의 병렬 디코딩 능력을 그대로 사용할 수 있다.

---





### 1. Main tasks studied in the paper

The paper studies how well Diffusion Large Language Models (dLLMs) explore and solve **mathematical reasoning and code-generation problems**.

It compares two decoding orders:

- **AR Order**: Generate tokens strictly from left to right.
- **Arbitrary Order**: Generate or revise tokens in an adaptive order, usually prioritizing high-confidence tokens.

The main claim is:

> Although arbitrary-order generation theoretically provides more possible trajectories, it can reduce practical reasoning coverage by allowing the model to avoid uncertain but logically important tokens.

---

### 2. Training data and input/output examples

The paper does not provide the complete text of every individual training example. It specifies the datasets and the general input/output formats.

#### 2.1 Mathematical reasoning data

**Training data**

- GSM8K
- MATH-500-related mathematical data
- Official training splits of the datasets

**Example input**

```text
A store has 180 items. The number of items sold in the third month
was 30% lower than in the second month, which was also 30% lower
than in the first month. How many items were sold in the third month?
```

A typical instruction-style prompt can be represented as:

```text
Solve the following problem. Explain your reasoning and give the final answer.

[Mathematical problem]
```

**Example output**

```text
The number sold in the second month was:
180 × 0.7 = 126.

The number sold in the third month was:
126 × 0.7 = 88.2.

Therefore, the answer is 88.2.
```

This is an illustrative example of the task format, not a verbatim training sample reproduced in the paper.

**Reward**

Mathematical tasks use a binary reward:

- Mathematically correct final answer: `reward = 1`
- Incorrect final answer: `reward = 0`

Thus, the main reward signal is whether the final answer is correct.

---

#### 2.2 Code-generation data

**Training data**

- AceCoder-87K
- A selected subset of approximately 21K difficult examples with executable unit tests

**Example input**

```text
Write a Python function that returns the sum of all even numbers
in a list.

The function must pass the provided unit tests.
```

**Example output**

```python
```python
def sum_even_numbers(nums):
    return sum(x for x in nums if x % 2 == 0)
```
```

The generated code is executed against unit tests.

**Code reward**

The total code reward consists of:

1. **Correctness reward**
   - The fraction of unit tests passed, ranging from 0 to 1.

2. **Format reward**
   - Valid Markdown code block with correct Python syntax: `1.0`
   - Valid code block but with syntax errors: `0.5`
   - No valid Markdown code block: `0.0`

---

### 3. Test datasets and evaluation

The paper evaluates four benchmarks.

| Domain | Benchmark | Evaluation |
|---|---|---|
| Mathematics | GSM8K | Final-answer accuracy |
| Mathematics | MATH-500 | Final-answer accuracy |
| Coding | HumanEval | Unit-test pass rate |
| Coding | MBPP | Unit-test pass rate |

The main generation lengths are:

- 128 tokens
- 256 tokens
- 512 tokens

The paper also evaluates **Pass@k**:

- `Pass@1`: Probability that one generated answer is correct.
- `Pass@k`: Probability that at least one of k generated answers is correct.

AR Order may be similar to or slightly worse than arbitrary order at `k=1`, but its Pass@k curve increases more strongly as `k` grows. This indicates better coverage of diverse reasoning solutions.

---

### 4. Why arbitrary-order decoding can hurt reasoning

Reasoning often depends on **forking tokens**, such as:

```text
Therefore, ...
Since, ...
However, ...
First, ...
Then, ...
```

These tokens are not merely grammatical. They determine the direction of the subsequent reasoning chain.

#### AR Order

```text
Problem → First calculation → Therefore → Second calculation → Final answer
```

The model must make a decision at the uncertain logical point. Different samples can therefore explore different reasoning branches.

#### Arbitrary Order

The model may instead generate:

```text
Problem → Later calculation → Final number → Therefore → Connecting phrase
```

It postpones the difficult logical connector and generates easier, high-confidence tokens first. Once the future context is fixed, the connector no longer represents an open branching decision; it merely fills the gap between already-decided pieces.

The paper calls this **entropy degradation**:

- AR Order preserves high entropy at logical forks.
- Arbitrary Order lowers entropy at those forks.
- The reasoning space collapses prematurely.

---

### 5. Concrete malformed-output example

The paper gives the following example of a structurally broken output under random or poorly chosen generation order:

```text
3. Third Month:
- The number of downloads reduced by
30% compared to the second month.
\[ 180 - ( × 180 - ( - 180)
= 180 - 0.30 × 180 = 54 ... \]
```

The output contains:

- Missing operands and operators
- Broken parentheses
- Corrupted mathematical notation
- Decisions made before sufficient context was available

Such outputs are unlikely to receive a positive reward, limiting what reinforcement learning can learn.

---

### 6. JustGRPO training input/output structure

JustGRPO uses the dLLM as an autoregressive policy **only during RL training**.

#### Input state

Tokens before the current position are visible, while future tokens are masked:

```text
[Question] [answer token 1] [answer token 2] [MASK] [MASK] [MASK] ...
```

For example:

```text
The answer is  [therefore]  [we]  [MASK] [MASK] ...
```

The dLLM predicts all masked positions, but GRPO uses the probability of the next token only.

#### Group-based training

For one question, the old policy generates multiple answers:

```text
Question q
 ├─ Answer 1 → reward 1
 ├─ Answer 2 → reward 0
 ├─ Answer 3 → reward 1
 └─ ...
```

GRPO compares the rewards within the group and increases the probability of tokens belonging to relatively better answers.

Crucially:

- Training exploration uses AR order.
- The model architecture is not converted into a causal transformer.
- No causal mask is imposed.
- Parallel diffusion decoding remains available at inference time.

Thus, training-time exploration and inference-time execution are decoupled.

---

### 7. Main results

On LLaDA-Instruct with generation length 256, JustGRPO achieves:

| Task | JustGRPO accuracy |
|---|---:|
| GSM8K | 89.1% |
| MATH-500 | 45.1% |
| HumanEval | 49.4% |
| MBPP | 52.4% |

The trained model also remains compatible with parallel decoding. Therefore, the paper’s main conclusion is:

> AR-order exploration is useful for learning reasoning capabilities, while the model can still use parallel dLLM decoding during inference.

<br/>
# 요약

 
1. dLLM의 임의 순서 생성을 그대로 활용하지 않고, RL 학습 중에는 토큰을 왼쪽에서 오른쪽으로 생성하는 AR 방식에 표준 GRPO를 적용한 **JustGRPO**를 제안했다.  
2. 임의 순서는 ‘Therefore’, ‘Since’ 같은 불확실성이 높은 논리적 분기 토큰을 건너뛰고 쉬운 미래 토큰부터 생성해 엔트로피와 해답 탐색 범위를 줄였지만, AR 순서는 이러한 분기를 직접 결정하게 해 Pass@k를 높였다.  
3. JustGRPO는 LLaDA-Instruct에서 GSM8K **89.1%**, MATH-500 **45.1%**, HumanEval **49.4%**, MBPP **52.4%**를 달성하면서도 추론 시 dLLM의 병렬 디코딩 능력을 유지했다.  



1. The paper proposes **JustGRPO**, which applies standard GRPO with left-to-right autoregressive generation during RL training instead of preserving arbitrary token-order generation.  
2. Arbitrary ordering often bypasses high-uncertainty logical fork tokens such as “Therefore” and “Since” by generating easier future tokens first, causing entropy degradation and reducing solution-space exploration, whereas AR order preserves these branching decisions.  
3. JustGRPO achieves **89.1% on GSM8K**, **45.1% on MATH-500**, **49.4% on HumanEval**, and **52.4% on MBPP**, while retaining the dLLM’s parallel decoding capability at inference.

<br/>
# 기타



## 1. 핵심 주장

이 논문의 중심 결론은 **dLLM의 임의 순서 생성(arbitrary-order generation)이 이론적으로는 더 넓은 해 공간을 제공하지만, 수학·코딩과 같은 일반 추론 과제에서는 실제 탐색 범위를 오히려 좁힐 수 있다**는 것입니다.

임의 순서 디코딩은 모델이 확신이 높은 쉬운 토큰을 먼저 생성하도록 만들고, “Therefore”, “Since”, “Thus”와 같은 **논리적 분기 토큰(forking tokens)** 을 뒤로 미룹니다. 그 결과 미래 문맥이 먼저 고정되면서 원래 여러 가능성이 있던 분기점의 불확실성이 줄어들고, 다양한 추론 경로를 탐색하지 못하게 됩니다. 논문은 이를 **entropy degradation**이라고 부릅니다.

---

## 2. 다이어그램 및 주요 피규어

### Figure 1. 불확실성에 대응하는 AR 순서 vs 우회하는 임의 순서

- **AR Order**: 왼쪽에서 오른쪽으로 생성하므로 불확실한 토큰을 즉시 결정해야 함.
- **Arbitrary Order**: 쉬운 토큰을 먼저 생성하고 어려운 토큰은 뒤로 미룸.
- 미래 문맥이 이미 생성된 뒤에는 원래의 여러 추론 가능성이 사실상 사라짐.

**인사이트:**  
AR의 제약은 단순한 비효율이 아니라, 불확실한 지점에서 여러 추론 경로를 시도하게 하는 **탐색 장치**로 작용합니다.

---

### Figure 2. “유연성이 클수록 추론 잠재력이 높다”는 통념에 대한 반례

- 왼쪽: AR Order가 Arbitrary Order보다 더 높은 Pass@k를 보임.
- 오른쪽: 이 관찰을 바탕으로 복잡한 diffusion 전용 RL 대신 **JustGRPO**를 제안.
- 핵심 메시지는 **추론을 학습할 때는 AR 방식으로 탐색하고, 추론 실행 시에는 dLLM의 병렬 디코딩을 유지하자**는 것입니다.

**인사이트:**  
학습 단계의 순서 제약과 추론 단계의 병렬성을 분리할 수 있습니다.

---

### Figure 3. Pass@k 기반 추론 잠재력

대상 모델은 LLaDA-Instruct, Dream-Instruct, LLaDA 1.5이며, 과제는 GSM8K, MATH-500, HumanEval, MBPP입니다.

- Pass@1에서는 Arbitrary Order가 경쟁력 있거나 더 좋을 때도 있음.
- 그러나 k가 증가할수록 AR Order의 곡선이 더 가파르게 상승.
- 즉, AR Order가 더 많은 정답 풀이를 발견하며 **solution-space coverage가 큼**.

**인사이트:**  
임의 순서 방식은 한 번의 생성에서는 그럴듯한 답을 만들 수 있지만, 많은 샘플을 생성할 때 새로운 정답 경로를 확장하는 능력은 부족합니다.

---

### Figure 4. 해 공간의 실제 중복 및 포함 관계

LLaDA-Instruct에서 Pass@1024 기준으로 비교한 결과:

- Arbitrary Order로만 해결되는 문제는 매우 적음.
- AR Order로만 해결되는 문제는 훨씬 많음.
- HumanEval에서는:
  - AR만 해결: **21.3%**
  - Arbitrary Order만 해결: **0.6%**

**인사이트:**  
임의 순서가 이론적으로는 더 큰 해 공간을 갖더라도, 실제 샘플링에서 도달 가능한 해는 AR이 발견하는 해의 작은 부분집합에 가까웠습니다.

---

### Figure 5. 블록 크기 B와 추론 잠재력

Semi-autoregressive 디코딩에서:

- B=1: 순수한 AR 순서
- B가 커질수록 한 블록 안에서 더 자유로운 토큰 선택 가능
- B가 증가할수록 Pass@k가 일관되고 단조롭게 감소

**인사이트:**  
성능 저하는 AR과 Arbitrary Order라는 두 극단 사이에서만 나타나는 현상이 아닙니다. **순서의 자유도가 커질수록 추론 탐색 범위가 줄어드는 경향**이 점진적으로 나타납니다.

---

### Figure 6. 임의 순서에서 자주 뒤로 미뤄지는 토큰

대표적인 토큰:

- “Therefore”
- “Thus”
- “Since”
- 기타 논리 연결어와 전환 표현

이들은 추론 과정에서 다음 논리적 방향을 결정하는 토큰입니다.

**인사이트:**  
모델이 무작위로 토큰을 미루는 것이 아니라, 실제로는 추론의 핵심이 되는 고불확실성 토큰을 선택적으로 회피하고 있습니다.

---

### Figure 7. Entropy degradation

- 전체 평균 토큰 엔트로피는 두 방식이 크게 다르지 않을 수 있음.
- 그러나 논리적 분기 토큰의 엔트로피는:
  - AR Order: 높게 유지
  - Arbitrary Order: 크게 감소

**인사이트:**  
문제는 전반적인 무작위성의 차이가 아니라, **추론 방향을 결정하는 소수의 토큰에서 불확실성이 조기에 사라지는 것**입니다. 임의 순서는 분기점에서 탐색하기보다 이미 만들어진 미래 문맥에 맞춰 연결어를 사후적으로 채웁니다.

---

### Figure 8. JustGRPO 이후에도 병렬 디코딩 유지

JustGRPO는 RL 학습 중에는 AR 순서를 사용하지만, 모델 구조에 causal mask를 적용하지 않습니다.

- 따라서 추론 단계에서 dLLM의 병렬 디코딩 능력을 그대로 유지.
- 병렬 토큰 수가 많아져도 JustGRPO 모델의 성능 저하가 원래 모델보다 작음.
- 예시:
  - MBPP에서 보수적 설정: 약 **+10.6%**
  - 더 공격적인 병렬 설정: 약 **+25.5%**

**인사이트:**  
AR 순서로 학습한다고 해서 모델이 일반적인 AR 모델로 바뀌는 것은 아닙니다. **학습은 순차적으로, 추론은 병렬적으로** 수행할 수 있습니다.

---

### Figure 9. 학습 효율

비교 대상:

- ESPO: diffusion 특화 근사 likelihood 사용
- JustGRPO: 더 정확한 토큰별 likelihood 계산
- JustGRPO-Fast: 엔트로피가 높은 상위 25% 토큰에서만 likelihood ratio 계산

결과:

- 기본 JustGRPO는 iteration당 계산량이 더 많지만, 정확도와 학습 시간의 균형이 좋음.
- JustGRPO-Fast는 불필요한 75%의 확률비 계산을 제거하여 더 효율적.
- 고엔트로피 토큰만으로도 추론 학습에 필요한 핵심 신호를 상당 부분 포착.

**인사이트:**  
추론은 모든 토큰에 균등하게 분산되어 있지 않고, 소수의 고불확실성 토큰에 집중되어 있습니다.

---

### Figure 10. 온도(temperature) 분석

- AR Order는 대체로 중간 온도인 **T≈0.6**에서 가장 좋은 성능.
- Arbitrary Order는 더 높은 온도에서 성능이 개선되는 경향.
- 하지만 최적 온도를 사용해도 Arbitrary Order의 Pass@k 곡선은 AR Order를 따라잡지 못함.

**인사이트:**  
임의 순서 방식은 디코딩 과정에서 이미 억제된 불확실성을 온도 증가로 보완하려 하지만, 높은 온도는 코드 문법이나 수식의 결정적인 부분까지 불필요하게 흔들 수 있습니다.

---

### Figure 11. 샘플링 알고리즘 비교

비교 방식:

- Confidence-based sampling
- Negative entropy sampling
- Top-k margin sampling
- AR sampling

결과:

- 더 정교한 샘플러가 기본 confidence-based 방식보다 Pass@k를 개선할 수 있음.
- 그러나 여전히 AR Order보다 낮음.
- AR과 가장 유사한 성능 특성을 보인 방식은 Neg-Entropy였으며, 문제별 정확도 상관계수는 **0.970**.

**인사이트:**  
샘플링 알고리즘을 개선하면 임의 순서의 문제를 일부 완화할 수 있지만, 가장 효과적인 개선 방향은 결국 AR 방식처럼 동작하는 것입니다.

---

### Figure 12. 다양한 논리적 분기 토큰에서의 엔트로피

분석 대상에는 다음과 같은 토큰들이 포함됩니다.

- “Therefore”, “Thus”, “Since”
- “However”, “Because”, “But”, “Or”
- “First”, “Then”, “Finally”
- “Calculate”, “Solving”, “Notice”, “Specifically” 등

결과:

- 거의 모든 분기 토큰에서 AR Order의 엔트로피가 더 높음.
- Arbitrary Order는 일관되게 낮은 엔트로피를 보임.

**인사이트:**  
Entropy degradation은 특정 몇 개 연결어에 국한된 현상이 아니라, 다양한 추론 분기 표현에서 반복적으로 관찰됩니다.

---

## 3. 테이블 결과

### Table 1. 여러 diffusion RL 방법과의 시스템 수준 비교

JustGRPO의 주요 성능:

| 과제 | 128 tokens | 256 tokens | 512 tokens |
|---|---:|---:|---:|
| GSM8K | 83.8 | **89.1** | **89.8** |
| MATH-500 | 39.0 | **45.1** | **45.2** |
| HumanEval | 37.8 | **49.4** | 48.7 |
| MBPP | 50.6 | **52.4** | 49.0 |

- 복잡한 diffusion 전용 RL 기법 없이도 높은 성능.
- GSM8K에서는 256 토큰 기준 89.1% 달성.
- LLaDOU, ESPO, GDPO, SPG 등과 비교해 전반적으로 경쟁력 있음.

**주의:**  
각 baseline의 데이터 규모, fine-tuning 방식, 한 단계당 unmasking 토큰 수 등이 달라 완전히 동일한 조건은 아닙니다.

---

### Table 2. 통일된 조건에서의 재현 비교

모든 방법을 다음 조건에 맞춰 비교했습니다.

- Full fine-tuning
- 한 decoding step에서 한 토큰 해제
- 생성 길이 256

결과:

- d1: GSM8K 83.8, MATH-500 39.2
- ESPO: GSM8K 84.7, MATH-500 40.3
- SPG: GSM8K 86.9, MATH-500 41.8
- **JustGRPO: GSM8K 89.1, MATH-500 45.1**

**인사이트:**  
성능 향상이 단순히 baseline의 실험 설정 차이 때문만은 아닙니다. 동일한 조건에서도 JustGRPO가 뚜렷한 우위를 보입니다.

---

### Table 3. JustGRPO 학습 설정

주요 설정:

- Base model: LLaDA 8B Instruct
- RL 알고리즘: GRPO
- Learning rate: 5×10⁻⁶
- Global batch size: 64
- Group size: 16
- Training steps: 125
- 최대 생성 길이: 256
- Sampling temperature: 1.0
- KL penalty: 0

**인사이트:**  
추가 모듈이나 복잡한 diffusion-specific objective 없이, 비교적 단순한 표준 GRPO를 직접 적용했습니다.

---

### Table 4. 무작위 순서 디코딩 비교

Pass@128:

| 방식 | GSM8K | MATH-500 | MBPP | HumanEval |
|---|---:|---:|---:|---:|
| Confidence-based | 97.0 | 71.4 | 67.1 | 67.1 |
| Fully Random | 97.5 | 70.6 | 71.8 | 64.9 |
| AR | **99.0** | **75.6** | **78.7** | **83.0** |

Pass@1에서는 Fully Random이 특히 낮습니다.

- GSM8K: 43.3%
- MATH-500: 14.1%
- MBPP: 15.0%
- HumanEval: 12.4%

**인사이트:**  
단순히 순서를 무작위화한다고 탐색 범위가 넓어지지는 않습니다. 충분한 문맥이 형성되기 전에 토큰을 예측하면 문법적으로 깨진 출력이 발생합니다.

---

### Table 5. Random order의 RL 학습 결과

GSM8K:

- LLaDA-Instruct: 78.6%
- JustGRPO-Random: 82.2%
- JustGRPO(AR): **89.1%**

**인사이트:**  
Random order는 baseline보다는 개선되지만 AR 기반 JustGRPO보다 크게 낮습니다. 따라서 중요한 것은 단순한 “임의 순서”가 아니라 **논리적·구조적 의미를 가진 left-to-right 순서**입니다.

---

### Table 6. 일반 능력 보존

| 모델 | MMLU | MMLU-Pro | HellaSwag | ARC-C |
|---|---:|---:|---:|---:|
| LLaDA-Instruct | 65.5 | 37.0 | 74.6 | 88.5 |
| JustGRPO | 65.8 | 36.7 | 74.8 | 87.5 |

**인사이트:**  
JustGRPO는 추론 성능을 높이면서도 일반적인 지식·상식 능력을 거의 유지합니다. 이는 AR 제약이 모델 구조나 추론 시 attention을 바꾸는 것이 아니라, RL 학습 과정의 탐색 방식에만 적용되기 때문입니다.

---

## 4. 어펜딕스의 추가 인사이트

### Appendix A. 데이터 및 보상

- 수학 과제는 정답 일치 여부에 따른 binary reward 사용.
- 코딩 과제는:
  - 코드 실행 정답률
  - Markdown 코드 블록 및 Python 문법 형식 보상
  을 함께 사용.
- 코드 학습에는 AceCoder-87K에서 unit test가 있는 21K개의 어려운 샘플을 사용.

---

### Appendix B. 추론 경계에 대한 추가 검증

- 온도와 샘플링 방식을 바꿔도 AR Order의 우수한 Pass@k scaling은 유지됨.
- 더 좋은 임의 순서 샘플러는 AR과 비슷한 성능 패턴을 보이지만, AR을 넘지는 못함.
- 따라서 결과가 특정 온도나 하나의 confidence sampler에만 의존한다고 보기 어렵습니다.

---

### Appendix C. Random order 분석

- Random order는 Pass@128에서 일부 과제에서 confidence-based 방식과 비슷하지만 AR보다 낮음.
- Pass@1은 크게 악화됨.
- 문맥이 충분히 형성되기 전에 토큰을 생성하여 dangling operator나 missing operand 같은 구조적 오류가 발생.
- RL에서도 Random order는 82.2%에 그쳐 AR 기반 JustGRPO의 89.1%보다 낮음.

**핵심:**  
“임의 순서”의 대안이 random order인 것은 아닙니다. 구조 없는 자유도는 탐색이 아니라 출력 붕괴를 유발할 수 있습니다.

---

### Appendix D. Bidirectional refinement 및 일반 능력

- JustGRPO는 학습 중 AR factorization을 사용하지만, 추론 시에는 dLLM의 양방향 문맥 활용과 병렬 디코딩을 보존합니다.
- 이미 생성한 토큰을 다시 수정하는 corrective decoding과 결합하는 것은 향후 연구 방향으로 제시됩니다.

---

## 5. 최종 정리

이 논문의 실험 결과는 다음 세 문장으로 요약할 수 있습니다.

1. **임의 순서 생성은 한 번의 답변 품질은 높일 수 있지만, 여러 추론 경로를 탐색하는 능력은 약화시킬 수 있다.**
2. **논리적 분기 토큰의 entropy degradation이 해 공간 축소의 주요 원인이다.**
3. **RL 학습에서는 AR 순서를 사용하고, 추론 시에는 dLLM의 병렬 디코딩을 유지하는 JustGRPO가 단순하면서도 효과적인 해법이다.**

---





## 1. Core Claim

The paper argues that although arbitrary-order generation theoretically gives diffusion LLMs a larger solution space, it can actually reduce the effective reasoning space on general reasoning tasks such as mathematics and coding.

Arbitrary-order decoding tends to generate easy, high-confidence tokens first and postpone uncertain logical connectors such as “Therefore,” “Since,” and “Thus.” Once the future context has been fixed, the original ambiguity at these logical forks disappears. The paper calls this phenomenon **entropy degradation**.

---

## 2. Figures and Main Insights

### Figure 1. Confronting vs. bypassing uncertainty

- **AR order** forces the model to resolve uncertainty immediately.
- **Arbitrary order** allows the model to generate easy tokens first and delay difficult decisions.
- Once future context is established, many alternative reasoning branches are effectively pruned.

**Insight:**  
The AR constraint is not merely an efficiency limitation; it forces exploration at important decision points.

---

### Figure 2. Less flexibility can produce better reasoning

- AR decoding achieves higher Pass@k than arbitrary-order decoding.
- This motivates **JustGRPO**, which uses AR-style exploration during RL while preserving parallel decoding at inference.

**Insight:**  
Training-time exploration order and inference-time parallelism can be separated.

---

### Figure 3. Pass@k reasoning potential

Across LLaDA-Instruct, Dream-Instruct, and LLaDA 1.5:

- Arbitrary order can be competitive at Pass@1.
- As k increases, AR decoding scales more strongly.
- AR discovers more correct reasoning solutions.

**Insight:**  
Arbitrary order may produce a good single answer, but it is less effective at expanding solution coverage through repeated sampling.

---

### Figure 4. Solution-space coverage

At Pass@1024 with LLaDA-Instruct:

- Solutions found by arbitrary order largely overlap with those found by AR.
- AR-only solutions are much more common.
- On HumanEval:
  - AR-only: **21.3%**
  - Arbitrary-order-only: **0.6%**

**Insight:**  
The theoretically larger arbitrary-order space is not effectively reachable under practical sampling. In practice, it behaves like a smaller subset of the AR solution space.

---

### Figure 5. Effect of block size

In semi-autoregressive decoding:

- B=1 corresponds to pure AR decoding.
- Larger B gives the sampler more freedom.
- Pass@k decreases consistently as B increases.

**Insight:**  
The degradation is gradual, not limited to a comparison between two extreme decoding modes. More order flexibility is consistently associated with lower reasoning potential.

---

### Figure 6. Frequently bypassed tokens

Arbitrary-order decoding frequently postpones logical connectors and transition words such as:

- “Therefore”
- “Thus”
- “Since”

**Insight:**  
The sampler selectively avoids tokens that determine the direction of the reasoning chain.

---

### Figure 7. Entropy degradation

- Global average entropy is not necessarily very different between the two modes.
- However, entropy at logical fork tokens is:
  - high under AR order,
  - sharply reduced under arbitrary order.

**Insight:**  
The problem is localized at a small number of high-entropy reasoning tokens. Arbitrary order turns an open-ended decision into a retrospective attempt to connect already-fixed future content.

---

### Figure 8. Parallel decoding is preserved

JustGRPO applies the AR constraint only during RL training and does not impose causal masking.

- The trained model remains compatible with parallel decoding.
- The performance gap over the original model can become larger under more aggressive parallelism.
- On MBPP, the gain grows from approximately **+10.6%** under conservative decoding to **+25.5%** under more aggressive parallel decoding.

**Insight:**  
The model can be trained sequentially for better exploration and still be executed in parallel at inference.

---

### Figure 9. Training efficiency

- JustGRPO uses more exact per-position probability calculations than approximation-based methods such as ESPO.
- Nevertheless, it has a competitive accuracy/wall-clock trade-off.
- **JustGRPO-Fast** computes ratios only at the top 25% highest-entropy positions and further improves efficiency.

**Insight:**  
Reasoning-relevant learning signals are concentrated in a small set of high-entropy tokens.

---

### Figure 10. Temperature analysis

- AR decoding generally performs best around **T≈0.6**.
- Arbitrary-order decoding benefits from higher temperatures.
- Even with optimized temperatures, arbitrary order does not match AR in Pass@k scaling.

**Insight:**  
Increasing temperature can partly compensate for suppressed uncertainty, but it also injects noise into deterministic components such as code syntax and mathematical expressions.

---

### Figure 11. Sampling algorithm comparison

Compared methods include confidence-based sampling, negative-entropy sampling, margin sampling, and AR sampling.

- More sophisticated samplers can improve arbitrary-order Pass@k.
- However, they still do not surpass AR.
- Negative-entropy sampling shows the strongest similarity to AR, with a per-problem accuracy correlation of **0.970**.

**Insight:**  
Improving the sampler helps, but the most effective direction is to behave more like AR decoding.

---

### Figure 12. Robustness across logical fork tokens

The analysis covers many tokens, including:

- “Therefore,” “Thus,” “Since”
- “However,” “Because,” “But,” “Or”
- “First,” “Then,” “Finally”
- “Calculate,” “Solving,” “Notice,” “Specifically”

Across these tokens:

- AR maintains higher entropy.
- Arbitrary order consistently shows lower entropy.

**Insight:**  
Entropy degradation is a broad phenomenon across many reasoning-related tokens, not an artifact of a few specific connectors.

---

## 3. Tables

### Table 1. System-level comparison

JustGRPO achieves:

| Task | 128 tokens | 256 tokens | 512 tokens |
|---|---:|---:|---:|
| GSM8K | 83.8 | **89.1** | **89.8** |
| MATH-500 | 39.0 | **45.1** | **45.2** |
| HumanEval | 37.8 | **49.4** | 48.7 |
| MBPP | 50.6 | **52.4** | 49.0 |

It performs competitively with or better than several diffusion-specific RL methods despite using standard GRPO without additional diffusion-specific machinery.

---

### Table 2. Reproduced baselines under unified settings

Under full fine-tuning, one token per decoding step, and generation length 256:

- d1: 83.8 GSM8K, 39.2 MATH-500
- ESPO: 84.7, 40.3
- SPG: 86.9, 41.8
- **JustGRPO: 89.1, 45.1**

**Insight:**  
The advantage is not merely caused by inconsistent baseline settings; it remains under a unified protocol.

---

### Table 3. JustGRPO training configuration

Main settings:

- Base model: LLaDA 8B Instruct
- Algorithm: GRPO
- Learning rate: 5×10⁻⁶
- Global batch size: 64
- Group size: 16
- Training steps: 125
- Maximum completion length: 256
- KL penalty: 0

**Insight:**  
The method is deliberately simple and does not require an auxiliary trainable module.

---

### Table 4. Fully random decoding

For Pass@128, AR is consistently strongest:

- GSM8K: AR 99.0 vs. confidence-based 97.0
- MATH-500: AR 75.6 vs. 71.4
- MBPP: AR 78.7 vs. 67.1
- HumanEval: AR 83.0 vs. 67.1

Random order performs especially poorly at Pass@1, for example:

- GSM8K: 43.3%
- MATH-500: 14.1%
- MBPP: 15.0%
- HumanEval: 12.4%

**Insight:**  
Randomizing the order does not automatically improve exploration. Predicting tokens before sufficient context exists often produces structurally invalid outputs.

---

### Table 5. Random order after RL

On GSM8K:

- LLaDA-Instruct: 78.6%
- JustGRPO-Random: 82.2%
- **JustGRPO with AR order: 89.1%**

**Insight:**  
A fixed random permutation gives some improvement over the base model, but it is far inferior to semantically meaningful left-to-right order.

---

### Table 6. General capability preservation

Performance remains nearly unchanged on non-reasoning benchmarks:

- MMLU: 65.5 → 65.8
- MMLU-Pro: 37.0 → 36.7
- HellaSwag: 74.6 → 74.8
- ARC-C: 88.5 → 87.5

**Insight:**  
JustGRPO improves reasoning without substantially damaging general knowledge or commonsense capabilities.

---

## 4. Appendix Insights

### Appendix A. Data and rewards

- Mathematical tasks use binary correctness rewards.
- Code tasks combine:
  - unit-test correctness,
  - output-format and Python-syntax rewards.
- Training uses 21K challenging AceCoder samples with verifiable unit tests.

### Appendix B. Robustness checks

- The AR advantage remains under different temperatures and sampling strategies.
- Better arbitrary-order samplers can narrow the gap but do not surpass AR.
- This suggests the main conclusion is not specific to one temperature or one sampler.

### Appendix C. Random-order analysis

- Random order is at best comparable to confidence-based decoding at high k.
- Its single-shot accuracy is much worse.
- It often generates malformed structures, such as dangling operators or missing operands.
- The same issue remains after RL training.

### Appendix D. Capability preservation and future directions

- AR factorization is used only as a training scaffold.
- Bidirectional attention and parallel inference remain available.
- Combining JustGRPO with corrective or iterative refinement is suggested as a future direction.

---

## Final Takeaway

The experimental evidence supports three conclusions:

1. **Arbitrary-order decoding may improve single-shot coherence but reduce broad reasoning exploration.**
2. **Entropy degradation at logical fork tokens is the main mechanism behind the reduced solution coverage.**
3. **JustGRPO uses AR order for RL exploration and preserves dLLM parallel decoding at inference, providing a simple and effective alternative to complex diffusion-specific RL methods.**

<br/>
# refer format:



```bibtex
@inproceedings{ni2026flexibility,
  author    = {Ni, Zanlin and Wang, Shenzhi and Yue, Yang and Yu, Tianyu and Zhao, Weilin and Hua, Yeguo and Chen, Tianyi and Song, Jun and Yu, Cheng and Zheng, Bo and Huang, Gao},
  title     = {The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  address   = {Seoul, South Korea},
  publisher = {PMLR},
  url       = {https://github.com/LeapLabTHU/JustGRPO}
}
```

### 시카고 스타일  

Zanlin Ni, Shenzhi Wang, Yang Yue, Tianyu Yu, Weilin Zhao, Yeguo Hua, Tianyi Chen, Jun Song, Cheng Yu, Bo Zheng, and Gao Huang, “The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models,” *Proceedings of the 43rd International Conference on Machine Learning*, vol. 306 (2026), published by PMLR.





