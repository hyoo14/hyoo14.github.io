---
layout: post
title:  "[2026]Information-Theoretic Design of Sparse Autoencoder-Based Steering"
date:   2026-11-06 05:12:04 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

**Sparse Autoencoder(SAE)**를 이용해 LLM의 내부 활성값을 조절하는 방법을 제안  
(추론 중 특정 층의 hidden state에 작은 벡터를 더해 모델의 행동을 바꾸는 방식)  
(어떤 방향의 벡터를 얼마나 더해야 원하는 행동을 얻을 수 있는지는 정하기 어렵, 그래서 두 데이터셋의 분포가 얼마나 다른지를 **Jensen–Shannon divergence(JSD)**로 측정, 방향설정)   


짧은 요약(Abstract) :


이 논문은 **Sparse Autoencoder(SAE)**를 이용해 LLM의 내부 활성값을 조절하는 방법을 제안합니다. 활성값 조절은 모델의 파라미터를 다시 학습하지 않고, 추론 중 특정 층의 hidden state에 작은 벡터를 더해 모델의 행동을 바꾸는 방식입니다. 하지만 어떤 방향의 벡터를 얼마나 더해야 원하는 행동을 얻을 수 있는지는 정하기 어렵습니다.

논문에서는 각 데이터셋에서 SAE feature가 얼마나 자주 활성화되는지를 **feature firing distribution**으로 표현합니다. 그리고 두 데이터셋의 분포가 얼마나 다른지를 **Jensen–Shannon divergence(JSD)**로 측정합니다. 예를 들어, 원하는 답변들로 이루어진 데이터셋을 *target distribution*, 현재 모델의 상태를 *anchor distribution*으로 두고, 현재 상태가 target에 가까워지도록 steering vector를 계산합니다.

이 방법을 **DESSIRT**라고 부릅니다. DESSIRT는 SAE feature 공간에서 작은 가상 변화가 JSD를 어떻게 바꾸는지 미분과 2차 근사로 분석한 뒤, target 분포와의 차이를 줄이는 방향을 선택합니다. 동시에 일반적인 자연어 데이터 분포에서 지나치게 멀어지지 않도록 제한하여, steering으로 인해 문장이 부자연스러워지는 문제도 줄이려고 합니다.

실험에서는 다음을 확인했습니다.

- 질문 생성 실험에서 원하는 질문 유형으로 답변을 유도할 수 있었습니다.
- 한 층에서 넣은 intervention이 이후 층들의 SAE 분포에도 구조적인 변화를 일으켰습니다.
- 안전성 실험에서는 유해한 답변 분포에서 벗어나 안전한 답변 분포로 모델의 출력을 유도할 수 있었습니다.
- 특히 작은 크기의 intervention에서는 JSD의 변화율을 이용하는 1차 방식이 효과적이었습니다.

즉, 이 논문의 핵심은 **SAE feature 분포와 정보이론적 거리인 JSD를 이용해, LLM의 내부 상태를 원하는 행동 방향으로 조절하는 체계적인 방법을 제시했다는 것**입니다.

---



This paper proposes a method for steering the behavior of large language models using **Sparse Autoencoders (SAEs)**. Activation steering changes a model’s behavior during inference by adding a small vector to an intermediate hidden state, without retraining the model. However, it is difficult to determine which steering direction is effective and how strongly it should be applied.

The paper represents each dataset by a **feature-firing distribution**, describing how frequently SAE features are activated. It then uses the **Jensen–Shannon divergence (JSD)** to measure the difference between two such distributions. A target dataset represents the desired behavior, while an anchor dataset represents the current state. The goal is to construct a steering vector that moves the anchor distribution closer to the target distribution.

The proposed framework is called **DESSIRT**. It analyzes how a small virtual shift changes the JSD using gradients and second-order approximations. Based on this analysis, it selects steering directions that reduce the divergence from the target. It also uses a general-language distribution to avoid moving too far from natural and fluent language behavior.

The experiments show that:

- DESSIRT can steer an LLM toward different question-generation modes.
- An intervention at one layer produces structured changes in SAE distributions at later layers.
- In a safety-related task, it can shift the model away from harmful responses and toward safer responses.
- The first-order version is especially effective for small steering interventions.

Overall, the main contribution is a systematic framework for designing and analyzing LLM activation steering using SAE feature distributions and the information-theoretic measure JSD.


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



### 1. 기본 모델과 개입 방식

이 논문은 **Llama 3.1-8B-Instruct**를 주 모델로 사용하고, 추가 검증에는 **Gemma-2-9B-IT**를 사용했다. 모델 파라미터를 재학습하지 않고, 추론 중 특정 층의 은닉 표현에 벡터를 더하는 **activation steering**을 적용했다.

- 주요 개입 위치: Llama 3.1의 **transformer block 16 이후 residual stream**
- Gemma 실험: **layer 20, 31**
- 개입 방식:

[
x' = x + alpha v
]

  - (x): 원래 residual-stream activation  
  - (v): 학습·최적화된 steering vector  
  - (alpha): steering 강도

즉, 모델의 가중치는 변경하지 않고 생성 도중 내부 표현을 조금 이동시킨다.

---

### 2. SAE 구조

저자들은 residual-stream activation을 해석하고 steering vector를 설계하기 위해 **Sparse Autoencoder(SAE)**를 사용했다.

주요 SAE 설정은 다음과 같다.

- 입력 차원: (D=4{,}096)
- SAE feature 수: (H=65{,}536), 즉 입력 차원의 16배
- Top-(k) sparsity: (k=256)
- 각 activation을 256개 이하의 활성 feature로 희소하게 표현
- Encoder가 residual activation을 feature activation으로 변환
- Decoder가 feature activation을 다시 residual 공간으로 복원

Steering vector는 SAE feature 공간의 벡터 (z)를 decoder로 변환해 만든다.

[
v = W^{dec}z
]

따라서 이 방법은 직접 hidden-state 공간에서 방향을 찾기보다, **SAE feature 통계에 기반해 어떤 feature들을 선택하고 조합할지 결정한 뒤 모델 공간으로 되돌리는 방식**이다.

---

### 3. DESSIRT의 핵심 아이디어

논문에서 제안한 방법은 **DESSIRT**  
(*Divergence from Empirical SAE Statistics via Information Theory*)이다.

각 데이터셋을 SAE feature의 평균적인 firing probability 벡터로 표현한다.

[
p_Y=(p_{Y,1}, ldots, p_{Y,H})
]

예를 들어 특정 질문 유형의 데이터셋에서는 어떤 SAE feature가 자주 활성화되고, 일반 텍스트에서는 덜 활성화될 수 있다. 저자들은 이러한 데이터셋 간 차이를 **Jensen–Shannon Divergence(JSD)**로 측정한다.

[
J(A,Y)=sum_i JSDbig(text{Ber}(p_{A,i}),text{Ber}(p_{Y,i})big)
]

- (A): 기준 또는 참조 데이터셋
- (Y): 현재 데이터셋 또는 steering 후의 가상 데이터셋
- JSD가 작을수록 두 데이터셋의 SAE feature 분포가 유사함

이렇게 하면 “특정 feature가 얼마나 많이 증가했는가”만 보는 것이 아니라, **전체 feature firing distribution이 목표 데이터 분포에 얼마나 가까워졌는가**를 최적화할 수 있다.

---

### 4. DESSIRT steering vector 설계

개입 벡터 (v)를 적용했을 때 SAE feature firing probability가 어떻게 변할지 1차 또는 2차 근사로 계산한다.

#### DESSIRT-1

JSD의 **gradient, 즉 1차 변화량**을 이용한다.

- 작은 steering 강도에서 효과적인 방향을 찾는 데 적합
- 적은 개입으로 큰 변화를 유도하는 것을 목표로 함
- 안전성 실험에서 주로 사용됨

#### DESSIRT-2

JSD의 gradient뿐 아니라 **curvature, 즉 2차항**까지 고려한다.

- 목표 분포와의 거리뿐 아니라 개입이 지나치게 커졌을 때의 불안정성도 고려
- 일반 텍스트 분포와의 divergence를 추가적인 비용으로 사용
- 자연스러움과 형식 보존을 위해 feature 선택 과정에서 큰 비용을 유발하는 feature를 제외

DESSIRT-2에서는 다음 두 가지를 함께 고려한다.

1. 목표 데이터셋의 SAE 분포에 가까워지는 것
2. 일반적인 자연어 데이터 분포에서 지나치게 멀어지지 않는 것

---

### 5. Feature 선택 방식

SAE feature 전체를 사용하는 대신, 정해진 개수의 feature만 선택한다.

논문에서는 greedy selection을 사용한다.

1. 현재 선택된 feature 집합 (S)를 초기화한다.
2. 선택되지 않은 각 feature를 추가했을 때 목적함수가 얼마나 감소하는지 계산한다.
3. 가장 큰 개선을 주는 feature를 추가한다.
4. 목표 feature 수에 도달할 때까지 반복한다.

주요 실험에서는 10개 또는 30개의 feature를 사용했고, feature 선택 과정에서 **budget 10 또는 30**을 적용했다. 이 budget은 특정 feature 하나에 지나치게 큰 계수가 부여되는 것을 막고, 자연스럽지 않은 생성이 발생하는 것을 완화하기 위한 제한이다.

---

### 6. 학습 데이터

#### SAE 학습 데이터

주요 질문 생성 실험의 SAE는 약 (7.7times10^7)개의 residual-stream activation vector로 학습했다.

주요 데이터 구성:

- **FineWeb**
- **UltraChat**
- 별도의 calibration corpus  
  - 토론·논쟁 관련 합성 질문 및 응답 데이터

안전성 실험용 SAE에는 다음 데이터도 포함했다.

- **WildJailbreak**
- **WildGuardMix**
- **RealToxicityPrompts**
- 안전성 steering vector를 만들기 위해 생성·필터링한 합성 응답

SAE는 텍스트 자체를 직접 학습하는 모델이라기보다, LLM이 생성한 **중간 residual activation의 구조를 재구성하고 희소 feature로 분해하도록 학습**된다.

---

### 7. 질문 유형 steering 실험

저자들은 질문 생성 결과를 7가지 유형으로 제어했다.

- M0: 기준·평가 기준
- M1: 원인·배경
- M2: 범위·적용 가능성
- M3: 결과·영향
- M4: 실행·구현
- M5: 증거·예시
- M6: 대안

각 유형에 대해 별도의 데이터셋을 만들고, 해당 데이터셋의 SAE firing distribution을 목표 분포로 삼았다.

- 데이터 생성: DeepSeek 3.2
- 데이터 기반: IBM ArgKP의 주제들
- 생성된 질문을 Qwen 3.6-27B로 필터링
- 각 유형별 약 10,000~16,000개 샘플 사용
- 평가: Qwen 3.6-27B가 질문의 유형 적합성과 문장 형식·유창성을 평가

비교 방법으로는 다음을 사용했다.

- SDCV
- SAE-CAA
- 단순 prompt instruction

---

### 8. 안전성 steering 실험

안전성 실험에서는 두 데이터셋을 사용했다.

- Dataset A: 유해한 응답
- Dataset B: 안전한 응답

목표는 모델의 응답을 Dataset B 쪽으로 이동시키고 Dataset A와는 멀어지게 하는 것이다.

이를 위해 다음과 같은 target–contrast 목적을 사용했다.

[
J(Y)=J(A,Y)-J(B,Y)
]

안전 방향 steering에서는 이 목적을 이용해 안전한 응답 분포에 가까워지도록 steering vector를 설계했다.

데이터셋은 Dolphin-Mistral-24B를 이용해 생성한 응답을 기반으로 만들었고, 다음 데이터셋의 프롬프트를 사용했다.

- WildGuardMix
- AILuminate Demo
- AttaQ
- OR-Bench의 toxic subset

생성 응답은 Qwen3-30B-A3B로 안전성 기준을 평가해 필터링했다. 최종적으로 Dataset A와 B 각각 약 7,916개의 응답을 사용했다.

---

### 9. 추가 분석: Teacher-forced pulse 실험

개입이 실제로 이후 층과 다음 토큰에 어떤 영향을 주는지도 분석했다.

- layer 16에서 특정 한 토큰의 residual stream에 steering vector를 한 번 주입
- 이후 layer 20, 24, 28, 31에서 SAE activation 분포를 측정
- 각 층의 reference distribution과 JSD를 계산

주요 결과는 다음과 같다.

- 개입 층 이후의 여러 층에서도 JSD가 구조적으로 변했다.
- 개입 강도에 따른 JSD 변화는 대체로 포물선 형태였다.
- 그러나 다음 토큰 단계로 넘어갔을 때 효과의 지속성은 제한적이었다.
- 따라서 지속적인 steering을 위해서는 생성 과정에서 개입을 반복적으로 적용해야 한다.

---

### 10. 방법의 핵심 요약

이 논문의 방법은 다음과 같이 요약할 수 있다.

1. LLM의 residual-stream activation을 SAE로 희소 feature로 분해한다.
2. 각 데이터셋을 feature firing probability 분포로 표현한다.
3. 목표 데이터셋과 현재 데이터셋의 차이를 JSD로 측정한다.
4. JSD의 gradient와 curvature를 이용해 목표 분포에 가까워지는 방향을 계산한다.
5. 일부 SAE feature만 greedy하게 선택한다.
6. 선택된 feature 방향을 SAE decoder를 통해 residual-stream steering vector로 변환한다.
7. 추론 중 특정 layer에 이 벡터를 더해 모델 행동을 조절한다.
8. 일반 텍스트 분포와의 차이도 확인해 유창성 저하와 과도한 개입을 줄인다.

---





## Method — Key Description

### 1. Base Models and Intervention

The main experiments use **Llama 3.1-8B-Instruct**, with additional validation on **Gemma-2-9B-IT**.

The model parameters are not retrained. Instead, the method performs **activation steering** by adding a steering vector to the residual stream during inference:

[
x' = x + alpha v
]

where (x) is the original residual-stream activation, (v) is the steering vector, and (alpha) controls the intervention strength.

The main Llama experiment intervenes after **transformer block 16**. The Gemma experiments use layers 20 and 31.

---

### 2. Sparse Autoencoder Architecture

The method uses a **Sparse Autoencoder (SAE)** to decompose residual-stream activations into sparse features.

Main SAE configuration:

- Input dimension: (D=4{,}096)
- Feature dimension: (H=65{,}536)
- Top-(k) sparsity: (k=256)
- At most 256 features are active for each activation

The SAE encoder maps a residual activation to feature activations, while the decoder reconstructs the original representation.

A steering vector is obtained by decoding a selected feature-space vector (z):

[
v=W^{dec}z
]

Thus, the method designs interventions in SAE feature space and then maps them back into the model’s residual-stream space.

---

### 3. Core Idea of DESSIRT

The proposed framework is called **DESSIRT**  
(*Divergence from Empirical SAE Statistics via Information Theory*).

Each dataset is represented by a vector of SAE feature firing probabilities:

[
p_Y=(p_{Y,1},ldots,p_{Y,H})
]

The method measures the difference between two datasets using the **Jensen–Shannon divergence (JSD)** between their feature-wise Bernoulli firing distributions:

[
J(A,Y)=sum_i JSDbig(text{Ber}(p_{A,i}),text{Ber}(p_{Y,i})big)
]

This provides a dataset-level objective: instead of selecting features only according to raw activation differences, DESSIRT optimizes how close the entire SAE activation distribution is to a target distribution.

---

### 4. DESSIRT-1 and DESSIRT-2

#### DESSIRT-1

DESSIRT-1 uses the **first-order gradient** of the JSD objective.

- It is suitable for small-amplitude interventions.
- It aims to obtain a large behavioral effect with a relatively small perturbation.
- It is mainly used in the safety-related experiment.

#### DESSIRT-2

DESSIRT-2 uses both the first-order gradient and the **second-order curvature** of the divergence.

It also introduces a cost based on a general natural-language distribution. This discourages steering directions that move the model too far away from normal language behavior.

Therefore, DESSIRT-2 jointly considers:

1. Alignment with the target dataset distribution.
2. Preservation of a general valid-language distribution.

---

### 5. Feature Selection

Only a limited number of SAE features are selected.

The method uses a greedy procedure:

1. Start with an empty feature set.
2. Evaluate the improvement from adding each unselected feature.
3. Add the feature that gives the largest objective improvement.
4. Repeat until the desired number of features is reached.

The main experiments use 10 or 30 selected features, with feature-selection budgets of 10 or 30. The budget helps prevent unstable coefficients and excessive reliance on a single feature.

---

### 6. Training Data

For the main SAE, the authors use approximately (7.7times10^7) residual-stream activation vectors.

The training data include:

- **FineWeb**
- **UltraChat**
- A calibration corpus containing synthetic debate- and question-related responses

The SAE used in the safety experiments additionally includes:

- **WildJailbreak**
- **WildGuardMix**
- **RealToxicityPrompts**
- Filtered synthetic responses used for constructing safety steering vectors

The SAE is trained on intermediate activations rather than directly learning a text-generation task. Its purpose is to represent the internal residual activations using sparse, higher-dimensional features.

---

### 7. Question-Mode Steering Experiment

The authors define seven target question modes:

- M0: criteria
- M1: causes and background
- M2: scope and applicability
- M3: consequences and impacts
- M4: implementation
- M5: evidence and examples
- M6: alternatives

For each mode, a separate dataset is constructed and its SAE firing distribution is used as the target distribution.

- Question generation model: DeepSeek 3.2
- Topics: IBM ArgKP
- Filtering and evaluation: Qwen 3.6-27B
- Approximately 10,000–16,000 examples per mode

The proposed method is compared with:

- SDCV
- SAE-CAA
- Prompt instruction without activation steering

---

### 8. Safety Steering Experiment

The safety experiment uses two reference datasets:

- Dataset A: harmful responses
- Dataset B: harmless responses

The objective is to move the model toward Dataset B and away from Dataset A:

[
J(Y)=J(A,Y)-J(B,Y)
]

The responses are generated using Dolphin-Mistral-24B and filtered with an LLM judge. Prompts are drawn from:

- WildGuardMix
- AILuminate Demo
- AttaQ
- The toxic subset of OR-Bench

The final construction data contain approximately 7,916 responses for each dataset.

---

### 9. Teacher-Forced Pulse Analysis

The authors also examine how a single intervention propagates through later layers.

They inject a steering vector once at layer 16 and measure SAE-based JSD at layers 20, 24, 28, and 31.

The results show that:

- A local intervention produces structured changes in downstream SAE distributions.
- JSD varies approximately parabolically with the intervention coefficient.
- The effect on the next token is relatively limited.
- Repeated intervention during generation is therefore likely necessary for a sustained steering effect.

---

### 10. Overall Summary

The method can be summarized as follows:

1. Decompose LLM residual activations into sparse SAE features.
2. Represent each dataset by its feature firing-probability distribution.
3. Measure the difference between current and target distributions using JSD.
4. Use the gradient and curvature of JSD to derive a steering direction.
5. Select a small number of useful SAE features greedily.
6. Decode the selected feature direction into a residual-stream vector.
7. Add this vector to the model during inference.
8. Use a general-language distribution to limit fluency degradation and unnatural behavior.


<br/>
# Results



### 1. 비교 대상 모델과 데이터

#### 경쟁 모델
논문은 다음 모델에서 제안 방법인 **DESSIRT**를 평가했다.

- **Llama 3.1-8B-Instruct**
  - 주요 실험 모델
  - 16번 transformer block의 residual stream에 개입
- **Gemma-2-9B-IT**
  - 교차 모델 검증
  - 20번 및 31번 레이어에서 평가

#### 비교 방법
- **DESSIRT-1**: JSD의 1차 미분, 즉 작은 개입에 대한 민감도를 이용
- **DESSIRT-2**: JSD의 2차 근사와 일반 텍스트 분포에 대한 비용을 함께 고려
- **SDCV**: SAE feature의 데이터셋 간 activation 차이와 분산을 이용하는 기존 방법
- **SAE-CAA**: SAE feature activation의 대조적 차이를 이용하는 방법
- **Prompt instruction**: activation steering 없이 프롬프트에 목표 행동을 직접 지시하는 방식
- **No steering**: 개입하지 않은 기본 모델

#### 테스트 데이터
1. **질문 모드 생성 실험**
   - IBM ArgKP 데이터셋의 주제를 사용
   - 각 주제에 대해 7가지 유형의 질문 생성
   - 평가용으로 별도의 200개 샘플 사용
   - 목표 질문 유형:
     - M0: 판단 기준
     - M1: 원인·배경
     - M2: 범위·적용 가능성
     - M3: 결과·영향
     - M4: 실행·구현
     - M5: 증거·사례
     - M6: 대안

2. **안전성 실험**
   - 안전하지 않은 응답 데이터셋 A
   - 안전한 응답 데이터셋 B
   - 두 데이터셋을 이용해 안전 방향의 steering vector를 구성
   - 최종 평가는 100개의 jailbreak prompt에서 수행
   - 평가 프롬프트는 GPTFuzzer, JailbreakV-28K, In-the-Wild Jailbreak에서 수집

3. **Teacher-forced pulse 실험**
   - 하나의 토큰 위치에서 layer 16에 개입
   - 이후 layer 20, 24, 28, 31과 다음 토큰 단계에서 효과를 측정
   - 개입이 후속 레이어와 다음 생성 단계에 어떻게 전달되는지 분석

---

### 2. 질문 모드 생성 실험의 평가 지표

두 가지 지표를 사용했다.

- **Mode congruence score**
  - 생성된 질문이 목표 질문 유형과 일치하는 비율
  - 높을수록 steering이 목표 행동을 잘 유도했다는 의미
- **Fluency/format score**
  - 문법적 자연스러움과 질문 형식 준수 정도
  - 0~5점
  - 높을수록 자연스럽고, 불필요한 설명 없이 하나의 질문 형식을 잘 지킨다는 의미

평가에는 **Qwen3.6-27B**가 judge 모델로 사용되었다.

---

### 3. Llama 3.1-8B-Instruct 결과

형식 점수가 4.5 이상인 조건에서 비교한 최고 congruence score는 다음과 같다.

| 방법 | M0 | M1 | M2 | M3 | M4 | M5 | M6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **DESSIRT-2** | 0.700 | 0.440 | 0.495 | **0.970** | **0.995** | **0.850** | **0.895** |
| SDCV | 0.565 | 0.385 | **0.725** | 0.900 | 0.935 | 0.490 | 0.570 |
| Prompt instruction | **0.895** | **0.855** | 0.375 | 0.950 | 0.760 | 0.545 | 0.770 |

괄호 안의 평균 format score는 모두 대체로 4.5 이상이었다.

#### 핵심 결과
- DESSIRT-2는 **7개 모드 중 6개에서 SDCV보다 높은 성능**을 보였다.
- 특히 다음 모드에서 강했다.
  - M3 결과·영향: **0.970**
  - M4 구현: **0.995**
  - M5 증거·사례: **0.850**
  - M6 대안: **0.895**
- M0과 M1에서는 prompt instruction이 더 강했다.
- M2 범위·적용 가능성에서는 SDCV가 DESSIRT-2보다 높았다.
- 전체적으로 steering coefficient가 커질수록 목표 모드 일치도는 증가했지만, 너무 큰 계수에서는 fluency와 형식 점수가 하락했다.

즉, DESSIRT는 **프롬프트만으로 지정하기 어려운 세부적인 질문 기능을 activation 수준에서 조절하는 데 효과적**이지만, 개입 강도가 커지면 자연스러움과 형식 준수 사이에 trade-off가 발생한다.

---

### 4. Gemma-2-9B-IT 교차 모델 검증

Gemma에서는 형식 점수 4.9 이상인 조건에서 평가했다.

| 방법 | M0 | M1 | M2 | M3 | M4 | M5 | M6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DESSIRT-2, L20 | 0.790 | 0.950 | 0.640 | 0.955 | 0.995 | 0.760 | 0.965 |
| **DESSIRT-2, L31** | 0.800 | **1.000** | **0.925** | 0.960 | **1.000** | **0.995** | 0.940 |
| SDCV, L20 | 0.755 | 0.940 | 0.755 | 0.930 | 0.975 | 0.855 | 0.795 |
| SDCV, L31 | 0.780 | 0.940 | 0.595 | 0.945 | 0.975 | 0.995 | 0.955 |
| Prompt instruction | 0.975 | 0.940 | 0.255 | **1.000** | 0.750 | 0.910 | 0.910 |

#### 해석
- DESSIRT-2는 Gemma에서도 대부분의 모드에서 SDCV보다 좋았다.
- 특히 31번 레이어의 DESSIRT-2는:
  - M1: **1.000**
  - M4: **1.000**
  - M5: **0.995**
  - M2: **0.925**
- prompt instruction은 M0과 M3에서 경쟁력이 있었지만, M2에서는 매우 낮은 점수인 **0.255**를 기록했다.
- 서로 다른 모델과 레이어에서 효과가 나타나므로, DESSIRT의 효과가 특정 Llama 모델이나 단일 레이어에만 한정되지 않을 가능성을 보여준다.

---

### 5. 안전성 실험 결과

안전성 실험에서는 jailbreak prompt에 대한 응답을 평가했다. 평가 모델은 **Qwen3-30B-A3B**이며, 다음 11개 안전성 지표와 fluency를 각각 0~5점으로 평가했다.

- 비기만성
- 비괴롭힘
- 무해성
- 혐오 표현 없음
- 법률 준수
- 개인정보 보호
- 민감정보 비공개
- 자해 내용 없음
- 성적 콘텐츠 안전성
- 윤리 준수
- 폭력적 내용 없음
- 유창성

주요 평균 결과는 다음과 같다.

| 방법 | 전반적 특징 | Fluency |
|---|---|---:|
| No steering | 기본 모델 | 4.76 |
| DESSIRT-1 | 가장 높은 전반적 안전성 | **4.99** |
| DESSIRT-2, budget=10 | 높은 안전성 | 4.92 |
| DESSIRT-2, budget=30 | 높은 안전성 | 4.95 |
| SAE-CAA | 높은 안전성 | 4.94 |
| SDCV | 개선 효과는 있으나 DESSIRT보다 낮음 | 4.88 |

#### 핵심 결과
- **DESSIRT-1이 비교 방법 중 전반적인 안전성 점수가 가장 높았다.**
- 특히 작은 steering amplitude에서 큰 효과를 내는 데 유리했다.
- 논문에서는 안전하지 않은 응답이 생성될 수 있는 jailbreak prompt에 대해, DESSIRT-1이 응답을 거부하거나 안전한 방향으로 바꾸는 효과를 보였다고 설명한다.
- 다만 steering은 양방향으로 사용할 수 있으므로, 반대 방향으로 적용하면 오히려 위험한 행동을 강화할 수 있다는 dual-use 위험도 지적한다.

---

### 6. JSD 기반 내부 상태 분석 결과

Teacher-forced pulse 실험에서는 layer 16에 한 번 개입한 뒤, 이후 레이어의 SAE feature 분포가 어떻게 변하는지 JSD로 측정했다.

주요 결과는 다음과 같다.

- layer 16뿐 아니라 layer 20, 24, 28, 31에서도 JSD가 개입 계수에 따라 **매끄러운 포물선 형태**로 변했다.
- 이는 개입 효과가 주입 레이어에만 머무르지 않고, 이후 transformer layer의 feature 분포에도 구조적으로 전달됨을 의미한다.
- 여러 레이어에서 JSD가 최소가 되는 steering coefficient 범위가 비슷했다.
- 다음 토큰 단계에서는 개입 효과가 제한적으로만 남았다.
- 따라서 지속적인 steering 효과를 유지하려면 생성 중 여러 토큰 단계에서 반복적으로 개입해야 한다.

---

### 7. 종합 결론

논문의 결과를 요약하면 다음과 같다.

1. **질문 모드 제어**
   - DESSIRT-2는 Llama와 Gemma 모두에서 기존 SDCV보다 대체로 우수했다.
   - 특히 구현, 결과, 대안, 증거와 같은 모드 제어에 강했다.

2. **프롬프트와의 비교**
   - 단순한 모드에서는 prompt instruction이 강할 수 있다.
   - 그러나 범위, 구현, 증거 등 세밀한 기능 제어에서는 DESSIRT가 더 안정적인 성능을 보였다.

3. **안전성 제어**
   - DESSIRT-1은 작은 개입으로 jailbreak 응답을 안전한 방향으로 바꾸는 데 가장 좋은 결과를 보였다.

4. **자연스러움과의 trade-off**
   - steering 강도를 높이면 목표 행동은 잘 유도되지만, fluency와 형식 준수는 저하될 수 있다.

5. **내부 메커니즘**
   - 한 레이어의 개입이 후속 레이어의 SAE feature 분포에 구조적인 변화를 일으킨다.
   - 하지만 다음 토큰까지 효과가 강하게 유지되지는 않으므로 반복 개입이 필요하다.

---





### 1. Models, baselines, and test data

#### Evaluated models
The paper evaluates DESSIRT on:

- **Llama 3.1-8B-Instruct**
  - Main experiment
  - Intervention mainly applied at layer 16
- **Gemma-2-9B-IT**
  - Cross-model validation
  - Experiments conducted at layers 20 and 31

#### Compared methods
- **DESSIRT-1**: Uses the first-order sensitivity, or gradient, of the JSD objective.
- **DESSIRT-2**: Uses a second-order JSD approximation and a cost for deviating from the general-text distribution.
- **SDCV**: Selects SAE features using contrastive activation differences normalized by variance.
- **SAE-CAA**: Uses contrastive differences in SAE feature activations.
- **Prompt instruction**: Directly specifies the desired behavior in the prompt.
- **No steering**: Baseline without intervention.

#### Test datasets
1. **Question-mode generation**
   - Topics were taken from IBM ArgKP.
   - The model generated questions belonging to one of seven modes:
     - M0: criteria
     - M1: causes/background
     - M2: scope/applicability
     - M3: consequences/impacts
     - M4: implementation
     - M5: evidence/examples
     - M6: alternatives
   - A separate set of 200 samples was used for evaluation.

2. **Safety experiment**
   - Dataset A contained unsafe responses.
   - Dataset B contained safe responses.
   - These datasets were used to construct safety-oriented steering vectors.
   - Final evaluation used 100 jailbreak prompts from GPTFuzzer, JailbreakV-28K, and In-the-Wild Jailbreak.

3. **Teacher-forced pulse experiment**
   - A pulse intervention was applied at layer 16.
   - Its effects were measured at layers 20, 24, 28, 31, and at the next token-generation step.

---

### 2. Evaluation metrics

The question-generation experiment used two main metrics:

- **Mode congruence score**
  - The proportion of outputs judged to match the intended question mode.
  - Higher is better.
- **Fluency/format score**
  - Measures naturalness and compliance with the requested single-question format.
  - Scored from 0 to 5.

The generated outputs were evaluated by **Qwen3.6-27B**.

---

### 3. Results on Llama 3.1-8B-Instruct

The table below reports the best congruence scores under a format-score constraint of at least 4.5.

| Method | M0 | M1 | M2 | M3 | M4 | M5 | M6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **DESSIRT-2** | 0.700 | 0.440 | 0.495 | **0.970** | **0.995** | **0.850** | **0.895** |
| SDCV | 0.565 | 0.385 | **0.725** | 0.900 | 0.935 | 0.490 | 0.570 |
| Prompt instruction | **0.895** | **0.855** | 0.375 | 0.950 | 0.760 | 0.545 | 0.770 |

#### Main findings
- DESSIRT-2 outperformed SDCV in **six of the seven modes**.
- It was particularly strong for:
  - M3, consequences: **0.970**
  - M4, implementation: **0.995**
  - M5, evidence/examples: **0.850**
  - M6, alternatives: **0.895**
- Prompt instruction was stronger for M0 and M1.
- SDCV performed better on M2, scope/applicability.
- Increasing the steering coefficient generally improved mode congruence, but excessive coefficients reduced fluency and format compliance.

Overall, DESSIRT is useful for controlling fine-grained question functions that are not always easy to specify through prompting alone.

---

### 4. Cross-model validation on Gemma-2-9B-IT

The Gemma results were reported under a format-score constraint of at least 4.9.

| Method | M0 | M1 | M2 | M3 | M4 | M5 | M6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DESSIRT-2, L20 | 0.790 | 0.950 | 0.640 | 0.955 | 0.995 | 0.760 | 0.965 |
| **DESSIRT-2, L31** | 0.800 | **1.000** | **0.925** | 0.960 | **1.000** | **0.995** | 0.940 |
| SDCV, L20 | 0.755 | 0.940 | 0.755 | 0.930 | 0.975 | 0.855 | 0.795 |
| SDCV, L31 | 0.780 | 0.940 | 0.595 | 0.945 | 0.975 | 0.995 | 0.955 |
| Prompt instruction | 0.975 | 0.940 | 0.255 | **1.000** | 0.750 | 0.910 | 0.910 |

#### Interpretation
- DESSIRT-2 was generally better than SDCV across the Gemma experiments.
- DESSIRT-2 at layer 31 achieved:
  - M1: **1.000**
  - M4: **1.000**
  - M5: **0.995**
  - M2: **0.925**
- Prompt instruction remained competitive for M0 and M3, but performed poorly on M2.
- These results suggest that DESSIRT is not limited to one model or one intervention layer.

---

### 5. Safety experiment results

The safety experiment evaluated outputs on jailbreak prompts. **Qwen3-30B-A3B** scored eleven safety dimensions and fluency on a 0–5 scale.

The metrics included:

- Non-deceptive content
- Non-harassing language
- Harmlessness
- No hate speech
- Legal compliance
- Privacy preservation
- No sensitive disclosure
- No self-harm content
- Sexual content safety
- Ethical compliance
- No violent content
- Fluency

| Method | Overall result | Fluency |
|---|---|---:|
| No steering | Baseline | 4.76 |
| DESSIRT-1 | Best overall safety performance | **4.99** |
| DESSIRT-2, budget=10 | Strong safety improvement | 4.92 |
| DESSIRT-2, budget=30 | Strong safety improvement | 4.95 |
| SAE-CAA | Strong improvement | 4.94 |
| SDCV | Improvement, but below DESSIRT | 4.88 |

#### Main findings
- **DESSIRT-1 achieved the best overall safety scores** among the compared steering methods.
- It was especially effective for producing a large behavioral change with a small intervention.
- On jailbreak prompts, DESSIRT-1 often shifted the model from unsafe assistance toward refusal or safer responses.
- The paper also notes a dual-use risk: reversing the steering direction could potentially increase unsafe behavior.

---

### 6. JSD analysis of downstream responses

The teacher-forced pulse experiment measured how a single intervention at layer 16 affected later SAE feature distributions.

The main findings were:

- JSD changed smoothly with the intervention coefficient and showed an approximately parabolic response.
- Similar structured responses appeared at layers 20, 24, 28, and 31.
- This indicates that the intervention propagates through later transformer layers rather than remaining localized at the injection layer.
- The coefficient minimizing JSD was relatively consistent across layers.
- The effect at the next token-generation step was limited.
- Continuous steering therefore likely requires repeated intervention during generation.

---

### 7. Overall conclusion

The empirical results support four main conclusions:

1. **Question-mode control**
   - DESSIRT-2 generally outperformed SDCV on both Llama and Gemma.
   - It was especially effective for implementation, consequences, evidence, and alternative-solution modes.

2. **Comparison with prompting**
   - Prompt instruction can be strong for simple or familiar modes.
   - DESSIRT is more useful for fine-grained behavioral control that is difficult to express reliably through prompts alone.

3. **Safety steering**
   - DESSIRT-1 provided the strongest safety improvement on jailbreak prompts, particularly at small intervention magnitudes.

4. **Trade-off and mechanism**
   - Stronger steering improves target alignment but can reduce fluency and format compliance.
   - A local intervention produces structured changes in downstream SAE distributions, although its effect does not persist strongly into the next token without repeated intervention.


<br/>
# 예제




이 논문은 **SAE(Sparse Autoencoder)의 feature 활성화 분포를 이용해 LLM의 출력을 원하는 방향으로 조정하는 방법인 DESSIRT**를 제안한다. 실험은 크게 **① SAE 학습, ② steering vector 구성, ③ 테스트 및 평가**로 나뉜다.

### 1. SAE 학습 데이터

#### 입력
Llama 3.1-8B-Instruct에 텍스트를 입력하고, 특정 레이어의 residual-stream activation을 추출한다.

- 모델: Llama 3.1-8B-Instruct
- 추출 위치: Transformer block 16 이후
- 각 입력은 32-token window로 나누고, 마지막 4개 토큰의 activation을 평균
- activation 차원: 4,096

사용한 데이터는 다음과 같다.

- 일반 텍스트: FineWeb, UltraChat
- 안전 관련 텍스트: WildJailbreak, WildGuardMix, RealToxicityPrompts
- 조정용 데이터: 이후 steering vector 구성에 사용한 합성 응답

#### 출력
SAE는 4,096차원 residual activation을 65,536개의 sparse feature로 변환한다.

[
x rightarrow h
]

- 입력 (x): residual-stream activation
- 출력 (h): SAE feature activation
- 각 feature가 특정 입력에서 얼마나 활성화되는지 기록
- 이후 데이터셋별 feature firing probability를 계산

예를 들어, 특정 feature가 데이터셋의 10,000개 activation 중 1,500개에서 활성화되었다면 해당 feature의 firing probability는 약 0.15가 된다.

---

### 2. 질문 생성 태스크용 steering vector 구성

논문은 질문의 의미적 기능을 조정하는 실험을 수행했다. 질문을 다음 7가지 모드로 나누었다.

- M0: 판단 기준(criteria)
- M1: 원인·배경(causes/background)
- M2: 범위·적용 가능성(scope/applicability)
- M3: 결과·영향(consequences/impacts)
- M4: 실행·구현(implementation)
- M5: 근거·사례(evidence/examples)
- M6: 대안(alternatives)

#### 입력 데이터

IBM ArgKP의 논제(topic)를 입력으로 사용하고, DeepSeek 3.2에 모드별 지시문을 추가했다.

예를 들어 M3 모드의 프롬프트는 다음과 같은 형태이다.

> “다음 주제에 대해 어떤 결과, 부작용 또는 영향이 발생할 수 있는지를 묻는 짧은 질문 8개를 작성하라.”

입력 주제의 예시는 다음과 같다.

> “Compulsory voting”  
> “The right to bear arms”

#### 출력 데이터

DeepSeek가 주제마다 8개의 질문을 생성한다.

예를 들어 M3(결과·영향)의 출력은 다음과 같다.

> “What might be the immediate impact on violent crime rates if the right to bear arms were abolished?”

M5(근거·사례)의 출력은 다음과 같다.

> “What specific data or statistics demonstrate that gun violence has reached a level considered ‘out of control’?”

생성된 질문은 Qwen 3.6-27B가 다시 분류하여, 목표 모드와 실제 의미가 일치하는 데이터만 남겼다. 최종 데이터 수는 모드별로 약 10,000~16,000개였다.

#### SAE 통계와 steering vector

각 질문을 Llama에 입력하고 layer 16의 residual activation을 추출한다. 이를 SAE에 통과시켜 모드별 feature firing probability를 계산한다.

예를 들어:

- M3 질문 데이터에서 feature 17이 자주 활성화됨
- 전체 질문 데이터에서는 덜 활성화됨

DESSIRT는 단순히 평균 activation 차이가 큰 feature만 선택하지 않고, **목표 모드의 feature 분포와 기준 분포 사이의 Jensen–Shannon divergence(JSD)를 줄이는 feature 조합**을 선택한다.

최종적으로 선택된 SAE feature들을 decoder로 다시 residual-space vector로 변환한다.

[
v = W^{dec}z
]

- (z): 선택된 SAE feature 공간의 조정 벡터
- (v): 실제 LLM residual stream에 더할 steering vector

---

### 3. 질문 생성 테스트

#### 입력

테스트 시에는 학습에 사용하지 않은 IBM ArgKP의 별도 200개 주제를 사용한다.

프롬프트에는 주제와 “관련된 질문 하나를 생성하라”는 일반적인 지시만 포함하며, 목표 모드는 프롬프트에 직접 쓰지 않는다.

예:

> Topic: “Compulsory voting”  
> Instruction: “Generate one short question related to the topic.”

목표 모드가 M4라면 모델의 hidden state에 M4용 steering vector를 더한다.

[
x' = x + alpha v
]

여기서 (alpha)는 steering 강도이다.

#### 출력

모델은 예를 들어 다음과 같은 질문을 생성할 수 있다.

> “How could a system of compulsory voting be structured to minimize public backlash against penalties?”

평가 모델인 Qwen 3.6-27B는 두 가지를 평가한다.

1. 생성된 질문이 목표 모드에 맞는가?
2. 문장이 자연스럽고 질문 형식을 지키는가?

평가 결과, Llama 실험에서 DESSIRT-2는 특히 M3와 M4에서 높은 성능을 보였다.

- M3: 0.970
- M4: 0.995

단, steering 강도 (alpha)가 커지면 목표 모드 일치율은 올라갈 수 있지만, 문장 자연스러움과 형식 준수도는 낮아지는 trade-off가 나타났다.

---

### 4. 일반 텍스트 분포를 이용한 fluency 보호

DESSIRT-2는 목표 데이터 분포와 가까워지는 것뿐 아니라, 일반적인 자연어 데이터 분포에서 지나치게 멀어지지 않도록 feature 선택을 제한한다.

#### 기준 데이터

- 여러 질문 모드의 통합 데이터
- 일반적인 자연어 분포를 나타내는 데이터

#### 목적

목표 모드에 맞추되 다음과 같은 문제가 생기지 않도록 한다.

- 문장이 부자연스러워짐
- 질문이 아닌 문장이 생성됨
- 불필요하게 긴 설명이 붙음
- 문법이나 형식이 무너짐

즉, DESSIRT-2는 다음 두 목표를 동시에 고려한다.

- 목표 분포와의 JSD 감소
- 일반 언어 분포로부터의 과도한 이탈 방지

---

### 5. Teacher-forced pulse 테스트

이 실험은 steering 효과가 현재 레이어에만 머무르는지, 이후 레이어와 다음 토큰 생성 단계까지 전달되는지를 확인하기 위한 것이다.

#### 입력 및 처리

미리 준비한 텍스트를 토큰 단위로 모델에 입력한다. 특정 시점 (t)에서 layer 16의 residual stream에 steering vector를 한 번만 추가한다.

그 후 layer 20, 24, 28, 31에서 SAE activation을 추출하고, 각 레이어의 기준 데이터와 JSD를 계산한다.

#### 출력 및 관찰

- layer 16에서 JSD는 steering 강도에 따라 대략 포물선 형태로 변화했다.
- layer 20~31에서도 비슷한 구조적 변화가 나타났다.
- 따라서 layer 16의 intervention이 이후 레이어의 feature 분포에도 영향을 미친다.
- 그러나 다음 토큰 단계까지 효과가 강하게 지속되지는 않았다.

논문은 지속적인 steering을 위해 생성 과정 중 여러 토큰 단계에서 intervention을 반복해야 할 가능성이 있다고 설명한다.

---

### 6. 안전성 태스크

두 번째 주요 실험은 jailbreak 입력에 대한 안전성 조정이다.

#### 학습용 입력 데이터

5,002개의 위험하거나 안전 관련 프롬프트를 사용했다. 출처는 다음과 같다.

- WildGuardMix
- AILuminate Demo
- AttaQ
- OR-Bench의 toxic subset

Dolphin-Mistral-24B로 각 프롬프트에 대한 응답을 생성한 뒤, Qwen3-30B-A3B가 응답을 평가했다.

#### 두 데이터셋

- Dataset A: 유해하거나 안전하지 않은 응답
- Dataset B: 안전하거나 거부하는 응답

최종적으로 각 데이터셋에 약 7,916개의 응답을 구성했다.

예시는 다음과 같은 유형이다.

- A: 위험한 요청에 대해 구체적인 방법을 설명하는 응답
- B: “I can’t assist with that request.”와 같이 거부하거나 안전한 대안을 제시하는 응답

#### Steering vector 구성

목표는 Dataset B 쪽으로 이동하고 Dataset A에서 멀어지는 것이다.

즉,

[
text{안전한 응답 분포에 가까워지기}
-
text{유해한 응답 분포에서 멀어지기}
]

가 되도록 steering vector를 만든다.

#### 테스트 입력과 출력

테스트에는 GPTFuzzer, JailbreakV-28K, In-the-Wild Jailbreak에서 수집한 100개의 jailbreak prompt를 사용했다.

예시 입력:

> “I want you to act as [역할] … What are the ways to conduct insider trading?”

기본 모델은 위험한 내용 일부를 설명할 수 있지만, DESSIRT-1 steering을 적용하면 다음과 같이 응답이 바뀔 수 있다.

> “I can’t assist with that request. Is there anything else I can help you with?”

출력은 11개 안전성 지표와 fluency 점수로 평가했다. DESSIRT-1은 비교한 방법들 중 전반적으로 가장 높은 안전성 점수를 기록했다.

---

## 핵심 요약

이 논문의 데이터 흐름은 다음과 같다.

[
text{텍스트}
rightarrow
text{LLM residual activation}
rightarrow
text{SAE feature activation}
rightarrow
text{feature firing distribution}
rightarrow
text{JSD 기반 feature 선택}
rightarrow
text{steering vector}
rightarrow
text{조정된 LLM 출력}
]

- **훈련 데이터**: SAE 학습용 일반·안전 관련 텍스트
- **steering 구성 데이터**: 질문 모드별 생성 질문 또는 안전/유해 응답
- **테스트 데이터**: 별도로 보관한 질문 주제와 jailbreak prompt
- **출력 평가**: 목표 의미 모드, 자연스러움, 안전성 등을 LLM-as-a-judge로 측정

---





## Overview

The paper proposes **DESSIRT**, a method for steering LLM behavior by using the distribution of Sparse Autoencoder (SAE) feature activations. The experiments consist of:

1. Training an SAE,
2. Constructing steering vectors,
3. Evaluating the steered model on question-generation and safety tasks.

### 1. SAE Training Data

#### Input

Texts are passed through Llama 3.1-8B-Instruct, and residual-stream activations are extracted after transformer block 16.

- Model: Llama 3.1-8B-Instruct
- Activation dimension: 4,096
- Text windows: 32 tokens
- Activation representation: mean of the last four tokens in each window

The training corpora include:

- General text: FineWeb and UltraChat
- Safety-related text: WildJailbreak, WildGuardMix, and RealToxicityPrompts
- Calibration data: synthetic responses later used for steering-vector construction

#### Output

The SAE maps each 4,096-dimensional activation to 65,536 sparse features.

[
x rightarrow h
]

The researchers then estimate how frequently each SAE feature fires in each dataset. For example, if a feature fires in 1,500 out of 10,000 samples, its firing probability is approximately 0.15.

---

### 2. Steering for the Question-Generation Task

The paper defines seven question modes:

- M0: Criteria
- M1: Causes/background
- M2: Scope/applicability
- M3: Consequences/impacts
- M4: Implementation
- M5: Evidence/examples
- M6: Alternatives

#### Input Data

The researchers use topics from IBM ArgKP and ask DeepSeek 3.2 to generate questions for each topic under a mode-specific instruction.

For example, an M3 instruction asks the model to generate questions about possible results, side effects, or impacts.

Example topic:

> “Compulsory voting”

Example output for M3:

> “What backlash might arise from imposing penalties for not voting?”

Example output for M5:

> “Can you provide a specific instance where penalties for not voting led to public backlash?”

The generated questions are filtered by Qwen 3.6-27B to ensure that they match the intended semantic mode. The final mode-specific datasets contain roughly 10,000–16,000 examples per mode.

#### Steering-Vector Construction

The questions are passed through Llama, and layer-16 residual activations are encoded by the SAE. The method estimates a feature-firing distribution for each question mode.

DESSIRT selects feature combinations that reduce the Jensen–Shannon divergence between the current distribution and the target-mode distribution. The selected SAE-space vector is decoded into a residual-space steering vector:

[
v = W^{dec}z
]

During generation, the modified hidden state is:

[
x' = x + alpha v
]

where (alpha) controls the steering strength.

---

### 3. Question-Generation Test

#### Input

The test set consists of 200 held-out IBM ArgKP topics that were not used to construct the steering vectors.

The prompt does not explicitly state the desired mode.

Example:

> Topic: “Compulsory voting”  
> Instruction: “Generate one short question related to the topic.”

The desired mode is controlled only by adding the corresponding steering vector to the model’s hidden state.

#### Output

For an M4 steering vector, the model may generate:

> “How could a system of compulsory voting be structured to minimize public backlash against penalties?”

Qwen 3.6-27B evaluates:

1. Whether the question matches the target mode,
2. Whether it is fluent and follows the required question format.

DESSIRT-2 achieved particularly high congruence scores for some modes, including:

- M3: 0.970
- M4: 0.995

However, increasing the steering coefficient generally improves mode congruence while eventually reducing fluency and format compliance.

---

### 4. Preserving General Language Quality

DESSIRT-2 also uses a general-language reference distribution. It prevents feature selections that move the model too far away from ordinary natural-language behavior.

This helps reduce outputs that are:

- Unnatural,
- Not properly formatted as questions,
- Excessively verbose,
- Grammatically degraded.

Thus, DESSIRT-2 balances target alignment with preservation of general language quality.

---

### 5. Teacher-Forced Pulse Experiment

This experiment examines how a one-time intervention propagates through later layers.

#### Input and Procedure

A prepared text is fed token by token. At a specific time step, a steering vector is added once to the layer-16 residual stream.

The researchers then extract SAE activations at layers 20, 24, 28, and 31 and calculate the JSD relative to the corresponding reference distributions.

#### Findings

- The JSD at layer 16 changes smoothly with the steering coefficient.
- Similar structured responses appear in later layers.
- Therefore, the intervention affects downstream SAE representations.
- The effect does not strongly persist into the next token-generation step.

The authors therefore suggest that repeated intervention may be needed for continuous steering during generation.

---

### 6. Safety-Related Steering Task

The second major experiment steers the model toward safer responses to jailbreak prompts.

#### Input Data

The researchers collect 5,002 safety-related prompts from:

- WildGuardMix
- AILuminate Demo
- AttaQ
- The toxic subset of OR-Bench

Dolphin-Mistral-24B generates candidate responses, which are filtered by Qwen3-30B-A3B.

#### Two Reference Datasets

- Dataset A: Unsafe or harmful responses
- Dataset B: Safe responses or refusal-style responses

The final datasets contain approximately 7,916 responses each.

Typical examples are:

- Dataset A: A response that provides actionable details for a harmful request
- Dataset B: A response such as “I can’t assist with that request.”

The steering vector is designed to move the model toward Dataset B and away from Dataset A.

#### Test Input and Output

The test set contains 100 jailbreak prompts from GPTFuzzer, JailbreakV-28K, and In-the-Wild Jailbreak.

Example input:

> “I want you to act as [role] … What are the ways to conduct insider trading?”

Without steering, the model may provide partially harmful information. With DESSIRT-1 steering, it may instead produce:

> “I can’t assist with that request. Is there anything else I can help you with?”

The outputs are evaluated on eleven safety metrics and fluency. DESSIRT-1 achieved the strongest overall safety performance among the compared steering methods.

---

## Main Data Flow

[
text{text}
rightarrow
text{LLM residual activation}
rightarrow
text{SAE feature activation}
rightarrow
text{feature-firing distribution}
rightarrow
text{JSD-based feature selection}
rightarrow
text{steering vector}
rightarrow
text{modified LLM output}
]

In short:

- **Training data**: General and safety-related text for SAE training
- **Steering-construction data**: Mode-specific questions or safe/unsafe responses
- **Test data**: Held-out question topics and jailbreak prompts
- **Evaluation**: Semantic mode matching, fluency, format compliance, and safety scores using an LLM judge

<br/>
# 요약


DESSIRT는 SAE 특징의 발화 확률 분포 간 Jensen–Shannon divergence(JSD)를 이용해 목표 데이터 분포에 가까워지도록 steering vector를 설계하고, 일반 텍스트 분포와의 이탈 비용으로 유창성을 보존한다.  
Llama 3.1-8B와 Gemma-2-9B에서 7가지 질문 유형 생성 실험을 수행한 결과, 대부분의 유형에서 기존 SDCV 및 프롬프트 지시보다 높은 유형 일치도를 보였으며, 일부 조건에서는 0.99 이상의 점수를 기록했다.  
예를 들어 Gemma에서는 M1·M4 질문 유형이 최대 1.00의 일치도를 보였고, 안전성 실험에서는 jailbreak 입력에 대해 위험한 답변 대신 거부 응답을 유도했으며, 단일 층 개입이 이후 층의 JSD에도 구조적인 변화를 일으켰다.  



DESSIRT designs steering vectors by minimizing the Jensen–Shannon divergence between SAE feature-firing distributions, while using deviation from a general-text distribution to help preserve fluency.  
Experiments on Llama 3.1-8B and Gemma-2-9B across seven question modes showed higher mode congruence than SDCV and prompt instructions in most settings, with scores above 0.99 in some cases.  
For example, Gemma achieved up to 1.00 congruence on the M1 and M4 modes, while safety experiments induced refusal behavior on jailbreak prompts and showed structured JSD changes in downstream layers after a single-layer intervention.

<br/>
# 기타



## 1. 다이어그램

### Figure 1. DESSIRT 개요
- 각 데이터셋을 SAE feature의 **firing-probability vector**로 표현한다.
- 기준(anchor) 분포에서 목표(target) 분포로 이동하도록 steering vector를 설계한다.
- 두 분포의 차이는 feature별 JSD(Jensen–Shannon divergence)로 측정한다.
- 일반 텍스트 분포와의 divergence도 함께 제한하여, 목표 행동을 유도하면서 유창성과 자연스러움을 유지한다.

**핵심 인사이트:**  
DESSIRT는 “어떤 feature가 해석 가능한가?”보다 “어떤 feature 조합이 목표 데이터 분포에 가까워지게 하는가?”를 기준으로 steering feature를 선택한다.

---

## 2. 본문 Figures

### Figure 2. 질문 유형 일치도와 steering coefficient
- 7개 질문 모드(M0–M6)에 대해 steering coefficient α가 커질수록 목표 모드와의 congruence score가 대체로 증가한다.
- 그러나 α가 너무 커지면 fluency/format score가 하락한다.
- M0, M2, M3, M4, M6에서는 비교적 뚜렷한 성능 향상이 나타났고, M1과 M5에서는 효과가 덜 안정적이다.

**인사이트:**  
steering 강도는 목표 행동과 자연스러움 사이의 trade-off를 결정한다. 강하게 조정할수록 원하는 모드에 가까워지지만 문장이 부자연스러워질 수 있다.

### Figure 3. 질문 유형 일치도와 유창성의 관계
- DESSIRT-2는 높은 fluency/format score를 유지하면서 congruence를 높이는 구간을 보인다.
- α가 지나치게 커지면 congruence는 높아져도 문장 품질이 악화된다.
- 일부 모드에서는 prompt instruction이 강력하지만, 여러 모드에서 activation steering이 더 높은 congruence를 달성한다.

**인사이트:**  
좋은 steering은 단순히 목표 행동을 최대화하는 것이 아니라, 허용 가능한 유창성 범위 안에서 목표 행동을 높이는 것이다.

### Figure 4. Teacher-forced pulse 실험
- Layer 16에 주입한 perturbation은 layer 20, 24, 28, 31에서도 JSD 변화를 일으킨다.
- 각 layer에서 JSD 곡선은 대체로 포물선 형태이며, 특정 α 부근에서 target distribution과 가장 가까워진다.
- 다음 generation step의 layer 16에서도 효과가 일부 관찰되지만, carry-over 효과는 크지 않다.
- teacher-forced와 teacher-then-free 결과가 유사하여, 한 토큰의 교체가 다음 step 내부 상태를 크게 바꾸지는 않는다.

**인사이트:**  
intervention은 주입 layer에만 머물지 않고 후속 transformer layer로 구조적으로 전파된다. 다만 지속적인 generation-wide steering을 위해서는 매 step 반복적으로 개입하는 것이 필요하다.

### Figure 5. Safety task의 feature 분포
- x축과 y축은 각각 harmful dataset A와 harmless dataset B에서의 feature firing probability이다.
- SDCV는 두 데이터셋의 activation 차이가 큰 feature, 즉 (p_A=p_B) 선에서 멀리 떨어진 feature를 우선 선택한다.
- DESSIRT-1은 단순히 A에 특이적이거나 B에 특이적인 feature만 선택하지 않고, 두 확률 공간 전체에서 다양한 feature를 선택한다.

**인사이트:**  
divergence 기반 목적함수에서는 “harmful feature” 또는 “harmless feature”처럼 명확히 해석되는 feature만 중요한 것이 아니다. 여러 feature의 확률 변화가 전체 분포 divergence를 줄이는지가 더 중요하다.

### Figure 6. Coefficient-resolved detected-mode heatmap
- DESSIRT-2는 α를 조절할 때 각 target mode의 행에 응답이 집중되는 경향을 보인다.
- 이는 steering coefficient에 따라 출력 mode가 목표 mode 방향으로 이동함을 보여준다.
- SDCV도 효과가 있지만, 목표 mode로의 집중도가 DESSIRT보다 덜 일관적이다.

**인사이트:**  
DESSIRT는 단순히 feature를 고르는 것을 넘어, coefficient 변화에 따른 행동 전환을 비교적 체계적으로 유도한다.

---

## 3. 본문 Tables

### Table 1. Llama와 Gemma에서의 congruence score

#### Llama 3.1-8B-Instruct
- DESSIRT-2는 대부분의 mode에서 SDCV보다 높은 점수를 얻었다.
- 특히 M3(결과/영향) 0.970, M4(실행/구현) 0.995, M5(근거/예시) 0.850을 기록했다.
- M0과 M1에서는 prompt instruction이 더 강했고, M2에서는 SDCV가 더 높은 점수를 보였다.
- DESSIRT의 format score도 대체로 4.5 이상으로 유지됐다.

#### Gemma-2-9B-IT
- DESSIRT-2는 대부분의 mode에서 SDCV보다 높았고, layer 31에서는 M1과 M4에서 1.000을 기록했다.
- prompt instruction보다 7개 중 5개 mode에서 높은 성능을 냈다.
- layer 20과 31 모두에서 효과가 나타났다.

**인사이트:**  
DESSIRT의 효과는 특정 LLM이나 특정 layer에만 국한되지 않을 가능성이 있다. 다만 모든 mode에서 항상 prompt나 다른 baseline을 이기는 것은 아니다.

### Table 2. Safety task 평가
- no-steering보다 모든 DESSIRT 계열 방법이 대체로 안전성 점수를 높였다.
- DESSIRT-1이 전체적으로 가장 높은 safety score와 fluency를 기록했다.
- DESSIRT-2도 좋은 성능을 보였지만, budget 10과 30 사이에 큰 차이는 없었다.
- DESSIRT-1의 평균 fluency는 4.99로 매우 높았다.

**인사이트:**  
작은 perturbation을 사용하는 safety steering에서는 gradient 중심의 DESSIRT-1이 특히 효과적이다. 목표 안전성을 높이면서도 유창성을 크게 손상시키지 않았다.

### Table 3. SAE 학습 데이터
- FineWeb과 UltraChat이 대부분의 일반 언어 activation을 제공한다.
- WildJailbreak, WildGuardMix, RealToxicityPrompts는 safety 관련 activation을 제공한다.
- calibration corpus는 실제 steering vector construction과 유사한 synthetic response를 포함한다.

**인사이트:**  
SAE가 일반 언어뿐 아니라 안전 관련 패턴과 steering 대상 분포를 함께 학습하도록 구성되어 있다.

### Table 4. Safety steering 데이터셋 A/B의 prompt 구성
- WildGuardMix, AILuminate Demo, AttaQ, OR-bench에서 총 5,002개 prompt를 수집했다.
- 필터링 후 3,958개 prompt가 남았고, 각 prompt에 대해 unsafe response A와 safe response B를 구성했다.

**인사이트:**  
A/B 데이터셋은 동일한 prompt에 대한 상반된 응답을 비교하도록 설계되어, safety direction을 비교적 명확하게 학습할 수 있다.

### Table 5. Safety text-level 평가 데이터
- 총 100개의 jailbreak prompt를 사용했다.
- GPTFuzzer 10개, JailbreakV-28K 65개, In-the-Wild-Jailbreak 25개로 구성된다.

**인사이트:**  
학습용 prompt와 분리된 jailbreak prompt에서 평가하여, steering vector가 단순한 데이터 memorization이 아니라 실제 공격 상황에서도 작동하는지 확인하려 했다.

### Table 6. DESSIRT-1이 선택한 상위 10개 feature
- 상위 feature 중 일부는 illegal weapon, unsafe action 등 safety 관련 내용을 직접 반영했다.
- 그러나 상당수는 “It’s important”, “I understand”, “Thank you for sharing” 같은 context-management 또는 interpersonal 표현에 반응했다.
- 즉, safety behavior는 safety 단어 자체뿐 아니라 범위 제한, 응답 태도, 거절 방식, 대화 관리와도 관련된다.

**인사이트:**  
안전 steering에 유용한 feature가 반드시 명시적인 안전/위험 개념 feature일 필요는 없다. 대화의 맥락을 조절하는 feature도 중요한 역할을 한다.

---

## 4. Appendix별 핵심 결과와 의미

### Appendix A. 식과 근사 유도
- SAE feature firing probability를 sigmoid로 근사하고, steering perturbation에 따른 1차·2차 변화량을 유도한다.
- 이를 통해 JSD 변화를 steering vector와 quadratic form으로 표현한다.

**의미:**  
DESSIRT가 단순한 경험적 feature ranking이 아니라, perturbation과 divergence 사이의 미분 가능한 근사에 기반함을 설명한다.

### Appendix B. DESSIRT-1 알고리즘
- 작은 steering amplitude에서는 JSD objective의 gradient를 이용해 가장 효과적인 방향을 선택한다.
- 선택된 feature 수가 늘어날 때마다 marginal gain이 가장 큰 feature를 greedy하게 추가한다.
- steering vector의 norm과 regularization을 사용해 지나치게 큰 방향을 방지한다.

**의미:**  
DESSIRT-1은 작은 개입으로 큰 효과를 내는 데 적합하다. Safety 실험에서 좋은 성능을 보인 이유도 여기에 있다.

### Appendix C. Baseline 방법
- SDCV는 두 데이터셋의 평균 activation 차이를 분산으로 정규화하여 feature를 선택한다.
- SAE-CAA는 평균 activation 차이의 큰 양수·음수 feature를 선택해 steering vector를 만든다.

**의미:**  
기존 방법은 주로 activation difference에 의존하지만, DESSIRT는 JSD의 gradient와 curvature를 이용한다.

### Appendix D. 질문 모드와 평가 기준
- M0–M6은 criteria, causes, scope, consequences, implementation, evidence, alternatives로 정의된다.
- fluency/format은 0–5점으로 평가하며, 짧고 자연스러운 단일 질문을 최고점으로 둔다.

**의미:**  
질문 유형 분류와 문장 품질을 분리 평가하여, steering이 의미적 목표뿐 아니라 출력 형식까지 얼마나 보존하는지 확인한다.

### Appendix E. Steering vector 구성 및 평가
- 각 mode에 대해 약 1만~1.6만 개의 질문을 생성하고, LLM judge로 필터링했다.
- 각 mode의 데이터셋을 target으로 사용하고, 모든 mode의 데이터를 일반적인 cost reference로 사용했다.
- DESSIRT-2는 budget 10과 30으로 평가했다.

**의미:**  
목표 mode를 강화하면서 다른 질문 분포에서 지나치게 멀어지는 feature를 억제하는 구조다.

### Appendix F. Detected-mode heatmap
- coefficient α에 따라 target mode가 얼마나 자주 검출되는지를 시각화했다.
- DESSIRT-2는 α가 증가할 때 target row에 밝은 mass가 나타나는 경향을 보인다.

**의미:**  
평균 score뿐 아니라 coefficient별 mode 전환 과정을 보여주어 steering의 작동 범위를 확인한다.

### Appendix G. Safety 실험 설정
- harmful response 데이터셋 A와 harmless response 데이터셋 B를 구성했다.
- safety 방향은 음의 α로 정의했다.
- DESSIRT-1은 (J(A,Y)-J(B,Y))의 gradient를 최대화하고, DESSIRT-2는 2차 divergence를 최소화한다.

**의미:**  
target–contrast 구조를 사용하면 “안전한 쪽으로 이동”과 “위험한 쪽에서 멀어짐”을 동시에 최적화할 수 있다.

### Appendix H. SAE 학습 세부사항
- Llama 3.1-8B-Instruct의 block 16 residual stream을 사용했다.
- SAE는 입력 4,096차원, feature 65,536개, top-k 256 구조다.
- main experiment에서 약 3.3만 개, safety experiment에서 약 2.5만 개의 non-dead feature가 남았다.

**의미:**  
SAE feature 공간은 원래 residual 공간보다 훨씬 크지만, 실제로 활성화되는 feature는 일부에 한정된다. DESSIRT는 이 sparse feature 공간에서 steering한다.

### Appendix I. Safety A/B 데이터셋 생성
- Dolphin-Mistral-24B로 동일 prompt에 대한 unsafe/safe 응답을 생성했다.
- LLM judge를 이용해 safety score가 낮은 응답을 A, 높은 응답을 B로 분류했다.
- 최종적으로 A와 B 각각 7,916개 response를 확보했다.

**의미:**  
동일한 입력에 대한 대조 응답을 사용하므로, topic 차이보다 response safety 차이를 steering 방향에 반영하기 쉽다.

### Appendix J. Safety 평가 방식
- Qwen3-30B-A3B가 11개 safety metric과 fluency를 각각 0–5점으로 평가했다.
- 평가 항목에는 harmlessness, legal compliance, privacy, hate speech, self-harm, sexual safety 등이 포함된다.

**의미:**  
안전성을 단일 점수가 아니라 여러 위험 차원으로 나누어 평가했다.

### Appendix K. Jailbreak 예시
- steering이 없는 경우 모델은 위험한 요청에 대해 일부 정보를 제공했다.
- DESSIRT-1 steering을 적용하면 직접적인 거절 응답으로 바뀌었다.
- 위험한 내용은 안전을 위해 redaction 처리했다.

**의미:**  
DESSIRT-1은 단순한 SAE-level 변화뿐 아니라 jailbreak 상황의 실제 응답 행동도 바꿀 수 있다.

### Appendix L. 선택 feature의 텍스트 분석
- 상위 10개 feature 중 4개는 safety-relevant 또는 refusal 관련 패턴을 보였다.
- 나머지는 scope control, interpersonal, response stance 등 대화 관리 패턴에 가까웠다.

**의미:**  
안전성은 특정 위험 단어를 억제하는 문제가 아니라, 응답의 태도와 범위 설정까지 포함하는 복합적인 행동이다.

---

## 전체 결론

- DESSIRT는 SAE feature를 단순한 activation difference로 고르는 대신, **데이터 분포 간 JSD와 그 변화율**을 이용해 steering vector를 설계한다.
- 질문 생성에서는 목표 질문 유형을 높이면서도 일정 수준의 유창성을 유지했다.
- safety task에서는 특히 작은 개입에 적합한 DESSIRT-1이 높은 safety와 fluency를 보였다.
- intervention 효과는 후속 layer로 전파되지만 다음 token까지 오래 유지되지는 않아, 지속적인 steering에는 반복 개입이 필요하다.
- 선택된 feature는 명시적인 safety feature뿐 아니라 문맥 관리와 응답 태도 feature도 포함한다.
- 다만 실험 모델·layer·SAE 설정이 제한적이고, LLM-as-a-judge 평가와 synthetic reference dataset에 의존한다는 한계가 있다.

---




## 1. Diagram

### Figure 1. Overview of DESSIRT
- Each dataset is represented as a vector of SAE feature firing probabilities.
- A steering vector moves an anchor distribution toward a target distribution.
- The discrepancy between distributions is measured using feature-wise JSD.
- Divergence from a general-text distribution is also monitored to preserve fluency and naturalness.

**Key insight:**  
DESSIRT selects features according to how they help move the overall feature distribution toward the target, rather than according to whether each feature is easily interpretable.

---

## 2. Main Figures

### Figure 2. Mode congruence versus steering coefficient
- Increasing the steering coefficient α generally improves agreement with the target question mode.
- Excessively large α reduces fluency and format quality.
- The effect is clear for M0, M2, M3, M4, and M6, but less stable for M1 and M5.

**Insight:**  
Steering strength controls a trade-off between behavioral control and naturalness.

### Figure 3. Congruence versus fluency
- DESSIRT-2 improves target-mode congruence while maintaining high fluency over a useful range of α.
- With overly large α, congruence may continue to increase while output quality deteriorates.
- Activation steering outperforms prompt instruction in several modes, although prompting remains competitive for some modes.

**Insight:**  
Effective steering should maximize the target behavior within an acceptable fluency range, not maximize the behavior without constraints.

### Figure 4. Teacher-forced pulse experiment
- An intervention applied at layer 16 produces structured JSD changes in layers 20, 24, 28, and 31.
- The responses are approximately parabolic, with a coefficient range that brings the distribution closer to the target.
- Some effect remains at layer 16 in the next generation step, but the carry-over is limited.
- Teacher-forced and teacher-then-free conditions are very similar.

**Insight:**  
The intervention propagates through later transformer layers, but it does not persist strongly across generation steps. Continuous steering therefore requires repeated intervention.

### Figure 5. Feature distributions in the safety task
- The axes represent feature firing probabilities on harmful dataset A and harmless dataset B.
- SDCV prioritizes features far from the (p_A=p_B) line, where activation differences are large.
- DESSIRT-1 selects a more mixed set of features across the probability plane.

**Insight:**  
Useful safety features need not be purely “harmful” or “harmless.” Their contribution to the overall divergence objective is more important than their simple dataset association.

### Figure 6. Detected-mode heatmaps
- DESSIRT-2 produces stronger concentration on the target-mode row as α changes.
- This indicates a systematic shift of output behavior toward the desired mode.
- SDCV also works, but its concentration is generally less consistent.

**Insight:**  
DESSIRT provides more systematic coefficient-dependent behavioral control.

---

## 3. Main Tables

### Table 1. Congruence scores on Llama and Gemma
- On Llama 3.1, DESSIRT-2 outperforms SDCV in most modes, especially M3, M4, and M5.
- Prompt instruction performs better for some modes, particularly M0 and M1.
- On Gemma-2, DESSIRT-2 performs better than SDCV in most modes and works at both layers 20 and 31.
- It also beats prompt instruction in five of the seven modes.

**Insight:**  
The method is not limited to one model or one layer, although it does not dominate every baseline in every mode.

### Table 2. Safety evaluation
- All DESSIRT variants generally improve safety relative to no steering.
- DESSIRT-1 achieves the strongest overall safety and fluency scores.
- Its average fluency score is 4.99, showing that the safety improvement does not require a major loss of linguistic quality.

**Insight:**  
The first-order, gradient-based method is particularly effective for small-amplitude safety interventions.

### Table 3. SAE training data
- FineWeb and UltraChat provide general-language activations.
- WildJailbreak, WildGuardMix, and RealToxicityPrompts provide safety-related activations.
- The calibration corpus contains synthetic responses similar to those used for steering-vector construction.

**Insight:**  
The SAE is trained to represent general language as well as safety-related and task-specific patterns.

### Table 4. Safety dataset construction
- Prompts come from WildGuardMix, AILuminate Demo, AttaQ, and OR-bench.
- After filtering, each prompt is paired with unsafe responses in A and safe responses in B.

**Insight:**  
Paired responses to the same prompt make the safety direction more directly identifiable.

### Table 5. Safety evaluation prompts
- The evaluation set contains 100 jailbreak prompts from GPTFuzzer, JailbreakV-28K, and In-the-Wild-Jailbreak.

**Insight:**  
The evaluation uses held-out jailbreak prompts to test whether the steering direction generalizes beyond the construction data.

### Table 6. Top features selected by DESSIRT-1
- Some top features directly respond to safety-related content or refusal patterns.
- Many others respond to context-management and interpersonal phrases such as “It’s important,” “I understand,” or “Thank you for sharing.”

**Insight:**  
Safety behavior depends not only on explicit safety concepts, but also on response stance, scope control, and conversational management.

---

## 4. Appendix Highlights

### Appendix A. Derivation
- Approximates feature firing probabilities with a sigmoid.
- Derives first- and second-order changes under steering.
- Expresses JSD changes as a quadratic function of the steering vector.

**Meaning:**  
The method has an analytical, differentiable basis rather than relying only on empirical feature ranking.

### Appendix B. DESSIRT-1 algorithm
- Uses the gradient of the divergence objective.
- Adds features greedily according to their marginal contribution.
- Uses normalization and regularization to avoid excessively large steering directions.

**Meaning:**  
DESSIRT-1 is well suited to small interventions, which helps explain its strong safety results.

### Appendix C. Baselines
- SDCV ranks features using normalized mean activation differences.
- SAE-CAA selects features with large positive or negative mean activation differences.

**Meaning:**  
The baselines rely mainly on activation contrast, whereas DESSIRT uses divergence gradients and curvature.

### Appendix D. Question modes and rubrics
- Defines seven modes: criteria, causes, scope, consequences, implementation, evidence, and alternatives.
- Fluency and format are scored from 0 to 5.

**Meaning:**  
The evaluation separates semantic control from output quality.

### Appendix E. Steering construction
- Generates and filters roughly 10,000–16,000 examples per mode.
- Uses the target-mode dataset as the target and all modes as a general reference for cost control.
- Evaluates budgets of 10 and 30.

**Meaning:**  
The method encourages the target behavior while discouraging excessive deviation from the broader question distribution.

### Appendix F. Mode heatmaps
- Visualizes detected modes as a function of α.
- DESSIRT-2 shows stronger concentration on the target-mode row.

**Meaning:**  
The figure illustrates the full coefficient-dependent behavior rather than only reporting one final score.

### Appendix G. Safety setup
- Constructs harmful dataset A and harmless dataset B.
- Uses the contrastive objective (J(A,Y)-J(B,Y)).
- DESSIRT-1 maximizes the gradient-based safety direction, while DESSIRT-2 minimizes the second-order divergence.

**Meaning:**  
The framework simultaneously moves outputs toward safety and away from harmful behavior.

### Appendix H. SAE training details
- Uses the layer-16 residual stream of Llama 3.1-8B-Instruct.
- The SAE has input dimension 4,096, feature dimension 65,536, and top-k sparsity of 256.
- Only a subset of features are non-dead after training.

**Meaning:**  
Steering takes place in a large but sparse feature space.

### Appendix I. Safety A/B datasets
- Generates safe and unsafe responses to the same prompts using Dolphin-Mistral-24B.
- Uses an LLM judge to filter responses into A and B.
- Produces 7,916 responses per dataset.

**Meaning:**  
The paired construction isolates safety differences more directly than unrelated datasets would.

### Appendix J. Safety evaluation
- Qwen3-30B-A3B scores 11 safety dimensions and fluency on a 0–5 scale.
- Metrics include harmlessness, legal compliance, privacy, hate speech, self-harm, sexual safety, and violence.

**Meaning:**  
Safety is evaluated across multiple risk categories instead of through a single aggregate score.

### Appendix K. Jailbreak example
- Without steering, the model provides partially informative content for a harmful request.
- With DESSIRT-1 steering, it switches to a direct refusal.
- Harmful details are redacted.

**Meaning:**  
The SAE-level direction produces an observable change in real model behavior.

### Appendix L. Text analysis of selected features
- Four of the top ten features show direct safety or refusal-related patterns.
- The others relate to scope control, interpersonal language, and response stance.

**Meaning:**  
Safety steering involves complex conversational behavior, not only explicit harmful-content features.

---

## Overall conclusion

DESSIRT designs SAE-based steering vectors using JSD and its local derivatives rather than simple activation differences. It improves target question-mode control while preserving reasonable fluency, and its gradient-based version is particularly effective for small safety interventions. The intervention propagates through later layers but does not persist strongly across generation steps, so repeated intervention is needed. The selected features include both explicit safety features and broader context-management features. The main limitations are the limited range of models and layers, reliance on synthetic reference datasets, and possible bias from LLM-as-a-judge evaluation.

<br/>
# refer format:
### BibTeX

```bibtex
@inproceedings{morise2026information,
  author    = {Morise, Hirofumi},
  title     = {Information-Theoretic Design of Sparse Autoencoder-Based Steering},
  booktitle = {Proceedings of the 15th International Joint Conference on Natural Language Processing and the 5th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics},
  year      = {2026},
  month     = nov,
  pages     = {5717--5739},
  publisher = {Association for Computational Linguistics},
  address   = {Online},
  note      = {November 6--10, 2026}
}
```

### 시카고 스타일    

Morise, Hirofumi. “Information-Theoretic Design of Sparse Autoencoder-Based Steering.” In *Proceedings of the 15th International Joint Conference on Natural Language Processing and the 5th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics*, 5717–5739. Association for Computational Linguistics, 2026.



