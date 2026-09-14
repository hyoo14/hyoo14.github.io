---
layout: post
title:  "[2025]Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach"
date:   2026-09-14 18:26:03 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 루프 트랜스포머, 리커런트 트랜스포머? astra의 그것.. CoT같은 자연어 추론 대신 트랜스포머단에서 여러번 반복 계산하도록..  
자연히 추론 과정을 텍스트적으로 볼 수는 없어서 사람들이 어려워하기도하는  


짧은 요약(Abstract) :


이 논문은 **언어 모델이 답을 생성하기 전에 잠재 공간(latent space)에서 여러 번 반복 계산하도록 만드는 구조**를 제안합니다. 기존 모델은 더 많이 생각하게 하려면 긴 Chain-of-Thought(CoT)를 텍스트 토큰으로 생성하지만, 이 논문의 모델은 문장으로 생각을 표현하지 않고 내부 벡터 상태를 반복적으로 갱신합니다.

핵심 구조는 하나의 Transformer 블록을 여러 번 반복 실행하는 **recurrent depth** 방식입니다. 따라서 테스트 시 반복 횟수 (r)을 늘리면 모델이 더 많은 계산을 수행할 수 있고, 문제에 따라 필요한 만큼 추론 깊이를 조절할 수 있습니다.

논문에서 제안한 모델은 다음과 같은 특징을 가집니다.

- 별도의 CoT 전용 학습 데이터 없이 일반적인 언어 모델 사전학습만으로 잠재적 추론을 학습함
- 긴 텍스트 컨텍스트나 긴 추론 문장을 반드시 필요로 하지 않음
- 수학·코딩처럼 복잡한 문제에서 테스트 시 계산량을 늘릴수록 성능이 향상됨
- 35억 개 파라미터 모델을 처음부터 학습하고, 8000억 개 토큰으로 실험함
- 쉬운 문제에는 적은 반복을, 어려운 문제에는 많은 반복을 사용하는 적응형 계산이 가능함
- 반복 구조 덕분에 KV-cache 공유와 자기 자신을 이용한 speculative decoding도 자연스럽게 지원함

즉, 이 논문의 핵심 주장은 **모델이 더 긴 답변을 생성하는 대신, 내부의 연속적인 벡터 공간에서 더 오래 계산하도록 하면 테스트 시 계산량을 확장하면서 추론 능력을 높일 수 있다**는 것입니다.

---



This paper proposes a language model that can perform additional reasoning directly in its **continuous latent space** before producing an answer. Instead of generating longer Chain-of-Thought text, the model repeatedly applies a shared recurrent Transformer block and updates its hidden state.

The number of recurrent iterations can be increased at test time, allowing the model to use more computation when solving difficult problems. The approach does not require specialized chain-of-thought training data and can work with relatively small context windows.

The authors train a 3.5-billion-parameter model from scratch on 800 billion tokens. They show that increasing the recurrent depth improves performance, particularly on reasoning-intensive tasks such as mathematics and coding. The model can also support per-token adaptive computation, KV-cache sharing, and self-speculative decoding.

Overall, the paper argues that **scaling test-time computation through latent recurrent reasoning can complement parameter scaling and verbalized chain-of-thought reasoning**.


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


## 1. 핵심 아이디어: 잠재 공간에서 반복적으로 추론

이 논문의 핵심은 **출력 토큰을 길게 생성하는 대신, 모델 내부의 연속적인 은닉 상태(latent state)를 여러 번 갱신하여 테스트 시 계산량을 늘리는 것**이다.

일반적인 Chain-of-Thought(CoT) 모델은 중간 추론 과정을 텍스트 토큰으로 생성한다. 반면 이 모델은 중간 과정을 사람이 읽을 수 있는 언어로 출력하지 않고, 고차원 벡터 공간의 은닉 상태 안에서 반복적으로 처리한다.

따라서 테스트 시 반복 횟수 (r)을 늘리면 모델의 계산 깊이가 증가하고, 수학·코딩·논리 문제처럼 어려운 작업에서 성능이 향상된다.

---

## 2. 모델 구조: Prelude–Recurrent Core–Coda

모델은 세 부분으로 구성된다.

### 2.1 Prelude

입력 토큰을 임베딩하고 여러 Transformer 층을 통과시켜 초기 잠재 표현 (e)를 만든다.

[
e=P(x)
]

여기서 (P)는 prelude 블록이다.

### 2.2 Recurrent Core

핵심 반복 블록 (R)은 동일한 Transformer 블록을 여러 번 재사용한다.

[
s_i=R(e,s_{i-1})
]

- (s_0): 무작위로 초기화된 잠재 상태
- (e): 모든 반복 단계에 다시 주입되는 입력 표현
- (s_i): (i)번째 반복 후의 잠재 상태
- (r): 반복 횟수

즉, 각 단계에서 모델은 입력 표현 (e)와 이전 상태 (s_{i-1})를 함께 사용해 더 깊은 계산을 수행한다.

논문에서는 입력 (e)와 상태 (s_i)를 연결(concatenation)한 뒤 adapter를 통과시켜 recurrent block에 입력한다. 입력을 매 반복마다 주입하는 방식은 반복 과정이 안정적으로 수렴하도록 돕는다.

### 2.3 Coda

마지막 잠재 상태 (s_r)를 다시 여러 Transformer 층과 출력 헤드에 통과시켜 다음 토큰의 확률을 계산한다.

[
p=C(s_r)
]

---

## 3. 파라미터 공유와 실질적 깊이

반복 블록은 같은 가중치를 계속 재사용한다. 따라서 모델의 실제 파라미터 수는 크게 늘리지 않으면서도 계산 깊이를 증가시킬 수 있다.

최종 모델의 구조는 다음과 같다.

[
(l_P,l_R,l_C)=(2,4,2)
]

즉,

- Prelude: 2개 층
- Recurrent core: 4개 층
- Coda: 2개 층

실제 고유 층은 8개뿐이지만, recurrent core를 (r=32)번 반복하면 유효 깊이는 다음과 같다.

[
2+4times32+2=132
]

따라서 이 모델은 **작은 파라미터 수로 매우 깊은 계산 그래프**를 만들 수 있다.

---

## 4. 학습 방법

### 4.1 반복 횟수를 무작위로 샘플링

테스트 시 다양한 반복 횟수를 사용할 수 있도록, 학습 중에도 반복 횟수 (r)을 고정하지 않는다.

각 학습 단계마다 반복 횟수를 확률적으로 샘플링하며, 로그정규-포아송 분포(log-normal Poisson distribution)를 사용한다.

이 분포는 보통 평균보다 적은 반복 횟수를 선택하지만, 가끔 매우 큰 반복 횟수도 선택하는 heavy-tailed 특성을 갖는다.

이를 통해 모델은 적은 계산량과 많은 계산량 모두에서 작동하도록 학습된다.

### 4.2 표준 언어 모델링 목적함수

특별한 CoT 데이터나 긴 풀이 예제를 사용하지 않고, 일반적인 다음 토큰 예측 손실을 사용한다.

[
mathcal{L}(theta)
=
mathbb{E}_{x,r}
left[
mathcal{L}(m_theta(x,r),x')
right]
]

여기서 (x')는 입력을 한 칸 이동한 다음 토큰 시퀀스다.

즉, 모델은 “생각 과정을 텍스트로 출력하라”는 별도의 감독 없이, 일반적인 언어 모델 사전학습만으로 잠재적 반복 추론 능력을 학습한다.

### 4.3 Truncated Backpropagation

반복 횟수가 커지면 모든 반복 단계에 대해 역전파할 때 메모리 비용이 커진다. 이를 줄이기 위해 마지막 (k=8)개 반복 단계에 대해서만 역전파한다.

이는 시간축 RNN에서 사용하는 truncated backpropagation through time과 유사하지만, 이 논문에서는 시간 방향이 아니라 **깊이 방향의 반복**에 적용된다.

---

## 5. Transformer 내부 설계

각 블록은 일반적인 decoder-only Transformer를 기반으로 한다.

주요 구성은 다음과 같다.

- Causal self-attention
- RoPE 위치 임베딩
- Gated SiLU MLP
- RMSNorm
- Q와 K에 대한 학습 가능한 bias
- Sandwich 형태의 정규화 구조

각 층은 대략 다음과 같은 형태다.

[
hat{x}=n_2(x+mathrm{Attn}(n_1(x)))
]

[
x'=n_4(hat{x}+mathrm{MLP}(n_3(hat{x})))
]

저자들은 대규모 recurrent 모델에서 이 sandwich normalization 구조가 반복 학습 안정성에 중요하다고 보고한다. 부적절한 정규화나 초기화는 은닉 상태가 모든 토큰에서 비슷해지는 표현 붕괴 또는 recurrent block이 반복을 무시하는 현상을 일으킬 수 있다.

---

## 6. 학습 데이터와 모델 규모

최종 모델은 약 **35억 개 파라미터**이며, 약 **8000억 토큰**으로 사전학습되었다.

데이터는 한 종류에만 집중하지 않고 다음과 같이 혼합했다.

- 일반 웹 텍스트
- 코드
- 과학 텍스트
- 수학 데이터
- 합성 교과서 데이터
- 수학·코딩·일반 지시 데이터
- Lean 및 정리 증명 관련 데이터
- 체스와 알고리즘 추론 데이터

데이터 비중은 대략 다음과 같다.

- 일반 텍스트: 28.71%
- 코드: 25.36%
- 과학 텍스트: 18.73%
- 합성 텍스트: 8.14%
- 장문 텍스트: 7.50%
- 수학: 6.14%
- 일반 instruction: 2.09%
- 수학 instruction: 1.51%

토크나이저는 65,536개 vocabulary를 갖는 BPE이며, 입력 시퀀스 길이는 4096 토큰이다.

---

## 7. 테스트 시 계산량 조절

테스트에서는 같은 모델을 여러 반복 횟수로 실행할 수 있다.

- (r=1): 기본적인 얕은 계산
- (r=4) 또는 (8): 쉬운 문제에 적합
- (r=16) 또는 (32): 더 어려운 문제에 사용
- (r=64) 이상: 추가적인 잠재 추론

쉬운 사실 회상이나 문법 문제는 적은 반복만으로 성능이 포화된다. 반면 GSM8K, HumanEval, MMLU, ARC Challenge와 같은 추론 중심 과제는 반복 횟수를 늘릴수록 성능이 향상된다.

중요한 점은 모델이 문제마다 필요한 계산량이 다르다는 것이다.

---

## 8. 추가적인 추론 효율화 기법

### 8.1 Zero-shot adaptive compute

각 토큰에서 연속된 반복 단계의 출력 분포를 비교한다. 두 단계의 다음 토큰 분포 사이 KL divergence가 충분히 작아지면 모델의 상태가 수렴했다고 보고 반복을 중단한다.

논문에서 사용한 기준은 다음과 같다.

[
D_{mathrm{KL}}(p_i|p_{i-1}) < 5times10^{-4}
]

쉬운 토큰은 일찍 종료하고, 어려운 토큰만 더 오래 반복한다. 별도의 early-exit 학습 없이도 이런 적응적 계산이 가능하다는 점이 특징이다.

### 8.2 KV-cache sharing

일반적으로 recurrent 단계마다 KV-cache를 저장하면 메모리가 크게 증가한다. 이 모델은 여러 recurrent 단계가 같은 K/V projection을 공유하므로, 일정한 개수의 cache 슬롯을 반복해서 사용할 수 있다.

예를 들어 16개 슬롯만 유지하면서 17번째 반복에서는 첫 번째 슬롯을 덮어쓴다. 이를 통해 KV-cache 메모리를 줄일 수 있다.

### 8.3 Self-speculative decoding

같은 모델을 적은 반복 횟수로 실행해 여러 토큰을 빠르게 초안 생성한 뒤, 더 많은 반복 횟수로 해당 토큰들을 검증한다.

별도의 draft model 없이 하나의 recurrent-depth 모델 자체가 초안 모델과 검증 모델 역할을 모두 수행한다.

### 8.4 Continuous latent CoT

새 토큰을 생성할 때마다 잠재 상태를 무작위로 초기화하는 대신, 이전 토큰의 마지막 상태를 다음 토큰의 초기 상태로 사용할 수 있다.

이렇게 하면 이전 토큰에서 수행한 잠재 계산을 이어받아 더 긴 계산 그래프를 구성할 수 있다. 논문은 이를 continuous chain-of-thought와 유사한 방식으로 설명한다.

---

## 9. 이 방법의 차별점

이 방법은 다음 세 가지 축을 통해 테스트 시 계산량을 확장한다.

1. **파라미터 수 확장**: 더 큰 모델 사용
2. **텍스트 기반 추론 확장**: 더 긴 CoT 생성
3. **잠재 깊이 확장**: 같은 recurrent block을 더 많이 반복

이 논문의 방법은 세 번째 축에 해당한다.

주요 장점은 다음과 같다.

- 특수한 CoT 학습 데이터가 필요하지 않음
- 긴 context window에 의존하지 않음
- 중간 추론을 토큰으로 출력하지 않음
- 작은 파라미터 수로 큰 계산량을 수행할 수 있음
- 문제와 토큰별로 계산량을 다르게 할 수 있음
- 코드, 수학, 논리 문제에서 반복 계산의 효과가 뚜렷함

다만 잠재 공간의 추론 과정은 사람이 직접 읽을 수 없기 때문에, 일반적인 CoT보다 감독과 해석이 어려울 수 있다는 한계가 있다.

---




## 1. Core idea: reasoning through latent recurrence

The central idea is to increase test-time computation by repeatedly updating a continuous hidden state instead of generating a long textual chain of thought.

A conventional Chain-of-Thought model externalizes intermediate reasoning as tokens. This model performs the intermediate computation internally in a high-dimensional latent space. Increasing the number of recurrent iterations therefore increases the effective reasoning depth without producing additional reasoning tokens.

---

## 2. Architecture: Prelude–Recurrent Core–Coda

The model consists of three parts.

### Prelude

The prelude embeds the input tokens and processes them with several Transformer layers:

[
e=P(x)
]

### Recurrent core

The recurrent core is a shared Transformer block that is applied repeatedly:

[
s_i=R(e,s_{i-1})
]

Here:

- (s_0) is a randomly initialized latent state.
- (e) is the encoded input.
- (s_i) is the latent state after the (i)-th iteration.
- (r) is the number of recurrent iterations.

The input representation (e) is injected at every iteration. In the large model, the input and recurrent state are concatenated and passed through a learned adapter before entering the recurrent Transformer block.

### Coda

After the final iteration, the coda decodes the final latent state into next-token probabilities:

[
p=C(s_r)
]

---

## 3. Parameter sharing and effective depth

The recurrent core reuses the same parameters at every iteration. This keeps the number of unique parameters relatively small while allowing the model to perform very deep computations.

The final architecture has:

[
(l_P,l_R,l_C)=(2,4,2)
]

That is:

- 2 prelude layers
- 4 recurrent-core layers
- 2 coda layers

Although the model has only 8 unique Transformer layers, using (r=32) recurrent iterations gives an effective depth of:

[
2+4times32+2=132
]

Thus, the model obtains a deep computation graph without storing a separate set of parameters for every depth.

---

## 4. Training procedure

### Randomized recurrence during training

The number of recurrent iterations is randomly sampled for each training step. The authors use a log-normal Poisson distribution, which usually samples relatively small depths but occasionally produces much larger depths.

This trains the model to operate under different compute budgets and enables test-time scaling.

### Standard language-modeling objective

The model is trained with the usual next-token prediction loss:

[
mathcal{L}(theta)
=
mathbb{E}_{x,r}
left[
mathcal{L}(m_theta(x,r),x')
right]
]

No specialized chain-of-thought demonstrations are required. The model learns latent reasoning through ordinary language-model pretraining with randomized unrolling.

### Truncated backpropagation

To control memory and training cost, gradients are propagated only through the last (k=8) recurrent iterations.

This is analogous to truncated backpropagation through time, except that the recurrence is along model depth rather than sequence time.

---

## 5. Transformer and stabilization details

Each block uses standard Transformer components:

- Causal self-attention
- RoPE positional embeddings
- Gated SiLU MLPs
- RMSNorm
- Learned query and key biases
- Sandwich-style normalization

The sandwich block is important for large-scale stability. The authors found that inappropriate normalization or initialization can cause hidden-state collapse, where all tokens receive nearly identical representations, or can cause the model to ignore the recurrent state entirely.

---

## 6. Training data and model scale

The final model has approximately **3.5 billion parameters** and is trained on roughly **800 billion tokens**.

The training mixture includes:

- General web text
- Source code
- Scientific text
- Mathematical data
- Synthetic textbooks
- General instruction data
- Math and coding instruction data
- Lean and theorem-proving data
- Chess and algorithmic-reasoning data

The tokenizer uses a 65,536-token BPE vocabulary, and the training sequence length is 4096 tokens.

---

## 7. Test-time compute scaling

At inference time, the same model can be run with different recurrence depths:

- (r=1): shallow computation
- (r=4) or (8): suitable for easier tasks
- (r=16) or (32): more difficult reasoning tasks
- (r=64+): extended latent computation

Easy recall and language tasks often saturate quickly. Math, coding, logic, and multi-step reasoning tasks benefit more from additional iterations.

The model therefore learns to use additional computation when the task requires it.

---

## 8. Inference-time efficiency techniques

### Zero-shot adaptive computation

The model compares the next-token distributions from two consecutive recurrent steps. If their KL divergence falls below a threshold, the model stops iterating for that token.

The paper uses:

[
D_{mathrm{KL}}(p_i|p_{i-1}) < 5times10^{-4}
]

This enables easy tokens to exit early while difficult tokens receive more computation, without training a separate early-exit mechanism.

### KV-cache sharing

Instead of storing a separate KV-cache entry for every recurrent step, the model reuses a fixed number of cache slots. This reduces memory usage while preserving performance reasonably well.

### Self-speculative decoding

The model can use a small number of recurrent iterations to draft multiple tokens and then use more iterations to verify them. No separate draft model is required.

### Continuous latent CoT

The final latent state from one generated token can be reused as the initial state for the next token. This allows latent computation to continue across tokens and creates a deeper computational process without explicitly generating a textual chain of thought.

---

## 9. Main contribution and limitation

The paper introduces recurrent depth as a third way to scale language-model computation, alongside:

1. Scaling parameter count
2. Generating longer verbalized chains of thought
3. Repeating a shared latent recurrent block

Its main advantages are that it does not require specialized CoT data, does not rely on long context windows, and can allocate computation adaptively.

The main limitation is interpretability: the model’s reasoning occurs in latent vectors rather than human-readable intermediate text, making oversight and analysis more difficult than with explicit chains of thought.


<br/>
# Results



### 1. 실험 모델과 비교 대상

논문은 **latent recurrent depth** 구조를 가진 3.5B 파라미터 언어 모델을 학습했다.

- 학습 데이터: 약 **800B 토큰**
- 구조: `(Prelude 2층, Recurrent Core 4층, Coda 2층)`
- 반복 횟수 (r): 테스트 시 1, 4, 8, 16, 32회 등으로 조절
- (r)이 클수록 같은 recurrent block을 더 많이 반복하므로 테스트 시 계산량이 증가한다.
- 모델의 실제 파라미터 수는 3.5B이지만, (r=32)에서는 유효 깊이가  
  [
  2+4times32+2=132
  ]
  층에 해당한다.

비교 모델로는 Pythia, Amber, OLMo-1/7B, OLMo-2-7B/32B, StarCoder2 등이 사용되었다. 다만 비교 모델마다 학습 토큰 수와 데이터 구성이 크게 다르므로, 모든 비교가 완전히 공정한 것은 아니다.

---

## 2. 일반 언어·상식 벤치마크

주요 평가 데이터셋은 다음과 같다.

- **ARC-Easy / ARC-Challenge**: 초등·중등 수준 과학 및 추론
- **HellaSwag**: 문맥에 맞는 문장 완성
- **MMLU**: 다양한 전문·학술 분야의 객관식 문제
- **OpenBookQA**: 과학 지식 및 추론
- **PiQA**: 물리적 상식
- **SciQ**: 과학 지식
- **WinoGrande**: 대명사·상식 추론

### 성능 변화

| 반복 횟수 | ARC-E | ARC-C | HellaSwag | MMLU | OBQA | PiQA | SciQ | WinoGrande |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (r=4) | 49.07 | 27.99 | 43.46 | 23.39 | 28.20 | 64.96 | 80.00 | 55.24 |
| (r=8) | 65.11 | 35.15 | 58.54 | 25.29 | 35.40 | 73.45 | 92.10 | 55.64 |
| (r=16) | 69.49 | 37.71 | 64.67 | 31.25 | 37.60 | 75.79 | 93.90 | 57.77 |
| (r=32) | **69.91** | **38.23** | **65.21** | **31.38** | **38.80** | **76.22** | **93.50** | **59.43** |

핵심 결과는 **반복 횟수를 늘릴수록 대부분의 벤치마크 성능이 향상**된다는 점이다. 특히 ARC-Challenge와 MMLU처럼 추론이 필요한 과제에서 증가 폭이 컸다.

반면 SciQ처럼 비교적 직접적인 과학 지식 회상 문제는 적은 반복 횟수에서도 빠르게 성능이 포화되었다. 즉, 이 모델은 문제의 난이도에 따라 필요한 latent computation 양이 달라지는 경향을 보였다.

(r=32) 기준으로 모델은 이전 세대 Pythia 모델과 대체로 비슷하거나 더 높은 성능을 보였고, 일부 지표에서는 OLMo-7B와 비슷한 수준에 도달했다. 그러나 OLMo-2와 같이 더 많은 데이터와 개선된 학습 방법을 사용한 최신 모델보다는 전반적으로 낮았다.

---

## 3. 수학적 추론 결과

수학 평가에는 다음 데이터셋이 사용되었다.

- **GSM8K**: 초등 수준 수학 문장제
- **GSM8K CoT**: chain-of-thought 형식의 GSM8K
- **Minerva MATH**: 고등·대학 수준 수학 문제
- **MathQA**: 수학 문제 풀이 및 연산 선택

### (r=32) 성능

| 모델 | GSM8K | GSM8K CoT | Minerva MATH | MathQA |
|---|---:|---:|---:|---:|
| Ours, (r=32) | 28.05 / 28.20 | 32.60 / 34.57 | 12.58 | 26.60 |
| Ours, template 사용 | 24.87 / 38.13 | 34.87 / 42.84 | 11.24 | 27.97 |
| OLMo-2-7B | 66.72 / 66.79 | 61.94 / 66.19 | 19.08 | 37.59 |
| OLMo-2-32B | 28.43 / 28.51 | 26.76 / 32.37 | 5.72 | 33.90 |

GSM8K의 두 숫자는 논문에서 보고한 **flexible match / strict match** 결과다. 평가 형식과 템플릿에 따라 결과가 상당히 달라진다.

이 모델은 OLMo-2-7B보다는 낮지만, 3.5B 모델이라는 점을 고려하면 수학 추론에서 유의미한 성능을 보였다. 또한 논문은 테스트 시 recurrent iteration을 늘릴 때 수학 성능이 지속적으로 향상된다고 보고한다. 이는 단순한 지식 회상보다는 **추론에 추가 계산을 활용하는 능력**이 모델에 학습되었음을 시사한다.

특히 동일한 학습 설정의 non-recurrent baseline과 비교했을 때 차이가 뚜렷하다.

- 180B 토큰 시점의 non-recurrent baseline: GSM8K CoT **1.82 / 2.20**
- 같은 시점의 recurrent model, (r=32): **9.02 / 10.24**
- 최종 recurrent model, (r=32): **34.80 / 42.08**

논문은 recurrent model이 어려운 GSM8K 문제에서 non-recurrent baseline보다 약 **5배 높은 성능**을 보였다고 강조한다.

---

## 4. 코드 생성 결과

코드 평가에는 다음 데이터셋이 사용되었다.

- **MBPP**: 간단한 Python 프로그램 생성
- **HumanEval**: 함수 구현 문제

평가지표는 생성한 코드가 테스트를 통과하는 비율인 **pass@1**이다.

| 모델 | MBPP | HumanEval |
|---|---:|---:|
| Ours, (r=32) | **24.80** | **23.17** |
| OLMo-2-7B | 21.80 | 10.36 |
| OLMo-2-32B | 19.80 | 17.68 |
| StarCoder2-3B | 43.00 | 31.09 |
| StarCoder2-7B | 43.80 | 31.70 |

제안 모델은 일반 목적 오픈소스 모델들과 비교하면 좋은 성능을 보였으며, 특히 OLMo 계열보다 HumanEval에서 높았다. 그러나 코드 전용 모델인 StarCoder2보다는 낮았다. StarCoder2는 수조 개 수준의 코드 중심 데이터를 학습했기 때문에 이 비교는 모델의 전문화 정도를 함께 고려해야 한다.

---

## 5. 반복 횟수에 따른 테스트 시점 계산 확장

이 논문의 가장 중요한 결과는 테스트 시 (r)을 증가시키면 별도의 모델 재학습 없이 성능이 향상된다는 점이다.

- (r=1): recurrent reasoning을 거의 사용하지 않는 상태
- (r=4): 일부 기본적인 개선
- (r=8sim16): 일반적인 언어·상식 문제에서 큰 성능 향상
- (r=32) 이상: 수학, 코드, MMLU, ARC-Challenge 등 어려운 문제에서 추가 개선

작업별로 포화 지점은 달랐다.

- **SciQ, BLiMP 등 쉬운 문제**: 적은 반복으로 빠르게 포화
- **HellaSwag**: 약 (r=8)에서 거의 포화
- **ARC-Challenge, MMLU, GSM8K, HumanEval**: 더 많은 반복에서 계속 개선
- **Mastermind 같은 복잡한 추론 과제**: 높은 테스트 계산량의 이점을 크게 활용

또한 모델은 최대 (r=1024), 유효 깊이 약 4100층까지도 안정적으로 실행되었다. 다만 일반 벤치마크 성능은 대체로 (r=64sim72) 부근에서 포화되었다.

---

## 6. 동일 파라미터의 non-recurrent 모델과 비교

논문은 동일한 파라미터 규모와 학습 환경에서 recurrent block을 한 번만 실행하는 non-recurrent baseline도 학습했다.

비교 결과:

- 쉬운 지식 회상 과제에서는 두 모델의 차이가 상대적으로 작았다.
- ARC-Challenge, GSM8K처럼 다단계 추론이 필요한 과제에서는 recurrent model이 크게 우세했다.
- 최종 recurrent model을 (r=1)로 실행하면 성능이 크게 떨어지고, (r=32)에서 성능이 회복되었다.

예를 들어 최종 모델의 GSM8K CoT 결과는 다음과 같다.

- (r=1): **0.00 / 0.00**
- (r=32): **34.80 / 42.08**

이는 성능 향상이 단순히 모델 파라미터나 학습 데이터에만 의한 것이 아니라, **recurrent block을 반복 실행하면서 추가 계산을 수행하는 데서 발생**한다는 근거로 제시된다.

---

## 7. 효율성과 추가 기능

이 구조는 성능뿐 아니라 추론 최적화에도 유리한 특성을 보였다.

### Zero-shot adaptive compute

각 토큰의 successive recurrence 결과 사이의 KL divergence가 작아지면 계산을 조기에 중단한다.

- 쉬운 질문이나 토큰은 적은 반복으로 종료
- 어려운 질문이나 핵심 추론 토큰은 더 많은 반복 수행
- 별도의 early-exit 학습 없이 적용 가능

예를 들어 MMLU에서 고등학교 수학 문제는 평균적으로 더 빨리 종료되었고, 도덕적 시나리오 문제는 더 많은 반복을 필요로 했다.

### KV-cache sharing

recurrent step 사이의 KV-cache를 공유하거나 제한된 크기로 재사용해 메모리를 줄였다. 논문에서는 GSM8K와 MT-Bench에서 성능 저하가 크지 않았으며, 일부 설정에서는 소폭 향상도 관찰했다.

### Self-speculative decoding

적은 반복 횟수로 초안을 생성한 뒤, 더 많은 반복 횟수로 결과를 검증하는 speculative decoding이 가능하다. 별도의 draft model 없이 동일 모델의 recurrent depth를 이용한다.

---

## 8. 종합 평가

이 논문의 실험은 latent recurrent reasoning이 다음과 같은 가능성을 보여준다.

1. **테스트 시 계산량을 반복 횟수로 연속적으로 확장할 수 있다.**
2. 추가 반복은 특히 수학·코드·다단계 추론 과제에서 효과적이다.
3. 동일 파라미터의 fixed-depth 모델보다 어려운 문제에서 훨씬 강한 성능을 보인다.
4. Chain-of-thought 토큰을 길게 생성하지 않고도 latent space에서 추론할 수 있다.
5. 토큰별 adaptive compute, KV-cache sharing, self-speculative decoding을 자연스럽게 지원한다.

다만 저자들도 이 결과를 **proof-of-concept**로 규정한다. 모델 하나의 대규모 학습 결과에 크게 의존하고 있으며, 고정 깊이 모델과 동일 FLOPs로 비교한 실험이나 CoT 모델과의 완전히 공정한 비교는 아직 충분하지 않다. 또한 latent reasoning은 사람이 직접 읽을 수 있는 chain-of-thought보다 감독과 해석이 어렵다는 한계가 있다.

---




## Results Summary

### 1. Model and baselines

The paper trains a **3.5B-parameter recurrent-depth language model** on approximately **800B tokens**.

- Architecture: 2-layer prelude, 4-layer recurrent core, and 2-layer coda
- Test-time recurrence (r): typically 1, 4, 8, 16, or 32
- Increasing (r) repeatedly applies the same recurrent block and increases test-time computation.
- At (r=32), the effective depth is:
  [
  2+4times32+2=132
  ]
  layers.

The model is compared with Pythia, Amber, OLMo, OLMo-2, and StarCoder2. These comparisons are not fully controlled because the models differ substantially in training data, token budgets, and specialization.

---

## 2. General benchmarks

The main benchmarks are ARC-Easy, ARC-Challenge, HellaSwag, MMLU, OpenBookQA, PiQA, SciQ, and WinoGrande.

| Recurrence | ARC-E | ARC-C | HellaSwag | MMLU | OBQA | PiQA | SciQ | WinoGrande |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (r=4) | 49.07 | 27.99 | 43.46 | 23.39 | 28.20 | 64.96 | 80.00 | 55.24 |
| (r=8) | 65.11 | 35.15 | 58.54 | 25.29 | 35.40 | 73.45 | 92.10 | 55.64 |
| (r=16) | 69.49 | 37.71 | 64.67 | 31.25 | 37.60 | 75.79 | 93.90 | 57.77 |
| (r=32) | **69.91** | **38.23** | **65.21** | **31.38** | **38.80** | **76.22** | **93.50** | **59.43** |

The main finding is that performance generally improves as the number of recurrent iterations increases. The gains are particularly visible on ARC-Challenge and MMLU, which require more reasoning.

Easier knowledge-oriented tasks such as SciQ saturate earlier, while harder reasoning tasks continue to benefit from additional computation. At (r=32), the model is generally competitive with older Pythia models and reaches performance comparable to some OLMo-7B results, although it remains behind newer OLMo-2 models trained with more data and improved recipes.

---

## 3. Mathematical reasoning

The paper evaluates GSM8K, GSM8K-CoT, Minerva MATH, and MathQA.

| Model | GSM8K | GSM8K CoT | Minerva MATH | MathQA |
|---|---:|---:|---:|---:|
| Ours, (r=32) | 28.05 / 28.20 | 32.60 / 34.57 | 12.58 | 26.60 |
| Ours with template | 24.87 / 38.13 | 34.87 / 42.84 | 11.24 | 27.97 |
| OLMo-2-7B | 66.72 / 66.79 | 61.94 / 66.19 | 19.08 | 37.59 |
| OLMo-2-32B | 28.43 / 28.51 | 26.76 / 32.37 | 5.72 | 33.90 |

The two GSM8K values correspond to flexible and strict matching. The results vary depending on the evaluation template.

The recurrent model does not outperform OLMo-2-7B, but it achieves meaningful mathematical reasoning performance for a 3.5B model. More importantly, its performance improves substantially when more recurrent computation is allowed.

For the same training setup, the recurrent model also strongly outperforms the non-recurrent baseline on difficult reasoning tasks:

- Non-recurrent baseline at 180B tokens: GSM8K-CoT **1.82 / 2.20**
- Recurrent model at the same stage with (r=32): **9.02 / 10.24**
- Final recurrent model with (r=32): **34.80 / 42.08**

The authors describe this as roughly a five-fold advantage over the non-recurrent baseline on GSM8K at the early checkpoint.

---

## 4. Code generation

The code benchmarks are MBPP and HumanEval. The metric is **pass@1**.

| Model | MBPP | HumanEval |
|---|---:|---:|
| Ours, (r=32) | **24.80** | **23.17** |
| OLMo-2-7B | 21.80 | 10.36 |
| OLMo-2-32B | 19.80 | 17.68 |
| StarCoder2-3B | 43.00 | 31.09 |
| StarCoder2-7B | 43.80 | 31.70 |

The recurrent model performs better than several general-purpose open models, especially on HumanEval. However, it remains below the code-specialized StarCoder2 models, which were trained on much larger code-focused datasets.

---

## 5. Scaling test-time computation

The central result is that increasing (r) improves performance without retraining the model.

- (r=1): little or no latent reasoning
- (r=4): initial improvements
- (r=8)–16: strong gains on many general tasks
- (r=32) or more: continued gains on mathematics, coding, MMLU, and ARC-Challenge

The saturation point depends on the task:

- SciQ and BLiMP saturate quickly.
- HellaSwag approaches saturation around (r=8).
- ARC-Challenge, MMLU, GSM8K, and HumanEval continue improving with more recurrence.
- Complex reasoning tasks such as Mastermind benefit substantially from extended computation.

The model remains numerically stable up to (r=1024), corresponding to roughly 4100 effective layers, although ordinary benchmark performance generally saturates around (r=64)–72.

---

## 6. Recurrent versus non-recurrent models

The recurrent model and the non-recurrent baseline have comparable parameter counts and training setups.

The difference is small on simple factual tasks but large on multi-step reasoning tasks. For the final model:

- (r=1): GSM8K-CoT **0.00 / 0.00**
- (r=32): GSM8K-CoT **34.80 / 42.08**

This supports the claim that the gains are not merely caused by parameter count or additional training data. They arise from repeatedly applying the recurrent computation block at inference time.

---

## 7. Efficiency-related results

The architecture also naturally supports several inference optimizations.

### Adaptive computation

The model can stop recurrence early when the output distribution stabilizes, using the KL divergence between successive recurrence steps.

- Easy tokens terminate early.
- Difficult or reasoning-critical tokens receive more computation.
- No special early-exit training is required.

### KV-cache sharing

The recurrent KV cache can be shared or compressed to reduce memory usage. The paper reports little performance degradation on GSM8K and MT-Bench.

### Self-speculative decoding

The model can generate drafts using fewer recurrent iterations and later verify them using more iterations. This enables speculative decoding without a separate draft model.

---

## 8. Overall assessment

The experiments suggest that latent recurrent reasoning can:

1. Scale test-time computation through recurrent depth.
2. Improve performance especially on mathematical, coding, and multi-step reasoning tasks.
3. Outperform a same-size fixed-depth model on difficult benchmarks.
4. Perform additional reasoning in latent space without producing long verbal chain-of-thought traces.
5. Support adaptive compute, KV-cache sharing, and self-speculative decoding.

However, the authors present the work as a **proof of concept**. The study is based mainly on one large-scale training run, and fully compute-matched comparisons against fixed-depth or verbal-CoT models remain future work. Latent reasoning also introduces an interpretability challenge because its internal reasoning is not directly human-readable.


<br/>
# 예제



### 1. 모델이 학습하는 방식

이 논문의 모델은 일반적인 언어 모델처럼 **입력 문맥을 보고 다음 토큰을 예측**하도록 학습됩니다.

- **입력:** 토큰으로 변환된 텍스트 시퀀스  
  - 예:  
    `Claire makes a 3 egg omelette every morning ...`
  - 코드, 수학 문제, 과학 문서, 웹 문서, 지시문 등
- **정답 출력:** 입력 문장의 다음 토큰들  
  - 예:  
    입력이 `Claire makes a`라면 정답은 다음 토큰인 `3`
    - 이후 `egg`, `omelette` 등도 순차적으로 예측
- **학습 목표:** 각 위치에서 다음 토큰의 확률을 높이는 next-token prediction loss

일반 Transformer와 달리, 이 모델은 한 번의 Transformer 계산으로 바로 출력하지 않습니다. 입력을 latent space에 넣은 뒤, 같은 recurrent block을 여러 번 반복합니다.

[
s_i = R(e, s_{i-1})
]

여기서 (s_i)는 (i)번째 반복 후의 latent state입니다. 모델은 이 상태를 여러 차례 갱신한 뒤 다음 토큰을 예측합니다.

### 2. 트레이닝 데이터의 구체적인 종류

논문에서는 특정한 “긴 Chain-of-Thought 정답 데이터”만을 사용하지 않고, 다양한 일반 텍스트를 섞어 학습했습니다.

주요 데이터 비율은 다음과 같습니다.

| 데이터 종류 | 비율 | 예시 |
|---|---:|---|
| 일반 웹 텍스트 | 28.71% | 웹 문서, 일반적인 설명문 |
| 코드 | 25.36% | Python, GitHub 코드, Lean 등 |
| 과학 텍스트 | 18.73% | arXiv 논문, 과학 문서 |
| 합성 텍스트 | 8.14% | 교과서 형식의 생성 데이터 |
| 장문 텍스트 | 7.50% | 책, 위키, 장문 문서 |
| 수학 데이터 | 6.14% | 수학 문제, 증명, 수식 |
| 지시문·질의응답 데이터 | 일부 | 수학·코딩·일반 질의응답 |

전체 모델은 약 **3.5B 파라미터**이며, 약 **800B 토큰**으로 사전학습되었습니다. 입력 시퀀스의 최대 길이는 **4096 토큰**입니다.

중요한 점은, 학습 데이터에 항상 사람이 작성한 명시적인 사고 과정이 포함되어야 하는 것은 아니라는 것입니다. 모델은 일반적인 next-token prediction을 수행하면서 latent recurrent computation을 사용하는 방법을 스스로 학습합니다.

### 3. 학습 중 recurrent depth의 예

학습 시에는 매번 같은 반복 횟수를 사용하지 않고, 입력마다 recurrent iteration 수 (r)를 무작위로 정합니다.

예를 들어 하나의 학습 배치가 다음과 같이 처리될 수 있습니다.

```text
입력:
Claire makes a 3 egg omelette every morning for breakfast.
How many dozens of eggs will she eat in 4 weeks?

정답:
다음 토큰들
```

모델은 이 입력을 latent state로 변환한 후 다음처럼 처리할 수 있습니다.

```text
r = 4  → recurrent block 4회 반복 후 다음 토큰 예측
r = 16 → recurrent block 16회 반복 후 다음 토큰 예측
r = 32 → recurrent block 32회 반복 후 다음 토큰 예측
```

학습 중에는 모델이 짧은 계산과 긴 계산 모두에 익숙해지도록 반복 횟수를 다양하게 샘플링합니다. 따라서 테스트 시에도 (r=1,4,8,16,32,64) 등 원하는 계산량을 선택할 수 있습니다.

### 4. 테스트 데이터와 테스트 출력

테스트에서는 입력 문제를 넣고, 모델이 정답 토큰이나 정답 문장을 생성하는지 평가합니다.

#### 4.1 수학 추론: GSM8K

- **입력:** 초등학교 수준의 수학 문장제
- **예시 입력:**  
  “Claire makes a 3 egg omelette every morning for breakfast. How many dozens of eggs will she eat in 4 weeks?”
- **기대 출력:**  
  문제를 계산한 최종 답변  
  - 3개 × 28일 = 84개
  - 84개 ÷ 12 = 7 dozen
  - 따라서 정답은 `7`
- **평가:** 생성된 답이 정답과 일치하는지 확인

이 모델은 (r=1)에서는 거의 추론을 수행하지 못하지만, recurrent iteration을 늘리면 GSM8K 성능이 크게 향상됩니다. 논문에서는 3.5B recurrent 모델이 (r=32)에서 GSM8K CoT 성능을 약 **34.8% flexible / 42.1% strict**로 기록했습니다. 템플릿과 평가 방식에 따라 수치는 달라집니다.

#### 4.2 코드 생성: HumanEval, MBPP

- **입력:** 함수 설명 또는 코드 작성 지시
- **예시 입력:**  
  “Write a function that returns the factorial of a non-negative integer.”
- **기대 출력:**  
  실행 가능한 코드

```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
```

- **평가:** 생성된 코드를 실제 테스트 케이스에 실행
- **주요 지표:** pass@1

논문의 모델은 (r=32)에서 다음 성능을 보였습니다.

- MBPP: **24.80%**
- HumanEval: **23.17%**

#### 4.3 일반 지식·상식·언어 이해

다음과 같은 객관식 또는 언어 이해 태스크도 평가했습니다.

- **MMLU:** 다양한 전문 분야 지식과 추론
- **ARC-Challenge / ARC-Easy:** 과학·상식 문제
- **OpenBookQA:** 과학 사실을 이용한 질의응답
- **PiQA:** 물리적 상식
- **HellaSwag:** 문장 이어 쓰기 및 상황 이해
- **SciQ:** 과학 지식
- **WinoGrande:** 대명사 및 문맥 해석

이 경우 입력은 질문과 선택지이고, 출력은 선택지에 해당하는 토큰 또는 답입니다.

예시:

```text
입력:
Which of the following is most likely to conduct electricity?
A. A wooden stick
B. A copper wire
C. A rubber ball
D. A glass cup

출력:
B. A copper wire
```

이러한 쉬운 지식·언어 태스크는 적은 recurrent iteration만으로도 성능이 포화되는 경우가 많았습니다. 반대로 MMLU, ARC-Challenge, 수학·코딩 문제처럼 추론이 필요한 태스크는 반복 횟수를 늘릴수록 더 좋아졌습니다.

### 5. 테스트 시 “출력 토큰을 늘리는 방식”과의 차이

일반적인 Chain-of-Thought 모델은 다음과 같이 중간 사고를 텍스트로 출력합니다.

```text
3 eggs/day × 28 days = 84 eggs
84 / 12 = 7 dozen
Answer: 7
```

반면 이 논문의 모델은 중간 사고를 반드시 텍스트로 출력하지 않습니다.

```text
입력 → latent state 반복 갱신 → 최종 답변 토큰 출력
```

즉, 계산 과정은 주로 고차원 연속 벡터 공간에서 수행되고, 최종 출력만 텍스트로 나타납니다. 그래서 작은 context window에서도 계산할 수 있고, 사고 과정을 매번 토큰으로 기록하지 않아도 됩니다.

### 6. 테스트 시 계산량을 조절하는 방법

같은 입력이라도 recurrent iteration 수를 다르게 줄 수 있습니다.

| 반복 횟수 | 일반적인 용도 |
|---:|---|
| (r=1) | 빠른 기본 예측 |
| (r=4sim8) | 쉬운 질문 |
| (r=16sim32) | 수학·코딩·복잡한 추론 |
| (r=64) 이상 | 더 많은 테스트 시간 계산 |

논문에서는 쉬운 문제는 적은 반복으로 해결하고, 어려운 문제는 더 많은 반복을 사용하는 현상이 학습만으로 나타났습니다. 또한 각 토큰의 연속적인 출력 분포가 충분히 안정되면 반복을 중단하는 **zero-shot adaptive compute**도 실험했습니다.

예를 들어 연속된 두 반복 단계의 다음 토큰 분포 사이 KL divergence가 (5times10^{-4})보다 작아지면 해당 토큰의 계산을 종료합니다.

---




### 1. How the model is trained

The model is trained like a standard language model: it receives a text sequence and predicts the next token.

- **Input:** Tokenized text, such as web text, code, mathematics, scientific writing, or instructions
- **Target output:** The next token at every position
- **Training objective:** Standard next-token prediction loss

The main difference is that the model does not immediately produce an output after one Transformer pass. It repeatedly applies the same recurrent core block:

[
s_i = R(e, s_{i-1})
]

The hidden state is refined in latent space for several iterations before the next-token distribution is produced.

### 2. Training data

The model was pretrained on approximately **800B tokens** with about **3.5B parameters**. The main data categories were:

- Generic web text: **28.71%**
- Code: **25.36%**
- Scientific text: **18.73%**
- Synthetic textbook-style text: **8.14%**
- Long-form text: **7.50%**
- Mathematics: **6.14%**
- Instruction, question-answering, and reasoning data: smaller portions

The training data was not primarily composed of explicit long Chain-of-Thought demonstrations. The model was trained mostly with the ordinary language-modeling objective, while the recurrent computation was encouraged by randomly varying the number of recurrent iterations during training.

### 3. Example of training

A training example may look like this:

```text
Input:
Claire makes a 3 egg omelette every morning for breakfast.
How many dozens of eggs will she eat in 4 weeks?

Target:
The next tokens of the sequence
```

The model may process the example with different recurrence depths:

```text
r = 4  → apply the recurrent block 4 times
r = 16 → apply it 16 times
r = 32 → apply it 32 times
```

The model is therefore trained to operate under different computation budgets.

### 4. Test tasks and expected outputs

#### 4.1 Mathematical reasoning: GSM8K

- **Input:** A grade-school word problem
- **Example:**  
  “Claire makes a 3 egg omelette every morning for breakfast. How many dozens of eggs will she eat in 4 weeks?”
- **Expected answer:** `7`
  - 3 eggs/day × 28 days = 84 eggs
  - 84 / 12 = 7 dozen

The model may internally perform the calculation through repeated latent-state updates rather than printing every intermediate step. Increasing the recurrence depth substantially improves mathematical reasoning performance.

#### 4.2 Code generation: HumanEval and MBPP

- **Input:** A natural-language programming prompt
- **Example:**  
  “Write a function that returns the factorial of a non-negative integer.”
- **Expected output:** Executable code

```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
```

The generated program is executed against test cases. The paper reports, at (r=32):

- MBPP: **24.80% pass@1**
- HumanEval: **23.17% pass@1**

#### 4.3 Knowledge and language-understanding tasks

The model was also evaluated on:

- MMLU
- ARC-Challenge and ARC-Easy
- OpenBookQA
- PiQA
- HellaSwag
- SciQ
- WinoGrande

For example:

```text
Input:
Which of the following is most likely to conduct electricity?
A. A wooden stick
B. A copper wire
C. A rubber ball
D. A glass cup

Output:
B. A copper wire
```

Easier factual or linguistic tasks often reach near-maximum performance with relatively few recurrent steps. Harder reasoning tasks, such as MMLU, ARC-Challenge, GSM8K, and HumanEval, benefit more from additional iterations.

### 5. Difference from verbalized Chain-of-Thought

A conventional Chain-of-Thought model may produce:

```text
3 eggs/day × 28 days = 84 eggs
84 / 12 = 7 dozen
Answer: 7
```

This recurrent-depth model instead performs most of the computation internally:

```text
Input → repeated latent-state updates → final answer tokens
```

The intermediate reasoning is represented in continuous hidden states rather than necessarily being written as text. Therefore, it does not require specialized long CoT training data or a very long context window.

### 6. Controlling test-time computation

The same input can be evaluated with different recurrence depths:

| Recurrence depth | Typical use |
|---:|---|
| (r=1) | Fast baseline prediction |
| (r=4sim8) | Easy questions |
| (r=16sim32) | Mathematics, coding, and harder reasoning |
| (r=64+) | Extended test-time computation |

The paper also demonstrates zero-shot adaptive computation. If the next-token distributions at two successive recurrent steps become sufficiently similar—for example, if their KL divergence falls below (5times10^{-4})—the model can stop iterating for that token and move on to the next one.

<br/>
# 요약

 
1. 이 논문은 Transformer의 일부 블록을 반복 실행해 언어 모델이 토큰을 생성하기 전에 연속적인 잠재공간에서 추가로 추론하도록 하고, 학습 중 반복 횟수를 무작위로 바꿔 테스트 시 계산량을 조절하는 방법을 제안한다.  
2. 3.5B 파라미터 모델을 800B 토큰으로 학습한 결과, 반복 횟수를 1회에서 32회로 늘리면 GSM8K CoT가 사실상 0%에서 약 35~42%로, HumanEval은 23.17%까지 향상되는 등 특히 수학·코딩 같은 어려운 과제에서 성능이 크게 좋아졌다.  
3. 예를 들어 쉬운 상식·언어 문제는 적은 반복만으로 해결되는 반면 수학 문제는 더 오래 추론하며, 모델은 토큰의 잠재 상태에서 수렴·궤도·방향 이동 같은 구조를 스스로 형성하고 토큰별 조기 종료와 KV-cache 공유도 별도 학습 없이 지원한다.  



1. The paper proposes a Transformer that repeatedly applies a shared recurrent block, allowing the model to reason in continuous latent space before generating tokens, while training with randomly sampled recurrence depths so test-time compute can be adjusted.  
2. In a 3.5B-parameter model trained on 800B tokens, increasing recurrence from 1 to 32 steps raised GSM8K CoT performance from essentially 0% to roughly 35–42% and achieved 23.17% on HumanEval, with the largest gains on difficult math and coding tasks.  
3. For example, easy factual or language questions are solved with few iterations, whereas math problems receive more latent computation; the model also spontaneously develops convergent paths, orbits, and drifts in latent space, while supporting zero-shot token-wise early exits and KV-cache sharing.

<br/>
# 기타



아래는 논문의 **다이어그램·피규어·테이블·어펜딕스**를 중심으로, 수치 자체보다 **무엇을 보여주며 어떤 인사이트를 주는지**에 초점을 맞춘 정리입니다.

---

## 1. 핵심 다이어그램과 모델 구조

### Figure 1 — 전체 성능 개요
- **왼쪽:** 테스트 시 recurrent iteration 수 (r)를 늘리면 ARC, GSM8K, OpenBookQA 등에서 성능이 향상됨.
- **오른쪽:** 쉬운 과제는 적은 recurrence로 해결되지만, MMLU·GSM8K·HumanEval·Mastermind처럼 어려운 과제는 더 많은 recurrence가 필요함.
- **핵심 인사이트:** 모델이 모든 입력에 동일한 계산량을 쓰는 것이 아니라, 과제 난이도에 따라 latent space에서 더 오래 “생각”하는 능력이 학습됨.

### Figure 2 — Recurrent-depth 아키텍처와 recurrence 샘플링
모델은 세 부분으로 구성됩니다.

1. **Prelude (P):** 입력 토큰을 latent representation으로 변환  
2. **Core recurrent block (R):** 같은 Transformer block을 반복 적용  
3. **Coda (C):** 최종 latent state를 다음 토큰 확률로 변환  

수식으로는 다음과 같습니다.

[
e=P(x), qquad s_i=R(e,s_{i-1}), qquad p=C(s_r)
]

- 입력 (e)는 각 recurrence 단계에 반복 주입됩니다.
- 초기 상태 (s_0)는 random state에서 시작합니다.
- 학습 시 recurrence 수는 고정하지 않고 log-normal Poisson 분포에서 무작위로 샘플링합니다.
- 대부분은 평균 근처의 작은 recurrence를 사용하지만, 가끔 매우 긴 recurrence도 학습합니다.

**핵심 인사이트:** 테스트 시 (r)을 임의로 늘릴 수 있는 이유는 학습 중부터 다양한 깊이에서 동작하도록 훈련했기 때문입니다.

---

## 2. 모델 성능과 테스트 타임 스케일링

### Table 1 — 일반 언어·상식·추론 벤치마크
3.5B recurrent 모델의 recurrence별 성능은 다음과 같습니다.

| Recurrence | ARC-E | ARC-C | HellaSwag | MMLU | OBQA | PiQA | SciQ |
|---|---:|---:|---:|---:|---:|---:|---:|
| (r=4) | 49.07 | 27.99 | 43.46 | 23.39 | 28.20 | 64.96 | 80.00 |
| (r=8) | 65.11 | 35.15 | 58.54 | 25.29 | 35.40 | 73.45 | 92.10 |
| (r=16) | 69.49 | 37.71 | 64.67 | 31.25 | 37.60 | 75.79 | 93.90 |
| (r=32) | 69.91 | 38.23 | 65.21 | 31.38 | 38.80 | 76.22 | 93.50 |

- (r=4)에서 (r=32)로 늘리면 ARC-E가 약 49%에서 70%로 상승합니다.
- MMLU도 23.39%에서 31.38%로 증가합니다.
- SciQ처럼 비교적 쉬운 지식 문제는 (r=8) 정도에서 이미 높은 성능에 도달합니다.

**인사이트:** recurrence는 단순한 지식 암기보다, 여러 단계의 추론이 필요한 문제에서 더 큰 효과를 보입니다.

---

### Table 2 — 수학 추론 성능
주요 결과:

- GSM8K CoT: **32.60/34.57**
- 템플릿을 적용한 GSM8K CoT: **34.87/42.84**
- Minerva MATH: **12.58**
- MathQA: **26.60**

EMA(weight averaging)를 적용하고 (r=64)로 실행하면 GSM8K 성능이 다음까지 증가합니다.

- Flexible match: **47.23%**
- Strict match: **38.59%**

**인사이트:** 모델은 별도의 긴 CoT 출력 없이도 latent recurrence만으로 수학 문제를 더 오래 처리할 수 있습니다. 다만 최신 대형 reasoning 모델과 비교하면 아직 proof-of-concept 수준입니다.

---

### Table 3 — 코드 생성
| 모델 | MBPP | HumanEval |
|---|---:|---:|
| Ours, (r=32) | 24.80 | 23.17 |
| StarCoder2-3B | 43.00 | 31.09 |
| OLMo-2-32B | 19.80 | 17.68 |

- 일반 목적 오픈소스 모델 대부분보다 높은 성능을 보입니다.
- 하지만 코드에 특화되어 수조 토큰으로 학습된 StarCoder2보다는 낮습니다.

**인사이트:** recurrence는 코드 생성에서도 유용하지만, 충분한 코드 데이터와 특화 모델의 효과를 완전히 대체하지는 못합니다.

---

### Table 4 — 동일 파라미터의 non-recurrent baseline과 비교
동일한 설정에서 학습한 고정 깊이 모델과 비교한 결과입니다.

| 모델 | GSM8K CoT | ARC-C |
|---|---:|---:|
| Non-recurrent baseline, 180B tokens | 1.82/2.20 | 26.96 |
| Recurrent, (r=32), 180B tokens | 9.02/10.24 | 29.18 |
| Recurrent, (r=32), 800B tokens | 34.80/42.08 | 38.23 |
| Recurrent, (r=1), 800B tokens | 0.00/0.00 | 24.06 |

- 초기 180B 토큰 시점에도 recurrent 모델은 GSM8K에서 baseline보다 약 5배 높은 성능을 보입니다.
- (r=1)에서는 성능이 매우 낮고, recurrence를 늘릴 때 어려운 과제에서 큰 폭으로 개선됩니다.
- 저자들은 이러한 결과를 통해 **추론 능력이 주로 recurrent block 내부에 저장된다**고 해석합니다.

---

## 3. 학습 과정과 recurrence scaling

### Figure 3 — 학습 토큰 수에 따른 성능
GSM8K CoT, HellaSwag, HumanEval을 (r=1,4,8,16,32,64)에서 평가합니다.

- 어려운 과제인 GSM8K와 HumanEval은 학습 토큰이 늘어날수록 높은 recurrence에서 계속 개선됩니다.
- HellaSwag는 비교적 빨리 포화됩니다.
- (r=1)에서는 학습이 진행되어도 어려운 추론 문제에서 성능이 제한됩니다.

**인사이트:** 모델의 “추론 능력”은 단순히 파라미터에 저장된 정보만으로 결정되지 않고, 테스트 시 recurrent computation을 얼마나 수행하는지에 크게 의존합니다.

---

### Figure 10 — 과제 및 문맥에 따른 recurrence 효과
- HellaSwag는 약 8회 recurrence에서 거의 포화됩니다.
- GSM8K와 HumanEval은 recurrence를 계속 늘릴수록 향상됩니다.
- ARC-Challenge에서는 few-shot 예제가 많을수록 더 많은 recurrence가 필요합니다.
  - 예제가 없을 때: 약 8–12회에서 포화
  - 25–50개 예제: 약 32회까지 계속 개선

**인사이트:** recurrence는 단순히 입력 자체를 처리하는 계산량뿐 아니라, 긴 문맥에서 제공된 정보를 조합하는 데에도 사용됩니다.

---

### Figure 11 — Continuous CoT와 일반 recurrence의 수렴 속도
이전 토큰의 latent state를 다음 토큰의 초기 상태로 재사용하는 “continuous CoT” 방식을 비교합니다.

- 철학·도덕·논리적 오류 등 일부 MMLU 범주에서 평균 recurrence 수가 감소합니다.
- 특히 이전에 계산한 latent representation을 재사용하면 일부 토큰은 매우 빠르게 수렴합니다.

**인사이트:** 모델은 토큰마다 독립적으로 처음부터 생각하지 않고, 앞선 토큰에서 수행한 latent computation을 이어서 사용할 수 있습니다.

---

## 4. 적응형 계산과 추론 효율

### Figure 4 — Zero-shot per-token adaptive compute
연속 recurrence 단계의 다음 토큰 분포 사이 KL divergence가

[
D_{mathrm{KL}}(p_i | p_{i-1}) < 5times10^{-4}
]

가 되면 해당 토큰의 recurrence를 중단합니다.

결과:

- 고등학교 수학 문제는 평균적으로 빨리 종료됩니다.
- 논리적 오류, 철학, 도덕적 시나리오는 더 많은 계산이 필요합니다.
- 도덕적 시나리오는 고등학교 수학보다 평균 약 3.5회 더 많은 recurrence를 사용합니다.
- MT-Bench 점수도 일반 방식과 큰 차이가 없습니다.
  - Standard: 5.63
  - Early exit: 5.56

**인사이트:** 별도의 early-exit head를 추가로 학습하지 않아도, recurrence 자체의 수렴 정도를 이용해 토큰별 계산량을 조절할 수 있습니다.

---

### Table 8 — 효율화 기법과 MT-Bench
주요 설정의 MT-Bench 점수는 대체로 비슷합니다.

- Baseline, 64 iterations: 5.693
- Baseline, 32 iterations: 5.662
- KV-cache compression, (s=4): 5.856
- KL early exit: 5.562

차이는 통계적으로 유의하지 않습니다.

**인사이트:** KV-cache 공유와 early exit를 사용해 메모리와 연산량을 줄여도 대화 성능이 크게 손상되지 않습니다.

---

### Table 9 — GSM8K에서 적응형 종료와 KV-cache 공유
기준 성능은 **46.63%**입니다.

- KV-cache sharing:
  - compression (s=16): 47.16%
  - compression (s=4): 47.08%
- KL 기반 early exit:
  - threshold (5times10^{-4}): 약 44.8%
- 너무 공격적인 argmax 안정성 기준:
  - 성능이 26% 수준까지 하락

**인사이트:** 적당한 cache sharing은 거의 손실이 없지만, 너무 이른 종료 기준은 어려운 수학 문제의 정확도를 크게 떨어뜨릴 수 있습니다.

---

### Zero-shot KV-cache Sharing
일반 Transformer에서는 보통 layer별 KV-cache 공유를 위해 별도 학습이 필요합니다. 그러나 이 모델에서는 모든 recurrence 단계가 같은 K/V projection을 공유하므로, 최신 recurrence의 cache를 사용해도 성능 손실이 작습니다.

- 메모리 예산 (k)를 정하고, 이후 recurrence에서는 오래된 cache를 덮어씁니다.
- 예: (k=16)이면 17번째 단계가 1번째 단계의 cache를 덮어씁니다.

---

### Zero-shot Self-speculative Decoding
- 적은 recurrence (N)으로 후보 토큰을 빠르게 생성합니다.
- 이후 더 많은 recurrence (M>N)으로 후보를 검증합니다.
- 별도의 draft model이 필요하지 않습니다.
- draft 단계에서 계산한 latent state를 검증 단계에서 재사용할 수 있습니다.

**인사이트:** recurrent 모델의 얕은 계산과 깊은 계산을 각각 draft model과 target model처럼 사용할 수 있습니다.

---

## 5. Latent space에서 나타나는 계산 메커니즘

### Figure 5 — 토큰별 수렴 속도
각 토큰의 latent state가 충분히 많은 recurrence 이후의 상태 (s^*)에 얼마나 가까워지는지를 표시합니다.

- 쉬운 토큰은 빠르게 수렴합니다.
- 질문의 핵심 부분과 답변 시작 부분은 더 오래 변합니다.
- 동일한 토큰이라도 문맥에 따라 수렴 속도가 달라집니다.
- 수렴이 항상 단조롭게 진행되지는 않습니다.

**인사이트:** 계산량은 sequence 전체에 균일하지 않고, 토큰별·문맥별로 다르게 배분됩니다.

---

### Figure 6 — Latent trajectory의 기하학적 구조
PCA로 recurrent trajectory를 시각화한 결과입니다.

관찰된 패턴:

- **Fixed point:** 상태가 한 점으로 수렴
- **Orbit:** latent space에서 반복적인 궤도를 형성
- **Slider/drift:** 한 방향으로 계속 이동하며 recurrence 횟수나 진행 단계를 표현하는 것으로 추정

**인사이트:** 모델은 언어로 표현되는 명시적 CoT 대신, latent space의 회전·이동·수렴 같은 기하학적 구조를 계산 과정으로 활용하는 것으로 보입니다.

---

### Figures 14–17 — 문제 유형별 latent trajectory
- **Figure 14:** 수학, trivia, unsafe 질문의 전체 latent space 방향을 비교
- **Figure 15:** GSM8K 문제의 숫자 “3” 토큰이 회전하는 궤도를 보임
- **Figure 16:** Goethe의 *Faust*를 묻는 간단한 trivia에서는 많은 토큰이 고정점으로 빠르게 수렴
- **Figure 17:** unsafe 질문에서는 특정 핵심 토큰의 trajectory가 비정상적으로 오래 변화함

**인사이트:** 문제의 의미와 난이도에 따라 latent trajectory의 형태가 달라집니다. 수학적 계산에는 orbit 같은 구조가, 단순 지식 회상에는 빠른 수렴이 나타납니다.

---

### Figures 18–20 — 수렴 패턴의 추가 예시
- 문장 전체에서 토큰마다 수렴 시점이 다릅니다.
- system prompt나 일반적인 문맥은 빠르게 안정화됩니다.
- 질문의 핵심 단어나 답변을 결정하는 부분은 더 오래 deliberation합니다.

---

### Figures 21–24 — Path independence
서로 다른 random initial state에서 시작해도 latent trajectory가 비슷한 fixed point 또는 orbit으로 수렴합니다.

- Figure 21: 여러 초기 상태의 trajectory 비교
- Figure 22: 수학 문제의 숫자 “3”
- Figure 23: trivia 질문
- Figure 24: unsafe 질문

**인사이트:** 초기 random state가 달라도 최종 계산 결과가 크게 달라지지 않습니다. 이는 모델이 초기 상태를 극복하고 문맥에 의해 결정되는 안정적인 iterative procedure를 학습했음을 의미합니다.

---

## 6. 데이터와 대규모 학습

### Figure 7 — Pretraining 데이터 구성
주요 데이터 비중:

- Generic text: 28.71%
- Code: 25.36%
- Scientific text: 18.73%
- Synthetic text: 8.14%
- Long-form text: 7.50%
- Math: 6.14%
- Instruction 데이터 전체도 포함

**인사이트:** 일반 언어 능력을 유지하면서도 코드·수학·과학 데이터를 비교적 많이 포함해 추론 행동이 나타나도록 구성했습니다.

---

### Figure 8 — 실패한 학습 설정과 성공한 설정
두 번의 실패 사례가 분석됩니다.

1. **Bad Run 1**
   - hidden state가 token 차원에서 collapse
   - 모든 토큰이 비슷한 representation을 갖게 됨
   - loss 개선이 멈춤

2. **Bad Run 2**
   - 초기에는 회복하지만 recurrent state (s)를 무시하는 local minimum에 빠짐
   - recurrence 수를 늘려도 성능이 증가하지 않음

3. **Main run**
   - sandwich normalization
   - learned adapter
   - 더 작은 learning rate
   - 안정적인 recurrence 활용

**인사이트:** recurrence 모델은 일반 Transformer보다 normalization과 initialization에 훨씬 민감합니다.

---

### Figure 9 — 전체 학습 곡선
- 800B 토큰 동안 training loss가 꾸준히 감소합니다.
- recurrence (r=1,4,8,16,32,64)의 validation perplexity가 모두 개선됩니다.
- 다양한 recurrence 깊이에서 동시에 학습이 진행됩니다.

---

## 7. 어펜딕스의 추가 결과

### Table 5 — Recurrence sampling distribution ablation
132M 모델에서 여러 recurrence 샘플링 분포를 비교했습니다.

| Sampling | Validation PPL |
|---|---:|
| Log-normal Poisson | 12.97 |
| Irwin-Hall | 12.99 |
| Schwarzschild-Bansal | 13.16 |
| Exponential | 13.26 |
| Gamma | 13.31 |
| Geometric | 13.33 |
| Uniform | 13.33 |

**결과:** log-normal Poisson 방식이 가장 안정적이고 낮은 validation perplexity를 보였습니다.

---

### Table 6 — Open-book와 Closed-book QA
OpenBookQA에서 관련 사실을 문맥에 제공하면 성능이 크게 올라갑니다.

- Closed-book: 38.2%
- Open-book: 49.2%

**인사이트:** 이 모델은 대량의 사실을 암기하는 능력보다, 주어진 문맥을 처리하고 추론하는 능력이 상대적으로 강합니다.

---

### Table 7 — 더 최신 오픈 모델과 비교
- 최신 Qwen, Llama, OLMo 모델은 대부분 더 많은 토큰과 compute로 학습되었습니다.
- recurrent 모델은 3.5B 파라미터와 0.8T 토큰만으로 일부 reasoning 및 code 과제에서 경쟁력 있는 결과를 보입니다.
- 그러나 전반적인 지식·언어·MMLU 성능은 최신 대형 모델보다 낮습니다.

**핵심:** 이 논문의 목적은 최신 모델을 능가하는 것이 아니라, recurrent depth를 이용한 test-time scaling이 실제 대규모 언어 모델에서도 작동함을 보이는 것입니다.

---

### Table 10 — 극단적인 recurrence 깊이에서의 안정성
ARC-Challenge 결과:

| Recurrence | Effective layers | Accuracy |
|---|---:|---:|
| 32 | 132 | 37.88 |
| 64 | 260 | 37.37 |
| 128 | 516 | 37.63 |
| 256 | 1,028 | 37.03 |
| 512 | 2,052 | 37.20 |
| 1,024 | 4,100 | 37.37 |

- 1,024회 recurrence, 즉 약 4,100개 effective layers에서도 모델이 붕괴하지 않습니다.
- 다만 성능은 약 32–64회 이후 포화됩니다.

**인사이트:** 모델이 특정 학습 깊이에만 의존하는 것이 아니라, 비교적 안정적인 iterative computation을 학습했다는 증거입니다.

---

### Table 11 — 추론 속도
3.5B recurrent 모델의 대략적인 속도:

- (r=4): 약 1,327 tokens/s
- (r=8): 약 726 tokens/s
- (r=16): 약 380 tokens/s
- (r=32): 약 201 tokens/s

**인사이트:** recurrence를 늘리면 성능은 좋아지지만 추론 속도는 거의 비례해서 감소합니다. 따라서 adaptive compute와 KV-cache sharing이 실제 배포에서 중요합니다.

---

### Tables 12–13 — Pretraining 데이터 목록
- 공개 Hugging Face 데이터셋, 코드·수학·과학·instruction 데이터가 사용됩니다.
- 각 데이터셋의 주소, 라이선스, 카테고리, 가중치가 정리되어 있습니다.
- 저자들은 재현성을 위해 데이터 처리 코드와 모델 checkpoint도 공개할 계획을 밝힙니다.

---

## 8. 어펜딕스의 방법론적 내용

### Appendix C — 대규모 학습 설정
주요 설정:

- 모델 크기: 약 3.5B parameters
- 구조: ((l_P,l_R,l_C)=(2,4,2))
- hidden size: 5,280
- 평균 recurrence: 32
- context length: 4,096
- vocabulary: 65,536 BPE tokens
- 학습 데이터: 약 800B tokens
- 하드웨어: AMD MI250X GPU 4,096개
- 학습 기간: 약 10일
- precision: bfloat16
- optimizer: Adam with decoupled weight decay

### Truncated Backpropagation
- 학습 중에는 마지막 (k=8) recurrence만 역전파합니다.
- recurrence가 매우 긴 샘플에서도 메모리와 backward cost를 제한할 수 있습니다.
- Prelude는 입력이 매 단계 주입되기 때문에 전체 recurrence의 gradient 영향을 계속 받습니다.

### Appendix D — 평가 세부사항
- 모든 평가에서 기본적으로 temperature 0을 사용합니다.
- 초기 latent state는 학습과 같은 분포의 Gaussian random vector로 설정합니다.
- (r=32)와 같이 표기된 경우 모든 query에 정확히 32회 recurrence를 적용합니다.
- lm-eval harness와 Hugging Face 기반 구현으로 결과를 재현할 수 있습니다.

### Appendix E — Latent visualization
- PCA 기반 trajectory 분석
- 초기 상태 변화에 대한 path independence 검증
- fixed point, orbit, drift 패턴의 추가 예시 제공

### Appendix F — Pretraining data
- 실제 학습에 사용된 데이터셋과 라이선스, 데이터 유형을 상세히 기록합니다.
- 데이터 구성은 재현성과 라이선스 검토에 유용합니다.

---

## 9. Appendix A/B와 논문의 한계

### Appendix A — Broader Impact
긍정적 영향:

- 파라미터 수가 작은 모델로 더 많은 test-time computation을 사용할 수 있음
- commodity hardware에서 모델을 배포하기 쉬울 가능성
- 긴 verbal CoT 없이 latent space에서 추론 가능

우려 사항:

- latent reasoning은 사람이 읽을 수 있는 CoT가 아니므로 감독과 해석이 어려울 수 있음
- 모델의 내부 computation을 검증하려면 white-box interpretability가 필요함
- 모델 성능과 reasoning 능력이 향상될수록 일반적인 LLM 오용 위험도 존재함

### Appendix B — Future Work
저자들이 제시한 방향:

- recurrence를 압축하는 post-training
- reasoning 데이터나 RL을 이용한 추가 학습
- CoT를 latent recurrence로 internalize
- recurrent model과 MoE의 결합
- 여러 recurrent stage를 갖는 구조
- linear attention과 recurrence의 결합

### Section 8 — 주요 한계
- 아직 proof-of-concept 모델입니다.
- 동일 FLOP 예산의 고정 깊이 모델과 완전히 공정한 비교는 수행하지 않았습니다.
- verbal CoT 모델과도 동일한 test-time compute 기준의 직접 비교가 부족합니다.
- 단일 대규모 학습 run에 의존합니다.
- recurrence가 늘어날수록 성능은 어느 시점에서 포화됩니다.
- latent reasoning은 인간이 직접 읽고 평가하기 어렵습니다.

---

## 전체 결론

이 논문의 가장 중요한 메시지는 다음과 같습니다.

1. **추론을 더 많은 출력 토큰으로 표현하지 않고, latent state를 반복 갱신하는 방식으로 test-time compute를 확장할 수 있다.**
2. **추가 recurrence는 쉬운 과제보다 수학·코딩·논리 추론 같은 어려운 과제에서 더 큰 효과를 낸다.**
3. **모델은 학습 중 무작위 recurrence 깊이를 경험함으로써, 테스트 시 계산량을 늘리거나 줄이는 능력을 얻는다.**
4. **토큰별 adaptive exit, KV-cache sharing, self-speculative decoding이 별도의 특수 학습 없이 가능하다.**
5. **latent space에는 fixed point, orbit, drift 등 구조적인 계산 패턴이 자연스럽게 나타난다.**
6. **다만 최신 reasoning 모델을 능가하는 완성된 시스템이라기보다, recurrent depth를 통한 latent reasoning의 가능성을 입증한 대규모 proof-of-concept에 가깝다.**

---




## 1. Core diagrams and architecture

### Figure 1 — Overall performance picture
- Increasing the recurrence count (r) improves performance on ARC, GSM8K, OpenBookQA, and other tasks.
- Easy tasks saturate quickly, while MMLU, GSM8K, HumanEval, and Mastermind benefit from extended recurrence.
- **Main insight:** The model learns to allocate more latent computation to harder problems.

### Figure 2 — Recurrent-depth architecture
The model consists of:

1. **Prelude (P):** embeds input tokens into latent space  
2. **Core recurrent block (R):** repeatedly updates the latent state  
3. **Coda (C):** converts the final latent state into next-token probabilities  

[
e=P(x), qquad s_i=R(e,s_{i-1}), qquad p=C(s_r)
]

The input embedding is injected at every recurrent step, and the initial state is randomly initialized. During training, the recurrence count is sampled from a log-normal Poisson distribution.

**Main insight:** Because the model is trained with variable recurrence depths, it can use more or less computation at test time.

---

## 2. Benchmark results

### Table 1 — Standard benchmarks
The 3.5B recurrent model improves substantially as recurrence increases.

| Recurrence | ARC-E | ARC-C | HellaSwag | MMLU | OBQA | PiQA | SciQ |
|---|---:|---:|---:|---:|---:|---:|---:|
| (r=4) | 49.07 | 27.99 | 43.46 | 23.39 | 28.20 | 64.96 | 80.00 |
| (r=8) | 65.11 | 35.15 | 58.54 | 25.29 | 35.40 | 73.45 | 92.10 |
| (r=16) | 69.49 | 37.71 | 64.67 | 31.25 | 37.60 | 75.79 | 93.90 |
| (r=32) | 69.91 | 38.23 | 65.21 | 31.38 | 38.80 | 76.22 | 93.50 |

- ARC-E increases from 49.07% to 69.91%.
- MMLU improves from 23.39% to 31.38%.
- SciQ saturates earlier, suggesting that simple knowledge questions require less computation.

**Insight:** Recurrent depth is especially helpful for multi-step reasoning rather than simple factual recall.

---

### Table 2 — Mathematical reasoning
At (r=32), the model achieves:

- GSM8K CoT: **32.60/34.57**
- Templated GSM8K CoT: **34.87/42.84**
- Minerva MATH: **12.58**
- MathQA: **26.60**

With exponential moving average weights and (r=64):

- Flexible GSM8K match: **47.23%**
- Strict match: **38.59%**

**Insight:** The model can perform additional mathematical reasoning internally, without producing a long verbal chain of thought. However, it remains a proof-of-concept compared with modern reasoning systems.

---

### Table 3 — Code generation
The model obtains:

- MBPP: **24.80**
- HumanEval: **23.17**

It outperforms many general-purpose open models, but remains below code-specialized models such as StarCoder2.

**Insight:** Recurrent computation helps code generation, but it does not replace large-scale code-specific pretraining.

---

### Table 4 — Recurrent versus non-recurrent baseline
At the same 180B-token training point:

- Non-recurrent GSM8K CoT: **1.82/2.20**
- Recurrent GSM8K CoT at (r=32): **9.02/10.24**

At 800B tokens:

- Recurrent (r=32): **34.80/42.08**
- Recurrent (r=1): **0.00/0.00**

**Insight:** Much of the reasoning ability is encoded in the repeated application of the recurrent block, not merely in the fixed-depth components.

---

## 3. Training and scaling behavior

### Figure 3 — Performance over training
- GSM8K and HumanEval continue improving when sufficient recurrence is provided.
- HellaSwag reaches saturation earlier.
- A single recurrence is insufficient for difficult reasoning tasks.

**Insight:** The model benefits from both more pretraining data and more test-time recurrence.

### Figure 10 — Task and context dependence
- HellaSwag saturates around 8 iterations.
- GSM8K and HumanEval keep improving with more recurrence.
- ARC-Challenge requires more recurrence when more few-shot examples are provided.

**Insight:** Recurrent computation is used not only to process the input, but also to combine and reason over longer contexts.

### Figure 11 — Continuous latent CoT
By reusing the previous token’s latent state:

- Some MMLU categories converge with fewer iterations.
- Previous latent computation can be carried over to subsequent tokens.

**Insight:** The model can perform a form of continuous latent reasoning across generated tokens.

---

## 4. Adaptive computation and efficiency

### Figure 4 — Zero-shot token-level adaptive compute
The model exits when the KL divergence between successive next-token distributions falls below

[
5times10^{-4}.
]

- High-school mathematics exits relatively early.
- Philosophy, logical fallacies, and moral scenarios require more iterations.
- MT-Bench remains almost unchanged:
  - Standard: 5.63
  - Early exit: 5.56

**Insight:** The model can adapt computation per token without a separately trained early-exit head.

### Table 8 — Efficiency methods
Representative MT-Bench scores remain close:

- Baseline, 64 iterations: 5.693
- Baseline, 32 iterations: 5.662
- KV-cache compression, (s=4): 5.856
- KL early exit: 5.562

**Insight:** Cache compression and early exit can reduce inference cost without a major conversational-quality loss.

### Table 9 — GSM8K with cache sharing and early exit
Baseline accuracy is **46.63%**.

- KV-cache sharing with (s=16): **47.16%**
- KV-cache sharing with (s=4): **47.08%**
- KL-based early exit: approximately **44.8%**
- Aggressive argmax-based exits can reduce accuracy to about **26%**

**Insight:** Moderate cache sharing is robust, but overly aggressive early stopping harms difficult reasoning tasks.

### Self-speculative decoding
The model can:

1. Draft tokens with fewer recurrence steps.
2. Verify them using more recurrence steps.
3. Reuse latent states from the drafting phase.

No separate draft model is required.

---

## 5. Emergent latent-space mechanisms

### Figure 5 — Token-specific convergence
- Easy tokens converge quickly.
- Key parts of questions and answer beginnings continue changing for longer.
- Identical tokens can have different convergence behavior depending on context.

**Insight:** Computation is allocated unevenly across tokens and depends strongly on context.

### Figure 6 — Geometric trajectories
The authors observe:

- **Fixed points:** convergence to a stable state
- **Orbits:** repeated motion through latent space
- **Sliders or drifts:** directional movement that may encode iteration count or progress

**Insight:** Instead of verbal CoT, the model appears to organize computation through geometric structures in high-dimensional latent space.

### Figures 14–17 — Different types of questions
- Mathematical questions show more structured trajectories and rotations.
- Trivia questions often converge quickly.
- Certain key tokens in unsafe questions exhibit unusually long or complex trajectories.

### Figures 18–20 — Additional convergence examples
Different tokens converge at different speeds. System prompts and unimportant words stabilize quickly, while question-critical tokens receive more latent deliberation.

### Figures 21–24 — Path independence
Different random initial states eventually converge to similar fixed points or orbit patterns.

**Insight:** The model learns a relatively stable iterative procedure determined by the input context rather than by the random initial state.

---

## 6. Data and large-scale training

### Figure 7 — Pretraining data mixture
Main components include:

- Generic text: 28.71%
- Code: 25.36%
- Scientific text: 18.73%
- Synthetic text: 8.14%
- Long-form text: 7.50%
- Mathematics: 6.14%

**Insight:** The data mixture aims to preserve general language ability while encouraging mathematical and algorithmic reasoning.

### Figure 8 — Failed and successful training runs
Two failed runs are analyzed:

- One suffers from hidden-state collapse.
- Another learns to ignore the recurrent state.
- The successful run uses sandwich normalization, a learned adapter, and a smaller learning rate.

**Insight:** Large recurrent-depth models are highly sensitive to initialization, normalization, and optimization settings.

### Figure 9 — Training curves
- Training loss steadily decreases across 800B tokens.
- Validation perplexity improves at recurrence depths from 1 to 64.
- The model learns to operate at multiple computational depths.

---

## 7. Additional appendix results

### Table 5 — Recurrence sampling ablation
The log-normal Poisson distribution gives the best validation perplexity:

- Log-normal Poisson: **12.97**
- Irwin-Hall: 12.99
- Uniform: 13.33

**Insight:** Heavy-tailed recurrence sampling is useful because it exposes the model to both ordinary and unusually deep computations.

### Table 6 — Open-book versus closed-book QA
- Closed-book OBQA: **38.2%**
- Open-book OBQA: **49.2%**

**Insight:** The model is relatively better at reasoning over provided information than memorizing large amounts of factual knowledge.

### Table 7 — Comparison with newer open models
The recurrent model is competitive on some reasoning and coding tasks, despite having fewer parameters and fewer training tokens. However, it trails more recent models on broad language and knowledge benchmarks.

### Table 10 — Stability at extreme depths
The model remains stable up to 1,024 recurrence steps, equivalent to about 4,100 effective layers. Performance saturates around 32–64 steps but does not collapse at larger depths.

**Insight:** The model appears to learn a stable iterative algorithm rather than a computation tied to only one specific depth.

### Table 11 — Inference speed
Approximate throughput:

- (r=4): 1,327 tokens/s
- (r=8): 726 tokens/s
- (r=16): 380 tokens/s
- (r=32): 201 tokens/s

**Insight:** More reasoning improves accuracy but reduces throughput, making adaptive computation and cache sharing important for deployment.

### Tables 12–13 — Pretraining data
These tables list the datasets, sources, licenses, categories, and sampling weights used for pretraining. They support reproducibility and licensing review.

---

## 8. Appendix C–F and limitations

### Appendix C — Training details
Key settings:

- Approximately 3.5B parameters
- Architecture ((2,4,2))
- Hidden size 5,280
- Mean recurrence 32
- Context length 4,096
- Vocabulary size 65,536
- Around 800B training tokens
- 4,096 AMD MI250X GPUs
- bfloat16 training

Truncated backpropagation is applied only through the last 8 recurrent iterations to control memory and backward computation.

### Appendix D — Evaluation details
- Temperature 0 is used by default.
- Initial latent states follow the training distribution.
- (r=32) means exactly 32 recurrence steps for every query.
- Evaluations can be reproduced with the provided Hugging Face and lm-eval implementations.

### Appendix E — Latent visualization
Provides additional PCA plots, path-independence experiments, and examples of fixed points, orbits, and directional drift.

### Appendix F — Pretraining data
Documents the datasets and licenses used in the training mixture.

### Appendix A — Broader impact
Potential benefits include smaller models with scalable inference-time reasoning and easier deployment on commodity hardware.

Potential concerns include:

- Latent reasoning is less human-readable than verbal CoT.
- Oversight and interpretability may therefore be more difficult.
- White-box analysis is needed to understand the internal computation.

### Appendix B — Future work
Suggested directions include:

- Recurrence compression
- Reinforcement learning with difficulty-aware data
- Internalizing verbal CoT into latent recurrence
- Combining recurrent depth with mixture-of-experts models
- Multiple recurrent stages
- Combining recurrence with linear attention

---

## Overall conclusion

The paper’s central message is:

1. Test-time computation can be scaled by repeatedly updating latent states rather than generating longer verbal chains of thought.
2. Additional recurrence is most useful for mathematics, coding, and other multi-step reasoning tasks.
3. Randomized recurrence during pretraining enables flexible computation at inference time.
4. Adaptive exits, KV-cache sharing, and self-speculative decoding arise naturally in this architecture.
5. Structured latent trajectories—fixed points, orbits, and drifts—emerge as possible internal reasoning mechanisms.
6. The work is best viewed as a large-scale proof of concept for latent recurrent reasoning, not yet as a fully competitive replacement for modern reasoning models.

<br/>
# refer format:
### BibTeX

```bibtex
@inproceedings{geiping2025scaling,
  author    = {
    Geiping, Jonas and
    McLeish, Sean and
    Jain, Neel and
    Kirchenbauer, John and
    Singh, Siddharth and
    Bartoldson, Brian R. and
    Kailkhura, Bhavya and
    Bhatele, Abhinav and
    Goldstein, Tom
  },
  title     = {Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {39},
  year      = {2025},
  note      = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)}
}
```


### Chicago Style    

Geiping, Jonas, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. “Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.” In *Advances in Neural Information Processing Systems*, vol. 39. Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025), 2025.


