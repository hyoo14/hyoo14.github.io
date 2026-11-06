---
layout: post
title:  "[2026]Prototype-Based Dynamic Steering for Large Language Models"
date:   2026-11-06 05:19:15 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

추론 시점에 내부 엑티베이션 값 조정해서 추론 성능 높임  
(내부 액티베이션 값 차이는 CoT와 nonCoT로 수집한다고 함)  


짧은 요약(Abstract) :




이 논문은 **Prototype-Based Dynamic Steering(PDS)**라는 방법을 제안합니다. PDS는 모델을 재학습하거나 프롬프트에 새로운 지시문을 추가하지 않고, **추론 시점에 모델 내부의 활성값을 조정하여 추론 능력을 높이는 방법**입니다.

구체적으로, 연구진은 같은 문제에 대해 **Chain-of-Thought(CoT) 프롬프트**와 일반적인 **중립 프롬프트**를 입력했을 때 발생하는 내부 활성값의 차이를 수집합니다. 그런 다음 이 차이들을 여러 그룹으로 묶어, 서로 다른 문제 해결 방식이나 추론 전략을 나타내는 **추론 프로토타입(reasoning prototypes)**을 만듭니다.

새로운 문제가 입력되면, PDS는 해당 문제의 내부 상태가 각 프로토타입과 얼마나 관련 있는지 계산하고, 그 결과를 이용해 **문제별로 맞춤화된 steering vector**를 구성합니다. 따라서 모든 문제에 같은 방향으로 개입하는 기존의 Difference-of-Means 방식보다 더 유연하게 작동합니다.

GSM8K, AQuA-RAT, BIG-Bench에서 실험한 결과, PDS는 파인튜닝이나 프롬프트 설계 없이도 전반적으로 정확도를 높였습니다. 특히 단계별 풀이를 금지하는 **Anti-CoT 조건에서도 성능 향상이 유지**되었습니다. 이는 PDS가 단순히 CoT 문장을 출력하게 만드는 것이 아니라, 모델 내부의 **잠재적인 추론 과정 자체를 강화할 수 있음**을 시사합니다.

---



This paper introduces **Prototype-Based Dynamic Steering (PDS)**, a test-time method for improving the reasoning ability of large language models without fine-tuning the model or modifying the instructions.

The method compares the model’s internal activations under a **Chain-of-Thought (CoT) prompt** and a neutral prompt. The differences between these activations are clustered into several **reasoning prototypes**, which represent different reasoning strategies.

When a new problem is given, PDS measures how strongly its hidden state aligns with each prototype and combines the corresponding directions into an **input-specific steering vector**. This makes the intervention adaptive to each problem, unlike the conventional Difference-of-Means method, which applies the same steering direction to every input.

Experiments on GSM8K, AQuA-RAT, and BIG-Bench show that PDS generally improves accuracy without fine-tuning or prompt engineering. Importantly, the gains remain even under **Anti-CoT prompts**, which explicitly discourage step-by-step explanations. This suggests that PDS enhances the model’s **latent reasoning processes**, rather than merely forcing it to produce visible chain-of-thought text.


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


### 1. 방법의 핵심: PDS
논문은 **Prototype-Based Dynamic Steering(PDS)**라는 추론 시점(inference-time) 개입 방법을 제안합니다.  
모델의 가중치를 재학습하거나 프롬프트를 수정하지 않고, **LLM 내부의 은닉 상태(hidden activation)**를 조정하여 문제별로 적합한 추론 방향을 유도합니다.

기존의 Difference-of-Means(DoM)가 모든 입력에 동일한 하나의 steering vector를 적용하는 것과 달리, PDS는 입력마다 다른 steering vector를 동적으로 계산합니다.

---

### 2. 사용 모델과 아키텍처
- 주요 모델: **LLaMA-3-Instruct-8B**
- BIG-Bench 추가 실험: **LLaMA-3-Instruct-70B**
- 보조 실험: **Gemma-2-9B-IT**
- 새로운 신경망 구조나 별도의 모듈을 추가하지 않음
- 모델 가중치와 기존 Transformer 구조는 그대로 유지
- PyTorch와 Hugging Face Transformers를 사용하고, forward hook으로 은닉 상태에 개입

즉, PDS는 새로운 모델 아키텍처라기보다 **기존 LLM의 내부 표현을 추론 중 수정하는 activation steering 기법**입니다.

---

### 3. 학습 또는 데이터 구성 방식
PDS는 모델을 fine-tuning하지 않습니다. 대신 각 데이터셋의 training split을 사용해 추론용 steering prototype을 미리 만듭니다.

각 문제 (x_i)에 대해 두 가지 프롬프트를 구성합니다.

- **CoT 프롬프트**:  
  “Let’s think step by step.”처럼 단계별 추론을 유도
- **Neutral 프롬프트**:  
  특별한 추론 지시 없이 바로 답변하도록 구성

두 프롬프트를 모델에 입력한 뒤, 특정 중간~후반 Transformer layer의 마지막 입력 토큰 은닉 상태를 추출합니다.

[
d_i = h_l(p_{i,text{CoT}})-h_l(p_{i,text{neutral}})
]

이 차이 벡터 (d_i)는 해당 문제에서 CoT식 추론을 유도할 때 발생하는 activation 변화입니다.

사용한 데이터는 다음과 같습니다.

- **GSM8K**: 훈련 예제 7,470개
- **AQuA-RAT**: 훈련 예제 10,000개 샘플
- **BIG-Bench 일부 과제**: Checkmate-in-One, Physics Questions, Rhyming, Temporal Sequences, Intersect Geometry

각 데이터셋별로 별도의 prototype을 생성하여 다른 데이터셋으로부터의 정보 누수를 피했습니다.

---

### 4. Reasoning Prototype 생성
수집한 activation 차이 벡터들을 하나의 평균으로 합치지 않고, **K-means clustering**으로 여러 군집으로 나눕니다.

[
{mu_1,mu_2,ldots,mu_k}=text{K-means}(D,k)
]

각 군집의 중심 (mu_j)가 하나의 **reasoning prototype**입니다.  
Prototype은 서로 다른 문제 해결 전략이나 추론 패턴을 나타내는 대표적인 activation 방향으로 해석됩니다.

예를 들어 GSM8K에서는 다음과 같은 유형의 군집이 발견되었습니다.

- 다단계 산술 문제
- 비율 및 비교 문제
- 금융·비용 계산
- 단위 변환
- 순차 연산 및 패턴 문제

군집 수 (k)는 주로 **elbow method**로 결정합니다.

---

### 5. 추론 시 동적 steering
새로운 입력이 들어오면 모델을 먼저 생성 없이 실행하여 입력의 은닉 상태 (h_{text{input}})을 얻습니다.

그 후 입력 은닉 상태를 각 prototype에 투영하고, 그 결과를 더해 입력별 steering vector를 만듭니다.

[
v_{text{steer}}(h_{text{input}})
=
sum_{j=1}^{k}text{proj}_{mu_j}(h_{text{input}})
]

즉,

- 현재 입력과 잘 맞는 prototype은 더 크게 기여하고
- 관련성이 낮은 prototype은 적게 기여합니다.

최종적으로 다음과 같이 은닉 상태를 수정합니다.

[
h'_{text{input}}
=
h_{text{input}}
+
alpha v_{text{steer}}(h_{text{input}})
]

여기서 (alpha)는 steering 강도입니다. 논문에서는 주로 중간 layer에 개입하며, **첫 번째 출력 토큰 생성 시점에 선택적으로 적용**합니다. 이를 통해 추론의 시작 방향만 조정하고 이후 생성 과정에 대한 과도한 간섭은 줄입니다.

---

### 6. 평가 프롬프트와 비교 방법
PDS는 세 가지 프롬프트 조건에서 평가되었습니다.

1. **CoT**: 단계별 추론을 명시적으로 요구
2. **Neutral**: 특별한 추론 지시 없음
3. **Anti-CoT**: “설명하지 말고 즉시 답하라”고 지시

비교 대상은 다음과 같습니다.

- **No Steering**: 개입 없음
- **DoM**: CoT와 Neutral activation 차이의 평균 벡터를 모든 입력에 동일하게 적용
- **PDS**: 입력마다 prototype 투영을 사용해 다른 steering vector 적용

---

### 7. 주요 결과와 의미
PDS는 특히 **Neutral**과 **Anti-CoT** 조건에서 성능 향상이 두드러졌습니다.

예를 들어 LLaMA-3-8B의 결과에서:

- GSM8K Anti-CoT: **11% → 21%**
- AQuA-RAT Anti-CoT: **29% → 31%**

Anti-CoT에서는 단계별 풀이를 출력하지 않으면서도 정확도가 향상되었습니다. 따라서 PDS는 단순히 CoT 텍스트를 강제로 생성하는 것이 아니라, 모델 내부의 **잠재적 추론 과정(latent reasoning)**을 강화할 수 있다고 논문은 해석합니다.

또한 PDS는 DoM보다 문제별 차이를 반영하기 때문에, 하나의 평균 방향만 사용하는 방식보다 유연하고 입력 적응적입니다. 다만 BIG-Bench 일부 과제에서는 성능이 개선되지 않거나 소폭 하락하는 경우도 있었습니다.

---

### 한 줄 요약
**PDS는 CoT와 일반 프롬프트의 내부 activation 차이를 여러 reasoning prototype으로 분해한 뒤, 새로운 입력과 잘 맞는 prototype들을 조합해 모델의 은닉 상태를 추론 시점에 동적으로 조정하는 학습 없는 activation steering 방법입니다.**

---



### 1. Core idea: Prototype-Based Dynamic Steering
The paper proposes **Prototype-Based Dynamic Steering (PDS)**, an inference-time method for improving LLM reasoning.

PDS does not fine-tune the model, modify its weights, or add new prompt instructions. Instead, it modifies the model’s **internal hidden activations** so that the model is guided toward a more suitable reasoning pattern for each input.

Unlike Difference-of-Means (DoM), which applies the same steering vector to every example, PDS constructs an input-specific steering vector dynamically.

---

### 2. Models and architecture
The experiments use:

- **LLaMA-3-Instruct-8B** as the main model
- **LLaMA-3-Instruct-70B** for additional BIG-Bench experiments
- **Gemma-2-9B-IT** in the supplementary model-scale analysis

PDS does not introduce a new neural architecture or additional trainable module. The original Transformer model remains unchanged, and activation interventions are implemented with forward hooks.

Thus, PDS is best understood as an **activation steering method applied to existing LLMs**, rather than as a new model architecture.

---

### 3. Activation-difference collection
For every training example, the authors create two prompt variants:

- **CoT prompt**: explicitly encourages step-by-step reasoning
- **Neutral prompt**: asks the model to answer without special reasoning instructions

The model processes both prompts, and the hidden state of the final input token is extracted from a middle-to-late Transformer layer.

The activation difference is computed as:

[
d_i =
h_l(p_{i,text{CoT}})
-
h_l(p_{i,text{neutral}})
]

Each difference vector represents how the model’s internal representation changes when explicit reasoning is encouraged for a particular problem.

The datasets include:

- GSM8K
- AQuA-RAT
- Selected BIG-Bench tasks

Prototypes are constructed separately for each dataset to avoid cross-task leakage.

---

### 4. Discovering reasoning prototypes
Instead of averaging all activation differences into one global vector, PDS applies **K-means clustering**:

[
{mu_1,mu_2,ldots,mu_k}
=
text{K-means}(D,k)
]

Each cluster centroid (mu_j) is treated as a **reasoning prototype**. These prototypes are intended to capture different reasoning strategies or activation patterns.

For example, GSM8K clusters were interpreted as involving:

- Multi-step arithmetic
- Comparative and ratio problems
- Financial or cost calculations
- Unit conversion
- Sequential operations and patterns

The number of clusters is mainly selected using the elbow method.

---

### 5. Dynamic steering during inference
For a new input, the model first produces an input representation (h_{text{input}}) without generating a response.

This representation is projected onto all learned prototypes:

[
v_{text{steer}}(h_{text{input}})
=
sum_{j=1}^{k}
text{proj}_{mu_j}(h_{text{input}})
]

Prototypes that are more aligned with the current input contribute more strongly. The resulting input-specific steering vector is then added to the hidden state:

[
h'_{text{input}}
=
h_{text{input}}
+
alpha v_{text{steer}}(h_{text{input}})
]

Here, (alpha) controls the steering strength. The intervention is applied mainly at a middle layer and selectively to the first output token, encouraging a useful reasoning trajectory while minimizing disruption to later generation.

---

### 6. Evaluation settings
The method is tested under three prompting conditions:

1. **CoT**: explicitly asks for step-by-step reasoning
2. **Neutral**: gives no reasoning instruction
3. **Anti-CoT**: asks the model to answer immediately without elaboration

The main comparisons are:

- **No Steering**
- **Difference-of-Means (DoM)**
- **Prototype-Based Dynamic Steering (PDS)**

---

### 7. Main findings
PDS produces its strongest and most consistent gains under **Neutral** and **Anti-CoT** prompting.

For example, with LLaMA-3-8B:

- GSM8K Anti-CoT: **11% → 21%**
- AQuA-RAT Anti-CoT: **29% → 31%**

Under Anti-CoT prompting, PDS improves accuracy without forcing the model to output long step-by-step explanations. This suggests that the method can strengthen **latent reasoning processes** rather than merely inducing visible CoT text.

Overall, PDS is more flexible than DoM because it preserves multiple reasoning directions and selects their combination according to the current input. However, its gains are not uniform: some BIG-Bench tasks show little improvement or slight regressions.

---

### One-sentence summary
**PDS learns multiple reasoning prototypes from CoT–Neutral activation differences and dynamically combines the prototypes that best match each new input, enabling training-free, input-adaptive steering of an existing LLM’s latent reasoning process.**


<br/>
# Results


### 1. 비교 대상과 실험 설정

논문은 **PDS(Prototype-Based Dynamic Steering)**를 기존의 **DoM(Difference-of-Means) steering** 및 **무개입 모델(No Steering)**과 비교했다.

- **No Steering**: 활성값 개입 없이 모델을 그대로 사용
- **DoM**: CoT 프롬프트와 Neutral 프롬프트의 활성값 차이를 평균 내어 하나의 고정 steering vector를 사용
- **PDS**: 활성값 차이를 여러 prototype으로 군집화한 뒤, 각 입력과 prototype의 유사도에 따라 입력별 steering vector를 동적으로 구성

사용한 모델은 다음과 같다.

- **LLaMA-3-Instruct-8B**: 주요 실험 모델
- **LLaMA-3-Instruct-70B**: 모델 규모 확장성 확인
- **Gemma-2-9B-IT**: 추가적인 중간 규모 모델 실험

프롬프트 조건은 세 가지다.

1. **CoT**: “Let’s think step by step”과 같이 단계별 추론을 유도
2. **Neutral**: 별도의 추론 지시 없음
3. **Anti-CoT**: “설명 없이 즉시 답하라”고 지시하여 단계별 출력을 억제

---

### 2. 테스트 데이터와 평가 지표

평가는 다음 세 종류의 벤치마크에서 수행되었다.

| 데이터셋 | 특징 | 평가 데이터 |
|---|---|---:|
| **GSM8K** | 초등 수준의 다단계 수학 문장제 | 1,319개 |
| **AQuA-RAT** | 대수적 조작과 객관식 추론 문제 | 254개 |
| **BIG-Bench Subset** | 체스, 물리, 운율, 시간 순서, 기하 문제 | 태스크별 상이 |

주요 평가지표는 **Accuracy@1**, 즉 모델의 최종 답이 정답인지 여부에 따른 정확도다.  
추가로 출력 토큰 수를 분석하여, PDS의 성능 향상이 단순히 더 긴 답변을 생성한 결과인지도 확인했다.

---

### 3. GSM8K와 AQuA-RAT 결과

#### LLaMA-3-Instruct-8B 정확도

| 프롬프트 조건 | GSM8K No Steering | GSM8K DoM | GSM8K PDS | AQuA-RAT No Steering | AQuA-RAT DoM | AQuA-RAT PDS |
|---|---:|---:|---:|---:|---:|---:|
| **CoT** | 71% | 75% | **78%** | 49% | 50% | **53%** |
| **Neutral** | 68% | 74% | **78%** | 46% | 50% | **53%** |
| **Anti-CoT** | 11% | 18% | **21%** | 29% | 30% | **31%** |

핵심 결과는 다음과 같다.

- PDS는 **GSM8K의 모든 프롬프트 조건에서 DoM보다 높은 정확도**를 보였다.
- 특히 **Neutral 조건**에서 GSM8K 정확도를 68%에서 78%로 높였다.
- **Anti-CoT 조건**에서도 GSM8K 정확도가 11%에서 21%로 상승했다.
- AQuA-RAT에서도 PDS는 모든 조건에서 DoM을 소폭 앞섰다.
- 따라서 PDS의 효과는 모델에게 명시적으로 “단계별로 생각하라”고 지시한 경우뿐 아니라, **추론 지시가 없거나 추론을 억제한 경우에도 나타났다.**

---

### 4. BIG-Bench 결과

BIG-Bench에서는 다음 다섯 태스크를 평가했다.

- Checkmate-in-One
- Physics Questions
- Rhyming
- Temporal Sequences
- Intersect Geometry

#### LLaMA-3-Instruct-8B

PDS는 일부 태스크에서 개선을 보였지만, 모든 태스크에서 일관되게 우수하지는 않았다.

예를 들어:

- **Temporal Sequences, Neutral**: No Steering 57.0% → PDS 58.5%
- **Intersect Geometry, Anti-CoT**: 31.2% → 32.2%
- **Checkmate-in-One, CoT**: 0.2% → 0.6%
- 반면 **Temporal Sequences, CoT**에서는 DoM보다 낮았고, 일부 태스크에서는 No Steering보다 낮은 결과도 나타났다.
- **Physics Questions**는 8B 모델에서 거의 모든 조건의 성능이 0%였다.

#### LLaMA-3-Instruct-70B

70B 모델에서는 기본 성능이 높아져 일부 태스크에서 PDS의 추가 이득이 줄어들었다.

예를 들어:

- **Checkmate-in-One, CoT**: 9.8% → PDS 10.4%
- **Checkmate-in-One, Anti-CoT**: 14.8% → **15.8%**
- **Physics Questions, Anti-CoT**: 0% → **6.2%**
- **Temporal Sequences**: 이미 98% 이상으로 높아 PDS의 추가 개선이 거의 없었다.
- **Intersect Geometry**에서는 PDS가 DoM이나 기본 모델보다 낮은 조건도 있었다.

즉, BIG-Bench 결과는 **PDS가 대체로 DoM과 비슷하거나 더 좋지만, 태스크에 따라 성능 변동과 소폭의 회귀가 존재한다**는 점을 보여준다.

---

### 5. 모델 규모에 따른 결과

논문은 모델이 커질수록 PDS의 개선 폭이 줄어드는 경향을 보고했다.

- **8B급 모델**: 기본 추론 능력이 충분히 사용되지 않아 PDS의 개선 폭이 큼
- **70B급 모델**: 이미 높은 수준의 추론 능력을 보유하여 추가적인 개선 여지가 작음
- **Gemma-2-9B-IT**에서도 같은 경향이 관찰됨

Gemma-2-9B-IT 결과의 예시는 다음과 같다.

| 데이터셋 및 조건 | Baseline | DoM | PDS |
|---|---:|---:|---:|
| GSM8K Neutral | 0.85 | 0.89 | 0.89 |
| GSM8K Anti-CoT | 0.28 | 0.28 | **0.29** |
| AQuA-RAT Neutral | 0.54 | 0.61 | **0.65** |
| AQuA-RAT Anti-CoT | 0.13 | 0.15 | **0.16** |

PDS는 특히 **Neutral 조건의 AQuA-RAT에서 0.54에서 0.65로 크게 향상**되었다.

---

### 6. 출력 길이와 추론 방식에 대한 분석

PDS의 성능 향상이 단순히 더 많은 토큰을 생성했기 때문인지를 확인하기 위해 출력 길이도 측정했다.

- PDS는 DoM보다 출력이 짧거나 비슷한 경우가 많았다.
- AQuA-RAT에서는 PDS의 평균 출력 길이가 No Steering과 DoM보다 짧았지만 정확도는 더 높았다.
- Anti-CoT 조건에서는 모든 방법이 짧은 답변을 생성했다.
- 따라서 PDS는 **출력 길이를 늘려 성능을 높이는 방식이 아니라, 내부 활성값의 방향을 조정하여 추론 품질을 개선하는 방식**으로 해석된다.

또한 Anti-CoT 조건에서도 모델이 단계별 풀이를 길게 출력하지 않으면서 정확도가 향상되었다. 이는 PDS가 명시적인 CoT 행동을 강제로 유도하기보다는 **잠재적인 추론 능력을 강화할 수 있음**을 시사한다.

---

### 7. 종합 평가

논문의 실험 결과를 종합하면 다음과 같다.

- PDS는 GSM8K와 AQuA-RAT에서 **No Steering과 DoM보다 전반적으로 높은 성능**을 보였다.
- 가장 큰 효과는 **Neutral 및 Anti-CoT 조건**에서 나타났다.
- CoT가 이미 제공된 상황에서도 성능을 유지하거나 추가로 개선했다.
- BIG-Bench에서는 결과가 태스크별로 혼합되어, PDS가 모든 문제 유형에 보편적으로 효과적인 것은 아니었다.
- PDS의 이점은 특히 **다단계·구성적 추론이 필요하고, 모델의 잠재적 추론 능력이 충분히 활용되지 않은 경우**에 큰 것으로 보인다.
- 반대로 기본 성능이 이미 포화된 태스크나 사실 기억 중심의 태스크에서는 개선 폭이 작거나 성능이 하락할 수 있다.

결론적으로, PDS는 고정된 하나의 steering 방향을 사용하는 DoM보다 입력별 상황에 맞는 추론 전략을 선택할 수 있으며, **프롬프트나 모델 가중치를 변경하지 않고 테스트 시점에 추론 성능을 높이는 방법**으로 제시되었다.

---



### 1. Compared Methods and Experimental Setup

The paper compares **Prototype-Based Dynamic Steering (PDS)** with:

- **No Steering**: no activation intervention
- **Difference-of-Means (DoM)**: a single fixed steering vector obtained by averaging activation differences between CoT and Neutral prompts
- **PDS**: activation differences are clustered into multiple prototypes, and an input-specific steering vector is constructed according to the alignment between the current input and the prototypes

The evaluated models are:

- **LLaMA-3-Instruct-8B**: main experimental model
- **LLaMA-3-Instruct-70B**: model-scale analysis
- **Gemma-2-9B-IT**: supplementary mid-sized model analysis

Three prompting conditions are used:

1. **CoT**: explicitly encourages step-by-step reasoning
2. **Neutral**: provides no reasoning instruction
3. **Anti-CoT**: asks the model to answer immediately without elaboration

---

### 2. Test Data and Metrics

The evaluation uses three benchmark groups:

| Dataset | Description | Evaluation Size |
|---|---|---:|
| **GSM8K** | Multi-step grade-school math word problems | 1,319 |
| **AQuA-RAT** | Multiple-choice algebraic reasoning problems | 254 |
| **BIG-Bench Subset** | Chess, physics, rhyming, temporal, and geometry tasks | Varies by task |

The main metric is **Accuracy@1**, measuring whether the final answer is correct. The paper also analyzes output token length to determine whether PDS gains simply come from generating longer responses.

---

### 3. GSM8K and AQuA-RAT Results

#### LLaMA-3-Instruct-8B Accuracy

| Prompt Condition | GSM8K No Steering | GSM8K DoM | GSM8K PDS | AQuA-RAT No Steering | AQuA-RAT DoM | AQuA-RAT PDS |
|---|---:|---:|---:|---:|---:|---:|
| **CoT** | 71% | 75% | **78%** | 49% | 50% | **53%** |
| **Neutral** | 68% | 74% | **78%** | 46% | 50% | **53%** |
| **Anti-CoT** | 11% | 18% | **21%** | 29% | 30% | **31%** |

The main findings are:

- PDS outperforms DoM under all tested conditions on GSM8K.
- On GSM8K Neutral, PDS improves accuracy from 68% to 78%.
- Under Anti-CoT prompting, GSM8K accuracy increases from 11% to 21%.
- PDS also slightly outperforms DoM across all AQuA-RAT conditions.
- Therefore, PDS is effective not only when CoT is explicitly requested, but also when reasoning is not encouraged or is actively discouraged.

---

### 4. BIG-Bench Results

The BIG-Bench subset contains:

- Checkmate-in-One
- Physics Questions
- Rhyming
- Temporal Sequences
- Intersect Geometry

For **LLaMA-3-Instruct-8B**, PDS improves performance on some tasks but is not uniformly better across all tasks.

Examples include:

- Temporal Sequences, Neutral: 57.0% → 58.5%
- Intersect Geometry, Anti-CoT: 31.2% → 32.2%
- Checkmate-in-One, CoT: 0.2% → 0.6%
- However, PDS sometimes underperforms DoM or even the unsteered baseline.
- Physics Questions remains near 0% for the 8B model in most conditions.

For **LLaMA-3-Instruct-70B**:

- Checkmate-in-One, CoT: 9.8% → 10.4%
- Checkmate-in-One, Anti-CoT: 14.8% → **15.8%**
- Physics Questions, Anti-CoT: 0% → **6.2%**
- Temporal Sequences is already above 98%, leaving little room for improvement.
- PDS is slightly worse than competing methods on some Intersect Geometry conditions.

Overall, BIG-Bench results are **mixed**: PDS usually matches or outperforms DoM, but task-specific regressions remain.

---

### 5. Effect of Model Scale

The paper reports that PDS gains generally become smaller as model capacity increases.

- **Smaller or mid-sized models** have more unused reasoning capacity, so PDS can produce larger gains.
- **Larger models**, such as the 70B model, are closer to reasoning saturation and have less room for improvement.
- The same trend appears in the Gemma-2-9B-IT experiments.

For example:

| Dataset and Condition | Baseline | DoM | PDS |
|---|---:|---:|---:|
| GSM8K Neutral | 0.85 | 0.89 | 0.89 |
| GSM8K Anti-CoT | 0.28 | 0.28 | **0.29** |
| AQuA-RAT Neutral | 0.54 | 0.61 | **0.65** |
| AQuA-RAT Anti-CoT | 0.13 | 0.15 | **0.16** |

The largest improvement appears on AQuA-RAT Neutral, where PDS increases accuracy from 0.54 to 0.65.

---

### 6. Output Length and Reasoning Behavior

The authors also examine whether PDS improves accuracy simply by producing longer answers.

- PDS outputs are often shorter than or comparable to DoM outputs.
- On AQuA-RAT, PDS produces shorter responses than both the baseline and DoM while achieving higher accuracy.
- Under Anti-CoT prompting, all methods generate short outputs.
- Thus, the gains are attributed to **activation-level refinement**, rather than increased output length.

PDS can also improve accuracy under Anti-CoT prompting without forcing the model to generate visible step-by-step reasoning. This suggests that PDS strengthens latent reasoning processes rather than merely changing the surface form of the response.

---

### 7. Overall Assessment

The results indicate that:

- PDS generally outperforms both No Steering and DoM on GSM8K and AQuA-RAT.
- Its strongest gains appear under Neutral and Anti-CoT prompting.
- It remains useful even when CoT is already present.
- BIG-Bench results are more variable and show that PDS is not universally beneficial for every task.
- PDS is most effective for compositional, multi-step reasoning problems where the model has underused latent reasoning capacity.
- Improvements are smaller or may disappear on tasks with near-saturated performance or tasks dominated by factual recall.

Overall, PDS is presented as a test-time steering method that improves reasoning without changing the model weights or prompts. Its main advantage over DoM is that it uses multiple reasoning prototypes and selects an input-specific combination instead of applying one fixed steering direction to every problem.


<br/>
# 예제
## 1. 논문에서 다루는 전체 구조

PDS는 **트레이닝 데이터를 이용해 모델을 재학습하는 방법이 아니라**, 추론 시점에 모델 내부 활성값을 조정하는 방법입니다.

전체 과정은 다음과 같습니다.

1. **트레이닝 데이터에서 CoT 프롬프트와 Neutral 프롬프트를 각각 입력**
2. 두 입력의 내부 활성값 차이 계산  
   [
   d_i=h(text{CoT})-h(text{Neutral})
   ]
3. 여러 차이 벡터를 K-means로 군집화하여 여러 개의 **reasoning prototype** 생성
4. 테스트 입력의 활성값이 각 prototype과 얼마나 유사한지 계산
5. 테스트 입력마다 서로 다른 steering vector를 만들고 모델 내부에 주입
6. 모델의 답변 정확도 비교

즉, 일반적인 DoM은 모든 문제에 같은 방향을 적용하지만, PDS는 **문제마다 다른 reasoning 전략의 조합을 적용**합니다.

---

## 2. 입력 프롬프트의 세 가지 유형

논문은 같은 문제에 대해 다음 세 가지 프롬프트를 사용합니다.

| 유형 | 입력 형식 | 모델에게 요구하는 것 |
|---|---|---|
| Neutral | `Question: [문제]. Answer:` | 별도의 추론 지시 없음 |
| CoT | `Question: [문제]. Let's think step by step. Answer:` | 단계별 풀이 유도 |
| Anti-CoT | `Question: [문제]. Answer immediately, without elaboration. Answer:` | 풀이 과정 없이 즉시 답변 |

### 예시: 산수 문제

문제:

> A pencil costs $1.25, and a notebook costs twice as much. How much do five notebooks and three pencils cost in total?

#### Neutral 입력

```text
Question: A pencil costs $1.25, and a notebook costs twice as much.
How much do five notebooks and three pencils cost in total?
Answer:
```

가능한 출력:

```text
$16.25
```

#### CoT 입력

```text
Question: A pencil costs $1.25, and a notebook costs twice as much.
How much do five notebooks and three pencils cost in total?
Let's think step by step. Answer:
```

가능한 출력:

```text
A notebook costs $2.50.
Five notebooks cost $12.50.
Three pencils cost $3.75.
Therefore, the total cost is $16.25.
```

#### Anti-CoT 입력

```text
Question: A pencil costs $1.25, and a notebook costs twice as much.
How much do five notebooks and three pencils cost in total?
Answer immediately, without elaboration. Answer:
```

가능한 출력:

```text
$16.25
```

Anti-CoT에서는 출력은 짧게 유지하지만, PDS가 모델의 내부 reasoning-related activation을 조정하여 정확도를 높이는지를 평가합니다.

> 위 계산은 논문에 제시된 예시 문제를 바탕으로 한 설명입니다. 논문은 모든 개별 학습·테스트 샘플의 실제 모델 출력과 정답을 전부 공개하지는 않습니다.

---

## 3. 트레이닝 데이터에서 하는 일

### 3.1 GSM8K의 예시

GSM8K는 초등학교 수준의 여러 단계 수학 문장제입니다.

트레이닝 데이터의 하나의 문제 (x_i)에 대해 두 가지 입력을 만듭니다.

```text
CoT:
Question: [문제]
Let's think step by step. Answer:

Neutral:
Question: [문제]
Answer:
```

두 입력을 모델에 통과시킨 뒤, 특정 중간 레이어의 마지막 입력 토큰 hidden state를 추출합니다.

```text
h_CoT      = CoT 프롬프트의 hidden state
h_Neutral  = Neutral 프롬프트의 hidden state
d_i        = h_CoT - h_Neutral
```

여기서 (d_i)는 단순히 정답 문장 자체가 아니라, 해당 문제가 **직접 답하기에서 단계적 추론 방향으로 바뀔 때 나타나는 내부 활성값 변화**입니다.

GSM8K의 예시적인 문제 유형은 다음과 같습니다.

- 여러 수의 계산
- 단위 변환
- 가격·수량 계산
- 순차적인 산술 연산
- 여러 단계의 문장제 풀이

논문에서는 GSM8K 학습 세트 7,470개를 prototype 생성에 사용하고, 테스트 세트 1,319개를 평가에 사용했습니다.

---

### 3.2 AQuA-RAT의 예시

AQuA-RAT는 선택지가 있는 대수·수리 문제입니다.

논문에 제시된 예시 유형은 다음과 같습니다.

> What is the value of (x) if three times the difference between (x) and 4 is equal to 18?

이를 수식으로 표현하면 다음과 같습니다.

[
3(x-4)=18
]

정답은:

[
x-4=6,quad x=10
]

실제 평가에서는 보통 모델이 숫자 자체 또는 정답 선택지에 해당하는 답을 출력하고, 그 결과가 정답 선택지와 일치하는지를 측정합니다.

AQuA-RAT의 주요 문제 유형은 다음과 같습니다.

- 대수식 조작
- 비율과 변화율 계산
- 패턴 인식
- 선택지 비교 및 오답 제거
- 문장 표현의 의미 해석

논문에서는 AQuA-RAT 학습 데이터에서 10,000개를 prototype 생성에 사용하고, 254개 평가 샘플에서 성능을 측정했습니다.

---

### 3.3 BIG-Bench의 예시

BIG-Bench에서는 서로 다른 능력을 평가하는 다섯 가지 태스크를 사용했습니다.

| 태스크 | 평가하는 능력 | 가능한 입력과 출력의 형태 |
|---|---|---|
| Checkmate-in-One | 체스판에서 한 수 만에 체크메이트 찾기 | 체스판 설명 → 최선의 수 또는 정답 선택지 |
| Physics Questions | 물리 개념 및 상식적 물리 추론 | 물리 질문 → 정답 선택지 |
| Rhyming | 단어의 운율·음운 관계 판단 | 단어 또는 보기 → 운율이 맞는 답 |
| Temporal Sequences | 사건의 시간적 순서 추론 | 사건 목록 → 올바른 순서 |
| Intersect Geometry | 도형 및 교차 관계 판단 | 도형 설명 → 교차 개수·관계 |

예를 들어 Temporal Sequences는 다음과 같은 형태일 수 있습니다.

```text
Event A happened before Event B.
Event C happened after Event A but before Event B.
What is the correct order?
```

가능한 정답:

```text
A, C, B
```

다만 논문은 BIG-Bench의 모든 실제 문항과 정답 문장을 본문에 열거하지 않고, 태스크 이름과 샘플 수만 제시합니다.

BIG-Bench의 prototype 생성용 트레이닝 샘플과 테스트 샘플 수는 다음과 같습니다.

| 태스크 | 트레이닝 샘플 | 테스트 샘플 |
|---|---:|---:|
| Checkmate-in-One | 1,000 | 699 |
| Physics Questions | 38 | 16 |
| Rhyming | 763 | 190 |
| Temporal Sequences | 800 | 200 |
| Intersect Geometry | 1,000 | 1,000 |

---

## 4. 테스트 데이터에서 하는 일

테스트 단계에서는 새로운 문제를 입력합니다. 이때 테스트 문제 자체에 대한 정답이나 CoT 출력이 필요한 것은 아닙니다.

예를 들어 테스트 입력이 다음과 같다고 하겠습니다.

```text
Question: Solve 2(x + 3) = x + 9.
Answer:
```

모델은 이 입력을 생성 없이 먼저 처리하고, 마지막 입력 토큰의 hidden state를 얻습니다.

```text
h_input
```

그 다음 학습 단계에서 얻은 여러 prototype과의 projection을 계산합니다.

[
v_{text{steer}}(h_{text{input}})
=
sum_{j=1}^{k}text{proj}_{mu_j}(h_{text{input}})
]

이 벡터를 모델의 residual stream에 추가합니다.

[
h'_{text{input}}
=
h_{text{input}}
+
alpha v_{text{steer}}(h_{text{input}})
]

그 후 모델이 답을 생성합니다.

가능한 출력:

```text
x = 3
```

중요한 점은 모든 테스트 문제에 같은 steering vector를 적용하지 않는다는 것입니다.

- 산술 문제가 현재 입력과 유사하면 산술 관련 prototype의 영향이 커짐
- 대수 문제가 현재 입력과 유사하면 대수·관계 추론 prototype의 영향이 커짐
- 시간 순서나 도형 문제라면 해당 문제와 가까운 prototype들이 더 많이 기여함

따라서 PDS의 출력은 문제에 따라 다음처럼 달라질 수 있습니다.

```text
산술 문제 → 계산 관련 prototype 조합
대수 문제 → 식 변환·관계 추론 prototype 조합
시간 순서 문제 → 순차적 논리 추론 prototype 조합
```

---

## 5. 실제 평가에서 비교한 방법

각 테스트 문제에 대해 다음 세 가지 방법을 비교했습니다.

| 방법 | 설명 |
|---|---|
| No Steering / Baseline | 내부 활성값을 수정하지 않음 |
| DoM | 모든 문제에 하나의 평균 steering vector 적용 |
| PDS | 입력과 prototype의 유사도에 따라 문제별 steering vector 적용 |

### GSM8K와 AQuA-RAT 결과

논문 Table 1의 주요 결과는 다음과 같습니다.

| 조건 | GSM8K Baseline | GSM8K DoM | GSM8K PDS | AQuA-RAT Baseline | DoM | PDS |
|---|---:|---:|---:|---:|---:|---:|
| CoT | 71% | 75% | **78%** | 49% | 50% | **53%** |
| Neutral | 68% | 74% | **78%** | 46% | 50% | **53%** |
| Anti-CoT | 11% | 18% | **21%** | 29% | 30% | **31%** |

특히 Anti-CoT 조건에서 PDS는 풀이 과정을 길게 출력하지 않으면서도 성능을 높였습니다. 즉, 출력 형식을 바꾸기보다는 모델 내부의 문제 해결 능력을 보강한 것으로 해석됩니다.

---

## 6. 핵심 정리

- **트레이닝 데이터**: CoT와 Neutral 프롬프트를 모두 넣어 두 hidden state의 차이를 추출합니다.
- **Prototype 생성**: 문제별 활성값 차이를 K-means로 묶어 여러 reasoning prototype을 만듭니다.
- **테스트 데이터**: 새로운 입력의 hidden state를 prototype들과 비교하여 해당 문제에 맞는 steering vector를 만듭니다.
- **출력**: GSM8K는 수치 답, AQuA-RAT는 선택지 또는 수치 답, BIG-Bench는 각 태스크의 정답 형식으로 평가합니다.
- **PDS의 특징**: 모든 문제에 같은 벡터를 적용하지 않고, 문제별로 다른 reasoning 전략을 조합합니다.
- **논문의 주요 주장**: 명시적인 CoT 지시가 없거나 Anti-CoT로 추론을 억제한 경우에도, 출력은 간결하게 유지하면서 정확도를 높일 수 있습니다.

---




## 1. Overall structure

PDS is not a training or fine-tuning method. It modifies the model’s internal activations at inference time.

The process is:

1. Feed each training problem with both a CoT prompt and a Neutral prompt.
2. Compute the activation difference:
   [
   d_i=h(text{CoT})-h(text{Neutral})
   ]
3. Cluster these difference vectors with K-means.
4. Treat the cluster centroids as reasoning prototypes.
5. For a new test input, compare its hidden state with the prototypes.
6. Construct an input-specific steering vector and inject it into the model.

DoM uses one global average direction for every problem, whereas PDS dynamically combines different reasoning prototypes for each input.

---

## 2. Three prompt conditions

| Condition | Prompt format | Purpose |
|---|---|---|
| Neutral | `Question: [problem]. Answer:` | No explicit reasoning instruction |
| CoT | `Question: [problem]. Let's think step by step. Answer:` | Encourages step-by-step reasoning |
| Anti-CoT | `Question: [problem]. Answer immediately, without elaboration. Answer:` | Requires a short answer without explanation |

### Example: arithmetic word problem

```text
Question: A pencil costs $1.25, and a notebook costs twice as much.
How much do five notebooks and three pencils cost in total?
```

A possible Neutral or Anti-CoT output is:

```text
$16.25
```

A possible CoT output is:

```text
A notebook costs $2.50.
Five notebooks cost $12.50.
Three pencils cost $3.75.
Therefore, the total cost is $16.25.
```

The paper uses this type of contrast to identify activation changes associated with reasoning. It does not provide every individual training and test example with its exact model output.

---

## 3. Training data and prototype construction

### GSM8K

GSM8K contains grade-school mathematical word problems involving multi-step arithmetic.

For each training problem, the authors create:

```text
CoT:
Question: [problem]
Let's think step by step. Answer:

Neutral:
Question: [problem]
Answer:
```

They extract hidden states from both versions and compute:

```text
d_i = h_CoT - h_Neutral
```

These vectors may reflect differences related to:

- Multi-step arithmetic
- Unit conversion
- Price and quantity calculations
- Sequential operations
- Decomposition of word problems

The paper uses 7,470 GSM8K training examples to generate prototypes and evaluates on 1,319 test examples.

---

### AQuA-RAT

AQuA-RAT contains multiple-choice algebraic word problems.

An example type given in the paper is:

> What is the value of (x) if three times the difference between (x) and 4 is equal to 18?

This corresponds to:

[
3(x-4)=18
]

so:

[
x=10
]

Typical AQuA-RAT tasks involve:

- Algebraic manipulation
- Ratios and rates
- Pattern recognition
- Distractor elimination
- Interpreting mathematical wording

The authors use 10,000 training questions for prototype construction and evaluate on 254 examples.

---

### BIG-Bench subset

The paper evaluates five BIG-Bench tasks:

| Task | Main ability | Typical input/output form |
|---|---|---|
| Checkmate-in-One | Chess reasoning | Board description → best move |
| Physics Questions | Physical reasoning | Physics question → answer choice |
| Rhyming | Phonetic reasoning | Words/options → rhyming answer |
| Temporal Sequences | Temporal ordering | Events → correct order |
| Intersect Geometry | Geometric reasoning | Shape description → intersection relation or count |

For example:

```text
Event A happened before Event B.
Event C happened after A but before B.
What is the correct order?
```

Possible answer:

```text
A, C, B
```

The paper does not list every actual BIG-Bench item and gold answer, but it reports the number of training and test examples:

| Task | Training examples | Test examples |
|---|---:|---:|
| Checkmate-in-One | 1,000 | 699 |
| Physics Questions | 38 | 16 |
| Rhyming | 763 | 190 |
| Temporal Sequences | 800 | 200 |
| Intersect Geometry | 1,000 | 1,000 |

---

## 4. Inference on test data

At test time, a new problem is first processed without generation to obtain its hidden state.

For example:

```text
Question: Solve 2(x + 3) = x + 9.
Answer:
```

The model produces an input representation (h_{text{input}}). PDS then projects this representation onto the learned prototypes:

[
v_{text{steer}}(h_{text{input}})
=
sum_{j=1}^{k}text{proj}_{mu_j}(h_{text{input}})
]

The steering vector is injected as:

[
h'_{text{input}}
=
h_{text{input}}
+
alpha v_{text{steer}}(h_{text{input}})
]

The model may then produce:

```text
x = 3
```

Different inputs activate different combinations of prototypes:

- Arithmetic problem → arithmetic-related prototypes
- Algebra problem → symbolic manipulation prototypes
- Temporal task → sequential reasoning prototypes
- Geometry task → spatial reasoning prototypes

Thus, PDS does not apply exactly the same intervention to every test example.

---

## 5. Evaluation methods and results

The paper compares:

| Method | Description |
|---|---|
| No Steering / Baseline | No activation intervention |
| DoM | One global average steering vector |
| PDS | Input-specific vector built from prototype projections |

Main results on GSM8K and AQuA-RAT:

| Condition | GSM8K Baseline | DoM | PDS | AQuA-RAT Baseline | DoM | PDS |
|---|---:|---:|---:|---:|---:|---:|
| CoT | 71% | 75% | **78%** | 49% | 50% | **53%** |
| Neutral | 68% | 74% | **78%** | 46% | 50% | **53%** |
| Anti-CoT | 11% | 18% | **21%** | 29% | 30% | **31%** |

The Anti-CoT results are especially important: PDS improves accuracy while preserving the requested concise output format rather than forcing the model to produce a visible chain of thought.

---

## 6. Key takeaway

- **Training stage**: Extract CoT–Neutral activation differences from training problems.
- **Prototype stage**: Cluster those differences into multiple reasoning prototypes.
- **Test stage**: Project each new input onto the prototypes and construct a customized steering vector.
- **Outputs**: Numeric answers for GSM8K, answer choices or values for AQuA-RAT, and task-specific answers for BIG-Bench.
- **Main advantage**: PDS selects a problem-specific combination of reasoning directions instead of applying one universal vector.
- **Main claim**: PDS can improve latent reasoning even without explicit CoT instructions and even when the prompt asks the model not to elaborate.

<br/>
# 요약


PDS는 CoT와 중립 프롬프트의 활성화 차이를 여러 프로토타입으로 군집화한 뒤, 입력과의 유사도에 따라 맞춤형 조향 벡터를 만들어 추론 시 주입하는 방법이다.  
LLaMA-3-8B에서 GSM8K는 중립 조건 기준 68%→78%, AQuA-RAT는 46%→53%로 향상됐고, Anti-CoT에서도 GSM8K 11%→21%로 개선되어 추가적인 단계별 출력 없이 잠재적 추론을 강화했다.  
예를 들어 산술·단위 변환·비율 문제처럼 서로 다른 풀이 전략이 필요한 문제에 각기 다른 프로토타입 조합을 적용하며, BIG-Bench에서는 체스·물리·시간 순서 등 과제별 성능이 혼합된 결과를 보였다.  



PDS clusters activation differences between Chain-of-Thought and neutral prompts into multiple prototypes, then constructs and injects an input-specific steering vector based on the current input’s similarity to them.  
With LLaMA-3-8B, accuracy improved from 68% to 78% on GSM8K and from 46% to 53% on AQuA-RAT under neutral prompting, while GSM8K under Anti-CoT rose from 11% to 21%, indicating stronger latent reasoning without requiring step-by-step outputs.  
For example, different prototype combinations can support arithmetic, unit-conversion, and ratio problems with distinct solution strategies, although results on BIG-Bench were mixed across tasks such as chess, physics, and temporal reasoning.

<br/>
# 기타



### 1. 다이어그램·피규어

**그림 1: DoM과 PDS의 비교**

- **DoM(Difference-of-Means)**은 모든 입력에 동일한 하나의 steering vector를 적용한다.
- **PDS**는 여러 reasoning prototype을 학습한 뒤, 각 입력의 activation이 어떤 prototype과 유사한지에 따라 입력별 steering vector를 만든다.
- 핵심 인사이트는 reasoning을 하나의 고정된 방향으로 보지 않고, 문제 유형에 따라 달라지는 여러 전략의 조합으로 본다는 점이다.

**그림 2: Prototype discovery 과정**

1. 동일한 문제에 대해 CoT prompt와 Neutral prompt를 각각 입력한다.
2. 두 activation의 차이  
   [
   d_i=h_l(p_{i,mathrm{CoT}})-h_l(p_{i,mathrm{neutral}})
   ]
   를 계산한다.
3. 여러 문제에서 얻은 activation difference를 K-means로 clustering한다.
4. 각 cluster centroid를 하나의 reasoning prototype으로 사용한다.

- 즉, prototype은 단순히 평균적인 “reasoning 방향”이 아니라, 산술·비율·논리·다단계 계산 등 서로 다른 문제 해결 패턴을 나타내도록 설계된다.

**그림 3: Inference-time steering**

- 새로운 입력의 hidden state를 각 prototype에 projection한다.
- projection 결과를 합쳐 입력별 steering vector를 만들고, 이를 residual stream에 주입한다.
- 개입은 주로 **첫 번째 output token**에 적용된다.
- 목적은 이후 전체 문장을 강제로 바꾸기보다, reasoning을 시작하는 초기 activation을 조정하는 것이다.

---

### 2. 주요 결과 테이블

#### Table 1: GSM8K와 AQuA-RAT

| 조건 | GSM8K Baseline / DoM / PDS | AQuA-RAT Baseline / DoM / PDS |
|---|---:|---:|
| CoT | 71 / 75 / **78** | 49 / 50 / **53** |
| Neutral | 68 / 74 / **78** | 46 / 50 / **53** |
| Anti-CoT | 11 / 18 / **21** | 29 / 30 / **31** |

**인사이트**

- PDS는 두 데이터셋과 세 prompting 조건에서 모두 baseline 및 DoM보다 높은 성능을 보였다.
- 특히 **Neutral**과 **Anti-CoT**에서 효과가 크다.
- CoT를 명시적으로 주지 않아도 PDS가 reasoning 성능을 높이며, Anti-CoT에서도 출력 형식을 크게 바꾸지 않고 정확도를 개선한다.
- GSM8K에서 Anti-CoT 성능은 11%에서 21%로 상승해, 명시적인 step-by-step 출력 없이도 latent reasoning이 강화될 가능성을 보여준다.

#### Table 2: BIG-Bench, LLaMA-3-8B

- 결과는 작업별로 혼합되어 있다.
- **Temporal Sequences**와 **Intersect Geometry** 등에서는 PDS가 baseline 또는 DoM을 개선하는 경우가 있다.
- 반면 Checkmate-in-One, Rhyming 등에서는 개선이 작거나 baseline과 동일하며, 일부 조건에서는 감소도 나타난다.
- 따라서 PDS의 효과는 모든 task에 동일하지 않고, task가 요구하는 reasoning 구조와 모델 능력에 의존한다.

#### Table 3: BIG-Bench, LLaMA-3-70B

- 70B 모델에서는 전반적으로 정확도가 더 높다.
- PDS는 Checkmate-in-One, Rhyming, Temporal Sequences 등에서 baseline과 비슷하거나 일부 개선을 보인다.
- 그러나 DoM이나 PDS가 항상 baseline을 능가하지는 않는다.
- 이는 모델 규모가 커질수록 이미 reasoning 능력이 충분히 활성화되어 있어 추가 steering의 여지가 줄어드는 **reasoning saturation** 현상과 연결된다.

#### Table 4: Anti-CoT 성능

- GSM8K: 11.60% → 21.12%, **+9.52%**
- AQuA-RAT: 29.50% → 31.50%, **+2.00%**
- Physics Questions: 0% → 6.2%, **+6.2%**
- Checkmate-in-One: 14.8% → 15.8%, **+1.0%**
- Rhyming과 Intersect Geometry에서는 각각 소폭 하락했다.

**핵심 해석**

- PDS는 “설명하지 말고 즉시 답하라”는 지시를 반드시 CoT 출력으로 바꾸지 않는다.
- 즉, 출력 길이나 형식은 유지하면서 내부 activation 수준에서 문제 해결 능력을 높이는 방향으로 작동할 수 있다.
- 다만 모든 task에서 안정적으로 이득을 보장하지는 않는다.

---

### 3. Prototype 해석 가능성

#### Table 5: GSM8K prototype cluster

GPT-4.5, Gemini 2.5 Pro, Claude Sonnet 4가 각 cluster의 문제를 독립적으로 분류했다.

주요 cluster 유형은 다음과 같다.

- 다단계 산술
- 비교·관계량 문제
- 비율 및 나이 문제
- 금융·비용 계산
- 단위 변환
- 백분율·분수
- 순차 연산 및 패턴
- 복합 조건 문제

세 모델의 label이 대체로 유사했다.

**인사이트**

- activation difference를 clustering한 결과가 단순한 임의의 통계적 그룹이 아니라, 사람이 이해할 수 있는 문제 해결 전략과 대응한다.
- 다만 이는 prototype이 실제로 특정 reasoning mechanism을 인과적으로 구현한다는 증명이라기보다, cluster의 **행동적·의미적 해석 가능성**을 보여주는 결과다.

#### Table 6: BIG-Bench prototype cluster

- Rhyming
- Geometric intersection
- Chess checkmate-in-one
- Time interval 및 schedule reasoning

처럼 서로 다른 task 유형이 비교적 명확히 분리되었다.

#### Table 7: AQuA-RAT prototype cluster

- 기본 수치 계산
- 금융·work-rate 문제
- 기하·확률
- 복잡한 대수 및 변화율
- 실생활 기반 수량 문제

등으로 해석되었다.

**종합 인사이트**

- PDS prototype은 데이터셋마다 따로 학습된다.
- 따라서 GSM8K에서 얻은 prototype을 AQuA-RAT이나 BIG-Bench에 직접 적용하는 방식이 아니다.
- 이는 task-specific reasoning 구조를 반영하는 장점이 있지만, 동시에 다른 도메인으로의 일반화에는 제한이 될 수 있다.

---

### 4. 어펜딕스 결과

#### 데이터 규모

- **GSM8K**: 학습 7,470개, 테스트 1,319개
- **AQuA-RAT**: 학습에서 10,000개 샘플, 평가 254개
- **BIG-Bench**: task별로 38~1,000개의 training sample과 16~1,000개의 test sample 사용

각 데이터셋의 training split에서 prototype을 만들고, 공식 test set에서 평가하여 test leakage를 피하려 했다.

#### Table 8: 공통 inference 설정

- 모델: LLaMA-3-Instruct-8B
- Temperature: 0.7
- Top-p: 0.9
- 최대 생성 길이: 4,096 tokens
- Sampling 방식: temperature sampling

#### Table 9: Gemma-2-9B-IT

| 데이터셋·조건 | Baseline | DoM | PDS |
|---|---:|---:|---:|
| GSM8K Neutral | .85 | .89 | .89 |
| GSM8K Anti-CoT | .28 | .28 | **.29** |
| AQuA-RAT Neutral | .54 | .61 | **.65** |
| AQuA-RAT Anti-CoT | .13 | .15 | **.16** |

**인사이트**

- PDS의 효과가 LLaMA 계열에만 국한되지 않고 Gemma에서도 관찰된다.
- 특히 AQuA-RAT Neutral에서 DoM보다 더 큰 개선을 보였다.
- 그러나 모델의 reasoning capacity가 제한되거나 Anti-CoT처럼 출력이 강하게 제한된 조건에서는 이득이 작다.

#### Table 10: BIG-Bench 샘플 수

- Checkmate-in-One: 1,000 / 699
- Physics Questions: 38 / 16
- Rhyming: 763 / 190
- Temporal Sequences: 800 / 200
- Intersect Geometry: 1,000 / 1,000  
  *(training / test)*

Physics Questions처럼 데이터가 매우 적은 task는 결과 변동성이 클 가능성이 있다.

#### Table 11: 출력 token 길이

- PDS는 대체로 DoM보다 짧거나 비슷한 길이의 출력을 생성했다.
- AQuA-RAT에서 PDS는 더 짧은 completion을 생성하면서도 높은 정확도를 보였다.
- Anti-CoT에서는 모든 방법의 출력 길이가 매우 짧았다.

**인사이트**

- 성능 향상이 단순히 더 많은 token을 생성했기 때문은 아니다.
- PDS는 추가적인 reasoning token을 늘리기보다, 초기 activation을 조정해 더 효율적으로 답을 찾을 가능성을 보여준다.

#### Table 12: Intervention layer와 steering strength

- GSM8K와 AQuA-RAT: layer 16
- BIG-Bench: layer 15
- 일반적으로 CoT와 Neutral에서는 α=1 또는 중간 수준의 값을 사용했다.
- Anti-CoT에서는 reasoning 억제가 강하기 때문에 더 큰 steering strength가 필요했다.
  - GSM8K: α=10
  - AQuA-RAT: α=10

**인사이트**

- 중간~후반 layer가 reasoning-related signal을 잘 포함하는 것으로 관찰되었다.
- PDS와 DoM은 동일한 layer와 α를 사용해 construction method 자체를 비교했다.

---

### 5. 어펜딕스의 추가 분석과 한계

- prototype 수를 elbow method로 정한 뒤, 그 주변에서 약간 바꾸어도 성능이 크게 변하지 않았다.
- intervention layer도 중간 layer 범위에서는 비교적 안정적이었다.
- prototype을 normalize해도 성능이 거의 동일했으므로, PDS의 효과는 activation의 단순한 크기 증폭보다는 **방향성 조정**에 가깝다.
- 큰 모델일수록 baseline reasoning이 이미 강해 PDS의 추가 이득이 줄어드는 경향이 있다.
- 사실 회상이나 domain-specific knowledge가 핵심인 task보다, compositional·step-by-step reasoning task에서 PDS가 더 유리하다.
- prototype 생성에 contrastive dataset이 필요하며, 다른 도메인으로 일반화가 제한될 수 있다.
- activation steering은 reasoning 향상뿐 아니라 jailbreak나 원치 않는 행동을 만드는 데 악용될 가능성도 있다.

### 전체 결론

표와 부록의 결과는 PDS가 단순히 CoT 문구를 출력하도록 유도하는 방법이 아니라, 문제별 activation에 맞춰 reasoning 방향을 선택하는 방법임을 뒷받침한다. 가장 강한 효과는 **명시적인 reasoning 지시가 없거나 억제된 조건**에서 나타났고, 반대로 이미 높은 성능을 내는 대형 모델이나 reasoning이 핵심이 아닌 task에서는 효과가 제한적이었다.

---




### 1. Diagrams and Figures

**Figure 1: DoM vs. PDS**

- **Difference-of-Means (DoM)** applies one global steering vector to every input.
- **PDS** learns multiple reasoning prototypes and constructs an input-specific steering vector according to the similarity between the current activation and each prototype.
- The main idea is that reasoning is not a single fixed behavior, but a combination of different strategies depending on the problem.

**Figure 2: Prototype discovery**

1. Run CoT and Neutral prompts for the same problem.
2. Compute the activation difference:
   [
   d_i=h_l(p_{i,mathrm{CoT}})-h_l(p_{i,mathrm{neutral}})
   ]
3. Cluster these differences using K-means.
4. Use each cluster centroid as a reasoning prototype.

The prototypes are intended to capture patterns such as multi-step arithmetic, ratios, logical deduction, and other problem-solving strategies rather than only one average reasoning direction.

**Figure 3: Inference-time steering**

- The new input activation is projected onto each learned prototype.
- The projections are summed to produce an input-specific steering vector.
- This vector is injected into the residual stream, mainly at the first output token.
- The goal is to influence the initiation of reasoning without forcing the entire completion into a different output format.

---

### 2. Main Results

#### Table 1: GSM8K and AQuA-RAT

| Condition | GSM8K Baseline / DoM / PDS | AQuA-RAT Baseline / DoM / PDS |
|---|---:|---:|
| CoT | 71 / 75 / **78** | 49 / 50 / **53** |
| Neutral | 68 / 74 / **78** | 46 / 50 / **53** |
| Anti-CoT | 11 / 18 / **21** | 29 / 30 / **31** |

**Key findings**

- PDS outperforms both the unsteered baseline and DoM across all conditions.
- The gains are particularly strong under **Neutral** and **Anti-CoT** prompting.
- PDS improves reasoning even without an explicit step-by-step instruction.
- On GSM8K Anti-CoT, accuracy increases from 11% to 21%, suggesting that latent reasoning can be enhanced without requiring visible CoT output.

#### Tables 2 and 3: BIG-Bench with 8B and 70B models

- Results are more mixed than on GSM8K and AQuA-RAT.
- PDS helps on some tasks, such as Temporal Sequences and certain geometry or chess conditions, but produces little improvement or occasional regressions on others.
- The 70B model generally achieves higher baseline accuracy, leaving less room for additional steering gains.
- This supports the paper’s claim of **reasoning saturation**: larger models may already encode strong reasoning directions, reducing the amount of underused structure that PDS can exploit.

#### Table 4: Anti-CoT results

- GSM8K: 11.60% → 21.12%, **+9.52%**
- AQuA-RAT: 29.50% → 31.50%, **+2.00%**
- Physics Questions: 0% → 6.2%, **+6.2%**
- Checkmate-in-One: 14.8% → 15.8%, **+1.0%**
- Rhyming and Intersect Geometry show small decreases.

The important point is that PDS does not necessarily override the instruction to answer immediately. It can preserve concise output while improving problem-solving at the activation level. However, the gains are not uniform across all tasks.

---

### 3. Interpretability of the Prototypes

#### Table 5: GSM8K clusters

The clusters were independently labeled by GPT-4.5, Gemini 2.5 Pro, and Claude Sonnet 4. The labels included:

- Multi-step arithmetic
- Comparative and relational quantities
- Ratios and age problems
- Financial and cost calculations
- Unit conversions
- Percentages and fractions
- Sequential operations and patterns
- Complex multi-condition problems

The labels from the three models were broadly consistent.

**Insight:** The clusters appear to correspond to human-interpretable problem-solving types rather than arbitrary statistical groups. This supports the interpretability of the prototypes, although it does not by itself prove that each prototype is a causally independent reasoning mechanism.

#### Table 6: BIG-Bench clusters

The clusters corresponded to:

- Rhyming
- Geometric intersections
- Chess checkmate-in-one
- Time-interval and schedule reasoning

#### Table 7: AQuA-RAT clusters

The clusters included:

- Basic numerical computation
- Financial and work-rate problems
- Geometry and probability
- Complex algebra and rate-of-change problems
- Applied real-world quantitative reasoning

PDS learns prototypes separately for each dataset. This helps align the prototypes with task-specific reasoning patterns, but it also limits cross-domain generalization.

---

### 4. Appendix Results

#### Dataset sizes

- **GSM8K:** 7,470 training examples and 1,319 test examples
- **AQuA-RAT:** 10,000 sampled training examples and 254 evaluation examples
- **BIG-Bench:** 38–1,000 training examples and 16–1,000 test examples depending on the task

The prototypes were learned from training data and evaluated on separate test data.

#### Table 8: Inference setup

- Model: LLaMA-3-Instruct-8B
- Temperature: 0.7
- Top-p: 0.9
- Maximum generation length: 4,096 tokens
- Sampling: temperature sampling

#### Table 9: Gemma-2-9B-IT

| Dataset and condition | Baseline | DoM | PDS |
|---|---:|---:|---:|
| GSM8K Neutral | .85 | .89 | .89 |
| GSM8K Anti-CoT | .28 | .28 | **.29** |
| AQuA-RAT Neutral | .54 | .61 | **.65** |
| AQuA-RAT Anti-CoT | .13 | .15 | **.16** |

PDS also helps on Gemma-2-9B-IT, especially on AQuA-RAT under Neutral prompting. The gains are smaller when reasoning is strongly constrained by Anti-CoT instructions.

#### Table 10: BIG-Bench sample counts

The number of training/test examples varies substantially across tasks. Physics Questions has especially few examples, so its results may be more variable than those of larger tasks.

#### Table 11: Completion length

- PDS generally produces outputs that are shorter than or comparable to DoM.
- On AQuA-RAT, PDS achieves higher accuracy with shorter completions.
- Under Anti-CoT, all methods generate very short outputs.

This suggests that the gains are not simply caused by producing more reasoning tokens. PDS may improve reasoning efficiency by modifying the initial activation state.

#### Table 12: Intervention layers and steering strength

- GSM8K and AQuA-RAT: layer 16
- BIG-Bench: layer 15
- Anti-CoT uses stronger steering because the prompt actively suppresses reasoning:
  - GSM8K: α=10
  - AQuA-RAT: α=10

The use of mid-to-late layers is consistent with the claim that these layers contain useful reasoning-related representations.

---

### 5. Additional Findings and Limitations

- Performance is relatively stable when the number of prototypes is varied moderately around the elbow-selected value.
- Results are also fairly stable across reasonable mid-layer choices.
- Normalizing the prototypes produces nearly unchanged performance, suggesting that PDS works mainly through **directional steering**, not simple activation magnitude scaling.
- Larger models show smaller gains because their reasoning capacity is closer to saturation.
- PDS is more suitable for compositional and step-by-step reasoning tasks than for factual recall or domain-specific knowledge tasks.
- The method requires contrastive data for prototype construction and may not generalize well to domains with very different reasoning styles.
- Activation steering could also be misused to produce jailbreaks or other undesirable behaviors.

### Overall conclusion

The figures, tables, and appendix results support the view that PDS does more than induce visible CoT behavior. It selects and combines reasoning directions according to the current input. Its strongest benefits appear when explicit reasoning guidance is absent or suppressed, while its impact is more limited for already capable large models and tasks where structured reasoning is not central.

<br/>
# refer format:




### BibTeX

```bibtex
@inproceedings{KayanZhang2026PDS,
  author    = {Kayan, Ceyhun Efe and Zhang, Li},
  title     = {Prototype-Based Dynamic Steering for Large Language Models},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  pages     = {55412--55425},
  year      = {2026},
  month     = oct,
  publisher = {Association for Computational Linguistics},
  note      = {Conference held October 24--29, 2026}
}
```

### 시카고 스타일   

Kayan, Ceyhun Efe, and Li Zhang. “Prototype-Based Dynamic Steering for Large Language Models.” In *Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing*, 55412–55425. Association for Computational Linguistics, 2026.



