---
layout: post
title:  "[2026]Towards a Mechanistic Understanding of Propositional Logical Reasoning in Large Language Models"
date:   2026-11-06 05:16:18 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

어떤 층과 토큰이 어떤 순서와 방식으로 잘 종합하는지..?  (모델 내부에 형성된 일반적인 계산 전략일 가능성을 밝힘)   


짧은 요약(Abstract) :



이 논문은 **대규모 언어 모델(LLM)이 명제 논리 문제를 내부적으로 어떻게 푸는지**를 분석한 연구입니다. 단순히 어떤 부품이 필요한지를 찾는 대신, 모델의 여러 층과 토큰이 **어떤 순서와 방식으로 협력하는지**를 인과적으로 조사했습니다.

연구진은 11가지 명제 논리 규칙과 1-hop·2-hop 문제로 구성된 **PropLogic-MI** 벤치마크를 사용하고, Qwen3·Llama-3.1·Mistral 계열 모델을 분석했습니다. 그 결과, 모델의 논리 추론에서 다음 네 가지 메커니즘이 반복적으로 나타났습니다.

1. **단계적 계산(Staged Computation)**  
   - 초기 층: 주어진 사실 처리  
   - 중간 층: 논리식 계산 및 변수와 사실의 연결  
   - 후기 층: 질문에 대한 최종 답 계산  

2. **정보 전달(Information Transmission)**  
   정보가 모든 토큰에 균등하게 퍼지는 것이 아니라, 사실·논리식·질문을 끝내는 **경계 토큰(boundary token)**에 모인 뒤 다음 단계로 전달됩니다.

3. **사실 회고(Fact Retrospection)**  
   모델은 사실을 초기에 한 번만 읽고 버리지 않습니다. 중간 층에서는 사실의 참·거짓 값을 다시 조회하고, 어려운 문제에서는 후기 층에서도 사실 토큰을 계속 참조합니다.

4. **특화된 어텐션 헤드(Specialized Attention Heads)**  
   특정 어텐션 헤드들이 각각  
   - 논리 구간을 나누고(Splitting),  
   - 정보를 구간의 마지막 토큰으로 모으며(Transmission),  
   - 필요한 사실의 값을 다시 찾아오는(Fact-Retrieval)  
   역할을 수행합니다.

또한 사실을 먼저 제시하는 프롬프트와 질문·논리식을 먼저 제시하는 프롬프트를 비교했는데도 같은 계산 구조가 나타났습니다. 이는 이러한 추론 방식이 단순히 입력 순서나 표면적인 문장 배치 때문에 생긴 것이 아니라, **모델 내부에 형성된 일반적인 계산 전략일 가능성**을 보여줍니다.

즉, 이 연구는 LLM이 명제 논리를 단순한 패턴 matching으로만 처리하는 것이 아니라, 여러 층과 토큰 위치에 걸쳐 역할을 나누는 **구조화된 내부 추론 과정**을 사용한다는 점을 제시합니다.

---




This paper investigates **how large language models internally solve propositional logic problems**. Instead of asking only which individual components are necessary, the authors study how layers, tokens, and attention heads work together as an organized computational process.

They introduce **PropLogic-MI**, a controlled benchmark covering 11 propositional rules and both one-hop and two-hop reasoning. Experiments are conducted on Qwen3, Llama-3.1, and Mistral models. The analysis identifies four recurring mechanisms:

1. **Staged Computation**  
   Early layers mainly process the given facts, middle layers handle logical expressions and variable binding, and later layers compute the final answer.

2. **Information Transmission**  
   Information is not distributed uniformly across all tokens. It tends to accumulate at **semantic boundary tokens**, such as the tokens that end a fact, an expression, or a query.

3. **Fact Retrospection**  
   Fact tokens remain causally important after their initial processing. Middle layers retrieve fact values to bind them to variables, and difficult tasks may cause the model to keep accessing fact tokens even in later layers.

4. **Specialized Attention Heads**  
   Different attention heads specialize in different operations:  
   - **Splitting heads** identify logical segment boundaries,  
   - **Transmission heads** route information toward boundary tokens, and  
   - **Fact-Retrieval heads** retrieve the relevant truth values from the facts.

The authors also reverse the order of the prompt—placing the query before the facts—and still observe the same overall organization. This suggests that the reasoning structure is not merely caused by the surface order of the input, but reflects a more internal and general computational strategy.

Overall, the paper argues that LLMs solve propositional logic through a **structured, layer-organized reasoning process**, rather than relying only on superficial pattern matching.


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



### 1. 사용 모델
이 논문은 세 가지 모델 계열의 **일반적인 사전학습 언어 모델**을 분석했습니다.

- **Qwen3-8B**
- **Qwen3-14B**
- **Llama-3.1-8B-Instruct**
- **Mistral-7B-Instruct**

모두 트랜스포머 기반의 자기회귀 언어 모델이며, 논리 문제를 풀 때 모델 내부에서 정보가 어떻게 이동하는지를 비교했습니다. 모델을 새로 학습하거나 구조를 수정하지 않고, 이미 학습된 모델을 대상으로 분석했습니다.

### 2. 특별한 아키텍처 사용 여부
논문에서는 새로운 신경망 구조나 별도의 논리 모듈을 추가하지 않았습니다.

대신 일반적인 트랜스포머의 다음 구성 요소를 분석했습니다.

- **Attention layer**: 서로 다른 토큰의 정보를 연결하고 전달
- **MLP layer**: 각 토큰의 표현을 변환하고 계산
- **Residual stream**: 여러 층의 정보가 누적·전달되는 경로
- **Attention head**: 특정 기능을 담당하는 세부 attention 단위

분석 결과, 특정 head와 층이 논리 추론 과정에서 다음과 같은 역할을 보이는 것으로 나타났습니다.

1. **Splitting heads**  
   사실, 논리식, 질의와 같은 구간의 경계를 구분합니다.

2. **Transmission heads**  
   한 논리 블록 내부의 정보를 마지막 또는 경계 토큰으로 모읍니다.

3. **Fact-Retrieval heads**  
   논리식이나 질의 위치에서 사실의 True/False 값을 다시 참조합니다.

즉, 특별한 아키텍처를 설계한 것이 아니라, 기존 트랜스포머 내부에서 기능이 분화된 attention head를 발견하고 검증했습니다.

### 3. 학습 데이터와 과제 데이터
모델 자체를 추가로 학습시키기 위한 새로운 트레이닝 데이터는 사용하지 않았습니다. 대신 모델의 논리 추론을 분석하기 위해 **PropLogic-MI**라는 통제된 평가·분석용 데이터셋을 만들었습니다.

데이터셋의 특징은 다음과 같습니다.

- 11가지 명제논리 규칙
  - 예: 드모르간 법칙, 분배법칙, 교환법칙, 결합법칙, 흡수법칙 등
- **One-hop** 문제: 사실에서 답을 직접 계산
- **Two-hop** 문제: 중간 변수나 중간 결과를 먼저 계산한 뒤 최종 답을 계산
- 각 문제에 대해:
  - 원래의 **clean prompt**
  - 사실 하나 등을 바꾸어 정답이 뒤집히도록 만든 **corrupted prompt**
  를 구성
- 사실과 질의의 순서를 바꾼 두 가지 프롬프트 형식 사용
  - 사실 → 논리식 → 질의
  - 논리식·질의 → 사실

또한 모델이 clean과 corrupted 문제를 모두 맞힌 사례만 주요 기계적 분석에 사용했습니다. 따라서 논문의 결론은 모든 실패 사례가 아니라, **모델이 성공적으로 추론한 경우의 내부 계산 방식**에 대한 것입니다.

### 4. 핵심 분석 기법

#### (1) Mean ablation
특정 층, 토큰 위치 또는 블록의 activation을 평균값으로 바꿉니다.

이를 통해 해당 정보가 제거되었을 때 정답 확률이 얼마나 감소하는지 측정했습니다. 분석 대상은 다음과 같습니다.

- Facts block
- Expression block
- Query block
- Attention 출력
- MLP 출력
- 특정 attention head

#### (2) Activation patching
정답을 내는 clean prompt의 내부 activation을 corrupted prompt에 복사합니다.

예를 들어 사실의 True/False 값을 바꾼 corrupted prompt에 clean prompt의 특정 토큰 activation을 넣었을 때 정답 확률이 회복되면, 그 위치와 층이 논리 정보 전달에 인과적으로 관여한다고 판단합니다.

#### (3) Probability Difference
True와 False의 확률 차이를 사용해 개입 효과를 측정했습니다.

- 올바른 답의 확률이 커지면 긍정적 효과
- 올바른 답의 확률이 줄어들면 부정적 효과

이 지표를 통해 어떤 층, 토큰, head가 실제로 추론에 필요한지를 비교했습니다.

#### (4) 토큰 단위 정보 흐름 분석
토큰을 기능별로 나누어 분석했습니다.

- `fact_value`: 사실의 True/False 값
- `fact_var_ref`: 사실을 가리키는 변수 토큰
- `expr_last`: 논리식의 마지막 토큰
- `query_last`: 질의의 마지막 토큰
- 연산자 및 기타 토큰

그 결과 정보가 모든 토큰에 균일하게 퍼지는 것이 아니라, 각 논리 블록의 **경계 토큰**에 집중되는 현상을 확인했습니다.

#### (5) Attention head 분류와 검증
attention pattern을 바탕으로 head를 Splitting, Transmission, Fact-Retrieval 역할로 분류했습니다. 이후 해당 head만 제거했을 때 성능이 크게 떨어지는지 확인하여 분류의 인과적 유효성을 검증했습니다.

### 5. 연구 방법의 핵심 결론
이 논문은 새로운 모델을 훈련한 연구가 아니라, 기존 LLM에 인과적 개입을 가해 내부 추론 과정을 역공학한 연구입니다.

핵심적으로 다음과 같은 계산 순서를 제안합니다.

1. **초기 층**: 사실과 입력 구간을 분리하고 사실 값을 표현
2. **중간 층**: 변수와 사실 값을 연결하고 논리식을 계산
3. **후기 층**: 질의 위치에 정보를 모아 True/False를 결정
4. **어려운 문제**: 중간 계산 이후에도 사실 토큰을 다시 참조

---




### 1. Models
The study analyzes four pretrained language models from three model families:

- **Qwen3-8B**
- **Qwen3-14B**
- **Llama-3.1-8B-Instruct**
- **Mistral-7B-Instruct**

They are standard autoregressive Transformer-based LLMs. The authors do not retrain or modify the models; instead, they investigate how the pretrained models internally solve propositional-logic problems.

### 2. Special Architecture
The paper does not introduce a new neural architecture or an external symbolic reasoning module.

It analyzes existing Transformer components:

- **Attention layers** for routing information between tokens
- **MLP layers** for transforming token representations
- **Residual stream** for carrying information across layers
- **Attention heads** as specialized computational units

The authors identify three functional types of attention heads:

1. **Splitting heads**  
   Detect boundaries between facts, expressions, and queries.

2. **Transmission heads**  
   Aggregate information within a logical block and move it toward a boundary or terminal token.

3. **Fact-Retrieval heads**  
   Retrieve True/False values from fact tokens when processing expressions or queries.

Thus, the study discovers functional specialization inside ordinary Transformers rather than designing a special architecture.

### 3. Data and Training Setup
The models are not further trained on a new dataset. The authors construct **PropLogic-MI**, a controlled benchmark for analysis.

Its main properties are:

- 11 propositional-logic rule categories, including De Morgan’s laws, distributivity, commutativity, associativity, and absorption
- **One-hop reasoning**, where the answer follows directly from the facts
- **Two-hop reasoning**, where an intermediate assignment must be computed first
- Matched **clean/corrupted prompt pairs**
  - The corrupted version changes a fact so that the correct answer flips
- Two prompt orders:
  - Facts before the expression and query
  - Expression/query before the facts

Mechanistic analyses are restricted to examples that the model answers correctly on both clean and corrupted prompts. Therefore, the findings describe the internal computation underlying **successful reasoning**, not necessarily failure cases.

### 4. Main Analytical Techniques

#### (1) Mean ablation
The activation of a selected block, layer, or attention head is replaced with a mean activation.

The authors then measure how much the model’s support for the correct answer decreases. They ablate:

- Fact blocks
- Expression blocks
- Query blocks
- Attention outputs
- MLP outputs
- Individual attention heads

#### (2) Activation patching
An internal activation from a clean prompt is copied into a corrupted prompt.

If patching a particular token and layer restores the correct answer, that location is considered causally involved in transmitting the relevant logical information.

#### (3) Probability-difference metric
The study measures the difference between the probabilities of the True and False tokens.

- A positive shift indicates stronger support for the correct answer.
- A negative shift indicates that the intervention damages the reasoning process.

#### (4) Token-level information-flow analysis
Tokens are grouped according to their structural roles, such as:

- `fact_value`
- `fact_var_ref`
- `expr_last`
- `query_last`
- Operators and other tokens

The analysis shows that information is concentrated at **semantic boundary tokens**, rather than being distributed uniformly across the prompt.

#### (5) Attention-head classification and validation
Attention heads are classified according to their attention patterns into Splitting, Transmission, and Fact-Retrieval roles. The authors then ablate the classified heads and compare the resulting performance with a random-head control.

### 5. Overall Methodological Finding
This is not a paper that trains a new model. It is a causal mechanistic-interpretability study of existing LLMs.

The proposed computation is organized roughly as follows:

1. **Early layers**: identify input segments and encode fact values
2. **Middle layers**: bind variables to fact values and process logical expressions
3. **Late layers**: converge information at the query position and produce True/False
4. **Harder tasks**: continue retrieving fact tokens even in late layers


<br/>
# Results



### 1. 비교 대상 모델

논문은 세 모델 계열의 총 4개 LLM을 비교했다.

- **Qwen3-8B**
- **Qwen3-14B**
- **Llama-3.1-8B-Instruct**
- **Mistral-7B-Instruct**

비교의 핵심은 단순한 정확도뿐 아니라, 서로 다른 모델이 **명제 논리 문제를 내부적으로 어떤 방식으로 처리하는지**가 공통적인지를 확인하는 데 있었다.

---

### 2. 테스트 데이터와 문제 구성

실험 데이터셋은 저자들이 구축한 **PropLogic-MI**이다.

- **명제 논리 규칙 11종**
  - 예: Identity, Contradiction, Commutativity, De Morgan, Distributivity, Absorption 등
- **추론 깊이**
  - **One-hop**: 주어진 사실에서 바로 답을 계산
  - **Two-hop**: 중간 변수의 값을 먼저 계산한 뒤 최종 답을 계산
- **프롬프트 순서**
  - **πFF**: 사실과 논리식이 먼저 나오고 질문이 뒤에 나옴
  - **πEF**: 논리식과 질문이 먼저 나오고 사실이 뒤에 나옴
- 각 순서별로 **20,000개의 clean/corrupted pair**를 사용
  - clean: 원래 문제
  - corrupted: 사실 하나 등을 바꾸어 정답이 바뀌도록 만든 문제
- 모델의 내부 메커니즘 분석은 clean과 corrupted 문제를 **둘 다 맞힌 사례(dual-correct pool)**에 한정했다.

따라서 mechanistic analysis의 결과는 모든 문제에서의 일반적인 추론이 아니라, 각 모델이 **성공적으로 답한 경우의 내부 계산 과정**을 분석한 것이다.

---

### 3. 행동 성능 비교

| 모델 | πFF Clean | πFF Dual | πEF Clean | πEF Dual |
|---|---:|---:|---:|---:|
| Qwen3-8B | 87.0% | 75.4% | 74.2% | 53.9% |
| Qwen3-14B | 87.4% | 76.6% | 74.4% | 52.8% |
| Llama-3.1-8B | 63.9% | 38.4% | 60.3% | 37.0% |
| Mistral-7B | 65.1% | 35.1% | 66.7% | 40.0% |

#### 주요 비교 결과

1. **Qwen3가 전반적으로 가장 높은 성능**
   - 두 크기 모두 Llama와 Mistral보다 높은 정확도를 보였다.
   - Qwen3-14B는 Qwen3-8B보다 약간 더 높지만, 차이는 크지 않았다.

2. **질문 순서가 성능에 큰 영향을 줌**
   - 대부분의 모델에서 πEF 성능이 πFF보다 낮았다.
   - 즉, 사실이 질문보다 뒤에 배치되면 모델이 논리적으로 같은 내용을 처리하더라도 더 어려워했다.

3. **Dual-correct 비율에서 모델 간 차이가 더 크게 나타남**
   - Qwen3는 약 **75%** 수준이었다.
   - Llama-3.1과 Mistral은 약 **35–40%** 수준에 머물렀다.
   - 이는 두 모델이 clean 문제뿐 아니라, 사실이 바뀐 corrupted 문제에서도 일관되게 추론하는 능력이 상대적으로 약하다는 뜻이다.

4. **규칙별 난이도 차이**
   - **Commutativity**는 비교적 쉬웠다.
   - **Contradiction**은 일관되게 어려웠다.
   - Associative, Domination, Excluded Middle 등은 프롬프트 순서에 따라 성능 변동이 컸다.

---

### 4. 사용한 주요 메트릭

#### 4.1 Clean accuracy

원래의 clean 문제에 대해 모델이 정답을 맞힌 비율이다.

#### 4.2 Dual-correct rate

동일한 논리 구조에 대해 clean과 corrupted 문제를 **모두 맞힌 비율**이다.

이 지표는 단순히 답을 맞혔는지보다, 모델이 사실의 변경에 따라 답을 실제로 수정했는지 평가한다. 따라서 mechanistic analysis의 표본을 고르는 기준으로 사용되었다.

#### 4.3 Probability Difference, PD

정답 토큰과 오답 토큰의 확률 차이다.

[
PD = P(text{True}) - P(text{False})
]

정답이 False인 경우에는 부호를 조정하여, 값이 클수록 항상 정답을 지지하도록 계산했다.

#### 4.4 Probability Difference Shift, dPD

activation patching이나 ablation을 수행한 뒤 PD가 얼마나 변했는지를 나타낸다.

- **양의 dPD**: 개입이 모델의 출력을 정답 방향으로 회복
- **음의 dPD**: 개입이 정답에 대한 확신을 감소

이를 통해 특정 층, 토큰, attention head가 추론에 인과적으로 얼마나 중요한지 측정했다.

---

### 5. 내부 추론 과정에 대한 핵심 결과

논문은 네 가지 공통 메커니즘을 발견했다.

#### 5.1 Staged Computation: 층별 단계적 계산

모델은 논리 문제를 모든 층에서 동시에 처리하지 않고, 대략 다음과 같이 나누어 처리했다.

- **초기 층**: 사실(facts)과 truth value 처리
- **중간 층**: 논리식과 변수의 binding, 정보 결합
- **후기 층**: 질문(query)에 대한 최종 계산과 답 선택

이 순서는 Qwen3, Llama, Mistral 모두에서 관찰되었으며, one-hop과 two-hop에서도 유지되었다. 모델마다 단계의 정확한 위치와 강도는 달랐지만, **사실 → 논리식 → 질문**이라는 상대적 순서는 공통적이었다.

#### 5.2 Information Transmission: 경계 토큰으로의 정보 집중

정보는 논리 블록 전체에 균등하게 퍼지지 않고, 각 블록의 끝이나 경계를 표시하는 토큰에 집중되었다.

- 사실 블록: **fact_value**
- 논리식 블록: **expr_last**
- 질문 블록: **query_last**

즉, 모델은 각 의미 단위의 대표적인 마지막 토큰을 정보 집약 지점 또는 허브처럼 사용했다.

#### 5.3 Fact Retrospection: 사실 토큰의 반복적 재참조

사실 토큰은 초기에 한 번 인코딩되고 끝나는 것이 아니었다.

- 중간 층에서는 변수 참조 토큰이 사실의 True/False 값을 다시 조회
- two-hop처럼 어려운 문제에서는 후기 층에서도 사실 토큰이 계속 인과적으로 활성화

특히 과제가 어려울수록 후기 층에서 사실 토큰의 영향력이 커졌다. 저자들은 이를 **Fact Retrospection**이라고 부른다.

#### 5.4 Specialized Attention Heads

Attention head는 기능별로 어느 정도 분업되어 있었다.

- **Splitting Heads**
  - 논리 블록과 절의 경계를 구분
  - 주로 초기 층에 위치
- **Transmission Heads**
  - 블록 내부 정보를 마지막 토큰으로 모음
  - 주로 중간 층에 위치
- **Fact-Retrieval Heads**
  - 질문이나 논리식 위치에서 사실의 True/False 값을 조회
  - 중간에서 후기 층에 집중

이 head들을 제거하면 무작위 head를 제거할 때보다 정확도와 dPD가 크게 떨어졌다. 따라서 이 분류가 단순한 attention 패턴이 아니라 실제 계산 기능과 관련 있음을 확인했다.

---

### 6. 모델별 내부 메커니즘 차이

- **Qwen3**
  - 가장 선명한 단계적 계산 구조
  - 사실 토큰 중심의 정보 흐름이 강함
  - 역할별 head를 제거했을 때 선택적으로 성능이 크게 감소
  - 특히 Fact-Retrieval head와 Transmission head의 기능 분리가 뚜렷함

- **Llama-3.1**
  - 상대적으로 query 중심이며 정보 흐름이 더 분산됨
  - 세 가지 head 역할은 관찰되지만, 작은 수의 head를 제거해도 성능이 비슷한 수준까지 빠르게 하락
  - 즉, head별 기능 분리가 Qwen3만큼 선명하지 않음

- **Mistral**
  - Qwen3와 Llama의 중간적인 특성
  - 역할별 head 제거에 민감하지만, 무작위 head 제거에도 성능이 크게 손상됨
  - 이는 특정 head만이 아니라 모델 전체가 논리 추론에 전반적으로 민감하다는 뜻이다.

---

### 7. 전체 결론

이 논문은 LLM의 명제 논리 추론이 단순한 표면 패턴 매칭만은 아니며, 성공적인 추론에서는 다음과 같은 구조가 반복된다고 주장한다.

> **초기 층에서 사실을 처리하고, 중간 층에서 논리적 binding과 정보 결합을 수행하며, 후기 층에서 질문에 대한 답을 수렴한다.**

이 과정에서 경계 토큰이 정보 허브 역할을 하고, 사실 토큰은 여러 단계에서 다시 참조되며, 특화된 attention head가 정보 분할·전달·조회 기능을 담당한다. 다만 분석 대상이 dual-correct 사례에 한정되므로, 모델이 실패하는 방식까지 설명하는 결과는 아니다.

---





### 1. Compared Models

The study evaluates four models from three model families:

- **Qwen3-8B**
- **Qwen3-14B**
- **Llama-3.1-8B-Instruct**
- **Mistral-7B-Instruct**

The goal is not only to compare accuracy, but also to examine whether different models use a common internal strategy for propositional reasoning.

---

### 2. Test Data

The experiments use **PropLogic-MI**, a controlled propositional-logic benchmark.

- **11 logical rule categories**
  - Including Identity, Contradiction, Commutativity, De Morgan, Distributivity, and Absorption
- **Two reasoning depths**
  - **One-hop**: the answer follows directly from the facts
  - **Two-hop**: an intermediate variable must be derived before answering
- **Two prompt orders**
  - **πFF**: facts appear before the expression and query
  - **πEF**: the expression/query appears before the facts
- Each prompt order contains **20,000 clean/corrupted pairs**
  - Clean prompts contain the original facts
  - Corrupted prompts modify a fact so that the correct answer changes

Mechanistic analyses are restricted to the **dual-correct pool**, namely examples that the model answers correctly in both the clean and corrupted versions.

---

### 3. Behavioral Results

| Model | πFF Clean | πFF Dual | πEF Clean | πEF Dual |
|---|---:|---:|---:|---:|
| Qwen3-8B | 87.0% | 75.4% | 74.2% | 53.9% |
| Qwen3-14B | 87.4% | 76.6% | 74.4% | 52.8% |
| Llama-3.1-8B | 63.9% | 38.4% | 60.3% | 37.0% |
| Mistral-7B | 65.1% | 35.1% | 66.7% | 40.0% |

#### Main comparisons

1. **Qwen3 performs best overall.**
   Both Qwen3 models substantially outperform Llama-3.1 and Mistral.

2. **Prompt order matters.**
   Performance generally drops under πEF, where the facts appear after the query. This shows that the models are sensitive to input layout even when the underlying logic is identical.

3. **The gap is larger for dual-correct performance.**
   Qwen3 retains roughly 75% of examples in the dual-correct pool, whereas Llama and Mistral retain only about 35–40%. This indicates that Qwen3 is more consistent when factual assignments are changed.

4. **Rule difficulty varies.**
   Commutativity is relatively easy, while Contradiction is consistently difficult. Associative, Domination, and Excluded Middle show stronger sensitivity to prompt order.

---

### 4. Evaluation Metrics

- **Clean accuracy**: accuracy on the original prompts.
- **Dual-correct rate**: the percentage of examples answered correctly in both clean and corrupted conditions.
- **Probability Difference (PD)**: the probability gap between the True and False answer tokens.
- **Probability Difference Shift (dPD)**: the change in PD after activation patching or ablation.

A positive dPD means that an intervention moves the model toward the correct answer, while a negative dPD means that it reduces support for the correct answer.

---

### 5. Main Mechanistic Findings

#### 5.1 Staged Computation

The models divide reasoning across depth:

- **Early layers**: process factual assignments
- **Middle layers**: process logical expressions and variable binding
- **Late layers**: integrate the query and produce the final answer

The exact layer ranges differ across models, but the relative order—**facts → expression → query**—is consistently observed.

#### 5.2 Information Transmission through Boundary Tokens

Information concentrates at semantic boundary tokens rather than being distributed uniformly.

- **fact_value** for factual blocks
- **expr_last** for expression blocks
- **query_last** for query blocks

These tokens function as local information hubs that pass information from one logical stage to the next.

#### 5.3 Fact Retrospection

Fact tokens remain causally active after their initial encoding.

- In middle layers, expression-side variable references retrieve the relevant truth values.
- In difficult two-hop problems, fact tokens remain influential even in late layers.

The paper calls this continued access to fact representations **Fact Retrospection**. It becomes stronger as the task becomes more difficult.

#### 5.4 Specialized Attention Heads

The authors identify three functional head types:

- **Splitting Heads**
  - Mark logical boundaries and separate blocks
  - Concentrate in early layers
- **Transmission Heads**
  - Aggregate information within a block and route it to its terminal token
  - Concentrate in middle layers
- **Fact-Retrieval Heads**
  - Retrieve True/False values from factual positions
  - Concentrate from middle to late layers

Ablating these role-specific heads causes larger performance drops than ablating randomly selected heads, supporting their functional interpretation.

---

### 6. Differences across Model Families

- **Qwen3**
  - Shows the clearest staged computation
  - Has a strong fact-centered information profile
  - Exhibits the cleanest separation between specialized head roles

- **Llama-3.1**
  - Shows a more query-centered and diffuse information flow
  - The three head roles are detectable, but their functional separation is weaker

- **Mistral**
  - Lies between Qwen3 and Llama in its token-level profile
  - Is globally sensitive: even random head ablations can substantially damage performance

---

### 7. Overall Conclusion

The paper argues that successful propositional reasoning in these LLMs is not merely surface pattern matching. Instead, it follows a recurring internal organization:

> **Facts are processed early, logical binding and information integration occur in middle layers, and the final query is resolved in late layers.**

Boundary tokens act as information hubs, fact tokens are repeatedly accessed during difficult reasoning, and specialized attention heads implement splitting, transmission, and fact retrieval. However, because the mechanistic analysis uses only dual-correct examples, these findings describe successful reasoning rather than the mechanisms underlying model failures.


<br/>
# 예제



### 1. 이 논문에서 수행한 태스크

이 논문은 LLM이 **명제논리(propositional logic)** 문제를 풀 때 내부적으로 어떤 계산 과정을 사용하는지 분석한다.  
모델은 주어진 사실과 논리식을 바탕으로 마지막 질의의 값이 **True인지 False인지** 예측한다.

중요한 점은 이 논문이 모델을 새로 학습시킨 것이 아니라, 이미 학습된 모델에 프롬프트를 입력하는 **zero-shot 추론 및 기계적 해석(mechanistic interpretability) 실험**이라는 것이다. 따라서 전통적인 의미의 “훈련 데이터와 테스트 데이터”가 따로 제시되지는 않는다.

---

### 2. 기본 입력과 출력 형식

모델 입력은 다음 세 부분으로 구성된다.

1. **Facts**: 변수의 참·거짓 값  
2. **Logical expression**: 변수와 논리 연산자로 구성된 식  
3. **Query**: 식의 결과를 출력해야 하는 위치

출력은 항상 다음 두 값 중 하나이다.

```text
True
```

또는

```text
False
```

데이터셋의 프롬프트는 보통 다음과 같은 형태로 끝난다.

```text
Answer with one word only: True or False.
```

---

### 3. One-hop 문제 예시

#### 입력

```text
A is True, B is False, (¬A or ¬B) is
```

#### 추론

- A = True 이므로 ¬A = False
- B = False 이므로 ¬B = True
- 따라서 `False or True = True`

#### 출력

```text
True
```

이 문제는 주어진 사실에서 바로 정답을 계산할 수 있으므로 **one-hop reasoning** 문제이다.

---

### 4. Clean–corrupted 쌍의 예시

논문의 인과 분석에서는 같은 논리 문제에 대해 정답이 달라지는 두 입력을 만든다.

#### Clean 입력

```text
A is True, B is False, (¬A or ¬B) is
```

정답:

```text
True
```

#### Corrupted 입력

```text
A is True, B is True, (¬A or ¬B) is
```

추론:

- A = True → ¬A = False
- B = True → ¬B = False
- `False or False = False`

정답:

```text
False
```

즉, 두 입력은 거의 동일하지만 사실 하나만 바뀌며, 그 결과 정답도 바뀐다. 연구진은 clean 입력의 내부 활성값을 corrupted 입력에 복사하는 **activation patching**을 사용해 어떤 토큰과 어떤 층이 정답에 영향을 주는지 측정했다.

---

### 5. Two-hop 문제 예시

two-hop 문제는 중간 결과를 먼저 계산한 뒤 최종 답을 계산해야 한다.

#### 입력

```text
A is True, B is A and True, C is False, B and C is
```

#### 추론

1. 먼저 중간 변수 B를 계산한다.

```text
B = A and True
  = True and True
  = True
```

2. 최종 질의를 계산한다.

```text
B and C
= True and False
= False
```

#### 출력

```text
False
```

이처럼 `A → B → 최종 질의`의 두 단계가 필요하므로 two-hop 문제이다.

논문에서는 다음과 같은 형태도 사용한다.

```text
A is True, B is False, C is ¬(A and B), D is True, C or D is
```

계산 과정은 다음과 같다.

```text
C = ¬(True and False) = True
C or D = True or True = True
```

출력:

```text
True
```

---

### 6. 다루는 구체적인 논리 규칙

PropLogic-MI 데이터셋은 다음 11개 규칙을 포함한다.

- Identity
- Domination
- Idempotent
- Double Negation
- Excluded Middle
- Contradiction
- Commutativity
- Associativity
- Distributivity
- De Morgan’s Law
- Absorption

예를 들어 De Morgan 규칙은 다음과 같다.

```text
¬(P and Q) ≡ ¬P or ¬Q
```

구체적인 입력:

```text
A is True, B is False, (¬A or ¬B) is
```

출력:

```text
True
```

다만 모델이 반드시 규칙 이름을 입력받는 것은 아니다. 대부분의 경우 모델은 사실과 논리식만 보고 결과를 계산해야 한다.

---

### 7. 프롬프트 순서 실험

논문은 같은 논리 내용을 두 가지 순서로 제시한다.

#### πFF: 사실이 먼저 나오는 경우

```text
A is True, B is False, (¬A or ¬B) is
```

#### πEF: 논리식이 먼저 나오고 사실이 뒤에 나오는 경우

개념적으로는 다음과 같은 순서이다.

```text
(¬A or ¬B) is, A is True, B is False
```

두 경우 논리적 내용과 정답은 같지만 입력 순서가 다르다.  
연구 결과, 입력 순서가 바뀌어도 다음과 같은 계산 순서가 대체로 유지되었다.

```text
Facts → Expression → Query → Answer
```

따라서 연구진은 이 구조가 단순히 “입력에서 먼저 등장한 정보가 먼저 처리되는 현상”이 아니라, 모델 내부에 어느 정도 조직된 계산 전략일 가능성이 있다고 해석한다.

---

### 8. 데이터셋과 분석용 데이터

- 11개 명제논리 규칙
- one-hop 및 two-hop 문제
- 두 가지 프롬프트 순서
- 각 순서마다 약 20,000개의 clean/corrupted 쌍
- 전체적으로 one-hop 9,523개, two-hop 10,477개
- 변수 이름은 매번 무작위로 바뀜
- 예: A, B, C 대신 다른 대문자 변수가 사용됨

이 데이터는 모델을 학습시키기 위한 training set이라기보다, 이미 학습된 모델의 추론 과정을 관찰하기 위한 **통제된 평가·분석 데이터셋**이다.

또한 기계적 분석은 각 모델이 clean과 corrupted 입력을 모두 맞힌 문제만 골라 수행했다. 이를 **dual-correct subset**이라고 한다. 따라서 논문의 결론은 모델이 성공적으로 푼 문제에서의 내부 계산에 관한 것이며, 모델이 실패하는 과정까지 직접 설명하는 것은 아니다.

---




### 1. Task studied in the paper

The paper studies how large language models solve **propositional logic problems** internally.

Given factual assignments and a logical expression, the model must predict whether the queried expression is **True** or **False**.

The models are not newly trained in this work. Instead, the authors use pretrained models in a **zero-shot inference setting** and analyze their internal activations. Therefore, the paper does not define a conventional training/test split in the usual machine-learning sense.

---

### 2. Basic input and output format

Each prompt contains:

1. **Facts**, such as `A is True`
2. **A logical expression**, such as `(¬A or ¬B)`
3. **A query position** where the answer must be produced

The output is one of:

```text
True
```

or

```text
False
```

The prompt usually ends with an instruction such as:

```text
Answer with one word only: True or False.
```

---

### 3. One-hop example

#### Input

```text
A is True, B is False, (¬A or ¬B) is
```

#### Reasoning

- A = True, so ¬A = False
- B = False, so ¬B = True
- Therefore, `False or True = True`

#### Output

```text
True
```

This is a **one-hop** problem because the answer can be computed directly from the given facts.

---

### 4. Clean and corrupted examples

For causal analysis, the authors create minimally different input pairs.

#### Clean input

```text
A is True, B is False, (¬A or ¬B) is
```

Output:

```text
True
```

#### Corrupted input

```text
A is True, B is True, (¬A or ¬B) is
```

Reasoning:

```text
¬A = False
¬B = False
False or False = False
```

Output:

```text
False
```

Only one fact is changed, which flips the answer. The authors use **activation patching** to copy internal activations from the clean run into the corrupted run and identify which layers and token positions carry the relevant information.

---

### 5. Two-hop example

A two-hop problem requires computing an intermediate variable first.

#### Input

```text
A is True, B is A and True, C is False, B and C is
```

#### Reasoning

First compute the intermediate variable:

```text
B = A and True
  = True and True
  = True
```

Then compute the final query:

```text
B and C
= True and False
= False
```

#### Output

```text
False
```

This is a two-hop problem because the model must follow the chain:

```text
A → B → final query
```

Another example is:

```text
A is True, B is False, C is ¬(A and B), D is True, C or D is
```

The reasoning is:

```text
C = ¬(True and False) = True
C or D = True or True = True
```

Therefore, the output is:

```text
True
```

---

### 6. Logical rules covered

PropLogic-MI contains 11 rule categories:

- Identity
- Domination
- Idempotent
- Double Negation
- Excluded Middle
- Contradiction
- Commutativity
- Associativity
- Distributivity
- De Morgan’s Law
- Absorption

For example, De Morgan’s Law is:

```text
¬(P and Q) ≡ ¬P or ¬Q
```

A concrete prompt is:

```text
A is True, B is False, (¬A or ¬B) is
```

The correct output is:

```text
True
```

The rule name is generally not explicitly provided to the model; it must infer the result from the facts and expression.

---

### 7. Prompt-order experiment

The same logical content is presented in two orders.

#### πFF: facts first

```text
A is True, B is False, (¬A or ¬B) is
```

#### πEF: expression first, facts later

Conceptually:

```text
(¬A or ¬B) is, A is True, B is False
```

The logical content and answer remain unchanged, but the surface order differs. The paper finds that the same broad computational organization generally remains:

```text
Facts → Expression → Query → Answer
```

This suggests that the organization is not merely caused by the order in which information appears in the prompt.

---

### 8. Dataset and analysis data

The dataset contains:

- 11 propositional-logic rule categories
- One-hop and two-hop problems
- Two prompt orderings
- About 20,000 clean/corrupted pairs for each ordering
- 9,523 one-hop and 10,477 two-hop instances
- Randomized variable names

The dataset is mainly a controlled benchmark for evaluating and interpreting pretrained models, not a training set used to teach the models logic.

For mechanistic analysis, the authors keep only examples that the model answers correctly on both the clean and corrupted versions. This is called the **dual-correct subset**. Consequently, the paper primarily explains the internal computation underlying successful reasoning, not the mechanisms behind failed answers.

<br/>
# 요약


1. 연구진은 PropLogic-MI의 11개 명제논리 규칙과 1·2-hop 문제를 네 LLM에 적용하고, activation patching과 mean ablation으로 층·토큰·attention head의 인과적 역할을 분석했다.  
2. 그 결과 모델은 초기층에서 사실을 처리하고 중간층에서 논리식과 변수-사실을 결합하며 후기층에서 질의 답을 수렴시키는 단계적 계산을 보였고, 정보는 각 블록의 경계 토큰에 모이며 어려운 문제에서는 사실 토큰을 후기층까지 재참조했다.  
3. 예를 들어 “A는 True, B는 False, (¬A 또는 ¬B)는?”에서 모델은 사실값을 논리식으로 전달해 True를 계산하며, 이를 Splitting·Transmission·Fact-Retrieval attention head가 구현했고 입력 순서를 바꿔도 같은 구조가 유지됐다.  



1. The authors applied PropLogic-MI, covering 11 propositional rules and one- and two-hop tasks, to four LLMs and used activation patching and mean ablation to causally analyze layers, tokens, and attention heads.  
2. The models showed staged computation: early layers processed facts, middle layers bound facts to logical expressions, and late layers converged on the query answer; information accumulated at boundary tokens, while difficult tasks caused fact tokens to remain active in later layers.  
3. For example, given “A is True, B is False, (¬A or ¬B) is?”, the model routes the fact values into the expression to derive True, with Splitting, Transmission, and Fact-Retrieval heads implementing the process, which remained similar when prompt order was reversed.

<br/>
# 기타



### 1. 전체 개념도와 분석 틀

**Figure 1**은 논문의 핵심 주장을 네 가지 메커니즘으로 요약한다.

1. **Staged Computation**
   - 초기 층: 사실(facts) 처리
   - 중간 층: 논리식과 변수의 결합(binding)
   - 후기 층: 질의(query)와 최종 답 계산

2. **Information Transmission**
   - 정보가 모든 토큰에 균등하게 퍼지는 것이 아니라, 각 의미 단위를 끝내는 **경계 토큰(boundary token)**으로 모인다.
   - 예: 사실문의 truth-value 토큰, 논리식의 마지막 토큰, 질의문의 마지막 토큰

3. **Fact Retrospection**
   - 모델은 사실 토큰을 초기에 한 번만 읽고 버리지 않는다.
   - 중간 층에서는 사실값을 변수 참조 위치에 전달하고, 어려운 문제에서는 후기 층까지 사실 토큰을 다시 참조한다.

4. **Specialized Attention Heads**
   - **Splitting heads**: 논리 블록의 경계를 구분
   - **Transmission heads**: 블록 내부 정보를 마지막 토큰으로 모음
   - **Fact-Retrieval heads**: 질의나 표현식에서 사실값을 조회

**Figure 2**는 분석 단위를 보여준다.  
논리 블록을 크게 Facts–Expression–Query로 나누고, 그 안에서 `fact_value`, `expr_last`, `query_last` 같은 구조적 토큰을 세밀하게 분석한다. 즉, “어떤 층에서 처리되는가”와 “어떤 토큰을 통해 정보가 이동하는가”를 함께 측정한다.

---

### 2. 데이터 및 행동 성능 결과

#### Table 1, Table 5: 전체 성능과 분석 대상

- PropLogic-MI는 **11개 명제논리 규칙**과 **1-hop/2-hop 추론**을 포함한다.
- 각 문제는 사실이 먼저 나오는 `πFF`와 논리식·질의가 먼저 나오는 `πEF` 두 가지 순서로 제공된다.
- `πFF`에서 전반적으로 성능이 더 높았고, `πEF`에서는 모든 모델의 성능과 dual-correct 비율이 낮아졌다.
- 예를 들어 Qwen3-8B의 dual-correct rate는:
  - `πFF`: 75.4%
  - `πEF`: 53.9%

**핵심 인사이트**
- 입력 순서를 바꾸면 성능은 떨어지지만, 뒤의 기계적 분석에서는 동일한 단계적 구조가 나타난다.
- 따라서 논리 처리 방식 자체는 단순히 “사실이 먼저 제시되었기 때문”에 생긴 것이 아니라, 모델 내부 계산 구조에 더 가깝다.
- 단, 분석은 clean과 corrupted 입력을 모두 맞힌 **dual-correct subset**에 한정된다. 따라서 실패 사례의 내부 메커니즘까지 설명하는 결과는 아니다.

#### Table 6: 규칙별 성능

- **Commutativity**는 대체로 매우 쉬운 규칙이었다.
- **Contradiction**은 모델과 프롬프트 순서에 관계없이 어려운 편이었다.
- **Associative, Domination, Excluded Middle**은 프롬프트 순서에 따른 성능 변화가 컸다.
- 즉, 명제논리 전체를 하나의 난이도로 볼 수 없으며, 규칙의 구조와 입력 배치가 함께 영향을 준다.

#### Table 7: 모델별 비교

- **Qwen3**: 사실 중심의 정보 흐름이 가장 뚜렷하고, 역할별 attention head의 선택성이 가장 명확하다.
- **Llama-3.1**: 질의 토큰 중심이며 정보 흐름이 더 분산되어 있다. head ablation에서 역할 간 구분이 약하다.
- **Mistral**: Qwen3와 Llama의 중간 성격을 보이지만, 역할별 head뿐 아니라 무작위 head 제거에도 성능이 크게 저하된다.
- 세 모델 모두 기본적인 메커니즘은 공유하지만, 그 구현의 선명도와 안정성은 다르다.

---

### 3. 층별 계산 구조

#### Figure 3 및 Figures 7–29: Block ablation

Facts, Expression, Query 블록의 activation을 평균값으로 대체해 어느 층이 각 블록에 의존하는지 측정했다.

결과는 일관되게 다음 순서를 보인다.

> **Facts → Expression → Query**

- Facts의 영향은 주로 초기 층에 집중
- Expression의 영향은 중간 층에 집중
- Query의 영향은 후기 층에 집중
- 2-hop 문제에서는 후기 계산이 더 넓게 퍼지지만, 위 순서는 유지된다.

이 패턴은 다음 조건에서도 유지된다.

- MLP ablation
- Attention ablation
- Attention+MLP 공동 ablation
- Qwen3-8B/14B
- Llama-3.1
- Mistral
- `πFF`와 `πEF`
- 1-hop과 2-hop

**중요한 해석**
- 특정 layer cutoff가 모든 모델에서 동일하다는 뜻은 아니다.
- 모델마다 단계의 위치와 폭은 다르지만, **상대적인 계산 순서**가 공통적으로 유지된다는 의미다.
- Llama와 Mistral은 Qwen3보다 Fact와 Expression 처리가 더 이른 층에서 나타나고, Query 관련 late-layer 통합은 더 넓게 분포한다.

---

### 4. 토큰 수준 정보 흐름

#### Figure 4 및 Figures 30–37

토큰 단위 activation patching 결과는 정보가 블록 전체에 균등하게 분포하지 않고, 특정 경계 토큰에 집중된다는 점을 보여준다.

- 초기 층: `fact_value`
- 중간 층: `expr_last`
- 후기 층: `query_last`

즉, 정보 이동은 대략 다음 경로를 따른다.

> **Fact boundary → Expression boundary → Query boundary**

`πEF`에서는 autoregressive 구조상 expression token의 일부를 패치해도 정보가 전달되지 않는 경우가 있다. 그러나 `fact_value`와 `query_last`의 지배적 역할은 유지된다. 따라서 boundary-token 현상은 단순한 입력 순서 효과가 아니다.

#### Fact Retrospection

Figure 4c와 Figures 32–33, 36–37은 중간 층에서 다음과 같은 교환을 보여준다.

- 초기에는 `fact_value`의 인과적 영향이 큼
- 중간 층으로 갈수록 `fact_var_ref`의 영향이 증가
- 이는 변수 참조 위치가 사실값을 조회하고 전달한다는 의미
- 어려운 2-hop 문제에서는 후기 층까지 `fact_value`의 영향이 남음

논문은 이 현상을 **Fact Retrospection**이라고 부른다.

또한 Figure 5는 문제 난이도가 높을수록 후기 층의 사실값 복원이 강해진다는 양의 상관을 보고한다(`r̄ = 0.449`).

**해석**
- 쉬운 문제에서는 사실값이 중간에 압축되어 이후 계산으로 전달될 수 있다.
- 어려운 문제에서는 모델이 사실 토큰을 계속 참조하면서 계산을 유지한다.
- 이는 “모델이 사실을 한 번 인코딩한 뒤 순방향으로만 계산한다”는 단순한 그림과 다르다.

---

### 5. Attention head 분석

#### Figure 6 및 Figure 38

세 가지 head 역할이 층별로 분리되어 나타난다.

1. **Splitting heads**
   - 주로 초기 층
   - 쉼표나 구간 경계를 주목
   - Facts, Expression, Query 같은 논리 블록을 분리

2. **Transmission heads**
   - 주로 중간 층
   - 블록 내부 토큰에서 블록의 마지막 토큰으로 정보를 전달
   - Figure 38에서는 논리 블록 내부의 삼각형 형태 attention 패턴으로 나타난다.

3. **Fact-Retrieval heads**
   - 중간부터 후기 층
   - 질의나 표현식 위치에서 `True/False` 사실값 토큰으로 attention
   - 변수에 대응하는 사실값을 조회하는 역할

#### Table 2: Head ablation

역할별 상위 head를 제거하면 성능이 크게 하락한다.

- Fact-Retrieval head 제거가 대체로 가장 큰 손상을 유발
- Transmission head가 그다음
- Splitting head는 상대적으로 영향이 작지만 여전히 무작위 대조군보다 중요
- 역할과 무관한 무작위 head 제거는 Qwen3에서 성능 손상이 거의 없다.

다만 모델별 차이가 있다.

- **Qwen3**: 역할별 ablation과 무작위 ablation의 차이가 뚜렷함
- **Llama**: 적은 수의 head만 제거해도 성능이 공통 하한에 도달해 역할 분리가 덜 선명함
- **Mistral**: 역할 head뿐 아니라 무작위 head 제거에도 민감하여 네트워크 전체가 전반적으로 취약함

따라서 attention head taxonomy가 단순한 시각적 분류가 아니라 실제 계산 기능과 관련됨을 보여준다.

---

### 6. 부록의 추가 결과

#### Appendix A: 데이터 생성 및 템플릿

- 11개 논리 규칙의 형식적 정의와 예시를 제공한다.
- 변수 이름을 무작위로 바꾸어 표면 패턴에 의존하는 것을 줄였다.
- corrupted prompt는 사실값을 바꾸어 정답이 뒤집히도록 구성했다.
- 2-hop 문제에서는 중간 변수와 gate 변수를 도입해 두 단계 계산을 요구한다.

**인사이트**
- 데이터가 통제되어 있어 모델이 단순한 자연어 상식이나 고정된 변수 이름을 이용하기 어렵다.
- 따라서 관찰된 회로는 명제논리 계산과 더 직접적으로 연결된다.

#### Appendix B: 방법론 및 재현성

- 주요 도구는 activation patching과 mean ablation이다.
- 분석은 residual stream, MLP, attention output, 개별 attention head에 대해 수행된다.
- 각 모델에 대해 동일한 probability-difference 지표를 사용한다.
- Early/Middle/Late 구간은 모델별 층 수와 경험적 계산 위치에 맞게 설정했다.

**주의점**
- 평균값 ablation은 특정 정보의 필요성을 측정하는 데 유용하지만, 그 정보가 정확히 어떤 Boolean 연산으로 변환되는지까지 보여주지는 않는다.
- 특히 MLP 내부 연산은 여전히 충분히 해석되지 않았다.

#### Appendix C: 행동 성능

- 모델별·hop별·규칙별 성능을 세분화한다.
- 2-hop 성능 저하는 특히 Llama와 Mistral에서 더 뚜렷하다.
- 이는 이후 관찰되는 후기 층의 Fact Retrospection과 연결된다.

#### Appendix D: Staged Computation 추가 검증

- 모든 모델, 프롬프트 순서, hop에서 Facts–Expression–Query 순서가 반복된다.
- Attention ablation은 MLP보다 중간·후기 층의 정보 전달 효과를 더 강하게 보여준다.
- Joint ablation은 초기 Fact 단계의 우세를 더 선명하게 만든다.

#### Appendix E: Information Routing 추가 검증

- boundary-token convergence가 네 모델 모두에서 재현된다.
- `πEF`에서는 `expr_last`가 autoregressive 제약으로 약해지지만, 초기 Fact와 후기 Query 중심 구조는 유지된다.
- Qwen3에서 중간 층의 `fact_var_ref` 신호가 가장 뚜렷하고, Llama와 Mistral에서는 약하다.

#### Appendix F: Specialized Heads 추가 검증

- 모든 모델과 프롬프트 순서에서 Splitting, Transmission, Fact-Retrieval head가 식별된다.
- 역할의 존재 자체는 공통적이지만, ablation 곡선의 선명도는 모델마다 다르다.
- 즉, “같은 전략이 항상 같은 head에 의해 구현된다”기보다는, 모델별로 다른 head 집합이 유사한 기능적 역할을 수행한다고 보는 것이 적절하다.

---

### 종합 결론

이 논문의 부록·그림·표가 뒷받침하는 핵심 결론은 다음과 같다.

> LLM의 명제논리 추론은 단순히 모든 층과 토큰이 동시에 처리하는 과정이 아니라,  
> **Facts → Expression → Query**의 단계적 계산과  
> **boundary-token 중심의 정보 집약**,  
> **사실값의 반복 조회**,  
> **기능별 attention head 분업**으로 구성된다.

이 구조는 모델, 논리 규칙, 추론 hop, 입력 순서가 바뀌어도 대체로 유지된다. 다만 모델이 추론에 실패하는 경우의 내부 과정, MLP가 실제로 수행하는 Boolean 변환, 더 긴 추론이나 자연어 논리에서의 일반화는 아직 남은 과제다.

---




### 1. Overall framework

**Figure 1** summarizes the paper’s four main mechanisms:

1. **Staged Computation**
   - Early layers process facts.
   - Middle layers bind variables and process logical expressions.
   - Late layers integrate the query and produce the answer.

2. **Information Transmission**
   - Information is not distributed uniformly across all tokens.
   - It converges on **semantic boundary tokens**, such as fact-value tokens, expression-final tokens, and query-final tokens.

3. **Fact Retrospection**
   - Fact tokens remain causally active after their initial encoding.
   - In middle layers, fact values are retrieved and transferred to variable-reference positions.
   - On harder tasks, especially two-hop tasks, fact tokens remain influential even in late layers.

4. **Specialized Attention Heads**
   - **Splitting heads** identify logical boundaries.
   - **Transmission heads** aggregate information within a logical block.
   - **Fact-Retrieval heads** retrieve truth values from fact positions.

**Figure 2** explains the analysis granularity: the prompt is divided into Facts, Expression, and Query blocks, which are then analyzed at the level of structural token categories such as `fact_value`, `expr_last`, and `query_last`.

---

### 2. Dataset and behavioral results

#### Tables 1 and 5

PropLogic-MI contains:

- 11 propositional-rule categories
- One-hop and two-hop reasoning problems
- Two prompt orders:
  - `πFF`: facts appear before the expression and query
  - `πEF`: the expression/query appears before the facts

Performance is generally higher under `πFF`. For example, Qwen3-8B has a dual-correct rate of:

- 75.4% under `πFF`
- 53.9% under `πEF`

The important point is that reversing the prompt order reduces accuracy, but the same internal staged organization still appears. This suggests that the mechanism is not merely caused by the surface ordering of the input.

However, all mechanistic analyses use the **dual-correct subset**, consisting of examples that the model answers correctly in both clean and corrupted versions. Therefore, the results mainly describe successful reasoning, not failure mechanisms.

#### Table 6: Rule-wise performance

- **Commutativity** is generally easy.
- **Contradiction** is consistently difficult.
- **Associativity, Domination, and Excluded Middle** are strongly affected by prompt order.

Thus, different logical rules impose different computational difficulties.

#### Table 7: Cross-model comparison

- **Qwen3** shows the clearest fact-centered information flow and the cleanest separation of attention-head roles.
- **Llama-3.1** is more query-centered and has more diffuse information routing.
- **Mistral** lies between Qwen3 and Llama in its token-level profile, but is globally sensitive to head ablation.

All models exhibit the same broad mechanisms, but the functional specialization is implemented more cleanly in Qwen3.

---

### 3. Layer-wise computation

#### Figure 3 and Figures 7–29

Block-level ablation consistently reveals the following order:

> **Facts → Expression → Query**

- Fact information is concentrated in early layers.
- Expression processing dominates middle layers.
- Query processing dominates late layers.
- Two-hop reasoning broadens the later computation but preserves the same order.

This pattern is reproduced across:

- MLP ablation
- Attention ablation
- Joint attention+MLP ablation
- Different model families
- Both prompt orders
- One-hop and two-hop tasks

The exact layer boundaries differ between models, but the **relative ordering of the stages** is stable.

---

### 4. Token-level information flow

#### Figure 4 and Figures 30–37

Causal influence is concentrated on a few boundary tokens:

- Early layers: `fact_value`
- Middle layers: `expr_last`
- Late layers: `query_last`

This produces an approximate information path:

> **Fact boundary → Expression boundary → Query boundary**

The same pattern survives prompt-order reversal. Under `πEF`, some expression-token effects are structurally suppressed by autoregressive constraints, but fact and query boundary tokens remain important.

#### Fact Retrospection

The zoomed-in results show a dynamic exchange:

- `fact_value` is strongest in early layers.
- `fact_var_ref` becomes stronger in middle layers.
- This indicates that variable-reference positions retrieve the corresponding fact values.
- On harder two-hop tasks, `fact_value` remains causally influential in late layers.

Figure 5 further shows that late-layer fact restoration increases with task difficulty, with a mean correlation of approximately `r = 0.449`.

The interpretation is that easy problems allow the model to compress fact information into a forward computation, whereas harder problems lead the model to repeatedly access the original fact tokens.

---

### 5. Attention-head analysis

#### Figures 6 and 38

The three head types occupy different depth ranges:

1. **Splitting heads**
   - Mostly early-layer heads
   - Attend to commas and other boundary markers
   - Separate facts, expressions, and queries

2. **Transmission heads**
   - Mostly middle-layer heads
   - Move information within a logical block toward its terminal token
   - Show block-like or triangular attention patterns

3. **Fact-Retrieval heads**
   - Concentrated from middle to late layers
   - Attend from expression/query positions back to `True` or `False` fact-value tokens

#### Table 2: Head ablation

Removing the identified heads damages performance substantially.

- Fact-Retrieval heads usually cause the largest degradation.
- Transmission heads cause the next largest degradation.
- Splitting heads have a smaller but still meaningful effect.
- Random heads outside the identified pool have little effect in Qwen3.

The model differences are important:

- **Qwen3**: clear separation between functional heads and random controls
- **Llama-3.1**: role-specific heads are less cleanly separated
- **Mistral**: both role-targeted and random ablations are damaging, suggesting global network sensitivity

This supports the claim that the head taxonomy reflects real computational functions rather than merely visual attention patterns.

---

### 6. Appendix results

#### Appendix A: Dataset construction

The dataset uses formal rule templates, random variable remapping, and label-flipping corruption. Two-hop problems introduce intermediate and gate variables.

This design reduces reliance on fixed variable names or superficial linguistic patterns and makes the observed mechanisms more directly related to logical computation.

#### Appendix B: Methodology

The study combines:

- Activation patching
- Mean ablation
- Residual-stream interventions
- MLP and attention ablation
- Individual attention-head ablation

The same probability-difference metric is used throughout. The main limitation is that these methods identify when MLPs are important, but not exactly which Boolean transformations the MLPs implement.

#### Appendix C: Behavioral results

The detailed breakdown confirms that two-hop reasoning is more difficult, especially for Llama and Mistral. This aligns with the stronger late-layer Fact Retrospection observed on harder tasks.

#### Appendix D: Additional staged-computation results

Across all models, prompt orders, and reasoning depths, the Facts–Expression–Query order remains stable.

Attention ablation places relatively more causal mass in middle and late layers, while joint ablation makes the early Fact stage more pronounced.

#### Appendix E: Additional information-routing results

Boundary-token convergence generalizes across all four models. Under `πEF`, `expr_last` is weakened by autoregressive constraints, but early fact concentration and late query concentration remain.

Qwen3 shows the clearest middle-layer exchange between `fact_value` and `fact_var_ref`.

#### Appendix F: Additional head results

All models and prompt orders contain identifiable Splitting, Transmission, and Fact-Retrieval heads. However, the sharpness of their causal validation differs across models.

This suggests that the models share a common computational strategy, while the exact heads implementing that strategy may vary.

---

### Overall conclusion

The figures, tables, and appendices support the following conclusion:

> Propositional reasoning in LLMs is not a uniformly distributed computation.  
> It is organized as a staged process—**Facts → Expression → Query**—with information concentrated at semantic boundary tokens, repeated retrieval of fact values, and specialized attention-head roles.

This organization is robust across models, rules, reasoning depths, and prompt orders. Nevertheless, the paper does not yet fully explain failure cases, the precise Boolean transformations performed by MLPs, or whether the same mechanisms generalize to longer chains and natural-language reasoning.

<br/>
# refer format:
### BibTeX

```bibtex
@inproceedings{chen-yan-wang-pan-2026-mechanistic,
  author    = {Chen, Danchun and Yan, Qiyao and Wang, Chenpeng and Pan, Liangming},
  title     = {Towards a Mechanistic Understanding of Propositional Logical Reasoning in Large Language Models},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  month     = oct,
  pages     = {28708--28737},
  publisher = {Association for Computational Linguistics},
  address   = {October 24--29, 2026}
}
```

### Chicago 스타일    

Chen, Danchun, Qiyao Yan, Chenpeng Wang, and Liangming Pan. “Towards a Mechanistic Understanding of Propositional Logical Reasoning in Large Language Models.” In *Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing*, 28708–28737. Association for Computational Linguistics, 2026.



