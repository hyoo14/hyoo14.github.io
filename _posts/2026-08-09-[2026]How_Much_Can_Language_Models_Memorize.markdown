---
layout: post
title:  "[2026]How Much Can Language Models Memorize?"
date:   2026-08-09 17:41:00 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 큰 모델은 더 많은 데이터를 암기할 수 있지만, 데이터셋이 모델 용량에 비해 지나치게 크면 평균적인 학습 샘플에 대한 멤버십 추론은 어려워진다  
( **멤버십 추론 공격**—특정 데이터가 학습에 사용되었는지, 로스값을 통해 알아내는 공격, 로스값 낮으면 사용된거고 높으면 안 사용된걸로 추론   )   


짧은 요약(Abstract) :


이 논문은 언어 모델이 학습 데이터를 **얼마나 기억할 수 있는지**를 연구한다. 기존 연구는 모델이 학습 데이터를 외워서 출력하는 것과, 언어의 일반적인 패턴을 학습해 자연스럽게 생성하는 것을 명확히 구분하기 어려웠다.

논문에서는 기억을 두 가지로 나눈다.

- **의도하지 않은 암기(unintended memorization)**: 특정 학습 데이터에 포함된 고유한 정보를 모델이 추가로 기억하는 것  
- **일반화(generalization)**: 여러 데이터에 공통된 언어적 규칙이나 데이터 생성 과정의 일반적인 패턴을 학습하는 것

연구진은 먼저 무작위 비트 문자열처럼 **일반화가 불가능한 데이터**로 모델을 학습시켰다. 이 경우 모델이 저장한 정보는 거의 전부 암기로 볼 수 있기 때문에, 모델의 실제 기억 용량을 측정할 수 있다. 그 결과 GPT 계열 모델은 대략 **파라미터 하나당 3.6비트**의 정보를 저장하는 것으로 추정되었다.

또한 실제 텍스트에서는 데이터가 적을 때 모델이 개별 샘플을 많이 암기하지만, 데이터가 모델의 기억 용량을 넘어서면 암기가 더 이상 증가하지 않고 일반적인 언어 패턴을 학습하기 시작한다. 즉, 모델의 용량이 차면 개별 데이터에 대한 암기는 줄어들고 일반화가 늘어난다.

연구진은 50만 개에서 15억 개 파라미터 규모의 수백 개 Transformer 모델을 실험해, 모델 크기와 데이터셋 크기가 **멤버십 추론 공격**—특정 데이터가 학습에 사용되었는지 알아내는 공격—의 성능에 어떤 영향을 주는지도 분석했다. 결론적으로, 큰 모델은 더 많은 데이터를 암기할 수 있지만, 데이터셋이 모델 용량에 비해 지나치게 크면 평균적인 학습 샘플에 대한 멤버십 추론은 어려워진다.

---



This paper studies **how much information language models can memorize**. Previous work often struggled to distinguish between memorization of specific training examples and generalization of common language patterns.

The authors separate memorization into two components:

- **Unintended memorization**: information the model retains about particular training examples  
- **Generalization**: reusable knowledge about the underlying data-generating process

To measure pure memorization, they first train models on uniformly random bitstrings, where generalization is impossible. Under this setting, nearly all information stored by the model can be attributed to memorization. They estimate that GPT-style models can store approximately **3.6 bits per parameter**.

On real text, models initially memorize individual examples. Once the dataset becomes larger than the model’s memorization capacity, memorization reaches a plateau and the model increasingly relies on generalization instead. Thus, the model’s capacity determines when it shifts from storing sample-specific information to learning broader patterns.

Finally, the authors train hundreds of Transformer models ranging from 500K to 1.5B parameters and study membership inference. Larger models can memorize more data, while membership inference becomes more difficult as the dataset grows relative to model capacity. Their scaling laws suggest that many modern language models are trained on so much data that reliable membership inference for an average training example is likely to be difficult.


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



### 1. 연구 목표와 핵심 아이디어
이 논문은 언어 모델의 **암기(memorization)** 를 단순한 텍스트 생성 여부가 아니라, 모델이 특정 데이터를 얼마나 짧게 압축해서 표현할 수 있는지로 측정한다.

암기를 두 부분으로 나눈다.

- **의도하지 않은 암기(Unintended memorization)**: 특정 학습 데이터 샘플 자체에 대해 모델이 추가로 저장한 정보
- **일반화(Generalization, intended memorization)**: 여러 데이터에 공통적으로 존재하는 언어적·통계적 규칙을 학습한 정도

이를 구분하기 위해 다음과 같이 측정한다.

\[
\text{Unintended memorization}
\approx
\text{Reference model의 정보량}
-
\text{Target model의 정보량}
\]

실제로는 데이터의 압축 길이 또는 모델의 negative log-likelihood를 이용한다. 즉, **타깃 모델이 기준 모델보다 특정 샘플을 더 잘 압축하면, 그 차이를 해당 샘플에 대한 의도하지 않은 암기로 간주**한다.

---

### 2. 모델 구조
모델은 처음부터 학습한 **GPT-2 계열의 decoder-only Transformer 언어 모델**이다.

주요 설정은 다음과 같다.

- Transformer decoder 구조
- 1~8개 층
- hidden dimension: 32~512
- 약 10만~2,000만 개 파라미터 규모의 모델을 주로 사용
- GPT-style 모델의 암기 용량을 확인하기 위해 더 큰 모델도 추가 검증
  - 약 500K~1.5B 파라미터
  - GPT-2 Small 및 GPT-2 XL 포함
- 대부분의 실험은 **bfloat16 정밀도**
- 정밀도의 영향을 보기 위해 일부 모델은 **float32**로도 재학습

이 논문에서 특별한 새로운 Transformer 아키텍처를 제안한 것은 아니다. 기존 GPT-2 구조를 사용해 **모델 크기와 데이터 크기의 관계**를 통제하면서 측정하는 것이 핵심이다.

---

### 3. 학습 데이터

#### 3.1 합성 데이터: 균등 무작위 시퀀스
먼저 일반화가 전혀 일어나지 않는 환경을 만들기 위해 합성 데이터를 사용했다.

- 각 토큰을 정해진 vocabulary에서 독립적으로 균등 샘플링
- 이전 토큰과 무관한 완전한 무작위 시퀀스
- 기본 설정:
  - vocabulary size: \(V=2048\)
  - sequence length: \(S=64\)
  - 데이터셋 크기: 여러 단계로 변화
- 데이터의 실제 정보량을 정확히 계산할 수 있음

각 데이터셋의 엔트로피는 다음과 같이 계산된다.

\[
H(X)=N S \log_2 V
\]

무작위 데이터이므로 모델이 학습할 수 있는 일반적인 패턴이 없다. 따라서 모델이 학습 데이터에 대해 얻은 정보는 거의 전부 **의도하지 않은 암기**로 해석할 수 있다.

#### 3.2 실제 텍스트 데이터
일반화가 가능한 자연어 환경에서는 **FineWeb 데이터셋**을 사용했다.

- 64-token 시퀀스로 분할
- 기존 deduplication에 더해 추가 중복 제거 수행
- 중복 데이터가 암기 측정을 부풀리지 않도록 학습 데이터 샘플을 완전히 중복 제거
- 여러 모델 크기와 여러 데이터셋 크기 조합으로 학습

실제 텍스트에서는 모델이 언어 규칙을 일반화할 수 있으므로, 기준 모델과 비교하여 샘플 수준의 추가 정보만 의도하지 않은 암기로 측정했다.

---

### 4. 기준 모델과 타깃 모델
실제 텍스트 실험에서는 두 종류의 모델을 구분한다.

- **Target model**: 특정 크기의 FineWeb 데이터로 학습한 실험 대상 모델
- **Reference/oracle model**: 더 큰 데이터와 충분한 학습을 통해 실제 텍스트 분포를 더 잘 근사한다고 가정한 모델

주요 기준 모델은 약 **774M 파라미터의 FineWeb 학습 GPT-2 모델**이다. 기준 모델의 영향을 확인하기 위해 124M, 355M, 774M 모델도 비교했다.

기준 모델이 작을수록 텍스트를 덜 잘 예측하므로 타깃 모델과의 차이가 커져 절대적인 암기량이 더 크게 측정된다. 따라서 논문은 **같은 기준 모델을 사용할 때 모델 간 상대적 비교가 안정적**이라고 설명한다.

---

### 5. 학습 방법
모든 모델은 다음과 같은 방식으로 학습했다.

- 모델을 무작위 초기화한 뒤 처음부터 학습
- Adam optimizer 사용
- batch size: 2048
- 최대 \(10^6\) training steps
- GPU: 주로 단일 NVIDIA A100
- 데이터셋 크기와 모델 크기를 독립적으로 변화
- 각 설정을 여러 random seed로 반복

\(10^6\) step은 모델이 충분히 포화되도록 설정했다. 여기서 포화란 추가 학습을 해도 training loss가 더 이상 크게 개선되지 않는 상태를 의미한다.

---

### 6. 암기량 측정 방법

#### 6.1 압축 기반 측정
논문은 Kolmogorov complexity에서 영감을 받아 데이터 샘플의 정보량을 측정한다.

- 모델이 샘플 \(x\)의 확률을 높게 부여할수록 해당 샘플을 더 짧게 압축할 수 있음
- 따라서 모델의 log-likelihood 또는 entropy를 압축 길이의 근사값으로 사용
- 데이터 자체의 정보량에서 모델 조건부 압축 길이를 빼서 암기량을 계산

합성 무작위 데이터에서는 데이터의 엔트로피를 정확히 알고 있으므로,

\[
\text{Memorization}
\approx
H(X)-H(X|\hat{\theta})
\]

로 계산한다.

#### 6.2 실제 텍스트에서의 의도하지 않은 암기
실제 텍스트에서는 타깃 모델 \(\hat{\theta}\)와 기준 모델 \(\theta\)의 손실 차이를 사용한다.

개념적으로는 다음과 같다.

\[
\text{MEM}_{U}(x)
\approx
\max\{L_{\theta}(x)-L_{\hat{\theta}}(x),0\}
\]

즉,

- 타깃 모델이 기준 모델보다 특정 샘플을 훨씬 더 잘 예측하면
- 그 샘플에 대한 추가 정보가 타깃 모델에 저장되었다고 보고
- 그 차이를 의도하지 않은 암기로 계산한다.

---

### 7. 모델 용량 측정
모델의 **암기 용량(capacity)** 은 데이터 크기를 증가시키면서 측정한 총 암기량의 최대값으로 정의한다.

\[
\text{Capacity}(L)
=
\max_X \text{Mem}(X,L(X))
\]

합성 무작위 데이터에서는 일반화가 없으므로, 암기량이 모델이 저장할 수 있는 총 정보량을 직접 나타낸다.

실험 결과:

- GPT 계열 모델은 대략 **파라미터당 3.5~3.6 bits**를 저장
- float32에서는 평균 약 **3.83 bits per parameter**
- bfloat16에서는 평균 약 **3.51 bits per parameter**
- 정밀도를 2배 높여도 용량은 2배가 되지 않음

데이터셋이 작을 때는 모델이 데이터를 거의 모두 암기하지만, 데이터가 커져 모델 용량을 초과하면 총 암기량은 plateau에 도달한다.

---

### 8. 추가 평가: 일반화, double descent, 추출, membership inference
논문은 암기량 측정 외에도 다음을 평가했다.

#### 일반화와 double descent
데이터셋 크기가 모델 용량보다 작을 때는 모델이 샘플을 직접 암기할 수 있다. 데이터 크기가 용량을 넘으면 개별 샘플 암기가 어려워지고, 모델은 여러 샘플에 공통된 패턴을 학습하기 시작한다. 논문은 이 지점에서 **double descent 현상**이 나타난다고 설명한다.

#### Extraction
학습 데이터의 일부 prefix를 입력하고 greedy decoding으로 나머지 문장을 복원하는 방식으로 추출률을 측정했다. 하지만 큰 데이터셋에서는 학습 샘플의 추출률이 테스트 샘플과 비슷해지므로, 추출 성공만으로는 의도하지 않은 암기의 증거가 될 수 없다고 주장한다.

#### Membership inference
loss-based membership inference를 사용했다.

- 모델의 loss가 낮으면 training member로 분류
- loss가 높으면 non-member로 분류
- F1 score로 성능 평가

결과적으로:

- 모델이 클수록 더 많은 데이터를 암기할 수 있음
- 데이터셋이 커질수록 평균 샘플의 membership inference가 어려워짐
- 현대 언어 모델처럼 파라미터당 매우 많은 토큰으로 학습한 모델에서는 평균 샘플에 대한 membership inference가 거의 무작위 수준이 될 수 있음

---



## Method

### 1. Main idea
The paper measures memorization as the amount of information a model can use to **compress a particular data point**. It distinguishes:

- **Unintended memorization**: information stored about a specific training sample
- **Generalization**: information about reusable patterns in the underlying data distribution

For real text, unintended memorization is estimated by comparing a target model with a larger reference model. If the target model compresses a sample substantially better than the reference model, the difference is attributed to sample-level memorization.

---

### 2. Model architecture
The experiments use **GPT-2-style decoder-only Transformers** trained from scratch.

Main settings include:

- 1–8 Transformer layers
- Hidden dimensions from 32 to 512
- Approximately 100K–20M parameters in the main synthetic experiments
- Additional validation with models from roughly 500K to 1.5B parameters
- Mostly bfloat16 training
- Some fp32 experiments to study numerical precision

The paper does not introduce a new Transformer architecture. Instead, it uses a standard GPT-style architecture to systematically vary model size and dataset size.

---

### 3. Training data

#### Synthetic random data
To remove generalization completely, the authors first train on uniformly random sequences.

- Each token is sampled independently
- Vocabulary size: \(V=2048\)
- Sequence length: \(S=64\)
- Dataset size is varied across experiments

Because every token is independent, the entropy of the dataset is known exactly:

\[
H(X)=NS\log_2 V
\]

There are no reusable patterns to learn, so information learned from the data is interpreted as unintended memorization.

#### Real text
For natural-language experiments, the authors use the **FineWeb** dataset.

- Text is divided into sequences of 64 tokens
- Additional deduplication is applied
- Multiple dataset sizes and model sizes are tested

This setting contains both memorization and generalization, so a reference model is needed to separate the two.

---

### 4. Target and reference models
The **target model** is trained on the experimental dataset. The **reference model** is a larger model trained on a much broader amount of text and is assumed to better approximate the underlying data distribution.

The main reference is a 774M-parameter FineWeb-trained GPT-2 model. The authors also compare 124M, 355M, and 774M reference models.

A smaller reference model produces larger absolute memorization estimates because it assigns higher loss to the text. Therefore, the absolute value depends on the reference model, while relative comparisons are more reliable when the reference is fixed.

---

### 5. Training procedure
The models are:

- Randomly initialized and trained from scratch
- Optimized with Adam
- Trained with batch size 2048
- Trained for up to \(10^6\) steps
- Usually trained on a single NVIDIA A100 GPU
- Repeated across multiple random seeds

The long training schedule is intended to ensure that the models reach a saturation regime in which additional training no longer substantially improves the training loss.

---

### 6. Measuring memorization

#### Compression-based measurement
The method is inspired by Kolmogorov complexity.

- A sequence assigned a higher likelihood by a model can be encoded more compactly
- Model likelihood or entropy is used as a practical approximation to compression length
- Memorization is estimated as the reduction in description length produced by the model

For synthetic random data:

\[
\text{Memorization}
\approx
H(X)-H(X|\hat{\theta})
\]

#### Real-text measurement
For real text, the authors compare the losses of the reference and target models:

\[
\text{MEM}_{U}(x)
\approx
\max\{L_{\theta}(x)-L_{\hat{\theta}}(x),0\}
\]

A positive gap means that the target model predicts the sample better than the reference model, suggesting that the target has stored sample-specific information.

---

### 7. Estimating model capacity
Model capacity is defined as the maximum total memorization observed as dataset size increases.

The synthetic-data experiments show that GPT-style Transformers store approximately:

- **3.5–3.6 bits per parameter** in bfloat16
- Approximately **3.83 bits per parameter** in fp32

Memorization increases with dataset size until the model reaches its capacity. After that point, total memorization plateaus.

---

### 8. Additional evaluations
The paper also evaluates:

- **Generalization and double descent**: double descent begins when dataset information exceeds model memorization capacity.
- **Extraction**: greedy decoding is used to reconstruct training sequences, but extraction can also result from generalization.
- **Membership inference**: a loss-based attack classifies samples as training members or non-members.

The main findings are that larger models memorize more, while larger datasets make average-sample membership inference more difficult.


<br/>
# Results



### 1. 실험 목적과 비교 대상

이 논문은 언어 모델이 학습 데이터를 얼마나 저장하는지 측정할 때, **단순 암기(memorization)**와 **일반화(generalization)**를 분리하는 것을 목표로 합니다.

논문에서 비교한 모델은 특정 상용 모델 간 경쟁이라기보다, 다음과 같은 **GPT-2 계열 Transformer 모델들의 크기·정밀도·학습 데이터 조건**입니다.

- **모델 구조:** GPT-2 스타일 Transformer
- **주요 모델 크기:** 약 10만~2,000만 파라미터
- **추가 검증 모델:** GPT-2 Medium 약 1.24억, GPT-2 XL 약 15.6억 파라미터
- **모델 크기 비교:** 약 0.5M~20M 파라미터 중심
- **수치 정밀도 비교:** bfloat16과 float32
- **참조 모델 비교:** FineWeb으로 학습한 124M, 355M, 774M 파라미터 모델

즉, 핵심 비교축은 **모델 크기, 데이터셋 크기, 정밀도, 참조 모델의 성능**입니다.

---

### 2. 테스트 데이터

#### (1) 합성 무작위 데이터

- 각 토큰을 균등 무작위로 생성한 비트열 또는 토큰열
- 토큰들이 서로 독립적이므로 데이터에 학습할 규칙이나 패턴이 없음
- 따라서 이 데이터에서는 모델이 얻을 수 있는 **일반화가 사실상 0**
- 모델이 학습 후 더 잘 압축하는 정보는 거의 전부 **의도하지 않은 샘플 수준 암기**로 해석 가능

이 데이터는 모델의 **순수한 암기 용량(capacity)**을 측정하기 위한 기준 데이터입니다.

#### (2) 실제 텍스트 데이터

- FineWeb 데이터셋 사용
- 학습 데이터에 대해 추가 중복 제거를 수행해 완전한 중복을 최소화
- 학습 데이터와 겹치지 않는 FineWeb 텍스트를 테스트 데이터로 사용
- 실제 텍스트에는 문법, 단어 사용, 사실적 지식 등의 규칙이 있으므로 암기와 일반화가 함께 나타남

---

### 3. 주요 평가 지표

#### (1) 비의도적 암기량

논문은 데이터가 모델 안에 얼마나 추가로 저장되었는지를 **비트(bit)** 단위로 측정합니다.

개념적으로는 다음과 같습니다.

- 모델이 없을 때 데이터를 표현하는 데 필요한 정보량
- 모델이 있을 때 데이터를 압축하는 데 필요한 정보량
- 두 값의 차이 = 모델이 해당 데이터에 대해 보유한 정보량

실험에서는 정확한 Kolmogorov 복잡도를 계산할 수 없기 때문에, 모델의 **로그 likelihood 또는 cross-entropy loss를 이용한 압축률**로 근사합니다.

#### (2) 모델 용량

여러 크기의 데이터셋을 학습시키면서 암기량이 더 이상 증가하지 않고 plateau에 도달하는 지점을 모델 용량으로 정의합니다.

\[
\text{Capacity} \approx \text{모델이 저장할 수 있는 총 암기 정보량}
\]

#### (3) 일반화

실제 텍스트에서는 큰 참조 모델(oracle model)이 이미 설명할 수 있는 정보와, 학습 대상 모델이 추가로 학습한 정보를 구분합니다.

예를 들어:

- `2^100`의 값을 계산하는 능력은 일반화된 능력
- 특정 인물의 이름, 점수, 날짜처럼 데이터셋에만 있는 세부 내용은 비의도적 암기에 가까움

#### (4) Extraction rate

특정 prefix를 주었을 때 모델이 학습 문장을 그대로 이어서 생성하는 비율입니다.

다만 논문은 **문장을 생성할 수 있다는 사실만으로 암기를 증명할 수 없다**고 지적합니다. 일반화된 언어 능력만으로도 학습하지 않은 테스트 문장을 생성할 수 있기 때문입니다.

#### (5) Membership inference F1

loss를 기준으로 어떤 샘플이 학습 데이터에 포함되었는지 판별하는 공격의 F1 점수입니다.

- F1 = 0.5: 사실상 무작위 추측
- F1이 높을수록 학습 데이터와 테스트 데이터를 잘 구분
- 평균적인 학습 샘플의 노출 위험을 평가하는 지표

---

### 4. 핵심 결과

#### 결과 1. GPT 스타일 모델은 파라미터당 약 3.6비트를 암기한다

무작위 데이터 실험에서 모델의 암기 용량은 대체로 다음과 같이 나타났습니다.

- **bfloat16:** 평균 약 **3.51 bits per parameter**
- **float32:** 평균 약 **3.83 bits per parameter**
- 전체적으로 약 **3.5~4 bits per parameter**
- 대표적인 요약값은 약 **3.6 bits per parameter**

따라서 파라미터 수가 증가하면 암기 가능한 정보량도 거의 선형적으로 증가합니다.

예를 들어 수백만 개의 파라미터를 가진 모델은 수천만 비트 규모의 무작위 정보를 저장할 수 있습니다.

단, 이 수치는 최적화가 전역 최적해를 찾았다고 보장되지 않으므로 **실제 이론적 최대 용량이라기보다 실험적으로 측정한 하한에 가깝습니다.**

---

#### 결과 2. 데이터가 작을 때는 거의 전부 암기하지만, 용량을 넘으면 암기량이 포화된다

- 작은 데이터셋에서는 모델이 학습 샘플 대부분을 직접 저장할 수 있음
- 데이터셋 크기가 모델 용량보다 커지면 총 암기량이 더 이상 증가하지 않음
- 이후에는 모델이 개별 샘플을 저장하는 대신 여러 샘플에 공통적인 패턴을 학습함

즉, 모델은 다음과 같은 단계적 동작을 보입니다.

1. 데이터가 적음 → 샘플을 주로 암기
2. 모델 용량에 접근 → 암기량 증가가 둔화
3. 용량 초과 → 암기 대신 일반화 비중 증가

---

#### 결과 3. 정밀도를 두 배로 높여도 암기 용량은 두 배가 되지 않는다

bfloat16에서 float32로 바꾸면 파라미터 하나를 표현하는 비트 수는 크게 증가하지만, 실제 암기 용량은 다음 정도만 증가했습니다.

- bfloat16: 약 3.51 bpp
- float32: 약 3.83 bpp

즉, 저장 가능한 원시 파라미터 비트 수가 두 배가 되어도 실제 학습 과정에서 데이터 암기에 사용되는 용량은 약간만 증가했습니다. 추가 정밀도의 상당 부분은 암기 저장에 직접 활용되지 않는 것으로 해석됩니다.

---

#### 결과 4. 실제 텍스트에서는 데이터가 많아질수록 샘플별 암기가 감소한다

실제 텍스트 데이터에서는 일반화가 가능하기 때문에 결과가 더 복잡합니다.

- 모델이 클수록 더 많은 샘플 수준 정보를 암기
- 데이터셋이 작을수록 개별 샘플 암기가 증가
- 데이터셋이 커져 모델 용량을 초과하면 비의도적 암기가 감소
- 대신 문법, 표현 방식, 반복되는 지식 등 재사용 가능한 패턴에 대한 일반화가 증가

따라서 대규모 데이터셋에서 모델의 학습 데이터가 그대로 저장되는 것이 아니라, 많은 경우 학습 데이터와 테스트 데이터에 공통적인 언어 규칙이 학습됩니다.

---

#### 결과 5. Double descent는 데이터 용량이 모델 용량을 넘는 지점에서 나타난다

논문은 double descent 현상을 다음과 같이 설명합니다.

- 데이터셋이 작을 때는 모델이 데이터를 거의 완전히 암기할 수 있음
- 데이터셋 크기가 모델의 암기 용량에 가까워지면 성능이 악화될 수 있음
- 데이터가 용량을 초과하면 모델은 샘플별 암기 대신 정보 공유와 일반화를 시작함

논문에서는 **데이터셋의 정보량이 모델 용량을 초과하는 지점이 double descent가 시작되는 지점과 일치한다**고 관찰했습니다.

---

#### 결과 6. Extraction은 일반화와 암기를 구분하지 못한다

학습 데이터에서 문장을 그대로 생성하는 extraction rate를 측정한 결과:

- 작은 데이터셋에서는 학습 문장의 extraction rate가 높음
- 데이터셋이 커지면 extraction rate가 감소
- 그러나 충분히 큰 데이터셋에서도 extraction이 완전히 0이 되지는 않음
- 이때 학습 데이터와 테스트 데이터의 extraction rate가 거의 같아짐

이는 대규모 데이터에서 성공적인 문장 생성이 발생하더라도, 그것이 특정 학습 샘플의 암기 때문이라기보다 **일반화된 언어 능력 때문일 수 있음**을 의미합니다.

따라서 extraction만으로 암기를 정의하면 일반화와 비의도적 암기를 혼동하게 됩니다.

---

#### 결과 7. Membership inference는 모델이 커질수록 강해지고, 데이터가 커질수록 약해진다

Membership inference 결과는 비교적 일관된 경향을 보였습니다.

- 모델이 클수록 더 많은 정보를 암기하므로 공격 성능이 높음
- 데이터셋이 작을수록 과적합이 심해져 membership inference가 쉬움
- 데이터셋이 커질수록 학습 샘플과 테스트 샘플의 loss 차이가 줄어들어 공격이 어려워짐
- 모델 용량에 비해 데이터셋이 충분히 크면 F1은 0.5에 가까워짐

실제 텍스트 실험에서는 일부 조건에서 F1이 약 **0.97**까지 올라갔지만, 데이터셋 규모가 커지면 급격히 낮아졌습니다.

논문이 제안한 membership inference 예측식은 대략 다음 형태입니다.

\[
F1 = \frac{1}{2}\left(1+c_1\sigma\left(c_2
\frac{\text{Capacity}}{|D|}+c_3\right)\right)
\]

핵심은 **모델 용량 대비 데이터셋 크기 비율**이 membership inference 성능을 설명한다는 것입니다.

---

### 5. 모델 크기와 데이터 크기의 비교

논문의 전체적인 결론은 다음과 같습니다.

| 조건 | 예상되는 현상 |
|---|---|
| 작은 모델 + 작은 데이터셋 | 데이터 암기 가능 |
| 큰 모델 + 같은 데이터셋 | 더 강한 암기 및 membership inference |
| 같은 모델 + 더 큰 데이터셋 | 샘플별 암기와 membership inference 감소 |
| 데이터셋 정보량이 모델 용량보다 작음 | 주로 암기 |
| 데이터셋 정보량이 모델 용량보다 큼 | 암기 포화 후 일반화 증가 |
| 매우 큰 데이터셋 | 평균 샘플에 대한 membership inference가 거의 불가능 |

논문은 현재 많은 대규모 언어 모델이 파라미터 수에 비해 매우 많은 토큰으로 학습되므로, **평균적인 학습 샘플에 대한 membership inference는 신뢰성 있게 수행하기 어려울 것**이라고 예측합니다.

---

### 6. 해석상의 주의점

- 3.6 bits-per-parameter는 모든 언어 모델에 적용되는 보편적 상수가 아니라, 주로 GPT 스타일 Transformer와 해당 학습 조건에서 얻은 실험값입니다.
- 실제 텍스트의 비의도적 암기량은 어떤 참조 모델을 사용하느냐에 따라 절대값이 달라집니다.
- 다만 참조 모델을 고정하면 모델 간 상대적 비교는 비교적 안정적입니다.
- extraction rate는 일반화된 언어 능력도 반영하므로 암기의 직접적인 증거로 사용하기 어렵습니다.
- 실험 결과는 특정 데이터셋, 아키텍처, 학습 설정에 의존하며 모든 현대 LLM에 그대로 일반화된다고 보기는 어렵습니다.

---




### 1. Experimental setup and comparison axes

The paper studies how much information language models retain from their training data while separating **unintended memorization** from **generalization**.

Rather than comparing commercial models directly, the experiments compare GPT-style Transformer models across:

- Model size: roughly 0.1M–20M parameters in the main experiments
- Larger validation models: GPT-2 Medium, about 124M parameters, and GPT-2 XL, about 1.56B parameters
- Numerical precision: bfloat16 versus float32
- Dataset size
- Reference-model size: 124M, 355M, and 774M FineWeb-matched models

---

### 2. Test data

#### Synthetic random data

The authors generate uniformly random token or bit sequences.

- Each token is sampled independently.
- There is no meaningful pattern to generalize.
- Therefore, generalization is effectively zero.
- Information learned beyond the known entropy of the data is interpreted as unintended memorization.

This setting is used to estimate the model’s pure memorization capacity.

#### Real text

The authors also train on deduplicated FineWeb text.

- Additional deduplication is applied to the training sequences.
- Non-overlapping FineWeb data is used for evaluation.
- Unlike random data, real text contains linguistic and semantic regularities.
- Consequently, both memorization and generalization can occur.

---

### 3. Evaluation metrics

The main metrics are:

1. **Unintended memorization:**  
   The amount of additional information about a sample stored in the trained model, measured in bits.

2. **Model capacity:**  
   The maximum total memorization observed as the dataset size increases.

3. **Extraction rate:**  
   The fraction of sequences that can be reproduced by decoding from a prefix.

4. **Membership-inference F1:**  
   The performance of a loss-based attack that predicts whether a sample was included in training. An F1 score of 0.5 corresponds to random guessing.

The memorization measure is based on compression. In practice, Kolmogorov complexity is approximated using model likelihoods or cross-entropy losses.

---

### 4. Main findings

#### Finding 1: GPT-style models memorize approximately 3.6 bits per parameter

On uniformly random data, the measured capacity is approximately:

- **3.51 bits per parameter** in bfloat16
- **3.83 bits per parameter** in float32
- Overall, roughly **3.5–4 bits per parameter**

Thus, memorization capacity scales approximately linearly with the number of parameters.

The authors emphasize that these are empirical measurements and likely lower bounds, since gradient descent is not guaranteed to find the global optimum.

---

#### Finding 2: Memorization increases until capacity is reached

For small datasets, models can memorize nearly all available information.

As the dataset grows:

1. Memorization initially increases.
2. It reaches a plateau near the model’s capacity.
3. Additional data is increasingly represented through shared, generalizable patterns rather than sample-specific storage.

This behavior appears clearly in the synthetic-data experiments.

---

#### Finding 3: Higher numerical precision does not double practical memorization capacity

Although float32 uses twice as many bits per parameter as bfloat16, the measured capacity only increases modestly:

- bfloat16: about 3.51 bits per parameter
- float32: about 3.83 bits per parameter

Therefore, much of the additional raw parameter precision is not used directly for storing memorized information.

---

#### Finding 4: On real text, sample-level memorization decreases after capacity is filled

For real text:

- Larger models memorize more sample-specific information.
- Smaller datasets produce more memorization.
- Once the dataset exceeds model capacity, unintended memorization decreases.
- The model increasingly learns reusable linguistic and semantic patterns.

This demonstrates why memorization and generalization must be measured separately.

---

#### Finding 5: Double descent begins when data capacity exceeds model capacity

The authors observe that double descent appears near the point where the information content of the dataset becomes larger than the model’s memorization capacity.

The proposed interpretation is:

- Before the capacity limit, the model can fit samples individually.
- Near the limit, performance can deteriorate.
- Beyond the limit, the model must share information across samples and rely more on generalization.

---

#### Finding 6: Extraction does not reliably distinguish memorization from generalization

Extraction rates decrease as the training dataset becomes larger. However, they do not necessarily converge to zero.

In the large-data regime, the extraction rate on training examples becomes nearly the same as on unseen test examples. This means that successful generation of a training sequence may be explained by general language modeling ability rather than unintended memorization.

Therefore, extraction alone is not sufficient evidence that a model memorized a particular training sample.

---

#### Finding 7: Membership inference becomes easier for larger models and smaller datasets

Membership inference follows a clear pattern:

- Larger models memorize more and are easier to attack.
- Smaller datasets encourage overfitting and improve attack performance.
- Larger datasets reduce the loss gap between train and test samples.
- When the dataset is sufficiently large relative to model capacity, F1 approaches 0.5.

In some text experiments, membership inference achieved an F1 score close to **0.97**, but performance declined as the dataset grew.

The proposed scaling law mainly depends on the ratio:

\[
\frac{\text{Model capacity}}{\text{Dataset size}}
\]

A larger ratio implies stronger membership-inference performance.

---

### 5. Overall interpretation

The paper’s central conclusion is:

- **Small dataset relative to model capacity:** mostly sample-level memorization
- **Large model on the same dataset:** more memorization and stronger membership inference
- **Large dataset relative to model capacity:** memorization saturates and generalization becomes more important
- **Very large training datasets:** reliable membership inference on an average training example becomes difficult

The paper predicts that many modern language models are trained on so many tokens relative to their parameter counts that average-case membership inference should be close to random guessing.

---

### 6. Important caveats

- The estimate of 3.6 bits per parameter is not a universal constant for all language models.
- Absolute unintended-memorization values on real text depend on the chosen reference model.
- Relative comparisons are more reliable when the reference model is fixed.
- Extraction can reflect generalization and should not be treated as direct proof of memorization.
- The findings depend on the tested architectures, datasets, optimization procedures, and evaluation settings.


<br/>
# 예제



이 논문은 언어 모델이 학습 데이터를 **그대로 기억하는지(unintended memorization)**, 아니면 데이터에 존재하는 일반적인 규칙을 **일반화하는지(generalization)**를 구분하기 위해 여러 실험을 수행합니다. 핵심 과제는 모두 **다음 토큰 예측(next-token prediction)**입니다.

---

## 1. 기본 언어 모델 학습 과제

### 입력과 출력

언어 모델에는 문장의 앞부분을 입력하고, 그다음 토큰을 예측하게 합니다.

**예시**

- 입력:  
  `The capital of France is`
- 정답 출력:  
  `Paris`

또는 토큰 단위로 보면 다음과 같습니다.

- 입력: `The capital of France`
- 출력: `is`
- 입력: `The capital of France is`
- 출력: `Paris`

논문에서는 일반적인 GPT 구조의 Transformer를 처음부터 학습시켰습니다. 입력 데이터는 길이 64 토큰의 시퀀스로 구성되며, 모델은 각 위치에서 다음 토큰을 예측합니다.

중요한 점은 별도의 정답 라벨이 있는 분류 문제가 아니라, **문맥 다음에 올 토큰의 확률을 높이는 문제**라는 것입니다.

---

## 2. 합성 무작위 데이터 실험: 순수한 암기 측정

이 실험은 일반화의 가능성을 완전히 제거하기 위해 수행되었습니다.

### 데이터 생성 방식

각 토큰을 이전 토큰과 무관하게 무작위로 뽑습니다.

- 어휘 크기: `V = 2048`
- 시퀀스 길이: `S = 64`
- 각 토큰: 2048개 토큰 중 하나를 균등하게 무작위 선택
- 데이터셋: 이러한 시퀀스를 여러 개 모은 것

### 축약된 예시

실제 토큰 대신 숫자로 표시하면 다음과 같습니다.

#### 학습 데이터

```text
[17, 804, 3, 1991, 45, 722, ...]
[901, 12, 1777, 88, 631, ...]
[4, 1450, 92, 18, 2030, ...]
```

#### 모델의 학습 과제

```text
입력: [17, 804, 3, 1991]
출력: [45]
```

또는 전체적으로는 각 위치의 다음 토큰을 예측합니다.

```text
입력: [17, 804, 3, 1991, 45]
출력: [722]
```

그러나 이 데이터에는 반복되는 문법, 의미, 사실 또는 규칙이 없습니다. 따라서 모델이 학습 후 특정 시퀀스를 잘 예측한다면, 그것은 일반화가 아니라 **그 시퀀스에 대한 정보가 모델 파라미터에 저장되었기 때문**이라고 해석할 수 있습니다.

### 테스트 데이터

테스트 데이터도 동일한 방식으로 새롭게 무작위 생성합니다.

```text
[17, 804, 3, 1991, 45, 722, ...]  ← 학습 데이터에 있던 시퀀스
[512, 76, 1880, 6, 1433, ...]      ← 테스트 데이터의 새로운 시퀀스
```

- 학습 시퀀스: 모델이 본 적 있음
- 테스트 시퀀스: 모델이 본 적 없음
- 둘 다 같은 무작위 생성 규칙에서 생성됨

### 측정하는 것

학습 시퀀스와 테스트 시퀀스에 대해 모델이 부여하는 확률 또는 손실을 비교합니다.

- 학습 데이터의 확률이 특별히 높음  
  → 해당 샘플을 암기했을 가능성
- 학습 데이터와 테스트 데이터의 성능이 비슷함  
  → 무작위 데이터에서는 일반화가 거의 일어나지 않음

이 실험에서 모델이 저장할 수 있는 정보량은 일정 수준에서 더 이상 증가하지 않았습니다. 이를 모델의 **암기 용량(capacity)**으로 정의했습니다.

논문은 GPT 계열 모델이 대략 다음 정도의 정보를 저장한다고 측정했습니다.

> **약 3.5~3.6 bits per parameter**  
> 정밀도를 높인 FP32에서는 평균 약 **3.83 bits per parameter**

예를 들어 파라미터가 1,000,000개라면 대략 수백만 비트 수준의 무작위 정보를 저장할 수 있다는 뜻입니다. 이는 파라미터의 물리적인 비트 수와 동일하다는 의미는 아닙니다.

---

## 3. 실제 텍스트 실험: 암기와 일반화의 분리

무작위 데이터와 달리 실제 텍스트에는 문법, 단어 사용 패턴, 사실과 같은 일반적인 구조가 있습니다.

논문에서는 FineWeb에서 중복 제거한 텍스트를 사용하고, 텍스트를 길이 64 토큰의 시퀀스로 나누었습니다.

### 학습 데이터 예시

```text
입력:
"Marie Curie was a Polish and naturalized-French physicist ..."
```

모델은 다음 토큰들을 예측합니다.

```text
출력:
"who conducted pioneering research on radioactivity ..."
```

또는 여러 위치에 대해 다음과 같이 학습합니다.

```text
입력: "Marie Curie was a"
출력: "Polish"

입력: "Marie Curie was a Polish"
출력: "and"

입력: "Marie Curie was a Polish and naturalized-French"
출력: "physicist"
```

### 테스트 데이터 예시

테스트 데이터는 학습에 사용되지 않은, 겹치지 않는 FineWeb 시퀀스입니다.

```text
입력:
"Albert Einstein developed the theory of ..."
```

```text
정답 출력:
"relativity"
```

테스트 데이터에 대해서도 모델은 같은 다음 토큰 예측 과제를 수행합니다.

### 여기서 구분하는 두 가지 정보

#### 1) 일반화

모델이 학습 데이터에 직접 등장하지 않은 문장도 잘 예측하는 경우입니다.

예를 들어 학습 중에 “physicist”라는 단어와 문법을 많이 배웠다면, 새로운 문장에서 물리학 관련 문맥을 어느 정도 예측할 수 있습니다.

#### 2) 의도하지 않은 암기

특정 학습 문장의 이름, 숫자, 희귀한 표현 등 개별적인 세부사항을 모델이 특별히 기억하는 경우입니다.

예를 들어 다음 문장이 학습 데이터에 있었다고 합시다.

```text
John Smith scored 147 points in the 2019 regional bowling championship.
```

모델이 일반적으로 알고 있는 것은 다음과 같은 문장 구조일 수 있습니다.

```text
[사람 이름] scored [점수] points in the [연도] ...
```

하지만 `John Smith`, `147`, `2019`라는 구체적인 조합까지 높은 확률로 복원한다면, 이는 해당 샘플에 대한 암기에 가깝습니다.

---

## 4. “일반 지식”과 “샘플 암기”를 구분하는 예시

논문은 다음과 같은 비교를 제시합니다.

### 예시 A: 일반화에 가까운 경우

```text
Q: What is 2^100?
A: 1267650600228229401496703205376
```

계산 능력이 있는 참조 모델이라면 이 답을 학습 데이터에 있는 정확한 문장을 보지 않고도 계산할 수 있습니다. 따라서 이 답이 학습 데이터에 등장했다는 이유만으로 암기라고 볼 수 없습니다.

### 예시 B: 샘플 암기에 가까운 경우

```text
John Smith scored 147 points in the
2019 regional bowling championship.
```

문장의 문법적 형태는 일반화할 수 있지만, 이름·점수·연도라는 구체적인 조합은 일반적인 언어 지식만으로 예측하기 어렵습니다. 모델이 이 문장을 유난히 낮은 손실로 예측한다면, 그 부분을 학습 샘플에서 기억했을 가능성이 큽니다.

논문의 기준에서는 참조 모델도 잘 설명할 수 있는 정보는 **generalization**, 참조 모델보다 학습 모델이 추가로 알고 있는 정보는 **unintended memorization**으로 봅니다.

---

## 5. 압축률을 이용한 암기 측정

논문은 “모델이 샘플을 얼마나 짧게 설명할 수 있게 해주는가?”를 암기의 양으로 측정합니다.

### 직관

어떤 데이터 `x`가 원래 1000비트로 표현되는데, 모델을 이용하면 700비트로 표현할 수 있다고 합시다.

```text
모델이 없을 때: 1000 bits
모델이 있을 때:  700 bits
차이:            300 bits
```

이 경우 모델이 해당 샘플에 대해 약 300비트의 정보를 갖고 있다고 해석합니다.

실제 계산에서는 Kolmogorov complexity를 직접 계산할 수 없기 때문에, 모델의 **negative log-likelihood 또는 cross-entropy loss**를 압축 길이의 근사값으로 사용합니다.

실제 텍스트에서는 다음과 같은 방식으로 계산합니다.

```text
unintended memorization
≈ max(
    참조 모델의 손실 - 학습 모델의 손실,
    0
  )
```

즉, 학습 모델이 참조 모델보다 특정 샘플을 더 잘 설명하는 부분을 암기로 계산합니다.

---

## 6. 추출(extraction) 실험

추출은 모델에게 학습 데이터의 앞부분을 보여주고 뒷부분을 정확히 생성할 수 있는지 확인하는 실험입니다.

### 입력과 출력

#### 학습 데이터에 있는 문장

```text
전체 문장:
"The rare chemical compound was discovered in 1897 by ..."
```

#### 추출 프롬프트

```text
입력:
"The rare chemical compound was discovered in"
```

#### 모델 출력

```text
예측 출력:
"1897 by ..."
```

모델이 이후 내용을 정확히 생성하면 해당 문장이 “추출 가능하다”고 기록합니다.

논문에서는 prefix 길이를 8, 16, 32 토큰 등으로 바꾸어 평가했습니다.

### 테스트 데이터에도 동일하게 적용

학습에 사용되지 않은 테스트 문장에도 같은 방식으로 프롬프트를 제공합니다.

```text
테스트 입력:
"The rare chemical compound was first reported ..."
```

테스트 문장도 모델이 일반적인 언어 패턴만으로 잘 완성할 수 있다면 추출될 수 있습니다.

따라서 논문의 중요한 결론은 다음과 같습니다.

> 학습 문장이 성공적으로 생성되었다는 사실만으로는 의도하지 않은 암기의 증거가 아니다.

데이터셋이 충분히 커지면 학습 데이터와 테스트 데이터의 추출률이 거의 같아졌습니다. 이 경우 학습 데이터가 생성된 것은 샘플을 개별적으로 기억했기 때문이 아니라, 모델이 일반적인 언어 구조를 학습했기 때문일 수 있습니다.

---

## 7. 멤버십 추론(membership inference) 실험

멤버십 추론은 특정 문장이 모델의 학습 데이터에 포함되었는지를 맞히는 과제입니다.

### 입력

공격자는 다음과 같은 문장 하나를 모델에 제시합니다.

```text
"John Smith scored 147 points in the
2019 regional bowling championship."
```

### 정답 라벨

- `member = 1`: 학습 데이터에 포함된 문장
- `member = 0`: 학습에 사용되지 않은 테스트 문장

### 공격 방식

논문에서는 문장 전체의 손실을 이용합니다.

- 손실이 낮음 → 모델이 잘 기억했을 가능성 → `member`
- 손실이 높음 → 모델이 본 적 없을 가능성 → `non-member`

예를 들어:

| 문장 | 모델 손실 | 예측 |
|---|---:|---|
| 학습 문장 A | 1.2 | member |
| 테스트 문장 B | 3.8 | non-member |

하지만 학습 데이터가 커지면 모델 용량이 여러 샘플에 분산되고, 학습 문장과 테스트 문장의 손실 차이가 줄어듭니다. 그러면 멤버십 추론은 점점 어려워집니다.

논문은 대체로 다음과 같은 경향을 확인했습니다.

- 모델이 클수록 더 많은 데이터를 암기할 수 있음
- 데이터셋이 클수록 개별 샘플에 대한 암기는 약해짐
- 데이터가 모델 용량에 비해 지나치게 많으면 평균 샘플에 대한 멤버십 추론은 거의 무작위 추측 수준이 됨
- 현대 언어 모델처럼 파라미터당 매우 많은 토큰을 학습한 모델에서는 평균적인 샘플에 대한 멤버십 추론이 어려울 것으로 예측됨

---

## 8. 전체 실험의 핵심 비교

| 실험 | 학습 입력 | 테스트 입력 | 모델이 해야 하는 일 | 측정 목적 |
|---|---|---|---|---|
| 무작위 시퀀스 | 무작위 토큰 시퀀스 | 새로운 무작위 토큰 시퀀스 | 다음 토큰 예측 | 일반화를 제거하고 순수 암기 측정 |
| 실제 텍스트 | FineWeb의 학습 문장 | 겹치지 않는 FineWeb 문장 | 다음 토큰 예측 | 암기와 일반화 분리 |
| 추출 | 문장의 앞부분 | 학습/테스트 문장의 앞부분 | 뒷부분을 생성 | 문장 재생성 가능성 측정 |
| 멤버십 추론 | 후보 문장 하나 | 후보 문장 하나 | 학습 포함 여부 분류 | 학습 데이터 노출 여부 측정 |

---



This paper studies whether language models **memorize individual training examples** or instead **generalize reusable patterns**. The main task in all experiments is standard **next-token prediction**.

---

## 1. Basic language-modeling task

A language model receives a prefix and predicts the next token.

**Example**

```text
Input:  The capital of France is
Output: Paris
```

At the token level:

```text
Input:  The capital of France
Target: is

Input:  The capital of France is
Target: Paris
```

The paper does not use a separate classification label. The models are trained to increase the probability of the correct next token.

---

## 2. Synthetic random-data experiment

The authors first use completely random sequences so that there is no meaningful structure to generalize.

### Data generation

- Vocabulary size: `V = 2048`
- Sequence length: `S = 64` tokens
- Each token is sampled independently and uniformly
- A dataset contains many such random sequences

### Simplified example

#### Training data

```text
[17, 804, 3, 1991, 45, 722, ...]
[901, 12, 1777, 88, 631, ...]
[4, 1450, 92, 18, 2030, ...]
```

#### Training objective

```text
Input:  [17, 804, 3, 1991]
Target: [45]
```

and then:

```text
Input:  [17, 804, 3, 1991, 45]
Target: [722]
```

Because the sequences are random, the model cannot learn grammar, facts, or semantic regularities. If it predicts a particular training sequence unusually well, this must come from information stored about that sequence.

### Test data

Test sequences are generated independently using the same random process.

```text
Training sequence:
[17, 804, 3, 1991, 45, 722, ...]

Unseen test sequence:
[512, 76, 1880, 6, 1433, ...]
```

The training sequence has been seen by the model; the test sequence has not.

The authors find that memorization increases with dataset size at first, but eventually reaches a plateau. This plateau is interpreted as the model’s memorization capacity.

Their estimate for GPT-style Transformers is approximately:

> **3.5–3.6 bits per parameter** in bfloat16  
> approximately **3.83 bits per parameter** in fp32

---

## 3. Real-text experiment

The authors then use deduplicated FineWeb text. Unlike random sequences, real text contains grammar, facts, and reusable patterns.

### Training example

```text
Input:
"Marie Curie was a Polish and naturalized-French physicist ..."
```

The model predicts the following tokens:

```text
Target:
"who conducted pioneering research on radioactivity ..."
```

At different positions, the task can be viewed as:

```text
Input:  "Marie Curie was a"
Target: "Polish"

Input:  "Marie Curie was a Polish"
Target: "and"

Input:  "Marie Curie was a Polish and naturalized-French"
Target: "physicist"
```

### Test example

The test set contains non-overlapping text that was not used for training:

```text
Input:
"Albert Einstein developed the theory of ..."

Target:
"relativity"
```

The model performs the same next-token prediction task on both training and test sequences.

### Generalization versus memorization

A model may generalize the fact that a sentence has the form:

```text
[person] scored [number] points in [year] ...
```

But remembering the exact combination in:

```text
John Smith scored 147 points in the
2019 regional bowling championship.
```

is closer to sample-level memorization.

The paper treats information that a capable reference model can already explain as **generalization**. Information that the trained target model knows in addition to the reference model is treated as **unintended memorization**.

---

## 4. The paper’s illustrative comparison

### Mostly generalization

```text
Q: What is 2^100?
A: 1267650600228229401496703205376
```

A capable language model may compute this answer without memorizing the exact training sentence. Therefore, seeing the exact answer in the training set does not automatically prove memorization.

### More sample-specific information

```text
John Smith scored 147 points in the
2019 regional bowling championship.
```

The syntactic pattern may be generalizable, but the exact name, score, and year are difficult to infer from general language knowledge alone. If the target model predicts this particular combination much better than the reference model, the difference is counted as unintended memorization.

---

## 5. Measuring memorization through compression

The paper defines memorization in terms of how much shorter a sample can be described when the model is available.

For example:

```text
Without the model: 1000 bits
With the model:     700 bits
Memorized information: 300 bits
```

Exact Kolmogorov complexity is not computable in practice, so the authors approximate it using model likelihoods or cross-entropy loss.

For real text, the unintended memorization is approximately based on:

```text
max(
    reference-model loss − target-model loss,
    0
)
```

If the target model assigns much higher probability to a particular training sample than the reference model does, that additional compression is interpreted as sample-specific memorization.

---

## 6. Extraction experiment

Extraction tests whether the model can reproduce the continuation of a sequence from a prefix.

### Training example

```text
Full sequence:
"The rare chemical compound was discovered in 1897 by ..."
```

The model receives:

```text
Prompt:
"The rare chemical compound was discovered in"
```

and may generate:

```text
"1897 by ..."
```

The authors evaluate different prefix lengths, such as 8, 16, and 32 tokens.

The same procedure is applied to unseen test sequences:

```text
Test prompt:
"The rare chemical compound was first reported ..."
```

A key conclusion is:

> Successful generation of a training sequence is not by itself proof of unintended memorization.

When the dataset becomes sufficiently large, the extraction rate on training data approaches the extraction rate on test data. In that regime, successful extraction can be explained by generalization rather than by memorization of individual training examples.

---

## 7. Membership-inference experiment

Membership inference asks whether a candidate sequence was included in the training set.

### Input

```text
"John Smith scored 147 points in the
2019 regional bowling championship."
```

### Ground-truth label

- `member = 1`: the sequence was in the training data
- `member = 0`: the sequence came from the held-out test set

### Attack rule

The attack uses the model’s loss on the candidate sequence.

```text
Lower loss  → likely a training member
Higher loss → likely a non-member
```

Example:

| Sequence | Loss | Prediction |
|---|---:|---|
| Training sequence A | 1.2 | member |
| Test sequence B | 3.8 | non-member |

As the dataset becomes larger, the model’s capacity is spread across more examples. The loss gap between training and test samples becomes smaller, making membership inference more difficult.

The paper finds that:

- Larger models can memorize more samples.
- Larger datasets make membership inference harder.
- Once the dataset exceeds model capacity, average-sample membership inference approaches random guessing.
- For modern models trained on very large token-per-parameter ratios, reliable membership inference for an average data point is predicted to be difficult.

---

## 8. Summary of the experimental setups

| Experiment | Training input | Test input | Task | Purpose |
|---|---|---|---|---|
| Random sequences | Random token sequences | New random sequences | Next-token prediction | Measure pure memorization without generalization |
| Real text | FineWeb training sequences | Non-overlapping FineWeb sequences | Next-token prediction | Separate memorization from generalization |
| Extraction | Prefixes of training sequences | Prefixes of training/test sequences | Generate the continuation | Measure reproducibility |
| Membership inference | Candidate sequences | Candidate sequences | Classify train-member vs. non-member | Measure whether training membership can be inferred |

<br/>
# 요약
**한국어 요약**  
1. 연구진은 모델이 데이터를 얼마나 짧게 압축할 수 있는지를 바탕으로, 일반화와 구분되는 ‘비의도적 암기’를 Kolmogorov·Shannon 정보량으로 측정했다.  
2. 무작위 비트열과 실제 텍스트로 다양한 크기의 GPT형 모델을 실험한 결과, 모델은 매개변수당 약 3.5~3.6비트(정밀도에 따라 최대 3.83비트)를 저장하며, 용량이 차면 암기는 포화되고 일반화가 증가했다.  
3. 예를 들어 “2¹⁰⁰의 값”처럼 모델이 일반적으로 학습할 수 있는 지식은 암기로 보지 않지만, 특정 인물의 경기 점수·연도처럼 데이터에만 있는 세부정보는 암기로 분류되며, 데이터가 커질수록 평균 샘플의 멤버십 추론은 어려워졌다.  

**English version**  
1. The authors measure unintended memorization through how much shorter a datapoint can be compressed using the model, using Kolmogorov- and Shannon-information-based quantities to separate memorization from generalization.  
2. Experiments on random bitstrings and real text with GPT-style models of different sizes show a storage capacity of roughly 3.5–3.6 bits per parameter, rising to about 3.83 bits with higher precision; once capacity is filled, memorization plateaus and generalization increases.  
3. For example, computing “the value of 2¹⁰⁰” is treated as generalization, whereas recalling a specific person’s bowling score and year is treated as memorization; as datasets grow, membership inference on an average training example becomes more difficult.

<br/>
# 기타



## 1. 다이어그램·피규어별 결과와 인사이트

### Figure 1 — 무작위 데이터의 비의도적 암기
- 데이터셋이 작을 때는 모델이 거의 모든 정보를 암기한다.
- 데이터셋 크기가 커지면 암기량이 증가하다가 일정 지점에서 plateau에 도달한다.
- 이 plateau가 모델의 **암기 용량(capacity)** 이다.
- 모델이 클수록 더 많은 비트를 저장할 수 있다.

**핵심 인사이트:** 모델은 데이터를 무한히 암기하지 못하며, 파라미터 수에 비례하는 저장 한계가 있다.

---

### Figure 2 — 실제 텍스트에서 암기와 일반화
- 작은 데이터셋에서는 모델이 학습 샘플에 대한 정보를 많이 저장한다.
- 데이터셋이 커지면 모델은 개별 샘플 암기보다 언어의 일반적인 패턴을 학습한다.
- 따라서 어느 지점 이후에는 **비의도적 암기량이 감소**한다.
- 여기서 암기량은 1B 파라미터 oracle/reference 모델과 비교해 측정한다.

**핵심 인사이트:** 실제 텍스트에서는 모델 용량이 차면 샘플별 암기를 줄이고 일반화로 전환한다.

---

### Figure 3 — 데이터 용량과 double descent
- x축은 데이터셋 크기와 모델 용량의 비율이다.
- 데이터셋 용량이 모델 용량에 가까워지는 지점에서 테스트 손실이 악화된다.
- 데이터셋이 모델 용량을 초과하면 모델은 샘플을 개별적으로 저장하지 못하고 정보 공유와 일반화를 시작한다.

**핵심 인사이트:** 논문은 double descent가 데이터셋의 정보량이 모델의 암기 용량을 초과하는 지점에서 발생한다고 해석한다.

---

### Figure 4 — 텍스트 모델의 train/test loss
- 작은 데이터셋에서는 큰 모델이 학습 데이터를 쉽게 외우므로 train loss와 test loss 모두 낮다.
- 데이터셋이 커지면 모델이 모든 샘플을 개별적으로 저장하지 못해 test loss가 다시 증가하는 구간이 나타난다.
- 이후에는 모델이 일반적인 언어 패턴을 학습하면서 일반화가 진행된다.

**핵심 인사이트:** 텍스트에서도 double descent가 나타나며, 그 전환점은 모델의 암기 용량과 관련된다.

---

### Figure 5 — membership inference scaling law
논문은 membership inference F1을 다음과 같은 sigmoid 함수로 근사한다.

\[
F1 \approx \frac{1}{2}
\left(1+c_1\sigma\left(c_2\frac{\text{Capacity}}{|D|}+c_3\right)\right)
\]

- 모델 용량이 크고 데이터셋이 작을수록 membership inference가 쉽다.
- 데이터셋이 커질수록 F1은 0.5, 즉 무작위 추측 수준으로 감소한다.
- 500K~1.5B 파라미터 모델에서 예측값과 실제값이 대체로 1~2% 이내로 일치했다.

**핵심 인사이트:** membership inference 위험은 단순히 모델 크기만이 아니라 **모델 용량 대비 데이터셋 크기**로 예측할 수 있다.

---

### Figure 6 — extraction rate
- 짧은 prefix를 주면 학습 데이터를 모델이 그대로 완성하는 extraction이 많이 발생한다.
- prefix가 길어질수록 extraction은 쉬워진다.
- 그러나 데이터셋이 커질수록 train extraction rate가 감소한다.
- 충분히 큰 데이터셋에서는 train과 test extraction rate가 거의 같아진다.

**핵심 인사이트:** 큰 데이터셋에서 학습 데이터가 생성된다는 사실만으로는 개별 샘플의 비의도적 암기를 증명할 수 없다. 일반화만으로도 동일한 출력이 가능하다.

---

### Figure 7 — 텍스트 membership inference
- 같은 모델에서는 학습 데이터가 많아질수록 membership inference F1이 하락한다.
- 큰 모델은 더 많은 데이터를 암기할 수 있어 같은 데이터셋 크기에서는 더 높은 F1을 보인다.
- 작은 데이터셋에서는 F1이 약 0.97까지 올라가지만, 데이터가 충분히 많아지면 0.5에 가까워진다.

**핵심 인사이트:** 모델이 클수록 더 많은 샘플을 구분할 수 있지만, 데이터셋이 모델 용량에 비해 충분히 크면 평균적인 샘플에 대한 membership inference는 어려워진다.

---

### Figure 8 — 무작위 데이터와 텍스트의 압축률 분포
- 무작위 bitstring에서는 train과 test의 압축률 분포가 비교적 뚜렷하게 분리된다.
- 텍스트에서는 train loss가 낮지만 분포가 넓고 train/test가 많이 겹친다.
- 이 겹침 때문에 텍스트의 membership inference가 더 어렵다.

**핵심 인사이트:** 무작위 데이터는 암기 여부를 명확히 드러내지만, 텍스트는 원래 압축 가능한 구조와 일반화가 존재하므로 암기와 일반화를 구분하기 어렵다.

---

### Figure 9 — TF-IDF와 비의도적 암기
- TF-IDF가 높은, 즉 희귀한 단어가 많은 문서일수록 더 많이 암기되는 경향이 있다.
- 특히 일본어·중국어·히브리어 등 영어 데이터에서 드문 토큰을 포함한 문서가 강하게 암기되었다.
- 데이터셋은 deduplication되었으므로 단순 중복만으로 설명되지 않는다.

**핵심 인사이트:** 모델은 일반적인 문장보다 희귀하고 특이한 토큰 조합을 더 강하게 암기할 수 있다.

---

### Figure 10 — 학습 진행에 따른 암기량
- 학습 초기에 암기량이 빠르게 증가한다.
- 이후 일정 수준에서 안정화되며, 더 오래 학습해도 크게 증가하지 않는다.
- 약 6.86M 파라미터 모델의 예시에서 용량은 약 23.9MB, 즉 약 190M bits 수준으로 제시된다.

**핵심 인사이트:** 암기량은 학습 시간이 무한히 늘어난다고 계속 증가하지 않으며, 모델의 실질적인 저장 용량에 의해 제한된다.

---

### Figure 11 — 파라미터 수와 capacity
- GPT 계열 모델의 capacity는 파라미터 수와 거의 선형 관계를 보인다.
- bfloat16 실험에서 평균적으로 약 **3.5~3.6 bits/parameter**, 전체 추정치는 약 **3.64 bits/parameter**이다.
- 이는 파라미터 하나가 물리적으로 가진 비트 수보다 훨씬 작은 실효 저장량이다.

**핵심 인사이트:** 파라미터 수는 단순한 모델 크기가 아니라 암기 가능한 정보량의 유용한 예측 변수다.

---

### Figure 12 — sequence length 변화
- 데이터 샘플 수를 고정하고 sequence length를 늘리면 샘플 하나의 정보량이 증가한다.
- 총 암기량은 대략 다음과 같이 예측된다.

\[
\text{Memorization} \approx \min(\text{Capacity}, H(X))
\]

- 데이터의 총 정보량이 모델 용량보다 작으면 대부분 암기한다.
- 정보량이 용량을 넘으면 암기량은 capacity 부근에서 포화된다.

---

### Figure 13 — vocabulary size 변화
- vocabulary가 커질수록 각 토큰의 정보량이 늘어난다.
- 동시에 embedding 등 모델 파라미터 수도 증가할 수 있다.
- 따라서 이 실험에서는 capacity plateau가 sequence length 실험만큼 뚜렷하지 않다.
- 그럼에도 예측식의 평균 오차는 약 1.8%로 작았다.

**핵심 인사이트:** 모델 용량뿐 아니라 데이터 자체의 entropy도 실제 암기량을 결정한다.

---

### Figure 14 — synthetic 데이터의 train/test loss
- 무작위 데이터에서는 일반화할 구조가 거의 없기 때문에 test loss가 크게 개선되지 않는다.
- 모델은 주로 학습 데이터를 암기하는 방향으로 동작한다.
- 데이터셋 크기가 모델 capacity를 넘으면 train loss와 암기 성능이 더 이상 충분히 개선되지 않는다.

---

### Figure 15 — synthetic 데이터 membership inference
- 작은 데이터셋에서는 membership inference가 매우 잘 작동한다.
- 데이터셋이 커질수록 성능이 감소한다.
- 다만 실험 범위에서는 F1이 약 0.54 아래로 완전히 내려가지는 않았다.

**핵심 인사이트:** 무작위 데이터에서는 train/test 압축률 차이가 비교적 명확해 텍스트보다 membership inference가 쉽다.

---

### Figure 16 — membership inference scaling law 적합
- empirical 결과와 sigmoid scaling law가 전반적으로 잘 맞는다.
- 데이터셋 크기 대비 capacity 비율이 낮을 때 F1이 높고, 비율이 높아질수록 F1이 0.5에 수렴한다.
- 따라서 대규모 모델이라도 훈련 데이터가 충분히 많으면 평균 샘플에 대한 membership inference가 어려워진다.

---

### Figure 17 — reference 모델 크기 ablation
- reference 모델이 작을수록 측정되는 \(MEM_U\)가 커진다.
- 작은 reference 모델은 텍스트를 덜 잘 예측하므로 target 모델과의 loss 차이가 커지기 때문이다.
- reference 모델 크기가 달라도 데이터셋 크기에 따른 곡선의 질적 형태는 대체로 유지된다.

**핵심 인사이트:** 실제 텍스트의 절대적인 암기량은 reference 모델 선택에 의존한다. 따라서 reference는 고정하고 상대적인 비교를 하는 것이 중요하다.

---

### Figure 18 — reference 모델 크기에 대한 민감도
- 모든 target 모델과 데이터셋 크기에서 reference 모델이 커질수록 \(MEM_U\)는 감소한다.
- 암기량이 큰 조건일수록 reference 크기에 따른 변화 폭도 더 크다.
- 다만 reference가 target과 비슷한 크기가 되면 일반화까지 암기로 잘못 계산할 위험이 있다.

---

## 2. 테이블별 결과와 인사이트

### Table 1 — precision과 모델 구조별 capacity
- bfloat16 평균: 약 **3.51 bits/parameter**
- fp32 평균: 약 **3.83 bits/parameter**
- precision을 2배로 높여도 capacity는 약 9% 정도만 증가한다.

**인사이트:** 파라미터의 저장 비트 수가 늘어난다고 암기 용량이 비례해서 증가하지 않는다. 추가 정밀도의 상당 부분은 raw storage에 사용되지 않는다.

---

### Table 2 — scaling law 검증
GPT-2 XL과 GPT-2 Medium에 대해 목표 F1 0.55, 0.75, 0.95를 만들 것으로 예측한 데이터셋 크기를 사용했다.

- 예측 F1과 실제 F1이 대체로 근접한다.
- 예측 오차는 약 1~1.5 percentage points 수준이다.

**인사이트:** 작은 실험 모델에서 얻은 scaling law가 125M 및 1.5B 파라미터 모델에도 어느 정도 외삽된다.

---

### Table 3 — sequence length별 capacity
- sequence length를 바꾸어도 capacity 예측은 실제 측정값과 잘 맞는다.
- 평균 오차는 약 **1.7%**다.

**인사이트:** capacity는 단순히 샘플 개수가 아니라 전체 데이터 정보량과 비교해야 한다.

---

### Table 4 — vocabulary size별 capacity
- vocabulary size가 증가하면 데이터 entropy와 embedding 파라미터가 모두 변한다.
- 예측 오차는 평균 약 **1.8%**다.

**인사이트:** 모델 capacity와 데이터 entropy를 함께 고려하면 다양한 입력 설정에서도 암기량을 예측할 수 있다.

---

### Table 5 — 가장 많이 암기된 샘플
- 높은 TF-IDF를 가진 샘플들이 대부분 강하게 암기되었다.
- 비영어권 토큰이나 매우 희귀한 토큰이 포함된 경우가 많았다.
- 어떤 샘플은 단 하나의 토큰만으로 전체 sequence를 재생성할 수 있었다.

**인사이트:** deduplication 이후에도 희귀성 자체가 암기를 유발하는 중요한 요인이다.

---

## 3. 어펜딕스의 주요 결과와 인사이트

### A.1~A.3 — 기존 암기 정의와의 비교
논문은 기존 방법의 한계를 정리한다.

- **Extraction 기반 정의:** 출력이 가능하다고 해서 암기를 의미하지는 않는다. 일반화나 쉬운 패턴 때문일 수 있다.
- **Membership inference:** 모집단 수준의 프라이버시 위험은 측정하지만 개별 샘플의 암기량을 직접 측정하기 어렵다.
- **Stability/DP 기반 정의:** 학습 알고리즘에 의존하며 최종 모델과 샘플만으로 평가하기 어렵다.
- **기존 Kolmogorov 기반 정의:** 압축을 사용하지만 일반화와 비의도적 암기를 분리하지 않는다.

**핵심 기여:** 이 논문은 reference 모델을 사용해 “모델이 원래 데이터 분포에서 배울 수 있는 정보”와 “특정 데이터셋에서 추가로 저장한 정보”를 분리한다.

---

### A.4 — language model compression
- 모델 likelihood를 이용한 arithmetic coding을 기본 압축 방법으로 사용한다.
- training sample에서는 greedy decoding이 가능한 구간이 있으므로, \(k\)와 temperature를 동적으로 바꾸는 ensemble compression도 제안한다.
- 더 좋은 압축기는 \(H_K(x|\hat\theta)\)를 더 정확히 추정하므로 암기량 측정도 더 정밀해진다.

**인사이트:** 논문의 암기 측정은 특정 decoding 방식 자체가 아니라, 모델이 샘플을 얼마나 짧게 설명할 수 있는가에 기반한다.

---

### A.5 — capacity 추정의 신뢰성
- sequence length와 vocabulary size를 바꾸어도

\[
\text{memorization}\approx \min(\text{capacity}, H(X))
\]

관계가 잘 유지된다.
- 이는 약 3.64 bits/parameter라는 선형 capacity 추정이 특정 실험 설정에만 맞춘 결과가 아님을 뒷받침한다.

---

### A.6 — 추가 텍스트 암기 결과
- 텍스트 데이터에서도 총 암기량은 모델 capacity 부근에서 plateau한다.
- 데이터가 커지면 모델은 암기 정보를 더 많은 샘플에 나누어 저장한다.
- 데이터셋이 capacity보다 작을 때는 여러 모델이 비슷하게 동작하지만, 데이터가 커지면 작은 모델부터 일반화와 암기 사이의 차이가 나타난다.

---

### A.7 — 고암기 샘플의 수동 분석
- 가장 많이 암기된 샘플에는 희귀 단어, 외국어 토큰, 비정상적인 문자열 조합이 많았다.
- 높은 TF-IDF와 비의도적 암기 사이에 강한 상관관계가 관찰되었다.

---

### A.8 — scaling law 적합도
- sigmoid 형태의 membership inference 법칙은 완벽하지는 않지만 실험값을 대체로 1~2% 이내로 설명한다.
- 데이터셋 크기가 무한히 커지면 membership inference와 extraction 모두 무작위 추측 수준으로 수렴한다.

---

### A.9~A.12 — 이론적 증명
- 전체 데이터셋의 비의도적 암기량은 각 샘플의 암기량 합보다 크거나 같다.
- 동시에 모델 자체의 entropy보다 클 수 없다.

\[
\sum_i MEM_U(X_i)\le MEM_U(X)\le H(\hat\Theta)
\]

- uniform random 데이터에서는 일반화가 0이므로, 측정된 암기량 전체를 비의도적 암기로 해석할 수 있다.

**인사이트:** synthetic random 데이터는 모델 capacity를 직접 측정하기 위한 통제된 실험 환경이다.

---

### A.13 — reference 모델 ablation
- reference 모델이 작을수록 절대적인 \(MEM_U\)가 과대평가된다.
- 그러나 동일한 reference를 사용하면 모델 간 상대 비교는 여전히 유효하다.
- reference 모델이 target 모델과 비슷해지면 일반화와 암기의 구분이 약해진다.

---

### A.14 — 한계
논문의 결과는 다음 조건에 의존한다.

- GPT 계열 Transformer
- 특정 optimizer와 학습 단계
- bfloat16/fp32 정밀도
- deduplicated FineWeb 또는 synthetic random 데이터
- likelihood 기반 압축 측정
- 고정된 reference 모델

따라서 3.6 bits/parameter가 모든 LLM, 모든 architecture, 모든 학습 방식에 적용되는 보편적인 상수라고 해석해서는 안 된다. 저자들도 이를 모델·데이터·학습 환경에 대한 경험적 하한 또는 추정치로 본다.

---

## 전체 결론

1. GPT 스타일 모델은 대략 **3.5~3.6 bits/parameter**의 비의도적 암기 용량을 보였다. fp32에서는 약 3.83 bits/parameter까지 증가했다.  
2. 데이터셋이 모델 capacity보다 작으면 샘플을 많이 암기한다.  
3. 데이터셋이 capacity를 넘으면 개별 샘플 암기는 감소하고 일반화가 증가한다.  
4. extraction은 일반화와 암기를 혼동할 수 있으므로 단독 증거로 사용하기 어렵다.  
5. membership inference는 모델 크기보다 **capacity 대비 데이터셋 크기**에 크게 좌우된다.  
6. 실제 LLM처럼 tokens-per-parameter 비율이 매우 큰 모델에서는 평균적인 샘플에 대한 membership inference가 거의 불가능할 수 있다.  

---




## 1. Figures and diagrams

### Figure 1 — Unintended memorization of random data
- Small datasets are almost completely memorized.
- Memorization grows with dataset size and then reaches a plateau.
- The plateau represents the model’s memorization capacity.
- Larger models store more information.

**Key insight:** A model cannot memorize unlimited information; its memorization capacity is bounded and roughly scales with parameter count.

---

### Figure 2 — Memorization and generalization on text
- On small datasets, models store substantial sample-specific information.
- As the dataset grows, models increasingly learn reusable language patterns.
- After capacity is filled, unintended memorization decreases while generalization increases.
- The measurement is made relative to a large oracle/reference model.

**Key insight:** On real text, models gradually replace sample-level memorization with generalization.

---

### Figure 3 — Dataset capacity and double descent
- Test performance deteriorates when dataset information approaches model capacity.
- Once dataset capacity exceeds model capacity, the model can no longer store every sample independently.
- It must share information across examples and generalize.

**Key insight:** The paper interprets double descent as beginning when the information content of the dataset exceeds the model’s memorization capacity.

---

### Figure 4 — Train and test loss on text
- On small datasets, larger models can fit and memorize the training data.
- As the dataset becomes larger, test loss can worsen near the capacity boundary.
- Further scaling leads to more generalization.

**Key insight:** The transition between memorization and generalization is closely related to the model-capacity boundary.

---

### Figure 5 — Membership-inference scaling law
The paper models membership-inference F1 with a sigmoid function of model capacity divided by dataset size.

- Large models and small datasets yield strong membership inference.
- Increasing dataset size drives F1 toward 0.5, or random guessing.
- The law predicts experiments on models from roughly 500K to 1.5B parameters with about 1–2% error.

**Key insight:** Membership-inference risk is governed not only by model size, but by the ratio between model capacity and dataset size.

---

### Figure 6 — Extraction rate
- Shorter prefixes make exact extraction easier.
- Extraction rates decrease as the training set grows.
- With sufficiently large datasets, train and test extraction rates become nearly identical.

**Key insight:** A training example being extractable is not necessarily evidence of unintended memorization; generalization can produce the same behavior.

---

### Figure 7 — Membership inference on text
- For a fixed model, membership inference becomes harder as the dataset grows.
- Larger models perform better at the same dataset size.
- F1 can be close to 0.97 on very small datasets but approaches 0.5 at large scales.

**Key insight:** Larger models can memorize more, but membership inference on an average example becomes difficult when the dataset is sufficiently large.

---

### Figure 8 — Compression distributions
- Random bitstrings show a clearer separation between train and test compression rates.
- Text has broader and more overlapping train/test loss distributions.
- This overlap makes membership inference on text more difficult.

**Key insight:** Text contains reusable structure and generalizable patterns, which makes memorization harder to isolate than in random data.

---

### Figure 9 — TF-IDF and memorization
- Samples with high TF-IDF—many rare words—tend to be memorized more strongly.
- Highly memorized examples often contain Japanese, Chinese, Hebrew, or other rare tokens.
- This effect remains even after deduplication.

**Key insight:** Rare and unusual token combinations are particularly vulnerable to memorization.

---

### Figure 10 — Memorization during training
- Memorization increases rapidly early in training.
- It eventually stabilizes near a capacity limit.
- Additional training steps do not substantially increase total memorization.

**Key insight:** Training longer does not provide unlimited storage; memorization is ultimately constrained by model capacity.

---

### Figure 11 — Capacity versus parameter count
- Capacity scales approximately linearly with parameter count.
- The main estimate is about **3.5–3.6 bits per parameter**, with a fitted value around 3.64 bpp in half precision.
- This is much smaller than the raw number of bits used to represent the parameters.

**Key insight:** Parameter count is a useful predictor of effective memorization capacity, but raw numerical precision does not translate directly into storage capacity.

---

### Figures 12 and 13 — Sequence length and vocabulary size
- Increasing sequence length increases the entropy of each example.
- Increasing vocabulary size also increases the information content of tokens.
- Memorization follows approximately:

\[
\text{Memorization}\approx\min(\text{Capacity},H(X)).
\]

- Predictions remain accurate, with roughly 1.7% error for sequence length and 1.8% for vocabulary size.

**Key insight:** Total memorization is limited by the smaller of the data’s information content and the model’s capacity.

---

### Figure 14 — Synthetic train/test loss
- Random data contains almost no structure to generalize.
- Models mainly memorize the training set.
- Once the dataset exceeds capacity, additional data cannot be individually stored.

---

### Figure 15 — Membership inference on synthetic data
- Membership inference is strong on small random datasets.
- Performance decreases as the dataset grows.
- In the tested range, F1 remains slightly above 0.5.

**Key insight:** Random data gives a clearer train/test separation than text, making membership inference easier.

---

### Figure 16 — Scaling-law fit
- The sigmoid law closely follows the empirical results.
- F1 is high when dataset size is small relative to capacity.
- It approaches 0.5 as the ratio increases.

---

### Figures 17 and 18 — Reference-model ablation
- Smaller reference models produce larger absolute \(MEM_U\) values.
- A weaker reference assigns higher loss to the text, increasing the apparent target/reference gap.
- The qualitative trends remain similar across reference models.
- However, if the reference becomes close in size to the target, generalization may be incorrectly attributed to memorization.

**Key insight:** Absolute unintended-memorization values depend on the reference model, so the reference should be fixed for meaningful comparisons.

---

## 2. Tables

### Table 1 — Precision and architecture
- bfloat16: about **3.51 bits/parameter**
- fp32: about **3.83 bits/parameter**
- Doubling numerical precision produces only a modest capacity increase.

**Insight:** Extra numerical precision is not used proportionally for raw memorization.

---

### Table 2 — Validation of the scaling law
- The authors select dataset sizes predicted to produce F1 values of 0.55, 0.75, and 0.95.
- Observed values are close, with roughly 1–1.5 percentage points of error.

**Insight:** The scaling law extrapolates reasonably well from smaller models to GPT-2 Medium and GPT-2 XL.

---

### Table 3 — Sequence-length scaling
- Capacity estimates remain consistent across sequence lengths.
- Average prediction error is about 1.7%.

---

### Table 4 — Vocabulary-size scaling
- Increasing vocabulary changes both data entropy and embedding-related parameter count.
- Prediction error is about 1.8%.

**Insight:** Capacity and data entropy together explain memorization across different input configurations.

---

### Table 5 — Highly memorized examples
- The most memorized samples have high TF-IDF and rare tokens.
- Many contain non-English or unusual token sequences.
- Some can be regenerated from extremely short prompts.

**Insight:** Rarity, rather than duplication alone, is an important driver of strong memorization.

---

## 3. Appendix results and insights

### Appendices A.1–A.3 — Comparison with prior definitions
The paper argues that:

- Extraction-based definitions can confuse generalization with memorization.
- Membership inference is mainly a population-level measure, not a precise sample-level quantity.
- Stability-based and differential-privacy definitions depend on the training algorithm.
- Earlier compression-based definitions do not explicitly separate generalization from unintended memorization.

**Main contribution:** The reference-model framework separates information learned from the true data distribution from information stored about a specific dataset.

---

### Appendix A.4 — Compression with language models
- Model likelihoods provide a practical approximation to Kolmogorov complexity.
- Arithmetic coding is extended with adaptive decoding settings.
- Better compression gives a tighter estimate of \(H_K(x|\hat\theta)\), and therefore a more accurate estimate of memorization.

---

### Appendix A.5 — Reliability of capacity estimates
Changing sequence length and vocabulary size supports the approximation:

\[
\text{Memorization}\approx\min(\text{Capacity},H(X)).
\]

This supports the robustness of the approximately 3.64 bits-per-parameter estimate.

---

### Appendix A.6 — Additional text results
- Total memorization on text also plateaus near model capacity.
- As the dataset grows, the model distributes its limited memorization across more examples.
- Larger datasets cause a shift from sample-level memorization toward generalization.

---

### Appendix A.7 — Manual analysis
Highly memorized samples tend to contain rare words, foreign-language tokens, or unusual strings. TF-IDF is strongly correlated with unintended memorization.

---

### Appendix A.8 — Scaling-law fit
The sigmoid membership-inference law is not exact, but it predicts empirical performance within approximately 1–2%. As dataset size tends to infinity, both extraction and membership inference approach random performance.

---

### Appendices A.9–A.12 — Theoretical results
The paper proves:

\[
\sum_i MEM_U(X_i)\le MEM_U(X)\le H(\hat\Theta).
\]

Thus, total unintended memorization is at least the sum of per-example memorization but cannot exceed the information content of the trained model.

For uniformly random data, generalization is exactly zero, so all measured information can be interpreted as unintended memorization.

---

### Appendix A.13 — Reference-model sensitivity
- Smaller references inflate absolute \(MEM_U\).
- Relative comparisons are still meaningful when the reference is fixed.
- A reference model close to the target may incorrectly classify generalization as memorization.

---

### Appendix A.14 — Limitations
The numerical estimates depend on:

- GPT-style Transformer architecture
- Training duration and optimizer
- bfloat16 or fp32 precision
- Synthetic random data and deduplicated FineWeb
- Likelihood-based compression
- Choice of reference model

Therefore, **3.6 bits per parameter should not be treated as a universal constant for every LLM or architecture**. It is an empirical estimate under the studied conditions.

---

## Overall conclusion

1. GPT-style models show an effective unintended-memorization capacity of roughly **3.5–3.6 bits per parameter**.  
2. Small datasets are memorized heavily; large datasets force a shift toward generalization.  
3. Extraction alone cannot reliably distinguish memorization from generalization.  
4. Membership inference is mainly determined by the ratio of model capacity to dataset size.  
5. For modern models trained on very large datasets, membership inference on an average training example may be close to random guessing.

<br/>
# refer format:


### BibTeX

```bibtex
@inproceedings{morris2026language,
  author    = {Morris, John X. and Sitawarin, Chawin and Guo, Chuan and
               Kokhlikyan, Narine and Suh, G. Edward and Rush, Alexander M. and
               Chaudhuri, Kamalika and Mahloujifar, Saeed},
  title     = {How Much Can Language Models Memorize?},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  publisher = {PMLR},
  address   = {Seoul, South Korea}
}
```

### 시카고 스타일   
Morris, John X., Chawin Sitawarin, Chuan Guo, Narine Kokhlikyan, G. Edward Suh, Alexander M. Rush, Kamalika Chaudhuri, and Saeed Mahloujifar. “How Much Can Language Models Memorize?” In *Proceedings of the 43rd International Conference on Machine Learning*. Vol. 306 of *Proceedings of Machine Learning Research*. Seoul, South Korea: PMLR, 2026.





