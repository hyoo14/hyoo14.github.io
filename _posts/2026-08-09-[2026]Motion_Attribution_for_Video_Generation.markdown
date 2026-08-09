---
layout: post
title:  "[2026]Motion Attribution for Video Generation"
date:   2026-08-09 17:43:01 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: MOTIVE는 AllTracker의 optical flow로 움직이는 영역을 찾아 loss에 가중하고, 고정된 timestep의 motion-weighted gradient를 Fastfood로 투영해 학습 클립과 생성 영상의 motion 영향도를 비교한다.  
(Fastfood projection은 random features를 싸고 빠르게 만들어준다고 이름붙인것)  
이 연구의 핵심은 단순히 “움직임이 많은 영상”을 고르는 것이 아니라, **특정한 움직임을 모델이 학습하고 생성하는 데 실제로 도움이 되는 영상**을 찾아내는 것  


짧은 요약(Abstract) :


이 논문은 **비디오 생성 모델이 어떤 학습 영상으로부터 움직임을 배우는지 분석하는 방법**을 제안합니다. 기존의 데이터 기여도 분석은 주로 이미지의 객체나 배경 같은 **정적인 시각 정보**에 초점을 맞추기 때문에, 비디오의 핵심인 움직임과 시간적 일관성을 제대로 설명하기 어려웠습니다.

저자들은 **MOTIVE**라는 그래디언트 기반 분석 프레임워크를 개발했습니다. 이 방법은 광학 흐름(optical flow)을 이용해 영상에서 움직이는 영역을 찾고, 해당 영역의 손실에 더 큰 가중치를 주어 **움직임에 특화된 데이터 영향도**를 계산합니다. 또한 긴 영상이 무조건 더 큰 영향력을 갖는 문제를 보정하고, 그래디언트 투영과 단일 시점 샘플링을 사용해 대규모 비디오 데이터셋과 생성 모델에서도 효율적으로 동작하도록 설계했습니다.

MOTIVE로 움직임에 큰 영향을 주는 학습 영상을 선별해 파인튜닝하면, 전체 데이터의 약 **10%만 사용해도** 비디오의 움직임이 더 부드럽고 자연스러워지며, 물리적으로 그럴듯한 동작을 생성할 수 있었습니다. VBench의 동적 정도(dynamic degree)와 움직임 부드러움(motion smoothness) 지표에서도 성능이 향상되었고, 사람 평가에서는 사전학습된 기본 모델과 비교해 **74.1%의 선호 승률**을 기록했습니다.

즉, 이 연구의 핵심은 단순히 “움직임이 많은 영상”을 고르는 것이 아니라, **특정한 움직임을 모델이 학습하고 생성하는 데 실제로 도움이 되는 영상**을 찾아내는 것입니다.

---



This paper introduces **MOTIVE**, a motion-centric data attribution framework for video generation models. Its goal is to identify which training clips influence the motion and temporal dynamics produced by a model.

Existing attribution methods mainly focus on static visual content, such as objects, textures, and backgrounds. They therefore struggle to explain motion-related properties, including trajectories, deformations, interactions, and temporal consistency.

MOTIVE addresses this issue by using optical flow to locate dynamic regions in each video. It assigns larger weights to these regions when computing loss gradients, allowing the method to measure the influence of training clips specifically on motion rather than appearance. The framework also corrects biases caused by different video lengths and uses efficient gradient sampling and projection techniques to scale to large video datasets and models.

By selecting high-influence clips for fine-tuning, MOTIVE improves motion smoothness and physical plausibility while using only about **10% of the training data**. It achieves stronger dynamic-degree scores on VBench and obtains a **74.1% human preference win rate** over the pretrained base model.

The main idea is not simply to select videos with large amounts of motion, but to find the clips that genuinely help the model learn and generate particular motion patterns.


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



### 1. 메서드의 핵심 목적

**MOTIVE**는 비디오 생성 모델이 특정한 움직임을 생성하는 데 어떤 학습 비디오가 영향을 주었는지를 추적하는 **모션 중심 데이터 어트리뷰션(data attribution) 프레임워크**입니다.

기존의 데이터 어트리뷰션은 이미지나 비디오 전체의 픽셀·외형 정보에 주로 반응하기 때문에, 비슷한 배경이나 물체를 포함한 영상이 높은 영향력을 가질 수 있습니다. MOTIVE는 optical flow 기반의 모션 마스크를 사용하여 **정적인 외형보다 움직이는 영역에 집중**합니다.

중요하게도 MOTIVE는 새로운 비디오 생성 아키텍처가 아니라, 기존 diffusion/flow-matching 비디오 생성 모델에 적용할 수 있는 **gradient 기반 분석 및 데이터 선택 방법**입니다.

---

### 2. 사용 모델과 아키텍처

실험에서는 다음 비디오 생성 모델에 적용했습니다.

- **Wan2.1-T2V-1.3B**
- **Wan2.2-TI2V-5B**
- 추가 실험: **LTX-2B**

모델 구조 자체를 새로 설계하지 않고, 기존 모델의 학습 손실에 대해 gradient를 계산합니다.

실제 fine-tuning에서는 다음과 같이 구성했습니다.

- 학습 대상: **DiT backbone**
- 고정: **T5 text encoder**
- 고정: **VAE**
- 입력 해상도: **480×832**
- 학습률: **1×10⁻⁵**
- optimizer: **AdamW**
- 선택된 데이터: 전체 학습 데이터의 약 **10%**

MOTIVE는 diffusion loss뿐 아니라 flow-matching loss에도 적용 가능한 모델 독립적인 방식입니다.

---

### 3. 학습 및 질의 데이터

#### Fine-tuning 데이터

다음 대규모 비디오 데이터셋을 사용했습니다.

- **VIDGEN-1M**
- **4DNeX-10M**

실험에서는 각 데이터셋에서 약 **10,000개 비디오**를 사용했습니다.

#### Query 비디오

어떤 움직임에 영향을 주는 데이터를 찾을지 정의하기 위해 query 비디오를 사용합니다.

10개의 모션 카테고리를 구성했습니다.

- compress
- bounce
- roll
- explode
- float
- free fall
- slide
- spin
- stretch
- swing

각 카테고리마다 5개씩, 총 **50개의 query 비디오**를 사용했습니다. 이 query들은 통제된 움직임을 만들기 위해 **Veo-3로 생성한 뒤 수작업으로 선별**했습니다. Query 데이터는 학습 데이터가 아니라, 원하는 움직임의 기준으로만 사용됩니다.

---

### 4. MOTIVE의 주요 처리 과정

#### Step 1. 비디오를 VAE latent로 변환

입력 비디오를 VAE encoder로 변환합니다.

\[
h = E(v)
\]

모델의 gradient는 픽셀 공간이 아니라 비디오 latent 공간에서 계산되므로, 모션 정보도 latent 해상도에 맞춰 변환합니다.

---

#### Step 2. 움직임 검출

**AllTracker**를 이용해 각 프레임 사이의 픽셀 이동을 추출합니다.

각 위치에서의 displacement vector는 다음과 같습니다.

\[
D_f(h,w)=(d_w,d_h)
\]

이 벡터의 크기

\[
M_f(h,w)=\|D_f(h,w)\|_2
\]

를 해당 위치의 모션 크기로 사용합니다.

즉, optical flow가 큰 영역은 많이 움직이는 영역이고, flow가 작은 영역은 정적인 배경에 가깝습니다.

---

#### Step 3. 모션 마스크 생성

모션 크기를 전체 프레임과 공간 위치에 대해 min-max 정규화하여 \([0,1]\) 범위의 가중치를 만듭니다.

\[
W(f,h,w)=
\frac{M_f(h,w)-M_{\min}}
{M_{\max}-M_{\min}+\zeta}
\]

그 후 이 마스크를 VAE latent 해상도로 bilinear downsampling합니다.

- 움직임이 큰 영역: 높은 가중치
- 정적인 영역: 낮은 가중치

---

#### Step 4. 모션 가중 손실 계산

각 latent 위치의 diffusion 예측 오차를 계산한 뒤, 모션 마스크를 곱합니다.

\[
L_{\text{mot}}
=
\frac{1}{F}
\sum_{f,\tilde h,\tilde w}
\tilde W(f,\tilde h,\tilde w)
\tilde L(f,\tilde h,\tilde w)
\]

따라서 일반적인 비디오 loss와 달리, MOTIVE의 loss는 **움직임이 있는 부분의 예측 오차를 더 중요하게 반영**합니다.

이 손실을 모델 파라미터에 대해 미분하여 모션 gradient를 얻습니다.

\[
g_{\text{mot}}=\nabla_\theta L_{\text{mot}}
\]

이 gradient는 해당 비디오가 모델의 모션 생성 능력과 어떤 방향으로 연결되어 있는지를 나타냅니다.

---

### 5. 계산량을 줄이는 특별한 기법

#### 고정 timestep과 공통 noise

모든 학습 비디오와 query 비디오에 대해 동일한 timestep \(t_{\text{fix}}\)와 noise \(\epsilon_{\text{fix}}\)를 사용합니다.

이를 통해:

- 비디오 간 gradient 비교가 안정적이고
- 여러 timestep과 noise를 반복 샘플링할 필요가 없으며
- 계산 비용을 크게 줄일 수 있습니다.

실험에서는 denoising trajectory의 중간 지점에 해당하는 **\(t=751\)**을 사용했습니다.

---

#### 프레임 길이 보정

긴 비디오는 단순히 프레임 수가 많기 때문에 gradient 크기가 커질 수 있습니다. 이를 방지하기 위해 gradient를 프레임 수 \(F\)로 나눕니다.

\[
g_{\text{mot}} \leftarrow \frac{1}{F}g_{\text{mot}}
\]

또한 실제 실험에서는 모든 비디오를 **81 프레임, 16 fps**로 표준화했습니다.

이 보정이 없으면 모션의 질보다 비디오 길이가 영향력 순위를 결정하는 문제가 발생합니다.

---

#### Fastfood projection

모델의 파라미터가 약 14억 개이므로 전체 gradient를 저장하는 것은 비현실적입니다. MOTIVE는 gradient를 **Fastfood 기반 Johnson–Lindenstrauss random projection**으로 저차원 벡터로 압축합니다.

- 원래 gradient 차원: 약 \(1.4\) billion
- 투영 후 차원: \(D'=512\)

그 후 벡터를 정규화합니다.

\[
\tilde g_{\text{mot}}(v)
=
\frac{P g_{\text{mot}}(v)}
{\|P g_{\text{mot}}(v)\|}
\]

이렇게 하면 각 비디오를 512차원 벡터로 저장할 수 있고, 대규모 데이터셋에서도 효율적으로 비교할 수 있습니다.

---

### 6. 영향력 점수 계산

query 비디오 \(\hat v\)와 학습 비디오 \(v_n\)의 모션 gradient를 cosine similarity로 비교합니다.

\[
I_{\text{mot}}(v_n,\hat v)
=
\tilde g_{\text{mot}}(\hat v)^\top
\tilde g_{\text{mot}}(v_n)
\]

- 높은 양의 점수: query의 움직임 생성에 도움이 될 가능성이 큰 비디오
- 낮은 점수 또는 음의 영향: query의 움직임과 잘 맞지 않거나 방해할 가능성이 있는 비디오

여러 query가 있을 경우, 각 query에서 상위 percentile에 든 비디오에 투표를 부여하고, 투표 수가 많은 비디오를 최종 선택합니다. 이를 **majority vote aggregation**이라고 합니다.

---

### 7. 선택된 데이터로 fine-tuning

최종적으로 MOTIVE 점수가 높은 학습 비디오의 상위 10%를 선택하여 fine-tuning합니다.

이 방식은 단순히 움직임이 큰 비디오를 고르는 것이 아닙니다. 실제 분석 결과, 선택된 데이터의 평균 모션 크기는 하위 데이터와 크게 다르지 않았습니다. 즉, MOTIVE는 “많이 움직이는 영상”이 아니라 **목표 움직임을 모델이 학습하는 데 gradient 관점에서 유용한 영상**을 선택합니다.

---

### 8. 성능 결과

Wan 모델에서 MOTIVE는 전체 데이터의 10%만 사용했음에도 다음과 같은 효과를 보였습니다.

- Wan2.1-T2V-1.3B dynamic degree: **47.6%**
- Wan2.2-TI2V-5B dynamic degree: **48.3%**
- 전체 데이터 fine-tuning보다 높은 dynamic degree
- 기본 모델과 비교한 인간 평가에서 **74.1% win rate**
- random selection이나 motion-unaware attribution보다 우수한 temporal dynamics

즉, MOTIVE는 모델 구조를 바꾸기보다 **모션에 실제로 기여하는 데이터를 골라 fine-tuning하는 방식**으로 모션의 일관성, 부드러움, 물리적 타당성을 개선합니다.

---




### 1. Main idea

**MOTIVE** is a motion-centric, gradient-based data attribution framework for video generation models. It identifies which training videos influence a generated video’s temporal dynamics.

Unlike vanilla attribution methods, which often focus on static appearance, objects, or backgrounds, MOTIVE uses optical-flow-based motion masks to emphasize dynamic regions and suppress static content.

MOTIVE is not a new video-generation architecture. It is an attribution and data-selection method that can be applied to existing diffusion or flow-matching video generators.

---

### 2. Models and architecture

The experiments use:

- **Wan2.1-T2V-1.3B**
- **Wan2.2-TI2V-5B**
- Additional experiments on **LTX-2B**

The model architecture is kept unchanged.

During fine-tuning:

- Trainable component: **DiT backbone**
- Frozen components: **T5 text encoder and VAE**
- Resolution: **480×832**
- Learning rate: **1×10⁻⁵**
- Optimizer: **AdamW**
- Selected training data: approximately **10%** of the dataset

The attribution formulation is compatible with both diffusion and flow-matching objectives.

---

### 3. Training and query data

#### Fine-tuning data

The method is evaluated on:

- **VIDGEN-1M**
- **4DNeX-10M**

The experiments use approximately 10,000 videos from each dataset.

#### Query videos

Query videos define the target motion that the system should analyze. The paper uses ten motion categories:

- compress
- bounce
- roll
- explode
- float
- free fall
- slide
- spin
- stretch
- swing

There are five query videos per category, for a total of 50 queries. These videos are generated with Veo-3 and manually screened for clear and physically plausible motion. They are used as attribution targets, not as training data.

---

### 4. MOTIVE pipeline

#### Step 1: Encode the video into latent space

The input video is encoded with the VAE:

\[
h=E(v)
\]

Since gradients are computed in latent space, the motion information is also mapped to the latent resolution.

---

#### Step 2: Detect motion

MOTIVE uses **AllTracker** to estimate pixel displacement between frames:

\[
D_f(h,w)=(d_w,d_h)
\]

The motion magnitude is:

\[
M_f(h,w)=\|D_f(h,w)\|_2
\]

Large values correspond to dynamic regions, while small values usually correspond to static backgrounds.

---

#### Step 3: Build a motion mask

The motion magnitudes are min-max normalized across frames and spatial locations:

\[
W(f,h,w)=
\frac{M_f(h,w)-M_{\min}}
{M_{\max}-M_{\min}+\zeta}
\]

The resulting mask is bilinearly downsampled to the latent resolution.

- High motion: high weight
- Static content: low weight

---

#### Step 4: Compute the motion-weighted loss

The per-location denoising error is weighted by the motion mask:

\[
L_{\text{mot}}
=
\frac{1}{F}
\sum_{f,\tilde h,\tilde w}
\tilde W(f,\tilde h,\tilde w)
\tilde L(f,\tilde h,\tilde w)
\]

The corresponding gradient is:

\[
g_{\text{mot}}=\nabla_\theta L_{\text{mot}}
\]

This gradient represents how the video is related to the model’s ability to represent motion, rather than just its visual appearance.

---

### 5. Efficiency techniques

#### Fixed timestep and shared noise

MOTIVE uses the same fixed timestep \(t_{\text{fix}}\) and noise sample \(\epsilon_{\text{fix}}\) for all train-query comparisons.

This reduces stochastic variance and avoids averaging over many timesteps and noise samples. The experiments use a midpoint timestep, \(t=751\).

---

#### Frame-length normalization

Longer videos naturally produce larger raw gradient magnitudes. MOTIVE corrects this by dividing the gradient by the number of frames:

\[
g_{\text{mot}}\leftarrow \frac{1}{F}g_{\text{mot}}
\]

The experiments also standardize videos to 81 frames at 16 fps. This prevents video length from dominating the attribution ranking.

---

#### Fastfood projection

Full gradients are too large to store for billion-parameter models. MOTIVE therefore projects them into a low-dimensional space using a Fastfood-based Johnson–Lindenstrauss projection.

- Original gradient dimension: approximately 1.4 billion
- Projected dimension: \(D'=512\)

The normalized projected gradient is:

\[
\tilde g_{\text{mot}}(v)
=
\frac{P g_{\text{mot}}(v)}
{\|P g_{\text{mot}}(v)\|}
\]

This makes it practical to store and compare gradients for thousands of videos.

---

### 6. Motion influence score

For a training video \(v_n\) and query video \(\hat v\), MOTIVE computes cosine similarity between their motion gradients:

\[
I_{\text{mot}}(v_n,\hat v)
=
\tilde g_{\text{mot}}(\hat v)^\top
\tilde g_{\text{mot}}(v_n)
\]

- High positive score: likely helpful for generating the target motion
- Low or negative score: potentially mismatched or harmful for that motion

For multiple queries, videos receive votes when they rank above a percentile threshold for each query. The final subset is selected using majority-vote aggregation.

---

### 7. Fine-tuning data selection

The top-scoring 10% of training videos are selected for fine-tuning.

MOTIVE does not simply select videos with the largest overall motion magnitude. Its scores are based on gradient alignment with the target motion. Therefore, a video with moderate motion can be highly influential if its motion patterns are useful for the target behavior.

---

### 8. Main results

Using only 10% of the training data, MOTIVE achieves:

- Dynamic degree of **47.6%** on Wan2.1-T2V-1.3B
- Dynamic degree of **48.3%** on Wan2.2-TI2V-5B
- Better dynamic degree than full-dataset fine-tuning in the reported experiments
- A **74.1% human preference win rate** over the base model
- Better motion quality than random selection and motion-unaware attribution

Overall, MOTIVE improves video motion by identifying and fine-tuning on training clips whose gradients are specifically aligned with the desired temporal dynamics.


<br/>
# Results



### 1. 비교한 생성 모델과 데이터

- **비디오 생성 모델**
  - **Wan2.1-T2V-1.3B**
  - **Wan2.2-TI2V-5B**
  - 추가 실험으로 **LTX-2B**도 평가했다.
- **파인튜닝 데이터**
  - **VIDGEN-1M**
  - **4DNeX-10M**
  - 각 데이터셋에서 실험용으로 10,000개 비디오를 사용했다.
- **쿼리 영상**
  - 특정 움직임의 영향이 어떤 학습 데이터와 관련되는지 확인하기 위한 영상이다.
  - `compress, bounce, roll, explode, float, free fall, slide, spin, stretch, swing`의 10개 움직임을 사용했다.
  - 각 움직임마다 5개씩, 총 **50개 쿼리 영상**을 구성했다.
  - Veo-3로 생성한 뒤 물리적으로 자연스럽고 움직임이 명확한 영상만 선별했다.
- **테스트 데이터**
  - 파인튜닝 후 성능 평가는 쿼리 영상과 다른 외형을 가진 새로운 프롬프트로 수행했다.
  - 즉, 단순히 쿼리 영상을 암기했는지가 아니라, 새로운 객체와 장면에서도 목표 움직임을 생성하는지 평가했다.

---

### 2. 비교 방법

모든 데이터 선택 기반 방법은 전체 파인튜닝 데이터의 **10%**만 사용했다.

- **Base**: 사전학습 모델, 추가 파인튜닝 없음
- **Full fine-tuning**: 전체 데이터로 파인튜닝
- **Random**: 무작위로 10% 선택
- **Motion magnitude**: 평균 움직임 크기가 큰 영상 선택
- **V-JEPA**: V-JEPA 영상 특징을 이용해 움직임 패턴이 대표적인 영상 선택
- **Ours w/o MM**: MOTIVE의 그래디언트 기반 영향도는 사용하지만, motion masking은 사용하지 않음
- **MOTIVE**: 움직이는 영역에 더 큰 가중치를 주는 motion-aware gradient attribution 사용

MOTIVE는 단순히 움직임이 큰 영상을 고르는 것이 아니라, 특정 목표 움직임의 손실을 줄이는 데 실제로 도움이 되는 학습 영상을 선택한다.

---

### 3. 평가 메트릭

VBench의 다음 6개 지표를 사용했다.

- **Subject Consistency**: 프레임 간 객체의 정체성 및 형태 일관성
- **Background Consistency**: 배경의 안정성
- **Motion Smoothness**: 시간에 따른 움직임의 부드러움
- **Dynamic Degree**: 영상에 나타나는 동적 움직임의 정도
- **Aesthetic Quality**: 미적 품질
- **Imaging Quality**: 전반적인 화질

이 중 논문의 핵심 목표는 **Motion Smoothness**와 **Dynamic Degree**이다.

---

### 4. Wan 모델에서의 주요 정량 결과

#### Wan2.1-T2V-1.3B

| 방법 | Motion Smoothness | Dynamic Degree | Aesthetic |
|---|---:|---:|---:|
| Base | 96.3 | 39.6 | 45.3 |
| Full fine-tuning | 96.3 | 42.0 | 45.0 |
| Random | 96.3 | 41.3 | 45.7 |
| Motion magnitude | 95.7 | 40.1 | 45.1 |
| V-JEPA | 95.6 | 41.6 | 44.9 |
| Ours w/o MM | 96.3 | 43.8 | 45.7 |
| **MOTIVE** | **96.3** | **47.6** | **46.0** |

#### Wan2.2-TI2V-5B

| 방법 | Motion Smoothness | Dynamic Degree | Aesthetic |
|---|---:|---:|---:|
| Base | 97.5 | 42.0 | 44.4 |
| Full fine-tuning | 97.5 | 45.3 | 44.8 |
| Random | 97.3 | 41.6 | 44.6 |
| Motion magnitude | 97.4 | 44.9 | 45.0 |
| V-JEPA | 97.3 | 45.6 | 44.9 |
| Ours w/o MM | 97.4 | 43.8 | 45.2 |
| **MOTIVE** | **97.6** | **48.3** | **45.6** |

#### 해석

- MOTIVE는 두 Wan 모델 모두에서 **Dynamic Degree가 가장 높았다.**
  - Wan2.1: **47.6%**
  - Wan2.2: **48.3%**
- 전체 데이터로 파인튜닝한 경우보다도 높은 동적 움직임 점수를 얻었다.
  - Wan2.1: 42.0% → 47.6%
  - Wan2.2: 45.3% → 48.3%
- 전체 데이터의 10%만 사용했음에도, 무작위 선택이나 단순 움직임 크기 선택보다 우수했다.
- Motion Smoothness는 기존 방법과 비슷하거나 소폭 향상되었으며, 동적 움직임을 증가시키면서도 시간적 안정성을 유지했다.
- Subject 및 Background Consistency도 크게 저하되지 않았다.

---

### 5. 추가 모델: LTX-2B

LTX-2B에서도 MOTIVE의 효과가 확인되었다.

| 방법 | Motion Smoothness | Dynamic Degree |
|---|---:|---:|
| Base | 94.6 | 36.5 |
| Full fine-tuning | 94.8 | 38.9 |
| Random | 94.7 | 37.8 |
| Motion magnitude | 95.2 | 39.6 |
| V-JEPA | 95.1 | 40.2 |
| Ours w/o MM | 95.3 | 41.8 |
| **MOTIVE** | **95.5** | **45.1** |

MOTIVE는 LTX-2B에서도 가장 높은 **Motion Smoothness 95.5%**와 **Dynamic Degree 45.1%**를 기록했다. 따라서 특정 Wan 구조에만 의존하는 방법이 아니라, 서로 다른 비디오 생성 모델에도 적용 가능함을 보였다.

---

### 6. 사람 평가 결과

17명의 평가자가 50개 테스트 영상에 대해 생성된 움직임이 더 좋은 영상을 선택했다.

| 비교 대상 | MOTIVE 승리 | 동률 | MOTIVE 패배 |
|---|---:|---:|---:|
| Base 모델 | **74.1%** | 12.3% | 13.6% |
| Random 선택 | **58.9%** | 12.1% | 29.0% |
| Full fine-tuning | **53.1%** | 14.8% | 32.1% |
| MOTIVE w/o motion masking | 46.9% | 20.0% | 33.1% |

핵심적으로, 사람들은 MOTIVE로 파인튜닝한 결과를 기본 모델보다 **74.1%의 경우에 더 선호**했다. 또한 전체 데이터 파인튜닝보다도 53.1%의 승률을 보여, 데이터 양보다 목표 움직임에 적합한 데이터 선택이 중요할 수 있음을 보여준다.

---

### 7. 정성적 분석과 추가 검증

- MOTIVE가 높은 점수를 준 학습 영상에는 목표 움직임과 유사한 **연속적인 궤적, 부드러운 변형, 물리적으로 자연스러운 상호작용**이 포함되어 있었다.
- 낮은 점수의 영상은 정적인 장면, 카메라만 움직이는 영상, 또는 실제 물리와 차이가 큰 만화 스타일 영상이 많았다.
- MOTIVE는 단순히 평균 움직임이 큰 영상을 선택하지 않았다.
  - 상위 10% 영상의 평균 motion magnitude: **3.85**
  - 하위 10% 영상의 평균 motion magnitude: **3.69**
  - 차이는 약 **4.3%**에 불과했다.
- 따라서 MOTIVE의 점수는 “얼마나 많이 움직이는가”보다, “목표 움직임을 모델이 학습하는 데 얼마나 유용한가”를 반영한다.

---

### 8. 주요 분석 결과

- **프레임 길이 보정**
  - 보정하지 않으면 긴 영상이 실제 움직임과 관계없이 높은 영향도를 받았다.
  - 영상 길이와 영향도 사이의 잘못된 상관이 약 **54% 감소**했다.
- **단일 타임스텝**
  - 중간 디노이징 시점인 `t=751` 하나만 사용해도 다중 타임스텝 방식과 약 **66%의 순위 일치도**를 보였다.
  - 계산량을 크게 줄이면서도 실용적인 영향도 순위를 유지했다.
- **그래디언트 투영**
  - 512차원 Fastfood projection을 사용했을 때 전체 그래디언트와의 순위 상관이 **74.7%**였다.
  - 대규모 모델에서 저장 및 계산 비용을 줄이기 위한 절충안으로 사용되었다.

---




## Results Summary

### 1. Models and Data

The method was evaluated on three video generation models:

- **Wan2.1-T2V-1.3B**
- **Wan2.2-TI2V-5B**
- **LTX-2B** as an additional model

The fine-tuning data came from:

- **VIDGEN-1M**
- **4DNeX-10M**

For the experiments, 10,000 videos were used from each dataset.

The query set contained 50 videos covering 10 motion categories:

`compress, bounce, roll, explode, float, free fall, slide, spin, stretch, swing`.

Each category had five query videos. These videos were generated with Veo-3 and manually screened for clear and physically plausible motion. Evaluation used new prompts and different visual appearances, so the models were tested on motion generalization rather than memorization.

---

### 2. Compared Methods

All data-selection methods used only **10% of the fine-tuning data**.

- **Base**: pretrained model without fine-tuning
- **Full fine-tuning**: fine-tuning on the complete dataset
- **Random**: random 10% subset
- **Motion magnitude**: videos with the largest average motion
- **V-JEPA**: representative videos selected using V-JEPA features
- **Ours w/o MM**: gradient-based attribution without motion masking
- **MOTIVE**: motion-aware gradient attribution using motion-weighted loss masks

Unlike motion-magnitude selection, MOTIVE does not simply choose videos that move the most. It selects videos whose gradients are most aligned with the target motion and therefore most useful for improving that motion.

---

### 3. Evaluation Metrics

The paper uses six VBench metrics:

- Subject Consistency
- Background Consistency
- Motion Smoothness
- Dynamic Degree
- Aesthetic Quality
- Imaging Quality

The primary targets are **Motion Smoothness** and **Dynamic Degree**.

---

### 4. Main Results on Wan Models

For **Wan2.1-T2V-1.3B**, MOTIVE achieved:

- **Motion Smoothness: 96.3%**
- **Dynamic Degree: 47.6%**
- **Aesthetic Quality: 46.0%**

The Dynamic Degree scores of the main baselines were:

- Base: 39.6%
- Full fine-tuning: 42.0%
- Random: 41.3%
- Motion magnitude: 40.1%
- V-JEPA: 41.6%
- Ours w/o MM: 43.8%
- **MOTIVE: 47.6%**

For **Wan2.2-TI2V-5B**, MOTIVE achieved:

- **Motion Smoothness: 97.6%**
- **Dynamic Degree: 48.3%**
- **Aesthetic Quality: 45.6%**

The corresponding Dynamic Degree scores were:

- Base: 42.0%
- Full fine-tuning: 45.3%
- Random: 41.6%
- Motion magnitude: 44.9%
- V-JEPA: 45.6%
- Ours w/o MM: 43.8%
- **MOTIVE: 48.3%**

Thus, MOTIVE produced the highest Dynamic Degree on both Wan models, even though it used only 10% of the training data. It also maintained or improved motion smoothness without substantially harming subject or background consistency.

---

### 5. Additional Results on LTX-2B

MOTIVE also performed best on LTX-2B:

- Motion Smoothness: **95.5%**
- Dynamic Degree: **45.1%**

The Dynamic Degree scores were:

- Base: 36.5%
- Full fine-tuning: 38.9%
- Random: 37.8%
- Motion magnitude: 39.6%
- V-JEPA: 40.2%
- Ours w/o MM: 41.8%
- **MOTIVE: 45.1%**

This suggests that the method generalizes across different video-generation architectures.

---

### 6. Human Evaluation

Seventeen annotators compared videos generated by different models across 50 test cases.

MOTIVE was preferred:

- Over the Base model: **74.1% win rate**
- Over Random selection: **58.9%**
- Over Full fine-tuning: **53.1%**
- Over attribution without motion masking: **46.9%**

The 74.1% preference over the Base model indicates that the improvements were perceptually meaningful, not limited to automated benchmark scores.

---

### 7. Qualitative and Additional Findings

High-influence samples selected by MOTIVE generally contained:

- Smooth and continuous trajectories
- Physically plausible interactions
- Consistent object deformation
- Motion patterns transferable to the query videos

Low-influence samples often contained static scenes, camera-only motion, or cartoon-like content with limited physical transferability.

Importantly, MOTIVE was not simply selecting high-motion videos. The average motion magnitude was:

- Top 10% influential videos: **3.85**
- Bottom 10% influential videos: **3.69**

The difference was only about **4.3%**, showing that influence depends on how useful a video is for learning the target dynamics, not on its overall motion magnitude.

---

### 8. Ablation Findings

- **Frame-length normalization** reduced spurious correlations between video length and influence scores by approximately **54%**.
- A **single fixed timestep** achieved about **66% ranking agreement** with a multi-timestep estimator while greatly reducing computation.
- A **512-dimensional Fastfood projection** preserved approximately **74.7%** Spearman correlation with full-gradient rankings, providing a practical balance between accuracy, storage, and scalability.

Overall, the experiments show that MOTIVE can identify motion-relevant training clips and use only a small, targeted subset of data to improve the temporal dynamics and physical plausibility of video generation models.


<br/>
# 예제



### 1. 이 논문이 해결하려는 과제

MOTIVE의 목표는 다음 질문에 답하는 것입니다.

> **비디오 생성 모델이 특정한 움직임을 생성하는 데 어떤 트레이닝 비디오가 가장 큰 영향을 주었는가?**

기존 데이터 어트리뷰션은 주로 이미지의 객체, 배경, 질감 같은 **정적인 외관**을 기준으로 학습 데이터의 영향을 계산합니다. 하지만 비디오에서는 물체의 이동, 회전, 변형, 낙하, 충돌처럼 **시간에 따른 움직임**이 중요합니다.

따라서 MOTIVE는 단순히 “비슷하게 생긴 비디오”가 아니라, **비슷한 움직임을 학습하는 데 도움이 되는 비디오**를 찾습니다.

---

### 2. 트레이닝 데이터와 테스트 데이터의 구체적인 예

#### 트레이닝 데이터

대규모 비디오 데이터셋에서 가져온 비디오 클립입니다.

예를 들어 다음과 같은 비디오가 후보가 될 수 있습니다.

- 물 위에서 흔들리며 떠 있는 물체
- 파도에 의해 물체가 움직이는 장면
- 실이 감긴 실패가 굴러가는 장면
- 공이 바닥에 튕기는 장면
- 물체가 압력에 의해 찌그러지는 장면
- 물체가 회전하거나 미끄러지는 장면
- 카메라만 움직이고 물체는 거의 정지한 장면
- 애니메이션이나 물리적으로 부자연스러운 장면

각 트레이닝 샘플은 일반적으로 다음과 같은 입력으로 구성됩니다.

\[
(v_n, c_n)
\]

- \(v_n\): 비디오 클립
- \(c_n\): 비디오를 설명하는 텍스트 캡션

예시:

> 입력 비디오: “나무 바닥에서 농구공이 위아래로 튀는 81프레임 비디오”  
> 입력 캡션: “A basketball bouncing vertically on a wooden court.”

---

#### 테스트 또는 쿼리 데이터

테스트 데이터는 “어떤 움직임에 관심이 있는가?”를 지정하는 기준 영상입니다. 논문에서는 움직임을 명확하게 통제하기 위해 Veo-3로 만든 합성 비디오를 사용했습니다.

총 10개의 움직임 유형을 사용했습니다.

- compress: 눌려 찌그러짐
- bounce: 튀어 오름
- roll: 굴러감
- explode: 폭발
- float: 떠 있음
- free fall: 자유 낙하
- slide: 미끄러짐
- spin: 제자리 회전
- stretch: 늘어남
- swing: 흔들림

각 유형마다 5개의 쿼리 비디오를 만들었으므로 총 50개입니다.

예를 들어 **float** 쿼리는 다음과 같습니다.

> 입력 쿼리 영상: “투명한 수조의 물 위에서 스펀지 큐브가 부력에 의해 떠 있고 약간 흔들리는 영상”  
> 쿼리 캡션:  
> “A foam cube floating on the surface of water; emphasize buoyancy and slight rocking.”

또 다른 예로 **roll** 쿼리는 다음과 같습니다.

> 입력 쿼리 영상: “실패가 고정된 카메라 앞에서 좌우로 부드럽게 굴러가는 영상”  
> 쿼리 캡션:  
> “A spool of thread rolling from left to right; highlight axle rotation and smooth travel.”

중요한 점은 쿼리 데이터가 트레이닝에 직접 사용되는 데이터가 아니라는 것입니다. 쿼리는 **찾고 싶은 움직임을 정의하는 기준 또는 검색 질의**로 사용됩니다.

---

### 3. MOTIVE의 입력과 출력

#### 전체 입력

MOTIVE에는 다음이 입력됩니다.

1. 사전 학습된 비디오 생성 모델
2. 파인튜닝 데이터셋의 비디오와 캡션
3. 움직임을 나타내는 쿼리 비디오와 캡션
4. 비디오에서 움직임을 검출하는 추적기(AllTracker)

모델은 Wan2.1-T2V-1.3B, Wan2.2-TI2V-5B, LTX-2B 등으로 실험되었습니다.

---

#### 내부 처리 과정

##### ① 비디오에서 움직임 검출

AllTracker를 사용해 연속 프레임 사이의 픽셀 이동을 계산합니다.

각 위치의 움직임은 다음과 같은 변위 벡터로 나타냅니다.

\[
D_f(h,w)=(d_w,d_h)
\]

이 벡터의 크기

\[
M_f(h,w)=\|D_f(h,w)\|_2
\]

가 해당 위치의 움직임 크기입니다.

예를 들어:

- 움직이는 공, 물체, 파도: 큰 motion magnitude
- 벽, 바닥, 배경: 작은 motion magnitude

이 값을 정규화하여 움직임 마스크를 만듭니다.

---

##### ② 움직이는 영역에 더 큰 손실 가중치 부여

각 비디오를 VAE latent 공간으로 변환한 뒤, 움직임 마스크를 latent 해상도에 맞게 축소합니다.

그 다음 일반적인 비디오 확산 손실을 모든 위치에 동일하게 적용하지 않고, 움직이는 영역에 더 큰 가중치를 줍니다.

\[
L_{\text{mot}}
=
\frac{1}{F}
\sum_{f,h,w}
W(f,h,w)\tilde{L}(f,h,w)
\]

- \(W(f,h,w)\): 해당 위치의 움직임 가중치
- \(\tilde{L}\): 해당 위치에서의 예측 오차
- \(F\): 프레임 수

따라서 배경의 색이나 질감보다 공의 이동, 물체의 회전, 변형 등의 정보가 gradient에 더 강하게 반영됩니다.

---

##### ③ 움직임 관련 gradient 계산

트레이닝 비디오와 쿼리 비디오 각각에 대해 motion-weighted loss의 gradient를 계산합니다.

예를 들어:

- 쿼리: 물 위에 물체가 떠 있는 움직임
- 트레이닝 샘플 A: 파도에 따라 물체가 떠서 흔들리는 영상
- 트레이닝 샘플 B: 정적인 제품 촬영 영상

쿼리와 샘플 A의 motion gradient는 비슷할 가능성이 높고, 샘플 B의 gradient는 덜 비슷할 가능성이 높습니다.

---

##### ④ 영향력 점수 계산

두 gradient를 정규화하고 내적하여 영향력 점수를 계산합니다.

\[
I_{\text{mot}}(v_n,\hat v)
=
\tilde g_{\text{mot}}(\hat v)^\top
\tilde g_{\text{mot}}(v_n)
\]

점수의 의미는 다음과 같습니다.

- **높은 양의 점수**: 쿼리의 움직임을 학습하는 데 도움이 될 가능성이 높은 샘플
- **낮거나 음의 점수**: 쿼리 움직임과 관련성이 낮거나, 해당 움직임 학습에 방해가 될 가능성이 있는 샘플

실제 계산에서는 거대한 gradient를 그대로 저장하지 않고 Fastfood 기반 랜덤 프로젝션으로 512차원 벡터로 줄여 효율성을 높입니다.

---

### 4. 구체적인 예시

#### 예시 A: “떠 있는 움직임(float)” 학습 데이터 선택

##### 쿼리

> 투명한 수조에서 초록색 잎이 물 위에 떠 있고 잔잔한 물결과 함께 천천히 움직이는 영상

##### 후보 트레이닝 데이터

| 샘플 | 내용 | 예상 영향 |
|---|---|---|
| A | 파도 위에서 작은 물체가 떠서 흔들림 | 높은 양의 영향 |
| B | 물 위에서 보트가 천천히 움직임 | 중간 또는 높은 영향 |
| C | 고정된 카메라로 촬영한 정적인 제품 영상 | 낮은 영향 |
| D | 카메라가 이동하지만 물체는 정지 | 낮은 영향 또는 부정적 영향 |
| E | 만화 속 물체가 비현실적으로 떠 있음 | 낮은 영향 |

##### MOTIVE의 출력

각 후보에 대해 점수를 출력합니다.

예:

- A: 0.82
- B: 0.67
- C: 0.08
- D: -0.12
- E: 0.03

그 후 점수가 높은 샘플을 정렬하여 상위 10%를 파인튜닝 데이터로 선택합니다.

##### 최종 결과

선택된 데이터로 모델을 파인튜닝하면 다음과 같은 프롬프트에서 더 나은 결과를 기대할 수 있습니다.

> “A leaf floating gently on still water.”

기존 모델보다 물체가 갑자기 사라지거나 튀지 않고, 물결과 물체의 움직임이 더 자연스럽고 일관되게 나타날 수 있습니다.

---

#### 예시 B: “굴러가는 움직임(roll)” 학습 데이터 선택

##### 쿼리

> 실패가 일정한 속도로 굴러가며 축도 함께 회전하는 영상

##### 높은 영향력을 가질 가능성이 높은 데이터

- 타이어가 흔들림 없이 굴러가는 영상
- 공이 바닥에서 회전하며 이동하는 영상
- 원통형 물체가 일정한 궤적으로 이동하는 영상

##### 낮은 영향력을 가질 가능성이 높은 데이터

- 카메라만 좌우로 이동하는 영상
- 물체가 정지해 있고 배경만 움직이는 영상
- 물체가 순간적으로 위치를 바꾸는 비현실적인 애니메이션

여기서 MOTIVE는 단순히 “움직임이 큰 영상”을 선택하지 않습니다. 실제 분석에서 높은 영향력을 가진 영상의 평균 움직임 크기는 낮은 영향력 영상보다 약간 높을 뿐이었습니다. 즉, 중요한 기준은 움직임의 양이 아니라 **쿼리 움직임을 모델이 학습하는 데 gradient 방향이 얼마나 일치하는가**입니다.

---

### 5. 여러 쿼리를 사용하는 경우

하나의 움직임 유형에 쿼리가 여러 개 있을 수 있습니다.

예를 들어 float에 대해 다음 5개의 쿼리가 있을 수 있습니다.

- 잎이 물에 떠 있음
- 스펀지 큐브가 물 위에 떠 있음
- 작은 장난감이 물결에 흔들림
- 보트가 잔잔한 물 위에 떠 있음
- 공기 중 풍선이 천천히 떠 있음

각 쿼리에서 높은 점수를 받은 트레이닝 샘플에 투표를 부여합니다.

\[
\text{MajVote}(v_n)
=
\sum_q
\mathbf{1}
[I_{\text{mot}}(v_n,\hat v_q)>\tau]
\]

여러 쿼리에서 반복적으로 높은 점수를 받은 샘플을 최종 선택합니다.

즉, 한 영상에만 우연히 비슷한 샘플보다 **여러 형태의 float 움직임에 공통으로 도움이 되는 샘플**을 우선합니다.

---

### 6. MOTIVE의 최종 출력

MOTIVE 자체가 새로운 비디오를 생성하는 모델은 아닙니다. 주요 출력은 다음과 같습니다.

1. 각 트레이닝 비디오의 motion influence score
2. 쿼리 움직임에 긍정적·부정적인 트레이닝 샘플 목록
3. 상위 영향력 데이터로 구성된 파인튜닝 subset
4. 해당 subset으로 파인튜닝한 비디오 생성 모델

논문에서는 전체 데이터의 10%만 사용해도 전체 데이터 파인튜닝보다 더 높은 dynamic degree를 얻었습니다.

Wan2.1-T2V-1.3B 기준:

- Base model: Dynamic Degree 39.6%
- Random selection: 41.3%
- Full fine-tuning: 42.0%
- MOTIVE: **47.6%**

또한 사람 평가에서는 MOTIVE 모델이 Base model보다 더 나은 움직임을 보였다는 선호율이 **74.1%**였습니다.

---




### 1. What task does MOTIVE solve?

MOTIVE addresses the following question:

> **Which training videos most strongly influence a video generation model’s ability to produce a particular motion?**

Existing data attribution methods often focus on static appearance, such as objects, textures, or backgrounds. MOTIVE instead focuses on temporal dynamics, including motion trajectories, deformation, rotation, falling, bouncing, and interactions.

The goal is not merely to find visually similar videos, but to find training videos that are useful for learning similar motion patterns.

---

### 2. Concrete training and query data

#### Training data

The training data consist of video clips and their captions:

\[
(v_n,c_n)
\]

For example:

> Video: A basketball bouncing on a wooden floor for 81 frames  
> Caption: “A basketball bouncing vertically on a wooden court.”

The candidate dataset may also contain videos of:

- Objects floating on water
- Waves moving objects
- Tires or spools rolling
- Balls bouncing
- Objects being compressed or stretched
- Objects sliding or spinning
- Static product shots
- Camera-only motion
- Cartoon-like or physically implausible motion

---

#### Query data

The query data specify the motion that we want to analyze or improve. The paper uses synthetic query videos generated with Veo-3 so that the motion is clear and controlled.

The ten motion categories are:

- compress
- bounce
- roll
- explode
- float
- free fall
- slide
- spin
- stretch
- swing

There are five query videos per category, for a total of 50 queries.

Example of a **float** query:

> Video: A foam cube floating on water and gently rocking  
> Caption: “A foam cube floating on the surface of water; emphasize buoyancy and slight rocking.”

Example of a **roll** query:

> Video: A spool of thread rolling smoothly from left to right  
> Caption: “A spool of thread rolling from left to right; highlight axle rotation and smooth travel.”

The query videos are not used as training data. They serve as target examples that define the motion for which influential training clips should be found.

---

### 3. MOTIVE input and output

#### Inputs

MOTIVE receives:

1. A pretrained video generation model
2. Training videos and captions
3. A query video and caption
4. A motion estimator such as AllTracker

The experiments use Wan2.1-T2V-1.3B, Wan2.2-TI2V-5B, and LTX-2B.

---

#### Processing steps

##### Step 1: Detect motion

AllTracker estimates pixel displacement between consecutive frames:

\[
D_f(h,w)=(d_w,d_h)
\]

The magnitude

\[
M_f(h,w)=\|D_f(h,w)\|_2
\]

indicates how much each location moves.

Moving objects receive larger motion values, while static backgrounds receive smaller values.

---

##### Step 2: Weight the loss toward dynamic regions

The motion map is downsampled to the VAE latent resolution and used as a spatial loss mask.

The motion-weighted loss is:

\[
L_{\text{mot}}
=
\frac{1}{F}
\sum_{f,h,w}
W(f,h,w)\tilde L(f,h,w)
\]

This makes the gradient emphasize moving regions rather than static appearance.

---

##### Step 3: Compute motion gradients

MOTIVE computes a motion-weighted gradient for every training video and for the query video.

For example:

- Query: an object floating and rocking on water
- Training clip A: an object being carried by waves
- Training clip B: a static product video

The gradient of clip A is expected to be more similar to the query gradient than the gradient of clip B.

---

##### Step 4: Compute influence scores

The normalized motion gradients are compared using a dot product:

\[
I_{\text{mot}}(v_n,\hat v)
=
\tilde g_{\text{mot}}(\hat v)^\top
\tilde g_{\text{mot}}(v_n)
\]

- High positive score: likely helpful for learning the query motion
- Low or negative score: weakly related or potentially conflicting

For scalability, the full gradients are projected into a 512-dimensional space using a Fastfood-based random projection.

---

### 4. Concrete example: selecting data for “floating”

#### Query

> A green leaf floating gently on still water, with slight rocking caused by surface tension.

#### Candidate training clips

| Clip | Content | Expected influence |
|---|---|---|
| A | A small object floating and rocking on waves | High positive influence |
| B | A boat drifting slowly on water | Medium or high influence |
| C | A static product video | Low influence |
| D | A moving camera observing a stationary object | Low or negative influence |
| E | An unrealistic cartoon object floating | Low influence |

#### Example output scores

- A: 0.82
- B: 0.67
- C: 0.08
- D: -0.12
- E: 0.03

MOTIVE ranks the clips and selects the top 10% for fine-tuning.

The resulting model may generate more stable videos for prompts such as:

> “A leaf floating gently on still water.”

The object is less likely to disappear, jump between positions, or move inconsistently.

---

### 5. Multiple query videos

For one motion category, MOTIVE can use several query videos. For example, the “float” queries may show:

- A leaf floating on water
- A foam cube floating
- A toy rocking on waves
- A boat drifting
- A balloon slowly floating in air

A training clip receives a vote whenever its influence score is above a percentile threshold for a query:

\[
\text{MajVote}(v_n)
=
\sum_q
\mathbf{1}
[I_{\text{mot}}(v_n,\hat v_q)>\tau]
\]

Clips that receive votes from many queries are selected first. This favors data that are consistently useful across different visual instances of the same motion.

---

### 6. Final output and results

MOTIVE does not directly generate videos. Its main outputs are:

1. Motion influence scores for training videos
2. Positive and negative training examples for a target motion
3. A high-influence fine-tuning subset
4. A fine-tuned video generation model

Using only 10% of the training data, MOTIVE achieved a higher dynamic degree than full fine-tuning in the reported experiments.

For Wan2.1-T2V-1.3B:

- Base model: 39.6%
- Random selection: 41.3%
- Full fine-tuning: 42.0%
- MOTIVE: **47.6%**

In human evaluation, videos generated with MOTIVE-based fine-tuning were preferred over the base model in **74.1%** of comparisons.

<br/>
# 요약

MOTIVE는 AllTracker의 optical flow로 움직이는 영역을 찾아 loss에 가중하고, 고정된 timestep의 motion-weighted gradient를 Fastfood로 투영해 학습 클립과 생성 영상의 motion 영향도를 비교한다.  
영향도가 높은 상위 10% 데이터를 골라 fine-tuning한 결과, Wan2.1과 Wan2.2의 VBench dynamic degree가 각각 47.6%, 48.3%로 올라가 전체 데이터 fine-tuning과 random selection을 능가했으며, motion smoothness도 유지됐다.  
예를 들어 roll·float·bounce 같은 질의에는 연속적인 궤적과 물리적으로 자연스러운 움직임을 가진 클립이 선택된 반면, 정적인 장면·카메라만 움직이는 영상·만화풍 클립은 낮게 평가됐고, 사람 평가에서 기본 모델 대비 74.1%의 선호율을 기록했다.  




MOTIVE detects dynamic regions with AllTracker optical flow, weights the loss accordingly, and compares Fastfood-projected motion-weighted gradients to estimate how strongly each training clip affects generated motion.  
Fine-tuning with the top 10% most influential clips raises VBench dynamic degree to 47.6% for Wan2.1 and 48.3% for Wan2.2, outperforming full-data fine-tuning and random selection while preserving motion smoothness.  
For queries such as roll, float, and bounce, it selects clips with continuous trajectories and physically plausible dynamics, while downranking static scenes, camera-only motion, and cartoon-style videos; human raters preferred it over the base model 74.1% of the time.

<br/>
# 기타



## 1. 다이어그램·피규어

### Figure 1 — MOTIVE 전체 구조
MOTIVE는 다음 순서로 동작한다.

1. **AllTracker**로 영상의 픽셀별 움직임과 optical flow를 추출한다.
2. 움직임 크기를 latent 해상도로 변환하고 motion mask를 만든다.
3. 움직임이 큰 위치의 denoising loss에 더 큰 가중치를 부여한다.
4. motion-weighted loss의 gradient를 계산한다.
5. gradient를 Fastfood 방식으로 저차원 투영한 뒤, query 영상과 training 영상의 cosine similarity를 계산한다.
6. 영향력이 높은 영상을 ranking하고 fine-tuning 데이터로 선택한다.

**핵심 인사이트:** 단순히 영상 전체의 시각적 유사도를 비교하는 것이 아니라, “어떤 학습 영상이 특정 움직임을 생성하는 데 유용한가”를 gradient 수준에서 측정한다.

### Figure 2 — Motion attribution 사례
Float와 roll query에 대해 MOTIVE가 선택한 상위 영상은 물체의 연속적인 궤적, 부드러운 움직임, 물리적으로 자연스러운 dynamics를 포함한다. 반면 낮은 영향력의 영상은 정적인 장면, 카메라만 움직이는 영상, cartoon 스타일 영상이 많다.

**인사이트:** MOTIVE는 단순히 움직임이 많은 영상을 고르는 것이 아니라, query의 motion pattern과 모델의 motion 학습에 실제로 도움이 되는 영상을 선택한다.

### Figure 3 — 정성적 생성 결과 비교
Base model, random selection fine-tuning, MOTIVE fine-tuning을 비교한다. MOTIVE를 사용한 모델은 roll, float, deformation, physics-based motion에서 더 자연스러운 궤적과 시간적 일관성을 보인다.

**인사이트:** attribution 결과가 단순한 ranking에 그치지 않고, 실제 생성 영상의 motion fidelity와 temporal consistency 개선으로 이어진다.

### Figure 4 — Frame-length normalization 효과
정규화를 적용하지 않으면 긴 영상이 gradient 크기 때문에 높은 영향력을 받는다. 이 경우 상위 영상들 사이에 일관된 motion pattern이 나타나지 않는다. 프레임 수로 gradient를 정규화하면 float query에 대해 파도, 부유 물체, surfing 등 관련 움직임이 일관되게 선택된다.

**인사이트:** video attribution에서는 영상 길이가 중요한 편향 요인이다. motion similarity를 보려면 frame-length correction이 필요하다.

### Figure 5 — Gradient projection 차원 분석
Fastfood projection 차원이 커질수록 full-gradient ranking과의 Spearman correlation이 증가한다.

- \(D'=128\): 약 46.9%
- \(D'=512\): 약 74.7%
- \(D'=1024\): 약 75.7%
- \(D'=2048\): 약 76.1%

**인사이트:** 512차원만으로도 ranking 품질과 메모리·계산량 사이에서 좋은 절충점을 얻는다. 더 큰 차원은 성능 향상이 작다.

### Figure 6 — MOTIVE는 “움직임이 큰 영상”만 선택하지 않음
Top 10%와 bottom 10% 영상의 평균 motion magnitude는 각각 3.85와 3.69로 큰 차이가 없다. 영향력이 높은 데이터는 낮은 움직임부터 높은 움직임까지 전체 motion spectrum에 걸쳐 분포한다.

**인사이트:** MOTIVE의 기준은 raw motion magnitude가 아니라, 해당 영상의 gradient가 query motion을 학습하는 데 얼마나 직접적으로 기여하는지이다.

### Figure 7 — Motion category 간 influence overlap
서로 다른 motion category 사이에도 영향력 높은 데이터가 공유된다.

- 높은 overlap: **bounce–float**, **compress–float**, **compress–spin**
- 낮은 overlap: **free fall–stretch**, **float–slide**

두 데이터셋에서 유사한 패턴이 반복되며, 평균 overlap은 약 24% 수준이다.

**인사이트:** 모델은 일부 motion primitive를 공유된 학습 데이터로 학습하지만, 물리적 특성이 크게 다른 움직임은 더 전문화된 데이터를 필요로 한다. 행렬이 비대칭적인 것은 category마다 고유한 influential sample의 수가 다르기 때문이다.

### Figure 8 — Query dataset 구성
10개 motion category마다 5개씩, 총 50개의 query 영상을 구성했다. Query는 Veo-3로 합성한 뒤 물리적 타당성과 시각적 품질을 검수했다.

**인사이트:** 통제된 synthetic query를 사용함으로써 배경, 물체 종류, 카메라 움직임보다 특정 motion primitive에 집중할 수 있다.

### Figure 9 — Motion overlay 시각화
Optical flow 기반 motion magnitude를 overlay로 표시한다. 움직이는 물체나 영역은 강조되고, 정적인 배경은 회색에 가깝게 약화된다.

**인사이트:** MOTIVE의 masking이 appearance 전체를 동일하게 취급하지 않고 dynamic region에 gradient를 집중시키는 방식을 직관적으로 보여준다.

---

## 2. 주요 테이블

### Table 1 — VBench 결과
모든 선택 방법은 학습 데이터의 10%만 사용한다. MOTIVE는 두 Wan 모델에서 가장 높은 Dynamic Degree를 기록했다.

- Wan2.1-T2V-1.3B: **47.6%**
- Wan2.2-TI2V-5B: **48.3%**

이는 다음보다 높다.

- Random: 41.3%, 41.6%
- Whole-video attribution, motion masking 없음: 43.8%, 43.8%
- Full fine-tuning: 42.0%, 45.3%

Motion smoothness도 높게 유지되었고, aesthetic quality 역시 경쟁력 있었다.

**핵심 결과:** 전체 데이터로 fine-tuning하는 것보다 적은 10%의 데이터만 사용해도, motion-specific data selection이 dynamic degree 개선에 더 효과적일 수 있다. 또한 motion masking이 없는 attribution보다 masking을 사용하는 MOTIVE가 더 좋다.

### Table 2 — Human evaluation
17명이 50개 영상에 대해 pairwise 평가를 수행했다.

- MOTIVE vs Base: **74.1% win**
- MOTIVE vs Random: **58.9% win**
- MOTIVE vs Full fine-tuning: **53.1% win**
- MOTIVE vs Ours w/o motion masking: **46.9% win**

**인사이트:** 자동 평가뿐 아니라 사람의 지각에서도 MOTIVE가 더 자연스러운 motion을 만든다는 결과를 얻었다. 다만 motion masking이 없는 방법과의 차이는 상대적으로 작아, 모든 경우에 압도적인 것은 아니다.

### Table 5 — LTX-2B 추가 실험
다른 구조와 학습 방식을 사용하는 LTX-2B에서도 MOTIVE는 Dynamic Degree **45.1%**를 달성했다.

비교 결과:

- Base: 36.5%
- Full fine-tuning: 38.9%
- Random: 37.8%
- V-JEPA: 40.2%
- Ours w/o masking: 41.8%
- MOTIVE: **45.1%**

**인사이트:** MOTIVE의 효과가 Wan 계열 모델에만 한정되지 않고, 다른 video generation architecture에도 어느 정도 일반화된다.

### Table 6 — Runtime breakdown
10k training samples와 Wan2.1-T2V-1.3B 기준으로 가장 큰 비용은 sample별 gradient 계산이다.

- Training gradient 계산: 약 **150시간 / 1 A100**
- 64 GPU 병렬화 시: 약 **2.3시간**
- Projection: sample당 약 **1.97초**
- Query influence 계산: 약 **46ms**
- Majority vote aggregation: 약 **139ms**

**인사이트:** 초기 gradient 계산은 매우 비싸지만, 한 번 계산한 training gradient를 여러 query에 재사용할 수 있다. 따라서 query가 많아질수록 amortization 효과가 커진다.

### Table 7 — Baseline과의 처리 시간 비교
10k samples를 한 GPU에서 처리하는 총 시간은 다음과 같다.

- Random: 1초 미만
- Motion Magnitude: 약 5.5시간
- Optical Flow: 약 5.7시간
- V-JEPA: 약 3시간
- MOTIVE: 약 150시간

**인사이트:** MOTIVE는 baseline보다 훨씬 비싸다. 따라서 실제 활용에서는 한 번의 대규모 gradient 계산을 데이터 큐레이션 인프라에서 수행하고, 이후 여러 query에 재사용하는 방식이 적합하다.

---

## 3. 어펜딕스별 핵심

### Appendix A — Notation
비디오, latent, diffusion/flow-matching loss, motion mask, influence score, projection 관련 기호를 정리한다. 특히 \(L_{\text{mot}}\), \(g_{\text{mot}}\), \(I_{\text{mot}}\)이 MOTIVE의 핵심 구성요소다.

### Appendix B — Related work
기존 data attribution은 주로 image diffusion과 static appearance를 다룬다. MOTIVE는 optical flow 기반 motion saliency와 gradient attribution을 결합해 temporal dynamics에 초점을 맞춘다는 차이가 있다.

### Appendix C — 추가 모델 결과
LTX-2B 실험을 통해 모델 구조에 관계없이 motion-weighted gradient attribution을 적용할 수 있음을 보인다.

### Appendix D — Motion distribution analysis
MOTIVE는 motion magnitude가 높은 영상만 선택하지 않는다. 영향력은 영상의 전체 움직임 크기보다 query motion과의 gradient-level compatibility에 의해 결정된다.

### Appendix E — Method generality
- **Tracker-agnostic:** AllTracker 대신 dense optical flow나 다른 point tracker도 사용할 수 있다.
- **Model-agnostic:** diffusion denoiser뿐 아니라 flow-matching velocity field에도 적용 가능하다.
- 필요한 것은 matched timestep/noise 조건에서 계산한 per-example gradient이다.

### Appendix F — Experiment details
주요 설정은 다음과 같다.

- 고정 timestep: denoising trajectory의 중간 지점
- 공통 noise: 모든 train-query pair에 동일한 \(\epsilon_{\text{fix}}\)
- Projection 차원: \(D'=512\)
- 원래 gradient 차원: 약 14억
- Precision: bfloat16
- 선택 데이터: top 10%
- 해상도: 480×832
- Fine-tuning: DiT backbone만 업데이트
- Text encoder와 VAE는 frozen
- 1 epoch, dataset 50회 반복

**인사이트:** 단일 timestep과 공통 noise를 사용해 계산량과 variance를 줄이면서도 full multi-timestep ranking과 상당히 유사한 결과를 얻는다.

### Appendix G — Discussion, limitations, future work
주요 한계는 다음과 같다.

1. sample별 gradient 계산 비용이 매우 크다.
2. 영상 전체를 하나의 단위로 처리해 특정 motion segment가 static 구간에 의해 희석될 수 있다.
3. camera motion과 object motion을 완전히 분리하지 못한다.
4. classifier-free guidance가 inference-time dynamics에 미치는 영향을 명시적으로 반영하지 않는다.
5. 특정 motion을 개선하는 과정에서 base model의 다른 능력이 저하될 수 있다.

향후에는 motion segment-level attribution, tracker ensemble, self-generated failure query, iterative closed-loop curation 등이 제안된다.

### Appendix H — Visualization
Motion overlay를 통해 MOTIVE가 실제로 dynamic region의 loss를 강조하고 static background의 영향을 줄이는지 시각적으로 확인한다.

---

## 전체적인 결론

MOTIVE의 가장 중요한 기여는 **“움직임이 많이 포함된 영상”과 “특정 움직임을 모델이 학습하는 데 실제로 유용한 영상”을 구분한 것**이다. Gradient-based attribution, motion mask, frame-length normalization, low-dimensional projection을 결합해 대규모 video fine-tuning 데이터에서 motion-specific selection을 수행한다.

실험적으로는 데이터의 10%만 사용해도 Dynamic Degree와 사람이 평가한 motion quality를 개선했지만, sample별 gradient 계산 비용이 매우 크다는 점은 실제 적용 시 가장 큰 trade-off다.

---




## 1. Diagrams and Figures

### Figure 1 — Overall MOTIVE pipeline
MOTIVE follows six main steps:

1. Extract pixel-level motion using AllTracker.
2. Convert motion magnitudes to the latent resolution.
3. Weight denoising loss more heavily in dynamic regions.
4. Compute motion-weighted gradients.
5. Project gradients into a compact space and compare train/query videos using cosine similarity.
6. Rank influential clips and select them for fine-tuning.

**Main insight:** MOTIVE measures which training clips help generate a target motion, rather than simply measuring visual or semantic similarity between videos.

### Figure 2 — Motion attribution examples
For float and roll queries, highly ranked clips contain continuous trajectories, smooth movement, and physically plausible dynamics. Low-influence clips are often static, contain only camera motion, or use cartoon-style content.

**Insight:** MOTIVE does not simply select videos with large motion. It selects videos whose gradients are useful for learning the target motion pattern.

### Figure 3 — Qualitative generation comparisons
Compared with the base model and random-selection fine-tuning, MOTIVE produces more coherent trajectories and more realistic temporal evolution, especially for rolling, floating, deformation, and physics-driven motion.

**Insight:** The attribution rankings lead to visible improvements in actual video generation, not just better data rankings.

### Figure 4 — Effect of frame-length normalization
Without normalization, longer videos receive larger gradient magnitudes and are ranked highly regardless of their motion relevance. After normalization, the top-ranked clips for floating motion consistently contain waves, floating objects, and surfing-like dynamics.

**Insight:** Video length is a major source of attribution bias. Frame-length correction is necessary for fair motion attribution.

### Figure 5 — Projection dimension
The correlation with full-gradient rankings improves as the projection dimension increases:

- \(D'=128\): 46.9%
- \(D'=512\): 74.7%
- \(D'=1024\): 75.7%
- \(D'=2048\): 76.1%

**Insight:** A 512-dimensional projection provides a strong accuracy–efficiency trade-off. Larger projections offer only marginal gains.

### Figure 6 — MOTIVE is not selecting only high-motion videos
The average motion magnitude of the top and bottom 10% samples is 3.85 and 3.69, respectively. Influential samples appear across the full motion spectrum, from low-motion to high-motion videos.

**Insight:** Selection is based on gradient-level compatibility with the query motion, not on raw motion magnitude.

### Figure 7 — Cross-motion influence overlap
Some motion categories share influential training data:

- High overlap: **bounce–float**, **compress–float**, **compress–spin**
- Low overlap: **free fall–stretch**, **float–slide**

The same pattern appears in both datasets, with an average overlap of about 24%.

**Insight:** Some motion primitives rely on shared training examples, while mechanically different motions require more specialized data. The matrices are asymmetric because the number of unique influential samples differs across categories.

### Figure 8 — Query dataset
The authors use 50 query videos covering 10 motion categories, with five videos per category. The clips are generated with Veo-3 and manually screened for realism and physical plausibility.

**Insight:** Controlled synthetic queries reduce confounding factors such as background and camera movement, allowing the evaluation to focus on specific motion primitives.

### Figure 9 — Motion overlay visualization
Optical-flow magnitude is visualized as an overlay. Dynamic regions remain emphasized, while static backgrounds are attenuated toward gray.

**Insight:** The visualization provides an intuitive explanation of how MOTIVE concentrates attribution on motion-relevant regions.

---

## 2. Main Tables

### Table 1 — VBench results
All selection methods use only 10% of the training data. MOTIVE achieves the highest Dynamic Degree on both Wan models:

- Wan2.1-T2V-1.3B: **47.6%**
- Wan2.2-TI2V-5B: **48.3%**

This exceeds:

- Random selection: 41.3% and 41.6%
- Whole-video attribution without masking: 43.8% and 43.8%
- Full fine-tuning: 42.0% and 45.3%

Motion smoothness remains strong, and aesthetic quality is also competitive.

**Main result:** Motion-specific data selection can outperform full-dataset fine-tuning on targeted motion metrics while using only 10% of the data.

### Table 2 — Human evaluation
The study uses 17 annotators and 50 generated-video comparisons.

- MOTIVE vs Base: **74.1% win**
- MOTIVE vs Random: **58.9% win**
- MOTIVE vs Full fine-tuning: **53.1% win**
- MOTIVE vs Without motion masking: **46.9% win**

**Insight:** Human judgments confirm that MOTIVE produces perceptually better motion. However, the advantage over the non-masked attribution baseline is relatively modest.

### Table 5 — LTX-2B results
On the different LTX-2B architecture, MOTIVE achieves a Dynamic Degree of **45.1%**, compared with:

- Base: 36.5%
- Full fine-tuning: 38.9%
- Random: 37.8%
- V-JEPA: 40.2%
- Without masking: 41.8%
- MOTIVE: **45.1%**

**Insight:** The method is not limited to the Wan architecture and shows some generalization across video-generation models.

### Table 6 — Runtime breakdown
The dominant cost is per-sample gradient computation.

- Training-gradient computation: about **150 hours on one A100** for 10k samples
- With 64 GPUs: about **2.3 hours**
- Projection: about **1.97 seconds per sample**
- Influence computation for a query: about **46 ms**
- Majority-vote aggregation: about **139 ms**

**Insight:** The initial computation is expensive, but projected training gradients can be reused for many future queries, making the cost amortizable.

### Table 7 — Runtime comparison
For 10k samples on one GPU:

- Random: less than 1 second
- Motion Magnitude: about 5.5 hours
- Optical Flow: about 5.7 hours
- V-JEPA: about 3 hours
- MOTIVE: about 150 hours

**Insight:** MOTIVE is substantially more expensive than simpler baselines. Its practical use therefore depends on precomputing gradients once and reusing them for many data-selection queries.

---

## 3. Appendix Highlights

### Appendix A — Notation
Defines the notation for video latents, diffusion or flow-matching losses, motion masks, influence scores, and projected gradients. The key quantities are \(L_{\text{mot}}\), \(g_{\text{mot}}\), and \(I_{\text{mot}}\).

### Appendix B — Related work
Prior attribution methods mainly target image diffusion and static appearance. MOTIVE differs by combining optical-flow-based motion saliency with gradient attribution for temporal dynamics.

### Appendix C — Additional model results
The LTX-2B experiments show that motion-weighted attribution can be applied beyond the Wan family of models.

### Appendix D — Motion distribution analysis
MOTIVE does not select clips merely because they contain more motion. Influence depends on whether a clip’s gradients help the model learn the query motion.

### Appendix E — Generality of the method
- **Tracker-agnostic:** AllTracker can be replaced by dense optical flow or another point tracker.
- **Model-agnostic:** The method applies to both diffusion denoisers and flow-matching velocity fields.
- The essential requirement is per-example gradients computed under matched timestep and noise conditions.

### Appendix F — Experimental details
Key settings include:

- One fixed midpoint timestep
- Shared Gaussian noise across train/query pairs
- Projection dimension \(D'=512\)
- Original gradient dimension of about 1.4 billion
- bfloat16 computation
- Top-10% data selection
- Resolution of 480×832
- Only the DiT backbone is fine-tuned
- Text encoder and VAE are frozen
- One epoch with 50 dataset repetitions

**Insight:** A single timestep and shared noise substantially reduce cost and variance while preserving useful ranking behavior.

### Appendix G — Limitations and future directions
Main limitations include:

1. Very high upfront cost for per-video gradients.
2. Whole-video attribution may dilute important motion segments with static content.
3. Camera motion and object motion are not fully disentangled.
4. Classifier-free guidance is not explicitly modeled.
5. Targeted motion improvement may trade off against broader base-model capabilities.

Suggested future directions include segment-level attribution, tracker ensembles, self-generated failure queries, and iterative closed-loop data curation.

### Appendix H — Visualization
The motion overlays visually confirm that MOTIVE emphasizes dynamic regions and reduces the contribution of static background regions.

---

## Overall conclusion

The central contribution of MOTIVE is distinguishing between **videos that contain a lot of motion** and **videos that are genuinely useful for learning a particular motion pattern**. By combining gradient attribution, motion masks, frame-length correction, and low-dimensional projection, it enables motion-specific data selection for large-scale video fine-tuning.

The experiments show improved Dynamic Degree and human-perceived motion quality using only 10% of the data. The main practical limitation is the very high cost of computing per-sample gradients, although this cost can be amortized across many query videos.

<br/>
# refer format:  

### BibTeX  

```bibtex
@inproceedings{Wu2026MotionAttribution,
  author    = {Wu, Xindi and Paschalidou, Despoina and Gao, Jun and
               Torralba, Antonio and Leal-Taix{\'e}, Laura and
               Russakovsky, Olga and Fidler, Sanja and Lorraine, Jonathan},
  title     = {Motion Attribution for Video Generation},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  volume    = {306},
  year      = {2026},
  publisher = {PMLR},
  address   = {Seoul, South Korea},
  url       = {https://research.nvidia.com/labs/sil/projects/MOTIVE/}
}
```

### Chicago 스타일 참고문헌  

Wu, Xindi, Despoina Paschalidou, Jun Gao, Antonio Torralba, Laura Leal-Taixé, Olga Russakovsky, Sanja Fidler, and Jonathan Lorraine. “Motion Attribution for Video Generation.” In *Proceedings of the 43rd International Conference on Machine Learning*, vol. 306. Seoul, South Korea: PMLR, 2026. https://research.nvidia.com/labs/sil/projects/MOTIVE/.

  
