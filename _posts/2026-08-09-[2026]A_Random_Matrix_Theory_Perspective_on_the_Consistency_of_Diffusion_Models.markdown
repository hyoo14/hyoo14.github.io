---
layout: post
title:  "[2026]A Random Matrix Theory Perspective on the Consistency of Diffusion Models"
date:   2026-08-09 17:45:02 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 평균과 공분산 같은 Gaussian 통계로 같은 노이즈 씨드에서 디퓨전 모델이 비슷한 이미지를 생성하는 이유를 설명(데이터의 스펙트럼 구조와 공유된 Gaussian 통계가 이유)    


짧은 요약(Abstract) :


이 논문은 **서로 겹치지 않는 데이터로 따로 학습한 diffusion model들이 같은 noise seed에서 비슷한 이미지를 생성하는 이유**를 분석합니다.

핵심 주장은 다음과 같습니다.

- 이런 생성 결과의 일관성은 복잡한 신경망의 고유한 능력만으로 생기는 것이 아니라, 데이터가 공유하는 **평균과 공분산 같은 Gaussian 통계**만으로도 상당 부분 설명할 수 있습니다.
- 이를 분석하기 위해 저자들은 **Random Matrix Theory(RMT)**를 사용해, 유한한 학습 데이터가 선형 denoiser와 sampling map에 어떤 영향을 주는지 연구합니다.
- 유한한 데이터에서는 실제 noise 수준 \(\sigma^2\)가 그대로 사용되지 않고, 더 큰 유효 noise 수준인 \(\kappa(\sigma^2)\)로 바뀐 것처럼 작용합니다. 그 결과 모델은 특히 데이터 분산이 작은 방향을 지나치게 축소하며, 생성 결과를 데이터 평균 쪽으로 끌어당깁니다. 이를 **overshrinkage**라고 합니다.
- 서로 다른 데이터 분할에서 생성 결과가 달라지는 정도는 세 가지 요인으로 설명됩니다.
  1. **Anisotropy**: 데이터의 고유 방향마다 불확실성이 다름  
  2. **Inhomogeneity**: 입력 noise가 데이터 공간의 어느 위치에 있는지에 따라 차이가 다름  
  3. **Dataset size**: 학습 데이터가 많을수록 분할 간 차이가 감소함
- 또한 저자들은 이러한 분석을 한 번의 denoising 단계뿐 아니라 전체 diffusion sampling 과정과 생성 궤적에도 확장합니다.
- 이론은 선형 diffusion model에서 정확하게 작동하며, UNet과 DiT 같은 실제 deep diffusion model에서도 **평균 쪽으로의 과도한 축소, 고유 방향별 차이, 입력별 불균일성**이 정성적으로 관찰된다고 검증합니다.

즉, 이 논문은 diffusion model의 재현성과 일관성이 우연한 현상이 아니라, **데이터의 스펙트럼 구조와 공유된 Gaussian 통계에 의해 상당 부분 결정된다**는 설명을 제시합니다.

---




This paper studies why diffusion models trained on **different, non-overlapping subsets of the same dataset** often generate very similar images when given the same noise seed.

The main points are:

- The consistency does not arise only from complex neural-network behavior. Much of it can already be explained by the shared **Gaussian statistics** of the data, such as its mean and covariance.
- The authors use **Random Matrix Theory (RMT)** to analyze how finite training datasets affect a linear denoiser and the full diffusion sampling map.
- With limited data, the original noise variance \(\sigma^2\) effectively becomes a larger, renormalized value \(\kappa(\sigma^2)\). This causes the model to **overshrink low-variance directions** and pull generated samples toward the dataset mean.
- Disagreement between models trained on different data splits is governed by three factors:
  1. **Anisotropy** across covariance eigenmodes  
  2. **Inhomogeneity** across different input noise locations  
  3. **Dataset-size scaling**, with fluctuations generally decreasing as the dataset grows
- The theory is extended from individual denoising steps to entire diffusion sampling trajectories.
- Although the theory is exact mainly for linear diffusion models, experiments with UNet and DiT models show similar qualitative behaviors, including overshrinkage, spectral anisotropy, and input-dependent inconsistency.

Overall, the paper argues that diffusion-model consistency is largely rooted in the **shared Gaussian statistics and spectral structure of the data**, providing a theoretical baseline for understanding reproducibility across different training splits.


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



이 논문은 **확산 모델의 학습 데이터가 달라도 같은 노이즈 시드에서 비슷한 샘플을 생성하는 현상**을 설명하기 위해, 선형 모델에 대한 무작위 행렬 이론(RMT)을 세우고 이를 실제 딥러닝 확산 모델에 검증했다.

### 1. 기본 확산 모델 설정

- EDM(Elucidated Diffusion Models) 방식의 확산 모델을 사용한다.
- 데이터 \(x_0\)에 가우시안 노이즈 \(\sigma z\)를 더한 입력
  \[
  x=x_0+\sigma z,\quad z\sim\mathcal N(0,I)
  \]
  을 만들고, 모델이 원본 \(x_0\)를 복원하도록 학습한다.
- 학습 목적함수는 denoising score matching(DSM)이다.
- EDM의 preconditioning과 noise weighting을 사용한다.
- 생성 시에는 확률 흐름 ODE(PF-ODE)를 결정론적으로 풀며, 딥러닝 실험에서는 Heun sampler를 사용한다. 따라서 동일한 초기 노이즈 시드는 동일한 생성 경로의 입력이 된다.

### 2. 이론적 기준 모델: 선형 가우시안 denoiser

논문의 핵심 분석 대상은 다음과 같은 affine 선형 denoiser이다.

\[
D(x;\sigma)=W_\sigma x+b_\sigma
\]

데이터의 평균과 공분산을 각각 \(\mu,\Sigma\), 유한 학습 데이터에서 계산한 값을 \(\hat\mu,\hat\Sigma\)라고 할 때, 최적 선형 denoiser는

\[
D^*_{\hat\Sigma}(x;\sigma)
=
\hat\mu+
(\hat\Sigma+\sigma^2I)^{-1}
\hat\Sigma(x-\hat\mu)
\]

로 표현된다.

이 모델은 데이터의 고차원 구조를 직접 학습하지 않고, 주로 **평균과 공분산, 즉 공유된 가우시안 통계**만 사용한다. 따라서 서로 겹치지 않는 데이터 분할로 학습해도 비슷한 평균과 공분산을 공유하면 유사한 생성 결과를 낼 수 있다.

### 3. 유한 데이터 효과를 분석하는 RMT

학습 데이터가 유한하면 \(\hat\Sigma\)는 모집단 공분산 \(\Sigma\)와 다르다. 논문은 이 차이를 RMT의 deterministic equivalence로 분석한다.

핵심 변수는 실제 노이즈 분산 \(\sigma^2\)를 대신하는 **renormalized noise scale** \(\kappa(\sigma^2)\)이다. 이는 다음 자기일관성 방정식으로 정의된다.

\[
\kappa(\lambda)-\lambda
=
\gamma\kappa(\lambda)
\operatorname{tr}
\left[
\Sigma(\Sigma+\kappa(\lambda)I)^{-1}
\right],
\qquad
\gamma=\frac dn
\]

- \(d\): 데이터 차원
- \(n\): 학습 샘플 수
- \(\gamma=d/n\): 차원 대비 데이터 수

이 결과에 따르면 유한 데이터의 평균적인 효과는 대략

\[
\sigma^2\rightarrow \kappa(\sigma^2)
\]

로 노이즈를 더 크게 만든 것과 같다. 그 결과 저분산 고유방향이 실제보다 강하게 축소되고, 생성 샘플이 데이터 평균 쪽으로 끌려간다.

### 4. 샘플링 맵 분석

선형 denoiser를 여러 노이즈 수준에서 연결하면 확산 sampling map을 닫힌 형태로 얻을 수 있다.

\[
\hat x_{\hat\Sigma}(x_{\sigma_T},0)
=
\hat\mu+
\hat\Sigma^{1/2}
(\hat\Sigma+\sigma_T^2I)^{-1/2}
(x_{\sigma_T}-\hat\mu)
\]

즉, 초기 노이즈에서 최종 샘플로 가는 변환은 경험적 공분산의 **행렬 제곱근**에 의해 결정된다.

논문은 fractional matrix power를 적분 표현으로 바꾸고 RMT를 적용하여, 전체 확산 궤적과 최종 샘플의 기대값 및 분산을 분석했다. 이를 통해 다음을 예측한다.

- 유한 데이터에서는 저분산 세부 특징이 과도하게 축소된다.
- 데이터 분할 간 차이는 모든 고유방향에서 같지 않다.
- 특정 데이터 고유방향과 특정 초기 노이즈가 더 큰 생성 불일치를 만든다.

### 5. 분할 간 변동성 분석

서로 독립적인 학습 데이터 분할에서 생성 결과가 얼마나 달라지는지를 다음 세 요소로 분해한다.

1. **Anisotropy(방향별 비등방성)**  
   공분산 고유값 \(\lambda_k\)가 renormalized noise \(\kappa(\sigma^2)\)와 비슷한 고유방향에서 denoiser 불확실성이 가장 크다.

2. **Inhomogeneity(입력별 불균일성)**  
   초기 노이즈나 입력이 데이터 공분산의 큰 고유방향으로 많이 벗어나 있을수록 분할 간 생성 차이가 커진다.

3. **Dataset-size scaling(데이터 수에 따른 스케일링)**  
   전체 변동성은 일반적으로 데이터 수 \(n\)이 증가할수록 감소하며, 충분히 많은 데이터에서는 모집단 모델에 가까워진다.

### 6. 딥러닝 확산 모델을 이용한 검증

이론이 실제 신경망에도 적용되는지 확인하기 위해 다음 모델을 학습했다.

- **UNet 기반 denoiser**
  - 기본 채널 수: 128
  - 채널 배율: \(\{1,2,2,2\}\)
  - 해상도 8에서 self-attention 사용

- **DiT(Diffusion Transformer) 기반 denoiser**
  - hidden size: 384
  - Transformer depth: 6
  - attention head: 6
  - MLP ratio: 4
  - patch size: 주로 2 또는 4

두 모델 모두 EDM 기반 DSM으로 학습했으며, 일반적인 실험 설정은 다음과 같다.

- 학습 데이터셋: FFHQ, AFHQ, CIFAR10, CIFAR100, LSUN Church, LSUN Bedroom
- 해상도: 32×32 및 64×64
- 데이터 수: \(n\in\{300,1000,3000,10000,30000\}\)
- 각 데이터 크기마다 서로 겹치지 않는 두 개의 학습 분할 구성
- 학습: 50,000 step, batch size 256, Adam, learning rate \(10^{-4}\)
- 평가: 동일한 noise seed 사용
- 샘플링: 35 step Heun sampler
- 비교 지표: 두 분할에서 생성한 이미지의 pixel MSE, 고유방향별 분산 및 불일치, 학습 세트와 control 세트에 대한 nearest-neighbor 거리

또한 데이터 분할의 평균이나 특정 주성분 방향의 분산을 의도적으로 다르게 만든 counterfactual 실험도 수행했다. 이를 통해 데이터 분할의 1·2차 통계가 다르면 생성 일관성이 크게 감소하는지 확인했다.

---




This paper explains why diffusion models trained on different data splits can generate similar samples from the same noise seed. The authors first develop a random matrix theory (RMT) for a linear Gaussian diffusion model and then test its predictions on practical neural diffusion networks.

### 1. Diffusion model setup

- The experiments use the EDM formulation of diffusion models.
- A clean data sample is corrupted as
  \[
  x=x_0+\sigma z,\quad z\sim\mathcal N(0,I).
  \]
- The denoiser is trained to reconstruct \(x_0\) using a denoising score matching objective.
- EDM preconditioning and noise-dependent loss weighting are used.
- Sampling is performed through the probability-flow ODE. For deep networks, a deterministic Heun sampler is used, so the same initial noise seed defines the same sampling input.

### 2. Linear Gaussian denoiser

The main theoretical model is an affine linear denoiser,

\[
D(x;\sigma)=W_\sigma x+b_\sigma.
\]

Let \(\mu,\Sigma\) denote the population mean and covariance, and let \(\hat\mu,\hat\Sigma\) be their empirical estimates from a finite training set. The optimal empirical linear denoiser is

\[
D^*_{\hat\Sigma}(x;\sigma)
=
\hat\mu+
(\hat\Sigma+\sigma^2I)^{-1}
\hat\Sigma(x-\hat\mu).
\]

This model only uses first- and second-order data statistics. Therefore, two non-overlapping data splits can produce similar outputs if they share similar means and covariances.

### 3. Random matrix theory analysis

Because the empirical covariance \(\hat\Sigma\) differs from the population covariance \(\Sigma\), the authors analyze its effect using deterministic-equivalence results from RMT.

The central quantity is a renormalized noise scale \(\kappa(\sigma^2)\), defined by

\[
\kappa(\lambda)-\lambda
=
\gamma\kappa(\lambda)
\operatorname{tr}
\left[
\Sigma(\Sigma+\kappa(\lambda)I)^{-1}
\right],
\qquad
\gamma=\frac dn.
\]

Here, \(d\) is the data dimension, \(n\) is the number of training samples, and \(\gamma=d/n\).

The main interpretation is that finite data approximately replace the true noise variance by

\[
\sigma^2\rightarrow \kappa(\sigma^2).
\]

Since \(\kappa(\sigma^2)>\sigma^2\), finite data behave like additional regularization. Low-variance eigenmodes are therefore overshrunk, pulling generated samples toward the dataset mean.

### 4. Sampling-map analysis

For the linear model, the full diffusion sampling map has the closed form

\[
\hat x_{\hat\Sigma}(x_{\sigma_T},0)
=
\hat\mu+
\hat\Sigma^{1/2}
(\hat\Sigma+\sigma_T^2I)^{-1/2}
(x_{\sigma_T}-\hat\mu).
\]

Thus, the mapping from initial noise to the final sample is controlled by a fractional matrix power of the empirical covariance.

The authors use integral representations for fractional matrix powers and apply deterministic-equivalence techniques to study the expectation and variance of the entire sampling map.

### 5. Decomposing cross-split variability

The variation between models trained on independent data splits is decomposed into three components:

1. **Anisotropy**  
   Variability is largest along covariance eigenmodes whose eigenvalues are close to the renormalized noise scale \(\kappa(\sigma^2)\).

2. **Inhomogeneity**  
   Initial noise vectors or inputs displaced along high-variance data directions tend to produce larger cross-split disagreement.

3. **Dataset-size scaling**  
   Overall variance decreases as the training-set size increases, approaching the population model in the large-data limit.

### 6. Validation with deep networks

The theory is tested using two neural architectures:

- **UNet denoisers**
  - Base channels: 128
  - Channel multipliers: \(\{1,2,2,2\}\)
  - Self-attention at resolution 8

- **DiT diffusion transformers**
  - Hidden size: 384
  - 6 Transformer layers
  - 6 attention heads
  - MLP ratio: 4
  - Patch sizes of 2 or 4

The models are trained with EDM-based denoising score matching on:

- FFHQ
- AFHQ
- CIFAR10
- CIFAR100
- LSUN Church
- LSUN Bedroom

Datasets are used at 32×32 or 64×64 resolution, with training sizes

\[
n\in\{300,1000,3000,10000,30000\}.
\]

For each size, two non-overlapping training splits are created. The common training setup is:

- 50,000 training steps
- Batch size 256
- Adam optimizer
- Learning rate \(10^{-4}\)
- 35-step Heun sampling
- Identical noise seeds for cross-split comparison

The evaluation measures pixel-level MSE between corresponding samples, spectral variance and disagreement along covariance eigenmodes, and nearest-neighbor distances to the training and control sets.

The authors also construct counterfactual splits with deliberately different means or variances along selected principal components. This tests whether mismatched first- and second-order statistics reduce generation consistency.


<br/>
# Results



### 1. 비교한 모델과 실험 설정

논문은 **선형 확산 모델**을 이론적 기준선으로 삼고, 실제 **UNet 기반 CNN**과 **DiT(Transformer)** 확산 모델이 같은 현상을 보이는지 비교했다.

- **모델**
  - 선형 가우시안 denoiser 및 Wiener filter 기반 sampling map
  - UNet-CNN
  - DiT
- **학습 데이터**
  - FFHQ, AFHQ, CIFAR10, CIFAR100
  - LSUN church, LSUN bedroom
  - 해상도는 주로 32×32 및 64×64
- **데이터 분할**
  - 서로 겹치지 않는 두 개의 학습 데이터 split을 구성
  - 데이터 수: 대략 \(n=300, 1{,}000, 3{,}000, 10{,}000, 30{,}000\)
- **샘플링**
  - 두 모델에 동일한 초기 noise seed를 입력
  - 결정론적 Heun sampler 사용
  - 따라서 출력 차이는 주로 학습 데이터 split과 모델 학습의 차이에서 발생

---

### 2. 주요 평가 지표

논문은 다음 지표를 사용해 split 간 일관성과 모델의 일반화 정도를 평가했다.

1. **Cross-split MSE**
   - 동일한 noise seed로 서로 다른 split에서 학습한 모델의 생성 이미지 사이의 pixel-wise MSE
   - 값이 낮을수록 같은 seed에 대해 더 일관된 출력을 생성한다는 의미

2. **Nearest-neighbor MSE**
   - 생성 이미지와 학습 split 또는 학습에 사용하지 않은 control split의 가장 가까운 이미지 사이의 거리
   - 생성 결과가 특정 학습 샘플을 단순히 암기했는지 확인하는 데 사용

3. **선형 예측값과의 MSE**
   - DNN의 생성 결과가 데이터의 평균과 공분산만 사용한 Gaussian linear predictor에 얼마나 가까운지 측정

4. **고유공간별 분산 및 cross-split MSE**
   - 데이터 공분산의 principal component(PC) 방향별로 생성 분산과 split 간 불일치를 분석
   - 저주파·고분산 방향과 고주파·저분산 방향의 차이를 확인

5. **Seed-wise correlation**
   - 초기 noise별 이론적 불일치 예측값과 실제 DNN 출력 차이 사이의 상관관계
   - 예를 들어 FFHQ64 UNet에서는 Spearman correlation 약 0.33을 보고했다.

---

### 3. 데이터 split이 달라도 생성 결과가 유사함

FFHQ를 비롯한 여러 데이터셋에서, 서로 겹치지 않는 학습 split으로 훈련한 모델들이 **같은 noise seed에 대해 시각적으로 매우 유사한 이미지를 생성**했다.

- 이 현상은 UNet과 DiT 사이에서도 관찰되었다.
- 서로 다른 아키텍처를 사용한 경우에도 출력의 대응 관계가 유지되었다.
- 생성 이미지 간 MSE는 생성 이미지와 학습 데이터의 nearest neighbor 사이의 MSE보다 낮은 경우가 많았다.
- 따라서 이 결과는 단순한 training-set memorization만으로 설명되기 어렵다.

특히 데이터의 평균과 공분산이 비슷한 무작위 split에서는 consistency가 높았지만, 주성분 방향으로 데이터를 의도적으로 나누어 평균이나 분산을 다르게 만든 경우에는 출력 차이가 크게 증가했다. 이는 **공유된 1·2차 통계량이 cross-split consistency의 중요한 원인**임을 보여준다.

---

### 4. 선형 Gaussian predictor가 consistency의 상당 부분을 설명함

각 split의 평균과 공분산으로 만든 선형 denoiser 및 Wiener filter가 실제 DNN 출력과 상당히 유사한 결과를 냈다.

즉, 확산 모델이 데이터의 모든 고차 통계량을 활용하기 전에도 다음과 같은 단순한 구조가 이미 많은 정보를 제공한다.

- 데이터 평균
- 공분산의 고유값 및 고유벡터
- noise와 데이터 분산의 상대적 크기

이 결과는 diffusion consistency가 부분적으로 **데이터에 공통으로 존재하는 Gaussian 구조를 학습하기 때문에 발생한다**는 해석을 지지한다.

---

### 5. 유한한 데이터는 유효 noise를 증가시키며 overshrinkage를 일으킴

RMT 분석에 따르면, 유한한 학습 데이터의 효과는 대략 다음과 같이 표현된다.

\[
\sigma^2 \rightarrow \kappa(\sigma^2), \qquad \kappa(\sigma^2)>\sigma^2
\]

즉, 모델이 실제 noise보다 더 큰 **renormalized noise scale**을 경험하는 것처럼 동작한다.

그 결과:

- 저분산 또는 저고유값 방향이 noise로 간주되어 더 강하게 축소된다.
- 생성 결과가 데이터 평균 쪽으로 당겨진다.
- 얼굴 이미지에서는 평균 얼굴에 가까워지고, 질감·배경·세부 묘사가 부드러워진다.
- 이러한 overshrinkage는 데이터 수가 증가할수록 감소한다.

DNN 실험에서도 \(n\leq 1{,}000\)에서는 memorization이 강하게 나타났고, \(n\geq 3{,}000\)에서는 일반화 및 renormalization regime가 나타났다. 데이터 수가 약 30,000에 가까워지면 생성된 이미지의 스펙트럼과 선형 population predictor가 점점 일치했다.

---

### 6. 일관성은 모든 방향에서 동일하지 않음: anisotropy

모델 간 불일치는 데이터 공간의 모든 방향에서 균일하지 않았다.

선형 denoiser의 이론에서는 고유값이 \(\lambda_k\)인 주성분 방향의 불확실성이

\[
\chi(\lambda_k,\kappa)
=
\frac{\lambda_k}{(\lambda_k+\kappa)^2}
\]

에 비례한다. 이 값은 대략

\[
\lambda_k \approx \kappa
\]

인 방향에서 최대가 된다.

실험적으로는:

- 높은 noise 단계에서는 저주파·고분산 구조에서 차이가 두드러졌다.
- 낮은 noise 단계에서는 고주파·세부 질감 방향의 차이가 커졌다.
- 최종 sampling map에서는 고유공간 상위 방향의 절대적인 차이가 크게 나타나기도 했지만, 저분산 세부 방향은 더 많은 데이터가 있어야 안정화되었다.
- 데이터 수를 늘리면 주로 상위 고유공간의 불일치가 먼저 감소하고, 낮은 고유공간은 상대적으로 오래 남았다.

따라서 “모델이 더 일관적인가?”라는 질문은 전체 MSE 하나만으로는 충분하지 않고, **어떤 spectral mode에서 차이가 발생하는지**도 함께 봐야 한다.

---

### 7. 입력 noise에 따라서도 consistency가 달라짐: inhomogeneity

동일한 모델이라도 초기 noise seed의 데이터 공분산 방향 정렬 상태에 따라 출력 차이가 달라졌다.

- 초기 noise가 고분산 principal component 방향으로 크게 이동한 경우, split 간 생성 결과의 차이가 더 커졌다.
- 반대로 저분산 방향에 주로 놓인 noise는 상대적으로 안정적인 출력을 만들었다.
- 이론적으로 계산한 noise별 불일치 예측값은 실제 DNN의 cross-split 차이와 유의한 상관을 보였다.
- 다만 실제 DNN의 차이 크기는 선형 이론보다 더 컸으며, 이는 비선형 네트워크의 추가적인 변동성을 반영한다.

---

### 8. 모델 간 비교

- **UNet과 DiT 모두** 데이터 수가 증가할수록 cross-split MSE가 감소했다.
- 동일한 데이터 수에서는 대체로 **DiT가 UNet보다 더 높은 consistency**를 보였다.
- 그러나 두 모델 모두 다음 현상을 공유했다.
  - 작은 데이터에서 memorization
  - 중간 데이터 규모에서 평균 방향으로의 overshrinkage
  - spectral anisotropy
  - 초기 noise 위치에 따른 inhomogeneity
- 따라서 이러한 현상은 특정 아키텍처에만 국한되지 않고, 확산 모델의 공통적인 통계적 구조와 관련된 것으로 해석된다.

---

### 9. 종합 결론

이 논문의 실험 결과는 다음 세 가지로 요약할 수 있다.

1. **동일한 noise seed에 대한 diffusion 모델의 높은 consistency는 실제로 관찰되며, 데이터 split과 아키텍처가 달라도 유지된다.**
2. **데이터 평균과 공분산만을 사용하는 선형 Gaussian predictor가 이 consistency의 상당 부분을 설명한다.**
3. **유한 데이터에서는 저분산 세부 방향이 과도하게 축소되고, split 간 차이는 데이터의 spectral 구조와 초기 noise의 방향에 따라 달라진다.**

다만 선형 RMT 이론은 실제 DNN의 불일치 크기를 과소평가한다. 따라서 이 이론은 DNN의 모든 변동을 설명하는 완전한 모델이라기보다는, **확산 모델의 재현성과 split 간 consistency를 분석하기 위한 핵심적인 기준선**으로 제시된다.

---


## Summary of Results

### 1. Models and experimental setup

The paper uses a **linear diffusion model** as a theoretical baseline and compares it with practical diffusion models based on **UNets** and **DiT transformers**.

- **Models**
  - Linear Gaussian denoiser and Wiener-filter sampling map
  - UNet-based CNN
  - DiT
- **Datasets**
  - FFHQ, AFHQ, CIFAR10, CIFAR100
  - LSUN Church and LSUN Bedroom
  - Mostly 32×32 and 64×64 resolutions
- **Training splits**
  - Two non-overlapping training splits were created
  - Dataset sizes ranged from approximately \(n=300\) to \(30{,}000\)
- **Sampling**
  - The same initial noise seed was used for both models
  - Sampling was performed deterministically with a Heun solver

---

### 2. Evaluation metrics

The main metrics were:

1. **Cross-split MSE**
   - Pixel-wise MSE between samples generated by models trained on different data splits using the same noise seed.
   - Lower MSE indicates stronger consistency.

2. **Nearest-neighbor MSE**
   - Distance between generated samples and their nearest training or control-set examples.
   - Used to distinguish generalization from memorization.

3. **MSE to the linear predictor**
   - Measures how closely DNN-generated samples match the Gaussian predictor based only on the data mean and covariance.

4. **Per-eigenmode variance and MSE**
   - Measures generated variance and cross-split disagreement along covariance principal components.

5. **Seed-wise correlation**
   - Compares theoretical predictions of seed-dependent disagreement with empirical DNN deviations.
   - For example, FFHQ64 UNets achieved a Spearman correlation of approximately 0.33.

---

### 3. Strong consistency across different training splits

Across FFHQ and other datasets, models trained on disjoint subsets often generated **visually similar images from the same noise seed**.

- This occurred for both UNets and DiTs.
- Similar outputs were obtained even across different architectures.
- Cross-split sample MSE was often lower than the distance to the nearest training example.
- Therefore, the effect cannot be explained solely by memorization.

When the data splits were intentionally constructed to have different means or covariances along a principal-component direction, consistency decreased substantially. This shows that **shared first- and second-order statistics are important drivers of diffusion consistency**.

---

### 4. The linear Gaussian predictor explains much of the effect

A linear denoiser or Wiener filter constructed from the empirical mean and covariance of each split produced outputs that were qualitatively similar to the DNN samples.

This suggests that diffusion consistency is partly driven by shared:

- Data means
- Covariance eigenvalues and eigenvectors
- Relative scales of signal variance and diffusion noise

Thus, a substantial part of the common mapping from noise to image can be explained without modeling higher-order statistics.

---

### 5. Finite data increase the effective noise level

The RMT analysis describes the finite-sample effect as an effective noise renormalization:

\[
\sigma^2 \rightarrow \kappa(\sigma^2), \qquad \kappa(\sigma^2)>\sigma^2.
\]

This produces several observable effects:

- Low-variance directions are treated more strongly as noise.
- Samples are pulled toward the dataset mean.
- Generated faces become smoother and more average-like.
- Fine textures and background details are reduced.
- The bias decreases as the training-set size grows.

The DNN experiments showed a transition from memorization at small sample sizes, roughly \(n\leq 1{,}000\), to a generalization and renormalization regime for \(n\geq 3{,}000\). At larger sizes, especially near 30,000 samples, the outputs increasingly approached the population linear predictor.

---

### 6. Consistency is anisotropic across spectral directions

Disagreement is not uniform across the data space. For a covariance eigenmode with eigenvalue \(\lambda_k\), the theoretical uncertainty is proportional to

\[
\chi(\lambda_k,\kappa)
=
\frac{\lambda_k}{(\lambda_k+\kappa)^2},
\]

which is maximized when

\[
\lambda_k \approx \kappa.
\]

Empirically:

- At high noise levels, differences mainly involved low-frequency, large-scale image structure.
- At low noise levels, differences shifted toward high-frequency details and textures.
- High-variance modes tended to stabilize earlier as the dataset grew.
- Low-variance detail modes required substantially more data to become consistent.

Therefore, overall MSE alone does not fully describe consistency; the spectral location of the disagreement is also important.

---

### 7. Consistency depends on the initial noise realization

Different initial noise seeds produced different levels of cross-split disagreement.

- Noise aligned with high-variance covariance directions led to larger output deviations.
- Noise concentrated in low-variance directions produced more stable outputs.
- Theoretical predictions based only on the population covariance and sample size correlated significantly with observed DNN deviations.
- However, the absolute deviations in DNNs were larger than those predicted by the linear theory, reflecting additional nonlinear variability.

---

### 8. Comparison between UNet and DiT

- Both UNet and DiT became more consistent as the dataset size increased.
- At comparable dataset sizes, **DiT was generally more consistent than UNet**.
- Nevertheless, both architectures exhibited the same qualitative phenomena:
  - Memorization with very limited data
  - Overshrinkage toward the mean
  - Spectral anisotropy
  - Dependence on the initial noise location
- These effects therefore appear to reflect general statistical properties of diffusion models rather than the behavior of one particular architecture.

---

### 9. Overall conclusion

The results support three main conclusions:

1. Diffusion models trained on different data splits can produce highly consistent outputs from the same noise seed.
2. A simple linear Gaussian predictor based on the data mean and covariance explains a substantial fraction of this consistency.
3. Finite data cause overshrinkage in low-variance detail directions, while cross-split disagreement depends on both the spectral structure of the data and the alignment of the initial noise.

The linear RMT theory underestimates the absolute variability of practical DNNs, but it successfully predicts **where and under what conditions** diffusion models disagree. It therefore serves as a useful theoretical baseline for analyzing reproducibility and consistency in diffusion training.


<br/>
# 예제



이 논문은 일반적인 분류 문제처럼 **트레이닝 입력과 정답 라벨을 이용해 테스트 정확도를 측정하는 연구가 아닙니다.**  
대신, 서로 다른 데이터 부분집합으로 학습한 diffusion model이 **같은 초기 노이즈를 넣었을 때 얼마나 비슷한 이미지를 생성하는지**를 분석합니다.

### 1. 데이터와 트레이닝 입력

#### 트레이닝 데이터
예를 들어 FFHQ32 얼굴 데이터셋을 두 개의 겹치지 않는 부분집합으로 나눕니다.

- Split 1: 얼굴 이미지 30,000장
- Split 2: 서로 겹치지 않는 얼굴 이미지 30,000장

각 모델은 한쪽 split만 사용해 학습합니다.

- Model 1: Split 1로 학습
- Model 2: Split 2로 학습

두 모델은 같은 이미지들을 보지 않았지만, 전체 데이터 분포의 평균과 공분산 같은 통계적 특성은 상당 부분 공유합니다.

#### 모델에 들어가는 실제 입력
Diffusion 학습에서는 깨끗한 이미지 \(x_0\)에 Gaussian noise를 추가합니다.

\[
x_\sigma=x_0+\sigma z,\qquad z\sim \mathcal N(0,I)
\]

- 입력: 노이즈가 섞인 이미지 \(x_\sigma\), noise scale \(\sigma\)
- 학습 목표: 원래 깨끗한 이미지 \(x_0\) 복원

즉, 구체적인 학습 예시는 다음과 같습니다.

- 트레이닝 입력: 고양이 또는 얼굴 이미지 \(x_0\)에 noise를 추가한 이미지 \(x_\sigma\)
- 정답: 원본 이미지 \(x_0\)
- 모델 출력: 복원된 이미지 \(D_\theta(x_\sigma;\sigma)\)

### 2. 테스트 또는 생성 단계의 입력과 출력

일반적인 테스트 이미지 대신, 이 논문에서는 **동일한 초기 Gaussian noise**를 두 모델에 넣습니다.

예를 들어,

- 같은 초기 noise seed \(s=42\)
- 동일한 초기 노이즈 \(x_{\sigma_T}\)
- Model 1과 Model 2에 각각 입력
- deterministic probability-flow ODE 또는 Heun sampler로 생성

출력은 다음과 같습니다.

- Model 1의 생성 이미지: \(G_1(x_{\sigma_T})\)
- Model 2의 생성 이미지: \(G_2(x_{\sigma_T})\)

그 후 두 출력의 픽셀 MSE 등을 계산합니다.

\[
\text{MSE}
=
\|G_1(x_{\sigma_T})-G_2(x_{\sigma_T})\|^2
\]

MSE가 작으면, 서로 다른 학습 데이터로 학습했어도 같은 noise seed가 비슷한 이미지로 변환된다는 뜻입니다.

### 3. 이 논문이 답하려는 구체적인 테스크

핵심 테스크는 다음과 같습니다.

> **서로 겹치지 않는 트레이닝 데이터로 학습한 두 diffusion model이 같은 초기 노이즈를 얼마나 일관되게 같은 이미지로 변환하는가?**

이를 위해 다음을 비교합니다.

1. **Split 1 모델과 Split 2 모델의 동일 seed 생성 결과**
2. 서로 다른 architecture  
   - UNet
   - DiT
3. 서로 다른 데이터셋 크기  
   - \(n=300, 1{,}000, 3{,}000, 10{,}000, 30{,}000\)
4. 생성 이미지와 학습 데이터의 nearest neighbor 거리  
   - 생성 결과가 특정 학습 이미지를 단순히 복사한 것인지 확인
5. 선형 Gaussian 모델의 예측과 실제 deep diffusion model의 출력 비교

### 4. 선형 모델에서의 구체적인 입력과 출력

선형 denoiser는 다음과 같은 형태입니다.

\[
D(x;\sigma)=W_\sigma x+b_\sigma
\]

트레이닝 데이터에서 평균 \(\hat\mu\)와 공분산 \(\hat\Sigma\)를 계산하고, 이를 이용해 noisy input을 복원합니다.

\[
D^*_{\hat\Sigma}(x;\sigma)
=
\hat\mu+
(\hat\Sigma+\sigma^2I)^{-1}\hat\Sigma(x-\hat\mu)
\]

- 입력: noisy image \(x\)
- 출력: denoised image
- 학습에 사용되는 정보: 주로 데이터의 평균과 공분산

이 모델은 이미지의 고차원 의미나 개별 객체를 직접 기억하지 않지만, 얼굴 이미지에 공통적인 저주파 구조, 평균적인 얼굴 형태, 주요 변동 방향 등을 어느 정도 재현합니다.

### 5. 논문의 주요 결과를 예시로 설명

#### 데이터가 적을 때
공분산을 정확하게 추정하지 못하므로, 모델은 일부 방향을 noise로 잘못 판단합니다.

특히 낮은 분산의 eigenmode, 즉 세밀한 texture나 고주파 디테일을 과도하게 축소합니다.

결과:

- 이미지가 평균 얼굴에 가까워짐
- texture와 배경이 부드러워짐
- 세부 묘사가 부족해짐
- 이를 **overshrinkage**라고 함

#### 데이터가 많아질 때
표본 공분산이 실제 데이터 공분산에 가까워집니다.

결과:

- 평균으로 끌려가는 현상이 감소
- 서로 다른 split에서 생성한 이미지의 차이가 감소
- deep model의 결과가 population Gaussian predictor에 가까워짐

#### 두 모델이 다르게 생성하는 경우
차이는 모든 방향에서 동일하지 않습니다.

- 고분산 방향: 얼굴의 큰 구조나 전체적인 배치
- 저분산 방향: texture, 반사광, 미세한 고주파 패턴

논문은 생성 결과의 불일치가 다음 세 요인으로 나뉜다고 설명합니다.

1. **Anisotropy**: 데이터의 eigenmode 방향에 따라 차이가 다름
2. **Inhomogeneity**: 초기 noise가 특정 데이터 방향으로 크게 벗어날수록 차이가 커짐
3. **Dataset-size scaling**: 데이터가 많아질수록 일반적으로 변동성이 감소

### 6. 중요한 해석

이 논문에서 “일관성”은 두 모델이 같은 이미지를 외워서 출력한다는 의미가 아닙니다.

오히려 다음과 같은 의미입니다.

- 두 학습 split이 데이터의 평균과 공분산을 공유함
- diffusion의 고 noise 영역에서는 이런 Gaussian 통계가 특히 중요한 역할을 함
- 따라서 같은 초기 noise가 두 모델에서 비슷한 고수준 구조로 변환됨
- 데이터가 부족하면 세부 방향을 잘 학습하지 못해 평균 쪽으로 과도하게 수축함

---




This paper is not a standard supervised learning study where training inputs and test labels are used to measure classification accuracy. Instead, it studies whether diffusion models trained on different subsets of data produce similar images when given the **same initial noise**.

### 1. Training data and training inputs

#### Training data

For example, the FFHQ32 face dataset can be divided into two non-overlapping subsets:

- Split 1: 30,000 face images
- Split 2: another 30,000 face images with no overlap

Two models are trained separately:

- Model 1: trained on Split 1
- Model 2: trained on Split 2

Although the models see different individual images, the two splits share statistical properties such as the data mean and covariance.

#### Actual training input

During diffusion training, Gaussian noise is added to a clean image:

\[
x_\sigma=x_0+\sigma z,\qquad z\sim\mathcal N(0,I)
\]

- Input: noisy image \(x_\sigma\) and noise level \(\sigma\)
- Target: original clean image \(x_0\)
- Output: denoised prediction \(D_\theta(x_\sigma;\sigma)\)

For example:

- Training input: a noisy face image
- Target output: the original clean face image
- Model output: a reconstructed or denoised face image

### 2. Test or generation input and output

Instead of using ordinary test images, the paper gives the two models the **same initial Gaussian noise**.

For example:

- Use the same noise seed, such as \(s=42\)
- Generate the same initial noise \(x_{\sigma_T}\)
- Run both models with a deterministic probability-flow ODE or Heun sampler

The outputs are:

- \(G_1(x_{\sigma_T})\): image generated by Model 1
- \(G_2(x_{\sigma_T})\): image generated by Model 2

Their consistency is measured using pixel-level MSE:

\[
\text{MSE}
=
\|G_1(x_{\sigma_T})-G_2(x_{\sigma_T})\|^2
\]

A small MSE means that the two independently trained models map the same noise seed to similar images.

### 3. Main task of the paper

The central task is:

> How consistently do diffusion models trained on disjoint datasets transform the same initial noise into the same generated image?

The authors compare:

1. Outputs from models trained on Split 1 and Split 2
2. Different architectures, including UNet and DiT
3. Different training-set sizes, such as 300, 1,000, 3,000, 10,000, and 30,000 samples
4. Generated images versus nearest training samples, to distinguish generalization from memorization
5. Deep diffusion outputs versus predictions from a linear Gaussian model

### 4. Linear model example

The linear denoiser has the form

\[
D(x;\sigma)=W_\sigma x+b_\sigma.
\]

It uses the empirical mean \(\hat\mu\) and covariance \(\hat\Sigma\) of the training data:

\[
D^*_{\hat\Sigma}(x;\sigma)
=
\hat\mu+
(\hat\Sigma+\sigma^2I)^{-1}\hat\Sigma(x-\hat\mu).
\]

- Input: a noisy image \(x\)
- Output: a denoised image
- Information learned: mainly the mean and covariance of the dataset

Although this model cannot represent detailed semantic structure or memorize individual images, it can still capture shared global patterns such as average face structure and dominant image variations.

### 5. Main findings

#### With limited data

The empirical covariance is estimated inaccurately. The model may treat some low-variance directions as noise and suppress them too strongly.

This causes:

- Outputs to move toward the average face
- Smoother textures and backgrounds
- Loss of fine details
- The phenomenon called **overshrinkage**

#### With more data

The empirical covariance becomes closer to the population covariance.

As a result:

- Overshrinkage becomes weaker
- Outputs from different data splits become more consistent
- Deep diffusion outputs approach the population Gaussian predictor

#### Where disagreement occurs

Disagreement is not uniform across all image directions.

- High-variance directions often correspond to large-scale structure
- Low-variance directions often correspond to fine textures, highlights, and high-frequency details

The paper explains cross-split disagreement using three factors:

1. **Anisotropy**: different covariance eigen-directions have different levels of uncertainty
2. **Inhomogeneity**: some initial noise locations lead to larger disagreement than others
3. **Dataset-size scaling**: disagreement generally decreases as the training dataset grows

### 6. Overall interpretation

The consistency does not mainly mean that the two models memorize the same images. Rather:

- The two data splits share similar first- and second-order statistics
- These Gaussian statistics strongly influence the score field, especially at high noise levels
- Therefore, the same initial noise is transformed into similar high-level image structure
- When data are limited, the model fails to estimate fine-variance directions accurately and pulls samples toward the dataset mean

<br/>
# 요약


1. 연구진은 선형 가우시안 확산 모델과 무작위 행렬 이론(RMT)을 사용해, 유한한 학습 데이터의 공분산 변동이 디노이저와 샘플링 경로에 미치는 영향을 분석했다.  
2. 유한 데이터는 실제 노이즈 수준을 \(\sigma^2\)에서 더 큰 유효 노이즈 \(\kappa(\sigma^2)\)로 바꾸어 저분산·고주파 성분을 과도하게 축소하고, 데이터 평균 쪽으로 샘플을 끌어당기며, 분할 간 차이는 고유모드의 비등방성·입력 위치의 이질성·데이터 수에 의해 결정됐다.  
3. FFHQ, CIFAR, AFHQ, LSUN에서 UNet·DiT와 선형 모델을 비교한 결과, 같은 노이즈 시드의 생성물이 서로 유사해지는 현상과 제한된 데이터에서 평균적인 얼굴·매끄러운 질감이 나타나는 과도한 축소 효과가 이론적으로 예측되고 실험적으로 확인됐다.  



1. The authors use linear Gaussian diffusion models and random matrix theory (RMT) to analyze how finite-sample covariance fluctuations affect denoisers and sampling trajectories.  
2. Finite data effectively replace the noise level \(\sigma^2\) with a larger renormalized scale \(\kappa(\sigma^2)\), causing overshrinkage of low-variance, high-frequency modes toward the data mean, while cross-split differences depend on spectral anisotropy, input inhomogeneity, and dataset size.  
3. Experiments with linear models, UNets, and DiTs on FFHQ, CIFAR, AFHQ, and LSUN confirmed the predicted same-seed consistency and the tendency of limited-data models to generate more average-looking faces with smoother textures.

<br/>
# 기타



이 논문의 그림·다이어그램·부록은 본문 이론을 **직관적으로 설명하고, 선형 모델에서 검증한 뒤, 실제 UNet/DiT로 확장되는지 확인**하는 역할을 한다. 핵심 메시지는 다음과 같다.

> 데이터 분할이 달라도 평균과 공분산 같은 Gaussian 통계가 비슷하면, 같은 noise seed에 대한 diffusion sampling map도 비슷해진다.  
> 유한한 데이터에서는 noise scale이 재규격화되고, 저분산 방향은 과도하게 축소되며, split 간 차이는 특정 eigenmode와 noise 입력에 집중된다.

---

## 1. 주요 다이어그램 및 그림

### Figure 1 — 일관성 현상과 선형 Gaussian predictor

**결과**
- 서로 겹치지 않는 두 데이터 split으로 학습한 UNet과 DiT가 동일한 noise seed에서 매우 유사한 이미지를 생성한다.
- 서로 다른 architecture인 CNN과 DiT 사이에서도 같은 현상이 나타난다.
- 단순한 Gaussian linear theory, 즉 데이터의 평균과 공분산만 사용하는 Wiener filter가 실제 diffusion 출력의 상당 부분을 설명한다.
- 생성 결과 간 MSE는 생성 이미지와 training-set nearest neighbor 사이의 MSE보다 작다.

**인사이트**
- 이 일관성은 특정 학습 샘플을 외워서 생기는 현상이라기보다, 여러 split이 공유하는 **저차 Gaussian 통계**에서 이미 발생한다.
- diffusion model의 동일 seed 대응 관계는 고차원 비선형 구조를 학습하기 전부터 어느 정도 정해진다.

---

### Figure 2 — 유한 데이터가 noise를 재규격화하는 효과

**결과**
- Figure 2A는 이론의 흐름을 보여준다:  
  empirical covariance → deterministic equivalent → renormalized noise \(\sigma^2 \mapsto \kappa(\sigma^2)\) → denoiser와 sampling map의 expectation/variance 분석.
- \(\kappa(\sigma^2)\)는 항상 \(\sigma^2\)보다 크며, 데이터 수가 적거나 \(d/n\)이 클수록 차이가 커진다.
- 특히 낮은 noise scale에서 재규격화 효과가 강하다.
- 낮은 고유값을 갖는 covariance 방향에서 empirical denoiser가 population denoiser보다 더 강하게 shrink한다.

**인사이트**
- 유한 데이터는 단순히 covariance 추정에 오차를 추가하는 것이 아니라, 평균적으로 모델이 더 큰 noise를 본 것처럼 만든다.
- 그 결과 저분산·고주파·세부 묘사 방향이 noise로 간주되어 평균 얼굴이나 평균적인 구조 쪽으로 축소된다.

---

### Figure 3 — Denoiser 불일치의 세 가지 구조

#### A. 시각적 예시

**결과**
- 높은 noise에서는 split 간 차이가 얼굴의 전체 윤곽이나 저주파 구조에 나타난다.
- 낮은 noise에서는 차이가 반짝임, 질감, 세부 패턴 같은 고주파 영역으로 이동한다.

**인사이트**
- 어떤 spectral mode가 불안정한지는 noise scale에 따라 변한다.

#### B. Anisotropy

**결과**
- covariance eigenvalue가 재규격화된 noise \(\kappa\)와 비슷한 방향에서 denoiser variance가 가장 크다.
- 방향별 variance는
  \[
  \chi(\lambda,\kappa)=\frac{\lambda}{(\lambda+\kappa)^2}
  \]
  형태이며 \(\lambda=\kappa\)에서 최대가 된다.

**인사이트**
- 모든 방향이 동일하게 불안정한 것이 아니다.
- 현재 noise scale과 분산이 비슷한 eigenmode가 가장 학습하기 어렵고 split 간 차이가 크다.

#### C. Inhomogeneity

**결과**
- 입력 \(x\)가 데이터 분포의 높은 분산 방향으로 많이 벗어날수록 denoiser 예측의 variance가 커진다.
- 입력 위치에 따라 split 간 불일치 정도가 달라진다.

**인사이트**
- 같은 모델이라도 어떤 noise seed 또는 어떤 입력 위치를 사용하느냐에 따라 일관성이 달라진다.

#### D. Dataset size scaling

**결과**
- 전체 variance는 대체로 \(1/n\)에 비례해 감소한다.
- 단, 작은 데이터 영역에서는 \(\kappa\)의 재규격화 때문에 단순한 \(1/n\) 법칙에서 벗어난다.

**인사이트**
- 데이터가 많아질수록 split 간 차이는 줄어들지만, 모든 spectral band가 같은 속도로 안정화되지는 않는다.

---

### Figure 4 — 최종 diffusion sampling map의 유한 데이터 효과

#### A. Overshrinkage

**결과**
- empirical sampling map은 population sampling map보다 낮은 eigenmode의 크기를 더 작게 만든다.
- 생성 샘플의 저분산 방향 variance가 부족해진다.

**인사이트**
- 데이터가 부족하면 생성물이 평균적인 형태에 가까워지고, 질감·세부 구조가 부드러워진다.

#### B. Sampling-map anisotropy

**결과**
- sampling map의 split 간 MSE는 covariance의 상위 eigenmode에서 더 크게 나타난다.

**인사이트**
- denoiser의 anisotropy와 sampling trajectory 전체의 anisotropy가 연결되어 있다.

#### C. Sampling-map inhomogeneity

**결과**
- 초기 noise가 높은 variance eigenmode에 더 많이 정렬될수록 최종 생성 결과의 split 간 차이가 커진다.

**인사이트**
- 초기 latent/noise 공간도 완전히 isotropic하지 않다. 데이터 covariance가 noise 공간에 의미 있는 방향을 부여한다.

#### D. Eigenband별 sample-size 효과

**결과**
- 상위 eigenmode의 불일치는 비교적 적은 데이터에서도 빠르게 줄어든다.
- 낮은 eigenmode, 즉 세부 묘사와 관련된 방향은 더 많은 데이터가 필요하다.

**인사이트**
- “전체 MSE가 감소한다”는 사실만으로는 충분하지 않다. 어떤 주파수·spectral band가 안정화되었는지를 봐야 한다.

---

### Figure 5 — UNet과 DiT에 대한 검증

#### A–B. Memorization에서 renormalization으로의 전환

**결과**
- \(n\leq 1000\) 정도에서는 모델이 training sample을 재현하는 memorization 경향을 보인다.
- \(n\geq 3000\)에서는 training split과 control split의 nearest-neighbor 거리가 비슷해지며 generalization regime으로 들어간다.
- 이때 생성물은 점점 linear Gaussian predictor에 가까워진다.

**인사이트**
- RMT 이론은 주로 memorization regime이 아니라 **generalization/renormalization regime**을 설명한다.

#### C. 데이터 크기와 일관성

**결과**
- 데이터셋 크기가 커질수록 cross-split MSE가 감소한다.
- DiT가 동일한 데이터 크기에서 UNet보다 더 일관적인 경우가 관찰된다.

**인사이트**
- 아키텍처에 따라 절대적인 변동 크기는 다르지만, 데이터 covariance가 만드는 spectral 패턴은 공통적으로 나타난다.

#### D. 생성 variance의 overshrinkage

**결과**
- 제한된 데이터에서는 중간 및 낮은 eigenmode의 생성 variance가 감소한다.
- 데이터가 충분해지면 이 bias가 줄고 population spectrum에 가까워진다.

#### E. Eigenmode별 cross-split MSE

**결과**
- 데이터 증가에 따른 일관성 향상은 주로 상위 eigenmode에서 먼저 나타난다.
- 낮은 eigenmode의 불일치는 오랫동안 남는다.

#### F. Noise seed별 inhomogeneity

**결과**
- RMT가 예측한 초기 noise별 불일치와 실제 UNet 생성 결과의 차이가 양의 상관을 보인다.
- FFHQ64, \(n=30{,}000\)에서 Spearman correlation이 약 0.33으로 보고된다.
- 다만 deep network의 실제 변동 크기는 선형 이론보다 훨씬 크다.

**인사이트**
- RMT는 정확한 절대 오차 크기보다는 “어떤 seed와 어떤 방향에서 더 불안정한가”를 예측하는 데 유용하다.

---

## 2. 부록의 추가 결과

### Appendix A — 관련 연구 정리

**주요 내용**
- diffusion consistency/reproducibility 연구를 정리한다.
- Gaussian linear score와 hidden linear structure가 여러 noise scale에서 강력한 근사라는 점을 연결한다.
- memorization, generalization, creativity를 architecture와 inductive bias의 관점에서 논의한다.
- GAN/VAE와 달리 diffusion은 deterministic flow를 사용하므로 같은 noise seed가 비교적 고정된 의미를 갖는다고 설명한다.

**인사이트**
- diffusion의 reproducibility는 단순한 학습 안정성만이 아니라, score field가 데이터 분포에 의해 정해지고 latent coordinate의 회전 자유도가 상대적으로 적다는 점에서 비롯된다.

---

### Appendix B.1 — 다양한 데이터셋에서의 시각적 검증

**대상**
- FFHQ32/64
- CIFAR10/100
- LSUN bedroom/church

**결과**
- 서로 다른 split과 architecture에서 일관성이 반복적으로 관찰된다.
- linear Gaussian predictor도 출력의 전체 구조와 일부 시각적 특성을 설명한다.
- 고해상도 이미지에서는 linear predictor가 edge나 국소 세부 묘사를 충분히 설명하지 못한다.

**인사이트**
- Gaussian 설명은 보편적인 baseline이지만, 고해상도·고차 구조에는 nonlinear statistics가 추가로 필요하다.

---

### Appendix B.2 및 Figures 13–14 — 평균·공분산을 의도적으로 다르게 만든 split

**결과**
- 데이터를 PC 방향으로 stratify하여 평균 또는 covariance를 인위적으로 다르게 만들면 split 간 생성 이미지의 MSE가 크게 증가한다.
- 반대로 random i.i.d. split은 훨씬 높은 일관성을 보인다.
- CNN과 DiT 모두에서 같은 경향이 나타난다.

**인사이트**
- 일관성의 핵심 조건은 단순히 데이터가 non-overlapping이라는 사실이 아니라, 두 split이 **첫 두 모멘트(mean와 covariance)를 공유하는지**이다.
- 데이터 통계가 실제로 다르면 deep network도 같은 seed에 대해 일관된 결과를 내기 어렵다.

---

### Appendix B.3 — 선형 denoiser의 추가 검증

#### Figure 15

**결과**
- 이론이 예측한 input별 denoiser variance와 실제 두 split 사이의 MSE가 높은 상관을 보인다.
- 낮은 noise scale에서 이론이 특히 정확하다.
- 높은 noise에서는 empirical mean의 차이가 추가적인 오차를 만든다.

#### Figure 16

**결과**
- sample size가 커질수록 전체 sampling-map MSE가 감소한다.
- 큰 데이터에서는 대략 \(1/n\) 스케일을 보인다.
- 작은 데이터에서는 rank deficiency와 renormalization 때문에 감소가 더 느리다.
- 작은 양의 \(\sigma_0\)를 사용하면 생성 variance에 floor가 생겨 overshrinkage가 완화된다.

**인사이트**
- 이론에서 평균을 동일하다고 가정한 부분은 실제 실험에서 중요한 근사다. 특히 데이터가 적을수록 mean fluctuation도 고려해야 한다.

---

### Appendix B.4 — Deep network 추가 검증

#### Figures 17–21: nearest-neighbor 분석

**결과**
- 작은 데이터에서는 training-set nearest neighbor가 control-set보다 훨씬 가깝다.
- 데이터가 증가하면 두 거리가 비슷해진다.

**인사이트**
- 작은 데이터에서의 memorization과 충분한 데이터에서의 generalization을 구분하는 실험적 근거다.

#### Figure 22: 데이터 크기별 cross-split consistency

**결과**
- 데이터 수가 증가하면 CNN과 DiT 모두 cross-split MSE가 감소한다.
- DiT가 대체로 더 낮은 MSE를 보인다.

#### Figure 23: DNN 출력과 Gaussian predictor의 접근

**결과**
- 데이터셋 크기가 증가할수록 DNN 출력이 empirical/population Gaussian predictor에 가까워진다.
- 이는 generalization regime에서 Gaussian 구조가 실제 모델 출력의 중요한 부분을 차지한다는 점을 보여준다.

#### Figures 24–32: 여러 데이터셋의 overshrinkage와 anisotropy

**결과**
- AFHQ, FFHQ, CIFAR, LSUN 등에서 낮은·중간 eigenmode의 variance 부족이 반복된다.
- split 간 차이는 eigenmode별로 균일하지 않다.

**인사이트**
- 현상은 특정 얼굴 데이터셋에 국한되지 않고 다양한 이미지 데이터셋과 architecture에서 나타난다.

#### Figures 33–35: spatial inhomogeneity

**결과**
- RMT가 예측한 noise/input별 불일치와 실제 DNN의 cross-split deviation 사이에 상관이 존재한다.
- 그러나 실제 DNN에서는 비선형성 때문에 절대 변동량이 더 크다.

#### Figures 36–38: 초기 noise 공간의 anisotropy

**결과**
- top eigenmode의 noise를 증폭하면 얼굴 구조가 과도하게 강조되거나 artifact가 증가한다.
- top eigenmode의 noise를 줄이면 더 단순하고 균질한 얼굴이 생성된다.
- bottom eigenmode을 조작해도 시각적 영향은 상대적으로 작다.

**인사이트**
- 초기 Gaussian noise는 생성 과정에서 완전히 의미 없는 좌표가 아니다.
- 데이터 covariance의 eigenbasis에 따라 noise 방향별 영향력이 다르며, 특정 “좋은 seed” 또는 “나쁜 seed”가 나타날 수 있다.

---

## 3. 이론 증명 및 수치 방법 부록

### Appendix C — 증명과 유도

**핵심 결과**
1. **Deterministic equivalence**  
   empirical covariance를 population covariance와 \(\kappa\)가 적용된 deterministic matrix로 치환한다.

2. **Denoiser expectation**  
   \[
   \mathbb{E}[\hat D(x;\sigma)]
   \approx D_\Sigma(x;\sqrt{\kappa(\sigma^2)})
   \]
   즉, 유한 데이터는 평균적으로 더 큰 noise를 적용한 것과 같다.

3. **Denoiser variance**  
   variance가 다음 세 요소로 분해된다.
   - 방향별 anisotropy
   - 입력별 inhomogeneity
   - sample size와 noise에 따른 global scaling

4. **Fractional matrix power**  
   \(\hat\Sigma^{1/2}\)를 적분 형태로 표현하여 최종 sampling map까지 분석한다.

5. **Sampling-map variance**  
   최종 생성 결과의 variance도 방향과 초기 noise에 대해 factorized form을 갖는다.

**인사이트**
- 이 논문의 핵심 기여는 단일 denoising step뿐 아니라 전체 diffusion trajectory와 최종 샘플까지 같은 RMT 언어로 분석했다는 점이다.

---

### Appendix D — 수치 및 실험 세부사항

**주요 내용**
- \(\kappa(z)\)는 Silverstein self-consistency equation을 Newton method와 continuation 방식으로 계산한다.
- 무한 적분은 \(u=\tan\theta\) 변환과 Gauss–Legendre quadrature로 안정적으로 계산한다.
- 선형 실험에서는 empirical covariance의 eigendecomposition으로 Wiener sampling map을 직접 계산한다.
- 작은 데이터에서 covariance가 rank deficient가 될 수 있어 float64와 eigenvalue clipping이 필요하다.
- DNN 실험은 EDM preconditioning, 50,000 training steps, Heun sampler 등을 사용했다.

**인사이트**
- 이론식은 단순한 폐쇄형 계산만으로 끝나는 것이 아니라, \(\kappa\)와 fractional-power integral을 안정적으로 계산해야 실제 데이터와 비교할 수 있다.

---

### Appendix E — LLM 사용

**내용**
- 저자들은 RMT 도구와 fractional matrix power의 적분 항등식을 탐색하는 연구 보조로 LLM을 사용했다.
- 코드 작성과 논문 문장 다듬기에도 활용했다.
- 수학적 유도와 결과는 저자들이 검증했다고 명시한다.

---

## 4. 표(Table)에 대한 설명

본문과 제공된 부록에는 주요 결과를 요약하는 전통적인 수치 Table은 거의 없고, 결과가 주로 **Figures와 Proposition/Eq.** 형태로 제시된다.

따라서 이 논문에서 표에 해당하는 핵심 정보는 다음과 같이 이해할 수 있다.

- **Proposition 4.1:** 유한 데이터의 평균 효과 = noise renormalization
- **Proposition 4.2:** denoiser variance = anisotropy × inhomogeneity × scaling
- **Proposition 5.1:** 최종 sampling map의 평균도 overshrinkage
- **Proposition 5.2:** 최종 샘플 variance도 방향·초기 noise에 의존
- **Figure 5 및 Appendix B.4:** 이 예측이 UNet/DiT에서도 정성적으로 성립

---

## 최종 핵심 정리

1. **일관성의 출발점은 Gaussian 통계다.**  
   서로 다른 split도 평균과 covariance가 비슷하면 같은 seed에서 비슷한 결과를 낸다.

2. **유한 데이터는 effective noise를 키운다.**  
   \(\sigma^2\)가 \(\kappa(\sigma^2)>\sigma^2\)로 바뀌어 저분산 방향이 과도하게 축소된다.

3. **불일치는 균일하지 않다.**  
   현재 noise와 비슷한 분산을 갖는 eigenmode, 그리고 높은 분산 방향으로 벗어난 입력에서 불일치가 커진다.

4. **세부 묘사는 더 많은 데이터를 필요로 한다.**  
   상위·저주파 구조는 빨리 안정화되지만, 저분산·고주파 세부 방향은 늦게 안정화된다.

5. **Deep network에서도 패턴은 유지된다.**  
   DNN은 선형 이론보다 변동량이 크지만, overshrinkage, anisotropy, inhomogeneity의 구조는 공유한다.

---



The figures, diagrams, and appendices mainly serve three purposes: **illustrating the consistency phenomenon, validating the RMT theory in linear models, and testing whether the same structure appears in UNet and DiT models**.

> When different dataset splits share similar mean and covariance, diffusion models tend to produce similar outputs from the same noise seed.  
> With finite data, the effective noise level is increased, low-variance directions are overshrunk, and cross-split disagreement becomes concentrated in particular eigenmodes and noise inputs.

---

## 1. Main figures and diagrams

### Figure 1 — Consistency and the Gaussian linear predictor

**Results**
- UNets and DiTs trained on disjoint data splits generate visually similar images from the same noise seed.
- The effect also appears across architectures.
- A simple Gaussian linear predictor based only on the data mean and covariance explains a substantial part of the generated output.
- Cross-split image MSE is smaller than the distance between generated images and their nearest training examples.

**Insight**
- Consistency is not primarily caused by memorizing the same examples.
- It already emerges from shared low-order Gaussian statistics.

---

### Figure 2 — Finite data renormalize the noise scale

**Results**
- The theoretical roadmap is:
  empirical covariance → deterministic equivalent → renormalized noise \(\sigma^2\mapsto\kappa(\sigma^2)\) → expectation and variance of denoisers and sampling maps.
- The effective noise satisfies \(\kappa(\sigma^2)>\sigma^2\).
- The effect is strongest when the dataset is small relative to the data dimension and at low noise levels.
- Low-variance covariance directions are shrunk more strongly by the empirical denoiser.

**Insight**
- Finite data behave as if the model were trained with a larger effective noise level.
- As a result, fine details and low-variance directions are treated as noise and pulled toward the dataset mean.

---

### Figure 3 — Three components of denoiser disagreement

#### A. Visual examples

**Results**
- At high noise, split differences appear in global, low-frequency structures.
- At low noise, differences move toward textures, highlights, and high-frequency details.

**Insight**
- The unstable spectral modes change with the noise scale.

#### B. Anisotropy

**Results**
- Variance is largest for eigenmodes whose covariance eigenvalue is close to the renormalized noise \(\kappa\).
- The directional dependence is
  \[
  \chi(\lambda,\kappa)=\frac{\lambda}{(\lambda+\kappa)^2},
  \]
  which peaks at \(\lambda=\kappa\).

**Insight**
- Disagreement is not equally distributed across directions.
- Modes whose variance matches the current effective noise are the most uncertain.

#### C. Inhomogeneity

**Results**
- Denoiser variance is larger for inputs displaced along high-variance covariance directions.
- Different noise inputs can therefore have very different levels of cross-split disagreement.

**Insight**
- Consistency depends not only on the model and dataset, but also on the particular input location or noise seed.

#### D. Dataset-size scaling

**Results**
- Overall variance decreases roughly as \(1/n\) for large datasets.
- At small sample sizes, renormalization modifies this simple scaling.

**Insight**
- More data improve consistency, but different spectral bands stabilize at different rates.

---

### Figure 4 — Finite-data effects on the full sampling map

#### A. Overshrinkage

**Results**
- The empirical sampling map produces smaller amplitudes than the population map, especially in low-variance modes.
- Generated samples therefore have reduced variance in fine-detail directions.

**Insight**
- Limited data produce smoother, more average-looking outputs.

#### B. Sampling-map anisotropy

**Results**
- Cross-split MSE is larger in certain high-variance eigenmodes.

**Insight**
- The anisotropy of the final sample is inherited from the anisotropy of the denoiser and covariance estimation process.

#### C. Sampling-map inhomogeneity

**Results**
- Initial noise aligned with high-variance eigenmodes leads to larger disagreement between generated samples.

**Insight**
- The initial noise space is not effectively isotropic after being processed by the data covariance.

#### D. Eigenband-dependent sample complexity

**Results**
- High-variance modes become consistent relatively quickly.
- Low-variance modes require substantially more data.

**Insight**
- Overall MSE can hide the fact that fine details remain unstable even after global structure has become consistent.

---

### Figure 5 — Validation on UNet and DiT

#### A–B. Memorization versus renormalization

**Results**
- At small \(n\), especially \(n\leq 1000\), models tend to reproduce training examples.
- At larger \(n\), training-set and control-set nearest-neighbor distances become similar, indicating generalization.
- In this regime, DNN samples increasingly resemble the Gaussian linear predictor.

**Insight**
- The RMT theory is mainly intended for the generalization/renormalization regime, not the strong memorization regime.

#### C. Consistency versus dataset size

**Results**
- Cross-split MSE decreases with dataset size.
- DiT is often more consistent than UNet at the same dataset size.

#### D. Generation variance

**Results**
- Limited data reduce the generated variance in middle- and low-spectrum modes.
- This bias decreases as the dataset becomes larger.

#### E. Eigenmode-specific disagreement

**Results**
- Improvements with increasing dataset size occur first in the top eigenmodes.
- Low-variance modes remain inconsistent for longer.

#### F. Seed-wise inhomogeneity

**Results**
- RMT predictions correlate with observed seed-wise disagreement in deep networks.
- For FFHQ64 with \(n=30{,}000\), a Spearman correlation of about 0.33 is reported.
- The absolute deviations in DNNs are much larger than those predicted by the linear theory.

**Insight**
- RMT is especially useful for predicting **where** disagreement occurs, even when it underestimates **how large** the disagreement is.

---

## 2. Additional appendix results

### Appendix A — Related work

**Main points**
- Reviews prior work on diffusion consistency and reproducibility.
- Connects diffusion consistency to the hidden Gaussian-linear structure of learned scores.
- Discusses memorization, generalization, and creativity through architectural inductive biases.
- Explains why deterministic diffusion sampling has more stable seed-to-output correspondence than GANs or VAEs.

**Insight**
- Diffusion reproducibility is related to the fact that the score field is tied to the data distribution, while GAN/VAE latent spaces have a larger rotational ambiguity.

---

### Appendix B.1 — Additional datasets

**Datasets**
- FFHQ32/64
- CIFAR10/100
- LSUN bedroom/church

**Results**
- Similar cross-split consistency appears across datasets and architectures.
- The Gaussian predictor captures global structure but misses some edges and local high-frequency details, especially at higher resolution.

**Insight**
- The Gaussian theory is a strong baseline, but nonlinear statistics are needed for detailed image structure.

---

### Appendix B.2 and Figures 13–14 — Deliberately mismatched moments

**Results**
- When splits are constructed to have different means or covariances along a principal component, generated samples become much less consistent.
- Random i.i.d. splits remain substantially more consistent.
- The effect appears in both CNN and DiT models.

**Insight**
- Non-overlap alone does not determine consistency. The crucial condition is whether the splits share the same first two moments.

---

### Appendix B.3 — Additional linear-model validation

#### Figure 15

**Results**
- The theoretical input-dependent denoiser variance agrees well with empirical cross-split MSE.
- Agreement is strongest at lower noise levels.
- At high noise, differences in empirical means introduce additional variation.

#### Figure 16

**Results**
- Sampling-map MSE decreases with dataset size and approaches approximately \(1/n\) at large \(n\).
- Small datasets show slower or irregular scaling because of rank deficiency and renormalization.
- A small positive final noise \(\sigma_0\) creates a variance floor and reduces the apparent overshrinkage.

**Insight**
- Assuming identical means isolates covariance effects, but mean fluctuations can matter in realistic finite datasets.

---

### Appendix B.4 — Further deep-network experiments

#### Figures 17–21: Nearest-neighbor analysis

**Results**
- For small datasets, generated samples are closer to the training split than to the control split.
- As the dataset grows, the two distances become similar.

**Insight**
- These experiments provide evidence for a transition from memorization to generalization.

#### Figure 22: Dataset-size scaling

**Results**
- Both CNN and DiT become more consistent as the dataset grows.
- DiT generally shows lower cross-split MSE.

#### Figure 23: DNNs approach the Gaussian predictor

**Results**
- With increasing dataset size, DNN outputs move closer to the empirical and population Gaussian predictors.

**Insight**
- In the generalization regime, the shared Gaussian structure becomes an increasingly important component of the learned model.

#### Figures 24–32: Overshrinkage and anisotropy

**Results**
- The same spectral patterns appear across FFHQ, AFHQ, CIFAR, and LSUN.
- Middle- and low-variance modes show reduced generation variance and slower consistency improvement.

#### Figures 33–35: Spatial inhomogeneity

**Results**
- RMT predictions correlate with pointwise cross-split deviations in DNNs.
- Nonlinear networks exhibit larger absolute variability than the linear theory predicts.

#### Figures 36–38: Anisotropic initial-noise space

**Results**
- Amplifying top-eigenmode noise can exaggerate dominant facial structure and create artifacts.
- Suppressing those modes produces simpler, more homogeneous outputs.
- Manipulating bottom eigenmodes has relatively little perceptual effect.

**Insight**
- Initial Gaussian noise is not semantically uniform after passing through the learned generative map. Its alignment with the data covariance eigenbasis affects output quality and consistency.

---

## 3. Theory and numerical-method appendices

### Appendix C — Proofs and derivations

The appendix establishes:

1. **Deterministic equivalence**  
   Empirical covariance matrices can be replaced by deterministic population-level expressions involving \(\kappa\).

2. **Denoiser expectation**  
   Finite-data denoising is equivalent, in expectation, to population denoising at a larger effective noise level.

3. **Denoiser variance**  
   Variance factorizes into:
   - directional anisotropy,
   - input-dependent inhomogeneity,
   - sample-size/noise scaling.

4. **Fractional matrix powers**  
   Integral representations make it possible to analyze \(\hat\Sigma^{1/2}\) and the full sampling map.

5. **Sampling-map variance**  
   Final-sample fluctuations inherit analogous dependence on output direction and initial noise.

**Insight**
- The main theoretical contribution is extending the analysis from a single denoising step to the entire diffusion sampling trajectory.

---

### Appendix D — Numerical and experimental details

**Main points**
- \(\kappa(z)\) is solved from the Silverstein equation using Newton’s method and continuation.
- Infinite integrals are evaluated using a tangent transformation and Gauss–Legendre quadrature.
- Linear sampling maps are computed through covariance eigendecompositions.
- Float64 precision and eigenvalue clipping are important when the empirical covariance is rank deficient.
- Deep models use EDM preconditioning, 50,000 training steps, and Heun sampling.

**Insight**
- The theory requires careful numerical treatment, especially for small datasets and fractional matrix-power integrals.

---

### Appendix E — Use of LLMs

**Content**
- LLMs were used as research assistants for locating RMT tools and integral identities.
- They also assisted with code generation and writing.
- The authors state that the mathematical derivations and results were independently verified.

---

## 4. Tables and proposition-style summaries

The paper contains few conventional numerical tables. Its main results are instead summarized by propositions and figures:

- **Proposition 4.1:** finite data renormalize the effective noise level.
- **Proposition 4.2:** denoiser variance decomposes into anisotropy, inhomogeneity, and scaling.
- **Proposition 5.1:** the final sampling map exhibits finite-data overshrinkage.
- **Proposition 5.2:** final-sample variance depends on output direction and initial noise.
- **Figures 5 and B.4:** these structures extend qualitatively to UNet and DiT.

---

## Overall takeaway

1. **Consistency begins with shared Gaussian statistics.**
2. **Finite datasets increase the effective noise level.**
3. **Low-variance and high-frequency modes are overshrunk.**
4. **Disagreement is anisotropic and input-dependent.**
5. **Fine details require more data to become consistent.**
6. **Deep networks show the same qualitative structure, although nonlinearities amplify the magnitude of the variation.**

<br/>
# refer format:



### BibTeX

```bibtex
@inproceedings{wang2026random,
  author    = {Wang, Binxu and Zavatone-Veth, Jacob A. and Pehlevan, Cengiz},
  title     = {A Random Matrix Theory Perspective on the Consistency of Diffusion Models},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  publisher = {PMLR},
  address   = {Seoul, South Korea},
  year      = {2026}
}
```

### Chicago 스타일 참고문헌

Wang, Binxu, Jacob A. Zavatone-Veth, and Cengiz Pehlevan. “A Random Matrix Theory Perspective on the Consistency of Diffusion Models.” In *Proceedings of the 43rd International Conference on Machine Learning*. Vol. 306 of *Proceedings of Machine Learning Research*. Seoul, South Korea: PMLR, 2026.

### Chicago 스타일 각주 예시

1. Binxu Wang, Jacob A. Zavatone-Veth, and Cengiz Pehlevan, “A Random Matrix Theory Perspective on the Consistency of Diffusion Models,” in *Proceedings of the 43rd International Conference on Machine Learning*, vol. 306, *Proceedings of Machine Learning Research* (Seoul, South Korea: PMLR, 2026).


