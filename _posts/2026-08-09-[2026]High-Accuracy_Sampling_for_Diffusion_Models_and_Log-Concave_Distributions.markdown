---
layout: post
title:  "[2026]High-Accuracy Sampling for Diffusion Models and Log-Concave Distributions"
date:   2026-08-09 17:47:18 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 디퓨전 모델에서 확률밀도 대신 로그밀도의 기울기인 스코어함수만 사용해서 확률분포 정확하게 샘플링하는 방법 제안  


짧은 요약(Abstract) :


이 논문은 **확률밀도 자체를 계산하지 않고, 로그밀도의 기울기인 score 함수만 이용해 확률분포를 매우 정확하게 샘플링하는 방법**을 제안합니다.

핵심은 다음과 같습니다.

- 목표 오차를 \(\delta\)라고 할 때, 기존 확산 모델 샘플러들은 보통 \(1/\delta\) 또는 \(1/\delta^2\)에 비례하는 많은 반복이 필요했습니다.
- 이 논문은 **First-Order Rejection Sampling (FORS)**라는 방법을 사용해, 필요한 반복 횟수를 \(\mathrm{polylog}(1/\delta)\), 즉 \(\log(1/\delta)\)의 다항식 수준으로 줄입니다.
- 따라서 \(\delta\)가 매우 작아져도 샘플링 복잡도가 비교적 천천히 증가합니다. 이것이 논문에서 말하는 **high-accuracy sampling**입니다.
- score 함수가 정확하지 않아도 되며, \(L^2\) 의미에서 대략 \(\widetilde O(\delta)\) 정도의 오차를 허용합니다. 최종 샘플의 오차는 목표 오차 \(\delta\)와 score 추정 오차의 영향을 함께 받습니다.

데이터 분포에 대한 조건에 따른 복잡도는 다음과 같습니다.

1. **최소한의 가정**  
   데이터 분포가 유한한 2차 모멘트만 가지면, 복잡도는 대략  
   \[
   \widetilde O\bigl(d\,\mathrm{polylog}(1/\delta)\bigr)
   \]
   입니다. 여기서 \(d\)는 데이터 차원이고, \(\widetilde O\)는 추가적인 로그 항을 생략한 표기입니다.

2. **score 함수가 비균일하게 \(L\)-Lipschitz인 경우**  
   복잡도는 대략  
   \[
   \widetilde O\bigl(\sqrt{dL}\,\mathrm{polylog}(1/\delta)\bigr)
   \]
   로 줄어듭니다. 즉, 차원 \(d\)에 대해 선형이 아니라 제곱근 수준의 의존성을 얻습니다.

3. **데이터가 저차원 구조를 가지는 경우**  
   데이터가 고차원 공간에 놓여 있더라도 실제로는 낮은 intrinsic dimension \(d^\star\)를 가진다면, 복잡도는  
   \[
   \widetilde O\bigl(d^\star\,\mathrm{polylog}(1/\delta)\bigr)
   \]
   수준까지 낮아집니다.

또한 같은 아이디어를 이용해, **로그-오목(log-concave) 분포**에서도 함수값이나 밀도값을 직접 계산하지 않고 **gradient 평가만으로** \(\mathrm{polylog}(1/\delta)\) 복잡도의 고정밀 샘플러를 제시합니다.

한마디로, 이 논문의 주요 기여는 **score 평가만으로도 확산 모델 샘플링을 기존의 다항식 정확도 의존성에서 로그 수준의 정확도 의존성으로 개선했다는 것**입니다.

---




This paper proposes a method for **high-accuracy sampling using only score-function evaluations**, without evaluating the probability density itself.

The main ideas are:

- Let \(\delta\) be the target sampling error. Previous diffusion samplers often required a number of steps scaling polynomially with \(1/\delta\), such as \(1/\delta\) or \(1/\delta^2\).
- The paper introduces **First-Order Rejection Sampling (FORS)**, which reduces the dependence on accuracy to \(\mathrm{polylog}(1/\delta)\).
- Thus, even when \(\delta\) is extremely small, the number of iterations grows only polylogarithmically. This is what the paper calls a **high-accuracy guarantee**.
- The score estimates do not need to be exact. The method tolerates score errors that are approximately \(\widetilde O(\delta)\) in \(L^2\), with the final sampling error depending on both \(\delta\) and the score-estimation error.

The complexity depends on the structure of the data distribution:

1. **Minimal assumptions**  
   If the data distribution only has a finite second moment, the complexity is approximately  
   \[
   \widetilde O\bigl(d\,\mathrm{polylog}(1/\delta)\bigr),
   \]
   where \(d\) is the ambient dimension.

2. **Non-uniform \(L\)-Lipschitz score condition**  
   The complexity improves to roughly  
   \[
   \widetilde O\bigl(\sqrt{dL}\,\mathrm{polylog}(1/\delta)\bigr).
   \]

3. **Low intrinsic dimension**  
   If the data has intrinsic dimension \(d^\star\), the complexity can be reduced to approximately  
   \[
   \widetilde O\bigl(d^\star\,\mathrm{polylog}(1/\delta)\bigr).
   \]

The paper also applies the same framework to **log-concave distributions**, obtaining the first sampler with polylogarithmic dependence on \(1/\delta\) that requires only gradient evaluations, rather than direct density evaluations.

In short, the central contribution is showing that **diffusion-model sampling can achieve extremely high accuracy with only a polylogarithmic dependence on the inverse error, using score evaluations alone**.


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



### 1. 이 논문이 해결하려는 문제

확산 모델(diffusion model)은 데이터 분포를 직접 학습하는 대신, 확산 과정 중 각 시점의 **score function**

\[
s_t^*(x)=\nabla \log p_t(x)
\]

을 학습합니다. 기존 DDPM은 이 score를 이용해 한 단계씩 역방향 샘플링하지만, 각 단계를 가우시안으로 근사하기 때문에 목표 오차 \(\delta\)를 매우 작게 만들면 필요한 단계 수가 대체로 \(1/\delta\) 또는 \(1/\delta^2\)까지 증가합니다.

이 논문의 목표는 다음과 같습니다.

- 확률밀도값이나 log-density를 직접 계산하지 않고
- score, 즉 gradient 평가만 사용하며
- 오차 \(\delta\)에 대해 **polylogarithmic**, 즉 \(\mathrm{polylog}(1/\delta)\) 복잡도로
- 매우 정확한 샘플을 생성하는 것

---

### 2. 핵심 방법: First-Order Rejection Sampling (FORS)

논문의 핵심 아이디어는 **gradient만으로 rejection sampling을 구현하는 것**입니다.

일반적인 rejection sampling에서는 다음과 같은 확률을 계산해야 합니다.

\[
\Pr(\text{accept }x)\propto e^{w(x)}.
\]

하지만 \(w(x)\) 자체를 계산하려면 log-density가 필요합니다. 논문은 \(w(x)\)를 gradient의 적분으로 표현하고, 이를 확률적으로 추정합니다.

예를 들어,

\[
f(x)-f(x_0)
=
\int_0^1
\langle x-x_0,\nabla f(x_0+r(x-x_0))\rangle dr.
\]

따라서 \(f(x)\)를 직접 계산하지 않고도, 임의의 \(r\sim\mathrm{Unif}[0,1]\)를 뽑아 gradient를 평가하면 \(f(x)\) 차이의 unbiased estimator를 만들 수 있습니다.

그 다음 논문은 **Bernoulli factory** 아이디어를 사용합니다. 즉, \(w(x)\)의 unbiased estimator만 주어져도 \(e^{w(x)}\)에 비례하는 acceptance decision을 생성할 수 있도록 합니다.

FORS의 주요 절차는 다음과 같습니다.

1. 제안분포 \(q(x)\)에서 후보 \(x\)를 샘플링합니다.
2. \(J\sim\mathrm{Poisson}(2B)\)를 샘플링합니다.
3. \(w(x)\)의 unbiased estimator \(W_1,\ldots,W_J\)를 생성합니다.
4. 다음 확률로 후보를 accept합니다.

\[
\prod_{j=1}^{J}\frac{B+W_j}{2B}.
\]

여기서 \(W_j\in[-B,B]\)가 되도록 clipping을 적용합니다.

이 방식은 density나 log-density를 직접 계산하지 않고도, rejection sampling과 같은 보정 효과를 구현합니다. \(B\)가 상수이면 필요한 gradient query 수는 높은 확률로 \(\mathrm{polylog}(1/\delta)\) 수준이 됩니다.

---

### 3. 확산 모델에 적용하는 방식

확산 모델의 역방향 전이분포는 다음과 같은 Gaussian tilt 형태입니다.

\[
\rho_t(x\mid x')
\propto
p_t(x)
\exp\left(
-\frac{\|x-\alpha_t^{-1}x'\|^2}{2\eta_t}
\right).
\]

즉, 가우시안 분포에 \(p_t(x)\)라는 density tilt가 곱해진 형태입니다. 논문은 이 분포를 FORS로 직접 샘플링합니다.

각 역방향 단계에서:

1. 현재 \(X_{t+1}\)으로부터 가우시안 제안분포를 만듭니다.
2. 학습된 score \(s_t(x)\)를 여러 지점에서 평가합니다.
3. score의 경로 적분으로 실제 Gaussian tilt와 제안분포의 차이를 추정합니다.
4. FORS rejection correction을 적용해 \(X_t\)를 생성합니다.
5. \(t=T-1,\ldots,1\)까지 반복합니다.

기존 DDPM이

\[
\log p_t(x)
\approx
\log p_t(x')
+
\langle s_t(x'),x-x'\rangle
\]

처럼 한 지점의 1차 근사만 사용하는 것과 달리, 이 논문은 여러 중간 지점에서 score를 평가하여 경로 적분 형태로 보정합니다.

---

### 4. 사용되는 모델과 학습 방식

이 논문은 새로운 신경망 아키텍처를 제안하지 않습니다.

- U-Net, Transformer 등 기존 score network를 사용할 수 있습니다.
- 네트워크는 각 noise level \(t\)에서 score \(s_t(x)\approx\nabla\log p_t(x)\)를 예측합니다.
- 일반적인 denoising score matching 또는 diffusion model training을 사용할 수 있습니다.
- 핵심 개선은 **학습 단계가 아니라 샘플링 단계의 알고리즘**에 있습니다.

논문은 score가 다음과 같은 평균제곱 오차를 가진다고 가정합니다.

\[
\varepsilon_{t,\mathrm{score}}^2
=
\mathbb E_{X_t\sim p_t}
\left[
\|s_t(X_t)-s_t^*(X_t)\|^2
\right].
\]

따라서 score가 완벽하지 않아도 샘플링 오차를 분석할 수 있습니다.

---

### 5. 주요 샘플러 변형

#### (1) Simple method

가우시안 부분만을 제안분포로 사용합니다.

- 데이터 분포에 특별한 구조를 거의 가정하지 않습니다.
- 유한한 2차 모멘트만 있으면 됩니다.
- 차원 \(d\)에 대해 대략

\[
O\!\left(
d\log^2\frac{M}{\delta}
+
\log^3\frac{M}{\delta}
\right)
\]

queries를 사용합니다.

이는 기존의 \(1/\delta\) 또는 \(1/\delta^2\) 의존성을 \(\mathrm{polylog}(1/\delta)\)로 줄이는 결과입니다.

#### (2) DDPM-like method

score를 이용해 더 정교한 가우시안 제안분포를 구성합니다.

\[
q_t
=
\mathcal N\left(
\alpha_t^{-1}X_{t+1}
+\eta_t s_t(\alpha_t^{-1}X_{t+1}),
\eta_t I
\right).
\]

score가 높은 확률 영역에서 Lipschitz하다는 조건하에 복잡도가 대략

\[
\widetilde O\left(
\sqrt{dL}\,
\mathrm{polylog}\frac{1}{\delta}
\right)
\]

까지 감소합니다.

#### (3) Intrinsic-dimension method

데이터가 주변 공간 \(\mathbb R^d\) 전체에 퍼져 있지 않고 저차원 구조를 가진 경우, shifted score

\[
\widetilde s_t(x)=s_t(x)+\frac{x}{\sigma_t^2}
\]

를 사용합니다.

이 경우 복잡도가 ambient dimension \(d\)가 아니라 intrinsic dimension \(d^\star\)에 의존합니다.

\[
O\!\left(
d^\star\log^2\frac{M}{\delta}
+
\log^3\frac{M}{\delta}
\right).
\]

---

### 6. 로그-오목 분포 샘플링에도 적용

논문은 diffusion model뿐 아니라 로그-오목 분포

\[
\mu(x)\propto e^{-f(x)}
\]

샘플링에도 FORS를 적용합니다.

기존 proximal sampler는 다음 restricted Gaussian oracle(RGO)를 필요로 합니다.

\[
\mathrm{RGO}_{f,\eta,y}(x)
\propto
\exp\left(
-f(x)-\frac{\|y-x\|^2}{2\eta}
\right).
\]

이 분포 역시 Gaussian tilt이므로, FORS로 샘플링할 수 있습니다. 결과적으로 \(f(x)\)의 함수값이나 density값 없이 \(\nabla f(x)\)만 사용하여 높은 정확도의 로그-오목 샘플링이 가능합니다.

---

### 7. 핵심 요약

- **특별한 신경망 구조:** 제안하지 않음.
- **학습 데이터:** 기존 diffusion model과 동일하게 데이터에 Gaussian noise를 추가하여 score를 학습.
- **학습 기법:** 일반적인 score matching/diffusion training 사용 가능.
- **핵심 신규 기법:** gradient만으로 rejection sampling을 구현하는 FORS.
- **샘플링 방식:** Gaussian proposal + score 기반 경로 적분 + rejection correction.
- **장점:** 정확도 \(\delta\)에 대한 query 복잡도를 기존 다항식 수준에서 \(\mathrm{polylog}(1/\delta)\)로 개선.
- **추가 장점:** Lipschitz 조건이나 저차원 구조를 이용하면 차원 의존성도 줄일 수 있음.
- **주의점:** 한 샘플링 단계마다 여러 번 score를 평가해야 하므로, 이론적인 단계 수는 적어도 실제 wall-clock 비용이 항상 더 낮다고 보장되지는 않습니다.

---




## 1. Problem addressed by the paper

Diffusion models learn the score function

\[
s_t^*(x)=\nabla\log p_t(x)
\]

at different noise levels instead of directly learning the data density.

Standard DDPM sampling approximates each reverse transition with a Gaussian distribution. Because this approximation introduces discretization error, obtaining a very small target error \(\delta\) usually requires polynomial dependence on \(1/\delta\), such as \(1/\delta\) or \(1/\delta^2\).

The paper aims to construct samplers that:

- use only score or gradient evaluations,
- do not require density or log-density evaluations,
- achieve very small error \(\delta\),
- and have only \(\mathrm{polylog}(1/\delta)\) query complexity.

---

## 2. Main idea: First-Order Rejection Sampling

The central technique is called **First-Order Rejection Sampling (FORS)**.

Ordinary rejection sampling requires an acceptance probability proportional to

\[
e^{w(x)}.
\]

However, evaluating \(w(x)\) usually requires the function value or log-density. The paper instead writes function differences as path integrals of gradients:

\[
f(x)-f(x_0)
=
\int_0^1
\langle x-x_0,
\nabla f(x_0+r(x-x_0))
\rangle dr.
\]

By sampling a random point along this path, one can construct an unbiased estimator of the function difference using only gradient evaluations.

FORS then uses a Bernoulli-factory construction. Given unbiased random estimators \(W_j\in[-B,B]\), it accepts a proposal \(x\) with probability

\[
\prod_{j=1}^{J}\frac{B+W_j}{2B},
\qquad
J\sim\mathrm{Poisson}(2B).
\]

This produces the same exponential tilting effect as rejection sampling without explicitly computing the density or log-density.

When \(B\) is constant, the number of gradient queries is polylogarithmic in \(1/\delta\), with high probability.

---

## 3. Applying FORS to diffusion sampling

The reverse diffusion transition has the form

\[
\rho_t(x\mid x')
\propto
p_t(x)
\exp\left(
-\frac{\|x-\alpha_t^{-1}x'\|^2}{2\eta_t}
\right).
\]

This is a **Gaussian tilt**: a Gaussian factor multiplied by the density \(p_t(x)\).

At each reverse step, the proposed method:

1. Constructs a Gaussian proposal distribution from \(X_{t+1}\).
2. Evaluates the learned score \(s_t(x)\) at several points.
3. Uses a path-integral representation to estimate the correction between the Gaussian proposal and the true reverse transition.
4. Applies the FORS rejection correction.
5. Repeats the procedure from \(t=T-1\) down to \(t=1\).

Unlike standard DDPM, which uses a local first-order approximation at a single point, this method uses score evaluations along a path and corrects the resulting approximation through rejection sampling.

---

## 4. Model architecture and training

The paper does **not** introduce a new neural-network architecture.

Existing score networks such as U-Nets or Transformers can be used. The model is trained to estimate

\[
s_t^*(x)=\nabla\log p_t(x)
\]

at each noise level, using standard denoising score matching or diffusion-model training.

The main contribution is therefore not a new training objective, dataset, or architecture. It is a new **sampling algorithm** that uses the learned score more accurately.

The analysis allows imperfect score estimates with mean-squared error

\[
\varepsilon_{t,\mathrm{score}}^2
=
\mathbb E_{X_t\sim p_t}
\left[
\|s_t(X_t)-s_t^*(X_t)\|^2
\right].
\]

Thus, the method remains robust when the score network is approximate.

---

## 5. Main sampler variants

### Simple method

This method uses the Gaussian part of the reverse transition as the proposal.

It requires very mild assumptions, essentially a finite second moment of the data distribution, and achieves a complexity roughly of

\[
O\!\left(
d\log^2\frac{M}{\delta}
+
\log^3\frac{M}{\delta}
\right).
\]

The key improvement is replacing polynomial dependence on \(1/\delta\) with polylogarithmic dependence.

### DDPM-like method

This method uses a score-informed Gaussian proposal:

\[
q_t
=
\mathcal N\left(
\alpha_t^{-1}X_{t+1}
+\eta_t s_t(\alpha_t^{-1}X_{t+1}),
\eta_t I
\right).
\]

Under a non-uniform Lipschitz condition on the score, the complexity becomes approximately

\[
\widetilde O\left(
\sqrt{dL}\,
\mathrm{polylog}\frac{1}{\delta}
\right).
\]

### Intrinsic-dimension method

For data concentrated near a low-dimensional structure, the method uses the shifted score

\[
\widetilde s_t(x)
=
s_t(x)+\frac{x}{\sigma_t^2}.
\]

Then the complexity depends on the intrinsic dimension \(d^\star\), rather than the ambient dimension \(d\):

\[
O\!\left(
d^\star\log^2\frac{M}{\delta}
+
\log^3\frac{M}{\delta}
\right).
\]

---

## 6. Application to log-concave sampling

The same Gaussian-tilt machinery can be applied to sampling from a log-concave distribution

\[
\mu(x)\propto e^{-f(x)}.
\]

The proximal sampler requires a restricted Gaussian oracle:

\[
\mathrm{RGO}_{f,\eta,y}(x)
\propto
\exp\left(
-f(x)-\frac{\|y-x\|^2}{2\eta}
\right).
\]

Since this is also a Gaussian tilt, FORS can implement it using only \(\nabla f(x)\), without evaluating \(f(x)\) itself.

Therefore, the paper provides high-accuracy sampling guarantees for log-concave and more general isoperimetric distributions using first-order information only.

---

## 7. Key takeaway

- **New neural architecture:** No.
- **Special training dataset:** No.
- **Training method:** Standard score-matching or diffusion training can be used.
- **Main innovation:** First-Order Rejection Sampling.
- **Sampling mechanism:** Gaussian proposal + path-integrated score evaluations + rejection correction.
- **Main benefit:** Polylogarithmic dependence on \(1/\delta\).
- **Additional benefit:** Improved dependence on dimension under smoothness or low-dimensional structure.
- **Practical limitation:** Each sampling step may require multiple score evaluations, so fewer theoretical steps do not automatically imply lower wall-clock cost.


<br/>
# Results




### 1. 논문의 핵심 결과

이 논문은 **정규화되지 않은 밀도값 \(f(x)\) 자체를 계산하지 않고, score 또는 log-density의 gradient만 사용해 고정밀 샘플링**하는 방법을 제안한다. 핵심 알고리즘은 **FORS(First-Order Rejection Sampling)**이다.

기존 방법들은 목표 오차 \(\delta\)를 매우 작게 만들 때 반복 횟수가 대체로 \(1/\delta\), \(1/\sqrt{\delta}\) 또는 그와 유사하게 증가했다. 반면 이 논문은 score가 \(L^2\) 의미에서 충분히 정확하다는 조건하에 **\(\log(1/\delta)\)의 다항식 정도만 증가하는 고정밀 보장**을 제시한다.

여기서 \(\widetilde O(\cdot)\)는 차원이나 오차에 대한 로그 인자를 생략한 표기이다.

---

### 2. 확산 모델 샘플링 결과

논문은 실제 데이터 분포 \(p_{\mathrm{data}}\)에서 직접 샘플을 생성하기보다, 먼저 작은 노이즈가 추가된 **early-stopped 분포 \(p_1\)**에서 샘플링한 뒤 원래 분포와의 차이를 제어한다.

#### 결과 A: 최소 가정

데이터 분포가 유한한 2차 모멘트만 가진다고 가정한다.

- 데이터 차원: \(d\)
- \(M_2^2=\mathbb E\|X_0\|^2\)
- 출력 오차: bounded-Lipschitz metric 기준
- 필요한 score query 수:

\[
\widetilde O\!\left(
\max\{d,\log(1/\delta)\}
\log^2\!\frac{d+M_2^2}{\delta^2}
\right).
\]

보다 구체적으로, 논문은 다음 형태의 보장을 제시한다.

\[
D_{\mathrm{BL}}(p_{\mathrm{data}},\widehat p_1)^2
\lesssim
\delta^2+
\sum_t \eta_t\varepsilon_{t,\mathrm{score}}^2.
\]

즉,

- 알고리즘 자체의 오차는 \(\delta\)까지 줄일 수 있고,
- score 추정 오차는 \(\sum_t\eta_t\varepsilon_{t,\mathrm{score}}^2\) 형태로 추가된다.

특징은 데이터 분포에 대한 가정이 사실상 **유한한 2차 모멘트뿐**이라는 점이다.

#### 결과 B: 비균일 Lipschitz score

score의 Hessian 또는 Jacobian이 모든 점에서 균일하게 매끄럽다고 가정하지 않고, 대부분의 확률 질량이 있는 영역에서만 매끄럽다고 가정한다.

비균일 Lipschitz 상수를 \(L_\delta\)라 하면 필요한 단계 수는 대략

\[
\widetilde O\!\left(
\max\{\sqrt{dL_\delta},L_\delta\}
\log\frac{d+M_2^2}{\delta^2}
\right).
\]

이 경우 total variation metric에 대해

\[
D_{\mathrm{TV}}(p_1,\widehat p_1)^2
\lesssim
\delta^2+
\sqrt{\frac d{L_\delta}}
\sum_t\eta_t\varepsilon_{t,\mathrm{score}}^2
\]

형태의 보장을 얻는다.

따라서 기본적인 \(d\) 의존성이 \(\sqrt d\) 수준까지 개선될 수 있다. 다만 score 오차에 대한 민감도는 최소 가정 결과보다 나빠진다.

#### 결과 C: 저차원 또는 intrinsic dimension

데이터가 \(d\)차원 공간에 놓여 있어도 실제 지지집합의 구조적 복잡도가 낮을 수 있다. 이를 covering number로 측정해 intrinsic dimension \(d^\star\)를 정의한다.

이때 query 복잡도는

\[
\widetilde O\!\left(
\max\{d^\star,\log(d/\delta)\}
\log^2\!\frac{d+M_2^2}{\delta^2}
\right)
\]

까지 줄어든다.

오차는 대략

\[
D_{\mathrm{BL}}(p_{\mathrm{data}},\widehat p_1)^2
\lesssim
\delta^2+
\frac d{d^\star}
\sum_t\eta_t\varepsilon_{t,\mathrm{score}}^2
\]

형태이다.

즉, iteration 수는 intrinsic dimension에 비례하지만, score 오차 항에는 \(d/d^\star\)가 추가로 들어간다.

---

### 3. 사용한 메트릭

논문은 상황에 따라 서로 다른 분포 거리 또는 발산을 사용한다.

- **Bounded-Lipschitz metric \(D_{\mathrm{BL}}\)**  
  최소 가정에서 사용한다. 약한 수렴을 측정하며, 매우 일반적인 데이터 분포에 적합하다.

- **Total variation \(D_{\mathrm{TV}}\)**  
  두 분포의 확률 질량 차이를 직접 측정한다. Lipschitz score 조건하에서 사용된다.

- **KL divergence \(D_{\mathrm{KL}}\)**  
  확산 과정의 각 역방향 전이 오차를 합산하거나 score 오차의 영향을 분석할 때 사용된다.

- **Hellinger distance \(D_{\mathrm H}\)**  
  Lipschitz score 결과에서 transition kernel 오차를 누적하는 데 사용된다.

- **Chi-squared divergence \(D_{\chi^2}\)**  
  Gaussian tilt 및 log-concave 샘플링 분석에서 사용된다.

- **Wasserstein distance \(W_2\)**  
  early stopping을 통해 \(p_1\)과 원래 데이터 분포 \(p_{\mathrm{data}}\) 사이의 노이즈 오차를 제어할 때 사용된다.

---

### 4. 기존 방법 및 경쟁 결과와의 비교

#### DDPM 계열

기존 DDPM 이론은 매우 약한 데이터 가정에서 성립하지만, 고정밀도 측면에서는 한계가 있다.

- 초기 이론: 대략 \(\widetilde O(d/\delta^2)\)
- 후속 개선: 대략 \(\widetilde O(d/\delta)\)
- Li & Cai: 대략 \(1/\sqrt{\delta}\) 의존성
- DDPM 자체에 대해서는 \(\Omega(1/\delta)\) lower bound가 알려져 있음

따라서 표준 DDPM의 단순한 시간 이산화만으로는 일반적으로 polylogarithmic accuracy를 얻기 어렵다.

이 논문은 DDPM의 각 backward transition을 그대로 근사하는 대신, **FORS를 사용해 Gaussian tilt를 rejection sampling 방식으로 보정**한다. 그 결과 \(\delta\) 의존성이 다항식에서

\[
\operatorname{polylog}(1/\delta)
\]

로 개선된다.

#### 고차 차분 또는 고차 solver 방법

고차 방법들은 대략

\[
\frac{d^{1+1/p}}{\delta^{1/p}}
\]

형태의 복잡도를 얻는다. \(p\)를 크게 하면 \(1/\delta\) 의존성이 줄어들지만, \(p\)에 대한 상수가 보통 매우 크게 증가하거나 사실상 지수적으로 증가한다.

따라서 이 방법들은 sub-polynomial 의존성은 제공하지만, 본 논문의 **진정한 polylogarithmic 의존성**에는 미치지 못한다.

#### 밀도값을 추가로 사용하는 방법

Huang et al. 및 Wainwright의 방법은 정규화되지 않은 log-density를 평가할 수 있다는 가정하에 다음과 같은 결과를 얻는다.

- \(\widetilde O(d^2\log(1/\delta))\)
- \(\widetilde O(\sqrt d\log^3(1/\delta))\)

그러나 실제 diffusion model은 일반적으로 density 자체가 아니라 score만 학습한다. 본 논문은 density query 없이 gradient/score query만 사용한다는 점에서 더 현실적인 설정을 다룬다.

#### 동시 연구인 Gatmiry et al.

Gatmiry et al.은 특정한 구조를 가진 데이터 분포와 강한 sub-exponential score error 조건하에서

\[
\widetilde O\!\left((R/\sigma)^2\log^2(1/\delta)\right)
\]

복잡도를 얻었다.

이 결과는 특정 상황에서 차원에 독립적일 수 있지만,

- 데이터가 Gaussian convolution이어야 하고,
- 지지집합이 반경 \(R\)인 공 안에 있어야 하며,
- score 오차가 sub-exponential tail을 가져야 한다.

반면 본 논문은 일반적인 \(L^2\) score error 조건과 훨씬 약한 데이터 가정에서 결과를 제시한다. 두 결과의 복잡도는 서로 직접적으로 우열을 비교하기 어렵다.

---

### 5. Log-concave sampling 결과

논문은 diffusion model뿐 아니라 log-concave 분포 샘플링에도 FORS를 적용한다.

목표 분포가

\[
\mu(x)\propto e^{-f(x)}
\]

이고 \(f\)의 gradient만 사용할 수 있다고 하자. 기존 proximal sampler는 restricted Gaussian oracle(RGO)를 필요로 하는데, FORS를 이용하면 이 RGO를 **density evaluation 없이 gradient query만으로 구현**할 수 있다.

대표적인 결과는 다음과 같다.

- \(f\)가 smooth하고 log-Sobolev inequality를 만족하는 경우:  
  \(d^{1/2}\)와 \(\log(1/\varepsilon)\)에 대해 고정밀 복잡도
- Poincaré inequality를 만족하는 경우:  
  \(\chi^2\) 오차를 \(\varepsilon^2\)까지 polylogarithmic accuracy로 감소
- 일반 log-concave 분포:  
  KL 오차를 \(\varepsilon^2\)까지 줄이는 고정밀 샘플러 제공
- \(f\)가 Lipschitz인 경우에도 Poincaré 또는 log-concavity 가정하에서 결과 제시

이 부분의 의의는 기존 rejection sampling이나 Metropolis–Hastings처럼 \(f(x)\) 자체를 평가하지 않고도, **gradient만으로 high-accuracy sampling**이 가능하다는 점이다.

---

### 6. 실험 데이터와 테스트 결과에 대한 주의점

이 논문은 실험 논문이라기보다 **이론 논문**이다.

따라서 다음과 같은 내용은 보고되지 않는다.

- MNIST, CIFAR-10, ImageNet 등의 실제 데이터셋 실험
- FID, IS, precision/recall 등의 생성모델 평가
- 실제 wall-clock time이나 GPU 비교
- 경험적 샘플 품질 그래프

논문에서 말하는 “결과”는 실제 테스트 데이터에서 측정한 수치가 아니라, 특정 분포 가정하에서 증명된 **query complexity와 분포 거리 보장**이다.

또한 저자들은 FORS가 한 단계마다 여러 score evaluation을 요구하기 때문에 실제 구현에서는 계산량이 증가할 수 있으며, 이론적으로 허용되는 큰 step size가 실제로도 유리한지는 별도의 실험적 검증이 필요하다고 명시한다.

---




### 1. Main contribution

The paper proposes **First-Order Rejection Sampling (FORS)**, a method for sampling from distributions using only first-order information, such as score or gradient evaluations. It does not require evaluating the unnormalized density itself.

The main improvement is in the target-accuracy dependence. Previous diffusion samplers typically required polynomial dependence on \(1/\delta\), while this work achieves

\[
\operatorname{polylog}(1/\delta)
\]

query complexity under \(L^2\)-accurate score estimates.

Here, \(\widetilde O(\cdot)\) suppresses logarithmic factors.

---

### 2. Diffusion sampling results

The analysis first samples from an early-stopped distribution \(p_1\), which is a slightly smoothed version of the data distribution, and then controls the distance between \(p_1\) and \(p_{\mathrm{data}}\).

#### A. Minimal data assumptions

Assume only that the data distribution has a finite second moment,

\[
M_2^2=\mathbb E\|X_0\|^2.
\]

The query complexity is approximately

\[
\widetilde O\!\left(
\max\{d,\log(1/\delta)\}
\log^2\!\frac{d+M_2^2}{\delta^2}
\right).
\]

The bounded-Lipschitz error satisfies

\[
D_{\mathrm{BL}}(p_{\mathrm{data}},\widehat p_1)^2
\lesssim
\delta^2+
\sum_t\eta_t\varepsilon_{t,\mathrm{score}}^2.
\]

Thus, the algorithmic error can be made arbitrarily small with only polylogarithmic dependence on \(1/\delta\). The score estimation error enters additively through the weighted sum of per-time-step \(L^2\) errors.

#### B. Non-uniform Lipschitz scores

Under a high-probability, or non-uniform, Lipschitz condition on the score Jacobian, the complexity becomes roughly

\[
\widetilde O\!\left(
\max\{\sqrt{dL_\delta},L_\delta\}
\log\frac{d+M_2^2}{\delta^2}
\right).
\]

The total variation guarantee is of the form

\[
D_{\mathrm{TV}}(p_1,\widehat p_1)^2
\lesssim
\delta^2+
\sqrt{\frac d{L_\delta}}
\sum_t\eta_t\varepsilon_{t,\mathrm{score}}^2.
\]

This improves the dimension dependence from \(d\) to approximately \(\sqrt d\), at the cost of somewhat worse sensitivity to score error.

#### C. Intrinsic dimension

If the data distribution has low intrinsic dimension \(d^\star\), measured through a covering number, the query complexity becomes

\[
\widetilde O\!\left(
\max\{d^\star,\log(d/\delta)\}
\log^2\!\frac{d+M_2^2}{\delta^2}
\right).
\]

The error bound has the form

\[
D_{\mathrm{BL}}(p_{\mathrm{data}},\widehat p_1)^2
\lesssim
\delta^2+
\frac d{d^\star}
\sum_t\eta_t\varepsilon_{t,\mathrm{score}}^2.
\]

Thus, the number of diffusion steps adapts to the intrinsic rather than ambient dimension, although the score-error term contains a \(d/d^\star\) factor.

---

### 3. Metrics used

The paper uses different divergences depending on the setting:

- **Bounded-Lipschitz distance:** for minimal assumptions and weak convergence.
- **Total variation distance:** for stronger regularity assumptions on the score.
- **KL divergence:** for analyzing errors accumulated across reverse diffusion transitions.
- **Hellinger distance:** for comparing transition kernels and summing local errors.
- **Chi-squared divergence:** mainly in Gaussian-tilt and log-concave analyses.
- **Wasserstein distance:** for controlling the gap between the early-stopped distribution and the original data distribution.

---

### 4. Comparison with prior and competing methods

#### Standard DDPM methods

Previous DDPM analyses under weak data assumptions typically obtained complexities such as

\[
\widetilde O(d/\delta^2)
\quad\text{or}\quad
\widetilde O(d/\delta).
\]

Other work achieved roughly \(1/\sqrt{\delta}\) dependence. Moreover, an \(\Omega(1/\delta)\) lower bound is known for DDPM in general.

The present work avoids this limitation by using FORS to correct each reverse Gaussian-tilt transition through rejection sampling rather than relying only on a local first-order discretization.

#### Higher-order methods

Higher-order discretization methods achieve complexities roughly of the form

\[
\frac{d^{1+1/p}}{\delta^{1/p}}.
\]

They improve the dependence on \(1/\delta\), but the constants often grow very rapidly, sometimes implicitly exponentially in the order \(p\). Therefore, they remain sub-polynomial rather than genuinely polylogarithmic in \(1/\delta\).

#### Methods using density evaluations

Some previous high-accuracy samplers assume access to unnormalized log-density evaluations and achieve bounds such as

\[
\widetilde O(d^2\log(1/\delta))
\]

or

\[
\widetilde O(\sqrt d\log^3(1/\delta)).
\]

The distinction is important: diffusion models normally provide score estimates, not density evaluations. This paper works in the score-only setting.

#### Concurrent work

Gatmiry et al. obtain a bound of approximately

\[
\widetilde O\!\left((R/\sigma)^2\log^2(1/\delta)\right)
\]

under stronger structural assumptions, including Gaussian-convolution data, bounded support, Lipschitz scores, and sub-exponential score errors.

Their result can be dimension-free in some regimes, but it relies on substantially stronger assumptions. The two complexity bounds are therefore not directly comparable.

---

### 5. Log-concave sampling

The same FORS framework is applied to sampling from

\[
\mu(x)\propto e^{-f(x)}
\]

using only \(\nabla f\).

The method implements the restricted Gaussian oracle required by the proximal sampler without evaluating \(f(x)\) itself. The paper obtains high-accuracy guarantees under:

- log-Sobolev inequalities,
- Poincaré inequalities,
- log-concavity,
- smooth or merely Lipschitz potentials.

The main significance is that it provides a polylogarithmic-accuracy sampling framework for general log-concave distributions using only first-order queries, whereas many classical rejection or Metropolis–Hastings methods require zeroth-order density evaluations.

---

### 6. Experimental data and empirical comparisons

This is primarily a **theoretical paper**, not an empirical benchmark paper.

It does not report:

- experiments on MNIST, CIFAR-10, ImageNet, or other datasets,
- FID or Inception Score,
- GPU runtime comparisons,
- empirical sample-quality plots.

The “results” are theoretical guarantees on query complexity and distributional divergences under explicit assumptions.

The authors also note that FORS may require several score evaluations per diffusion step. Consequently, its practical computational advantage over standard DDPM is not established experimentally and remains an open implementation question.


<br/>
# 예제
## 1. 논문이 다루는 핵심 과제

이 논문은 **확률분포의 값을 직접 계산하지 않고, 로그밀도의 기울기(score 또는 gradient)만 이용해 샘플을 생성하는 방법**을 제안합니다.

주요 대상은 두 가지입니다.

1. **Diffusion model sampling**
   - 학습된 score 함수만 이용해 데이터와 유사한 샘플 생성
2. **Log-concave distribution sampling**
   - 로그밀도 \( \log p(x)=-f(x)+\text{const} \)의 gradient \( \nabla f(x) \)만 이용해 샘플 생성

특히 기존 방법보다 목표 오차 \(\delta\)에 대한 반복 횟수를 크게 줄여, 대략  
\[
\operatorname{polylog}(1/\delta)
\]
수준의 높은 정확도(high-accuracy)를 목표로 합니다.

---

## 2. Diffusion model의 구체적인 예시

### 예시 데이터

2차원 이미지 데이터라고 가정합니다.

- 학습 데이터 입력:
  \[
  x_0=
  \begin{bmatrix}
  0.8\\
  0.6
  \end{bmatrix}
  \]
  또는 실제로는 이미지 한 장을 벡터로 표현한 값
- 학습 데이터 출력:
  - 일반적인 지도학습처럼 클래스 라벨을 예측하는 것이 아니라,
  - 해당 데이터에 노이즈를 추가했을 때의 **score 함수**를 학습합니다.

### 2.1 학습 단계

Diffusion 학습에서는 원본 데이터 \(x_0\)에 Gaussian noise를 추가해 다음과 같은 \(x_t\)를 만듭니다.

\[
x_t=\bar\alpha_t x_0+\sigma_t\epsilon,
\qquad
\epsilon\sim \mathcal N(0,I).
\]

예를 들어,

- 원본 데이터:
  \[
  x_0=(0.8,0.6)
  \]
- noise:
  \[
  \epsilon=(0.2,-1.0)
  \]
- \(\bar\alpha_t=0.7,\ \sigma_t=0.5\)

이면

\[
x_t
=0.7(0.8,0.6)+0.5(0.2,-1.0)
=(0.66,-0.08).
\]

신경망의 학습 입력과 출력은 다음과 같습니다.

| 구분 | 내용 |
|---|---|
| 학습 입력 | noisy sample \(x_t\), 시간 \(t\) |
| 학습 목표 | \(s_t^\star(x_t)=\nabla\log p_t(x_t)\) |
| 실제 학습 방식 | 보통 noise \(\epsilon\) 또는 이에 대응하는 값을 예측 |
| 학습된 함수 | \(s_t(x)\approx s_t^\star(x)\) |

즉, 모델은 “이 noisy point에서 데이터가 많이 존재하는 방향이 어디인가?”를 학습합니다.

논문에서는 score estimation error를 다음처럼 측정합니다.

\[
\epsilon_{t,\text{score}}^2
=
\mathbb E_{X_t\sim p_t}
\left\|s_t(X_t)-s_t^\star(X_t)\right\|^2.
\]

---

### 2.2 테스트 또는 샘플링 단계

이 논문에서 “테스트 데이터에 대한 예측”은 일반적인 분류 문제의 테스트 예측과는 다릅니다.  
**새로운 데이터 \(x_0\)를 입력으로 넣는 것이 아니라, Gaussian noise에서 시작하여 새로운 sample을 생성**합니다.

#### 샘플링 입력

\[
X_T\sim \mathcal N(0,\sigma_T^2I)
\]

예를 들어 2차원에서는

\[
X_T=(1.3,-0.4)
\]

처럼 Gaussian noise를 입력으로 사용합니다.

#### 각 역방향 단계

논문은 다음과 같은 backward transition을 직접 근사합니다.

\[
\rho_t(x\mid x')
\propto
p_t(x)
\exp\left(
-\frac{\|x-\alpha_t^{-1}x'\|^2}{2\eta_t}
\right).
\]

일반 DDPM은 score를 한 지점에서만 사용해 Gaussian proposal을 만들지만, 이 논문은 **FORS(First-Order Rejection Sampling)**를 사용하여 해당 분포를 보다 정확하게 샘플링합니다.

샘플링 과정은 다음과 같습니다.

1. 현재 상태 \(X_{t+1}\)를 입력으로 받음
2. Gaussian proposal \(q_t\)에서 후보 \(x\)를 생성
3. score 함수 \(s_t\)를 여러 지점에서 평가
4. 후보를 accept/reject
5. 이를 \(t=T-1,\ldots,1\)까지 반복
6. 최종적으로 \(X_1\)을 출력

#### 샘플링 출력

예를 들어 최종 출력이

\[
\hat x_0=(0.76,0.58)
\]

이라면, 이는 학습 데이터 분포에서 생성된 새로운 샘플입니다. 이미지라면 새로운 이미지가 됩니다.

평가는 보통 다음과 같은 분포 간 거리로 합니다.

- bounded Lipschitz distance
- total variation distance
- KL divergence
- Hellinger distance
- Wasserstein distance

논문의 목표는 대략 다음과 같은 형태입니다.

\[
D(p_{\text{data}},\hat p)
\le
\delta+C_{\text{apx}}\epsilon_{\text{score}}.
\]

즉,

- \(\delta\): 알고리즘 자체의 샘플링 오차
- \(\epsilon_{\text{score}}\): score 모델의 학습 오차
- \(C_{\text{apx}}\): score 오차가 최종 결과에 미치는 영향

입니다.

---

## 3. Diffusion sampling에서의 구체적인 task

### Task

> 학습된 score 함수 \(s_t\)에 질의할 수 있을 때, Gaussian noise로부터 데이터 분포와 가까운 새로운 샘플을 생성하라.

### 입력과 출력

| 단계 | 입력 | 출력 |
|---|---|---|
| 학습 | 원본 데이터 \(x_0\), 시간 \(t\), noisy data \(x_t\) | score 추정값 \(s_t(x_t)\) |
| 샘플링 초기화 | Gaussian noise \(X_T\) | 초기 noisy state |
| 역방향 단계 | \(X_{t+1}\), score 함수 \(s_t\) | \(X_t\) |
| 최종 결과 | 전체 score 모델과 random seed | 새로운 데이터 샘플 \(X_1\) |

### 논문의 개선점

기존 DDPM은 높은 정확도를 얻으려면 대략 \(1/\delta\) 또는 \(1/\delta^2\)에 가까운 많은 단계가 필요할 수 있습니다.

반면 이 논문은 다음과 같은 복잡도를 제시합니다.

- 최소 가정:
  \[
  \widetilde O\left(
  d\operatorname{polylog}(1/\delta)
  \right)
  \]
- score가 비균일 Lipschitz인 경우:
  \[
  \widetilde O\left(
  \sqrt{dL}\operatorname{polylog}(1/\delta)
  \right)
  \]
- intrinsic dimension이 \(d^\star\)인 경우:
  \[
  \widetilde O\left(
  d^\star\operatorname{polylog}(1/\delta)
  \right)
  \]

여기서 \(d\)는 ambient dimension, \(d^\star\)는 데이터의 intrinsic dimension입니다.

---

## 4. FORS의 직관적인 예시

FORS는 함수값 \(f(x)\)를 직접 계산하지 않고, gradient만 이용해 rejection sampling을 흉내 냅니다.

### 일반적인 rejection sampling

목표 분포가

\[
p(x)\propto e^{-f(x)}
\]

라고 하면, 보통 \(f(x)\)의 값을 알아야 accept/reject 확률을 계산할 수 있습니다.

### FORS의 입력

- proposal distribution \(q(x)\)
- gradient oracle \(\nabla f(x)\)
- gradient를 이용해 만든 unbiased estimator \(W\)

즉,

\[
\mathbb E[W\mid x]=w(x)
\]

가 되도록 여러 random gradient query를 사용합니다.

### FORS의 출력

proposal \(x\)를 accept할지 결정하고, accept된 \(x\)를 출력합니다.

이 방식은 함수 \(f(x)\) 자체를 계산하지 않고도

\[
q(x)e^{w(x)}
\]

형태의 target distribution에서 샘플을 얻도록 해줍니다.

---

## 5. Log-concave sampling의 구체적인 예시

이번에는 목표 분포가 다음과 같다고 하겠습니다.

\[
\mu(x)\propto e^{-f(x)}.
\]

예를 들어 1차원 Gaussian 분포를 생각하면

\[
f(x)=\frac{x^2}{2},
\qquad
\nabla f(x)=x.
\]

### 입력

알고리즘이 사용할 수 있는 것은 다음뿐입니다.

- 현재 샘플 \(X_n\)
- gradient query:
  \[
  \nabla f(X_n)
  \]
- Gaussian random variable

밀도 \(e^{-f(x)}\)의 실제 값은 직접 계산하지 않아도 됩니다.

### Proximal sampler

논문의 Algorithm 3은 다음 과정을 반복합니다.

1. 현재 값 \(X_n\)에서 Gaussian perturbation 생성:
   \[
   Y_n\sim\mathcal N(X_n,\eta I)
   \]
2. 다음 restricted Gaussian oracle 분포에서 \(X_{n+1}\) 샘플링:
   \[
   X_{n+1}
   \sim
   \operatorname{RGO}_{f,\eta,Y_n},
   \]
   \[
   p(x\mid Y_n)
   \propto
   \exp\left(
   -f(x)-\frac{\|Y_n-x\|^2}{2\eta}
   \right).
   \]
3. 이 RGO 샘플을 FORS로 구현
4. 충분히 반복한 뒤 \(X_N\) 출력

### 구체적인 1차원 예시

\[
f(x)=\frac{x^2}{2}
\]

이고 현재 값이 \(X_n=1.0\), step size가 \(\eta=0.5\)라고 하겠습니다.

1. Gaussian perturbation:
   \[
   Y_n\sim\mathcal N(1.0,0.5)
   \]
2. \(Y_n=0.7\)이 나왔다고 하면, 다음 분포는
   \[
   p(x\mid 0.7)
   \propto
   \exp\left(
   -\frac{x^2}{2}
   -\frac{(0.7-x)^2}{1.0}
   \right)
   \]
3. 이 분포에서 FORS가 gradient \(\nabla f(x)=x\)만 사용해 샘플 생성
4. 예를 들어 \(X_{n+1}=0.4\) 출력

반복하면 샘플들은 목표 Gaussian 분포에 가까워집니다.

---

## 6. Log-concave sampling에서의 구체적인 task

### Task

> 로그밀도 함수 \(f\)의 값은 모르거나 계산하지 않고, gradient \(\nabla f\)만 이용하여 \(\mu(x)\propto e^{-f(x)}\)에서 샘플을 생성하라.

### 입력과 출력

| 구분 | 입력 | 출력 |
|---|---|---|
| Oracle 입력 | 점 \(x\) | gradient \(\nabla f(x)\) |
| Proximal 단계 입력 | 현재 상태 \(X_n\), Gaussian noise | RGO target distribution |
| FORS 입력 | proposal \(q\), gradient 기반 estimator | accept/reject된 샘플 |
| 최종 출력 | 여러 반복 후의 \(X_N\) | 목표 log-concave 분포의 근사 샘플 |

논문은 이 방법이 log-concave 또는 더 일반적인 isoperimetric distribution에 대해 높은 정확도 보장을 제공한다고 설명합니다.

---

## 7. 핵심 요약

- **학습 단계:** noisy data \(x_t\)를 입력받아 score \(s_t(x_t)\)를 학습
- **Diffusion 샘플링 단계:** Gaussian noise \(X_T\)에서 시작해 score query와 FORS를 사용하여 \(X_1\) 생성
- **출력:** 데이터 분포와 가까운 새로운 샘플
- **FORS의 역할:** density 또는 log-density 값을 직접 계산하지 않고 gradient/score만으로 rejection sampling 수행
- **Log-concave sampling:** \(\nabla f\)만 이용해 \(e^{-f(x)}\) 분포에서 샘플 생성
- **주요 장점:** 목표 정확도 \(\delta\)에 대해 반복 복잡도가 \(\operatorname{polylog}(1/\delta)\) 수준으로 감소

> 참고로, 제시된 논문은 이론 논문이므로 특정 이미지 데이터셋에 대한 실제 실험 결과나 고정된 train/test 데이터 예시는 제시하지 않습니다. 위 예시는 논문의 알고리즘을 이해하기 위한 구체적인 적용 예시입니다.

---




## 1. Main task of the paper

The paper studies how to generate samples from a probability distribution using only evaluations of its **score function** or the gradient of its log-density.

It focuses on two settings:

1. **Diffusion model sampling**
   - Generate new data samples using learned score functions.
2. **Log-concave sampling**
   - Sample from \(p(x)\propto e^{-f(x)}\) using only gradient evaluations \(\nabla f(x)\).

The main goal is to achieve high accuracy with a number of iterations that depends polylogarithmically on the target error:

\[
\operatorname{polylog}(1/\delta).
\]

---

## 2. Concrete diffusion-model example

Assume that the data are 2D images represented as vectors.

For example,

\[
x_0=(0.8,0.6).
\]

### 2.1 Training stage

The diffusion process adds Gaussian noise:

\[
x_t=\bar\alpha_t x_0+\sigma_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
\]

For example, if

\[
\epsilon=(0.2,-1.0),\quad
\bar\alpha_t=0.7,\quad
\sigma_t=0.5,
\]

then

\[
x_t
=
0.7(0.8,0.6)+0.5(0.2,-1.0)
=
(0.66,-0.08).
\]

The neural network receives:

- noisy input \(x_t\),
- diffusion time \(t\),

and learns to estimate

\[
s_t^\star(x_t)=\nabla\log p_t(x_t).
\]

In practice, score matching is often implemented by predicting the added noise or an equivalent denoising target.

| Component | Description |
|---|---|
| Training input | \(x_t\) and time \(t\) |
| Training target | The score \(s_t^\star(x_t)\) or an equivalent noise-prediction target |
| Learned output | \(s_t(x)\approx s_t^\star(x)\) |

The score estimation error is measured by

\[
\epsilon_{t,\text{score}}^2
=
\mathbb E
\left[
\|s_t(X_t)-s_t^\star(X_t)\|^2
\right].
\]

---

### 2.2 Sampling stage

At sampling time, the algorithm does not receive a real test image as input. Instead, it starts from Gaussian noise:

\[
X_T\sim\mathcal N(0,\sigma_T^2I).
\]

For example,

\[
X_T=(1.3,-0.4).
\]

It then runs the reverse process:

1. Use \(X_{t+1}\) to construct a proposal distribution.
2. Query the score function \(s_t\).
3. Use FORS to accept or reject candidate points.
4. Generate \(X_t\).
5. Repeat until \(X_1\).

The final output may be, for example,

\[
\hat x_0=(0.76,0.58),
\]

which is a newly generated sample from the learned data distribution.

The quality of the generated distribution can be measured using:

- bounded Lipschitz distance,
- total variation distance,
- KL divergence,
- Hellinger distance,
- Wasserstein distance.

The guarantee has the general form

\[
D(p_{\text{data}},\hat p)
\le
\delta+C_{\text{apx}}\epsilon_{\text{score}}.
\]

Here:

- \(\delta\) is the algorithmic sampling error,
- \(\epsilon_{\text{score}}\) is the score estimation error,
- \(C_{\text{apx}}\) controls how score error affects the final distribution.

---

## 3. Concrete diffusion-sampling task

### Task

> Given access to learned score functions \(s_t\), generate a new sample close to the data distribution, starting only from Gaussian noise.

| Stage | Input | Output |
|---|---|---|
| Training | \(x_0\), \(t\), noisy sample \(x_t\) | Estimated score \(s_t(x_t)\) |
| Initialization | Gaussian noise \(X_T\) | Initial noisy state |
| Reverse step | \(X_{t+1}\), score function \(s_t\) | \(X_t\) |
| Final output | Score model and random seed | New generated sample \(X_1\) |

The paper gives approximate complexities of:

\[
\widetilde O\left(d\operatorname{polylog}(1/\delta)\right)
\]

under minimal assumptions,

\[
\widetilde O\left(\sqrt{dL}\operatorname{polylog}(1/\delta)\right)
\]

under a non-uniform Lipschitz condition, and

\[
\widetilde O\left(d^\star\operatorname{polylog}(1/\delta)\right)
\]

when the data have intrinsic dimension \(d^\star\).

---

## 4. Intuition behind FORS

Suppose the target density is

\[
p(x)\propto e^{-f(x)}.
\]

Standard rejection sampling usually requires evaluating \(f(x)\). FORS avoids this requirement.

### FORS input

- A proposal distribution \(q(x)\)
- Gradient evaluations \(\nabla f(x)\)
- A random estimator \(W\) satisfying
  \[
  \mathbb E[W\mid x]=w(x)
  \]

### FORS output

FORS generates a proposal \(x\), decides whether to accept it using random gradient-based estimates, and outputs the accepted point.

Thus, it simulates rejection sampling without directly evaluating the density or log-density.

---

## 5. Concrete log-concave-sampling example

Consider

\[
\mu(x)\propto e^{-f(x)}
\]

with

\[
f(x)=\frac{x^2}{2}.
\]

Then

\[
\nabla f(x)=x,
\]

and the target distribution is a standard Gaussian.

### Proximal sampler

The algorithm repeatedly performs:

1. Gaussian perturbation:
   \[
   Y_n\sim\mathcal N(X_n,\eta I)
   \]
2. Sample from
   \[
   p(x\mid Y_n)
   \propto
   \exp\left(
   -f(x)-\frac{\|Y_n-x\|^2}{2\eta}
   \right).
   \]
3. Implement this restricted Gaussian oracle using FORS and gradient evaluations.
4. Output the final state \(X_N\).

For example, if

\[
X_n=1.0,\qquad \eta=0.5,
\]

and the Gaussian perturbation produces \(Y_n=0.7\), then the next target distribution is

\[
p(x\mid 0.7)
\propto
\exp\left(
-\frac{x^2}{2}
-\frac{(0.7-x)^2}{1.0}
\right).
\]

FORS samples from this distribution using only

\[
\nabla f(x)=x.
\]

After many iterations, the output approaches the desired Gaussian distribution.

---

## 6. Main takeaway

- **Training:** Learn the score from noisy data.
- **Diffusion sampling:** Start from Gaussian noise and use score queries plus FORS to generate a new sample.
- **FORS:** Performs rejection-sampling-like correction without evaluating the density.
- **Log-concave sampling:** Uses only \(\nabla f\) to sample from \(e^{-f(x)}\).
- **Main benefit:** The number of iterations can depend polylogarithmically on \(1/\delta\), instead of polynomially on \(1/\delta\).

The paper is primarily theoretical, so it does not present a fixed image dataset with an empirical train/test experiment. The examples above are concrete illustrations of how the proposed algorithms would be used.

<br/>
# 요약

 
1. 본 논문은 score/gradient 값만으로 rejection sampling을 모사하는 **FORS(First-Order Rejection Sampling)**를 제안해, 밀도값 계산 없이 확산 모델의 역전이와 Gaussian tilt를 정확하게 샘플링한다.  
2. 그 결과 목표 오차 \(\delta\)에 대해 반복·질의 복잡도를 \(\mathrm{polylog}(1/\delta)\)로 낮추고, 최소 가정에서는 약 \(\tilde O(d)\), 비균일 Lipschitz 조건에서는 \(\tilde O(\sqrt{dL})\), 내재 차원 \(d^\star\)를 활용하면 \(\tilde O(d^\star)\)까지 달성한다.  
3. 또한 이 방법은 bounded Lipschitz 오차와 \(L^2\) score 오차를 허용하면서도 적용 가능하며, 예를 들어 일반 데이터 분포의 diffusion sampling과 log-concave 분포의 gradient-only sampling에서 기존 방법보다 높은 정확도와 효율을 제공한다.  



1. The paper introduces **First-Order Rejection Sampling (FORS)**, which simulates rejection sampling using only score/gradient evaluations, without evaluating the density itself.  
2. It achieves \(\mathrm{polylog}(1/\delta)\) iteration and query complexity, with roughly \(\tilde O(d)\) under minimal assumptions, \(\tilde O(\sqrt{dL})\) under a non-uniform Lipschitz condition, and \(\tilde O(d^\star)\) when exploiting intrinsic dimension.  
3. The framework remains robust to \(L^2\) score-estimation error and applies to diffusion-model sampling and gradient-only sampling of log-concave distributions, substantially improving the accuracy dependence of previous methods.

<br/>
# 기타



### 1. 다이어그램·피규어·테이블

제공된 논문 본문과 부록에는 **별도의 그림(Figure), 다이어그램, 표(Table)가 포함되어 있지 않습니다.**  
따라서 핵심 결과는 그림이나 표가 아니라, **정리(Theorem)·알고리즘·부록의 증명**으로 제시됩니다.

논문의 전체 흐름은 다음과 같이 이해할 수 있습니다.

> **score/gradient 평가 → Gaussian tilt 형태로 역전이 분포 구성 → FORS로 rejection sampling 구현 → 여러 diffusion step을 연결 → high-accuracy sampling 보장**

---

## 2. Appendix A — Discussion

### 주요 결과

Appendix A.1은 동시 연구인 **Gatmiry et al. (2026)**와 비교합니다.

그 연구는 다음과 같은 강한 조건을 가정합니다.

- 데이터 분포가  
  \[
  p_{\mathrm{data}}=p_\star * \mathcal N(0,\sigma^2I)
  \]
  형태
- \(p_\star\)가 반지름 \(R\)인 공 안에 지지됨
- score error가 평균제곱 오차가 아니라 **sub-exponential tail bound**를 만족
- score 함수가 Lipschitz

그 결과 query complexity는 대략

\[
\widetilde O\left((R/\sigma)^2\log^2(1/\delta)\right)
\]

입니다.

### 인사이트

- 동시 연구는 특정 조건에서 **dimension-free** 결과를 얻을 수 있지만, \(R/\sigma\)에 크게 의존합니다.
- 특히 실제 데이터 분포에 가까워지기 위해 \(\sigma\)를 매우 작게 선택하면 \(R/\sigma\)가 커져 high-accuracy 이점이 약화될 수 있습니다.
- 본 논문은 더 약한 조건인 **L2 평균 score error**만 사용하면서도, 데이터 분포에 대한 가정이 훨씬 적습니다.
- 즉, 본 논문의 장점은 최적의 dimension dependence라기보다 **약한 가정과 일반성**입니다.

---

## 3. Appendix B — Additional Notation and Technical Tools

Appendix B는 본문 정리들의 증명을 위한 확률·정보이론 도구를 정리합니다.

### B.1 Functional inequalities

- **Poincaré inequality (PI)**  
  분산을 gradient energy로 제어합니다.
- **Log-Sobolev inequality (LSI)**  
  entropy 또는 KL divergence의 수렴을 제어합니다.
- **Bakry–Émery 결과**  
  \(\alpha\)-strongly log-concave 분포는 LSI constant가 \(1/\alpha\) 이하입니다.
- **Holley–Stroock perturbation**  
  두 분포의 density ratio가 제한되어 있으면 LSI constant도 안정적으로 변합니다.

### B.2 핵심 보조정리

주요 도구는 다음과 같습니다.

1. Gaussian 벡터의 지수 모멘트와 quadratic form concentration
2. Hellinger distance와 KL/chi-square divergence 사이의 관계
3. 서로 다른 tilt를 적용한 두 분포의 divergence 비교
4. Markov chain의 여러 transition error를 전체 Hellinger error로 합치는 chain rule
5. Gaussian convolution이 score와 Hessian을 얼마나 regularize하는지에 대한 bound

### 인사이트

Appendix B의 핵심 역할은 다음입니다.

> **한 번의 FORS step에서 생기는 작은 local error를 diffusion 전체의 global error로 변환하는 수학적 연결고리**

특히 squared Hellinger distance의 sub-additivity는 각 backward step의 오차를 대략 \(T\)배로 합칠 수 있게 해줍니다.

---

## 4. Appendix C — Proof of Theorem 3.1: FORS의 정확성

### 주요 결과

FORS는 다음과 같은 분포를 샘플링합니다.

\[
\hat p(x)\propto q(x)\exp(\mathbb E[W\mid x]),
\]

여기서 \(W\)는 \(w(x)\)의 unbiased estimator이며 \([-B,B]\)에 속합니다.

알고리즘은

- \(x\sim q\)를 제안하고
- \(J\sim\mathrm{Poisson}(2B)\)를 뽑은 뒤
- \(W_1,\ldots,W_J\)를 샘플링하고
- 다음 확률로 accept합니다.

\[
\prod_{j=1}^{J}\frac{B+W_j}{2B}.
\]

조건부 기대값을 계산하면 acceptance probability가

\[
\exp(\mathbb E[W\mid x]-B)
\]

가 되므로, 최종 출력 density는 정확히 \(q(x)e^{\mathbb E[W\mid x]}\)에 비례합니다.

또한 \(1-\delta\) 확률로 필요한 \(W_j\)의 수는

\[
O\left(Be^{2B}\log(1/\delta)\right)
\]

수준입니다.

### 인사이트

FORS의 핵심은 다음과 같습니다.

> 함수값 \(w(x)\) 자체를 계산하지 않고도, unbiased한 gradient 기반 추정치만으로 \(e^{w(x)}\) 형태의 rejection probability를 구현한다.

이는 Bernoulli factory 아이디어를 rejection sampling에 적용한 것입니다.  
따라서 expensive한 적분이나 log-density 계산 없이도 rejection correction을 수행할 수 있습니다.

---

## 5. Appendix D — Gaussian tilt 분석

### 주요 결과

Gaussian tilt

\[
\nu(x)\propto
\exp\left(-f(x)-\frac{\|x-x_0\|^2}{2\eta}\right)
\]

를 gradient query만으로 샘플링할 때, \(f\)의 gradient가 Hölder 연속이라고 가정합니다.

\[
\|\nabla f(x)-\nabla f(y)\|
\le \beta_s\|x-y\|^s.
\]

그 결과, 충분히 작은 step size에 대해 FORS가 생성하는 분포 \(\hat\nu\)는 \(\nu\)와 매우 가깝습니다. 대표적인 조건은

\[
\frac{1}{\eta^{1+s}}
\gtrsim
\beta_s^2\left(
d^s\log(1/\delta)
+
d^{1-s}\log^2(1/\delta)
\right).
\]

### 두 가지 중요한 경우

- \(s=0\): gradient가 단순히 bounded/Lipschitz 수준인 경우
- \(s=1\): \(f\)가 smooth한 경우

\(s=1\)이면 필요한 inverse step size가 대략

\[
\eta^{-1}\gtrsim \beta_1 d^{1/2}\log(1/\delta)
\]

정도로 되어, dimension dependence가 \(\sqrt d\)까지 개선됩니다.

### 인사이트

Appendix D는 본 논문의 기본 원리를 설명합니다.

> Gaussian proposal이 이미 target distribution의 대부분을 설명하고, FORS는 gradient 정보로 남은 local nonlinearity만 correction한다.

또한 path를 단순한 직선 대신 Gaussian bridge 형태로 설계하면, 고차원 Gaussian fluctuation을 더 잘 평균화할 수 있어 \(d\) 의존성이 개선됩니다.

---

## 6. Appendix E — Diffusion sampling 증명

Appendix E는 diffusion 모델에 FORS를 적용한 세 가지 핵심 결과를 뒷받침합니다.

### E.1 Theorem 4.1 증명 — Minimal assumption

역방향 transition은

\[
\rho_t(x\mid x')
\propto
p_t(x)
\exp\left(-\frac{\|x-\alpha_t^{-1}x'\|^2}{2\eta_t}\right)
\]

인 Gaussian tilt입니다.

FORS가 사용하는 \(s_t\)가 정확한 score가 아닐 경우, local error는 대략

\[
\eta_t\epsilon_{t,\mathrm{score}}^2
\]

로 제어됩니다.

전체 chain에 대해

\[
D_{\mathrm{KL}}(p_1\|\hat p_1)
\lesssim
D_{\mathrm{KL}}(p_T\|\hat p_T)
+
T\delta
+
\sum_t\eta_t\epsilon_{t,\mathrm{score}}^2.
\]

#### 인사이트

- score error가 각 step에서 누적되지만, 단순히 \(\epsilon^2\)만이 아니라 step size \(\eta_t\)가 곱해집니다.
- 따라서 큰 step을 사용할 수 있으면 전체 query 수를 줄일 수 있습니다.
- 이 분석은 데이터 분포에 거의 smoothness 가정을 두지 않고도 성립합니다.

---

### E.2 DDPM-like 방법의 증명 — \(\sqrt d\) 의존성

여기서는 proposal을

\[
q_t=
\mathcal N\left(
\bar x_t+\eta_t s_t(\bar x_t),\,
\eta_t I
\right)
\]

로 선택합니다.

score Hessian이 높은 확률로 bounded하다는 non-uniform Lipschitz 조건을 사용합니다. 그 결과 필요한 step size 조건은 대략

\[
\frac{\sigma_t^2}{\eta_t}
\gtrsim
\sqrt{dL_\delta}\log(d/\delta)
+
L_\delta\log(d/\delta).
\]

전체 query complexity는 대략

\[
\widetilde O\left(
\max\{\sqrt{dL_\delta},L_\delta\}
\log\frac{d+M_2^2}{\delta^2}
\right).
\]

#### 인사이트

- score가 모든 곳에서 smooth할 필요는 없습니다.
- 실제 \(X_t\sim p_t\)에서 대부분의 확률 질량이 smooth한 영역에 있으면 충분합니다.
- 이 때문에 전역 Lipschitz 조건보다 훨씬 약한 **high-probability smoothness**로 \(\sqrt d\) scaling을 얻습니다.
- 단점은 score error가 minimal-assumption 결과보다 더 민감하게 들어간다는 점입니다.

---

### E.3 Low-dimensional structure — intrinsic dimension

shifted score를 사용합니다.

\[
\tilde s_t(x)=s_t(x)+\frac{x}{\sigma_t^2}.
\]

이는 Gaussian smoothing으로 생기는 \(-I/\sigma_t^2\) Hessian 성분을 제거하는 효과가 있습니다.

데이터의 covering number로 intrinsic dimension을

\[
d_\star
=
1+\log N_{\mathrm{data}}(\sigma_0/\sqrt d)
\]

로 정의하고, 필요한 조건은

\[
\frac{\sigma_t^2}{\eta_t}
\gtrsim
d_\star\log(d/\delta)+\log^2(d/\delta)
\]

입니다.

복잡도는 대략

\[
\widetilde O\left(
\max\{d_\star,\log(d/\delta)\}
\log^2\frac{d+M_2^2}{\delta^2}
\right).
\]

#### 인사이트

- ambient dimension \(d\)가 아니라 데이터 지지집합의 covering complexity가 중요합니다.
- 데이터가 낮은 차원의 manifold 또는 구조화된 집합 근처에 있으면 \(d_\star\ll d\)가 될 수 있습니다.
- shifted score는 데이터 구조와 무관한 Gaussian noise 방향을 제거하여 이 개선을 가능하게 합니다.
- 다만 score error 항에는 \(d/d_\star\) factor가 들어가므로, intrinsic dimension 개선은 score 정확도 조건과 함께 해석해야 합니다.

---

### E.4 Technical lemmas

Appendix E의 후반부에는 다음이 증명됩니다.

- Gaussian convolution 이후 score와 Hessian의 exponential moment bound
- posterior covariance와 covering number의 관계
- intrinsic dimension이 낮을 때 shifted Hessian의 trace와 quadratic form이 작다는 사실
- Gaussian quadratic form concentration
- approximate score와 exact score 사이의 Hellinger/KL error decomposition

#### 특히 중요한 결과

posterior가 data support 주변에 집중된다는 것을 covering number로 제어합니다.  
즉, 데이터가 낮은 복잡도를 가지면 posterior covariance도 대부분의 방향에서 작아지고, shifted Hessian의 effective complexity도 \(d_\star\) 수준으로 감소합니다.

---

## 7. Appendix F — Log-concave sampling

### 주요 결과

FORS를 proximal sampler의 restricted Gaussian oracle(RGO)에 적용합니다.

RGO는 다음 분포입니다.

\[
\mathrm{RGO}_{f,\eta,y}(x)
\propto
\exp\left(
-f(x)-\frac{\|y-x\|^2}{2\eta}
\right).
\]

Proximal sampler는 매 step마다

1. \(Y_n\sim\mathcal N(X_n,\eta I)\)를 샘플링하고
2. \(X_{n+1}\sim\mathrm{RGO}_{f,\eta,Y_n}\)를 샘플링합니다.

Appendix F의 Theorem F.1은 \(f\)의 gradient만 사용해 이 RGO를 구현하면서 high-accuracy 보장을 얻습니다.

### 대표적인 복잡도

\(f\)가 smooth한 경우에는 log-Sobolev 조건 아래 대략

\[
\widetilde O\left(
\kappa\sqrt d\log^{3/2}(1/\varepsilon)
+
\kappa\log^2(1/\varepsilon)
\right)
\]

수준의 first-order query가 필요합니다.

단순 log-concavity와 Wasserstein 초기화 조건 아래에서는 대략

\[
\widetilde O\left(
\frac{\beta_1 d^{1/2}W_2^2(\mu_0,\mu)}
{\varepsilon^2}
\right)
\]

형태의 결과를 얻습니다.

또한 \(f\)가 Lipschitz인 경우에는 smooth case와 다른 형태의 복잡도 bound가 제시됩니다.

### 인사이트

Appendix F의 가장 중요한 의미는 다음입니다.

> FORS는 diffusion 모델에만 특화된 기법이 아니라, 일반적인 gradient-only sampling에도 적용할 수 있다.

기존 rejection sampling이나 Metropolis–Hastings는 보통 density value가 필요하지만, FORS 기반 RGO는 **gradient 평가만으로** 구현됩니다.  
따라서 discretization bias를 갖는 Langevin 방식과 달리, rejection correction을 통해 \(\log(1/\varepsilon)\) 의존성을 유지할 수 있습니다.

---

## 전체 부록의 핵심 메시지

| 부록 | 핵심 역할 | 주요 인사이트 |
|---|---|---|
| A | 동시 연구와 비교 | 본 논문은 더 약한 가정과 L2 score error를 사용 |
| B | 확률·정보이론 도구 | local error를 global divergence bound로 변환 |
| C | FORS 정확성 증명 | unbiased gradient estimator로 exponential rejection probability 구현 |
| D | Gaussian tilt 분석 | Gaussian proposal + first-order correction으로 high accuracy 달성 |
| E | Diffusion 적용 | minimal assumption, \(\sqrt d\), intrinsic dimension 결과 증명 |
| F | Log-concave 적용 | gradient-only proximal sampling으로 일반화 |

### 한 문장 요약

이 논문의 부록들은 **“gradient만 알고 density는 모르는 상황에서도, Gaussian proposal과 Bernoulli-factory 방식의 rejection correction을 결합하면 diffusion 및 log-concave sampling에서 \(\log(1/\delta)\) 수준의 high-accuracy 복잡도를 얻을 수 있다”**는 주장을 단계적으로 뒷받침합니다.

---




## 1. Diagrams, figures, and tables

The provided paper does not contain separate **figures, diagrams, or tables**.  
Its main results are presented through algorithms, theorems, and appendix proofs.

The overall pipeline is:

> **score/gradient evaluations → Gaussian-tilt representation of the reverse transition → FORS rejection correction → composition over diffusion steps → high-accuracy sampling**

---

## 2. Appendix A — Discussion

### Main result

Appendix A.1 compares the paper with the concurrent work of Gatmiry et al. (2026).

That work assumes:

- \(p_{\mathrm{data}}=p_\star*\mathcal N(0,\sigma^2I)\),
- \(p_\star\) is supported on a ball of radius \(R\),
- score errors have sub-exponential tails,
- the score is Lipschitz.

Its query complexity is roughly

\[
\widetilde O\left((R/\sigma)^2\log^2(1/\delta)\right).
\]

### Insight

The concurrent result can be dimension-free when \(R/\sigma\) is controlled, but \(R/\sigma\) may become very large when the smoothing scale \(\sigma\) must be small. In contrast, this paper works under much weaker assumptions, using only average \(L^2\) score error and minimal assumptions on the data distribution.

The main advantage of this paper is therefore **generality and weaker assumptions**, rather than uniformly optimal dimension dependence.

---

## 3. Appendix B — Additional notation and technical tools

Appendix B collects the analytical tools used in the proofs.

### Main components

- Poincaré and log-Sobolev inequalities
- Bakry–Émery and Holley–Stroock results
- Gaussian exponential-moment and quadratic-form concentration
- Relations among KL, chi-square, Hellinger, and Rényi divergences
- Hellinger chain rules for Markov processes
- Bounds for Gaussian convolution, scores, and Hessians

### Insight

The main purpose of Appendix B is to connect local and global errors:

> It converts the approximation error of one FORS or reverse-diffusion step into a total divergence bound over the entire chain.

In particular, sub-additivity of squared Hellinger distance allows the errors of individual backward steps to be accumulated approximately linearly in the number of steps.

---

## 4. Appendix C — Proof of the FORS guarantee

### Main result

FORS produces a sample from

\[
\hat p(x)\propto q(x)\exp(\mathbb E[W\mid x]),
\]

where \(W\in[-B,B]\) is an unbiased estimator of the desired tilt \(w(x)\).

The algorithm samples \(J\sim\mathrm{Poisson}(2B)\) and accepts using

\[
\prod_{j=1}^J\frac{B+W_j}{2B}.
\]

Taking expectations shows that the acceptance probability is

\[
\exp(\mathbb E[W\mid x]-B),
\]

which gives the desired tilted density exactly.

With probability at least \(1-\delta\), the number of \(W_j\) evaluations is

\[
O\left(Be^{2B}\log(1/\delta)\right).
\]

### Insight

FORS implements an exponential rejection probability without evaluating the function value \(w(x)\) itself. It only needs unbiased first-order estimators. This is the key Bernoulli-factory idea behind the whole paper.

---

## 5. Appendix D — Analysis of Gaussian tilts

### Main result

For

\[
\nu(x)\propto
\exp\left(-f(x)-\frac{\|x-x_0\|^2}{2\eta}\right),
\]

the paper assumes Hölder continuity of the gradient:

\[
\|\nabla f(x)-\nabla f(y)\|
\le \beta_s\|x-y\|^s.
\]

FORS approximates \(\nu\) accurately when

\[
\eta^{-(1+s)}
\gtrsim
\beta_s^2\left(
d^s\log(1/\delta)
+
d^{1-s}\log^2(1/\delta)
\right).
\]

For the smooth case \(s=1\), this gives a roughly \(\sqrt d\)-dependent step-size requirement.

### Insight

The Gaussian proposal captures the quadratic part of the target distribution. FORS only corrects the remaining nonlinear part using gradient information.

The specially designed Gaussian-bridge path reduces the effect of high-dimensional Gaussian fluctuations and improves the dimension dependence compared with a naive line-segment construction.

---

## 6. Appendix E — Proofs for diffusion sampling

### E.1 Minimal-assumption method

The reverse transition is a Gaussian tilt:

\[
\rho_t(x\mid x')
\propto
p_t(x)
\exp\left(-\frac{\|x-\alpha_t^{-1}x'\|^2}{2\eta_t}\right).
\]

Using an approximate score \(s_t\), the local error is controlled by

\[
\eta_t\epsilon_{t,\mathrm{score}}^2.
\]

After composing all reverse steps,

\[
D_{\mathrm{KL}}(p_1\|\hat p_1)
\lesssim
D_{\mathrm{KL}}(p_T\|\hat p_T)
+
T\delta
+
\sum_t\eta_t\epsilon_{t,\mathrm{score}}^2.
\]

#### Insight

Score error is weighted by the step size. This makes it possible to use relatively large steps while retaining a polylogarithmic dependence on the target accuracy.

---

### E.2 DDPM-like method and \(\sqrt d\) dependence

This method uses

\[
q_t=
\mathcal N\left(
\bar x_t+\eta_t s_t(\bar x_t),\eta_tI
\right).
\]

It assumes that the score Hessian is bounded with high probability rather than everywhere. The resulting condition is approximately

\[
\frac{\sigma_t^2}{\eta_t}
\gtrsim
\sqrt{dL_\delta}\log(d/\delta)
+
L_\delta\log(d/\delta).
\]

The total complexity is roughly

\[
\widetilde O\left(
\max\{\sqrt{dL_\delta},L_\delta\}
\log\frac{d+M_2^2}{\delta^2}
\right).
\]

#### Insight

Global smoothness is unnecessary. It is enough that the score is smooth on most of the probability mass. This yields the improved \(\sqrt d\) dependence, although the method is more sensitive to score-estimation error.

---

### E.3 Low-dimensional structure

The paper introduces the shifted score

\[
\tilde s_t(x)=s_t(x)+\frac{x}{\sigma_t^2}.
\]

This removes the dominant Hessian contribution caused by Gaussian smoothing.

The intrinsic dimension is defined through a covering number:

\[
d_\star
=
1+\log N_{\mathrm{data}}(\sigma_0/\sqrt d).
\]

The step-size condition becomes

\[
\frac{\sigma_t^2}{\eta_t}
\gtrsim
d_\star\log(d/\delta)+\log^2(d/\delta),
\]

and the complexity is roughly

\[
\widetilde O\left(
\max\{d_\star,\log(d/\delta)\}
\log^2\frac{d+M_2^2}{\delta^2}
\right).
\]

#### Insight

The relevant complexity can be the covering complexity of the data support rather than the ambient dimension. When \(d_\star\ll d\), the method adapts to low-dimensional structure.

The tradeoff is that the score-error term contains a factor of approximately \(d/d_\star\), so the intrinsic-dimension gain depends on sufficiently accurate score estimates.

---

## 7. Appendix F — Log-concave sampling

### Main result

The paper applies FORS to the restricted Gaussian oracle

\[
\mathrm{RGO}_{f,\eta,y}(x)
\propto
\exp\left(
-f(x)-\frac{\|y-x\|^2}{2\eta}
\right).
\]

This is the key step in the proximal sampler:

1. Sample \(Y_n\sim\mathcal N(X_n,\eta I)\).
2. Sample \(X_{n+1}\) from the RGO distribution.

FORS implements the RGO using only gradient evaluations.

For smooth log-concave targets under functional inequalities, the resulting complexities are of the form

\[
\widetilde O\left(
\kappa\sqrt d\log^{3/2}(1/\varepsilon)
+
\kappa\log^2(1/\varepsilon)
\right)
\]

under log-Sobolev assumptions, with related bounds under Poincaré inequalities and general log-concavity.

### Insight

This appendix shows that FORS is not specific to diffusion models:

> It is a general gradient-only high-accuracy sampling primitive.

Unlike standard rejection sampling or Metropolis–Hastings, it does not require zeroth-order density evaluations. Unlike ordinary discretized Langevin methods, its rejection correction avoids the usual discretization bias that prevents polylogarithmic accuracy dependence.

---

## Summary of the appendices

| Appendix | Main role | Key insight |
|---|---|---|
| A | Comparison with concurrent work | Weaker assumptions and average \(L^2\) score errors |
| B | Analytical tools | Converts local errors into global divergence bounds |
| C | FORS proof | Unbiased gradient estimators can implement exponential rejection |
| D | Gaussian-tilt analysis | Gaussian proposal plus first-order correction gives high accuracy |
| E | Diffusion applications | Minimal-assumption, \(\sqrt d\), and intrinsic-dimension guarantees |
| F | Log-concave sampling | The same method extends to gradient-only proximal sampling |

### One-sentence takeaway

The appendices establish that **Gaussian proposals combined with Bernoulli-factory-style rejection correction can achieve polylogarithmic high-accuracy sampling using only gradients or score evaluations, for both diffusion models and log-concave distributions.**

<br/>
# refer format:



### BibTeX

```bibtex
@inproceedings{chen2026high,
  author    = {Fan Chen and Sinho Chewi and Constantinos Daskalakis and Alexander Rakhlin},
  title     = {High-Accuracy Sampling for Diffusion Models and Log-Concave Distributions},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  address   = {Seoul, South Korea},
  publisher = {PMLR},
  year      = {2026}
}
```

### 시카고 스타일 참고문헌

Chen, Fan, Sinho Chewi, Constantinos Daskalakis, and Alexander Rakhlin. “High-Accuracy Sampling for Diffusion Models and Log-Concave Distributions.” In *Proceedings of the 43rd International Conference on Machine Learning*. Vol. 306 of *Proceedings of Machine Learning Research*. Seoul, South Korea: PMLR, 2026.

### 시카고 스타일 각주 예시

1. Fan Chen, Sinho Chewi, Constantinos Daskalakis, and Alexander Rakhlin, “High-Accuracy Sampling for Diffusion Models and Log-Concave Distributions,” in *Proceedings of the 43rd International Conference on Machine Learning*, vol. 306, *Proceedings of Machine Learning Research* (Seoul, South Korea: PMLR, 2026).





