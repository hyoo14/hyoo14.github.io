---
layout: post
title:  "[2026]TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate"
date:   2026-03-27 02:59:57 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: TURBO QUANT는 입력 벡터를 **랜덤 회전**해 각 좌표가 Beta(고차원에서 거의 Normal) 분포를 따르게 만든 뒤, 좌표 간 **거의 독립성**을 이용해 **좌표별 최적 스칼라(Lloyd–Max) 양자화**로 MSE를 최소화하고, 내적에서는 **(b−1)-bit MSE 양자화 + 잔차에 1-bit QJL**을 추가하는 2-stage로 **편향 없는(unbiased) 내적 추정**을 만든다.


짧은 요약(Abstract) :


이 논문은 **고차원 벡터를 낮은 비트로 양자화(벡터 양자화, VQ)** 하면서도 원래 벡터의 **기하학적 구조(거리/내적)** 가 최대한 덜 망가지게(왜곡 최소화) 하는 **온라인(데이터-오블리비어스)** 알고리즘 **TURBO QUANT**를 제안합니다.

핵심 기여는 다음과 같습니다.

- 기존 방법들은 보통 **계산이 느리거나(오프라인 학습/튜닝 필요)**, 혹은 **이론적으로 최적 왜곡-비트율(distortion-rate)을 달성하지 못하는** 문제가 있었는데, TURBO QUANT는 이를 개선합니다.
- 입력 벡터에 **무작위 회전(random rotation)** 을 적용하면, 회전된 벡터의 각 좌표가 **Beta 분포(고차원에서는 거의 정규분포)** 를 따르고 좌표 간 **거의 독립**이 되므로, 좌표별로 **최적 스칼라 양자화기(각 좌표에 Lloyd-Max 최적 양자화)** 를 적용하는 단순한 방식으로도 **(상수배 오차만 남기고) 거의 최적의 MSE 왜곡률**을 달성합니다.
- 하지만 **MSE에 최적인 양자화는 내적 추정에서 편향(bias)** 을 만들 수 있음을 보이고, 이를 해결하기 위해  
  1) 먼저 MSE 양자화로 근사한 뒤  
  2) 남은 **잔차(residual)** 에 대해 **1-bit QJL(Quantized Johnson–Lindenstrauss) 변환**을 추가 적용하는 **2단계 방식**을 제안합니다.  
  이로써 **내적 추정이 “불편(unbiased)”** 이면서도 낮은 왜곡을 갖게 됩니다.
- 또한 어떤 벡터 양자화도 넘을 수 없는 **정보이론적 하한(lower bound)** 을 증명하고, TURBO QUANT가 그 하한과 **작은 상수(약 2.7) 배 차이**로 매우 근접함을 보입니다.
- 실험적으로는 LLM의 **KV cache 양자화**에서 **채널당 3.5비트로 품질 저하 없이(absolute quality neutrality)**, **2.5비트에서도 소폭 저하**만 보이며,  
  **최근접 이웃 검색**에서는 기존 **Product Quantization(PQ)** 계열보다 **리콜이 더 좋고 인덱싱 시간이 거의 0**에 가깝다고 보고합니다.

---



This paper studies vector quantization for high-dimensional Euclidean vectors, aiming to minimize both mean-squared error (MSE) and inner-product distortion. It proposes **TURBO QUANT**, a **data-oblivious (online-friendly)** quantization approach that achieves **near-optimal distortion–rate performance** (within a small constant factor) for all bit-widths and dimensions.

TURBO QUANT first applies a **random rotation** to the input, which induces a **concentrated Beta distribution** on coordinates and makes different coordinates **nearly independent** in high dimensions. This enables the method to apply **optimal scalar quantizers (Lloyd–Max) per coordinate**, yielding near-optimal MSE distortion.

Since MSE-optimal quantizers can be **biased for inner-product estimation**, the paper introduces a **two-stage scheme**: (i) an MSE quantizer, and (ii) a **1-bit Quantized JL (QJL)** transform applied to the residual, resulting in an **unbiased** inner-product quantizer with low distortion. The paper also proves **information-theoretic lower bounds** on achievable distortion rates and shows TURBO QUANT matches them up to a small constant (≈2.7). Experiments validate the theory, showing strong results for **KV cache quantization** (quality-neutral at 3.5 bits/channel) and improved **nearest-neighbor search** recall with near-zero indexing time compared to product quantization baselines.

원하시면 abstract 문장을 **문장별로 직역/의역** 형태로도 다시 써드릴게요.


* Useful sentences(information) :


**  Lloyd–Max 색을 몇 개만 남기고 단순화 
원래: 수천 가지 색
→ 몇 개 대표 색으로 바꿈
"디테일 줄이고 비슷한 색끼리 뭉침"

** (b−1)-bit MSE
 전체 그림을 저화질로 저장
해상도는 그대로 (차원 유지)
대신 색이 뭉개짐 (정밀도 감소)
"이미지는 그대로인데 좀 흐릿해짐"

* residual + 1-bit QJL
"틀린 부분만 대충 표시"
어디가 틀렸는지 체크
근데 정확한 색은 안 저장하고
"이쪽으로 더 밝음 / 어두움" 정도만 기록
"여기 좀 더 밝게! (정확한 값은 몰라도 방향은 앎)"

** 온라인
데이터를 한 번에 하나씩 보면서 처리

전체 데이터를 미리 다 보지 않음
들어오는 대로 바로 처리

** 데이터-오블리비언스
데이터 내용과 상관없이 항상 같은 방식으로 처리

값이 뭐든 상관없이 같은 규칙
데이터 보고 알고리즘이 바뀌지 않음


** Gaussian:
"데이터 값 자체의 분포"
(예: 키, 노이즈)

** Beta:
"확률 p 자체의 분포"
(예: 성공 확률이 얼마인지)


{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
*


<br/>
# Methodology


이 논문은 “모델 아키텍처”를 새로 만들기보다, **고차원 벡터(예: LLM의 KV cache, 임베딩 벡터)**를 **온라인(데이터-오블리비어스)** 방식으로 빠르고 이론보장 있게 양자화하는 **벡터 양자화(VQ) 알고리즘**을 제안합니다.

---

### 1) 문제 설정: 무엇을 최소화하려는가?
논문은 벡터 \(x\in\mathbb{R}^d\)를 **총 \(B=b\cdot d\) 비트(좌표당 b비트 평균)**로 압축하는 양자화기 \(Q:\mathbb{R}^d\to\{0,1\}^B\)와 복원기 \(Q^{-1}\)를 설계합니다(0.1절).

목표 왜곡(distortion)은 두 가지입니다.

1. **MSE 왜곡**
\[
D_{\text{mse}}=\mathbb{E}\|x-Q^{-1}(Q(x))\|_2^2
\]
2. **내적 왜곡(Inner product error)**  
\[
D_{\text{prod}}=\mathbb{E}\left|\langle y,x\rangle-\langle y,Q^{-1}(Q(x))\rangle\right|^2
\]
추가로 **내적 추정의 불편성(unbiasedness)**:
\[
\mathbb{E}\langle y,Q^{-1}(Q(x))\rangle=\langle y,x\rangle
\]

핵심은 “데이터 분포 가정 없이(worst-case)”, 그리고 “온라인/즉시 적용 가능(data-oblivious)”하면서도, **정보이론적 최저 왜곡률에 거의 근접**하는 방법을 만드는 것입니다(0.3절, Lower bound 요약).

---

### 2) TURBO QUANT의 핵심 아이디어(기법): “랜덤 회전 → 좌표별 스칼라 최적 양자화”
기존 VQ/PQ 계열은 (i) 오프라인 학습/캘리브레이션이 필요하거나 (ii) 왜곡률이 비트폭에 대해 최적이 아니거나 (iii) GPU/가속기 벡터화가 어렵다는 문제가 있다고 보고(서론, 0.2절), TURBO QUANT는 다음 구조로 해결합니다.

#### 2.1 랜덤 회전(Random Rotation)으로 “최악 입력”을 “균일 구면 분포”로 바꾸기
- 입력 \(x\)에 대해 **무작위 회전행렬 \(\Pi\)** 를 곱해 \(y=\Pi x\)로 바꿉니다(Algorithm 1 line 5).
- 그러면 \(y\)는 **단위구 \(S^{d-1}\)** 위에서 “균일”하게 분포한 것처럼 취급 가능해지고(1.1절), 각 좌표 \(y_j\)의 분포가 **Beta 분포**를 따른다는 성질을 사용합니다(**Lemma 1**).

> Lemma 1: \(x\)가 구면 균일이면 각 좌표 \(x_j\)는 (스케일된) Beta 분포  
> 고차원에서 이 분포는 \( \mathcal{N}(0,1/d)\)에 수렴.

또 고차원에서는 좌표 간 상관이 매우 작아지고 “거의 독립”이 되어(0.3절, 1.1절), **벡터 전체를 복잡하게 양자화**하지 않아도, **좌표별로 독립 스칼라 양자화**를 적용해도 전체 왜곡이 거의 최적 수준까지 내려간다는 논리를 씁니다.

#### 2.2 좌표별 “최적 스칼라 양자화”를 Lloyd–Max(=1D k-means)로 미리 계산
- 각 좌표가 따르는 분포 \(f_X\) (Beta 형태)가 주어졌을 때,
- \([-1,1]\) 구간을 \(2^b\)개 버킷으로 나누고 중심값(centroid) \(c_1,\dots,c_{2^b}\)를 찾아
- 기대 제곱오차를 최소화하는 **Lloyd-Max 양자화기**를 설계합니다(식 (3), 1.1절).

중요: 이 코드북(codebook)은 **데이터가 아니라 “차원 d와 비트폭 b에 의해” 결정**되므로,
- 한 번 오프라인으로 풀어 **미리 저장(precompute)** 해두고,
- 실제 온라인 양자화에서는 “가장 가까운 centroid 인덱스만 저장”하면 됩니다(0.3절).

---

### 3) 알고리즘 1: **TURBO QUANT\_mse** (MSE 최적화)
(Algorithm 1, 1.1절, Theorem 1)

#### 3.1 양자화(QUANTmse)
1. 랜덤 회전: \(y=\Pi x\)
2. 각 좌표 \(y_j\)에 대해 가장 가까운 centroid \(c_k\)의 **인덱스 idx\_j** 저장  
   \[
   idx_j=\arg\min_k |y_j-c_k|
   \]
3. 출력은 인덱스 벡터 \(idx\in[2^b]^d\) (좌표당 b비트)

#### 3.2 복원(DEQUANTmse)
1. 각 좌표를 centroid로 복원: \(\tilde y_j=c_{idx_j}\)
2. 역회전: \(\tilde x=\Pi^\top \tilde y\)

#### 3.3 이론 보장(왜곡률)
**Theorem 1**에서 단위노름 \(\|x\|=1\)일 때
\[
D_{\text{mse}}\le \sqrt{\frac{3\pi}{2}}\cdot 4^{-b}
\]
또 작은 비트폭에서의 수치 예: \(b=1,2,3,4\)일 때 \(D_{\text{mse}}\approx 0.36,0.117,0.03,0.009\).

---

### 4) 왜 MSE 최적 양자화만으로는 내적이 “편향(bias)”되는가?
논문은 **MSE 최적 양자화는 내적 추정에 편향을 유발**함을 명시합니다(1.2절).

예를 들어 \(b=1\)이고 충분히 큰 d에서,
- \(Q_{\text{mse}}(x)=\mathrm{sign}(\Pi x)\),
- 복원은 스케일이 붙어 \(Q^{-1}_{\text{mse}}(z)=\sqrt{2/(\pi d)}\,\Pi^\top z\) 꼴이 되어,
- 내적의 기대값이
\[
\mathbb{E}\langle y,\tilde x\rangle = \frac{2}{\pi}\langle y,x\rangle
\]
즉 **곱셈 편향(2/π)** 이 생깁니다(1.2절, Lemma 4 언급과 함께 서술).

이 편향은 b가 커지면 줄지만, **“모든 비트폭에서 불편(unbiased) 내적”**을 원하면 별도 장치가 필요합니다.

---

### 5) 알고리즘 2: **TURBO QUANT\_prod** (불편 내적 + 낮은 왜곡)
(Algorithm 2, 1.2절, Theorem 2, QJL Lemma 4)

핵심은 **2단계(two-stage)** 입니다.

#### 5.1 1단계: (b−1)-비트 MSE 양자화로 “잔차(residual)”를 작게 만들기
- 먼저 TURBO QUANT\_mse를 **b−1 비트폭**으로 적용:
  \[
  \tilde x_{\text{mse}} = Q^{-1}_{\text{mse}}(Q_{\text{mse}}(x))
  \]
- 잔차:
  \[
  r = x - \tilde x_{\text{mse}}
  \]
MSE 최적화는 곧 \(\|r\|\)를 작게 만드는 방향이므로, 2단계에서 처리할 “남은 오차”가 작아집니다.

#### 5.2 2단계: 잔차에 **QJL(Quantized Johnson–Lindenstrauss) 1-bit** 적용
- 무작위 가우시안 행렬 \(S\in\mathbb{R}^{d\times d}\), \(S_{ij}\sim\mathcal{N}(0,1)\) 생성(Algorithm 2 line 3)
- 잔차를 1-bit로:
  \[
  qjl = \mathrm{sign}(Sr)\in\{-1,+1\}^d
  \]
- 복원 시 QJL 디코딩은 (Definition 1, Algorithm 2 line 11):
  \[
  \tilde x_{\text{qjl}} = \sqrt{\frac{\pi}{2}}\frac{\gamma}{d}S^\top qjl
  \quad (\gamma=\|r\|_2)
  \]
- 최종 복원:
  \[
  \tilde x = \tilde x_{\text{mse}} + \tilde x_{\text{qjl}}
  \]

여기서 QJL은 **내적 불편성**을 보장합니다(**Lemma 4: Unbiased**).
따라서 전체도 불편해집니다(Theorem 2 증명 아이디어: 조건부기대 → 전체기대).

#### 5.3 이론 보장(불편성 + 내적 왜곡률)
**Theorem 2**:
- \(\mathbb{E}\langle y,\tilde x\rangle = \langle y,x\rangle\) (불편)
- 내적 왜곡 상계:
\[
D_{\text{prod}}
\le \sqrt{3\pi^2}\cdot \frac{\|y\|_2^2}{d}\cdot 4^{-b}
\]
작은 비트폭 수치 예: \(b=1,2,3,4\)일 때
\[
D_{\text{prod}}\approx \frac{1.57}{d},\frac{0.56}{d},\frac{0.18}{d},\frac{0.047}{d}
\]

---

### 6) “트레이닝 데이터”가 필요한가? (온라인/데이터-오블리비어스 특성)
- TURBO QUANT는 **데이터셋으로 codebook을 학습하지 않습니다.**
- 필요한 것은:
  1) 차원 \(d\), 비트폭 \(b\)에 따른 **이론적 분포(Beta/근사 Normal)** 기반의 centroid 테이블(식 (3))을 **미리 계산**해 두는 것  
  2) 랜덤 회전(및 QJL의 랜덤 행렬/구조화 변환)
- 그래서 논문이 강조하는 “online(data-oblivious) quantization” 범주에 속합니다(0.2절, 0.3절).

---

### 7) 구현/가속기 친화(특별한 아키텍처/커널 관점)
논문 구현 파트(부록 E)는 “이론상의 랜덤 회전 행렬”을 그대로 \(O(d^2)\)로 쓰지 않고, 실제로는:

- **구조화된 랜덤 회전(랜덤 Hadamard 변환)** 사용  
  좌표 부호 랜덤 플립 + FHT(Fast Hadamard Transform)로 \(O(d\log d)\)에 회전 적용(Implementation E).
- KV cache에서는 “양자화 값을 HBM에서 다시 full dequant해서 쓰지 말고”,  
  **fused GPU kernel**로 “온더플라이 dequant + GEMM(M=1)”을 수행해 메모리 트래픽을 줄이는 방향을 설명합니다(Implementation E, FLUTE 인용).

---

### 8) 정보이론적 하한과의 근접성(near-optimal rate)
논문은 Shannon lower bound + Yao minimax로 어떤 양자화도 피할 수 없는 하한을 제시합니다(Overview의 Lower Bound, Appendix Theorem 3).

- 어떤 \(b\)-bit/coord 양자화라도 어떤 어려운 입력에 대해:
  \[
  D_{\text{mse}}\ge 4^{-b}
  \]
  \[
  D_{\text{prod}}\ge \frac{\|y\|_2^2}{d}\cdot 4^{-b}
  \]
TURBO QUANT는 상계가 하계의 **상수배(약 2.7배)** 안에 들어 **거의 최적**이라고 주장합니다(0.3절).

---

##  Systematic description of the TURBO QUANT method

This summary is based on the paper’s relevant sections: **Overview (0.3)**, **Algorithms 1–2**, **Sections 1.1–1.2**, **Lemma 1 & Lemma 4**, **Theorems 1–2**, and **Implementation (Appendix E)**. The paper does not introduce a new neural architecture; instead, it proposes an **online, data-oblivious vector quantization algorithm** for high-dimensional vectors (e.g., LLM KV cache, embedding databases) with near-optimal distortion–rate guarantees.

### 1) Objective and distortion metrics
Given a randomized quantizer \(Q:\mathbb{R}^d\to\{0,1\}^{B}\) with \(B=b\cdot d\) bits (average \(b\) bits per coordinate) and a dequantizer \(Q^{-1}\), the paper targets:

- **MSE distortion**:  
  \(D_{\text{mse}}=\mathbb{E}\|x-Q^{-1}(Q(x))\|_2^2\)
- **Inner-product distortion**:  
  \(D_{\text{prod}}=\mathbb{E}|\langle y,x\rangle-\langle y,Q^{-1}(Q(x))\rangle|^2\)
- plus **unbiasedness** for inner products:  
  \(\mathbb{E}\langle y,Q^{-1}(Q(x))\rangle=\langle y,x\rangle\)

All guarantees are in a **worst-case input** sense (no assumptions on the data distribution).

### 2) Core technique: random rotation + coordinate-wise optimal scalar quantization
The key idea is to apply a **random rotation** \(y=\Pi x\). Then:
- each coordinate of \(y\) follows a **Beta distribution** (Lemma 1),
- and in high dimension coordinates become **nearly independent**, enabling **per-coordinate scalar quantization** without losing near-optimal distortion–rate behavior.

For the scalar quantizer, the centroids \(c_1,\dots,c_{2^b}\) are computed by solving a **1D continuous k-means / Lloyd–Max** problem (Eq. (3)) for the Beta (or near-Gaussian) coordinate distribution. These codebooks are **precomputed** for practical \(b\) and reused online.

### 3) Algorithm 1: TURBO QUANT\_mse
- Quantize: rotate \(x\), then for each coordinate store the nearest centroid index (b bits).
- Dequantize: replace indices by centroids and rotate back.

**Theorem 1**: for \(\|x\|=1\),
\[
D_{\text{mse}}\le \sqrt{\frac{3\pi}{2}}\cdot 4^{-b}
\]
with refined constants for small \(b\).

### 4) Why MSE-optimal quantizers are biased for inner products
Section 1.2 shows TURBO QUANT\_mse can be **biased** for inner-product estimation. For \(b=1\) in large \(d\),
\[
\mathbb{E}\langle y,\tilde x\rangle=\frac{2}{\pi}\langle y,x\rangle
\]
a multiplicative bias of \(2/\pi\).

### 5) Algorithm 2: TURBO QUANT\_prod (two-stage, unbiased)
To get unbiased inner products with low distortion:

**Stage 1:** apply TURBO QUANT\_mse with bit-width \(b-1\) to get \(\tilde x_{\text{mse}}\) and residual \(r=x-\tilde x_{\text{mse}}\).

**Stage 2:** apply **1-bit QJL** (Quantized Johnson–Lindenstrauss) to the residual:
\[
qjl=\text{sign}(Sr)
\]
and reconstruct using
\[
\tilde x_{\text{qjl}}=\sqrt{\frac{\pi}{2}}\frac{\gamma}{d}S^\top qjl,\quad \gamma=\|r\|_2.
\]
Final reconstruction: \(\tilde x=\tilde x_{\text{mse}}+\tilde x_{\text{qjl}}\).

By Lemma 4, QJL is unbiased for inner products; thus the combined estimator is unbiased (Theorem 2).

**Theorem 2**:
- Unbiased: \(\mathbb{E}\langle y,\tilde x\rangle=\langle y,x\rangle\)
- Distortion:
\[
D_{\text{prod}}\le \sqrt{3\pi^2}\cdot \frac{\|y\|_2^2}{d}\cdot 4^{-b}.
\]

### 6) Online / data-oblivious nature (no training data)
No dataset-specific training is required. Codebooks depend on \(d,b\) and the induced coordinate distribution after rotation, hence suitable for streaming/online scenarios (e.g., KV cache).

### 7) Implementation and accelerator-friendliness
Appendix E explains practical speedups:
- random rotations are implemented via **structured randomized Hadamard transforms** (sign flips + FHT), reducing cost to \(O(d\log d)\),
- for KV cache, a **fused GPU kernel** performs on-the-fly dequantization + matmul to reduce memory traffic (building on ideas like FLUTE).

### 8) Near-optimality via information-theoretic lower bounds
The paper proves lower bounds (Theorem 3) showing any \(b\)-bit/coord quantizer must have, for some hard instances:
\[
D_{\text{mse}}\ge 4^{-b},\quad
D_{\text{prod}}\ge \frac{\|y\|_2^2}{d}\cdot 4^{-b}.
\]
TURBO QUANT matches these rates up to a small constant factor (~2.7).

---




<br/>
# Results


---

## 1) 실험 환경(공통)
- **하드웨어**: “All experiments are performed using a single NVIDIA A100 GPU.” (Section 2)
- 실험은 크게
  1) 이론적 왜곡(distortion) 보장 검증(2.1),
  2) 다운스트림 태스크(KV cache quantization, long-context, NN search) 평가(2.2~2.4)
  로 구성됩니다.

---

## 2) 실험 2.1: 이론 검증(왜곡/바이어스/분산) — 데이터·메트릭·비교대상·결과

### (1) 테스트 데이터
- **DBpedia Entities dataset**
- 임베딩: “embedded in a **1536-dimensional** space via **OpenAI3 embeddings**”
- 분할:
  - 학습(quantize 대상/DB): **100,000 points**
  - 쿼리: **1,000 distinct entries**
  (Section 2.1)

### (2) 비교 방법(경쟁 모델/방법)
- **TURBO QUANTprod**: “optimized for unbiased inner product estimation”
- **TURBO QUANTmse**: “minimizes mean squared error (MSE)”
(Section 2.1)

### (3) 평가 메트릭(그림/수치에서 본 핵심)
- **Inner product distortion/error 분포**: Figure 1
- **Average inner product error** 및 **MSE**: Figure 2(a)(b)
- 또한 이론 **상한/하한**과 함께 플로팅:
  - inner-product error: 하한 \( \frac{1}{d}4^{-b} \), 상한 \( \sqrt{\frac{3\pi}{2}}\frac{1}{d}4^{-b}\) 형태로 표시(그림 2(a) 축 설명)
  - MSE: 하한 \(4^{-b}\), 상한 \( \sqrt{\frac{3\pi}{2}}4^{-b}\) (그림 2(b) 설명)
  (이 상·하한 형태는 본문 Theorem 1~3의 요약과 일치)

### (4) 핵심 비교 결과(논문이 직접 말하는 결론)
- 비트폭(b) 증가 시 두 방법 모두 **분산이 감소**:
  - “increasing the bit widths reduces variance in both methods.” (2.1)
- 하지만 **TURBO QUANTmse는 inner product 추정에 바이어스**가 존재:
  - “TURBO QUANTmse introduces bias … diminishes and converges to zero with higher bit widths.” (2.1)
- 반대로 **TURBO QUANTprod는 모든 비트폭에서 unbiased**:
  - “TURBO QUANTprod remains unbiased across all bit widths” (2.1)
- 추가 해석(2.1 마지막 문장):
  - 낮은 bit ratio에서는 **prod가 더 유리**
  - 비트 수가 커지면 **mse의 bias가 줄고**, 결국 inner product 추정에서도 **mse가 더 좋아질 수 있음**
  - “as the bit count increases… TurboQuant mse… ultimately achieves superior performance in inner product estimation.” (2.1)

---

## 3) 실험 2.2: Needle-In-A-Haystack(장문 컨텍스트 검색/회수) — 비교 및 결론

### (1) 테스트/모델/세팅
- 태스크: “Needle-In-A-Haystack Test”
  - 긴 문서(haystack) 안에 숨겨진 문장(needle)을 찾아야 함
- 모델: “**Llama-3.1-8B-Instruct**”
- 문서 길이: “from **4k to 104k tokens**”
- 메트릭: “**recall score**”
(Section 2.2)

### (2) 비교 방법(경쟁 모델)
- **PolarQuant** (Han et al. 2025a)
- **SnapKV** (Li et al. 2024)
- **PyramidKV** (Cai et al. 2024)
- **KIVI** (Liu et al. 2024b)
- **Full-Precision** baseline
- **TURBO QUANT**
(Section 2.2, Figure 3)

### (3) 비교 조건(공정 비교 조건)
- “all evaluated under a **memory compression ratio of 0.25** (25% of full KV cache)”  
  즉 **모두 KV cache를 4× 압축한 조건**에서 비교 (2.2)

### (4) 결과 요약(논문 서술 그대로)
- 이론적 보장이 있는 방법(PolarQuant, TURBO QUANT)이 토큰 압축(SnapKV/PyramidKV) 및 보장 없는 스칼라 양자화(KIVI)보다 낫다고 주장:
  - “methods with theoretical guarantees… outperform token-level compression … and scalar quantization methods without formal guarantees” (2.2)
- **가장 중요한 결론**:  
  - “**TURBO QUANT matches the performance of the full-precision model even at 4× compression**” (2.2)
- Figure 3 캡션도 동일 취지:
  - “despite being more than 4× quantized, achieves the same exact performance as the uncompressed baseline.”

---

## 4) 실험 2.3: LongBench(장문 이해/생성) — 테이블 기반 정량 비교

### (1) 데이터/설정
- 데이터: “**LongBench** dataset… using the more uniformly distributed **LongBench-E subset**”
- 비교 모델: **Llama-3.1-8B-Instruct**, **Ministral-7B-Instruct**
- 특징: TURBO QUANT는 스트리밍 생성 동안 **generated token도 계속 quantization 적용**  
  - “Unlike KIVI and PolarQuant, which skip quantization for generated tokens, TURBO QUANT applies quantization throughout the streaming process.” (2.3)

### (2) 메트릭(테이블 컬럼)
Table 1에 태스크별 점수:
- SingleQA, MultiQA, Summarization, Few shot, Synthetic, Code, Average  
(표 제목/열)

### (3) 비교 방법(표에 등장)
- (Llama-3.1-8B) Full Cache, KIVI(3bit/5bit로 표기), PolarQuant(3.9), TURBO QUANT(2.5 / 3.5)
- (Ministral-7B) Full Cache, TURBO QUANT(2.5)
(Table 1)

### (4) 결과 해석(논문이 강조하는 포인트)
- TURBO QUANT는 **낮은 비트(2.5-bit, 3.5-bit)에서도 강한 성능**
- “3.5 bits per channel”에서는 “absolute quality neutrality(절대적 품질 중립)”를 초록에서 주장했고,
- 표(Table 1)에서도 **Llama-3.1-8B에서 TURBO QUANT 3.5의 Average가 Full Cache Average와 동일(50.06)**로 제시됨.
  - Full Cache Average: **50.06**
  - TURBO QUANT 3.5 Average: **50.06**
- 또한 2.5-bit도 평균이 거의 비슷한 수준(49.74)로 제시됨.
- 압축률 관련 주장:
  - “matches the performance… while achieving over **4.5× compression**.” (2.3)

> 참고: 비정수 bit(2.5/3.5)의 이유도 본문에 직접 설명이 있습니다.  
> “two-tier channel-wise quantization strategy: outlier channels are allocated more bits …” (2.3)

---

## 5) 실험 2.4: Near Neighbor Search(최근접 탐색) — 데이터·메트릭·비교대상·결과

### (1) 테스트 데이터
- DBpedia Entities (OpenAI3 embeddings)
  - **d=1536**, **d=3072**
- 추가로 저차원 데이터: “standard **GloVe embeddings**”
  - 그림 4(a)에 **d=200**으로 표시
(Section 2.4, Figure 4)

### (2) 세팅
- 보통:
  - train/database: **100,000 points**
  - query: **1,000**
- 예외:
  - GloVe는 “pre-existing **10,000-query** set”
(2.4)

### (3) 메트릭
- **recall@k** (top-k 안에 정답 top inner product가 포함되는 비율)
- 그림 4: Recall@1@k 형태로 top-k(1,2,4,...,64) 비교
(2.4, Figure 4 설명)

### (4) 비교 방법(경쟁 모델)
- **TURBO QUANT**
- **Product Quantization (PQ)** (Douze et al. 2024 / Faiss)
- **RabitQ** (Gao et al. 2024)
(2.4)

### (5) 결과 결론(논문 서술)
- “Despite these advantages favoring the baselines, **TURBO QUANT consistently achieves higher recall@k** across all datasets and bit-widths…”
- 또한 indexing/전처리 시간 측면에서도 서론/초록에서 “reducing indexing time to virtually zero” 주장.

---

## 6) (추가) 정량 속도/시간 비교 근거: Figure 2(c), Table 2
- Figure 2(c): KV-cache에서 **QKᵀ 연산 speedup**을 PyTorch einsum 대비로 제시(1-bit/2-bit/4-bit)
- Table 2: “Quantization time (seconds) … using 4-bit quantization”
  - PQ: 37.04~494.42초(차원 증가에 따라 증가)
  - RabitQ: 597.25~3957.19초
  - TURBO QUANT: 0.0007~0.0021초
  → TURBO QUANT가 **수 자릿수(orders of magnitude) 더 빠르다**는 주장 뒷받침
(Appendix D, Table 2 캡션/본문)

---




Below is a structured summary of the paper’s **experimental results** 

## 1) Common setup
- Hardware: “All experiments are performed using a single NVIDIA A100 GPU.” (Sec. 2)
- Two groups of experiments:
  1) Empirical validation of theoretical distortion/bias results (2.1),
  2) Downstream tasks: KV cache quantization & long-context tasks, and NN search (2.2–2.4).

## 2) Experiment 2.1 (Empirical validation)
**Dataset**
- DBpedia Entities, embedded into **1536-D** using **OpenAI3 embeddings**
- **100,000** training/database points, **1,000** query points (Sec. 2.1)

**Methods compared**
- **TURBO QUANTprod** (unbiased inner-product focused)
- **TURBO QUANTmse** (MSE-focused)

**Metrics**
- Inner-product distortion/error histograms (Fig. 1)
- Average inner-product error and MSE vs theoretical upper/lower bounds (Fig. 2a,b)

**Key findings**
- Increasing bit-width reduces variance for both (Sec. 2.1)
- **TURBO QUANTmse is biased** for inner-product estimation; bias decreases with bit-width (Sec. 2.1)
- **TURBO QUANTprod remains unbiased** across all bit-widths (Sec. 2.1)
- At low bit ratios, **prod performs better** for inner products; at higher bits, mse bias vanishes and mse can become better for inner products (Sec. 2.1)

## 3) Experiment 2.2 (Needle-in-a-Haystack)
**Task/Model/Setting**
- Needle-in-a-Haystack retrieval test on **Llama-3.1-8B-Instruct**
- Context length: **4k to 104k tokens**
- Metric: **recall score** (Sec. 2.2)

**Baselines**
- PolarQuant, SnapKV, PyramidKV, KIVI, Full-Precision, and TURBO QUANT (Fig. 3)

**Comparison condition**
- All methods at **memory compression ratio 0.25** (i.e., 4× KV compression) (Sec. 2.2)

**Result**
- Methods with theoretical guarantees (PolarQuant, TURBO QUANT) outperform token compression (SnapKV/PyramidKV) and methods without formal guarantees (KIVI) (Sec. 2.2)
- **TURBO QUANT matches full-precision performance even at 4× compression** (Sec. 2.2 and Fig. 3 caption)

## 4) Experiment 2.3 (LongBench end-to-end generation)
**Dataset**
- LongBench, using **LongBench-E** subset (Sec. 2.3)

**Models**
- Llama-3.1-8B-Instruct, Ministral-7B-Instruct

**Notable setting**
- TURBO QUANT quantizes throughout streaming, unlike KIVI/PolarQuant which skip generated tokens (Sec. 2.3)

**Metrics**
- Task scores (SingleQA, MultiQA, Summarization, Few shot, Synthetic, Code) and Average (Table 1)

**Result highlights**
- Strong performance even at **2.5-bit and 3.5-bit**
- For Llama-3.1-8B, **TURBO QUANT 3.5-bit matches Full Cache Average (50.06)** (Table 1)
- Claims **>4.5× compression** while matching unquantized baselines (Sec. 2.3)

## 5) Experiment 2.4 (Near neighbor search)
**Datasets**
- DBpedia Entities with OpenAI3 embeddings: **d=1536** and **d=3072**
- GloVe embeddings: **d=200** (Fig. 4a) (Sec. 2.4)

**Split**
- Typically 100k database / 1k queries; for GloVe 10k queries (Sec. 2.4)

**Metric**
- **recall@k** (whether true top inner product appears in top-k approximated results) (Sec. 2.4)

**Baselines**
- **Product Quantization (PQ)** and **RabitQ** (Sec. 2.4)

**Result**
- TURBO QUANT achieves **higher recall@k consistently across datasets and bit-widths** (Sec. 2.4)

## 6) Extra efficiency evidence (Fig. 2c, Table 2)
- Fig. 2c: speedup for QKᵀ in KV-cache vs PyTorch einsum (1/2/4-bit)
- Table 2: 4-bit quantization time (seconds): TURBO QUANT is orders of magnitude faster than PQ and RabitQ (Appendix D)

---



<br/>
# 예제
## (논문에서 관련 부분 발췌 기반) 실험 예시를 “입력/출력/테스크” 관점에서 길고 체계적으로 정리 

아래 예시는 질문하신 것처럼 **트레이닝 데이터/테스트 데이터의 구체적인 인풋·아웃풋 형태**, 그리고 **어떤 테스크를 어떻게 수행했는지**를 논문 본문에 나온 실험 섹션(2.1~2.4)과 표/그림 설명을 근거로 재구성한 것입니다.

---

# 1) 예시 A: “왜 TurboQuant-prod는 inner product가 unbiased인가?”를 검증하는 실험 (Section 2.1 Empirical Validation)

## 1.1 테스크(무엇을 했나?)
- 목적: 벡터를 저비트로 양자화했을 때  
  1) **내적(Inner Product) 오차 분포가 어떻게 되는지**,  
  2) 특히 **TurboQuant-mse는 편향(bias)이 생기고**, **TurboQuant-prod는 모든 비트폭에서 unbiased**인지 확인.
- 비교 대상:
  - **TURBO QUANT\_prod**: 내적 추정이 unbiased가 되도록 설계된 2-stage 방식 (MSE 양자화 + 잔차에 1-bit QJL)
  - **TURBO QUANT\_mse**: MSE 최소화가 목표(내적은 bias가 있을 수 있음)

## 1.2 데이터(트레이닝/테스트가 구체적으로 뭐냐?)
- 데이터셋: **DBpedia Entities**
- 임베딩: **OpenAI3 embeddings**
- 벡터 차원: **d = 1536**
- 분할:
  - **Training set(데이터베이스 역할)**: 무작위로 **100,000개 벡터**
  - **Query set(테스트 쿼리 역할)**: 서로 다른 엔트리에서 **1,000개 벡터**

즉, 벡터 검색 세팅처럼:
- “DB(학습/저장용) 벡터”는 10만 개,
- “질의(query) 벡터”는 1천 개를 따로 뽑아 실험합니다.

## 1.3 인풋/아웃풋(구체적인 형태)
### (1) 인풋
- 양자화 대상 입력: training set의 각 벡터 \(x_i \in \mathbb{R}^{1536}\)
- 내적 평가용 입력: query 벡터 \(q_j \in \mathbb{R}^{1536}\)

### (2) 양자화 출력(저장되는 것)
- **TurboQuant-mse** (Algorithm 1)
  - 출력: 각 좌표별로 codebook index  
    \[
    \text{idx} \in [2^b]^{d}
    \]
  - 복원: centroid로 좌표 복원 후, 회전 되돌림(\(\Pi^\top\))

- **TurboQuant-prod** (Algorithm 2)
  - 출력이 더 복합적(2-stage):
    1) MSE 부분 인덱스: \(\text{idx} \in [2^{b-1}]^d\)
    2) 잔차(residual) 부호벡터(QJL): \(\text{qjl} \in \{-1,+1\}^d\)
    3) 잔차 크기 스칼라: \(\gamma = \|r\|_2\)
  - 즉 출력은:
    \[
    (\text{idx}, \text{qjl}, \gamma)
    \]

### (3) 평가 출력(우리가 관찰하는 최종 출력)
- 목표는 내적 \(\langle q_j, x_i\rangle\)를 양자화된 복원값으로 근사:
  \[
  \langle q_j, \tilde{x}_i\rangle
  \]
- 그래서 최종적으로 측정하는 출력은:
  - 내적 오차: \(\langle q, x\rangle - \langle q, \tilde{x}\rangle\)
  - 내적 오차 제곱의 기대값(분산/왜곡): \(D_{prod}\)
  - MSE 왜곡: \(D_{mse}\)

## 1.4 실험 절차(“어떻게” 했나?)
1. Training set 10만 개 벡터를 비트폭 \(b=1,2,3,4\) 등으로 양자화
2. Query set 1천 개 벡터로 training 벡터들과의 내적을 추정
3. 각 비트폭에서:
   - **오차 분포 히스토그램**(Fig.1)
   - **평균 내적오차 및 MSE를 이론 상/하한과 같이 plot**(Fig.2)

## 1.5 관찰/결론(논문이 말하는 핵심 결과)
- **TurboQuant-mse는 낮은 비트폭에서 내적에 bias가 생김**  
  - 논문은 b=1의 경우 예시로 “곱셈 편향이 \(2/\pi\)”가 생긴다고 설명합니다(Section 1.2).
- **TurboQuant-prod는 모든 비트폭에서 unbiased** (Fig.1/2에서 확인)
- 비트폭이 커지면 mse 방식도 bias가 줄어드는 경향.

---

# 2) 예시 B: KV Cache 양자화로 “Needle-In-A-Haystack” 장문 검색(회수) 능력 평가 (Section 2.2)

## 2.1 테스크(무엇을 했나?)
- “Needle-in-a-haystack” 테스트: 긴 문서(수만 토큰) 안에 숨겨둔 문장(needle)을 모델이 정확히 찾아 답변/회수하는지 평가
- 평가 모델: **Llama-3.1-8B-Instruct**
- 컨텍스트 길이: **4k ~ 104k 토큰**까지 변화
- 평가 지표: **recall score** (숨겨진 문장을 제대로 회수했는지)

## 2.2 인풋/아웃풋(구체적으로 뭐가 들어가고 나오나?)
### (1) 인풋
- 모델 입력 프롬프트(긴 문서):  
  - 일반 텍스트 토큰 시퀀스(길이가 4k~104k)
  - 어딘가에 특정 문장(needle)이 삽입됨

### (2) 양자화 대상
- Transformer가 이전 토큰들의 attention을 위해 저장하는 **KV cache**:
  - Key/Value 행렬 \(K,V \in \mathbb{R}^{n \times d}\) (n은 시퀀스 길이)
- TurboQuant를 이용해 KV를 저비트로 저장

### (3) 아웃풋
- 모델이 최종 생성하는 답변 텍스트
- 그리고 그 답변이 needle을 정확히 포함/회수했는지로 recall을 계산

## 2.3 비교 조건(논문에 명시)
- 비교 방법: PolarQuant, SnapKV, PyramidKV, KIVI, Full-Precision, TurboQuant
- 메모리 조건: **compression ratio 0.25** (즉 KV 캐시를 25% 메모리만 사용하도록 제한)
- 결과: Fig.3에서 **TurboQuant가 Full-Precision과 동일한 성능(점수 0.997)을 달성**했다고 보고

---

# 3) 예시 C: LongBench 장문 생성 성능(정답률) 평가 (Section 2.3 + Table 1)

## 3.1 테스크
- 데이터셋: **LongBench** (LongBench-E subset)
- 모델:
  - Llama-3.1-8B-Instruct
  - Ministral-7B-Instruct
- 여러 태스크 카테고리:
  - SingleQA / MultiQA / Summarization / Few-shot / Synthetic / Code 등

## 3.2 인풋/아웃풋
- 인풋: 각 LongBench 태스크의 입력 문서/질문(장문 컨텍스트 포함)
- 중간 양자화: 생성 중 계속 누적되는 KV cache를 TurboQuant로 양자화(논문은 **생성 토큰도 스트리밍 중 양자화**한다고 명시)
- 아웃풋: 모델이 생성한 답변 텍스트
- 평가: 태스크별 점수 및 평균(Table 1)

## 3.3 “2.5-bit, 3.5-bit”는 무엇을 의미하나? (논문 설명)
- 비정수 비트폭은 “채널별로 outlier 채널에 더 많은 비트를 배정”하는 혼합 전략에서 발생:
  - 예: 2.5-bit = 일부 채널은 3비트, 일부는 2비트

## 3.4 결과 요약
- Table 1에서:
  - **3.5 bits/channel에서 full cache와 평균 성능이 동일(50.06)** 수준으로 보고
  - 2.5 bits/channel에서도 근소한 하락 또는 유사

---

# 4) 예시 D: 최근접 이웃(ANN) 벡터 검색 recall@k 비교 (Section 2.4 + Fig.4)

## 4.1 테스크
- 목표: 데이터베이스 벡터를 양자화해 저장한 뒤, 쿼리 벡터가 왔을 때 “내적(top inner product)” 기준 상위 k개를 얼마나 잘 맞추는지 평가
- 지표: **recall@k**

## 4.2 데이터/차원/분할(논문 명시)
- DBpedia Entities (OpenAI3 임베딩):
  - d=1536, d=3072 실험
- GloVe 임베딩:
  - d=200 실험
- 분할:
  - training set(=DB): 100,000
  - query: 1,000 (GloVe는 10,000 queries)

## 4.3 인풋/아웃풋
- 인풋:
  - DB 벡터 \(x_i\)
  - 쿼리 벡터 \(q\)
- 출력:
  - 근사 내적으로 정렬한 top-k 결과 리스트
  - 정답 top-1(또는 top-k)이 그 리스트에 포함되는 비율 = recall@k

## 4.4 비교 대상 및 결론
- 비교:
  - Product Quantization(PQ)
  - RabitQ
  - TurboQuant
- 결과:
  - Fig.4에서 TurboQuant가 여러 데이터/차원/비트폭에서 **더 높은 recall@k**를 보고

---



## Systematic, example-driven description of the experiments (inputs/outputs/tasks), based on the paper’s relevant sections

---

# 1) Example A: Empirical validation of unbiased inner-product estimation (Section 2.1)

**Task.** Evaluate how vector quantization affects (i) inner-product distortion and (ii) bias, comparing **TURBO QUANT\_mse** vs **TURBO QUANT\_prod** across bit-widths.

**Dataset.**
- DBpedia Entities, embedded with **OpenAI3 embeddings**
- Dimension: **d = 1536**
- Split:
  - “Training set” (database vectors): **100,000 vectors**
  - “Query set”: **1,000 vectors**

**Inputs.**
- Quantization input: each database vector \(x_i \in \mathbb{R}^{1536}\)
- Evaluation input: each query vector \(q_j \in \mathbb{R}^{1536}\)

**Quantized outputs stored.**
- **TurboQuant-mse (Alg. 1):** index vector \(\text{idx} \in [2^b]^d\)
- **TurboQuant-prod (Alg. 2):**  
  \(\text{idx} \in [2^{b-1}]^d\), \(\text{qjl} \in \{-1,+1\}^d\), and \(\gamma=\|r\|_2\), i.e., \((\text{idx},\text{qjl},\gamma)\)

**Evaluation output.**
- Approximate inner products \(\langle q_j,\tilde{x}_i\rangle\) and errors  
  \(\langle q,x\rangle - \langle q,\tilde{x}\rangle\)
- Plots:
  - Error histograms (Fig. 1)
  - Average distortions vs theoretical bounds (Fig. 2)

**Key finding.**
- **TurboQuant-mse is biased at low bit-widths** (paper gives the \(2/\pi\) multiplicative bias example for \(b=1\)).
- **TurboQuant-prod remains unbiased at all bit-widths**, matching theory.

---

# 2) Example B: KV-cache quantization on Needle-in-a-Haystack (Section 2.2)

**Task.** Long-context retrieval: the model must recover a hidden sentence (“needle”) embedded in a very long prompt (“haystack”).

**Model & setup.**
- Llama-3.1-8B-Instruct
- Context length: **4k to 104k tokens**
- Metric: recall score

**Inputs.**
- A long token sequence prompt with an inserted needle sentence.

**What is quantized?**
- Transformer **KV cache** \(K,V \in \mathbb{R}^{n\times d}\) during decoding, using TurboQuant under a memory compression ratio of **0.25**.

**Outputs.**
- Generated answer text; recall computed from whether the needle is correctly recovered.

**Key result.**
- TurboQuant matches **full precision** performance in Fig. 3 under 4× compression (reported score 0.997).

---

# 3) Example C: End-to-end generation on LongBench (Section 2.3, Table 1)

**Task.** Evaluate long-context generation performance across multiple categories (SingleQA, MultiQA, Summarization, Few-shot, Synthetic, Code).

**Inputs/Outputs.**
- Inputs: LongBench prompts (long contexts + questions/tasks)
- Outputs: model-generated responses; task scores aggregated in Table 1

**Quantization detail.**
- TurboQuant applies quantization **throughout streaming**, including generated tokens.
- Non-integer bit-widths (e.g., 2.5, 3.5 bits/channel) come from a mixed-precision, channel-wise allocation strategy (outlier channels get more bits).

---

# 4) Example D: Approximate nearest neighbor search recall@k (Section 2.4, Fig. 4)

**Task.** Quantize database vectors, approximate inner products with queries, and measure **recall@k**.

**Datasets & dimensions.**
- DBpedia + OpenAI3 embeddings: **d=1536, 3072**
- GloVe: **d=200**
- Split: database = 100,000; queries = 1,000 (GloVe uses 10,000 queries)

**Inputs/Outputs.**
- Inputs: DB vectors \(x_i\), query vectors \(q\)
- Output: top-k list by approximate inner product; recall@k computed.

**Key result.**
- TurboQuant achieves higher recall@k than PQ and RabitQ in Fig. 4.

---



<br/>
# 요약
TURBO QUANT는 입력 벡터를 **랜덤 회전**해 각 좌표가 Beta(고차원에서 거의 Normal) 분포를 따르게 만든 뒤, 좌표 간 **거의 독립성**을 이용해 **좌표별 최적 스칼라(Lloyd–Max) 양자화**로 MSE를 최소화하고, 내적에서는 **(b−1)-bit MSE 양자화 + 잔차에 1-bit QJL**을 추가하는 2-stage로 **편향 없는(unbiased) 내적 추정**을 만든다.  
이때 왜곡률은 정보이론 하한(Shannon lower bound)과 거의 일치하며, **MSE는 ≤ (√(3π)/2)·4^{-b}**, **내적오차는 ≤ (√(3)π/2)·(||y||²/d)·4^{-b}**로 하한 대비 상수배(≈2.7) 이내이고, 작은 비트폭에서도 예시로 **b=1..4에서 MSE≈0.36/0.117/0.03/0.009**, **내적오차≈1.57/d, 0.56/d, 0.18/d, 0.047/d**를 제시한다.  
실험적으로 KV-cache 양자화에서 **3.5 bits/channel은 품질 완전 유지**, **2.5 bits/channel은 경미한 저하**를 보였고, 최근접 이웃 검색에서는 기존 PQ류보다 **recall이 높고 인덱싱 시간이 거의 0**에 가깝다고 보고한다.  


 
TURBO QUANT first applies a **random rotation** so each coordinate follows a Beta distribution (nearly Normal in high dimensions), then exploits **near-independence across coordinates** to run **per-coordinate optimal scalar (Lloyd–Max) quantization** for MSE; for inner products it uses a 2-stage scheme: **(b−1)-bit MSE quantization + 1-bit QJL on the residual** to obtain an **unbiased** estimator.  
Its distortion rates nearly match information-theoretic limits: **MSE ≤ (√(3π)/2)·4^{-b}** and **inner-product error ≤ (√(3)π/2)·(||y||²/d)·4^{-b}**, within a small constant factor (~2.7) of the lower bound, with examples **MSE≈0.36/0.117/0.03/0.009 for b=1..4** and **inner-product error≈1.57/d, 0.56/d, 0.18/d, 0.047/d**.  
Empirically, for KV-cache quantization it achieves **quality neutrality at 3.5 bits/channel** and only **marginal degradation at 2.5 bits/channel**, and for nearest-neighbor search it beats prior PQ methods in **recall** while making **indexing time virtually zero**.

<br/>
# 기타




---

## 1) Figure 1: Inner Product error 분포 히스토그램 (TurboQuant\_prod vs TurboQuant\_mse)
**관련 부분(Section 2.1 Empirical Validation)**  
- “As shown in Fig. 1… TURBO QUANTmse introduces bias… In contrast, TURBO QUANTprod remains unbiased across all bit widths…”

**결과 요약**
- (a) **TURBO QUANT\_prod**: 비트폭(b=1~4)이 증가할수록 분산이 줄어들며, 분포가 **0을 중심**으로 대칭 → **항상 unbiased**.
- (b) **TURBO QUANT\_mse**: 분산은 줄어도 분포가 **0에서 치우침(바이어스)**, 특히 낮은 비트에서 두드러짐. 비트폭이 커지면 바이어스가 줄어 0으로 수렴.

**인사이트**
- **MSE 최적화(좌표별 Lloyd-Max)**는 복원 벡터의 내적을 “정확히” 맞추는 목적함수가 아니라서, **내적 추정에는 구조적 바이어스가 생길 수 있음**을 실험적으로 보여줌.
- 제안한 **2-stage(TurboQuant\_prod)**가 “잔차(residual)에 QJL 1-bit를 추가”하여 **unbiased 내적**을 달성한다는 이론(정리 2)을 분포 수준에서 확인.

---

## 2) Figure 2: (a) 내적 오류 vs 이론 경계, (b) MSE vs 이론 경계, (c) KV-cache QKᵀ speedup
### Figure 2(a): Inner-product error (D_prod)
**관련 부분(Section 2.1)**  
- “we also plot in Fig. 2 the average inner product error… against the upper and lower bounds…”

**결과 요약**
- b가 증가할수록 D_prod가 감소하며, 관측값이 **이론적 upper/lower bound 사이**에 잘 놓임.
- 낮은 비트폭에서는 **TurboQuant\_prod가 더 유리**(바이어스 없이 설계된 효과).
- 비트가 커지면 TurboQuant\_mse도 바이어스가 줄어들어 내적에서도 좋아질 수 있음을 언급.

**인사이트**
- 논문의 핵심 주장인 “**near-optimal distortion rate**”가 단지 이론이 아니라 **실측 평균 오류가 경계에 근접**함을 보여줌.
- 특히 low-bit에서 **unbiased 설계의 실익**이 큼.

### Figure 2(b): MSE (D_mse)
**관련 부분(Section 2.1 + Theorem 1)**

**결과 요약**
- TurboQuant\_mse의 MSE가 b 증가에 따라 빠르게 감소하며, 이론 bound(대략 4^{-b} 스케일) 트렌드와 일치.

**인사이트**
- 랜덤 회전 후 좌표별 스칼라 양자화가 “벡터 양자화” 문제에서 **rate–distortion을 거의 최적으로** 따라간다는 근거.

### Figure 2(c): Speedup over einsum (KV-cache의 QKᵀ)
**관련 부분(캡션 + 실험 섹션 2, 구현 Appendix E와 연결됨)**  
- “Speedup is measured relative to the PyTorch einsum baseline.”

**결과 요약**
- 시퀀스 길이가 길어질수록(16k→1M) TurboQuant 저비트 커널이 **einsum 대비 큰 속도 향상**을 보임.
- 1/2/4-bit 모두 가속이 나타나며, 압축으로 인해 메모리 병목이 줄어든 효과를 시사.

**인사이트**
- KV-cache는 HBM↔SRAM/연산 이동이 병목인데, **저비트 저장+온더플라이 연산**이 실제로 end-to-end 성능 이득을 만든다는 근거.
- 이 그래프는 “정확도”가 아니라 **서빙 효율(속도/메모리)** 관점에서 TurboQuant의 실용성을 뒷받침.

---

## 3) Figure 3: Needle-in-a-Haystack (Llama-3.1-8B-Instruct, 4k~104k)
**관련 부분(Section 2.2)**  
- “As shown in Fig. 3… TURBO QUANT matches the performance of the full-precision model even at 4× compression…”

**결과 요약**
- 동일 압축비(0.25, 즉 KV를 25%만 사용) 조건에서:
  - SnapKV/PyramidKV(토큰 압축)는 일부 구간에서 리콜 저하.
  - PolarQuant와 TurboQuant(이론 보장 기반 양자화)는 강함.
  - 특히 **TurboQuant가 Full-Precision과 사실상 동일한 점수(Score 0.997)**.

**인사이트**
- 장문 컨텍스트에서 “needle”을 찾는 능력은 attention의 내적/거리 구조 보존이 중요한데, TurboQuant가 **내적 왜곡을 작게/편향 없이** 유지하는 설계가 실제 장문 태스크에서 유효함을 보여줌.
- “4× 압축인데도 원본과 동일 성능”은 KV-cache 양자화가 **품질-중립(quality-neutral)**이 될 수 있음을 시사.

---

## 4) Table 1: LongBench-V1 (여러 태스크 평균, KV size 비교)
**관련 부분(Section 2.3)**  
- “As shown in Table 1… 2.5-bit and 3.5-bit… matches the performance of unquantized baselines…”

**결과 요약**
- Llama-3.1-8B-Instruct:
  - Full Cache(16) 평균 50.06
  - TurboQuant 3.5-bit(KV size 3.5) 평균 **50.06 (동일)**  
  - TurboQuant 2.5-bit(KV size 2.5) 평균 49.74(소폭 하락)
- Ministral-7B-Instruct에서도 2.5-bit가 거의 유지(49.89→49.62)

**인사이트**
- “3.5 bits per channel에서 **absolute quality neutrality**”라는 초록의 주장 근거가 되는 표.
- 2.5-bit도 평균 성능이 거의 유지되어, **대규모 KV 메모리 절감(4.5× 이상)**과 실사용 가능성을 강조.
- 또한 “비정수 비트폭”은 채널별 mixed precision(아웃라이어 채널에 더 많은 비트) 전략에서 옴(논문에 설명).

---

## 5) Figure 4: Nearest Neighbor Search recall@k (GloVe d=200, OpenAI3 d=1536/3072)
**관련 부분(Section 2.4)**  
- “TURBO QUANT consistently achieves higher recall@k…”

**결과 요약**
- 모든 데이터셋/차원에서 TurboQuant(2-bit, 4-bit)가 PQ 및 RabitQ 대비 **recall@k가 더 높게** 나오는 경향.

**인사이트**
- 기존 PQ는 오프라인 k-means codebook 학습이 필요하고(시간/비용), low-bit LUT16에서 품질이 흔들릴 수 있는데, TurboQuant는 **온라인/데이터-불요(data-oblivious)**임에도 recall이 좋음.
- 차원이 클수록 랜덤 회전 후 좌표 분포/준독립성 가정이 더 잘 맞아, TurboQuant 설계가 **고차원에서 특히 강점**을 가질 가능성을 뒷받침.

---

## 6) Appendix D Figure 5: 평균 inner product 크기에 따른 왜곡 특성(특히 바이어스)
**관련 부분(Appendix D)**  
- “variance remains constant… TurboQuant_mse bias dependent on average inner product…”

**결과 요약**
- TurboQuant\_prod: 평균 내적(Avg IP)이 달라도 **오류 분산이 거의 일정**.
- TurboQuant\_mse: Avg IP가 커질수록 **바이어스가 커짐**(분포 중심이 더 이동).

**인사이트**
- MSE 최적 양자화의 내적 바이어스가 “일정한 상수”가 아니라 **질의/데이터의 내적 스케일에 의해 체감이 달라질 수 있음**을 시각적으로 보여줌.
- 반대로 TurboQuant\_prod는 unbiased 설계라 “내적 크기”가 달라도 통계적 성질이 안정적 → 검색/attention 같이 다양한 내적 범위를 다루는 작업에 유리.

---

## 7) Appendix D Table 2: Quantization time(인덱싱/양자화 시간) 비교
**관련 부분(Appendix D)**  
- “TURBO QUANT is significantly faster—by several orders of magnitude…”

**결과 요약(4-bit, 100k 벡터)**
- d=1536: PQ 239.75s, RabitQ 2267.59s, TurboQuant **0.0013s**
- d=3072: PQ 494.42s, RabitQ 3957.19s, TurboQuant **0.0021s**

**인사이트**
- TurboQuant의 “온라인 적용/인덱싱 시간이 사실상 0”이라는 초록의 주장에 대한 직접 근거.
- PQ는 k-means 학습(오프라인)이 병목, RabitQ는 GPU/벡터화가 어려움. TurboQuant는 **완전 벡터화 가능** 설계(랜덤 회전+좌표별 양자화+간단 LUT)라 GPU에서 압도적으로 빠름.

---

## 8) Appendix E: Implementation 인사이트(구조적 회전, 검색/LLM 커널 결합)
**관련 부분(Appendix E Implementation)**

**핵심 내용/인사이트**
- 랜덤 회전 Π를 “일반 행렬”이 아니라 **구조적 랜덤 Hadamard 변환(FHT)**로 구현해 O(d log d)로 가속.
- 벡터 검색에서는 쿼리 q에 대해 LUT를 미리 만들고(PSHUFB 등 SIMD), 데이터의 양자화 인덱스로 빠르게 내적을 근사.
- KV-cache는 “복원값을 HBM에 다시 쓰지 않고”, **fused GPU kernel에서 on-the-fly dequant + GEMM**으로 처리(FLUTE류 커널 아이디어 활용).  
  → 메모리 트래픽 최소화가 핵심.

---

# Results & Insights for figures/tables/appendices

## Fig. 1 (Histograms): Inner-product error distribution
**Where:** Section 2.1  
**Result:** TurboQuant_prod stays centered at zero (unbiased) for all bit-widths; TurboQuant_mse shows a clear bias at low bits that shrinks as b increases.  
**Insight:** MSE-optimal quantization can be systematically biased for inner products; the proposed 2-stage residual+1-bit QJL fix achieves unbiasedness in practice.

## Fig. 2(a)(b): Distortion vs theory; Fig. 2(c): KV-cache speedup
**Where:** Section 2.1 + captions; ties to Theorems 1–2 and Appendix E  
- **(a)** Observed inner-product error tracks the theoretical upper/lower bounds; TurboQuant_prod is especially strong at low bits.  
- **(b)** MSE decreases with b following the predicted ~4^{-b} trend.  
- **(c)** Large speedups over PyTorch einsum for QKᵀ as sequence length grows, showing practical system gains from low-bit KV quantization.  
**Insight:** Confirms “near-optimal rate–distortion” empirically and demonstrates real inference acceleration benefits.

## Fig. 3: Needle-in-a-Haystack (long-context retrieval)
**Where:** Section 2.2  
**Result:** Under 4× KV compression (ratio 0.25), TurboQuant matches full precision performance, outperforming token compression baselines.  
**Insight:** Preserving geometric structure (inner products/distances) via principled quantization can be quality-neutral even for very long contexts.

## Table 1: LongBench results + KV size
**Where:** Section 2.3  
**Result:** 3.5-bit TurboQuant matches the full-cache average score on Llama-3.1-8B; 2.5-bit shows only marginal degradation; similar trend on Ministral.  
**Insight:** Supports the “quality neutrality at 3.5 bits/channel” claim and shows strong trade-offs at 2.5 bits with major memory savings.

## Fig. 4: ANN recall@k across datasets/dimensions
**Where:** Section 2.4  
**Result:** TurboQuant yields higher recall@k than PQ and RabitQ at 2/4 bits across dimensions (200/1536/3072).  
**Insight:** Data-oblivious, online-friendly quantization can outperform data-dependent PQ, especially in high dimensions.

## Appendix D Fig. 5: Dependence on average inner product
**Where:** Appendix D  
**Result:** TurboQuant_prod’s variance is stable across different Avg IP; TurboQuant_mse’s bias increases with Avg IP.  
**Insight:** MSE quantizers’ inner-product bias can worsen depending on query/data regimes; unbiased design provides robustness.

## Appendix D Table 2: Quantization time
**Where:** Appendix D  
**Result:** TurboQuant quantization time is orders of magnitude smaller than PQ/RabitQ (near-zero indexing time).  
**Insight:** Vectorizable, GPU-friendly design is a major practical advantage.

## Appendix E: Implementation
**Where:** Appendix E  
**Insight:** Structured random rotations (Hadamard/FHT), LUT-based inner products, and fused on-the-fly dequant+GEMM kernels explain why the method is both fast and scalable.

---



<br/>
# refer format:
```bibtex
@inproceedings{ZandiehDaliriHadianMirrokni2026TurboQuant,
  title        = {TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate},
  author       = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle    = {International Conference on Learning Representations (ICLR)},
  year         = {2026},
  note         = {Published as a conference paper at ICLR 2026}
}
```

Zandieh, Amir, Majid Daliri, Majid Hadian, and Vahab Mirrokni. “TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate.” In *International Conference on Learning Representations (ICLR)*, 2026. Published as a conference paper at ICLR 2026.
