---
layout: post
title:  "[2025]Emergence in Non-Neural Models: Grokking Modular Arithmetic via Average Gradient Outer Product"  
date:   2025-07-27 02:13:40 +0200
categories: study
---

{% highlight ruby %}


한줄 요약: 


이 논문은 AGOP(평균 그래디언트 외적)를 활용한 RFM(Recursive Feature Machine)이라는 비신경망 모델을 통해 모듈러 산술 문제에서의 feature 학습과 grokking 현상을 분석(RFM과 신경망이 공통적으로 학습하는 feature가 **block-circulant 구조**를 가지며, 이는 이전에 제안된 **Fourier multiplication algorithm**과 동일한 알고리즘을 구현한다는 점도 밝힘)     


짧은 요약(Abstract) :    




이 논문은 **모듈러 산술(modular arithmetic)** 문제를 푸는 동안 **grokking 현상**이 신경망뿐만 아니라 비신경망(non-neural) 모델에서도 나타난다는 점을 보여줍니다. 일반적으로 grokking은 훈련 정확도는 이미 100%인데 테스트 정확도는 한참 뒤에야 갑자기 향상되는 ‘급격한 전이(phase transition)’ 현상을 의미하며, 종종 “emergence”의 예로 언급됩니다.

이 논문에서는 **Gradient Descent(경사 하강법)** 없이 동작하는 \*\*RFM(Recursive Feature Machines)\*\*이라는 모델이 \*\*AGOP(Average Gradient Outer Product)\*\*를 사용해 **task-specific features**를 학습하고, 이 과정에서 grokking이 발생함을 보여줍니다. 즉, 모듈러 산술 문제를 푸는 데 있어서 **신경망도, 경사 기반 최적화도 필수적이지 않다**는 결론입니다.

또한 RFM과 신경망이 공통적으로 학습하는 feature가 **block-circulant 구조**를 가지며, 이는 이전에 제안된 **Fourier multiplication algorithm**과 동일한 알고리즘을 구현한다는 점도 밝혔습니다.

---



Neural networks trained to solve modular arithmetic tasks exhibit **grokking**, the phenomenon where the test accuracy improves only long after the model achieves 100% training accuracy in the training process. It is often taken as an example of “**emergence**”, where model ability manifests sharply through a phase transition.

In this work, we show that the phenomenon of grokking is not specific to neural networks nor to gradient descent-based optimization. Specifically, we show that **grokking occurs when learning modular arithmetic with Recursive Feature Machines (RFM)**, an iterative algorithm that uses the **Average Gradient Outer Product (AGOP)** to enable task-specific feature learning with kernel machines.

We show that RFM and, furthermore, neural networks that solve modular arithmetic **learn block-circulant feature transformations** which implement the previously proposed **Fourier multiplication algorithm**.

---






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




이 논문은 **모듈러 산술(modular arithmetic)** 문제를 학습할 때 나타나는 grokking 현상을 신경망이 아닌 비신경망 모델인 \*\*Recursive Feature Machine (RFM)\*\*을 통해 탐구합니다. RFM은 \*\*AGOP(Average Gradient Outer Product)\*\*라는 기법을 사용하여, 일반적으로 feature learning 능력이 없는 \*\*커널 머신(kernel machine)\*\*에서도 feature를 점진적으로 학습할 수 있도록 만듭니다.

모델 구조는 다음 세 가지 단계를 반복하는 방식입니다:

1. **커널 머신 학습**: 현재 feature 행렬을 기반으로 훈련 데이터를 학습
2. **AGOP 행렬 계산**: 학습된 모델의 기울기를 기반으로 평균 그래디언트 외적(AGOP)을 계산
3. **입력 변환**: AGOP의 제곱근(또는 다른 제곱 지수)을 통해 입력 데이터를 변환

이렇게 학습된 feature 행렬은 결국 **block-circulant 구조**를 가지게 되며, 이는 모듈러 산술 연산에 필요한 \*\*Fourier Multiplication Algorithm (FMA)\*\*과 동일한 구조임이 이론적으로도 증명됩니다.

트레이닝 데이터는 소수 $p$에 대해 정의된 정수 집합 $\mathbb{Z}_p$ 상의 연산 (예: $(a + b) \mod p$)으로 구성되며, 각 입력은 one-hot 인코딩된 벡터 $[e_a, e_b]$, 출력은 $e_{(a + b) \mod p}$ 형식입니다.

---


The authors introduce a non-neural model called the **Recursive Feature Machine (RFM)** to study the grokking phenomenon in **modular arithmetic** learning tasks. RFM enables **feature learning in kernel machines**, which traditionally do not learn features, through a method based on the **Average Gradient Outer Product (AGOP)**.

The algorithm proceeds in three main steps that are repeated iteratively:

1. **Train a kernel machine** on the current feature matrix.
2. **Compute the AGOP matrix** from the model gradients over the training data.
3. **Transform the input data** using the matrix square root of the AGOP to update features.

The features learned by RFM eventually develop a **block-circulant structure**, which the authors theoretically prove corresponds to the **Fourier Multiplication Algorithm (FMA)** for modular arithmetic.

The training data consists of operations defined over the finite field $\mathbb{Z}_p$, such as $(a + b) \mod p$. Inputs are one-hot encoded as vectors $[e_a \oplus e_b]$, and outputs are encoded as $e_{(a + b) \mod p}$.

---




   
 
<br/>
# Results  





이 논문은 grokking 현상이 신경망에서만 발생하는 것이 아니라 \*\*비신경망 모델(RFM)\*\*에서도 나타날 수 있음을 실험적으로 입증했습니다. 주요 실험은 소수 $p=61$을 기반으로 한 \*\*모듈러 연산(addition, subtraction, multiplication, division)\*\*에 대해 수행되었으며, 입력과 출력은 one-hot 벡터로 구성되어 있습니다.

**핵심 결과는 다음과 같습니다:**

1. **RFM은 정확히 0의 훈련 손실을 보이며도 테스트 정확도는 한참 후에야 급격히 향상됨(grokking)**
   → 초기 수 회(iterations) 동안은 테스트 손실 및 정확도 변화가 없지만, 이후 급격한 정확도 상승과 손실 감소가 나타남.

2. **기존의 일반적인 메트릭(정확도, 테스트 손실)로는 초기 학습 진행 상황을 설명할 수 없음**
   → 대신, 새로운 **진행도 측정 지표**인 **Circulant Deviation**과 **AGOP Alignment**가 feature 학습의 점진적 진전을 포착함.

3. **신경망(1-hidden layer MLP)과 RFM은 유사한 block-circulant feature를 학습함**
   → 두 모델 모두 Fourier 기반 알고리즘을 학습하고 있으며, AGOP와 신경망의 Feature Matrix 간 높은 상관관계를 보임 (피어슨 상관 0.92 이상).

4. **Random circulant feature로 변환된 입력을 사용하면 일반적인 커널 머신도 높은 성능을 달성함**
   → 이 구조가 일반화 성능의 핵심임을 실험적으로 입증.

5. **RFM과 신경망 모두 일반적인 커널/MLP보다 적은 데이터로도 일반화(generalization)에 성공함**
   → 예: 훈련 데이터가 17.5%만 있어도 random circulant 변환을 통해 수백 에폭 안에 100% 정확도 도달.

---



This study demonstrates that the **grokking phenomenon**—a sudden increase in test accuracy long after perfect training accuracy—is not exclusive to neural networks but also arises in **Recursive Feature Machines (RFMs)**, a non-neural model.

**Key experimental results include:**

1. **RFM shows grokking even with zero training loss from the start**
   → Both test loss and accuracy remain flat for several iterations, then suddenly improve, confirming the presence of grokking.

2. **Standard metrics like test loss and accuracy fail to track early learning progress**
   → New progress measures—**Circulant Deviation** and **AGOP Alignment**—reveal gradual structural improvements in learned features.

3. **Neural networks and RFMs learn similar block-circulant features**
   → Neural Feature Matrices and AGOPs are highly correlated (Pearson correlation > 0.92), indicating convergence to the same Fourier-based algorithm.

4. **Standard kernel machines generalize well when input data is transformed with random circulant features**
   → This shows that block-circulant structure is sufficient for generalization.

5. **Both RFM and neural networks generalize from limited data**
   → With just 17.5% of the data and circulant transformations, models reach 100% test accuracy within a few hundred training steps.

---



<br/>
# 예제  




이 논문에서는 **모듈러 산술(modular arithmetic)** 연산을 학습하는 태스크를 예시로 다룹니다. 이는 수학적으로 매우 단순하지만, 머신러닝 모델에게는 일반화가 어려운 **알고리즘적 학습 문제**입니다.

####  주요 태스크:

다음과 같은 연산을 예측하는 함수 $f^*(a, b) = (a \, \text{op} \, b) \mod p$
여기서 op는 덧셈(+), 뺄셈(−), 곱셈(×), 나눗셈(÷), $p$는 소수(예: 59 또는 61)

####  입력 (Input):

* 각 입력은 정수 $a, b \in \mathbb{Z}_p$의 쌍으로 구성됨
* 이 정수들은 각각 **one-hot 벡터**로 표현되고, 최종 입력은 두 벡터의 **concatenation**
  → $x_{ab} = e_a \oplus e_b \in \mathbb{R}^{2p}$

예시)
$a = 5, b = 12$, $p = 59$일 때
→ 입력은 $e_5 \oplus e_{12}$: 길이 $2 \times 59 = 118$의 벡터에서 5번째와 71번째가 1인 벡터

####  출력 (Output):

* 정답 레이블 $y = f^*(a, b)$ 역시 one-hot 벡터로 표현됨
  → $y_{ab} = e_{(a \,\text{op}\, b) \mod p} \in \mathbb{R}^{p}$

예시)
$f^*(5, 12) = (5 + 12) \mod 59 = 17$
→ 출력 벡터 $y = e_{17}$

####  데이터셋 구성:

* 총 입력 쌍의 수는 $N = p^2$ (곱셈/나눗셈 제외 시), 또는 $N = p(p-1)$ (나눗셈의 경우 0 제외)
* 학습 데이터는 무작위로 선택된 비율 $r \times N$ 개로 구성 (예: 17.5%, 50%)
* 테스트는 전체 입력 공간에서 진행

---


This paper focuses on **modular arithmetic** learning tasks, which involve predicting the result of arithmetic operations modulo a prime number. These tasks are mathematically simple but pose a generalization challenge for learning models.

####  Main Task:

Learn a function $f^*(a, b) = (a \,\text{op}\, b) \mod p$
where `op` is one of addition, subtraction, multiplication, or division, and $p$ is a prime (e.g., 59 or 61)

####  Input Format:

* Each input is a pair of integers $a, b \in \mathbb{Z}_p$
* Each integer is represented as a **one-hot vector**, and the full input is the **concatenation**:
  → $x_{ab} = e_a \oplus e_b \in \mathbb{R}^{2p}$

Example:
If $a = 5, b = 12$, and $p = 59$, then
→ Input is a 118-dimensional vector with 1s at positions 5 and 71

####  Output Format:

* The target label is also a one-hot vector encoding $f^*(a, b) = (a \,\text{op}\, b) \mod p$
  → $y_{ab} = e_{(a \,\text{op}\, b) \mod p} \in \mathbb{R}^{p}$

Example:
For $a = 5, b = 12$, the correct output is
→ $f^*(5,12) = 17$, so output is $e_{17}$

####  Dataset Details:

* The full input space contains $N = p^2$ samples (or $p(p-1)$ for division)
* The training set is a random subset of $r \times N$ samples (e.g., 17.5%, 50%)
* Testing is performed over the entire discrete input space

---





<br/>  
# 요약   




이 논문은 AGOP(평균 그래디언트 외적)를 활용한 RFM(Recursive Feature Machine)이라는 비신경망 모델을 통해 모듈러 산술 문제에서의 feature 학습과 grokking 현상을 분석한다. 실험 결과, RFM과 신경망 모두 명확한 성능 전이 없이 점진적으로 block-circulant feature를 학습하며, 이는 Fourier 알고리즘과 높은 유사성을 보인다. 입력은 $a, b \in \mathbb{Z}_p$를 one-hot 인코딩한 $2p$-차원 벡터이고, 출력은 $(a \,\text{op}\, b) \mod p$의 one-hot 벡터이다.

---



This paper investigates feature learning and grokking in modular arithmetic using a non-neural model called Recursive Feature Machine (RFM), which leverages the Average Gradient Outer Product (AGOP). Experimental results show that both RFM and neural networks gradually learn block-circulant feature structures that closely resemble the Fourier Multiplication Algorithm. Inputs are one-hot encoded vectors of integers $a, b \in \mathbb{Z}_p$, and outputs are one-hot vectors of $(a \,\text{op}\, b) \mod p$.




<br/>  
# 기타  




####  Figure 1 – RFM의 grokking 과정 시각화

모듈로 연산 $f^*(x, y) = (x + y) \mod 59$에 대해, RFM의 반복(iteration)에 따른 **정확도 및 손실 변화**를 보여줌. **초기에는 테스트 정확도와 손실 변화가 없지만**, 특정 반복 이후 급격한 성능 향상이 발생함. 이는 전형적인 grokking 현상을 잘 보여주는 예시로, feature matrix의 구조가 점차 **block-circulant 패턴**으로 발전함을 시각적으로 확인할 수 있음.

####  Figure 2 – 진행도 측정 지표들 (Circulant Deviation, AGOP Alignment)

전통적인 메트릭(정확도, 손실)은 변화가 없다가 급격히 좋아지지만, **Circulant Deviation**과 **AGOP Alignment**는 초기부터 **점진적으로 증가**함. 이는 모델이 **점진적으로 구조적 feature를 학습**한다는 것을 보여주는 핵심 인사이트.

####  Figure 3 & 6 – Feature Matrix의 구조

RFM과 신경망이 학습한 **AGOP 및 Neural Feature Matrix**를 시각화한 결과로, **블록-순환(block-circulant)** 구조가 명확히 드러남. 특히 곱셈과 나눗셈 연산의 경우에는 **입력 순서를 이산 로그(discrete log)로 재정렬**해야 이러한 구조가 나타나며, 이는 **Fourier 기반 구조**를 모델이 학습하고 있음을 보여줌.

####  Figure 4 & 7 – Random Circulant 변환 효과

랜덤 블록-순환 행렬로 입력을 변환한 경우, 일반적인 커널 모델이나 신경망도 **빠르게 일반화**하며 적은 데이터로도 높은 정확도 도달. 이는 구조 자체가 핵심임을 실험적으로 입증.

####  부록(Appendix) – 추가 실험, 시각화, 알고리즘

* Appendix C: **Neural Feature Ansatz (NFA)** 설명
* Appendix D: 실험 세팅 및 하이퍼파라미터
* Appendix E: **이산 로그 재정렬 알고리즘**
* Appendix F–I: 곱셈/나눗셈 결과 시각화, Fourier 알고리즘 증명
  → 본문에서 다루지 못한 구조적 통찰 및 재현성 보장을 위한 핵심 자료 포함

---


####  Figure 1 – Visualization of Grokking in RFM

Shows how accuracy and loss behave during RFM iterations for $f^*(x, y) = (x + y) \mod 59$. Despite constant loss/accuracy at first, there is a **sharp test accuracy transition**, visually aligned with the emergence of **striped, block-circulant patterns** in the feature matrix.

####  Figure 2 – Progress Metrics (Circulant Deviation, AGOP Alignment)

While traditional metrics remain flat early in training, **Circulant Deviation** and **AGOP Alignment** steadily improve, revealing **underlying structural progress** in feature learning even before generalization becomes evident.

####  Figure 3 & 6 – Feature Matrix Structures

Visualizations of AGOPs (RFM) and Neural Feature Matrices (NN) show **block-circulant structure**. For multiplication/division tasks, the structure becomes clear only after **reordering inputs using discrete logarithms**, linking learned features to **Fourier-based representations**.

####  Figure 4 & 7 – Effect of Random Circulant Features

Applying random block-circulant transformations to input vectors significantly **accelerates generalization** in both kernel machines and neural networks. This supports the claim that structure, rather than optimization, is key.

####  Appendix – Additional Experiments, Visuals, Algorithms

* Appendix C: Details of the **Neural Feature Ansatz (NFA)**
* Appendix D: Experimental setup and hyperparameters
* Appendix E: **Discrete log reordering algorithm**
* Appendix F–I: Visual evolution of feature matrices, theoretical derivations of Fourier algorithm
  → Provides key technical foundations and extended results to ensure **reproducibility and deeper insight**.




<br/>
# refer format:     



@inproceedings{mallinar2025emergence,
  title     = {Emergence in Non-Neural Models: Grokking Modular Arithmetic via Average Gradient Outer Product},
  author    = {Neil Mallinar and Daniel Beaglehole and Libin Zhu and Adityanarayanan Radhakrishnan and Parthe Pandit and Mikhail Belkin},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  publisher = {PMLR},
  volume    = {267},
  address   = {Vancouver, Canada},
  url       = {https://proceedings.mlr.press/v267/mallinar25a.html}
}



Mallinar, Neil, Daniel Beaglehole, Libin Zhu, Adityanarayanan Radhakrishnan, Parthe Pandit, and Mikhail Belkin. “Emergence in Non-Neural Models: Grokking Modular Arithmetic via Average Gradient Outer Product.” In Proceedings of the 42nd International Conference on Machine Learning, vol. 267. Vancouver, Canada: PMLR, 2025.





