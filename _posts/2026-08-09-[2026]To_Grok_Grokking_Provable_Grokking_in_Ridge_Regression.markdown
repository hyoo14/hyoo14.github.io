---
layout: post
title:  "[2026]To Grok Grokking: Provable Grokking in Ridge Regression"
date:   2026-08-09 17:36:38 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 트레이닝 일정 구간동안 성능 상승않는 그로킹이 릿지 리그레션에서도 발생함을 보임   
또한 가중치 감쇠, 학습률, 데이터 수, 모델 차원, 초기화 크기와 같은 학습 하이퍼파라미터에 의해 조절될 수 있음을 보임   
이 논문은 그로킹이 발생하기까지 걸리는 시간, 즉 **grokking time**에 대해 하이퍼파라미터와 관련된 정량적인 하한을 제시한 최초의 엄밀한 결과라고 주장   

과적합과 일반화가 서로 다른 파라미터 방향에서 일어난다는 점  




짧은 요약(Abstract) :
## 초록(Abstract) 설명

이 논문은 **리지 회귀(ridge regression)**라는 단순한 선형 모델에서도 *그로킹(grokking)*이 발생할 수 있음을 이론적으로 증명한 연구입니다. 그로킹은 모델이 훈련 데이터를 이미 완벽하게 맞춘 뒤에도 한동안 테스트 성능이 좋지 않다가, 훨씬 나중에 일반화 성능이 갑자기 좋아지는 현상을 말합니다.

저자들은 **가중치 감쇠(weight decay)를 포함한 경사하강법**으로 과매개변수화된 선형 회귀 모델을 학습할 때 다음 세 단계가 나타난다는 것을 보였습니다.

1. **초기 과적합:** 모델이 훈련 데이터의 오차를 빠르게 거의 0으로 만든다.  
2. **일반화 지연:** 훈련 오차가 낮아진 뒤에도 테스트 오차는 오랫동안 높게 유지된다.  
3. **뒤늦은 일반화:** 충분히 학습하면 테스트 오차도 임의로 작아진다.

또한 그로킹이 단순히 우연히 나타나는 현상이 아니라, **가중치 감쇠, 학습률, 데이터 수, 모델 차원, 초기화 크기**와 같은 학습 하이퍼파라미터에 의해 조절될 수 있음을 보였습니다. 특히 가중치 감쇠를 작게 하면 일반화가 시작되는 시점이 늦어져 그로킹 시간이 길어질 수 있고, 적절히 조정하면 그로킹을 약화하거나 없앨 수도 있습니다.

이 논문은 그로킹이 발생하기까지 걸리는 시간, 즉 **grokking time**에 대해 하이퍼파라미터와 관련된 정량적인 하한을 제시한 최초의 엄밀한 결과라고 주장합니다. 마지막으로 비선형 신경망 실험에서도 선형 리지 회귀에서 얻은 이론적 예측과 비슷한 경향이 나타남을 확인했습니다. 따라서 그로킹은 심층 신경망의 구조 자체가 가진 필연적인 문제라기보다, **특정한 정규화와 학습 조건에서 발생하는 현상**일 수 있다고 결론내립니다.

---




This paper theoretically proves that **grokking can occur even in ridge regression**, a simple linear model. Grokking refers to the phenomenon in which a model first fits the training data perfectly, while its test performance remains poor for a long time, and only later begins to generalize well.

Using gradient descent with weight decay, the authors prove three stages:

1. **Early overfitting:** the training error quickly becomes very small.  
2. **Delayed generalization:** the test error remains large long after overfitting.  
3. **Eventual generalization:** after sufficiently long training, the test error becomes arbitrarily small.

The paper also derives quantitative bounds on the **grokking time**, or the delay between overfitting and generalization. In particular, decreasing the weight-decay parameter can greatly increase this delay, while suitable hyperparameter tuning can reduce or eliminate grokking.

Finally, experiments on nonlinear neural networks show qualitatively similar dependencies on the hyperparameters. These results suggest that grokking is not necessarily an inherent problem of deep learning architectures, but may instead result from particular training conditions and regularization choices.


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



### 1. 연구의 핵심 목적
이 논문은 **릿지 회귀(ridge regression)**라는 매우 단순한 선형 모델에서도 그로킹(grokking)이 발생할 수 있음을 이론적으로 증명한다.  
그로킹은 다음과 같은 학습 과정을 의미한다.

1. 초기에 훈련 데이터는 거의 완벽하게 맞춘다.
2. 그러나 테스트 성능은 오랫동안 좋지 않다.
3. 충분히 시간이 지나면 테스트 성능이 급격히 또는 점진적으로 좋아진다.

즉, **과적합 이후 한참 뒤에 일반화가 나타나는 현상**을 분석한다.

---

### 2. 기본 모델: 과매개변수화된 선형 회귀

학생 모델(student model)은 다음과 같은 선형 모델이다.

\[
N(x;\theta)=\langle \theta,\phi(x)\rangle
\]

- \(x\): 입력 데이터
- \(\phi(x)\in\mathbb{R}^m\): 고정된 특징(feature) 변환
- \(\theta\in\mathbb{R}^m\): 학습되는 가중치
- \(m\): 특징 차원 또는 모델 파라미터 수

교사 함수(ground-truth teacher)는

\[
N^*(x)=\langle \theta^*,\phi(x)\rangle
\]

로 설정한다. 즉, 교사 함수 역시 동일한 특징 공간에서 표현 가능한 **realizable setting**을 가정한다.

특징 차원 \(m\)이 훈련 샘플 수 \(n\)보다 훨씬 큰

\[
m\gg n
\]

상황을 주로 다룬다. 따라서 모델은 훈련 데이터를 완벽하게 맞출 수 있지만, 훈련 데이터가 관측하지 못한 방향에 대해서는 잘못된 예측을 할 수 있다.

---

### 3. 트레이닝 데이터 생성 방식

훈련 데이터는 다음 절차로 생성된다.

1. 입력 \(x_i\)를 어떤 분포 \(D_x\)에서 샘플링한다.
2. 교사 함수로 정답을 생성한다.

\[
y_i=N^*(x_i)
\]

따라서 이론적 핵심 설정에서는 라벨 노이즈가 없는 **정확한 teacher–student 데이터**를 사용한다.

훈련 특징을 행렬로 모으면

\[
\Phi=
\begin{pmatrix}
\phi(x_1)^\top\\
\vdots\\
\phi(x_n)^\top
\end{pmatrix}
\in\mathbb{R}^{n\times m}
\]

이 된다.

논문은 주로 다음 두 가지 성능을 구분한다.

- **훈련 손실**
  \[
  L_n(\theta)=\frac{1}{2n}\sum_{i=1}^n
  (N(x_i;\theta)-N^*(x_i))^2
  \]

- **일반화 손실**
  \[
  L(\theta)=\mathbb{E}_{x}
  [(N(x;\theta)-N^*(x))^2]
  \]

---

### 4. 특별한 학습 기법: 릿지 정규화와 weight decay

학습 목적 함수는 평균제곱오차에 \(\ell_2\) 정규화를 더한 릿지 회귀 목적 함수이다.

\[
L_n(\theta;\lambda)
=
\frac{1}{2n}\sum_{i=1}^n
(N(x_i;\theta)-N^*(x_i))^2
+
\frac{\lambda}{2}\|\theta\|_2^2
\]

여기서 \(\lambda>0\)는 **weight decay의 세기**이다.

- 작은 \(\lambda\): 정규화가 약함
- 큰 \(\lambda\): 가중치 크기를 강하게 줄임

이 논문에서 weight decay는 단순한 과적합 방지 기법이 아니라, 모델이 장기적으로 **작은 노름의 일반화 가능한 해**로 이동하도록 만드는 핵심 요소다.

---

### 5. 최적화 알고리즘

고정 학습률 \(\eta\)를 사용하는 vanilla gradient descent를 적용한다.

\[
\theta^{(t+1)}
=
\theta^{(t)}
-\eta\nabla_\theta L_n(\theta^{(t)};\lambda)
\]

구체적으로는

\[
\theta^{(t+1)}
=
\theta^{(t)}
-\frac{\eta}{n}\Phi^\top
(\Phi\theta^{(t)}-\mathbf y)
-\eta\lambda\theta^{(t)}
\]

이다.

초기 가중치는 무작위로 설정한다.

\[
\theta^{(0)}\sim\mathcal N(0,\nu^2 I_m)
\]

여기서 \(\nu^2\)는 초기화 규모(initialization scale)를 조절한다.

---

### 6. 그로킹이 발생하는 핵심 메커니즘

훈련 데이터가 포함하는 부분공간과 그에 수직인 부분공간을 분리하면 현상을 쉽게 이해할 수 있다.

#### 데이터가 관측한 방향
\(\Phi\)의 row space에 해당하는 가중치 성분은 훈련 데이터의 오차를 줄이는 방향으로 빠르게 업데이트된다.

따라서 훈련 손실은 비교적 빠르게 감소한다.

#### 데이터가 관측하지 못한 방향
\(\Phi\)의 null space에 해당하는 가중치 성분은 훈련 데이터의 손실에 거의 영향을 주지 않는다. 이 성분은 주로 weight decay에 의해 다음처럼 천천히 감소한다.

\[
\theta_\perp^{(t)}
=
(1-\eta\lambda)^t\theta_\perp^{(0)}
\]

과매개변수화된 경우 \(m\gg n\)이므로 이러한 미관측 방향이 많이 남는다. 초기화된 가중치가 이 방향에 상당히 남아 있기 때문에:

- 훈련 손실은 빠르게 작아진다.
- 테스트 오차는 오랫동안 높게 유지된다.
- weight decay가 충분히 누적된 후에야 미관측 방향의 가중치가 사라진다.
- 그 결과 일반화 성능이 좋아진다.

즉, 이 논문에서 그로킹은 **훈련 데이터가 관측한 성분의 빠른 수렴과, 관측되지 않은 성분의 느린 weight-decay 수렴 사이의 시간 차이**에서 발생한다.

---

### 7. 이론적으로 정의한 그로킹 시간

논문은 다음 두 시점을 정의한다.

- \(t_1\): 훈련 손실이 기준값 \(\epsilon\)보다 작아지는 시점
- \(t_2\): 일반화 손실이 기준값 \(c\)보다 작아지는 시점

\[
t_1=\max\{t:L_n(\theta^{(t)})\ge \epsilon\}
\]

\[
t_2=\min\{t:L(\theta^{(t)})\le c\}
\]

따라서 그로킹 시간 또는 일반화 지연은

\[
t_2-t_1
\]

이다.

논문의 주요 결론은 적절한 조건에서 이 차이를 **임의로 크게 만들 수 있다**는 것이다.

특히 작은 \(\lambda\)에서는

\[
t_2\propto \frac{1}{\lambda}
\]

이므로 weight decay를 약하게 할수록 일반화가 늦어진다. 반면 훈련 손실을 줄이는 시간 \(t_1\)은 주로 데이터 행렬의 고유값과 학습률에 의해 결정되며, 충분히 작은 \(\lambda\)의 영향은 상대적으로 작다.

---

### 8. 주요 하이퍼파라미터의 영향

- **Weight decay \(\lambda\) 감소**
  - 일반화가 늦어진다.
  - \(t_2\)가 커진다.
  - 그로킹 현상이 강해진다.
  - 너무 작으면 일반화 자체가 지나치게 느려질 수 있다.

- **훈련 샘플 수 \(n\) 감소**
  - 훈련 데이터를 더 빠르게 맞출 수 있는 조건이 형성될 수 있다.
  - \(t_1\)이 감소하여 과적합과 일반화 사이의 간격이 커질 수 있다.

- **특징 차원 \(m\) 증가**
  - 미관측 방향이 많아진다.
  - 초기화된 잔여 성분이 커질 수 있다.
  - 과매개변수화에 의한 그로킹을 가능하게 한다.
  - 다만 논문 실험에서는 \(t_1,t_2\)에 대한 영향이 크지 않은 경우도 관찰된다.

- **초기화 규모 \(\nu^2\) 증가**
  - 초기 가중치 노름이 커진다.
  - 훈련과 일반화 모두 늦어질 수 있다.
  - 이론적으로 \(t_1,t_2\)는 대체로 \(\log(\nu^2)\)에 비례한다.

- **학습률 \(\eta\)**
  - 안정적인 수렴을 위해 충분히 작아야 한다.
  - 이론에서는 대략
    \[
    \eta<\frac{1}{\lambda+b^2}
    \]
    또는 데이터 행렬의 최대 고유값을 사용하는 조건을 둔다.

---

### 9. 실험 모델 및 확장

#### 9.1 선형 릿지 회귀 실험
- 특징 함수는 항등 함수로 설정:
  \[
  \phi(x)=x
  \]
- 특징 벡터는 주로 Gaussian 분포에서 샘플링
- 기본 설정 예:
  - \(n=100\)
  - \(m=1000\)
  - \(\eta=1\)
  - \(\nu^2=1\)
  - \(\lambda=10^{-4}\)

훈련 손실은 먼저 감소하지만 테스트 손실은 오랫동안 높게 유지되다가 나중에 감소하는 패턴을 확인했다.

#### 9.2 Random-feature ReLU 네트워크
모델은

\[
N(x)=\sum_{j=1}^m a_j\sigma(w_j^\top x)
\]

형태의 2층 ReLU 네트워크이다.

- hidden weights \(w_j\): 무작위 초기화 후 고정
- output weights \(a_j\): 학습
- ReLU 특징을 고정된 feature map으로 사용

이 경우 모델은 output layer에 대해서는 선형 회귀와 동일하게 다룰 수 있다.

#### 9.3 일반적인 2층 ReLU 네트워크
추가 실험에서는 hidden layer와 output layer를 모두 학습했다.

- 교사 함수: 주로 zero function
- weight decay를 포함한 gradient descent 사용
- 이론을 직접 증명한 것은 아니지만, \(\lambda,n,m,\nu^2\) 변화에 따른 그로킹 경향이 선형 모델의 예측과 질적으로 일치했다.

#### 9.4 추가 실험
논문은 다음 환경에서도 그로킹을 관찰했다.

- 라벨 노이즈가 있는 릿지 회귀
- Random Fourier feature 모델
- 정확도 대신 임계값 기반 surrogate accuracy를 사용하는 평가

---

### 10. 한계와 의의

이 논문의 이론은 다음 설정에 집중한다.

- realizable teacher
- 고정 feature map
- 선형 모델
- 제곱 손실
- \(\ell_2\) weight decay
- gradient descent

따라서 일반적인 비선형 신경망의 lazy-to-rich 전환을 직접 증명한 것은 아니다. 하지만 복잡한 네트워크 구조가 없어도 그로킹이 발생할 수 있음을 보이고, **weight decay와 과매개변수화가 일반화 지연을 어떻게 제어하는지 정량적으로 설명했다**는 점이 핵심 의의다.

---



### 1. Main goal
The paper studies grokking in a classical ridge-regression setting. Grokking is characterized by:

1. The training loss becoming very small early.
2. Poor test performance persisting for a long time.
3. Generalization improving only much later.

The main contribution is an end-to-end theoretical proof of this behavior.

---

### 2. Model: over-parameterized linear regression

The student model is

\[
N(x;\theta)=\langle \theta,\phi(x)\rangle,
\]

where:

- \(x\) is the input,
- \(\phi(x)\in\mathbb{R}^m\) is a fixed feature map,
- \(\theta\in\mathbb{R}^m\) is the trainable parameter vector.

The teacher is realizable in the same feature space:

\[
N^*(x)=\langle \theta^*,\phi(x)\rangle.
\]

The main regime is highly over-parameterized:

\[
m\gg n,
\]

where \(n\) is the number of training samples. This creates many parameter directions that are not constrained by the training data.

---

### 3. Training data

Inputs \(x_i\) are sampled from an arbitrary input distribution \(D_x\), and labels are generated exactly by the teacher:

\[
y_i=N^*(x_i).
\]

Thus, the main theoretical setting is noiseless and realizable.

The paper distinguishes between:

- **Empirical training loss**
  \[
  L_n(\theta)=\frac{1}{2n}\sum_{i=1}^n
  (N(x_i;\theta)-N^*(x_i))^2,
  \]

- **Population generalization loss**
  \[
  L(\theta)=\mathbb{E}_x[
  (N(x;\theta)-N^*(x))^2].
  \]

---

### 4. Training objective and regularization

The model is trained with ridge regression:

\[
L_n(\theta;\lambda)
=
\frac{1}{2n}\sum_{i=1}^n
(N(x_i;\theta)-N^*(x_i))^2
+
\frac{\lambda}{2}\|\theta\|_2^2.
\]

The second term is \(\ell_2\) regularization, implemented as weight decay.

- Small \(\lambda\): weak regularization and slower long-term shrinkage.
- Large \(\lambda\): stronger parameter shrinkage.

Weight decay is the key mechanism that eventually removes parameter components that are not supported by the training data.

---

### 5. Optimization

The paper uses vanilla gradient descent with a fixed learning rate \(\eta\):

\[
\theta^{(t+1)}
=
\theta^{(t)}
-\eta\nabla_\theta L_n(\theta^{(t)};\lambda).
\]

The initialization is Gaussian:

\[
\theta^{(0)}\sim\mathcal N(0,\nu^2 I_m),
\]

where \(\nu^2\) controls the initialization scale.

---

### 6. Why grokking occurs

The parameter vector can be decomposed into:

- a component in the row space of the training feature matrix \(\Phi\),
- a component in its null space.

The row-space component is directly affected by the training data and therefore converges quickly, reducing the training error.

The null-space component is not constrained by the training examples. It changes mainly through weight decay:

\[
\theta_\perp^{(t)}
=
(1-\eta\lambda)^t\theta_\perp^{(0)}.
\]

When \(m\gg n\), the null space is large. Consequently:

- the training loss decreases quickly,
- the unseen-direction parameters remain large,
- the generalization error stays high,
- only after prolonged weight decay does the model move toward a small-norm, well-generalizing solution.

Therefore, the paper interprets grokking as a mismatch between:

- fast fitting of data-observed directions, and
- slow decay of data-unobserved directions.

---

### 7. Definition of grokking time

The paper defines:

- \(t_1\): the time by which the training loss falls below a threshold \(\epsilon\),
- \(t_2\): the time by which the generalization loss falls below a threshold \(c\).

The grokking delay is

\[
t_2-t_1.
\]

The main theorem shows that, under suitable conditions, this delay can be made arbitrarily large. In particular, for sufficiently small weight decay,

\[
t_2\propto \frac{1}{\lambda}.
\]

Thus, decreasing \(\lambda\) can strongly amplify grokking.

---

### 8. Hyperparameter effects

- **Weight decay \(\lambda\)**
  - Smaller \(\lambda\) delays generalization.
  - The grokking time increases approximately as \(1/\lambda\).

- **Number of samples \(n\)**
  - Fewer samples can make training fitting faster.
  - This reduces \(t_1\) and can enlarge the gap between overfitting and generalization.

- **Feature dimension \(m\)**
  - Larger \(m\) creates more unconstrained directions.
  - This supports persistent poor generalization in the over-parameterized regime.

- **Initialization scale \(\nu^2\)**
  - Larger initialization increases the magnitude of the remaining parameter components.
  - Both \(t_1\) and \(t_2\) grow roughly logarithmically with \(\nu^2\).

- **Learning rate \(\eta\)**
  - It must be small enough to guarantee stable convergence.

---

### 9. Experimental models

#### Linear ridge regression
The main experiments use the identity feature map:

\[
\phi(x)=x.
\]

Features are sampled from Gaussian distributions, with settings such as \(n=100\), \(m=1000\), and small \(\lambda\). The experiments reproduce the theoretically predicted delayed-generalization pattern.

#### Random-feature ReLU network
The model is

\[
N(x)=\sum_{j=1}^m a_j\sigma(w_j^\top x).
\]

The hidden weights \(w_j\) are randomly initialized and fixed, while the output weights \(a_j\) are trained. Since the model is linear in \(a\), it is a special case of linear regression with a nonlinear feature map.

#### Fully trained two-layer ReLU network
The paper also trains both hidden and output layers. Although the theoretical proof does not directly cover this nonlinear setting, the experiments show qualitatively similar dependencies on weight decay, sample size, width, and initialization.

Additional experiments demonstrate grokking with label noise and random Fourier features.

---

### 10. Significance and limitations

The theory is developed for a realizable linear model with a fixed feature map, squared loss, \(\ell_2\) regularization, and gradient descent. It does not directly prove grokking caused by a lazy-to-rich transition in general deep networks.

Its main significance is that grokking does not require a complex architecture. Even a simple over-parameterized ridge-regression model can exhibit provable grokking, and the delay can be quantitatively controlled through weight decay and other training hyperparameters.


<br/>
# Results



### 1. 연구에서 비교한 모델과 설정

이 논문은 전통적인 모델과 신경망에서 **grokking이 어떻게 나타나는지**를 비교한다. 다만 별도의 경쟁 알고리즘이나 다른 최적화 방법을 주된 baseline으로 두기보다는, 동일한 Gradient Descent(GD) 학습 과정에서 **하이퍼파라미터를 바꾸며 학습·일반화 성능을 비교**한다.

| 실험 | 모델 | 교사 함수 및 데이터 |
|---|---|---|
| Ridge regression | 과매개변수화 선형 회귀 모델 | 실현 가능한 선형 교사 함수. 주로 \(\|\theta^*\|_2=1\), \(\phi(x)=x\) 사용 |
| Random-feature ReLU network | 고정된 은닉층과 학습 가능한 출력층을 가진 2층 ReLU 네트워크 | 단일 ReLU 뉴런 교사 \(x\mapsto \sigma(\langle w^*,x\rangle)\) |
| Nonlinear neural network | 두 층 모두 학습하는 2층 ReLU 네트워크 | 주로 zero teacher, 즉 항상 0인 함수 |
| 추가 실험 | Random Fourier feature 모델, label-noise ridge regression | 실현 가능한 교사 또는 잡음이 섞인 라벨 |

Ridge regression과 random-feature 모델에서는 사실상 선형 회귀 문제로 분석할 수 있으며, 두 층 ReLU 네트워크 실험은 이론이 비선형 모델에서도 어느 정도 성립하는지 확인하기 위한 실험이다.

---

### 2. 사용한 데이터와 테스트 방식

- 학습 데이터는 교사 함수에서 생성된 **합성 데이터(synthetic data)**이다.
- 실현 가능한 설정에서는 학습 라벨이 교사 함수의 출력과 정확히 일치한다.
- 테스트 데이터는 학습 데이터와 독립적으로 추출한 새로운 입력에 대해 교사 함수의 출력을 계산하여 구성한다.
- Ridge regression의 주요 실험에서는 다음과 같은 설정을 사용한다.
  - feature map: \(\phi(x)=x\)
  - 입력 특징: 대체로 \(N(0,I_m/m)\) 형태의 Gaussian feature
  - 기본값: \(n=100,\;m=1000,\;\eta=1,\;\nu^2=1,\;\lambda=10^{-4}\)
- 여기서 \(n\)은 학습 샘플 수, \(m\)은 feature dimension 또는 모델 폭, \(\lambda\)는 weight decay, \(\nu^2\)는 초기화 규모이다.

---

### 3. 평가 메트릭

#### 주요 메트릭: 학습 및 테스트 제곱 손실

\[
L_n(\theta)
= \frac{1}{n}\sum_{i=1}^n
\bigl(N(x_i;\theta)-N^*(x_i)\bigr)^2
\]

- **Training loss**: 학습 데이터에서의 평균 제곱 오차
- **Test/generalization loss**:

\[
L(\theta)
= \mathbb{E}_x
\left[
\bigl(N(x;\theta)-N^*(x)\bigr)^2
\right]
\]

논문에서 grokking은 다음과 같은 순서로 정의된다.

1. 학습 손실이 빠르게 작아져 학습 데이터를 거의 완벽히 맞춘다.
2. 그 이후에도 테스트 손실은 오랫동안 높게 유지된다.
3. 충분히 긴 학습 뒤에 테스트 손실이 급격히 또는 점진적으로 감소한다.

이를 위해 다음 시간을 사용한다.

- \(t_1\): training loss가 임계값 \(\epsilon\) 이하로 내려가는 시점
- \(t_2\): test loss가 임계값 \(c\) 이하로 내려가는 시점
- **Grokking time**: \(t_2-t_1\)

---

### 4. Ridge regression에서의 핵심 결과

#### 학습 손실은 빠르게 감소

정리 4.4와 정리 4.2에 따르면, 학습 손실은 대략 다음 속도로 감소한다.

\[
t_1
\lesssim
\frac{n}{\eta\lambda_{\min}^{+}(\Phi^\top\Phi)}
\log\frac{1}{\epsilon}
\]

즉, 데이터가 차지하는 부분공간(row space)의 방향은 GD가 비교적 빠르게 학습한다. 과매개변수화된 모델에서는 학습 데이터에 맞추는 데 필요한 방향만 빠르게 업데이트된다.

#### 테스트 성능은 느리게 개선

학습 데이터가 관측하지 못한 부분, 즉 \(\Phi\)의 null space에 있는 파라미터 성분은 데이터 손실의 gradient로는 직접 수정되지 않는다. 이 성분은 weight decay에 의해서만 감소하며, 그 속도는 대략

\[
(1-\eta\lambda)^t
\]

이다. 따라서 \(\lambda\)가 작으면 테스트 성능 개선이 매우 느려진다.

논문은 다음과 같은 테스트 손실 하한을 제시한다.

\[
L(\theta(t))
\gtrsim
\lambda_{\min}(\Sigma)
(1-\eta\lambda)^{2t}
(m-n)\nu^2
\]

이는 모델이 학습 데이터를 이미 잘 맞춘 뒤에도, 테스트 오차가 일정 시간 동안 계속 크게 남을 수 있음을 보인다.

#### 결국 일반화가 좋아짐

GD는 장기적으로 ridge regression의 전역 최적점으로 수렴한다. bounded feature와 충분한 학습 샘플이라는 조건에서 최종 해는 다음과 같은 일반화 보장을 가진다.

\[
L(\theta_\lambda^*)
\leq
2L_n(\theta_\lambda^*)+\epsilon
\]

따라서 이론적으로 다음 세 단계가 모두 증명된다.

- 초기 학습 데이터 과적합
- 과적합 이후의 장기적인 나쁜 일반화
- 충분히 긴 시간이 지난 뒤의 좋은 일반화

이것이 논문의 **end-to-end provable grokking** 결과이다.

---

### 5. 하이퍼파라미터에 따른 비교 결과

#### Weight decay \(\lambda\)

가장 중요한 조절 변수이다.

- \(\lambda\)를 작게 하면 training loss가 내려가는 시점 \(t_1\)에는 큰 영향을 주지 않는다.
- 반면 테스트 성능을 회복하는 시간 \(t_2\)는 대략 \(1/\lambda\)에 비례해 증가한다.
- 따라서 작은 weight decay는 grokking delay를 크게 늘린다.
- 충분히 큰 weight decay를 사용하면 grokking이 약해지거나 사실상 사라질 수 있다.

#### 학습 샘플 수 \(n\)

- 샘플 수가 작을수록 모델이 관측하지 못하는 방향이 많아지고, 학습 손실은 빠르게 감소한다.
- 결과적으로 \(t_1\)이 작아져 과적합이 더 빨리 발생한다.
- 따라서 일반화가 늦게 시작되는 grokking 현상이 더 두드러진다.

#### Feature dimension 또는 모델 폭 \(m\)

- \(m\gg n\)인 과매개변수화 상황에서 grokking이 뚜렷하게 나타난다.
- 하지만 실험에서는 \(m\)을 증가시켜도 \(t_1\)과 \(t_2\)가 크게 늘어나지는 않았다.
- 핵심은 단순히 모델을 크게 만드는 것보다는, 학습 데이터가 설명하지 못하는 파라미터 방향이 충분히 존재하는지 여부이다.

#### Initialization scale \(\nu^2\)

- 초기화 규모가 커지면 \(t_1\)과 \(t_2\)가 모두 대략 \(\log(\nu^2)\)에 비례해 증가한다.
- 이론적으로는 \(t_2\)가 \(t_1\)보다 더 빠르게 증가하므로 grokking gap이 커질 수 있다.
- 다만 비선형 네트워크에서는 초기화 규모를 늘렸을 때 delay 자체보다 training/test loss 간 격차가 커지는 효과가 더 뚜렷하게 나타났다.

---

### 6. Random-feature ReLU 네트워크 결과

이 실험에서는 은닉층의 가중치는 고정하고 출력층만 학습한다. 따라서 이 모델은 고정 feature map을 사용하는 선형 회귀로 볼 수 있다.

- 작은 weight decay에서 grokking time이 길어졌다.
- 학습 샘플 수를 줄이면 training loss가 더 빨리 감소하여 grokking이 강해졌다.
- 모델 폭을 증가시켜도 grokking delay가 크게 길어지지는 않았다.
- Ridge regression에서 관찰된 하이퍼파라미터 의존성이 random-feature ReLU 모델에서도 대체로 반복되었다.

단, 교사 ReLU 네트워크와 고정 random feature가 정확히 일치하지 않을 수 있기 때문에 이 실험은 엄밀한 realizable 설정이라기보다는 충분히 넓은 모델에서 실현 가능한 상황에 근접한 설정이다.

---

### 7. 두 층 모두 학습하는 비선형 신경망 결과

두 층 ReLU 네트워크에서는 은닉층과 출력층을 모두 GD로 학습했다.

- 교사는 zero function으로 설정했다.
- training loss와 test loss의 차이를 비교했다.
- ridge regression과 마찬가지로:
  - 작은 weight decay는 generalization을 지연시켰다.
  - 작은 학습 샘플 수는 training convergence를 빠르게 하여 grokking을 강화했다.
  - feature dimension 또는 width 변화의 영향은 상대적으로 작았다.
- 따라서 선형 ridge regression에서 얻은 하이퍼파라미터 의존성이 비선형 네트워크에서도 **정성적으로** 유지되었다.

다만 이 비선형 모델에 대해서는 논문이 ridge regression과 같은 수준의 엄밀한 end-to-end 증명을 제공하지 않고, 실험적 관찰만 제시한다.

---

### 8. Accuracy형 지표와 추가 결과

회귀 문제에서는 테스트 제곱 손실이 보통 매끄럽게 감소하기 때문에, 분류 문제에서 흔히 보이는 뚜렷한 “plateau”가 약하게 보일 수 있다.

이를 보완하기 위해 다음과 같은 threshold-based accuracy를 추가로 측정했다.

\[
\Pr_x\left[
\bigl(N(x;\theta)-N^*(x)\bigr)^2\leq\epsilon
\right]
\]

이 지표를 사용하면:

- 초기에는 정확도가 낮게 유지되고,
- 학습 손실이 이미 작아진 뒤에도 일정 기간 plateau가 지속되며,
- 이후 테스트 정확도가 상승하는

전형적인 grokking 형태가 나타났다.

또한 label noise가 있는 ridge regression과 random Fourier feature 모델에서도 적절한 하이퍼파라미터를 선택하면 grokking이 관찰되었다. 다만 이러한 비실현 가능 설정은 논문의 주된 이론적 보장 범위 밖이다.

---

### 9. 전체 결론

논문의 실험 결과는 다음을 보여준다.

1. Grokking은 신경망에만 특유한 현상이 아니다.
2. 과매개변수화된 ridge regression에서도 학습 손실과 테스트 손실의 수렴 속도 차이만으로 grokking이 발생할 수 있다.
3. 핵심 원인은 학습 데이터가 보지 못한 파라미터 성분이 weight decay에 의해 매우 천천히 제거되는 것이다.
4. Weight decay, sample size, initialization scale 등을 조절하면 grokking을 증폭하거나 약화할 수 있다.
5. Ridge regression에서 얻은 예측은 random-feature 모델과 비선형 ReLU 네트워크에서도 대체로 실험적으로 확인되었다.

---



### 1. Models and experimental comparisons

The paper studies grokking in both linear models and neural networks. It does not primarily compare against a separate competing optimizer or algorithm. Instead, it compares training and test behavior under different hyperparameter settings.

| Experiment | Model | Teacher/data |
|---|---|---|
| Ridge regression | Over-parameterized linear regression | Realizable linear teacher, mainly with \(\|\theta^*\|_2=1\) and \(\phi(x)=x\) |
| Random-feature ReLU network | Fixed hidden layer, trainable output layer | A single ReLU teacher \(x\mapsto \sigma(\langle w^*,x\rangle)\) |
| Nonlinear neural network | Both layers of a two-layer ReLU network are trained | Mainly the zero teacher |
| Additional experiments | Random Fourier features and noisy-label ridge regression | Realizable or noisy settings |

The ridge and random-feature experiments are effectively linear regression problems with a fixed feature map. The fully trained ReLU network is used to test whether the same qualitative behavior extends beyond the linear setting.

---

### 2. Data and test evaluation

- The experiments use synthetic teacher-student data.
- In the realizable setting, training labels are generated exactly by the teacher function.
- Test data are independently sampled inputs, with targets computed from the same teacher.
- In the main ridge regression experiments:
  - \(\phi(x)=x\)
  - features are typically Gaussian, approximately \(N(0,I_m/m)\)
  - default parameters are \(n=100\), \(m=1000\), \(\eta=1\), \(\nu^2=1\), and \(\lambda=10^{-4}\)

Here, \(n\) is the number of training samples, \(m\) is the feature dimension or width, \(\lambda\) is the weight-decay coefficient, and \(\nu^2\) is the initialization scale.

---

### 3. Evaluation metrics

The main metrics are training and test squared losses:

\[
L_n(\theta)
=
\frac{1}{n}\sum_{i=1}^n
\bigl(N(x_i;\theta)-N^*(x_i)\bigr)^2
\]

and

\[
L(\theta)
=
\mathbb{E}_x
\left[
\bigl(N(x;\theta)-N^*(x)\bigr)^2
\right].
\]

Grokking is characterized by three stages:

1. The training loss becomes small quickly.
2. The test loss remains high for a long period after training overfit has occurred.
3. The test loss eventually becomes small.

The paper defines:

- \(t_1\): the time when the training loss falls below \(\epsilon\)
- \(t_2\): the time when the test loss falls below \(c\)
- grokking time: \(t_2-t_1\)

---

### 4. Main ridge regression results

#### Fast training convergence

The training loss decreases at a rate controlled by the smallest positive eigenvalue of the empirical feature covariance:

\[
t_1
\lesssim
\frac{n}{\eta\lambda_{\min}^{+}(\Phi^\top\Phi)}
\log\frac{1}{\epsilon}.
\]

The components of the parameter vector lying in the data-spanned subspace are therefore learned relatively quickly.

#### Slow generalization

The components in the null space of the training feature matrix are not directly updated by the data gradient. They are reduced only through weight decay, at a rate approximately given by

\[
(1-\eta\lambda)^t.
\]

When \(\lambda\) is small, this process is very slow. The paper proves a lower bound of the form

\[
L(\theta(t))
\gtrsim
\lambda_{\min}(\Sigma)
(1-\eta\lambda)^{2t}
(m-n)\nu^2.
\]

Thus, the model can fit the training data while maintaining a large test error for a long time.

#### Eventual generalization

Gradient descent eventually converges to the global minimizer of the ridge objective. Under bounded features and a sufficiently large sample size, the limiting solution satisfies a generalization guarantee such as

\[
L(\theta_\lambda^*)
\leq
2L_n(\theta_\lambda^*)+\epsilon.
\]

Together, these results establish the full sequence of early overfitting, delayed generalization, and eventual good generalization.

---

### 5. Effects of hyperparameters

#### Weight decay \(\lambda\)

This is the most important control parameter.

- Decreasing \(\lambda\) has little effect on the early training convergence time \(t_1\).
- It increases the generalization time \(t_2\), approximately as \(t_2\propto 1/\lambda\).
- Therefore, smaller weight decay produces a longer grokking delay.
- Larger weight decay can weaken or eliminate grokking.

#### Number of training samples \(n\)

- Fewer samples make the training loss decrease more quickly.
- The model then overfits earlier, reducing \(t_1\).
- Consequently, the gap between fitting the training data and generalizing becomes more pronounced.

#### Feature dimension or width \(m\)

- Grokking is most visible in the over-parameterized regime \(m\gg n\).
- Increasing \(m\) alone, however, does not substantially increase \(t_1\) or \(t_2\) in the experiments.
- The important factor is the existence of many parameter directions that are not constrained by the training data.

#### Initialization scale \(\nu^2\)

- Larger initialization increases both \(t_1\) and \(t_2\), approximately logarithmically.
- The theoretical result suggests that \(t_2\) can grow faster than \(t_1\), amplifying the grokking gap.
- In nonlinear networks, the main observed effect was often a larger separation between training and test losses rather than a large increase in the absolute delay.

---

### 6. Random-feature ReLU results

The hidden layer is fixed and only the output layer is trained, so this model is equivalent to linear regression over a nonlinear fixed feature map.

The experiments show that:

- Smaller weight decay produces a longer generalization delay.
- Smaller training sets make training convergence faster and grokking more pronounced.
- Increasing the width does not substantially prolong grokking.
- The qualitative hyperparameter dependencies predicted for ridge regression also appear in the random-feature model.

Because the fixed random features do not exactly match the teacher features, this experiment is approximately realizable rather than strictly realizable. A sufficiently wide model can nevertheless approximate the teacher well.

---

### 7. Fully trained nonlinear neural networks

For the two-layer ReLU network, both hidden and output layers are trained.

- The teacher is mainly the zero function.
- Training and test losses are compared over time.
- The same qualitative trends appear:
  - smaller weight decay delays generalization,
  - fewer samples accelerate training convergence and strengthen grokking,
  - changes in width have a relatively limited effect.

These results suggest that the ridge-regression mechanism captures behavior that also appears in nonlinear networks. However, the paper provides empirical evidence rather than a theorem of the same strength as the ridge-regression result.

---

### 8. Threshold-based accuracy and additional experiments

Since regression loss usually decreases smoothly, the classic test-accuracy plateau may be less visually obvious. The authors therefore also evaluate

\[
\Pr_x\left[
\bigl(N(x;\theta)-N^*(x)\bigr)^2\leq\epsilon
\right].
\]

This threshold-based accuracy displays a more familiar grokking pattern:

- low test accuracy initially,
- a plateau after training loss has already become small,
- a later increase in test accuracy.

Grokking is also observed experimentally in noisy-label ridge regression and random Fourier feature models, although these settings are outside the main theoretical guarantee.

---

### 9. Overall conclusion

The experimental results support five main conclusions:

1. Grokking is not specific to deep neural networks.
2. Over-parameterized ridge regression can exhibit grokking solely because training and generalization converge at different rates.
3. The main mechanism is the slow decay of parameter components that are invisible to the training data.
4. Weight decay, sample size, and initialization can systematically amplify or suppress grokking.
5. The qualitative predictions from ridge regression are also observed in random-feature and fully trained ReLU networks.


<br/>
# 예제


이 논문에서 다루는 핵심 과제는 **훈련 데이터에는 매우 잘 맞지만 테스트 데이터에서는 오랫동안 성능이 나쁘다가, 학습을 계속하면 뒤늦게 일반화하는 현상(grokking)**을 선형 회귀와 신경망에서 재현하고 이론적으로 설명하는 것입니다.

### 1. 기본적인 학습 문제

각 데이터는 다음과 같은 형태입니다.

- 입력: \(x\)
- 정답 출력: \(y=N^*(x)\)
- 학생 모델의 예측:  
  \[
  N(x;\theta)=\langle \theta,\phi(x)\rangle
  \]

여기서 \(\phi(x)\)는 입력을 \(m\)차원 특징 벡터로 바꾸는 함수입니다.

- 훈련 데이터: \((x_1,y_1),\ldots,(x_n,y_n)\)
- 테스트 데이터: 훈련에 사용하지 않은 새로운 \(x\)를 같은 분포에서 뽑아 평가
- 손실: 예측값과 정답의 제곱 오차

\[
\text{training loss}
=\frac{1}{n}\sum_{i=1}^{n}(N(x_i;\theta)-y_i)^2
\]

\[
\text{test loss}
=\mathbb{E}_{x}\left[(N(x;\theta)-N^*(x))^2\right]
\]

학습 시에는 여기에 가중치 감소(weight decay)를 추가합니다.

\[
L_n(\theta;\lambda)
=\frac{1}{2n}\sum_{i=1}^{n}(N(x_i;\theta)-y_i)^2
+\frac{\lambda}{2}\|\theta\|_2^2
\]

---

## 2. 예시 A: 0 함수(zero teacher)를 학습하는 선형 모델

논문에서 가장 단순하게 분석한 예시입니다.

### 구체적인 테스크

어떤 입력이 들어와도 출력이 0이어야 합니다.

\[
N^*(x)=0
\]

예를 들어 입력 차원이 3이라면 다음과 같습니다.

| 입력 \(x\) | 정답 \(y\) |
|---|---:|
| \((1.2,-0.4,0.7)\) | 0 |
| \((-0.5,0.8,1.1)\) | 0 |
| \((0.2,0.1,-1.3)\) | 0 |

테스트 데이터도 새로운 입력을 사용하지만 정답은 모두 0입니다.

| 테스트 입력 \(x\) | 정답 \(y\) |
|---|---:|
| \((0.6,-1.0,0.3)\) | 0 |
| \((-1.1,0.2,0.4)\) | 0 |

### 모델

가장 단순하게 \(\phi(x)=x\)라고 하면,

\[
N(x;\theta)=\theta^\top x
\]

입니다. 예를 들어 \(\theta=(0.5,-0.2,0.1)\)이면 \(x=(1,2,-1)\)에 대한 예측은

\[
0.5(1)-0.2(2)+0.1(-1)=0
\]

입니다.

### 어떻게 grokking이 발생하는가?

모델의 파라미터를 훈련 데이터가 포함하는 방향과 포함하지 않는 방향으로 나눌 수 있습니다.

- **훈련 데이터가 관측한 방향**: 훈련 오차를 빠르게 줄임
- **훈련 데이터가 관측하지 못한 방향**: 초기 랜덤 가중치가 남아 있어 테스트 오차를 크게 만들 수 있음

따라서 학습 초기에 다음과 같은 현상이 나타납니다.

1. 훈련 데이터의 정답은 빠르게 맞춤
2. 테스트 입력에 대해서는 여전히 큰 출력이 나옴
3. 가중치 감소가 아주 천천히 작동하면서 남아 있는 가중치가 서서히 0으로 감소
4. 시간이 충분히 지나면 테스트 출력도 0에 가까워짐

즉,

\[
\text{training error}\downarrow \quad\text{빠르게}
\]

하지만

\[
\text{test error}\downarrow \quad\text{매우 느리게}
\]

됩니다. 논문은 이 차이를 grokking의 핵심 원인으로 설명합니다.

---

## 3. 예시 B: 일반적인 실현 가능한(realizable) 선형 교사

0 함수가 아닌 선형 함수도 같은 방식으로 다룰 수 있습니다.

### 구체적인 테스크

교사 함수가 다음과 같다고 가정합니다.

\[
N^*(x)=\theta^{*T}\phi(x)
\]

예를 들어 \(\phi(x)=x\), \(\theta^*=(2,-1)\)이면

\[
N^*(x_1,x_2)=2x_1-x_2
\]

입니다.

### 훈련 데이터

| 입력 \(x\) | 정답 \(y=2x_1-x_2\) |
|---|---:|
| \((1,0)\) | 2 |
| \((0,1)\) | -1 |
| \((2,1)\) | 3 |

### 테스트 데이터

| 새로운 입력 \(x\) | 정답 |
|---|---:|
| \((1,2)\) | 0 |
| \((-1,1)\) | -3 |
| \((0.5,-2)\) | 3 |

학생 모델도

\[
N(x;\theta)=\theta^\top x
\]

형태입니다.

### 학습 과정

초기에는 랜덤하게 \(\theta\)를 정하므로 훈련 및 테스트 오차가 모두 클 수 있습니다. 이후:

- 훈련에 사용된 입력 방향에 해당하는 파라미터는 빠르게 조정됨
- 훈련 데이터에 나타나지 않은 고차원 방향의 초기 랜덤 성분은 남음
- 따라서 훈련 오차는 작아졌지만 테스트 오차는 계속 큼
- weight decay가 모든 파라미터를 천천히 줄이면서 불필요한 성분을 제거
- 최종적으로 일반화가 좋아짐

논문의 Theorem 4.2는 이러한 현상이 임의의 실현 가능한 선형 교사 함수에 대해서도 발생할 수 있음을 보입니다.

---

## 4. 논문 실험의 구체적인 선형 회귀 설정

논문 Section 5.1에서는 다음 설정을 사용합니다.

### 데이터와 모델

- 특징 함수:
  \[
  \phi(x)=x\in\mathbb{R}^m
  \]
- 교사 파라미터:
  \[
  \|\theta^*\|_2=1
  \]
- 훈련 입력:
  \[
  x_i\sim\mathcal{N}(0,\frac{1}{m}I_m)
  \]
- 훈련 정답:
  \[
  y_i=\theta^{*T}x_i
  \]
- 테스트 입력: 같은 가우시안 분포에서 새롭게 샘플링
- 기본 실험 설정:
  - \(n=100\): 훈련 샘플 수
  - \(m=1000\): 특징 차원
  - \(\nu^2=1\): 초기화 크기
  - \(\lambda=10^{-4}\): weight decay
  - \(\eta=1\): 학습률

즉, 샘플 수 100개로 차원 1000의 모델을 학습하므로 **매우 과매개변수화된 상황**입니다.

### 관찰되는 현상

- 초기에 훈련 손실이 빠르게 감소
- 테스트 손실은 일정 시간 동안 높은 상태로 유지
- 이후 테스트 손실이 천천히 감소
- 학습을 충분히 오래 하면 테스트 성능이 좋아짐

논문은 특히 \(\lambda\)를 작게 하면 테스트 성능이 좋아지는 시점 \(t_2\)가 늦어져 grokking 시간이 길어진다고 설명합니다.

---

## 5. 예시 C: Random ReLU features 네트워크

논문은 선형 회귀와 유사한 현상이 비선형 함수에서도 나타나는지 확인하기 위해 random ReLU features 모델을 사용합니다.

### 모델

\[
N(x;a)=\sum_{j=1}^{m}a_j\sigma(w_j^\top x)
\]

여기서

\[
\sigma(z)=\max(0,z)
\]

는 ReLU 함수입니다.

- \(w_j\): 무작위로 초기화한 뒤 고정
- \(a_j\): 학습하는 출력층 가중치

이 모델은 \(a\)에 대해서는 선형이므로, 다음과 같은 특징 벡터를 사용하는 선형 회귀로 볼 수 있습니다.

\[
\phi(x)=
\big(
\sigma(w_1^\top x),\ldots,\sigma(w_m^\top x)
\big)
\]

### 교사 함수

교사는 하나의 ReLU 뉴런입니다.

\[
N^*(x)=\sigma(w^{*T}x)
\]

예를 들어 \(w^*=(1,-1)\)이면:

| 입력 \(x\) | 교사 출력 \(N^*(x)=\max(0,x_1-x_2)\) |
|---|---:|
| \((2,1)\) | 1 |
| \((1,2)\) | 0 |
| \((3,0)\) | 3 |
| \((-1,1)\) | 0 |

학생 네트워크는 여러 개의 랜덤 ReLU 특징을 조합해 이 함수를 학습합니다.

### 실험에서 확인한 효과

- weight decay를 작게 하면 일반화가 늦어짐
- 훈련 샘플 수 \(n\)을 줄이면 훈련 오차가 더 빨리 감소하여 grokking이 강해짐
- 모델 폭 \(m\)을 늘리는 것만으로는 grokking 시간이 크게 늘지 않음
- 선형 회귀에서 얻은 이론적 예측과 비슷한 경향이 나타남

다만 이 실험은 유한한 random feature 모델이므로 교사 함수와 완전히 일치하는 실현 가능 조건은 아니며, 충분히 넓은 모델에서 실현 가능한 경우에 가까워진다고 설명합니다.

---

## 6. 예시 D: 두 층 ReLU 신경망

Section 5.3에서는 두 층 ReLU 네트워크의 **두 층 모두**를 학습합니다.

\[
N(x;W,a)=\sum_{j=1}^{m}a_j\sigma(w_j^\top x)
\]

### 테스크

교사 함수는 다시 0 함수입니다.

\[
N^*(x)=0
\]

따라서 훈련 및 테스트 데이터의 정답은 모두 0입니다.

예를 들어:

| 입력 | 정답 |
|---|---:|
| \((0.4,-1.2,\ldots)\) | 0 |
| \((-0.7,0.3,\ldots)\) | 0 |

### 실험 설정의 예

- 입력 차원 \(d=50\)
- 훈련 샘플 수 \(n=50\)
- 은닉 뉴런 수 \(m=1000\)
- 학습률 \(\eta=10^{-4}\)
- weight decay \(\lambda=0.05\)

이 실험에서도 훈련 손실은 먼저 작아지고, 테스트 손실은 뒤늦게 감소하는 경향이 나타납니다. 논문은 이 결과가 선형 ridge regression에서 얻은 다음 예측과 질적으로 일치한다고 설명합니다.

- 작은 \(\lambda\): 일반화 지연 증가
- 작은 \(n\): 훈련 오차 감소가 빨라져 grokking 강화
- \(m\) 증가: grokking 시간에 큰 영향이 없을 수 있음

---

## 7. 논문에서 말하는 grokking의 시간적 구조

논문은 두 시점을 정의합니다.

- \(t_1\): 훈련 손실이 임계값 \(\epsilon\)보다 작아지는 시점
- \(t_2\): 테스트 손실이 임계값 \(c\)보다 작아지는 시점

따라서 grokking 시간은 대략

\[
t_2-t_1
\]

입니다.

논문의 핵심 결론은 다음과 같습니다.

1. \(t_1\)은 비교적 빠르게 도달할 수 있다.
2. \(t_1\) 이후에도 테스트 오차가 오랫동안 높게 유지될 수 있다.
3. weight decay가 장기간 작동하면 결국 테스트 오차가 감소한다.
4. 특히 작은 \(\lambda\)는 일반화 시점을 늦추므로 grokking을 증폭한다.
5. 충분히 오래 학습하면 ridge 해에 수렴하여 좋은 일반화를 얻는다.

---



## 1. Basic learning setup

The paper studies **grokking**, where the model fits the training data early but generalizes only much later.

Each example consists of:

- Input: \(x\)
- Target output: \(y=N^*(x)\)
- Student prediction:
  \[
  N(x;\theta)=\langle\theta,\phi(x)\rangle
  \]

The training set contains \(n\) examples \((x_i,y_i)\), while the test set consists of new inputs sampled from the same input distribution.

The training objective is ridge regression:

\[
L_n(\theta;\lambda)
=
\frac{1}{2n}\sum_{i=1}^{n}
\big(N(x_i;\theta)-y_i\big)^2
+
\frac{\lambda}{2}\|\theta\|_2^2.
\]

The first term measures training error, and the second term is weight decay.

---

## 2. Example A: Learning the zero function

The simplest task studied in the paper is

\[
N^*(x)=0
\]

for every input.

For example:

| Input \(x\) | Target \(y\) |
|---|---:|
| \((1.2,-0.4,0.7)\) | 0 |
| \((-0.5,0.8,1.1)\) | 0 |
| \((0.2,0.1,-1.3)\) | 0 |

The test set contains new inputs, but their targets are also all zero.

With the identity feature map \(\phi(x)=x\), the student model is

\[
N(x;\theta)=\theta^\top x.
\]

### Why grokking occurs

The parameter vector can be decomposed into:

- A component in directions represented by the training data
- A component in directions not represented by the training data

The first component is rapidly adjusted to reduce the training loss. The second component remains close to its random initialization and can produce large outputs on unseen test inputs.

Weight decay gradually shrinks this remaining component. Consequently:

1. Training error becomes small quickly.
2. Test error remains large for a long time.
3. The test error eventually decreases as weight decay removes unnecessary parameters.

---

## 3. Example B: A realizable linear teacher

Suppose the teacher function is

\[
N^*(x)=\theta^{*T}\phi(x).
\]

For example, with \(\phi(x)=x\) and \(\theta^*=(2,-1)\),

\[
N^*(x_1,x_2)=2x_1-x_2.
\]

### Training examples

| Input \(x\) | Target \(y\) |
|---|---:|
| \((1,0)\) | 2 |
| \((0,1)\) | -1 |
| \((2,1)\) | 3 |

### Test examples

| New input \(x\) | Target \(y\) |
|---|---:|
| \((1,2)\) | 0 |
| \((-1,1)\) | -3 |
| \((0.5,-2)\) | 3 |

The student uses the same linear form \(N(x;\theta)=\theta^\top x\). It can fit the training examples quickly while still retaining large random components in directions that are not sufficiently constrained by the training data. Those components cause poor test performance until weight decay has acted for a long time.

Theorem 4.2 shows that this phenomenon can occur for any realizable linear teacher under suitable over-parameterization and hyperparameter choices.

---

## 4. Concrete ridge-regression experiment

In Section 5.1, the paper uses:

- Feature map:
  \[
  \phi(x)=x\in\mathbb{R}^m
  \]
- Teacher parameter with:
  \[
  \|\theta^*\|_2=1
  \]
- Training inputs:
  \[
  x_i\sim\mathcal{N}(0,\frac{1}{m}I_m)
  \]
- Labels:
  \[
  y_i=\theta^{*T}x_i
  \]
- Test inputs: independently sampled from the same Gaussian distribution
- Typical parameters:
  - \(n=100\) training samples
  - \(m=1000\) feature dimensions
  - initialization scale \(\nu^2=1\)
  - weight decay \(\lambda=10^{-4}\)
  - learning rate \(\eta=1\)

This is strongly over-parameterized because there are 1000 parameters but only 100 training examples.

The observed behavior is:

- Training loss decreases rapidly.
- Test loss stays high for a long period.
- Test loss eventually decreases.
- Smaller \(\lambda\) produces a longer generalization delay.

---

## 5. Example C: Random ReLU features

The paper also studies a random-feature ReLU model:

\[
N(x;a)=\sum_{j=1}^{m}a_j\sigma(w_j^\top x),
\qquad
\sigma(z)=\max(0,z).
\]

The hidden weights \(w_j\) are randomly initialized and fixed, while the output weights \(a_j\) are trained.

This is equivalent to linear regression on the feature map

\[
\phi(x)=
\big(
\sigma(w_1^\top x),\ldots,\sigma(w_m^\top x)
\big).
\]

The teacher is a single ReLU neuron:

\[
N^*(x)=\sigma(w^{*T}x).
\]

For example, if \(w^*=(1,-1)\),

| Input \(x\) | Target \(\max(0,x_1-x_2)\) |
|---|---:|
| \((2,1)\) | 1 |
| \((1,2)\) | 0 |
| \((3,0)\) | 3 |
| \((-1,1)\) | 0 |

The experiments show trends similar to the linear theory:

- Smaller weight decay delays generalization.
- Fewer training samples make training fit faster and amplify grokking.
- Increasing the width does not necessarily increase the grokking time substantially.

---

## 6. Example D: A two-layer ReLU neural network

In Section 5.3, both layers of the network are trained:

\[
N(x;W,a)
=
\sum_{j=1}^{m}a_j\sigma(w_j^\top x).
\]

The teacher is again the zero function:

\[
N^*(x)=0.
\]

Thus, all training and test inputs have target output zero. A typical experimental setting is:

- Input dimension: \(d=50\)
- Number of training samples: \(n=50\)
- Hidden width: \(m=1000\)
- Learning rate: \(\eta=10^{-4}\)
- Weight decay: \(\lambda=0.05\)

The network first obtains a small training loss, while its test loss decreases only later. The qualitative dependence on hyperparameters is similar to the ridge-regression results.

---

## 7. Definition of grokking time

The paper defines:

- \(t_1\): the last time at which training loss is still above a threshold \(\epsilon\)
- \(t_2\): the first time at which test loss becomes smaller than a threshold \(c\)

The grokking delay is therefore approximately

\[
t_2-t_1.
\]

The main conclusion is:

1. The model can overfit the training data quickly.
2. Poor generalization can persist long after overfitting.
3. Weight decay eventually removes unnecessary parameter components.
4. The model then converges toward a well-generalizing ridge solution.
5. Smaller weight decay generally makes the grokking delay longer.

<br/>
# 요약


1. 과매개변수 선형회귀에서 \(\ell_2\) 가중치 감쇠를 적용한 경사하강법을 분석해, 학습 오차와 일반화 오차가 서로 다른 속도로 감소하는 현상을 이론적으로 모델링했다.  
2. 모델은 초기에 학습 데이터를 빠르게 과적합하지만 일반화 성능은 오래 정체된 뒤, 가중치 감쇠에 의해 점차 최소 노름 해로 수렴하며 좋은 일반화를 달성하고, 저자들은 이 지연 시간의 정량적 하한을 제시했다.  
3. 특히 작은 가중치 감쇠 \(\lambda\)는 grokking 시간을 대략 \(1/\lambda\)에 비례해 늘리며, 초기화 크기·샘플 수·특징 차원도 지연에 영향을 주고, 이러한 경향은 영점/실현 가능한 교사 선형회귀와 ReLU 신경망 실험에서 확인됐다.  


 
1. The paper analyzes gradient descent with \(\ell_2\) weight decay in over-parameterized linear regression, showing that training and generalization errors can decrease at different rates.  
2. The model quickly overfits the training data, remains poorly generalized for a long period, and eventually converges toward a minimum-norm solution with good generalization; the paper provides quantitative lower bounds on this delay.  
3. In particular, smaller weight decay \(\lambda\) increases the grokking time roughly as \(1/\lambda\), while initialization scale, sample size, and feature dimension also affect the delay, with these trends verified in realizable/zero-teacher regression and ReLU-network experiments.

<br/>
# 기타



### 1. 핵심 메커니즘: 데이터가 보지 못하는 방향에서 발생하는 지연

이 논문의 가장 중요한 인사이트는 과적합과 일반화가 서로 다른 파라미터 방향에서 일어난다는 점입니다.

- 학습 데이터가 span하는 부분공간(row space)의 성분은 데이터에 의해 빠르게 업데이트됩니다.
- 반대로 데이터가 관측하지 못하는 직교 부분공간(null space)의 성분은 주로 weight decay에 의해서만 천천히 감소합니다.
- 따라서 모델은 먼저 학습 데이터를 거의 완벽하게 맞추지만, 초기화에서 비롯된 null-space 성분 때문에 테스트 오차는 오랫동안 크게 유지됩니다.
- 시간이 충분히 지나면 weight decay가 이 성분까지 서서히 줄이고, 결국 작은 norm의 ridge 해에 도달하여 일반화가 좋아집니다.

즉, 이 논문에서 grokking은 다음 전환으로 설명됩니다.

> 빠른 데이터 적합 단계 → 장기간의 과적합 단계 → 느린 정규화 및 일반화 단계

---

## 2. Figure 1: Ridge regression과 신경망의 학습·테스트 손실

### 구성

- 왼쪽: zero teacher를 학습하는 ridge regression
- 오른쪽: zero teacher를 학습하는 two-layer ReLU neural network
- x축은 로그 스케일
- 여러 독립적인 데이터와 초기화에 대한 평균적인 학습/테스트 손실을 비교

### 결과

두 모델 모두 다음과 같은 패턴을 보입니다.

1. training loss는 초기에 빠르게 감소합니다.
2. training loss가 거의 0이 된 뒤에도 test loss는 높은 수준에 머뭅니다.
3. 충분히 오랜 시간이 지나면 test loss도 감소합니다.

### 인사이트

Grokking은 신경망의 비선형성이나 깊이에만 의존하지 않습니다. 매우 단순한 선형 ridge regression에서도 동일한 현상이 나타납니다.

따라서 논문은 grokking을 다음과 같이 해석합니다.

- 구조적으로 복잡한 모델의 고유한 현상이라기보다,
- over-parameterization, 작은 weight decay, 랜덤 초기화, 장시간 학습이 결합되어 나타나는 최적화 현상입니다.

---

## 3. Figure 2: Ridge regression에서 하이퍼파라미터의 영향

이 그림은 이론식 (8)의 예측과 실제 실험 결과를 비교합니다.

### (a) Weight decay \(\lambda\)

#### 결과

- \(\lambda\)를 작게 할수록 generalization이 시작되는 시점 \(t_2\)가 늦어집니다.
- training loss가 감소하는 시점 \(t_1\)은 상대적으로 크게 변하지 않습니다.
- 따라서 grokking delay \(t_2-t_1\)가 증가합니다.

#### 인사이트

논문에서 가장 직접적으로 grokking을 조절하는 변수는 weight decay입니다.

\[
t_2 \propto \frac{1}{\lambda}
\]

작은 weight decay는 null-space 성분을 매우 천천히 제거하므로, 과적합 상태가 오래 지속됩니다. 반대로 \(\lambda\)를 크게 하면 일반화가 빨라져 grokking이 약해지거나 사라질 수 있습니다.

---

### (b) Sample size \(n\)

#### 결과

- 샘플 수가 적을수록 training loss가 더 빠르게 감소합니다.
- \(t_1\)이 작아지므로, 모델이 더 일찍 과적합합니다.
- 결과적으로 training과 generalization 사이의 간격이 커집니다.

#### 인사이트

적은 데이터는 모델이 학습 데이터를 빠르게 암기하게 하지만, 테스트 분포에 대한 정보는 충분히 제공하지 못합니다. 따라서 적은 \(n\)은 grokking을 증폭시킬 수 있습니다.

---

### (c) Feature dimension \(m\)

#### 결과

- feature dimension을 증가시켜도 \(t_1\)과 \(t_2\)는 크게 변하지 않는 것으로 관찰됩니다.
- 즉, 일정 수준 이상의 over-parameterization에서는 width를 더 키우는 것이 grokking 시간을 크게 늘리지 않습니다.

#### 인사이트

중요한 것은 단순히 모델을 크게 만드는 것이 아니라, 데이터가 관측하지 못하는 방향의 차원 \(m-n\)이 충분히 크고, 동시에 학습 방향과 null-space 방향의 수렴 속도가 크게 다른 상황을 만드는 것입니다.

---

### (d) Initialization scale \(\nu^2\)

#### 결과

- 초기화 규모를 키우면 \(t_1\)과 \(t_2\)가 모두 증가합니다.
- 증가율은 대략 로그 스케일입니다.

\[
t_1,t_2 \propto \log(\nu^2)
\]

- 이론적으로는 \(t_2\)가 \(t_1\)보다 더 빠르게 증가하여 grokking gap이 커질 수 있습니다.

#### 인사이트

큰 초기화는 모델이 제거해야 할 파라미터 에너지를 증가시킵니다. 특히 null-space에 남아 있는 초기화 성분이 커져서 일반화까지 더 오래 걸립니다.

---

## 4. Figure 3: Random-feature ReLU network

### 설정

- hidden layer는 랜덤하게 초기화한 뒤 고정
- output layer만 학습
- 따라서 이 모델은 고정 feature map을 사용하는 선형 regression과 수학적으로 매우 유사합니다.

### 결과

Figure 2와 비슷한 경향이 관찰됩니다.

- 작은 weight decay: generalization 지연 증가
- 작은 sample size: training convergence 가속 및 grokking 증폭
- 큰 width: grokking 시간에 큰 영향 없음
- 큰 initialization scale: training/test loss 사이의 간격을 넓힘

### 인사이트

Grokking의 핵심 원인은 반드시 feature learning에 있지 않습니다. feature가 고정된 random-feature 모델에서도 다음 구조만 있으면 grokking이 나타납니다.

1. feature dimension이 샘플 수보다 큼
2. training data를 빠르게 fitting
3. 데이터가 관측하지 못하는 성분이 weight decay로 천천히 제거됨

다만 이 실험은 teacher가 student feature map과 정확히 일치하지 않아 엄밀히는 non-realizable 설정입니다. 저자들은 충분히 넓은 네트워크에서는 이 차이가 작아져 realizable setting에 가까워진다고 설명합니다.

---

## 5. Figure 4: 두 층 ReLU 신경망에서의 실험

### 설정

- 두 층 ReLU 네트워크의 hidden layer와 output layer를 모두 학습
- zero teacher 사용
- 선형 모델이 아닌 실제 nonlinear neural network를 대상으로 실험

### 결과

하이퍼파라미터별 경향은 Figure 2와 질적으로 일치합니다.

- 작은 weight decay: generalization 지연
- 작은 sample size: 더 빠른 overfitting
- 큰 width: grokking 시간이 크게 늘어나지는 않음
- initialization scale 변화: training/test loss 간의 격차 변화

### 인사이트

ridge regression에서 얻은 이론적 예측이 단순한 선형 모델에만 국한되지 않을 가능성을 보여줍니다.

다만 이는 실험적 증거일 뿐, nonlinear network에 대한 end-to-end 정리나 엄밀한 증명은 아닙니다. 저자들도 향후 과제로 다음을 지적합니다.

- lazy regime에서 rich regime으로의 전환을 이용한 엄밀한 grokking 증명
- 두 층 신경망에서 overfitting을 이론적으로 보장하는 방법

---

## 6. Figure 5: 회귀 손실과 threshold-based accuracy

### 문제의식

회귀에서는 test loss가 일반적으로 부드럽게 감소하므로, 분류 문제에서 자주 보이는 뚜렷한 “test accuracy plateau”가 잘 나타나지 않습니다.

### 사용한 지표

저자들은 다음과 같은 threshold-based accuracy를 정의합니다.

\[
P_x\left((N(x;\theta(t))-N^*(x))^2\leq \epsilon\right)
\]

즉, 예측 오차가 \(\epsilon\) 이하인 테스트 샘플의 비율을 측정합니다.

### 결과

- 실제 squared loss는 비교적 부드럽게 감소합니다.
- 그러나 threshold-based accuracy를 사용하면,
  - 오랫동안 거의 변화가 없는 plateau가 나타나고,
  - 이후 급격히 상승하는 전형적인 grokking 형태가 나타납니다.

### 인사이트

Grokking의 plateau는 반드시 분류 모델에만 존재하는 것이 아닙니다. 연속적인 회귀 손실을 이진 성공 여부로 변환하면 유사한 plateau 현상이 나타납니다.

따라서 “grokking의 전형적인 시각적 모양”은 모델 자체뿐 아니라 평가 지표의 선택에도 영향을 받습니다.

---

## 7. Figure 6: Label noise가 있는 ridge regression

### 설정

- 학습 레이블에 평균 0인 Gaussian noise를 추가
- noise의 표준편차를 바꾸어가며 실험

### 결과

- label noise가 있어도 grokking이 관찰됩니다.
- 서로 다른 noise 수준에서 grokking time은 크게 변하지 않는 것으로 나타납니다.

### 인사이트

이 결과는 grokking이 완전히 깨끗하고 realizable한 데이터에만 국한되지 않을 수 있음을 시사합니다.

다만 중요한 한계가 있습니다.

- 이 논문의 주요 정리는 realizable teacher를 가정합니다.
- label noise가 있는 경우는 non-realizable setting이므로, Figure 6은 실험적 관찰입니다.
- 저자들은 non-realizable ridge regression에 대한 엄밀한 grokking 증명을 향후 연구 문제로 남겨둡니다.

---

## 8. Figure 7: Random Fourier features

### 설정

- Random Fourier feature map 사용:

\[
\phi(x)=\sqrt{\frac{2}{m}}\cos(Wx+b)
\]

- teacher도 같은 종류의 Fourier feature로 구성
- realizable한 random-feature regression 문제를 구성

### 결과

- 적절한 하이퍼파라미터 조정이 있을 때 grokking이 나타납니다.
- 하지만 다른 실험보다 조건에 민감하고, weight decay와 feature 관련 파라미터를 세밀하게 조정해야 합니다.

### 인사이트

Grokking은 특정한 identity feature나 ReLU feature에만 의존하지 않습니다. 다양한 feature map에서 나타날 수 있습니다.

동시에, grokking은 자동으로 발생하는 보편적 현상이라기보다 다음 조건의 균형에 민감합니다.

- feature covariance의 구조
- sample size와 feature dimension
- initialization scale
- weight decay
- 학습률

---

# Appendix의 주요 결과와 인사이트

## 9. Appendix A: 증명

### Theorem A.1 / Theorem 4.1: Zero teacher

zero teacher에서는 모델의 목표가 \(\theta=0\)입니다.

저자들은 다음 세 가지를 보입니다.

1. **Training loss의 빠른 감소**

   데이터가 span하는 row space 성분은 대략 다음 속도로 감소합니다.

   \[
   \left(1-\eta\frac{\lambda_{\min}^{+}(\Phi^\top\Phi)}{n}-\eta\lambda\right)^{2t}
   \]

2. **Generalization loss의 느린 감소**

   null space 성분은 데이터 gradient의 영향을 받지 않고 weight decay로만 감소합니다.

   \[
   (1-\eta\lambda)^{2t}
   \]

3. **파라미터 norm의 최종 감소**

   충분히 오래 학습하면 weight decay가 전체 파라미터 norm을 줄여 \(\theta=0\)에 도달하게 합니다.

### 핵심 인사이트

training과 test loss의 수렴 속도가 다르다는 사실만으로도 provable grokking이 발생합니다.

---

## 10. Appendix A: Realizable teacher의 증명

### Theorem A.7–A.9

realizable teacher \(\theta^*\)에 대해서도 세 단계가 성립함을 보입니다.

#### 초기 단계: 빠른 training fitting

row-space 성분이 빠르게 감소하여 training loss가 \(\epsilon\) 이하가 됩니다.

#### 중간 단계: 지속적인 poor generalization

초기화의 null-space 성분은 다음처럼 유지됩니다.

\[
\theta_\perp^{(t)}
=(1-\eta\lambda)^t\theta_\perp^{(0)}
\]

따라서 충분한 시간 동안 test error가 일정한 상수 \(c\)보다 크게 유지됩니다.

#### 최종 단계: 좋은 generalization

GD는 ridge objective의 전역 최적점 \(\theta_\lambda^*\)로 수렴합니다. 또한 uniform convergence를 이용해 이 해의 population loss가 작음을 보입니다.

### 핵심 인사이트

이 부분이 논문의 핵심 공헌입니다. 단순히 “training loss와 test loss가 다르게 움직일 수 있다”는 설명을 넘어,

- overfitting이 먼저 발생하고,
- 그 상태가 오래 지속되며,
- 이후 generalization이 좋아진다는

전체 grokking 과정을 하나의 정리로 보장합니다.

---

## 11. Appendix A: 하이퍼파라미터 조건

주요 조건은 다음과 같습니다.

- 충분히 큰 sample size \(n\)
- 충분히 큰 feature dimension \(m\)
- 충분히 작은 weight decay \(\lambda\)
- 안정적인 학습률 \(\eta\)

특히 \(m>n\)이어야 데이터가 보지 못하는 방향이 생기며, \(m-n\)이 클수록 초기화 성분이 남을 가능성이 커집니다.

### 주의할 점

이론적 bound는 다소 보수적입니다.

- \(t_1\)은 \(\lambda_{\min}^{+}(\Phi^\top\Phi)\)에 의존합니다.
- 이 값이 \(m,n\)에 따라 정확히 어떻게 변하는지는 일반적인 feature distribution에 대해 충분히 알려져 있지 않습니다.
- 따라서 \(n\)과 \(m\)에 대한 간단하고 보편적인 grokking-time 공식은 제공하지 못합니다.

---

## 12. Appendix A: Generalization bound

Theorem A.10은 ridge 해 \(\theta_\lambda^*\)가 population loss 측면에서도 좋은 성능을 보임을 증명합니다.

핵심 과정은 다음과 같습니다.

1. ridge penalty 때문에

   \[
   \|\theta_\lambda^*\|_2\leq \|\theta^*\|_2
   \]

2. 파라미터 norm이 제한된 선형 함수 클래스의 Rademacher complexity를 계산
3. uniform convergence를 적용
4. 충분한 샘플 수가 있으면 empirical loss와 population loss의 차이를 작게 제어

대략적으로 필요한 샘플 수는

\[
n=\Omega\left(
\frac{b^4\|\theta^*\|_2^4}{\epsilon^2}
\log\frac1\delta
\right)
\]

입니다.

### 인사이트

최종 일반화는 단순히 GD가 수렴하기 때문만이 아니라,

- ridge 해의 norm이 제한되고,
- 제한된 함수 클래스에서 uniform convergence가 성립하기 때문에

보장됩니다.

---

## 13. Appendix C: Chi-squared concentration

랜덤 초기화

\[
\theta^{(0)}\sim\mathcal N(0,\nu^2 I_m)
\]

의 norm이 평균적으로 \(m\nu^2\) 근처에 있다는 사실을 사용합니다.

이 보조정리는 다음을 보장합니다.

- null-space에 들어가는 초기화 성분이 충분히 큼
- 높은 확률로 \(\|\theta_\perp^{(0)}\|^2\)가 \(m-n\)에 비례
- 따라서 과적합 이후에도 테스트 오차가 일정 시간 동안 작아지지 않음

### 인사이트

랜덤 초기화는 단순한 기술적 가정이 아니라 grokking을 만들어내는 중요한 원인 중 하나입니다. 특히 고차원 모델에서는 초기화 에너지의 상당 부분이 데이터가 관측하지 못하는 방향에 놓일 수 있습니다.

---

# 전체 결론

이 논문에서 그림과 부록이 함께 보여주는 핵심은 다음과 같습니다.

1. **Grokking은 선형 ridge regression에서도 엄밀히 증명될 수 있다.**
2. **Training error와 generalization error는 서로 다른 파라미터 성분에 의해 지배된다.**
3. **작은 weight decay가 generalization을 지연시키는 가장 직접적인 원인이다.**
4. **Over-parameterization은 데이터가 보지 못하는 null-space를 만들어 장기 과적합을 가능하게 한다.**
5. **적은 sample size는 과적합을 앞당기고, 큰 초기화는 일반화를 늦출 수 있다.**
6. **이론적 경향은 random features와 nonlinear neural networks에서도 실험적으로 확인된다.**
7. **다만 noisy 또는 non-realizable 데이터와 fully trained nonlinear network에 대한 엄밀한 증명은 아직 남은 과제이다.**

---



## 1. Core mechanism: delayed decay in the data-invisible directions

The central insight is that training and generalization are controlled by different components of the parameter vector.

- The component in the row space of the training feature matrix is rapidly updated by the data.
- The component in the null space is invisible to the training data and is reduced mainly through weight decay.
- As a result, the model fits the training data quickly, while the initialization-dependent null-space component keeps the test error large.
- Only after a long period does weight decay remove this component, leading to a low-norm ridge solution with good generalization.

Thus, the paper interprets grokking as:

> fast fitting → prolonged overfitting → slow regularization-driven generalization

---

## 2. Figure 1: Ridge regression versus neural networks

### Setup

- Left: ridge regression trained on a zero teacher
- Right: a two-layer ReLU network trained on a zero teacher
- Training and test losses are plotted over logarithmic time

### Result

Both models show the same three-stage behavior:

1. Training loss decreases rapidly.
2. Test loss remains high long after training error becomes nearly zero.
3. Test loss eventually decreases.

### Insight

Grokking does not fundamentally require depth or nonlinear representations. It can already occur in a simple linear ridge-regression model.

This suggests that grokking can arise from a combination of:

- over-parameterization,
- small weight decay,
- random initialization,
- and sufficiently long optimization.

---

## 3. Figure 2: Hyperparameter control in ridge regression

### Weight decay \(\lambda\)

- Smaller \(\lambda\) delays the onset of generalization.
- The training-fitting time \(t_1\) changes little.
- The generalization time \(t_2\) increases approximately as

\[
t_2\propto \frac{1}{\lambda}.
\]

Therefore, reducing weight decay amplifies grokking, while increasing it can weaken or eliminate the delay.

### Sample size \(n\)

- Smaller \(n\) makes the training loss decrease faster.
- The model overfits earlier.
- The gap between training and test performance becomes larger.

Thus, fewer samples can amplify grokking by making memorization easier.

### Feature dimension \(m\)

- Increasing \(m\) has relatively little effect on \(t_1\) and \(t_2\) once the model is sufficiently over-parameterized.
- The important feature is the existence of a large data-invisible subspace, rather than width alone.

### Initialization scale \(\nu^2\)

- Increasing \(\nu^2\) increases both \(t_1\) and \(t_2\).
- The dependence is approximately logarithmic:

\[
t_1,t_2\propto \log(\nu^2).
\]

A larger initialization leaves more energy in the null space, which takes longer to decay.

---

## 4. Figure 3: Random-feature ReLU networks

### Setup

The hidden layer is randomly initialized and fixed, while only the output layer is trained. This makes the model mathematically similar to linear regression with a fixed feature map.

### Result

The experiments reproduce the trends from Figure 2:

- smaller weight decay delays generalization,
- smaller sample size accelerates overfitting,
- increasing width has limited impact on the grokking time,
- larger initialization increases the gap between training and test losses.

### Insight

Feature learning is not necessary for grokking. A fixed random-feature model can exhibit grokking as long as it has:

1. more features than samples,
2. rapid fitting of the training data,
3. slowly decaying components outside the data span.

The experiment is technically non-realizable because the teacher is not exactly represented by the student features, but a sufficiently wide model can approximate it closely.

---

## 5. Figure 4: Fully trained two-layer ReLU networks

### Setup

Both the hidden and output layers are trained, using a zero teacher.

### Result

The qualitative effects agree with the ridge-regression experiments:

- smaller weight decay delays generalization,
- smaller datasets lead to earlier overfitting,
- increasing width does not strongly prolong grokking,
- initialization affects the separation between training and test losses.

### Insight

The ridge-regression theory may capture mechanisms that also appear in nonlinear networks. However, this remains empirical evidence; the paper does not provide an end-to-end proof for the fully trained nonlinear model.

---

## 6. Figure 5: Threshold-based accuracy

### Motivation

In regression, squared test loss usually decreases smoothly, so the classic accuracy plateau seen in classification is not always visible.

The authors therefore measure

\[
P_x\left((N(x;\theta(t))-N^*(x))^2\leq\epsilon\right),
\]

the fraction of test inputs whose squared error is below a threshold.

### Result

- The ordinary regression loss decreases smoothly.
- The threshold-based accuracy shows a long plateau followed by a sharp increase.

### Insight

A plateau-like grokking curve is not exclusive to classification. It can emerge in regression when continuous errors are converted into a thresholded success metric.

---

## 7. Figure 6: Ridge regression with label noise

### Setup

Gaussian noise with mean zero is added to the training labels.

### Result

Grokking is still observed, and different noise levels do not substantially change the grokking time in the reported experiments.

### Insight

Grokking may persist beyond the perfectly realizable setting.

However, this is only an empirical result. The main theoretical analysis assumes a realizable teacher, and rigorous results for noisy or non-realizable ridge regression are left open.

---

## 8. Figure 7: Random Fourier features

### Setup

The model uses the random Fourier feature map

\[
\phi(x)=\sqrt{\frac{2}{m}}\cos(Wx+b),
\]

and the teacher is generated using the same type of feature.

### Result

Grokking can be observed, but the phenomenon is more sensitive to hyperparameter choices than in the basic ridge-regression experiments.

### Insight

Grokking is not specific to identity or ReLU features. It can occur with a variety of feature maps, although it depends sensitively on:

- the feature covariance,
- sample size,
- feature dimension,
- initialization,
- weight decay,
- and learning rate.

---

# Main appendix results

## Appendix A: Proof structure

### Zero teacher

For the zero teacher, the paper proves:

1. The training loss decreases at a fast rate governed by the smallest positive eigenvalue of the empirical feature covariance.
2. The generalization loss decreases at the slower rate

\[
(1-\eta\lambda)^{2t},
\]

because the null-space component is affected only by weight decay.
3. The parameter norm eventually decreases to zero.

This establishes provable grokking through a separation of convergence rates.

### Realizable teacher

For a general realizable teacher, the paper proves three stages:

- early training fitting,
- persistent poor generalization,
- eventual convergence to a well-generalizing ridge solution.

This is the main end-to-end theoretical contribution.

---

## Hyperparameter conditions

The theorems require:

- sufficiently many training samples,
- sufficiently large feature dimension,
- sufficiently small weight decay,
- and a stable learning rate.

The condition \(m>n\) is especially important because it creates a nontrivial null space. A larger \(m-n\) typically means more initialization energy remains outside the data span.

The bounds are somewhat conservative because the training-time bound depends on

\[
\lambda_{\min}^{+}(\Phi^\top\Phi),
\]

whose precise dependence on \(m\) and \(n\) is not generally known.

---

## Generalization theorem

The final ridge solution \(\theta_\lambda^*\) generalizes because:

1. ridge regularization controls its norm,
2. the corresponding linear function class has bounded Rademacher complexity,
3. uniform convergence connects empirical and population losses.

The required sample size is roughly

\[
n=\Omega\left(
\frac{b^4\|\theta^*\|_2^4}{\epsilon^2}
\log\frac1\delta
\right).
\]

Thus, eventual generalization is not merely a consequence of optimization convergence; it also relies on statistical control of the low-norm hypothesis class.

---

## Appendix C: Gaussian initialization concentration

The chi-squared concentration lemma shows that for

\[
\theta^{(0)}\sim\mathcal N(0,\nu^2I_m),
\]

the initialization norm in the null space is typically proportional to \(m-n\).

This establishes that the null-space component is not a rare event. In high dimensions, a substantial amount of initialization energy naturally lies in directions unseen by the training data, providing the source of prolonged overfitting.

---

## Overall conclusion

The figures and appendices together support the following picture:

1. Grokking can be rigorously established in linear ridge regression.
2. Training and generalization are governed by different parameter components.
3. Small weight decay is the main mechanism that delays generalization.
4. Over-parameterization creates data-invisible directions that sustain overfitting.
5. Smaller datasets can make overfitting happen earlier.
6. Larger initialization can delay the eventual removal of harmful components.
7. Similar qualitative behavior appears in random-feature and nonlinear neural networks.
8. Rigorous results for noisy, non-realizable data and fully trained nonlinear networks remain open problems.

<br/>
# refer format:



### BibTeX

```bibtex
@inproceedings{xu2026grok,
  author    = {Mingyue Xu and Gal Vardi and Itay Safran},
  title     = {To Grok Grokking: Provable Grokking in Ridge Regression},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  address   = {Seoul, South Korea},
  publisher = {PMLR}
}
```

### 시카고 스타일 참고문헌

Xu, Mingyue, Gal Vardi, and Itay Safran. “To Grok Grokking: Provable Grokking in Ridge Regression.” In *Proceedings of the 43rd International Conference on Machine Learning*. Vol. 306 of *Proceedings of Machine Learning Research*. Seoul, South Korea: PMLR, 2026.

### 시카고 스타일 각주

1. Mingyue Xu, Gal Vardi, and Itay Safran, “To Grok Grokking: Provable Grokking in Ridge Regression,” in *Proceedings of the 43rd International Conference on Machine Learning*, vol. 306, *Proceedings of Machine Learning Research* (Seoul, South Korea: PMLR, 2026).



