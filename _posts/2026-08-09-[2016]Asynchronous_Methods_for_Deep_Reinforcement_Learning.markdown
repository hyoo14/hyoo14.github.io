---
layout: post
title:  "[2016]Asynchronous Methods for Deep Reinforcement Learning"
date:   2026-08-09 17:53:18 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 비동기식(asynchronous) 강화학습 방법 제안, 특히 A3C(Asynchronous Advantage Actor-Critic) 제안   
이는 하나의 공유 신경망을 동시에 업데이트, 병렬 학습 방식으로서 에이전트들이 서로 다른 상태를 경험하게 하여 데이터 간의 상관관계를 줄이고, 딥러닝 기반 강화학습의 불안정성을 완화   


짧은 요약(Abstract) :


이 논문은 **여러 에이전트가 동시에 학습하는 비동기식(asynchronous) 강화학습 방법**을 제안합니다. 각 에이전트는 서로 다른 환경에서 경험을 수집하고, 그 경험을 바탕으로 하나의 공유 신경망을 동시에 업데이트합니다. 이러한 병렬 학습 방식은 에이전트들이 서로 다른 상태를 경험하게 하여 데이터 간의 상관관계를 줄이고, 딥러닝 기반 강화학습의 불안정성을 완화합니다.

논문에서는 **Sarsa, Q-learning, n-step Q-learning, actor-critic**이라는 네 가지 강화학습 알고리즘을 비동기 방식으로 구현했습니다. 실험 결과, 네 방법 모두 신경망 정책을 안정적으로 학습할 수 있었으며, 특히 **A3C(Asynchronous Advantage Actor-Critic)**가 가장 좋은 성능을 보였습니다.

A3C는 Atari 게임에서 기존 최고 수준의 성능을 뛰어넘었고, GPU가 아닌 **하나의 멀티코어 CPU에서 기존 방법의 절반 정도 시간**만으로 학습되었습니다. 또한 Atari 게임뿐 아니라 연속적인 동작 제어 문제와 시각 정보만으로 무작위 3D 미로를 탐색하는 문제에서도 성공적으로 학습했습니다.

> **핵심:** 경험 replay 없이도 여러 에이전트를 병렬로 실행하면 학습이 더 안정적이고 빨라질 수 있으며, A3C는 다양한 환경에 적용 가능한 강력한 강화학습 방법이라는 내용입니다.

---




This paper proposes an **asynchronous reinforcement learning framework** in which multiple agents learn in parallel. Each agent interacts with its own copy of the environment, collects experience, and asynchronously updates a shared neural network. Because the agents encounter different states at the same time, their experiences become less correlated, which helps reduce the instability of deep reinforcement learning.

The authors develop asynchronous versions of four standard algorithms: **Sarsa, Q-learning, n-step Q-learning, and actor-critic**. All four methods are able to train neural-network controllers successfully, but the best-performing method is **A3C (Asynchronous Advantage Actor-Critic)**.

A3C outperforms previous state-of-the-art methods on Atari games while training in about half the time on a single multi-core CPU, without relying on a GPU. It also performs well on continuous motor-control tasks and on a visual 3D maze-navigation task.

> **Key idea:** Parallel actor-learners can stabilize and accelerate deep reinforcement learning without experience replay, and A3C is effective across a wide range of environments.


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



### 1) 핵심 아이디어: 경험 재생 대신 비동기 병렬 학습

이 논문은 여러 개의 **actor-learner**를 동시에 실행하여 심층 강화학습을 안정화하는 방법을 제안한다.

- 각 actor-learner는 자신의 환경(environment) 사본과 상호작용한다.
- 각 스레드는 현재 상태에서 행동을 선택하고 보상 및 다음 상태를 수집한다.
- 여러 스레드가 계산한 gradient를 하나의 **공유 전역 네트워크(shared global network)**에 비동기적으로 적용한다.
- 데이터가 서로 다른 환경과 상태에서 동시에 생성되므로, 시간적으로 강하게 상관된 연속 데이터가 완화된다.
- 따라서 DQN처럼 큰 **experience replay memory**를 사용하지 않고도 학습을 안정화할 수 있다.

논문에서는 하나의 머신에서 여러 CPU 스레드를 사용하며, 파라미터와 gradient 업데이트는 **Hogwild! 방식의 lock-free 비동기 업데이트**를 사용한다.

---

### 2) 제안한 네 가지 강화학습 방법

논문은 다음 네 가지 알고리즘의 비동기 버전을 제안한다.

#### a. Asynchronous one-step Q-learning

Q-learning의 1-step target을 사용한다.

\[
y =
\begin{cases}
r & \text{terminal state}\\
r+\gamma \max_{a'} Q(s',a';\theta^-) & \text{otherwise}
\end{cases}
\]

- 현재 Q-network와 별도로 천천히 갱신되는 **target network** \(\theta^-\)를 사용한다.
- 여러 시간 단계에서 gradient를 누적한 후 한 번에 공유 네트워크에 적용한다.
- 각 스레드는 서로 다른 \(\epsilon\)-greedy 탐색률을 사용할 수 있다.

#### b. Asynchronous one-step Sarsa

Q-learning과 구조는 유사하지만, 다음 상태에서 실제로 선택한 행동의 값을 target으로 사용한다.

\[
y = r+\gamma Q(s',a';\theta^-)
\]

즉, 최대 Q값을 사용하는 Q-learning과 달리 현재 정책이 실제로 선택한 행동을 반영한다.

#### c. Asynchronous n-step Q-learning

한 번의 보상만 사용하는 대신, 여러 단계의 보상을 포함한 **n-step return**을 사용한다.

\[
R_t^{(n)}
= r_t+\gamma r_{t+1}+\cdots+
\gamma^{n-1}r_{t+n-1}
+\gamma^n\max_a Q(s_{t+n},a)
\]

- 보상이 여러 이전 상태-행동 쌍에 빠르게 전달된다.
- 논문에서는 backward view가 아니라 **forward view**로 n-step return을 직접 계산한다.
- 최대 \(t_{\max}\)개 행동을 수행하거나 episode가 끝나면 gradient를 계산하여 업데이트한다.

#### d. Asynchronous Advantage Actor-Critic, A3C

논문의 대표적인 방법이다. A3C는 다음 두 네트워크 또는 두 출력값을 함께 학습한다.

- **Actor:** 정책 \(\pi(a_t|s_t;\theta)\)를 출력
- **Critic:** 상태 가치 \(V(s_t;\theta_v)\)를 추정

정책 업데이트에는 advantage를 사용한다.

\[
A_t =
\sum_{i=0}^{k-1}\gamma^i r_{t+i}
+\gamma^k V(s_{t+k})
-V(s_t)
\]

정책은 advantage가 큰 행동의 확률을 높이고, critic은 가치 함수 예측 오차를 줄인다.

실제 구현에서는 일반적으로 하나의 convolutional neural network에서 다음을 출력한다.

- 정책을 위한 softmax output
- 가치 함수를 위한 linear output
- 두 출력 이전의 convolutional 및 hidden layer는 공유

또한 정책의 엔트로피를 목적 함수에 추가한다.

\[
\nabla_\theta \log \pi(a_t|s_t)
(R_t-V(s_t))
+\beta \nabla_\theta H(\pi)
\]

엔트로피 보너스는 정책이 너무 이르게 결정적인 행동 하나로 수렴하는 것을 막고 탐색을 촉진한다.

---

### 3) 네트워크 구조

Atari 실험에서는 기존 DQN과 유사한 시각 입력 기반 네트워크를 사용한다.

- 입력: 게임 화면 이미지
- convolutional neural network를 이용해 시각적 특징 추출
- A3C의 경우:
  - 정책 확률을 출력하는 softmax head
  - 상태 가치를 출력하는 linear head
- 일부 실험에서는 convolutional layer 뒤에 **256개의 LSTM cell**을 추가한 recurrent agent를 사용한다.

LSTM은 과거 관측 정보를 기억할 수 있으므로, 현재 화면만으로 충분한 정보를 얻기 어려운 부분관측 환경이나 미로 탐색에 유리하다.

---

### 4) 학습 데이터와 환경

이 논문은 고정된 지도학습 데이터셋을 사용하지 않는다. 데이터는 agent가 환경과 직접 상호작용하면서 온라인으로 생성한다.

주요 실험 환경은 다음과 같다.

- **Atari 2600:** 57개 게임, 화면 이미지 입력
- **TORCS:** 3D 자동차 경주, RGB 이미지와 속도 기반 보상
- **MuJoCo:** 연속적인 운동 제어 및 조작 문제
- **Labyrinth:** 매 episode마다 새로운 3D 미로가 생성되는 시각 기반 탐색 문제

각 병렬 스레드는 서로 다른 환경 인스턴스에서 경험을 수집하므로, 전체 학습 데이터가 다양한 상태와 궤적을 포함하게 된다.

---

### 5) 주요 학습 기법

#### 비동기 actor-learner

각 스레드는 다음 과정을 반복한다.

1. 공유 네트워크에서 자신의 로컬 파라미터를 읽는다.
2. 자신의 환경에서 여러 step을 실행한다.
3. 수집한 보상으로 gradient를 계산한다.
4. gradient를 공유 전역 네트워크에 비동기적으로 적용한다.
5. 최신 전역 파라미터를 다시 사용한다.

#### 경험 재생 메모리 미사용

DQN처럼 과거 transition을 저장하고 무작위로 재사용하지 않는다.

그 대신 병렬 actor들이 서로 다른 상태를 경험하도록 하여 데이터 상관관계를 줄이고 학습을 안정화한다.

#### 다양한 탐색 정책

각 스레드의 \(\epsilon\)-greedy 탐색률 또는 탐색 방식이 다를 수 있다. 이는 여러 스레드가 비슷한 행동만 반복하는 것을 막고 탐색 다양성을 높인다.

#### Gradient 누적

각 step마다 즉시 업데이트하지 않고 여러 step의 gradient를 누적한 뒤 한 번에 적용한다. 이는 여러 스레드가 서로의 업데이트를 지나치게 덮어쓰는 문제를 줄인다.

#### RMSProp

저자들은 SGD with momentum, RMSProp 등을 비교했고, 스레드 간 RMSProp 통계량을 공유하는 방식이 가장 안정적이었다고 보고한다.

---

### 6) 방법의 장점

- experience replay 없이도 심층 신경망을 안정적으로 학습 가능
- on-policy 알고리즘인 Sarsa와 actor-critic에도 적용 가능
- off-policy 알고리즘인 Q-learning에도 적용 가능
- GPU가 아닌 멀티코어 CPU에서도 빠르게 학습
- discrete action뿐 아니라 continuous action에도 적용 가능
- feedforward 네트워크와 recurrent 네트워크 모두 지원
- 여러 actor가 동시에 데이터를 생성하므로 학습 시간이 크게 감소

특히 A3C는 Atari, TORCS, MuJoCo, Labyrinth에서 좋은 성능을 보였으며, Atari 57개 게임에서 기존 방법보다 짧은 시간과 적은 하드웨어 자원으로 높은 평균 성능을 달성했다.

---




### 1) Main idea: Asynchronous parallel learning instead of experience replay

The paper proposes stabilizing deep reinforcement learning by running multiple **actor-learners** in parallel.

- Each actor-learner interacts with its own copy of the environment.
- It collects states, actions, rewards, and next states online.
- The gradients computed by different threads are asynchronously applied to a shared global network.
- Since parallel actors experience different states at the same time, the training data become less temporally correlated.
- Therefore, the method can train deep networks without using a large experience replay memory.

The implementation uses multiple CPU threads on a single machine. Parameter updates follow a **Hogwild!-style lock-free asynchronous update scheme**.

---

### 2) Four asynchronous algorithms

The paper introduces asynchronous versions of four standard RL algorithms.

#### a. Asynchronous one-step Q-learning

It uses the one-step Q-learning target:

\[
y =
\begin{cases}
r & \text{for terminal states}\\
r+\gamma\max_{a'}Q(s',a';\theta^-)
& \text{otherwise}
\end{cases}
\]

A slowly updated target network \(\theta^-\) is used to improve stability. Gradients may be accumulated over several time steps before being applied to the shared network.

#### b. Asynchronous one-step Sarsa

The structure is similar to Q-learning, but the target uses the action actually selected by the policy:

\[
y=r+\gamma Q(s',a';\theta^-)
\]

Thus, Sarsa follows the behavior policy, whereas Q-learning uses the maximum estimated action value.

#### c. Asynchronous n-step Q-learning

Instead of using only the immediate reward, n-step Q-learning uses multiple future rewards:

\[
R_t^{(n)}
= r_t+\gamma r_{t+1}+\cdots+
\gamma^{n-1}r_{t+n-1}
+\gamma^n\max_a Q(s_{t+n},a)
\]

This allows rewards to propagate more quickly to preceding state-action pairs. The paper explicitly computes these returns using the **forward view**, up to a maximum number of steps \(t_{\max}\).

#### d. Asynchronous Advantage Actor-Critic, A3C

A3C is the main method proposed in the paper. It jointly learns:

- **Actor:** a policy \(\pi(a_t|s_t;\theta)\)
- **Critic:** a state-value function \(V(s_t;\theta_v)\)

The advantage estimate is computed as:

\[
A_t =
\sum_{i=0}^{k-1}\gamma^i r_{t+i}
+\gamma^k V(s_{t+k})
-V(s_t)
\]

The actor increases the probability of actions with positive advantage, while the critic learns to predict state values accurately.

In practice, the policy and value function usually share the same convolutional feature layers, followed by two output heads:

- a softmax policy output
- a linear value output

An entropy regularization term is also added to encourage exploration and prevent premature convergence to a deterministic policy.

---

### 3) Network architecture

For Atari experiments, the agents use a convolutional neural network similar to the architecture used in DQN.

- Input: game-screen images
- Convolutional layers: visual feature extraction
- A3C output heads:
  - softmax policy head
  - linear value head

For some experiments, the authors add **256 LSTM cells** after the final hidden layer. This recurrent architecture helps the agent retain information from previous observations and is useful in partially observable tasks such as visual maze navigation.

---

### 4) Training data and environments

The method does not use a fixed supervised-learning dataset. Training data are generated online through interaction with the environment.

The experiments include:

- **Atari 2600:** 57 games using visual input
- **TORCS:** 3D car racing with RGB images and speed-related rewards
- **MuJoCo:** continuous motor-control and manipulation tasks
- **Labyrinth:** visual navigation in randomly generated 3D mazes

Each parallel thread collects experience from a different environment instance, producing diverse trajectories and reducing correlation between updates.

---

### 5) Important training techniques

#### Asynchronous actor-learners

Each thread:

1. Reads the current shared parameters.
2. Interacts with its own environment for several steps.
3. Computes gradients from the collected rewards.
4. Applies the gradients asynchronously to the shared network.
5. Continues using updated parameters.

#### No experience replay

Past transitions are not stored and randomly sampled as in DQN. Instead, parallel actors provide decorrelated and diverse online experience.

#### Diverse exploration

Different threads can use different \(\epsilon\)-greedy exploration rates or exploration policies. This increases behavioral diversity and improves exploration.

#### Accumulated gradients

Gradients from multiple time steps are accumulated before an update. This reduces the chance that concurrent threads overwrite one another’s updates too aggressively.

#### RMSProp optimization

The authors compare SGD with momentum and different RMSProp variants. RMSProp with statistics shared across threads is reported to be the most robust option.

---

### 6) Main advantages

- Stable deep RL training without experience replay
- Supports both on-policy methods, such as Sarsa and actor-critic, and off-policy methods, such as Q-learning
- Efficient training on a standard multi-core CPU
- Applicable to both discrete and continuous action spaces
- Supports both feedforward and recurrent neural networks
- Parallel actors improve training speed and exploration

Among the proposed methods, A3C achieved particularly strong results across Atari, TORCS, MuJoCo, and Labyrinth, outperforming previous Atari approaches while using less training time and less specialized hardware.


<br/>
# Results



### 1. 비교 대상과 평가 환경

논문은 제안한 **비동기 강화학습(asynchronous RL)** 프레임워크를 다음 네 가지 알고리즘에 적용했다.

- **Asynchronous one-step Q-learning**
- **Asynchronous one-step SARSA**
- **Asynchronous n-step Q-learning**
- **A3C(Asynchronous Advantage Actor-Critic)**

주요 비교 대상은 다음과 같다.

- **DQN**
- **Double DQN**
- **Dueling Double DQN**
- **Prioritized DQN**
- **Gorila**
- 인간 플레이어 성능

평가 환경은 네 종류였다.

1. **Atari 2600 게임**
2. **TORCS 3D 자동차 경주**
3. **MuJoCo 연속 제어 과제**
4. **Labyrinth 3D 시각적 미로 탐색**

---

### 2. Atari 2600: 학습 속도 비교

먼저 5개 Atari 게임에서 학습 속도를 비교했다.

- Beamrider
- Breakout
- Pong
- Q*bert
- Space Invaders

#### 실험 조건

- DQN: **Nvidia K40 GPU 1개**
- 비동기 알고리즘: **CPU 코어 16개**
- 여러 개의 actor-learner가 각자 환경과 상호작용하면서 하나의 공유 모델을 비동기적으로 업데이트

#### 주요 결과

- 네 가지 비동기 알고리즘 모두 신경망 기반 Atari 에이전트를 성공적으로 학습했다.
- 많은 게임에서 비동기 방식이 DQN보다 더 빠르게 학습했다.
- 일부 게임에서는 **n-step 방법이 one-step 방법보다 빠르게 학습**했다.
- 네 방법 중에서는 **A3C가 전반적으로 가장 우수한 성능**을 보였다.

즉, 경험 재생 메모리(experience replay)를 사용하지 않고도 여러 actor-learner의 병렬 경험 수집을 통해 학습 데이터를 시간적으로 덜 상관되게 만들 수 있었고, 이 때문에 안정적인 학습이 가능했다.

---

### 3. Atari 2600: 57개 게임에서의 성능

A3C를 57개 Atari 게임에 적용하고, 기존 최신 방법들과 비교했다.

#### 평가 지표

- **Human-normalized score**
  - 인간 점수와 무작위 에이전트 점수를 기준으로 정규화한 성능
- **Mean**: 57개 게임의 평균 점수
- **Median**: 57개 게임 점수의 중앙값

#### 비교 결과

| 방법 | 학습 시간 및 하드웨어 | 평균 점수 | 중앙값 |
|---|---:|---:|---:|
| DQN | GPU 8일 | 121.9% | 47.5% |
| Gorila | 100대 머신, 4일 | 215.2% | 71.3% |
| Double DQN | GPU 8일 | 332.9% | 110.9% |
| Dueling Double DQN | GPU 8일 | 343.8% | 117.1% |
| Prioritized DQN | GPU 8일 | 463.6% | 127.6% |
| A3C, Feedforward | CPU 1일 | 344.1% | 68.2% |
| A3C, Feedforward | CPU 4일 | 496.8% | 116.6% |
| A3C, LSTM | CPU 4일 | **623.0%** | 112.6% |

#### 해석

- **A3C LSTM**은 평균 human-normalized score **623.0%**로 가장 높은 평균 성능을 기록했다.
- 이는 Prioritized DQN의 평균 점수인 **463.6%**보다 높다.
- A3C feedforward 모델도 4일 학습 기준으로 Prioritized DQN보다 높은 평균 점수를 보였다.
- A3C는 다른 방법들이 GPU를 사용해 8일 정도 학습한 것과 비교해, **16개 CPU 코어만으로 4일 학습**했다.
- A3C는 **1일 학습만으로도 Dueling Double DQN의 평균 성능에 근접**했고, Gorila의 중앙값 성능에도 거의 도달했다.

다만 중앙값에서는 Prioritized DQN이 127.6%로 A3C보다 높았으므로, A3C가 모든 게임에서 일관되게 최고였다는 의미는 아니다. A3C의 가장 큰 강점은 **전체 게임 평균 성능과 학습 효율**이었다.

---

### 4. TORCS 자동차 경주

TORCS에서는 시각 입력을 이용해 자동차를 조종하도록 학습했다.

#### 평가 조건

다음 네 가지 설정을 사용했다.

- 느린 자동차 / 상대 차량 없음
- 느린 자동차 / 상대 차량 있음
- 빠른 자동차 / 상대 차량 없음
- 빠른 자동차 / 상대 차량 있음

#### 메트릭과 결과

- 인간 플레이어 점수 대비 성능을 비교했다.
- A3C가 네 가지 설정 모두에서 네 알고리즘 중 가장 좋은 성능을 보였다.
- 약 **12시간의 학습** 후, 인간 테스터 점수의 약 **75~90%** 수준에 도달했다.

이는 A3C가 Atari와 같은 단순한 이산 행동 게임뿐 아니라, 시각 정보와 물리적 동역학이 필요한 자동차 제어 문제에도 적용될 수 있음을 보여준다.

---

### 5. MuJoCo 연속 행동 제어

MuJoCo에서는 로봇의 조작 및 이동과 같은 **연속 행동 공간** 과제를 평가했다.

#### 평가 방식

- 물리 상태 또는 픽셀 이미지를 입력으로 사용
- A3C만 평가
  - Q-learning 계열은 연속 행동 공간에 직접 적용하기 어렵기 때문

#### 결과

- 다양한 조작 및 이동 과제에서 A3C가 좋은 정책을 학습했다.
- 대부분의 과제에서 **수 시간 이내**, 모두 **24시간 이내**에 성공적인 정책을 찾았다.
- 연속 행동 제어에서도 A3C가 효과적으로 작동함을 확인했다.

---

### 6. Labyrinth 3D 미로 탐색

Labyrinth는 매 에피소드마다 새로운 3D 미로가 생성되는 시각 기반 탐색 과제다.

#### 입력과 목표

- 입력: **84×84 RGB 이미지**
- 미로 안에서 사과와 포털을 찾아 보상 획득
- 포털 발견 후 반복적으로 포털로 돌아가는 일반적인 탐색 전략을 학습해야 함
- A3C에 **LSTM**을 추가해 부분적으로 관측 가능한 환경에서 과거 정보를 활용

#### 결과

- 최종 평균 점수는 약 **50점**
- 에이전트는 보지 못한 새로운 미로에서도 합리적인 탐색 전략을 학습했다.
- 이는 A3C가 단순한 반응형 행동뿐 아니라, 시각적 입력과 기억이 필요한 3D 탐색 문제에도 적용될 수 있음을 보여준다.

---

### 7. 병렬 actor-learner 수에 따른 확장성

1, 2, 4, 8, 16개의 스레드를 사용해 학습 속도 향상을 측정했다.

#### 평균 학습 속도 향상

| 방법 | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads |
|---|---:|---:|---:|---:|---:|
| One-step Q | 1.0 | 3.0 | 6.3 | 13.3 | **24.1** |
| One-step SARSA | 1.0 | 2.8 | 5.9 | 13.1 | **22.1** |
| n-step Q | 1.0 | 2.7 | 5.9 | 10.7 | **17.2** |
| A3C | 1.0 | 2.1 | 3.7 | 6.9 | **12.5** |

- 16개 스레드 사용 시 모든 방법에서 최소 약 10배 이상의 속도 향상이 나타났다.
- 특히 one-step Q-learning과 SARSA는 이상적인 선형 속도 향상보다 높은 **초선형(superlinear) speedup**을 보였다.
- 이는 계산량 증가뿐 아니라, 병렬 actor들이 서로 다른 상태를 탐색하면서 데이터 효율성과 학습 편향도 개선했기 때문으로 해석된다.
- 더 많은 actor-learner를 사용할수록 one-step 방법의 데이터 효율이 향상되는 경향도 관찰됐다.

---

### 8. 안정성과 강건성

각 알고리즘을 5개 Atari 게임에서 다음 조건으로 반복 실험했다.

- 학습률 50가지
- 무작위 초기화 50가지

#### 결과

- 각 알고리즘과 게임 조합에서 좋은 성능을 내는 학습률 범위가 존재했다.
- 적절한 학습률 범위에서는 초기 가중치가 달라도 대체로 높은 성능을 얻었다.
- 학습이 진행되던 조건에서 점수가 0으로 붕괴하거나 발산하는 경우가 거의 없었다.

따라서 비동기 actor-learner 구조가 신경망 기반 강화학습의 학습을 안정화하는 데 효과적이라는 결론을 내릴 수 있다.

---

## 종합 결론

이 논문의 핵심 실험 결과는 다음과 같다.

1. 여러 actor-learner가 공유 모델을 비동기적으로 업데이트하면 경험 재생 없이도 안정적인 학습이 가능하다.
2. A3C는 Atari에서 기존 DQN 계열 방법보다 높은 평균 성능과 빠른 학습을 보였다.
3. 16개 CPU 코어만으로 GPU 기반 방법과 경쟁하거나 이를 능가했다.
4. A3C는 이산 행동뿐 아니라 연속 행동, 시각 입력, LSTM 기반 기억, 3D 탐색에도 적용 가능했다.
5. 병렬 스레드 수를 늘리면 학습 속도가 크게 향상되며, 일부 방법에서는 단순한 계산 속도 이상의 성능 개선이 나타났다.
6. 다만 A3C가 모든 게임에서 중앙값 기준으로 항상 최고였던 것은 아니며, 평균 성능과 계산 효율에서 특히 강점을 보였다.

---




## Summary of Experimental Results

### 1. Compared Algorithms and Evaluation Domains

The paper applies the asynchronous reinforcement learning framework to four algorithms:

- **Asynchronous one-step Q-learning**
- **Asynchronous one-step SARSA**
- **Asynchronous n-step Q-learning**
- **A3C (Asynchronous Advantage Actor-Critic)**

The main comparison methods include:

- DQN
- Double DQN
- Dueling Double DQN
- Prioritized DQN
- Gorila
- Human performance

The experiments were conducted on:

1. Atari 2600 games
2. TORCS 3D car racing
3. MuJoCo continuous-control tasks
4. Labyrinth, a visual 3D maze environment

---

### 2. Atari 2600: Learning-Speed Comparison

The authors first compared the methods on five Atari games:

- Beamrider
- Breakout
- Pong
- Q*bert
- Space Invaders

#### Experimental setup

- DQN was trained on **one Nvidia K40 GPU**.
- The asynchronous methods were trained using **16 CPU cores**.
- Multiple actor-learners interacted with separate environment instances and asynchronously updated a shared model.

#### Main findings

- All four asynchronous methods successfully trained neural-network controllers.
- They generally learned faster than DQN, with particularly large improvements on some games.
- The n-step methods often learned faster than the one-step methods.
- Among the four methods, **A3C achieved the best overall performance**.

The results suggest that parallel actors can decorrelate the training data and provide a stabilizing effect, reducing the need for experience replay.

---

### 3. Atari 2600: Results on 57 Games

A3C was evaluated on all 57 Atari games and compared with several state-of-the-art methods.

#### Evaluation metrics

The paper reports:

- **Human-normalized score**
- **Mean score** across 57 games
- **Median score** across 57 games

| Method | Training time and hardware | Mean | Median |
|---|---:|---:|---:|
| DQN | 8 days on GPU | 121.9% | 47.5% |
| Gorila | 4 days on 100 machines | 215.2% | 71.3% |
| Double DQN | 8 days on GPU | 332.9% | 110.9% |
| Dueling Double DQN | 8 days on GPU | 343.8% | 117.1% |
| Prioritized DQN | 8 days on GPU | 463.6% | **127.6%** |
| A3C, feedforward | 1 day on CPU | 344.1% | 68.2% |
| A3C, feedforward | 4 days on CPU | 496.8% | 116.6% |
| A3C, LSTM | 4 days on CPU | **623.0%** | 112.6% |

#### Interpretation

- **A3C with LSTM achieved the highest mean score: 623.0%.**
- This was higher than Prioritized DQN’s mean score of 463.6%.
- A3C also achieved strong performance with only 16 CPU cores, while other methods typically used GPUs or large distributed systems.
- After only one day, A3C nearly matched the average performance of Dueling Double DQN and approached Gorila’s median performance.

However, Prioritized DQN achieved a higher median score than A3C. Therefore, A3C was not the best on every individual game; its main advantage was its strong overall average performance and computational efficiency.

---

### 4. TORCS Car Racing

In TORCS, the agent controlled a car using visual input.

#### Evaluation settings

Four configurations were tested:

- Slow car without opponent bots
- Slow car with opponent bots
- Fast car without opponent bots
- Fast car with opponent bots

#### Metric and results

- Performance was compared with the score of a human tester.
- A3C performed best among the four asynchronous methods.
- After approximately **12 hours of training**, it reached roughly **75–90% of human performance** across the four configurations.

This demonstrates that A3C can handle not only Atari games but also visually controlled tasks involving realistic dynamics.

---

### 5. MuJoCo Continuous Control

MuJoCo was used to evaluate continuous-action tasks involving manipulation and locomotion.

#### Evaluation setup

- The agent used either physical-state inputs or pixel inputs.
- Only A3C was evaluated because policy-based methods are easier to extend to continuous action spaces than value-based Q-learning methods.

#### Results

- A3C found good policies across a range of continuous-control tasks.
- Most tasks were solved within a few hours, and all were solved within **24 hours**.
- The results show that A3C is effective in continuous-action environments as well as discrete-action games.

---

### 6. Labyrinth 3D Maze Navigation

Labyrinth is a visual 3D environment in which a new maze is generated at the beginning of each episode.

#### Input and objective

- Input: **84×84 RGB images**
- The agent had to find apples and portals to receive rewards.
- It needed to learn a general exploration strategy rather than memorize a single maze.
- An A3C agent with an LSTM was used to retain information from previous observations.

#### Results

- The final average score was approximately **50**.
- The agent learned a reasonable strategy for exploring previously unseen random mazes.
- This shows that A3C can handle visual navigation tasks requiring memory and generalization.

---

### 7. Scalability with the Number of Actor-Learners

The authors measured the speedup obtained with 1, 2, 4, 8, and 16 threads.

| Method | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads |
|---|---:|---:|---:|---:|---:|
| One-step Q-learning | 1.0 | 3.0 | 6.3 | 13.3 | **24.1** |
| One-step SARSA | 1.0 | 2.8 | 5.9 | 13.1 | **22.1** |
| n-step Q-learning | 1.0 | 2.7 | 5.9 | 10.7 | **17.2** |
| A3C | 1.0 | 2.1 | 3.7 | 6.9 | **12.5** |

#### Findings

- All methods obtained substantial speedups from parallelism.
- With 16 threads, each method achieved at least an order-of-magnitude speedup.
- One-step Q-learning and SARSA showed **superlinear speedups**, meaning the improvement was larger than what could be explained by computation alone.
- The authors attribute this partly to better exploration and improved data efficiency caused by multiple actors visiting different parts of the environment.

---

### 8. Robustness and Stability

Each algorithm was trained on five Atari games using:

- 50 different learning rates
- 50 random initializations

#### Results

- There was usually a range of learning rates that produced good performance.
- Within this range, the methods were relatively insensitive to random initialization.
- The learning curves rarely collapsed to zero or diverged once training had started successfully.

These results support the claim that asynchronous actor-learners provide a stabilizing effect for deep reinforcement learning.

---

## Overall Conclusion

The main experimental conclusions are:

1. Asynchronous actor-learners can stabilize deep RL without relying on experience replay.
2. A3C achieved strong average performance on Atari and exceeded several DQN-based methods.
3. It achieved these results using 16 CPU cores, without a GPU or a large distributed system.
4. A3C worked across discrete actions, continuous actions, visual inputs, recurrent memory, and 3D navigation.
5. Increasing the number of parallel workers substantially reduced training time and sometimes improved data efficiency.
6. A3C was not the best method by every metric on every game, but it showed particularly strong overall performance, robustness, and computational efficiency.


<br/>
# 예제



이 논문은 일반적인 지도학습처럼 **고정된 입력–정답 출력 데이터셋**을 사용하는 것이 아니라, 에이전트가 환경과 상호작용하면서 데이터를 직접 수집하는 **강화학습**을 다룹니다. 따라서 “트레이닝데이터”는 상태·행동·보상으로 이루어진 경험이며, “테스트데이터”는 학습에 사용하지 않은 에피소드나 새로운 환경에서의 수행 결과입니다.

### 1. 강화학습 데이터의 기본 형태

| 구분 | 내용 | 예시 |
|---|---|---|
| 입력 | 현재 상태 \(s_t\) | 게임 화면, 자동차 화면, 로봇의 물리 상태 |
| 에이전트 출력 | 행동 \(a_t\) 또는 행동별 가치 | 왼쪽 이동, 가속, 로봇 관절 토크 |
| 환경 출력 | 다음 상태 \(s_{t+1}\), 보상 \(r_t\) | 다음 화면, 점수 변화, 속도에 따른 보상 |
| 학습 목표 | 누적 보상 \(R_t\) 최대화 | 게임 점수, 주행 속도, 목표 도달 보상 |

예를 들어 Atari 게임에서 다음과 같은 상호작용이 반복됩니다.

```text
입력: 현재 게임 화면 s_t
출력: 행동 a_t = "공을 치기 위해 패들을 왼쪽으로 이동"
환경 반응: 다음 화면 s_{t+1}, 보상 r_t = 0
...
입력: 공을 성공적으로 맞힌 뒤의 화면
출력: 행동 = "오른쪽 이동"
환경 반응: 보상 r_t = +1
```

에이전트는 특정 화면에 대한 정답 행동을 제공받지 않습니다. 대신 어떤 행동이 장기적으로 더 많은 보상을 주는지를 학습합니다.

---

### 2. Atari 2600 게임

#### 트레이닝 입력과 출력

- **입력:** 현재 게임의 시각적 상태
  - 예: Pong의 화면, Breakout의 화면, Q*bert의 게임 장면
- **출력:** 게임에서 선택할 행동
  - 예: 왼쪽, 오른쪽, 발사, 점프 등
- **환경의 피드백:**
  - 다음 게임 화면
  - 게임 점수 변화에 해당하는 보상
  - 게임 종료 여부

Q-learning 계열에서는 신경망이 각 행동에 대한 \(Q(s,a)\) 값을 출력합니다.

```text
입력: 현재 Pong 화면
출력: [왼쪽: 0.8, 오른쪽: 1.2, 정지: 0.4]
선택: Q값이 가장 큰 "오른쪽"
```

A3C에서는 하나의 네트워크가 두 가지 출력을 냅니다.

1. **정책 출력:** 각 행동을 선택할 확률  
   예: `[왼쪽 0.2, 오른쪽 0.7, 정지 0.1]`
2. **가치 출력:** 현재 상태에서 앞으로 얻을 것으로 예상되는 누적 보상  
   예: \(V(s_t)=15.3\)

#### 테스트 및 평가

학습이 끝난 뒤에는 학습에 사용하지 않은 게임 플레이 에피소드에서 에이전트를 평가합니다. 논문은 Atari 57개 게임에서 A3C를 평가했으며, 사람이 시작한 상태에서의 성능도 비교했습니다.

- **테스트 입력:** 새로운 게임 에피소드의 화면
- **테스트 출력:** 에이전트가 선택한 행동들
- **평가 결과:** 최종 게임 점수 또는 인간 점수 대비 정규화된 점수

즉, 분류 문제의 “정답 라벨 정확도”가 아니라 **얼마나 높은 게임 점수를 얻는가**가 평가 기준입니다.

---

### 3. TORCS 자동차 경주

#### 구체적인 태스크

에이전트가 자동차를 조종하여 트랙을 빠르고 안정적으로 주행해야 합니다.

- **입력:** 현재 자동차의 RGB 이미지
- **출력:** 자동차 제어 행동
  - 조향
  - 가속
  - 제동 등
- **보상:** 자동차의 진행 속도와 트랙 중앙선에서의 위치를 반영한 값

예시:

```text
입력: 자동차가 커브에 진입한 화면
출력: "왼쪽 조향 + 속도 감소"
환경 반응: 자동차가 트랙 중앙에 가까워짐
보상: 양의 보상
```

반대로 자동차가 트랙을 벗어나거나 진행 방향과 반대로 움직이면 보상이 낮아질 수 있습니다.

논문에서는 느린 차와 빠른 차, 상대 차량의 유무를 조합한 네 가지 설정을 평가했습니다. A3C는 인간 플레이어 점수의 대략 75~90%에 도달했습니다.

- **트레이닝:** 다양한 트랙 주행 경험을 수집하며 정책 학습
- **테스트:** 학습 이후의 주행 에피소드에서 평균 주행 점수 측정
- **핵심 평가:** 얼마나 빠르고 안정적으로 트랙을 주행하는가

---

### 4. MuJoCo 연속 제어 태스크

#### 구체적인 태스크

MuJoCo에서는 로봇이나 물리적 물체를 제어합니다.

- 걷기
- 달리기
- 물체 조작
- 관절을 이용한 이동
- 접촉이 포함된 물리 제어

#### 입력과 출력

- **입력:**
  - 로봇의 관절 각도
  - 관절 속도
  - 몸체 위치와 속도
  - 또는 로봇을 촬영한 이미지
- **출력:**
  - 각 관절에 가할 연속적인 토크나 힘
  - 예: 왼쪽 다리 토크 \(0.35\), 오른쪽 다리 토크 \(-0.12\)
- **보상:**
  - 목표 방향으로 이동한 거리
  - 속도
  - 에너지 사용량
  - 자세 유지 정도 등

Atari처럼 “왼쪽/오른쪽” 중 하나를 선택하는 것이 아니라, 행동이 실수값일 수 있다는 점이 다릅니다. 논문에서는 A3C가 이러한 연속 행동 문제에도 적용되어 몇 시간에서 24시간 이내에 좋은 정책을 학습했다고 보고합니다.

---

### 5. Labyrinth 3D 미로

#### 구체적인 태스크

매 에피소드마다 새로운 3D 미로가 생성됩니다. 에이전트는 시각 정보만 보고 미로를 탐색하여 사과와 포털을 찾아야 합니다.

- 사과 획득: 보상 \(+1\)
- 포털 진입: 보상 \(+10\)
- 포털에 들어가면 새로운 위치에서 다시 시작
- 에피소드는 60초 후 종료

#### 입력과 출력

- **입력:** \(84 \times 84\) RGB 이미지
  - 현재 미로의 벽, 통로, 사과, 포털 등이 보임
- **출력:** 이동 또는 시점 조작 행동
  - 앞으로 이동
  - 왼쪽·오른쪽 회전
  - 기타 환경에서 제공하는 행동
- **환경 반응:** 다음 시점의 미로 이미지와 보상

예시:

```text
입력: 앞에 통로와 사과가 보이는 이미지
출력: 앞으로 이동
환경 반응: 사과 획득
보상: +1

입력: 포털 위치를 발견한 이미지
출력: 포털 방향으로 이동
환경 반응: 포털 진입 및 +10 보상
```

#### 테스트 방식

학습 중 보지 못한 새로운 무작위 미로에서도 포털을 찾는 전략을 수행하는지를 평가합니다. 따라서 단순히 특정 미로의 경로를 외우는 것이 아니라, **새로운 미로에 일반화된 탐색 전략**을 학습했는지가 중요합니다.

논문에서는 A3C LSTM 에이전트가 약 50점의 평균 점수를 얻어, 시각 입력만으로도 무작위 3D 미로를 탐색하는 합리적인 전략을 학습했다고 설명합니다.

---

### 6. 이 논문에서 트레이닝과 테스트의 차이

| 항목 | 트레이닝 | 테스트/평가 |
|---|---|---|
| 데이터 생성 | 에이전트가 환경에서 직접 수집 | 새로운 에피소드 또는 새로운 상태 사용 |
| 정답 라벨 | 없음 | 없음 |
| 학습 여부 | 네트워크 파라미터 업데이트 | 파라미터 고정 |
| 목적 | 누적 보상 최대화 | 실제 수행 점수 측정 |
| 예시 | 여러 Atari 게임에서 반복 플레이 | 새로운 게임 에피소드에서 최종 점수 평가 |
| Labyrinth | 여러 무작위 미로에서 탐색 학습 | 학습하지 않은 미로에서 포털 탐색 |

또한 이 논문의 핵심은 경험 재생 메모리를 사용하는 대신, 여러 개의 **비동기 actor-learner**가 서로 다른 환경에서 동시에 경험을 수집한다는 점입니다. 각 스레드는 자신의 환경에서 상태와 보상을 얻고, 공유된 신경망을 업데이트합니다. 여러 에이전트가 서로 다른 상태를 경험하므로 데이터의 시간적 상관이 줄어들고 학습이 안정화됩니다.

---




This paper studies **reinforcement learning**, not ordinary supervised learning. Therefore, it does not use a fixed dataset of input–label pairs. Instead, the agent generates training data by interacting with an environment. The “test data” consists of new episodes, unseen states, or newly generated environments used for evaluation.

### 1. Basic reinforcement-learning data format

| Component | Description | Example |
|---|---|---|
| Input | Current state \(s_t\) | Game image, car image, robot state |
| Agent output | Action \(a_t\) or action values | Move left, accelerate, joint torque |
| Environment output | Next state \(s_{t+1}\) and reward \(r_t\) | Next image, score change, movement reward |
| Learning objective | Maximize cumulative return | Game score, driving performance, task reward |

For example, in an Atari game:

```text
Input: Current game screen s_t
Output: Move the paddle to the left
Environment response: Next screen and reward r_t = 0
...
Input: Screen after successfully hitting the ball
Output: Move to the right
Environment response: Reward r_t = +1
```

There is no teacher providing the correct action. The agent learns which actions lead to higher long-term rewards.

---

### 2. Atari 2600 games

#### Training inputs and outputs

- **Input:** Visual state of the current game
  - Pong, Breakout, Q*bert, Beamrider, etc.
- **Output:** A game action
  - Move left, move right, fire, jump, and so on
- **Environment feedback:**
  - Next game screen
  - Reward based on score changes
  - Whether the episode has terminated

For Q-learning methods, the network outputs an action value \(Q(s,a)\) for each possible action.

```text
Input: Current Pong screen
Output: [left: 0.8, right: 1.2, no-op: 0.4]
Selected action: right
```

For A3C, the network produces:

1. **Policy output:** Probability of each action  
   Example: `[left: 0.2, right: 0.7, no-op: 0.1]`
2. **Value output:** Expected future cumulative reward  
   Example: \(V(s_t)=15.3\)

#### Testing and evaluation

After training, the agent is evaluated on new game episodes. The paper evaluates A3C on 57 Atari games and also uses a human-start evaluation protocol.

- **Test input:** A new game state or screen
- **Test output:** Sequence of selected actions
- **Evaluation metric:** Final game score or human-normalized score

Thus, performance is not measured by classification accuracy, but by **how much reward or game score the agent obtains**.

---

### 3. TORCS car racing

#### Task

The agent must control a car and drive quickly and stably around a race track.

- **Input:** Current RGB image of the car’s view
- **Output:** Driving actions
  - Steering
  - Acceleration
  - Braking
- **Reward:** Related to the car’s velocity along the center of the track

Example:

```text
Input: Image showing the car entering a left curve
Output: Steer left and reduce speed
Environment response: The car remains near the center of the track
Reward: Positive reward
```

The paper evaluates four configurations involving slow or fast cars and the presence or absence of opponent cars. A3C achieved roughly 75–90% of the score of a human tester.

- **Training:** Collect driving trajectories and update the policy
- **Testing:** Run new driving episodes and measure the average score
- **Main criterion:** Fast and stable driving

---

### 4. MuJoCo continuous-control tasks

#### Task

MuJoCo is used for physical control problems such as:

- Walking
- Running
- Locomotion
- Object manipulation
- Tasks with contact dynamics

#### Inputs and outputs

- **Input:**
  - Joint angles
  - Joint velocities
  - Body position and velocity
  - Or visual observations such as images
- **Output:**
  - Continuous torques or forces applied to the joints
  - Example: left-leg torque \(0.35\), right-leg torque \(-0.12\)
- **Reward:**
  - Forward movement
  - Speed
  - Energy efficiency
  - Maintaining a stable posture

Unlike Atari, the agent does not choose only from discrete actions such as “left” or “right.” Its actions can be real-valued continuous controls. The paper reports that A3C learned good policies within several hours to less than 24 hours.

---

### 5. Labyrinth 3D maze

#### Task

A new random 3D maze is generated at the beginning of each episode. The agent must explore the maze using visual input and find apples and portals.

- Collecting an apple: reward \(+1\)
- Entering a portal: reward \(+10\)
- After entering a portal, the agent is respawned at a new location
- The episode ends after 60 seconds

#### Inputs and outputs

- **Input:** An \(84 \times 84\) RGB image
  - Walls, corridors, apples, and portals may appear in the image
- **Output:** Navigation actions
  - Move forward
  - Turn left or right
  - Other actions available in the environment
- **Environment response:** Next visual observation and reward

Example:

```text
Input: Image showing a corridor and an apple
Output: Move forward
Environment response: Apple collected
Reward: +1

Input: Image showing the portal
Output: Move toward the portal
Environment response: Portal entered
Reward: +10
```

#### Testing

The agent is evaluated on new randomly generated mazes that were not seen during training. The goal is therefore not to memorize a particular route, but to learn a general exploration strategy. The paper reports that the A3C LSTM agent achieved an average score of approximately 50, indicating reasonable exploration behavior from visual input alone.

---

### 6. Difference between training and testing

| Aspect | Training | Testing/Evaluation |
|---|---|---|
| Data generation | The agent collects trajectories by interacting with the environment | New episodes or unseen environments are used |
| Labels | No target labels | No target labels |
| Parameter updates | Network parameters are updated | Parameters are fixed |
| Objective | Maximize cumulative reward | Measure actual task performance |
| Example | Repeatedly play Atari games | Evaluate final score on new episodes |
| Labyrinth | Learn exploration in random mazes | Find portals in previously unseen mazes |

The central idea of the paper is to replace experience replay with multiple **asynchronous actor-learners**. Each worker interacts with its own environment, collects states, actions, and rewards, and updates a shared neural network. Since different workers experience different states at the same time, their data are less correlated, which helps stabilize and accelerate training.

<br/>
# 요약

 
여러 에이전트가 병렬로 환경을 탐색하며 공유 신경망을 비동기적으로 업데이트하는 A3C 등 네 가지 강화학습 방법을 제안하고, 경험 재생 없이도 학습을 안정화했다.  
A3C는 16개 CPU 코어만으로 Atari 57개 게임에서 기존 방법보다 높은 성능을 절반의 학습 시간에 달성했으며, 병렬 스레드 수가 늘어날수록 학습 속도도 크게 향상되었다.  
또한 Atari 게임뿐 아니라 TORCS 자동차 경주, MuJoCo 연속 제어, 시각 입력을 이용한 무작위 3D 미로 탐색에서도 효과적인 정책을 학습했다.  



The paper proposes asynchronous methods such as A3C, in which multiple agents explore environments in parallel and asynchronously update a shared neural network, stabilizing learning without experience replay.  
Using only 16 CPU cores, A3C achieved higher performance than previous methods on 57 Atari games in half the training time, while increasing the number of parallel threads substantially accelerated learning.  
The method also learned effective policies in TORCS car racing, MuJoCo continuous-control tasks, and visual navigation through randomly generated 3D mazes.

<br/>
# 기타



논문에서 **그림·표·의사코드·부록에 해당하는 결과와 인사이트**를 중심으로 정리하면 다음과 같습니다.  
※ 제공된 본문에는 부록 전체가 포함되어 있지 않으므로, 본문에서 언급된 부록 내용까지만 설명합니다.

---

### 1. Figure 1 — Atari 학습 속도 비교

- **비교 대상:** DQN, 비동기 1-step Q-learning, 1-step Sarsa, n-step Q-learning, A3C
- **환경:** Atari 2600 게임 5개
- **하드웨어:**  
  - DQN: Nvidia K40 GPU 1개  
  - 비동기 방법: CPU 코어 16개

#### 결과

- 네 가지 비동기 알고리즘 모두 신경망 기반 Atari 에이전트를 성공적으로 학습했다.
- 비동기 방법들은 전반적으로 DQN보다 빠르게 학습했다.
- 특히 일부 게임에서는 학습 속도 차이가 크게 나타났다.
- n-step 방법은 일부 게임에서 1-step 방법보다 빠르게 학습했다.
- 전체적으로 **A3C가 세 가지 value-based 방법보다 가장 좋은 성능**을 보였다.

#### 인사이트

경험 재생 메모리 없이도 여러 actor-learner가 서로 다른 상태를 병렬로 경험하게 하면 데이터 상관성이 감소한다. 이 효과가 학습을 안정화하고, GPU가 아닌 멀티코어 CPU만으로도 빠른 학습을 가능하게 했다.

---

### 2. Table 1 — 57개 Atari 게임의 성능 비교

표는 57개 게임에서의 **인간 정규화 점수(human-normalized score)**의 평균과 중앙값을 보여준다.

| 방법 | 학습 시간 | 평균 | 중앙값 |
|---|---:|---:|---:|
| DQN | GPU 8일 | 121.9% | 47.5% |
| Gorila | 100대 머신, 4일 | 215.2% | 71.3% |
| Double DQN | GPU 8일 | 332.9% | 110.9% |
| Dueling Double DQN | GPU 8일 | 343.8% | 117.1% |
| Prioritized DQN | GPU 8일 | 463.6% | 127.6% |
| A3C, Feedforward | CPU 1일 | 344.1% | 68.2% |
| A3C, Feedforward | CPU 4일 | 496.8% | 116.6% |
| A3C, LSTM | CPU 4일 | **623.0%** | 112.6% |

#### 결과

- A3C LSTM은 평균 점수 **623.0%**로 가장 높았다.
- A3C Feedforward도 CPU 4일 학습으로 평균 **496.8%**를 달성해 기존 방법보다 높은 성능을 보였다.
- A3C는 기존 GPU 기반 방법보다 짧은 시간과 적은 하드웨어로 경쟁력 있는 결과를 냈다.
- A3C는 1일 만에 Dueling Double DQN의 평균 성능에 근접했다.
- LSTM A3C는 평균은 가장 높지만 중앙값은 Feedforward A3C 4일보다 약간 낮다. 이는 일부 게임에서 매우 높은 점수를 얻어 평균을 크게 끌어올렸음을 의미한다.

#### 인사이트

A3C의 장점은 단순히 평균 점수가 높은 것뿐 아니라 다음과 같다.

1. CPU만으로 학습 가능하다.
2. 짧은 시간 안에 높은 성능에 도달한다.
3. Feedforward 네트워크와 recurrent LSTM 모두 지원한다.
4. 부분 관측 또는 시간적 기억이 필요한 문제에서 LSTM이 유리할 수 있다.

다만 평균과 중앙값의 차이를 보면, 모든 게임에서 균일하게 강하다기보다는 일부 게임에서 특히 뛰어난 성능을 보였다고 해석할 수 있다.

---

### 3. Table 2 — 병렬 actor-learner 수에 따른 학습 속도 향상

| 방법 | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads |
|---|---:|---:|---:|---:|---:|
| 1-step Q | 1.0 | 3.0 | 6.3 | 13.3 | **24.1** |
| 1-step Sarsa | 1.0 | 2.8 | 5.9 | 13.1 | **22.1** |
| n-step Q | 1.0 | 2.7 | 5.9 | 10.7 | **17.2** |
| A3C | 1.0 | 2.1 | 3.7 | 6.9 | **12.5** |

#### 결과

- 모든 알고리즘에서 thread 수가 증가할수록 학습 속도가 크게 향상되었다.
- 16개 thread를 사용하면 모든 방법이 최소 약 10배 이상의 속도 향상을 보였다.
- 1-step Q-learning과 Sarsa는 각각 24.1배, 22.1배로 **이론적인 선형 증가보다 더 큰(superlinear) 속도 향상**을 보였다.
- A3C도 16개 thread에서 12.5배의 속도 향상을 보였다.

#### 인사이트

속도 향상은 단순히 여러 CPU가 동시에 계산했기 때문만은 아니다.

- 여러 actor가 서로 다른 상태를 탐색한다.
- 데이터의 시간적 상관성이 줄어든다.
- 특히 1-step 방법에서는 병렬성이 업데이트 편향을 줄여, 목표 점수에 도달하는 데 필요한 데이터 자체도 감소할 수 있다.

다만 thread 수가 증가해도 속도 향상이 완전히 선형적이지 않은 이유는 CPU 자원 경쟁, 비동기 업데이트 충돌, 환경 처리 비용 등의 오버헤드 때문이다.

---

### 4. Figure 2 — 학습 안정성과 하이퍼파라미터 강건성

- **대상:** A3C
- **실험:** 5개 Atari 게임에서 50개의 learning rate와 random initialization을 사용
- **표현:** 각 실험의 최종 점수를 scatter plot으로 표시

#### 결과

- 좋은 성능을 내는 learning rate의 범위가 비교적 넓었다.
- 해당 범위에서 random initialization이 달라도 대부분 좋은 점수를 얻었다.
- 좋은 learning rate 영역에서 점수가 0으로 무너지는 사례가 거의 없었다.

#### 인사이트

A3C는 특정 초기값이나 매우 정교한 learning rate 설정에만 의존하지 않는다. 즉, 학습이 발산하거나 갑자기 붕괴하는 현상이 비교적 적으며, **실제 적용에 필요한 튜닝 부담이 낮다**는 점을 보여준다.

---

### 5. Figure 3 — 데이터 효율성 비교

- **x축:** 전체 actor들이 사용한 training frames  
  - 한 epoch는 전체 thread를 합쳐 400만 frame
- **y축:** 평균 게임 점수
- **비교:** 서로 다른 수의 actor-learner를 사용한 비동기 방법

#### 결과

- 병렬 actor 수가 증가할수록 1-step Q-learning의 데이터 효율성이 좋아졌다.
- 즉, 단순히 더 빠르게 데이터를 처리한 것뿐 아니라, 같은 양의 경험으로도 더 높은 점수에 도달하는 경우가 있었다.
- Sarsa에서도 유사한 결과가 관찰되었으며, 상세 결과는 Supplementary Figure S5에 제시되었다.

#### 인사이트

일반적으로 병렬화는 wall-clock time만 줄일 것으로 예상할 수 있다. 하지만 이 결과는 병렬 actor가 탐색 데이터를 다양하게 만들고 업데이트 상관성을 낮추어, **학습에 필요한 경험의 양 자체를 줄일 수 있음**을 보여준다.

---

### 6. Figure 4 — 실제 시간 기준 학습 속도

- **x축:** 실제 학습 시간(hours)
- **y축:** 평균 점수
- **대상:** Atari 5개 게임
- **비교:** 1, 2, 4, 8, 16개의 actor-learner

#### 결과

- 모든 비동기 알고리즘에서 thread 수가 증가할수록 일정 점수에 더 빨리 도달했다.
- 16개 thread 설정이 대체로 가장 빠른 학습 곡선을 보였다.
- Figure 3가 “사용한 데이터 양”을 비교했다면, Figure 4는 “실제 걸린 시간”에서의 이득을 보여준다.

#### 인사이트

A3C 프레임워크는 이론적인 데이터 효율성뿐 아니라 실제 시스템의 wall-clock 성능도 개선한다. 따라서 환경과 모델 업데이트를 병렬화할 수 있는 경우, 실험 반복 시간을 크게 줄일 수 있다.

---

### 7. Algorithm 1 — 비동기 1-step Q-learning 의사코드

주요 흐름은 다음과 같다.

1. 각 thread가 독립적인 환경에서 상태를 관찰한다.
2. 공유된 네트워크로 행동을 선택한다.
3. 1-step Q-learning target을 계산한다.
4. 여러 step의 gradient를 누적한다.
5. 일정 주기마다 공유 파라미터에 비동기적으로 gradient를 적용한다.
6. 별도의 target network를 주기적으로 갱신한다.

#### 인사이트

이 구조는 DQN의 핵심 요소인 target network는 유지하면서, experience replay 대신 여러 thread의 병렬 경험을 사용한다. 또한 gradient를 매 step 즉시 적용하지 않고 누적하여 여러 thread가 서로의 업데이트를 지나치게 덮어쓰는 문제를 줄인다.

---

### 8. Supplementary Algorithm S1 — n-step Q-learning

부록의 n-step Q-learning은 **forward view** 방식으로 동작한다.

- 최대 \(t_{\max}\) step까지 행동을 수행한다.
- 여러 reward를 이용해 n-step return을 계산한다.
- 마지막 상태에는 1-step return, 그 이전 상태에는 2-step return 등 서로 다른 길이의 return을 적용한다.
- 계산된 gradient를 한 번에 업데이트한다.

#### 인사이트

n-step return은 한 번의 보상이 여러 이전 상태-행동 쌍에 빠르게 전달되도록 한다. 따라서 보상이 긴 시간 뒤에 나타나는 문제에서 1-step 방법보다 reward propagation이 빠를 수 있다.

저자들은 신경망, momentum 기반 최적화, backpropagation through time과 결합할 때 backward view보다 forward view가 구현하기 쉽다고 설명한다.

---

### 9. Supplementary Algorithm S2 — A3C

A3C는 다음 두 출력을 가진 네트워크를 사용한다.

- 정책 출력: \(\pi(a_t|s_t)\)
- 가치 출력: \(V(s_t)\)

일반적으로 두 출력층을 제외한 convolutional feature layer는 공유한다.

#### 주요 구성

- n-step return을 사용해 policy와 value function을 함께 업데이트한다.
- advantage는 다음 형태로 추정된다.

\[
A_t =
\sum_{i=0}^{k-1}\gamma^i r_{t+i}
+\gamma^k V(s_{t+k})
-V(s_t)
\]

- 정책 gradient에는 entropy regularization도 추가한다.

#### 인사이트

A3C는 가치 함수만 학습하는 것이 아니라 정책을 직접 개선한다. 또한 entropy 항을 통해 정책이 너무 일찍 결정적인 행동 하나로 수렴하는 것을 막아 탐색을 유지한다. 이 때문에 Atari뿐 아니라 연속 행동 제어와 시각 기반 3D 미로에도 적용할 수 있었다.

---

### 10. Supplementary Section 1 — 최적화 알고리즘 비교

논문은 다음 세 가지 최적화 방법을 비교했다.

1. Momentum SGD
2. thread 간 통계량을 공유하지 않는 RMSProp
3. **thread 간 통계량을 공유하는 RMSProp**

#### 결과

- 공유 통계량을 사용하는 RMSProp이 다른 두 방법보다 상당히 강건했다.
- 비동기 thread들이 동일한 RMSProp 통계량을 공유하는 방식이 안정적인 학습에 도움이 되었다.

#### 인사이트

비동기 학습에서는 각 thread가 서로 다른 gradient를 계산하므로, optimizer의 상태를 어떻게 관리하는지가 중요하다. 단순히 모델 파라미터만 공유하는 것이 아니라 optimizer 통계량도 공유하는 것이 성능과 안정성에 영향을 준다.

---

### 11. Supplementary Figure S1 / Table S1

본문에서 Supplementary Table S1은 57개 Atari 게임 각각의 raw score를 제공한다고 설명한다.

#### 역할

- Table 1의 평균·중앙값을 구성하는 개별 게임 결과를 확인할 수 있다.
- A3C가 특정 게임에서 특히 강한지, 또는 일부 게임에서 약한지를 분석할 수 있다.
- 평균 점수만으로는 알기 어려운 게임별 편차를 확인하는 자료다.

제공된 본문에는 실제 raw score 표가 포함되어 있지 않으므로, 구체적인 게임별 수치는 여기서 확인할 수 없다.

---

### 12. Supplementary Figure S2 — TORCS 결과

본문에 따르면 네 가지 비동기 방법을 TORCS에서 비교했다.

- 느린 차량 / opponent 없음
- 느린 차량 / opponent 있음
- 빠른 차량 / opponent 없음
- 빠른 차량 / opponent 있음

#### 결과

- A3C가 네 가지 설정 모두에서 가장 좋은 성능을 보였다.
- 약 12시간 학습 후 인간 점수의 대략 75~90%에 도달했다.

#### 인사이트

A3C의 장점은 Atari와 같은 이산 행동 게임에만 한정되지 않는다. 시각 입력과 차량 dynamics가 함께 필요한 더 복잡한 제어 문제에서도 정책 기반 학습이 효과적임을 보여준다.

---

### 13. Supplementary Figures S5와 S6 — Sarsa의 추가 결과

- **Figure S5:** Sarsa의 데이터 효율성
- **Figure S6:** Sarsa의 wall-clock 학습 속도

본문은 Sarsa도 다른 비동기 방법과 마찬가지로 병렬 actor 수 증가에 따라 데이터 효율성과 학습 속도가 개선된다고 설명한다.

#### 인사이트

비동기 병렬화의 효과는 A3C나 Q-learning에만 해당하지 않는다. on-policy 방법인 Sarsa에도 적용되므로, experience replay 없이도 여러 종류의 RL 알고리즘을 안정적으로 학습시킬 수 있다.

---

### 14. Supplementary Figure S7 — 다른 알고리즘의 안정성

Figure 2는 A3C의 강건성을 보여주며, Supplementary Figure S7은 다음 세 방법에 대한 유사한 결과를 제공한다.

- 1-step Q-learning
- 1-step Sarsa
- n-step Q-learning

#### 결과 및 인사이트

네 알고리즘 모두 적절한 learning rate 영역에서 안정적으로 학습되었다. 따라서 병렬 actor-learner 구조는 policy-based A3C뿐 아니라 value-based 방법의 학습 안정성도 높인다.

---

## 전체적으로 얻을 수 있는 핵심 인사이트

1. **경험 재생 없이도 안정적인 deep RL이 가능하다.**  
   여러 actor가 서로 다른 데이터를 수집함으로써 경험 재생의 decorrelation 효과를 일부 대체한다.

2. **비동기 병렬화는 학습 속도와 안정성을 동시에 개선한다.**  
   단순한 계산 병렬화 이상의 효과가 있으며, 경우에 따라 데이터 효율성도 좋아진다.

3. **A3C가 가장 범용적인 방법으로 나타났다.**  
   Atari, TORCS, MuJoCo의 연속 제어, 시각 기반 Labyrinth까지 적용되었다.

4. **n-step return은 보상 전파를 빠르게 한다.**  
   긴 시간 뒤에 얻은 보상을 여러 이전 상태에 직접 전달할 수 있다.

5. **A3C의 entropy regularization과 LSTM은 탐색 및 부분 관측 문제에 유리하다.**

6. **다만 병렬 thread 수가 늘어난다고 항상 완벽한 선형 가속이 발생하는 것은 아니다.**  
   시스템 오버헤드와 비동기 업데이트 충돌 때문에 알고리즘별로 scaling 정도가 다르다.

---




## Results and Insights from Figures, Tables, Pseudocode, and Supplementary Material

The following summarizes the paper’s results mainly from its figures, tables, algorithms, and supplementary references.  
The full appendix is not included in the provided text, so the supplementary discussion is limited to what the main paper explicitly describes.

---

### 1. Figure 1 — Atari Learning Speed

- **Compared methods:** DQN, asynchronous 1-step Q-learning, 1-step Sarsa, n-step Q-learning, and A3C
- **Environment:** Five Atari 2600 games
- **Hardware:**  
  - DQN: one Nvidia K40 GPU  
  - Asynchronous methods: 16 CPU cores

#### Results

- All four asynchronous algorithms successfully trained neural-network Atari agents.
- They generally learned faster than DQN.
- n-step methods were faster than one-step methods on some games.
- Overall, **A3C performed best among the proposed methods**.

#### Insight

Parallel actor-learners collect experience from different parts of the environment. This reduces temporal correlation in the training data and provides a stabilizing effect similar to experience replay, while using only a multi-core CPU.

---

### 2. Table 1 — Performance on 57 Atari Games

The table reports mean and median human-normalized scores.

| Method | Training time | Mean | Median |
|---|---:|---:|---:|
| DQN | 8 days on GPU | 121.9% | 47.5% |
| Gorila | 4 days, 100 machines | 215.2% | 71.3% |
| Double DQN | 8 days on GPU | 332.9% | 110.9% |
| Dueling Double DQN | 8 days on GPU | 343.8% | 117.1% |
| Prioritized DQN | 8 days on GPU | 463.6% | 127.6% |
| A3C, Feedforward | 1 day on CPU | 344.1% | 68.2% |
| A3C, Feedforward | 4 days on CPU | 496.8% | 116.6% |
| A3C, LSTM | 4 days on CPU | **623.0%** | 112.6% |

#### Results

- A3C with LSTM achieved the highest mean score: **623.0%**.
- Feedforward A3C trained for four days reached a mean score of 496.8%.
- A3C achieved strong performance using only 16 CPU cores and no GPU.
- After only one day, A3C was already close to the average performance of Dueling Double DQN.
- The LSTM version had the highest mean but a slightly lower median than four-day feedforward A3C, suggesting that it achieved extremely high scores on some games.

#### Insight

A3C is attractive because it is computationally efficient, works with both feedforward and recurrent networks, and can handle tasks that require temporal memory. However, the difference between mean and median indicates that performance was not perfectly uniform across all games.

---

### 3. Table 2 — Speedup from More Actor-Learners

| Method | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads |
|---|---:|---:|---:|---:|---:|
| 1-step Q | 1.0 | 3.0 | 6.3 | 13.3 | **24.1** |
| 1-step Sarsa | 1.0 | 2.8 | 5.9 | 13.1 | **22.1** |
| n-step Q | 1.0 | 2.7 | 5.9 | 10.7 | **17.2** |
| A3C | 1.0 | 2.1 | 3.7 | 6.9 | **12.5** |

#### Results

- All methods benefited substantially from additional threads.
- With 16 threads, every method achieved at least roughly an order-of-magnitude speedup.
- One-step Q-learning and Sarsa showed superlinear speedups of 24.1× and 22.1×.
- A3C achieved a 12.5× speedup with 16 threads.

#### Insight

The gains are not purely computational. Parallel agents also improve exploration diversity and reduce correlation among updates. For one-step methods, this can reduce the amount of experience needed to reach a target score.

Perfect linear scaling is not achieved because of CPU contention, asynchronous update conflicts, and environment-processing overhead.

---

### 4. Figure 2 — Robustness and Stability

- **Method:** A3C
- **Experiment:** 50 learning rates and random initializations on five Atari games
- **Visualization:** Scatter plots of final scores

#### Results

- There was a relatively broad range of learning rates that produced good performance.
- Performance was generally robust to different random initializations.
- Very few runs collapsed to a score of zero in regions with otherwise good learning rates.

#### Insight

A3C does not depend heavily on a single carefully chosen learning rate or initialization. This indicates good practical robustness and a low probability of catastrophic divergence once learning begins.

---

### 5. Figure 3 — Data Efficiency

- **x-axis:** Total training frames collected across all actors  
  - One epoch corresponds to four million frames across all threads
- **y-axis:** Average score

#### Results

- Increasing the number of parallel actors improved the data efficiency of one-step Q-learning.
- More workers sometimes allowed the agent to reach the same score using fewer total frames.
- Similar Sarsa results are shown in Supplementary Figure S5.

#### Insight

Parallelism improves more than wall-clock speed. Diverse experiences and less-correlated updates can reduce the amount of data required for learning.

---

### 6. Figure 4 — Wall-Clock Training Speed

- **x-axis:** Actual training time in hours
- **y-axis:** Average score
- **Compared settings:** 1, 2, 4, 8, and 16 actor-learners

#### Results

- All asynchronous methods learned faster in real time when more threads were used.
- The 16-thread configurations generally produced the fastest learning curves.

#### Insight

Figure 3 demonstrates data-efficiency improvements, while Figure 4 demonstrates practical time savings. Together, they show that asynchronous learning can improve both sample usage and wall-clock training time.

---

### 7. Algorithm 1 — Asynchronous One-Step Q-Learning

The main procedure is:

1. Each thread interacts with its own environment.
2. It selects actions using the shared network.
3. It computes a one-step Q-learning target.
4. It accumulates gradients over several steps.
5. It asynchronously applies the gradients to the shared parameters.
6. It periodically updates a target network.

#### Insight

The method retains the target-network idea from DQN but replaces experience replay with parallel experience collection. Gradient accumulation also reduces the chance that concurrent workers will overwrite each other’s updates too aggressively.

---

### 8. Supplementary Algorithm S1 — n-step Q-Learning

The supplementary n-step method uses a **forward-view** implementation.

- The agent acts for up to \(t_{\max}\) steps.
- It computes multi-step returns using several rewards.
- Different earlier states receive returns of different lengths.
- The accumulated gradients are applied together.

#### Insight

n-step returns propagate rewards more quickly because a reward can directly influence multiple preceding state-action pairs. This is especially useful when rewards are delayed.

The authors also found the forward view easier to combine with neural networks, momentum-based optimization, and backpropagation through time than a backward-view implementation.

---

### 9. Supplementary Algorithm S2 — A3C

A3C uses a network with two outputs:

- Policy output: \(\pi(a_t|s_t)\)
- Value output: \(V(s_t)\)

The lower convolutional layers are generally shared.

#### Main components

- The policy and value function are updated using n-step returns.
- The advantage is estimated as

\[
A_t =
\sum_{i=0}^{k-1}\gamma^i r_{t+i}
+\gamma^k V(s_{t+k})
-V(s_t)
\]

- An entropy regularization term is added to the policy objective.

#### Insight

A3C directly improves the policy while simultaneously learning a value baseline. Entropy regularization prevents premature convergence to a deterministic policy and encourages continued exploration. This helps explain its applicability to Atari, continuous control, and visual 3D navigation.

---

### 10. Supplementary Section 1 — Optimization Comparison

The paper compares:

1. Momentum SGD
2. RMSProp without shared statistics
3. **RMSProp with shared statistics across threads**

#### Result and insight

RMSProp with shared statistics was substantially more robust. In asynchronous learning, not only the network parameters but also the optimizer statistics can affect stability. Sharing these statistics across workers helped coordinate updates.

---

### 11. Supplementary Table S1 — Per-Game Atari Scores

The paper states that Supplementary Table S1 reports the raw score for each of the 57 Atari games.

#### Purpose

- It provides the individual game scores underlying Table 1.
- It shows whether A3C is particularly strong or weak on specific games.
- It reveals game-to-game variation that is hidden by the mean and median.

The actual raw-score table is not included in the provided text, so specific per-game values cannot be listed here.

---

### 12. Supplementary Figure S2 — TORCS Results

The four settings were:

- Slow car without opponents
- Slow car with opponents
- Fast car without opponents
- Fast car with opponents

#### Results

- A3C was the best-performing method in all four configurations.
- After approximately 12 hours, it reached roughly 75–90% of the score achieved by a human tester.

#### Insight

A3C is not limited to discrete Atari actions. It also works in visually rich control tasks where the agent must learn vehicle dynamics and react to opponents.

---

### 13. Supplementary Figures S5 and S6 — Additional Sarsa Results

- **Figure S5:** Sarsa data efficiency
- **Figure S6:** Sarsa wall-clock training speed

The paper indicates that Sarsa also benefits from increasing the number of parallel actor-learners.

#### Insight

The advantages of asynchronous parallelism extend beyond A3C and Q-learning. Even an on-policy algorithm such as Sarsa can be trained effectively without experience replay.

---

### 14. Supplementary Figure S7 — Stability of Other Algorithms

Figure S7 provides robustness plots similar to Figure 2 for:

- 1-step Q-learning
- 1-step Sarsa
- n-step Q-learning

#### Result and insight

All four asynchronous algorithms showed stable learning over suitable learning-rate ranges. Thus, the stabilizing effect of parallel actor-learners applies to both policy-based and value-based methods.

---

## Overall Takeaways

1. **Deep RL can be trained stably without experience replay** when multiple actors generate diverse, less-correlated experience.
2. **Asynchronous parallelism improves both stability and training speed**, and can sometimes improve data efficiency as well.
3. **A3C is the most general method in the paper**, working on Atari, TORCS, MuJoCo continuous control, and visual 3D mazes.
4. **n-step returns accelerate reward propagation** through earlier state-action pairs.
5. **Entropy regularization and LSTM memory improve exploration and temporal reasoning.**
6. More threads do not guarantee perfectly linear scaling because of synchronization overhead, resource contention, and asynchronous update conflicts.

<br/>
# refer format:
### BibTeX    

```bibtex
@inproceedings{mnih2016asynchronous,
  author    = {Mnih, Volodymyr and Puigdomènech Badia, Adrià and Mirza, Mehdi
               and Graves, Alex and Harley, Tim and Lillicrap, Timothy P.
               and Silver, David and Kavukcuoglu, Koray},
  title     = {Asynchronous Methods for Deep Reinforcement Learning},
  booktitle = {Proceedings of the 33rd International Conference on Machine Learning},
  editor    = {Balcan, Maria-Florina and Weinberger, Kilian Q.},
  series    = {Proceedings of Machine Learning Research},
  volume    = {48},
  pages     = {1928--1937},
  year      = {2016},
  publisher = {PMLR},
  address   = {New York, NY, USA},
  url       = {https://proceedings.mlr.press/v48/mniha16.html}
}
```

### Chicago 스타일   

Mnih, Volodymyr, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Tim Harley, Timothy P. Lillicrap, David Silver, and Koray Kavukcuoglu. “Asynchronous Methods for Deep Reinforcement Learning.” In *Proceedings of the 33rd International Conference on Machine Learning*, edited by Maria-Florina Balcan and Kilian Q. Weinberger, 1928–1937. Vol. 48 of *Proceedings of Machine Learning Research*. New York, NY: PMLR, 2016. https://proceedings.mlr.press/v48/mniha16.html.


