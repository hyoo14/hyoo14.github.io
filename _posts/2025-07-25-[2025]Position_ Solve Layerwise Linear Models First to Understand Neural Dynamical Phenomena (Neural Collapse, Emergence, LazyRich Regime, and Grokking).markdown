---
layout: post
title:  "[2025]Position: Solve Layerwise Linear Models First to Understand Neural Dynamical Phenomena (Neural Collapse, Emergence, Lazy/Rich Regime, and Grokking)"  
date:   2025-07-25 14:23:40 +0900
categories: study
---

{% highlight ruby %}


한줄 요약: 

딥러닝의 4가지 현상을 층별 선형 모델로 설명(dynamical feedback principle)   
또한 이를 통한 현상 완화의 가능성 제시   



짧은 요약(Abstract) :    


물리학에서는 복잡한 시스템을 최소한의 핵심 원리만 담은 해석 가능한 모델로 단순화하는 경우가 많습니다. 이 논문에서는 딥러닝에서도 유사한 접근을 제안하며, 복잡한 비선형 딥러닝 모델을 분석하기 위해 **layerwise linear model (층별 선형 모델)**을 먼저 해결할 것을 주장합니다. 이러한 선형 모델은 각 층의 동적 상호작용 원리를 담고 있으며, 이를 통해 신경망에서 나타나는 다양한 현상(예: Neural Collapse, Emergence, Lazy/Rich Regime, Grokking 등)을 설명할 수 있습니다. 저자들은 **dynamical feedback principle (동적 피드백 원리)**을 제안하며, 이 원리를 통해 비선형 요소 없이도 주요 딥러닝 동작을 이해할 수 있다고 주장합니다. 이러한 선형 모델은 수학적으로 해석 가능하고, 신경망 과학의 발전을 가속화할 수 있는 도구로 제시됩니다.



In physics, complex systems are often simplified into minimal, solvable models that retain only the core principles. In machine learning, layerwise linear models (e.g., linear neural networks) act as simplified representations of neural network dynamics. These models follow the dynamical feedback principle, which describes how layers mutually govern and amplify each other’s evolution. This principle extends beyond the simplified models, successfully explaining a wide range of dynamical phenomena in deep neural networks, including neural collapse, emergence, lazy and rich regimes, and grokking. In this position paper, we call for the use of layerwise linear models retaining the core principles of neural dynamical phenomena to accelerate the science of deep learning.





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



이 논문에서는 복잡한 딥러닝 현상을 해석하기 위해 층별 선형 모델(layerwise linear models) 을 사용합니다. 여기서 선형 모델은 입력과 출력 사이의 관계가 선형인 모델이지만, 각 층을 나누어 층간 곱(product of parameters) 으로 구성되는 구조를 갖습니다. 예를 들어, 
𝑓
(
𝑥
)
=
𝑥
⊤
𝑊
1
𝑊
2
f(x)=x 
⊤
 W 
1
​
 W 
2
​
  같은 2-layer linear neural network나 
𝑓
(
𝑥
)
=
∑
𝑖
𝑥
𝑖
𝑎
𝑖
𝑏
𝑖
f(x)=∑ 
i
​
 x 
i
​
 a 
i
​
 b 
i
​
  형태의 대각선형(diagonal) 모델이 사용됩니다.

논문은 이러한 구조에서 발생하는 **동적 피드백 원리(dynamical feedback principle)**를 중심으로 분석합니다. 이 원리는 한 층의 파라미터 크기가 다른 층의 변화 속도에 영향을 주며, 이로 인해 각 층이 서로를 증폭하는 비선형적 동작을 만들어냅니다. 이러한 원리를 통해 emergence, grokking, neural collapse 등 다양한 현상이 설명됩니다.

또한, 논문에서는 다양한 초기화 조건(작은 초기화, 계층 간 불균형 등)이나 목표 함수 크기(target scale)에 따라 발생하는 **lazy regime (선형적 학습)**과 rich regime (비선형적 특성 학습) 의 전이를 해석합니다. 이 과정을 통해 복잡한 딥러닝 현상을 단순화된 수학적 모델로 설명할 수 있다는 것이 주요 기법입니다.




This paper employs layerwise linear models—such as two-layer linear neural networks and diagonal linear networks—to analyze and understand deep learning dynamics. These models maintain a layerwise multiplicative structure, where the output is a product of parameters across layers (e.g., 
𝑓
(
𝑥
)
=
𝑥
⊤
𝑊
1
𝑊
2
f(x)=x 
⊤
 W 
1
​
 W 
2
​
  or 
𝑓
(
𝑥
)
=
∑
𝑖
𝑥
𝑖
𝑎
𝑖
𝑏
𝑖
f(x)=∑ 
i
​
 x 
i
​
 a 
i
​
 b 
i
​
 ). Although they lack non-linear activations, the training dynamics are inherently non-linear due to inter-layer interactions.

A central methodological contribution is the introduction of the dynamical feedback principle, which describes how the magnitude of one layer’s parameters governs the rate of change in another layer. This principle leads to amplifying feedback dynamics that are used to explain diverse deep learning phenomena such as emergence, grokking, neural collapse, and the lazy/rich regime transition.

The models are analyzed under various initial conditions (e.g., small initialization, imbalanced layers, target scaling), and are often exactly solvable, allowing clear mathematical insight into training behavior. By focusing on the dynamics of these simplified but expressive models, the authors demonstrate that much of deep learning’s complexity can be explained without relying on non-linearity.




   
 
<br/>
# Results  



이 논문은 기존의 복잡한 딥러닝 모델들과 달리, 층별 선형 모델(layerwise linear models) 을 통해 다음과 같은 여러 현상을 수학적으로 해석 가능하게 만들고, 실험적으로도 유사한 동작을 보임을 확인합니다.

Emergence (돌발적 성능 향상): 다층 선형 모델에서 각 feature 또는 skill이 시그모이드 형태의 성장 곡선을 따라 학습되며, 학습 순서가 feature의 분산(E[x²])이나 빈도에 따라 달라짐을 보였습니다. 이로 인해 특정 시점에서 갑작스러운 성능 향상(emergence) 이 발생합니다. 이는 실제 대형 언어 모델에서도 관측된 현상과 유사합니다.

Neural Collapse (신경 붕괴): 분류 문제에서 학습이 진행됨에 따라 마지막 레이어의 피처들이 각 클래스별 평균 중심으로 수렴하고, 정규화된 단순 구조(simplex ETF) 를 이루는 현상이 선형 모델에서도 수학적으로 도출됩니다. 기존의 ResNet18 등 실제 모델에서도 동일한 구조가 나타나는 것과 일치합니다.

Lazy vs. Rich Regime: 초기화 조건이나 weight-to-target 비율에 따라 선형적 동역학(lazy) 또는 비선형적 특성 학습(rich) 으로 나뉘며, rich regime에서는 특징 학습(feature learning)이 더욱 활발히 일어납니다. 이 변화는 grokking (지연된 일반화) 현상과도 연결되어, weight 초기화나 타겟 크기를 조정함으로써 grokking 없이 빠른 일반화를 유도할 수 있음을 실험적으로 보였습니다.

비교 모델: 대부분의 결과는 2-layer ReLU 신경망이나 MLP(Multilayer Perceptron)와 비교되었으며, 실제 비선형 모델이 보이는 emergence, neural collapse, grokking 등의 현상이 선형 모델로도 정확히 예측 가능함을 보여주었습니다.

메트릭: 테스트 정확도, 학습 곡선, feature 분산, rank 추정, correlation (skill strength), NTK distance 등을 메트릭으로 사용했습니다.




This paper demonstrates that layerwise linear models, despite their simplicity, can effectively reproduce and explain several complex phenomena observed in modern deep neural networks:

Emergence: The model exhibits sigmoidal saturation in learning different features, leading to abrupt performance jumps when certain feature variances or frequencies are reached. This behavior matches empirical observations in large language models.

Neural Collapse: The model shows that, over training, the final layer features converge to class-specific mean vectors forming a simplex equiangular tight frame (ETF)—a phenomenon observed in models like ResNet18 trained on CIFAR-10. This behavior is mathematically derived in the linear setting.

Lazy vs. Rich Regimes: By varying initialization imbalance or weight-to-target ratios, the model transitions between lazy (linear) dynamics and rich (feature-learning) dynamics. Notably, entering the rich regime removes the delayed generalization phase (grokking), enabling faster learning.

Comparative Models: The linear models were compared against nonlinear neural networks such as 2-layer ReLU models and 4-layer tanh MLPs. In all cases, the layerwise linear models accurately predicted the empirical trends of the nonlinear networks.

Evaluation Metrics: Key metrics included test accuracy, learning curves, feature rank, correlation metrics (skill strength), and NTK kernel distance, showing that linear models can quantitatively align with complex network behavior.





<br/>
# 예제  



이 논문에서는 다양한 딥러닝 현상(emergence, grokking, neural collapse 등)을 설명하기 위해 단순화된 예제 데이터와 테스크를 활용합니다.

Emergence 실험 예시

입력 (Input): 각 feature가 power-law 분포를 따르는 **다중 패리티 함수 (multitask sparse parity)**에서 나온 pre-defined skill functions 
𝑔
𝑘
(
𝑥
)
g 
k
​
 (x)

출력 (Output): 특정 skill function들의 선형 결합으로 구성된 목표 함수 
𝑓
∗
(
𝑥
)
=
∑
𝑆
𝑘
𝑔
𝑘
(
𝑥
)
f 
∗
 (x)=∑S 
k
​
 g 
k
​
 (x)

테스크 (Task): 어떤 skill이 더 자주 등장하는지에 따라 모델이 그 skill을 먼저 학습하고, 적은 빈도의 skill은 나중에 학습함 → 시간, 데이터, 파라미터 증가에 따라 돌발적 성능 향상(emergence) 발생

Grokking 실험 예시

입력 (Input): MNIST 이미지 중 1000개 샘플 사용, 4-layer tanh MLP 모델 학습

출력 (Output): 숫자 클래스 (0~9)

테스크 (Task): 작은 데이터로 훈련할 때 일반화가 지연되다가 특정 시점 이후에 갑자기 test accuracy가 상승 → grokking 현상

조절 실험: 초기 weight 크기, target 값 크기, 입력 스케일 등을 조절하여 grokking을 제거하고 빠른 일반화를 유도

Neural Collapse 예시

입력 (Input): 클래스별 입력 데이터 
𝑥
x

출력 (Output): 클래스별 one-hot label

테스크 (Task): 훈련이 진행되며 마지막 피처 벡터들이 클래스별 평균으로 모이고, 서로 직교하는 simplex ETF 구조를 형성함

이러한 예시들은 모두 layerwise linear model로 수학적으로 기술 가능하며, ReLU MLP 등 실제 모델과 매우 유사한 동작을 보임을 실험으로 증명합니다.




The paper uses concrete example tasks and datasets to demonstrate that layerwise linear models can replicate complex neural phenomena:

Emergence Example

Input: Predefined skill functions 
𝑔
𝑘
(
𝑥
)
g 
k
​
 (x) derived from a multitask sparse parity problem, with features following a power-law distribution

Output: Target function 
𝑓
∗
(
𝑥
)
=
∑
𝑆
𝑘
𝑔
𝑘
(
𝑥
)
f 
∗
 (x)=∑S 
k
​
 g 
k
​
 (x) as a weighted sum of skills

Task: The model learns frequently occurring skills earlier and rare ones later, leading to abrupt performance improvements (emergence) as time, data, or parameters increase

Grokking Example

Input: 1000 MNIST digit images used to train a 4-layer tanh MLP

Output: Digit classification labels (0–9)

Task: The model overfits training data but delays generalization, suddenly improving test accuracy after many epochs (grokking)

Intervention: By adjusting initial weight scale, target scaling, or input magnitude, the authors eliminate grokking and achieve early generalization

Neural Collapse Example

Input: Class-specific input vectors

Output: One-hot encoded class labels

Task: As training progresses, final-layer feature vectors collapse to class means forming a simplex ETF structure, matching phenomena observed in networks like ResNet18

These examples are all mathematically modeled using layerwise linear networks, and their behavior closely aligns with that of more complex ReLU-based networks in practice.




<br/>  
# 요약   



이 논문은 복잡한 신경망 현상을 해석하기 위해 **층별 선형 모델(layerwise linear models)**과 **동적 피드백 원리(dynamical feedback principle)**를 제안합니다. 이 모델은 emergence, neural collapse, lazy/rich regime, grokking 등 다양한 현상을 수학적으로 정확히 설명하고, 실제 비선형 신경망과 유사한 실험 결과를 보여줍니다. 예시로는 패리티 기반 다중 과제 학습, MNIST 숫자 분류, 클래스별 피처 붕괴 등이 사용되며, 선형 모델로도 이러한 현상이 재현됨을 입증합니다.




This paper proposes layerwise linear models and the dynamical feedback principle to analyze complex neural network phenomena. These models successfully explain a range of behaviors—including emergence, neural collapse, lazy/rich regimes, and grokking—with exact mathematical solutions and empirical consistency with nonlinear networks. Example tasks include multitask parity learning, MNIST classification, and class-wise feature collapse, all of which are reproduced using linear models.



<br/>  
# 기타  



Figure 1 (전체 구조도)

논문의 흐름을 시각적으로 요약한 다이어그램으로, 각 현상이 어떤 초기 조건과 동적 원리로부터 파생되는지를 색상(초록–핵심 원리, 노랑–조건, 파랑–수학적 성질, 빨강–실제 현상)으로 구분하여 정리

인사이트: **단일한 원리(피드백 동역학)**가 다양한 현상을 관통한다는 것을 강조

Figure 3 (선형 vs. 층별 선형 모델의 학습 곡선 비교)

동일한 입력 분산 조건에서 선형 모델은 빠르게 포화되는 반면, 층별 선형 모델은 지연된 시그모이드 학습을 보여 emergence를 설명

인사이트: 시그모이드 학습과 모드 간 시간차가 돌발적 성능 향상과 관련 있음

Figure 4 (스킬 학습 곡선)

multitask parity 문제에서 각 skill이 시간, 데이터, 파라미터의 증가에 따라 어떤 시점에서 급격히 학습되는지를 시각화

인사이트: emergence가 계단식으로 일어나는 이유를 수학적으로 예측 가능함을 보여줌

Figure 5 (Neural Collapse 구조 시각화)

마지막 레이어 피처들이 클래스 평균을 중심으로 simplex ETF 형태로 수렴하는 구조를 도식화

인사이트: 이 단순한 기하 구조가 학습된 피처 분산의 최소화 및 일반화와 연결됨

Figure 6 & 7 (Lazy/Rich Regime 및 Grokking 시각화)

Layer imbalance 및 weight-to-target ratio를 조절하면서 NTK 변화나 학습 속도의 차이를 시각화

인사이트: 작은 초기화 또는 큰 타겟 스케일이 피드백 효과를 유도하여 feature learning을 강화함 → grokking 제거 가능

Figure 8 (MNIST에서 grokking 제거 실험)

weight/downscaling/input-scaling 등 다양한 기법이 grokking 지연 없이 일반화를 유도함을 학습곡선으로 보여줌

인사이트: 초기 조건만으로도 rich regime을 유도하여 grokking을 제어할 수 있음

Appendix A–H

모든 주요 수식(예: 시그모이드 학습곡선, 보존량, θ 변화 등)의 수학적 유도 과정 포함

인사이트: 실험이 아닌 이론적 모델링만으로 복잡한 현상을 예측 가능함을 보여주는 핵심 뒷받침 자료





Figure 1 (Paper Roadmap Diagram)

A color-coded diagram summarizes how each phenomenon arises from specific initial conditions and a common dynamical feedback principle

Insight: Reinforces the central claim that a single unified principle explains diverse neural behaviors

Figure 3 (Comparison of Dynamics)

Shows learning curves of linear vs. diagonal linear networks: linear models saturate quickly, whereas layerwise models show delayed sigmoidal growth

Insight: Highlights how delayed saturation of modes leads to emergence

Figure 4 (Skill Emergence Curves)

Tracks how individual skills in a multitask parity problem emerge abruptly as training time, data, or parameter count increases

Insight: Demonstrates that emergence is mathematically predictable and staged

Figure 5 (Neural Collapse Geometry)

Illustrates how final-layer features converge to class means forming a simplex ETF structure

Insight: Shows geometric organization of features that supports generalization

Figures 6 & 7 (Lazy/Rich Regime and Grokking Visualization)

Visualizes how layer imbalance or weight-to-target ratios affect training dynamics (e.g., NTK distances, learning speed)

Insight: Amplifying feedback induced by certain initializations leads to rich, feature-learning regimes and eliminates grokking delays

Figure 8 (Grokking Removal on MNIST)

Shows that techniques like weight or target scaling allow fast generalization without grokking on a 4-layer MLP

Insight: Proper initialization alone can place the model in a rich regime, avoiding overfitting phases

Appendices A–H

Contain full mathematical derivations for all key dynamics: sigmoidal learning, conservation laws, rank constraints, etc.

Insight: These provide theoretical rigor to show that complex DNN behavior can be captured by simple solvable models




<br/>
# refer format:     



@inproceedings{nam2025position,
  title={Position: Solve Layerwise Linear Models First to Understand Neural Dynamical Phenomena (Neural Collapse, Emergence, Lazy/Rich Regime, and Grokking)},
  author={Nam, Yoonsoo and Lee, Seok Hyeong and Domine, Clementine Carla Juliette and Park, Yeachan and London, Charles and Choi, Wonyl and Goring, Niclas Alexander and Lee, Seungjai},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025},
  series={Proceedings of Machine Learning Research},
  volume={267},
  address={Vancouver, Canada},
  publisher={PMLR}
}




Nam, Yoonsoo, Seok Hyeong Lee, Clementine Carla Juliette Domine, Yeachan Park, Charles London, Wonyl Choi, Niclas Alexander Goring, and Seungjai Lee. “Position: Solve Layerwise Linear Models First to Understand Neural Dynamical Phenomena (Neural Collapse, Emergence, Lazy/Rich Regime, and Grokking).” In Proceedings of the 42nd International Conference on Machine Learning, PMLR 267, Vancouver, Canada, 2025.




