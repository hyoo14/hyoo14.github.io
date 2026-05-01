---
layout: post
title:  "[2018]Neural Relational Inference for Interacting Systems"
date:   2026-05-01 04:43:05 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 Neural Relational Inference (NRI) 모델을 제안하여 상호작용 시스템의 동역학을 관찰 데이터로부터 비지도 학습 방식으로 추론하고, 상호작용 그래프를 동시에 학습한다.


짧은 요약(Abstract) :


이 논문에서는 상호작용하는 시스템을 모델링하기 위해 신경 관계 추론(Neural Relational Inference, NRI) 모델을 소개합니다. 이 모델은 관찰 데이터만을 사용하여 상호작용을 추론하고 동역학을 학습하는 비지도 학습 모델입니다. NRI 모델은 잠재 코드가 기본 상호작용 그래프를 나타내고, 그래프 신경망을 기반으로 한 재구성을 수행하는 변분 오토인코더 형태로 구성됩니다. 실험 결과, NRI 모델은 시뮬레이션된 물리 시스템에서 실제 상호작용을 정확하게 복원할 수 있으며, 실제 모션 캡처 및 스포츠 추적 데이터에서도 해석 가능한 구조를 찾아 복잡한 동역학을 예측할 수 있음을 보여줍니다.


In this paper, we introduce the Neural Relational Inference (NRI) model for modeling interacting systems. This model is an unsupervised learning framework that infers interactions and learns dynamics solely from observational data. The NRI model takes the form of a variational auto-encoder, where the latent code represents the underlying interaction graph, and the reconstruction is based on graph neural networks. Experimental results demonstrate that the NRI model can accurately recover ground-truth interactions in simulated physical systems and can also find interpretable structures and predict complex dynamics in real motion capture and sports tracking data.


* Useful sentences :

참고로 그래프뉴럴네트워크가 쓰이는데.. 진짜 그래프가 있다고 생각하면 안되고 매트릭스가 있다고 생각하면 이해가 편함  
엣지 타입 수, 노드 수, 시간 변화 데이터를 가지고 행렬 만든다  
->즉, 임의로 구간을 설정해놓고 그 구간에 해당되는 데이터들 바탕으로 그 구간용 매트릭스를 만든다는거  
근데 여러 구간을 계속 계속 학습하니까 결국 잘 제너럴 된다... 글쿤..  

{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
*


<br/>
# Methodology



**모델 개요: Neural Relational Inference (NRI)**

Neural Relational Inference (NRI) 모델은 상호작용하는 시스템의 동역학을 학습하고, 동시에 상호작용 구조를 추론하는 비지도 학습 모델입니다. 이 모델은 변분 오토인코더(Variational Autoencoder, VAE)의 형태를 취하고 있으며, 잠재 코드(latent code)는 기본 상호작용 그래프를 나타내고, 재구성(reconstruction)은 그래프 신경망(Graph Neural Network, GNN)을 기반으로 합니다.

**모델 아키텍처**

1. **인코더(Encoder)**: 
   - 인코더는 관찰된 궤적(trajectories)으로부터 상호작용 유형을 추론합니다. 입력은 N개의 객체의 궤적이며, 각 객체의 특성 벡터는 위치와 속도를 포함합니다. 인코더는 GNN을 사용하여 완전 연결 그래프에서 잠재 그래프 구조를 예측합니다.
   - 인코더는 여러 번의 메시지 전달(message passing) 과정을 통해 각 객체 간의 상호작용을 모델링합니다. 이 과정에서 각 노드의 임베딩을 계산하고, 이를 통해 엣지 임베딩을 생성합니다.

2. **디코더(Decoder)**: 
   - 디코더는 인코더에서 추론된 그래프 구조를 기반으로 미래의 동역학을 예측합니다. 디코더는 GNN을 사용하여 현재 상태와 잠재 그래프를 입력으로 받아 다음 상태를 예측합니다.
   - 디코더는 마르코프 가정(Markov assumption)을 따르며, 이전 상태와 잠재 그래프를 기반으로 다음 상태를 예측합니다.

**트레이닝 기법**

- NRI 모델은 ELBO(변분 하한)를 최대화하는 방식으로 학습됩니다. ELBO는 두 가지 항으로 구성됩니다: 재구성 손실(reconstruction loss)과 KL 발산(KL divergence)입니다. 재구성 손실은 모델이 예측한 궤적과 실제 궤적 간의 차이를 측정하며, KL 발산은 추론된 분포와 사전 분포 간의 차이를 측정합니다.
- 모델은 비지도 방식으로 학습되며, 상호작용 그래프의 구조를 명시적으로 추론할 수 있습니다. 이를 통해 모델은 다양한 동적 시스템에서의 상호작용을 효과적으로 학습할 수 있습니다.

**특별한 기법**

- **Concrete Distribution**: NRI 모델은 잠재 변수가 이산적(discrete)인 경우에도 재파라미터화 기법을 사용할 수 있도록 Concrete 분포를 사용합니다. 이를 통해 모델은 이산 샘플링을 통해 그래디언트를 전파할 수 있습니다.
- **다단계 예측**: 디코더는 단일 시간 단계가 아닌 여러 시간 단계를 예측하도록 훈련되어, 상호작용이 단기 동역학에 미치는 영향을 더 잘 반영합니다.




**Model Overview: Neural Relational Inference (NRI)**

The Neural Relational Inference (NRI) model is an unsupervised learning model that learns the dynamics of interacting systems while simultaneously inferring the interaction structure. This model takes the form of a Variational Autoencoder (VAE), where the latent code represents the underlying interaction graph, and the reconstruction is based on Graph Neural Networks (GNNs).

**Model Architecture**

1. **Encoder**: 
   - The encoder infers interaction types from observed trajectories. The input consists of trajectories of N objects, where each object's feature vector includes position and velocity. The encoder uses a GNN to predict the latent graph structure from a fully connected graph.
   - The encoder performs multiple rounds of message passing to model interactions between objects. During this process, it computes embeddings for each node and generates edge embeddings.

2. **Decoder**: 
   - The decoder predicts future dynamics based on the graph structure inferred by the encoder. It uses a GNN to take the current state and the latent graph as inputs to predict the next state.
   - The decoder follows the Markov assumption, predicting the next state based on the previous state and the latent graph.

**Training Techniques**

- The NRI model is trained to maximize the ELBO (Evidence Lower Bound). The ELBO consists of two terms: the reconstruction loss and the KL divergence. The reconstruction loss measures the difference between the predicted and actual trajectories, while the KL divergence measures the difference between the inferred distribution and the prior distribution.
- The model is trained in an unsupervised manner, allowing it to explicitly infer the structure of the interaction graph. This enables the model to effectively learn interactions in various dynamic systems.

**Special Techniques**

- **Concrete Distribution**: The NRI model uses the Concrete distribution to allow for the reparameterization trick even when the latent variables are discrete. This enables the model to propagate gradients through discrete sampling.
- **Multi-step Prediction**: The decoder is trained to predict multiple time steps rather than a single time step, which better reflects the impact of interactions on short-term dynamics.


<br/>
# Results



이 논문에서는 Neural Relational Inference (NRI) 모델을 사용하여 상호작용 시스템의 동역학을 학습하고, 이를 통해 상호작용 그래프를 추론하는 방법을 제안합니다. 실험 결과는 세 가지 주요 시뮬레이션(스프링, 전하 입자, 그리고 쿠라모토 모델)과 실제 데이터(모션 캡처 및 NBA 데이터)를 포함합니다.

#### 1. 경쟁 모델
NRI 모델은 여러 경쟁 모델과 비교되었습니다. 주요 비교 모델은 다음과 같습니다:
- **Corr. (path)**: 경로 간의 상관관계를 기반으로 상호작용 그래프를 추정합니다.
- **Corr. (LSTM)**: LSTM을 사용하여 각 개체의 경로를 개별적으로 모델링하고, 마지막 숨겨진 상태 간의 상관관계를 계산하여 상호작용 행렬을 도출합니다.
- **NRI (sim.)**: NRI 모델의 디코더를 실제 시뮬레이터로 교체하여 훈련합니다.
- **Supervised**: 상호작용 그래프의 실제 레이블을 사용하여 인코더를 훈련합니다.

#### 2. 테스트 데이터
실험은 50,000개의 훈련 예제와 10,000개의 검증 및 테스트 예제를 포함하는 세 가지 시뮬레이션 데이터셋에서 수행되었습니다. 각 데이터셋은 5개 또는 10개의 상호작용 개체로 구성되어 있으며, 각 개체의 위치와 속도를 포함합니다. 모션 캡처 데이터는 31개의 3D 경로로 구성되어 있으며, NBA 데이터는 25프레임의 피크 앤 롤(PnR) 인스턴스를 포함합니다.

#### 3. 메트릭
모델의 성능은 두 가지 주요 메트릭으로 평가되었습니다:
- **정확도 (Accuracy)**: 상호작용 그래프의 정확도를 측정합니다. NRI 모델은 대부분의 작업에서 높은 정확도로 실제 상호작용 그래프를 복원했습니다.
- **평균 제곱 오차 (MSE)**: 미래 상태 예측의 정확성을 평가합니다. NRI 모델은 LSTM 및 정적 기준선보다 우수한 성능을 보였습니다.

#### 4. 결과
NRI 모델은 스프링 모델에서 99.9%의 정확도로 상호작용을 복원했으며, 전하 입자 모델에서는 82.1%의 정확도를 기록했습니다. 쿠라모토 모델에서는 96.0%의 정확도를 달성했습니다. 또한, NRI 모델은 미래 상태 예측에서 LSTM 모델보다 우수한 성능을 보였으며, 특히 장기 예측에서 그 차이가 두드러졌습니다.

모션 캡처 데이터와 NBA 데이터에서도 NRI 모델은 기존의 LSTM 모델보다 더 나은 성능을 보였으며, 특히 동적 상호작용을 모델링하는 데 있어 효과적이었습니다. NBA 데이터에서는 피크 앤 롤 상황에서의 상호작용을 잘 포착하여 의미 있는 예측을 수행했습니다.





This paper introduces the Neural Relational Inference (NRI) model, which learns the dynamics of interacting systems and infers interaction graphs. The experimental results include three main simulations (springs, charged particles, and the Kuramoto model) and real data (motion capture and NBA data).

#### 1. Competing Models
The NRI model was compared with several competing models. The main comparison models are as follows:
- **Corr. (path)**: Estimates the interaction graph based on correlations between trajectories.
- **Corr. (LSTM)**: Uses an LSTM to model each object's trajectory individually and calculates correlations between the final hidden states to derive an interaction matrix.
- **NRI (sim.)**: Replaces the decoder of the NRI model with the actual simulator for training.
- **Supervised**: Trains the encoder using the actual labels of the interaction graph.

#### 2. Test Data
Experiments were conducted on three simulation datasets containing 50,000 training examples and 10,000 validation and test examples. Each dataset consists of 5 or 10 interacting objects, including their positions and velocities. The motion capture data consists of 31 3D trajectories, while the NBA data includes 25 frames of pick-and-roll (PnR) instances.

#### 3. Metrics
The performance of the models was evaluated using two main metrics:
- **Accuracy**: Measures the accuracy of the interaction graph. The NRI model accurately recovered the ground-truth interaction graph with high precision in most tasks.
- **Mean Squared Error (MSE)**: Assesses the accuracy of future state predictions. The NRI model outperformed LSTM and static baselines.

#### 4. Results
The NRI model achieved an accuracy of 99.9% in recovering interactions in the springs model and 82.1% in the charged particles model. It reached 96.0% accuracy in the Kuramoto model. Additionally, the NRI model demonstrated superior performance in future state predictions compared to LSTM models, particularly in long-term predictions.

In the motion capture and NBA datasets, the NRI model outperformed existing LSTM models, effectively modeling dynamic interactions. In the NBA data, it captured interactions during pick-and-roll situations, making meaningful predictions.


<br/>
# 예제




이 논문에서는 Neural Relational Inference (NRI) 모델을 사용하여 상호작용하는 시스템의 동역학을 학습하고, 이를 통해 상호작용 구조를 추론하는 방법을 제시합니다. 실험은 물리적 시뮬레이션, 모션 캡처 데이터, 그리고 NBA 스포츠 트래킹 데이터에서 수행되었습니다. 각 실험의 구체적인 입력과 출력, 그리고 작업(Task)에 대해 설명하겠습니다.

#### 1. 물리적 시뮬레이션
- **트레이닝 데이터**: 5개 또는 10개의 입자로 구성된 2D 박스에서의 시뮬레이션. 각 입자는 스프링으로 연결될 수 있으며, 초기 위치와 속도는 무작위로 설정됩니다. 각 시뮬레이션에서 50,000개의 훈련 예제가 생성됩니다.
- **테스트 데이터**: 훈련 데이터와 동일한 설정으로 생성된 10,000개의 검증 및 테스트 예제.
- **입력**: 각 입자의 위치와 속도를 포함하는 피처 벡터.
- **출력**: 다음 시간 단계의 입자 위치 예측.
- **작업**: 상호작용 그래프를 추론하고, 미래의 입자 위치를 예측하는 것.

#### 2. 모션 캡처 데이터
- **트레이닝 데이터**: CMU 모션 캡처 데이터베이스에서 수집된 걷기 동작의 11개 트라이얼로 구성된 데이터. 각 트라이얼은 31개의 3D 관절의 위치를 추적합니다.
- **테스트 데이터**: 7개의 트라이얼로 구성된 테스트 세트.
- **입력**: 각 관절의 위치와 속도 데이터.
- **출력**: 다음 시간 단계의 관절 위치 예측.
- **작업**: 동작의 동역학을 모델링하고, 미래의 관절 위치를 예측하는 것.

#### 3. NBA 데이터
- **트레이닝 데이터**: 2016 시즌의 NBA 경기에서 추출한 12,000개의 픽 앤 롤(Pick and Roll) 세그먼트. 각 세그먼트는 25프레임 길이로, 5개의 노드(공, 공을 가진 선수, 스크리너, 수비수)로 구성됩니다.
- **테스트 데이터**: 1,000개의 검증 및 테스트 세트.
- **입력**: 각 프레임에서의 선수 위치.
- **출력**: 다음 시간 단계의 선수 위치 예측.
- **작업**: 픽 앤 롤 상황에서 선수 간의 상호작용을 모델링하고, 미래의 선수 위치를 예측하는 것.

이러한 실험을 통해 NRI 모델은 상호작용 그래프를 정확하게 복원하고, 동역학을 예측하는 데 있어 높은 성능을 보였습니다.

---




This paper presents the Neural Relational Inference (NRI) model, which learns the dynamics of interacting systems and infers the interaction structure. Experiments were conducted on physical simulations, motion capture data, and NBA sports tracking data. Below is a detailed explanation of the specific inputs, outputs, and tasks for each experiment.

#### 1. Physical Simulations
- **Training Data**: Simulations consisting of 5 or 10 particles in a 2D box. Each particle can be connected by springs, and initial positions and velocities are set randomly. A total of 50,000 training examples are generated for each simulation.
- **Test Data**: 10,000 validation and test examples generated under the same conditions as the training data.
- **Input**: Feature vectors containing the positions and velocities of each particle.
- **Output**: Predictions of the next time step's particle positions.
- **Task**: To infer the interaction graph and predict future particle positions.

#### 2. Motion Capture Data
- **Training Data**: Data collected from walking motions in the CMU Motion Capture Database, consisting of 11 trials. Each trial tracks the positions of 31 3D joints.
- **Test Data**: A test set consisting of 7 trials.
- **Input**: Position and velocity data for each joint.
- **Output**: Predictions of the next time step's joint positions.
- **Task**: To model the dynamics of the motion and predict future joint positions.

#### 3. NBA Data
- **Training Data**: 12,000 segments extracted from the NBA 2016 season, each segment representing a Pick and Roll (PnR) instance, consisting of 25 frames and 5 nodes (the ball, ball handler, screener, and defender).
- **Test Data**: 1,000 validation and test examples.
- **Input**: Player positions at each frame.
- **Output**: Predictions of the next time step's player positions.
- **Task**: To model interactions between players during the Pick and Roll situation and predict future player positions.

Through these experiments, the NRI model demonstrated high performance in accurately recovering interaction graphs and predicting dynamics.

<br/>
# 요약


이 논문에서는 Neural Relational Inference (NRI) 모델을 제안하여 상호작용 시스템의 동역학을 관찰 데이터로부터 비지도 학습 방식으로 추론하고, 상호작용 그래프를 동시에 학습한다. 실험 결과, NRI 모델은 물리적 시뮬레이션과 실제 모션 캡처 데이터에서 높은 정확도로 상호작용을 복원하고 복잡한 동역학을 예측할 수 있음을 보여주었다. 예를 들어, 농구 경기 데이터에서 NRI 모델은 선수 간의 상호작용을 효과적으로 학습하여 미래의 궤적을 예측하는 데 성공하였다.

---

In this paper, the authors propose the Neural Relational Inference (NRI) model to infer the dynamics of interacting systems and learn the interaction graph simultaneously from observational data in an unsupervised manner. Experimental results demonstrate that the NRI model can accurately recover interactions and predict complex dynamics in both physical simulations and real motion capture data. For instance, in basketball game data, the NRI model successfully learns player interactions to predict future trajectories.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 이 그림은 2D 입자들이 보이지 않는 스프링으로 연결된 물리적 시뮬레이션을 보여줍니다. 왼쪽은 입자들의 움직임을, 오른쪽은 잠재적 상호작용 그래프를 나타냅니다. 이 그림은 NRI 모델이 어떻게 상호작용을 추론하는지를 시각적으로 설명합니다.
   - **Figure 3**: NRI 모델의 구조를 보여주는 다이어그램으로, 인코더와 디코더의 상호작용을 설명합니다. 인코더는 입력 궤적을 기반으로 상호작용을 예측하고, 디코더는 이 상호작용 그래프를 기반으로 동적 모델을 학습합니다.
   - **Figure 6**: 모션 캡처 데이터와 스포츠 추적 데이터에서의 예측 결과를 보여줍니다. NRI 모델이 실제 궤적과 비교하여 얼마나 정확하게 예측했는지를 시각적으로 나타냅니다.

2. **테이블**
   - **Table 1**: 다양한 모델의 상호작용 회복 정확도를 비교한 결과를 보여줍니다. NRI(learned) 모델은 다른 모델들에 비해 높은 정확도로 상호작용 그래프를 복원하는 데 성공했습니다. 이는 NRI 모델이 비지도 학습 방식으로도 효과적으로 상호작용을 추론할 수 있음을 나타냅니다.
   - **Table 2**: 미래 상태 예측의 평균 제곱 오차(MSE)를 비교한 결과입니다. NRI(learned) 모델은 LSTM 및 정적 기준선보다 우수한 성능을 보였으며, 이는 NRI 모델이 동적 시스템의 예측에 효과적임을 시사합니다.

3. **어펜딕스**
   - 어펜딕스에서는 실험 세부사항, 데이터 생성 방법, 모델 아키텍처 및 하이퍼파라미터 설정에 대한 추가 정보를 제공합니다. 이는 연구의 재현성을 높이고, 다른 연구자들이 NRI 모델을 이해하고 적용하는 데 도움을 줍니다.




1. **Diagrams and Figures**
   - **Figure 1**: This figure illustrates a physical simulation of 2D particles connected by invisible springs. The left side shows the movement of the particles, while the right side represents the latent interaction graph. This visual representation helps explain how the NRI model infers interactions.
   - **Figure 3**: A diagram depicting the structure of the NRI model, illustrating the interaction between the encoder and decoder. The encoder predicts interactions based on input trajectories, while the decoder learns the dynamic model based on this interaction graph.
   - **Figure 6**: This figure presents the prediction results from motion capture and sports tracking data. It visually compares the NRI model's predictions against the actual trajectories, demonstrating the model's accuracy.

2. **Tables**
   - **Table 1**: This table compares the accuracy of interaction recovery across various models. The NRI(learned) model significantly outperformed others, indicating its effectiveness in inferring interactions in an unsupervised manner.
   - **Table 2**: This table shows the mean squared error (MSE) in predicting future states. The NRI(learned) model exhibited superior performance compared to LSTM and static baselines, suggesting its efficacy in predicting dynamics of interacting systems.

3. **Appendices**
   - The appendices provide additional details on experimental setups, data generation methods, model architectures, and hyperparameter settings. This information enhances the reproducibility of the research and aids other researchers in understanding and applying the NRI model.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Kipf2018,
  author = {Thomas Kipf and Ethan Fetaya and Kuan-Chieh Wang and Max Welling and Richard Zemel},
  title = {Neural Relational Inference for Interacting Systems},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  year = {2018},
  pages = {80},
  publisher = {PMLR},
  address = {Stockholm, Sweden},
  url = {http://arxiv.org/abs/1802.04687}
}
```

### 시카고 스타일

Kipf, Thomas, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, and Richard Zemel. 2018. "Neural Relational Inference for Interacting Systems." In *Proceedings of the 35th International Conference on Machine Learning*, 80. Stockholm, Sweden: PMLR. http://arxiv.org/abs/1802.04687.
