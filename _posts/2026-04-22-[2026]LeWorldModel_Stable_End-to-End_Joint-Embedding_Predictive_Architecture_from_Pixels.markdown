---
layout: post
title:  "[2026]LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"
date:   2026-04-22 13:04:46 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: LeWorldModel (LeWM)은 두 개의 손실 항목만을 사용하여 원시 픽셀에서 안정적으로 학습할 수 있는 첫 번째 Joint Embedding Predictive Architecture(JEPA)로, 예측 손실과 가우시안 분포를 강제하는 정규화 항을 포함한다.


짧은 요약(Abstract) :

이 논문에서는 LeWorldModel(LeWM)이라는 새로운 Joint Embedding Predictive Architecture (JEPA)를 소개합니다. LeWM은 복잡한 다중 손실 함수, 지수 이동 평균, 사전 훈련된 인코더 또는 보조 감독 없이 원시 픽셀에서 안정적으로 end-to-end 학습할 수 있는 첫 번째 JEPA입니다. 이 모델은 다음 임베딩 예측 손실과 가우시안 분포의 잠재 임베딩을 강제하는 정규화기 두 가지 손실 항만을 사용하여 훈련됩니다. 이로 인해 조정해야 할 손실 하이퍼파라미터의 수가 기존의 대안보다 여섯 개에서 하나로 줄어듭니다. LeWM은 1500만 개의 파라미터를 가지고 있으며, 단일 GPU에서 몇 시간 내에 훈련할 수 있습니다. 이 모델은 기초 모델 기반의 세계 모델보다 최대 48배 빠른 계획을 가능하게 하면서도 다양한 2D 및 3D 제어 작업에서 경쟁력을 유지합니다. 또한, LeWM의 잠재 공간은 물리적 구조를 의미 있게 인코딩하고 있으며, 물리적으로 불가능한 사건을 신뢰성 있게 감지하는 능력을 보여줍니다.



This paper introduces a new Joint Embedding Predictive Architecture (JEPA) called LeWorldModel (LeWM). LeWM is the first JEPA that can be trained stably end-to-end from raw pixels without relying on complex multi-term losses, exponential moving averages, pre-trained encoders, or auxiliary supervision. It uses only two loss terms: a next-embedding prediction loss and a regularizer that enforces Gaussian-distributed latent embeddings. This reduces the number of tunable loss hyperparameters from six to one compared to existing alternatives. LeWM has 15 million parameters and can be trained on a single GPU in a few hours. It enables planning up to 48 times faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Additionally, LeWM's latent space encodes meaningful physical structure and demonstrates a reliable ability to detect physically implausible events.


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



LeWorldModel(LeWM)은 Joint Embedding Predictive Architecture(JEPA) 기반의 모델로, 원시 픽셀 데이터에서 안정적으로 학습할 수 있도록 설계되었습니다. LeWM의 주요 구성 요소는 인코더와 예측기로, 인코더는 주어진 프레임 관측치를 저차원 잠재 표현으로 매핑하고, 예측기는 현재 잠재 상태와 행동을 기반으로 다음 프레임의 잠재 표현을 예측합니다.

#### 모델 아키텍처
1. **인코더**: LeWM의 인코더는 Vision Transformer(ViT) 아키텍처를 사용하여, 입력된 이미지 프레임을 저차원 잠재 표현으로 변환합니다. 인코더는 [CLS] 토큰의 임베딩을 사용하여 최종 잠재 표현을 생성하며, 이 과정에서 1층의 MLP와 배치 정규화를 통해 최종 임베딩을 새로운 표현 공간으로 매핑합니다.

2. **예측기**: 예측기는 6개의 레이어로 구성된 트랜스포머로, 행동을 인코딩하기 위해 Adaptive Layer Normalization(AdaLN)을 사용합니다. 예측기는 과거의 N 프레임 표현을 입력으로 받아 다음 프레임 표현을 오토회귀적으로 예측합니다.

#### 트레이닝 데이터
LeWM은 보상 신호 없이 오프라인 데이터셋에서 학습됩니다. 데이터는 행동 정책에 따라 수집된 원시 픽셀 관측치와 행동의 시퀀스로 구성됩니다. 이 데이터는 환경의 동적 특성을 충분히 포괄해야 하며, LeWM은 이러한 데이터를 통해 환경의 동적 모델을 학습합니다.

#### 트레이닝 기법
LeWM의 학습 목표는 두 가지 손실 항목으로 구성됩니다:
1. **예측 손실 (Lpred)**: 다음 잠재 상태의 예측과 실제 잠재 상태 간의 차이를 최소화합니다.
2. **정규화 손실 (SIGReg)**: 잠재 임베딩이 이소트로픽 가우시안 분포를 따르도록 유도하여 표현의 다양성을 촉진합니다. 이 정규화는 고차원 공간에서의 정규성 검정을 통해 수행됩니다.

LeWM은 복잡한 다중 손실 항목이나 외부 정보에 의존하지 않고, 단순하고 안정적인 두 항목의 목표를 통해 학습할 수 있습니다. 이로 인해 LeWM은 하이퍼파라미터 조정이 용이하고, 단일 GPU에서 몇 시간 내에 학습할 수 있는 효율성을 제공합니다.




LeWorldModel (LeWM) is a model based on the Joint Embedding Predictive Architecture (JEPA), designed to learn stably from raw pixel data. The key components of LeWM are the encoder and the predictor. The encoder maps given frame observations into low-dimensional latent representations, while the predictor models the environment dynamics by predicting the next latent representation based on the current latent state and action.

#### Model Architecture
1. **Encoder**: The encoder of LeWM utilizes a Vision Transformer (ViT) architecture to convert input image frames into low-dimensional latent representations. It employs the [CLS] token embedding to generate the final latent representation, followed by a projection step using a 1-layer MLP with Batch Normalization to map the final embedding into a new representation space.

2. **Predictor**: The predictor consists of a transformer with 6 layers, which incorporates actions through Adaptive Layer Normalization (AdaLN). It takes a history of N frame representations as input and autoregressively predicts the next frame representation.

#### Training Data
LeWM is trained in an offline setting without reward signals. The data consists of sequences of raw pixel observations and actions collected from behavior policies. This data must sufficiently cover the dynamics of the environment, allowing LeWM to learn a dynamic model of the environment.

#### Training Technique
The training objective of LeWM consists of two loss terms:
1. **Prediction Loss (Lpred)**: Minimizes the difference between the predicted latent state and the actual latent state.
2. **Regularization Loss (SIGReg)**: Encourages the latent embeddings to follow an isotropic Gaussian distribution, promoting feature diversity. This regularization is performed through normality testing in high-dimensional spaces.

LeWM can learn through a simple and stable two-term objective without relying on complex multi-term losses or external information. This allows for easy hyperparameter tuning and efficient training on a single GPU within a few hours.


<br/>
# Results



LeWorldModel(LeWM)은 다양한 2D 및 3D 제어 작업에서 경쟁 모델들과 비교하여 우수한 성능을 보였습니다. LeWM은 Push-T, OGBench-Cube, Two-Room, Reacher와 같은 여러 환경에서 평가되었습니다. 각 환경에서 LeWM의 성공률은 다음과 같았습니다:

1. **Push-T**: LeWM은 96%의 성공률을 기록하여 PLDM(78%)과 DINO-WM(92%)을 초과했습니다. 이는 LeWM이 더 복잡한 계획 작업에서 더 나은 성능을 발휘함을 나타냅니다.
   
2. **OGBench-Cube**: LeWM은 DINO-WM과 PLDM에 비해 경쟁력 있는 성능을 보였지만, DINO-WM이 약간 더 높은 성능을 보였습니다. 이는 DINO-WM이 더 많은 데이터로 사전 훈련된 모델을 사용했기 때문일 수 있습니다.

3. **Two-Room**: LeWM은 PLDM과 DINO-WM에 비해 낮은 성공률을 보였지만, 이는 환경의 낮은 복잡성으로 인해 SIGReg 정규화가 고차원 잠재 공간에서 이소트로픽 가우시안 분포를 맞추는 데 어려움을 겪었기 때문일 수 있습니다.

4. **Reacher**: LeWM은 PLDM과 DINO-WM에 비해 경쟁력 있는 성능을 보였으며, 특히 DINO-WM이 더 많은 사전 훈련 데이터를 활용했음에도 불구하고 LeWM이 유사한 성능을 유지했습니다.

LeWM의 훈련 과정은 안정적이며, 단일 하이퍼파라미터(정규화 가중치 λ)만을 조정하면 되므로 하이퍼파라미터 선택이 간소화되었습니다. LeWM은 예측 손실과 SIGReg 정규화 손실의 두 가지 항목으로 구성된 간단한 훈련 목표를 사용하여 안정적인 훈련을 가능하게 했습니다. 

LeWM은 또한 물리적 이해를 평가하기 위해 물리적 양을 추출하는 프로빙 실험에서 PLDM보다 우수한 성능을 보였으며, DINO-WM과도 경쟁력을 유지했습니다. 이는 LeWM의 잠재 공간이 환경의 동적 구조를 잘 캡처하고 있음을 나타냅니다.




LeWorldModel (LeWM) demonstrated superior performance compared to competing models across various 2D and 3D control tasks. It was evaluated in several environments, including Push-T, OGBench-Cube, Two-Room, and Reacher. The success rates of LeWM in each environment were as follows:

1. **Push-T**: LeWM achieved a success rate of 96%, surpassing PLDM (78%) and DINO-WM (92%). This indicates that LeWM performs better in more complex planning tasks.

2. **OGBench-Cube**: LeWM showed competitive performance compared to DINO-WM and PLDM, although DINO-WM had a slight edge. This advantage may be attributed to DINO-WM's use of a model pre-trained on a larger dataset.

3. **Two-Room**: LeWM exhibited lower success rates compared to PLDM and DINO-WM, which may be due to the low complexity of the environment, making it challenging for SIGReg regularization to match the isotropic Gaussian distribution in a high-dimensional latent space.

4. **Reacher**: LeWM maintained competitive performance against PLDM and DINO-WM, particularly as DINO-WM leveraged more pre-training data, yet LeWM still achieved similar results.

The training process of LeWM was stable, requiring only a single hyperparameter (the regularization weight λ) for tuning, which simplified hyperparameter selection. LeWM utilized a straightforward training objective consisting of a prediction loss and a SIGReg regularization loss, enabling stable training dynamics.

LeWM also outperformed PLDM in probing experiments designed to evaluate physical understanding, while remaining competitive with DINO-WM. This suggests that LeWM's latent space effectively captures the dynamic structure of the environment.


<br/>
# 예제



LeWorldModel(LeWM) 논문에서는 다양한 환경에서의 제어 작업을 위해 훈련된 모델을 평가합니다. 이 모델은 주로 두 가지 유형의 데이터로 훈련됩니다: 트레이닝 데이터와 테스트 데이터. 

1. **트레이닝 데이터**:
   - **환경**: Push-T, TwoRoom, OGBench-Cube와 같은 다양한 환경에서 수집됩니다.
   - **구체적인 인풋**: 각 환경에서 에이전트는 연속적인 픽셀 관찰을 입력으로 받습니다. 예를 들어, Push-T 환경에서는 에이전트가 T자형 블록을 목표 위치로 밀어야 하며, 이 과정에서 에이전트의 위치와 블록의 위치가 포함된 이미지가 입력으로 사용됩니다.
   - **구체적인 아웃풋**: 에이전트는 다음 상태의 임베딩을 예측해야 하며, 이 예측은 주어진 행동에 따라 달라집니다. 예를 들어, 에이전트가 블록을 밀기 위해 취한 행동에 따라 다음 상태의 임베딩이 생성됩니다.

2. **테스트 데이터**:
   - **환경**: 테스트 데이터는 훈련 데이터와 동일한 환경에서 수집되지만, 에이전트가 새로운 목표를 달성하기 위해 사용됩니다.
   - **구체적인 인풋**: 테스트 환경에서도 에이전트는 연속적인 픽셀 관찰을 입력으로 받습니다. 예를 들어, TwoRoom 환경에서는 에이전트가 두 개의 방을 이동하여 목표 위치에 도달해야 합니다.
   - **구체적인 아웃풋**: 테스트 중 에이전트는 주어진 목표에 도달하기 위해 최적의 행동 시퀀스를 생성해야 하며, 이 과정에서 LeWM 모델이 예측한 다음 상태의 임베딩을 기반으로 행동을 결정합니다.

이러한 방식으로 LeWM은 다양한 환경에서의 제어 작업을 수행하며, 훈련 데이터와 테스트 데이터 모두에서 에이전트의 성능을 평가합니다.



In the LeWorldModel (LeWM) paper, the model is evaluated on control tasks trained in various environments. The model is primarily trained on two types of data: training data and testing data.

1. **Training Data**:
   - **Environments**: Collected from various environments such as Push-T, TwoRoom, and OGBench-Cube.
   - **Specific Input**: In each environment, the agent receives continuous pixel observations as input. For example, in the Push-T environment, the agent must push a T-shaped block to a target configuration, and the input consists of images that include the agent's position and the block's position.
   - **Specific Output**: The agent is required to predict the embedding of the next state, which varies based on the actions taken. For instance, the next state embedding is generated based on the action taken by the agent to push the block.

2. **Testing Data**:
   - **Environments**: The testing data is collected from the same environments as the training data but is used for the agent to achieve new goals.
   - **Specific Input**: In the testing environment, the agent also receives continuous pixel observations as input. For example, in the TwoRoom environment, the agent must navigate between two rooms to reach a target position.
   - **Specific Output**: During testing, the agent generates an optimal sequence of actions to reach the given goal, using the next state embeddings predicted by the LeWM model as a basis for decision-making.

In this way, LeWM performs control tasks across various environments, evaluating the agent's performance on both training and testing data.

<br/>
# 요약


LeWorldModel (LeWM)은 두 개의 손실 항목만을 사용하여 원시 픽셀에서 안정적으로 학습할 수 있는 첫 번째 Joint Embedding Predictive Architecture(JEPA)로, 예측 손실과 가우시안 분포를 강제하는 정규화 항을 포함한다. LeWM은 15M 파라미터로 구성되어 있으며, 다양한 2D 및 3D 제어 작업에서 경쟁력 있는 성능을 보이며, 기존 방법보다 최대 48배 빠른 계획 속도를 달성한다. 실험 결과, LeWM은 물리적 구조를 효과적으로 인코딩하고, 비물리적 사건을 신뢰성 있게 감지하는 능력을 보여준다.

---

LeWorldModel (LeWM) is the first Joint Embedding Predictive Architecture (JEPA) that can be trained stably from raw pixels using only two loss terms: a prediction loss and a regularizer enforcing Gaussian-distributed embeddings. With 15M parameters, LeWM achieves competitive performance across diverse 2D and 3D control tasks while planning up to 48 times faster than existing methods. Experimental results demonstrate that LeWM effectively encodes physical structure and reliably detects physically implausible events.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: LeWorldModel의 훈련 파이프라인을 보여줍니다. 이 다이어그램은 인코더가 관측된 프레임을 저차원 잠재 표현으로 매핑하고, 예측기가 현재 잠재 상태와 행동을 기반으로 다음 잠재 상태를 예측하는 과정을 설명합니다. 이 구조는 LeWM이 어떻게 환경의 동적 모델을 학습하는지를 시각적으로 나타냅니다.
   - **Figure 3**: LeWM과 DINO-WM의 계획 시간 및 성능을 비교합니다. LeWM은 DINO-WM보다 최대 48배 빠른 계획 시간을 기록하며, 이는 LeWM의 효율성을 강조합니다.
   - **Figure 6**: 다양한 환경에서의 계획 성능을 보여줍니다. LeWM은 Push-T 및 Reacher 환경에서 PLDM과 DINO-WM을 초과하는 성공률을 기록하며, 이는 LeWM의 우수한 성능을 나타냅니다.
   - **Figure 10**: Violation-of-expectation 평가 결과를 보여줍니다. LeWM은 물리적 위반이 발생한 경우 더 높은 놀라움을 감지하며, 이는 모델이 물리적 규칙을 잘 이해하고 있음을 나타냅니다.

2. **테이블**
   - **Table 1**: Push-T 환경에서의 물리적 양을 프로빙한 결과를 보여줍니다. LeWM은 PLDM보다 일관되게 우수한 성능을 보이며, 이는 LeWM의 잠재 표현이 물리적 상태를 잘 캡처하고 있음을 시사합니다.
   - **Table 4**: OGBench-Cube 환경에서의 물리적 프로빙 결과를 보여줍니다. LeWM은 블록 위치와 엔드 이펙터 위치에서 가장 좋은 결과를 기록하며, 이는 LeWM의 잠재 공간이 물리적 정보를 잘 보존하고 있음을 나타냅니다.
   - **Table 5**: 훈련의 안정성을 평가하기 위한 결과를 보여줍니다. LeWM은 낮은 분산으로 높은 성공률을 기록하여 훈련의 안정성을 강조합니다.

3. **어펜딕스**
   - **Appendix A**: SIGReg의 수학적 기초를 설명합니다. 이 정규화 방법은 잠재 표현이 이소트로픽 가우시안 분포를 따르도록 유도하여 표현 붕괴를 방지합니다.
   - **Appendix B**: Cross-Entropy Method (CEM)의 작동 방식을 설명합니다. CEM은 샘플링 기반 최적화 알고리즘으로, 후보 계획을 반복적으로 개선합니다.
   - **Appendix D**: 구현 세부 사항을 제공합니다. LeWM의 인코더와 예측기 아키텍처, 훈련 과정에서의 하이퍼파라미터 설정 등을 설명합니다.

---

### Results and Insights from Other Components (Diagrams, Figures, Tables, Appendices)

1. **Diagrams and Figures**
   - **Figure 1**: Illustrates the training pipeline of LeWorldModel. This diagram visually represents how the encoder maps observed frames into low-dimensional latent representations, and how the predictor models the environment dynamics by predicting the next latent state based on the current latent state and action. It highlights the learning process of the dynamic model.
   - **Figure 3**: Compares planning time and performance between LeWM and DINO-WM. LeWM achieves planning times up to 48 times faster than DINO-WM, emphasizing its efficiency.
   - **Figure 6**: Displays planning performance across various environments. LeWM outperforms PLDM and DINO-WM in success rates for the Push-T and Reacher environments, indicating its superior performance.
   - **Figure 10**: Shows results from the Violation-of-expectation evaluation. LeWM detects higher surprise in cases of physical violations, indicating that the model has a good understanding of physical rules.

2. **Tables**
   - **Table 1**: Presents probing results for physical quantities in the Push-T environment. LeWM consistently outperforms PLDM, suggesting that its latent representations effectively capture physical states.
   - **Table 4**: Displays probing results in the OGBench-Cube environment. LeWM achieves the best results for block position and end-effector position, indicating that its latent space retains significant physical information.
   - **Table 5**: Shows results assessing training stability. LeWM exhibits high success rates with low variance, underscoring the stability of its training process.

3. **Appendices**
   - **Appendix A**: Explains the mathematical foundation of SIGReg. This regularization method encourages latent representations to follow an isotropic Gaussian distribution, preventing representation collapse.
   - **Appendix B**: Describes the workings of the Cross-Entropy Method (CEM). CEM is a sampling-based optimization algorithm that iteratively refines candidate plans.
   - **Appendix D**: Provides implementation details, including the architecture of the encoder and predictor, as well as hyperparameter settings during training.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@article{maes2026leworldmodel,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:2603.19312},
  year={2026},
  url={https://arxiv.org/abs/2603.19312}
}
```

### 시카고 스타일

Maes, Lucas, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. 2026. "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels." arXiv preprint arXiv:2603.19312. https://arxiv.org/abs/2603.19312.
