---
layout: post
title:  "[2023]OpenCon: Open-world Contrastive Learning"
date:   2025-12-10 22:32:04 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: OpenCon은 알려진 클래스와 새로운 클래스 모두에 대해 구별 가능한 표현을 학습하기 위해 프로토타입 기반의 오픈 월드 대조 학습 프레임워크를 제안합니다.


짧은 요약(Abstract) :


이 논문에서는 "Open-world Contrastive Learning"이라는 새로운 학습 프레임워크를 소개합니다. 이 프레임워크는 기계 학습 모델이 실제 환경에서 레이블이 없는 샘플을 포함한 데이터에 직면할 때 발생하는 문제를 해결하는 데 중점을 둡니다. OpenCon은 알려진 클래스와 새로운 클래스 모두에 대해 컴팩트한 표현을 학습하고, 새로운 클래스를 발견하는 과정을 촉진합니다. 실험 결과, OpenCon은 ImageNet 데이터셋에서 현재 가장 좋은 방법보다 11.9%와 7.4% 각각 새로운 클래스와 전체 분류 정확도에서 크게 향상된 성능을 보였습니다. 이 프레임워크는 EM 알고리즘 관점에서 엄밀하게 해석될 수 있으며, 유사한 샘플을 임베딩 공간에서 클러스터링하여 대조 손실을 최소화하는 방식으로 작동합니다.



This paper introduces a new learning framework called "Open-world Contrastive Learning" (OpenCon). This framework focuses on addressing the challenges that arise when machine learning models encounter unlabeled samples in real-world settings. OpenCon aims to learn compact representations for both known and novel classes while facilitating the discovery of new classes. Experimental results demonstrate that OpenCon significantly outperforms the current best method on the ImageNet dataset by 11.9% and 7.4% in terms of novel and overall classification accuracy, respectively. The framework can be rigorously interpreted from an EM algorithm perspective, operating by minimizing the contrastive loss through clustering similar samples in the embedding space.


* Useful sentences :


** How to detect OOD?

라벨된 데이터로 클래스 프로토타입(평균 임베딩)을 만들고, unlabeled 샘플의 임베딩이 어떤 프로토타입과도 충분히 가깝지 않으면(OOD threshold 이하) novel/OOD로 판정한다.
즉, contrastive SSL로 표현을 학습하고, 그 위에 프로토타입 + 임계값 기반 스코어링을 추가해서 OOD를 거르는 방식이다.

Class prototypes (mean embeddings per labeled class) are learned, and an unlabeled sample is marked as novel/OOD if its maximum similarity to any prototype falls below a threshold.
In other words, it uses contrastive SSL for representation learning, then adds a prototype + threshold scoring rule on top to detect OOD samples.



{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
*


<br/>
# Methodology



OpenCon(오픈월드 대조 학습)은 알려진 클래스와 새로운 클래스 모두에 대해 구별 가능한 표현을 학습하기 위해 설계된 새로운 학습 프레임워크입니다. 이 프레임워크는 두 가지 주요 구성 요소로 구성됩니다: 프로토타입 기반 학습 전략과 대조 손실 함수입니다.

1. **프로토타입 기반 학습 전략**: OpenCon은 각 클래스에 대해 프로토타입 벡터를 유지합니다. 이 프로토타입은 각 클래스의 대표적인 임베딩 벡터로, 학습 과정에서 업데이트됩니다. 모델은 주어진 샘플의 임베딩과 프로토타입 간의 코사인 유사도를 측정하여, 샘플이 알려진 클래스에 속하는지 새로운 클래스에 속하는지를 판단합니다. 이 과정에서 OOD(Out-of-Distribution) 탐지를 통해 알려진 클래스와 새로운 클래스를 구분합니다.

2. **대조 손실 함수**: OpenCon은 새로운 클래스에 대한 대조 손실을 정의하여, 예측된 레이블이 동일한 샘플들 간의 임베딩을 가깝게 만들고, 서로 다른 클래스의 샘플 간의 임베딩은 멀어지도록 유도합니다. 이 손실 함수는 주어진 샘플의 긍정적 및 부정적 샘플 세트를 구성하는 데 사용됩니다. 긍정적 샘플은 예측된 레이블이 동일한 샘플로 구성되며, 부정적 샘플은 나머지 샘플로 구성됩니다.

3. **훈련 과정**: OpenCon은 라벨이 있는 데이터셋(Dl)과 라벨이 없는 데이터셋(Du)을 동시에 사용하여 훈련됩니다. 라벨이 있는 데이터셋은 알려진 클래스의 샘플로 구성되며, 라벨이 없는 데이터셋은 알려진 클래스와 새로운 클래스의 샘플이 혼합되어 있습니다. 이 두 데이터셋을 통해 모델은 알려진 클래스와 새로운 클래스 모두에 대해 구별 가능한 표현을 학습합니다.

4. **이론적 해석**: OpenCon은 EM(Expectation-Maximization) 알고리즘의 관점에서 해석될 수 있습니다. 대조 손실을 최소화하는 것은 유사한 샘플을 임베딩 공간에서 클러스터링하여 우도(likelihood)를 부분적으로 최대화하는 효과를 가집니다.

OpenCon은 기존의 방법들에 비해 뛰어난 성능을 보여주며, 특히 새로운 클래스의 정확도를 크게 향상시킵니다. 이 프레임워크는 CNN 기반 아키텍처와 Transformer 기반 아키텍처 모두에 호환되며, 엔드 투 엔드로 훈련할 수 있는 장점이 있습니다.




OpenCon (Open-world Contrastive Learning) is a new learning framework designed to learn distinguishable representations for both known and novel classes. This framework consists of two main components: a prototype-based learning strategy and a contrastive loss function.

1. **Prototype-based Learning Strategy**: OpenCon maintains prototype vectors for each class. These prototypes serve as representative embedding vectors for each class and are updated during the training process. The model measures the cosine similarity between the embedding of a given sample and the prototypes to determine whether the sample belongs to a known class or a novel class. This process involves OOD (Out-of-Distribution) detection to separate known and novel classes.

2. **Contrastive Loss Function**: OpenCon defines a contrastive loss for novel classes that encourages embeddings of samples with the same predicted label to be close together while pushing embeddings of samples from different classes apart. This loss function is used to construct the positive and negative sample sets for a given sample. The positive samples consist of those with the same predicted label, while the negative samples consist of the remaining samples.

3. **Training Process**: OpenCon is trained using both labeled datasets (Dl) and unlabeled datasets (Du). The labeled dataset consists of samples from known classes, while the unlabeled dataset contains a mixture of samples from both known and novel classes. By leveraging both datasets, the model learns distinguishable representations for both known and novel classes.

4. **Theoretical Interpretation**: OpenCon can be interpreted from the perspective of the EM (Expectation-Maximization) algorithm. Minimizing the contrastive loss partially maximizes the likelihood by clustering similar samples in the embedding space.

OpenCon demonstrates superior performance compared to existing methods, significantly improving the accuracy of novel classes. This framework is compatible with both CNN-based and Transformer-based architectures and offers the advantage of being end-to-end trainable.


<br/>
# Results



이 논문에서는 OpenCon이라는 새로운 오픈 월드 대조 학습 프레임워크를 제안하고, 이를 통해 기존의 경쟁 모델들과 비교하여 성능을 평가하였다. OpenCon은 이미지 분류 작업에서 특히 효과적이며, 다양한 데이터셋에서 실험을 통해 그 성능을 입증하였다.

#### 실험 데이터셋
1. **CIFAR-100**: 100개의 클래스로 구성된 데이터셋으로, 각 클래스는 600개의 이미지를 포함하고 있다.
2. **ImageNet-100**: 100개의 클래스를 포함하는 ImageNet의 하위 집합으로, 각 클래스는 1300개의 이미지를 포함하고 있다.
3. **ImageNet-1k**: 1000개의 클래스를 포함하는 대규모 데이터셋으로, OpenCon의 성능을 대규모 환경에서 평가하기 위해 사용되었다.

#### 성능 메트릭
- **전체 정확도 (Overall Accuracy)**: 모든 클래스에 대한 정확도를 측정.
- **신규 클래스 정확도 (Novel Class Accuracy)**: 모델이 신규 클래스에 대해 얼마나 잘 분류하는지를 측정.
- **기존 클래스 정확도 (Seen Class Accuracy)**: 모델이 기존 클래스에 대해 얼마나 잘 분류하는지를 측정.

#### 경쟁 모델
OpenCon은 다음과 같은 기존 모델들과 비교되었다:
- **ORCA (Open-world semi-supervised learning)**: 2022년에 제안된 모델로, OpenCon과 유사한 설정에서 성능을 평가받았다.
- **GCD (Generalized Category Discovery)**: 2022년에 제안된 모델로, 신규 클래스 발견 문제를 다루고 있다.

#### 결과
- **CIFAR-100**에서 OpenCon은 전체 정확도 53.7%, 신규 클래스 정확도 48.7%, 기존 클래스 정확도 69.0%를 기록하였다. 이는 ORCA보다 7.3% 향상된 결과이다.
- **ImageNet-100**에서 OpenCon은 전체 정확도 83.8%, 신규 클래스 정확도 80.8%, 기존 클래스 정확도 90.6%를 기록하였다. 이는 ORCA보다 7.4% 향상된 결과이다.
- **ImageNet-1k**에서 OpenCon은 전체 정확도 44.6%, 신규 클래스 정확도 42.5%, 기존 클래스 정확도 73.9%를 기록하였다. GCD보다 8.8% 향상된 성능이다.

이러한 결과는 OpenCon이 기존의 경쟁 모델들보다 더 나은 성능을 발휘하며, 특히 신규 클래스에 대한 분류 성능이 크게 향상되었음을 보여준다. OpenCon은 대조 학습을 통해 알려진 클래스와 신규 클래스를 동시에 효과적으로 학습할 수 있는 능력을 갖추고 있다.

---



In this paper, a new open-world contrastive learning framework called OpenCon is proposed, and its performance is evaluated against existing competitive models. OpenCon is particularly effective in image classification tasks, and its performance is validated through experiments on various datasets.

#### Experimental Datasets
1. **CIFAR-100**: A dataset consisting of 100 classes, each containing 600 images.
2. **ImageNet-100**: A subset of ImageNet with 100 classes, each containing approximately 1300 images.
3. **ImageNet-1k**: A large-scale dataset containing 1000 classes, used to evaluate the performance of OpenCon in a large-scale setting.

#### Performance Metrics
- **Overall Accuracy**: Measures the accuracy across all classes.
- **Novel Class Accuracy**: Measures how well the model classifies novel classes.
- **Seen Class Accuracy**: Measures how well the model classifies seen (known) classes.

#### Competitive Models
OpenCon was compared with the following existing models:
- **ORCA (Open-world semi-supervised learning)**: A model proposed in 2022, evaluated under similar settings as OpenCon.
- **GCD (Generalized Category Discovery)**: A model proposed in 2022 that addresses the problem of discovering novel classes.

#### Results
- On **CIFAR-100**, OpenCon achieved an overall accuracy of 53.7%, a novel class accuracy of 48.7%, and a seen class accuracy of 69.0%, which is a 7.3% improvement over ORCA.
- On **ImageNet-100**, OpenCon achieved an overall accuracy of 83.8%, a novel class accuracy of 80.8%, and a seen class accuracy of 90.6%, which is a 7.4% improvement over ORCA.
- On **ImageNet-1k**, OpenCon achieved an overall accuracy of 44.6%, a novel class accuracy of 42.5%, and a seen class accuracy of 73.9%, outperforming GCD by 8.8%.

These results demonstrate that OpenCon outperforms existing competitive models, particularly in terms of classification performance for novel classes. OpenCon effectively learns to distinguish between known and novel classes through contrastive learning.


<br/>
# 예제



이 논문에서는 Open-world Contrastive Learning (OpenCon)이라는 새로운 학습 프레임워크를 제안합니다. OpenCon은 레이블이 있는 데이터와 레이블이 없는 데이터 모두에서 알려진 클래스와 새로운 클래스를 동시에 학습하는 것을 목표로 합니다. 이 과정에서 모델은 새로운 클래스를 발견하고, 두 클래스 간의 구별 가능한 표현을 학습합니다.

#### 예시

1. **트레이닝 데이터**:
   - **레이블이 있는 데이터 (Dl)**: 예를 들어, 100개의 클래스 중 50개 클래스에 대한 이미지가 레이블이 붙어 있습니다. 이 데이터는 다음과 같습니다:
     - 클래스 1: 고양이 이미지 100장
     - 클래스 2: 개 이미지 100장
     - 클래스 3: 새 이미지 100장
     - ...
     - 클래스 50: 자동차 이미지 100장
   - **레이블이 없는 데이터 (Du)**: 이 데이터는 알려진 클래스와 새로운 클래스의 이미지가 혼합되어 있습니다. 예를 들어:
     - 고양이 이미지 50장
     - 개 이미지 50장
     - 새로운 클래스 (예: 비행기) 이미지 100장
     - 새로운 클래스 (예: 자전거) 이미지 100장

2. **테스트 데이터**:
   - 테스트 데이터는 레이블이 없는 데이터와 유사하게 구성되며, 모델이 새로운 클래스를 얼마나 잘 인식하는지를 평가합니다. 예를 들어:
     - 고양이 이미지 20장
     - 개 이미지 20장
     - 비행기 이미지 30장 (새로운 클래스)
     - 자전거 이미지 30장 (새로운 클래스)

3. **구체적인 작업**:
   - 모델은 주어진 트레이닝 데이터에서 고양이, 개, 자동차와 같은 알려진 클래스를 학습하고, 비행기와 자전거와 같은 새로운 클래스를 발견해야 합니다. 
   - 모델의 성능은 다음과 같은 지표로 평가됩니다:
     - 알려진 클래스의 정확도
     - 새로운 클래스의 정확도
     - 전체 정확도

이러한 방식으로 OpenCon은 레이블이 있는 데이터와 레이블이 없는 데이터 모두를 활용하여, 알려진 클래스와 새로운 클래스를 효과적으로 구별할 수 있는 표현을 학습합니다.

---




This paper introduces a new learning framework called Open-world Contrastive Learning (OpenCon). OpenCon aims to learn from both labeled and unlabeled data, simultaneously addressing known and novel classes. In this process, the model discovers new classes and learns distinguishable representations between the two classes.

#### Example

1. **Training Data**:
   - **Labeled Data (Dl)**: For instance, among 100 classes, there are images labeled for 50 classes. This data might look like:
     - Class 1: 100 images of cats
     - Class 2: 100 images of dogs
     - Class 3: 100 images of birds
     - ...
     - Class 50: 100 images of cars
   - **Unlabeled Data (Du)**: This data contains a mixture of known and novel class images. For example:
     - 50 images of cats
     - 50 images of dogs
     - 100 images of a new class (e.g., airplanes)
     - 100 images of another new class (e.g., bicycles)

2. **Test Data**:
   - The test data is structured similarly to the unlabeled data and is used to evaluate how well the model recognizes new classes. For example:
     - 20 images of cats
     - 20 images of dogs
     - 30 images of airplanes (new class)
     - 30 images of bicycles (new class)

3. **Specific Task**:
   - The model is required to learn known classes such as cats, dogs, and cars from the provided training data while discovering new classes like airplanes and bicycles.
   - The model's performance is evaluated using metrics such as:
     - Accuracy on known classes
     - Accuracy on novel classes
     - Overall accuracy

In this way, OpenCon effectively utilizes both labeled and unlabeled data to learn representations that can distinguish between known and novel classes.

<br/>
# 요약


OpenCon은 알려진 클래스와 새로운 클래스 모두에 대해 구별 가능한 표현을 학습하기 위해 프로토타입 기반의 오픈 월드 대조 학습 프레임워크를 제안합니다. 실험 결과, OpenCon은 CIFAR-100 및 ImageNet-100 데이터셋에서 기존 최상의 방법인 ORCA보다 각각 7.4% 및 11.9% 더 높은 정확도를 달성했습니다. 이 방법은 알려진 클래스와 새로운 클래스의 혼합 데이터에서 효과적으로 학습할 수 있는 능력을 보여줍니다.

---

OpenCon introduces a prototype-based open-world contrastive learning framework to learn distinguishable representations for both known and novel classes. Experimental results demonstrate that OpenCon achieves 7.4% and 11.9% higher accuracy than the current best method, ORCA, on the CIFAR-100 and ImageNet-100 datasets, respectively. This method showcases the ability to effectively learn from a mixture of known and novel class data.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: OpenCon의 학습 프레임워크를 설명하는 다이어그램으로, 라벨이 있는 데이터(Dl)와 라벨이 없는 데이터(Du)에서 알려진 클래스와 새로운 클래스를 동시에 학습하는 구조를 보여줍니다. 이 구조는 알려진 클래스와 새로운 클래스의 구분을 통해 효과적인 표현 학습을 가능하게 합니다.
   - **Figure 2**: UMAP을 사용하여 OpenCon, ORCA, GCD의 임베딩을 시각화한 결과로, OpenCon이 더 밀집되고 구별 가능한 임베딩 공간을 생성함을 보여줍니다. 이는 OpenCon이 새로운 클래스에 대한 표현을 더 잘 학습했음을 나타냅니다.
   - **Figure 3**: OOD(Out-of-Distribution) 탐지의 중요성을 보여주는 그래프입니다. OOD 탐지를 통해 알려진 클래스와 새로운 클래스를 구분함으로써 성능이 향상됨을 확인할 수 있습니다.
   - **Figure 5**: 20개 클래스의 임베딩을 시각화한 결과로, OpenCon이 GCD 및 ORCA보다 더 구별 가능한 임베딩을 생성함을 보여줍니다.

2. **테이블**
   - **Table 1**: 다양한 문제 설정을 비교한 표로, Open-world Semi-supervised Learning이 기존의 방법들과 어떻게 다른지를 명확히 보여줍니다.
   - **Table 2**: CIFAR-100 및 ImageNet-100에서 OpenCon의 성능을 다른 방법들과 비교한 결과로, OpenCon이 기존의 최선의 방법들보다 우수한 성능을 보임을 나타냅니다.
   - **Table 4**: 손실 구성 요소에 대한 ablation study 결과로, 각 손실 구성 요소가 OpenCon의 성능에 미치는 영향을 보여줍니다. 모든 손실이 함께 작용하여 표현 품질을 향상시키는 데 기여함을 확인할 수 있습니다.
   - **Table 5**: 알려지지 않은 클래스 수에 대한 OpenCon의 성능을 보여주는 표로, OpenCon이 클래스 수를 미리 알지 못해도 경쟁력 있는 성능을 유지함을 나타냅니다.

3. **어펜딕스**
   - 어펜딕스에서는 OpenCon의 알고리즘, 이론적 배경, 하이퍼파라미터 설정, OOD 탐지 방법 비교 등 다양한 추가 정보를 제공합니다. 특히, OOD 탐지의 중요성과 OpenCon의 이론적 정당성을 설명하는 부분이 인상적입니다.




1. **Diagrams and Figures**
   - **Figure 1**: This diagram illustrates the learning framework of OpenCon, showing how it simultaneously learns from labeled data (Dl) and unlabeled data (Du) containing both known and novel classes. This structure enables effective representation learning by distinguishing between known and novel classes.
   - **Figure 2**: The UMAP visualization demonstrates the embeddings produced by OpenCon, ORCA, and GCD. OpenCon generates a more compact and distinguishable embedding space, indicating its superior ability to learn representations for novel classes.
   - **Figure 3**: This graph highlights the importance of OOD (Out-of-Distribution) detection. It shows that performance improves when OOD detection is employed to separate known and novel classes.
   - **Figure 5**: The visualization of embeddings for 20 classes further confirms that OpenCon produces more distinguishable representations compared to GCD and ORCA.

2. **Tables**
   - **Table 1**: This table compares different problem settings, clearly illustrating how Open-world Semi-supervised Learning differs from existing methods.
   - **Table 2**: The performance comparison of OpenCon on CIFAR-100 and ImageNet-100 against other methods shows that OpenCon significantly outperforms the current best methods.
   - **Table 4**: The ablation study results on loss components reveal the impact of each loss component on OpenCon's performance, confirming that all losses work synergistically to enhance representation quality.
   - **Table 5**: This table presents OpenCon's performance with an unknown number of classes, demonstrating that it maintains competitive performance even without prior knowledge of the class count.

3. **Appendices**
   - The appendices provide additional information on the algorithm of OpenCon, theoretical background, hyperparameter settings, and comparisons of OOD detection methods. Notably, the section on the importance of OOD detection and the theoretical justification for OpenCon's approach is particularly insightful.

<br/>
# refer format:




### BibTeX 형식
```bibtex
@article{sun2023opencon,
  title={OpenCon: Open-world Contrastive Learning},
  author={Sun, Yiyou and Li, Yixuan},
  journal={Transactions on Machine Learning Research},
  year={2023},
  volume={1},
  pages={1--20},
  url={https://openreview.net/forum?id=2wWJxtpFer},
  note={Code available at: https://github.com/deeplearning-wisc/opencon}
}
```

### 시카고 스타일 인용
Yiyou Sun and Yixuan Li. 2023. "OpenCon: Open-world Contrastive Learning." *Transactions on Machine Learning Research* 1: 1-20. https://openreview.net/forum?id=2wWJxtpFer. Code available at: https://github.com/deeplearning-wisc/opencon.
