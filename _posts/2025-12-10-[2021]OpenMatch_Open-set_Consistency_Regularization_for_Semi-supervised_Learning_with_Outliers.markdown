---
layout: post
title:  "[2021]OpenMatch: Open-set Consistency Regularization for Semi-supervised Learning with Outliers"
date:   2025-12-10 22:29:56 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: OpenMatch는 Open-set Semi-Supervised Learning (OSSL) 접근 방식을 통해 라벨이 없는 데이터에서 아웃라이어를 효과적으로 탐지하고 분류하는 새로운 프레임워크입니다.


짧은 요약(Abstract) :


이 논문에서는 반지도 학습(Semi-supervised Learning, SSL)의 효과를 높이기 위해 새로운 접근법인 OpenMatch를 제안합니다. 기존의 SSL 방법들은 레이블이 있는 데이터와 레이블이 없는 데이터가 동일한 레이블 공간을 공유한다고 가정하지만, 실제로는 레이블이 없는 데이터에 레이블이 없는 새로운 카테고리(즉, 아웃라이어)가 포함될 수 있습니다. 이러한 아웃라이어는 SSL 알고리즘의 성능에 심각한 영향을 미칠 수 있습니다. OpenMatch는 FixMatch와 아웃라이어 탐지를 위한 일대다(One-vs-All, OVA) 분류기를 통합하여 아웃라이어를 거부하고 인라이어의 표현을 학습하는 방법을 제안합니다. 또한, 입력 변환에 대한 OVA 분류기의 부드러움을 향상시키는 오픈셋 소프트 일관성 정규화 손실을 도입하여 아웃라이어 탐지를 크게 개선합니다. OpenMatch는 세 가지 데이터셋에서 최첨단 성능을 달성하며, CIFAR10에서는 레이블이 없는 데이터에서 보지 못한 아웃라이어를 탐지하는 데 있어 완전 감독 모델보다 더 나은 성능을 보입니다.



This paper proposes a novel approach called OpenMatch to enhance the effectiveness of semi-supervised learning (SSL). Existing SSL methods assume that labeled and unlabeled data share the same label space; however, in practice, unlabeled data can contain novel categories (i.e., outliers) that are not present in the labeled set. These outliers can significantly harm the performance of SSL algorithms. OpenMatch unifies FixMatch with one-vs-all (OVA) classifiers for novelty detection, allowing the model to learn representations of inliers while rejecting outliers. Additionally, it introduces an open-set soft-consistency regularization loss that enhances the smoothness of the OVA classifier with respect to input transformations, greatly improving outlier detection. OpenMatch achieves state-of-the-art performance on three datasets and even outperforms a fully supervised model in detecting outliers unseen in unlabeled data on CIFAR10.


* Useful sentences :



** How to detect OOD?


기본 분류기는 FixMatch 스타일 SSL이고, 여기에 각 클래스별 OVA(One-vs-All) outlier head를 붙여서 "이 샘플이 예측 클래스의 inlier인지 OOD인지"를 판정.  
OVA와 **SOCR(soft open-set consistency)**로 outlier 판정을 안정화한 뒤, inlier로 판정된 unlabeled 샘플만 FixMatch loss에 사용해 학습.  


The backbone is FixMatch-style SSL, augmented with class-wise OVA (One-vs-All) outlier heads that decide whether a sample is an inlier or OOD for its predicted class.
Using the OVA heads plus SOCR (soft open-set consistency), it stabilizes outlier detection and then applies the FixMatch loss only to unlabeled samples predicted as inliers.



{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
*


<br/>
# Methodology



OpenMatch는 오픈셋 반지도 학습(Open-set Semi-Supervised Learning, OSSL)을 위한 새로운 프레임워크로, 주로 레이블이 없는 데이터에 포함된 아웃라이어(outlier)를 효과적으로 탐지하고, 인라이어(inlier)를 올바르게 분류하는 데 중점을 두고 있습니다. 이 방법은 FixMatch와 One-Vs-All (OVA) 분류기를 통합하여 아웃라이어를 탐지하고, 인라이어에 대한 정확한 분류를 수행합니다.

#### 1. 모델 아키텍처
OpenMatch는 세 가지 주요 구성 요소로 이루어져 있습니다:
- **공유 피처 추출기 (Feature Extractor)**: 입력 데이터를 처리하여 특징 벡터를 생성합니다.
- **아웃라이어 탐지기 (Outlier Detector)**: 각 클래스에 대해 OVA 분류기를 사용하여 아웃라이어를 탐지합니다. 각 OVA 분류기는 특정 클래스에 대한 인라이어와 아웃라이어를 구분하는 역할을 합니다.
- **클로즈드셋 분류기 (Closed-set Classifier)**: 인라이어에 대한 K-클래스 분류 확률 벡터를 출력합니다.

#### 2. 훈련 과정
OpenMatch의 훈련 과정은 다음과 같은 단계로 진행됩니다:
1. **라벨이 있는 데이터와 라벨이 없는 데이터 샘플링**: 라벨이 있는 데이터와 라벨이 없는 데이터를 배치로 샘플링합니다.
2. **아웃라이어 탐지기 훈련**: OVA 분류기를 훈련하여 각 클래스에 대한 인라이어와 아웃라이어를 구분합니다. 이 과정에서 아웃라이어 탐지기를 통해 아웃라이어의 확률을 계산하고, 이를 기반으로 아웃라이어를 식별합니다.
3. **소프트 오픈셋 일관성 정규화 (Soft Open-set Consistency Regularization, SOCR)**: 두 개의 서로 다른 데이터 증강을 통해 생성된 입력에 대해 아웃라이어 탐지기의 출력을 일관되게 유지하도록 정규화합니다. 이 과정은 아웃라이어 탐지기의 결정 경계를 부드럽게 만들어 아웃라이어 탐지 성능을 향상시킵니다.
4. **FixMatch 적용**: 아웃라이어 탐지기를 통해 인라이어로 분류된 라벨이 없는 샘플에 대해 FixMatch를 적용하여 추가적인 학습을 진행합니다.

#### 3. 손실 함수
OpenMatch는 여러 손실 함수를 결합하여 최적화합니다:
- **라벨이 있는 데이터에 대한 손실 (Lsup)**: 인라이어에 대한 크로스 엔트로피 손실과 OVA 분류기의 손실을 포함합니다.
- **오픈셋 엔트로피 최소화 손실 (Lem)**: 라벨이 없는 데이터에 대한 아웃라이어 탐지기의 엔트로피를 최소화합니다.
- **소프트 일관성 손실 (Loc)**: 아웃라이어 탐지기의 출력을 부드럽게 유지하기 위해 두 개의 증강된 입력에 대한 출력을 비교합니다.
- **FixMatch 손실 (Lfm)**: 아웃라이어 탐지기에 의해 인라이어로 분류된 샘플에 대해 적용됩니다.

이러한 구성 요소와 훈련 과정을 통해 OpenMatch는 아웃라이어를 효과적으로 탐지하고, 인라이어를 정확하게 분류하는 성능을 보여줍니다.

---




OpenMatch is a novel framework for Open-set Semi-Supervised Learning (OSSL), focusing on effectively detecting outliers present in unlabeled data and accurately classifying inliers. This method integrates FixMatch with One-Vs-All (OVA) classifiers to detect outliers while ensuring correct classification of inliers.

#### 1. Model Architecture
OpenMatch consists of three main components:
- **Feature Extractor**: Processes input data to generate feature vectors.
- **Outlier Detector**: Utilizes OVA classifiers for each class to detect outliers. Each OVA classifier is responsible for distinguishing between inliers and outliers for its corresponding class.
- **Closed-set Classifier**: Outputs a probability vector for K-class classification of inliers.

#### 2. Training Process
The training process of OpenMatch proceeds through the following steps:
1. **Sampling Labeled and Unlabeled Data**: A batch of labeled and unlabeled data is sampled.
2. **Training the Outlier Detector**: The OVA classifiers are trained to differentiate between inliers and outliers for each class. The outlier detector computes the probability of a sample being an outlier, which is used to identify outliers.
3. **Soft Open-set Consistency Regularization (SOCR)**: This regularization encourages consistency in the outputs of the outlier detector for two different augmentations of the same input, smoothing the decision boundary and enhancing outlier detection performance.
4. **Applying FixMatch**: FixMatch is applied to unlabeled samples classified as inliers by the outlier detector for additional training.

#### 3. Loss Functions
OpenMatch optimizes a combination of several loss functions:
- **Loss for Labeled Data (Lsup)**: Includes cross-entropy loss for inliers and the loss from the OVA classifiers.
- **Open-set Entropy Minimization Loss (Lem)**: Minimizes the entropy of the outlier detector for unlabeled data.
- **Soft Consistency Loss (Loc)**: Compares outputs of the outlier detector for two augmented inputs to maintain smoothness.
- **FixMatch Loss (Lfm)**: Applied to samples classified as inliers by the outlier detector.

Through these components and training processes, OpenMatch demonstrates effective outlier detection and accurate inlier classification performance.


<br/>
# Results



**결과 요약:**
OpenMatch는 여러 데이터셋에서 경쟁 모델들과 비교하여 우수한 성능을 보였습니다. 특히 CIFAR10, CIFAR100, 그리고 ImageNet-30 데이터셋에서의 성능이 두드러졌습니다.

1. **CIFAR10 데이터셋:**
   - OpenMatch는 50개의 레이블이 있는 경우 10.4%의 오류율을 기록했습니다. 이는 이전의 최첨단 모델인 MTC의 20.3% 오류율에 비해 상당한 개선을 보여줍니다.
   - AUROC(Area Under the Receiver Operating Characteristic Curve) 측면에서도 OpenMatch는 99.3%의 성능을 기록하여, 레이블이 없는 데이터에서의 아웃라이어 탐지에서 우수한 성능을 보였습니다.

2. **CIFAR100 데이터셋:**
   - OpenMatch는 400개의 레이블이 있는 경우 5.9%의 오류율을 기록했습니다. 이는 MTC의 9.0% 오류율보다 낮습니다.
   - AUROC는 87.0%로, MTC의 81.2%보다 높았습니다.

3. **ImageNet-30 데이터셋:**
   - OpenMatch는 10%의 레이블이 있는 경우 96.3%의 AUROC를 기록하여, FixMatch의 88.6%와 비교하여 우수한 성능을 보였습니다.

**비교 모델:**
- **Labeled Only:** 레이블이 있는 데이터만 사용한 모델로, CIFAR10에서 35.7%의 오류율을 기록했습니다.
- **FixMatch:** 43.2%의 오류율을 기록하여, OpenMatch보다 성능이 낮았습니다.
- **MTC:** OpenMatch와 비교했을 때, 성능이 떨어지는 경향을 보였습니다.

**테스트 데이터 및 메트릭:**
- 테스트 데이터는 CIFAR10, CIFAR100, ImageNet-30으로 구성되었으며, 각 데이터셋에서의 성능은 오류율과 AUROC로 평가되었습니다. 
- OpenMatch는 아웃라이어 탐지에서 특히 뛰어난 성능을 보였으며, 이는 레이블이 없는 데이터에서의 아웃라이어를 효과적으로 탐지할 수 있는 능력을 보여줍니다.



**Summary of Results:**
OpenMatch demonstrated superior performance compared to several competing models across multiple datasets, particularly on CIFAR10, CIFAR100, and ImageNet-30.

1. **CIFAR10 Dataset:**
   - OpenMatch achieved an error rate of 10.4% with 50 labeled samples, significantly improving over the previous state-of-the-art model, MTC, which had an error rate of 20.3%.
   - In terms of AUROC (Area Under the Receiver Operating Characteristic Curve), OpenMatch recorded a performance of 99.3%, showcasing excellent outlier detection capabilities on unlabeled data.

2. **CIFAR100 Dataset:**
   - With 400 labeled samples, OpenMatch achieved an error rate of 5.9%, which is lower than MTC's 9.0% error rate.
   - The AUROC was 87.0%, surpassing MTC's 81.2%.

3. **ImageNet-30 Dataset:**
   - OpenMatch recorded an AUROC of 96.3% with 10% labeled data, outperforming FixMatch, which achieved 88.6%.

**Comparison Models:**
- **Labeled Only:** A model using only labeled data, which recorded an error rate of 35.7% on CIFAR10.
- **FixMatch:** Achieved an error rate of 43.2%, which is lower than OpenMatch.
- **MTC:** Tended to perform worse compared to OpenMatch.

**Test Data and Metrics:**
- The test data consisted of CIFAR10, CIFAR100, and ImageNet-30, with performance evaluated based on error rates and AUROC.
- OpenMatch particularly excelled in outlier detection, demonstrating its ability to effectively identify outliers in unlabeled data.


<br/>
# 예제



OpenMatch는 오픈셋 반지도 학습(Open-set Semi-Supervised Learning, OSSL) 문제를 해결하기 위해 설계된 프레임워크입니다. 이 방법은 레이블이 있는 데이터와 레이블이 없는 데이터가 동일한 레이블 공간을 공유하지 않는 경우, 즉 레이블이 없는 데이터에 새로운 카테고리(아웃라이어)가 포함될 수 있는 상황을 다룹니다. 

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **레이블이 있는 데이터**: 예를 들어, CIFAR-10 데이터셋에서 50개의 레이블이 있는 샘플을 선택합니다. 이 샘플들은 '개', '고양이', '자동차' 등 6개의 알려진 클래스에 속합니다.
   - **레이블이 없는 데이터**: 같은 CIFAR-10 데이터셋에서 1000개의 레이블이 없는 샘플을 선택합니다. 이 샘플들은 알려진 클래스 외에도 '새', '기타'와 같은 새로운 카테고리를 포함할 수 있습니다.

2. **테스트 데이터**:
   - 테스트 데이터는 레이블이 있는 데이터와 레이블이 없는 데이터가 혼합되어 있습니다. 예를 들어, 100개의 레이블이 있는 샘플과 200개의 레이블이 없는 샘플이 포함된 테스트 세트를 사용할 수 있습니다. 이 테스트 세트는 모델이 아웃라이어를 얼마나 잘 식별하는지를 평가하는 데 사용됩니다.

#### 구체적인 작업(Task)

- **작업 목표**: 모델은 레이블이 있는 샘플을 올바른 클래스에 분류하고, 레이블이 없는 샘플 중 아웃라이어를 식별해야 합니다. 
- **입력**: 모델은 두 가지 형태의 입력을 받습니다. 첫 번째는 레이블이 있는 샘플(예: '개' 또는 '고양이'로 레이블이 붙은 이미지)이고, 두 번째는 레이블이 없는 샘플(예: 아웃라이어로 분류될 수 있는 이미지)입니다.
- **출력**: 모델은 각 입력 샘플에 대해 두 가지 출력을 생성합니다. 첫 번째는 샘플이 특정 클래스에 속할 확률(예: '개', '고양이', '자동차')이고, 두 번째는 샘플이 아웃라이어일 확률입니다.

이러한 방식으로 OpenMatch는 레이블이 없는 데이터에서 아웃라이어를 효과적으로 탐지하고, 레이블이 있는 데이터의 분류 성능을 향상시킵니다.

---




OpenMatch is a framework designed to address the problem of Open-set Semi-Supervised Learning (OSSL). This approach deals with situations where labeled and unlabeled data do not share the same label space, meaning that the unlabeled data may contain new categories (outliers).

#### Training Data and Testing Data

1. **Training Data**:
   - **Labeled Data**: For example, we select 50 labeled samples from the CIFAR-10 dataset. These samples belong to 6 known classes such as 'dog', 'cat', and 'car'.
   - **Unlabeled Data**: We select 1000 unlabeled samples from the same CIFAR-10 dataset. These samples may include new categories such as 'bird' or 'other' that are not part of the known classes.

2. **Testing Data**:
   - The testing data consists of a mix of labeled and unlabeled data. For instance, we could have a test set containing 100 labeled samples and 200 unlabeled samples. This test set is used to evaluate how well the model identifies outliers.

#### Specific Task

- **Task Objective**: The model aims to classify the labeled samples into the correct classes while identifying outliers among the unlabeled samples.
- **Input**: The model receives two types of inputs. The first is labeled samples (e.g., images labeled as 'dog' or 'cat'), and the second is unlabeled samples (e.g., images that could be classified as outliers).
- **Output**: The model generates two outputs for each input sample. The first output is the probability of the sample belonging to a specific class (e.g., 'dog', 'cat', 'car'), and the second output is the probability of the sample being an outlier.

In this way, OpenMatch effectively detects outliers in unlabeled data and improves the classification performance of labeled data.

<br/>
# 요약


OpenMatch는 Open-set Semi-Supervised Learning (OSSL) 접근 방식을 통해 라벨이 없는 데이터에서 아웃라이어를 효과적으로 탐지하고 분류하는 새로운 프레임워크입니다. 이 방법은 One-Vs-All (OVA) 분류기와 FixMatch를 결합하여 아웃라이어 탐지의 정확성을 높이고, CIFAR10 데이터셋에서 300개의 라벨이 있는 예제에 대해 10.4%의 오류율을 달성하여 기존의 최첨단 성능을 초과했습니다. OpenMatch는 라벨이 없는 데이터에서 보지 못한 아웃라이어를 탐지하는 데 있어 완전 감독 모델보다 더 나은 성능을 보여주었습니다.

---

OpenMatch is a novel framework for Open-set Semi-Supervised Learning (OSSL) that effectively detects and classifies outliers from unlabeled data. This method combines One-Vs-All (OVA) classifiers with FixMatch to enhance the accuracy of outlier detection, achieving a 10.4% error rate on the CIFAR10 dataset with 300 labeled examples, surpassing the previous state-of-the-art performance. OpenMatch also demonstrates superior performance in detecting unseen outliers in unlabeled data compared to fully supervised models.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: Open-set soft-consistency loss의 구조를 보여줍니다. 두 개의 서로 다른 증강 입력이 네트워크에 입력되어 아웃라이어 탐지기의 예측을 얻습니다. 이 구조는 아웃라이어 탐지의 일관성을 높이는 데 기여합니다.
   - **Figure 2**: SSL과 OSSL의 차이를 설명합니다. SSL에서는 라벨이 있는 샘플로부터 신호를 전파하는 것이 가능하지만, OSSL에서는 아웃라이어가 라벨이 없기 때문에 신뢰할 수 없는 라벨이 생성될 수 있습니다. 이로 인해 OSSL에서는 부드러운 일관성 손실이 필요합니다.
   - **Figure 3**: 아웃라이어 탐지기의 점수 분포를 보여줍니다. SOCR을 사용하지 않을 경우 아웃라이어와 인라이어가 혼동되지만, SOCR을 사용하면 아웃라이어의 점수가 증가하고 분리가 향상됩니다.
   - **Figure 4**: T-SNE를 사용하여 피처 분포를 시각화합니다. OpenMatch는 인라이어와 아웃라이어를 잘 분리하는 반면, 라벨이 있는 샘플로만 훈련된 모델은 이들을 혼동합니다.

2. **테이블**
   - **Table 1**: CIFAR10 및 CIFAR100에서의 오류율을 보여줍니다. OpenMatch는 모든 경우에서 최첨단 성능을 달성하며, 특히 CIFAR10에서 50개의 라벨로 10%의 개선을 보였습니다.
   - **Table 2**: AUROC 값을 보여줍니다. OpenMatch는 모든 경우에서 AUROC에서 우수한 성능을 보이며, 특히 MTC와 비교하여 더 나은 성능을 나타냅니다.
   - **Table 3**: SOCR의 효과를 보여주는 수치 비교입니다. SOCR을 도입하면 AUROC가 CIFAR10에서 20% 이상 증가하는 등 아웃라이어 탐지 성능이 향상됩니다.
   - **Table 4**: 자가 감독 학습 손실이 OpenMatch에 미치는 영향을 보여줍니다. SimCLR 초기화가 성능에 큰 차이를 만들지 않음을 나타냅니다.
   - **Table 5**: 아웃라이어 탐지 성능을 보여주는 표입니다. OpenMatch는 OSSL 기준선보다 평균 5.8% 더 높은 성능을 보이며, 감독 모델보다도 3.4% 더 높은 성능을 나타냅니다.

3. **어펜딕스**
   - 어펜딕스에는 실험에 사용된 하이퍼파라미터와 추가적인 실험 결과가 포함되어 있습니다. 이는 연구의 재현성을 높이고, 다양한 설정에서의 성능을 비교하는 데 유용합니다.

---




1. **Diagrams and Figures**
   - **Figure 1**: Illustrates the structure of the open-set soft-consistency loss. Two differently augmented inputs are fed into the network to obtain predictions from the outlier detector. This structure contributes to enhancing the consistency of outlier detection.
   - **Figure 2**: Explains the difference between SSL and OSSL. In SSL, it is possible to propagate signals from labeled samples, but in OSSL, the absence of labels for outliers can lead to unreliable labels. This necessitates the use of soft consistency loss in OSSL.
   - **Figure 3**: Shows the score distribution of the outlier detector. Without using SOCR, inliers and outliers are confused, but using SOCR increases the scores of outliers and enhances separation.
   - **Figure 4**: Visualizes feature distributions using T-SNE. OpenMatch effectively separates inliers and outliers, while a model trained only with labeled samples confuses them.

2. **Tables**
   - **Table 1**: Displays error rates on CIFAR10 and CIFAR100. OpenMatch achieves state-of-the-art performance in all cases, with a notable 10% improvement on CIFAR10 with 50 labels.
   - **Table 2**: Shows AUROC values. OpenMatch demonstrates superior performance across all cases, particularly outperforming MTC.
   - **Table 3**: Numerical comparison demonstrating the effectiveness of SOCR. Introducing SOCR increases AUROC by over 20% in CIFAR10, indicating improved outlier detection performance.
   - **Table 4**: Compares the impact of self-supervised learning loss on OpenMatch. It shows that initializing with SimCLR does not significantly differ from random initialization.
   - **Table 5**: Displays outlier detection performance. OpenMatch outperforms OSSL baselines by an average of 5.8% and even the supervised model by 3.4%.

3. **Appendix**
   - The appendix includes hyperparameters used in experiments and additional experimental results. This enhances the reproducibility of the research and is useful for comparing performance across various settings.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@inproceedings{saito2021openmatch,
  title={OpenMatch: Open-set Consistency Regularization for Semi-supervised Learning with Outliers},
  author={Kuniaki Saito and Donghyun Kim and Kate Saenko},
  booktitle={Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS)},
  year={2021},
  organization={MIT-IBM Watson AI Lab, Boston University},
  url={https://github.com/VisionLearningGroup/OP_Match}
}
```

### 시카고 스타일 인용
Kuniaki Saito, Donghyun Kim, and Kate Saenko. "OpenMatch: Open-set Consistency Regularization for Semi-supervised Learning with Outliers." In *Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS)*, 2021. MIT-IBM Watson AI Lab, Boston University. https://github.com/VisionLearningGroup/OP_Match.
