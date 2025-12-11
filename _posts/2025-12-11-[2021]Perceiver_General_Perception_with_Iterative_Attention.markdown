---
layout: post
title:  "[2021]Perceiver: General Perception with Iterative Attention"
date:   2025-12-11 17:57:38 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: Perceiver 모델은 크로스 어텐션 모듈과 트랜스포머를 결합하여 다양한 입력 모달리티를 처리할 수 있도록 설계되었으며, 이를 통해 입력 크기와 네트워크 깊이를 분리하여 효율성을 높였다.


짧은 요약(Abstract) :


이 논문에서는 생물학적 시스템이 시각, 청각, 촉각, 고유감각 등 다양한 고차원 입력을 동시에 처리하여 세상을 인식하는 방식을 모방한 '퍼시버(Perceiver)'라는 모델을 소개합니다. 기존의 딥러닝 인식 모델은 개별적인 모달리티에 맞춰 설계되어 있으며, 일반적으로 2D 격자 구조와 같은 도메인 특화 가정을 기반으로 합니다. 이러한 가정은 유용한 유도 편향을 제공하지만, 모델을 개별 모달리티에 고정시키는 단점이 있습니다. 퍼시버는 트랜스포머를 기반으로 하여 입력 간의 관계에 대한 구조적 가정을 최소화하고, 수십만 개의 입력을 처리할 수 있도록 확장할 수 있는 모델입니다. 이 모델은 비대칭 주의 메커니즘을 활용하여 입력을 반복적으로 정제하여 밀집된 잠재 병목으로 변환함으로써 매우 큰 입력을 처리할 수 있도록 합니다. 우리는 이 아키텍처가 이미지, 포인트 클라우드, 오디오, 비디오 및 이들의 조합과 같은 다양한 모달리티의 분류 작업에서 강력한 전문 모델과 경쟁하거나 이를 초월하는 성능을 보인다는 것을 보여줍니다. 퍼시버는 50,000개의 픽셀에 직접 주의를 기울여 2D 합성곱 없이도 ImageNet에서 ResNet-50 및 ViT와 유사한 성능을 달성합니다. 또한 AudioSet의 모든 모달리티에서도 경쟁력 있는 성능을 보입니다.



This paper introduces the Perceiver, a model that mimics how biological systems perceive the world by simultaneously processing high-dimensional inputs from diverse modalities such as vision, audition, touch, and proprioception. In contrast, existing deep learning perception models are designed for individual modalities and often rely on domain-specific assumptions, such as the local grid structures exploited by virtually all existing vision models. These priors provide helpful inductive biases but also lock models to individual modalities. The Perceiver builds upon Transformers, making few architectural assumptions about the relationships between its inputs while also scaling to handle hundreds of thousands of inputs. The model leverages an asymmetric attention mechanism to iteratively distill inputs into a tight latent bottleneck, allowing it to scale to handle very large inputs. We demonstrate that this architecture is competitive with or outperforms strong, specialized models on classification tasks across various modalities: images, point clouds, audio, video, and video+audio. The Perceiver achieves performance comparable to ResNet-50 and ViT on ImageNet without 2D convolutions by directly attending to 50,000 pixels. It is also competitive across all modalities in AudioSet.


* Useful sentences :


Set Transformer (특히 PMA)와 천천히 비교해보자면 우선 셋트랜스포머 구조는 집합(Set) 연산에 특화되어 있어, 여러 요소가 합쳐질 때(Pooling) 그 요소들의 **밀도(Density)나 개수(Count)**가 자연스럽게 결과값에 반영되도록 설계하기 쉬움.
Perceiver는 "압축"이 목적으로 입력이 100만 개라도 결과는 512개. 즉, **"개수(Count) 정보를 잃어버리기 딱 좋은 구조"**입니다. 100만 개 중 90만 개가 같은 조각이라도, Latent가 꽉 차면 그 비율을 정확히 대변하지 못할 수 있음..  

뭘 써야하느냐 고민해본다면:  
"쓰레기 더미 속에서 보물찾기"라면 → Perceiver
엄청나게 많은 파편(Abundance) 중에, 의미 있는 아주 소수의 패턴을 찾아내야 한다면 Perceiver의 '반복 탐색(Iterative Attention)'이 유리    

"재료의 배합 비율 맞추기"라면 → Set Transformer
파편들이 얼마나 많이(Count), 어떤 비율로(Ratio) 존재하는지가 그 샘플의 성격을 결정한다면, Perceiver는 그 통계 정보를 압축하다가 잃어버릴 수 있음. 반대 의견대로 Set Transformer가 훨씬 유리   


{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
*


<br/>
# Methodology



**Perceiver 아키텍처**

Perceiver는 다양한 입력 모달리티를 처리할 수 있도록 설계된 모델로, Transformer 아키텍처를 기반으로 합니다. 이 모델은 입력 데이터의 차원에 대한 가정이 적고, 수백만 개의 입력을 처리할 수 있는 능력을 가지고 있습니다. Perceiver의 핵심 아이디어는 입력을 작은 잠재 공간(latent space)으로 압축하는 비대칭적 주의(attention) 메커니즘을 도입하는 것입니다. 이를 통해 입력의 크기와 네트워크의 깊이를 분리하여, 매우 깊은 모델을 구축할 수 있습니다.

**모델 구성**

Perceiver는 두 가지 주요 구성 요소로 이루어져 있습니다: 
1. **크로스 어텐션 모듈(Cross-Attention Module)**: 이 모듈은 고차원 입력 배열(예: 이미지의 픽셀 배열)을 잠재 배열로 매핑합니다. 이 과정에서 입력의 차원 수는 줄어들지만, 정보는 유지됩니다.
2. **Transformer 타워(Transformer Tower)**: 이 모듈은 잠재 배열을 다시 잠재 배열로 매핑합니다. 이 두 모듈은 번갈아 가며 적용되어 입력을 처리합니다.

**트레이닝 데이터 및 기법**

Perceiver는 다양한 데이터셋에서 훈련될 수 있으며, 특히 이미지, 비디오, 오디오 및 포인트 클라우드와 같은 고차원 데이터를 처리하는 데 강점을 보입니다. 예를 들어, ImageNet 데이터셋에서 Perceiver는 50,176개의 픽셀을 직접 처리하여 ResNet-50 및 ViT와 유사한 성능을 달성했습니다. 또한, AudioSet 데이터셋에서도 경쟁력 있는 성능을 보였습니다.

Perceiver는 위치 인코딩을 사용하여 입력의 공간적 또는 시간적 정보를 보존합니다. Fourier 특징을 사용한 위치 인코딩은 입력 데이터의 구조를 직접적으로 표현할 수 있도록 도와줍니다. 이 모델은 또한 여러 번의 크로스 어텐션을 통해 입력의 정보를 반복적으로 추출하여, 필요한 정보를 점진적으로 수집할 수 있습니다.

**결론**

Perceiver는 다양한 입력 모달리티를 처리할 수 있는 유연한 아키텍처로, 기존의 도메인 특화된 모델들과는 달리, 입력 데이터의 구조에 대한 가정이 적습니다. 이는 Perceiver가 다양한 데이터셋에서 강력한 성능을 발휘할 수 있도록 합니다.

---



**Perceiver Architecture**

The Perceiver is a model designed to handle various input modalities, built upon the Transformer architecture. This model makes few assumptions about the dimensionality of the input data and has the capability to process millions of inputs. The core idea of the Perceiver is to introduce an asymmetric attention mechanism that compresses the inputs into a small latent space. This allows for the decoupling of the input size from the depth of the network, enabling the construction of very deep models.

**Model Composition**

The Perceiver consists of two main components:
1. **Cross-Attention Module**: This module maps a high-dimensional input array (e.g., pixel array of an image) to a latent array. During this process, the dimensionality of the input is reduced while preserving the information.
2. **Transformer Tower**: This module maps the latent array back to another latent array. These two modules are applied alternately to process the input.

**Training Data and Techniques**

The Perceiver can be trained on various datasets, particularly excelling in processing high-dimensional data such as images, videos, audio, and point clouds. For instance, on the ImageNet dataset, the Perceiver directly processed 50,176 pixels to achieve performance comparable to ResNet-50 and ViT. It also demonstrated competitive performance on the AudioSet dataset.

The Perceiver employs position encodings to retain spatial or temporal information of the inputs. The use of Fourier features for position encoding helps to directly represent the structure of the input data. Additionally, the model iteratively extracts information from the inputs through multiple cross-attention layers, gradually collecting the necessary information.

**Conclusion**

The Perceiver is a flexible architecture capable of handling various input modalities, differing from traditional domain-specific models by making fewer assumptions about the structure of the input data. This flexibility allows the Perceiver to achieve strong performance across diverse datasets.


<br/>
# Results



이 논문에서는 Perceiver라는 새로운 모델을 소개하며, 이 모델이 다양한 입력 모달리티(이미지, 오디오, 비디오 등)에서 어떻게 경쟁력 있는 성능을 발휘하는지를 보여줍니다. Perceiver는 Transformer 아키텍처를 기반으로 하며, 입력 데이터의 구조에 대한 도메인 특화 가정을 최소화하여 다양한 입력을 처리할 수 있는 유연성을 제공합니다.

#### 1. 경쟁 모델
Perceiver는 여러 경쟁 모델과 비교되었습니다. 예를 들어, 이미지 분류 작업에서는 ResNet-50과 Vision Transformer(ViT)와 같은 강력한 모델과 성능을 비교했습니다. Perceiver는 ImageNet 데이터셋에서 ResNet-50과 ViT에 비해 경쟁력 있는 성능을 보였으며, 특히 50,176개의 픽셀에 직접 주의를 기울여 78.0%의 Top-1 정확도를 달성했습니다. 이는 ResNet-50의 77.6% 및 ViT-B-16의 77.9%와 유사한 성능입니다.

#### 2. 테스트 데이터
Perceiver는 다양한 데이터셋에서 테스트되었습니다. 주요 데이터셋으로는 ImageNet, AudioSet, ModelNet40 등이 있습니다. ImageNet은 이미지 분류를 위한 데이터셋으로, 1,000개의 클래스가 포함되어 있습니다. AudioSet은 1.7M 개의 10초 길이의 비디오로 구성된 데이터셋으로, 다양한 오디오 이벤트를 분류하는 데 사용됩니다. ModelNet40은 3D 포인트 클라우드 데이터셋으로, 40개의 객체 카테고리를 포함하고 있습니다.

#### 3. 메트릭
모델의 성능은 주로 Top-1 정확도와 평균 정밀도(mAP)로 평가되었습니다. ImageNet에서는 Top-1 정확도를 사용하여 모델의 분류 성능을 측정하였고, AudioSet에서는 mAP를 사용하여 오디오 이벤트 분류 성능을 평가했습니다. ModelNet40에서는 Top-1 정확도를 사용하여 포인트 클라우드 분류 성능을 측정했습니다.

#### 4. 비교
Perceiver는 각 데이터셋에서 다른 모델들과 비교하여 성능을 평가했습니다. 예를 들어, ImageNet에서 Perceiver는 78.0%의 Top-1 정확도를 기록하여 ResNet-50(77.6%) 및 ViT-B-16(77.9%)과 유사한 성능을 보였습니다. AudioSet에서는 Perceiver가 38.4%의 mAP를 기록하여 CNN-14(43.1%)와 같은 다른 모델들과 비교할 수 있는 성능을 보였습니다. ModelNet40에서는 Perceiver가 85.7%의 정확도로 PointNet++(91.9%)에 비해 경쟁력 있는 성능을 보였습니다.

이러한 결과들은 Perceiver가 다양한 입력 모달리티에서 강력한 성능을 발휘할 수 있는 능력을 보여주며, 도메인 특화 가정 없이도 효과적으로 작동할 수 있음을 시사합니다.

---




This paper introduces a new model called Perceiver and demonstrates how it achieves competitive performance across various input modalities (images, audio, video, etc.). The Perceiver is based on the Transformer architecture and minimizes domain-specific assumptions about the structure of input data, allowing for flexibility in processing diverse inputs.

#### 1. Competing Models
The Perceiver was compared with several competing models. For instance, in image classification tasks, it was compared to strong models like ResNet-50 and Vision Transformer (ViT). The Perceiver achieved a Top-1 accuracy of 78.0% on the ImageNet dataset, which is competitive with ResNet-50's 77.6% and ViT-B-16's 77.9%, particularly by directly attending to 50,176 pixels.

#### 2. Test Data
The Perceiver was tested on various datasets, including ImageNet, AudioSet, and ModelNet40. ImageNet is a dataset for image classification containing 1,000 classes. AudioSet consists of 1.7 million 10-second long videos used for classifying various audio events. ModelNet40 is a 3D point cloud dataset that includes 40 object categories.

#### 3. Metrics
The performance of the models was primarily evaluated using Top-1 accuracy and mean average precision (mAP). For ImageNet, Top-1 accuracy was used to measure the classification performance of the models, while mAP was used for evaluating audio event classification performance on AudioSet. For ModelNet40, Top-1 accuracy was used to measure point cloud classification performance.

#### 4. Comparison
The Perceiver's performance was assessed by comparing it with other models on each dataset. For example, on ImageNet, the Perceiver recorded a Top-1 accuracy of 78.0%, which is similar to ResNet-50 (77.6%) and ViT-B-16 (77.9%). On AudioSet, the Perceiver achieved a mAP of 38.4%, demonstrating competitive performance compared to other models like CNN-14 (43.1%). On ModelNet40, the Perceiver achieved an accuracy of 85.7%, showing competitive performance against PointNet++ (91.9%).

These results indicate that the Perceiver can deliver strong performance across various input modalities and operate effectively without domain-specific assumptions.


<br/>
# 예제



**Perceiver 모델의 훈련 및 테스트 데이터 예시**

1. **훈련 데이터 (Training Data)**:
   - **입력 (Input)**: Perceiver 모델은 다양한 모달리티의 데이터를 처리할 수 있습니다. 예를 들어, 이미지 분류 작업을 위해 ImageNet 데이터셋을 사용할 수 있습니다. 이 데이터셋은 1,000개의 클래스에 대한 1,000,000개 이상의 이미지를 포함하고 있습니다. 각 이미지는 224x224 픽셀 크기로 리사이즈되고, RGB 색상 채널을 가집니다.
   - **출력 (Output)**: 모델의 출력은 각 이미지에 대한 클래스 확률입니다. 예를 들어, 특정 이미지가 '고양이'일 확률이 0.85, '개'일 확률이 0.10, '새'일 확률이 0.05일 수 있습니다. 이 확률은 소프트맥스 함수를 통해 계산됩니다.

2. **테스트 데이터 (Test Data)**:
   - **입력 (Input)**: 테스트 데이터는 훈련 데이터와 동일한 형식의 이미지로 구성됩니다. 예를 들어, 테스트 세트에서 새로운 이미지가 주어질 수 있습니다. 이 이미지는 훈련 데이터와 같은 방식으로 전처리되어야 합니다.
   - **출력 (Output)**: 테스트 데이터에 대한 출력은 모델이 예측한 클래스입니다. 예를 들어, 모델이 테스트 이미지에 대해 '고양이'로 예측했다면, 이는 모델이 해당 이미지를 '고양이'로 분류했음을 의미합니다. 모델의 성능은 정확도(accuracy)로 평가되며, 이는 올바르게 분류된 이미지의 비율로 계산됩니다.

3. **구체적인 작업 (Specific Task)**:
   - **작업 설명**: Perceiver 모델은 이미지 분류 작업을 수행합니다. 주어진 이미지가 어떤 클래스에 속하는지를 예측하는 것이 목표입니다. 예를 들어, '고양이', '개', '새', '자동차' 등 다양한 클래스 중 하나로 이미지를 분류합니다.
   - **훈련 과정**: 모델은 훈련 데이터셋을 사용하여 각 이미지에 대한 정답 레이블(클래스)을 학습합니다. 이 과정에서 손실 함수(예: 크로스 엔트로피 손실)를 최소화하는 방향으로 가중치를 업데이트합니다.

### English Version




1. **Training Data**:
   - **Input**: The Perceiver model can process data from various modalities. For instance, for an image classification task, the ImageNet dataset can be used. This dataset contains over 1,000,000 images across 1,000 classes. Each image is resized to a size of 224x224 pixels and has RGB color channels.
   - **Output**: The model's output is the class probabilities for each image. For example, for a specific image, the probability of it being a 'cat' might be 0.85, 'dog' 0.10, and 'bird' 0.05. These probabilities are computed using a softmax function.

2. **Testing Data**:
   - **Input**: The test data consists of images in the same format as the training data. For example, a new image may be presented from the test set. This image should be preprocessed in the same way as the training data.
   - **Output**: The output for the test data is the class predicted by the model. For instance, if the model predicts 'cat' for the test image, it means the model classified that image as 'cat'. The model's performance is evaluated based on accuracy, which is calculated as the ratio of correctly classified images.

3. **Specific Task**:
   - **Task Description**: The Perceiver model performs an image classification task. The goal is to predict which class a given image belongs to. For example, it classifies images into various classes such as 'cat', 'dog', 'bird', 'car', etc.
   - **Training Process**: The model learns from the training dataset, which includes the correct labels (classes) for each image. During this process, the weights are updated in a way that minimizes a loss function (e.g., cross-entropy loss).

<br/>
# 요약


Perceiver 모델은 크로스 어텐션 모듈과 트랜스포머를 결합하여 다양한 입력 모달리티를 처리할 수 있도록 설계되었으며, 이를 통해 입력 크기와 네트워크 깊이를 분리하여 효율성을 높였다. 실험 결과, Perceiver는 ImageNet, AudioSet, ModelNet40 등 여러 데이터셋에서 기존의 강력한 모델들과 경쟁할 수 있는 성능을 보였다. 특히, Perceiver는 2D 컨볼루션을 사용하지 않고도 ImageNet에서 ResNet-50과 유사한 성능을 달성하였다.

---

The Perceiver model is designed to handle various input modalities by combining cross-attention modules and transformers, increasing efficiency by decoupling input size from network depth. Experimental results show that the Perceiver performs competitively with existing strong models across multiple datasets, including ImageNet, AudioSet, and ModelNet40. Notably, the Perceiver achieves performance comparable to ResNet-50 on ImageNet without using 2D convolutions.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: Perceiver 아키텍처의 구조를 보여주며, 입력 데이터가 어떻게 고차원 배열에서 잠재적 배열로 변환되는지를 설명합니다. 이 구조는 다양한 입력 모달리티를 처리할 수 있는 유연성을 제공합니다.
   - **Figure 3**: ImageNet 데이터셋에서의 주의 맵을 시각화한 것으로, 모델이 입력 이미지의 특정 부분에 어떻게 주의를 기울이는지를 보여줍니다. 초기 모듈에서는 이미지의 주요 특징이 강조되며, 후속 모듈에서는 더 복잡한 패턴이 나타납니다.
   - **Figure 6**: 초기화 스케일과 Fourier 주파수 위치 인코딩 매개변수가 성능에 미치는 영향을 보여줍니다. 더 많은 주파수 대역과 높은 최대 해상도가 성능을 향상시키는 경향이 있음을 나타냅니다.

2. **테이블**
   - **Table 1**: ImageNet에서의 Perceiver 모델의 성능을 다른 모델들과 비교합니다. Perceiver는 ResNet-50 및 ViT와 유사한 성능을 보이며, 도메인 특화된 구조에 의존하지 않고도 경쟁력 있는 결과를 보여줍니다.
   - **Table 3**: AudioSet에서의 Perceiver 성능을 다른 최신 모델들과 비교합니다. Perceiver는 오디오 및 비디오 입력 모두에서 강력한 성능을 보이며, 멜 스펙트로그램을 사용할 때도 경쟁력 있는 결과를 나타냅니다.
   - **Table 4**: ModelNet40 데이터셋에서의 Perceiver 성능을 다른 모델들과 비교합니다. Perceiver는 포인트 클라우드 데이터에 대해 높은 정확도를 달성하며, 이는 고정된 격자 구조에 의존하지 않는 모델의 유연성을 보여줍니다.

3. **어펜딕스**
   - **Appendix A**: Perceiver 아키텍처의 효율성에 대한 추가적인 논의가 포함되어 있으며, Set Transformer와의 관계를 설명합니다. Perceiver는 입력 크기와 깊이를 분리하여 더 깊은 네트워크를 구축할 수 있는 장점을 가지고 있습니다.
   - **Appendix B**: 다양한 하이퍼파라미터의 영향을 평가한 실험 결과를 포함하고 있습니다. 모델 크기와 깊이를 증가시키는 것이 성능 향상에 기여하는 경향이 있음을 보여줍니다.
   - **Appendix D**: Fourier 주파수 위치 인코딩의 매개변수화에 대한 설명이 포함되어 있으며, 이는 입력 신호의 샘플링 주파수에 따라 조정될 수 있습니다.





1. **Diagrams and Figures**
   - **Figure 1**: Illustrates the structure of the Perceiver architecture, showing how input data is transformed from high-dimensional arrays to latent arrays. This structure provides flexibility to handle various input modalities.
   - **Figure 3**: Visualizes attention maps from the ImageNet dataset, demonstrating how the model focuses on specific parts of the input image. Initial modules highlight key features of the image, while subsequent modules reveal more complex patterns.
   - **Figure 6**: Shows the impact of initialization scale and Fourier frequency position encoding parameters on performance. It indicates that a higher number of frequency bands and maximum resolution tend to improve performance.

2. **Tables**
   - **Table 1**: Compares the performance of the Perceiver model on ImageNet with other models. The Perceiver shows competitive results similar to ResNet-50 and ViT, demonstrating effectiveness without relying on domain-specific structures.
   - **Table 3**: Compares the performance of the Perceiver on AudioSet with other state-of-the-art models. The Perceiver achieves strong performance on both audio and video inputs, showing competitive results even when using mel spectrograms.
   - **Table 4**: Compares the performance of the Perceiver on the ModelNet40 dataset with other models. The Perceiver achieves high accuracy on point cloud data, showcasing the flexibility of models that do not rely on fixed grid structures.

3. **Appendix**
   - **Appendix A**: Contains additional discussions on the efficiency of the Perceiver architecture and its relationship with the Set Transformer. The Perceiver's ability to decouple input size and depth allows for the construction of deeper networks.
   - **Appendix B**: Includes results from experiments evaluating the effects of various hyperparameters. It shows that increasing model size and depth tends to contribute to performance improvements.
   - **Appendix D**: Provides explanations of the parameterization of Fourier frequency position encodings, which can be adjusted based on the sampling frequency of the input signal.

<br/>
# refer format:



### BibTeX 
```bibtex
@inproceedings{jaegle2021perceiver,
  title={Perceiver: General Perception with Iterative Attention},
  author={Andrew Jaegle and Felix Gimeno and Andrew Brock and Andrew Zisserman and Oriol Vinyals and Joao Carreira},
  booktitle={Proceedings of the 38th International Conference on Machine Learning},
  pages={139--150},
  year={2021},
  publisher={PMLR},
  url={https://arxiv.org/abs/2103.03206}
}
```

### 시카고 스타일 인용
Jaegle, Andrew, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, and Joao Carreira. 2021. "Perceiver: General Perception with Iterative Attention." In *Proceedings of the 38th International Conference on Machine Learning*, 139–150. PMLR. https://arxiv.org/abs/2103.03206.
