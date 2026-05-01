---
layout: post
title:  "[2026]Memorization Through the Lens of Sample Gradients"
date:   2026-05-01 04:15:30 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 Cumulative Sample Gradient (CSG)를 제안하여 메모리제이션을 효율적으로 추정하는 방법을 소개합니다.


짧은 요약(Abstract) :


이 논문에서는 딥 뉴럴 네트워크가 종종 저대표성의 어려운 예제를 암기하는 경향이 있으며, 이는 일반화와 개인 정보 보호에 영향을 미친다고 설명합니다. Feldman & Zhang(2020)은 암기에 대한 엄격한 개념을 정의했지만, 이 점수를 계산하는 것은 데이터 포인트를 포함한 모델과 포함하지 않은 모델을 모두 훈련해야 하므로 비용이 많이 듭니다. 저자들은 덜 암기된 샘플이 훈련 초기에 학습되는 경향이 있고, 반면에 고도로 암기된 샘플은 나중에 학습된다는 점에 주목했습니다. 이를 바탕으로, Cumulative Sample Gradient (CSG)라는 계산적으로 효율적인 암기 프록시를 도입합니다. CSG는 훈련 과정에서 입력 샘플에 대한 손실의 기울기를 누적한 것입니다. CSG는 암기와 학습 시간에 대한 이론적 경계를 제공하며, 훈련 중 이러한 기울기를 추적하면 모델의 가중치 노름과 유사한 특성의 상승-정점-하강 궤적을 보여줍니다. 이로 인해 검증 세트 없이도 조기 중단 기준을 제공하며, CSG는 Feldman & Zhang(2020)의 암기 점수보다 최대 5배 더 효율적입니다. 또한, CSG는 잘못 레이블이 붙은 샘플 탐지 및 편향 발견과 같은 실제 데이터셋 진단에서 최첨단 성능을 달성합니다.



This paper discusses how deep neural networks often tend to memorize underrepresented, hard examples, which has implications for generalization and privacy. Feldman & Zhang (2020) defined a rigorous notion of memorization, but computing this score is prohibitively expensive as it requires training models both with and without the data point of interest. The authors observe that less memorized samples tend to be learned earlier in training, while highly memorized samples are learned later. Motivated by this observation, they introduce Cumulative Sample Gradient (CSG), a computationally efficient proxy for memorization. CSG is defined as the gradient of the loss with respect to input samples, accumulated over the course of training. It provides theoretical bounds on memorization and learning time, and tracking these gradients during training reveals a characteristic rise–peak–decline trajectory that mirrors the model's weight norm. This yields an early-stopping criterion that does not require a validation set, making CSG up to five orders of magnitude more efficient than the memorization score from Feldman & Zhang (2020). Additionally, CSG achieves state-of-the-art performance on practical dataset diagnostics, such as mislabeled-sample detection and bias discovery.


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


이 논문에서는 Cumulative Sample Gradient (CSG)라는 새로운 메모리 측정 방법을 제안합니다. CSG는 입력 샘플에 대한 손실의 그래디언트를 누적하여 계산되며, 이는 훈련 과정에서 각 샘플이 얼마나 잘 학습되었는지를 나타냅니다. CSG는 메모리 측정의 효율적인 대리자로, 기존의 메모리 점수보다 최대 5배 빠르며, 이전의 최첨단 메모리 대리자들보다도 140배 더 빠릅니다.

CSG의 주요 아이디어는 훈련 초기에는 덜 기억되는 샘플이 더 빨리 학습되고, 반대로 높은 메모리 점수를 가진 샘플은 훈련 후반에 학습된다는 관찰에 기반합니다. 이를 통해 CSG는 메모리와 학습 시간 간의 이론적 연결을 제공합니다. CSG는 훈련 중에 샘플 그래디언트를 추적하여, 샘플의 학습 상태를 평가하고, 이를 통해 조기 중단 기준을 설정할 수 있습니다. 즉, 가중치 노름의 피크에서 훈련을 중단함으로써, 검증 세트 없이도 최적의 성능을 달성할 수 있습니다.

CSG는 또한 Sample Gradient Assisted Loss (SGAL)라는 추가적인 프로시저를 통해 메모리와의 정렬을 더욱 개선합니다. SGAL은 샘플 손실을 누적하여 그래디언트가 최적의 중단 지점을 나타낼 때 훈련을 중단하는 방법입니다. 이 방법은 훈련의 10%에서 30%만으로도 강력한 성능을 발휘할 수 있도록 해줍니다.

마지막으로, CSG는 데이터셋의 편향을 발견하고, 잘못 레이블이 붙은 샘플을 식별하는 데 있어 최첨단 성능을 달성합니다. 이는 데이터 중심의 딥러닝 연구에 있어 실용적이고 이론적으로 기반이 있는 도구로 자리 잡을 수 있습니다.



This paper introduces a novel memory measurement method called Cumulative Sample Gradient (CSG). CSG is calculated by accumulating the gradients of the loss with respect to input samples, which indicates how well each sample is learned during the training process. CSG serves as an efficient proxy for measuring memorization, being up to five orders of magnitude faster than the memorization score from Feldman & Zhang (2020) and 140 times faster than previous state-of-the-art memorization proxies.

The key idea behind CSG is based on the observation that less memorized samples tend to be learned earlier in training, while highly memorized samples are learned later. This allows CSG to provide a theoretical link between memorization and learning time. By tracking sample gradients during training, CSG evaluates the learning status of samples and establishes an early stopping criterion. Specifically, it suggests stopping training at the peak of the weight norm, enabling optimal performance without the need for a validation set.

Additionally, CSG is complemented by a further procedure called Sample Gradient Assisted Loss (SGAL), which improves alignment with memorization. SGAL accumulates sample loss until the gradient indicates an optimal stopping point, allowing strong performance with only 10% to 30% of the training epochs.

Finally, CSG achieves state-of-the-art performance in identifying dataset biases and detecting mislabeled samples, positioning itself as a practical and theoretically grounded tool for data-centric deep learning research.


<br/>
# Results


이 논문에서는 Cumulative Sample Gradient (CSG)라는 새로운 메트릭을 제안하고, 이를 통해 딥러닝 모델의 메모리제이션을 효율적으로 측정하는 방법을 소개합니다. CSG는 기존의 메모리제이션 점수와 비교하여 최대 5배 빠르며, 140배 더 효율적인 계산 속도를 자랑합니다. 이 연구는 CIFAR-100 및 ImageNet 데이터셋을 사용하여 CSG의 성능을 평가했습니다.

#### 실험 결과
1. **경쟁 모델**: CSG는 기존의 메모리제이션 점수(Feldman & Zhang, 2020)와 비교하여 높은 상관관계를 보였습니다. CSG는 0.84의 코사인 유사도와 0.72의 피어슨 상관계수를 기록하여, 기존 메모리제이션 점수와 유사한 성능을 나타냈습니다.

2. **테스트 데이터**: CIFAR-100 및 ImageNet 데이터셋을 사용하여 실험을 진행했습니다. 특히, CIFAR-100에서는 Inception 모델을, ImageNet에서는 ResNet50 모델을 사용했습니다.

3. **메트릭**: CSG는 메모리제이션 점수와의 상관관계 외에도, 잘못 레이블이 붙은 샘플을 탐지하는 데 있어 최첨단 성능을 보였습니다. CSG는 5%에서 30%의 레이블 노이즈가 있는 CIFAR-10 및 CIFAR-100 데이터셋에서 잘못 레이블이 붙은 샘플을 탐지하는 데 있어 모든 기준선 모델을 초과하는 성능을 보였습니다.

4. **비교**: CSG는 Curvature (Garg et al., 2024) 및 CSL (Ravikumar et al., 2025a)와 같은 기존의 메모리제이션 프록시와 비교하여 각각 140배 및 10배 더 빠른 성능을 보였습니다. CSG는 또한 잘못 레이블이 붙은 샘플 탐지에서 0.9896의 AUROC를 기록하여, 기존 방법들보다 우수한 성능을 입증했습니다.

5. **결론**: CSG는 메모리제이션을 측정하는 데 있어 매우 효율적이고 신뢰할 수 있는 도구로 자리 잡을 수 있으며, 데이터 중심의 딥러닝 연구 및 응용에 있어 중요한 기여를 할 것으로 기대됩니다.

---



This paper introduces a new metric called Cumulative Sample Gradient (CSG) and presents an efficient method for measuring the memorization of deep learning models. CSG boasts a computation speed that is up to five orders of magnitude faster than traditional memorization scores, and it is 140 times more efficient. The study evaluates the performance of CSG using the CIFAR-100 and ImageNet datasets.

#### Experimental Results
1. **Competing Models**: CSG demonstrated a high correlation with existing memorization scores (Feldman & Zhang, 2020). It achieved a cosine similarity of 0.84 and a Pearson correlation coefficient of 0.72, indicating performance comparable to traditional memorization scores.

2. **Test Data**: Experiments were conducted using the CIFAR-100 and ImageNet datasets. Specifically, the Inception model was used for CIFAR-100, while the ResNet50 model was employed for ImageNet.

3. **Metrics**: In addition to its correlation with memorization scores, CSG exhibited state-of-the-art performance in detecting mislabeled samples. It outperformed all baseline models in detecting mislabeled samples on CIFAR-10 and CIFAR-100 datasets with 5% to 30% label noise.

4. **Comparison**: CSG was shown to be 140 times faster than Curvature (Garg et al., 2024) and 10 times faster than CSL (Ravikumar et al., 2025a), two existing memorization proxies. CSG also achieved an AUROC of 0.9896 in mislabeled sample detection, surpassing the performance of previous methods.

5. **Conclusion**: CSG is expected to become a highly efficient and reliable tool for measuring memorization, making significant contributions to data-centric deep learning research and applications.


<br/>
# 예제


이 논문에서는 Cumulative Sample Gradient (CSG)라는 새로운 메트릭을 제안하여 딥러닝 모델의 메모리제이션을 측정하는 방법을 설명합니다. CSG는 훈련 과정에서 각 샘플에 대한 손실의 그래디언트를 누적하여 계산됩니다. 이 메트릭은 메모리제이션과 학습 시간 간의 이론적 연결을 제공하며, 실험적으로도 메모리제이션 점수와 높은 상관관계를 보입니다.

#### 예시 설명

1. **데이터셋**: CIFAR-100 데이터셋을 사용합니다. 이 데이터셋은 100개의 클래스로 구성된 60,000개의 32x32 픽셀 컬러 이미지로 이루어져 있습니다. 각 클래스는 600개의 이미지로 구성되어 있습니다.

2. **훈련 데이터와 테스트 데이터**: 
   - 훈련 데이터: CIFAR-100의 50,000개의 이미지를 훈련 데이터로 사용합니다. 예를 들어, '사슴' 클래스의 이미지는 600개 중 일부가 훈련 데이터로 사용됩니다.
   - 테스트 데이터: 나머지 10,000개의 이미지는 테스트 데이터로 사용되며, 모델의 성능을 평가하는 데 사용됩니다.

3. **구체적인 인풋과 아웃풋**:
   - 인풋: 모델에 입력되는 이미지는 32x32 픽셀 크기의 RGB 이미지입니다. 예를 들어, '사슴' 이미지는 모델에 입력될 때 (32, 32, 3) 형태의 텐서로 변환됩니다.
   - 아웃풋: 모델의 출력은 각 클래스에 대한 확률 분포입니다. 예를 들어, '사슴' 클래스에 대한 확률이 0.85라면, 모델은 해당 이미지가 '사슴'일 확률이 85%라고 예측합니다.

4. **테스크**: 모델의 목표는 주어진 이미지를 올바른 클래스(예: '사슴', '자동차', '비행기' 등)로 분류하는 것입니다. 훈련 과정에서 모델은 각 이미지에 대한 손실을 계산하고, 이 손실을 최소화하기 위해 가중치를 업데이트합니다.

5. **CSG의 활용**: CSG는 훈련 과정에서 각 샘플의 그래디언트를 누적하여 계산되며, 이를 통해 모델이 특정 샘플을 얼마나 잘 기억하는지를 평가합니다. 예를 들어, '사슴' 클래스의 이미지가 훈련 초기에 잘 학습되면 CSG 값이 낮고, 후기에 잘 학습되면 CSG 값이 높아집니다. 이를 통해 모델이 메모리제이션을 어떻게 수행하는지를 분석할 수 있습니다.



This paper introduces a new metric called Cumulative Sample Gradient (CSG) to measure the memorization of deep learning models. CSG is calculated by accumulating the gradients of the loss with respect to each sample during the training process. This metric provides a theoretical link between memorization and learning time, and it has been experimentally shown to have a high correlation with memorization scores.

#### Example Explanation

1. **Dataset**: The CIFAR-100 dataset is used. This dataset consists of 60,000 32x32 pixel color images divided into 100 classes. Each class contains 600 images.

2. **Training and Testing Data**: 
   - Training Data: 50,000 images from CIFAR-100 are used as training data. For example, some images from the 'deer' class, which has 600 images, are included in the training set.
   - Testing Data: The remaining 10,000 images are used as testing data to evaluate the model's performance.

3. **Specific Inputs and Outputs**:
   - Input: The images fed into the model are 32x32 pixel RGB images. For instance, a 'deer' image is transformed into a tensor of shape (32, 32, 3) when inputted into the model.
   - Output: The model's output is a probability distribution over each class. For example, if the probability for the 'deer' class is 0.85, the model predicts that there is an 85% chance that the input image is a 'deer'.

4. **Task**: The goal of the model is to classify the given image into the correct class (e.g., 'deer', 'car', 'airplane', etc.). During the training process, the model calculates the loss for each image and updates its weights to minimize this loss.

5. **Utilization of CSG**: CSG is calculated by accumulating the gradients for each sample during training, allowing the evaluation of how well the model memorizes specific samples. For example, if an image from the 'deer' class is learned well early in training, the CSG value will be low, while if it is learned well later, the CSG value will be high. This analysis helps understand how the model performs memorization.

<br/>
# 요약


이 논문에서는 Cumulative Sample Gradient (CSG)를 제안하여 메모리제이션을 효율적으로 추정하는 방법을 소개합니다. CSG는 훈련 과정에서 입력 샘플에 대한 손실의 그래디언트를 누적하여 계산되며, 기존 메모리제이션 점수보다 최대 5배 빠른 성능을 보입니다. 실험 결과, CSG는 잘못 레이블이 붙은 샘플을 탐지하고 데이터 편향을 발견하는 데 있어 최첨단 성능을 달성했습니다.

---

This paper introduces the Cumulative Sample Gradient (CSG) as an efficient method for estimating memorization. CSG is computed by accumulating the gradients of the loss with respect to input samples during training, achieving up to five times faster performance than existing memorization scores. Experimental results demonstrate that CSG achieves state-of-the-art performance in detecting mislabeled samples and uncovering data biases.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: CSG의 낮은 값과 높은 값의 이미지를 보여줍니다. 낮은 CSG 이미지는 클래스의 전형적인 예시(쉬운 예시)를 나타내고, 높은 CSG 이미지는 비전형적인 예시(어려운 예시)를 나타냅니다. 이는 CSG가 샘플의 난이도를 잘 포착하고 있음을 보여줍니다.
   - **Figure 2**: 입력 그래디언트 노름을 시각화하여 쉬운 샘플과 어려운 샘플의 학습 과정을 비교합니다. 쉬운 샘플은 빠르게 학습되고 낮은 그래디언트를 유지하는 반면, 어려운 샘플은 높은 그래디언트를 오랫동안 유지합니다. 이는 CSG가 샘플의 난이도를 구별하는 데 유용하다는 것을 나타냅니다.
   - **Figure 3**: 평균 샘플 그래디언트와 검증 손실의 관계를 보여줍니다. 샘플 그래디언트의 피크는 검증 손실의 최소값과 일치하며, 이는 조기 중단 기준을 설정하는 데 유용합니다.
   - **Figure 5**: CSG 분석을 통해 FM-NIST 데이터셋의 편향을 발견한 사례를 보여줍니다. 어두운 의류와 낮은 대비의 샘플이 높은 CSG로 식별되는 경향이 있습니다.
   - **Figure 16**: ImageNet에서 CSG가 낮은 이미지와 높은 이미지를 보여줍니다. 이는 CSG가 샘플의 난이도를 잘 포착하고 있음을 다시 한번 강조합니다.

2. **테이블**
   - **Table 1**: CSG와 다른 메모리제이션 프로시의 유사성을 비교합니다. CSG는 메모리제이션 점수와 높은 상관관계를 보이며, 계산 비용이 매우 낮습니다. 이는 CSG가 메모리제이션을 효과적으로 추정할 수 있는 강력한 도구임을 나타냅니다.
   - **Table 2**: CIFAR-100 ResNet18 체크포인트의 캘리브레이션 결과를 보여줍니다. CSG를 사용한 조기 중단이 마지막 에포크 체크포인트보다 더 나은 캘리브레이션 성능을 보입니다.
   - **Table 4**: CIFAR-10 및 CIFAR-100에서 잘못 레이블이 붙은 샘플을 감지하는 성능을 비교합니다. CSG는 모든 노이즈 수준에서 최고의 성능을 보이며, 이는 CSG가 잘못된 레이블을 효과적으로 감지할 수 있는 도구임을 나타냅니다.

3. **어펜딕스**
   - 어펜딕스에서는 실험 세부사항, 추가 결과, 이론적 증명 등을 제공합니다. 특히, CSG의 이론적 기초와 메모리제이션 및 학습 시간과의 관계를 설명하는 증명이 포함되어 있습니다. 이는 CSG가 메모리제이션을 추정하는 데 있어 신뢰할 수 있는 방법임을 뒷받침합니다.

---




1. **Figures and Diagrams**
   - **Figure 1**: Displays images with low and high CSG values. Low CSG images represent typical examples for the class (easy examples), while high CSG images represent atypical examples (hard examples). This indicates that CSG effectively captures the difficulty of samples.
   - **Figure 2**: Visualizes the input gradient norms for easy and hard samples, showing that easy samples are learned quickly and maintain low gradients, while hard samples retain high gradients for longer. This suggests that CSG is useful for distinguishing sample difficulty.
   - **Figure 3**: Illustrates the relationship between average sample gradient and validation loss. The peak in sample gradient corresponds to the minimum validation loss, providing a useful criterion for early stopping.
   - **Figure 5**: Shows a case where CSG analysis uncovers bias in the FM-NIST dataset, indicating that darker clothing with lower contrast is often identified as high CSG (i.e., harder).
   - **Figure 16**: Displays low and high CSG images from ImageNet, reinforcing that CSG effectively captures sample difficulty.

2. **Tables**
   - **Table 1**: Compares CSG with other memorization proxies, showing that CSG has a high correlation with memorization scores while being computationally efficient. This indicates that CSG is a powerful tool for estimating memorization.
   - **Table 2**: Presents calibration results for CIFAR-100 ResNet18 checkpoints, demonstrating that early stopping using CSG leads to better calibration performance compared to the last epoch checkpoint.
   - **Table 4**: Compares performance in detecting mislabeled samples on CIFAR-10 and CIFAR-100, showing that CSG outperforms all baselines at various noise levels, indicating its effectiveness in identifying mislabeled data.

3. **Appendix**
   - The appendix provides experimental details, additional results, and theoretical proofs. It includes proofs that establish the theoretical foundation of CSG and its relationship with memorization and learning time, supporting the reliability of CSG as a method for estimating memorization.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Ravikumar2026,
  author = {Deepak Ravikumar and Efstathia Soufleri and Abolfazl Hashemi and Kaushik Roy},
  title = {Memorization Through the Lens of Sample Gradients},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year = {2026},
  address = {New Orleans, LA, USA},
  url = {https://github.com/DeepakTatachar/Sample-Gradient-Memorization}
}
```

### 시카고 스타일

Deepak Ravikumar, Efstathia Soufleri, Abolfazl Hashemi, and Kaushik Roy. 2026. "Memorization Through the Lens of Sample Gradients." In *Proceedings of the International Conference on Learning Representations (ICLR)*. New Orleans, LA, USA. https://github.com/DeepakTatachar/Sample-Gradient-Memorization.
