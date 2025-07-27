---
layout: post
title:  "[2025]An analytic theory of creativity in convolutional diffusion models"  
date:   2025-07-27 02:40:40 +0200
categories: study
---

{% highlight ruby %}


한줄 요약: 


CNN 기반 디퓨전(확산) 모델이 훈련 데이터의 조각들을 조합해 새롭고 창의적인 이미지를 만들어내는 과정을 수학적으로 설명하고 예측할 수 있는 이론 모델(LS, ELS)을 제시(저자들의 해석 기반 수학적 근사 이론)     
(LS(Local Score)는 각 픽셀이 자신의 주변 지역 정보만 보고 생성 방향을 결정하는 모델이고,
ELS(Equivariant Local Score)는 그 지역 정보를 이미지 어디서 왔는지 모른 채 추론하며 더 자유롭게 패치를 조합하는 모델)    
(기존 CNN 기반 확산 모델이 어떻게 창의적 이미지를 만들어내는지를 설명하고 예측하는 것이 목표)   



짧은 요약(Abstract) :    




---


이 논문은 \*\*합성곱 기반 확산 모델(convolutional diffusion models)\*\*에서 나타나는 \*\*창의성(creativity)\*\*을 **이론적으로 분석**하고 예측 가능한 방식으로 설명하려는 시도입니다. 기존의 **스코어 매칭(score-matching)** 이론에 따르면 확산 모델은 학습 데이터만 재생산할 수 있어야 하지만, 실제로는 전혀 새로운 이미지들을 생성할 수 있습니다. 저자들은 이 간극을 설명하기 위해 두 가지 단순한 귀납적 편향, 즉 \*\*지역성(locality)\*\*과 \*\*등변성(equivariance)\*\*을 도입합니다. 이로부터 **Local Score (LS) 기계**와 **Equivariant Local Score (ELS) 기계**라는 완전히 해석 가능한 모델을 수학적으로 유도합니다. 이 모델들은 시간에 따라 조정되는 단일 하이퍼파라미터만으로도, CIFAR10, MNIST, FashionMNIST, CelebA 등의 데이터셋에 대해 실제 확산 모델의 출력을 높은 정확도로 예측합니다(r² ≈ 0.94\~0.96). 이 분석은 **이미지를 다양한 위치의 작은 패치로 분해하고 조합하는 방식**, 즉 **패치 모자이크 방식의 창의성** 메커니즘을 설명합니다. 또한, 주의(attention)를 사용하는 모델에 대해서도 어느 정도 설명력이 있음을 보여줍니다(r² ≈ 0.77). 이로써 저자들은 확산 모델의 창의성 메커니즘을 명확하게 밝히고, 향후 주의 기반 모델에 대한 분석의 기반을 마련합니다.

---


> We obtain an analytic, interpretable and predictive theory of creativity in convolutional diffusion models. Indeed, score-matching diffusion models can generate highly original images that lie far from their training data. However, optimal score-matching theory suggests that these models should only be able to produce memorized training examples. To reconcile this theory-experiment gap, we identify two simple inductive biases, locality and equivariance, that: (1) induce a form of combinatorial creativity by preventing optimal score-matching; (2) result in fully analytic, completely mechanistically interpretable, local score (LS) and equivariant local score (ELS) machines that, (3) after calibrating a single time-dependent hyperparameter can quantitatively predict the outputs of trained convolution only diffusion models (like ResNets and UNets) with high accuracy (median r² of 0.95, 0.94, 0.94, 0.96 for our top model on CIFAR10, FashionMNIST, MNIST, and CelebA). Our model reveals a locally consistent patch mosaic mechanism of creativity, in which diffusion models create exponentially many novel images by mixing and matching different local training set patches at different scales and image locations. Our theory also partially predicts the outputs of pre-trained self-attention enabled UNets (median r² \~ 0.77 on CIFAR10), revealing an intriguing role for attention in carving out semantic coherence from local patch mosaics.

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





이 논문에서는 \*\*확산 모델(diffusion models)\*\*의 창의성을 분석하기 위해, 기존의 스코어 매칭(score matching) 이론이 설명하지 못하는 **새로운 샘플 생성 능력**을 설명하는 **해석 가능한 이론적 모델**을 제안합니다. 주요 기법은 다음과 같습니다:

1. **스코어 추정(score function approximation)**: 이상적인 스코어를 정확히 학습하면 모델은 학습 데이터만을 복원하게 되므로, 실제 확산 모델의 창의적 생성을 설명할 수 없습니다. 저자들은 **CNN 기반 모델이 학습 과정에서 자연스럽게 도입하는 두 가지 귀납적 편향**—① **지역성(Locality)**, ② **변환 등변성(Equivariance)**—에 주목합니다.

2. **모델 구성**: 이 편향들을 반영한 **세 가지 해석 가능한 모델**을 수학적으로 유도합니다:

   * **IS (Ideal Score) 머신**: 이상적인 스코어를 사용할 경우로, 항상 학습 데이터만 복원합니다.
   * **LS (Local Score) 머신**: 각 픽셀이 인접한 지역 정보만으로 스코어를 추정해, 서로 다른 위치에서 온 학습 이미지 패치를 조합할 수 있습니다.
   * **ELS (Equivariant Local Score) 머신**: 위치 정보까지 자유롭게 추론하여, 더욱 다양한 패치를 조합해 **지수적인 창의성**을 실현할 수 있습니다.

3. **훈련 데이터 및 실험 환경**:

   * 훈련은 **UNet**과 **ResNet** 기반의 확산 모델에서 진행되며, **MNIST, FashionMNIST, CIFAR10, CelebA** 데이터셋을 사용합니다.
   * 저자들이 제안한 이론 모델은 **시간에 따라 변화하는 지역성 크기(locality scale)** 하이퍼파라미터 하나만으로, 훈련된 모델의 출력을 거의 정확히 예측합니다 (r² ≈ 0.9 이상).

---


This paper proposes an **analytic and interpretable theory** to explain the **creativity** observed in convolutional diffusion models, which cannot be fully accounted for by traditional score-matching theory. The main methodological contributions are as follows:

1. **Score Function Approximation**: If a model perfectly learns the ideal score function, it will only reconstruct memorized training examples. The authors identify two key **inductive biases** naturally present in CNN-based architectures:

   * **Locality**: Each pixel only sees a limited neighborhood due to finite receptive fields.
   * **Equivariance**: CNNs preserve translation symmetry (i.e., translating the input translates the output).

2. **Model Construction**: Based on these constraints, the authors derive **three mechanistically interpretable models**:

   * **IS (Ideal Score) Machine**: Reconstructs only exact training data, with no creativity.
   * **LS (Local Score) Machine**: Each pixel uses only local context to reverse flow, allowing patch recombination from different training images.
   * **ELS (Equivariant Local Score) Machine**: Combines local and equivariant constraints, enabling each image location to borrow patches from any location in the training data—facilitating **exponential creativity** via locally consistent patch mosaics.

3. **Training Setup and Datasets**:

   * The experiments are conducted on convolutional diffusion models with **ResNet** and **UNet** backbones.
   * Datasets include **MNIST, FashionMNIST, CIFAR10, and CelebA**.
   * The ELS and LS machines can match the outputs of trained models with remarkable accuracy using only a single calibrated **time-dependent locality scale**, achieving median r² values of \~0.94–0.96.

---



   
 
<br/>
# Results  





이 논문에서는 제안된 **ELS(Local Score + Equivariance)** 및 **LS(Local Score)** 이론 모델이 실제 학습된 CNN 기반 확산 모델(ResNet, UNet)의 출력을 **정량적·정성적으로 얼마나 잘 예측하는지** 평가했습니다. 주요 실험 결과는 다음과 같습니다:

1. **데이터셋**:

   * 네 가지 표준 이미지 데이터셋을 사용했습니다: **MNIST, FashionMNIST, CIFAR10, CelebA**.

2. **경쟁 모델 및 비교 기준**:

   * 비교 대상은 실제 학습된 **ResNet 기반** 또는 **UNet 기반** 확산 모델의 출력이며,
   * 제안된 이론 모델인 **ELS 머신**, **LS 머신**, **IS (Ideal Score) 머신**이 이들과 얼마나 유사한 출력을 생성하는지를 \*\*r² (결정 계수)\*\*로 평가했습니다.

3. **주요 정량적 결과 (r² 값)**:

   * **ResNet**:

     * MNIST: 0.94
     * FashionMNIST: 0.90
     * CIFAR10: 0.90
     * CelebA: 0.96
   * **UNet**:

     * MNIST: 0.89
     * FashionMNIST: 0.93
     * CIFAR10: 0.90
     * CelebA: LS 머신이 더 적합 (r² ≈ 0.90)

4. **주의(attention) 기반 모델과의 비교**:

   * 자가주의(Self-Attention)를 포함한 UNet+SA 모델(CIFAR10 사전학습)을 대상으로도 실험을 수행했으며,
   * ELS 머신이 이 모델의 출력을 **부분적으로 예측**할 수 있었음 (r² ≈ 0.77).

5. **질적 비교 결과**:

   * ELS 이론은 **다리 개수 오류, 비대칭 구조 등** 실제 확산 모델에서 자주 관찰되는 **공간적 불일치 현상**도 정확히 모사할 수 있었으며,
   * 이는 **역전파 후반부의 과도한 지역성(locality)** 때문임을 이론적으로 설명함.

---


The authors evaluated their **ELS (Equivariant Local Score)** and **LS (Local Score)** machines by comparing their outputs to those of trained CNN-based diffusion models (ResNet and UNet) across several standard image datasets. Key results include:

1. **Datasets**:

   * Experiments were conducted on **MNIST, FashionMNIST, CIFAR10, and CelebA** datasets.

2. **Baselines and Evaluation Metric**:

   * The outputs of the **trained UNet and ResNet diffusion models** served as the reference.
   * The similarity between theory-based outputs (ELS, LS, and IS machines) and the trained model outputs was measured using the **coefficient of determination (r²)**.

3. **Quantitative Performance (r² values)**:

   * **ResNet**:

     * MNIST: 0.94
     * FashionMNIST: 0.90
     * CIFAR10: 0.90
     * CelebA: 0.96
   * **UNet**:

     * MNIST: 0.89
     * FashionMNIST: 0.93
     * CIFAR10: 0.90
     * CelebA: Better explained by **LS machine** (r² ≈ 0.90), likely due to full equivariance breaking.

4. **Comparison with Self-Attention Models**:

   * The ELS machine also partially predicted the outputs of a pre-trained **UNet+Self-Attention** model on CIFAR10,
   * Achieving a **median r² ≈ 0.77**, substantially outperforming the ideal score machine baseline.

5. **Qualitative Results**:

   * The ELS theory was able to **reproduce spatial inconsistencies** (e.g., wrong number of limbs) often observed in real diffusion models,
   * Providing a clear mechanistic explanation in terms of **excessive late-stage locality** during the reverse generation process.

---




<br/>
# 예제  




이 논문에서 제안된 \*\*이론 모델(ELS, LS)\*\*은 실제 학습된 확산 모델이 어떤 출력을 생성하는지를 예측하기 위한 것으로, 아래와 같은 구성으로 예제를 제시합니다.

1. **입력(Input)**:

   * 입력은 확산 모델에서 일반적으로 사용하는 방식처럼, **가우시안 노이즈 벡터**입니다. 예를 들어, MNIST나 CIFAR10 이미지와 동일한 크기의 랜덤 노이즈 이미지 `ϕ_T`를 생성합니다.
   * 같은 `ϕ_T`를 **이론 모델**과 **실제 학습된 모델(ResNet, UNet)** 모두에 입력으로 제공합니다.

2. **출력(Output)**:

   * 출력은 최종 생성된 이미지 `ϕ_0`이며, 이는 노이즈로부터 \*\*이미지 재구성(reverse generation)\*\*을 수행한 결과입니다.
   * 두 모델(이론 vs 학습 모델)의 생성 이미지가 **같은 초기 노이즈**에서 출발하여 **얼마나 유사한 이미지를 출력하는지** 비교합니다.
   * 예를 들어, Fig. 5에서 MNIST의 경우, ELS 이론이 생성한 숫자 '8' 이미지가 실제 UNet이 생성한 숫자 '8'과 매우 유사합니다.

3. **태스크(Task)**:

   * 이 예시는 \*\*이미지 생성(Image Generation)\*\*이라는 태스크에 해당합니다.
   * 주어진 노이즈로부터 실제로 학습된 확산 모델이 어떤 이미지를 생성할지를, 훈련 없이 제안된 이론 모델이 **정확히 예측할 수 있는지**를 보여주는 것이 목적입니다.
   * 특히 CIFAR10, FashionMNIST와 같은 복잡한 이미지에서도 제안된 이론 모델이 각 픽셀 수준에서 실제 모델의 동작을 예측함을 보여줍니다.

---



In this paper, the authors demonstrate the effectiveness of their **analytical theory (ELS and LS machines)** by showing how closely these models can **predict the outputs of trained diffusion models** on a **case-by-case basis**, using the following setup:

1. **Input**:

   * The input is a **Gaussian noise image** `ϕ_T`, similar in shape and size to the dataset images (e.g., 28×28 for MNIST, 32×32×3 for CIFAR10).
   * This same noise input is fed into both the **theoretical model** (e.g., ELS machine) and the **trained model** (ResNet or UNet).

2. **Output**:

   * The output is a **denoised, generated image** `ϕ_0`, which results from the reverse diffusion process.
   * The goal is to compare whether both models produce **visually and structurally similar images** when given the same starting noise.
   * For instance, in Figure 5, the ELS-generated digit ‘8’ closely resembles the ‘8’ produced by a trained UNet on MNIST.

3. **Task**:

   * The task is **image generation**: generating realistic images from noise via reverse diffusion.
   * The key objective is to evaluate whether the theory-based model can **accurately predict** the output of a trained diffusion model **without learning**, purely from inductive biases like locality and equivariance.
   * The examples span datasets like MNIST, CIFAR10, and FashionMNIST, showing the predictive capability even for relatively complex and diverse image structures.

---




<br/>  
# 요약   





이 논문은 CNN 기반 확산 모델의 창의성을 설명하기 위해 지역성(locality)과 변환 등변성(equivariance)에 기반한 해석 가능한 이론 모델(LS, ELS)을 제안한다.
제안된 모델은 MNIST, CIFAR10, CelebA 등의 실제 학습된 UNet 및 ResNet 모델의 출력을 r² ≈ 0.9 이상으로 정확히 예측하였다.
같은 초기 노이즈로부터 생성된 이미지가 이론 모델과 학습 모델 간에 매우 유사함을 통해, 이론이 실제 모델의 동작을 픽셀 수준까지 정량적으로 설명할 수 있음을 보여준다.

---



This paper introduces interpretable theoretical models (LS, ELS) based on locality and equivariance to explain the creativity of CNN-based diffusion models.
The proposed models accurately predict the outputs of trained UNet and ResNet diffusion models on datasets like MNIST, CIFAR10, and CelebA, achieving r² ≈ 0.9 or higher.
By comparing outputs from the same noise inputs, the theory is shown to closely match the behavior of trained models at the pixel level.

---



<br/>  
# 기타  





1. **Figure 1, 5, 6, 7: 이론 vs 실제 이미지 비교**

   * **Fig. 5**는 동일한 노이즈 입력으로부터 생성된 이론 모델(ELS)과 학습된 ResNet/UNet 모델의 이미지를 나란히 제시하며, **매우 유사한 결과**를 보여준다.
   * 특히 **Fig. 6**에서는 ELS 출력이 불완전한 조각(patch mosaic) 형태로 구성되었을 때, **Self-Attention이 이를 정합된 객체로 보정**하는 과정을 시각적으로 보여준다.
   * **Fig. 7**은 CelebA 데이터셋에서 UNet이 ResNet보다 더 구조화된 얼굴을 생성함을 보여주며, **UNet이 위치 정보를 더 잘 활용**하는 것으로 해석된다.

2. **Figure 4: 시간에 따른 지역성 변화**

   * **Fig. 4a**는 시간에 따라 수용영역(receptive field)이 **넓은 영역에서 좁은 영역으로 줄어드는 coarse-to-fine 경향**을 시각화하며,
   * 이 late-time 지역성이 \*\*신체 부위의 수 오류(팔이 3개, 다리가 1개 등)\*\*와 같은 공간적 불일치 현상의 원인임을 보여준다 (**Fig. 5c**에서 함께 설명됨).

3. **Figure 10, Table 2: 정량 평가**

   * **Fig. 10**과 **Table 2**는 데이터셋별로 이론 모델과 학습 모델 간 **r² 분포와 중앙값**을 수록하며,
   * 대부분의 경우 ELS 모델이 학습된 CNN보다 **현저히 높은 예측 정확도**를 보임을 입증한다 (e.g., CelebA에서 r² ≈ 0.96).

4. **Appendix C, D: 다양한 예제 및 설정**

   * **Appendix D**는 다양한 입력 노이즈에 대한 출력 비교 예시(Fig. 13\~24)를 포함하고 있어 **case-by-case 예측 능력**을 반복적으로 확인할 수 있다.
   * **Appendix C**에서는 실험 조건, 모델 구조, locality scale의 보정 방법 등이 기술되어 있어 **이론과 실험 간 정합성 확보**에 기여한다.

---


1. **Figures 1, 5, 6, 7: Side-by-side comparisons**

   * **Figure 5** shows side-by-side outputs from the ELS theory and trained ResNet/UNet using the **same noise input**, revealing **high visual similarity**.
   * **Figure 6** illustrates how Self-Attention in a UNet-SA model can **extract semantically coherent objects** from the locally incoherent mosaics generated by the ELS machine.
   * **Figure 7** highlights that UNet generates **more structured faces** than ResNet on CelebA, suggesting UNet’s stronger ability to **utilize absolute positional information**.

2. **Figure 4: Temporal locality dynamics**

   * **Figure 4a** visualizes the shrinking receptive field size over time during reverse diffusion, indicating a **coarse-to-fine progression**.
   * This helps explain **spatial inconsistencies** in generated outputs (e.g., wrong number of limbs), as confirmed in **Figure 5c**.

3. **Figure 10, Table 2: Quantitative accuracy**

   * **Figure 10** and **Table 2** report **r² values** between theoretical and trained model outputs across datasets,
   * Showing that ELS achieves **remarkably high predictive accuracy** (e.g., r² ≈ 0.96 on CelebA), often surpassing all other baselines.

4. **Appendices C & D: Experimental setup and extensive examples**

   * **Appendix D** provides **dozens of case-by-case visual comparisons** (Figures 13–24), reinforcing the **sample-level predictive validity** of the theory.
   * **Appendix C** describes architectural details, calibration of locality scales, and experimental setups, ensuring **rigorous alignment between theory and experiments**.

---




<br/>
# refer format:     




@inproceedings{kamb2025analytic,
  title     = {An Analytic Theory of Creativity in Convolutional Diffusion Models},
  author    = {Mason Kamb and Surya Ganguli},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  publisher = {PMLR},
  volume    = {267},
  address   = {Vancouver, Canada}
}



Kamb, Mason, and Surya Ganguli. “An Analytic Theory of Creativity in Convolutional Diffusion Models.” In Proceedings of the 42nd International Conference on Machine Learning (ICML), vol. 267. Vancouver, Canada: PMLR, 2025.


