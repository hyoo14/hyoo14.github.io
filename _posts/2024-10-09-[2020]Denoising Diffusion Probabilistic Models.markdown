---
layout: post
title:  "[2020]Denoising Diffusion Probabilistic Models"  
date:   2024-10-09 17:10:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    





이 논문은 비평형 열역학에서 영감을 받은 확률적 확산 모델을 사용해 고품질의 이미지 합성 결과를 제시합니다. 최적의 결과는 새로운 확산 확률 모델과 Langevin 동역학을 사용한 잡음 제거 점수 매칭 간의 연결을 바탕으로 설계된 가중 변분 경계를 학습하여 얻었습니다. 이러한 모델은 점진적인 손실 압축 방식을 자연스럽게 받아들이며, 이를 자기회귀 디코딩의 일반화로 해석할 수 있습니다. 무조건적인 CIFAR-10 데이터셋에서 모델은 Inception 점수 9.46과 FID 점수 3.17을 기록했으며, 256x256 LSUN 데이터셋에서는 ProgressiveGAN과 유사한 샘플 품질을 보였습니다. 이 구현은 GitHub에서 공개되었습니다.


This paper presents high-quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. The best results were obtained by training on a weighted variational bound, designed based on a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics. The models naturally support a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR-10 dataset, the models achieve an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On the 256x256 LSUN dataset, the models produce sample quality comparable to ProgressiveGAN. The implementation is available on GitHub.




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


이 논문에서 제안한 방법론은 확산 확률 모델을 활용하여 고품질 이미지 합성을 수행합니다. 이 방법은 크게 세 단계로 나뉩니다:

1. **확산 과정(Forward Process):**  
   원본 데이터를 점진적으로 잡음이 추가된 상태로 변화시키는 마코프 체인 형태의 확산 과정입니다. 이 과정은 여러 단계(t)에서 데이터에 작은 가우시안 잡음을 추가해 데이터의 신호를 손실시키는 방식으로 이루어집니다. 이 과정의 결과는 완전히 잡음화된 데이터(\(x_T\))입니다.

2. **역방향 과정(Reverse Process):**  
   역방향 과정은 확산 과정의 반대 방향으로, 잡음화된 데이터를 원본 데이터로 되돌리기 위해 학습된 마코프 체인입니다. 이 과정에서는 모델이 각 단계에서 주어진 \(x_t\)를 기반으로 \(x_{t-1}\)을 생성하는 조건부 가우시안 분포를 학습합니다. 이는 가우시안 조건부 분포를 사용해 각 단계의 전환을 단순하게 설정할 수 있게 하며, 뉴럴 네트워크가 이러한 전환을 학습합니다.

3. **변분 경계 학습(Variational Bound Training):**  
   모델을 학습하기 위해 변분 추론을 사용해, 확산 과정과 역방향 과정 간의 차이를 최소화하는 방식으로 학습됩니다. 이 논문에서는 특히 잡음 제거 점수 매칭(denoising score matching)과 Langevin 동역학을 사용하여 변분 경계를 최적화하는 새로운 접근법을 제안했습니다. 이는 학습 중 다양한 잡음 수준에서 데이터 분포의 경사를 예측하도록 모델을 조정하며, 이를 통해 역방향 과정이 데이터를 효과적으로 복원할 수 있게 합니다.

이 방법론의 핵심은 확산 과정에서 점진적으로 잡음을 제거하며 고품질의 샘플을 생성하는 것입니다. 이 과정은 모델이 각 잡음 단계에서 데이터의 원래 신호를 어떻게 복원할지 학습하도록 유도하며, 이를 통해 최종적으로 원본 데이터와 유사한 고해상도 이미지를 생성할 수 있습니다.



The methodology proposed in this paper leverages diffusion probabilistic models for high-quality image synthesis. The approach is divided into three main stages:

1. **Forward Process:**  
   This is a diffusion process where the original data is gradually transformed into a noisy state through a Markov chain. At each step (t), small amounts of Gaussian noise are added to the data, progressively destroying the signal. The process results in fully noise-corrupted data (\(x_T\)).

2. **Reverse Process:**  
   The reverse process is a Markov chain trained to reverse the forward diffusion process, restoring the noisy data back to its original state. At each step, the model learns a conditional Gaussian distribution to generate \(x_{t-1}\) given \(x_t\). This allows for a simple parameterization using Gaussian conditionals, with a neural network learning the transitions at each step.

3. **Variational Bound Training:**  
   To train the model, variational inference is used to minimize the discrepancy between the forward and reverse processes. The paper introduces a novel approach by optimizing the variational bound through denoising score matching and Langevin dynamics. This approach adjusts the model to predict gradients of the data distribution across various noise levels during training, enabling the reverse process to effectively restore data.

The key aspect of this methodology is the gradual removal of noise through the diffusion process to generate high-quality samples. The model learns how to recover the original signal from noisy data at each stage, ultimately producing high-resolution images that closely resemble the original data.

<br/>
# Results  


이 논문에서 제시된 확산 확률 모델의 성능은 주로 CIFAR-10과 LSUN 데이터셋에서 평가되었습니다.

1. **CIFAR-10 데이터셋:**  
   - 이 논문에서는 무조건적인(Unconditional) CIFAR-10 데이터셋에서 모델을 평가했습니다.  
   - 제안된 확산 모델은 Inception Score (IS) 9.46과 FID (Frechet Inception Distance) 3.17을 달성했습니다.  
   - 특히, FID 3.17은 이전의 많은 모델들보다 우수한 성능으로, ProgressiveGAN과 같은 최신 모델보다도 뛰어난 샘플 품질을 보여주었습니다. FID 점수는 낮을수록 실제 데이터와 생성된 데이터 간의 차이가 적음을 의미하므로, 이 모델은 매우 높은 샘플 품질을 나타냅니다.

2. **LSUN 데이터셋 (256x256 해상도):**  
   - LSUN Church와 LSUN Bedroom 데이터셋에서 256x256 해상도의 이미지 샘플을 생성했으며, 여기에서도 ProgressiveGAN과 유사한 샘플 품질을 보여주었습니다.  
   - FID 점수는 LSUN Church 데이터셋에서 7.89, LSUN Bedroom 데이터셋에서 4.90으로 측정되었습니다. 이는 해당 해상도에서 기존의 경쟁 모델과 유사하거나 더 나은 성능을 보여주는 결과입니다.

3. **성능 비교:**  
   - 제안된 모델은 확산 모델의 기존 결과보다 높은 샘플 품질을 기록했으며, 특히 가중치가 고정된 경우에 FID 성능이 향상되었습니다.  
   - 또한, 기존의 에너지 기반 모델(EBM) 및 NCSN (Noise Conditional Score Network)와 비교했을 때도 더 나은 성능을 보여주었습니다.


The performance of the proposed diffusion probabilistic models is evaluated primarily on the CIFAR-10 and LSUN datasets.

1. **CIFAR-10 Dataset:**  
   - The model was evaluated on the unconditional CIFAR-10 dataset.  
   - The proposed diffusion model achieved an Inception Score (IS) of 9.46 and a Frechet Inception Distance (FID) score of 3.17.  
   - The FID score of 3.17 is better than many previously reported models, showing superior sample quality compared to even state-of-the-art models like ProgressiveGAN. A lower FID indicates that the generated samples are closer to the real data, demonstrating the high quality of the generated images.

2. **LSUN Dataset (256x256 resolution):**  
   - The model generated image samples for the LSUN Church and LSUN Bedroom datasets at 256x256 resolution, achieving sample quality comparable to ProgressiveGAN.  
   - The FID scores were 7.89 for the LSUN Church dataset and 4.90 for the LSUN Bedroom dataset. These results indicate that the model's performance is comparable to or exceeds that of existing models at this resolution.

3. **Performance Comparison:**  
   - The proposed model surpassed previous diffusion models in terms of sample quality, particularly when using fixed-weight parameterization, leading to improved FID performance.  
   - Additionally, it outperformed other competing models such as energy-based models (EBMs) and Noise Conditional Score Networks (NCSN).


<br/>
# 예시  


이 논문에서 제안된 확산 확률 모델은 이미지가 점진적으로 변하고 복원되는 과정을 통해 고품질 샘플을 생성합니다. 이 과정은 확산 과정과 역방향 과정으로 나눌 수 있습니다:

1. **확산 과정 (이미지의 변화 양상):**  
   - 확산 과정에서는 원본 이미지에 점진적으로 가우시안 잡음이 추가됩니다. 이 과정은 마코프 체인을 통해 단계별로 이루어지며, 각 단계에서 이미지는 점점 더 노이즈가 많아져 원래의 시각적 특성이 사라집니다.
   - 예를 들어, 초기에는 이미지의 고해상도 세부 정보가 유지되지만, 시간이 지날수록 이미지가 흐려지고, 마지막에는 순수한 잡음(완전히 손실된 신호) 상태로 변합니다. 이 상태를 \(x_T\)라고 부릅니다. 이 과정은 생성된 데이터가 더 다양한 분포를 가질 수 있도록 돕는 역할을 합니다.

2. **역방향 과정 (이미지의 복원 양상):**  
   - 역방향 과정에서는 확산 과정에서 완전히 잡음화된 이미지(\(x_T\))를 점진적으로 원래의 상태로 되돌립니다. 이는 마코프 체인을 역으로 적용하여 각 단계에서 조금씩 잡음을 제거해 가는 과정입니다.
   - 구체적으로, 모델은 각 단계에서 현재 상태의 이미지(\(x_t\))를 기반으로 이전 단계(\(x_{t-1}\))의 상태를 예측합니다. 이 과정은 가우시안 분포를 사용해 조건부로 설정되며, 모델이 학습한 뉴럴 네트워크가 이를 수행합니다.
   - 예를 들어, 초기에는 매우 거친 이미지(거의 잡음으로 가득 찬 상태)에서 시작하여 큰 특징들이 복원되기 시작합니다. 이후 단계에서는 더 세부적인 부분들, 예를 들어 이미지의 텍스처나 색상 등이 점진적으로 복원됩니다. 최종 단계에 도달했을 때는, 원래 이미지와 유사한 고해상도 이미지가 생성됩니다.

이 두 과정은 마치 이미지의 점진적인 손실과 회복을 반복하는 과정처럼 작동하며, 모델이 다양한 잡음 수준에서 데이터를 복원하는 방법을 학습하도록 돕습니다. 이를 통해 생성된 이미지는 원본과 비슷한 고품질의 결과물을 만들어냅니다.


In this paper, the proposed diffusion probabilistic model generates high-quality samples by gradually altering and restoring images through a process divided into a forward diffusion process and a reverse process:

1. **Forward Process (Image Alteration):**  
   - In the forward process, Gaussian noise is gradually added to the original image. This process is implemented as a Markov chain, where at each step, more noise is introduced, progressively obscuring the original visual features of the image.
   - For example, initially, high-resolution details of the image are retained, but over time, the image becomes increasingly blurry, eventually turning into pure noise (a state where the signal is completely destroyed). This state is referred to as \(x_T\). This process allows the generated data to encompass a broader distribution.

2. **Reverse Process (Image Restoration):**  
   - The reverse process involves gradually restoring the fully noise-corrupted image (\(x_T\)) back to its original state. This is done by reversing the Markov chain, removing a small amount of noise at each step.
   - Specifically, the model predicts the state of the previous step (\(x_{t-1}\)) based on the current noisy image (\(x_t\)) at each step. This is formulated using a conditional Gaussian distribution, and the neural network learned by the model performs this restoration.
   - For example, the process begins with a highly rough image (almost entirely filled with noise), where large features start to emerge. As the process continues, finer details, such as texture and color, gradually reappear. By the final step, the restored image closely resembles the original high-resolution image.

These two processes function like a repeated cycle of degradation and recovery, guiding the model to learn how to restore data from various levels of noise. This results in generated images that are similar in quality to the original data.
 




<br/>  
# 요약 



이 논문은 확산 확률 모델을 사용하여 고품질 이미지 합성을 수행하는 방법을 제안합니다. 모델은 가우시안 잡음을 점진적으로 추가하는 확산 과정과 이를 반대로 제거해 원래 이미지를 복원하는 역방향 과정을 사용합니다. CIFAR-10 데이터셋에서 Inception Score 9.46, FID 3.17을 기록하며, 기존의 많은 경쟁 모델들보다 우수한 성능을 보여주었습니다. LSUN 데이터셋에서도 ProgressiveGAN과 비슷한 수준의 샘플 품질을 달성했습니다. 이 모델은 복잡한 잡음 수준에서도 데이터를 효과적으로 복원할 수 있어, 이미지 생성 및 복원에서 뛰어난 성능을 입증했습니다.



This paper proposes a method for high-quality image synthesis using diffusion probabilistic models. The model employs a forward process that gradually adds Gaussian noise and a reverse process that removes this noise to restore the original image. It achieved an Inception Score of 9.46 and an FID score of 3.17 on the CIFAR-10 dataset, outperforming many competing models. On the LSUN dataset, it demonstrated sample quality similar to that of ProgressiveGAN. The model effectively restores data even at complex noise levels, proving its strength in image generation and restoration tasks.



# 기타  




1. **Inception Score (IS):**  
   - Inception Score는 생성된 이미지의 품질을 평가하는 지표로, 두 가지 요소를 고려합니다: 이미지가 특정 클래스로 얼마나 뚜렷하게 분류되는지와 생성된 이미지들이 얼마나 다양한지를 평가합니다. 높은 IS는 생성된 이미지가 명확한 분류를 가지며, 다양한 샘플을 포함하고 있음을 의미합니다. 일반적으로 GAN 모델의 성능을 평가할 때 많이 사용됩니다.

2. **Frechet Inception Distance (FID):**  
   - FID는 생성된 이미지와 실제 데이터 사이의 분포 차이를 평가하는 지표입니다. FID는 생성된 이미지와 실제 데이터의 Inception 네트워크에서 추출한 피처 맵의 평균과 공분산 행렬을 비교하여 계산됩니다. 낮은 FID 점수는 생성된 이미지가 실제 데이터와 유사한 분포를 가진다는 것을 나타내며, 따라서 더 나은 샘플 품질을 의미합니다. FID는 Inception Score보다 실제 데이터와의 유사성을 더 잘 반영한다고 여겨집니다.


1. **Inception Score (IS):**  
   - The Inception Score evaluates the quality of generated images by considering two factors: how clearly the images can be classified into distinct categories and the diversity of the generated samples. A higher IS indicates that the generated images are both well-classified and diverse. It is commonly used to assess the performance of GAN models.

2. **Frechet Inception Distance (FID):**  
   - The FID measures the difference between the distribution of generated images and real images. It is calculated by comparing the means and covariance matrices of feature maps extracted from an Inception network for both the generated and real images. A lower FID score indicates that the generated images have a distribution closer to that of real images, implying better sample quality. FID is often considered to provide a more accurate measure of similarity to real data compared to the Inception Score.


<br/>
# refer format:     

@inproceedings{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={Proceedings of the 34th Conference on Neural Information Processing Systems (NeurIPS 2020)},
  year={2020},
  address={Vancouver, Canada},
  url={https://github.com/hojonathanho/diffusion},
 }


Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising Diffusion Probabilistic Models." Proceedings of the 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada, 2020. https://github.com/hojonathanho/diffusion.