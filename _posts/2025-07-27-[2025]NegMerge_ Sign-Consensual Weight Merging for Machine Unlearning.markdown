---
layout: post
title:  "[2025]NegMerge: Sign-Consensual Weight Merging for Machine Unlearning"  
date:   2025-07-27 02:00:40 +0200
categories: study
---

{% highlight ruby %}


한줄 요약: 


파인튜닝한 모델들의 task vector를 병합할 때, 부호가 일치하는 요소만 남기는 방식으로 더 안정적인 머신 언러닝을 달성   


짧은 요약(Abstract) :    



이 논문은 **머신 언러닝(machine unlearning)**을 위한 새로운 방법을 제안합니다. 머신 언러닝이란 학습된 모델에서 특정 데이터를 선택적으로 "잊게 만드는" 기술입니다. 기존 방법들은 'Task Arithmetic'처럼 특정 데이터를 기반으로 모델을 파인튜닝한 후 그 가중치 차이 벡터를 원래 모델에서 빼는 방식(Forget by Negation)을 사용합니다. 하지만 이 방식은 어떤 하이퍼파라미터로 파인튜닝하느냐에 따라 결과가 크게 달라지며, 최적의 벡터를 찾기 위해 많은 검증 과정이 필요합니다.

NegMerge는 이러한 문제를 해결하기 위해, 여러 하이퍼파라미터 설정에서 얻어진 파인튜닝 모델들의 벡터를 모두 활용합니다. 여러 벡터 중 부호(sign)가 일치하는 요소들만 남기고, 나머지는 0으로 설정한 후, 그 평균 벡터를 생성하여 원래 모델에서 빼는 방식으로 언러닝을 수행합니다. 이 방법은 단일 벡터 선택보다 성능이 뛰어나며, 유사하거나 더 적은 계산 자원으로 더 나은 성능을 보여줍니다.



This paper introduces NegMerge, a new method for machine unlearning, which aims to selectively remove specific knowledge from a trained model. Traditional approaches, like Task Arithmetic, subtract a "task vector"—obtained from a fine-tuned model on the forget set—from the original model weights. However, these methods are highly sensitive to hyperparameter selection, requiring costly validation to choose the best candidate.

NegMerge overcomes this by utilizing all fine-tuned models trained under different hyperparameters. It merges their task vectors by retaining only elements with consistent signs, masking the rest to zero. The merged task vector is then negated from the original model to induce forgetting. Experimental results on twelve datasets and four architectures show that NegMerge outperforms existing methods while requiring similar or fewer computational resources.




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



NegMerge는 기존의 Task Arithmetic 기반 머신 언러닝 방식에 개선을 가한 기법으로, 다음과 같은 세 가지 주요 단계로 구성됩니다:

다양한 하이퍼파라미터로 파인튜닝한 모델 생성: 단일 모델이 아닌 여러 하이퍼파라미터 설정(학습률, 에폭 수, 데이터 증강 등)으로 파인튜닝한 여러 모델을 생성합니다. 각 모델로부터 Task Vector(원래 모델과 파인튜닝 모델의 가중치 차이)를 계산합니다.

부호 일치 요소만 선택하여 병합: 각 Task Vector에서 동일한 부호(+, -)를 가진 요소만 유지하고, 부호가 일치하지 않는 요소는 0으로 마스킹합니다. 이렇게 하면 "잊혀야 할 데이터(Forget Set)"와 관련된 일관된 신호만 유지됩니다.

최종 Task Vector를 반영하여 언러닝 수행: 위에서 병합된 벡터를 원래 모델에서 빼는 방식으로 모델이 해당 지식을 잊게 만듭니다. 이 방식은 기존 방법보다 계산량은 비슷하지만 성능이 더 뛰어납니다.

이 방식은 기존에 하이퍼파라미터 튜닝에 의존하던 언러닝 과정을 개선하며, 여러 모델의 정보를 모두 활용함으로써 성능 편차를 줄이고, 보존 데이터(Retain Set)의 성능 손실도 최소화합니다.




NegMerge improves on traditional Task Arithmetic-based machine unlearning through the following three key steps:

Generating multiple fine-tuned models: Instead of selecting a single model, the method fine-tunes multiple models using diverse hyperparameters (e.g., learning rates, number of epochs, data augmentation settings). From each model, a task vector is computed by subtracting the original model weights from the fine-tuned model.

Merging sign-consistent elements: Across all task vectors, only elements with the same sign (+ or −) are retained. Inconsistent elements (with conflicting signs) are masked to zero. This sign-consensual filtering ensures that only consistent signals related to the forget set are preserved.

Final task vector and unlearning: The merged task vector is averaged and then subtracted from the original model weights, effectively inducing forgetting. This avoids relying on a single hyperparameter configuration and reduces performance instability.

This method leads to more reliable and efficient unlearning while maintaining performance on the retain set and requiring no additional computational cost.




   
 
<br/>
# Results  



NegMerge는 기존 머신 언러닝 기법들과 비교해 다양한 모델과 데이터셋에서 **최신 성능(state-of-the-art)**을 기록했습니다.

비교 대상: Task Arithmetic, Uniform Merge, Greedy Merge, TIES-Merging, MagMax, SalUn, Fine-tuning, Influence 기반 언러닝 등 총 10개 이상의 기존 방법들과 비교했습니다.

테스트 환경:

모델: CLIP (ViT-B/32, B/16, L/14), ResNet-18, VGG-16, Swin-T

데이터셋: Cars, SUN397, DTD, EuroSAT, MNIST, GTSRB, SVHN 등 총 12개

시나리오:

Vision-language 모델(CLIP)의 특정 데이터셋에 대한 제로샷 인식 능력 제거

일반 이미지 분류 모델이 학습한 특정 데이터의 지식 제거

평가 지표:

Forget Set Accuracy (Df): 낮을수록 언러닝 성능이 우수

Retain Set Accuracy (Dr): 높을수록 기존 지식 보존 성능이 우수

MIA (Membership Inference Attack): 낮을수록 개인정보 보호 성능 우수

Avg. Gap: 완전 재학습 모델과의 평균 성능 차이 (작을수록 좋음)

핵심 결과:

CLIP 모델에서 NegMerge는 모든 벤치마크에서 가장 낮은 Df(Forget Set 정확도)를 기록하며 가장 효과적인 언러닝을 달성

CIFAR-10에서 10% 데이터를 잊게 만드는 실험에서는 기존 SalUn보다 낮은 Avg. Gap (1.07 vs 1.15)으로 완전 재학습 수준에 가장 가까운 성능 달성

MagMax, TIES-Merging보다 빠르고 효과적이며, Greedy Merge보다 훨씬 계산 효율적임




NegMerge achieves state-of-the-art performance across multiple datasets and architectures when compared to over ten existing unlearning methods.

Baselines Compared: Task Arithmetic, Uniform Merge, Greedy Merge, TIES-Merging, MagMax, SalUn, Fine-tuning, Influence-based unlearning, and others.

Evaluation Setup:

Backbones: CLIP (ViT-B/32, B/16, L/14), ResNet-18, VGG-16, Swin-T

Datasets: Cars, SUN397, DTD, EuroSAT, MNIST, GTSRB, SVHN, CUB, CIFAR-10, Tiny ImageNet (12 in total)

Scenarios:

Zero-shot knowledge erasure in vision-language models (e.g., CLIP)

Knowledge removal of specific training subsets in standard image classifiers

Metrics Used:

Forget Set Accuracy (Df): Lower is better (more unlearning)

Retain Set Accuracy (Dr): Higher is better (knowledge retention)

Membership Inference Attack (MIA): Lower is better (privacy protection)

Average Gap: Mean performance difference from the retrained model (lower is better)

Key Findings:

On CLIP models, NegMerge achieves the lowest Df accuracy across all datasets, indicating the strongest unlearning.

On CIFAR-10 (10% data forgetting), it achieves an Avg. Gap of 1.07, outperforming SalUn (1.15), and getting closest to the fully retrained model.

NegMerge also outperforms MagMax and TIES-Merging in accuracy while being more computationally efficient than Greedy Merge.







<br/>
# 예제  



NegMerge는 두 가지 대표적인 머신 언러닝 시나리오에 적용되어 실험되었습니다.



1.  CLIP 모델 기반 제로샷 언러닝 시나리오
목표: CLIP 모델이 특정 데이터셋(예: Cars)을 인식하지 못하도록 만듦

입력 데이터:

Forget Set: 예를 들어 Cars 데이터셋 (자동차 이미지)

Retain Set: ImageNet (다양한 일반 이미지)

테스트 과정:

언러닝 후, CLIP 모델이 Cars 이미지를 올바르게 분류하지 못해야 함 (Df↓)

동시에 ImageNet 성능은 유지되어야 함 (Dr 유지)

입출력 구조:

입력: 이미지와 텍스트 쌍 (e.g., 자동차 이미지 + "car")

출력: 텍스트 라벨의 매칭 점수 (제로샷 인식 점수)



2.  표준 이미지 분류 모델 기반 언러닝 시나리오
목표: 학습 데이터 일부를 모델이 '잊도록' 만듦

사용 예시:

CIFAR-10에서 전체 학습 데이터 중 10%를 Forget Set으로 설정

입력 데이터:

Forget Set: 잊게 만들 클래스 예시들 (예: 개, 고양이 이미지 등)

Retain Set: 나머지 90% 학습 데이터

테스트 과정:

모델은 Forget Set에 대해 낮은 정확도를 보여야 하고 (Df↓)

Retain Set과 Test Set에 대해서는 정확도를 유지해야 함

입출력 구조:

입력: 이미지 (32×32 픽셀)

출력: 예측된 클래스 레이블 (0~9 사이의 숫자)




NegMerge was evaluated in two key machine unlearning scenarios, each with different data structures and goals.



1.  CLIP-based Zero-Shot Unlearning Scenario
Objective: Make CLIP unable to recognize data from specific datasets (e.g., Cars)

Input:

Forget Set: Cars dataset (car images)

Retain Set: ImageNet dataset (general image categories)

Evaluation:

After unlearning, the model should fail to recognize "Cars" images (low Df accuracy)

At the same time, it must retain ImageNet performance (high Dr accuracy)

I/O Structure:

Input: Image-text pairs (e.g., car image + "car")

Output: Zero-shot matching score for each text label



2.  Standard Classifier Unlearning Scenario
Objective: Make the model forget part of its training data

Use Case:

From CIFAR-10, 10% of the training data is used as the Forget Set

Input:

Forget Set: Specific images (e.g., some classes like "dog" or "cat")

Retain Set: Remaining 90% of training data

Evaluation:

Model should show degraded accuracy on Forget Set (low Df)

But maintain accuracy on Retain and Test Sets

I/O Structure:

Input: Image (e.g., 32×32 RGB image)

Output: Predicted class label (from 0 to 9)



<br/>  
# 요약   



NegMerge는 여러 하이퍼파라미터로 파인튜닝한 모델들의 task vector를 병합할 때, 부호가 일치하는 요소만 남기는 방식으로 더 안정적인 머신 언러닝을 달성합니다. 실험 결과, CLIP과 ResNet 등 다양한 모델과 12개 데이터셋에서 기존 SOTA 기법보다 더 강력한 unlearning 성능과 유지 성능을 동시에 달성했습니다. 예를 들어 CLIP 모델에서는 Cars 데이터셋을 잊게 하면서도 ImageNet 성능을 유지했고, CIFAR-10에서는 Forget Set과 Retain Set 간 성능 균형을 가장 잘 맞췄습니다.




NegMerge performs machine unlearning by merging task vectors from multiple fine-tuned models, keeping only sign-consistent elements to ensure stable forgetting. Experiments show that it outperforms prior state-of-the-art methods across 12 datasets and multiple backbones like CLIP and ResNet. For example, it makes CLIP forget the Cars dataset while preserving ImageNet performance, and achieves the best balance on CIFAR-10 between forget and retain sets.



<br/>  
# 기타  




 Figure 1: 하이퍼파라미터 민감도 시각화
(a) Forget Set과 Retain Set 정확도의 trade-off를 시각화. 기존 방법은 한 쪽을 희생해야 했지만, NegMerge는 둘 다 유지.

(b) 하이퍼파라미터 설정에 따라 Forget Set 정확도가 최대 15%까지 변동함 → NegMerge는 이 민감도를 완화.



 Figure 2: 방법론 도식화
여러 task vector를 병합할 때, 부호가 일치하는 요소만 합산하고 나머지는 0으로 마스킹.

이로 인해 불필요한 흔들림(noise)을 제거하고, 일관된 unlearning 방향으로 작동함.



 Table 1 & 2: 성능 비교 표 (CLIP, CIFAR-10 등)
CLIP ViT-B/32 기준으로 NegMerge가 Forget Set 정확도를 **20.76%**까지 낮춤 (기존 Task Arithmetic보다 낮음).

CIFAR-10 실험에서는 Avg. Gap이 1.07로 가장 낮아, 완전 재학습과 가장 유사한 언러닝 품질 달성.



 Table 3: 부호 일치 vs 불일치 비교 실험
Conflict(부호 불일치)만 사용하면 성능이 크게 저하됨 → 부호 일치 요소만 쓰는 것이 핵심 설계 포인트임을 증명.



 Table 4: 병합된 벡터의 희소성(sparsity)
병합된 task vector의 90% 이상 요소가 0, 즉 매우 sparse → 계산 비용 감소 및 retain 성능 유지에 유리함.



 Figure 3: Grad-CAM 시각화 (클래스 관련성 제거 확인)
Consensus 방식은 클래스 관련 위치에 활성도가 낮게 나타나 해당 개념이 제거되었음을 시각적으로 확인 가능.



 Appendix C, D, F 등:
C: 병합 연산별 비교 → 평균(average)이 최적

D: 다양한 모델 풀 구성에도 NegMerge 성능 유지 → 모델 구성에 강건함

F: 비전 모델 외에도 LLM과 VLM에도 적용 가능성 있음 논의




 Figure 1: Hyperparameter Sensitivity
(a) Visualizes the trade-off between forgetting and retaining accuracy. NegMerge breaks this trade-off.

(b) Forget accuracy fluctuates up to 15% with different hyperparameters → NegMerge smooths this instability.



 Figure 2: Method Diagram
Merges task vectors by retaining only elements with sign-consensus, masking conflicting ones to zero.

This reduces noise and ensures consistent forgetting direction.



 Table 1 & 2: Performance Tables (CLIP, CIFAR-10)
On CLIP ViT-B/32, NegMerge achieves 20.76% Df, outperforming Task Arithmetic and others.

On CIFAR-10, Avg. Gap is just 1.07, closest to retrained model, showing high-fidelity unlearning.



 Table 3: Sign Consistency vs. Conflict
Using only sign-conflict elements degrades performance → proves that sign-consensual elements are essential.



 Table 4: Sparsity of Merged Vectors
Final task vectors are 90%+ sparse, modifying only ~5–10% of weights → reduces overhead and preserves retention.



 Figure 3: Grad-CAM Visualization
Shows reduced class-relevant activations in the consensus-based model → visually confirms unlearning success.



 Appendix C, D, F:
C: Comparison of merge ops → averaging performs best

D: Works across different model pool compositions → robust to validation diversity

F: Discusses extension to LLMs and VLMs, beyond image classifiers




<br/>
# refer format:     



@inproceedings{kim2025negmerge,
  title     = {NegMerge: Sign-Consensual Weight Merging for Machine Unlearning},
  author    = {Kim, Hyo Seo and Han, Dongyoon and Choe, Junsuk},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  volume    = {267},
  publisher = {PMLR},
  address   = {Vancouver, Canada},
  url       = {https://github.com/naver-ai/negmerge}
}


Kim, Hyo Seo, Dongyoon Han, and Junsuk Choe. 2025. “NegMerge: Sign-Consensual Weight Merging for Machine Unlearning.” In Proceedings of the 42nd International Conference on Machine Learning (ICML), vol. 267. Vancouver, Canada: PMLR.




