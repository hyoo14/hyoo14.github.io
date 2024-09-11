---
layout: post
title:  "[2024]Distilling CLIP with Dual Guidance for Learning Discriminative Human Body Shape Representation"  
date:   2024-09-11 18:24:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    


이 논문은 사람 재식별(Person Re-Identification, ReID) 분야에서 새로운 접근 방식을 제시합니다. 기존의 재식별 방법은 옷이나 색상과 같은 외형적 속성에 의존하지만, 이러한 방법은 장기적인 상황이나 동적인 환경에서는 한계가 있습니다. 이를 해결하기 위해 저자들은 사전 학습된 CLIP 모델을 활용한 CLIP3DReID라는 새로운 접근 방식을 제안합니다. 이 방법은 CLIP 모델을 사용하여 신체 모양을 언어적으로 묘사하고, 최적 수송 이론을 적용하여 학생 모델의 시각적 특징을 CLIP의 언어적 출력과 일치시키는 방식입니다. 이 과정에서 3D SMPL 모델을 사용하여 전역적 시각적 특징을 강화하여 더 나은 도메인 적응성을 제공합니다. CLIP3DReID는 사람의 신체 모양을 구별하는 데 탁월하며, 기존의 방법들보다 더 나은 성능을 보입니다.


This paper presents a novel approach in the field of Person Re-Identification (ReID). Traditional ReID methods rely on appearance attributes such as clothing or color, but these methods face limitations in long-term scenarios or dynamic environments. To address this, the authors propose a new approach called CLIP3DReID, which leverages the pretrained CLIP model. This method uses CLIP to automatically describe body shapes with linguistic labels and applies optimal transport theory to align the visual features of the student model with the linguistic outputs of CLIP. Additionally, the 3D SMPL model is incorporated to enhance global visual features, improving domain adaptability. CLIP3DReID excels in distinguishing human body shapes and demonstrates superior performance compared to existing methods.



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


논문의 *Methodology* 부분에서는 CLIP3DReID라는 새로운 접근 방식을 제안하며, 이 방법은 시각적-언어적 모델인 CLIP의 지식을 학습자 모델로 전수하는 과정에서 다양한 기법을 활용합니다. 주요 내용은 다음과 같습니다:

### 모델 구성 요소
1. **CLIP 모델**: 이 연구에서 CLIP(Vision-Language Pretrained Model) 모델은 사전 학습된 교사 모델로 사용됩니다. CLIP은 이미지를 시각적으로 인코딩하고, 언어적으로 신체 모양을 표현하는 텍스트로부터 특징을 추출하는 두 가지 기능을 수행합니다. 여기서는 CLIP의 텍스트 인코더와 이미지 인코더를 활용하여 신체 모양에 대한 텍스트 설명과 시각적 특징을 연결합니다.

2. **학생 모델**: 학습 모델(Student Model)로는 **ResNet-50** 아키텍처가 사용됩니다. ResNet-50은 상대적으로 가벼운 구조로, CLIP 모델에서 전수된 시각적 지식(특히 신체 모양에 관련된 지식)을 학습합니다. 학습자의 시각적 특징은 CLIP 모델로부터 전달된 정보를 기반으로 강화됩니다.

### 학습 데이터
- **실제 이미지**와 **합성 신체 이미지**를 동시에 사용합니다. 실제 이미지는 대규모 사람 재식별 데이터셋에서 가져오고, 합성 이미지는 3D SMPL(Skinned Multi-Person Linear) 모델을 사용해 생성된 3D 신체 형태로부터 얻습니다.
- **합성 이미지**는 옷이나 텍스처가 없는 신체 모양만 포함하여, 학생 모델이 본질적인 신체 모양 특징을 학습하도록 유도합니다. 합성된 신체는 다양한 신체 자세를 취하고 있으며, 이는 Celeb-reID 데이터셋에서 선택된 인간 이미지에서 추정된 자세를 사용해 생성됩니다.

### 주요 설계 요소
1. **언어적 신체 설명을 통한 라벨링**:
   - CLIP 텍스트 인코더를 사용해 신체 모양에 대한 언어적 설명(예: "근육질", "날씬한")을 자동으로 이미지에 라벨링합니다. 이 과정에서 CLIP 모델이 사전 학습된 텍스트-이미지 유사성으로부터 신체 모양 라벨을 생성합니다.
   
2. **이중 지식 전수(Dual Distillation)**:
   - CLIP 모델에서 전수된 지식을 바탕으로, 학생 모델이 이미지의 국소적 특징과 전역적 특징을 동시에 학습하도록 합니다.
   - 국소적 특징은 신체의 특정 부분(예: 어깨, 다리 등)에 집중하며, 이 과정에서 **최적 수송 이론(Optimal Transport Theory)**을 적용해 CLIP 텍스트 인코더가 생성한 신체 모양 특징과 일치시킵니다.
   - 전역적 특징은 전체 신체 모양의 시각적 표현을 학습하며, 이 과정에서 MSE 손실 함수(MSE Loss)를 사용해 학생 모델과 CLIP 이미지 인코더 간의 특징 공간을 일치시킵니다.

3. **3D 재구성 규제**:
   - 학습자가 신체 모양을 더 잘 학습할 수 있도록 3D 신체 모형(SMPL)을 활용한 규제를 추가합니다. 이 규제는 학생 모델이 3D 신체를 옷이나 텍스처 정보 없이 재구성하는 과정에서 적용되며, 본질적인 신체 모양을 학습하는 데 도움을 줍니다.
   
### 최종 학습 목표
최종 손실 함수는 다음과 같이 구성됩니다:
- **Cross-Entropy Loss (LCE)**: 이미지 간 유사성을 비교하기 위한 기본적인 손실 함수.
- **Optimal Transport Loss (LOT)**: 신체 부분의 국소적 특징을 언어적 설명과 일치시키는 손실.
- **Global Feature Alignment Loss (Lglobal)**: 전역적 시각 특징을 CLIP 이미지 인코더와 일치시키는 손실.
- **3D Reconstruction Regularization (L3D-Regu)**: 3D 신체 모양을 학습하는 규제 손실.

이 방법론을 통해 CLIP3DReID는 기존의 사람 재식별 모델보다 더 나은 성능을 보여주며, 신체 모양의 구별 능력을 크게 향상시킵니다.



The *Methodology* section of the paper introduces the novel CLIP3DReID approach, which utilizes knowledge distillation from the vision-language model CLIP into a student model. The key details are as follows:

### Model Components
1. **CLIP Model**: In this research, the pretrained CLIP (Vision-Language Pretrained Model) is used as the teacher model. CLIP performs two main functions: it encodes images visually and extracts features from linguistic descriptions of body shapes. The text and image encoders of CLIP are used to link the linguistic descriptions of body shapes with visual features.

2. **Student Model**: The student model is based on the **ResNet-50** architecture. ResNet-50, a relatively lightweight structure, learns visual knowledge related to body shapes that is transferred from the CLIP model. The visual features of the student model are strengthened by the knowledge transferred from the CLIP model.

### Training Data
- **Real images** and **synthetic body images** are used simultaneously. The real images are sourced from large-scale person ReID datasets, while the synthetic images are generated using the 3D SMPL (Skinned Multi-Person Linear) model, which creates 3D body shapes.
- **Synthetic images** consist only of body shapes without clothing or texture, encouraging the student model to focus on learning essential body shape features. The synthetic bodies are rendered in various poses, with the poses generated from human images in the Celeb-reID dataset.

### Key Design Elements
1. **Labeling with Linguistic Body Descriptions**:
   - The CLIP text encoder is used to automatically label images with linguistic body shape descriptions (e.g., "muscular," "slender"). This process leverages the pretrained text-to-image similarity of the CLIP model to generate body shape labels.
   
2. **Dual Knowledge Distillation**:
   - Knowledge is transferred from the CLIP model to the student model, which learns both local and global visual features from images.
   - Local features focus on specific body parts (e.g., shoulders, legs), and **Optimal Transport Theory** is applied to align these local visual features with the linguistic descriptions generated by the CLIP text encoder.
   - Global features capture the overall visual representation of the body shape, and an MSE loss function is used to align the feature space between the student model and the CLIP image encoder.

3. **3D Reconstruction Regularization**:
   - To further enhance the learning of body shapes, the method introduces a regularization technique using the 3D SMPL model. This regularization encourages the student model to reconstruct the body shape without relying on clothing or texture, helping the model learn more generalizable features across different domains.

### Final Training Objective
The final loss function is composed of the following:
- **Cross-Entropy Loss (LCE)**: A basic loss function for comparing the similarity between images.
- **Optimal Transport Loss (LOT)**: A loss function that aligns local body shape features with linguistic descriptions.
- **Global Feature Alignment Loss (Lglobal)**: A loss function that aligns global visual features with the CLIP image encoder.
- **3D Reconstruction Regularization (L3D-Regu)**: A regularization loss to guide the learning of 3D body shapes.

Through this methodology, CLIP3DReID demonstrates superior performance compared to existing ReID models, significantly improving the ability to distinguish body shapes.



<br/>
# Results  



논문의 결과 부분에서는 CLIP3DReID 모델이 여러 사람 재식별(Person ReID) 데이터셋에서 기존 최신 모델들과 비교하여 우수한 성능을 보였음을 보여줍니다. 주요 결과는 다음과 같습니다.

### 비교 모델
CLIP3DReID는 여러 최신 ReID 모델들과 성능을 비교했습니다:
- **ReIDCaps** (TCSVT 2020)
- **RCSAnet** (ICCV 2021)
- **CASE-Net** (WACV 2021)
- **CAL** (CVPR 2022)
- **3DInvarReID** (ICCV 2023)

이들 모델은 주로 사람의 외형적 특징(옷, 색상 등)을 학습하는 기존 방법들을 사용하거나, 옷과 자세 변화에 불변하는 3D 신체 모양 특징을 학습하는 방식 등을 적용한 모델들입니다.

### 데이터셋
CLIP3DReID는 다양한 데이터셋에서 평가되었습니다. 이 데이터셋들은 옷을 바꾸거나 다양한 활동을 하는 사람들을 포함하는 **cloth-changing ReID** 상황을 주로 다룹니다:
- **Celeb-reID** 및 **Celeb-reID-light**: 옷을 바꾸는 장기적인 재식별 시나리오를 다룹니다.
- **PRCC**: 옷을 바꾸는 상황에서 촬영된 이미지 데이터셋.
- **LTCC**: 다양한 옷 변화와 관련된 데이터셋.
- **CCVID**, **DeepChange**, **CCDA**: 최근 소개된 cloth-changing ReID 데이터셋으로, 다양한 인간 활동을 포함합니다.
- **Market-1501** 및 **MSMT17**: 일반적인 단기 재식별을 위한 데이터셋.

### 성능 비교
#### Celeb-reID 및 Celeb-reID-light 데이터셋
- CLIP3DReID는 **Celeb-reID-light**에서 평균 정밀도(mAP)를 21.8%에서 **26.3%**로 향상시켰으며, Rank-1 정확도도 37.0%에서 **39.4%**로 향상되었습니다.
- **Celeb-reID**에서는 mAP가 19.2%, Rank-1 정확도가 63.1%로 기존 모델들보다 뛰어난 성능을 보였습니다.

#### 얼굴이 흐려진 Celeb-reID 데이터셋
- 얼굴이 흐려진(익명화된) 버전에서도 CLIP3DReID는 mAP가 **11.6%**, Rank-1 정확도가 **52.8%**로, 기존 모델들보다 높은 성능을 보였습니다. 이는 얼굴이 가려져도 신체 모양만으로 높은 재식별 성능을 유지할 수 있음을 보여줍니다.

#### PRCC 및 LTCC 데이터셋
- PRCC에서는 mAP가 59.3%, Rank-1 정확도가 60.6%로 기존의 **3DInvarReID** 모델(57.2% mAP, 56.5% Rank-1)을 능가했습니다.
- LTCC에서는 mAP가 21.7%, Rank-1 정확도가 42.1%로 기존 모델보다 성능이 더 높았습니다.

#### CCVID, DeepChange, CCDA 데이터셋
- CCDA에서는 mAP가 25.7%, Rank-1 정확도가 15.5%로 가장 높은 성능을 기록했습니다. 이는 특히 다양한 활동을 포함한 데이터셋에서 CLIP3DReID의 강점을 잘 보여줍니다.

#### 단기 ReID 데이터셋 (Market-1501, MSMT17)
- CLIP3DReID는 장기 재식별에 초점을 맞추었지만, 단기 재식별 데이터셋에서도 성능을 테스트했습니다. **Market-1501**에서는 mAP가 88.4%, Rank-1 정확도가 95.6%로 기존 성능을 약간 향상시켰습니다.
- **MSMT17**에서도 mAP가 61.2%, Rank-1 정확도가 81.5%로 기존 모델보다 성능이 향상되었습니다.

#### 교차 데이터셋 실험
- LTCC에서 훈련하고 PRCC에서 테스트한 결과, mAP가 37.5%, Rank-1 정확도가 41.7%로 기존 모델들보다 성능이 향상되었습니다. 이는 다른 데이터셋에서도 모델이 잘 일반화됨을 보여줍니다.

### 요약
CLIP3DReID는 기존 최신 모델들과 비교했을 때 **모든 데이터셋에서 성능이 향상**되었으며, 특히 신체 모양을 기반으로 한 재식별 성능이 두드러졌습니다. 이 모델은 얼굴을 흐리거나 옷을 바꾼 상황에서도 높은 정확도를 유지할 수 있음을 보여주었고, 다양한 옷 변화와 활동이 포함된 복잡한 상황에서도 기존 방법들보다 뛰어난 성능을 보였습니다.


The results section of the paper demonstrates that the CLIP3DReID model outperforms several state-of-the-art models across various person Re-Identification (ReID) datasets. The key findings are as follows:

### Compared Models
CLIP3DReID was compared with several state-of-the-art ReID models, including:
- **ReIDCaps** (TCSVT 2020)
- **RCSAnet** (ICCV 2021)
- **CASE-Net** (WACV 2021)
- **CAL** (CVPR 2022)
- **3DInvarReID** (ICCV 2023)

These models typically rely on learning appearance features (e.g., clothing and color) or focus on extracting clothing and pose-invariant 3D body shape features.

### Datasets
CLIP3DReID was evaluated on several datasets that primarily address **cloth-changing ReID** scenarios:
- **Celeb-reID** and **Celeb-reID-light**: Focus on long-term scenarios where individuals change clothing.
- **PRCC**: A dataset containing images of people changing their clothes.
- **LTCC**: A dataset that involves various clothing changes.
- **CCVID**, **DeepChange**, **CCDA**: Recent cloth-changing ReID datasets that include diverse human activities.
- **Market-1501** and **MSMT17**: Conventional short-term ReID datasets.

### Performance Comparison
#### Celeb-reID and Celeb-reID-light Datasets
- On **Celeb-reID-light**, CLIP3DReID improved the mean Average Precision (mAP) from 21.8% to **26.3%** and increased Rank-1 accuracy from 37.0% to **39.4%**.
- On **Celeb-reID**, CLIP3DReID achieved an mAP of 19.2% and a Rank-1 accuracy of 63.1%, outperforming previous models.

#### Face-Blurred Celeb-reID Datasets
- In face-anonymized versions of the Celeb-reID dataset, CLIP3DReID achieved an mAP of **11.6%** and a Rank-1 accuracy of **52.8%**, outperforming other baseline models. This indicates the model’s ability to maintain high ReID accuracy even when facial details are obscured.

#### PRCC and LTCC Datasets
- On **PRCC**, CLIP3DReID achieved an mAP of 59.3% and a Rank-1 accuracy of 60.6%, outperforming **3DInvarReID** (57.2% mAP, 56.5% Rank-1).
- On **LTCC**, it achieved an mAP of 21.7% and Rank-1 accuracy of 42.1%, showing higher performance than previous models.

#### CCVID, DeepChange, and CCDA Datasets
- On the **CCDA** dataset, CLIP3DReID achieved the highest mAP of 25.7% and Rank-1 accuracy of 15.5%. This highlights the model’s strength in diverse real-world scenarios with varied human activities.

#### Short-term ReID Datasets (Market-1501, MSMT17)
- Although CLIP3DReID is designed for long-term scenarios, it was also tested on short-term datasets. On **Market-1501**, it slightly improved performance with an mAP of 88.4% and a Rank-1 accuracy of 95.6%.
- On **MSMT17**, it achieved an mAP of 61.2% and a Rank-1 accuracy of 81.5%, outperforming existing models.

#### Cross-Dataset Experiments
- When trained on LTCC and tested on PRCC, CLIP3DReID achieved an mAP of 37.5% and Rank-1 accuracy of 41.7%, surpassing other models. This demonstrates the model’s strong generalization across different datasets.

### Summary
CLIP3DReID consistently outperformed state-of-the-art models across all datasets, showing significant improvements in distinguishing body shapes for person ReID. The model maintained high accuracy even in challenging scenarios, such as when faces were blurred or clothing was changed. It also demonstrated superior performance in complex datasets involving diverse clothing changes and activities.



<br/>
# 예시  


예를 들어, **Celeb-reID** 데이터셋을 기반으로 설명해보겠습니다.

### 이전의 문제점
기존의 사람 재식별(Person ReID) 시스템은 주로 옷, 색상, 액세서리 등 **외형적 특징**에 크게 의존했습니다. 예를 들어, **파란 셔츠를 입고 빨간 모자를 쓴 남성**이 특정 카메라에 잡혔다고 해봅시다. 이 시스템은 그 남성을 재식별할 때 **파란 셔츠와 빨간 모자**라는 외형적 요소에 의존하게 됩니다. 그런데 시간이 지나서 이 남성이 **흰 셔츠와 검정 모자**를 입고 다른 카메라에 등장했을 경우, 기존 시스템은 이 사람을 동일 인물로 인식하지 못하는 한계가 있었습니다. 즉, 외형이 바뀌면 동일 인물로 인식하기 어렵다는 것입니다.

### CLIP3DReID의 개선점
CLIP3DReID는 **신체 모양**에 중점을 둬서 사람을 재식별합니다. 같은 남성이 처음에는 파란 셔츠와 빨간 모자를 입고 있고, 나중에 흰 셔츠와 검정 모자를 입었을 때도 **그 사람의 신체 모양**(예를 들어, **넓은 어깨**나 **긴 다리**)을 통해 동일 인물임을 인식할 수 있습니다. 이 모델은 **옷이나 액세서리의 변화**에도 불구하고, 신체 모양이나 구조적 특징(예: 체격, 어깨 너비, 다리 길이)을 기반으로 사람을 재식별할 수 있습니다.

### 실험 결과
**Celeb-reID 데이터셋**에서, 기존 모델들은 이와 같은 상황에서 재식별 성능이 크게 떨어졌습니다. 예를 들어, Rank-1 정확도는 50-60% 정도였지만, CLIP3DReID는 이러한 변화에도 불구하고 **63.1%의 Rank-1 정확도**를 달성했습니다. 이 향상된 성능은 모델이 옷이나 액세서리와 상관없이 사람의 고유한 신체 모양을 효과적으로 학습했다는 것을 보여줍니다. 

따라서, 같은 사람이 다른 옷을 입고 다른 환경에 나타났을 때도 CLIP3DReID는 높은 정확도로 동일 인물임을 인식할 수 있었습니다.



### Previous Limitations
Traditional person re-identification (ReID) systems mainly rely on **appearance features** such as clothing, color, and accessories. For instance, imagine a man wearing a **blue shirt and a red hat** captured by one camera. The system would primarily identify this individual based on his **blue shirt and red hat**. However, after some time, if this man appears in another camera wearing a **white shirt and black hat**, the traditional system would struggle to recognize him as the same person. In other words, these systems have difficulty recognizing the same individual if their appearance changes.

### How CLIP3DReID Improved This
CLIP3DReID focuses on identifying individuals based on **body shape**. In this example, even though the man initially wore a blue shirt and red hat and later changed to a white shirt and black hat, CLIP3DReID could recognize him as the same person by focusing on **his body shape** (e.g., **broad shoulders** or **long legs**). The model uses structural features like physique, shoulder width, and leg length, which remain consistent regardless of changes in clothing or accessories.

### Experimental Results
In the **Celeb-reID dataset**, traditional models significantly underperformed in such scenarios, with Rank-1 accuracy around 50-60%. However, CLIP3DReID maintained a **63.1% Rank-1 accuracy**, even when individuals changed clothing. This improvement demonstrates that the model effectively learned to distinguish people based on **inherent body shape features**, rather than relying solely on superficial appearance details.

As a result, when the same individual appears in different clothing and environments, CLIP3DReID can still accurately recognize them based on their unique body shape, leading to higher identification accuracy.



<br/>  
# 요약 


CLIP3DReID는 사전 학습된 CLIP 모델을 사용하여 신체 모양을 언어적 설명과 연결하고, 최적 수송 이론을 통해 시각적 특징과 정렬하는 새로운 사람 재식별(ReID) 방법입니다. 학생 모델로 **ResNet-50**을 사용하며, 신체 모양과 같은 본질적인 특징을 학습해 옷이나 외형적 변화에도 동일 인물을 인식할 수 있도록 설계되었습니다. 이 모델은 **Celeb-reID**, **PRCC**, **LTCC** 등 다양한 옷 변화 데이터셋에서 기존 최신 모델들보다 성능을 크게 향상시켰습니다. 특히, Celeb-reID에서 CLIP3DReID는 Rank-1 정확도 63.1%를 달성하며, 얼굴이 흐려진 상황에서도 높은 정확도를 유지했습니다. 이를 통해 신체 모양을 중심으로 사람을 구별하는 능력이 크게 개선되었음을 입증했습니다.


CLIP3DReID is a novel person Re-Identification (ReID) method that uses a pretrained CLIP model to link body shape descriptions with visual features, aligning them through optimal transport theory. The student model, based on **ResNet-50**, learns essential body shape features, enabling it to recognize individuals even when clothing or appearance changes. The model significantly outperformed existing state-of-the-art models across various cloth-changing datasets, including **Celeb-reID**, **PRCC**, and **LTCC**. In particular, CLIP3DReID achieved a Rank-1 accuracy of 63.1% on Celeb-reID and maintained high accuracy even in face-blurred scenarios. This demonstrates a substantial improvement in distinguishing individuals based on body shape.  



# 기타  


<br/>
# refer format:     

@inproceedings{liu2024distilling,
  title={Distilling CLIP with Dual Guidance for Learning Discriminative Human Body Shape Representation},
  author={Liu, Feng and Kim, Minchul and Ren, Zhiyuan and Liu, Xiaoming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
  organization={IEEE}
}




Liu, Feng, Minchul Kim, Zhiyuan Ren, and Xiaoming Liu. "Distilling CLIP with Dual Guidance for Learning Discriminative Human Body Shape Representation." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.






