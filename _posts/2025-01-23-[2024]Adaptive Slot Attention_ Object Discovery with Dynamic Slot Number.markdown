---
layout: post
title:  "[2024]Adaptive Slot Attention: Object Discovery with Dynamic Slot Number"  
date:   2025-01-23 22:00:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

비전트랜스포머의 피처(패치)의 셀프어텐션 스코어 바탕으로 이미지의 복잡도(객체 수)에 따라 슬롯 개수(인식 가능한 객체 개수)를 동적으로 조정   
* 단순한 이미지: 모든 패치가 유사 → 어텐션 점수가 비슷 → 슬롯 수를 줄임.  
* 복잡한 이미지: 패치 간 차이가 큼 → 어텐션 점수가 다양 → 슬롯 수를 늘림.  

짧은 요약(Abstract) :    




본 논문은 객체 중심 학습(Object-centric Learning, OCL)에서 슬롯(Slot)을 활용하여 객체의 표현을 추출하는 방법론을 제안합니다. 기존 슬롯 주의(Slot Attention) 방법은 데이터셋에 따른 사전 정의된 슬롯 수에 의존하지만, 이는 객체 수의 가변성을 반영하지 못하는 한계가 있습니다. 이를 해결하기 위해, 데이터의 내용에 따라 최적의 슬롯 수를 동적으로 결정하는 **적응형 슬롯 주의(AdaSlot)** 메커니즘이 포함된 복잡도 인지 객체 오토인코더(Complexity-aware Object Auto-encoder) 프레임워크를 제안합니다. 이 메커니즘은 후보 슬롯 목록에서 적합한 슬롯 수를 선택하는 이산 슬롯 샘플링 모듈과, 선택되지 않은 슬롯을 디코딩 과정에서 억제하는 마스킹 슬롯 디코더를 도입합니다. 다양한 데이터셋에서의 실험 결과, 본 방법은 고정 슬롯 모델과 동등하거나 우수한 성능을 보이며, 각 인스턴스의 복잡성에 따라 슬롯 수를 동적으로 적응시킬 수 있는 잠재력을 입증하였습니다.

---



Object-centric learning (OCL) extracts the representation of objects with slots, offering an exceptional blend of flexibility and interpretability for abstracting low-level perceptual features. A widely adopted method within OCL is slot attention, which utilizes attention mechanisms to iteratively refine slot representations. However, a major drawback of most object-centric models, including slot attention, is their reliance on predefining the number of slots. This not only necessitates prior knowledge of the dataset but also overlooks the inherent variability in the number of objects present in each instance. To overcome this fundamental limitation, we present a novel complexity-aware object auto-encoder framework. Within this framework, we introduce an adaptive slot attention (AdaSlot) mechanism that dynamically determines the optimal number of slots based on the content of the data. This is achieved by proposing a discrete slot sampling module that is responsible for selecting an appropriate number of slots from a candidate list. Furthermore, we introduce a masked slot decoder that suppresses unselected slots during the decoding process. Our framework, tested extensively on object discovery tasks with various datasets, shows performance matching or exceeding top fixed-slot models. Moreover, our analysis substantiates that our method exhibits the capability to dynamically adapt the slot number according to each instance’s complexity, offering the potential for further exploration in slot attention research.



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





이 논문에서는 **복잡도 인지 객체 오토인코더(Complexity-aware Object Auto-encoder)**라는 새로운 프레임워크를 제안했습니다. 이를 통해 기존의 슬롯 수를 고정적으로 정의해야 하는 문제를 해결하고, 데이터의 복잡성에 따라 동적으로 슬롯 수를 조정할 수 있는 방법을 제공합니다.

1. **슬롯 수 상한 설정**: 최대 슬롯 수(\(K_{max}\))를 데이터셋의 최대 객체 수에 기반하여 설정합니다. 예를 들어, 특정 데이터셋에 최대 10개의 객체가 포함될 수 있다면 \(K_{max}=10\)으로 설정됩니다.

2. **슬롯 샘플링 및 선택**: 각 데이터 인스턴스에 대해 샘플링 방법(\(\pi\))을 학습하여 어떤 슬롯을 유지하거나 제거할지를 결정합니다. Gumbel-Softmax를 사용하여 이산 슬롯 샘플링 과정을 차별적으로 학습 가능하게 설계하였습니다. 샘플링은 각 슬롯에 대해 독립적으로 수행되며, 최적화 효율성을 높이기 위해 Mean-field 방식을 사용합니다.

3. **마스킹 슬롯 디코더**: 선택되지 않은 슬롯의 정보를 제거하기 위해 마스킹 슬롯 디코더를 도입했습니다. 이는 슬롯의 알파 마스크(가중치)를 직접 조정하여 정보 손실을 최소화하고, 디코딩 시 불필요한 슬롯의 영향을 억제합니다.

4. **복잡도 기반 정규화**: 슬롯 수를 과도하게 사용하는 것을 방지하기 위해 복잡도 기반 정규화 항(\(L_{reg}\))을 도입했습니다. 이 정규화 항은 인스턴스의 복잡성에 따라 적절한 슬롯 수를 유지하도록 설계되었습니다.

**트레이닝 데이터 및 구현**:
- **데이터셋**: CLEVR10(간단한 객체 중심 데이터셋), MOVi-C/E(복잡한 합성 데이터셋), MS COCO(복잡한 실제 이미지 데이터셋)를 사용하여 모델을 평가했습니다.
- **모델 구조**: DINO 기반 ViT/B-16을 사용하여 고정된 피처를 추출하고, MLP를 활용한 디코더를 통해 슬롯의 정보를 복원합니다.
- **트레이닝 설정**: Adam 옵티마이저를 사용하며, 학습률은 0.0004로 설정하고, 10k 스텝 동안 선형 웜업, 이후 지수 감소 방식을 적용하였습니다. 주요 실험은 50만 스텝 동안 수행되었으며, 랜덤 시드 3개를 사용해 평균 결과를 제시했습니다.

---



This paper introduces a novel **complexity-aware object auto-encoder framework** to overcome the limitations of fixed slot numbers in traditional slot attention methods. The proposed framework dynamically adjusts the number of slots based on the complexity of the data.

1. **Setting an Upper Bound for Slot Numbers**: The maximum slot number (\(K_{max}\)) is determined based on the dataset's maximum number of objects. For example, if a dataset contains up to 10 objects per instance, \(K_{max}=10\).

2. **Slot Sampling and Selection**: A sampling method (\(\pi\)) is learned for each instance to determine which slots to keep or drop. Gumbel-Softmax is used to make this discrete slot sampling process differentiable. Sampling is performed independently for each slot using the mean-field approach to enhance computational efficiency.

3. **Masked Slot Decoder**: A masked slot decoder is introduced to remove the influence of unused slots during decoding. By directly manipulating the alpha masks (weights) of dropped slots, this decoder suppresses unnecessary information while maintaining reconstruction quality.

4. **Complexity-aware Regularization**: A complexity-based regularization term (\(L_{reg}\)) is introduced to discourage excessive use of slots. This term ensures that the model retains an appropriate number of slots based on the complexity of each instance.

**Training Data and Implementation**:
- **Datasets**: The model is evaluated using CLEVR10 (a toy object-centric dataset), MOVi-C/E (synthetic datasets with complex scenes), and MS COCO (a challenging real-world dataset).
- **Model Architecture**: DINO-based ViT/B-16 is employed for frozen feature extraction, and an MLP-based decoder reconstructs slot information.
- **Training Setup**: The Adam optimizer is used with a learning rate of 0.0004, 10k steps of linear warmup, followed by exponential decay. The main experiments were conducted over 500k steps, with results averaged across three random seeds.


   
 
<br/>
# Results  




본 연구에서는 제안된 **AdaSlot 모델**이 다양한 데이터셋과 메트릭에서 기존 고정 슬롯 기반 모델들과 비교했을 때 뛰어난 성능을 보임을 확인했습니다.

1. **테스트 데이터**:
   - **CLEVR10**: 간단한 객체 중심 데이터셋으로, 픽셀 수준 재구성 품질 평가에 사용되었습니다.
   - **MOVi-C/E**: 최대 10개의 객체(MOVi-C)와 23개의 객체(MOVi-E)를 포함한 복잡한 합성 데이터셋.
   - **MS COCO**: 현실 세계의 복잡한 이미지 데이터셋으로, 불균형한 객체 수와 다양한 크기의 객체를 포함하고 있습니다.

2. **메트릭**:
   - **객체 분류**: ARI(Adjusted Rand Index), Precision, Recall, F1 Score.
   - **객체 매칭**: Purity, mBO(mean Intersection-Over-Union), CorLoc(정확히 로컬라이즈된 이미지 비율).
   - **정보 이론 기반 메트릭**: NMI(Normalized Mutual Information), AMI(Adjusted Mutual Information).

3. **비교 모델**:
   - **GENESIS-V2**: 확률적 클러스터링 기반 객체 중심 모델.
   - **DINOSAUR**: 고정된 슬롯 수를 사용하는 슬롯 주의 모델.
   - **AdaSlot**: 본 연구에서 제안된 복잡도 인지 슬롯 모델.

4. **결과 요약**:
   - **CLEVR10**: AdaSlot은 객체 수를 정확히 결정하여 슬롯 중복 문제를 해결했으며, 픽셀을 실제 객체 수에 따라 정확히 그룹화하였습니다.
   - **MOVi-C/E**:
     - AdaSlot은 **ARI**와 **F1 Score**에서 GENESIS-V2 및 DINOSAUR를 모두 능가했습니다.
     - 특히, MOVi-E에서 AdaSlot은 Recall에서 12포인트 이상 높은 성능을 보이며, 객체 수가 많은 이미지에서 성능이 탁월했습니다.
   - **MS COCO**: AdaSlot은 고정 슬롯 모델과 비슷한 성능을 보였지만, 복잡한 객체 구성이 포함된 이미지에서 여전히 경쟁력 있는 결과를 보였습니다.
   - 전반적으로, AdaSlot은 **동적으로 슬롯 수를 조정**하며 데이터 복잡성을 고려한 우수한 객체 중심 학습 성능을 입증했습니다.

---



The proposed **AdaSlot model** demonstrated superior performance compared to existing fixed-slot models across various datasets and metrics.

1. **Test Datasets**:
   - **CLEVR10**: A simple object-centric dataset used for evaluating pixel-level reconstruction quality.
   - **MOVi-C/E**: Complex synthetic datasets with up to 10 objects (MOVi-C) and 23 objects (MOVi-E).
   - **MS COCO**: A challenging real-world dataset with imbalanced object counts and varying object sizes.

2. **Metrics**:
   - **Object Grouping**: ARI (Adjusted Rand Index), Precision, Recall, F1 Score.
   - **Object Matching**: Purity, mBO (mean Intersection-Over-Union), CorLoc (proportion of images with correctly localized objects).
   - **Information-Theoretic Metrics**: NMI (Normalized Mutual Information), AMI (Adjusted Mutual Information).

3. **Comparison Models**:
   - **GENESIS-V2**: A probabilistic clustering-based object-centric model.
   - **DINOSAUR**: A fixed-slot model using slot attention.
   - **AdaSlot**: The complexity-aware slot model proposed in this study.

4. **Summary of Results**:
   - **CLEVR10**: AdaSlot effectively resolved slot duplication issues by accurately determining the number of objects and grouping pixels based on actual object counts.
   - **MOVi-C/E**:
     - AdaSlot outperformed both GENESIS-V2 and DINOSAUR in **ARI** and **F1 Score**.
     - On MOVi-E, AdaSlot achieved a significantly higher recall (over 12 points) compared to fixed-slot models, excelling in images with numerous objects.
   - **MS COCO**: AdaSlot achieved competitive performance with fixed-slot models, showing strong results even in complex, real-world image scenarios.
   - Overall, AdaSlot validated its ability to **dynamically adjust slot numbers**, achieving excellent object-centric learning performance by considering data complexity.



<br/>
# 예제  



다음은 AdaSlot 모델과 기존 고정 슬롯 모델(GENESIS-V2, DINOSAUR) 간의 성능 차이를 보여주는 예제입니다.

#### 1. 테스트 데이터 예시: MOVi-E
- MOVi-E는 현실적인 배경과 복잡한 객체 구성을 가진 합성 데이터셋으로, 한 이미지에 최대 23개의 객체가 포함됩니다.
- 예를 들어, 한 이미지에 다음과 같은 객체가 포함될 수 있습니다:
  - **이미지 구성**: 테이블 위에 있는 다양한 크기의 컵, 접시, 과일, 그리고 장식품.
  - **객체 수**: 13개.

#### 2. 제안 모델(AdaSlot) 성능:
- AdaSlot은 이미지 내 복잡성을 기반으로 슬롯 수를 **13개**로 조정하여 정확히 각 객체를 인식합니다.
- 객체 간 중복 슬롯이 없으며, 배경 픽셀을 제외하고 각 슬롯이 정확히 하나의 객체를 나타냅니다.
- **결과**: 높은 ARI와 F1 점수, 정확한 객체 그룹화.

#### 3. 비교 모델(GENESIS-V2, DINOSAUR) 성능:
- GENESIS-V2:
  - 고정된 슬롯 수로 인해 객체 수가 13개임에도 슬롯을 **6개**로 제한.
  - 결과적으로 일부 객체는 동일한 슬롯에 배치되거나, 배경이 잘못 분할됨.
- DINOSAUR:
  - 슬롯 수를 **24개**로 설정.
  - 일부 객체는 여러 슬롯에 중복 분할되며, 불필요한 슬롯이 생성됨.
- **결과**: 낮은 Recall과 Purity 점수, 과소/과잉 분할 문제.

---



Below is an example illustrating the performance difference between the proposed AdaSlot model and existing fixed-slot models (GENESIS-V2, DINOSAUR).

#### 1. Test Data Example: MOVi-E
- MOVi-E is a synthetic dataset with realistic backgrounds and complex object compositions, containing up to 23 objects per image.
- For instance, an image might include the following:
  - **Image Composition**: A table with various sizes of cups, plates, fruits, and decorations.
  - **Number of Objects**: 13.

#### 2. Performance of the Proposed Model (AdaSlot):
- AdaSlot dynamically adjusts the number of slots to **13** based on the image complexity.
- There is no overlap between slots, and each slot precisely represents a single object, excluding background pixels.
- **Result**: High ARI and F1 scores, accurate object grouping.

#### 3. Performance of Comparison Models (GENESIS-V2, DINOSAUR):
- GENESIS-V2:
  - Fixed slot count of **6** fails to capture all 13 objects.
  - Some objects are grouped into the same slot, while background segmentation errors occur.
- DINOSAUR:
  - Fixed slot count of **24** results in multiple slots representing the same object, creating redundant slots.
- **Result**: Low Recall and Purity scores, under-segmentation or over-segmentation issues.



<br/>  
# 요약   




AdaSlot 모델은 객체 중심 학습에서 슬롯 수를 동적으로 조정할 수 있는 프레임워크를 제안하여 기존 고정 슬롯 모델의 한계를 극복합니다. MOVi-E 데이터셋에서 AdaSlot은 이미지 내 객체 수(예: 13개)에 따라 슬롯 수를 자동으로 조정하며, 정확한 객체 분류와 배경 제거를 달성합니다. 비교 모델인 GENESIS-V2는 슬롯 수를 6개로 제한하여 객체를 충분히 캡처하지 못했고, DINOSAUR는 슬롯 수를 24개로 설정하여 중복 분할과 불필요한 슬롯이 생성되었습니다. AdaSlot은 ARI와 F1 점수에서 GENESIS-V2와 DINOSAUR를 모두 능가하며, 과소/과잉 분할 문제를 성공적으로 해결했습니다. 이러한 결과는 AdaSlot의 데이터 복잡성 기반 슬롯 조정 능력이 다양한 객체 중심 학습 작업에 유용함을 입증합니다.

---



The AdaSlot model introduces a framework that dynamically adjusts the number of slots in object-centric learning, addressing the limitations of fixed-slot models. On the MOVi-E dataset, AdaSlot automatically adapts the slot number to the number of objects in an image (e.g., 13), achieving precise object classification and background removal. In contrast, the comparison model GENESIS-V2 limited the slot count to 6, failing to capture all objects, while DINOSAUR used 24 slots, resulting in redundant slots and over-segmentation. AdaSlot outperformed both GENESIS-V2 and DINOSAUR in ARI and F1 scores, successfully mitigating under-segmentation and over-segmentation issues. These results demonstrate AdaSlot’s ability to dynamically adjust slot numbers based on data complexity, proving its utility for various object-centric learning tasks.


<br/>  
# 기타  




#### 1. **그림 1: 슬롯 수에 따른 분할 품질**
- 이 그림은 원본 이미지와 함께 슬롯 수를 다르게 설정했을 때 발생하는 **과소 분할(Under-segmentation)**, 적절한 분할(Proper-segmentation), 그리고 **과잉 분할(Over-segmentation)**의 사례를 보여줍니다.
- 슬롯 수가 적으면 객체를 제대로 구분하지 못하고(과소 분할), 슬롯 수가 너무 많으면 불필요한 객체로 분할이 이루어집니다(과잉 분할).

#### 2. **그림 2: AdaSlot의 파이프라인**
- AdaSlot은 이미지 특징을 추출한 뒤, Gumbel-Softmax를 활용해 중요한 슬롯을 선택하고, 선택된 슬롯만을 사용하여 디코딩하는 과정을 나타냅니다.
- 슬롯 선택 모듈과 마스킹 슬롯 디코더의 구조가 강조되어 있습니다.

#### 3. **그림 3: 데이터셋별 슬롯 수 비교**
- MOVi-C, MOVi-E, 그리고 COCO 데이터셋에서 AdaSlot과 고정 슬롯 모델(DINOSAUR)의 객체 분할 결과를 비교한 이미지입니다.
- AdaSlot은 인스턴스의 복잡성에 따라 슬롯 수를 조정하여 객체를 적절히 분할하며, 고정 슬롯 모델은 과소 분할 또는 과잉 분할 문제가 발생합니다.

#### 4. **그림 4: 슬롯별 분할 시각화**
- AdaSlot과 고정된 슬롯 수를 가진 모델(예: 11-slot)의 분할 결과를 비교한 그림입니다.
- AdaSlot은 불필요한 슬롯을 제거하고 객체 수에 맞는 정확한 분할을 수행합니다.

#### 5. **그림 5: 객체 수에 따른 메트릭 변화**
- MOVi-C와 MOVi-E 데이터셋에서 객체 수에 따른 ARI, Precision, Recall, mBO 변화를 나타냅니다.
- AdaSlot은 고정 슬롯 모델과 비교해 넓은 범위에서 우수한 성능을 보이며, 객체 수에 적응적으로 대응합니다.

#### 6. **그림 6: 슬롯 예측과 실제 객체 수 비교**
- MOVi-C와 MOVi-E 데이터셋에서 실제 객체 수와 AdaSlot의 예측 슬롯 수를 비교한 히트맵입니다.
- AdaSlot은 슬롯 수를 다이나믹하게 조정하며, 예측값과 실제값 간의 높은 일치를 보입니다.

#### 7. **테이블 1-3: 각 데이터셋의 성능 결과**
- MOVi-C, MOVi-E, COCO 데이터셋에서 AdaSlot과 기존 모델(GENESIS-V2, DINOSAUR)의 주요 메트릭 비교 결과를 제시합니다.
- AdaSlot은 ARI, F1 Score, Recall, Purity 등에서 기존 모델보다 전반적으로 높은 점수를 기록했습니다.

#### 8. **테이블 4: 객체 속성 예측**
- MOVi-C 데이터셋에서 AdaSlot과 고정 슬롯 모델의 객체 속성 예측 성능을 비교한 테이블입니다.
- AdaSlot은 Jaccard Index에서 가장 높은 점수를 기록하며, 속성 예측에서도 우수한 성능을 보였습니다.

---



#### 1. **Figure 1: Segmentation Quality Based on Slot Numbers**
- This figure illustrates examples of **under-segmentation**, proper segmentation, and **over-segmentation** based on varying slot numbers.
- Too few slots result in poor object distinction (under-segmentation), while too many slots lead to unnecessary splitting (over-segmentation).

#### 2. **Figure 2: AdaSlot Pipeline**
- The pipeline shows AdaSlot extracting image features, selecting important slots using Gumbel-Softmax, and decoding only the selected slots.
- The slot selection module and masked slot decoder are emphasized.

#### 3. **Figure 3: Slot Count Comparison Across Datasets**
- This figure compares the object segmentation results of AdaSlot and fixed-slot models (e.g., DINOSAUR) on MOVi-C, MOVi-E, and COCO datasets.
- AdaSlot dynamically adjusts slot numbers based on instance complexity, achieving better segmentation, while fixed-slot models suffer from under- or over-segmentation.

#### 4. **Figure 4: Per-Slot Segmentation Visualization**
- A comparison of AdaSlot and a fixed-slot model (e.g., 11-slot) shows AdaSlot removing unnecessary slots and performing accurate segmentation based on object counts.

#### 5. **Figure 5: Metric Variations by Object Count**
- This figure shows how ARI, Precision, Recall, and mBO vary with object counts on MOVi-C and MOVi-E datasets.
- AdaSlot outperforms fixed-slot models over a wide range of metrics, adapting effectively to varying object counts.

#### 6. **Figure 6: Comparison of Predicted and Actual Object Counts**
- A heatmap comparing actual object counts and AdaSlot’s predicted slot counts on MOVi-C and MOVi-E datasets.
- AdaSlot dynamically adjusts slot numbers, showing high alignment between predictions and ground truths.

#### 7. **Tables 1-3: Performance Results by Dataset**
- These tables present the performance metrics of AdaSlot and baseline models (GENESIS-V2, DINOSAUR) on MOVi-C, MOVi-E, and COCO datasets.
- AdaSlot achieves higher scores in ARI, F1 Score, Recall, and Purity compared to baseline models.

#### 8. **Table 4: Object Property Prediction**
- This table compares the object property prediction performance of AdaSlot and fixed-slot models on the MOVi-C dataset.
- AdaSlot achieves the highest Jaccard Index, demonstrating superior performance in property prediction tasks.


<br/>
# refer format:     



@inproceedings{fan2024adaptive,
  title={Adaptive Slot Attention: Object Discovery with Dynamic Slot Number},
  author={Fan, Ke and Bai, Zechen and Xiao, Tianjun and He, Tong and Horn, Max and Fu, Yanwei and Locatello, Francesco and Zhang, Zheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
  url={https://kfan21.github.io/AdaSlot/}
}



Fan, Ke, Zechen Bai, Tianjun Xiao, Tong He, Max Horn, Yanwei Fu, Francesco Locatello, and Zheng Zhang. "Adaptive Slot Attention: Object Discovery with Dynamic Slot Number." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. Accessible at https://kfan21.github.io/AdaSlot/.








