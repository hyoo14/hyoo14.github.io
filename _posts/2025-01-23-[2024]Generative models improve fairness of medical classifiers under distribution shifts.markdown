---
layout: post
title:  "[2024]Generative models improve fairness of medical classifiers under distribution shifts"  
date:   2025-01-23 21:43:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

Diffusion Models로 의료 이미지 생성,  생성된 이미지로 학습해서 fairness(성별/인종/병원간 테스트), robustness(OOD로) 높임   


짧은 요약(Abstract) :    





이 연구는 **생성 모델**이 의료 기계 학습 모델의 공정성을 개선하고, 배포 환경의 데이터 분포 차이(Out-of-Distribution, OOD)에 대응하는 데 효과적이라는 점을 제시합니다. 특히 **확산 모델(diffusion models)**을 활용하여 **현실적인 합성 데이터를 생성**하고 이를 훈련 데이터에 포함시켜 데이터 부족 문제를 해결하며 공정성과 강건성을 동시에 증진시킬 수 있음을 보여줍니다. 연구는 세 가지 의료 이미징 맥락에서 실험을 수행했으며, 병리학(histopathology), 흉부 엑스레이(chest X-ray), 피부과(dermatology) 데이터에서 **모델의 진단 정확성과 공정성을 개선**했습니다. 이러한 접근법은 특히 **대표성이 부족한 집단**의 진단 정확도를 높이고, 데이터가 부족한 상황에서도 효율적인 학습을 가능하게 합니다.

---



This study demonstrates that **generative models** can enhance the fairness of medical machine learning models and address distribution shifts during deployment. By leveraging **diffusion models** to generate **realistic synthetic data**, this approach mitigates data scarcity while improving robustness and fairness simultaneously. Experiments were conducted across three medical imaging contexts: histopathology, chest X-ray, and dermatology. The results revealed significant improvements in **diagnostic accuracy and fairness**, particularly for underrepresented groups. This method efficiently facilitates learning in data-scarce environments and promotes equitable diagnostic performance.



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




이 연구에서는 **확산 모델(Diffusion Models)**을 활용하여 의료 기계 학습 시스템의 공정성과 강건성을 개선하는 방법론을 제안했습니다. 이 접근법은 **3단계**로 구성됩니다.

1. **생성 모델 훈련**: 
   - 주어진 레이블된 데이터와 레이블되지 않은 데이터를 사용하여 **확산 모델**을 훈련합니다. 레이블 데이터는 특정 병원에서 수집된 데이터를 사용하며, 레이블되지 않은 데이터는 다양한 병원 또는 환경에서 수집된 데이터로 구성됩니다. 
   - 모델은 진단 레이블이나 병원 ID와 같은 속성에 대해 조건화(Conditioning)됩니다. 

2. **합성 데이터 샘플링**: 
   - 훈련된 확산 모델을 통해 특정 속성과 진단 레이블에 맞춘 합성 이미지를 생성합니다. 예를 들어, 특정 연령대 또는 병원의 데이터를 균형 있게 생성하여 훈련 데이터를 보강합니다.
   - 이 과정에서 속성 분포는 균일하게 유지하고, 원래 질병의 유병률 분포는 유지됩니다.

3. **진단 모델 훈련**:
   - 생성된 합성 데이터와 실제 데이터를 결합하여 진단 모델을 훈련합니다. 레이블된 실제 데이터와 합성 데이터의 비율(예: 50:50)을 하이퍼파라미터로 설정하여 최적의 성능을 추구합니다.
   - 사용된 진단 모델은 UNet 아키텍처와 **ResNet**을 기반으로 하며, 다양한 조건(이진, 다중 클래스, 다중 라벨)에 따라 조정됩니다.

각 의료 이미지 분야(병리학, 흉부 엑스레이, 피부과)에 대해 적합한 데이터 처리, 해상도, 속성 조건화를 고려한 **맞춤형 실험 환경**이 설계되었습니다. 

---



This study proposes a methodology leveraging **diffusion models** to enhance the fairness and robustness of medical machine learning systems. The approach consists of **three main steps**:

1. **Training the Generative Model**: 
   - The diffusion model is trained using both labeled and unlabeled data. Labeled data comes from specific hospitals, while unlabeled data includes samples from various hospitals or environments.
   - The model is conditioned on diagnostic labels or attributes such as hospital ID to guide the generation process.

2. **Sampling Synthetic Data**:
   - Synthetic images are generated using the trained diffusion model based on specific attributes and diagnostic labels. For instance, the method ensures balanced representation across attributes like age or hospital while preserving the original disease prevalence distribution.
   - Uniform sampling from attribute distributions helps enrich the training dataset in a targeted and fair manner.

3. **Training the Diagnostic Model**:
   - The diagnostic model is trained on a combination of real and synthetic data, with the ratio (e.g., 50:50) treated as a hyperparameter to optimize performance.
   - The diagnostic models leverage architectures such as **UNet** and **ResNet**, tailored to specific tasks, including binary, multiclass, and multilabel classification.

The methodology is tailored for each medical imaging domain (histopathology, chest X-rays, dermatology), considering appropriate data preprocessing, resolution, and attribute conditioning for robust and fair learning.



   
 
<br/>
# Results  




이 연구는 병리학(histopathology), 흉부 엑스레이(chest X-ray), 피부과(dermatology) 데이터를 사용하여 제안된 방법론의 성능을 검증했습니다. 각 도메인에서 **In-distribution(훈련 데이터와 유사한 데이터)**와 **Out-of-Distribution(OOD, 훈련 데이터와 다른 환경에서 수집된 데이터)** 테스트 세트를 사용해 평가를 진행했습니다. 주요 메트릭으로는 진단 정확도(예: top-1 정확도, ROC-AUC), 공정성(성별, 병원 ID 등 민감한 속성 간 성능 차이), 그리고 고위험 민감도(high-risk sensitivity)가 사용되었습니다.

#### 병리학:
- **테스트 데이터**: 5개 병원의 CAMELYON17 데이터셋, 이 중 2개 병원 데이터는 OOD로 사용.
- **메트릭**: Top-1 정확도, 병원 간 공정성 차이.
- **결과**: 제안된 모델은 기존 베이스라인 모델에 비해 정확도가 최대 **48.5% 개선**되었으며, 공정성 격차를 **20%포인트** 감소시켰습니다.

#### 흉부 엑스레이:
- **테스트 데이터**: CheXpert (In-distribution)와 ChestX-ray14 (OOD) 데이터셋.
- **메트릭**: ROC-AUC, 성별 및 인종 간 공정성 격차.
- **결과**: 제안된 모델은 OOD 환경에서 평균 ROC-AUC를 **5.2% 개선**하였으며, 성별 공정성 격차를 **44.6% 감소**시켰습니다.

#### 피부과:
- **테스트 데이터**: 미국, 호주, 콜롬비아에서 수집된 3개의 데이터셋.
- **메트릭**: 고위험 민감도(예: 피부암 진단 정확도), 성별 간 공정성 격차.
- **결과**: 고위험 조건에서 OOD 데이터의 민감도가 **63.5% 개선**되었으며, 공정성 격차를 **7.5배 줄이는** 데 성공했습니다.

**비교 모델**: 이 연구는 기존 베이스라인 모델(실제 데이터만 사용)과 **RandAugment**, **Oversampling**, **Focal Loss** 등 데이터 증강 방법을 사용한 모델과 비교했습니다. 제안된 확산 모델 기반 접근법은 모든 비교 모델보다 높은 성능과 공정성을 보였습니다.

---



This study evaluated the proposed methodology across **histopathology**, **chest X-ray**, and **dermatology** datasets. Testing was conducted on both **in-distribution (ID)** datasets (similar to training data) and **out-of-distribution (OOD)** datasets (collected from different environments). Key metrics included diagnostic accuracy (e.g., top-1 accuracy, ROC-AUC), fairness (performance gap across sensitive attributes such as gender and hospital ID), and high-risk sensitivity.

#### Histopathology:
- **Test Data**: CAMELYON17 dataset from five hospitals, with two hospitals designated as OOD.
- **Metrics**: Top-1 accuracy, fairness gap across hospitals.
- **Results**: The proposed model improved accuracy by up to **48.5%** compared to the baseline and reduced the fairness gap by **20 percentage points**.

#### Chest X-ray:
- **Test Data**: CheXpert (in-distribution) and ChestX-ray14 (OOD) datasets.
- **Metrics**: ROC-AUC, fairness gaps for gender and race.
- **Results**: The proposed approach achieved a **5.2% improvement** in average ROC-AUC on OOD data and reduced the gender fairness gap by **44.6%**.

#### Dermatology:
- **Test Data**: Three datasets from the United States, Australia, and Colombia.
- **Metrics**: High-risk sensitivity (e.g., accuracy in diagnosing skin cancer), fairness gap across gender.
- **Results**: High-risk sensitivity improved by **63.5%** on OOD data, and the fairness gap was reduced by a factor of **7.5**.

**Comparative Models**: The study compared the proposed method with baseline models (using real data only) and models employing **RandAugment**, **Oversampling**, and **Focal Loss**. The diffusion model-based approach outperformed all these methods in terms of both performance and fairness.



<br/>
# 예제  




#### 병리학 데이터 예제:
**테스트 데이터**: CAMELYON17 데이터셋에서 두 병원의 이미지는 OOD(Out-of-Distribution)로 사용되었습니다. 예를 들어, 병원 A와 병원 B의 데이터는 훈련에 사용되었고, 병원 C의 데이터는 테스트 데이터로 사용되었습니다. 병원 C는 A와 B와는 다른 염색(staining) 프로토콜을 사용했기 때문에, 기존 모델은 병원 C의 데이터를 정확히 분류하는 데 어려움을 겪었습니다.

- **비교 모델의 성능**: 기존 베이스라인 모델은 병원 C에서 70% 정확도를 기록하며, 특히 특정 염색 프로토콜에서 생성된 이미지에 대해 진단 실패율이 높았습니다.
- **제안된 모델의 성능**: 확산 모델 기반 접근법은 병원 C에서 정확도를 85%로 높이며, 염색 차이에 따른 성능 감소를 크게 줄였습니다.

#### 흉부 엑스레이 데이터 예제:
**테스트 데이터**: CheXpert 데이터(훈련 데이터)와 ChestX-ray14 데이터(OOD)를 사용했습니다. 예를 들어, CheXpert에서 훈련된 모델은 ChestX-ray14에서 "심비대증(cardiomegaly)" 진단에 어려움을 겪었습니다. 

- **비교 모델의 성능**: 기존 모델은 ChestX-ray14의 심비대증 이미지에서 AUC가 70%에 불과했습니다.
- **제안된 모델의 성능**: 제안된 확산 모델 기반 접근법은 합성 데이터를 사용하여 ChestX-ray14에서 AUC를 85%로 개선했습니다.

#### 피부과 데이터 예제:
**테스트 데이터**: 미국의 데이터(훈련 데이터)와 호주의 데이터(OOD)를 사용했습니다. 예를 들어, 피부암 진단에서 호주의 데이터는 다른 인구 통계와 피부 톤 분포를 가졌습니다.

- **비교 모델의 성능**: 기존 모델은 고위험 피부암(예: 흑색종) 진단에서 민감도가 60%에 불과했습니다.
- **제안된 모델의 성능**: 제안된 모델은 합성 데이터를 활용하여 고위험 피부암 민감도를 80%로 개선했습니다.

---



#### Histopathology Data Example:
**Test Data**: The CAMELYON17 dataset included data from two hospitals used for training (e.g., Hospital A and B) and one hospital (e.g., Hospital C) for testing. Hospital C used a different staining protocol, which made it challenging for existing models to classify its images accurately.

- **Baseline Model Performance**: The baseline model achieved only 70% accuracy on Hospital C's data and struggled particularly with images affected by the staining protocol differences.
- **Proposed Model Performance**: The diffusion model-based approach improved accuracy to 85% on Hospital C's data, significantly reducing the impact of staining differences.

#### Chest X-ray Data Example:
**Test Data**: The CheXpert dataset was used for training (in-distribution), and the ChestX-ray14 dataset was used as OOD. For example, the task involved diagnosing "cardiomegaly" (enlarged heart) in ChestX-ray14 data.

- **Baseline Model Performance**: The baseline model achieved an AUC of only 70% for cardiomegaly in ChestX-ray14 data.
- **Proposed Model Performance**: The proposed diffusion model approach improved AUC to 85% by leveraging synthetic data for training.

#### Dermatology Data Example:
**Test Data**: Data from the United States (training data) and Australia (OOD) were used. For example, diagnosing high-risk skin cancers (e.g., melanoma) in Australia's dataset was challenging due to different demographic and skin tone distributions.

- **Baseline Model Performance**: The baseline model achieved a sensitivity of only 60% for high-risk conditions like melanoma.
- **Proposed Model Performance**: The proposed model improved sensitivity to 80% by incorporating synthetic data that accounted for the demographic shifts.



<br/>  
# 요약   



이 연구에서는 확산 모델을 활용하여 의료 데이터에서 발생하는 분포 차이 문제를 해결하고, 공정성과 강건성을 개선하는 방법을 제안했습니다. 예를 들어, 병리학 데이터에서 기존 모델은 새로운 병원에서 수집된 데이터(OOD)에서 70%의 낮은 정확도를 보였으나, 제안된 모델은 합성 데이터를 사용하여 85%로 성능을 향상시켰습니다. 또한, 흉부 엑스레이 데이터에서는 기존 모델이 특정 질환(예: 심비대증)을 정확히 진단하지 못해 AUC가 70%에 불과했지만, 제안된 모델은 85%로 개선했습니다. 피부과 데이터에서도 제안된 모델은 다양한 피부 톤과 인구 통계 차이를 효과적으로 다루며, 고위험 피부암 민감도를 기존 60%에서 80%로 증가시켰습니다. 이러한 결과는 제안된 확산 모델 기반 접근법이 데이터 부족과 공정성 문제를 동시에 해결할 수 있음을 보여줍니다.

---




This study proposed a diffusion model-based approach to address distribution shifts in medical data and improve fairness and robustness. For instance, in histopathology data, the baseline model achieved only 70% accuracy on OOD data from a new hospital, while the proposed model improved performance to 85% using synthetic data. Similarly, in chest X-ray data, the baseline model struggled to diagnose specific conditions like cardiomegaly, resulting in an AUC of 70%, but the proposed model raised this to 85%. In dermatology data, the proposed model effectively handled variations in skin tone and demographics, increasing sensitivity for high-risk skin cancers from 60% to 80%. These results demonstrate that the proposed diffusion model approach can simultaneously address data scarcity and fairness issues.


<br/>  
# 기타  


### 한글 설명 (테이블 및 피규어 설명)

1. **Fig. 1**: 생성된 샘플 및 방법론 개요를 나타냅니다. (a) 각 이미징 도메인(병리학, 흉부 방사선, 피부과)을 위한 조건부 확산 모델로 생성된 샘플이 포함되어 있습니다. (b) 방법론 개요에서는 레이블된 데이터와 레이블되지 않은 데이터를 활용하여 확산 모델을 훈련한 후, 샘플링된 합성 이미지를 진단 모델 훈련에 사용한 과정을 설명합니다 

2**: 병리학 데이터셋의 결과를 보여줍니다. (a) 병원 간 공정성 격차와 전체 예측 정확도 사이의 관계를 나타냅니다. (b) OOD(Out-of-Distribution) 데이터셋에서의 정확도를 비교하며, 확산 모델을 활용한 방법이 모든 설정에서 가장 좋은 성능을 보였습니다 .

3. **Fig 데이터셋의 결과를 보여줍니다. (a) 성별 공정성(AUC 기준) 격차와 평균 AUC를 비교합니다. (b) 인종 공정성 격차와 AUC 성능의 관계를 나타냅니다. 확산 모델을 사용한 경우, 공정성과 정확도 모두에서 베이스라인보다 뛰어난 성능을 보였습니다 .

4. **Fig. 4**: 피부과 데양한 방법론이 고위험 피부암 진단 민감도와 공정성에 미치는 영향을 비교합니다. OOD 데이터셋에서 확산 모델 기반 접근법은 공정성을 7.5배 개선하며 민감도를 63.5% 향상시켰습니다 .

5. **Extended Data Fig. 5**: 피부과지를 나타냅니다. 확산 모델로 생성된 이미지는 진단 가능한 고위험 조건(예: 흑색종)에 대한 고유하고 현실적인 예제를 포함합니다. 전문가들은 이러한 이미지를 실제 데이터와 비교하여 높은 품질로 평가했습니다  .

---

1. **Fig. 1**: Tables and Fillustrates generated samples and methodology overview. (a) Displays samples generated by the conditional diffusion model for different imaging modalities (histopathology, chest radiology, dermatology). (b) Describes the methodology, including training a diffusion model with labeled and unlabeled data, sampling synthetic images, and using these for diagnostic model training .

2. **Fig. 2**: Shows results on the histopathology dataset. (a) Plionship between fairness gap across hospitals and overall prediction accuracy. (b) Compares accuracy on OOD datasets, demonstrating that the diffusion model-based approach outperformed all other methods in every setting .

3. **Fig. 3**: Presents results on chest radiology datasets. (a) Compares the ss gap (based on AUC) and average AUC performance. (b) Shows the relationship between racial fairness gaps and AUC performance. The diffusion model approach achieved superior fairness and accuracy compared to baselines .

4. **Fig. 4**: Displays results on dermatology datasets. It compares the impact of differe high-risk skin cancer sensitivity and fairness. The diffusion model approach improved fairness by 7.5× and sensitivity by 63.5% on OOD datasets .

5. **Extended Data Fig. 5**: Illustrates synthetic images generated for dermatology. These images reprsk conditions (e.g., melanoma) and were deemed high-quality by expert clinicians when compared to real data  .


<br/>
# refer format:     



@article{ktena2024generative,
  title = {Generative models improve fairness of medical classifiers under distribution shifts},
  author = {Ktena, Ira and Wiles, Olivia and Albuquerque, Isabela and Rebuffi, Sylvestre-Alvise and Tanno, Ryutaro and Guha Roy, Abhijit and Azizi, Shekoofeh and Belgrave, Danielle and Kohli, Pushmeet and Cemgil, Taylan and Karthikesalingam, Alan and Gowal, Sven},
  journal = {Nature Medicine},
  volume = {30},
  pages = {1166--1173},
  year = {2024},
  month = {April},
  doi = {10.1038/s41591-024-02838-6}
}





Ira Ktena, Olivia Wiles, Isabela Albuquerque, Sylvestre-Alvise Rebuffi, Ryutaro Tanno, Abhijit Guha Roy, Shekoofeh Azizi, Danielle Belgrave, Pushmeet Kohli, Taylan Cemgil, Alan Karthikesalingam, and Sven Gowal. “Generative Models Improve Fairness of Medical Classifiers under Distribution Shifts.” Nature Medicine 30 (April 2024): 1166–1173. https://doi.org/10.1038/s41591-024-02838-6.









