---
layout: post
title:  "[2025]MoE-Health: A Mixture of Experts Framework for Robust Multimodal Healthcare Prediction"
date:   2025-10-17 21:09:06 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: MoE-Health는 다양한 데이터 모달리티를 처리하기 위해 설계된 Mixture of Experts 프레임워크로, 각 모달리티에 특화된 전문가 네트워크와 동적 게이팅 메커니즘을 활용하여 결측 데이터에 강인한 예측을 수행한다.


짧은 요약(Abstract) :



이 논문에서는 MoE-Health라는 새로운 혼합 전문가(Mixture of Experts, MoE) 프레임워크를 제안합니다. 이 프레임워크는 다양한 형태의 의료 데이터를 효과적으로 활용하여 임상 예측을 수행하는 데 중점을 두고 있습니다. 현대의 의료 시스템은 전자 건강 기록(EHR), 임상 노트, 의료 이미지 등 다양한 다중 모달 데이터를 생성하지만, 실제 샘플은 종종 다양한 형태의 데이터가 결여되어 있거나 불완전합니다. 기존의 접근 방식은 일반적으로 모든 모달 데이터가 완전하게 필요하거나 수동 선택 전략에 의존하여, 데이터 가용성이 환자와 기관에 따라 다르게 나타나는 실제 임상 환경에서의 적용 가능성을 제한합니다. MoE-Health는 이러한 한계를 극복하기 위해 설계되었으며, 다양한 모달리티를 가진 샘플을 처리하고 중요한 임상 작업의 성능을 향상시키는 데 중점을 두고 있습니다. 이 프레임워크는 전문화된 전문가 네트워크와 동적 게이팅 메커니즘을 활용하여, 사용 가능한 데이터 모달리티에 따라 관련 전문가를 동적으로 선택하고 결합함으로써 데이터 가용성 시나리오에 유연하게 적응할 수 있습니다. MoE-Health는 MIMIC-IV 데이터셋을 사용하여 세 가지 중요한 임상 예측 작업(병원 내 사망 예측, 긴 입원 기간 예측, 병원 재입원 예측)에 대해 평가되었으며, 실험 결과는 MoE-Health가 기존의 다중 모달 융합 방법에 비해 우수한 성능을 달성하고 다양한 모달리티 가용성 패턴에서도 강건성을 유지함을 보여줍니다.



This paper proposes a novel Mixture of Experts (MoE) framework called MoE-Health, designed to effectively leverage diverse multimodal healthcare data for clinical prediction. Modern healthcare systems generate various types of multimodal data, including Electronic Health Records (EHR), clinical notes, and medical images. However, real-world samples often present with varied or incomplete modalities. Existing approaches typically require complete modality data or rely on manual selection strategies, limiting their applicability in real-world clinical settings where data availability varies across patients and institutions. To address these limitations, MoE-Health is specifically designed to handle samples with differing modalities and improve performance on critical clinical tasks. The framework leverages specialized expert networks and a dynamic gating mechanism to dynamically select and combine relevant experts based on available data modalities, enabling flexible adaptation to varying data availability scenarios. MoE-Health is evaluated on the MIMIC-IV dataset across three critical clinical prediction tasks: in-hospital mortality prediction, long length of stay, and hospital readmission prediction. Experimental results demonstrate that MoE-Health achieves superior performance compared to existing multimodal fusion methods while maintaining robustness across different modality availability patterns.


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



**MoE-Health 아키텍처**

MoE-Health는 다양한 데이터 모달리티를 효과적으로 통합하여 의료 예측을 수행하기 위해 설계된 Mixture of Experts (MoE) 프레임워크입니다. 이 아키텍처는 다음과 같은 주요 구성 요소로 이루어져 있습니다:

1. **모달리티 특화 인코딩**: 각 데이터 모달리티(예: 전자 건강 기록(EHR), 임상 노트, 의료 이미지)는 전용 인코더를 통해 고정 차원의 임베딩으로 변환됩니다. EHR 인코더는 정적 인구 통계 데이터와 동적 시계열 정보를 처리하며, 임상 노트 인코더는 비구조화된 텍스트 데이터를 Clinical-BERT 모델을 사용하여 인코딩합니다. 이미지 인코더는 DenseNet-121 아키텍처를 사용하여 X-ray 이미지를 처리합니다.

2. **모달리티 결합 처리**: MoE-Health는 각 모달리티의 결합을 처리하기 위해 전문가 네트워크를 사용합니다. 각 전문가 네트워크는 특정 모달리티 조합에 대해 사전 훈련되어 있으며, 이를 통해 각 조합에 최적화된 예측을 수행할 수 있습니다. 이 과정에서 동적 게이팅 메커니즘이 사용되어 입력 데이터에 따라 적절한 전문가를 선택하고 그들의 출력을 가중합하여 최종 예측을 생성합니다.

3. **결측 모달리티 처리**: MoE-Health는 결측 모달리티 문제를 해결하기 위해 학습 가능한 결측성 임베딩을 도입합니다. 각 모달리티가 결측일 경우, 해당 모달리티의 인코더 출력은 결측성 임베딩으로 대체되어 모델이 결측 정보를 학습할 수 있도록 합니다.

4. **훈련 목표**: 모델은 이진 분류 문제로 설정된 세 가지 예측 작업(병원 내 사망, 병원 체류 기간, 재입원 예측)을 위해 훈련됩니다. 손실 함수는 이진 교차 엔트로피(BCE) 손실과 전문가 활용 균형 손실을 결합하여 최적화됩니다.

5. **실험 및 평가**: MoE-Health는 MIMIC-IV 데이터셋을 사용하여 세 가지 주요 임상 예측 작업에서 성능을 평가합니다. 실험 결과, MoE-Health는 기존의 다중 모달리티 융합 방법들에 비해 우수한 성능을 보이며, 결측 모달리티에 대한 강인성을 유지합니다.

이러한 구조적 설계는 MoE-Health가 다양한 데이터 가용성 시나리오에 유연하게 적응할 수 있도록 하며, 실제 임상 환경에서의 적용 가능성을 높입니다.

---




**MoE-Health Architecture**

MoE-Health is a Mixture of Experts (MoE) framework designed to effectively integrate diverse data modalities for healthcare prediction. This architecture consists of the following key components:

1. **Modality-Specific Encoding**: Each data modality (e.g., Electronic Health Records (EHR), clinical notes, medical images) is transformed into a fixed-dimensional embedding through dedicated encoders. The EHR encoder processes static demographic data and dynamic time-series information, while the clinical notes encoder uses the Clinical-BERT model to encode unstructured text data. The image encoder processes X-ray images using the DenseNet-121 architecture.

2. **Modality Combination Processing**: MoE-Health employs expert networks to handle the combinations of modalities. Each expert network is pre-trained on specific modality combinations, allowing it to perform optimized predictions for those combinations. A dynamic gating mechanism is used to select the appropriate experts based on the input data and to combine their outputs through a weighted sum to generate the final prediction.

3. **Handling Missing Modalities**: To address the issue of missing modalities, MoE-Health introduces learnable missingness embeddings. When a modality is absent, the output of its encoder is replaced with a missingness embedding, enabling the model to learn from the absence of specific modalities.

4. **Training Objective**: The model is trained on three binary classification tasks (in-hospital mortality, length of stay, and readmission prediction). The loss function combines binary cross-entropy (BCE) loss with an auxiliary load balancing loss to optimize expert utilization.

5. **Experiments and Evaluation**: MoE-Health is evaluated on the MIMIC-IV dataset across three critical clinical prediction tasks. Experimental results demonstrate that MoE-Health outperforms existing multimodal fusion methods while maintaining robustness against missing modalities.

This structural design allows MoE-Health to flexibly adapt to varying data availability scenarios, enhancing its applicability in real-world clinical settings.


<br/>
# Results



이 논문에서는 MoE-Health라는 새로운 혼합 전문가(Mixture of Experts, MoE) 프레임워크를 제안하고, 이를 통해 다양한 임상 예측 작업에서의 성능을 평가하였다. MoE-Health는 MIMIC-IV 데이터셋을 사용하여 세 가지 주요 임상 예측 작업인 입원 중 사망 예측, 긴 병원 체류 예측, 그리고 병원 재입원 예측을 수행하였다.

#### 1. 경쟁 모델
MoE-Health는 여러 경쟁 모델과 비교되었다. 이들 모델은 단일 모달리티 모델, 전통적인 융합 방법, 그리고 최신 멀티모달 아키텍처를 포함한다. 단일 모달리티 모델은 EHR, CXR 이미지, 그리고 임상 노트 각각에 대해 훈련된 모델로, 각 데이터 유형의 성능 기준을 제공한다. 전통적인 융합 방법으로는 조기 융합, 늦은 융합, 그리고 공동 융합 방법이 포함되며, MPP라는 모델이 세 가지 융합 전략을 적용하여 성능을 평가하였다. 또한, TriMF라는 최신 멀티모달 아키텍처와 HAIM 방법도 비교에 포함되었다.

#### 2. 테스트 데이터
MoE-Health는 MIMIC-IV 데이터셋에서 31,088개의 고유한 병원 입원 사례를 사용하였다. 이 데이터셋은 EHR, CXR 이미지, 그리고 임상 노트의 세 가지 모달리티를 포함하고 있으며, 각 환자에 대해 다양한 모달리티의 가용성을 반영하고 있다. 데이터의 결측치는 실제 임상 환경에서의 데이터 가용성의 변동성을 나타낸다.

#### 3. 메트릭
모델의 성능 평가는 두 가지 주요 메트릭인 수신자 조작 특성 곡선 아래 면적(AUROC)과 F1 점수를 사용하여 수행되었다. AUROC는 모델의 분류 성능을 평가하는 데 유용하며, F1 점수는 정밀도와 재현율의 조화 평균을 나타내어 불균형 데이터셋에서의 성능을 평가하는 데 적합하다.

#### 4. 비교 결과
MoE-Health는 모든 예측 작업에서 경쟁 모델들보다 우수한 성능을 보였다. 특히, 입원 중 사망 예측에서 MoE-Health는 AUROC 0.818을 기록하여 두 번째로 좋은 성능을 보인 TriMF 모델보다 1.2% 포인트 높은 성과를 나타냈다. 긴 병원 체류 예측에서는 AUROC 0.794로 HAIM 모델보다 1.2% 포인트 높은 성과를 보였고, 병원 재입원 예측에서는 AUROC 0.643으로 TriMF 모델보다 0.5% 포인트 높은 성과를 기록하였다. F1 점수에서도 MoE-Health는 입원 중 사망 예측에서 0.465로 TriMF 모델보다 3.0% 포인트 높은 성과를 보였고, 긴 병원 체류 예측에서는 HAIM 모델과 동률을 기록하였다.

이러한 결과는 MoE-Health가 다양한 모달리티의 가용성에 적응하여 임상 예측 작업에서 우수한 성능을 발휘할 수 있음을 보여준다.

---

### English Version

In this paper, a novel Mixture of Experts (MoE) framework called MoE-Health is proposed, and its performance is evaluated across various clinical prediction tasks. MoE-Health was tested on the MIMIC-IV dataset, focusing on three key clinical prediction tasks: in-hospital mortality prediction, long length of stay prediction, and hospital readmission prediction.

#### 1. Competing Models
MoE-Health was compared against several competing models, including unimodal models, traditional fusion methods, and advanced multimodal architectures. Unimodal models were trained on each data type (EHR, CXR images, and clinical notes) to provide performance benchmarks. Traditional fusion methods included early fusion, late fusion, and joint fusion approaches, with the MPP model applying these three fusion strategies for performance evaluation. Additionally, the latest multimodal architecture, TriMF, and the HAIM method were included in the comparisons.

#### 2. Test Data
MoE-Health utilized a dataset of 31,088 unique hospital admissions from the MIMIC-IV dataset. This dataset includes three modalities: EHR, CXR images, and clinical notes, reflecting the variability in modality availability for each patient. The missingness in the data highlights the variability in data availability that is common in real clinical environments.

#### 3. Metrics
Model performance was evaluated using two primary metrics: Area Under the Receiver Operating Characteristic Curve (AUROC) and F1 Score. AUROC is useful for assessing the classification performance of the model, while the F1 Score represents the harmonic mean of precision and recall, making it suitable for evaluating performance on imbalanced datasets.

#### 4. Comparative Results
MoE-Health demonstrated superior performance across all prediction tasks compared to competing models. Specifically, in the in-hospital mortality prediction task, MoE-Health achieved an AUROC of 0.818, which is 1.2 percentage points higher than the second-best performing model, TriMF. For long length of stay prediction, it recorded an AUROC of 0.794, outperforming the HAIM model by 1.2 percentage points, and for hospital readmission prediction, it achieved an AUROC of 0.643, which is 0.5 percentage points higher than TriMF. In terms of F1 Score, MoE-Health achieved 0.465 in in-hospital mortality prediction, representing a substantial improvement of 3.0 percentage points over TriMF, and tied with HAIM in long length of stay prediction.

These results indicate that MoE-Health can adapt to varying modality availability and perform excellently in clinical prediction tasks.


<br/>
# 예제
### 한글 설명

MoE-Health 프레임워크는 다양한 모달리티(데이터 유형)를 활용하여 임상 예측을 수행하는 모델입니다. 이 모델은 MIMIC-IV 데이터셋을 사용하여 세 가지 주요 임상 예측 작업을 수행합니다: 입원 중 사망 예측, 긴 병원 체류 예측, 그리고 병원 재입원 예측입니다.

#### 1. 데이터셋 구성
MIMIC-IV 데이터셋은 약 300,000명의 환자에 대한 전자 건강 기록(EHR), 임상 노트, 그리고 의료 이미지를 포함하고 있습니다. 이 데이터셋에서 73,181개의 ICU 입원 사례를 선택하여 세 가지 모달리티(구조화된 EHR 데이터, 비구조화된 임상 노트, 그리고 흉부 X선 이미지)를 통합했습니다. 최종적으로 31,088개의 고유한 입원 사례가 생성되었으며, 이 중 37.4%는 모든 세 가지 모달리티를 포함하고 있습니다.

#### 2. 트레이닝 데이터와 테스트 데이터
- **트레이닝 데이터**: 모델은 80%의 데이터를 트레이닝에 사용합니다. 이 데이터는 각 환자의 EHR, 임상 노트, 그리고 X선 이미지로 구성되어 있으며, 각 환자의 상태에 대한 정보를 포함합니다. 예를 들어, 특정 환자가 입원한 후 48시간 이내에 사망했는지 여부(이진 분류)를 예측하는 데 사용됩니다.
- **테스트 데이터**: 나머지 20%의 데이터는 모델의 성능을 평가하는 데 사용됩니다. 이 데이터는 모델이 훈련 중에 보지 못한 새로운 환자 사례로 구성되어 있으며, 모델이 얼마나 잘 일반화되는지를 평가합니다.

#### 3. 구체적인 작업(Task)
- **입원 중 사망 예측**: 이 작업은 환자가 입원 후 48시간 이내에 사망할 확률을 예측하는 것입니다. 입력은 EHR, 임상 노트, X선 이미지의 조합이며, 출력은 사망 여부(1 또는 0)입니다.
- **긴 병원 체류 예측**: 이 작업은 환자의 병원 체류 기간이 7일을 초과할지 여부를 예측합니다. 입력은 입원 후 첫 48시간의 EHR 데이터이며, 출력은 긴 체류 여부(1 또는 0)입니다.
- **병원 재입원 예측**: 이 작업은 환자가 퇴원 후 30일 이내에 재입원할 확률을 예측합니다. 입력은 EHR, 임상 노트, X선 이미지의 조합이며, 출력은 재입원 여부(1 또는 0)입니다.

이러한 작업을 통해 MoE-Health는 다양한 모달리티를 효과적으로 통합하고, 결측 데이터에 강인한 예측 모델을 제공합니다.

---

### English Version

The MoE-Health framework is a model that utilizes various modalities (data types) to perform clinical predictions. This model uses the MIMIC-IV dataset to carry out three main clinical prediction tasks: in-hospital mortality prediction, long length of stay prediction, and hospital readmission prediction.

#### 1. Dataset Construction
The MIMIC-IV dataset includes electronic health records (EHR), clinical notes, and medical images for approximately 300,000 patients. From this dataset, 73,181 ICU admission cases were selected, integrating three modalities: structured EHR data, unstructured clinical notes, and chest X-ray images. Ultimately, 31,088 unique admission cases were generated, of which 37.4% contain all three modalities.

#### 2. Training Data and Test Data
- **Training Data**: The model uses 80% of the data for training. This data consists of EHR, clinical notes, and X-ray images for each patient, containing information about their condition. For example, it is used to predict whether a specific patient died within 48 hours of admission (binary classification).
- **Test Data**: The remaining 20% of the data is used to evaluate the model's performance. This data consists of new patient cases that the model has not seen during training, allowing for the assessment of how well the model generalizes.

#### 3. Specific Tasks
- **In-Hospital Mortality Prediction**: This task predicts the likelihood of a patient dying within 48 hours of admission. The input is a combination of EHR, clinical notes, and X-ray images, and the output is whether the patient died (1 or 0).
- **Long Length of Stay Prediction**: This task predicts whether a patient's hospital stay will exceed 7 days. The input is EHR data from the first 48 hours of admission, and the output is whether the stay is long (1 or 0).
- **Hospital Readmission Prediction**: This task predicts the likelihood of a patient being readmitted within 30 days of discharge. The input is a combination of EHR, clinical notes, and X-ray images, and the output is whether the patient was readmitted (1 or 0).

Through these tasks, MoE-Health effectively integrates diverse modalities and provides a robust predictive model that is resilient to missing data.

<br/>
# 요약

**한국어 요약:** MoE-Health는 다양한 데이터 모달리티를 처리하기 위해 설계된 Mixture of Experts 프레임워크로, 각 모달리티에 특화된 전문가 네트워크와 동적 게이팅 메커니즘을 활용하여 결측 데이터에 강인한 예측을 수행한다. MIMIC-IV 데이터셋을 사용한 실험 결과, MoE-Health는 병원 내 사망 예측, 입원 기간 예측, 재입원 예측에서 기존 방법들보다 우수한 성능을 보였다. 예를 들어, 병원 내 사망 예측에서 MoE-Health는 AUROC 0.818을 기록하며, 이는 두 번째로 좋은 방법인 TriMF보다 1.2% 포인트 높은 수치이다.

**English Summary:** MoE-Health is a Mixture of Experts framework designed to handle diverse data modalities, utilizing specialized expert networks and a dynamic gating mechanism to perform robust predictions in the presence of missing data. Experimental results on the MIMIC-IV dataset demonstrate that MoE-Health outperforms existing methods in critical tasks such as in-hospital mortality prediction, length of stay prediction, and hospital readmission prediction. For instance, in in-hospital mortality prediction, MoE-Health achieved an AUROC of 0.818, which is 1.2 percentage points higher than the second-best method, TriMF.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 이 그림은 MoE-Health 아키텍처의 전반적인 구조를 보여줍니다. 각 샘플은 EHR, 임상 노트, 그리고 이미지를 포함한 세 가지 주요 모달리티의 정보를 통합합니다. 이 구조는 다양한 모달리티의 조합을 처리할 수 있는 능력을 강조합니다.
   - **Figure 2**: MoE-Health 아키텍처의 세 가지 주요 단계(모달리티별 인코딩, MoE 기반 다중 모달 융합, 예측)를 설명합니다. 이 그림은 각 모달리티가 어떻게 인코딩되고, 결합되어 최종 예측으로 이어지는지를 시각적으로 나타냅니다.

2. **테이블**
   - **Table 1**: MoE-Health와 여러 기준 모델 간의 성능 비교를 보여줍니다. MoE-Health는 모든 예측 작업에서 AUROC 점수와 F1 점수에서 우수한 성능을 보였습니다. 특히, in-hospital mortality 예측에서 0.818의 AUROC 점수를 기록하며, 이는 다른 방법들보다 1.2% 포인트 높은 수치입니다.
   - **Table 2**: 다양한 모달리티 조합에 대한 MoE-Health의 성능을 보여줍니다. EHR 단일 모달리티가 가장 높은 AUROC 점수를 기록했으며, 모든 모달리티(EHR, Clinical Notes, Image)를 결합했을 때 가장 높은 성능을 보였습니다. 이는 모달리티 융합의 중요성을 강조합니다.
   - **Table 3**: MoE-Health의 주요 구성 요소에 대한 ablation 연구 결과를 보여줍니다. Expert Specialization이 가장 중요한 요소로 나타났으며, 이를 제거했을 때 AUROC 점수가 0.083 포인트 감소했습니다. 이는 특정 모달리티 조합에 대한 전문가의 전문성이 모델 성능에 미치는 영향을 강조합니다.

3. **어펜딕스**
   - 어펜딕스는 추가적인 실험 결과나 데이터 세트에 대한 세부 정보를 제공할 수 있습니다. 이 부분은 연구의 신뢰성을 높이고, 다른 연구자들이 동일한 방법론을 재현할 수 있도록 돕는 역할을 합니다.

---




1. **Diagrams and Figures**
   - **Figure 1**: This figure illustrates the overall structure of the MoE-Health architecture. Each sample integrates information from three major modalities: EHR, clinical notes, and images. This structure emphasizes the ability to handle various combinations of modalities.
   - **Figure 2**: It explains the three main stages of the MoE-Health architecture (Modality-Specific Encoding, MoE-based Multimodal Fusion, and Prediction). This figure visually represents how each modality is encoded, combined, and leads to the final prediction.

2. **Tables**
   - **Table 1**: It shows the performance comparison between MoE-Health and several baseline models. MoE-Health demonstrated superior performance across all prediction tasks, achieving an AUROC score of 0.818 for in-hospital mortality prediction, which is 1.2 percentage points higher than other methods.
   - **Table 2**: It presents the performance of MoE-Health using different combinations of modalities. The single modality of EHR achieved the highest AUROC score, and combining all modalities (EHR, Clinical Notes, Image) yielded the best performance. This highlights the importance of modality fusion.
   - **Table 3**: It shows the results of the ablation study on the key components of MoE-Health. Expert Specialization emerged as the most critical factor, with a 0.083 point drop in AUROC when removed. This underscores the impact of expert specialization on model performance.

3. **Appendix**
   - The appendix may provide additional experimental results or detailed information about the dataset. This section enhances the credibility of the research and helps other researchers replicate the methodology.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@inproceedings{WangYang2018,
  author = {Xiaoyang Wang and Christopher C. Yang},
  title = {MoE-Health: A Mixture of Experts Framework for Robust Multimodal Healthcare Prediction},
  booktitle = {Proceedings of the 16th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB '25)},
  year = {2025},
  month = {October},
  location = {Philadelphia, PA, USA},
  publisher = {ACM},
  pages = {1--12},
}
```

### 시카고 스타일 인용
Wang, Xiaoyang, and Christopher C. Yang. 2018. "MoE-Health: A Mixture of Experts Framework for Robust Multimodal Healthcare Prediction." In *Proceedings of the 16th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB '25)*, 1-12. Philadelphia, PA, USA: ACM. 
