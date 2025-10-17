---
layout: post
title:  "[2025]Rare Disease Phenotyping via Self-Supervised Multimodal Learning"
date:   2025-10-17 16:39:34 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 희귀 질환의 표현형을 파악하기 위해 유전자, 3D 이미징, 웨어러블 센서 데이터를 통합한 멀티모달 자기 지도 학습 모델을 제안합니다.


짧은 요약(Abstract) :


이 논문에서는 희귀 유전 질환 환자들이 겪는 진단 과정의 복잡성을 해결하기 위해, 전체 엑솜 시퀀싱(Whole Exome Sequencing, WES), 3D MRI 영상, 그리고 웨어러블 센서 데이터를 통합하여 희귀 질환의 표현형을 파악하는 다중 모달 자기 지도 학습 모델을 제안합니다. 기존의 개별적인 머신러닝 파이프라인과는 달리, 이 접근법은 세 가지 데이터 모달리티를 모두 처리할 수 있는 단일 변환기 기반의 인코더-디코더 아키텍처를 사용합니다. 우리는 환자별 다중 모달 특징을 정렬하기 위해 교차 모달 대조 손실을 최적화하고, 생물 의학 지식 그래프에 기반한 온톨로지 손실을 통해 표현을 강화합니다. 제로샷 검색 및 분류 작업에서, 우리의 모델은 최신 전문화된 기준선보다 우수한 성능을 보이며, 이전 방법보다 +5.6% AUROC 향상을 달성했습니다. 주요 기여로는 (i) 유전적 변이, 3D 이미지, 시간 시계열 신호를 처리하는 통합 다중 모달 변환기, (ii) 환자 데이터를 인간 표현형 온톨로지(HPO) 용어와 함께 임베딩하는 온톨로지 정렬 잠재 공간, (iii) 제로샷 희귀 질환 검색에서 향상된 재현율과 임상적으로 의미 있는 주의 기반 설명을 보여주는 광범위한 실험이 포함됩니다.



This paper proposes a multimodal self-supervised learning model to phenotype rare diseases by integrating whole exome sequencing (WES), volumetric MRI, and wearable sensor data, addressing the complexity of the diagnostic process faced by patients with rare genetic disorders. Unlike traditional siloed machine learning pipelines, this approach employs a single transformer-based encoder-decoder architecture capable of handling all three data modalities. We optimize a cross-modal contrastive loss to align patient-specific multimodal features, alongside an ontology-based loss that grounds representations in biomedical knowledge graphs. Across zero-shot retrieval and classification tasks, our model outperforms state-of-the-art specialized baselines, achieving a +5.6% AUROC gain over the best prior method. Key contributions include: (i) a unified multimodal transformer that processes genomic variants, 3D images, and time-series signals, (ii) an ontology-aligned latent space that embeds patient data alongside Human Phenotype Ontology (HPO) terms, and (iii) extensive experiments demonstrating improved recall in zero-shot rare disease retrieval and clinically meaningful attention-based explanations.


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



이 논문에서는 희귀 질병의 표현을 학습하기 위해 통합된 다중 모달 자기 지도 학습 접근 방식을 제안합니다. 이 모델은 유전자 변이, 3D 이미징, 웨어러블 센서 데이터를 포함한 세 가지 서로 다른 데이터 모달리티를 처리할 수 있는 통합된 인코더-디코더 아키텍처를 사용합니다. 

#### 모델 아키텍처
모델은 세 가지 모달리티(유전체, 이미징, 생체 신호)에 대해 각각의 인코더를 가지고 있으며, 이 인코더들은 공통의 잠재 공간으로 매핑됩니다. 각 인코더는 다음과 같은 방식으로 구성됩니다:

1. **유전체 인코더 (𝐸𝑔)**: 이 인코더는 전체 엑솜 시퀀싱(Whole Exome Sequencing, WES)에서 생성된 변이 호출 데이터를 입력으로 받습니다. BERT 스타일의 트랜스포머를 사용하여 변이 토큰을 인코딩하며, 하플로타입 마스킹 기법을 도입하여 연관된 변이 정보를 보존합니다.

2. **이미징 인코더 (𝐸𝑣)**: 이 인코더는 T1 가중치 뇌 MRI와 같은 3D 이미지를 처리합니다. 이미지는 패치로 나누어지고, 각 패치는 3D 비전 트랜스포머에 입력됩니다. 마스킹된 볼륨 모델링 기법을 사용하여 일부 패치를 마스킹하고 이를 복원하는 방식으로 학습합니다.

3. **생체 신호 인코더 (𝐸𝑠)**: 이 인코더는 다변량 생체 신호(예: ECG, PPG)를 처리합니다. 신호는 1D CNN을 통해 지역적 패턴을 추출한 후, 트랜스포머를 통해 장기 의존성을 모델링합니다.

#### 학습 기법
모델은 두 가지 주요 손실 함수를 사용하여 학습됩니다:

1. **교차 모달 대조 손실 (LMMCL)**: 이 손실 함수는 서로 다른 모달리티에서 동일한 환자의 임베딩이 유사하도록 유도합니다. 이를 통해 모델은 다양한 모달리티 간의 공통된 질병 관련 신호를 학습합니다.

2. **온톨로지 정렬 손실 (LKG)**: 이 손실 함수는 환자의 잠재 표현이 임상적 표현과 일치하도록 유도합니다. Human Phenotype Ontology (HPO)와 Unified Medical Language System (UMLS)을 활용하여 환자 임베딩을 전문가가 정의한 표현과 정렬합니다.

#### 데이터 처리
모델은 다양한 데이터셋을 사용하여 학습됩니다. 유전체 데이터는 TCGA-GBM 데이터셋을 사용하고, 이미징 데이터는 ADNI 데이터셋을, 생체 신호 데이터는 UK Biobank의 웨어러블 데이터셋을 사용합니다. 각 데이터셋은 사전 처리 과정을 거쳐 모델에 입력됩니다.

이러한 통합된 접근 방식은 희귀 질병의 진단을 위한 AI 기반의 차별 진단을 가능하게 하며, 환자에게 보다 신속하고 정확한 치료를 제공할 수 있는 잠재력을 가지고 있습니다.

---




This paper proposes a unified multimodal self-supervised learning approach for rare disease phenotyping. The model utilizes an integrated encoder-decoder architecture capable of processing three different data modalities: genomic variants, 3D imaging, and wearable sensor data.

#### Model Architecture
The model consists of three modality-specific encoders, each corresponding to genomics, imaging, and biosignals, which map to a common latent space. Each encoder is structured as follows:

1. **Genomic Encoder (𝐸𝑔)**: This encoder takes variant call data from Whole Exome Sequencing (WES) as input. It employs a BERT-style transformer to encode variant tokens and introduces haplotype masking techniques to preserve information about linked variants.

2. **Imaging Encoder (𝐸𝑣)**: This encoder processes 3D images such as T1-weighted brain MRIs. The images are divided into patches, and each patch is fed into a 3D vision transformer. A masked volume modeling technique is used, where some patches are masked, and the model is tasked with reconstructing them.

3. **Biosignal Encoder (𝐸𝑠)**: This encoder handles multivariate biosignals (e.g., ECG, PPG). The signals are first processed through a 1D CNN to extract local patterns, followed by a transformer to model long-range dependencies.

#### Training Techniques
The model is trained using two main loss functions:

1. **Cross-Modal Contrastive Loss (LMMCL)**: This loss function encourages the embeddings of the same patient from different modalities to be similar. This allows the model to learn common disease-relevant signals across various modalities.

2. **Ontology Alignment Loss (LKG)**: This loss function encourages the model's latent representation to align with clinical phenotypes. It leverages the Human Phenotype Ontology (HPO) and the Unified Medical Language System (UMLS) to align patient embeddings with expert-defined phenotypic profiles.

#### Data Processing
The model is trained on various datasets. Genomic data is sourced from the TCGA-GBM dataset, imaging data from the ADNI dataset, and biosignal data from the UK Biobank wearable dataset. Each dataset undergoes preprocessing before being input into the model.

This integrated approach enables AI-driven differential diagnosis for rare diseases, potentially providing patients with timely and accurate treatments.


<br/>
# Results



이 연구에서는 희귀 질환의 표현 학습을 위한 통합 다중 모달 자기 감독 학습 접근 방식을 제안하였으며, 여러 테스트 데이터셋에서 경쟁 모델과 비교하여 성능을 평가하였다. 주요 결과는 다음과 같다:

1. **테스트 데이터셋**:
   - **유전체 데이터**: TCGA-GBM 데이터셋을 사용하여 250명의 신경교종 환자와 50명의 정상 대조군을 포함하였다.
   - **신경영상 데이터**: ADNI 데이터셋에서 알츠하이머 환자 200명과 연령이 일치하는 대조군 200명의 T1 가중치 뇌 MRI를 사용하였다.
   - **생체 신호 데이터**: UK Biobank에서 100명의 피험자로부터 500시간의 PPG 데이터를 수집하였다.

2. **경쟁 모델**:
   - **ElasticNet (조기 융합)**: 유전체, 영상 및 신호 데이터를 수작업으로 결합하여 로지스틱 회귀를 수행하였다.
   - **단일 모달 CNN**: 각 모달리티에 대해 별도로 훈련된 모델로, 유전체 데이터에 대해 MLP, MRI에 대해 3D ResNet, 신호에 대해 1D CNN을 사용하였다.
   - **BioGPT-X**: 생물 의학 텍스트에 대해 사전 훈련된 강력한 변환기 모델로, 유전체 데이터에 대해 미세 조정하였다.
   - **MedCLIP**: 의료 이미지를 텍스트와 대조하여 학습하는 비전-언어 모델로, 두 모달리티를 결합하여 성능을 평가하였다.

3. **메트릭**:
   - **AUROC (Receiver Operating Characteristic Curve의 면적)**: 각 모델의 분류 성능을 평가하는 데 사용되었다.
   - **F1 점수**: 모델의 정밀도와 재현율을 종합적으로 평가하였다.
   - **Recall@K**: 주어진 K 값에 대해 올바른 사례를 검색하는 성능을 평가하였다.
   - **Mean Reciprocal Rank (MRR)**: 검색된 결과의 순위를 평가하는 데 사용되었다.

4. **비교 결과**:
   - **유전체 데이터 (GBM vs. 정상)**: 제안된 모델은 AUROC 0.93±0.02를 기록하여 BioGPT-X의 0.88±0.03을 초과하였다. 이는 5%의 성능 향상을 나타낸다.
   - **신경영상 데이터 (AD vs. 정상)**: 제안된 모델은 AUROC 0.89±0.01을 기록하여 ResNet-18의 0.81±0.02를 초과하였다.
   - **생체 신호 데이터 (AF vs. 정상)**: 제안된 모델은 AUROC 0.94±0.01을 기록하여 단일 모달 CNN의 0.87±0.02를 초과하였다.
   - **희귀 질환 검색**: 제안된 모델은 30개의 시뮬레이션된 환자 데이터에서 93.3%의 정확도로 올바른 사례를 검색하였다. 이는 기존의 경쟁 모델들이 수행할 수 없는 다중 모달 검색을 가능하게 하였다.

이러한 결과는 제안된 모델이 다중 모달 데이터를 통합하여 희귀 질환의 진단을 지원하는 데 있어 효과적임을 보여준다. 특히, 제안된 모델은 기존의 단일 모달 모델들보다 우수한 성능을 보였으며, 이는 다중 모달 학습의 이점을 잘 나타낸다.

---




This study proposed a unified multimodal self-supervised learning approach for rare disease representation learning and evaluated its performance against competitive models across various test datasets. The key results are as follows:

1. **Test Datasets**:
   - **Genomic Data**: The TCGA-GBM dataset was used, which includes 250 glioblastoma patients and 50 normal controls.
   - **Neuroimaging Data**: The ADNI dataset was utilized, comprising T1-weighted brain MRIs from 200 Alzheimer’s patients and 200 age-matched controls.
   - **Biosignal Data**: A subset of the UK Biobank was used, collecting 500 hours of PPG data from 100 subjects.

2. **Competitive Models**:
   - **ElasticNet (Early Fusion)**: A logistic regression model that combined handcrafted features from genomic, imaging, and signal data.
   - **Single-Modality CNNs**: Models trained separately for each modality, including MLP for genomic data, 3D ResNet for MRI, and 1D CNN for signals.
   - **BioGPT-X**: A strong transformer model pre-trained on biomedical text, fine-tuned on genomic data.
   - **MedCLIP**: A vision-language model that aligns medical images with text, adapted for the tri-modal problem.

3. **Metrics**:
   - **AUROC (Area Under the Receiver Operating Characteristic Curve)**: Used to evaluate the classification performance of each model.
   - **F1 Score**: A comprehensive measure of the model's precision and recall.
   - **Recall@K**: Evaluated the performance of retrieving correct cases for a given K value.
   - **Mean Reciprocal Rank (MRR)**: Used to assess the ranking of retrieved results.

4. **Comparison Results**:
   - **Genomic Data (GBM vs. Normal)**: The proposed model achieved an AUROC of 0.93±0.02, surpassing BioGPT-X's 0.88±0.03, indicating a 5% performance improvement.
   - **Neuroimaging Data (AD vs. Normal)**: The proposed model recorded an AUROC of 0.89±0.01, significantly higher than ResNet-18's 0.81±0.02.
   - **Biosignal Data (AF vs. Normal)**: The proposed model achieved an AUROC of 0.94±0.01, outperforming the single-modality CNN's 0.87±0.02.
   - **Rare Disease Retrieval**: The proposed model correctly retrieved matching cases in 93.3% of instances from a simulated dataset of 30 patients, demonstrating capabilities for cross-modal retrieval that existing models could not achieve.

These results indicate that the proposed model effectively supports the diagnosis of rare diseases by integrating multimodal data. Notably, the model outperformed existing single-modality models, highlighting the advantages of multimodal learning.


<br/>
# 예제



이 논문에서는 희귀 질병의 표현을 학습하기 위해 다중 모달 자기 지도 학습(self-supervised learning) 접근 방식을 제안합니다. 이 모델은 유전체 데이터(Whole Exome Sequencing, WES), 3D MRI 이미지, 그리고 웨어러블 센서 데이터(예: ECG, PPG)를 통합하여 환자의 표현을 학습합니다. 

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **유전체 데이터**: TCGA-GBM 데이터셋을 사용하여 250명의 신경교종(GBM) 환자와 50명의 정상 대조군의 WES 변이 데이터를 포함합니다. 각 변이는 VCF 형식으로 제공되며, 단일 뉴클레오타이드 변이(SNV)와 작은 인델을 포함합니다.
   - **MRI 데이터**: ADNI 데이터셋에서 200명의 알츠하이머 환자와 200명의 연령대가 일치하는 대조군의 T1 가중치 뇌 MRI 이미지를 사용합니다. 각 이미지는 1mm³의 등방성 복셀로 재샘플링되고, 두개골이 제거되며, 강도 정규화가 이루어집니다.
   - **웨어러블 센서 데이터**: UK Biobank에서 100명의 피험자로부터 수집된 500시간의 PPG 신호 데이터를 사용합니다. 이 데이터는 심장 부정맥 진단을 받은 피험자와 건강한 피험자를 포함합니다.

2. **테스트 데이터**:
   - **희귀 질병 시뮬레이션 데이터**: 30명의 "시뮬레이션 환자"를 구성하여 각 환자에 대해 10개의 TCGA-GBM 유전체, 10개의 ADNI MRI 스캔, 10개의 UK Biobank 신호를 매칭하여 10개의 서로 다른 희귀 질병을 나타냅니다. 이 데이터는 모델의 제로샷(zero-shot) 검색 성능을 평가하는 데 사용됩니다.

#### 구체적인 테스크

- **제로샷 희귀 질병 검색**: 주어진 유전체 데이터에 대해 가장 유사한 사례를 검색하는 작업입니다. 예를 들어, 특정 유전자 변이를 가진 환자의 유전체 데이터를 입력으로 제공하면, 모델은 해당 환자와 유사한 MRI 또는 PPG 기록을 검색합니다.
- **분류 작업**: 모델은 학습된 표현을 사용하여 특정 질병(예: GBM vs. 정상, AD vs. 정상, 심방세동 vs. 정상)을 분류하는 작업을 수행합니다. 이 작업에서는 AUROC(Receiver Operating Characteristic Area Under the Curve)와 F1 점수를 사용하여 성능을 평가합니다.

이러한 방식으로, 모델은 다양한 모달리티의 데이터를 통합하여 희귀 질병의 진단을 지원하는 데 기여할 수 있습니다.

---




This paper proposes a multimodal self-supervised learning approach for phenotyping rare diseases. The model integrates genomic data (Whole Exome Sequencing, WES), 3D MRI images, and wearable sensor data (e.g., ECG, PPG) to learn patient representations.

#### Training Data and Test Data

1. **Training Data**:
   - **Genomic Data**: The TCGA-GBM dataset is used, which includes WES variant data from 250 glioblastoma (GBM) patients and 50 normal controls. Each variant is provided in VCF format and includes single nucleotide variants (SNVs) and small indels.
   - **MRI Data**: The ADNI dataset provides T1-weighted brain MRI images from 200 Alzheimer's patients and 200 age-matched controls. Each image is resampled to 1mm³ isotropic voxels, skull-stripped, and intensity-normalized.
   - **Wearable Sensor Data**: The UK Biobank dataset includes 500 hours of PPG signal data collected from 100 subjects, some diagnosed with cardiac arrhythmias and others healthy.

2. **Test Data**:
   - **Rare Disease Simulation Data**: A set of 30 "simulated patients" is constructed, pairing 10 TCGA-GBM genomes, 10 ADNI MRI scans, and 10 UK Biobank signals to represent 10 distinct rare conditions. This data is used to evaluate the model's zero-shot retrieval performance.

#### Specific Tasks

- **Zero-Shot Rare Disease Retrieval**: This task involves retrieving the most similar case given a genomic input. For example, when provided with the genomic data of a patient with a specific genetic variant, the model retrieves the corresponding MRI or PPG record of that patient.
- **Classification Tasks**: The model performs classification tasks using the learned embeddings to distinguish between specific diseases (e.g., GBM vs. control, AD vs. control, atrial fibrillation vs. normal). Performance is evaluated using metrics such as AUROC (Area Under the Receiver Operating Characteristic Curve) and F1 score.

In this way, the model aims to contribute to the diagnosis of rare diseases by integrating data from various modalities.

<br/>
# 요약


이 논문에서는 희귀 질환의 표현형을 파악하기 위해 유전자, 3D 이미징, 웨어러블 센서 데이터를 통합한 멀티모달 자기 지도 학습 모델을 제안합니다. 실험 결과, 이 모델은 제로샷 환경에서 기존의 최첨단 방법보다 5% 이상의 AUROC 향상을 보여주었으며, 다양한 데이터 모달리티 간의 상관관계를 효과적으로 활용하여 진단 정확도를 높였습니다. 예를 들어, 특정 유전자 변이를 가진 환자의 MRI 이미지를 성공적으로 매칭하여 질병의 표현형을 파악하는 데 기여했습니다.

---

In this paper, a multimodal self-supervised learning model is proposed to phenotype rare diseases by integrating genomic, 3D imaging, and wearable sensor data. Experimental results show that this model achieves over a 5% AUROC improvement compared to state-of-the-art methods in a zero-shot setting, effectively leveraging correlations across different data modalities to enhance diagnostic accuracy. For instance, it successfully matched MRI images of patients with specific genetic variants, aiding in the identification of disease phenotypes.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 통합된 자기 지도 학습 아키텍처의 개요를 보여줍니다. 각 모달리티(유전체, 영상, 신호)에 대한 인코더가 공유 잠재 공간으로 수렴하는 과정을 시각화하여, 다양한 데이터 유형이 어떻게 통합되는지를 설명합니다.
   - **Figure 2**: t-SNE 시각화로, 환자 임베딩이 질병에 따라 클러스터링되는 모습을 보여줍니다. 이는 모델이 데이터 소스가 아닌 질병 상태에 따라 임베딩을 그룹화하고 있음을 나타냅니다.
   - **Figure 3**: 수신자 조작 특성(ROC) 곡선을 통해 모델과 베이스라인 간의 성능 비교를 시각적으로 나타냅니다. 모델이 여러 모달리티에서 높은 AUROC를 달성했음을 보여줍니다.
   - **Figure 4**: 신뢰성 곡선으로, 모델의 예측 확률이 실제 빈도와 얼마나 잘 일치하는지를 보여줍니다. 이는 모델의 예측이 잘 보정되었음을 나타냅니다.
   - **Figure 6**: 온톨로지 그래프 거리와 임베딩 공간 거리 간의 관계를 보여줍니다. 이는 온톨로지의 의미적 근접성이 학습된 표현에 반영되고 있음을 나타냅니다.

2. **테이블**
   - **Table 1**: 평가 데이터셋과 각 데이터셋의 특성을 요약합니다. 각 모달리티에 대한 긍정 클래스와 작업을 명시하여, 모델의 평가 기준을 명확히 합니다.
   - **Table 2**: 모델 성능을 베이스라인과 비교한 결과를 보여줍니다. 통합된 다중 모달 모델이 거의 모든 메트릭에서 가장 높은 점수를 기록했음을 강조합니다.
   - **Table 3**: 각 구성 요소 또는 모달리티를 제거했을 때의 성능 변화를 보여주는 절단 분석 결과입니다. 이는 각 구성 요소의 중요성을 강조합니다.

3. **어펜딕스**
   - **Appendix A**: 주요 하이퍼파라미터와 그 값을 나열하여, 모델 훈련에 사용된 설정을 명확히 합니다. 이는 모델의 재현성을 높이는 데 기여합니다.




1. **Diagrams and Figures**
   - **Figure 1**: Provides an overview of the unified self-supervised learning architecture, illustrating how modality-specific encoders converge into a shared latent space. This visualizes the integration of diverse data types.
   - **Figure 2**: Shows a t-SNE visualization of patient embeddings, indicating that the model clusters embeddings primarily by disease rather than by data source, demonstrating effective cross-modal alignment.
   - **Figure 3**: Displays receiver operating characteristic (ROC) curves comparing the model's performance against baselines across modalities, highlighting the model's superior AUROC scores.
   - **Figure 4**: Reliability curves illustrate how well the model's predicted probabilities align with observed frequencies, indicating good calibration of the model's predictions.
   - **Figure 6**: Shows the relationship between ontology graph distance and embedding space distance, suggesting that semantic proximity in the ontology is reflected in the learned representations.

2. **Tables**
   - **Table 1**: Summarizes the evaluation datasets and their characteristics, clarifying the positive classes and tasks for each modality, which helps define the evaluation criteria for the model.
   - **Table 2**: Presents performance metrics of the model compared to baselines, emphasizing that the unified multimodal model achieves the highest scores across nearly all metrics.
   - **Table 3**: Provides results from ablation studies that quantify the contribution of each component or modality, underscoring the importance of each part of the model.

3. **Appendix**
   - **Appendix A**: Lists key hyperparameters and their values, clarifying the settings used for model training, which contributes to the reproducibility of the model.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Uppalapati2025,
  author = {Khartik Uppalapati and Bora Yimenicioglu and Shakeel Abdulkareem and Adan Eftekhari},
  title = {Rare Disease Phenotyping via Self-Supervised Multimodal Learning},
  booktitle = {Proceedings of the 16th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB '25)},
  year = {2025},
  month = {October},
  location = {Philadelphia, PA, USA},
  publisher = {ACM},
  pages = {1--12},
  doi = {10.1145/3765612.3767304}
}
```

### 시카고 스타일

Uppalapati, Khartik, Bora Yimenicioglu, Shakeel Abdulkareem, and Adan Eftekhari. 2025. "Rare Disease Phenotyping via Self-Supervised Multimodal Learning." In *Proceedings of the 16th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB '25)*, 1-12. Philadelphia, PA, USA: ACM. https://doi.org/10.1145/3765612.3767304.
